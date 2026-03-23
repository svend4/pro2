"""
Официальный запуск Недели 4: solan_nautilus.

NautilusMoME + GlyphAwareEmbedding (SOLAN-76 Q6 токенизация).
Гипотеза: SOLAN Q6-глифы как дополнительный embedding улучшают
маршрутизацию и cross-domain diversity.

Метрики успеха:
  - ксерокс-тест > 80% (4/5 правильных routing + PPL)
  - cross_domain_diversity > 0.16 (std routing по доменам)
  - PPL не хуже lean_baseline (1.51)

Использование:
    python experiments/train_solan_nautilus.py
    python experiments/train_solan_nautilus.py --dry-run
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from yijing_transformer.models.lean_model import LeanYiJingGPT
from yijing_transformer.models.geometry.kasatkin_router import KasatkinQ6Router
from yijing_transformer.tokenizer.glyph_tokenizer import GlyphTokenizer
from corpus_loader import CorpusLoader


DEFAULTS = dict(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=4,
    block_size=256,
    dropout=0.0,
    steps=3000,
    batch_size=8,
    lr=3e-4,
    warmup_steps=200,
    grad_clip=1.0,
    n_experts=6,
    routing_temperature=0.5,
    routing_loss_weight=0.30,   # λ для routing supervision loss (сильный сигнал)
    log_every=100,
    xerox_every=500,
    save_every=1000,
    # Критерии успеха
    target_xerox_rate=0.80,
    target_diversity=0.16,
    baseline_ppl=1.51,
)

# Маппинг доменов корпуса → индекс эксперта
_DOMAIN_TO_EXPERT = {
    'GEO':    0,   # CODE  (+X)
    'HYDRO':  1,   # RECON (-X)
    'PYRO':   2,   # SYSTEM (+Y)
    'AERO':   3,   # MATH  (-Y)
    'COSMO':  4,   # HUMAN (+Z)
    'NOOS':   5,   # INFO  (-Z)
    'METHOD': 5,
    'YIJING': 4,
}


def detect_domain_from_text(text: str) -> int:
    """Точный детектор домена по содержимому текста.

    Приоритет выше, чем corpus domain labels — используется для
    routing supervision и ксерокс-теста.

    Returns:
        int: 0=CODE, 1=RECON, 2=SYSTEM, 3=MATH, 4=HUMAN, 5=INFO
    """
    # CODE: Python/JS ключевые слова
    if any(kw in text for kw in [
        'def ', 'class ', 'import ', 'return ', 'if __name__',
        'function ', 'const ', 'var ', 'let ', '# -*- coding',
    ]):
        return 0
    # RECON: Русский текст
    if sum(1 for c in text if 'а' <= c.lower() <= 'я') > len(text) * 0.1:
        return 1
    # SYSTEM: SQL/DevOps
    tl = text.lower()
    if any(kw in tl for kw in [
        'select ', 'from ', 'where ', 'docker', 'kubectl',
        'kubernetes', 'nginx', 'yaml:', 'k8s',
    ]):
        return 2
    # MATH: Математика
    if any(kw in text for kw in [
        '∑', '∫', '∂', 'theorem', 'proof', 'lemma', 'matrix',
        'σ', '∈', '∀', '∃',
    ]):
        return 3
    # HUMAN: Философия/общество
    if any(kw in tl for kw in [
        'philosophy', 'human', 'society', 'people', 'story',
        'consciousness', 'meaning', 'value',
    ]):
        return 4
    # INFO: знания/методология/остальное
    return 5

# Ксерокс-тест: (текст, имя_эксперта, индекс_эксперта, max_ppl)
_XEROX_TESTS = [
    ("def neural_network(x):",  "CODE",   0, 8.0),
    ("class GradientDescent:",   "CODE",   0, 8.0),
    ("SELECT * FROM experts",    "SYSTEM", 2, 8.0),
    ("Hexagram as archetype",    "HUMAN",  4, 8.0),
    ("knowledge concept theory", "INFO",   5, 8.0),
]


def _build_q6_byte_table() -> torch.Tensor:
    """Строит таблицу Q6-кодов для всех 256 байт.

    Каждый байт → 6 бит (LSB) → вектор {-1, +1}^6.
    Возвращает тензор (256, 6).
    """
    table = torch.zeros(256, 6)
    for b in range(256):
        for i in range(6):
            bit = (b >> i) & 1
            table[b, i] = 1.0 if bit else -1.0
    return table


class SolanNautilusModel(nn.Module):
    """LeanYiJingGPT + SOLAN Q6 glyph embedding + KasatkinQ6Router.

    Архитектура:
        bytes → byte_embed + q6_glyph_embed → LeanBlocks → Router
                                                              ↓
                                               expert_projs × 6 → weighted mix
                                                              ↓
                                                     LM head (vocab=256)

    Q6 glyph embedding: каждый байт → 6-битный Q6 код → q6_basis проекция.
    Это добавляет геометрическую информацию о структуре символа к стандартному
    обучаемому embedding.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        d_model = cfg['d_model']

        self.base = LeanYiJingGPT(
            vocab_size=cfg['vocab_size'],
            d_model=d_model,
            n_layers=cfg['n_layers'],
            n_heads=cfg['n_heads'],
            block_size=cfg['block_size'],
            dropout=cfg['dropout'],
        )

        # Q6 byte lookup table (256 байт → 6-битный код)
        q6_table = _build_q6_byte_table()
        self.register_buffer('q6_table', q6_table)

        # Q6-basis: случайная фиксированная проекция 6 → d_model
        q6_basis = torch.randn(6, d_model) / (6 ** 0.5)
        self.register_buffer('q6_basis', q6_basis)

        # Learnable масштаб Q6 glyph embedding
        self.glyph_scale = nn.Parameter(torch.tensor(0.1))

        # KasatkinQ6Router для измерения routing
        self.router = KasatkinQ6Router(
            d_model=d_model,
            n_experts=cfg['n_experts'],
            routing_temperature=cfg['routing_temperature'],
        )

        # 6 лёгких экспертных проекторов
        self.expert_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(cfg['n_experts'])
        ])

        self.d_model = d_model
        self.block_size = cfg['block_size']

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _q6_embed(self, idx: torch.Tensor) -> torch.Tensor:
        """Q6 glyph embedding для батча байт-индексов.

        Args:
            idx: (B, T) байтовые индексы

        Returns:
            (B, T, d_model) Q6 glyph embeddings
        """
        # (B, T, 6) — Q6 биты для каждого байта
        q6_codes = self.q6_table[idx]
        # (B, T, d_model) — проекция через basis
        return q6_codes @ self.q6_basis

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        domain_idx: int | None = None,
        routing_loss_weight: float = 0.05,
    ) -> tuple:
        """
        Args:
            idx: (B, T) байтовые индексы
            targets: (B, T) целевые индексы
            domain_idx: если задан — добавляет routing supervision loss
            routing_loss_weight: λ для routing loss

        Returns:
            (logits, total_loss, routing_weights)
        """
        base = self.base
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        # Стандартный byte embedding
        byte_emb = base.embed(idx) + base.pos_embed(pos)

        # Q6 glyph embedding (геометрический, SOLAN)
        glyph_emb = self._q6_embed(idx)

        # Объединение: byte + scaled Q6 glyph
        x = base.drop(byte_emb + self.glyph_scale * glyph_emb)

        # LeanYiJing блоки
        for block in base.blocks:
            x = block(x)

        # Routing
        routing_weights = self.router(x)

        # Expert mix
        expert_outs = torch.stack(
            [proj(x) for proj in self.expert_projs], dim=-1
        )  # (B, T, d_model, 6)
        routed = (expert_outs * routing_weights.unsqueeze(2)).sum(dim=-1)

        # Residual + norm + head
        x = base.norm(x + routed)
        logits = base.head(x)

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss = lm_loss

            # Routing supervision: cross-entropy к правильному эксперту
            # Сильнее чем MSE: прямо максимизирует P(correct_expert)
            if domain_idx is not None:
                avg_routing = routing_weights.mean(dim=(0, 1))  # (6,)
                target = torch.tensor(domain_idx, device=idx.device)
                routing_loss = F.cross_entropy(avg_routing.unsqueeze(0), target.unsqueeze(0))
                loss = lm_loss + routing_loss_weight * routing_loss

        return logits, loss, routing_weights

    def get_routing_confidence(self, x: torch.Tensor) -> float:
        return self.router.get_routing_confidence(x)


def build_dataset() -> tuple[list, list]:
    """Загружает корпус и возвращает:
    - texts_with_domains: все тексты [(encoded, expert_idx)]
    - domain_buckets: тексты сгруппированные по домену [6 списков]

    Использует detect_domain_from_text для точного определения домена.
    """
    loader = CorpusLoader()
    corpus = loader.as_training_corpus(max_per_source=1000)
    texts_with_domains = []
    domain_buckets: list[list] = [[] for _ in range(6)]

    domain_counts = [0] * 6
    for d in corpus:
        text = d['text']
        if len(text) < 50:
            continue
        expert_idx = detect_domain_from_text(text)
        domain_counts[expert_idx] += 1
        encoded = list(text.encode('utf-8', errors='replace'))
        texts_with_domains.append((encoded, expert_idx))
        domain_buckets[expert_idx].append(encoded)

    expert_names = KasatkinQ6Router.EXPERT_NAMES
    print(f"  Корпус: {len(texts_with_domains)} текстов")
    print(f"  Домены: " + " | ".join(
        f"{expert_names[i]}:{domain_counts[i]}" for i in range(6)
    ))
    return texts_with_domains, domain_buckets


def get_batch(texts_with_domains: list, block_size: int, batch_size: int,
              device: str) -> tuple:
    """Возвращает батч (x, y, domain_indices) — случайная выборка."""
    x_list, y_list, domain_list = [], [], []
    indices = torch.randint(len(texts_with_domains), (batch_size,))
    for idx in indices:
        encoded, domain_idx = texts_with_domains[idx.item()]
        if len(encoded) < block_size + 1:
            encoded = encoded + [0] * (block_size + 1 - len(encoded))
        start = torch.randint(0, max(1, len(encoded) - block_size), (1,)).item()
        chunk = encoded[start: start + block_size + 1]
        x_list.append(chunk[:block_size])
        y_list.append(chunk[1: block_size + 1])
        domain_list.append(domain_idx)
    return (
        torch.tensor(x_list, dtype=torch.long, device=device),
        torch.tensor(y_list, dtype=torch.long, device=device),
        domain_list,
    )


def get_balanced_domain_batch(domain_buckets: list, block_size: int,
                               device: str) -> tuple:
    """Батч из одного случайного домена — для routing supervision.

    Выбирает домен равномерно (не пропорционально размеру),
    чтобы routing учился всем 6 доменам одинаково.

    Returns:
        (x, y, domain_idx)
    """
    # Выбираем только домены с хотя бы одним текстом
    available = [i for i, bucket in enumerate(domain_buckets) if bucket]
    domain_idx = available[torch.randint(len(available), (1,)).item()]
    bucket = domain_buckets[domain_idx]

    seq_idx = torch.randint(len(bucket), (1,)).item()
    encoded = bucket[seq_idx]
    if len(encoded) < block_size + 1:
        encoded = encoded + [0] * (block_size + 1 - len(encoded))
    start = torch.randint(0, max(1, len(encoded) - block_size), (1,)).item()
    chunk = encoded[start: start + block_size + 1]

    x = torch.tensor([chunk[:block_size]], dtype=torch.long, device=device)
    y = torch.tensor([chunk[1: block_size + 1]], dtype=torch.long, device=device)
    return x, y, domain_idx


def get_lr(step: int, cfg: dict) -> float:
    if step < cfg['warmup_steps']:
        return cfg['lr'] * step / max(cfg['warmup_steps'], 1)
    progress = (step - cfg['warmup_steps']) / max(1, cfg['steps'] - cfg['warmup_steps'])
    return cfg['lr'] * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_xerox_test(model: SolanNautilusModel, step: int, device: str) -> dict:
    """Ксерокс-тест: маршрутизация и PPL для диагностических текстов."""
    model.eval()
    results = []

    for text, exp_name, exp_idx, max_ppl in _XEROX_TESTS:
        encoded = list(text.encode('utf-8', errors='replace'))
        T = min(len(encoded), model.block_size)
        encoded = encoded[:T]

        idx = torch.tensor([encoded], dtype=torch.long, device=device)
        targets = idx.clone()  # для PPL измерения

        with torch.no_grad():
            logits, _, routing_weights = model(idx)

        # PPL
        log_probs = F.log_softmax(logits, dim=-1)
        ppl = math.exp(min(
            -log_probs[0, :-1, :].gather(1, targets[0, 1:].unsqueeze(1)).mean().item(),
            20
        ))

        # Routing
        avg_weights = routing_weights.mean(dim=(0, 1))  # (6,)
        actual_expert_idx = avg_weights.argmax().item()
        expert_names = KasatkinQ6Router.EXPERT_NAMES
        routing_correct = (actual_expert_idx == exp_idx)

        results.append({
            'text': text,
            'expected': exp_name,
            'actual': expert_names[actual_expert_idx],
            'routing_correct': routing_correct,
            'ppl': round(ppl, 2),
            'ppl_ok': ppl < max_ppl,
            'weights': [round(w.item(), 3) for w in avg_weights],
        })

    passed = sum(r['routing_correct'] and r['ppl_ok'] for r in results)
    total = len(results)
    print(f"  [Ксерокс step={step}] {passed}/{total} пройдено")
    for r in results:
        status = "✓" if (r['routing_correct'] and r['ppl_ok']) else "✗"
        print(f"    {status} '{r['text'][:28]}' | {r['actual']:6s} "
              f"(ожид={r['expected']}) | PPL={r['ppl']:.1f}")

    model.train()
    return {'passed': passed, 'total': total, 'pass_rate': passed / total,
            'details': results}


def compute_cross_domain_diversity(model: SolanNautilusModel, device: str) -> float:
    """Считает std routing weights через 6 доменных текстов.

    Высокое значение = роутер чётко различает домены.
    Цель: > 0.16
    """
    domain_texts = [
        "def neural_network(x): return W @ x",          # CODE  (0)
        "Гексаграмма как архетип связи",                   # RECON (1)
        "SELECT id, name FROM users WHERE active = 1",   # SYSTEM(2)
        "f(x) = sum(w_i * phi_i(x)) для i=1..n",        # MATH  (3)
        "The universe expands with dark energy",          # HUMAN (4)
        "knowledge concept ontology methodology",        # INFO  (5)
    ]

    model.eval()
    all_weights = []
    for text in domain_texts:
        encoded = list(text.encode('utf-8', errors='replace'))
        T = min(len(encoded), model.block_size)
        idx = torch.tensor([encoded[:T]], dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, routing_weights = model(idx)
        avg = routing_weights.mean(dim=(0, 1)).cpu()  # (6,)
        all_weights.append(avg)

    # Матрица 6 × 6: строки = домены, столбцы = эксперты
    w_matrix = torch.stack(all_weights)   # (6, 6)
    # Diversity = std по среднему каждого эксперта через домены
    diversity = w_matrix.std(dim=0).mean().item()
    model.train()
    return diversity


def load_baseline_ppl() -> float | None:
    path = Path('experiments') / 'lean_baseline_log.json'
    if path.exists():
        return json.loads(path.read_text()).get('final_ppl')
    return None


def train(cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_ppl = load_baseline_ppl() or cfg['baseline_ppl']

    print(f"\n{'='*64}")
    print(f"  SOLAN NAUTILUS (Неделя 4)")
    print(f"{'='*64}")
    print(f"  Устройство        : {device}")
    print(f"  Шаги              : {cfg['steps']}")
    print(f"  Baseline PPL      : {baseline_ppl:.4f}")
    print(f"  Routing λ         : {cfg['routing_loss_weight']}")
    print(f"  Цель ксерокс      : > {cfg['target_xerox_rate']:.0%}")
    print(f"  Цель diversity    : > {cfg['target_diversity']}")

    texts_with_domains, domain_buckets = build_dataset()
    model = SolanNautilusModel(cfg).to(device)
    print(f"  Параметры         : {model.num_parameters:,}")
    print(f"  glyph_scale       : {model.glyph_scale.item():.3f} (обучаемый)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], betas=(0.9, 0.95), weight_decay=0.1
    )

    log = {
        'config': cfg, 'baseline_ppl': baseline_ppl,
        'steps': [], 'losses': [], 'ppls': [],
        'routing_confidences': [],
        'xerox_results': [],
        'diversity_history': [],
    }

    best_loss = float('inf')
    t0 = time.time()
    model.train()

    print(f"\n{'─'*64}")
    for step in range(1, cfg['steps'] + 1):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Чётные шаги: LM из общего корпуса (реалистичный mix)
        # Нечётные шаги: balanced domain batch (равномерно по доменам)
        if step % 2 == 0:
            from collections import Counter
            x, y, domain_list = get_batch(texts_with_domains, cfg['block_size'],
                                           cfg['batch_size'], device)
            domain_idx = Counter(domain_list).most_common(1)[0][0]
        else:
            x, y, domain_idx = get_balanced_domain_batch(
                domain_buckets, cfg['block_size'], device
            )

        logits, loss, routing_weights = model(
            x, y, domain_idx=domain_idx,
            routing_loss_weight=cfg['routing_loss_weight'],
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()

        loss_val = loss.item()
        # LM loss без routing для PPL
        with torch.no_grad():
            _, lm_only_loss, _ = model(x, y, domain_idx=None)
        ppl_val = math.exp(min(lm_only_loss.item(), 20))

        if loss_val < best_loss:
            best_loss = loss_val

        if step % cfg['log_every'] == 0 or step == 1:
            elapsed = time.time() - t0
            log['steps'].append(step)
            log['losses'].append(round(loss_val, 4))
            log['ppls'].append(round(ppl_val, 4))
            gs = model.glyph_scale.item()
            print(
                f"  step={step:4d}/{cfg['steps']} | "
                f"loss={loss_val:.4f} | ppl={ppl_val:.4f} | "
                f"glyph_scale={gs:.3f} | lr={lr:.2e} | {elapsed:.0f}s"
            )

        # Routing confidence
        if step % cfg['xerox_every'] == 0:
            with torch.no_grad():
                base = model.base
                byte_emb = base.embed(x) + base.pos_embed(
                    torch.arange(x.shape[1], device=device)
                )
                glyph_emb = model._q6_embed(x)
                h = base.drop(byte_emb + model.glyph_scale * glyph_emb)
                conf = model.get_routing_confidence(h)
            log['routing_confidences'].append({'step': step, 'conf': round(conf, 4)})

            avg_weights = routing_weights.mean(dim=(0, 1))
            experts_str = " ".join(
                f"{KasatkinQ6Router.EXPERT_NAMES[i]}:{avg_weights[i]:.2f}"
                for i in range(6)
            )
            print(f"  [Routing step={step}] conf={conf:.4f} | {experts_str}")

            # Ксерокс-тест
            xerox = run_xerox_test(model, step, device)
            log['xerox_results'].append({'step': step, **xerox})

            # Cross-domain diversity
            div = compute_cross_domain_diversity(model, device)
            log['diversity_history'].append({'step': step, 'diversity': round(div, 4)})
            print(f"  [Diversity step={step}] {div:.4f} (цель > {cfg['target_diversity']})")

        if step % cfg['save_every'] == 0:
            ckpt = Path('experiments') / f'solan_nautilus_step{step}.pt'
            torch.save({'step': step, 'model_state': model.state_dict(),
                        'ppl': ppl_val, 'config': cfg}, ckpt)

    # Финальные метрики
    final_ppl = log['ppls'][-1] if log['ppls'] else float('inf')
    final_xerox = log['xerox_results'][-1] if log['xerox_results'] else {'pass_rate': 0, 'passed': 0, 'total': 0}
    final_diversity = log['diversity_history'][-1]['diversity'] if log['diversity_history'] else 0.0
    total_time = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  РЕЗУЛЬТАТ solan_nautilus")
    print(f"{'='*64}")
    print(f"  Final PPL         : {final_ppl:.4f}")
    print(f"  Ксерокс-тест      : {final_xerox['passed']}/{final_xerox['total']} "
          f"({final_xerox['pass_rate']:.0%})")
    print(f"  Cross-domain div  : {final_diversity:.4f}")
    print(f"  Baseline PPL      : {baseline_ppl:.4f}")

    verdict = "inconclusive"
    if cfg['steps'] < 3000:
        print(f"  ⚠  < 3000 шагов — вердикт недействителен")
    else:
        xerox_ok = final_xerox['pass_rate'] >= cfg['target_xerox_rate']
        div_ok = final_diversity >= cfg['target_diversity']
        ppl_ok = final_ppl <= baseline_ppl * 1.02

        if xerox_ok and div_ok and ppl_ok:
            print(f"  ✓  УСПЕХ: xerox={final_xerox['pass_rate']:.0%}, "
                  f"diversity={final_diversity:.4f}, PPL={final_ppl:.4f}")
            verdict = "proven"
        elif not xerox_ok:
            print(f"  ✗  Ксерокс-тест недостаточен: "
                  f"{final_xerox['pass_rate']:.0%} < {cfg['target_xerox_rate']:.0%}")
            verdict = "disproven"
        elif not div_ok:
            print(f"  ✗  Cross-domain diversity мала: "
                  f"{final_diversity:.4f} < {cfg['target_diversity']}")
            verdict = "disproven"
        else:
            print(f"  ✗  PPL хуже baseline: {final_ppl:.4f} > {baseline_ppl:.4f}")
            verdict = "disproven"

    final_ckpt = Path('experiments') / 'solan_nautilus_final.pt'
    torch.save({'step': cfg['steps'], 'model_state': model.state_dict(),
                'final_ppl': final_ppl,
                'xerox_pass_rate': final_xerox['pass_rate'],
                'cross_domain_diversity': final_diversity,
                'verdict': verdict, 'config': cfg}, final_ckpt)

    log.update({
        'final_ppl': final_ppl,
        'xerox_pass_rate': final_xerox['pass_rate'],
        'cross_domain_diversity': final_diversity,
        'verdict': verdict,
        'total_time_s': round(total_time, 1),
    })
    log_path = Path('experiments') / 'solan_nautilus_log.json'
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"  Лог: {log_path}")

    return final_ppl, final_xerox['pass_rate'], final_diversity, verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.dry_run:
        cfg['steps'] = 50
        cfg['log_every'] = 10
        cfg['xerox_every'] = 25
        cfg['save_every'] = 9999

    train(cfg)
