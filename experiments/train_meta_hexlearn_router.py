"""
Официальный запуск Недели 6: meta_hexlearn_router.

LeanYiJingGPT + HexLearnRouter (геометрический роутер из meta/hexlearn).
Гипотеза: Q6/Hamming k-NN роутинг даёт routing_confidence > 20%.

Отличие от KasatkinQ6Router (Неделя 3):
  Касаткин: d_model → 3D → cosine similarity с ±X,±Y,±Z (6 осей)
  HexLearn: d_model → 6D Q6 координаты → soft Hamming dist к 6 unit-векторам

  Преимущество HexLearn: работает в родном пространстве Q6 (не 3D-проекция),
  использует геометрию из meta/projects/hexlearn/hexlearn.py (KNN, KMedoids).

Метрики успеха:
  - routing_confidence > 20% (= 0.20)
  - PPL ≤ lean_baseline * 1.02 (= 1.54)

Использование:
    python experiments/train_meta_hexlearn_router.py
    python experiments/train_meta_hexlearn_router.py --dry-run
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
    log_every=100,
    conf_every=500,
    save_every=1000,
    # Критерии успеха
    target_confidence=0.20,
    baseline_ppl=1.51,
)

# Имена экспертов — те же что у KasatkinQ6Router для совместимости
EXPERT_NAMES = ['CODE', 'RECON', 'SYSTEM', 'MATH', 'HUMAN', 'INFO']

# 6 единичных векторов Q6 (one-hot в {0,1}^6) — эксперты как Q6 вершины
# Это ортогональнее чем ±X,±Y,±Z Касаткина (которые образуют 3 пары)
_EXPERT_VERTICES_Q6 = torch.tensor([
    [1., 0., 0., 0., 0., 0.],   # CODE  = вершина 32 (100000)
    [0., 1., 0., 0., 0., 0.],   # RECON = вершина 16 (010000)
    [0., 0., 1., 0., 0., 0.],   # SYSTEM= вершина 8  (001000)
    [0., 0., 0., 1., 0., 0.],   # MATH  = вершина 4  (000100)
    [0., 0., 0., 0., 1., 0.],   # HUMAN = вершина 2  (000010)
    [0., 0., 0., 0., 0., 1.],   # INFO  = вершина 1  (000001)
], dtype=torch.float32)


class HexLearnRouter(nn.Module):
    """Q6/Hamming геометрический роутер из meta/hexlearn.

    Вместо 3D-проекции Касаткина использует родную 6D геометрию Q6:
      x → proj_q6 → sigmoid → [0,1]^6 мягкие координаты
      dist_i = Hamming-like(coords, expert_vertex_i) = sum|coords - ev_i|
      routing = softmax(-dist / temperature)

    Преимущество: 6 осей Q6 взаимно ортогональны (unit vectors),
    что даёт более чёткое разделение доменов чем ±X,±Y,±Z в 3D.

    Args:
        d_model: размерность входных векторов
        n_experts: ДОЛЖНО быть 6 (= числу вершин Q6-unit)
        routing_temperature: температура softmax
    """

    def __init__(
        self,
        d_model: int = 128,
        n_experts: int = 6,
        routing_temperature: float = 0.5,
    ):
        super().__init__()
        if n_experts != 6:
            raise ValueError(f"HexLearnRouter требует n_experts=6, получено {n_experts}")

        self.d_model = d_model
        self.n_experts = n_experts
        self.routing_temperature = routing_temperature

        # Проектор d_model → 6D Q6 (обучаемый)
        self.proj_q6 = nn.Linear(d_model, 6, bias=True)

        # 6 единичных экспертных вершин Q6 (фиксированы)
        self.register_buffer('expert_vertices', _EXPERT_VERTICES_Q6)

        # Инициализация: ортогональная матрица (6×d_model)
        nn.init.xavier_uniform_(self.proj_q6.weight)
        nn.init.zeros_(self.proj_q6.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет мягкие routing weights через Q6-геометрию.

        Args:
            x: (B, T, d_model)

        Returns:
            routing_weights: (B, T, n_experts)
        """
        # Проекция в мягкие Q6 координаты [0,1]^6 через sigmoid
        coords = torch.sigmoid(self.proj_q6(x))  # (B, T, 6)

        # Мягкое расстояние Хэмминга до каждой экспертной вершины
        # dist[i,j,k] = |coords[i,j] - expert_vertices[k]| сумма по битам
        ev = self.expert_vertices.view(1, 1, 6, 6)   # (1, 1, 6_experts, 6_bits)
        c = coords.unsqueeze(2)                        # (B, T, 1, 6)
        soft_hamming = (c - ev).abs().sum(dim=-1)      # (B, T, 6_experts)

        # Routing: близкие вершины получают больший вес
        routing_weights = F.softmax(
            -soft_hamming / self.routing_temperature,
            dim=-1,
        )  # (B, T, 6)

        return routing_weights

    def get_routing_confidence(self, x: torch.Tensor) -> float:
        """Метрика уверенности роутера: среднее (max_weight - 1/n_experts).

        Цель: > 20% (т.е. conf > 0.20).
        """
        weights = self.forward(x)
        max_weights = weights.max(dim=-1).values
        return (max_weights - 1.0 / self.n_experts).mean().item()

    def get_q6_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает Q6 координаты для визуализации.

        Returns:
            coords: (B, T, 6) — значения sigmoid в [0,1]^6
        """
        with torch.no_grad():
            return torch.sigmoid(self.proj_q6(x))


class LeanWithHexLearnRouter(nn.Module):
    """LeanYiJingGPT + HexLearnRouter как слой маршрутизации.

    Архитектурно идентичен LeanWithKasatkinRouter (Неделя 3),
    но использует HexLearnRouter вместо KasatkinQ6Router.

    Это позволяет прямое сравнение: только router меняется.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.base = LeanYiJingGPT(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
            n_heads=cfg['n_heads'],
            block_size=cfg['block_size'],
            dropout=cfg['dropout'],
        )
        self.router = HexLearnRouter(
            d_model=cfg['d_model'],
            n_experts=cfg['n_experts'],
            routing_temperature=cfg['routing_temperature'],
        )
        self.expert_projs = nn.ModuleList([
            nn.Linear(cfg['d_model'], cfg['d_model'], bias=False)
            for _ in range(cfg['n_experts'])
        ])
        self.d_model = cfg['d_model']
        self.block_size = cfg['block_size']

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple:
        base = self.base
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = base.drop(base.embed(idx) + base.pos_embed(pos))
        for block in base.blocks:
            x = block(x)

        routing_weights = self.router(x)

        expert_outs = torch.stack(
            [proj(x) for proj in self.expert_projs], dim=-1
        )  # (B, T, d_model, 6)
        routed = (expert_outs * routing_weights.unsqueeze(2)).sum(dim=-1)

        x = base.norm(x + routed)
        logits = base.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss, routing_weights

    def get_routing_confidence(self, x: torch.Tensor) -> float:
        return self.router.get_routing_confidence(x)

    def get_q6_coords_stats(self, x: torch.Tensor) -> dict:
        """Статистика Q6 координат — показывает геометрическое распределение."""
        coords = self.router.get_q6_coords(x)  # (B, T, 6)
        avg = coords.mean(dim=(0, 1))
        return {name: round(avg[i].item(), 3) for i, name in enumerate(EXPERT_NAMES)}


def build_dataset():
    loader = CorpusLoader()
    corpus = loader.as_training_corpus(max_per_source=1000)
    texts = [d['text'] for d in corpus if len(d['text']) > 50]
    print(f"  Корпус: {len(texts)} текстов")
    return [list(t.encode('utf-8', errors='replace')) for t in texts]


def get_batch(encoded, block_size, batch_size, device):
    x_list, y_list = [], []
    for _ in range(batch_size):
        idx = torch.randint(len(encoded), (1,)).item()
        seq = encoded[idx]
        if len(seq) < block_size + 1:
            seq = seq + [0] * (block_size + 1 - len(seq))
        start = torch.randint(0, max(1, len(seq) - block_size), (1,)).item()
        chunk = seq[start: start + block_size + 1]
        x_list.append(chunk[:block_size])
        y_list.append(chunk[1: block_size + 1])
    return (
        torch.tensor(x_list, dtype=torch.long, device=device),
        torch.tensor(y_list, dtype=torch.long, device=device),
    )


def get_lr(step, cfg):
    if step < cfg['warmup_steps']:
        return cfg['lr'] * step / max(cfg['warmup_steps'], 1)
    progress = (step - cfg['warmup_steps']) / max(1, cfg['steps'] - cfg['warmup_steps'])
    return cfg['lr'] * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_baseline_ppl():
    path = Path('experiments') / 'lean_baseline_log.json'
    if path.exists():
        return json.loads(path.read_text()).get('final_ppl')
    return 1.51


def compute_routing_diversity(model, encoded, device, n_samples=20) -> float:
    """Cross-domain diversity без supervision — из самого routing.

    Берём тексты из разных частей корпуса и меряем std routing weights.
    Высокое значение = роутер органически дифференцирует домены.
    """
    model.eval()
    all_avg_weights = []
    step = max(1, len(encoded) // n_samples)
    for i in range(0, min(len(encoded), n_samples * step), step):
        seq = encoded[i]
        T = min(len(seq), model.block_size)
        x = torch.tensor([seq[:T]], dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, routing_weights = model(x)
        all_avg_weights.append(routing_weights.mean(dim=(0, 1)).cpu())

    if not all_avg_weights:
        return 0.0
    w_matrix = torch.stack(all_avg_weights)
    diversity = w_matrix.std(dim=0).mean().item()
    model.train()
    return diversity


def train(cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_ppl = load_baseline_ppl() or cfg['baseline_ppl']

    print(f"\n{'='*64}")
    print(f"  META HEXLEARN ROUTER (Неделя 6)")
    print(f"{'='*64}")
    print(f"  Устройство     : {device}")
    print(f"  Шаги           : {cfg['steps']}")
    print(f"  Baseline PPL   : {baseline_ppl:.4f}")
    print(f"  n_experts      : {cfg['n_experts']} (= 6 вершин Q6 unit-basis)")
    print(f"  routing_T      : {cfg['routing_temperature']}")
    print(f"  Цель           : routing_conf > {cfg['target_confidence']}")
    print(f"  vs Kasatkin    : 0.3450 (Week 3, для сравнения)")

    encoded = build_dataset()
    model = LeanWithHexLearnRouter(cfg).to(device)
    print(f"  Параметры      : {model.num_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], betas=(0.9, 0.95), weight_decay=0.1
    )

    log = {
        'config': cfg, 'baseline_ppl': baseline_ppl,
        'steps': [], 'losses': [], 'ppls': [],
        'routing_confidences': [],
        'diversity_history': [],
        'kasatkin_comparison': {'kasatkin_conf': 0.3450, 'kasatkin_ppl': 1.0355},
    }

    best_loss = float('inf')
    t0 = time.time()
    model.train()

    print(f"\n{'─'*64}")
    for step in range(1, cfg['steps'] + 1):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch(encoded, cfg['block_size'], cfg['batch_size'], device)
        logits, loss, routing_weights = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()

        loss_val = loss.item()
        ppl_val = math.exp(min(loss_val, 20))
        if loss_val < best_loss:
            best_loss = loss_val

        if step % cfg['log_every'] == 0 or step == 1:
            elapsed = time.time() - t0
            log['steps'].append(step)
            log['losses'].append(round(loss_val, 4))
            log['ppls'].append(round(ppl_val, 4))
            print(
                f"  step={step:4d}/{cfg['steps']} | "
                f"loss={loss_val:.4f} | ppl={ppl_val:.4f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

        if step % cfg['conf_every'] == 0:
            with torch.no_grad():
                base = model.base
                h = base.drop(
                    base.embed(x) + base.pos_embed(torch.arange(x.shape[1], device=device))
                )
                conf = model.get_routing_confidence(h)
                q6_stats = model.get_q6_coords_stats(h)

            log['routing_confidences'].append({'step': step, 'conf': round(conf, 4)})

            avg_weights = routing_weights.mean(dim=(0, 1))
            w_str = " ".join(f"{EXPERT_NAMES[i]}:{avg_weights[i]:.2f}" for i in range(6))
            q6_str = " ".join(f"{k}:{v:.2f}" for k, v in q6_stats.items())
            print(f"  [Routing step={step}] conf={conf:.4f} | {w_str}")
            print(f"  [Q6 coords  step={step}] {q6_str}")

            # Diversity
            div = compute_routing_diversity(model, encoded, device)
            log['diversity_history'].append({'step': step, 'diversity': round(div, 4)})
            print(f"  [Diversity  step={step}] {div:.4f}")

        if step % cfg['save_every'] == 0:
            ckpt = Path('experiments') / f'hexlearn_router_step{step}.pt'
            torch.save({'step': step, 'model_state': model.state_dict(),
                        'loss': loss_val, 'config': cfg}, ckpt)

    final_ppl = log['ppls'][-1] if log['ppls'] else float('inf')
    final_conf = (log['routing_confidences'][-1]['conf']
                  if log['routing_confidences'] else 0.0)
    final_div = (log['diversity_history'][-1]['diversity']
                 if log['diversity_history'] else 0.0)
    total_time = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  РЕЗУЛЬТАТ meta_hexlearn_router")
    print(f"{'='*64}")
    print(f"  Final PPL          : {final_ppl:.4f}")
    print(f"  Routing confidence : {final_conf:.4f} (цель: > {cfg['target_confidence']})")
    print(f"  Routing diversity  : {final_div:.4f}")
    print(f"  Baseline PPL       : {baseline_ppl:.4f}")
    print(f"  KasatkinQ6Router   : conf=0.3450, PPL=1.0355 (для сравнения)")

    verdict = "inconclusive"
    if cfg['steps'] < 3000:
        print(f"  ⚠  < 3000 шагов — вердикт недействителен")
    else:
        conf_ok = final_conf > cfg['target_confidence']
        ppl_ok = final_ppl <= baseline_ppl * 1.02
        if conf_ok and ppl_ok:
            print(f"  ✓  УСПЕХ: conf={final_conf:.4f} > {cfg['target_confidence']} И PPL в норме")
            verdict = "proven"
        elif not conf_ok:
            print(f"  ✗  Routing confidence недостаточна: {final_conf:.4f}")
            verdict = "disproven"
        else:
            print(f"  ✗  PPL хуже baseline: {final_ppl:.4f} > {baseline_ppl:.4f}")
            verdict = "disproven"

    final_ckpt = Path('experiments') / 'hexlearn_router_final.pt'
    torch.save({'step': cfg['steps'], 'model_state': model.state_dict(),
                'final_ppl': final_ppl, 'routing_confidence': final_conf,
                'routing_diversity': final_div,
                'verdict': verdict, 'config': cfg}, final_ckpt)

    log.update({
        'final_ppl': final_ppl,
        'routing_confidence': final_conf,
        'routing_diversity': final_div,
        'verdict': verdict,
        'total_time_s': round(total_time, 1),
    })
    log_path = Path('experiments') / 'hexlearn_router_log.json'
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"  Лог: {log_path}")

    return final_ppl, final_conf, verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.dry_run:
        cfg['steps'] = 50
        cfg['log_every'] = 10
        cfg['conf_every'] = 25
        cfg['save_every'] = 9999

    train(cfg)
