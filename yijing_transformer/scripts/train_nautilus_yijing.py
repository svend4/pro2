"""
train_nautilus_yijing.py — обучение интегрированной модели NautilusYiJing.

Объединяет YiJingGPT и NautilusMoME через три варианта интеграции:
  1. Q6GeometricRouter  — геометрический роутер (Хэмминг вместо linear)
  2. YiJingCoreBlock    — YiJing-слои как ядро (вместо TransformerBlock)
  3. YiJingMicroExpert  — YiJing-эксперты с якорями гексаграмм

Использование:
    python scripts/train_nautilus_yijing.py --steps 500
    python scripts/train_nautilus_yijing.py --steps 2000 --text-file my_corpus.txt
    python scripts/train_nautilus_yijing.py --steps 500 --compare   # vs baseline
"""

import sys
import os
import math
import time
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.nautilus_yijing import NautilusYiJing, NautilusYiJingConfig
from tokenizer.char_tokenizer import CharTokenizer


# ─── Синтетический корпус (тот же что в train_text_demo.py) ───────────────────
SYNTHETIC_TEXT = """
The ancient Book of Changes, known as the I Ching or Yi Jing, is one of the oldest
Chinese classical texts. It describes a system of cosmology and philosophy based on
the interplay of yin and yang, represented by broken and unbroken lines.

Each hexagram consists of six lines, either yin (broken) or yang (unbroken).
The 64 hexagrams represent all possible combinations of these six lines.
This creates a natural binary encoding: each hexagram is a vertex of the
six-dimensional hypercube {-1, +1}^6.

The eight trigrams (bagua) are the building blocks: Heaven, Earth, Water, Fire,
Thunder, Wind, Mountain, and Lake. Each trigram has three lines and represents
a fundamental force of nature.

A hexagram is composed of two trigrams: lower (inner) and upper (outer).
This factorization is key to our transformer architecture: instead of computing
distances to all 64 hexagrams, we can factorize into two independent softmax
operations over 8 trigrams each.

Mathematics is the language of structure. Code is the language of process.
Human experience is the language of meaning. System thinking is the language of flow.
Recognition is the language of pattern. Information is the language of exchange.
Synthesis is the language of transformation.

The geometric router assigns each token to the nearest expert based on
Hamming distance in the Q6 hypercube. MATH tokens cluster near vertex 63
(all yang), HUMAN tokens near vertex 0 (all yin), reflecting the
fundamental polarity of structure versus reception.
""".strip()


def make_batches(data: torch.Tensor, block_size: int, batch_size: int):
    """Генератор батчей из токенизированного текста."""
    n = len(data) - block_size - 1

    def get_batch():
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x, y

    return get_batch


def cosine_lr(step: int, total: int, lr_max: float, warmup: int = 100) -> float:
    if step < warmup:
        return lr_max * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr_max * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def generate(model: NautilusYiJing, tokenizer: CharTokenizer,
             prompt: str, max_tokens: int = 150,
             temperature: float = 0.8, top_k: int = 40) -> str:
    """Генерирует продолжение текста."""
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    if not ids:
        return ''
    ctx = torch.tensor([ids], dtype=torch.long, device=device)

    generated = []
    for _ in range(max_tokens):
        logits, _, _ = model(ctx[:, -model.cfg.block_size:])
        logits = logits[0, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, nxt.unsqueeze(0)], dim=1)
        generated.append(nxt.item())

    return prompt + tokenizer.decode(generated)


def print_routing_stats(info: dict, expert_names: list):
    """Выводит статистику геометрического роутинга."""
    ew = info['expert_weights']  # (B, T, n_experts)
    usage = ew.mean(dim=(0, 1)).tolist()
    q6 = info['q6_coords']       # (B, T, 6)
    q6_mean = q6.mean(dim=(0, 1)).tolist()

    print("  Routing (expert usage):")
    for i, (name, u) in enumerate(zip(expert_names, usage)):
        bar = '█' * int(u * 30)
        print(f"    {name:8s} [{bar:<30s}] {u:.3f}")
    print(f"  Q6 coords mean: [{', '.join(f'{v:+.2f}' for v in q6_mean)}]")
    if info.get('synth'):
        s = info['synth']
        print(f"  SYNTH: entropy={s.get('avg_entropy', 0):.3f}, "
              f"active={s.get('synth_frac', 0):.3f}")


def train(args):
    device = torch.device('cpu')

    # ─── Корпус ───────────────────────────────────────────────────────────
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text):,} chars from {args.text_file}")
    else:
        text = SYNTHETIC_TEXT
        print(f"Using synthetic text ({len(text):,} chars)")

    tokenizer = CharTokenizer()
    ids = tokenizer.encode(text)
    print(f"Vocabulary size: {tokenizer.get_piece_size()}")
    print(f"Tokens: {len(ids):,}")

    data = torch.tensor(ids, dtype=torch.long, device=device)
    get_batch = make_batches(data, block_size=128, batch_size=args.batch_size)

    # ─── Модель NautilusYiJing ─────────────────────────────────────────────
    cfg = NautilusYiJingConfig(
        vocab_size=tokenizer.get_piece_size(),
        d_model=args.d_model,
        n_layers=4,
        n_heads=6,
        block_size=128,
        d_expert=args.d_model // 2,
        n_experts=6,
        top_k=2,
        dropout=0.05,
        use_rope=True,
        use_bian_gua=True,
        enable_synth=True,
    )
    model = NautilusYiJing(cfg).to(device)

    total_params, yijing_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"NautilusYiJing — интегрированная модель")
    print(f"{'='*60}")
    print(f"Параметров всего:    {total_params:>10,}")
    print(f"YiJing-компоненты:   {yijing_params:>10,} ({100*yijing_params/total_params:.1f}%)")
    print(f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
    print(f"n_experts={cfg.n_experts}, top_k={cfg.top_k}")
    print(f"{'='*60}")
    print(f"\nАрхитектура интеграции:")
    print(f"  Вариант 1: Q6GeometricRouter    — роутинг по расстоянию Хэмминга")
    print(f"  Вариант 2: YiJingCoreBlock ×{cfg.n_layers} — геометрическое ядро")
    print(f"  Вариант 3: YiJingMicroExpert ×{cfg.n_experts}  — эксперты с Q6-якорями")
    print()

    # ─── Baseline для сравнения (опционально) ─────────────────────────────
    baseline = None
    if args.compare:
        from models.baseline import VanillaGPT
        from config.config import YiJingConfig
        b_cfg = YiJingConfig(
            vocab_size=tokenizer.get_piece_size(),
            d_model=args.d_model, n_layers=4, n_heads=6, block_size=128,
        )
        baseline = VanillaGPT(b_cfg).to(device)
        b_params = sum(p.numel() for p in baseline.parameters())
        print(f"Baseline VanillaGPT: {b_params:,} параметров")
        b_opt = torch.optim.AdamW(baseline.parameters(), lr=3e-3, weight_decay=0.1)
        print()

    # ─── Оптимизатор ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95)
    )

    # ─── Обучение ─────────────────────────────────────────────────────────
    print(f"Обучение {args.steps} шагов...")
    t0 = time.time()
    last_info = {}

    for step in range(1, args.steps + 1):
        x, y = get_batch()

        # LR schedule
        lr = cosine_lr(step, args.steps, lr_max=3e-3, warmup=min(100, args.steps // 5))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        logits, loss, info = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_info = info

        # Baseline (параллельно)
        if baseline is not None:
            b_lr = cosine_lr(step, args.steps, lr_max=3e-3, warmup=min(100, args.steps // 5))
            for pg in b_opt.param_groups:
                pg['lr'] = b_lr
            b_out = baseline(x, y)
            b_loss = b_out[1] if isinstance(b_out, tuple) else b_out
            b_opt.zero_grad()
            b_loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            b_opt.step()

        if step % 50 == 0 or step == args.steps:
            elapsed = time.time() - t0
            speed = step / elapsed
            loss_val = loss.item()

            if baseline is not None:
                print(f"  Step {step:4d}/{args.steps}: "
                      f"NautilusYiJing loss={loss_val:.4f} | "
                      f"Baseline loss={b_loss.item():.4f} | "
                      f"{speed:.1f} steps/s")
            else:
                print(f"  Step {step:4d}/{args.steps}: "
                      f"loss={loss_val:.4f}, lr={lr:.6f}, {speed:.1f} steps/s")

    elapsed = time.time() - t0
    print(f"\nОбучение завершено: {args.steps} шагов за {elapsed:.1f}s")

    # ─── Статистика роутинга ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Геометрический роутинг (Q6-статистика):")
    print_routing_stats(last_info, list(model.experts.keys()))

    # Q6-якоря экспертов
    print(f"\nQ6-якоря экспертов (гексаграммы):")
    for i, name in enumerate(model.experts.keys()):
        anchor = model.expert_q6_anchors[i].tolist()
        anchor_str = ''.join(['☰' if v > 0 else '☷' for v in anchor])
        print(f"  {name:8s}: [{', '.join(f'{v:+.0f}' for v in anchor)}]  {anchor_str}")

    # ─── Генерация текста ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Генерация текста:")
    print(f"{'='*60}")
    prompts = ["The ", "Each hexagram ", "Mathematics is "]
    for p in prompts:
        gen = generate(model, tokenizer, p, max_tokens=120, temperature=0.8)
        print(f"\n>>> {p}")
        print(gen)
        print("---")

    # ─── Сохранение чекпоинта ─────────────────────────────────────────────
    if args.save:
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f'checkpoints/nautilus_yijing_step{args.steps}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'cfg': cfg,
            'step': args.steps,
            'loss': loss.item(),
        }, ckpt_path)
        print(f"\nЧекпоинт сохранён: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description='Train NautilusYiJing')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--d-model', type=int, default=192)
    parser.add_argument('--text-file', type=str, default=None)
    parser.add_argument('--compare', action='store_true',
                        help='Сравнить с VanillaGPT baseline')
    parser.add_argument('--save', action='store_true',
                        help='Сохранить чекпоинт после обучения')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
