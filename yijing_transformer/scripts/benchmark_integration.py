"""
benchmark_integration.py — сравнение четырёх моделей по единым метрикам.

Модели:
  1. VanillaGPT      — стандартный трансформер (baseline)
  2. YiJingGPT       — YiJing с Q6-квантизатором, без экспертов
  3. NautilusMoME    — MoE с обученным роутером, стандартные блоки
  4. NautilusYiJing  — полная интеграция (наш результат)

Метрики:
  - final_loss          — финальный train loss
  - convergence_auc     — площадь под кривой loss (меньше = быстрее сходится)
  - bigram_diversity    — разнообразие биграмм в генерации [0..1]
  - routing_entropy     — энтропия роутинга по экспертам (выше = равномернее)
  - params_M            — число параметров в миллионах
  - steps_per_sec       — скорость обучения

Использование:
    python scripts/benchmark_integration.py
    python scripts/benchmark_integration.py --steps 300
"""

import sys
import os
import math
import time
import json
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tokenizer.char_tokenizer import CharTokenizer


# ─── Корпус ───────────────────────────────────────────────────────────────────
CORPUS = """
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

Mathematics is the language of structure. Code is the language of process.
Human experience is the language of meaning. System thinking is the language of flow.
Recognition is the language of pattern. Information is the language of exchange.
Synthesis is the language of transformation.

A hexagram is composed of two trigrams: lower (inner) and upper (outer).
This factorization is key to our transformer architecture: instead of computing
distances to all 64 hexagrams, we can factorize into two independent softmax
operations over 8 trigrams each.

The geometric router assigns each token to the nearest expert based on
Hamming distance in the Q6 hypercube. MATH tokens cluster near vertex 63
all yang, HUMAN tokens near vertex 0 all yin, reflecting the fundamental
polarity of structure versus reception in the geometry of knowledge.
""".strip()


def make_batches(data, block_size, batch_size, device):
    n = len(data) - block_size - 1

    def get_batch():
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    return get_batch


def cosine_lr(step, total, lr_max=3e-3, warmup=50):
    if step < warmup:
        return lr_max * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr_max * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def bigram_diversity(model_fn, tokenizer, device, n_samples=5, max_tokens=80) -> float:
    """Разнообразие биграмм в генерации: уникальные биграммы / всего биграмм."""
    prompts = ["The ", "Each ", "Math", "Code", "Human"]
    all_bigrams = []
    for p in prompts:
        ids = tokenizer.encode(p)
        ctx = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = model_fn(ctx)          # (1, T, V)
            logits = logits[0, -1, :] / 0.8
            v, _ = torch.topk(logits, 40)
            logits[logits < v[-1]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ctx = torch.cat([ctx, nxt.unsqueeze(0)], dim=1)
        tokens = ctx[0, len(ids):].tolist()
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        all_bigrams.extend(bigrams)
    if not all_bigrams:
        return 0.0
    return len(set(all_bigrams)) / len(all_bigrams)


def train_model(model, get_batch, steps, device) -> Dict:
    """Обучает модель, возвращает метрики."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.1, betas=(0.9, 0.95)
    )
    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        lr = cosine_lr(step, steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch()
        out = model(x, y)

        # Унифицируем формат возврата
        if isinstance(out, tuple):
            logits, loss = out[0], out[1]
        else:
            loss = out

        if loss is None:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    elapsed = time.time() - t0
    auc = sum(losses) / len(losses) if losses else 99.0

    return {
        'final_loss': losses[-1] if losses else 99.0,
        'convergence_auc': auc,
        'steps_per_sec': steps / elapsed,
        'loss_curve': losses[::max(1, len(losses)//20)],  # прореженная кривая
    }


def get_routing_entropy(model, get_batch, model_name: str) -> float:
    """Энтропия роутинга: насколько равномерно загружены эксперты."""
    model.eval()
    entropies = []
    with torch.no_grad():
        for _ in range(10):
            x, _ = get_batch()
            out = model(x)
            if isinstance(out, tuple) and len(out) >= 3:
                info = out[2]
                if isinstance(info, dict):
                    # NautilusYiJing
                    if 'expert_weights' in info:
                        ew = info['expert_weights']  # (B, T, n_experts)
                        usage = ew.mean(dim=(0, 1))  # (n_experts,)
                        usage = usage.clamp(min=1e-8)
                        h = -(usage * usage.log()).sum().item()
                        entropies.append(h)
                    # NautilusMoME
                    elif 'routing' in info:
                        ew = info['routing']
                        usage = ew.mean(dim=(0, 1)).clamp(min=1e-8)
                        h = -(usage * usage.log()).sum().item()
                        entropies.append(h)
    model.train()
    return sum(entropies) / len(entropies) if entropies else 0.0


# ─── Создание моделей ─────────────────────────────────────────────────────────

def make_vanilla(vocab_size, d_model, block_size):
    from models.baseline import VanillaGPT
    from config.config import YiJingConfig
    cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=4,
        n_heads=6, block_size=block_size,
    )
    return VanillaGPT(cfg)


def make_yijing(vocab_size, d_model, block_size):
    from models.model import YiJingGPT
    from config.config import YiJingConfig
    cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=4,
        n_heads=6, block_size=block_size,
        use_rope=True, use_swiglu=True, use_bian_gua=True,
        quantizer_type='factored6',
    )
    return YiJingGPT(cfg)


def make_nautilus_mome(vocab_size, d_model, block_size):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_nautilus_mome import NautilusMoME
    return NautilusMoME(
        vocab_size=vocab_size, d_model=d_model, n_layers=4,
        n_heads=6, block_size=block_size,
        d_expert=d_model // 2, n_experts=6, top_k=2,
    )


def make_nautilus_yijing(vocab_size, d_model, block_size):
    from models.nautilus_yijing import NautilusYiJing, NautilusYiJingConfig
    cfg = NautilusYiJingConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=4,
        n_heads=6, block_size=block_size,
        d_expert=d_model // 2, n_experts=6, top_k=2,
    )
    return NautilusYiJing(cfg)


# ─── Главная функция ─────────────────────────────────────────────────────────

def run_benchmark(steps: int = 300, d_model: int = 192, block_size: int = 128,
                  batch_size: int = 8):
    device = torch.device('cpu')
    tokenizer = CharTokenizer()
    ids = tokenizer.encode(CORPUS)
    data = torch.tensor(ids, dtype=torch.long, device=device)
    vocab_size = tokenizer.get_piece_size()
    get_batch = make_batches(data, block_size, batch_size, device)

    print(f"\n{'='*65}")
    print(f"  BENCHMARK: NautilusYiJing vs остальные")
    print(f"{'='*65}")
    print(f"  Шагов: {steps} | d_model: {d_model} | vocab: {vocab_size} | corpus: {len(ids)} токенов")
    print(f"{'='*65}\n")

    models_def = [
        ('VanillaGPT',     lambda: make_vanilla(vocab_size, d_model, block_size)),
        ('YiJingGPT',      lambda: make_yijing(vocab_size, d_model, block_size)),
        ('NautilusMoME',   lambda: make_nautilus_mome(vocab_size, d_model, block_size)),
        ('NautilusYiJing', lambda: make_nautilus_yijing(vocab_size, d_model, block_size)),
    ]

    all_results = {}

    for name, factory in models_def:
        print(f"[{name}]", flush=True)
        model = factory().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Параметров: {params:,}")

        # Обучение
        metrics = train_model(model, get_batch, steps, device)
        print(f"  Loss:  {metrics['loss_curve'][0]:.4f} → {metrics['final_loss']:.4f}")
        print(f"  AUC:   {metrics['convergence_auc']:.4f}")
        print(f"  Speed: {metrics['steps_per_sec']:.1f} steps/s")

        # Routing entropy (только для моделей с экспертами)
        routing_h = 0.0
        if name in ('NautilusMoME', 'NautilusYiJing'):
            routing_h = get_routing_entropy(model, get_batch, name)
            print(f"  Routing entropy: {routing_h:.4f}  (max={math.log(6):.4f})")

        # Bigram diversity
        model.eval()
        def model_fn(ctx):
            out = model(ctx)
            logits = out[0] if isinstance(out, tuple) else out
            return logits
        div = bigram_diversity(model_fn, tokenizer, device)
        print(f"  Bigram diversity: {div:.4f}")
        print()

        all_results[name] = {
            'params': params,
            'final_loss': metrics['final_loss'],
            'convergence_auc': metrics['convergence_auc'],
            'steps_per_sec': metrics['steps_per_sec'],
            'routing_entropy': routing_h,
            'bigram_diversity': div,
            'loss_curve': metrics['loss_curve'],
        }

    # ─── Сводная таблица ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ИТОГОВАЯ ТАБЛИЦА ({steps} шагов)")
    print(f"{'='*65}")
    header = f"  {'Модель':<18} {'Params':>8} {'Loss↓':>8} {'AUC↓':>8} {'Div↑':>8} {'RouteH↑':>8}"
    print(header)
    print(f"  {'-'*62}")

    # Лучшие значения для подсветки
    best_loss = min(v['final_loss'] for v in all_results.values())
    best_auc  = min(v['convergence_auc'] for v in all_results.values())
    best_div  = max(v['bigram_diversity'] for v in all_results.values())
    best_rh   = max(v['routing_entropy'] for v in all_results.values())

    for name, r in all_results.items():
        star_loss = '*' if abs(r['final_loss']       - best_loss) < 1e-4 else ' '
        star_auc  = '*' if abs(r['convergence_auc']  - best_auc)  < 1e-4 else ' '
        star_div  = '*' if abs(r['bigram_diversity']  - best_div)  < 1e-4 else ' '
        star_rh   = '*' if r['routing_entropy'] > 0 and abs(r['routing_entropy'] - best_rh) < 1e-4 else ' '
        print(f"  {name:<18} {r['params']/1e6:>7.2f}M "
              f"{r['final_loss']:>7.4f}{star_loss} "
              f"{r['convergence_auc']:>7.4f}{star_auc} "
              f"{r['bigram_diversity']:>7.4f}{star_div} "
              f"{r['routing_entropy']:>7.4f}{star_rh}")

    print(f"\n  * = лучший результат в колонке")
    print(f"  AUC = площадь под кривой loss (меньше = быстрее сходится)")
    print(f"  RouteH = энтропия роутинга (выше = эксперты загружены равномернее)")

    # ─── Кривые loss ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Кривые loss (прореженные до 20 точек):")
    print(f"{'='*65}")
    for name, r in all_results.items():
        curve = r['loss_curve']
        bars = []
        for v in curve:
            h = max(1, min(8, int(v * 3)))
            bars.append('▇' * h if v > 0.5 else '▄' * h if v > 0.1 else '▁')
        print(f"  {name:<18} {''.join(bars)}")

    # ─── Сохранение ──────────────────────────────────────────────────────
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'benchmark_integration_results.json'
    )
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != 'loss_curve'}
                 for k, v in all_results.items()}
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Результаты сохранены: {os.path.basename(out_path)}")
    print(f"{'='*65}\n")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',      type=int, default=300)
    parser.add_argument('--d-model',    type=int, default=192)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    run_benchmark(
        steps=args.steps,
        d_model=args.d_model,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )
