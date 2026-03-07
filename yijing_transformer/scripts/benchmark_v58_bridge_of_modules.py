#!/usr/bin/env python3
"""
v58 Benchmark: Bridge of Modules — иерархическая медиация между источниками.

Сравнение трёх стратегий антиинтерференции:

  1. direct_mix     — прямая линейная смесь всех источников (baseline, v53)
  2. source_router  — MoE-style top-k маршрутизация (v54)
  3. source_mixer   — per-source sigmoid gates (v54)
  4. bridge_of_mods — попарная cross-attention медиация (v58, новый)

Задача: synthetic multi-pattern — смесь XOR + periodic + modular паттернов.
Это стресс-тест: каждый паттерн выигрывает от разных геометрических
источников. Прямое смешивание создаёт интерференцию.

Метрики:
  - val loss (ниже = лучше)
  - per-pattern accuracy (какие паттерны решает каждая стратегия)
  - gate/bridge statistics (что включается/выключается)
"""

import sys
import os
import math
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from yijing_transformer.models.geometry.routing import (
    GeometricSourceRouter,
    GeometricSourceMixer,
    BridgeOfModules,
)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


# ── Synthetic multi-source model ────────────────────────────────


class FakeGeometricSource(nn.Module):
    """Имитирует один геометрический источник (lightweight для бенчмарка)."""
    def __init__(self, d_model: int, source_type: str = 'generic'):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.source_type = source_type
        # Каждый источник инициализирован по-разному
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x):
        return self.norm(self.proj(x))


class MultiSourceModel(nn.Module):
    """Модель с несколькими геометрическими источниками и стратегией смешивания.

    Архитектура:
        Input → Embedding → [Source₁, ..., Sourceₙ] → Strategy → Head → Logits
    """
    def __init__(self, vocab_size: int, d_model: int, n_sources: int,
                 strategy: str = 'direct_mix', router_top_k: int = 2,
                 bridge_n_heads: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.sources = nn.ModuleList([
            FakeGeometricSource(d_model, source_type=f'source_{i}')
            for i in range(n_sources)
        ])
        self.strategy_name = strategy

        if strategy == 'direct_mix':
            # Простое среднее (baseline)
            self.mixer = None
        elif strategy == 'source_router':
            self.mixer = GeometricSourceRouter(
                d_model=d_model, n_sources=n_sources, top_k=router_top_k
            )
        elif strategy == 'source_mixer':
            self.mixer = GeometricSourceMixer(
                d_model=d_model, n_sources=n_sources
            )
        elif strategy == 'bridge_of_mods':
            self.mixer = BridgeOfModules(
                d_model=d_model, n_sources=n_sources, n_heads=bridge_n_heads
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.head = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, targets=None):
        emb = self.embedding(x)  # (B, T, d)

        # Каждый источник обрабатывает вход
        source_outputs = [src(emb) for src in self.sources]

        # Стратегия смешивания
        if self.strategy_name == 'direct_mix':
            # Простое среднее всех источников
            mixed = torch.stack(source_outputs, dim=0).mean(dim=0)
            out = emb + 0.1 * mixed
        elif self.strategy_name == 'source_router':
            out = emb + 0.1 * self.mixer(emb, source_outputs)
        elif self.strategy_name == 'source_mixer':
            out = self.mixer(emb, source_outputs)
        elif self.strategy_name == 'bridge_of_mods':
            out = self.mixer(emb, source_outputs)

        logits = self.head(self.norm(out))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# ── Data: multi-pattern synthetic ──────────────────────────────


def generate_multi_pattern_data(n_samples=5000, seq_len=32, vocab_size=64, seed=42):
    """Генерирует данные со смесью паттернов.

    Каждая последовательность — один из 4 типов:
    - xor: y[i] = x[i-1] XOR x[i-2]
    - periodic: y[i] = (x[i-1] + period) mod vocab_size
    - modular: y[i] = (x[i-1] * x[i-2]) mod vocab_size
    - echo: y[i] = x[i-lag]

    Разные паттерны требуют разных геометрических источников.
    """
    set_seed(seed)
    data = []
    labels = []  # pattern type per sample

    for _ in range(n_samples):
        pattern = random.choice(['xor', 'periodic', 'modular', 'echo'])
        seq = [random.randint(0, vocab_size - 1) for _ in range(3)]

        if pattern == 'xor':
            for i in range(3, seq_len):
                seq.append((seq[-1] ^ seq[-2]) % vocab_size)
        elif pattern == 'periodic':
            period = random.randint(2, 8)
            for i in range(3, seq_len):
                seq.append((seq[-1] + period) % vocab_size)
        elif pattern == 'modular':
            for i in range(3, seq_len):
                seq.append((seq[-1] * seq[-2]) % vocab_size)
        elif pattern == 'echo':
            lag = random.randint(1, 4)
            for i in range(3, seq_len):
                seq.append(seq[max(0, len(seq) - lag)])

        data.append(seq[:seq_len])
        labels.append(pattern)

    return torch.tensor(data, dtype=torch.long), labels


def split_data(data, labels, val_frac=0.2):
    n_val = int(len(data) * val_frac)
    perm = torch.randperm(len(data))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (data[train_idx], [labels[i] for i in train_idx],
            data[val_idx], [labels[i] for i in val_idx])


def get_batch(data, batch_size):
    idx = torch.randint(0, len(data), (batch_size,))
    batch = data[idx]
    return batch[:, :-1], batch[:, 1:]


# ── Training & Evaluation ──────────────────────────────────────


@torch.no_grad()
def evaluate(model, val_data, batch_size=32, n_eval=20):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, batch_size)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def per_pattern_accuracy(model, val_data, val_labels, vocab_size):
    """Оценивает accuracy для каждого типа паттерна."""
    model.eval()
    pattern_correct = {}
    pattern_total = {}

    for i in range(len(val_data)):
        x = val_data[i:i+1, :-1]
        y = val_data[i:i+1, 1:]
        logits, _ = model(x)
        preds = logits.argmax(dim=-1)
        # Accuracy на последних 10 позициях (где паттерн установился)
        correct = (preds[:, -10:] == y[:, -10:]).float().mean().item()
        pattern = val_labels[i]
        pattern_correct[pattern] = pattern_correct.get(pattern, 0) + correct
        pattern_total[pattern] = pattern_total.get(pattern, 0) + 1

    results = {}
    for p in sorted(pattern_correct.keys()):
        results[p] = pattern_correct[p] / max(pattern_total[p], 1)
    return results


def train_and_eval(strategy, train_data, val_data, val_labels,
                   vocab_size=64, d_model=64, n_sources=6,
                   n_steps=1500, batch_size=32, lr=3e-3):
    set_seed(42)
    model = MultiSourceModel(
        vocab_size=vocab_size, d_model=d_model, n_sources=n_sources,
        strategy=strategy, router_top_k=2, bridge_n_heads=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {strategy} ({n_params:,} params)")
    print(f"{'='*60}")

    t0 = time.time()
    best_val = float('inf')
    history = []

    for step in range(1, n_steps + 1):
        model.train()
        # Cosine LR
        progress = step / n_steps
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, batch_size)
        _, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 300 == 0 or step == 1:
            vl = evaluate(model, val_data, batch_size)
            best_val = min(best_val, vl)

            # Collect strategy-specific stats
            extra = ""
            if strategy == 'source_router' and model.mixer is not None:
                stats = model.mixer.get_routing_stats()
                probs = [f"{stats.get(f'source_{i}', 0):.2f}" for i in range(n_sources)]
                extra = f" routing=[{','.join(probs)}]"
            elif strategy == 'source_mixer' and model.mixer is not None:
                stats = model.mixer.get_gate_stats()
                gates = stats.get('gates', [])
                extra = f" gates=[{','.join(f'{g:.2f}' for g in gates)}]"
            elif strategy == 'bridge_of_mods' and model.mixer is not None:
                stats = model.mixer.get_bridge_stats()
                extra = f" global_gate={stats['global_gate']:.3f}"
                scales = [f"{s['scale']:.3f}" for s in stats['bridge_scales']]
                extra += f" scales=[{','.join(scales)}]"

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f}{extra}")
            history.append({'step': step, 'train_loss': loss.item(), 'val_loss': vl})

    elapsed = time.time() - t0
    final_vl = evaluate(model, val_data, batch_size)
    per_pattern = per_pattern_accuracy(model, val_data, val_labels, vocab_size)

    print(f"\n  FINAL: val_loss={final_vl:.4f} best_val={best_val:.4f} time={elapsed:.1f}s")
    print(f"  Per-pattern accuracy:")
    for p, acc in per_pattern.items():
        print(f"    {p:12s}: {acc:.4f}")

    # Final stats
    mixer_stats = {}
    if model.mixer is not None:
        if hasattr(model.mixer, 'get_routing_stats'):
            mixer_stats = model.mixer.get_routing_stats()
        elif hasattr(model.mixer, 'get_gate_stats'):
            mixer_stats = model.mixer.get_gate_stats()
        elif hasattr(model.mixer, 'get_bridge_stats'):
            mixer_stats = model.mixer.get_bridge_stats()

    return {
        'strategy': strategy,
        'params': n_params,
        'final_val_loss': final_vl,
        'best_val_loss': best_val,
        'per_pattern': per_pattern,
        'time': elapsed,
        'history': history,
        'mixer_stats': mixer_stats,
    }


# ── Main ───────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  v58 Benchmark: Bridge of Modules")
    print("  Иерархическая медиация vs линейное смешивание")
    print("=" * 60)

    vocab_size = 64
    seq_len = 32
    d_model = 64
    n_sources = 6

    print("\n[1] Generating multi-pattern data...")
    data, labels = generate_multi_pattern_data(
        n_samples=5000, seq_len=seq_len, vocab_size=vocab_size
    )
    train_data, train_labels, val_data, val_labels = split_data(data, labels)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)} sequences")

    # Count patterns
    from collections import Counter
    print(f"  Patterns: {dict(Counter(val_labels))}")

    strategies = ['direct_mix', 'source_router', 'source_mixer', 'bridge_of_mods']
    results = {}

    for strategy in strategies:
        print(f"\n[{strategies.index(strategy)+2}] Training {strategy}...")
        results[strategy] = train_and_eval(
            strategy, train_data, val_data, val_labels,
            vocab_size=vocab_size, d_model=d_model, n_sources=n_sources,
            n_steps=1500, batch_size=32, lr=3e-3,
        )

    # ── Summary ────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  SUMMARY: Anti-Interference Strategy Comparison")
    print("=" * 70)
    print(f"{'Strategy':<20s} {'Params':>8s} {'Val Loss':>10s} {'Best':>10s} {'Time':>8s}")
    print("-" * 70)
    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy:<20s} {r['params']:>8,d} {r['final_val_loss']:>10.4f} "
              f"{r['best_val_loss']:>10.4f} {r['time']:>7.1f}s")

    print(f"\n{'Per-Pattern Accuracy':}")
    patterns = sorted(set(val_labels))
    header = f"{'Strategy':<20s}" + "".join(f"{p:>12s}" for p in patterns)
    print(header)
    print("-" * len(header))
    for strategy in strategies:
        pp = results[strategy]['per_pattern']
        row = f"{strategy:<20s}" + "".join(f"{pp.get(p, 0):>12.4f}" for p in patterns)
        print(row)

    # ── Winner analysis ────────────────────────────────────────

    print("\n  Winner Analysis:")
    for p in patterns:
        best_strat = max(strategies, key=lambda s: results[s]['per_pattern'].get(p, 0))
        best_acc = results[best_strat]['per_pattern'].get(p, 0)
        print(f"    {p:12s}: {best_strat} ({best_acc:.4f})")

    overall_best = min(strategies, key=lambda s: results[s]['best_val_loss'])
    print(f"\n  Overall best val loss: {overall_best} "
          f"({results[overall_best]['best_val_loss']:.4f})")

    # ── Save results ───────────────────────────────────────────

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'benchmark_v58_bridge_results.json'
    )
    # Clean up non-serializable items
    for s in strategies:
        if 'mixer_stats' in results[s]:
            stats = results[s]['mixer_stats']
            for k, v in list(stats.items()):
                if isinstance(v, torch.Tensor):
                    stats[k] = v.tolist()

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
