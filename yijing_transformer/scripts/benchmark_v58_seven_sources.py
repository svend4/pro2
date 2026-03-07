#!/usr/bin/env python3
"""
v58 Seven-Source Benchmark: полный набор из 7 геометрических источников.

Это стресс-тест интерференции — ключевая проблема из ablation_six_sources.
Все 7 обогатительных модулей включены одновременно:

  1. HeisenbergAttention (Беляев)
  2. PalaceAttention (Склярова)
  3. PrivilegedAxisAttention (Касаткин)
  4. FlowerOfLifeGAT (Беляев)
  5. CubeDiagonalAttention (Касаткин)
  6. D4EquivariantLayer (Фомюк)
  7. DualEmbedding (Касаткин)

Конфигурации:
  1. vanilla         — без геометрии (baseline)
  2. seven_fixed     — все 7 с фиксированными коэффициентами (ожидается интерференция)
  3. seven_router    — MoE top-k router (v54)
  4. seven_bridge    — Bridge of Modules lightweight (v58)
"""

import sys
import os
import math
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def load_data(block_size=128, seed=42):
    random.seed(seed)
    words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "it", "was",
        "for", "on", "are", "with", "as", "his", "they", "be", "at", "one",
        "have", "this", "from", "by", "not", "but", "what", "all", "were", "when",
        "we", "there", "can", "an", "your", "which", "their", "said", "each", "she",
        "do", "how", "will", "up", "other", "about", "out", "many", "then", "them",
        "would", "like", "so", "these", "her", "long", "make", "thing", "see", "him",
        "two", "has", "look", "more", "day", "could", "go", "come", "did", "my",
        "no", "most", "who", "over", "know", "than", "call", "first", "people", "may",
        "down", "been", "now", "find", "any", "new", "work", "part", "take", "get",
        "place", "made", "after", "back", "only", "use", "where", "good", "very", "still",
    ]
    lines = []
    for _ in range(60000):
        n = random.randint(5, 25)
        line = ' '.join(random.choice(words) for _ in range(n))
        lines.append(line + '.')
    full = '\n'.join(lines)
    split = int(len(full) * 0.9)
    train_text, val_text = full[:split], full[split:]

    train_data = torch.tensor(list(train_text.encode('utf-8')), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode('utf-8')), dtype=torch.long)
    print(f"  Train: {len(train_data):,}, Val: {len(val_data):,} tokens (byte-level)")
    return train_data, val_data, 256


def get_batch(data, block_size, batch_size):
    n = len(data) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# ── All 7 enrichment sources ──────────────────────────────────

ALL_SOURCES = dict(
    hex_strength=0.05,
    quantizer_type='factored6',
    # Belyaev:
    use_heisenberg_attention=True,
    use_flower_gat=True,
    # Skliarova:
    use_palace_attention=True,
    # Kasatkin:
    use_privileged_axis=True,
    use_cube_diagonal=True,
    use_dual_embedding=True,
    # Fomyuk:
    use_d4_equivariant=True,
)


CONFIGS = {
    'vanilla': dict(
        architecture_mode='standard',
        hex_strength=0.0,
    ),
    'seven_fixed': dict(
        **ALL_SOURCES,
    ),
    'seven_router': dict(
        **ALL_SOURCES,
        use_source_router=True,
        source_router_top_k=3,
    ),
    'seven_bridge': dict(
        **ALL_SOURCES,
        use_bridge_of_modules=True,
        bridge_mode='lightweight',
    ),
}


def make_config(name, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=128):
    base = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, dropout=0.05,
        use_rope=True, use_swiglu=True, temp=0.3,
    )
    return YiJingConfig(**{**base, **CONFIGS[name]})


@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, n_eval=30):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, block_size, batch_size)
        logits, loss, _ = model(x, targets=y)
        if isinstance(loss, torch.Tensor):
            losses.append(loss.item())
        else:
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))


def train_and_eval(cfg, name, train_data, val_data,
                   n_steps=800, batch_size=16, lr=1e-3, eval_every=200):
    set_seed(42)
    model = YiJingGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    n_sources = 0
    for layer in model.core.layers:
        if hasattr(layer, '_enrichment_sources'):
            n_sources = len(layer._enrichment_sources)
            break

    print(f"\n{'='*60}")
    print(f"  {name} ({n_params:,} params, {n_sources} enrichment sources)")
    print(f"{'='*60}")

    history = []
    t0 = time.time()
    best_val = float('inf')

    for step in range(1, n_steps + 1):
        model.train()
        progress = step / n_steps
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, cfg.block_size, batch_size)
        logits, loss, _ = model(x, targets=y)
        if not isinstance(loss, torch.Tensor):
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            vl, ppl = evaluate(model, val_data, cfg.block_size, batch_size)
            best_val = min(best_val, vl)

            # Bridge stats
            extra = ""
            for layer in model.core.layers:
                if hasattr(layer, 'bridge_of_modules'):
                    s = layer.bridge_of_modules.get_bridge_stats()
                    extra = f" gate={s['global_gate']:.3f}"
                    break
                elif hasattr(layer, 'source_router'):
                    s = layer.source_router.get_routing_stats()
                    probs = [f"{s.get(f'source_{i}', 0):.2f}" for i in range(n_sources)]
                    extra = f" routing=[{','.join(probs)}]"
                    break

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f}{extra}")
            history.append({'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl})

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")

    return {
        'name': name, 'params': n_params, 'n_sources': n_sources,
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
    }


def main():
    print("=" * 60)
    print("  v58 Seven-Source Benchmark")
    print("  All 7 enrichment sources — interference stress test")
    print("=" * 60)

    print("\n[1/2] Loading data...")
    train_data, val_data, vocab_size = load_data(block_size=128)

    configs_to_run = ['vanilla', 'seven_fixed', 'seven_router', 'seven_bridge']
    results = []

    print(f"\n[2/2] Training {len(configs_to_run)} configurations (800 steps each)...")
    for name in configs_to_run:
        cfg = make_config(name, vocab_size)
        result = train_and_eval(cfg, name, train_data, val_data, n_steps=800)
        results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY: 7-Source Anti-Interference Comparison")
    print("=" * 80)
    print(f"{'Config':<20} {'Params':>10} {'Sources':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8}")
    print("-" * 80)
    vanilla_ppl = results[0]['best_ppl']
    for r in results:
        delta = r['best_ppl'] - vanilla_ppl
        sign = "+" if delta > 0 else ""
        print(f"{r['name']:<20} {r['params']:>10,} {r['n_sources']:>8} "
              f"{r['best_ppl']:>10.1f} {r['final_ppl']:>10.1f} {r['time']:>7.1f}s  {sign}{delta:>.1f}")

    # Save
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'benchmark_v58_seven_sources_results.json'
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
