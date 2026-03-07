#!/usr/bin/env python3
"""
v59 Full Benchmark: AbrialeBridge, AdaptiveBridge, SourceSpecializer + Ablation Study.

Конфигурации:

 ЧАСТЬ 1: Сравнение новых архитектур (все 7 источников)
  1. vanilla           — без геометрии (baseline)
  2. seven_bridge      — v58 LightweightBridge (reference)
  3. abriale_bridge    — v59 AbrialeBridge (гибрид Abriale + Bridge)
  4. adaptive_bridge   — v59 AdaptiveBridge (адаптивная глубина)
  5. specializer       — v59 SourceSpecializer (доменная специализация)

 ЧАСТЬ 2: Ablation Study — вклад каждого из 7 источников
  Для каждого из 7 источников: bridge с 6 из 7 (один убран)
  Показывает, насколько критичен каждый источник.
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
    use_heisenberg_attention=True,
    use_flower_gat=True,
    use_palace_attention=True,
    use_privileged_axis=True,
    use_cube_diagonal=True,
    use_dual_embedding=True,
    use_d4_equivariant=True,
)

SOURCE_NAMES = [
    'use_heisenberg_attention',
    'use_palace_attention',
    'use_privileged_axis',
    'use_flower_gat',
    'use_cube_diagonal',
    'use_d4_equivariant',
    'use_dual_embedding',
]

SOURCE_LABELS = [
    'Heisenberg (Belyaev)',
    'Palace (Skliarova)',
    'PrivilegedAxis (Kasatkin)',
    'FlowerGAT (Belyaev)',
    'CubeDiagonal (Kasatkin)',
    'D4Equivariant (Fomyuk)',
    'DualEmbedding (Kasatkin)',
]

# ── Configs ────────────────────────────────────────────────────

ARCH_CONFIGS = {
    'vanilla': dict(
        architecture_mode='standard',
        hex_strength=0.0,
    ),
    'seven_bridge': dict(
        **ALL_SOURCES,
        use_bridge_of_modules=True,
        bridge_mode='lightweight',
    ),
    'abriale_bridge': dict(
        **ALL_SOURCES,
        use_abriale_bridge=True,
        bridge_mode='lightweight',
        abriale_bridge_arity=2,
    ),
    'adaptive_bridge': dict(
        **ALL_SOURCES,
        use_adaptive_bridge=True,
        bridge_mode='lightweight',
    ),
    'specializer': dict(
        **ALL_SOURCES,
        use_source_specialization=True,
        n_domains=4,
    ),
}


def make_config(overrides, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=128):
    base = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, dropout=0.05,
        use_rope=True, use_swiglu=True, temp=0.3,
    )
    return YiJingConfig(**{**base, **overrides})


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

    # Detect bridge type
    bridge_type = 'none'
    for layer in model.core.layers:
        if hasattr(layer, 'bridge_of_modules'):
            bom = layer.bridge_of_modules
            bridge_type = type(bom).__name__
            break
        elif hasattr(layer, 'source_specializer'):
            bridge_type = 'SourceSpecializer'
            break

    print(f"\n{'='*60}")
    print(f"  {name} ({n_params:,} params, type={bridge_type})")
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

            extra = ""
            for layer in model.core.layers:
                if hasattr(layer, 'bridge_of_modules'):
                    s = layer.bridge_of_modules.get_bridge_stats()
                    extra = f" gate={s['global_gate']:.3f}"
                    if 'complexity' in s:
                        extra += f" cmplx={s['complexity']:.3f} lvls={s['active_levels']}"
                    if 'abriale_commit_rate' in s:
                        extra += f" commit={s['abriale_commit_rate']:.3f}"
                    break
                elif hasattr(layer, 'source_specializer'):
                    s = layer.source_specializer.get_specialization_stats()
                    if s.get('domain_probs'):
                        dp = [f"{p:.2f}" for p in s['domain_probs']]
                        extra = f" domains=[{','.join(dp)}]"
                    break

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f}{extra}")
            history.append({'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl})

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    # Collect final stats
    layer_stats = {}
    for layer in model.core.layers:
        if hasattr(layer, 'bridge_of_modules'):
            layer_stats = layer.bridge_of_modules.get_bridge_stats()
            break
        elif hasattr(layer, 'source_specializer'):
            layer_stats = layer.source_specializer.get_specialization_stats()
            break

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")

    return {
        'name': name, 'params': n_params,
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
        'layer_stats': layer_stats,
    }


def main():
    print("=" * 70)
    print("  v59 Full Benchmark: AbrialeBridge, AdaptiveBridge, Specializer + Ablation")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    train_data, val_data, vocab_size = load_data(block_size=128)

    # ── PART 1: Architecture comparison ──
    print(f"\n[2/3] Architecture comparison (5 configs × 800 steps)...")
    arch_results = []
    for name in ['vanilla', 'seven_bridge', 'abriale_bridge', 'adaptive_bridge', 'specializer']:
        cfg = make_config(ARCH_CONFIGS[name], vocab_size)
        result = train_and_eval(cfg, name, train_data, val_data, n_steps=800)
        arch_results.append(result)

    # ── PART 2: Ablation study ──
    print(f"\n[3/3] Ablation study: remove 1 source at a time (7 configs × 800 steps)...")
    ablation_results = []

    # Full 7-source bridge as reference
    full_ref = next(r for r in arch_results if r['name'] == 'seven_bridge')

    for i, (src_flag, src_label) in enumerate(zip(SOURCE_NAMES, SOURCE_LABELS)):
        # Copy all sources, remove one
        sources = {k: v for k, v in ALL_SOURCES.items() if k != src_flag}
        sources['use_bridge_of_modules'] = True
        sources['bridge_mode'] = 'lightweight'
        ablation_name = f'ablation_no_{src_flag.replace("use_", "")}'
        cfg = make_config(sources, vocab_size)
        result = train_and_eval(cfg, ablation_name, train_data, val_data, n_steps=800)
        result['removed_source'] = src_label
        result['removed_flag'] = src_flag
        ablation_results.append(result)

    # ── Summary ──
    print("\n" + "=" * 90)
    print("  PART 1: Architecture Comparison")
    print("=" * 90)
    print(f"{'Config':<22} {'Params':>10} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8}")
    print("-" * 90)
    vanilla_ppl = arch_results[0]['best_ppl']
    for r in arch_results:
        delta = r['best_ppl'] - vanilla_ppl
        sign = "+" if delta >= 0 else ""
        print(f"{r['name']:<22} {r['params']:>10,} "
              f"{r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} {r['time']:>7.1f}s  {sign}{delta:.2f}")

    print("\n" + "=" * 90)
    print("  PART 2: Ablation Study (7-source bridge, remove 1)")
    print("=" * 90)
    print(f"{'Removed Source':<30} {'Best PPL':>10} {'Delta':>10} {'Impact':>10}")
    print("-" * 90)
    ref_ppl = full_ref['best_ppl']
    for r in ablation_results:
        delta = r['best_ppl'] - ref_ppl
        impact = "CRITICAL" if delta > 0.3 else "HIGH" if delta > 0.1 else "MEDIUM" if delta > 0.05 else "LOW"
        print(f"{r['removed_source']:<30} {r['best_ppl']:>10.2f} {delta:>+10.2f} {impact:>10}")

    # ── Save ──
    output = {
        'architecture_comparison': arch_results,
        'ablation_study': ablation_results,
        'reference_full_bridge_ppl': ref_ppl,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'benchmark_v59_full_results.json'
    )
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
