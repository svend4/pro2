#!/usr/bin/env python3
"""
v64 Combined Benchmark: Nautilus + ConvergenceBridge + GlyphPrior.

Tests whether combining NautilusHierarchy (geometric module ordering)
with ConvergenceBridge + GlyphPrior (Q6 inductive bias) yields better
results than each approach alone.

Configs:
  1. vanilla                — no geometry (baseline)
  2. all_flat               — all 7 modules via BridgeOfModules lightweight (v58)
  3. nautilus_seq_lean      — Nautilus sequential, heisenberg + flower_gat only
  4. v64_bridge_prior       — bridge_lightweight + convergence + glyph_prior (v64)
  5. nautilus_plus_bridge    — Nautilus seq all + convergence + glyph_prior
  6. nautilus_lean_plus_bridge — Nautilus seq lean + convergence + glyph_prior
  7. nautilus_par_plus_bridge — Nautilus parallel + convergence + glyph_prior
  8. nautilus_mid_plus_bridge — Nautilus seq mid + convergence + glyph_prior

Hypotheses:
  H1: nautilus + bridge < nautilus alone (bridge adds Q6 inductive bias)
  H2: nautilus + bridge < bridge alone (nautilus orders modules better)
  H3: Combined < all_flat (synergy beats flat application)

Usage:
  python benchmark_v64_combined.py --group all
  python benchmark_v64_combined.py --run nautilus_plus_bridge
  python benchmark_v64_combined.py --summary
"""

import sys
import os
import math
import time
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT


RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmark_v64_combined_results.json'
)


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

# ── Convergence Bridge + GlyphPrior flags ──────────────────────

BRIDGE_PRIOR = dict(
    use_convergence_bridge=True,
    use_glyph_prior=True,
)


# ── Configs ────────────────────────────────────────────────────

CONFIGS = {
    # Baselines (re-run for fair comparison on same hardware)
    'vanilla': dict(
        architecture_mode='standard',
        hex_strength=0.0,
    ),
    'all_flat': dict(
        **ALL_SOURCES,
        use_bridge_of_modules=True,
        bridge_mode='lightweight',
    ),
    'nautilus_seq_lean': dict(
        hex_strength=0.05,
        quantizer_type='factored6',
        use_heisenberg_attention=True,
        use_flower_gat=True,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='heisenberg,flower_gat',
        nautilus_warmup_steps=100,
    ),

    # v64 standalone: bridge_lightweight + convergence + glyph_prior
    'v64_bridge_prior': dict(
        **ALL_SOURCES,
        use_bridge_of_modules=True,
        bridge_mode='lightweight',
        **BRIDGE_PRIOR,
    ),

    # ── Combined: Nautilus + ConvergenceBridge + GlyphPrior ──

    # Nautilus sequential all + bridge + prior
    'nautilus_plus_bridge': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='all',
        nautilus_warmup_steps=200,
        **BRIDGE_PRIOR,
    ),
    # Nautilus lean + bridge + prior
    'nautilus_lean_plus_bridge': dict(
        hex_strength=0.05,
        quantizer_type='factored6',
        use_heisenberg_attention=True,
        use_flower_gat=True,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='heisenberg,flower_gat',
        nautilus_warmup_steps=100,
        **BRIDGE_PRIOR,
    ),
    # Nautilus parallel + bridge + prior
    'nautilus_par_plus_bridge': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='parallel',
        nautilus_chambers='all',
        nautilus_warmup_steps=200,
        **BRIDGE_PRIOR,
    ),
    # Nautilus mid (4 chambers) + bridge + prior
    'nautilus_mid_plus_bridge': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='d4_equivariant,palace,heisenberg,flower_gat',
        nautilus_warmup_steps=200,
        **BRIDGE_PRIOR,
    ),
}

CONFIG_ORDER = [
    'vanilla', 'all_flat', 'nautilus_seq_lean',
    'v64_bridge_prior',
    'nautilus_plus_bridge', 'nautilus_lean_plus_bridge',
    'nautilus_par_plus_bridge', 'nautilus_mid_plus_bridge',
]


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


def _get_module_info(model):
    """Detect module type for display."""
    parts = []
    if hasattr(model, 'nautilus'):
        n = len(model.nautilus.chambers)
        parts.append(f"Nautilus({model.nautilus.mode},{n}ch)")
    if hasattr(model, 'convergence_bridge'):
        parts.append("ConvBridge")
    if hasattr(model, 'glyph_prior_gate'):
        parts.append("GlyphPrior")
    for layer in model.core.layers:
        if hasattr(layer, 'bridge_of_modules'):
            parts.append("Bridge")
            break
    return '+'.join(parts) if parts else 'none'


def train_and_eval(cfg, name, train_data, val_data,
                   n_steps=800, batch_size=16, lr=1e-3, eval_every=200):
    set_seed(42)
    model = YiJingGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    module_type = _get_module_info(model)

    print(f"\n{'='*70}")
    print(f"  {name} ({n_params:,} params, type={module_type})")
    print(f"{'='*70}")

    history = []
    t0 = time.time()
    best_val = float('inf')

    for step in range(1, n_steps + 1):
        model.train()
        progress = step / n_steps
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        # Update nautilus curriculum step
        if hasattr(model, 'nautilus'):
            model.nautilus.set_step(step)

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
            # Nautilus stats
            if hasattr(model, 'nautilus'):
                nstats = model.nautilus.get_nautilus_stats()
                rg = nstats.get('nautilus/residual_gate', 0)
                gates = []
                for cname in model.nautilus.chamber_names:
                    g = nstats.get(f'nautilus/{cname}/gate', 0)
                    gates.append(f"{cname[:4]}={g:.2f}")
                extra += f" [Naut] rg={rg:.3f} {' '.join(gates)}"

            # Convergence bridge stats
            if hasattr(model, 'convergence_bridge'):
                cb = model.convergence_bridge
                bs = cb.bridge_scale.item()
                extra += f" [CB] s={bs:.3f}"

            # Glyph prior gate
            if hasattr(model, 'glyph_prior_gate'):
                gp = torch.sigmoid(model.glyph_prior_gate).item()
                extra += f" [GP] g={gp:.3f}"

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f}{extra}")
            history.append({'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl})

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")

    return {
        'name': name, 'params': n_params, 'module_type': module_type,
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
    }


# ── Results storage ──────────────────────────────────────────

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_result(name, result):
    data = load_results()
    data[name] = result
    data['_last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  >> Result for '{name}' saved to {RESULTS_PATH}")


def run_single(name, train_data, val_data, vocab_size):
    if name not in CONFIGS:
        print(f"  ERROR: Unknown config '{name}'")
        print(f"  Available: {', '.join(CONFIG_ORDER)}")
        return None

    cfg = make_config(CONFIGS[name], vocab_size)
    result = train_and_eval(cfg, name, train_data, val_data, n_steps=800)
    save_result(name, result)
    return result


def print_summary():
    data = load_results()
    if not data:
        print("  No results yet. Run some configs first.")
        return

    meta_keys = {'_last_updated'}
    results = {k: v for k, v in data.items() if k not in meta_keys}

    if not results:
        print("  No results yet.")
        return

    print(f"\n  Last updated: {data.get('_last_updated', '?')}")

    print("\n" + "=" * 115)
    print("  v64 COMBINED BENCHMARK: Nautilus + ConvergenceBridge + GlyphPrior")
    print("=" * 115)
    print(f"  {'Config':<28} {'Type':<35} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8} {'Delta':>8}")
    print("  " + "-" * 112)

    vanilla_ppl = results.get('vanilla', {}).get('best_ppl')

    for name in CONFIG_ORDER:
        if name not in results:
            print(f"  {name:<28} {'':>35} {'---':>8} {'(pending)':>10}")
            continue
        r = results[name]
        delta_str = ""
        if vanilla_ppl is not None and name != 'vanilla':
            delta = r['best_ppl'] - vanilla_ppl
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}"
        mt = r.get('module_type', 'none')
        print(f"  {name:<28} {mt:<35} {r['params']:>8,} "
              f"{r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} "
              f"{r['time']:>7.1f}s {delta_str:>8}")

    # Hypothesis testing
    print(f"\n  Hypothesis Testing:")
    vanilla = results.get('vanilla', {}).get('best_ppl')
    all_flat = results.get('all_flat', {}).get('best_ppl')
    v64 = results.get('v64_bridge_prior', {}).get('best_ppl')
    naut_bridge = results.get('nautilus_plus_bridge', {}).get('best_ppl')
    naut_lean = results.get('nautilus_seq_lean', {}).get('best_ppl')
    naut_lean_bridge = results.get('nautilus_lean_plus_bridge', {}).get('best_ppl')

    if naut_bridge is not None and v64 is not None:
        print(f"  H1 (nautilus+bridge < bridge alone): {'CONFIRMED' if naut_bridge < v64 else 'REJECTED'}"
              f"  ({naut_bridge:.2f} vs {v64:.2f})")

    if naut_bridge is not None and naut_lean is not None:
        # Compare nautilus+bridge vs nautilus alone (using lean as proxy)
        print(f"  H2 (nautilus+bridge < nautilus lean): {'CONFIRMED' if naut_bridge < naut_lean else 'REJECTED'}"
              f"  ({naut_bridge:.2f} vs {naut_lean:.2f})")

    if naut_bridge is not None and all_flat is not None:
        print(f"  H3 (combined < all_flat):            {'CONFIRMED' if naut_bridge < all_flat else 'REJECTED'}"
              f"  ({naut_bridge:.2f} vs {all_flat:.2f})")

    if naut_lean_bridge is not None and naut_lean is not None:
        print(f"  H4 (lean+bridge < lean alone):       {'CONFIRMED' if naut_lean_bridge < naut_lean else 'REJECTED'}"
              f"  ({naut_lean_bridge:.2f} vs {naut_lean:.2f})")

    done = sum(1 for n in CONFIG_ORDER if n in results)
    total = len(CONFIG_ORDER)
    print(f"\n  Progress: {done}/{total} configs done")


def main():
    parser = argparse.ArgumentParser(description='v64 Combined Benchmark')
    parser.add_argument('--run', type=str, help='Run single config')
    parser.add_argument('--group', type=str,
                        choices=['baselines', 'combined', 'all'],
                        help='Run group of configs')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    print("  Loading data...")
    train_data, val_data, vocab_size = load_data()

    if args.run:
        run_single(args.run, train_data, val_data, vocab_size)
    elif args.group:
        if args.group == 'baselines':
            names = ['vanilla', 'all_flat', 'nautilus_seq_lean', 'v64_bridge_prior']
        elif args.group == 'combined':
            names = ['nautilus_plus_bridge', 'nautilus_lean_plus_bridge',
                     'nautilus_par_plus_bridge', 'nautilus_mid_plus_bridge']
        else:  # all
            names = CONFIG_ORDER

        for name in names:
            run_single(name, train_data, val_data, vocab_size)

        print_summary()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
