#!/usr/bin/env python3
"""
v63 Benchmark: NautilusHierarchy — иерархическое упорядочивание vs Lean Model vs Bridge.

Тестирует гипотезу: «каждый сверчок знай свой шесток» лучше, чем:
  - Удаление «вредных» модулей (Lean = только Heisenberg + FlowerGAT)
  - Плоское применение всех модулей (v58 Bridge)
  - Без геометрии вообще (vanilla)

Конфигурации:
  1. vanilla              — без геометрии
  2. all_flat             — все 7 модулей через BridgeOfModules (v58)
  3. lean_only            — только Heisenberg + FlowerGAT (без вредных)
  4. nautilus_seq_all     — Nautilus sequential, все 7 камер
  5. nautilus_par_all     — Nautilus parallel, все 7 камер
  6. nautilus_seq_5       — Nautilus sequential, 5 камер (без cube_diagonal, privileged_axis)
  7. nautilus_seq_lean    — Nautilus sequential, только heisenberg + flower_gat
  8. nautilus_seq_mid     — Nautilus sequential, d4 + palace + heisenberg + flower_gat
  9. nautilus_seq_warmup  — Nautilus sequential, все 7, длинный warmup (600 steps)
  10. nautilus_par_mid    — Nautilus parallel, d4 + palace + heisenberg + flower_gat

Гипотезы:
  H1: nautilus_seq_all < all_flat (иерархия снижает интерференцию)
  H2: nautilus_seq_all < lean_only (вредные модули полезны на своём месте)
  H3: sequential < parallel (каскад эффективнее параллельного merge)
  H4: nautilus_seq_mid ≈ nautilus_seq_all (4 камеры достаточно)
  H5: длинный warmup помогает стабильности

Использование:
  python benchmark_v63_nautilus.py --group all
  python benchmark_v63_nautilus.py --run nautilus_seq_all
  python benchmark_v63_nautilus.py --summary
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
    'benchmark_v63_nautilus_results.json'
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


# ── Configs ────────────────────────────────────────────────────

CONFIGS = {
    # Baselines
    'vanilla': dict(
        architecture_mode='standard',
        hex_strength=0.0,
    ),
    'all_flat': dict(
        **ALL_SOURCES,
        use_bridge_of_modules=True,
        bridge_mode='lightweight',
    ),
    'lean_only': dict(
        hex_strength=0.05,
        quantizer_type='factored6',
        use_heisenberg_attention=True,
        use_flower_gat=True,
    ),

    # v63: Nautilus sequential variants
    'nautilus_seq_all': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='all',
        nautilus_warmup_steps=200,
    ),
    'nautilus_par_all': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='parallel',
        nautilus_chambers='all',
        nautilus_warmup_steps=200,
    ),
    'nautilus_seq_5': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='dual_embedding,d4_equivariant,palace,heisenberg,flower_gat',
        nautilus_warmup_steps=200,
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
    'nautilus_seq_mid': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='d4_equivariant,palace,heisenberg,flower_gat',
        nautilus_warmup_steps=200,
    ),
    'nautilus_seq_warmup': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_chambers='all',
        nautilus_warmup_steps=600,
    ),
    'nautilus_par_mid': dict(
        **ALL_SOURCES,
        use_nautilus=True,
        nautilus_mode='parallel',
        nautilus_chambers='d4_equivariant,palace,heisenberg,flower_gat',
        nautilus_warmup_steps=200,
    ),
}

CONFIG_ORDER = [
    'vanilla', 'all_flat', 'lean_only',
    'nautilus_seq_all', 'nautilus_par_all',
    'nautilus_seq_5', 'nautilus_seq_lean',
    'nautilus_seq_mid', 'nautilus_seq_warmup',
    'nautilus_par_mid',
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


def _get_nautilus_stats(model):
    """Extract nautilus hierarchy stats if available."""
    if hasattr(model, 'nautilus'):
        return model.nautilus.get_nautilus_stats()
    return None


def _get_nautilus_chamber_details(model):
    """Get per-chamber gate and scale values."""
    if not hasattr(model, 'nautilus'):
        return None
    details = {}
    for name, chamber in zip(model.nautilus.chamber_names, model.nautilus.chambers):
        s = chamber.get_stats()
        details[name] = {'gate': s['gate_mean'], 'scale': s['scale']}
    details['residual_gate'] = model.nautilus.residual_gate.item()
    details['mode'] = model.nautilus.mode
    return details


def _get_bridge_type(model):
    """Detect module type for display."""
    if hasattr(model, 'nautilus'):
        n = len(model.nautilus.chambers)
        return f"Nautilus({model.nautilus.mode},{n}ch)"
    for layer in model.core.layers:
        if hasattr(layer, 'bridge_of_modules'):
            return type(layer.bridge_of_modules).__name__
    return 'none'


def train_and_eval(cfg, name, train_data, val_data,
                   n_steps=800, batch_size=16, lr=1e-3, eval_every=200):
    set_seed(42)
    model = YiJingGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    module_type = _get_bridge_type(model)

    print(f"\n{'='*70}")
    print(f"  {name} ({n_params:,} params, type={module_type})")
    print(f"{'='*70}")

    history = []
    nautilus_history = []
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
            nautilus_stats = _get_nautilus_stats(model)
            if nautilus_stats:
                # Show chamber gate means
                gates = []
                for cname in model.nautilus.chamber_names:
                    g = nautilus_stats.get(f'nautilus/{cname}/gate', 0)
                    gates.append(f"{cname[:4]}={g:.2f}")
                rg = nautilus_stats.get('nautilus/residual_gate', 0)
                extra = f" [Nautilus] rg={rg:.3f} {' '.join(gates)}"
                nautilus_history.append({'step': step, **nautilus_stats})
            else:
                for layer in model.core.layers:
                    if hasattr(layer, 'bridge_of_modules'):
                        s = layer.bridge_of_modules.get_bridge_stats()
                        extra = f" gate={s['global_gate']:.3f}"
                        break

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f}{extra}")
            history.append({'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl})

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    # Final diagnostics
    final_nautilus = _get_nautilus_chamber_details(model)

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")
    if final_nautilus:
        print(f"  Nautilus ({final_nautilus['mode']}):")
        print(f"    residual_gate={final_nautilus['residual_gate']:.4f}")
        for cname, cvals in final_nautilus.items():
            if isinstance(cvals, dict):
                print(f"    {cname}: gate={cvals['gate']:.4f}, scale={cvals['scale']:.4f}")

    return {
        'name': name, 'params': n_params, 'module_type': module_type,
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
        'nautilus_stats': final_nautilus,
        'nautilus_history': nautilus_history,
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
    print("  v63 BENCHMARK: NautilusHierarchy — «Каждый сверчок знай свой шесток»")
    print("=" * 115)
    print(f"  {'Config':<24} {'Type':<28} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8} {'Delta':>8}")
    print("  " + "-" * 110)

    vanilla_ppl = results.get('vanilla', {}).get('best_ppl')

    for name in CONFIG_ORDER:
        if name not in results:
            print(f"  {name:<24} {'':>28} {'---':>8} {'(pending)':>10}")
            continue
        r = results[name]
        delta_str = ""
        if vanilla_ppl is not None and name != 'vanilla':
            delta = r['best_ppl'] - vanilla_ppl
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}"
        mt = r.get('module_type', 'none')
        print(f"  {name:<24} {mt:<28} {r['params']:>8,} "
              f"{r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} "
              f"{r['time']:>7.1f}s {delta_str:>8}")

    # Nautilus chamber diagnostics
    nautilus_configs = [n for n in CONFIG_ORDER if n in results
                       and results[n].get('nautilus_stats')]
    if nautilus_configs:
        print(f"\n  Nautilus Chamber Diagnostics (final gate / scale):")
        # Collect all chamber names
        all_chambers = set()
        for n in nautilus_configs:
            ns = results[n].get('nautilus_stats', {})
            for k, v in ns.items():
                if isinstance(v, dict):
                    all_chambers.add(k)
        chamber_names = sorted(all_chambers)

        header = f"  {'Config':<24} {'RGate':>6}"
        for cn in chamber_names:
            header += f" {cn[:8]:>10}"
        print(header)
        print("  " + "-" * (30 + 10 * len(chamber_names)))

        for name in nautilus_configs:
            ns = results[name].get('nautilus_stats', {})
            rg = ns.get('residual_gate', 0)
            line = f"  {name:<24} {rg:>6.3f}"
            for cn in chamber_names:
                cv = ns.get(cn, {})
                if isinstance(cv, dict):
                    g = cv.get('gate', 0)
                    s = cv.get('scale', 0)
                    line += f" {g:.2f}/{s:.3f}"
                else:
                    line += f" {'---':>10}"
            print(line)

    # Hypothesis testing
    print(f"\n  Hypothesis Testing:")
    vanilla = results.get('vanilla', {}).get('best_ppl')
    all_flat = results.get('all_flat', {}).get('best_ppl')
    lean = results.get('lean_only', {}).get('best_ppl')
    naut_seq = results.get('nautilus_seq_all', {}).get('best_ppl')
    naut_par = results.get('nautilus_par_all', {}).get('best_ppl')
    naut_mid = results.get('nautilus_seq_mid', {}).get('best_ppl')
    naut_warmup = results.get('nautilus_seq_warmup', {}).get('best_ppl')

    if all_flat is not None and naut_seq is not None:
        print(f"  H1 (nautilus < flat bridge): {'CONFIRMED' if naut_seq < all_flat else 'REJECTED'}"
              f"  ({naut_seq:.2f} vs {all_flat:.2f})")

    if lean is not None and naut_seq is not None:
        print(f"  H2 (nautilus > lean):        {'CONFIRMED' if naut_seq < lean else 'REJECTED'}"
              f"  ({naut_seq:.2f} vs {lean:.2f})")

    if naut_seq is not None and naut_par is not None:
        print(f"  H3 (sequential < parallel):  {'CONFIRMED' if naut_seq < naut_par else 'REJECTED'}"
              f"  ({naut_seq:.2f} vs {naut_par:.2f})")

    if naut_seq is not None and naut_mid is not None:
        diff = abs(naut_seq - naut_mid)
        print(f"  H4 (mid ≈ all):              {'CONFIRMED' if diff < 0.3 else 'REJECTED'}"
              f"  (diff={diff:.2f}, {naut_mid:.2f} vs {naut_seq:.2f})")

    if naut_seq is not None and naut_warmup is not None:
        print(f"  H5 (long warmup helps):      {'CONFIRMED' if naut_warmup < naut_seq else 'REJECTED'}"
              f"  ({naut_warmup:.2f} vs {naut_seq:.2f})")

    done = sum(1 for n in CONFIG_ORDER if n in results)
    total = len(CONFIG_ORDER)
    print(f"\n  Progress: {done}/{total} configs done")


def main():
    parser = argparse.ArgumentParser(description='v63 NautilusHierarchy Benchmark')
    parser.add_argument('--run', type=str, help='Run single config')
    parser.add_argument('--group', type=str,
                        choices=['baselines', 'nautilus', 'all'],
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
            names = ['vanilla', 'all_flat', 'lean_only']
        elif args.group == 'nautilus':
            names = ['nautilus_seq_all', 'nautilus_par_all',
                     'nautilus_seq_5', 'nautilus_seq_lean',
                     'nautilus_seq_mid', 'nautilus_seq_warmup',
                     'nautilus_par_mid']
        else:  # all
            names = CONFIG_ORDER

        for name in names:
            run_single(name, train_data, val_data, vocab_size)

        print_summary()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
