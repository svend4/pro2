#!/usr/bin/env python3
"""
v59 Full Benchmark: AbrialeBridge, AdaptiveBridge, SourceSpecializer + Ablation Study.

Модульный запуск — каждый конфиг отдельно, результаты сохраняются инкрементально.

Использование:
  # Запустить один конфиг:
  python benchmark_v59_full.py --run vanilla
  python benchmark_v59_full.py --run seven_bridge
  python benchmark_v59_full.py --run abriale_bridge
  python benchmark_v59_full.py --run ablation_no_heisenberg_attention

  # Запустить группу:
  python benchmark_v59_full.py --group arch        # 5 архитектур
  python benchmark_v59_full.py --group ablation    # 7 ablation
  python benchmark_v59_full.py --group all          # всё (как раньше)

  # Посмотреть текущие результаты:
  python benchmark_v59_full.py --summary

  # Список всех доступных конфигов:
  python benchmark_v59_full.py --list
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
    'benchmark_v59_full_results.json'
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

SOURCE_NAMES = [
    'use_heisenberg_attention',
    'use_palace_attention',
    'use_privileged_axis',
    'use_flower_gat',
    'use_cube_diagonal',
    'use_d4_equivariant',
    'use_dual_embedding',
]

SOURCE_LABELS = {
    'use_heisenberg_attention': 'Heisenberg (Belyaev)',
    'use_palace_attention': 'Palace (Skliarova)',
    'use_privileged_axis': 'PrivilegedAxis (Kasatkin)',
    'use_flower_gat': 'FlowerGAT (Belyaev)',
    'use_cube_diagonal': 'CubeDiagonal (Kasatkin)',
    'use_d4_equivariant': 'D4Equivariant (Fomyuk)',
    'use_dual_embedding': 'DualEmbedding (Kasatkin)',
}

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

ARCH_ORDER = ['vanilla', 'seven_bridge', 'abriale_bridge', 'adaptive_bridge', 'specializer']


def get_ablation_configs():
    """Generate ablation configs: bridge with 6/7 sources (one removed)."""
    configs = {}
    for src_flag in SOURCE_NAMES:
        sources = {k: v for k, v in ALL_SOURCES.items() if k != src_flag}
        sources['use_bridge_of_modules'] = True
        sources['bridge_mode'] = 'lightweight'
        name = f'ablation_no_{src_flag.replace("use_", "")}'
        configs[name] = sources
    return configs


def get_all_config_names():
    ablation = get_ablation_configs()
    return ARCH_ORDER + list(ablation.keys())


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


# ── Incremental results storage ──────────────────────────────

def load_results():
    """Load existing results from disk."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_result(name, result):
    """Save one config result incrementally (merge into existing file)."""
    data = load_results()
    data[name] = result
    data['_last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  >> Result for '{name}' saved to {RESULTS_PATH}")


def run_single(name, train_data, val_data, vocab_size):
    """Run a single config by name, save result immediately."""
    ablation_configs = get_ablation_configs()

    if name in ARCH_CONFIGS:
        overrides = ARCH_CONFIGS[name]
    elif name in ablation_configs:
        overrides = ablation_configs[name]
    else:
        print(f"  ERROR: Unknown config '{name}'")
        print(f"  Available: {', '.join(get_all_config_names())}")
        return None

    cfg = make_config(overrides, vocab_size)
    result = train_and_eval(cfg, name, train_data, val_data, n_steps=800)

    # Add ablation metadata
    if name.startswith('ablation_no_'):
        src_flag = 'use_' + name.replace('ablation_no_', '')
        if src_flag in SOURCE_LABELS:
            result['removed_source'] = SOURCE_LABELS[src_flag]
            result['removed_flag'] = src_flag

    # Category tag
    result['category'] = 'ablation' if name.startswith('ablation_') else 'architecture'

    save_result(name, result)
    return result


def print_summary():
    """Print summary table from saved results."""
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

    # Split by category
    arch = {k: v for k, v in results.items() if v.get('category') != 'ablation'}
    ablation = {k: v for k, v in results.items() if v.get('category') == 'ablation'}

    # Architecture comparison
    if arch:
        print("\n" + "=" * 90)
        print("  PART 1: Architecture Comparison")
        print("=" * 90)
        print(f"  {'Config':<22} {'Params':>10} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8}")
        print("  " + "-" * 86)

        vanilla_ppl = arch.get('vanilla', {}).get('best_ppl')

        for name in ARCH_ORDER:
            if name not in arch:
                print(f"  {name:<22} {'---':>10} {'(pending)':>10}")
                continue
            r = arch[name]
            delta_str = ""
            if vanilla_ppl is not None and name != 'vanilla':
                delta = r['best_ppl'] - vanilla_ppl
                sign = "+" if delta >= 0 else ""
                delta_str = f"  {sign}{delta:.2f}"
            print(f"  {name:<22} {r['params']:>10,} "
                  f"{r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} "
                  f"{r['time']:>7.1f}s{delta_str}")

        done = len(arch)
        total = len(ARCH_ORDER)
        print(f"\n  Progress: {done}/{total} architecture configs done")

    # Ablation study
    if ablation:
        print("\n" + "=" * 90)
        print("  PART 2: Ablation Study (7-source bridge, remove 1)")
        print("=" * 90)
        print(f"  {'Removed Source':<30} {'Best PPL':>10} {'Delta':>10} {'Impact':>10}")
        print("  " + "-" * 66)

        ref_ppl = arch.get('seven_bridge', {}).get('best_ppl')

        for src_flag in SOURCE_NAMES:
            abl_name = f'ablation_no_{src_flag.replace("use_", "")}'
            if abl_name not in ablation:
                label = SOURCE_LABELS.get(src_flag, src_flag)
                print(f"  {label:<30} {'---':>10} {'(pending)':>10}")
                continue
            r = ablation[abl_name]
            label = r.get('removed_source', abl_name)
            if ref_ppl is not None:
                delta = r['best_ppl'] - ref_ppl
                impact = ("CRITICAL" if delta > 0.3 else "HIGH" if delta > 0.1
                          else "MEDIUM" if delta > 0.05 else "LOW")
                print(f"  {label:<30} {r['best_ppl']:>10.2f} {delta:>+10.2f} {impact:>10}")
            else:
                print(f"  {label:<30} {r['best_ppl']:>10.2f} {'(no ref)':>10}")

        done = len(ablation)
        print(f"\n  Progress: {done}/{len(SOURCE_NAMES)} ablation configs done")

    elif arch:
        print(f"\n  Ablation: 0/{len(SOURCE_NAMES)} configs done (run --group ablation)")

    # Overall
    total_done = len(results)
    total_all = len(ARCH_ORDER) + len(SOURCE_NAMES)
    print(f"\n  Overall: {total_done}/{total_all} configs complete")


def main():
    parser = argparse.ArgumentParser(description='v59 Modular Benchmark')
    parser.add_argument('--run', type=str, help='Run a single config by name')
    parser.add_argument('--group', type=str, choices=['arch', 'ablation', 'all'],
                        help='Run a group of configs')
    parser.add_argument('--summary', action='store_true', help='Show current results')
    parser.add_argument('--list', action='store_true', help='List all available configs')
    parser.add_argument('--reset', action='store_true', help='Clear saved results')
    args = parser.parse_args()

    if args.list:
        print("\nAvailable configs:")
        print("\n  Architecture (--group arch):")
        for name in ARCH_ORDER:
            print(f"    {name}")
        print("\n  Ablation (--group ablation):")
        for name in get_ablation_configs():
            print(f"    {name}")
        return

    if args.summary:
        print_summary()
        return

    if args.reset:
        if os.path.exists(RESULTS_PATH):
            os.remove(RESULTS_PATH)
            print("  Results cleared.")
        return

    if not args.run and not args.group:
        parser.print_help()
        print("\nExamples:")
        print("  python benchmark_v59_full.py --run vanilla")
        print("  python benchmark_v59_full.py --group arch")
        print("  python benchmark_v59_full.py --summary")
        return

    # Load data once
    print("Loading data...")
    train_data, val_data, vocab_size = load_data(block_size=128)

    if args.run:
        run_single(args.run, train_data, val_data, vocab_size)
        print("\n" + "-" * 40)
        print_summary()
        return

    if args.group:
        if args.group == 'arch':
            names = ARCH_ORDER
        elif args.group == 'ablation':
            names = list(get_ablation_configs().keys())
        else:  # all
            names = ARCH_ORDER + list(get_ablation_configs().keys())

        # Skip already completed
        existing = load_results()
        meta_keys = {'_last_updated'}
        remaining = [n for n in names if n not in existing or n in meta_keys]

        if not remaining:
            print(f"  All {len(names)} configs already done!")
            print_summary()
            return

        print(f"\n  Running {len(remaining)}/{len(names)} configs "
              f"({len(names) - len(remaining)} already cached)...\n")

        for i, name in enumerate(remaining, 1):
            print(f"\n{'#'*60}")
            print(f"  [{i}/{len(remaining)}] Running: {name}")
            print(f"{'#'*60}")
            run_single(name, train_data, val_data, vocab_size)

        print("\n" + "=" * 60)
        print("  ALL DONE")
        print("=" * 60)
        print_summary()


if __name__ == '__main__':
    main()
