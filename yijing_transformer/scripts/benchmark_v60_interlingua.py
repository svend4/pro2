#!/usr/bin/env python3
"""
v60 Benchmark: Archetypal Interlingua vs BridgeOfModules vs AbrialeBridge.

Тестирует предсказания P1–P5 из archetypal-interlingua-theory.md:
  P1: Interlingua масштабируется лучше Bridge при N > 4
  P2: Архетипы коррелируют с 64 гексаграммами Q6
  P3: Тернарные 0 концентрируются на "сложных" токенах
  P4: При ternary=False → деградация к TokenAbstractor-like
  P5: Diminishing returns при uncertainty_budget → 1

Использование:
  python benchmark_v60_interlingua.py --group arch      # 6 архитектур
  python benchmark_v60_interlingua.py --group ablation   # 4 ablation
  python benchmark_v60_interlingua.py --group all        # всё
  python benchmark_v60_interlingua.py --run interlingua_ternary
  python benchmark_v60_interlingua.py --summary
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
    'benchmark_v60_interlingua_results.json'
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
    'interlingua_ternary': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    'interlingua_binary': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=False,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    'interlingua_full_ternary': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.8,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
}

ARCH_ORDER = [
    'vanilla', 'seven_bridge', 'abriale_bridge',
    'interlingua_ternary', 'interlingua_binary', 'interlingua_full_ternary',
]

ABLATION_CONFIGS = {
    'interlingua_32_archetypes': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=32,
    ),
    'interlingua_128_archetypes': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=128,
    ),
    'interlingua_2heads': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.3,
        interlingua_n_heads=2,
    ),
    'interlingua_uncertainty_0': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.0,
    ),
}

ABLATION_ORDER = list(ABLATION_CONFIGS.keys())


def get_all_config_names():
    return ARCH_ORDER + ABLATION_ORDER


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


def _get_interlingua_stats(model):
    """Extract ArchetypalInterlingua stats from model layers."""
    for layer in model.core.layers:
        if hasattr(layer, 'archetypal_interlingua'):
            il = layer.archetypal_interlingua
            stats = il.get_interlingua_stats()
            stats['q6_correlation'] = il.archetype_q6_correlation().item()
            return stats
    return None


def _get_bridge_type(model):
    """Detect bridge/interlingua type for display."""
    for layer in model.core.layers:
        if hasattr(layer, 'archetypal_interlingua'):
            return 'ArchetypalInterlingua'
        if hasattr(layer, 'bridge_of_modules'):
            return type(layer.bridge_of_modules).__name__
        if hasattr(layer, 'source_specializer'):
            return 'SourceSpecializer'
    return 'none'


def train_and_eval(cfg, name, train_data, val_data,
                   n_steps=800, batch_size=16, lr=1e-3, eval_every=200):
    set_seed(42)
    model = YiJingGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    bridge_type = _get_bridge_type(model)

    print(f"\n{'='*60}")
    print(f"  {name} ({n_params:,} params, type={bridge_type})")
    print(f"{'='*60}")

    history = []
    interlingua_history = []
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
            il_stats = _get_interlingua_stats(model)
            if il_stats:
                gate = il_stats['global_gate']
                trit = il_stats.get('trit_distribution', {})
                active = il_stats.get('active_archetypes', '?')
                q6_corr = il_stats.get('q6_correlation', 0)
                extra = (f" gate={gate:.3f} active={active}"
                         f" trits=[+{trit.get('pos',0):.2f}/0:{trit.get('zero',0):.2f}/-{trit.get('neg',0):.2f}]"
                         f" q6={q6_corr:.3f}")
                # Temperature annealing stats
                if 'ternary_temperature' in il_stats:
                    extra += f" temp={il_stats['ternary_temperature']:.3f}"
                interlingua_history.append({
                    'step': step, **il_stats,
                })
            else:
                for layer in model.core.layers:
                    if hasattr(layer, 'bridge_of_modules'):
                        s = layer.bridge_of_modules.get_bridge_stats()
                        extra = f" gate={s['global_gate']:.3f}"
                        if 'abriale_commit_rate' in s:
                            extra += f" commit={s['abriale_commit_rate']:.3f}"
                        break

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f}{extra}")
            history.append({'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl})

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    # Final stats
    final_il_stats = _get_interlingua_stats(model) or {}

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")
    if final_il_stats:
        print(f"  Interlingua: gate={final_il_stats['global_gate']:.3f}"
              f" active={final_il_stats.get('active_archetypes', '?')}"
              f" q6_corr={final_il_stats.get('q6_correlation', 0):.3f}"
              f" trits={final_il_stats.get('trit_distribution', {})}")

    return {
        'name': name, 'params': n_params, 'bridge_type': bridge_type,
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
        'interlingua_stats': final_il_stats,
        'interlingua_history': interlingua_history,
    }


# ── Incremental results storage ──────────────────────────────

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
    all_configs = {**ARCH_CONFIGS, **ABLATION_CONFIGS}

    if name not in all_configs:
        print(f"  ERROR: Unknown config '{name}'")
        print(f"  Available: {', '.join(get_all_config_names())}")
        return None

    cfg = make_config(all_configs[name], vocab_size)
    result = train_and_eval(cfg, name, train_data, val_data, n_steps=800)
    result['category'] = 'ablation' if name in ABLATION_CONFIGS else 'architecture'
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

    arch = {k: v for k, v in results.items() if v.get('category') != 'ablation'}
    ablation = {k: v for k, v in results.items() if v.get('category') == 'ablation'}

    # Architecture comparison
    if arch:
        print("\n" + "=" * 100)
        print("  PART 1: Architecture Comparison (v60 Interlingua)")
        print("=" * 100)
        print(f"  {'Config':<28} {'Type':<22} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8}")
        print("  " + "-" * 96)

        vanilla_ppl = arch.get('vanilla', {}).get('best_ppl')

        for name in ARCH_ORDER:
            if name not in arch:
                print(f"  {name:<28} {'':>22} {'---':>8} {'(pending)':>10}")
                continue
            r = arch[name]
            delta_str = ""
            if vanilla_ppl is not None and name != 'vanilla':
                delta = r['best_ppl'] - vanilla_ppl
                sign = "+" if delta >= 0 else ""
                delta_str = f"  {sign}{delta:.2f}"
            bt = r.get('bridge_type', 'none')
            print(f"  {name:<28} {bt:<22} {r['params']:>8,} "
                  f"{r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} "
                  f"{r['time']:>7.1f}s{delta_str}")

        # Interlingua-specific stats
        il_configs = [n for n in ARCH_ORDER if 'interlingua' in n and n in arch]
        if il_configs:
            print(f"\n  Interlingua Diagnostics:")
            print(f"  {'Config':<28} {'Gate':>6} {'Active':>8} {'Q6 Corr':>9} {'Trits(+/0/-)':>16}")
            print("  " + "-" * 73)
            for name in il_configs:
                r = arch[name]
                il = r.get('interlingua_stats', {})
                gate = il.get('global_gate', 0)
                active = il.get('active_archetypes', '?')
                q6 = il.get('q6_correlation', 0)
                trit = il.get('trit_distribution', {})
                trit_str = f"{trit.get('pos',0):.2f}/{trit.get('zero',0):.2f}/{trit.get('neg',0):.2f}"
                print(f"  {name:<28} {gate:>6.3f} {str(active):>8} {q6:>9.3f} {trit_str:>16}")

        done = len(arch)
        total = len(ARCH_ORDER)
        print(f"\n  Progress: {done}/{total} architecture configs done")

    # Ablation study
    if ablation:
        print("\n" + "=" * 100)
        print("  PART 2: Interlingua Ablation Study")
        print("=" * 100)
        print(f"  {'Config':<30} {'Best PPL':>10} {'Delta':>10} {'Active':>8} {'Q6 Corr':>9}")
        print("  " + "-" * 73)

        ref_ppl = arch.get('interlingua_ternary', {}).get('best_ppl')

        for name in ABLATION_ORDER:
            if name not in ablation:
                print(f"  {name:<30} {'---':>10} {'(pending)':>10}")
                continue
            r = ablation[name]
            il = r.get('interlingua_stats', {})
            delta_str = ""
            if ref_ppl is not None:
                delta = r['best_ppl'] - ref_ppl
                delta_str = f"{delta:>+10.2f}"
            else:
                delta_str = f"{'(no ref)':>10}"
            active = il.get('active_archetypes', '?')
            q6 = il.get('q6_correlation', 0)
            print(f"  {name:<30} {r['best_ppl']:>10.2f} {delta_str} {str(active):>8} {q6:>9.3f}")

        done = len(ablation)
        print(f"\n  Progress: {done}/{len(ABLATION_ORDER)} ablation configs done")

    # Predictions validation
    print("\n" + "=" * 100)
    print("  PREDICTION VALIDATION (archetypal-interlingua-theory.md)")
    print("=" * 100)

    p1 = "?"
    il_tern = arch.get('interlingua_ternary', {})
    ab_bridge = arch.get('abriale_bridge', {})
    if il_tern and ab_bridge:
        if il_tern.get('best_ppl', 999) < ab_bridge.get('best_ppl', 999):
            p1 = "CONFIRMED"
        else:
            p1 = f"NOT YET (IL={il_tern.get('best_ppl','?'):.2f} vs AB={ab_bridge.get('best_ppl','?'):.2f})"
    print(f"  P1 (Interlingua < Bridge):     {p1}")

    p2 = "?"
    if il_tern.get('interlingua_stats', {}).get('q6_correlation'):
        q6 = il_tern['interlingua_stats']['q6_correlation']
        p2 = f"{'CONFIRMED' if q6 > 0.3 else 'PARTIAL'} (corr={q6:.3f})"
    print(f"  P2 (Q6 correlation > 0.3):     {p2}")

    p4 = "?"
    il_bin = arch.get('interlingua_binary', {})
    if il_tern and il_bin:
        tern_ppl = il_tern.get('best_ppl', 999)
        bin_ppl = il_bin.get('best_ppl', 999)
        if tern_ppl < bin_ppl:
            p4 = f"CONFIRMED (ternary={tern_ppl:.2f} < binary={bin_ppl:.2f})"
        else:
            p4 = f"REVERSED (ternary={tern_ppl:.2f} >= binary={bin_ppl:.2f})"
    print(f"  P4 (Ternary > Binary):         {p4}")

    p5 = "?"
    il_full = arch.get('interlingua_full_ternary', {})
    if il_tern and il_full:
        tern_ppl = il_tern.get('best_ppl', 999)
        full_ppl = il_full.get('best_ppl', 999)
        if full_ppl > tern_ppl:
            p5 = f"CONFIRMED (0.3={tern_ppl:.2f} < 0.8={full_ppl:.2f})"
        else:
            p5 = f"NOT YET (0.3={tern_ppl:.2f} vs 0.8={full_ppl:.2f})"
    print(f"  P5 (Diminishing returns):      {p5}")

    total_done = len(results)
    total_all = len(ARCH_ORDER) + len(ABLATION_ORDER)
    print(f"\n  Overall: {total_done}/{total_all} configs complete")


def main():
    parser = argparse.ArgumentParser(description='v60 Interlingua Benchmark')
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
        for name in ABLATION_ORDER:
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
        print("  python benchmark_v60_interlingua.py --run interlingua_ternary")
        print("  python benchmark_v60_interlingua.py --group arch")
        print("  python benchmark_v60_interlingua.py --summary")
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
            names = ABLATION_ORDER
        else:
            names = ARCH_ORDER + ABLATION_ORDER

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
