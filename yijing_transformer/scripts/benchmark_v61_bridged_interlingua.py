#!/usr/bin/env python3
"""
v61 Benchmark: BridgedInterlingua (двойная прослойка) vs v58 Bridge vs v60 Interlingua.

Тестирует гипотезу: Module → Bridge → 64 Archetype → Core лучше, чем:
  - Module → Bridge → Core (v58, одинарная прослойка мостов)
  - Module → Encoder → 64 Archetype → Core (v60, одинарная прослойка архетипов)

Предсказания:
  H1: BridgedInterlingua снижает интерференцию лучше v60 (мосты фильтруют конфликты)
  H2: BridgedInterlingua сохраняет семантику лучше v58 (архетипы дают единый язык)
  H3: Тернарное голосование по bridge-выходам точнее (меньше шума после медиации)
  H4: Двойная прослойка не хуже одинарных по скорости (O(N) для обоих слоёв)
  H5: Paired bit (строительная логика) лучше обычных тритов (нет STE-ловушки)
  H6: Paired bit активирует больше архетипов (градиент течёт через оба бита)

Использование:
  python benchmark_v61_bridged_interlingua.py --group all
  python benchmark_v61_bridged_interlingua.py --run bridged_lightweight
  python benchmark_v61_bridged_interlingua.py --summary
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
    'benchmark_v61_bridged_interlingua_results.json'
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
    'v58_bridge_lightweight': dict(
        **ALL_SOURCES,
        use_bridge_of_modules=True,
        bridge_mode='lightweight',
    ),
    'v58_bridge_full': dict(
        **ALL_SOURCES,
        use_bridge_of_modules=True,
        bridge_mode='full',
    ),
    'v60_interlingua': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    # v61: BridgedInterlingua variants
    'bridged_lightweight': dict(
        **ALL_SOURCES,
        use_bridged_interlingua=True,
        bridged_bridge_mode='lightweight',
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    'bridged_full': dict(
        **ALL_SOURCES,
        use_bridged_interlingua=True,
        bridged_bridge_mode='full',
        bridged_bridge_n_heads=2,
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    'bridged_no_ternary': dict(
        **ALL_SOURCES,
        use_bridged_interlingua=True,
        bridged_bridge_mode='lightweight',
        interlingua_use_ternary=False,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    'bridged_high_uncertainty': dict(
        **ALL_SOURCES,
        use_bridged_interlingua=True,
        bridged_bridge_mode='lightweight',
        interlingua_use_ternary=True,
        interlingua_uncertainty=0.7,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    # v62: Строительная логика — трит из пары битов
    'bridged_paired_bit': dict(
        **ALL_SOURCES,
        use_bridged_interlingua=True,
        bridged_bridge_mode='lightweight',
        interlingua_use_ternary=True,
        interlingua_use_paired_bit=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    'v60_paired_bit': dict(
        **ALL_SOURCES,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_use_paired_bit=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
    # v63: Geometric prior — SOLAN Q6 lookup table as inductive bias
    'v63_convergence_learned': dict(
        **ALL_SOURCES,
        use_convergence_bridge=True,
        use_glyph_prior=False,
    ),
    'v63_convergence_glyph_prior': dict(
        **ALL_SOURCES,
        use_convergence_bridge=True,
        use_glyph_prior=True,
    ),
    'v63_prior_plus_interlingua': dict(
        **ALL_SOURCES,
        use_convergence_bridge=True,
        use_glyph_prior=True,
        use_archetypal_interlingua=True,
        interlingua_use_ternary=True,
        interlingua_use_paired_bit=True,
        interlingua_uncertainty=0.3,
        interlingua_n_archetypes=64,
        interlingua_n_heads=4,
    ),
}

CONFIG_ORDER = [
    'vanilla', 'v58_bridge_lightweight', 'v58_bridge_full',
    'v60_interlingua',
    'bridged_lightweight', 'bridged_full',
    'bridged_no_ternary', 'bridged_high_uncertainty',
    'bridged_paired_bit', 'v60_paired_bit',
    'v63_convergence_learned', 'v63_convergence_glyph_prior',
    'v63_prior_plus_interlingua',
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


def _get_interlingua_stats(model):
    """Extract interlingua stats (works for both v60 and v61)."""
    for layer in model.core.layers:
        if hasattr(layer, 'archetypal_interlingua'):
            il = layer.archetypal_interlingua
            stats = il.get_interlingua_stats()
            stats['q6_correlation'] = il.archetype_q6_correlation().item()
            stats['class'] = type(il).__name__
            return stats
    return None


def _get_bridge_type(model):
    """Detect bridge/interlingua type for display."""
    for layer in model.core.layers:
        if hasattr(layer, 'archetypal_interlingua'):
            return type(layer.archetypal_interlingua).__name__
        if hasattr(layer, 'bridge_of_modules'):
            return type(layer.bridge_of_modules).__name__
    return 'none'


def train_and_eval(cfg, name, train_data, val_data,
                   n_steps=800, batch_size=16, lr=1e-3, eval_every=200):
    set_seed(42)
    model = YiJingGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    bridge_type = _get_bridge_type(model)

    print(f"\n{'='*70}")
    print(f"  {name} ({n_params:,} params, type={bridge_type})")
    print(f"{'='*70}")

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
                cls = il_stats.get('class', '?')
                extra = (f" [{cls}] gate={gate:.3f} active={active}"
                         f" trits=[+{trit.get('pos',0):.2f}/0:{trit.get('zero',0):.2f}/-{trit.get('neg',0):.2f}]"
                         f" q6={q6_corr:.3f}")
                # Paired bit direction stats (весна/осень)
                dir_stats = il_stats.get('direction_stats')
                if dir_stats and il_stats.get('paired_bit'):
                    extra += f" dir=[↑{dir_stats.get('spring',0):.2f}/↓{dir_stats.get('autumn',0):.2f}]"
                # Temperature annealing stats
                if 'ternary_temperature' in il_stats:
                    extra += f" temp={il_stats['ternary_temperature']:.3f}"
                # BridgedInterlingua-specific stats
                if 'n_bridges' in il_stats:
                    extra += f" bridges={il_stats['n_bridges']}"
                interlingua_history.append({'step': step, **il_stats})
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

    final_il_stats = _get_interlingua_stats(model) or {}

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")
    if final_il_stats:
        cls = final_il_stats.get('class', '?')
        print(f"  {cls}: gate={final_il_stats['global_gate']:.3f}"
              f" active={final_il_stats.get('active_archetypes', '?')}"
              f" q6_corr={final_il_stats.get('q6_correlation', 0):.3f}"
              f" trits={final_il_stats.get('trit_distribution', {})}")
        if 'bridge_scales' in final_il_stats:
            print(f"  Bridge scales: {final_il_stats['bridge_scales']}")

    return {
        'name': name, 'params': n_params, 'bridge_type': bridge_type,
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
        'interlingua_stats': final_il_stats,
        'interlingua_history': interlingua_history,
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

    print("\n" + "=" * 110)
    print("  v61 BENCHMARK: BridgedInterlingua (двойная прослойка) vs v58 Bridge vs v60 Interlingua")
    print("=" * 110)
    print(f"  {'Config':<28} {'Type':<24} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8} {'Delta':>8}")
    print("  " + "-" * 106)

    vanilla_ppl = results.get('vanilla', {}).get('best_ppl')

    for name in CONFIG_ORDER:
        if name not in results:
            print(f"  {name:<28} {'':>24} {'---':>8} {'(pending)':>10}")
            continue
        r = results[name]
        delta_str = ""
        if vanilla_ppl is not None and name != 'vanilla':
            delta = r['best_ppl'] - vanilla_ppl
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}"
        bt = r.get('bridge_type', 'none')
        print(f"  {name:<28} {bt:<24} {r['params']:>8,} "
              f"{r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} "
              f"{r['time']:>7.1f}s {delta_str:>8}")

    # Interlingua-specific diagnostics
    il_configs = [n for n in CONFIG_ORDER if n in results
                  and results[n].get('interlingua_stats')]
    if il_configs:
        print(f"\n  Interlingua / BridgedInterlingua Diagnostics:")
        print(f"  {'Config':<28} {'Class':<22} {'Gate':>6} {'Active':>8} {'Q6':>6} {'Trits(+/0/-)':>16} {'Dir(↑/↓)':>12}")
        print("  " + "-" * 104)
        for name in il_configs:
            r = results[name]
            il = r.get('interlingua_stats', {})
            cls = il.get('class', '?')
            gate = il.get('global_gate', 0)
            active = il.get('active_archetypes', '?')
            q6 = il.get('q6_correlation', 0)
            trit = il.get('trit_distribution', {})
            trit_str = f"{trit.get('pos',0):.2f}/{trit.get('zero',0):.2f}/{trit.get('neg',0):.2f}"
            dir_str = ""
            if il.get('paired_bit'):
                ds = il.get('direction_stats', {})
                dir_str = f"{ds.get('spring',0):.2f}/{ds.get('autumn',0):.2f}"
            print(f"  {name:<28} {cls:<22} {gate:>6.3f} {str(active):>8} {q6:>6.3f} {trit_str:>16} {dir_str:>12}")

    # Hypothesis testing
    print(f"\n  Hypothesis Testing:")
    v58_lw = results.get('v58_bridge_lightweight', {}).get('best_ppl')
    v60 = results.get('v60_interlingua', {}).get('best_ppl')
    bridged_lw = results.get('bridged_lightweight', {}).get('best_ppl')

    if all(x is not None for x in [v58_lw, v60, bridged_lw]):
        print(f"  H1 (bridged < v60):   {'CONFIRMED' if bridged_lw < v60 else 'REJECTED'}"
              f"  ({bridged_lw:.2f} vs {v60:.2f})")
        print(f"  H2 (bridged < v58):   {'CONFIRMED' if bridged_lw < v58_lw else 'REJECTED'}"
              f"  ({bridged_lw:.2f} vs {v58_lw:.2f})")

    bridged_nt = results.get('bridged_no_ternary', {}).get('best_ppl')
    if bridged_lw is not None and bridged_nt is not None:
        print(f"  H3 (ternary helps):   {'CONFIRMED' if bridged_lw < bridged_nt else 'REJECTED'}"
              f"  ({bridged_lw:.2f} vs {bridged_nt:.2f})")

    bridged_lw_time = results.get('bridged_lightweight', {}).get('time')
    v60_time = results.get('v60_interlingua', {}).get('time')
    if bridged_lw_time and v60_time:
        overhead = (bridged_lw_time / v60_time - 1) * 100
        print(f"  H4 (speed OK):        {'CONFIRMED' if overhead < 30 else 'REJECTED'}"
              f"  (overhead: {overhead:+.1f}%)")

    # H5: Paired bit (строительная логика) лучше обычных тритов
    bridged_pb = results.get('bridged_paired_bit', {}).get('best_ppl')
    if bridged_lw is not None and bridged_pb is not None:
        print(f"  H5 (paired_bit > ternary): {'CONFIRMED' if bridged_pb < bridged_lw else 'REJECTED'}"
              f"  ({bridged_pb:.2f} vs {bridged_lw:.2f})")

    # H6: Paired bit активирует больше архетипов (нет STE-ловушки)
    bridged_lw_active = results.get('bridged_lightweight', {}).get('interlingua_stats', {}).get('active_archetypes')
    bridged_pb_active = results.get('bridged_paired_bit', {}).get('interlingua_stats', {}).get('active_archetypes')
    if bridged_lw_active is not None and bridged_pb_active is not None:
        print(f"  H6 (more active archetypes): {'CONFIRMED' if bridged_pb_active > bridged_lw_active else 'REJECTED'}"
              f"  ({bridged_pb_active} vs {bridged_lw_active})")

    done = sum(1 for n in CONFIG_ORDER if n in results)
    total = len(CONFIG_ORDER)
    print(f"\n  Progress: {done}/{total} configs done")


def main():
    parser = argparse.ArgumentParser(description='v61 BridgedInterlingua Benchmark')
    parser.add_argument('--run', type=str, help='Run single config')
    parser.add_argument('--group', type=str, choices=['baselines', 'bridged', 'all'],
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
            names = ['vanilla', 'v58_bridge_lightweight', 'v58_bridge_full', 'v60_interlingua']
        elif args.group == 'bridged':
            names = ['bridged_lightweight', 'bridged_full',
                     'bridged_no_ternary', 'bridged_high_uncertainty',
                     'bridged_paired_bit', 'v60_paired_bit']
        else:  # all
            names = CONFIG_ORDER

        for name in names:
            run_single(name, train_data, val_data, vocab_size)

        print_summary()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
