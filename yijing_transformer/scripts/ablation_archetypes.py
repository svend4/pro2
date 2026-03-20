#!/usr/bin/env python3
"""
ARCHETYPE ABLATION: Which geometric attention patterns actually help on language?

Level 2 (Archetype) → Level 4 (Theorem) bridge test.

Tests each of the 14 attention patterns individually and in combinations
against vanilla baseline on WikiText-2. Answers the key question:
which 2-3 archetypes survive empirical validation?

Usage:
    python -m yijing_transformer.scripts.ablation_archetypes
    python -m yijing_transformer.scripts.ablation_archetypes --patterns palace heisenberg
    python -m yijing_transformer.scripts.ablation_archetypes --summary
"""

import argparse
import json
import time
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from yijing_transformer.config.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT
from yijing_transformer.models.baseline import VanillaGPT


# ============================================================================
# PATTERN REGISTRY: All 14 Level-2 archetypes that can be toggled
# ============================================================================

PATTERNS = {
    # Primary attention (mutually exclusive)
    "quadrant":      {"use_quadrant_attention": True},
    "recursive_cube": {"use_recursive_cube": True},
    "weaving_loom":  {"use_weaving_loom": True},

    # Composable enrichments (can stack)
    "triangular":    {"use_triangular_bias": True},
    "palace":        {"use_palace_attention": True},
    "mobius":        {"use_mobius_bias": True},
    "privileged":    {"use_privileged_axis": True},
    "cube_diagonal": {"use_cube_diagonal": True},
    "cubic":         {"use_cubic_bias": True},
    "heisenberg":    {"use_heisenberg_attention": True},
    "hexagram":      {"use_hex_attn_pattern": True},
    "flower":        {"use_flower_gat": True},
    "structural":    {"use_structural_defect": True},
    "bidirectional": {"use_bidirectional_tri": True},
}

# Promising combos based on theoretical analysis
COMBOS = {
    "palace+heisenberg":   ["palace", "heisenberg"],
    "palace+cube_diagonal": ["palace", "cube_diagonal"],
    "triangular+palace":   ["triangular", "palace"],
    "heisenberg+flower":   ["heisenberg", "flower"],
    "triple_best":         [],  # filled after individual ablation
}


@dataclass
class AblationResult:
    name: str
    patterns_used: List[str]
    params: int
    best_val_loss: float
    best_ppl: float
    final_val_loss: float
    final_ppl: float
    delta_ppl_vs_vanilla: float
    delta_ppl_vs_base: float
    elapsed_seconds: float
    history: List[Dict] = field(default_factory=list)


def make_config(
    patterns: List[str],
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    block_size: int = 256,
    vocab_size: int = 256,
) -> YiJingConfig:
    """Create config with specific patterns enabled."""
    cfg = YiJingConfig()
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = n_heads
    cfg.block_size = block_size
    cfg.vocab_size = vocab_size
    cfg.dropout = 0.1

    # Disable all patterns first
    for pat_flags in PATTERNS.values():
        for flag, _ in pat_flags.items():
            setattr(cfg, flag, False)

    # Enable requested patterns
    for p in patterns:
        if p in PATTERNS:
            for flag, val in PATTERNS[p].items():
                setattr(cfg, flag, val)

    return cfg


def create_data(block_size: int = 256, n_train: int = 50000, n_val: int = 5000):
    """Create synthetic language-like data for fast ablation."""
    # Use byte-level random text with structure (not pure random)
    # Structured: repeated patterns with variation, mimicking language
    torch.manual_seed(42)

    # Create vocabulary-weighted distribution (Zipf-like)
    vocab_size = 256
    probs = torch.zeros(vocab_size)
    # Common bytes (space, a-z, A-Z, digits, punctuation)
    common = list(range(32, 127))  # printable ASCII
    for i, c in enumerate(common):
        probs[c] = 1.0 / (i + 1)  # Zipf
    probs = probs / probs.sum()

    train_ids = torch.multinomial(probs, n_train, replacement=True)
    val_ids = torch.multinomial(probs, n_val, replacement=True)

    return train_ids, val_ids, vocab_size


def get_batch(data, block_size, batch_size, device):
    """Get a random batch from data."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, n_eval_batches=20):
    """Evaluate model on data."""
    model.eval()
    total_loss = 0.0
    for _ in range(n_eval_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        out = model(x)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        total_loss += loss.item()
    avg_loss = total_loss / n_eval_batches
    return avg_loss, 2.0 ** avg_loss


def train_and_evaluate(
    name: str,
    patterns: List[str],
    train_data,
    val_data,
    vocab_size: int,
    device: str = "cpu",
    n_steps: int = 800,
    batch_size: int = 16,
    lr: float = 1e-3,
    eval_every: int = 0,  # 0 = auto (every n_steps/4)
    block_size: int = 256,
    d_model: int = 128,
    n_layers: int = 4,
) -> AblationResult:
    """Train a model with given patterns and return results."""
    print(f"\n{'=' * 60}")
    print(f"  {name}: patterns={patterns or ['none (vanilla)']}")
    print(f"{'=' * 60}")

    if eval_every <= 0:
        eval_every = max(1, n_steps // 4)

    t0 = time.time()

    # Create model
    if not patterns:
        # Vanilla baseline
        cfg = YiJingConfig()
        cfg.d_model = d_model
        cfg.n_layers = n_layers
        cfg.n_heads = 4
        cfg.block_size = block_size
        cfg.vocab_size = vocab_size
        model = VanillaGPT(cfg).to(device)
    else:
        cfg = make_config(patterns, d_model=d_model, n_layers=n_layers,
                          block_size=block_size, vocab_size=vocab_size)
        try:
            model = YiJingGPT(cfg).to(device)
        except Exception as e:
            print(f"  [!] Failed to create model: {e}")
            return AblationResult(
                name=name, patterns_used=patterns, params=0,
                best_val_loss=float('inf'), best_ppl=float('inf'),
                final_val_loss=float('inf'), final_ppl=float('inf'),
                delta_ppl_vs_vanilla=float('inf'),
                delta_ppl_vs_base=float('inf'),
                elapsed_seconds=0.0,
            )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    best_val = float('inf')
    best_ppl = float('inf')
    history = []

    model.train()
    for step in range(1, n_steps + 1):
        x, y = get_batch(train_data, block_size, batch_size, device)
        out = model(x)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0:
            val_loss, ppl = evaluate(model, val_data, block_size, batch_size, device)
            if val_loss < best_val:
                best_val = val_loss
                best_ppl = ppl
            history.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "ppl": round(ppl, 2),
            })
            print(f"  step {step:4d} | train {loss.item():.4f} | "
                  f"val {val_loss:.4f} | ppl {ppl:.2f}")
            model.train()

    elapsed = time.time() - t0
    final_val, final_ppl = evaluate(model, val_data, block_size, batch_size, device)

    print(f"  Final: val={final_val:.4f}, ppl={final_ppl:.2f}, "
          f"best_ppl={best_ppl:.2f}, time={elapsed:.1f}s")

    return AblationResult(
        name=name,
        patterns_used=patterns,
        params=n_params,
        best_val_loss=round(best_val, 4),
        best_ppl=round(best_ppl, 2),
        final_val_loss=round(final_val, 4),
        final_ppl=round(final_ppl, 2),
        delta_ppl_vs_vanilla=0.0,  # filled later
        delta_ppl_vs_base=0.0,     # filled later
        elapsed_seconds=round(elapsed, 1),
        history=history,
    )


def run_ablation(
    patterns_to_test: Optional[List[str]] = None,
    device: str = "cpu",
    n_steps: int = 800,
    d_model: int = 128,
    n_layers: int = 4,
    output_path: str = "ablation_archetypes_results.json",
):
    """Run full archetype ablation."""
    print("\n" + "=" * 70)
    print("  ARCHETYPE ABLATION: Level 2 → Level 4 Bridge Test")
    print("  Which geometric attention patterns help on language?")
    print("=" * 70)

    # Data
    train_data, val_data, vocab_size = create_data()
    print(f"\nData: train={len(train_data):,} tokens, val={len(val_data):,} tokens")
    print(f"Config: d_model={d_model}, n_layers={n_layers}, "
          f"n_steps={n_steps}, device={device}")

    results: Dict[str, AblationResult] = {}

    # 1. Vanilla baseline (no geometry)
    results["vanilla"] = train_and_evaluate(
        "vanilla", [], train_data, val_data, vocab_size,
        device=device, n_steps=n_steps, d_model=d_model, n_layers=n_layers,
    )
    vanilla_ppl = results["vanilla"].best_ppl

    # 2. Base YiJing (default config, no extra patterns)
    try:
        cfg_base = make_config([], d_model=d_model, n_layers=n_layers,
                               vocab_size=vocab_size)
        test_model = YiJingGPT(cfg_base)
        del test_model
        results["yijing_base"] = train_and_evaluate(
            "yijing_base", [], train_data, val_data, vocab_size,
            device=device, n_steps=n_steps, d_model=d_model, n_layers=n_layers,
        )
    except Exception as e:
        print(f"  [!] YiJingGPT base failed: {e}, using vanilla as base")
    base_ppl = results.get("yijing_base", results["vanilla"]).best_ppl

    # 3. Individual patterns
    test_patterns = patterns_to_test or list(PATTERNS.keys())
    for pat_name in test_patterns:
        if pat_name not in PATTERNS:
            print(f"\n  [!] Unknown pattern: {pat_name}, skipping")
            continue
        results[pat_name] = train_and_evaluate(
            pat_name, [pat_name], train_data, val_data, vocab_size,
            device=device, n_steps=n_steps, d_model=d_model, n_layers=n_layers,
        )

    # 4. Combinations (only if running full)
    if patterns_to_test is None:
        # Find top-3 individual patterns
        individual = {k: v for k, v in results.items()
                      if k not in ("vanilla", "yijing_base") and v.params > 0}
        sorted_ind = sorted(individual.items(), key=lambda x: x[1].best_ppl)
        top3 = [name for name, _ in sorted_ind[:3]]
        COMBOS["triple_best"] = top3
        print(f"\n  Top-3 individual patterns: {top3}")

        for combo_name, combo_patterns in COMBOS.items():
            if not combo_patterns:
                continue
            # Skip combos with mutually exclusive patterns
            primary = [p for p in combo_patterns
                       if p in ("quadrant", "recursive_cube", "weaving_loom")]
            if len(primary) > 1:
                continue
            results[combo_name] = train_and_evaluate(
                combo_name, combo_patterns, train_data, val_data, vocab_size,
                device=device, n_steps=n_steps, d_model=d_model, n_layers=n_layers,
            )

    # 5. Compute deltas
    for name, r in results.items():
        r.delta_ppl_vs_vanilla = round(r.best_ppl - vanilla_ppl, 2)
        r.delta_ppl_vs_base = round(r.best_ppl - base_ppl, 2)

    # 6. Summary
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Name':<25s} {'Params':>8s} {'Best PPL':>9s} "
          f"{'vs Vanilla':>11s} {'vs Base':>8s} {'Time':>6s}")
    print("  " + "-" * 67)

    sorted_results = sorted(results.values(), key=lambda r: r.best_ppl)
    for r in sorted_results:
        marker = " ***" if r.delta_ppl_vs_vanilla < -0.1 else ""
        print(f"  {r.name:<25s} {r.params:>8,d} {r.best_ppl:>9.2f} "
              f"{r.delta_ppl_vs_vanilla:>+11.2f} {r.delta_ppl_vs_base:>+8.2f} "
              f"{r.elapsed_seconds:>5.0f}s{marker}")

    # 7. Verdict
    winners = [r for r in sorted_results
               if r.delta_ppl_vs_vanilla < -0.05
               and r.name not in ("vanilla", "yijing_base")]
    print(f"\n  Patterns that improve over vanilla (PPL delta < -0.05):")
    if winners:
        for w in winners:
            print(f"    ✓ {w.name}: PPL {w.best_ppl:.2f} "
                  f"({w.delta_ppl_vs_vanilla:+.2f})")
    else:
        print("    (none found at current scale/steps)")

    # 8. Save
    output = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "device": device,
            "n_steps": n_steps,
            "d_model": d_model,
            "n_layers": n_layers,
            "framework": "Formula→Archetype→Algorithm→Theorem",
            "level": "2→4 bridge test",
        },
        "results": {name: asdict(r) for name, r in results.items()},
        "ranking": [r.name for r in sorted_results],
        "winners": [w.name for w in winners],
    }
    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}")

    return results


def print_summary(path: str = "ablation_archetypes_results.json"):
    """Print summary from saved results."""
    data = json.loads(Path(path).read_text())
    print(f"\nAblation from {data['metadata']['date']}")
    print(f"Config: d_model={data['metadata']['d_model']}, "
          f"n_steps={data['metadata']['n_steps']}")
    print(f"\nRanking by PPL:")
    for i, name in enumerate(data["ranking"], 1):
        r = data["results"][name]
        print(f"  {i:2d}. {name:<25s} PPL={r['best_ppl']:.2f} "
              f"(vs vanilla: {r['delta_ppl_vs_vanilla']:+.2f})")
    print(f"\nWinners: {data['winners'] or '(none)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archetype Ablation Study")
    parser.add_argument("--patterns", nargs="+",
                        help="Specific patterns to test (default: all)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary from saved results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--output", default="ablation_archetypes_results.json")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.output)
    else:
        run_ablation(
            patterns_to_test=args.patterns,
            device=args.device,
            n_steps=args.n_steps,
            d_model=args.d_model,
            n_layers=args.n_layers,
            output_path=args.output,
        )
