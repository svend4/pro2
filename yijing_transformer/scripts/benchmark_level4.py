#!/usr/bin/env python3
"""
LEVEL 4 BENCHMARK: Does hypercube geometry work on real language?

Comprehensive benchmark answering the Theorem-level question.
Tests geometry vs vanilla on structured language tasks at sufficient scale.

Tasks:
  1. Language Modeling (PPL on structured text)
  2. Modular Arithmetic (a + b mod 64 — Z₂⁶ structured)
  3. XOR Classification (pure group-theoretic task)
  4. Sequence Copying (memory / attention quality)

Each task tests a different aspect of the Formula→Theorem bridge.

Usage:
    python -m yijing_transformer.scripts.benchmark_level4
    python -m yijing_transformer.scripts.benchmark_level4 --tasks lm xor
    python -m yijing_transformer.scripts.benchmark_level4 --summary
"""

import argparse
import json
import math
import time
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from yijing_transformer.config.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT
from yijing_transformer.models.baseline import VanillaGPT


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

def make_vanilla(d_model=128, n_layers=4, vocab_size=256, block_size=256):
    cfg = YiJingConfig()
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = max(1, d_model // 32)
    cfg.block_size = block_size
    cfg.vocab_size = vocab_size
    cfg.dropout = 0.1
    return VanillaGPT(cfg), "vanilla"


def make_geometric(d_model=128, n_layers=4, vocab_size=256, block_size=256,
                   patterns=None):
    """Create YiJingGPT with specified geometric patterns."""
    cfg = YiJingConfig()
    cfg.d_model = d_model
    cfg.n_layers = n_layers
    cfg.n_heads = max(1, d_model // 32)
    cfg.block_size = block_size
    cfg.vocab_size = vocab_size
    cfg.dropout = 0.1

    if patterns:
        from yijing_transformer.scripts.ablation_archetypes import PATTERNS
        for p in patterns:
            if p in PATTERNS:
                for flag, val in PATTERNS[p].items():
                    setattr(cfg, flag, val)

    return YiJingGPT(cfg), f"geometric({'_'.join(patterns or ['base'])})"


# ============================================================================
# TASK 1: LANGUAGE MODELING
# ============================================================================

def task_language_modeling(
    device="cpu", d_model=128, n_layers=4, n_steps=800, batch_size=16,
    block_size=256,
) -> Dict:
    """Language modeling on structured text (Zipf-distributed bytes)."""
    print("\n" + "=" * 60)
    print("  TASK 1: Language Modeling")
    print("  Level: 4 (Theorem) — does geometry help on text?")
    print("=" * 60)

    torch.manual_seed(42)
    vocab_size = 256

    # Structured data: repeating patterns with noise (language-like)
    n_tokens = 60000
    patterns_pool = []
    for _ in range(200):
        length = torch.randint(3, 15, (1,)).item()
        pat = torch.randint(32, 127, (length,))
        patterns_pool.append(pat)

    data = []
    while len(data) < n_tokens:
        idx = torch.randint(0, len(patterns_pool), (1,)).item()
        data.extend(patterns_pool[idx].tolist())
        if torch.rand(1).item() < 0.1:  # 10% noise
            data.append(torch.randint(0, vocab_size, (1,)).item())
    data = torch.tensor(data[:n_tokens], dtype=torch.long)
    split = int(n_tokens * 0.9)
    train_data, val_data = data[:split], data[split:]

    results = {}
    eval_every = max(1, n_steps // 4)

    for model_fn, name in [
        (lambda: make_vanilla(d_model, n_layers, vocab_size, block_size), "vanilla"),
        (lambda: make_geometric(d_model, n_layers, vocab_size, block_size), "geometric"),
    ]:
        model, label = model_fn()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}: {n_params:,} params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        best_ppl = float('inf')
        t0 = time.time()

        model.train()
        for step in range(1, n_steps + 1):
            ix = torch.randint(len(train_data) - block_size - 1, (batch_size,))
            x = torch.stack([train_data[i:i+block_size] for i in ix]).to(device)
            y = torch.stack([train_data[i+1:i+block_size+1] for i in ix]).to(device)

            out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    vix = torch.randint(len(val_data) - block_size - 1, (batch_size,))
                    vx = torch.stack([val_data[i:i+block_size] for i in vix]).to(device)
                    vy = torch.stack([val_data[i+1:i+block_size+1] for i in vix]).to(device)
                    vout = model(vx)
                    vlogits = vout if isinstance(vout, torch.Tensor) else vout[0]
                    vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                    ppl = 2.0 ** vloss.item()
                    if ppl < best_ppl:
                        best_ppl = ppl
                    print(f"    step {step:4d} | loss {vloss.item():.4f} | ppl {ppl:.2f}")
                model.train()

        elapsed = time.time() - t0
        results[label] = {
            "params": n_params, "best_ppl": round(best_ppl, 2),
            "elapsed": round(elapsed, 1),
        }
        del model

    return {"task": "language_modeling", "results": results}


# ============================================================================
# TASK 2: MODULAR ARITHMETIC (a + b mod 64)
# ============================================================================

def task_modular_arithmetic(
    device="cpu", d_model=128, n_layers=4, n_steps=1000, batch_size=64,
) -> Dict:
    """Modular arithmetic: a + b mod 64 (Z₂⁶ structured)."""
    print("\n" + "=" * 60)
    print("  TASK 2: Modular Arithmetic (a + b mod 64)")
    print("  Level: 1→4 (Formula→Theorem) — group theory on Z₂⁶")
    print("=" * 60)

    vocab_size = 64
    block_size = 3  # [a, b] -> predict c = (a+b) mod 64

    results = {}
    eval_every = max(1, n_steps // 4)

    for model_fn, name in [
        (lambda: make_vanilla(d_model, n_layers, vocab_size, block_size), "vanilla"),
        (lambda: make_geometric(d_model, n_layers, vocab_size, block_size), "geometric"),
    ]:
        model, label = model_fn()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}: {n_params:,} params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        best_acc = 0.0
        t0 = time.time()

        model.train()
        for step in range(1, n_steps + 1):
            a = torch.randint(0, 64, (batch_size,))
            b = torch.randint(0, 64, (batch_size,))
            c = (a + b) % 64
            x = torch.stack([a, b, c], dim=1).to(device)  # [B, 3]

            out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            # Predict c from position 1 (after seeing a, b)
            pred_logits = logits[:, 1, :vocab_size]  # [B, 64]
            loss = F.cross_entropy(pred_logits, c.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # Test on 1000 samples
                    ta = torch.randint(0, 64, (1000,))
                    tb = torch.randint(0, 64, (1000,))
                    tc = (ta + tb) % 64
                    tx = torch.stack([ta, tb, tc], dim=1).to(device)
                    tout = model(tx)
                    tlogits = tout if isinstance(tout, torch.Tensor) else tout[0]
                    preds = tlogits[:, 1, :vocab_size].argmax(dim=-1)
                    acc = (preds == tc.to(device)).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                    print(f"    step {step:4d} | loss {loss.item():.4f} | "
                          f"acc {acc:.1%}")
                model.train()

        elapsed = time.time() - t0
        results[label] = {
            "params": n_params, "best_accuracy": round(best_acc, 4),
            "elapsed": round(elapsed, 1),
        }
        del model

    return {"task": "modular_arithmetic_mod64", "results": results}


# ============================================================================
# TASK 3: XOR CLASSIFICATION
# ============================================================================

def task_xor_classification(
    device="cpu", d_model=128, n_layers=4, n_steps=1000, batch_size=64,
) -> Dict:
    """XOR on Z₂⁶: pure group-theoretic task where geometry should excel."""
    print("\n" + "=" * 60)
    print("  TASK 3: XOR on Z₂⁶ (should favor geometry)")
    print("  Level: 1→4 — pure algebraic structure")
    print("=" * 60)

    vocab_size = 64
    block_size = 3  # [a, b] -> predict c = a XOR b

    results = {}
    eval_every = max(1, n_steps // 4)

    for model_fn, name in [
        (lambda: make_vanilla(d_model, n_layers, vocab_size, block_size), "vanilla"),
        (lambda: make_geometric(d_model, n_layers, vocab_size, block_size), "geometric"),
    ]:
        model, label = model_fn()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}: {n_params:,} params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        best_acc = 0.0
        t0 = time.time()

        model.train()
        for step in range(1, n_steps + 1):
            a = torch.randint(0, 64, (batch_size,))
            b = torch.randint(0, 64, (batch_size,))
            c = a ^ b  # XOR
            x = torch.stack([a, b, c], dim=1).to(device)

            out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            pred_logits = logits[:, 1, :vocab_size]
            loss = F.cross_entropy(pred_logits, c.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    ta = torch.randint(0, 64, (1000,))
                    tb = torch.randint(0, 64, (1000,))
                    tc = ta ^ tb
                    tx = torch.stack([ta, tb, tc], dim=1).to(device)
                    tout = model(tx)
                    tlogits = tout if isinstance(tout, torch.Tensor) else tout[0]
                    preds = tlogits[:, 1, :vocab_size].argmax(dim=-1)
                    acc = (preds == tc.to(device)).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                    print(f"    step {step:4d} | loss {loss.item():.4f} | "
                          f"acc {acc:.1%}")
                model.train()

        elapsed = time.time() - t0
        results[label] = {
            "params": n_params, "best_accuracy": round(best_acc, 4),
            "elapsed": round(elapsed, 1),
        }
        del model

    return {"task": "xor_z2_6", "results": results}


# ============================================================================
# TASK 4: SEQUENCE COPYING
# ============================================================================

def task_sequence_copying(
    device="cpu", d_model=128, n_layers=4, n_steps=800, batch_size=32,
) -> Dict:
    """Copy a sequence after a separator: test attention quality."""
    print("\n" + "=" * 60)
    print("  TASK 4: Sequence Copying (attention quality)")
    print("  Level: 2→4 — do geometric attention patterns help memory?")
    print("=" * 60)

    vocab_size = 32  # small vocab
    seq_len = 8
    sep_token = 0
    block_size = seq_len * 2 + 1  # seq + sep + seq

    results = {}
    eval_every = max(1, n_steps // 4)

    for model_fn, name in [
        (lambda: make_vanilla(d_model, n_layers, vocab_size, block_size), "vanilla"),
        (lambda: make_geometric(d_model, n_layers, vocab_size, block_size), "geometric"),
    ]:
        model, label = model_fn()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {label}: {n_params:,} params")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        best_acc = 0.0
        t0 = time.time()

        model.train()
        for step in range(1, n_steps + 1):
            # Generate: [a1..a8, SEP, a1..a8]
            seq = torch.randint(1, vocab_size, (batch_size, seq_len))
            sep = torch.zeros(batch_size, 1, dtype=torch.long)
            x = torch.cat([seq, sep, seq], dim=1).to(device)  # [B, 17]

            out = model(x)
            logits = out if isinstance(out, torch.Tensor) else out[0]
            # Loss only on the copy part (positions seq_len+1 onwards)
            copy_logits = logits[:, seq_len:-1, :vocab_size]  # [B, 8, V]
            copy_targets = seq.to(device)  # [B, 8]
            loss = F.cross_entropy(
                copy_logits.reshape(-1, vocab_size), copy_targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    tseq = torch.randint(1, vocab_size, (200, seq_len))
                    tsep = torch.zeros(200, 1, dtype=torch.long)
                    tx = torch.cat([tseq, tsep, tseq], dim=1).to(device)
                    tout = model(tx)
                    tlogits = tout if isinstance(tout, torch.Tensor) else tout[0]
                    tpreds = tlogits[:, seq_len:-1, :vocab_size].argmax(dim=-1)
                    acc = (tpreds == tseq.to(device)).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                    print(f"    step {step:4d} | loss {loss.item():.4f} | "
                          f"copy_acc {acc:.1%}")
                model.train()

        elapsed = time.time() - t0
        results[label] = {
            "params": n_params, "best_copy_accuracy": round(best_acc, 4),
            "elapsed": round(elapsed, 1),
        }
        del model

    return {"task": "sequence_copying", "results": results}


# ============================================================================
# MAIN
# ============================================================================

TASK_REGISTRY = {
    "lm": task_language_modeling,
    "mod": task_modular_arithmetic,
    "xor": task_xor_classification,
    "copy": task_sequence_copying,
}


def run_benchmark(
    tasks: Optional[List[str]] = None,
    device: str = "cpu",
    n_steps: int = 800,
    d_model: int = 128,
    n_layers: int = 4,
    output_path: str = "benchmark_level4_results.json",
):
    """Run Level 4 benchmark suite."""
    print("\n" + "=" * 70)
    print("  LEVEL 4 BENCHMARK: Does Hypercube Geometry Work on Language?")
    print("  Framework: Formula → Archetype → Algorithm → Theorem")
    print("=" * 70)

    task_list = tasks or list(TASK_REGISTRY.keys())
    all_results = {}

    for task_name in task_list:
        if task_name not in TASK_REGISTRY:
            print(f"  [!] Unknown task: {task_name}")
            continue
        fn = TASK_REGISTRY[task_name]
        kwargs = {"device": device, "d_model": d_model, "n_layers": n_layers,
                  "n_steps": n_steps}
        result = fn(**kwargs)
        all_results[result["task"]] = result["results"]

    # Summary
    print("\n" + "=" * 70)
    print("  LEVEL 4 BENCHMARK SUMMARY")
    print("=" * 70)

    for task_name, task_results in all_results.items():
        print(f"\n  {task_name}:")
        metric = None
        for label, r in task_results.items():
            if "best_ppl" in r:
                metric = "PPL"
                val = r["best_ppl"]
            elif "best_accuracy" in r:
                metric = "Accuracy"
                val = f"{r['best_accuracy']:.1%}"
            elif "best_copy_accuracy" in r:
                metric = "Copy Acc"
                val = f"{r['best_copy_accuracy']:.1%}"
            else:
                metric = "?"
                val = "?"
            print(f"    {label:<30s} {metric}={val}  "
                  f"({r['params']:,} params, {r['elapsed']:.0f}s)")

        # Compare
        labels = list(task_results.keys())
        if len(labels) >= 2:
            v = task_results[labels[0]]
            g = task_results[labels[1]]
            if "best_ppl" in v and "best_ppl" in g:
                delta = g["best_ppl"] - v["best_ppl"]
                winner = "geometric" if delta < 0 else "vanilla"
                print(f"    → {winner} wins (delta PPL = {delta:+.2f})")
            elif "best_accuracy" in v and "best_accuracy" in g:
                delta = g["best_accuracy"] - v["best_accuracy"]
                winner = "geometric" if delta > 0 else "vanilla"
                print(f"    → {winner} wins (delta acc = {delta:+.1%})")
            elif "best_copy_accuracy" in v and "best_copy_accuracy" in g:
                delta = g["best_copy_accuracy"] - v["best_copy_accuracy"]
                winner = "geometric" if delta > 0 else "vanilla"
                print(f"    → {winner} wins (delta acc = {delta:+.1%})")

    # Save
    output = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "device": device,
            "n_steps": n_steps,
            "d_model": d_model,
            "n_layers": n_layers,
            "framework": "Formula→Archetype→Algorithm→Theorem",
            "level": "4 (Theorem)",
        },
        "results": all_results,
    }
    out_path = Path(output_path)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}")

    return all_results


def print_summary(path: str = "benchmark_level4_results.json"):
    """Print summary from saved results."""
    data = json.loads(Path(path).read_text())
    print(f"\nLevel 4 Benchmark from {data['metadata']['date']}")
    print(f"Config: d_model={data['metadata']['d_model']}, "
          f"n_steps={data['metadata']['n_steps']}")
    for task, results in data["results"].items():
        print(f"\n  {task}:")
        for label, r in results.items():
            vals = {k: v for k, v in r.items() if k not in ("params", "elapsed")}
            print(f"    {label}: {vals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Level 4 Benchmark")
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_REGISTRY.keys()),
                        help="Tasks to run (default: all)")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--output", default="benchmark_level4_results.json")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.output)
    else:
        run_benchmark(
            tasks=args.tasks,
            device=args.device,
            n_steps=args.n_steps,
            d_model=args.d_model,
            n_layers=args.n_layers,
            output_path=args.output,
        )
