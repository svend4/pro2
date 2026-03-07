#!/usr/bin/env python3
"""
Direction C: Pairwise source combination study.

Tests all 2-source combinations on modular arithmetic mod 64
to find the optimal subset. Key question: which pairs synergize
and which interfere?

Sources:
  F = Fomyuk (D4 + antipodal)
  B = Belyaev (Heisenberg + FlowerOfLife + Möbius + StructuralDefect)
  A = Andreev (triangular + 4-level PE + bidirectional)
  K = Kasatkin (dual embedding + cube diagonal + privileged axis)
  S = Sklyarova (palace attention + graduated biangua)
  H = Hermann (factored6 quantizer)
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT


def generate_modadd_data(n_samples, mod=64, device='cpu'):
    a = torch.randint(0, mod, (n_samples,), device=device)
    b = torch.randint(0, mod, (n_samples,), device=device)
    c = (a + b) % mod
    SEP = mod
    inputs = torch.stack([a, b, torch.full_like(a, SEP)], dim=1)
    targets = torch.stack([torch.full_like(a, -100), torch.full_like(a, -100), c], dim=1)
    return inputs, targets


# Individual source configs (flags only, no base params)
SOURCE_FLAGS = {
    'F': dict(use_d4_equivariant=True, use_antipodal_reg=True, antipodal_weight=0.01),
    'B': dict(use_heisenberg_attention=True, use_flower_gat=True,
              use_mobius_bias=True, use_structural_defect=True),
    'A': dict(use_triangular_bias=True, use_four_level_pe=True, use_bidirectional_tri=True),
    'K': dict(use_dual_embedding=True, use_cube_diagonal=True, use_privileged_axis=True),
    'S': dict(use_palace_attention=True, use_graduated_biangua=True),
    'H': dict(quantizer_type='factored6'),
}

SOURCE_NAMES = {
    'F': 'Fomyuk', 'B': 'Belyaev', 'A': 'Andreev',
    'K': 'Kasatkin', 'S': 'Sklyarova', 'H': 'Hermann',
}


def make_config(source_keys):
    """Merge flags from multiple sources."""
    base = dict(vocab_size=65, d_model=64, n_layers=2, n_heads=4,
                block_size=8, dropout=0.0, use_rope=False, use_swiglu=True,
                hex_strength=0.05, temp=0.3)
    merged = {}
    for k in source_keys:
        merged.update(SOURCE_FLAGS[k])
    return YiJingConfig(**{**base, **merged})


def train_and_eval(cfg, label, n_steps=3000, batch_size=256, lr=3e-4,
                   eval_every=200, device='cpu'):
    torch.manual_seed(42)
    model = YiJingGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {label} ({n_params:,} params)")

    accs = []
    losses = []
    t0 = time.time()

    for step in range(1, n_steps + 1):
        model.train()
        inputs, targets = generate_modadd_data(batch_size, device=device)
        logits, loss_base, _ = model(inputs, targets=targets)
        loss = F.cross_entropy(logits[:, 2, :], targets[:, 2])
        if isinstance(loss_base, torch.Tensor):
            loss = loss + 0.1 * loss_base

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vi, vt = generate_modadd_data(1024, device=device)
                vl, _, _ = model(vi)
                preds = vl[:, 2, :].argmax(dim=-1)
                acc = (preds == vt[:, 2]).float().mean().item()
                vloss = F.cross_entropy(vl[:, 2, :], vt[:, 2]).item()
            accs.append(acc)
            losses.append(vloss)

    elapsed = time.time() - t0
    print(f"    acc@200={accs[1] if len(accs)>1 else 0:.4f} "
          f"acc@600={accs[3] if len(accs)>3 else 0:.4f} "
          f"final={accs[-1]:.4f} loss={losses[-1]:.4f} time={elapsed:.1f}s")

    return {
        'label': label,
        'final_accuracy': accs[-1],
        'final_loss': losses[-1],
        'acc_at_200': accs[1] if len(accs) > 1 else 0,
        'acc_at_600': accs[3] if len(accs) > 3 else 0,
        'params': n_params,
        'time': elapsed,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 70)
    print("Direction C: Pairwise Source Combination Study (v53)")
    print("=" * 70)

    results = {}

    # Vanilla baseline
    print("\n--- Singles ---")
    cfg = make_config([])
    # Vanilla needs explicit config without merged source flags
    vanilla_cfg = YiJingConfig(vocab_size=65, d_model=64, n_layers=2, n_heads=4,
                               block_size=8, dropout=0.0, use_rope=False, use_swiglu=True,
                               hex_strength=0.05, temp=0.3)
    results['vanilla'] = train_and_eval(vanilla_cfg, 'vanilla', device=device)

    # Best singles (from ablation): F, B, H
    for k in ['F', 'B', 'H']:
        label = f"{k}({SOURCE_NAMES[k]})"
        results[k] = train_and_eval(make_config([k]), label, device=device)

    # All 2-source pairs from the top 4 sources: F, B, A, H
    # (Skip S=Sklyarova which was harmful, K=Kasatkin which was weak)
    print("\n--- Promising Pairs ---")
    promising = ['F', 'B', 'A', 'H']
    for i in range(len(promising)):
        for j in range(i + 1, len(promising)):
            k1, k2 = promising[i], promising[j]
            label = f"{k1}+{k2}"
            results[label] = train_and_eval(make_config([k1, k2]), label, device=device)

    # Best triple (top 3)
    print("\n--- Best Triple ---")
    results['F+B+H'] = train_and_eval(make_config(['F', 'B', 'H']), 'F+B+H', device=device)
    results['F+B+A'] = train_and_eval(make_config(['F', 'B', 'A']), 'F+B+A', device=device)

    # Summary
    print("\n\n" + "=" * 70)
    print("PAIRWISE COMBINATION SUMMARY (mod-add 64)")
    print("=" * 70)
    print(f"{'Config':<15} {'Acc@200':>8} {'Acc@600':>8} {'Final':>8} {'Loss':>10} {'Params':>8}")
    print("-" * 60)

    # Sort by final_accuracy descending, then by acc_at_200
    sorted_keys = sorted(results.keys(),
                         key=lambda k: (results[k]['final_accuracy'], results[k]['acc_at_200']),
                         reverse=True)
    for k in sorted_keys:
        r = results[k]
        print(f"{k:<15} {r['acc_at_200']:>8.4f} {r['acc_at_600']:>8.4f} "
              f"{r['final_accuracy']:>8.4f} {r['final_loss']:>10.4f} {r['params']:>8,}")

    # Synergy analysis
    print(f"\nSynergy analysis (pair acc vs best single acc):")
    for k in sorted_keys:
        if '+' not in k:
            continue
        parts = k.split('+')
        single_accs = [results[p]['final_accuracy'] for p in parts if p in results]
        if not single_accs:
            continue
        best_single = max(single_accs)
        pair_acc = results[k]['final_accuracy']
        synergy = pair_acc - best_single
        speed_pair = results[k]['acc_at_200']
        speed_singles = [results[p]['acc_at_200'] for p in parts if p in results]
        speed_best = max(speed_singles) if speed_singles else 0
        speed_syn = speed_pair - speed_best
        label = "SYNERGY" if synergy > 0 else ("NEUTRAL" if synergy == 0 else "INTERFERENCE")
        print(f"  {k:<12} Δfinal={synergy:+.4f} Δspeed@200={speed_syn:+.4f} [{label}]")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'benchmark_pairwise_v53_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
