#!/usr/bin/env python3
"""
Phase D: Six-sources ablation on modular arithmetic (a + b mod 64).

Tests each v51 source's contribution with a small model (d_model=64, 2 layers).
Task: Z₂⁶ group operation — geometry should provide natural advantage.
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


def make_config(source_name):
    base = dict(vocab_size=65, d_model=64, n_layers=2, n_heads=4,
                block_size=8, dropout=0.0, use_rope=False, use_swiglu=True,
                hex_strength=0.05, temp=0.3)

    source_configs = {
        'vanilla': {},
        'source1_sklyarova': dict(use_palace_attention=True, use_graduated_biangua=True),
        'source2_fomyuk': dict(use_d4_equivariant=True, use_antipodal_reg=True, antipodal_weight=0.01),
        'source3_andreev': dict(use_triangular_bias=True, use_four_level_pe=True, use_bidirectional_tri=True),
        'source4_kasatkin': dict(use_dual_embedding=True, use_cube_diagonal=True, use_privileged_axis=True),
        'source5_hermann': dict(quantizer_type='factored6'),
        'source6_belyaev': dict(use_heisenberg_attention=True, use_flower_gat=True,
                                use_mobius_bias=True, use_structural_defect=True),
        'all_sources': dict(
            use_palace_attention=True, use_graduated_biangua=True,
            use_d4_equivariant=True, use_antipodal_reg=True, antipodal_weight=0.01,
            use_triangular_bias=True, use_four_level_pe=True, use_bidirectional_tri=True,
            use_dual_embedding=True, use_cube_diagonal=True, use_privileged_axis=True,
            use_heisenberg_attention=True, use_flower_gat=True,
            use_mobius_bias=True, use_structural_defect=True,
        ),
    }
    return YiJingConfig(**{**base, **source_configs[source_name]})


def train_and_eval(cfg, source_name, n_steps=3000, batch_size=256, lr=3e-4,
                   eval_every=200, device='cpu'):
    model = YiJingGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- {source_name} ({n_params:,} params) ---")

    history = {'step': [], 'loss': [], 'accuracy': []}
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
            history['step'].append(step)
            history['loss'].append(vloss)
            history['accuracy'].append(acc)
            if step <= 1000 or step % 1000 == 0:
                print(f"  Step {step:5d}: loss={vloss:.4f}, acc={acc:.4f}")

    elapsed = time.time() - t0
    print(f"  FINAL: acc={history['accuracy'][-1]:.4f}, loss={history['loss'][-1]:.4f}, time={elapsed:.1f}s")

    return {
        'source': source_name,
        'final_accuracy': history['accuracy'][-1],
        'final_loss': history['loss'][-1],
        'params': n_params, 'time': elapsed,
        'acc_at_200': history['accuracy'][1] if len(history['accuracy']) > 1 else 0,
        'acc_at_600': history['accuracy'][3] if len(history['accuracy']) > 3 else 0,
        'history': history,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\nTask: (a + b) mod 64")

    sources = ['vanilla', 'source1_sklyarova', 'source2_fomyuk', 'source3_andreev',
               'source4_kasatkin', 'source5_hermann', 'source6_belyaev', 'all_sources']

    results = {}
    for s in sources:
        results[s] = train_and_eval(make_config(s), s, device=device)

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY: Modular Addition mod 64")
    print("=" * 70)
    print(f"{'Source':<25} {'Acc@200':>8} {'Acc@600':>8} {'Final':>8} {'Loss':>8} {'Params':>9}")
    print("-" * 68)
    for s in sources:
        r = results[s]
        print(f"{s:<25} {r['acc_at_200']:>8.4f} {r['acc_at_600']:>8.4f} "
              f"{r['final_accuracy']:>8.4f} {r['final_loss']:>8.4f} {r['params']:>9,}")

    vanilla_acc = results['vanilla']['final_accuracy']
    print(f"\nΔ vs vanilla (acc@200 = convergence speed):")
    for s in sources:
        if s == 'vanilla': continue
        r = results[s]
        da = r['final_accuracy'] - vanilla_acc
        speed = r['acc_at_200'] - results['vanilla']['acc_at_200']
        print(f"  {s:<25} Δacc={da:+.4f}  Δspeed@200={speed:+.4f}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'ablation_six_sources_results.json')
    # Save without history for compact JSON
    compact = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in results.items()}
    with open(out_path, 'w') as f:
        json.dump(compact, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
