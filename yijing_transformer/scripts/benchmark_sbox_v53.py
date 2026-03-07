#!/usr/bin/env python3
"""
Direction B: Crypto S-box experiment — Z₂⁶ domain where geometry is natural.

Task: Learn an AES-like S-box mapping (bijection on {0..255}).
The S-box is a nonlinear substitution — geometry on {-1,+1}⁶ should help
because the S-box is designed to maximize bit-diffusion properties.

Also tests XOR task (pure Z₂ group operation) as sanity check.
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT


# Simplified AES S-box (first 64 entries for Z₂⁶ compatibility)
AES_SBOX_256 = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]

# Project to mod 64 for Z₂⁶
SBOX_64 = [AES_SBOX_256[i] % 64 for i in range(64)]


def generate_sbox_data(n_samples, sbox, mod=64, device='cpu'):
    """Task: given input x, predict S-box(x). Sequence: [x, SEP] -> [_, target]."""
    x = torch.randint(0, mod, (n_samples,), device=device)
    y = torch.tensor([sbox[xi.item()] for xi in x], device=device)
    SEP = mod
    inputs = torch.stack([x, torch.full_like(x, SEP)], dim=1)
    targets = torch.stack([torch.full_like(x, -100), y], dim=1)
    return inputs, targets


def generate_xor_data(n_samples, mod=64, device='cpu'):
    """Task: a XOR b. Pure Z₂⁶ group operation."""
    a = torch.randint(0, mod, (n_samples,), device=device)
    b = torch.randint(0, mod, (n_samples,), device=device)
    c = a ^ b
    SEP = mod
    inputs = torch.stack([a, b, torch.full_like(a, SEP)], dim=1)
    targets = torch.stack([torch.full_like(a, -100), torch.full_like(a, -100), c], dim=1)
    return inputs, targets


def generate_double_sbox_data(n_samples, sbox, mod=64, device='cpu'):
    """Task: S(a XOR b). Composition of Z₂⁶ operation + nonlinear substitution."""
    a = torch.randint(0, mod, (n_samples,), device=device)
    b = torch.randint(0, mod, (n_samples,), device=device)
    c = torch.tensor([sbox[(ai ^ bi).item()] for ai, bi in zip(a, b)], device=device)
    SEP = mod
    inputs = torch.stack([a, b, torch.full_like(a, SEP)], dim=1)
    targets = torch.stack([torch.full_like(a, -100), torch.full_like(a, -100), c], dim=1)
    return inputs, targets


def make_config(source_name, task_type='sbox'):
    mod = 64
    vocab = mod + 1  # +1 for SEP token
    seq_len = 3 if task_type in ('xor', 'double_sbox') else 2
    base = dict(
        vocab_size=vocab, d_model=64, n_layers=2, n_heads=4,
        block_size=8, dropout=0.0, use_rope=False, use_swiglu=True,
        hex_strength=0.05, temp=0.3,
    )
    source_configs = {
        'vanilla': {},
        'fomyuk': dict(use_d4_equivariant=True, use_antipodal_reg=True, antipodal_weight=0.01),
        'belyaev': dict(use_heisenberg_attention=True, use_flower_gat=True,
                        use_mobius_bias=True, use_structural_defect=True),
        'fomyuk_belyaev': dict(
            use_d4_equivariant=True, use_antipodal_reg=True, antipodal_weight=0.01,
            use_heisenberg_attention=True, use_flower_gat=True,
            use_mobius_bias=True, use_structural_defect=True,
        ),
    }
    return YiJingConfig(**{**base, **source_configs[source_name]})


def train_task(task_name, data_fn, sources, n_steps=3000, batch_size=256,
               lr=3e-4, eval_every=200, device='cpu'):
    """Train all sources on a single task."""
    print(f"\n{'=' * 60}")
    print(f"Task: {task_name}")
    print(f"{'=' * 60}")

    results = {}
    for s in sources:
        task_type = 'sbox' if 'sbox' in task_name and 'xor' not in task_name else 'xor'
        cfg = make_config(s, task_type)
        model = YiJingGPT(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  {s} ({n_params:,} params)")

        history = {'step': [], 'loss': [], 'accuracy': []}
        t0 = time.time()
        pred_pos = -1  # last position in sequence

        for step in range(1, n_steps + 1):
            model.train()
            inputs, targets = data_fn(batch_size, device=device)
            logits, loss_base, _ = model(inputs, targets=targets)
            loss = F.cross_entropy(logits[:, pred_pos, :], targets[:, pred_pos])
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
                    vi, vt = data_fn(1024, device=device)
                    vl, _, _ = model(vi)
                    preds = vl[:, pred_pos, :].argmax(dim=-1)
                    acc = (preds == vt[:, pred_pos]).float().mean().item()
                    vloss = F.cross_entropy(vl[:, pred_pos, :], vt[:, pred_pos]).item()
                history['step'].append(step)
                history['loss'].append(vloss)
                history['accuracy'].append(acc)
                if step <= 600 or step % 1000 == 0 or step == n_steps:
                    print(f"    Step {step:5d}: loss={vloss:.4f} acc={acc:.4f}")

        elapsed = time.time() - t0
        print(f"    FINAL: acc={history['accuracy'][-1]:.4f} loss={history['loss'][-1]:.4f} time={elapsed:.1f}s")

        results[s] = {
            'source': s,
            'final_accuracy': history['accuracy'][-1],
            'final_loss': history['loss'][-1],
            'params': n_params, 'time': elapsed,
            'acc_at_200': history['accuracy'][1] if len(history['accuracy']) > 1 else 0,
            'acc_at_600': history['accuracy'][3] if len(history['accuracy']) > 3 else 0,
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 70)
    print("Direction B: Crypto S-box Domain Experiment (v53)")
    print("=" * 70)

    sources = ['vanilla', 'fomyuk', 'belyaev', 'fomyuk_belyaev']

    all_results = {}

    # Task 1: S-box lookup (memorization + nonlinear structure)
    sbox_fn = lambda n, device='cpu': generate_sbox_data(n, SBOX_64, device=device)
    all_results['sbox'] = train_task('S-box lookup (64→64)', sbox_fn, sources, device=device)

    # Task 2: XOR (pure Z₂⁶)
    all_results['xor'] = train_task('XOR (a ⊕ b)', generate_xor_data, sources, device=device)

    # Task 3: S(a XOR b) — composition
    dsbox_fn = lambda n, device='cpu': generate_double_sbox_data(n, SBOX_64, device=device)
    all_results['sbox_xor'] = train_task('S(a ⊕ b) composition', dsbox_fn, sources, device=device)

    # Grand summary
    print("\n\n" + "=" * 70)
    print("CRYPTO S-BOX BENCHMARK SUMMARY")
    print("=" * 70)
    for task_name, task_results in all_results.items():
        print(f"\n  Task: {task_name}")
        print(f"  {'Source':<20} {'Acc@200':>8} {'Acc@600':>8} {'Final':>8} {'Loss':>8}")
        print(f"  {'-' * 55}")
        for s in sources:
            r = task_results[s]
            print(f"  {s:<20} {r['acc_at_200']:>8.4f} {r['acc_at_600']:>8.4f} "
                  f"{r['final_accuracy']:>8.4f} {r['final_loss']:>8.4f}")

        vanilla_acc = task_results['vanilla']['final_accuracy']
        for s in sources:
            if s == 'vanilla':
                continue
            r = task_results[s]
            da = r['final_accuracy'] - vanilla_acc
            speed = r['acc_at_200'] - task_results['vanilla']['acc_at_200']
            print(f"    Δ {s}: final={da:+.4f} speed@200={speed:+.4f}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'benchmark_sbox_v53_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
