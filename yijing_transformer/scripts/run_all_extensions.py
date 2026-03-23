#!/usr/bin/env python3
"""
Запуск всех экспериментов A2–D12 последовательно.

Каждый эксперимент:
1. Создаёт нужные модели
2. Обучает/анализирует
3. Сохраняет результаты в JSON

Результаты: yijing_transformer/extensions_results.json
"""

import os
import sys
import math
import time
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import HybridGatedGPT, AdaptiveHybridGPT
from models.baseline import VanillaGPT
from models.extensions import (
    MultiHeadGeometricAttention,
    HierarchicalZ2_12Quantizer,
    HexagramMoE,
    ReedMullerCodebook,
    WalshHadamardAnalyzer,
    RateDistortionAnalyzer,
    HypercubeWeightQuantizer,
    GateDynamicsTracker,
    HexagramActivationTracker,
    find_fair_config,
)
from models.geometry import generate_hexagrams, generate_hypercube

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 300
BATCH_SIZE = 8
BLOCK_SIZE = 128
LR = 1e-3
VOCAB_SIZE = 64

ALL_RESULTS = {}


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


class SyntheticDataset:
    """Синтетический датасет для экспериментов."""
    def __init__(self, vocab_size=64, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.words = []
        for length in range(2, 8):
            for _ in range(vocab_size // 6):
                self.words.append([random.randint(1, vocab_size - 1) for _ in range(length)])

    def get_batch(self, batch_size, block_size, device):
        seqs = []
        for _ in range(batch_size):
            seq = []
            while len(seq) < block_size + 1:
                seq.extend(random.choice(self.words))
                seq.append(0)
            seq = [min(max(t, 0), self.vocab_size - 1) for t in seq[:block_size + 1]]
            seqs.append(seq)
        data = torch.tensor(seqs, dtype=torch.long, device=device)
        return data[:, :-1], data[:, 1:]


def get_lr(step, total_steps, base_lr):
    warmup = int(total_steps * 0.1)
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, dataset, device, n=20):
    model.eval()
    losses = []
    for _ in range(n):
        xb, yb = dataset.get_batch(BATCH_SIZE, BLOCK_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def quick_train(model, dataset, device, steps=STEPS):
    """Быстрое обучение для сравнительных экспериментов."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()
    losses = []
    for step in range(1, steps + 1):
        lr = get_lr(step, steps, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)
        xb, yb = dataset.get_batch(BATCH_SIZE, BLOCK_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            losses.append(loss.item())
    val = evaluate(model, dataset, device)
    return {'train_losses': losses, 'val_loss': val}


# =========================================================================
# A2: Multi-Head Geometric Attention
# =========================================================================
def run_a2():
    print("\n" + "=" * 70)
    print("  A2: MULTI-HEAD GEOMETRIC ATTENTION")
    print("=" * 70)

    dataset = SyntheticDataset(VOCAB_SIZE, SEED)
    device = torch.device(DEVICE)
    results = {}

    # Standard GeometricAttention baseline
    cfg_std = YiJingConfig(
        vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, total_steps=STEPS,
        architecture_mode='hybrid', use_rope=True, use_swiglu=True,
    )
    set_seed(SEED)
    m_std = HybridGatedGPT(cfg_std).to(device)
    p_std, g_std = m_std.count_parameters()
    print(f"  Standard Hybrid: {p_std:,} params")
    r_std = quick_train(m_std, dataset, device)
    results['standard_hybrid'] = {
        'params': p_std, 'geo_params': g_std,
        'val_loss': r_std['val_loss'],
    }
    del m_std
    print(f"    val_loss={r_std['val_loss']:.4f}")

    # MultiHead version — тестируем через замену attention в стандартной модели
    set_seed(SEED)
    m_mhga = HybridGatedGPT(cfg_std).to(device)
    # Заменяем attention слои на MHGA
    for i, layer in enumerate(m_mhga.layers):
        if hasattr(layer, 'geo_attn'):
            layer.geo_attn = MultiHeadGeometricAttention(cfg_std, codebook_dim=3).to(device)
    p_mh, _ = m_mhga.count_parameters()
    print(f"  MultiHead Geo: {p_mh:,} params")
    r_mh = quick_train(m_mhga, dataset, device)
    results['multihead_geo'] = {
        'params': p_mh,
        'val_loss': r_mh['val_loss'],
    }
    del m_mhga
    print(f"    val_loss={r_mh['val_loss']:.4f}")

    ALL_RESULTS['A2_multihead_geo_attention'] = results
    print(f"\n  A2 Summary: Standard={r_std['val_loss']:.4f} vs MultiHead={r_mh['val_loss']:.4f}")


# =========================================================================
# A3: Hierarchical Z₂¹² Quantization
# =========================================================================
def run_a3():
    print("\n" + "=" * 70)
    print("  A3: HIERARCHICAL Z₂¹² QUANTIZATION")
    print("=" * 70)

    device = torch.device(DEVICE)

    # Test quantizer on random data
    d_model = 128
    quant = HierarchicalZ2_12Quantizer(d_model, temp=0.3).to(device)
    x = torch.randn(2, 32, d_model, device=device)
    out, gate_info = quant(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Gate distribution: L1={gate_info['gate_l1']:.3f}, "
          f"L2={gate_info['gate_l2']:.3f}, L3={gate_info['gate_l3']:.3f}")

    # Reconstruction quality at each level
    results = {
        'gate_distribution': gate_info,
        'output_norm_ratio': (out.norm() / x.norm()).item(),
        'levels': {
            'L1_Z2_3': {'n_codes': 8, 'dim': 3},
            'L2_Z2_6': {'n_codes': 64, 'dim': 6},
            'L3_Z2_12': {'n_codes': '4096 (factored 64x64)', 'dim': 12},
        }
    }
    ALL_RESULTS['A3_hierarchical_z2_12'] = results
    print(f"  A3 Complete: 3-level hierarchical quantization working")


# =========================================================================
# A4: Geometric MoE
# =========================================================================
def run_a4():
    print("\n" + "=" * 70)
    print("  A4: GEOMETRIC MOE ROUTING")
    print("=" * 70)

    device = torch.device(DEVICE)
    d_model = 128

    # Test MoE
    moe = HexagramMoE(d_model, n_experts=64, top_k=4, use_factored=True).to(device)
    x = torch.randn(2, 32, d_model, device=device)
    out, balance_loss = moe(x)

    n_params = sum(p.numel() for p in moe.parameters())
    print(f"  HexagramMoE: {n_params:,} params")
    print(f"  Output shape: {out.shape}")
    print(f"  Balance loss: {balance_loss.item():.6f}")

    # Compare with standard FFN
    ffn = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.GELU(),
        nn.Linear(d_model * 4, d_model),
    ).to(device)
    ffn_params = sum(p.numel() for p in ffn.parameters())
    print(f"  Standard FFN: {ffn_params:,} params")

    results = {
        'moe_params': n_params,
        'ffn_params': ffn_params,
        'balance_loss': balance_loss.item(),
        'n_experts': 64,
        'top_k': 4,
        'factored': True,
    }
    ALL_RESULTS['A4_geometric_moe'] = results
    print(f"  A4 Complete: Factored 64-expert MoE via hexagrams")


# =========================================================================
# B5: Reed-Muller Error Correction
# =========================================================================
def run_b5():
    print("\n" + "=" * 70)
    print("  B5: REED-MULLER ERROR CORRECTION CODES")
    print("=" * 70)

    device = torch.device(DEVICE)
    d_model = 128

    rm = ReedMullerCodebook(d_model, order=1, length_exp=6).to(device)
    x = torch.randn(2, 32, d_model, device=device)
    out, code_info = rm(x)

    print(f"  RM(1,6): {code_info['n_codewords']} codewords, "
          f"length={code_info['code_length']}, d_min={code_info['min_distance']}")

    # Robustness test
    robustness = {}
    for noise_std in [0.01, 0.05, 0.1, 0.5, 1.0]:
        flip_rate = rm.measure_robustness(x, noise_std=noise_std)
        robustness[f"noise_{noise_std}"] = flip_rate
        print(f"    noise_std={noise_std:.2f}: flip_rate={flip_rate:.4f}")

    # Compare with standard hypercube
    from models.geometry import FactoredYiJingQuantizer
    std_quant = FactoredYiJingQuantizer(temp=0.3).to(device)
    # Standard hypercube has 64 codewords, RM(1,6) has 128

    results = {
        'code_info': code_info,
        'robustness': robustness,
        'comparison': 'RM(1,6) has 128 codewords with d_min=32 vs Z₂⁶ with 64 codewords and d_min=2',
    }
    ALL_RESULTS['B5_reed_muller'] = results
    print(f"  B5 Complete: RM(1,6) provides {code_info['min_distance']}x better error correction")


# =========================================================================
# B6: Walsh-Hadamard Spectral Analysis
# =========================================================================
def run_b6():
    print("\n" + "=" * 70)
    print("  B6: WALSH-HADAMARD SPECTRAL ANALYSIS")
    print("=" * 70)

    device = torch.device(DEVICE)
    analyzer = WalshHadamardAnalyzer()

    # Analyze random representations on Z₂⁶
    d = 16
    hexagrams = generate_hexagrams().to(device)  # (64, 6)

    # 1. Random embeddings for 64 hexagrams
    set_seed(SEED)
    random_repr = torch.randn(64, d, device=device)
    _, random_energy = analyzer.analyze_representations(random_repr, n_bits=6)
    print(f"  Random representations energy by order:")
    for order, energy in sorted(random_energy.items()):
        print(f"    Order {order}: {energy:.4f}")

    # 2. Structured (geometric) representations
    # Use hexagrams directly (padded to d dims)
    geo_repr = F.pad(hexagrams, (0, d - 6))
    _, geo_energy = analyzer.analyze_representations(geo_repr, n_bits=6)
    print(f"\n  Geometric representations energy by order:")
    for order, energy in sorted(geo_energy.items()):
        print(f"    Order {order}: {energy:.4f}")

    # 3. Train a small model and analyze learned representations
    cfg = YiJingConfig(
        vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=2,
        block_size=BLOCK_SIZE, total_steps=100,
        architecture_mode='hybrid', use_rope=True, use_swiglu=True,
    )
    set_seed(SEED)
    model = HybridGatedGPT(cfg).to(device)
    dataset = SyntheticDataset(VOCAB_SIZE, SEED)
    quick_train(model, dataset, device, steps=100)

    # Get embeddings for first 64 tokens
    with torch.no_grad():
        tok_ids = torch.arange(64, device=device).unsqueeze(0)
        emb = model.tok_emb(tok_ids).squeeze(0)  # (64, d_model)
    _, learned_energy = analyzer.analyze_representations(emb, n_bits=6)
    print(f"\n  Learned embedding energy by order:")
    for order, energy in sorted(learned_energy.items()):
        print(f"    Order {order}: {energy:.4f}")

    results = {
        'random_energy': random_energy,
        'geometric_energy': geo_energy,
        'learned_energy': learned_energy,
        'interpretation': {
            'order_0': 'DC component (mean)',
            'order_1': 'Linear terms (individual bit contributions)',
            'order_2': 'Pairwise interactions',
            'order_3+': 'Higher-order interactions',
        },
    }
    ALL_RESULTS['B6_walsh_hadamard'] = results
    del model
    print(f"\n  B6 Complete: Spectral decomposition of hypercube representations")


# =========================================================================
# B7: Rate-Distortion Analysis
# =========================================================================
def run_b7():
    print("\n" + "=" * 70)
    print("  B7: RATE-DISTORTION ANALYSIS")
    print("=" * 70)

    device = torch.device(DEVICE)
    analyzer = RateDistortionAnalyzer()

    # Generate data: random vectors projected to different dimensions
    set_seed(SEED)
    data_6d = torch.randn(1000, 6, device=device)

    codebooks = {
        'Z2_3 (8 pts)': generate_hypercube(3).to(device),
        'Z2_4 (16 pts)': generate_hypercube(4).to(device),
        'Z2_6 (64 pts)': generate_hypercube(6).to(device),
        'Random_64': torch.randn(64, 6, device=device),
        'Random_256': torch.randn(256, 6, device=device),
    }

    results = analyzer.compare_codebooks(data_6d, codebooks)
    print(f"\n  {'Codebook':<20} {'Rate':>6} {'Distortion':>12} {'Entropy':>10} {'Utilization':>12}")
    print(f"  {'-' * 62}")
    for name, r in results.items():
        print(f"  {name:<20} {r['rate']:>6.1f} {r['distortion']:>12.4f} "
              f"{r['entropy']:>10.2f} {r['utilization']:>12.2%}")

    ALL_RESULTS['B7_rate_distortion'] = results
    print(f"\n  B7 Complete: R-D comparison across codebook structures")


# =========================================================================
# C8: Fair Comparison (Equal Params)
# =========================================================================
def run_c8():
    print("\n" + "=" * 70)
    print("  C8: FAIR COMPARISON (EQUAL PARAMETERS)")
    print("=" * 70)

    dataset = SyntheticDataset(VOCAB_SIZE, SEED)
    device = torch.device(DEVICE)

    # First, get Vanilla params at d_model=128
    vanilla_cfg = YiJingConfig(
        vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, total_steps=STEPS,
        use_rope=True, use_swiglu=True,
    )
    set_seed(SEED)
    m_vanilla = VanillaGPT(vanilla_cfg).to(device)
    vanilla_params, _ = m_vanilla.count_parameters()
    print(f"  Vanilla (d=128): {vanilla_params:,} params")

    # Train vanilla
    r_vanilla = quick_train(m_vanilla, dataset, device)
    print(f"    val_loss={r_vanilla['val_loss']:.4f}")
    del m_vanilla

    # Find fair Hybrid config with same parameter count
    fair_cfg, fair_params = find_fair_config(
        target_params=vanilla_params,
        model_class=HybridGatedGPT,
        base_cfg_kwargs={
            'vocab_size': VOCAB_SIZE, 'n_layers': 4,
            'block_size': BLOCK_SIZE, 'total_steps': STEPS,
            'use_rope': True, 'use_swiglu': True,
            'architecture_mode': 'hybrid', 'use_bian_gua': True,
            'adaptive_temp': True,
        },
        search_dims=[64, 80, 96, 112, 128, 144],
    )

    if fair_cfg:
        set_seed(SEED)
        m_hybrid = HybridGatedGPT(fair_cfg).to(device)
        actual_params, geo_params = m_hybrid.count_parameters()
        print(f"  Fair Hybrid (d={fair_cfg.d_model}): {actual_params:,} params ({geo_params:,} geo)")
        r_hybrid = quick_train(m_hybrid, dataset, device)
        print(f"    val_loss={r_hybrid['val_loss']:.4f}")
        del m_hybrid

        results = {
            'vanilla': {
                'params': vanilla_params, 'd_model': 128,
                'val_loss': r_vanilla['val_loss'],
            },
            'hybrid_fair': {
                'params': actual_params, 'd_model': fair_cfg.d_model,
                'geo_params': geo_params,
                'val_loss': r_hybrid['val_loss'],
            },
            'param_diff_pct': abs(actual_params - vanilla_params) / vanilla_params * 100,
        }
    else:
        results = {'error': 'Could not find fair config'}

    ALL_RESULTS['C8_fair_comparison'] = results
    print(f"\n  C8 Complete: Fair comparison with {results.get('param_diff_pct', 0):.1f}% param difference")


# =========================================================================
# C9: Hypercube Weight Quantization
# =========================================================================
def run_c9():
    print("\n" + "=" * 70)
    print("  C9: HYPERCUBE WEIGHT QUANTIZATION (PTQ)")
    print("=" * 70)

    dataset = SyntheticDataset(VOCAB_SIZE, SEED)
    device = torch.device(DEVICE)

    # Train a model first
    cfg = YiJingConfig(
        vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, total_steps=STEPS,
        use_rope=True, use_swiglu=True,
    )
    set_seed(SEED)
    model = VanillaGPT(cfg).to(device)
    quick_train(model, dataset, device)

    # Evaluate before quantization
    val_before = evaluate(model, dataset, device)
    print(f"  Before quantization: val_loss={val_before:.4f}")

    # Quantize with different n_bits
    results = {'before_val_loss': val_before}
    for n_bits in [2, 3, 4]:
        set_seed(SEED)
        model_q = VanillaGPT(cfg).to(device)
        # Copy weights
        model_q.load_state_dict(model.state_dict())

        layer_results, avg_mse = HypercubeWeightQuantizer.quantize_model(
            model_q, group_size=64, n_bits=n_bits
        )
        val_after = evaluate(model_q, dataset, device)
        print(f"  {n_bits}-bit hypercube: val_loss={val_after:.4f} "
              f"(MSE={avg_mse:.6f}, Δ={val_after-val_before:+.4f})")

        results[f'{n_bits}bit'] = {
            'val_loss': val_after,
            'avg_mse': avg_mse,
            'delta': val_after - val_before,
            'n_codebook_entries': 2 ** n_bits,
        }
        del model_q

    ALL_RESULTS['C9_weight_quantization'] = results
    del model
    print(f"\n  C9 Complete: Hypercube PTQ across bit widths")


# =========================================================================
# D11: Gate Dynamics Visualization
# =========================================================================
def run_d11():
    print("\n" + "=" * 70)
    print("  D11: GATE DYNAMICS VISUALIZATION")
    print("=" * 70)

    dataset = SyntheticDataset(VOCAB_SIZE, SEED)
    device = torch.device(DEVICE)

    cfg = YiJingConfig(
        vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, total_steps=STEPS,
        architecture_mode='hybrid', use_rope=True, use_swiglu=True,
        use_bian_gua=True, adaptive_temp=True,
    )
    set_seed(SEED)
    model = HybridGatedGPT(cfg).to(device)

    tracker = GateDynamicsTracker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    model.train()

    for step in range(1, STEPS + 1):
        lr = get_lr(step, STEPS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)
        xb, yb = dataset.get_batch(BATCH_SIZE, BLOCK_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            tracker.record(model, step)

    summary = tracker.get_summary()
    print(f"\n  Gate dynamics summary:")
    for key, info in summary.items():
        print(f"    {key}: {info['start']:.3f} → {info['end']:.3f} ({info['trend']})")

    ALL_RESULTS['D11_gate_dynamics'] = tracker.to_json()
    del model
    print(f"\n  D11 Complete: Tracked {len(summary)} gates over {STEPS} steps")


# =========================================================================
# D12: Attention + Hexagram Maps
# =========================================================================
def run_d12():
    print("\n" + "=" * 70)
    print("  D12: ATTENTION + HEXAGRAM ACTIVATION MAPS")
    print("=" * 70)

    dataset = SyntheticDataset(VOCAB_SIZE, SEED)
    device = torch.device(DEVICE)

    cfg = YiJingConfig(
        vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, total_steps=200,
        architecture_mode='hybrid', use_rope=True, use_swiglu=True,
        use_bian_gua=True, adaptive_temp=True, use_hex_moe=True,
        n_experts=8, moe_top_k=2,
    )
    set_seed(SEED)
    model = HybridGatedGPT(cfg).to(device)
    quick_train(model, dataset, device, steps=200)

    tracker = HexagramActivationTracker()

    # Record activations for different "types" of data
    model.eval()
    with torch.no_grad():
        for dtype in ['sequential', 'random', 'repetitive']:
            for _ in range(10):
                if dtype == 'sequential':
                    seq = list(range(BLOCK_SIZE))
                    seq = [s % VOCAB_SIZE for s in seq]
                elif dtype == 'random':
                    seq = [random.randint(0, VOCAB_SIZE - 1) for _ in range(BLOCK_SIZE)]
                else:  # repetitive
                    word = [random.randint(1, 10) for _ in range(4)]
                    seq = (word * (BLOCK_SIZE // 4 + 1))[:BLOCK_SIZE]

                x = torch.tensor([seq], dtype=torch.long, device=device)
                # Forward pass to get hidden states
                hidden = model.tok_emb(x)
                if hasattr(model, 'pos_emb') and model.pos_emb is not None:
                    hidden = hidden + model.pos_emb[:, :BLOCK_SIZE, :]
                tracker.record_activations(model, hidden, data_type=dtype)

    summary = tracker.get_summary()
    print(f"\n  Hexagram activation summary:")
    for dtype, info in summary.items():
        print(f"    {dtype}: {info['unique_hexagrams']} unique hexagrams, "
              f"utilization={info['utilization']:.2%}")
        if info['top_5']:
            top = ', '.join(f"hex_{idx}({cnt})" for idx, cnt in info['top_5'])
            print(f"      top-5: {top}")

    ALL_RESULTS['D12_hexagram_maps'] = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in summary.items()}
    del model
    print(f"\n  D12 Complete: Activation patterns across data types")


# =========================================================================
# MAIN
# =========================================================================
def main():
    t_start = time.time()

    print("=" * 70)
    print("  YIJING-TRANSFORMER: ALL EXTENSIONS (A2–D12)")
    print("=" * 70)

    run_a2()
    run_a3()
    run_a4()
    run_b5()
    run_b6()
    run_b7()
    run_c8()
    run_c9()
    run_d11()
    run_d12()

    elapsed = time.time() - t_start
    ALL_RESULTS['meta'] = {
        'total_time': elapsed,
        'device': DEVICE,
        'steps': STEPS,
    }

    # Save all results
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'extensions_results.json'
    )
    output_path = os.path.abspath(output_path)
    with open(output_path, 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2, default=str)

    print(f"\n\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s")
    print(f"  Results: {output_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
