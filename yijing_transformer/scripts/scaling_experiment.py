#!/usr/bin/env python3
"""
Фаза 2: Scaling Experiment — проверка масштабирования геометрии.

Тестируем 3 размера модели (tiny/small/medium) × 3 режима (vanilla/pure_geo/hybrid)
для ответа на вопрос: сохраняется ли преимущество геометрии при увеличении масштаба?

Гипотеза: на больших масштабах геометрия может показать большее преимущество.
"""

import os
import sys
import math
import time
import json
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT, PureGeometricGPT, HybridGatedGPT
from models.baseline import VanillaGPT


# ==================== КОНФИГУРАЦИЯ ====================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 500  # шагов на каждый размер (меньше чем ablation — больше конфигов)
WARMUP_FRAC = 0.1
LOG_EVERY = 50
VAL_EVERY = 100
BATCH_SIZE = 8
VOCAB_SIZE = 256

# Три размера модели
SCALES = {
    'tiny': {'d_model': 64, 'n_layers': 2, 'n_heads': 2, 'block_size': 32},
    'small': {'d_model': 128, 'n_layers': 4, 'n_heads': 4, 'block_size': 64},
    'medium': {'d_model': 256, 'n_layers': 6, 'n_heads': 8, 'block_size': 64},
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def generate_batch(batch_size, block_size, vocab_size, device):
    """Синтетические данные с паттернами."""
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)
    for i in range(batch_size):
        pattern_type = i % 4
        if pattern_type == 0:  # XOR pattern
            seg = block_size // 3
            if 2 * seg + seg <= block_size + 1:
                x[i, 2*seg:3*seg] = x[i, :seg] ^ x[i, seg:2*seg]
        elif pattern_type == 1:  # Modular addition
            seg = block_size // 3
            if 2 * seg + seg <= block_size + 1:
                x[i, 2*seg:3*seg] = (x[i, :seg] + x[i, seg:2*seg]) % vocab_size
        elif pattern_type == 2:  # Copy
            seg = block_size // 4
            dst = block_size // 2
            if dst + seg <= block_size + 1:
                x[i, dst:dst+seg] = x[i, :seg]
    return x[:, :-1], x[:, 1:]


def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, block_size, device, num_batches=30):
    model.eval()
    losses = []
    for _ in range(num_batches):
        xb, yb = generate_batch(BATCH_SIZE, block_size, VOCAB_SIZE, device)
        result = model(xb, yb)
        if isinstance(result, tuple):
            loss = result[1] if len(result) >= 2 else result[0]
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_one(model, model_name, block_size, device):
    """Train one model, return metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    total_params, geo_params = model.count_parameters()
    warmup = int(STEPS * WARMUP_FRAC)

    print(f"  {model_name}: {total_params:,} params ({geo_params:,} geo)")

    best_val = float('inf')
    train_losses = []
    val_losses = []
    step_times = []

    model.train()
    accum = 0.0

    for step in range(1, STEPS + 1):
        t0 = time.time()
        lr = get_lr(step, STEPS, warmup, 1e-3)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)

        xb, yb = generate_batch(BATCH_SIZE, block_size, VOCAB_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        step_times.append(dt)
        accum += loss.item()

        if step % LOG_EVERY == 0:
            avg = accum / LOG_EVERY
            train_losses.append(avg)
            accum = 0.0

        if step % VAL_EVERY == 0:
            vl = evaluate(model, block_size, device)
            val_losses.append(vl)
            if vl < best_val:
                best_val = vl

    # Gate analysis for hybrid
    gate_info = {}
    if hasattr(model, 'get_gate_summary'):
        summary = model.get_gate_summary()
        geo_count = 0
        total_gates = 0
        for layer_name, gates in summary.items():
            for gate_name, stats in gates.items():
                total_gates += 1
                if stats['gate_mean'] > 0.5:
                    geo_count += 1
        gate_info = {'geo_gates': geo_count, 'total_gates': total_gates}

    avg_ms = sum(step_times) / len(step_times) * 1000

    print(f"    best_val={best_val:.4f} | {avg_ms:.1f}ms/step | "
          f"geo_gates={gate_info.get('geo_gates', '-')}/{gate_info.get('total_gates', '-')}")

    return {
        'params': total_params,
        'geo_params': geo_params,
        'best_val': best_val,
        'final_train': train_losses[-1] if train_losses else None,
        'avg_ms': avg_ms,
        'train_curve': train_losses,
        'val_curve': val_losses,
        'gate_info': gate_info,
    }


def run_scaling():
    device = torch.device(DEVICE)

    print("=" * 70)
    print("  PHASE 2: SCALING EXPERIMENT")
    print("  tiny → small → medium × {vanilla, pure_geo, hybrid}")
    print("=" * 70)

    all_results = {}

    for scale_name, scale_cfg in SCALES.items():
        d = scale_cfg['d_model']
        L = scale_cfg['n_layers']
        H = scale_cfg['n_heads']
        B = scale_cfg['block_size']

        print(f"\n{'▓' * 70}")
        print(f"  SCALE: {scale_name} (d={d}, L={L}, H={H}, block={B})")
        print(f"{'▓' * 70}")

        base = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=d, n_layers=L, n_heads=H,
            block_size=B, batch_size=BATCH_SIZE,
            use_rope=True, use_swiglu=True, use_bian_gua=True,
            adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
        )

        scale_results = {}

        # 1. Vanilla
        set_seed(SEED)
        vanilla = VanillaGPT(base).to(device)
        scale_results['vanilla'] = train_one(vanilla, "Vanilla", B, device)
        del vanilla
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 2. Pure Geometry
        set_seed(SEED)
        pure = PureGeometricGPT(base).to(device)
        scale_results['pure_geometry'] = train_one(pure, "PureGeo", B, device)
        del pure
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 3. Hybrid
        hybrid_cfg = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=d, n_layers=L, n_heads=H,
            block_size=B, batch_size=BATCH_SIZE,
            use_rope=True, use_swiglu=True, use_bian_gua=True,
            adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
            architecture_mode='hybrid', gate_init_bias=0.0,
            curriculum_strategy_geo='linear',
            curriculum_warmup_fraction=0.3,
            curriculum_target_strength=0.1,
        )

        set_seed(SEED)
        hybrid = HybridGatedGPT(hybrid_cfg).to(device)
        scale_results['hybrid'] = train_one(hybrid, "Hybrid", B, device)
        del hybrid
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        all_results[scale_name] = scale_results

    # ==================== SUMMARY ====================
    print(f"\n\n{'=' * 70}")
    print("  SCALING RESULTS SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Scale':<10} {'Mode':<12} {'Params':>10} {'BestVal':>10} {'ms/step':>10} {'GeoGates':>10}")
    print(f"  {'-' * 64}")

    for scale_name in ['tiny', 'small', 'medium']:
        for mode in ['vanilla', 'pure_geometry', 'hybrid']:
            r = all_results[scale_name][mode]
            gg = f"{r['gate_info'].get('geo_gates', '-')}/{r['gate_info'].get('total_gates', '-')}"
            print(f"  {scale_name:<10} {mode:<12} {r['params']:>10,} {r['best_val']:>10.4f} "
                  f"{r['avg_ms']:>10.1f} {gg:>10}")
        print()

    # Scaling analysis
    print(f"\n  SCALING ANALYSIS:")
    for mode in ['vanilla', 'pure_geometry', 'hybrid']:
        vals = [all_results[s][mode]['best_val'] for s in ['tiny', 'small', 'medium']]
        params = [all_results[s][mode]['params'] for s in ['tiny', 'small', 'medium']]
        print(f"  {mode}: loss {vals[0]:.4f} → {vals[1]:.4f} → {vals[2]:.4f}")
        # Scaling efficiency: loss reduction per log-params
        if vals[0] > vals[-1] and params[0] < params[-1]:
            import math
            eff = (vals[0] - vals[-1]) / math.log(params[-1] / params[0])
            print(f"    Efficiency: {eff:.6f} loss/log(params)")

    # Geometry advantage by scale
    print(f"\n  GEOMETRY ADVANTAGE (Hybrid - Vanilla):")
    for scale_name in ['tiny', 'small', 'medium']:
        v = all_results[scale_name]['vanilla']['best_val']
        h = all_results[scale_name]['hybrid']['best_val']
        delta = v - h  # positive = hybrid better
        pct = delta / v * 100 if v > 0 else 0
        trend = "BETTER" if delta > 0 else "WORSE"
        print(f"  {scale_name}: Δ = {delta:+.4f} ({pct:+.2f}%) — Hybrid {trend}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'scaling_results.json')
    output_path = os.path.abspath(output_path)

    # Clean for JSON
    save_data = {}
    for scale, modes in all_results.items():
        save_data[scale] = {}
        for mode, data in modes.items():
            save_data[scale][mode] = {k: v for k, v in data.items()}

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    run_scaling()
