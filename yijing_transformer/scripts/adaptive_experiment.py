#!/usr/bin/env python3
"""
Фаза 3: Adaptive Specialization Experiment.

Сравнивает 4 архитектуры:
1. Vanilla (baseline)
2. HybridGated (Phase 1 — простые гейты)
3. AdaptiveHybrid (Phase 3 — адаптивные гейты + task-aware routing + multi-scale)
4. AdaptiveHybrid + DynamicCurriculum (Phase 3 — полная адаптация)

Тестирует на 3 типах данных:
- XOR-heavy (геометрия должна помогать)
- Periodic (стандартный attention должен быть лучше)
- Mixed (комбинация)
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
from models.model import HybridGatedGPT, AdaptiveHybridGPT
from models.baseline import VanillaGPT

SEED = 42
DEVICE = 'cpu'
STEPS = 500
BATCH_SIZE = 8
BLOCK_SIZE = 64
VOCAB_SIZE = 64
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def generate_xor_batch(batch_size, block_size, vocab_size, device):
    """XOR-heavy данные."""
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)
    seg = block_size // 3
    for i in range(batch_size):
        if 3 * seg <= block_size + 1:
            x[i, 2*seg:3*seg] = x[i, :seg] ^ x[i, seg:2*seg]
    return x[:, :-1], x[:, 1:]


def generate_periodic_batch(batch_size, block_size, vocab_size, device):
    """Periodic данные."""
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)
    for i in range(batch_size):
        period = random.choice([2, 4, 8])
        base = x[i, :period].clone()
        for j in range(0, block_size + 1, period):
            end = min(j + period, block_size + 1)
            x[i, j:end] = base[:end-j]
    return x[:, :-1], x[:, 1:]


def generate_mixed_batch(batch_size, block_size, vocab_size, device):
    """Mixed данные: XOR + periodic + modular + random."""
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)
    seg = block_size // 4
    for i in range(batch_size):
        t = i % 4
        if t == 0 and 3 * seg <= block_size + 1:  # XOR
            x[i, 2*seg:3*seg] = x[i, :seg] ^ x[i, seg:2*seg]
        elif t == 1:  # Periodic
            period = 4
            base = x[i, :period].clone()
            for j in range(0, block_size + 1, period):
                end = min(j + period, block_size + 1)
                x[i, j:end] = base[:end-j]
        elif t == 2 and 3 * seg <= block_size + 1:  # Modular
            x[i, 2*seg:3*seg] = (x[i, :seg] + x[i, seg:2*seg]) % vocab_size
    return x[:, :-1], x[:, 1:]


def get_lr(step, total_steps, base_lr=1e-3):
    warmup = int(total_steps * 0.1)
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, gen_func, device, n=30):
    model.eval()
    losses = []
    for _ in range(n):
        xb, yb = gen_func(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_model(model, name, gen_func, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    total_params, geo_params = model.count_parameters()
    print(f"  {name}: {total_params:,} params ({geo_params:,} geo)")

    best_val = float('inf')
    train_losses = []
    val_losses = []
    t_start = time.time()

    model.train()
    accum = 0.0

    for step in range(1, STEPS + 1):
        lr = get_lr(step, STEPS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)

        xb, yb = gen_func(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        accum += loss.item()

        if step % 50 == 0:
            train_losses.append(accum / 50)
            accum = 0.0

        if step % 100 == 0:
            vl = evaluate(model, gen_func, device)
            val_losses.append(vl)
            if vl < best_val:
                best_val = vl

    elapsed = time.time() - t_start

    # Gate summary
    gate_info = {}
    if hasattr(model, 'get_gate_summary'):
        summary = model.get_gate_summary()
        geo_count = 0
        total_gates = 0
        for layer_name, gates in summary.items():
            for gate_name, stats in gates.items():
                if 'gate_mean' in stats:
                    total_gates += 1
                    if stats.get('prefers_geometry', False):
                        geo_count += 1
        gate_info = {
            'geo_gates': geo_count,
            'total_gates': total_gates,
            'details': summary,
        }

    # Dynamic curriculum history
    curriculum_info = {}
    if hasattr(model, 'dynamic_curriculum'):
        curriculum_info = {
            'final_strength': model.dynamic_curriculum.current_strength,
            'history_len': len(model.dynamic_curriculum.history),
        }

    print(f"    best_val={best_val:.4f} | time={elapsed:.1f}s | "
          f"geo_gates={gate_info.get('geo_gates', '-')}/{gate_info.get('total_gates', '-')}")

    return {
        'params': total_params,
        'geo_params': geo_params,
        'best_val': best_val,
        'train_curve': train_losses,
        'val_curve': val_losses,
        'elapsed_s': elapsed,
        'gate_info': gate_info,
        'curriculum_info': curriculum_info,
    }


def run_experiment():
    device = torch.device(DEVICE)

    print("=" * 70)
    print("  PHASE 3: ADAPTIVE SPECIALIZATION EXPERIMENT")
    print("=" * 70)

    data_types = {
        'xor_heavy': generate_xor_batch,
        'periodic': generate_periodic_batch,
        'mixed': generate_mixed_batch,
    }

    all_results = {}

    for data_name, gen_func in data_types.items():
        print(f"\n{'▓' * 70}")
        print(f"  DATA: {data_name}")
        print(f"{'▓' * 70}")

        base_cfg = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
            use_rope=True, use_swiglu=True, use_bian_gua=True,
            adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
        )

        results = {}

        # 1. Vanilla
        set_seed(SEED)
        m = VanillaGPT(base_cfg).to(device)
        results['vanilla'] = train_model(m, "Vanilla", gen_func, device)
        del m

        # 2. HybridGated (Phase 1)
        hybrid_cfg = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
            use_rope=True, use_swiglu=True, use_bian_gua=True,
            adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
            architecture_mode='hybrid', gate_init_bias=0.0,
            curriculum_strategy_geo='linear',
            curriculum_target_strength=0.1,
        )
        set_seed(SEED)
        m = HybridGatedGPT(hybrid_cfg).to(device)
        results['hybrid_gated'] = train_model(m, "HybridGated", gen_func, device)
        del m

        # 3. AdaptiveHybrid (Phase 3)
        adaptive_cfg = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
            use_rope=True, use_swiglu=True, use_bian_gua=True,
            adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
            architecture_mode='hybrid', gate_init_bias=0.0,
            curriculum_strategy_geo='none',
            curriculum_target_strength=0.1,
        )
        set_seed(SEED)
        m = AdaptiveHybridGPT(adaptive_cfg).to(device)
        results['adaptive_hybrid'] = train_model(m, "AdaptiveHybrid", gen_func, device)
        del m

        all_results[data_name] = results

    # ==================== SUMMARY ====================
    print(f"\n\n{'=' * 70}")
    print("  ADAPTIVE SPECIALIZATION RESULTS")
    print(f"{'=' * 70}")

    for data_name in data_types:
        print(f"\n  {data_name.upper()}:")
        print(f"  {'Model':<20} {'Params':>10} {'BestVal':>10} {'GeoGates':>10}")
        print(f"  {'-' * 52}")
        for mode in ['vanilla', 'hybrid_gated', 'adaptive_hybrid']:
            r = all_results[data_name][mode]
            gg = f"{r['gate_info'].get('geo_gates', '-')}/{r['gate_info'].get('total_gates', '-')}"
            print(f"  {mode:<20} {r['params']:>10,} {r['best_val']:>10.4f} {gg:>10}")

    # Task-aware analysis
    print(f"\n  TASK-AWARE ROUTING ANALYSIS:")
    for data_name in data_types:
        adaptive = all_results[data_name].get('adaptive_hybrid', {})
        if 'gate_info' in adaptive and 'details' in adaptive['gate_info']:
            details = adaptive['gate_info']['details']
            for layer, gates in details.items():
                router = gates.get('task_router', {})
                if router:
                    probs = [f"{router.get(f'strategy_{i}', 0):.2f}" for i in range(4)]
                    print(f"    {data_name}/{layer}: strategies=[{', '.join(probs)}]")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'adaptive_experiment_results.json')
    output_path = os.path.abspath(output_path)

    save_data = {}
    for data_name, modes in all_results.items():
        save_data[data_name] = {}
        for mode, data in modes.items():
            save_data[data_name][mode] = {
                k: v for k, v in data.items()
                if k != 'gate_info' or not isinstance(v, dict) or 'details' not in v
            }
            # Simplified gate info
            if 'gate_info' in data:
                save_data[data_name][mode]['gate_summary'] = {
                    'geo_gates': data['gate_info'].get('geo_gates', 0),
                    'total_gates': data['gate_info'].get('total_gates', 0),
                }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
