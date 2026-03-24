"""
Эксперимент: геометрия на данных с явными симметриями.

Гипотеза: геометрические компоненты Yi Jing (гиперкубы, триграммы)
должны лучше работать на данных с бинарными симметриями,
периодическими паттернами и групповыми структурами.

Типы данных:
1. Binary reflection: f(x) = bitflip(x) — прямая аналогия с BianGua
2. Modular arithmetic: c = (a + b) mod N — групповая операция
3. Periodic patterns: повторение с периодом 2^k (гиперкубная структура)
4. XOR patterns: a XOR b — основная операция в Z₂
5. Palindrome: зеркальная симметрия последовательности
6. Rotation: циклический сдвиг (связь с вращениями гиперкуба)

Per-pattern gate analysis: для каждого типа паттерна отслеживаем,
какие гейты модель предпочитает — это покажет, ГДЕ геометрия полезна.
"""

import os
import sys
import math
import time
import json
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT, PureGeometricGPT, HybridGatedGPT
from models.baseline import VanillaGPT


# ==================== КОНФИГУРАЦИЯ ====================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 1000
WARMUP_STEPS = 100
LOG_EVERY = 50
VAL_EVERY = 100
BATCH_SIZE = 8
BLOCK_SIZE = 64
VOCAB_SIZE = 64   # Уменьшенный словарь для чётких бинарных паттернов
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
LR = 1e-3

# 7 типов паттернов
PATTERN_NAMES = [
    'binary_reflection',
    'modular_addition',
    'periodic_2k',
    'xor_pattern',
    'palindrome',
    'rotation',
    'random',
]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


# ==================== ГЕНЕРАТОРЫ ДАННЫХ С СИММЕТРИЯМИ ====================

def generate_binary_reflection(batch_size, block_size, vocab_size):
    """Binary reflection: вторая половина = bitflip первой.
    Прямая аналогия с BianGua (变卦) — инверсия линий гексаграммы."""
    half = block_size // 2
    x = torch.randint(0, vocab_size, (batch_size, half))
    # Bitflip: XOR с маской всех единиц
    mask = vocab_size - 1
    reflected = x ^ mask
    seq = torch.cat([x, reflected], dim=1)
    return seq[:, :block_size]


def generate_modular_addition(batch_size, block_size, vocab_size):
    """Modular addition: c[i] = (a[i] + b[i]) mod vocab_size.
    Групповая операция Z_N — фундамент алгебраической структуры."""
    third = block_size // 3
    a = torch.randint(0, vocab_size, (batch_size, third))
    b = torch.randint(0, vocab_size, (batch_size, third))
    c = (a + b) % vocab_size
    # Паддинг до block_size
    rest = block_size - 3 * third
    pad = torch.randint(0, vocab_size, (batch_size, rest))
    return torch.cat([a, b, c, pad], dim=1)[:, :block_size]


def generate_periodic_2k(batch_size, block_size, vocab_size):
    """Периодические паттерны с периодом 2^k.
    Структура гиперкуба: 2^k вершин → период 2^k."""
    k = random.choice([2, 3, 4])  # период 4, 8, или 16
    period = 2 ** k
    base = torch.randint(0, vocab_size, (batch_size, period))
    repeats = (block_size + period - 1) // period
    seq = base.repeat(1, repeats)
    return seq[:, :block_size]


def generate_xor_pattern(batch_size, block_size, vocab_size):
    """XOR pattern: c[i] = a[i] XOR b[i].
    Основная операция в Z₂ — прямо соответствует координатам гиперкуба."""
    third = block_size // 3
    # Ограничиваем vocab_size степенью двойки для корректного XOR
    bits = int(math.log2(vocab_size)) if vocab_size > 1 else 1
    v = 2 ** bits
    a = torch.randint(0, v, (batch_size, third))
    b = torch.randint(0, v, (batch_size, third))
    c = a ^ b
    rest = block_size - 3 * third
    pad = torch.randint(0, v, (batch_size, rest))
    return torch.cat([a, b, c, pad], dim=1)[:, :block_size]


def generate_palindrome(batch_size, block_size, vocab_size):
    """Palindrome: вторая половина = reverse первой.
    Зеркальная симметрия — инволюция."""
    half = block_size // 2
    first = torch.randint(0, vocab_size, (batch_size, half))
    second = first.flip(dims=[1])
    return torch.cat([first, second], dim=1)[:, :block_size]


def generate_rotation(batch_size, block_size, vocab_size):
    """Rotation: циклический сдвиг подпоследовательности.
    Связь с вращениями в пространстве гиперкуба."""
    seg_len = block_size // 3
    base = torch.randint(0, vocab_size, (batch_size, seg_len))
    shift = random.randint(1, max(1, seg_len - 1))
    rotated = torch.roll(base, shifts=shift, dims=1)
    rest = block_size - 2 * seg_len
    pad = torch.randint(0, vocab_size, (batch_size, rest))
    return torch.cat([base, rotated, pad], dim=1)[:, :block_size]


def generate_random(batch_size, block_size, vocab_size):
    """Случайные данные — baseline без структуры."""
    return torch.randint(0, vocab_size, (batch_size, block_size))


GENERATORS = {
    'binary_reflection': generate_binary_reflection,
    'modular_addition': generate_modular_addition,
    'periodic_2k': generate_periodic_2k,
    'xor_pattern': generate_xor_pattern,
    'palindrome': generate_palindrome,
    'rotation': generate_rotation,
    'random': generate_random,
}


def generate_mixed_batch(batch_size, block_size, vocab_size, device):
    """Смешанный батч: каждый элемент — случайный тип паттерна."""
    sequences = []
    pattern_types = []
    target_len = block_size + 1
    for i in range(batch_size):
        ptype = PATTERN_NAMES[i % len(PATTERN_NAMES)]
        seq = GENERATORS[ptype](1, target_len, vocab_size)
        # Гарантируем одинаковую длину
        if seq.size(1) < target_len:
            pad = torch.randint(0, vocab_size, (1, target_len - seq.size(1)))
            seq = torch.cat([seq, pad], dim=1)
        seq = seq[:, :target_len]
        sequences.append(seq)
        pattern_types.append(ptype)
    x = torch.cat(sequences, dim=0).to(device)
    return x[:, :-1], x[:, 1:], pattern_types


def generate_single_pattern_batch(pattern_name, batch_size, block_size, vocab_size, device):
    """Батч из одного типа паттерна для изолированного тестирования."""
    seq = GENERATORS[pattern_name](batch_size, block_size + 1, vocab_size).to(device)
    return seq[:, :-1], seq[:, 1:]


# ==================== LR SCHEDULE ====================

def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ==================== ОБУЧЕНИЕ ====================

def train_model(model, model_name, device, log_gates=False):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
    )

    total_params, geo_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"  {model_name}: {total_params:,} params", end="")
    if geo_params > 0:
        print(f" (geo: {geo_params:,} = {100*geo_params/total_params:.1f}%)")
    else:
        print()
    print(f"{'='*60}")

    history = {
        'model': model_name,
        'total_params': total_params,
        'geo_params': geo_params,
        'train_loss': [],
        'val_loss': [],
        'steps': [],
        'val_steps': [],
        'best_val_loss': float('inf'),
        'avg_time_per_step': 0,
    }

    model.train()
    accum_loss = 0.0
    step_times = []

    for step in range(1, TOTAL_STEPS + 1):
        t0 = time.time()
        lr = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)

        xb, yb, _ = generate_mixed_batch(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        result = model(xb, yb)
        loss = result[1]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        step_times.append(dt)
        accum_loss += loss.item()

        if step % LOG_EVERY == 0:
            avg_loss = accum_loss / LOG_EVERY
            history['train_loss'].append(avg_loss)
            history['steps'].append(step)
            print(f"  Step {step:4d} | loss={avg_loss:.4f} | lr={lr:.2e} | {sum(step_times[-LOG_EVERY:])/LOG_EVERY*1000:.0f}ms")
            accum_loss = 0.0

        if log_gates and step % 100 == 0 and hasattr(model, 'log_gates'):
            model.log_gates(step)

        if step % VAL_EVERY == 0:
            val_loss = evaluate_mixed(model, device)
            history['val_loss'].append(val_loss)
            history['val_steps'].append(step)
            marker = " *" if val_loss < history['best_val_loss'] else ""
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
            print(f"  >>> VAL={val_loss:.4f}{marker}")

    history['avg_time_per_step'] = sum(step_times) / len(step_times)
    return history


@torch.no_grad()
def evaluate_mixed(model, device, num_batches=30):
    model.eval()
    losses = []
    for _ in range(num_batches):
        xb, yb, _ = generate_mixed_batch(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        result = model(xb, yb)
        losses.append(result[1].item())
    model.train()
    return sum(losses) / len(losses)


# ==================== PER-PATTERN EVALUATION ====================

@torch.no_grad()
def evaluate_per_pattern(model, device, num_batches=50):
    """Оценка loss отдельно для каждого типа паттерна."""
    model.eval()
    results = {}
    for pname in PATTERN_NAMES:
        losses = []
        for _ in range(num_batches):
            xb, yb = generate_single_pattern_batch(pname, BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
            result = model(xb, yb)
            losses.append(result[1].item())
        results[pname] = sum(losses) / len(losses)
    model.train()
    return results


@torch.no_grad()
def evaluate_per_pattern_gates(model, device, num_batches=30):
    """Для Hybrid модели: какие гейты активируются на каждом типе паттерна."""
    model.eval()
    results = {}

    for pname in PATTERN_NAMES:
        gate_accum = defaultdict(lambda: defaultdict(list))

        for _ in range(num_batches):
            xb, yb = generate_single_pattern_batch(pname, BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
            _ = model(xb, yb)

            # Собираем гейты
            for i, layer in enumerate(model.layers):
                stats = layer.get_all_gate_stats()
                for gate_name, gate_stats in stats.items():
                    gate_accum[f'layer_{i}'][gate_name].append(gate_stats['gate_mean'])

        # Усредняем
        pattern_gates = {}
        for layer_name, gates in gate_accum.items():
            pattern_gates[layer_name] = {}
            for gate_name, values in gates.items():
                avg = sum(values) / len(values)
                pattern_gates[layer_name][gate_name] = {
                    'mean': avg,
                    'prefers_geometry': avg > 0.5,
                }
        results[pname] = pattern_gates

    model.train()
    return results


# ==================== ОСНОВНОЙ ЭКСПЕРИМЕНТ ====================

def run_symmetry_experiment():
    device = torch.device(DEVICE)

    print("=" * 70)
    print("  SYMMETRY EXPERIMENT: Where Does Geometry Excel?")
    print("=" * 70)
    print(f"  Config: d={D_MODEL}, L={N_LAYERS}, H={N_HEADS}, V={VOCAB_SIZE}, steps={TOTAL_STEPS}")
    print(f"  Patterns: {', '.join(PATTERN_NAMES)}")

    base_cfg = YiJingConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        use_rope=True,
        use_swiglu=True,
        use_bian_gua=True,
        adaptive_temp=True,
        hex_strength=0.01,
        total_steps=TOTAL_STEPS,
    )

    hybrid_cfg = YiJingConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        use_rope=True,
        use_swiglu=True,
        use_bian_gua=True,
        adaptive_temp=True,
        hex_strength=0.01,
        total_steps=TOTAL_STEPS,
        architecture_mode='hybrid',
        gate_init_bias=0.0,
        curriculum_strategy_geo='linear',
        curriculum_target_strength=0.1,
    )

    results = {}

    # === Vanilla ===
    print("\n" + "▓" * 70)
    print("  VANILLA")
    print("▓" * 70)
    set_seed(SEED)
    vanilla = VanillaGPT(base_cfg).to(device)
    set_seed(SEED)
    results['vanilla'] = {'history': train_model(vanilla, "Vanilla", device)}

    # === Pure Geometry ===
    print("\n" + "▓" * 70)
    print("  PURE GEOMETRY")
    print("▓" * 70)
    set_seed(SEED)
    pure = PureGeometricGPT(base_cfg).to(device)
    set_seed(SEED)
    results['pure_geometry'] = {'history': train_model(pure, "PureGeo", device)}

    # === Hybrid ===
    print("\n" + "▓" * 70)
    print("  HYBRID GATED")
    print("▓" * 70)
    set_seed(SEED)
    hybrid = HybridGatedGPT(hybrid_cfg).to(device)
    set_seed(SEED)
    results['hybrid'] = {'history': train_model(hybrid, "Hybrid", device, log_gates=True)}

    # === PER-PATTERN EVALUATION ===
    print("\n" + "=" * 70)
    print("  PER-PATTERN LOSS (lower = better)")
    print("=" * 70)

    set_seed(SEED)
    v_pp = evaluate_per_pattern(vanilla, device)
    p_pp = evaluate_per_pattern(pure, device)
    h_pp = evaluate_per_pattern(hybrid, device)

    results['vanilla']['per_pattern'] = v_pp
    results['pure_geometry']['per_pattern'] = p_pp
    results['hybrid']['per_pattern'] = h_pp

    print(f"\n  {'Pattern':<22} {'Vanilla':>10} {'PureGeo':>10} {'Hybrid':>10} {'Best':>8}")
    print(f"  {'-'*62}")

    geo_wins = 0
    for pname in PATTERN_NAMES:
        vl, pl, hl = v_pp[pname], p_pp[pname], h_pp[pname]
        best = min(vl, pl, hl)
        if best == pl:
            winner = "PureGeo"
            geo_wins += 1
        elif best == hl:
            winner = "Hybrid"
            geo_wins += 1
        else:
            winner = "Vanilla"
        print(f"  {pname:<22} {vl:>10.4f} {pl:>10.4f} {hl:>10.4f} {winner:>8}")

    print(f"\n  Geometry-based models win on {geo_wins}/{len(PATTERN_NAMES)} patterns")

    # === PER-PATTERN GATE ANALYSIS ===
    print(f"\n{'='*70}")
    print(f"  PER-PATTERN GATE PREFERENCES (Hybrid)")
    print(f"  gate > 0.5 = prefers GEOMETRY, < 0.5 = prefers STANDARD")
    print(f"{'='*70}")

    gate_analysis = evaluate_per_pattern_gates(hybrid, device)
    results['hybrid']['gate_per_pattern'] = {}

    for pname in PATTERN_NAMES:
        gates = gate_analysis[pname]
        geo_count = 0
        total_gates = 0
        for layer_name, layer_gates in sorted(gates.items()):
            for gate_name, gs in layer_gates.items():
                total_gates += 1
                if gs['prefers_geometry']:
                    geo_count += 1

        geo_pct = 100 * geo_count / max(total_gates, 1)
        print(f"\n  {pname}: {geo_pct:.0f}% gates prefer geometry ({geo_count}/{total_gates})")

        # Детали по слоям
        for layer_name in sorted(gates.keys()):
            parts = []
            for gname in ['attention', 'quantization', 'ffn']:
                if gname in gates[layer_name]:
                    val = gates[layer_name][gname]['mean']
                    marker = "G" if val > 0.5 else "S"
                    parts.append(f"{gname[0].upper()}={val:.3f}({marker})")
            print(f"    {layer_name}: {' | '.join(parts)}")

        results['hybrid']['gate_per_pattern'][pname] = {
            'geo_fraction': geo_pct / 100,
            'details': {
                ln: {gn: gs['mean'] for gn, gs in lg.items()}
                for ln, lg in gates.items()
            }
        }

    # === ВЫВОДЫ ===
    print(f"\n{'='*70}")
    print(f"  CONCLUSIONS")
    print(f"{'='*70}")

    # Найти паттерны, где геометрия максимально выигрывает
    geo_advantage = {}
    for pname in PATTERN_NAMES:
        # Преимущество PureGeo или Hybrid над Vanilla
        best_geo = min(p_pp[pname], h_pp[pname])
        advantage = v_pp[pname] - best_geo
        geo_advantage[pname] = advantage

    sorted_advantage = sorted(geo_advantage.items(), key=lambda x: x[1], reverse=True)

    print("\n  Patterns ranked by geometry advantage (positive = geometry better):")
    for pname, adv in sorted_advantage:
        marker = "+++" if adv > 0.01 else "++" if adv > 0.001 else "+" if adv > 0 else "-"
        print(f"    {marker} {pname:<22} delta={adv:+.4f}")

    # Корреляция между gate preference и advantage
    print("\n  Gate preference vs advantage correlation:")
    for pname, adv in sorted_advantage:
        gate_pct = results['hybrid']['gate_per_pattern'].get(pname, {}).get('geo_fraction', 0)
        print(f"    {pname:<22} adv={adv:+.4f}, geo_gates={gate_pct:.0%}")

    # Сохранение
    save_data = {
        'config': {
            'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_heads': N_HEADS,
            'block_size': BLOCK_SIZE, 'vocab_size': VOCAB_SIZE,
            'steps': TOTAL_STEPS, 'seed': SEED,
            'patterns': PATTERN_NAMES,
        },
    }
    for mode in ['vanilla', 'pure_geometry', 'hybrid']:
        save_data[mode] = {
            'best_val_loss': results[mode]['history']['best_val_loss'],
            'per_pattern_loss': results[mode].get('per_pattern', {}),
        }
    if 'gate_per_pattern' in results['hybrid']:
        save_data['hybrid']['gate_per_pattern'] = results['hybrid']['gate_per_pattern']

    results_path = os.path.join(os.path.dirname(__file__), '..', 'symmetry_experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    run_symmetry_experiment()
