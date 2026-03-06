"""
Ablation Study: три архитектурных режима

Три условия эксперимента:
1. Vanilla — стандартный трансформер (baseline)
2. Pure Geometry — только геометрические компоненты (изоляция)
3. Hybrid Gated — оба пути + гейтовый выбор (финальная архитектура)

Цель: ответить на вопросы:
- Способна ли геометрия решать задачу самостоятельно? (Pure vs Vanilla)
- Выбирает ли модель геометрию, когда ей дают свободу? (Hybrid гейты)
- Где именно геометрия полезна? (анализ гейтов по слоям)

Принцип ненавязывания: геометрическая структура Yi Jing выступает
как предложение, а не ограничение. Модель сама определяет баланс.

Equal parameter budget: все три модели имеют сопоставимое число параметров.
"""

import os
import sys
import math
import time
import json
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT, PureGeometricGPT, HybridGatedGPT
from models.baseline import VanillaGPT


# ==================== КОНФИГУРАЦИЯ ====================

SEED = 42
DEVICE = 'cpu'
TOTAL_STEPS = 1000
WARMUP_STEPS = 100
LOG_EVERY = 50
VAL_EVERY = 100
GATE_LOG_EVERY = 50  # логировать гейты каждые N шагов
BATCH_SIZE = 8
BLOCK_SIZE = 64
VOCAB_SIZE = 256
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
LR = 1e-3


# ==================== УТИЛИТЫ ====================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_patterned_batch(batch_size, block_size, vocab_size, device):
    """Синтетические данные с паттернами: copy, reverse, modular add, random."""
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)
    for i in range(batch_size):
        pattern_type = i % 4
        if pattern_type == 0:  # Copy
            src_len = block_size // 4
            src_start = random.randint(0, block_size // 4)
            dst_start = block_size // 2
            if dst_start + src_len <= block_size + 1:
                x[i, dst_start:dst_start + src_len] = x[i, src_start:src_start + src_len]
        elif pattern_type == 1:  # Reverse
            seg_len = block_size // 4
            start = random.randint(0, block_size // 4)
            rev_start = block_size // 2
            if rev_start + seg_len <= block_size + 1:
                x[i, rev_start:rev_start + seg_len] = x[i, start:start + seg_len].flip(0)
        elif pattern_type == 2:  # Modular addition
            seg_len = block_size // 6
            c_start = 2 * seg_len
            if c_start + seg_len <= block_size + 1:
                x[i, c_start:c_start + seg_len] = (
                    x[i, 0:seg_len] + x[i, seg_len:2 * seg_len]
                ) % vocab_size
    return x[:, :-1], x[:, 1:]


def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def token_entropy(tokens):
    from collections import Counter
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ==================== ВАЛИДАЦИЯ ====================

@torch.no_grad()
def evaluate(model, device, num_batches=50):
    model.eval()
    losses = []
    for _ in range(num_batches):
        xb, yb = generate_patterned_batch(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        result = model(xb, yb)
        if isinstance(result, tuple):
            if len(result) == 3:
                _, loss, _ = result
            else:
                _, loss = result
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ==================== ОБУЧЕНИЕ ====================

def train_model(model, model_name, device, log_gates=False):
    """Обучает модель, возвращает историю метрик и (опционально) гейт-логи."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
    )

    total_params, geo_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Total params: {total_params:,}")
    if geo_params > 0:
        print(f"  Geometric params: {geo_params:,} ({100*geo_params/total_params:.1f}%)")
    print(f"{'='*60}")

    history = {
        'model': model_name,
        'total_params': total_params,
        'geo_params': geo_params,
        'train_loss': [],
        'val_loss': [],
        'steps': [],
        'val_steps': [],
        'time_per_step': [],
        'best_val_loss': float('inf'),
        'avg_time_per_step': 0,
    }

    gate_history = []  # для Hybrid модели

    model.train()
    accum_loss = 0.0
    step_times = []

    for step in range(1, TOTAL_STEPS + 1):
        t0 = time.time()

        lr = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Curriculum update (для Hybrid модели)
        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)

        xb, yb = generate_patterned_batch(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        result = model(xb, yb)
        if len(result) == 3:
            _, loss, _ = result
        else:
            _, loss = result

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        step_times.append(dt)
        accum_loss += loss.item()

        # Логирование train loss
        if step % LOG_EVERY == 0:
            avg_loss = accum_loss / LOG_EVERY
            avg_time = sum(step_times[-LOG_EVERY:]) / len(step_times[-LOG_EVERY:])
            history['train_loss'].append(avg_loss)
            history['steps'].append(step)
            history['time_per_step'].append(avg_time)
            print(f"  Step {step:4d} | loss={avg_loss:.4f} | lr={lr:.2e} | {avg_time*1000:.1f}ms/step")
            accum_loss = 0.0

        # Логирование гейтов (для Hybrid)
        if log_gates and step % GATE_LOG_EVERY == 0 and hasattr(model, 'log_gates'):
            entry = model.log_gates(step)
            gate_history.append(entry)
            # Печатаем сводку
            if step % (LOG_EVERY * 2) == 0:
                summary = model.gate_logger.summary()
                geo_count = summary.get('layers_prefer_geometry', 0)
                std_count = summary.get('layers_prefer_standard', 0)
                print(f"    [Gates] Geometry={geo_count} | Standard={std_count}")

        # Валидация
        if step % VAL_EVERY == 0:
            val_loss = evaluate(model, device)
            history['val_loss'].append(val_loss)
            history['val_steps'].append(step)
            marker = ""
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                marker = " *best*"
            print(f"  >>> VAL loss={val_loss:.4f}{marker}")

    history['avg_time_per_step'] = sum(step_times) / len(step_times)

    return history, gate_history


# ==================== ГЕНЕРАЦИЯ ====================

@torch.no_grad()
def test_generation(model, model_name, device):
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, 16), device=device)

    if hasattr(model, 'generate'):
        output = model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=40)
    else:
        idx = prompt
        for _ in range(32):
            idx_input = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits, _ = model(idx_input)
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, min(40, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        output = idx

    generated = output[0, 16:].tolist()
    unique_tokens = len(set(generated))
    repetition_rate = 1 - unique_tokens / max(len(generated), 1)
    entropy = token_entropy(generated)

    print(f"\n  {model_name} generation:")
    print(f"    Unique: {unique_tokens}/{len(generated)} | Rep: {repetition_rate:.2%} | Entropy: {entropy:.3f}")

    return {
        'unique_tokens': unique_tokens,
        'total_tokens': len(generated),
        'repetition_rate': repetition_rate,
        'entropy': entropy,
    }


# ==================== АНАЛИЗ ГЕЙТОВ ====================

def analyze_gate_trajectory(gate_history):
    """Анализирует, как гейты менялись во время обучения."""
    if not gate_history:
        return {}

    analysis = {}
    # Собираем данные по слоям
    for entry in gate_history:
        for layer_name, stats in entry.get('gates', {}).items():
            if layer_name not in analysis:
                analysis[layer_name] = {
                    'steps': [], 'means': [],
                    'initial_preference': None,
                    'final_preference': None,
                }
            analysis[layer_name]['steps'].append(entry['step'])
            analysis[layer_name]['means'].append(stats['gate_mean'])

    # Определяем тренды
    for layer_name, data in analysis.items():
        if data['means']:
            data['initial_preference'] = 'geometry' if data['means'][0] > 0.5 else 'standard'
            data['final_preference'] = 'geometry' if data['means'][-1] > 0.5 else 'standard'
            data['changed_preference'] = data['initial_preference'] != data['final_preference']
            data['avg_gate'] = sum(data['means']) / len(data['means'])
            data['trend'] = data['means'][-1] - data['means'][0]  # + = к геометрии

    return analysis


# ==================== ОСНОВНОЙ ЭКСПЕРИМЕНТ ====================

def run_ablation():
    device = torch.device(DEVICE)

    print("=" * 70)
    print("  THREE-MODE ABLATION STUDY")
    print("  Vanilla vs Pure Geometry vs Hybrid Gated")
    print("=" * 70)
    print(f"\nConfig: d={D_MODEL}, L={N_LAYERS}, H={N_HEADS}, "
          f"T={BLOCK_SIZE}, V={VOCAB_SIZE}, steps={TOTAL_STEPS}")

    # === Базовый конфиг ===
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

    results = {}

    # ==================== 1. VANILLA BASELINE ====================
    print("\n\n" + "▓" * 70)
    print("  MODE 1: VANILLA (Standard Transformer)")
    print("▓" * 70)

    set_seed(SEED)
    vanilla = VanillaGPT(base_cfg).to(device)
    set_seed(SEED)
    vanilla_history, _ = train_model(vanilla, "VanillaGPT", device)
    set_seed(SEED)
    vanilla_gen = test_generation(vanilla, "VanillaGPT", device)
    results['vanilla'] = {'history': vanilla_history, 'generation': vanilla_gen}

    # ==================== 2. PURE GEOMETRY ====================
    print("\n\n" + "▓" * 70)
    print("  MODE 2: PURE GEOMETRY (Only Geometric Components)")
    print("▓" * 70)

    set_seed(SEED)
    pure_geo = PureGeometricGPT(base_cfg).to(device)
    set_seed(SEED)
    pure_history, _ = train_model(pure_geo, "PureGeometricGPT", device)
    set_seed(SEED)
    pure_gen = test_generation(pure_geo, "PureGeometricGPT", device)
    results['pure_geometry'] = {'history': pure_history, 'generation': pure_gen}

    # ==================== 3. HYBRID GATED ====================
    print("\n\n" + "▓" * 70)
    print("  MODE 3: HYBRID GATED (Freedom of Choice)")
    print("▓" * 70)

    # Гибридный конфиг с curriculum
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
        gate_init_bias=0.0,  # равные шансы
        curriculum_strategy_geo='linear',
        curriculum_warmup_fraction=0.3,
        curriculum_target_strength=0.1,
    )

    set_seed(SEED)
    hybrid = HybridGatedGPT(hybrid_cfg).to(device)
    set_seed(SEED)
    hybrid_history, gate_history = train_model(hybrid, "HybridGatedGPT", device, log_gates=True)
    set_seed(SEED)
    hybrid_gen = test_generation(hybrid, "HybridGatedGPT", device)
    results['hybrid'] = {
        'history': hybrid_history,
        'generation': hybrid_gen,
        'gate_trajectory': analyze_gate_trajectory(gate_history),
    }

    # ==================== РЕЗУЛЬТАТЫ ====================
    print("\n\n" + "=" * 70)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 70)

    models_data = [
        ('VanillaGPT', results['vanilla']),
        ('PureGeoGPT', results['pure_geometry']),
        ('HybridGPT', results['hybrid']),
    ]

    # Таблица параметров
    print(f"\n  {'Model':<16} {'Params':>10} {'GeoParams':>10} {'GeoFrac':>8}")
    print(f"  {'-'*48}")
    for name, data in models_data:
        h = data['history']
        frac = f"{100*h['geo_params']/h['total_params']:.1f}%" if h['geo_params'] > 0 else "—"
        print(f"  {name:<16} {h['total_params']:>10,} {h['geo_params']:>10,} {frac:>8}")

    # Таблица loss
    print(f"\n  {'Model':<16} {'FinalTrain':>12} {'BestVal':>12} {'ms/step':>10}")
    print(f"  {'-'*52}")
    for name, data in models_data:
        h = data['history']
        final = h['train_loss'][-1] if h['train_loss'] else float('nan')
        best = h['best_val_loss']
        ms = h['avg_time_per_step'] * 1000
        print(f"  {name:<16} {final:>12.4f} {best:>12.4f} {ms:>10.1f}")

    # Таблица генерации
    print(f"\n  {'Model':<16} {'Unique':>8} {'RepRate':>10} {'Entropy':>10}")
    print(f"  {'-'*48}")
    for name, data in models_data:
        g = data['generation']
        print(f"  {name:<16} {g['unique_tokens']:>8} {g['repetition_rate']:>10.2%} {g['entropy']:>10.3f}")

    # Анализ гейтов (Hybrid)
    if results['hybrid'].get('gate_trajectory'):
        print(f"\n  {'='*70}")
        print(f"  GATE ANALYSIS (Hybrid — что модель предпочитает?)")
        print(f"  {'='*70}")

        traj = results['hybrid']['gate_trajectory']
        for layer_name, data in sorted(traj.items()):
            pref = "GEOMETRY" if data.get('final_preference') == 'geometry' else "STANDARD"
            avg = data.get('avg_gate', 0.5)
            trend = data.get('trend', 0.0)
            changed = " [CHANGED]" if data.get('changed_preference') else ""
            print(f"  {layer_name}: prefers {pref} (avg={avg:.3f}, trend={trend:+.3f}){changed}")

    # Вердикт
    print(f"\n  {'='*70}")
    print(f"  VERDICT")
    print(f"  {'='*70}")

    v_best = results['vanilla']['history']['best_val_loss']
    p_best = results['pure_geometry']['history']['best_val_loss']
    h_best = results['hybrid']['history']['best_val_loss']

    best_model = 'Vanilla'
    best_val = v_best
    if p_best < best_val:
        best_model, best_val = 'Pure Geometry', p_best
    if h_best < best_val:
        best_model, best_val = 'Hybrid Gated', h_best

    print(f"\n  Best model by val loss: {best_model} ({best_val:.4f})")

    if p_best < v_best:
        delta = (v_best - p_best) / v_best * 100
        print(f"  Pure Geometry beats Vanilla by {delta:.1f}% — geometry works independently!")
    else:
        delta = (p_best - v_best) / v_best * 100
        print(f"  Vanilla beats Pure Geometry by {delta:.1f}% — geometry needs standard components")

    if h_best < v_best:
        delta = (v_best - h_best) / v_best * 100
        print(f"  Hybrid beats Vanilla by {delta:.1f}% — gated selection helps!")
    else:
        delta = (h_best - v_best) / v_best * 100
        print(f"  Vanilla beats Hybrid by {delta:.1f}% — overhead outweighs geometry benefit")

    # Подсчёт слоёв, предпочитающих геометрию
    geo_layers = sum(
        1 for d in traj.values()
        if d.get('final_preference') == 'geometry'
    ) if traj else 0
    total_layers = len(traj) if traj else 0
    print(f"\n  Hybrid gate preferences: {geo_layers}/{total_layers} layers prefer geometry")

    # === Технологическая политика ===
    print(f"\n  {'='*70}")
    print(f"  TECHNOLOGICAL POLICY RECOMMENDATIONS")
    print(f"  {'='*70}")

    if h_best <= v_best and geo_layers > 0:
        print("""
  РЕКОМЕНДАЦИЯ: Дипломатический подход успешен.
  - Гейтовый выбор позволяет модели использовать геометрию там, где выгодно
  - Curriculum learning помогает геометрическим компонентам обучиться
  - ДАЛЬНЕЙШИЕ ШАГИ:
    1. Увеличить масштаб эксперимента (больше данных, шагов)
    2. Тестировать на реальных данных (не синтетических)
    3. Анализировать, на каких типах паттернов геометрия выигрывает
    4. Рассмотреть layer-specific гейты (разные стратегии для разных слоёв)
        """)
    elif p_best < v_best:
        print("""
  РЕКОМЕНДАЦИЯ: Чистая геометрия показывает потенциал.
  - Геометрические компоненты самодостаточны
  - ДАЛЬНЕЙШИЕ ШАГИ:
    1. Масштабировать PureGeometricGPT
    2. Оптимизировать GeometricAttention
    3. Исследовать, почему Hybrid не усилил результат
        """)
    else:
        print("""
  РЕКОМЕНДАЦИЯ: Геометрия пока не показала преимущества.
  - Стандартный трансформер эффективнее на текущем масштабе
  - ДАЛЬНЕЙШИЕ ШАГИ:
    1. Увеличить масштаб (геометрия может требовать больше данных)
    2. Использовать стратегию 'geometric_first' для curriculum
    3. Увеличить hex_strength до 0.1-1.0
    4. Тестировать на данных с явными симметриями
        """)

    # Сохраняем результаты
    save_data = {}
    for mode, data in results.items():
        save_data[mode] = {
            'params': data['history']['total_params'],
            'geo_params': data['history']['geo_params'],
            'final_train_loss': data['history']['train_loss'][-1] if data['history']['train_loss'] else None,
            'best_val_loss': data['history']['best_val_loss'],
            'avg_ms_per_step': data['history']['avg_time_per_step'] * 1000,
            'train_loss_curve': data['history']['train_loss'],
            'val_loss_curve': data['history']['val_loss'],
            'generation': data['generation'],
        }
    if results['hybrid'].get('gate_trajectory'):
        save_data['hybrid']['gate_analysis'] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ('steps', 'means')}
            for k, v in results['hybrid']['gate_trajectory'].items()
        }

    results_path = os.path.join(os.path.dirname(__file__), '..', 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    run_ablation()
