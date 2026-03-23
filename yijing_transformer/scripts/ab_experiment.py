"""
A/B Эксперимент: YiJingGPT vs VanillaGPT

Цель: Проверить, даёт ли геометрическая структура И-Цзин
(гексаграммы, BianGua, trigram bias) преимущество над
стандартным трансформером при равном числе параметров.

Протокол:
1. Одинаковая архитектура (d_model, n_layers, n_heads, RoPE, SwiGLU)
2. Одинаковые данные (синтетические паттерны + случайные)
3. Одинаковый оптимизатор и шедулер
4. Фиксированный seed для воспроизводимости
5. Сравнение: train loss, val loss, скорость сходимости, генерация
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
from models.model import YiJingGPT
from models.baseline import VanillaGPT


# ==================== КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА ====================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 1000
WARMUP_STEPS = 100
LOG_EVERY = 50
VAL_EVERY = 100
BATCH_SIZE = 8
BLOCK_SIZE = 64
VOCAB_SIZE = 256
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
LR = 1e-3


# ==================== ДАННЫЕ ====================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_patterned_batch(batch_size, block_size, vocab_size, device):
    """
    Синтетические данные с разными типами паттернов:
    - Копирование: повторение подпоследовательности
    - Реверс: обратный порядок
    - Арифметика: модульное сложение
    - Случайные: baseline шум
    """
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)

    for i in range(batch_size):
        pattern_type = i % 4

        if pattern_type == 0:
            # Копирование
            src_len = block_size // 4
            src_start = random.randint(0, block_size // 4)
            dst_start = block_size // 2
            if dst_start + src_len <= block_size + 1:
                x[i, dst_start:dst_start + src_len] = x[i, src_start:src_start + src_len]

        elif pattern_type == 1:
            # Реверс
            seg_len = block_size // 4
            start = random.randint(0, block_size // 4)
            rev_start = block_size // 2
            if rev_start + seg_len <= block_size + 1:
                x[i, rev_start:rev_start + seg_len] = x[i, start:start + seg_len].flip(0)

        elif pattern_type == 2:
            # Модульное сложение: c[i] = (a[i] + b[i]) % vocab_size
            seg_len = block_size // 6
            a_start = 0
            b_start = seg_len
            c_start = 2 * seg_len
            if c_start + seg_len <= block_size + 1:
                x[i, c_start:c_start + seg_len] = (
                    x[i, a_start:a_start + seg_len] + x[i, b_start:b_start + seg_len]
                ) % vocab_size

        # pattern_type == 3: чисто случайный (уже готово)

    return x[:, :-1], x[:, 1:]


# ==================== LR SCHEDULE ====================

def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ==================== ВАЛИДАЦИЯ ====================

@torch.no_grad()
def evaluate(model, device, num_batches=50):
    model.eval()
    losses = []
    for _ in range(num_batches):
        xb, yb = generate_patterned_batch(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        logits = model(xb, yb)
        # handle both (logits, loss) and (logits, loss, kv_cache)
        if isinstance(logits, tuple):
            if len(logits) == 3:
                _, loss, _ = logits
            else:
                _, loss = logits
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ==================== ОБУЧЕНИЕ ОДНОЙ МОДЕЛИ ====================

def train_model(model, model_name, device):
    """Обучает модель и возвращает историю метрик."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
    )

    history = {
        'model': model_name,
        'train_loss': [],
        'val_loss': [],
        'steps': [],
        'val_steps': [],
        'time_per_step': [],
    }

    total_params, hex_params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  Parameters: {total_params:,}")
    if hex_params > 0:
        print(f"  YiJing-specific: {hex_params:,} ({100*hex_params/total_params:.1f}%)")
    print(f"{'='*60}")

    model.train()
    accum_loss = 0.0
    step_times = []
    best_val_loss = float('inf')

    for step in range(1, TOTAL_STEPS + 1):
        t0 = time.time()

        # LR schedule
        lr = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        xb, yb = generate_patterned_batch(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE, device)
        result = model(xb, yb)
        if len(result) == 3:
            _, loss, _ = result
        else:
            _, loss = result

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        dt = time.time() - t0
        step_times.append(dt)
        accum_loss += loss.item()

        # Логирование
        if step % LOG_EVERY == 0:
            avg_loss = accum_loss / LOG_EVERY
            avg_time = sum(step_times[-LOG_EVERY:]) / len(step_times[-LOG_EVERY:])
            history['train_loss'].append(avg_loss)
            history['steps'].append(step)
            history['time_per_step'].append(avg_time)
            print(f"  Step {step:4d} | loss={avg_loss:.4f} | lr={lr:.2e} | {avg_time*1000:.1f}ms/step")
            accum_loss = 0.0

        # Валидация
        if step % VAL_EVERY == 0:
            val_loss = evaluate(model, device)
            history['val_loss'].append(val_loss)
            history['val_steps'].append(step)
            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                marker = " *best*"
            print(f"  >>> VAL loss={val_loss:.4f}{marker}")

    history['total_params'] = total_params
    history['hex_params'] = hex_params
    history['best_val_loss'] = best_val_loss
    history['avg_time_per_step'] = sum(step_times) / len(step_times)
    return history


# ==================== ГЕНЕРАЦИЯ ====================

@torch.no_grad()
def test_generation(model, model_name, device):
    """Тестирует генерацию с одинакового промпта."""
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, 16), device=device)

    if hasattr(model, 'generate'):
        output = model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=40)
    else:
        # VanillaGPT — простая авторегрессия
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

    # Статистики генерации
    generated = output[0, 16:].tolist()
    unique_tokens = len(set(generated))
    repetition_rate = 1 - unique_tokens / max(len(generated), 1)

    print(f"\n  {model_name} generation stats:")
    print(f"    Unique tokens: {unique_tokens}/{len(generated)}")
    print(f"    Repetition rate: {repetition_rate:.2%}")
    print(f"    Token distribution entropy: {token_entropy(generated):.3f}")
    return {
        'unique_tokens': unique_tokens,
        'total_tokens': len(generated),
        'repetition_rate': repetition_rate,
        'entropy': token_entropy(generated),
    }


def token_entropy(tokens):
    """Энтропия распределения токенов в сгенерированной последовательности."""
    from collections import Counter
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ==================== АНАЛИЗ ГЕОМЕТРИИ ====================

def analyze_geometry(model):
    """Анализирует, как модель использует геометрические компоненты."""
    analysis = {}
    for i, layer in enumerate(model.core.layers):
        info = {
            'hex_scale': layer.hex_scale.item(),
            'head_scales_mean': layer.attn.head_scales.data.abs().mean().item(),
            'head_scales_std': layer.attn.head_scales.data.abs().std().item(),
        }
        if hasattr(layer.quantizer, 'current_temp'):
            temp = layer.quantizer.current_temp
            info['quantizer_temp'] = temp.item() if isinstance(temp, torch.Tensor) else temp
        if layer.bian_gua is not None:
            info['bian_gua_scale'] = layer.bian_gua.scale.item()
            probs = torch.sigmoid(layer.bian_gua.change_logits).tolist()
            info['bian_gua_active_lines'] = sum(1 for p in probs if p > 0.5)
        analysis[f'layer_{i}'] = info
    return analysis


# ==================== ОСНОВНОЙ ЭКСПЕРИМЕНТ ====================

def run_experiment():
    device = torch.device(DEVICE)
    print("=" * 60)
    print("  A/B EXPERIMENT: YiJingGPT vs VanillaGPT")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
    print(f"  block_size={BLOCK_SIZE}, vocab_size={VOCAB_SIZE}")
    print(f"  batch_size={BATCH_SIZE}, steps={TOTAL_STEPS}")
    print(f"  lr={LR}, warmup={WARMUP_STEPS}")
    print(f"  seed={SEED}, device={DEVICE}")

    cfg = YiJingConfig(
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
    )

    # ===== YiJingGPT =====
    set_seed(SEED)
    yijing_model = YiJingGPT(cfg).to(device)

    set_seed(SEED)  # тот же seed для данных
    yijing_history = train_model(yijing_model, "YiJingGPT", device)

    # ===== VanillaGPT =====
    set_seed(SEED)
    vanilla_model = VanillaGPT(cfg).to(device)

    set_seed(SEED)  # тот же seed для данных
    vanilla_history = train_model(vanilla_model, "VanillaGPT", device)

    # ===== Генерация =====
    print("\n" + "=" * 60)
    print("  GENERATION TEST")
    print("=" * 60)
    set_seed(SEED)
    yijing_gen = test_generation(yijing_model, "YiJingGPT", device)
    set_seed(SEED)
    vanilla_gen = test_generation(vanilla_model, "VanillaGPT", device)

    # ===== Анализ геометрии =====
    print("\n" + "=" * 60)
    print("  YIJING GEOMETRY ANALYSIS (post-training)")
    print("=" * 60)
    geo_analysis = analyze_geometry(yijing_model)
    for layer_name, vals in geo_analysis.items():
        parts = []
        for k, v in vals.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(f"  {layer_name}: {', '.join(parts)}")

    # ===== РЕЗУЛЬТАТЫ =====
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    y_final = yijing_history['train_loss'][-1] if yijing_history['train_loss'] else float('nan')
    v_final = vanilla_history['train_loss'][-1] if vanilla_history['train_loss'] else float('nan')
    y_best_val = yijing_history['best_val_loss']
    v_best_val = vanilla_history['best_val_loss']
    y_time = yijing_history['avg_time_per_step'] * 1000
    v_time = vanilla_history['avg_time_per_step'] * 1000

    print(f"\n  {'Metric':<25} {'YiJingGPT':>12} {'VanillaGPT':>12} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Parameters':<25} {yijing_history['total_params']:>12,} {vanilla_history['total_params']:>12,} {yijing_history['total_params'] - vanilla_history['total_params']:>+10,}")
    print(f"  {'  (YiJing-specific)':<25} {yijing_history['hex_params']:>12,} {'—':>12} {'':>10}")
    print(f"  {'Final train loss':<25} {y_final:>12.4f} {v_final:>12.4f} {y_final - v_final:>+10.4f}")
    print(f"  {'Best val loss':<25} {y_best_val:>12.4f} {v_best_val:>12.4f} {y_best_val - v_best_val:>+10.4f}")
    print(f"  {'Avg ms/step':<25} {y_time:>12.1f} {v_time:>12.1f} {y_time - v_time:>+10.1f}")
    print(f"  {'Gen unique tokens':<25} {yijing_gen['unique_tokens']:>12} {vanilla_gen['unique_tokens']:>12} {yijing_gen['unique_tokens'] - vanilla_gen['unique_tokens']:>+10}")
    print(f"  {'Gen entropy':<25} {yijing_gen['entropy']:>12.3f} {vanilla_gen['entropy']:>12.3f} {yijing_gen['entropy'] - vanilla_gen['entropy']:>+10.3f}")

    # Вердикт
    print(f"\n  {'='*60}")
    if y_best_val < v_best_val:
        delta_pct = (v_best_val - y_best_val) / v_best_val * 100
        print(f"  VERDICT: YiJingGPT лучше на {delta_pct:.1f}% по val loss")
    elif v_best_val < y_best_val:
        delta_pct = (y_best_val - v_best_val) / y_best_val * 100
        print(f"  VERDICT: VanillaGPT лучше на {delta_pct:.1f}% по val loss")
    else:
        print(f"  VERDICT: Разницы нет")

    overhead_pct = (y_time - v_time) / v_time * 100 if v_time > 0 else 0
    print(f"  OVERHEAD: YiJingGPT {overhead_pct:+.1f}% по скорости")
    print(f"  {'='*60}")

    # Конвергенция: когда loss опустился ниже порога
    threshold = 5.0  # loss порог для "сходимости"
    y_conv_step = None
    v_conv_step = None
    for step, loss in zip(yijing_history['val_steps'], yijing_history['val_loss']):
        if loss < threshold and y_conv_step is None:
            y_conv_step = step
    for step, loss in zip(vanilla_history['val_steps'], vanilla_history['val_loss']):
        if loss < threshold and v_conv_step is None:
            v_conv_step = step

    if y_conv_step or v_conv_step:
        print(f"\n  Convergence (val loss < {threshold}):")
        print(f"    YiJingGPT:  step {y_conv_step or 'не достигнуто'}")
        print(f"    VanillaGPT: step {v_conv_step or 'не достигнуто'}")

    # Сохраняем результаты
    results = {
        'config': {
            'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_heads': N_HEADS,
            'block_size': BLOCK_SIZE, 'vocab_size': VOCAB_SIZE,
            'batch_size': BATCH_SIZE, 'steps': TOTAL_STEPS, 'seed': SEED,
        },
        'yijing': {
            'params': yijing_history['total_params'],
            'hex_params': yijing_history['hex_params'],
            'final_train_loss': y_final,
            'best_val_loss': y_best_val,
            'avg_ms_per_step': y_time,
            'train_loss_history': yijing_history['train_loss'],
            'val_loss_history': yijing_history['val_loss'],
            'generation': yijing_gen,
        },
        'vanilla': {
            'params': vanilla_history['total_params'],
            'final_train_loss': v_final,
            'best_val_loss': v_best_val,
            'avg_ms_per_step': v_time,
            'train_loss_history': vanilla_history['train_loss'],
            'val_loss_history': vanilla_history['val_loss'],
            'generation': vanilla_gen,
        },
        'geometry_analysis': {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                                   for kk, vv in v.items()}
                              for k, v in geo_analysis.items()},
    }

    results_path = os.path.join(os.path.dirname(__file__), '..', 'ab_experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    run_experiment()
