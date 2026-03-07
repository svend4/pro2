#!/usr/bin/env python3
"""
Эксперимент: доказательство преимущества геометрии на XOR-задачах.

Задача: предсказать XOR двух 6-битных чисел.
Вход: [a₁ a₂ a₃ a₄ a₅ a₆ SEP b₁ b₂ b₃ b₄ b₅ b₆ SEP]
Цель: [c₁ c₂ c₃ c₄ c₅ c₆] где c = a XOR b

Почему геометрия должна помочь:
- XOR ≡ умножение в Z₂⁶ (теорема 1: изоморфизм)
- Гиперкубный кодбук {-1,+1}⁶ = натуральное пространство для XOR
- Квантизация проецирует в пространство, где XOR тривиален

Три режима:
1. Vanilla — стандартный трансформер (без квантизации)
2. Geometry — с D4-эквивариантностью + квантизацией
3. Hybrid — с гейтовым выбором
"""

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from yijing_transformer.config.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT, HybridGatedGPT


# ==================== DATA ====================

VOCAB_SIZE = 4  # 0, 1, SEP=2, PAD=3
SEP_TOKEN = 2
SEQ_LEN = 14  # 6 + 1(SEP) + 6 + 1(SEP)
TARGET_LEN = 6


def generate_xor_batch(batch_size: int, device='cpu'):
    """Генерирует батч задач XOR на 6-битных числах."""
    a = torch.randint(0, 2, (batch_size, 6), device=device)
    b = torch.randint(0, 2, (batch_size, 6), device=device)
    c = a ^ b  # XOR

    # Вход: [a1 a2 a3 a4 a5 a6 SEP b1 b2 b3 b4 b5 b6 SEP]
    sep = torch.full((batch_size, 1), SEP_TOKEN, device=device)
    inputs = torch.cat([a, sep, b, sep], dim=1)  # (B, 14)

    # Таргет: [PAD]*8 + [c1 c2 c3 c4 c5 c6] — предсказываем после второго SEP
    # Для teacher forcing: сдвигаем таргет вправо
    # Проще: используем seq2seq подход — предсказываем каждый следующий токен
    # Но для нашей задачи: просто предсказываем XOR из последних 6 позиций

    return inputs, c


def generate_modular_add_batch(batch_size: int, mod: int = 64, device='cpu'):
    """Генерирует батч задач модулярного сложения (mod 64 = Z₂⁶)."""
    a = torch.randint(0, mod, (batch_size, 1), device=device)
    b = torch.randint(0, mod, (batch_size, 1), device=device)
    c = (a + b) % mod

    # Кодируем числа как 6 бит
    def to_bits(x, n_bits=6):
        bits = []
        for i in range(n_bits - 1, -1, -1):
            bits.append((x >> i) & 1)
        return torch.cat(bits, dim=1)

    a_bits = to_bits(a)
    b_bits = to_bits(b)
    c_bits = to_bits(c)

    sep = torch.full((batch_size, 1), SEP_TOKEN, device=device)
    inputs = torch.cat([a_bits, sep, b_bits, sep], dim=1)
    return inputs, c_bits


# ==================== MODEL WRAPPER ====================

class XORModel(nn.Module):
    """Обёртка: трансформер + голова для 6-битного XOR предсказания."""
    def __init__(self, base_model, d_model):
        super().__init__()
        self.base = base_model
        self.xor_head = nn.Linear(d_model, 2)  # бинарная классификация на каждый бит
        self.d_model = d_model

    def forward(self, inputs, targets=None):
        # Forward через трансформер
        logits, _, _ = self.base(inputs)  # (B, T, V)

        # Берём скрытые состояния из последних 6 позиций перед финальным SEP
        # Позиции 7-12 (0-indexed) = биты b, после них SEP
        # Но нам нужны скрытые состояния ПОСЛЕ обработки всего входа
        # Используем последние 6 позиций logits через отдельную голову

        # Получаем hidden states (обходим head модели)
        x = self.base.tok_emb(inputs)
        if self.base.pos_emb is not None:
            x = x + self.base.pos_emb[:, :inputs.size(1), :]
        elif hasattr(self.base, 'four_level_pe') and self.base.use_four_level_pe:
            x = x + self.base.four_level_pe(inputs.size(1), device=inputs.device)

        if hasattr(self.base, 'core'):
            hidden, _ = self.base.core(x)
        else:
            for layer in self.base.layers:
                x, _ = layer(x)
            hidden = self.base.final_norm(x)

        # Берём hidden states от последней позиции (SEP) — содержит всю информацию
        last_hidden = hidden[:, -1, :]  # (B, D)
        # Предсказываем 6 бит через 6 независимых линейных головок
        # Лучше: проецируем в 6×2
        bit_logits = self.xor_head(last_hidden.unsqueeze(1).expand(-1, 6, -1))  # (B, 6, 2)

        if targets is not None:
            loss = F.cross_entropy(
                bit_logits.reshape(-1, 2),
                targets.reshape(-1).long(),
            )
            return bit_logits, loss

        return bit_logits, None


class XORModelDirect(nn.Module):
    """Прямая модель: encoder → 6-bit predictor."""
    def __init__(self, base_model, d_model):
        super().__init__()
        self.base = base_model
        # 6 независимых бинарных головок
        self.bit_heads = nn.ModuleList([
            nn.Linear(d_model, 2) for _ in range(6)
        ])

    def forward(self, inputs, targets=None):
        # Forward через базовую модель
        x = self.base.tok_emb(inputs)
        if self.base.pos_emb is not None:
            x = x + self.base.pos_emb[:, :inputs.size(1), :]
        elif hasattr(self.base, 'four_level_pe') and self.base.use_four_level_pe:
            x = x + self.base.four_level_pe(inputs.size(1), device=inputs.device)

        if hasattr(self.base, 'core'):
            hidden, _ = self.base.core(x)
        else:
            for layer in self.base.layers:
                x, _ = layer(x)
            hidden = self.base.final_norm(x)

        # Используем последний токен (SEP) как summary
        summary = hidden[:, -1, :]  # (B, D)

        # 6 бинарных предсказаний
        bit_logits = torch.stack([head(summary) for head in self.bit_heads], dim=1)  # (B, 6, 2)

        if targets is not None:
            loss = F.cross_entropy(
                bit_logits.reshape(-1, 2),
                targets.reshape(-1).long(),
            )
            return bit_logits, loss

        return bit_logits, None


# ==================== TRAINING ====================

def train_model(model, task='xor', n_steps=2000, batch_size=128, lr=3e-4,
                device='cpu', eval_every=200):
    """Обучает модель и возвращает метрики."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    history = {'step': [], 'loss': [], 'accuracy': [], 'bit_accuracy': []}
    gen_fn = generate_xor_batch if task == 'xor' else generate_modular_add_batch

    model.train()
    for step in range(1, n_steps + 1):
        inputs, targets = gen_fn(batch_size, device=device)
        bit_logits, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                eval_inputs, eval_targets = gen_fn(1024, device=device)
                eval_logits, eval_loss = model(eval_inputs, eval_targets)
                preds = eval_logits.argmax(dim=-1)  # (B, 6)
                # Точность по отдельным битам
                bit_acc = (preds == eval_targets).float().mean().item()
                # Полная точность (все 6 бит верны)
                full_acc = (preds == eval_targets).all(dim=1).float().mean().item()

            history['step'].append(step)
            history['loss'].append(eval_loss.item())
            history['accuracy'].append(full_acc)
            history['bit_accuracy'].append(bit_acc)

            if step % (eval_every * 5) == 0 or step == eval_every:
                print(f"  Step {step:5d}: loss={eval_loss.item():.4f}, "
                      f"bit_acc={bit_acc:.4f}, full_acc={full_acc:.4f}")
            model.train()

    return history


# ==================== EXPERIMENT ====================

def run_experiment():
    device = 'cpu'
    n_steps = 3000
    batch_size = 128
    lr = 1e-3

    results = {}

    for task in ['xor', 'modular_add']:
        print(f"\n{'='*60}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*60}")

        task_results = {}

        # === 1. Vanilla (no geometry) ===
        print(f"\n--- Vanilla Transformer ---")
        cfg_vanilla = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=64, n_layers=3, n_heads=4,
            block_size=SEQ_LEN + 4, dropout=0.0,
            use_bian_gua=False, hex_strength=0.0,
            use_rope=True, use_swiglu=True,
        )
        base_vanilla = YiJingGPT(cfg_vanilla)
        model_vanilla = XORModelDirect(base_vanilla, cfg_vanilla.d_model).to(device)
        n_params_v = sum(p.numel() for p in model_vanilla.parameters())
        print(f"  Params: {n_params_v:,}")
        t0 = time.time()
        hist_vanilla = train_model(model_vanilla, task, n_steps, batch_size, lr, device)
        t_vanilla = time.time() - t0
        task_results['vanilla'] = {
            'history': hist_vanilla,
            'params': n_params_v,
            'time': t_vanilla,
            'final_accuracy': hist_vanilla['accuracy'][-1],
            'final_bit_accuracy': hist_vanilla['bit_accuracy'][-1],
            'final_loss': hist_vanilla['loss'][-1],
        }
        print(f"  FINAL: acc={hist_vanilla['accuracy'][-1]:.4f}, "
              f"bit_acc={hist_vanilla['bit_accuracy'][-1]:.4f}, time={t_vanilla:.1f}s")

        # === 2. Geometry (D4 + квантизация + graduated BianGua) ===
        print(f"\n--- Geometry Transformer (D4 + Quant + BianGua) ---")
        cfg_geo = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=64, n_layers=3, n_heads=4,
            block_size=SEQ_LEN + 4, dropout=0.0,
            use_bian_gua=True, hex_strength=0.1,
            use_rope=True, use_swiglu=True,
            use_d4_equivariant=True,
            use_graduated_biangua=True,
            use_antipodal_reg=True,
            antipodal_weight=0.005,
        )
        base_geo = YiJingGPT(cfg_geo)
        model_geo = XORModelDirect(base_geo, cfg_geo.d_model).to(device)
        n_params_g = sum(p.numel() for p in model_geo.parameters())
        print(f"  Params: {n_params_g:,}")
        t0 = time.time()
        hist_geo = train_model(model_geo, task, n_steps, batch_size, lr, device)
        t_geo = time.time() - t0
        task_results['geometry'] = {
            'history': hist_geo,
            'params': n_params_g,
            'time': t_geo,
            'final_accuracy': hist_geo['accuracy'][-1],
            'final_bit_accuracy': hist_geo['bit_accuracy'][-1],
            'final_loss': hist_geo['loss'][-1],
        }
        print(f"  FINAL: acc={hist_geo['accuracy'][-1]:.4f}, "
              f"bit_acc={hist_geo['bit_accuracy'][-1]:.4f}, time={t_geo:.1f}s")

        # === 3. QuadrantAttention + D4 ===
        print(f"\n--- Quadrant Attention + D4 ---")
        cfg_quad = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=64, n_layers=3, n_heads=4,
            block_size=SEQ_LEN + 4, dropout=0.0,
            use_bian_gua=True, hex_strength=0.1,
            use_rope=True, use_swiglu=True,
            use_d4_equivariant=True,
            use_quadrant_attention=True,
            use_antipodal_reg=True,
        )
        base_quad = YiJingGPT(cfg_quad)
        model_quad = XORModelDirect(base_quad, cfg_quad.d_model).to(device)
        n_params_q = sum(p.numel() for p in model_quad.parameters())
        print(f"  Params: {n_params_q:,}")
        t0 = time.time()
        hist_quad = train_model(model_quad, task, n_steps, batch_size, lr, device)
        t_quad = time.time() - t0
        task_results['quadrant_d4'] = {
            'history': hist_quad,
            'params': n_params_q,
            'time': t_quad,
            'final_accuracy': hist_quad['accuracy'][-1],
            'final_bit_accuracy': hist_quad['bit_accuracy'][-1],
            'final_loss': hist_quad['loss'][-1],
        }
        print(f"  FINAL: acc={hist_quad['accuracy'][-1]:.4f}, "
              f"bit_acc={hist_quad['bit_accuracy'][-1]:.4f}, time={t_quad:.1f}s")

        results[task] = task_results

    # ==================== SUMMARY ====================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for task_name, task_res in results.items():
        print(f"\n{task_name.upper()}:")
        print(f"  {'Model':<25} {'Full Acc':>10} {'Bit Acc':>10} {'Loss':>10} {'Params':>10}")
        print(f"  {'-'*65}")
        for model_name, res in task_res.items():
            print(f"  {model_name:<25} {res['final_accuracy']:>10.4f} "
                  f"{res['final_bit_accuracy']:>10.4f} "
                  f"{res['final_loss']:>10.4f} "
                  f"{res['params']:>10,}")

    # Сохраняем результаты
    output = {}
    for task_name, task_res in results.items():
        output[task_name] = {}
        for model_name, res in task_res.items():
            output[task_name][model_name] = {
                'final_accuracy': res['final_accuracy'],
                'final_bit_accuracy': res['final_bit_accuracy'],
                'final_loss': res['final_loss'],
                'params': res['params'],
                'time': res['time'],
                'history': {
                    'step': res['history']['step'],
                    'accuracy': res['history']['accuracy'],
                    'bit_accuracy': res['history']['bit_accuracy'],
                    'loss': res['history']['loss'],
                },
            }

    output_path = os.path.join(os.path.dirname(__file__), '..', 'xor_experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    run_experiment()
