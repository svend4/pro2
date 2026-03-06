"""
v15 утилиты: Z-Loss, Gradient Accumulation Profiler, Vocab Expansion.

Z-Loss: штрафует большие logits для численной стабильности.
Предотвращает log-sum-exp overflow при обучении с float16.
Ref: Chowdhery et al., "PaLM" (2022)

Gradient Accumulation Profiler: мониторинг gradient accumulation,
проверка эффективного batch size, tracking gradient norms per step.

Vocab Expansion: добавление новых токенов к обученной модели
без потери выученных представлений.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Z-Loss ====================

def z_loss(logits, weight=1e-4):
    """
    Z-Loss: штраф за большие logits.

    L_z = weight * mean(log(sum(exp(logits)))²)

    Это стабилизирует log-sum-exp в cross-entropy,
    предотвращая overflow при mixed precision.

    Args:
        logits: (B, T, V) или (B*T, V) — raw logits
        weight: масштаб Z-Loss

    Returns:
        z_loss_value: скаляр
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))

    # log(sum(exp(logits))) = logsumexp
    log_z = torch.logsumexp(logits, dim=-1)  # (N,)
    return weight * (log_z ** 2).mean()


def compute_loss_with_z(logits, targets, label_smoothing=0.0, z_weight=1e-4):
    """
    Cross-entropy + Z-Loss.

    Args:
        logits: (B, T, V)
        targets: (B, T)
        label_smoothing: label smoothing
        z_weight: вес Z-Loss (0 = выкл)

    Returns:
        total_loss, ce_loss, z_loss_value
    """
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        label_smoothing=label_smoothing,
    )

    zl = z_loss(logits, weight=z_weight) if z_weight > 0 else torch.tensor(0.0)
    total = ce + zl

    return total, ce, zl


# ==================== Gradient Accumulation Profiler ====================

class GradAccumProfiler:
    """
    Профайлер для gradient accumulation.

    Отслеживает:
    - Gradient norms per micro-step
    - Эффективный batch size
    - Variance градиентов между micro-steps
    - Рекомендации по оптимальному grad_accum_steps

    Args:
        model: модель для профилирования
        grad_accum_steps: число шагов аккумуляции
    """
    def __init__(self, model, grad_accum_steps=1):
        self.model = model
        self.grad_accum_steps = grad_accum_steps
        self.micro_step = 0
        self.micro_grad_norms = []
        self.step_history = []  # [(step, grad_norm, micro_norms)]

    def log_micro_step(self):
        """Логирует gradient norm текущего micro-step."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        self.micro_grad_norms.append(total_norm)
        self.micro_step += 1

    def log_optimizer_step(self, step):
        """Логирует после optimizer.step() (конец аккумуляции)."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        self.step_history.append({
            'step': step,
            'accumulated_grad_norm': total_norm,
            'micro_grad_norms': self.micro_grad_norms[:],
            'n_micro_steps': len(self.micro_grad_norms),
        })

        self.micro_grad_norms = []
        self.micro_step = 0

    def get_stats(self):
        """Возвращает статистику gradient accumulation."""
        if not self.step_history:
            return {
                'n_steps': 0,
                'avg_accumulated_norm': 0.0,
                'avg_micro_norm': 0.0,
                'grad_accum_steps': self.grad_accum_steps,
            }

        accum_norms = [s['accumulated_grad_norm'] for s in self.step_history]
        all_micro = []
        for s in self.step_history:
            all_micro.extend(s['micro_grad_norms'])

        return {
            'n_steps': len(self.step_history),
            'avg_accumulated_norm': sum(accum_norms) / len(accum_norms),
            'max_accumulated_norm': max(accum_norms),
            'avg_micro_norm': sum(all_micro) / len(all_micro) if all_micro else 0.0,
            'grad_accum_steps': self.grad_accum_steps,
            'effective_batch_multiplier': self.grad_accum_steps,
        }


# ==================== Vocab Expansion ====================

def expand_vocab(model, new_vocab_size, init_method='mean'):
    """
    Расширяет словарь модели, сохраняя выученные эмбеддинги.

    Новые токены инициализируются:
    - 'mean': средним существующих эмбеддингов
    - 'random': случайными значениями со std = std(existing)
    - 'zero': нулями

    Args:
        model: YiJingGPT модель
        new_vocab_size: новый размер словаря (> текущего)
        init_method: 'mean', 'random', 'zero'

    Returns:
        model: модель с расширенным словарём
    """
    old_vocab_size = model.cfg.vocab_size
    assert new_vocab_size > old_vocab_size, \
        f"new_vocab_size ({new_vocab_size}) must be > old ({old_vocab_size})"

    d_model = model.cfg.d_model
    n_new = new_vocab_size - old_vocab_size

    # === 1. Расширяем embedding ===
    old_emb_weight = model.tok_emb.weight.data.clone()
    new_emb = nn.Embedding(new_vocab_size, d_model)

    # Копируем старые веса
    new_emb.weight.data[:old_vocab_size] = old_emb_weight

    # Инициализируем новые
    if init_method == 'mean':
        mean_emb = old_emb_weight.mean(dim=0)
        new_emb.weight.data[old_vocab_size:] = mean_emb.unsqueeze(0).expand(n_new, -1)
    elif init_method == 'random':
        std = old_emb_weight.std().item()
        new_emb.weight.data[old_vocab_size:] = torch.randn(n_new, d_model) * std
    elif init_method == 'zero':
        new_emb.weight.data[old_vocab_size:] = 0
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    model.tok_emb = new_emb

    # === 2. Расширяем output head ===
    if model.cfg.weight_tying:
        # Weight tying: head.weight = tok_emb.weight
        new_head = nn.Linear(d_model, new_vocab_size, bias=False)
        new_head.weight = model.tok_emb.weight
        model.head = new_head
    else:
        old_head_weight = model.head.weight.data.clone()
        new_head = nn.Linear(d_model, new_vocab_size, bias=False)
        new_head.weight.data[:old_vocab_size] = old_head_weight

        if init_method == 'mean':
            mean_head = old_head_weight.mean(dim=0)
            new_head.weight.data[old_vocab_size:] = mean_head.unsqueeze(0).expand(n_new, -1)
        elif init_method == 'random':
            std = old_head_weight.std().item()
            new_head.weight.data[old_vocab_size:] = torch.randn(n_new, d_model) * std
        elif init_method == 'zero':
            new_head.weight.data[old_vocab_size:] = 0

        model.head = new_head

    # Обновляем конфиг
    model.cfg.vocab_size = new_vocab_size

    return model


def shrink_vocab(model, new_vocab_size):
    """
    Сжимает словарь модели (удаляет последние токены).

    Args:
        model: YiJingGPT модель
        new_vocab_size: новый размер (< текущего)

    Returns:
        model с уменьшенным словарём
    """
    old_vocab_size = model.cfg.vocab_size
    assert new_vocab_size < old_vocab_size

    d_model = model.cfg.d_model

    # Embedding
    new_emb = nn.Embedding(new_vocab_size, d_model)
    new_emb.weight.data = model.tok_emb.weight.data[:new_vocab_size].clone()
    model.tok_emb = new_emb

    # Head
    if model.cfg.weight_tying:
        new_head = nn.Linear(d_model, new_vocab_size, bias=False)
        new_head.weight = model.tok_emb.weight
        model.head = new_head
    else:
        new_head = nn.Linear(d_model, new_vocab_size, bias=False)
        new_head.weight.data = model.head.weight.data[:new_vocab_size].clone()
        model.head = new_head

    model.cfg.vocab_size = new_vocab_size
    return model
