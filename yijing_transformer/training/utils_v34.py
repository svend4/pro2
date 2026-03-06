"""
v34 утилиты: Safe Gradient Accumulation, LLRD, Token Dropout,
Convergence Detector, Activation Checkpointing Manager.

Safe Gradient Accumulation: FP16-safe accumulation с overflow detection.

LLRD: Layer-wise Learning Rate Decay.
Ref: Clark et al., "ELECTRA" (2020) — используется в fine-tuning

Token Dropout: случайное удаление токенов для регуляризации.
Ref: Hou et al., "Token Dropping for Efficient BERT" (2022)

Convergence Detector: обнаружение сходимости обучения.

Activation Checkpointing: выборочный gradient checkpointing.
Ref: Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ==================== Safe Gradient Accumulation ====================

class SafeGradAccumulator:
    """
    Безопасное накопление градиентов с overflow detection.

    Отслеживает inf/nan в градиентах и пропускает
    шаги с overflow. Полезно для mixed precision.

    Args:
        accum_steps: число шагов накопления
        max_grad_norm: клиппинг нормы (0 = выкл)
        skip_on_overflow: пропускать шаг при overflow
    """
    def __init__(self, accum_steps=4, max_grad_norm=1.0, skip_on_overflow=True):
        self.accum_steps = accum_steps
        self.max_grad_norm = max_grad_norm
        self.skip_on_overflow = skip_on_overflow
        self._micro_step = 0
        self._overflow_count = 0
        self._total_steps = 0
        self._skipped_steps = 0

    def check_overflow(self, model):
        """
        Проверяет наличие inf/nan в градиентах.

        Args:
            model: nn.Module

        Returns:
            bool: True если overflow обнаружен
        """
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    return True
        return False

    def accumulate_step(self, model, loss):
        """
        Один micro-step накопления.

        Args:
            model: nn.Module
            loss: Tensor loss (уже scaled)

        Returns:
            dict: {should_step, overflow, micro_step}
        """
        self._micro_step += 1
        scaled_loss = loss / self.accum_steps
        scaled_loss.backward()

        overflow = self.check_overflow(model)
        if overflow:
            self._overflow_count += 1

        should_step = self._micro_step >= self.accum_steps

        return {
            'should_step': should_step,
            'overflow': overflow,
            'micro_step': self._micro_step,
        }

    def optimizer_step(self, model, optimizer):
        """
        Выполняет optimizer step с проверками.

        Args:
            model: nn.Module
            optimizer: torch optimizer

        Returns:
            dict: {stepped, overflow, grad_norm}
        """
        self._total_steps += 1
        overflow = self.check_overflow(model)

        if overflow and self.skip_on_overflow:
            optimizer.zero_grad()
            self._micro_step = 0
            self._skipped_steps += 1
            return {'stepped': False, 'overflow': True, 'grad_norm': float('inf')}

        # Clip gradients
        grad_norm = 0.0
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_grad_norm
            ).item()
        else:
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = math.sqrt(grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        self._micro_step = 0

        return {'stepped': True, 'overflow': False, 'grad_norm': grad_norm}

    @property
    def overflow_rate(self):
        if self._total_steps == 0:
            return 0.0
        return self._overflow_count / self._total_steps

    @property
    def skip_rate(self):
        if self._total_steps == 0:
            return 0.0
        return self._skipped_steps / self._total_steps

    def get_stats(self):
        return {
            'total_steps': self._total_steps,
            'overflow_count': self._overflow_count,
            'skipped_steps': self._skipped_steps,
            'overflow_rate': self.overflow_rate,
        }

    def reset(self):
        self._micro_step = 0
        self._overflow_count = 0
        self._total_steps = 0
        self._skipped_steps = 0


# ==================== Layer-wise Learning Rate Decay ====================

class LayerwiseLRDecay:
    """
    Layer-wise Learning Rate Decay (LLRD).

    Нижние слои получают меньший LR, верхние — больший.
    LR_layer = base_lr * decay_rate^(n_layers - layer_idx)

    Args:
        model: nn.Module
        base_lr: базовый LR (для верхнего слоя)
        decay_rate: коэффициент затухания (0.9 типично)
        no_decay_names: имена параметров без weight decay
    """
    def __init__(self, model, base_lr=1e-3, decay_rate=0.9,
                 no_decay_names=('bias', 'LayerNorm', 'layer_norm')):
        self.model = model
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.no_decay_names = no_decay_names

    def get_param_groups(self, weight_decay=0.01):
        """
        Создаёт param groups с разными LR.

        Args:
            weight_decay: базовый weight decay

        Returns:
            list[dict]: param groups для optimizer
        """
        # Collect layers by depth
        layers = self._get_layer_depths()

        param_groups = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            depth = self._get_param_depth(name, layers)
            lr = self.base_lr * (self.decay_rate ** depth)

            # Check no_decay
            wd = weight_decay
            for nd_name in self.no_decay_names:
                if nd_name in name:
                    wd = 0.0
                    break

            param_groups.append({
                'params': [p],
                'lr': lr,
                'weight_decay': wd,
                'name': name,
            })

        return param_groups

    def _get_layer_depths(self):
        """Определяет глубину каждого именованного модуля."""
        layers = {}
        depth = 0
        for name, _ in self.model.named_modules():
            if name:
                layers[name] = depth
                depth += 1
        return layers

    def _get_param_depth(self, param_name, layers):
        """Определяет глубину параметра (сколько слоёв от верха)."""
        max_depth = max(layers.values()) if layers else 0
        best_depth = max_depth  # Default: deepest (highest LR)

        for layer_name, depth in layers.items():
            if param_name.startswith(layer_name):
                # Invert: lower layers get higher depth value
                best_depth = min(best_depth, max_depth - depth)

        return best_depth

    def get_lr_summary(self, weight_decay=0.01):
        """Сводка LR по слоям."""
        groups = self.get_param_groups(weight_decay)
        summary = []
        for g in groups:
            summary.append({
                'name': g['name'],
                'lr': g['lr'],
                'weight_decay': g['weight_decay'],
            })
        return summary


# ==================== Token Dropout ====================

class TokenDropout(nn.Module):
    """
    Случайное удаление токенов из последовательности.

    Во время обучения с вероятностью p удаляет токены,
    сжимая последовательность. Улучшает робастность.

    Args:
        drop_prob: вероятность удаления токена
        min_tokens: минимум оставшихся токенов
    """
    def __init__(self, drop_prob=0.1, min_tokens=2):
        super().__init__()
        self.drop_prob = drop_prob
        self.min_tokens = min_tokens

    def forward(self, input_ids, attention_mask=None):
        """
        Применяет token dropout.

        Args:
            input_ids: (B, T) token ids
            attention_mask: (B, T) optional mask

        Returns:
            dict: {input_ids, attention_mask, kept_indices, drop_rate}
        """
        if not self.training or self.drop_prob == 0:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'kept_indices': None,
                'drop_rate': 0.0,
            }

        B, T = input_ids.shape
        keep_len = max(self.min_tokens, int(T * (1 - self.drop_prob)))

        # Generate keep mask per batch
        keep_indices = []
        new_ids = []
        new_masks = []

        for b in range(B):
            # Random permutation, keep first keep_len
            perm = torch.randperm(T, device=input_ids.device)[:keep_len]
            perm = perm.sort().values  # Keep original order

            new_ids.append(input_ids[b, perm])
            if attention_mask is not None:
                new_masks.append(attention_mask[b, perm])
            keep_indices.append(perm)

        new_input_ids = torch.stack(new_ids)
        new_attention_mask = torch.stack(new_masks) if attention_mask is not None else None

        return {
            'input_ids': new_input_ids,
            'attention_mask': new_attention_mask,
            'kept_indices': keep_indices,
            'drop_rate': 1.0 - keep_len / T,
        }


# ==================== Training Convergence Detector ====================

class ConvergenceDetector:
    """
    Детектор сходимости обучения.

    Определяет, когда обучение сошлось (loss перестал улучшаться).

    Критерии:
    - loss plateau: loss не улучшается за patience шагов
    - gradient vanishing: градиенты стали очень малыми
    - lr exhausted: LR достиг минимума

    Args:
        patience: шагов терпения
        min_delta: минимальное улучшение для считания прогресса
        window_size: окно для сглаживания
    """
    def __init__(self, patience=100, min_delta=1e-4, window_size=50):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self._loss_history = deque(maxlen=window_size * 2)
        self._best_loss = float('inf')
        self._steps_without_improvement = 0
        self._converged = False
        self._convergence_step = None

    def check(self, loss_value, grad_norm=None, lr=None):
        """
        Проверяет сходимость.

        Args:
            loss_value: текущий loss
            grad_norm: норма градиента (опционально)
            lr: текущий LR (опционально)

        Returns:
            dict: {converged, reason, steps_without_improvement,
                   best_loss, current_smoothed}
        """
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()

        self._loss_history.append(loss_value)

        # Smoothed loss
        recent = list(self._loss_history)[-self.window_size:]
        smoothed = sum(recent) / len(recent)

        # Check improvement
        if smoothed < self._best_loss - self.min_delta:
            self._best_loss = smoothed
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        reason = None
        converged = False

        # Loss plateau
        if self._steps_without_improvement >= self.patience:
            converged = True
            reason = 'loss_plateau'

        # Gradient vanishing
        if grad_norm is not None and grad_norm < 1e-8:
            converged = True
            reason = 'gradient_vanishing'

        if converged and not self._converged:
            self._converged = True
            self._convergence_step = len(self._loss_history)

        return {
            'converged': converged,
            'reason': reason,
            'steps_without_improvement': self._steps_without_improvement,
            'best_loss': self._best_loss,
            'current_smoothed': smoothed,
        }

    @property
    def is_converged(self):
        return self._converged

    @property
    def convergence_step(self):
        return self._convergence_step

    def reset(self):
        self._loss_history.clear()
        self._best_loss = float('inf')
        self._steps_without_improvement = 0
        self._converged = False
        self._convergence_step = None


# ==================== Activation Checkpointing Manager ====================

class ActivationCheckpointManager:
    """
    Менеджер выборочного gradient checkpointing.

    Определяет, какие слои нужно checkpoint'ить
    для оптимального баланса memory/compute.

    Args:
        model: nn.Module
        checkpoint_ratio: доля слоёв для checkpointing (0-1)
        strategy: 'uniform' или 'deepest' (checkpoint самые глубокие)
    """
    def __init__(self, model, checkpoint_ratio=0.5, strategy='uniform'):
        self.model = model
        self.checkpoint_ratio = checkpoint_ratio
        self.strategy = strategy
        self._checkpointed_layers = set()
        self._original_forwards = {}

    def get_layers_to_checkpoint(self):
        """
        Определяет какие слои checkpoint'ить.

        Returns:
            list[str]: имена слоёв
        """
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer,
                                   nn.TransformerDecoderLayer,
                                   nn.MultiheadAttention)):
                layers.append(name)

        # Also check for custom transformer layers
        for name, module in self.model.named_modules():
            if hasattr(module, 'self_attn') or hasattr(module, 'attention'):
                if name not in layers:
                    layers.append(name)

        if not layers:
            # Fallback: use children
            for name, _ in self.model.named_children():
                layers.append(name)

        # Select subset
        n_to_checkpoint = max(1, int(len(layers) * self.checkpoint_ratio))

        if self.strategy == 'deepest':
            # Checkpoint deepest layers (most memory-intensive)
            selected = layers[-n_to_checkpoint:]
        elif self.strategy == 'uniform':
            # Evenly spaced
            step = max(1, len(layers) // n_to_checkpoint)
            selected = [layers[i] for i in range(0, len(layers), step)][:n_to_checkpoint]
        else:
            selected = layers[:n_to_checkpoint]

        return selected

    def wrap_checkpoint(self, module, *args, **kwargs):
        """
        Обёртка для gradient checkpointing одного модуля.

        Args:
            module: nn.Module
            *args: входные аргументы

        Returns:
            output
        """
        if self.model.training:
            return torch_checkpoint(module, *args, use_reentrant=False, **kwargs)
        return module(*args, **kwargs)

    def estimate_memory_savings(self):
        """
        Оценка экономии памяти.

        Returns:
            dict: {layers_checkpointed, estimated_savings_pct}
        """
        layers = self.get_layers_to_checkpoint()
        total_layers = sum(1 for _ in self.model.named_children())
        n_ckpt = len(layers)

        # Rough estimate: each checkpointed layer saves ~50% activation memory
        savings = n_ckpt / max(total_layers, 1) * 50

        return {
            'layers_checkpointed': layers,
            'n_checkpointed': n_ckpt,
            'n_total_layers': total_layers,
            'estimated_savings_pct': min(savings, 80),
        }

    def get_info(self):
        """Информация о checkpointing."""
        return {
            'checkpoint_ratio': self.checkpoint_ratio,
            'strategy': self.strategy,
            'selected_layers': self.get_layers_to_checkpoint(),
        }
