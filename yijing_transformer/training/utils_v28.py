"""
v28 утилиты: Lookahead Optimizer, SWA, Gradient Accumulation Manager,
Label Smoothing с Warmup, Batch Size Finder.

Lookahead: интерполяция fast/slow весов для стабильности.
Ref: Zhang et al., "Lookahead Optimizer" (2019)

SWA: усреднение весов по траектории для обобщения.
Ref: Izmailov et al., "Averaging Weights Leads to Wider Optima" (2018)

Gradient Accumulation Manager: динамическое накопление градиентов.

Label Smoothing with Warmup: прогрессивное сглаживание меток.

Batch Size Finder: автоматический подбор batch size.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ==================== Lookahead Optimizer ====================

class Lookahead:
    """
    Lookahead Optimizer wrapper.

    Поддерживает slow weights, которые обновляются раз в k шагов
    через интерполяцию с fast weights.

    θ_slow = θ_slow + α * (θ_fast - θ_slow)

    Args:
        optimizer: базовый оптимизатор (fast weights)
        k: число inner steps между slow updates
        alpha: интерполяция (0.5 типично)
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow_params = []

        # Cache slow weights
        for group in optimizer.param_groups:
            slow_group = []
            for p in group['params']:
                slow_group.append(p.data.clone())
            self._slow_params.append(slow_group)

    def step(self):
        """Шаг fast optimizer + periodic slow update."""
        self.optimizer.step()
        self._step_count += 1

        if self._step_count % self.k == 0:
            self._update_slow()

    def _update_slow(self):
        """Interpolate slow weights toward fast weights."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for p_idx, p in enumerate(group['params']):
                slow = self._slow_params[group_idx][p_idx]
                slow.add_(self.alpha * (p.data - slow))
                p.data.copy_(slow)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def sync_lookahead(self):
        """Force sync slow weights to fast."""
        self._update_slow()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_params': self._slow_params,
            'step_count': self._step_count,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self._slow_params = state['slow_params']
        self._step_count = state['step_count']


# ==================== Stochastic Weight Averaging ====================

class SWA:
    """
    Stochastic Weight Averaging.

    Усредняет веса модели по траектории обучения.
    Начинает сбор после swa_start шагов.

    Args:
        model: nn.Module
        swa_start: шаг начала SWA
        swa_freq: частота сбора snapshot'ов
    """
    def __init__(self, model, swa_start=100, swa_freq=10):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self._swa_state = {}
        self._n_averaged = 0

    def update(self, step):
        """
        Обновляет SWA средние если пора.

        Args:
            step: текущий шаг

        Returns:
            bool: True если обновлено
        """
        if step < self.swa_start:
            return False
        if (step - self.swa_start) % self.swa_freq != 0:
            return False

        self._n_averaged += 1
        for name, p in self.model.named_parameters():
            if name not in self._swa_state:
                self._swa_state[name] = p.data.clone()
            else:
                # Running average: avg = avg + (new - avg) / n
                self._swa_state[name].add_(
                    (p.data - self._swa_state[name]) / self._n_averaged
                )
        return True

    def apply_swa_weights(self):
        """Применяет SWA веса к модели."""
        if self._n_averaged == 0:
            return False
        for name, p in self.model.named_parameters():
            if name in self._swa_state:
                p.data.copy_(self._swa_state[name])
        return True

    def get_swa_weights(self):
        """Возвращает SWA веса как state_dict."""
        return dict(self._swa_state)

    @property
    def n_averaged(self):
        return self._n_averaged

    def reset(self):
        self._swa_state.clear()
        self._n_averaged = 0


# ==================== Gradient Accumulation Manager ====================

class GradAccumulationManager:
    """
    Умный менеджер gradient accumulation.

    Поддерживает:
    - Фиксированный accumulation
    - Динамическое масштабирование на основе loss variance
    - Автоматическая нормализация

    Args:
        base_accum_steps: базовое число шагов накопления
        dynamic: адаптивное число шагов
        max_accum_steps: максимальное число шагов
        variance_window: окно для оценки variance loss
    """
    def __init__(self, base_accum_steps=4, dynamic=False,
                 max_accum_steps=32, variance_window=50):
        self.base_accum_steps = base_accum_steps
        self.dynamic = dynamic
        self.max_accum_steps = max_accum_steps
        self.current_accum_steps = base_accum_steps
        self._micro_step = 0
        self._loss_history = deque(maxlen=variance_window)

    def should_step(self):
        """
        Нужно ли делать optimizer step?

        Returns:
            bool: True если пора сделать step
        """
        self._micro_step += 1
        return self._micro_step >= self.current_accum_steps

    def reset_micro_steps(self):
        """Сбросить micro step counter после optimizer step."""
        self._micro_step = 0

    def scale_loss(self, loss):
        """
        Масштабирует loss для accumulation.

        Args:
            loss: Tensor

        Returns:
            scaled loss
        """
        self._loss_history.append(loss.item())
        return loss / self.current_accum_steps

    def adapt(self):
        """
        Адаптирует число шагов на основе loss variance.

        Высокая variance → больше accumulation для стабильности.

        Returns:
            dict: {accum_steps, loss_variance}
        """
        if not self.dynamic or len(self._loss_history) < 10:
            return {
                'accum_steps': self.current_accum_steps,
                'loss_variance': 0.0,
            }

        losses = list(self._loss_history)
        mean = sum(losses) / len(losses)
        variance = sum((x - mean) ** 2 for x in losses) / len(losses)

        # High variance → more accumulation
        # Heuristic: scale base steps by (1 + variance)
        factor = min(1 + variance, self.max_accum_steps / self.base_accum_steps)
        new_steps = min(
            int(self.base_accum_steps * factor),
            self.max_accum_steps
        )
        # Round to power of 2
        new_steps = max(1, 2 ** round(math.log2(new_steps)))
        self.current_accum_steps = min(new_steps, self.max_accum_steps)

        return {
            'accum_steps': self.current_accum_steps,
            'loss_variance': variance,
        }

    @property
    def micro_step(self):
        return self._micro_step

    def get_progress(self):
        """Прогресс текущего accumulation."""
        return self._micro_step / max(self.current_accum_steps, 1)


# ==================== Label Smoothing with Warmup ====================

class WarmupLabelSmoothing(nn.Module):
    """
    Label smoothing с постепенным включением.

    В начале обучения — hard labels (smoothing=0).
    Постепенно увеличивается до target_smoothing.

    Args:
        vocab_size: размер словаря
        target_smoothing: целевой уровень сглаживания
        warmup_steps: шагов для разогрева
        ignore_index: игнорируемый индекс (padding)
    """
    def __init__(self, vocab_size, target_smoothing=0.1,
                 warmup_steps=1000, ignore_index=-100):
        super().__init__()
        self.vocab_size = vocab_size
        self.target_smoothing = target_smoothing
        self.warmup_steps = warmup_steps
        self.ignore_index = ignore_index
        self._step = 0

    @property
    def current_smoothing(self):
        if self.warmup_steps <= 0:
            return self.target_smoothing
        ratio = min(self._step / self.warmup_steps, 1.0)
        return self.target_smoothing * ratio

    def forward(self, logits, targets):
        """
        Compute smoothed cross-entropy.

        Args:
            logits: (B, T, V) or (B, V)
            targets: (B, T) or (B,)

        Returns:
            loss: scalar
        """
        self._step += 1
        smoothing = self.current_smoothing

        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)

        # Create mask for non-ignored tokens
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[mask]
        targets = targets[mask]

        log_probs = F.log_softmax(logits, dim=-1)

        # Hard label part
        nll = F.nll_loss(log_probs, targets, reduction='mean')

        if smoothing == 0:
            return nll

        # Smooth part: uniform over vocab
        smooth_loss = -log_probs.mean(dim=-1).mean()

        loss = (1 - smoothing) * nll + smoothing * smooth_loss
        return loss

    def reset(self):
        self._step = 0


# ==================== Batch Size Finder ====================

class BatchSizeFinder:
    """
    Автоматический поиск оптимального batch size.

    Пробует batch sizes от малого к большому,
    находит максимальный, который помещается в память.

    Args:
        model: nn.Module
        min_batch: минимальный batch size
        max_batch: максимальный batch size
        seq_len: длина последовательности
        vocab_size: размер словаря
    """
    def __init__(self, model, min_batch=1, max_batch=256,
                 seq_len=128, vocab_size=64):
        self.model = model
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def find(self, device='cpu'):
        """
        Находит максимальный batch size.

        Args:
            device: устройство для теста

        Returns:
            dict: {max_batch_size, tested_sizes, recommended}
        """
        self.model.to(device)
        self.model.train()
        tested = {}
        best = self.min_batch

        # Binary search
        low, high = self.min_batch, self.max_batch
        while low <= high:
            mid = (low + high) // 2
            success = self._try_batch(mid, device)
            tested[mid] = success
            if success:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        # Recommend 80% of max for safety
        recommended = max(self.min_batch, int(best * 0.8))
        # Round to nearest power of 2
        if recommended > 1:
            recommended = 2 ** int(math.log2(recommended))

        return {
            'max_batch_size': best,
            'tested_sizes': tested,
            'recommended': max(1, recommended),
        }

    def _try_batch(self, batch_size, device):
        """Пробует один batch size."""
        try:
            x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=device)
            y = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=device)

            self.model.zero_grad()
            output = self.model(x, y)
            if isinstance(output, tuple) and len(output) >= 2:
                loss = output[1]
            else:
                loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()
            loss.backward()

            # Clean up
            del x, y, output, loss
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    def estimate_memory(self, batch_size):
        """
        Оценивает потребление памяти.

        Args:
            batch_size: размер батча

        Returns:
            dict: {param_mb, activation_estimate_mb}
        """
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        # Rough estimate: activations ≈ 2-4x params per sample
        activation_estimate = param_bytes * batch_size * 3

        return {
            'param_mb': param_bytes / (1024 * 1024),
            'activation_estimate_mb': activation_estimate / (1024 * 1024),
            'total_estimate_mb': (param_bytes + activation_estimate) / (1024 * 1024),
        }
