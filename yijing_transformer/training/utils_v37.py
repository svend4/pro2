"""
v37 утилиты: Gradient Centralization, Lookahead, SWA,
Batch Size Warmup, Gradient Penalty (R1).

Gradient Centralization: центрирование градиентов.
Ref: Yong et al., "Gradient Centralization" (2020)

Lookahead: slow/fast weight updates.
Ref: Zhang et al., "Lookahead Optimizer" (2019)

SWA: stochastic weight averaging.
Ref: Izmailov et al., "Averaging Weights Leads to Wider Optima" (2018)

Batch Size Warmup: постепенное увеличение batch size.

Gradient Penalty (R1): регуляризация градиентов.
Ref: Mescheder et al., "Which Training Methods for GANs
do actually Converge?" (2018)
"""

import math
import copy
import torch
import torch.nn as nn
from collections import deque


# ==================== Gradient Centralization ====================

class GradientCentralization:
    """
    Центрирование градиентов: вычитание среднего по fan-in.

    Ускоряет сходимость и улучшает генерализацию
    без дополнительных гиперпараметров.

    Args:
        apply_to_conv: применять к conv слоям
        apply_to_linear: применять к linear слоям
    """
    def __init__(self, apply_to_conv=True, apply_to_linear=True):
        self.apply_to_conv = apply_to_conv
        self.apply_to_linear = apply_to_linear

    def centralize(self, model):
        """
        Центрирует градиенты модели in-place.

        Args:
            model: nn.Module (после backward)

        Returns:
            dict: {n_centralized, layer_names}
        """
        n_centralized = 0
        layer_names = []

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            if p.grad.dim() < 2:
                continue  # Skip biases and 1D params

            # Centralize: subtract mean over fan-in dims
            fan_in_dims = list(range(1, p.grad.dim()))
            mean = p.grad.mean(dim=fan_in_dims, keepdim=True)
            p.grad.data.sub_(mean)
            n_centralized += 1
            layer_names.append(name)

        return {
            'n_centralized': n_centralized,
            'layer_names': layer_names,
        }

    def centralize_optimizer(self, optimizer):
        """
        Центрирует градиенты через optimizer param groups.

        Args:
            optimizer: torch optimizer

        Returns:
            int: число центрированных параметров
        """
        count = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None or p.grad.dim() < 2:
                    continue
                fan_in_dims = list(range(1, p.grad.dim()))
                mean = p.grad.mean(dim=fan_in_dims, keepdim=True)
                p.grad.data.sub_(mean)
                count += 1
        return count


# ==================== Lookahead Optimizer ====================

class Lookahead:
    """
    Lookahead optimizer wrapper.

    Поддерживает «медленные» веса, которые обновляются
    раз в k шагов в сторону «быстрых» весов.

    fast_weights обновляются inner optimizer каждый шаг.
    slow_weights = slow + alpha * (fast - slow) каждые k шагов.

    Args:
        optimizer: внутренний optimizer
        k: шагов между slow updates
        alpha: шаг в сторону fast weights
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow_params = []

        # Save slow copies
        for group in optimizer.param_groups:
            slow_group = []
            for p in group['params']:
                slow_group.append(p.data.clone())
            self._slow_params.append(slow_group)

    def step(self):
        """
        Выполняет шаг optimizer + lookahead update.

        Returns:
            dict: {inner_step, lookahead_update}
        """
        self.optimizer.step()
        self._step_count += 1

        lookahead_update = False
        if self._step_count % self.k == 0:
            self._update_slow()
            lookahead_update = True

        return {
            'inner_step': self._step_count,
            'lookahead_update': lookahead_update,
        }

    def _update_slow(self):
        """Обновляет slow weights."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for p_idx, p in enumerate(group['params']):
                slow = self._slow_params[group_idx][p_idx]
                # slow = slow + alpha * (fast - slow)
                slow.add_(p.data - slow, alpha=self.alpha)
                # Set fast = slow
                p.data.copy_(slow)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def get_info(self):
        return {
            'step_count': self._step_count,
            'k': self.k,
            'alpha': self.alpha,
            'next_slow_update': self.k - (self._step_count % self.k),
        }


# ==================== Stochastic Weight Averaging ====================

class SWACollector:
    """
    Stochastic Weight Averaging.

    Усредняет веса модели на поздних этапах обучения
    для нахождения более плоских минимумов.

    Args:
        model: nn.Module
        swa_start: шаг начала SWA
        swa_freq: частота сбора весов
    """
    def __init__(self, model, swa_start=0, swa_freq=1):
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self._n_averaged = 0
        self._avg_params = {}
        self._step = 0

        for name, p in model.named_parameters():
            if p.requires_grad:
                self._avg_params[name] = torch.zeros_like(p.data)

    def update(self, model):
        """
        Обновляет running average.

        Args:
            model: nn.Module

        Returns:
            dict: {collected, n_averaged, step}
        """
        self._step += 1
        collected = False

        if self._step >= self.swa_start and self._step % self.swa_freq == 0:
            self._n_averaged += 1
            for name, p in model.named_parameters():
                if name in self._avg_params:
                    # Running average: avg = avg + (new - avg) / n
                    self._avg_params[name].add_(
                        (p.data - self._avg_params[name]) / self._n_averaged
                    )
            collected = True

        return {
            'collected': collected,
            'n_averaged': self._n_averaged,
            'step': self._step,
        }

    def apply_averaged(self, model):
        """
        Заменяет веса модели на усреднённые.

        Args:
            model: nn.Module
        """
        if self._n_averaged == 0:
            return

        for name, p in model.named_parameters():
            if name in self._avg_params:
                p.data.copy_(self._avg_params[name])

    def get_averaged_params(self):
        """Возвращает копию усреднённых параметров."""
        return {k: v.clone() for k, v in self._avg_params.items()}

    @property
    def n_averaged(self):
        return self._n_averaged


# ==================== Batch Size Warmup ====================

class BatchSizeWarmup:
    """
    Постепенное увеличение batch size.

    Начинает с малого batch size и линейно/экспоненциально
    увеличивает до целевого.

    Args:
        initial_batch_size: начальный batch size
        target_batch_size: целевой batch size
        warmup_steps: шагов разогрева
        strategy: 'linear' или 'exponential'
    """
    def __init__(self, initial_batch_size=4, target_batch_size=32,
                 warmup_steps=100, strategy='linear'):
        self.initial_batch_size = initial_batch_size
        self.target_batch_size = target_batch_size
        self.warmup_steps = warmup_steps
        self.strategy = strategy
        self._step = 0

    def get_batch_size(self):
        """
        Текущий batch size.

        Returns:
            int
        """
        if self._step >= self.warmup_steps:
            return self.target_batch_size

        progress = self._step / max(self.warmup_steps, 1)

        if self.strategy == 'linear':
            bs = self.initial_batch_size + (
                self.target_batch_size - self.initial_batch_size
            ) * progress
        elif self.strategy == 'exponential':
            log_start = math.log(max(self.initial_batch_size, 1))
            log_end = math.log(max(self.target_batch_size, 1))
            bs = math.exp(log_start + (log_end - log_start) * progress)
        else:
            bs = self.target_batch_size

        # Round to nearest power of 2 or just int
        return max(1, int(bs))

    def step(self):
        """Увеличить шаг."""
        self._step += 1

    def get_lr_scale(self):
        """
        Масштаб LR пропорционально batch size (linear scaling rule).

        Returns:
            float
        """
        return self.get_batch_size() / self.target_batch_size

    @property
    def current_step(self):
        return self._step

    @property
    def is_warmup_done(self):
        return self._step >= self.warmup_steps

    def get_info(self):
        return {
            'batch_size': self.get_batch_size(),
            'step': self._step,
            'warmup_done': self.is_warmup_done,
            'lr_scale': self.get_lr_scale(),
        }


# ==================== Gradient Penalty (R1) ====================

class GradientPenalty:
    """
    R1 Gradient Penalty.

    Штрафует норму градиентов выхода по входу.
    Стабилизирует обучение (особенно для GAN-like моделей).

    penalty = lambda * ||grad(output, input)||^2

    Args:
        lambda_gp: сила penalty
        norm_type: 'l2' или 'l1'
    """
    def __init__(self, lambda_gp=10.0, norm_type='l2'):
        self.lambda_gp = lambda_gp
        self.norm_type = norm_type

    def compute(self, model, real_input, output_fn=None):
        """
        Вычисляет gradient penalty.

        Args:
            model: nn.Module
            real_input: входной тензор (requires_grad)
            output_fn: callable(model, input) -> scalar output
                        если None, используется model(input).sum()

        Returns:
            dict: {penalty, grad_norm}
        """
        real_input = real_input.detach().float()
        real_input.requires_grad_(True)

        if output_fn is not None:
            output = output_fn(model, real_input)
        else:
            output = model(real_input)
            if isinstance(output, tuple):
                output = output[0]
            output = output.sum()

        grad = torch.autograd.grad(
            outputs=output,
            inputs=real_input,
            create_graph=True,
            retain_graph=True,
        )[0]

        if self.norm_type == 'l2':
            grad_norm = grad.reshape(grad.shape[0], -1).norm(2, dim=1)
            penalty = self.lambda_gp * (grad_norm ** 2).mean()
        else:  # l1
            grad_norm = grad.reshape(grad.shape[0], -1).norm(1, dim=1)
            penalty = self.lambda_gp * grad_norm.mean()

        return {
            'penalty': penalty,
            'grad_norm': grad_norm.mean().item(),
        }

    def compute_simple(self, model):
        """
        Простой gradient penalty на параметрах.

        Штрафует норму весов, не требует входных данных.

        Args:
            model: nn.Module

        Returns:
            Tensor: penalty
        """
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for p in model.parameters():
            if p.requires_grad:
                penalty += p.norm(2) ** 2
        return self.lambda_gp * penalty
