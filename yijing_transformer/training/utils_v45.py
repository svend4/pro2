"""
v45 утилиты: Gradient Centralization, AdaFactor-like LR Scaling,
Gradient Penalty, SAM, Lion Optimizer.

Gradient Centralization: центрирование градиентов.
Ref: Yong et al., "Gradient Centralization" (2020)

AdaFactor LR Scaling: масштабирование LR по √(size).
Ref: Shazeer & Stern, "Adafactor" (2018)

Gradient Penalty: штраф за норму градиентов.
Ref: Gulrajani et al., "Improved Training of WGANs" (2017)

SAM: Sharpness-Aware Minimization.
Ref: Foret et al., "Sharpness-Aware Minimization" (2021)

Lion: EvoLved Sign Momentum optimizer.
Ref: Chen et al., "Symbolic Discovery of Optimization Algorithms" (2023)
"""

import math
import torch
import torch.nn as nn


# ==================== Gradient Centralization ====================

class GradientCentralization:
    """
    Центрирование градиентов: вычитание среднего.

    GC(∇W) = ∇W - mean(∇W)

    Ускоряет сходимость и улучшает генерализацию.
    Применяется только к весам (не bias, не 1D).

    Args:
        apply_to_conv: применять к conv слоям
        apply_to_linear: применять к linear слоям
    """
    def __init__(self, apply_to_conv=True, apply_to_linear=True):
        self.apply_to_conv = apply_to_conv
        self.apply_to_linear = apply_to_linear
        self._stats = {'n_centralized': 0, 'n_skipped': 0}

    def centralize(self, model):
        """
        Центрирует градиенты модели (после backward).

        Args:
            model: nn.Module

        Returns:
            dict: {n_centralized, n_skipped}
        """
        n_cent = 0
        n_skip = 0

        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            # Only centralize weights with dim >= 2
            if p.grad.dim() < 2:
                n_skip += 1
                continue

            # Centralize: subtract mean over all dims except first
            dims = list(range(1, p.grad.dim()))
            p.grad.data.sub_(p.grad.data.mean(dim=dims, keepdim=True))
            n_cent += 1

        self._stats['n_centralized'] += n_cent
        self._stats['n_skipped'] += n_skip

        return {'n_centralized': n_cent, 'n_skipped': n_skip}

    def get_stats(self):
        return dict(self._stats)


# ==================== AdaFactor-like LR Scaling ====================

class AdaFactorLRScaling:
    """
    Масштабирование LR по размеру параметров.

    lr_scaled = lr * min(1/√step, 1/√d) * RMS(param)

    Автоматически подстраивает LR под масштаб параметров.

    Args:
        base_lr: базовый LR
        min_rms: минимальный RMS (для стабильности)
        warmup_steps: шагов warmup
    """
    def __init__(self, base_lr=1e-3, min_rms=1e-3, warmup_steps=1000):
        self.base_lr = base_lr
        self.min_rms = min_rms
        self.warmup_steps = warmup_steps
        self._step = 0

    def get_scaled_lr(self, param):
        """
        Вычисляет масштабированный LR для параметра.

        Args:
            param: nn.Parameter

        Returns:
            float: scaled LR
        """
        rms = max(param.data.norm() / max(param.data.numel() ** 0.5, 1), self.min_rms)

        # Step factor
        self._step += 1
        step_factor = min(1.0 / math.sqrt(self._step), 1.0)

        # Warmup
        warmup_factor = min(self._step / max(self.warmup_steps, 1), 1.0)

        return self.base_lr * rms * step_factor * warmup_factor

    def apply(self, optimizer):
        """
        Применяет scaling к param groups оптимизатора.

        Returns:
            list[float]: scaled LRs
        """
        self._step += 1
        step_factor = min(1.0 / math.sqrt(self._step), 1.0)
        warmup_factor = min(self._step / max(self.warmup_steps, 1), 1.0)

        lrs = []
        for group in optimizer.param_groups:
            # Compute average RMS across params in group
            total_rms = 0
            count = 0
            for p in group['params']:
                if p.requires_grad:
                    rms = p.data.norm() / max(p.data.numel() ** 0.5, 1)
                    total_rms += rms.item()
                    count += 1

            avg_rms = max(total_rms / max(count, 1), self.min_rms)
            lr = self.base_lr * avg_rms * step_factor * warmup_factor
            group['lr'] = lr
            lrs.append(lr)

        return lrs

    def get_info(self):
        return {
            'step': self._step,
            'base_lr': self.base_lr,
        }


# ==================== Gradient Penalty ====================

class GradientPenalty:
    """
    Штраф за норму градиентов параметров.

    L_gp = λ * ||∇_θ L||²

    Регуляризирует обучение, предотвращая резкие изменения.

    Args:
        lambda_gp: вес штрафа
        max_norm: максимальная норма (штрафовать только при превышении)
    """
    def __init__(self, lambda_gp=0.01, max_norm=None):
        self.lambda_gp = lambda_gp
        self.max_norm = max_norm
        self._history = []

    def compute(self, model):
        """
        Вычисляет gradient penalty.

        Args:
            model: nn.Module (после backward)

        Returns:
            dict: {penalty, grad_norm}
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if self.max_norm is not None and total_norm <= self.max_norm:
            penalty = 0.0
        else:
            penalty = self.lambda_gp * total_norm ** 2

        self._history.append(total_norm)
        return {'penalty': penalty, 'grad_norm': total_norm}

    def compute_differentiable(self, loss, model):
        """
        Дифференцируемый gradient penalty (через autograd).

        Args:
            loss: скалярный loss tensor
            model: nn.Module

        Returns:
            Tensor: penalty loss (можно добавить к основному loss)
        """
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True,
            allow_unused=True
        )

        penalty = torch.tensor(0.0, device=loss.device)
        for g in grads:
            if g is not None:
                penalty = penalty + g.norm(2) ** 2

        return self.lambda_gp * penalty

    def get_stats(self):
        if not self._history:
            return {'mean_norm': 0, 'max_norm': 0}
        return {
            'mean_norm': sum(self._history) / len(self._history),
            'max_norm': max(self._history),
        }


# ==================== SAM (Sharpness-Aware Minimization) ====================

class SAM:
    """
    Sharpness-Aware Minimization.

    Двухшаговая оптимизация:
    1. ε-шаг в направлении максимального loss (ascent)
    2. Обычный шаг минимизации в возмущённой точке

    Находит плоские минимумы для лучшей генерализации.

    Args:
        optimizer: базовый оптимизатор
        rho: радиус возмущения
    """
    def __init__(self, optimizer, rho=0.05):
        self.optimizer = optimizer
        self.rho = rho
        self._old_params = {}

    def first_step(self):
        """
        Шаг 1: возмущение параметров в направлении max loss.

        Вызывается после backward на обычном loss.
        """
        grad_norm = self._grad_norm()
        scale = self.rho / max(grad_norm, 1e-12)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Save original params
                self._old_params[id(p)] = p.data.clone()
                # Ascent step: move to adversarial point
                e_w = p.grad * scale
                p.data.add_(e_w)

        self.optimizer.zero_grad()

    def second_step(self):
        """
        Шаг 2: обычная минимизация в возмущённой точке.

        Вызывается после backward на loss в возмущённой точке.
        """
        # Restore original params
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if id(p) in self._old_params:
                    p.data.copy_(self._old_params[id(p)])

        # Normal optimizer step with gradients from perturbed point
        self.optimizer.step()
        self._old_params.clear()

    def _grad_norm(self):
        """L2 norm of gradients."""
        norm = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += p.grad.data.norm(2).item() ** 2
        return norm ** 0.5

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


# ==================== Lion Optimizer ====================

class Lion(torch.optim.Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer.

    update = sign(β₁ * m + (1 - β₁) * g)
    m = β₂ * m + (1 - β₂) * g

    Использует только sign — memory efficient.
    Рекомендуется LR в 3-10x меньше чем AdamW.

    Args:
        params: параметры модели
        lr: learning rate (рекомендуется 1e-4)
        betas: (β₁, β₂) momentum coefficients
        weight_decay: weight decay
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Один шаг оптимизации.

        Returns:
            Optional[float]: loss если closure предоставлен
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']

                # Weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # Update: sign of interpolation
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(torch.sign(update), alpha=-lr)

                # Momentum update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
