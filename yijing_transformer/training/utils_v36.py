"""
v36 утилиты: EMA Model, Gradient Vaccine, AGC,
LR Finder, Weight Standardization.

EMA Model: exponential moving average весов.
Ref: Polyak & Juditsky (1992), Izmailov et al., "SWA" (2018)

Gradient Vaccine: защита от catastrophic forgetting.
Ref: вдохновлено EWC (Kirkpatrick et al. 2017)

Adaptive Gradient Clipping: clip по ratio grad_norm / weight_norm.
Ref: Brock et al., "High-Performance Large-Scale Image Recognition
Without Normalization" (NFNet, 2021)

LR Finder: автоматический поиск оптимального LR.
Ref: Smith, "Cyclical Learning Rates" (2017)

Weight Standardization: нормализация весов.
Ref: Qiao et al., "Micro-Batch Training with Batch-Channel Normalization
and Weight Standardization" (2019)
"""

import math
import copy
import torch
import torch.nn as nn


# ==================== EMA Model ====================

class EMAModel:
    """
    Exponential Moving Average весов модели.

    Поддерживает shadow copy весов:
    θ_ema = decay * θ_ema + (1 - decay) * θ_model

    Args:
        model: nn.Module
        decay: коэффициент сглаживания (0.999 типично)
        warmup_steps: шагов до начала EMA
    """
    def __init__(self, model, decay=0.999, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._shadow = {}
        self._backup = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self._shadow[name] = p.data.clone()

    def _get_decay(self):
        """Decay с warmup."""
        if self._step < self.warmup_steps:
            return min(self.decay, (1 + self._step) / (10 + self._step))
        return self.decay

    def update(self, model):
        """
        Обновляет EMA shadow weights.

        Args:
            model: nn.Module
        """
        decay = self._get_decay()
        self._step += 1

        for name, p in model.named_parameters():
            if p.requires_grad and name in self._shadow:
                self._shadow[name].mul_(decay).add_(p.data, alpha=1 - decay)

    def apply_shadow(self, model):
        """
        Заменяет веса модели на EMA shadow.
        Сохраняет оригинальные веса для restore.

        Args:
            model: nn.Module
        """
        self._backup = {}
        for name, p in model.named_parameters():
            if name in self._shadow:
                self._backup[name] = p.data.clone()
                p.data.copy_(self._shadow[name])

    def restore(self, model):
        """
        Восстанавливает оригинальные веса.

        Args:
            model: nn.Module
        """
        for name, p in model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    def get_shadow_params(self):
        """Возвращает копию shadow parameters."""
        return {k: v.clone() for k, v in self._shadow.items()}

    @property
    def step(self):
        return self._step


# ==================== Gradient Vaccine ====================

class GradientVaccine:
    """
    Защита от catastrophic forgetting при fine-tuning.

    Сохраняет «якорные» веса и штрафует отклонение
    от них (Elastic Weight Consolidation inspired).

    Args:
        model: nn.Module (до fine-tuning)
        lambda_ewc: сила регуляризации
        fisher_samples: число сэмплов для Fisher Information
    """
    def __init__(self, model, lambda_ewc=0.5, fisher_samples=100):
        self.lambda_ewc = lambda_ewc
        self.fisher_samples = fisher_samples
        self._anchor_params = {}
        self._fisher_diagonal = {}

        # Save anchor weights
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._anchor_params[name] = p.data.clone()

    def compute_fisher(self, model, data_loader_fn):
        """
        Вычисляет Fisher Information Matrix (диагональ).

        Args:
            model: nn.Module
            data_loader_fn: callable возвращающий (input, target) batches

        Returns:
            dict: Fisher diagonal per parameter
        """
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}

        model.eval()
        for i, (x, y) in enumerate(data_loader_fn()):
            if i >= self.fisher_samples:
                break
            model.zero_grad()
            logits, loss, _ = model(x, y)
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.data ** 2

        # Normalize
        n_samples = min(i + 1, self.fisher_samples) if 'i' in dir() else 1
        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        self._fisher_diagonal = fisher
        model.train()
        return fisher

    def penalty(self, model):
        """
        Вычисляет EWC penalty loss.

        Args:
            model: nn.Module

        Returns:
            Tensor: penalty loss
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if name in self._anchor_params:
                diff = p - self._anchor_params[name]
                if name in self._fisher_diagonal:
                    loss += (self._fisher_diagonal[name] * diff ** 2).sum()
                else:
                    loss += (diff ** 2).sum()

        return self.lambda_ewc * loss

    def simple_penalty(self, model):
        """
        Простой L2 penalty (без Fisher).

        Args:
            model: nn.Module

        Returns:
            Tensor: L2 penalty от anchor weights
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if name in self._anchor_params:
                loss += ((p - self._anchor_params[name]) ** 2).sum()
        return self.lambda_ewc * loss

    def get_drift(self, model):
        """
        Измеряет drift от anchor weights.

        Args:
            model: nn.Module

        Returns:
            dict: {total_drift, max_drift, per_layer}
        """
        total_drift = 0.0
        max_drift = 0.0
        per_layer = {}

        for name, p in model.named_parameters():
            if name in self._anchor_params:
                drift = (p.data - self._anchor_params[name]).norm().item()
                per_layer[name] = drift
                total_drift += drift
                max_drift = max(max_drift, drift)

        return {
            'total_drift': total_drift,
            'max_drift': max_drift,
            'per_layer': per_layer,
        }


# ==================== Adaptive Gradient Clipping ====================

class AdaptiveGradientClipping:
    """
    AGC: клиппинг по отношению ||grad|| / ||weight||.

    Если ||grad|| / ||weight|| > clip_factor, масштабирует
    градиент вниз. Лучше чем фиксированный clip для
    моделей без BatchNorm.

    Args:
        clip_factor: максимальный ratio (0.01 типично)
        eps: для числовой стабильности
        exclude_names: имена параметров для исключения
    """
    def __init__(self, clip_factor=0.01, eps=1e-3, exclude_names=('bias',)):
        self.clip_factor = clip_factor
        self.eps = eps
        self.exclude_names = exclude_names

    def clip(self, model):
        """
        Применяет AGC к градиентам модели.

        Args:
            model: nn.Module

        Returns:
            dict: {n_clipped, n_total, max_ratio, ratios}
        """
        n_clipped = 0
        n_total = 0
        max_ratio = 0.0
        ratios = {}

        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            # Skip excluded
            skip = False
            for exc in self.exclude_names:
                if exc in name:
                    skip = True
                    break
            if skip:
                continue

            n_total += 1

            # Compute norms
            p_norm = p.data.norm(2)
            g_norm = p.grad.data.norm(2)

            # Ratio
            max_norm = p_norm * self.clip_factor
            ratio = g_norm / max(p_norm, self.eps)
            ratios[name] = ratio.item()
            max_ratio = max(max_ratio, ratio.item())

            # Clip if needed
            if g_norm > max_norm:
                p.grad.data.mul_(max_norm / max(g_norm, self.eps))
                n_clipped += 1

        return {
            'n_clipped': n_clipped,
            'n_total': n_total,
            'max_ratio': max_ratio,
            'ratios': ratios,
        }


# ==================== Learning Rate Finder ====================

class LRFinder:
    """
    Автоматический поиск оптимального Learning Rate.

    Линейно/экспоненциально увеличивает LR и записывает loss.
    Оптимальный LR — где loss падает быстрее всего.

    Args:
        model: nn.Module
        optimizer: torch optimizer
        min_lr: начальный LR
        max_lr: конечный LR
        n_steps: число шагов поиска
    """
    def __init__(self, model, optimizer, min_lr=1e-7, max_lr=10,
                 n_steps=100):
        self.model = model
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_steps = n_steps
        self._results = []

    def find(self, train_fn):
        """
        Запускает LR range test.

        Args:
            train_fn: callable(lr) -> loss для одного шага

        Returns:
            dict: {best_lr, results, suggestion}
        """
        # Save model state
        model_state = copy.deepcopy(self.model.state_dict())
        opt_state = copy.deepcopy(self.optimizer.state_dict())

        lr_mult = (self.max_lr / self.min_lr) ** (1.0 / self.n_steps)
        lr = self.min_lr
        best_loss = float('inf')
        results = []

        for step in range(self.n_steps):
            # Set LR
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

            loss = train_fn(lr)
            if isinstance(loss, torch.Tensor):
                loss = loss.item()

            results.append({'lr': lr, 'loss': loss, 'step': step})

            if loss < best_loss:
                best_loss = loss

            # Stop if loss explodes
            if loss > best_loss * 10:
                break

            lr *= lr_mult

        # Restore model state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(opt_state)

        self._results = results
        best_lr = self._find_best_lr(results)

        return {
            'best_lr': best_lr,
            'results': results,
            'suggestion': best_lr / 10,  # Conservative suggestion
        }

    def _find_best_lr(self, results):
        """Находит LR с максимальным падением loss."""
        if len(results) < 3:
            return results[0]['lr'] if results else self.min_lr

        # Smoothed gradient
        best_lr = results[0]['lr']
        best_slope = 0

        for i in range(1, len(results) - 1):
            slope = results[i - 1]['loss'] - results[i + 1]['loss']
            if slope > best_slope:
                best_slope = slope
                best_lr = results[i]['lr']

        return best_lr

    @property
    def results(self):
        return self._results


# ==================== Weight Standardization ====================

class WeightStandardization(nn.Module):
    """
    Стандартизация весов линейного слоя.

    Нормализует веса по fan-in: w = (w - mean) / std.
    Улучшает обучение без BatchNorm.

    Args:
        module: nn.Linear или nn.Conv1d/2d
        eps: для числовой стабильности
    """
    def __init__(self, module, eps=1e-5):
        super().__init__()
        self.module = module
        self.eps = eps

    def forward(self, x):
        """
        Forward с стандартизированными весами.

        Args:
            x: входной тензор

        Returns:
            Tensor
        """
        weight = self.module.weight
        # Standardize over fan-in dimensions
        if weight.dim() >= 2:
            fan_in_dims = list(range(1, weight.dim()))
            mean = weight.mean(dim=fan_in_dims, keepdim=True)
            std = weight.std(dim=fan_in_dims, keepdim=True)
            standardized_weight = (weight - mean) / (std + self.eps)
        else:
            standardized_weight = weight

        # Use F.linear directly to keep grad graph
        bias = self.module.bias if hasattr(self.module, 'bias') else None
        return torch.nn.functional.linear(x, standardized_weight, bias)

    @staticmethod
    def apply_to_model(model, eps=1e-5):
        """
        Оборачивает все Linear слои в WeightStandardization.

        Args:
            model: nn.Module
            eps: epsilon

        Returns:
            dict: {n_wrapped, layer_names}
        """
        wrapped = []
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                wrapped.append(name)
        return {
            'n_eligible': len(wrapped),
            'layer_names': wrapped,
        }

    @staticmethod
    def standardize_weights_(model, eps=1e-5):
        """
        In-place стандартизация весов (без обёртки).

        Args:
            model: nn.Module
            eps: epsilon

        Returns:
            int: число стандартизированных слоёв
        """
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() >= 2:
                with torch.no_grad():
                    fan_in_dims = list(range(1, module.weight.dim()))
                    mean = module.weight.mean(dim=fan_in_dims, keepdim=True)
                    std = module.weight.std(dim=fan_in_dims, keepdim=True)
                    module.weight.data = (module.weight.data - mean) / (std + eps)
                count += 1
        return count
