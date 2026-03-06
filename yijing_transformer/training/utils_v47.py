"""
v47 утилиты: Gradient Accumulation + Loss Scaling,
Parameter Noise, EMA Schedule, Multi-Objective Loss Balancer,
Training Checkpoint Manager.

Gradient Accumulation + Loss Scaling: накопление с масштабированием.
Ref: Micikevicius et al., "Mixed Precision Training" (2018)

Parameter Noise: шум в параметрах для exploration.
Ref: Plappert et al., "Parameter Space Noise for Exploration" (2018)

EMA Schedule: переменный decay для EMA.
Ref: Polyak averaging with schedule.

Multi-Objective Loss Balancer: автобалансировка losses.
Ref: Kendall et al., "Multi-Task Learning Using Uncertainty" (2018)

Checkpoint Manager: управление чекпоинтами обучения.
Ref: Standard training infrastructure.
"""

import math
import os
import json
import torch
import torch.nn as nn


# ==================== Gradient Accumulation + Loss Scaling ====================

class GradientAccumulatorWithScaling:
    """
    Градиентное накопление с динамическим loss scaling.

    Для mixed precision: масштабирует loss для предотвращения
    underflow в FP16, уменьшает scale при overflow.

    Args:
        accumulation_steps: число шагов накопления
        init_scale: начальный множитель loss
        growth_factor: множитель при увеличении scale
        backoff_factor: множитель при уменьшении scale
        growth_interval: шагов между увеличением scale
    """
    def __init__(self, accumulation_steps=4, init_scale=65536.0,
                 growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.accumulation_steps = accumulation_steps
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._step = 0
        self._accum_step = 0
        self._steps_since_growth = 0
        self._overflow_count = 0

    def scale_loss(self, loss):
        """
        Масштабирует loss.

        Args:
            loss: скалярный tensor

        Returns:
            Tensor: scaled loss / accumulation_steps
        """
        return loss * self.scale / self.accumulation_steps

    def should_step(self):
        """Нужно ли делать optimizer.step()."""
        self._accum_step += 1
        return self._accum_step >= self.accumulation_steps

    def step(self, optimizer, model):
        """
        Шаг оптимизатора с unscaling и overflow check.

        Args:
            optimizer: оптимизатор
            model: nn.Module

        Returns:
            dict: {stepped, overflow, scale}
        """
        self._accum_step = 0
        self._step += 1

        # Unscale gradients
        inv_scale = 1.0 / self.scale
        overflow = False

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(inv_scale)
                # Check for overflow
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    overflow = True
                    break

        if overflow:
            optimizer.zero_grad()
            self.scale *= self.backoff_factor
            self._overflow_count += 1
            self._steps_since_growth = 0
            return {'stepped': False, 'overflow': True, 'scale': self.scale}

        optimizer.step()
        optimizer.zero_grad()

        # Try growing scale
        self._steps_since_growth += 1
        if self._steps_since_growth >= self.growth_interval:
            self.scale *= self.growth_factor
            self._steps_since_growth = 0

        return {'stepped': True, 'overflow': False, 'scale': self.scale}

    def get_info(self):
        return {
            'scale': self.scale,
            'step': self._step,
            'overflow_count': self._overflow_count,
            'accum_steps': self.accumulation_steps,
        }


# ==================== Parameter Noise Injection ====================

class ParameterNoiseInjector:
    """
    Шум в параметрах модели (не градиентах).

    Добавляет шум перед forward pass, убирает после.
    Используется для exploration и регуляризации.

    Args:
        noise_std: стандартное отклонение шума
        adaptive: адаптировать по magnitude параметра
    """
    def __init__(self, noise_std=0.01, adaptive=True):
        self.noise_std = noise_std
        self.adaptive = adaptive
        self._saved_params = {}
        self._noised = False

    def inject(self, model):
        """
        Добавляет шум к параметрам.

        Args:
            model: nn.Module

        Returns:
            dict: {n_params, avg_noise_ratio}
        """
        n = 0
        total_ratio = 0.0

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._saved_params[name] = p.data.clone()

            if self.adaptive:
                param_std = max(p.data.std().item(), 1e-8)
                noise = torch.randn_like(p.data) * self.noise_std * param_std
            else:
                noise = torch.randn_like(p.data) * self.noise_std

            p.data.add_(noise)
            ratio = noise.norm().item() / max(p.data.norm().item(), 1e-8)
            total_ratio += ratio
            n += 1

        self._noised = True
        return {
            'n_params': n,
            'avg_noise_ratio': total_ratio / max(n, 1),
        }

    def restore(self, model):
        """
        Убирает шум — восстанавливает оригинальные параметры.

        Returns:
            bool: True если восстановлено
        """
        if not self._noised:
            return False

        for name, p in model.named_parameters():
            if name in self._saved_params:
                p.data.copy_(self._saved_params[name])

        self._saved_params.clear()
        self._noised = False
        return True


# ==================== EMA Schedule ====================

class EMASchedule:
    """
    EMA с переменным decay.

    Начинает с маленького decay (быстрое обновление),
    увеличивает до целевого (стабильность).

    decay(t) = min(target_decay, 1 - (1 + t)^(-power))

    Args:
        target_decay: целевой decay (0.999 типично)
        power: скорость роста
        warmup_steps: шагов до достижения целевого decay
    """
    def __init__(self, target_decay=0.999, power=2.0/3.0, warmup_steps=2000):
        self.target_decay = target_decay
        self.power = power
        self.warmup_steps = warmup_steps
        self._step = 0
        self._shadow = {}

    def get_decay(self):
        """Текущий decay."""
        self._step += 1
        if self.warmup_steps > 0:
            # Linear warmup to target
            progress = min(self._step / self.warmup_steps, 1.0)
            return self.target_decay * progress
        return min(self.target_decay, 1 - (1 + self._step) ** (-self.power))

    def update(self, model):
        """
        Обновляет EMA shadow параметры.

        Args:
            model: nn.Module

        Returns:
            dict: {decay, step}
        """
        decay = self.get_decay()

        for name, p in model.named_parameters():
            if name not in self._shadow:
                self._shadow[name] = p.data.clone()
            else:
                self._shadow[name].mul_(decay).add_(p.data, alpha=1 - decay)

        return {'decay': decay, 'step': self._step}

    def apply(self, model):
        """Применяет EMA к модели."""
        for name, p in model.named_parameters():
            if name in self._shadow:
                p.data.copy_(self._shadow[name])

    def get_info(self):
        return {
            'step': self._step,
            'n_params': len(self._shadow),
        }


# ==================== Multi-Objective Loss Balancer ====================

class MultiObjectiveLossBalancer:
    """
    Автоматическая балансировка нескольких loss-ов.

    Использует learnable log-variance (uncertainty weighting):
    L = Σ (1/(2σ²_i) * L_i + log(σ_i))

    Args:
        n_tasks: число задач/losses
        method: 'uncertainty', 'equal', 'gradnorm'
    """
    def __init__(self, n_tasks=2, method='uncertainty'):
        self.n_tasks = n_tasks
        self.method = method
        # Learnable log-variances
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self._history = {i: [] for i in range(n_tasks)}

    def combine(self, losses):
        """
        Комбинирует несколько losses.

        Args:
            losses: list[Tensor] или dict[str, Tensor]

        Returns:
            dict: {total_loss, weights, individual_losses}
        """
        if isinstance(losses, dict):
            loss_list = list(losses.values())
            names = list(losses.keys())
        else:
            loss_list = losses
            names = [f'task_{i}' for i in range(len(loss_list))]

        if self.method == 'uncertainty':
            total = torch.tensor(0.0, device=loss_list[0].device)
            weights = []
            for i, loss in enumerate(loss_list):
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * loss + self.log_vars[i]
                weights.append(precision.item())
        elif self.method == 'equal':
            total = sum(loss_list) / len(loss_list)
            weights = [1.0 / len(loss_list)] * len(loss_list)
        else:
            total = sum(loss_list) / len(loss_list)
            weights = [1.0 / len(loss_list)] * len(loss_list)

        # Record history
        for i, loss in enumerate(loss_list):
            self._history[i].append(loss.item())

        return {
            'total_loss': total,
            'weights': dict(zip(names, weights)),
            'individual_losses': dict(zip(names, [l.item() for l in loss_list])),
        }

    def parameters(self):
        """Возвращает learnable параметры."""
        return [self.log_vars]

    def get_stats(self):
        stats = {}
        for i, history in self._history.items():
            if history:
                stats[f'task_{i}_avg'] = sum(history[-100:]) / len(history[-100:])
        return stats


# ==================== Training Checkpoint Manager ====================

class CheckpointManager:
    """
    Управление чекпоинтами обучения.

    Сохраняет top-K лучших + периодические.
    Автоматически удаляет старые.

    Args:
        save_dir: директория для чекпоинтов
        max_to_keep: максимум чекпоинтов
        metric_name: имя метрики для сравнения
        mode: 'min' или 'max'
    """
    def __init__(self, save_dir='checkpoints', max_to_keep=5,
                 metric_name='loss', mode='min'):
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.metric_name = metric_name
        self.mode = mode
        self._checkpoints = []  # list of (metric, path, step)

    def save(self, model, optimizer, step, metric_value, extra=None):
        """
        Сохраняет чекпоинт если он в top-K.

        Args:
            model: nn.Module
            optimizer: оптимизатор
            step: номер шага
            metric_value: значение метрики
            extra: дополнительные данные

        Returns:
            dict: {saved, path, is_best}
        """
        is_best = self._is_best(metric_value)

        # Check if should save
        if len(self._checkpoints) >= self.max_to_keep:
            worst_idx = self._get_worst_idx()
            worst_metric = self._checkpoints[worst_idx][0]
            if self.mode == 'min' and metric_value >= worst_metric:
                return {'saved': False, 'path': None, 'is_best': False}
            elif self.mode == 'max' and metric_value <= worst_metric:
                return {'saved': False, 'path': None, 'is_best': False}
            # Remove worst
            _, old_path, _ = self._checkpoints.pop(worst_idx)

        path = os.path.join(self.save_dir, f'checkpoint_step{step}.pt')

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            self.metric_name: metric_value,
        }
        if extra:
            checkpoint.update(extra)

        self._checkpoints.append((metric_value, path, step))

        return {'saved': True, 'path': path, 'is_best': is_best}

    def _is_best(self, metric_value):
        if not self._checkpoints:
            return True
        best = self._get_best_metric()
        if self.mode == 'min':
            return metric_value < best
        return metric_value > best

    def _get_best_metric(self):
        metrics = [c[0] for c in self._checkpoints]
        return min(metrics) if self.mode == 'min' else max(metrics)

    def _get_worst_idx(self):
        metrics = [c[0] for c in self._checkpoints]
        if self.mode == 'min':
            return metrics.index(max(metrics))
        return metrics.index(min(metrics))

    def get_best(self):
        """Возвращает лучший чекпоинт."""
        if not self._checkpoints:
            return None
        if self.mode == 'min':
            return min(self._checkpoints, key=lambda x: x[0])
        return max(self._checkpoints, key=lambda x: x[0])

    def get_info(self):
        return {
            'n_checkpoints': len(self._checkpoints),
            'max_to_keep': self.max_to_keep,
            'best': self.get_best(),
        }
