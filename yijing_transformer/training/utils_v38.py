"""
v38 утилиты: Multi-Scale Loss, Gradient Accumulation Scheduler,
Parameter Freezing Scheduler, Checkpoint Manager, Cosine Warm Restarts.

Multi-Scale Loss: loss на разных уровнях модели.
Ref: Lee et al., "Deeply-Supervised Nets" (2015)

Gradient Accumulation Scheduler: динамические accum steps.

Parameter Freezing: постепенная разморозка.
Ref: Howard & Ruder, "ULMFiT" (2018)

Checkpoint Manager: автосохранение top-k чекпойнтов.

Cosine Warm Restarts: SGDR.
Ref: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent
with Warm Restarts" (2017)
"""

import math
import torch
import torch.nn as nn
from collections import deque


# ==================== Multi-Scale Loss ====================

class MultiScaleLoss:
    """
    Loss на разных масштабах/уровнях модели.

    Суммирует loss от промежуточных слоёв с весами,
    позволяя глубокому supervision.

    Args:
        scales: список весов для каждого масштаба
        reduction: 'mean' или 'sum'
    """
    def __init__(self, scales=None, reduction='mean'):
        self.scales = scales or [1.0, 0.5, 0.25]
        self.reduction = reduction

    def compute(self, losses):
        """
        Взвешенная сумма losses на разных масштабах.

        Args:
            losses: list[Tensor] — losses от разных слоёв

        Returns:
            dict: {total_loss, per_scale, weights_used}
        """
        n = min(len(losses), len(self.scales))
        weights = self.scales[:n]

        # Normalize weights
        if self.reduction == 'mean':
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]

        total = sum(w * l for w, l in zip(weights, losses[:n]))
        per_scale = [{'weight': w, 'loss': l.item()} for w, l in zip(weights, losses[:n])]

        return {
            'total_loss': total,
            'per_scale': per_scale,
            'weights_used': weights,
        }

    def compute_with_projections(self, hidden_states, target, loss_fn, projections):
        """
        Вычисляет loss на промежуточных hidden states через проекции.

        Args:
            hidden_states: list[Tensor] — (B, T, D) от разных слоёв
            target: target tensor
            loss_fn: callable(logits, target) -> loss
            projections: list[nn.Module] — проекции hidden->vocab

        Returns:
            dict: {total_loss, per_scale}
        """
        losses = []
        for h, proj in zip(hidden_states, projections):
            logits = proj(h)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target.view(-1))
            losses.append(loss)

        return self.compute(losses)


# ==================== Gradient Accumulation Scheduler ====================

class GradAccumScheduler:
    """
    Динамическое изменение числа gradient accumulation steps.

    Увеличивает effective batch size по мере обучения
    без изменения memory footprint.

    Args:
        initial_steps: начальное число accum steps
        max_steps: максимальное число
        increase_every: шагов между увеличениями
        increase_factor: множитель
    """
    def __init__(self, initial_steps=1, max_steps=16,
                 increase_every=1000, increase_factor=2):
        self.initial_steps = initial_steps
        self.max_steps = max_steps
        self.increase_every = increase_every
        self.increase_factor = increase_factor
        self._current_steps = initial_steps
        self._global_step = 0
        self._history = []

    def get_accum_steps(self):
        """Текущее число accumulation steps."""
        return self._current_steps

    def step(self):
        """
        Обновляет счётчик.

        Returns:
            dict: {accum_steps, changed, global_step}
        """
        self._global_step += 1
        changed = False

        if (self._global_step % self.increase_every == 0 and
                self._current_steps < self.max_steps):
            old = self._current_steps
            self._current_steps = min(
                self._current_steps * self.increase_factor,
                self.max_steps
            )
            if self._current_steps != old:
                changed = True
                self._history.append({
                    'step': self._global_step,
                    'old': old,
                    'new': self._current_steps,
                })

        return {
            'accum_steps': self._current_steps,
            'changed': changed,
            'global_step': self._global_step,
        }

    def get_effective_batch_size(self, micro_batch_size):
        """Эффективный batch size."""
        return micro_batch_size * self._current_steps

    def get_info(self):
        return {
            'current_steps': self._current_steps,
            'global_step': self._global_step,
            'history': self._history[-5:],
        }


# ==================== Parameter Freezing Scheduler ====================

class ParameterFreezingScheduler:
    """
    Постепенная разморозка слоёв (gradual unfreezing).

    Сначала замораживает все слои кроме верхних,
    затем постепенно размораживает снизу вверх.

    Args:
        model: nn.Module
        unfreeze_every: шагов между разморозкой слоёв
        initial_unfrozen: число начально разморожённых слоёв (сверху)
    """
    def __init__(self, model, unfreeze_every=500, initial_unfrozen=1):
        self.model = model
        self.unfreeze_every = unfreeze_every
        self._step = 0
        self._layer_names = []
        self._frozen_layers = set()

        # Collect named children as layers
        for name, _ in model.named_children():
            self._layer_names.append(name)

        # Freeze all except top initial_unfrozen
        n_freeze = max(0, len(self._layer_names) - initial_unfrozen)
        for i, name in enumerate(self._layer_names):
            if i < n_freeze:
                self._freeze_layer(name)
                self._frozen_layers.add(name)

    def _freeze_layer(self, name):
        """Замораживает слой."""
        for n, p in self.model.named_parameters():
            if n.startswith(name + '.') or n == name:
                p.requires_grad = False

    def _unfreeze_layer(self, name):
        """Размораживает слой."""
        for n, p in self.model.named_parameters():
            if n.startswith(name + '.') or n == name:
                p.requires_grad = True

    def step(self):
        """
        Шаг: возможно размораживает следующий слой.

        Returns:
            dict: {unfrozen_layer, n_frozen, n_total}
        """
        self._step += 1
        unfrozen_layer = None

        if (self._step % self.unfreeze_every == 0 and
                len(self._frozen_layers) > 0):
            # Unfreeze from top of frozen layers (deepest first)
            frozen_list = [n for n in self._layer_names if n in self._frozen_layers]
            if frozen_list:
                to_unfreeze = frozen_list[-1]
                self._unfreeze_layer(to_unfreeze)
                self._frozen_layers.discard(to_unfreeze)
                unfrozen_layer = to_unfreeze

        return {
            'unfrozen_layer': unfrozen_layer,
            'n_frozen': len(self._frozen_layers),
            'n_total': len(self._layer_names),
        }

    def unfreeze_all(self):
        """Размораживает все слои."""
        for name in list(self._frozen_layers):
            self._unfreeze_layer(name)
        self._frozen_layers.clear()

    def get_frozen_layers(self):
        return list(self._frozen_layers)

    def get_trainable_params(self):
        """Число trainable параметров."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_info(self):
        return {
            'step': self._step,
            'frozen_layers': self.get_frozen_layers(),
            'n_frozen': len(self._frozen_layers),
            'n_total': len(self._layer_names),
            'trainable_params': self.get_trainable_params(),
        }


# ==================== Checkpoint Manager ====================

class CheckpointManager:
    """
    Менеджер чекпойнтов: сохраняет top-k лучших.

    Отслеживает метрику и хранит только лучшие
    чекпойнты (state_dict).

    Args:
        max_checkpoints: максимум хранимых чекпойнтов
        metric_name: имя метрики для сравнения
        mode: 'min' (loss) или 'max' (accuracy)
    """
    def __init__(self, max_checkpoints=3, metric_name='loss', mode='min'):
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        self._checkpoints = []  # list of (metric, step, state_dict)
        self._step = 0

    def _is_better(self, new_metric, old_metric):
        if self.mode == 'min':
            return new_metric < old_metric
        return new_metric > old_metric

    def update(self, model, metric_value, extra_info=None):
        """
        Проверяет метрику и возможно сохраняет чекпойнт.

        Args:
            model: nn.Module
            metric_value: значение метрики
            extra_info: дополнительная информация

        Returns:
            dict: {saved, rank, best_metric}
        """
        self._step += 1

        # Check if should save
        should_save = False
        if len(self._checkpoints) < self.max_checkpoints:
            should_save = True
        else:
            worst_metric = self._checkpoints[-1][0]
            if self._is_better(metric_value, worst_metric):
                should_save = True

        if should_save:
            state = {
                'model_state_dict': {k: v.clone() for k, v in model.state_dict().items()},
                'metric': metric_value,
                'step': self._step,
                'extra': extra_info,
            }

            self._checkpoints.append((metric_value, self._step, state))

            # Sort: best first
            reverse = (self.mode == 'max')
            self._checkpoints.sort(key=lambda x: x[0], reverse=reverse)

            # Trim
            if len(self._checkpoints) > self.max_checkpoints:
                self._checkpoints = self._checkpoints[:self.max_checkpoints]

        rank = None
        for i, (m, s, _) in enumerate(self._checkpoints):
            if s == self._step and m == metric_value:
                rank = i + 1
                break

        best_metric = self._checkpoints[0][0] if self._checkpoints else None

        return {
            'saved': should_save,
            'rank': rank,
            'best_metric': best_metric,
            'n_checkpoints': len(self._checkpoints),
        }

    def get_best(self):
        """Возвращает лучший чекпойнт."""
        if not self._checkpoints:
            return None
        return self._checkpoints[0][2]

    def load_best(self, model):
        """
        Загружает лучший чекпойнт в модель.

        Args:
            model: nn.Module

        Returns:
            dict or None: info о загруженном чекпойнте
        """
        best = self.get_best()
        if best is None:
            return None
        model.load_state_dict(best['model_state_dict'])
        return {'metric': best['metric'], 'step': best['step']}

    def get_summary(self):
        return {
            'n_checkpoints': len(self._checkpoints),
            'best_metric': self._checkpoints[0][0] if self._checkpoints else None,
            'all_metrics': [(m, s) for m, s, _ in self._checkpoints],
        }


# ==================== Cosine Annealing with Warm Restarts ====================

class CosineWarmRestarts:
    """
    SGDR: Cosine Annealing with Warm Restarts.

    LR следует cosine schedule с периодическими
    перезапусками, каждый цикл длиннее предыдущего.

    Args:
        base_lr: начальный LR
        min_lr: минимальный LR
        T_0: длина первого цикла (шагов)
        T_mult: множитель длины цикла
        warmup_steps: шагов warmup в начале каждого цикла
    """
    def __init__(self, base_lr=1e-3, min_lr=1e-6, T_0=100,
                 T_mult=2, warmup_steps=10):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmup_steps = warmup_steps
        self._step = 0
        self._cycle = 0
        self._step_in_cycle = 0
        self._cycle_length = T_0

    def get_lr(self):
        """
        Текущий LR.

        Returns:
            float
        """
        # Warmup phase
        if self._step_in_cycle < self.warmup_steps:
            progress = self._step_in_cycle / max(self.warmup_steps, 1)
            return self.min_lr + (self.base_lr - self.min_lr) * progress

        # Cosine phase
        effective_step = self._step_in_cycle - self.warmup_steps
        effective_length = self._cycle_length - self.warmup_steps

        if effective_length <= 0:
            return self.base_lr

        cosine = math.cos(math.pi * effective_step / effective_length)
        return self.min_lr + (self.base_lr - self.min_lr) * (1 + cosine) / 2

    def step(self):
        """
        Шаг scheduler.

        Returns:
            dict: {lr, cycle, step_in_cycle, restart}
        """
        self._step += 1
        self._step_in_cycle += 1
        restart = False

        if self._step_in_cycle >= self._cycle_length:
            self._cycle += 1
            self._step_in_cycle = 0
            self._cycle_length = int(self._cycle_length * self.T_mult)
            restart = True

        return {
            'lr': self.get_lr(),
            'cycle': self._cycle,
            'step_in_cycle': self._step_in_cycle,
            'restart': restart,
        }

    def apply_to_optimizer(self, optimizer):
        """Применяет текущий LR к optimizer."""
        lr = self.get_lr()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

    @property
    def current_cycle(self):
        return self._cycle

    def get_info(self):
        return {
            'lr': self.get_lr(),
            'step': self._step,
            'cycle': self._cycle,
            'cycle_length': self._cycle_length,
            'step_in_cycle': self._step_in_cycle,
        }
