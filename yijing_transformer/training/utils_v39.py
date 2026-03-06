"""
v39 утилиты: Gradient Projection, Loss Spike Recovery,
Optimizer State Pruning, Scheduled Dropout, Weight Decay Scheduler.

Gradient Projection: проекция градиентов для multi-task.
Ref: Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad, 2020)

Loss Spike Recovery: откат при loss spike.

Optimizer State Pruning: очистка momentum/state.

Scheduled Dropout: динамический dropout.
Ref: Rennie et al., "Self-Critical Sequence Training" (2017)

Weight Decay Scheduler: динамический weight decay.
"""

import math
import copy
import torch
import torch.nn as nn
from collections import deque


# ==================== Gradient Projection ====================

class GradientProjection:
    """
    Проекция градиентов для multi-task learning.

    Если градиенты двух задач конфликтуют (отрицательное
    скалярное произведение), проецирует один на плоскость,
    ортогональную другому.

    Args:
        n_tasks: число задач
    """
    def __init__(self, n_tasks=2):
        self.n_tasks = n_tasks

    def project(self, grads):
        """
        PCGrad: проекция конфликтующих градиентов.

        Args:
            grads: list[Tensor] — градиенты от разных задач
                   (уже flattened в 1D)

        Returns:
            dict: {projected_grads, n_conflicts, conflict_pairs}
        """
        projected = [g.clone() for g in grads]
        n_conflicts = 0
        conflict_pairs = []

        for i in range(len(grads)):
            for j in range(len(grads)):
                if i == j:
                    continue
                dot = torch.dot(projected[i], grads[j])
                if dot < 0:
                    # Project: remove component along grads[j]
                    norm_sq = grads[j].norm() ** 2
                    if norm_sq > 1e-12:
                        projected[i] -= (dot / norm_sq) * grads[j]
                    n_conflicts += 1
                    conflict_pairs.append((i, j))

        return {
            'projected_grads': projected,
            'n_conflicts': n_conflicts,
            'conflict_pairs': conflict_pairs,
        }

    def project_model_grads(self, model, task_losses):
        """
        Проекция градиентов модели от нескольких loss.

        Args:
            model: nn.Module
            task_losses: list[Tensor] — losses от разных задач

        Returns:
            dict: {n_conflicts, applied}
        """
        # Compute per-task gradients
        task_grads = []
        for loss in task_losses:
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad = torch.cat([
                p.grad.flatten() for p in model.parameters()
                if p.grad is not None
            ])
            task_grads.append(grad)

        # Project
        result = self.project(task_grads)

        # Average projected grads
        avg_grad = sum(result['projected_grads']) / len(result['projected_grads'])

        # Apply back to model
        idx = 0
        model.zero_grad()
        for p in model.parameters():
            if p.requires_grad:
                n = p.numel()
                p.grad = avg_grad[idx:idx + n].reshape(p.shape).clone()
                idx += n

        return {
            'n_conflicts': result['n_conflicts'],
            'applied': True,
        }


# ==================== Loss Spike Recovery ====================

class LossSpikeRecovery:
    """
    Автоматическое восстановление после loss spike.

    Сохраняет snapshot модели и optimizer.
    При spike — откатывает к последнему хорошему состоянию.

    Args:
        spike_threshold: множитель для определения spike
        window_size: окно для baseline loss
        max_rollbacks: максимум откатов подряд
    """
    def __init__(self, spike_threshold=5.0, window_size=50, max_rollbacks=3):
        self.spike_threshold = spike_threshold
        self.window_size = window_size
        self.max_rollbacks = max_rollbacks
        self._loss_history = deque(maxlen=window_size)
        self._snapshot = None
        self._rollback_count = 0
        self._total_rollbacks = 0

    def save_snapshot(self, model, optimizer):
        """
        Сохраняет snapshot текущего состояния.

        Args:
            model: nn.Module
            optimizer: torch optimizer
        """
        self._snapshot = {
            'model': copy.deepcopy(model.state_dict()),
            'optimizer': copy.deepcopy(optimizer.state_dict()),
        }

    def check_and_recover(self, model, optimizer, loss_value):
        """
        Проверяет spike и восстанавливает при необходимости.

        Args:
            model: nn.Module
            optimizer: torch optimizer
            loss_value: текущий loss

        Returns:
            dict: {spike_detected, rolled_back, baseline_loss}
        """
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()

        # Check for nan/inf
        if math.isnan(loss_value) or math.isinf(loss_value):
            if self._snapshot and self._rollback_count < self.max_rollbacks:
                model.load_state_dict(self._snapshot['model'])
                optimizer.load_state_dict(self._snapshot['optimizer'])
                self._rollback_count += 1
                self._total_rollbacks += 1
                return {'spike_detected': True, 'rolled_back': True, 'baseline_loss': None}
            return {'spike_detected': True, 'rolled_back': False, 'baseline_loss': None}

        # Compute baseline
        baseline = None
        spike = False
        if len(self._loss_history) >= 10:
            baseline = sum(self._loss_history) / len(self._loss_history)
            if loss_value > baseline * self.spike_threshold:
                spike = True

        rolled_back = False
        if spike and self._snapshot and self._rollback_count < self.max_rollbacks:
            model.load_state_dict(self._snapshot['model'])
            optimizer.load_state_dict(self._snapshot['optimizer'])
            self._rollback_count += 1
            self._total_rollbacks += 1
            rolled_back = True
        else:
            self._loss_history.append(loss_value)
            self._rollback_count = 0  # Reset consecutive count
            # Periodically save snapshot
            if len(self._loss_history) % 10 == 0 and len(self._loss_history) >= 10:
                self.save_snapshot(model, optimizer)

        return {
            'spike_detected': spike,
            'rolled_back': rolled_back,
            'baseline_loss': baseline,
        }

    def get_stats(self):
        return {
            'total_rollbacks': self._total_rollbacks,
            'has_snapshot': self._snapshot is not None,
            'history_len': len(self._loss_history),
        }


# ==================== Optimizer State Pruning ====================

class OptimizerStatePruner:
    """
    Очистка optimizer state для экономии памяти.

    Сбрасывает momentum/variance для параметров
    с малым gradient flow или замороженных.

    Args:
        threshold: порог gradient norm для pruning
    """
    def __init__(self, threshold=1e-7):
        self.threshold = threshold

    def prune_dead_states(self, optimizer, model):
        """
        Удаляет optimizer state для параметров без градиентов.

        Args:
            optimizer: torch optimizer
            model: nn.Module

        Returns:
            dict: {n_pruned, memory_saved_params}
        """
        n_pruned = 0
        memory_saved = 0

        for group in optimizer.param_groups:
            for p in group['params']:
                if not p.requires_grad or p.grad is None:
                    if p in optimizer.state:
                        for v in optimizer.state[p].values():
                            if isinstance(v, torch.Tensor):
                                memory_saved += v.numel()
                        del optimizer.state[p]
                        n_pruned += 1

        return {
            'n_pruned': n_pruned,
            'memory_saved_params': memory_saved,
        }

    def prune_small_grad_states(self, optimizer, model):
        """
        Сбрасывает state для параметров с малыми градиентами.

        Args:
            optimizer: torch optimizer
            model: nn.Module

        Returns:
            dict: {n_reset, names}
        """
        n_reset = 0
        names = []

        param_to_name = {id(p): n for n, p in model.named_parameters()}

        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm = p.grad.data.norm().item()
                    if grad_norm < self.threshold and p in optimizer.state:
                        # Reset state
                        for key in list(optimizer.state[p].keys()):
                            v = optimizer.state[p][key]
                            if isinstance(v, torch.Tensor):
                                optimizer.state[p][key] = torch.zeros_like(v)
                        n_reset += 1
                        name = param_to_name.get(id(p), 'unknown')
                        names.append(name)

        return {
            'n_reset': n_reset,
            'names': names,
        }

    def get_state_memory(self, optimizer):
        """
        Оценка памяти optimizer state.

        Returns:
            dict: {total_params, total_bytes_estimate}
        """
        total = 0
        for state in optimizer.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel()

        return {
            'total_params': total,
            'total_bytes_estimate': total * 4,  # float32
        }


# ==================== Scheduled Dropout ====================

class ScheduledDropout(nn.Module):
    """
    Dropout с расписанием: rate меняется по ходу обучения.

    Начинает с высокого dropout и постепенно уменьшает
    (или наоборот).

    Args:
        initial_rate: начальный dropout rate
        final_rate: конечный dropout rate
        total_steps: шагов для перехода
        strategy: 'linear', 'cosine', 'step'
    """
    def __init__(self, initial_rate=0.5, final_rate=0.1,
                 total_steps=10000, strategy='linear'):
        super().__init__()
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_steps = total_steps
        self.strategy = strategy
        self._step = 0
        self._current_rate = initial_rate

    def get_rate(self):
        """Текущий dropout rate."""
        progress = min(self._step / max(self.total_steps, 1), 1.0)

        if self.strategy == 'linear':
            rate = self.initial_rate + (self.final_rate - self.initial_rate) * progress
        elif self.strategy == 'cosine':
            cosine = (1 + math.cos(math.pi * progress)) / 2
            rate = self.final_rate + (self.initial_rate - self.final_rate) * cosine
        elif self.strategy == 'step':
            if progress < 0.33:
                rate = self.initial_rate
            elif progress < 0.66:
                rate = (self.initial_rate + self.final_rate) / 2
            else:
                rate = self.final_rate
        else:
            rate = self.initial_rate

        self._current_rate = rate
        return rate

    def forward(self, x):
        """
        Применяет scheduled dropout.

        Args:
            x: входной тензор

        Returns:
            Tensor
        """
        if not self.training:
            return x
        rate = self.get_rate()
        return torch.nn.functional.dropout(x, p=rate, training=True)

    def step(self):
        """Увеличить счётчик."""
        self._step += 1

    def apply_to_model(self, model):
        """
        Обновляет все Dropout слои модели.

        Args:
            model: nn.Module

        Returns:
            int: число обновлённых слоёв
        """
        rate = self.get_rate()
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = rate
                count += 1
        return count

    def get_info(self):
        return {
            'current_rate': self.get_rate(),
            'step': self._step,
            'strategy': self.strategy,
        }


# ==================== Weight Decay Scheduler ====================

class WeightDecayScheduler:
    """
    Динамический weight decay.

    Изменяет weight decay по ходу обучения.
    Может увеличивать для лучшей регуляризации
    на поздних этапах.

    Args:
        initial_wd: начальный weight decay
        final_wd: конечный weight decay
        total_steps: шагов для перехода
        strategy: 'linear', 'cosine', 'constant_then_decay'
    """
    def __init__(self, initial_wd=0.01, final_wd=0.1,
                 total_steps=10000, strategy='linear'):
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.total_steps = total_steps
        self.strategy = strategy
        self._step = 0

    def get_weight_decay(self):
        """Текущий weight decay."""
        progress = min(self._step / max(self.total_steps, 1), 1.0)

        if self.strategy == 'linear':
            wd = self.initial_wd + (self.final_wd - self.initial_wd) * progress
        elif self.strategy == 'cosine':
            cosine = (1 - math.cos(math.pi * progress)) / 2
            wd = self.initial_wd + (self.final_wd - self.initial_wd) * cosine
        elif self.strategy == 'constant_then_decay':
            if progress < 0.5:
                wd = self.initial_wd
            else:
                sub_progress = (progress - 0.5) * 2
                wd = self.initial_wd + (self.final_wd - self.initial_wd) * sub_progress
        else:
            wd = self.initial_wd

        return wd

    def step(self):
        """Увеличить счётчик."""
        self._step += 1

    def apply_to_optimizer(self, optimizer):
        """
        Применяет текущий weight decay к optimizer.

        Args:
            optimizer: torch optimizer

        Returns:
            float: текущий weight decay
        """
        wd = self.get_weight_decay()
        for group in optimizer.param_groups:
            group['weight_decay'] = wd
        return wd

    def get_info(self):
        return {
            'weight_decay': self.get_weight_decay(),
            'step': self._step,
            'strategy': self.strategy,
        }
