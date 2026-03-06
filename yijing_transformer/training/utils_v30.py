"""
v30 утилиты: Gradient Centralization, Adaptive Gradient Clipping,
Loss Spike Detector, Parameter Freezing Scheduler, Training State Snapshotter.

Gradient Centralization: центрирование градиентов (zero-mean).
Ref: Yong et al., "Gradient Centralization" (2020)

AGC: клиппинг по соотношению grad/param норм.
Ref: Brock et al., "High-Performance Large-Scale Image Recognition Without Normalization" (2021)

Loss Spike Detector: обнаружение и обработка скачков loss.

Parameter Freezing Scheduler: постепенное размораживание слоёв.

Training State Snapshotter: лёгкие снимки для отката.
"""

import copy
import math
import torch
import torch.nn as nn
from collections import deque


# ==================== Gradient Centralization ====================

class GradientCentralization:
    """
    Gradient Centralization: вычитает среднее из градиентов.

    Для весов с ndim >= 2 центрирует градиенты по всем осям кроме выходной.
    Ускоряет сходимость и улучшает обобщение.

    Args:
        apply_to_conv: применять к conv слоям (ndim >= 4)
        apply_to_linear: применять к linear слоям (ndim == 2)
    """
    def __init__(self, apply_to_conv=True, apply_to_linear=True):
        self.apply_to_conv = apply_to_conv
        self.apply_to_linear = apply_to_linear

    def centralize(self, model):
        """
        Центрирует градиенты модели.

        Args:
            model: nn.Module

        Returns:
            dict: {n_centralized, n_skipped}
        """
        n_centralized = 0
        n_skipped = 0

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    n_skipped += 1
                    continue

                ndim = p.grad.ndim
                if ndim < 2:
                    n_skipped += 1
                    continue

                if ndim == 2 and not self.apply_to_linear:
                    n_skipped += 1
                    continue
                if ndim >= 4 and not self.apply_to_conv:
                    n_skipped += 1
                    continue

                # Centralize: subtract mean over all dims except output (dim 0)
                dims = tuple(range(1, ndim))
                p.grad.data.sub_(p.grad.data.mean(dim=dims, keepdim=True))
                n_centralized += 1

        return {'n_centralized': n_centralized, 'n_skipped': n_skipped}

    def centralize_params(self, parameters):
        """Центрирует градиенты заданных параметров."""
        n_centralized = 0
        with torch.no_grad():
            for p in parameters:
                if p.grad is None or p.grad.ndim < 2:
                    continue
                dims = tuple(range(1, p.grad.ndim))
                p.grad.data.sub_(p.grad.data.mean(dim=dims, keepdim=True))
                n_centralized += 1
        return n_centralized


# ==================== Adaptive Gradient Clipping ====================

class AdaptiveGradientClipping:
    """
    Adaptive Gradient Clipping (AGC).

    Клиппирует градиенты на основе соотношения ||grad|| / ||param||.
    Если ratio > clipping_factor, масштабирует градиент.

    Более стабильный чем фиксированный клиппинг для разных слоёв.

    Args:
        clipping_factor: максимальное допустимое соотношение (0.01 типично)
        eps: для числовой стабильности
    """
    def __init__(self, clipping_factor=0.01, eps=1e-3):
        self.clipping_factor = clipping_factor
        self.eps = eps
        self.last_stats = {}

    def clip(self, model):
        """
        Применяет AGC к модели.

        Args:
            model: nn.Module

        Returns:
            dict: {n_clipped, n_total, max_ratio}
        """
        n_clipped = 0
        n_total = 0
        max_ratio = 0.0

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                n_total += 1

                param_norm = p.data.norm(2)
                grad_norm = p.grad.data.norm(2)

                if param_norm < self.eps:
                    continue

                ratio = grad_norm / (param_norm + self.eps)
                max_ratio = max(max_ratio, ratio.item())

                if ratio > self.clipping_factor:
                    scale = self.clipping_factor / (ratio + self.eps)
                    p.grad.data.mul_(scale)
                    n_clipped += 1

        self.last_stats = {
            'n_clipped': n_clipped,
            'n_total': n_total,
            'max_ratio': max_ratio,
        }
        return self.last_stats

    def clip_params(self, named_parameters):
        """Применяет AGC к заданным параметрам."""
        n_clipped = 0
        with torch.no_grad():
            for name, p in named_parameters:
                if p.grad is None:
                    continue
                param_norm = p.data.norm(2)
                grad_norm = p.grad.data.norm(2)
                if param_norm < self.eps:
                    continue
                ratio = grad_norm / (param_norm + self.eps)
                if ratio > self.clipping_factor:
                    scale = self.clipping_factor / (ratio + self.eps)
                    p.grad.data.mul_(scale)
                    n_clipped += 1
        return n_clipped


# ==================== Loss Spike Detector ====================

class LossSpikeDetector:
    """
    Детектор скачков loss.

    Отслеживает loss и определяет аномальные скачки
    на основе скользящей статистики.

    Args:
        window_size: размер окна для статистики
        spike_threshold: порог в стандартных отклонениях
        patience: число последовательных спайков до тревоги
    """
    def __init__(self, window_size=100, spike_threshold=3.0, patience=3):
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.patience = patience
        self._history = deque(maxlen=window_size)
        self._consecutive_spikes = 0
        self._total_spikes = 0
        self._alarm_count = 0

    def check(self, loss_value):
        """
        Проверяет loss на спайк.

        Args:
            loss_value: текущее значение loss

        Returns:
            dict: {is_spike, is_alarm, loss, mean, std, z_score,
                   consecutive_spikes, total_spikes}
        """
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()

        # Need sufficient history
        if len(self._history) < 10:
            self._history.append(loss_value)
            return {
                'is_spike': False,
                'is_alarm': False,
                'loss': loss_value,
                'mean': loss_value,
                'std': 0.0,
                'z_score': 0.0,
                'consecutive_spikes': 0,
                'total_spikes': 0,
            }

        values = list(self._history)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1e-8

        z_score = (loss_value - mean) / max(std, 1e-8)
        is_spike = z_score > self.spike_threshold

        if is_spike:
            self._consecutive_spikes += 1
            self._total_spikes += 1
        else:
            self._consecutive_spikes = 0

        is_alarm = self._consecutive_spikes >= self.patience
        if is_alarm:
            self._alarm_count += 1

        self._history.append(loss_value)

        return {
            'is_spike': is_spike,
            'is_alarm': is_alarm,
            'loss': loss_value,
            'mean': mean,
            'std': std,
            'z_score': z_score,
            'consecutive_spikes': self._consecutive_spikes,
            'total_spikes': self._total_spikes,
        }

    @property
    def alarm_count(self):
        return self._alarm_count

    @property
    def total_spikes(self):
        return self._total_spikes

    def reset(self):
        self._history.clear()
        self._consecutive_spikes = 0
        self._total_spikes = 0
        self._alarm_count = 0


# ==================== Parameter Freezing Scheduler ====================

class FreezeScheduler:
    """
    Постепенное размораживание слоёв модели.

    Начинает с замороженных нижних слоёв и постепенно
    размораживает по расписанию.

    Стратегии:
    - bottom_up: размораживание снизу вверх
    - top_down: размораживание сверху вниз (типично для fine-tuning)
    - all_at_once: все слои одновременно в заданный шаг

    Args:
        model: nn.Module
        total_steps: общее число шагов
        strategy: стратегия размораживания
        freeze_until: шаг, до которого всё заморожено
    """
    def __init__(self, model, total_steps=1000, strategy='top_down',
                 freeze_until=0):
        self.model = model
        self.total_steps = max(total_steps, 1)
        self.strategy = strategy
        self.freeze_until = freeze_until
        self._layer_names = self._collect_layers()
        self._n_layers = max(len(self._layer_names), 1)
        self._unfrozen = set()

    def _collect_layers(self):
        """Собирает имена слоёв верхнего уровня."""
        names = []
        for name, _ in self.model.named_children():
            names.append(name)
        return names

    def freeze_all(self):
        """Замораживает все параметры."""
        for p in self.model.parameters():
            p.requires_grad = False
        self._unfrozen.clear()

    def unfreeze_all(self):
        """Размораживает все параметры."""
        for p in self.model.parameters():
            p.requires_grad = True
        self._unfrozen = set(self._layer_names)

    def _unfreeze_layer(self, layer_name):
        """Размораживает один слой."""
        module = getattr(self.model, layer_name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
            self._unfrozen.add(layer_name)

    def step(self, current_step):
        """
        Обновляет состояние заморозки.

        Args:
            current_step: текущий шаг

        Returns:
            dict: {n_unfrozen, n_total, unfrozen_layers}
        """
        if current_step < self.freeze_until:
            return {
                'n_unfrozen': len(self._unfrozen),
                'n_total': self._n_layers,
                'unfrozen_layers': list(self._unfrozen),
            }

        effective_step = current_step - self.freeze_until
        effective_total = max(self.total_steps - self.freeze_until, 1)
        progress = min(effective_step / effective_total, 1.0)

        n_to_unfreeze = max(1, int(progress * self._n_layers))

        if self.strategy == 'top_down':
            layers_order = list(reversed(self._layer_names))
        elif self.strategy == 'bottom_up':
            layers_order = list(self._layer_names)
        elif self.strategy == 'all_at_once':
            if progress > 0:
                layers_order = self._layer_names
                n_to_unfreeze = self._n_layers
            else:
                layers_order = []
        else:
            layers_order = self._layer_names

        for name in layers_order[:n_to_unfreeze]:
            if name not in self._unfrozen:
                self._unfreeze_layer(name)

        return {
            'n_unfrozen': len(self._unfrozen),
            'n_total': self._n_layers,
            'unfrozen_layers': list(self._unfrozen),
        }

    def get_trainable_params(self):
        """Число trainable параметров."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_pct': trainable / max(total, 1) * 100,
        }


# ==================== Training State Snapshotter ====================

class StateSnapshotter:
    """
    Лёгкие снимки состояния обучения для отката.

    Хранит N последних снимков модели и оптимизатора.
    Позволяет откатиться при обнаружении проблем.

    Args:
        max_snapshots: максимум снимков в памяти
    """
    def __init__(self, max_snapshots=3):
        self.max_snapshots = max_snapshots
        self._snapshots = deque(maxlen=max_snapshots)

    def save(self, model, optimizer=None, step=None, metadata=None):
        """
        Сохраняет снимок.

        Args:
            model: nn.Module
            optimizer: optional optimizer
            step: номер шага
            metadata: дополнительные данные
        """
        snapshot = {
            'model_state': copy.deepcopy(model.state_dict()),
            'step': step,
            'metadata': metadata or {},
        }
        if optimizer is not None:
            snapshot['optimizer_state'] = copy.deepcopy(optimizer.state_dict())
        self._snapshots.append(snapshot)

    def restore(self, model, optimizer=None, index=-1):
        """
        Восстанавливает снимок.

        Args:
            model: nn.Module
            optimizer: optional optimizer
            index: индекс снимка (-1 = последний)

        Returns:
            dict: metadata восстановленного снимка, или None
        """
        if not self._snapshots:
            return None

        snapshot = self._snapshots[index]
        model.load_state_dict(snapshot['model_state'])

        if optimizer is not None and 'optimizer_state' in snapshot:
            optimizer.load_state_dict(snapshot['optimizer_state'])

        return {
            'step': snapshot['step'],
            'metadata': snapshot['metadata'],
        }

    def restore_best(self, model, optimizer=None, metric_key='loss', mode='min'):
        """
        Восстанавливает лучший снимок по метрике.

        Args:
            model: nn.Module
            optimizer: optional optimizer
            metric_key: ключ метрики в metadata
            mode: 'min' или 'max'

        Returns:
            dict или None
        """
        if not self._snapshots:
            return None

        best_idx = None
        best_val = None

        for i, snap in enumerate(self._snapshots):
            val = snap['metadata'].get(metric_key)
            if val is None:
                continue
            if best_val is None:
                best_val = val
                best_idx = i
            elif mode == 'min' and val < best_val:
                best_val = val
                best_idx = i
            elif mode == 'max' and val > best_val:
                best_val = val
                best_idx = i

        if best_idx is None:
            return None

        return self.restore(model, optimizer, index=best_idx)

    @property
    def n_snapshots(self):
        return len(self._snapshots)

    def get_snapshot_info(self):
        """Информация о всех снимках."""
        return [
            {'step': s['step'], 'metadata': s['metadata']}
            for s in self._snapshots
        ]

    def clear(self):
        self._snapshots.clear()
