"""
v50 утилиты: Entropy Regularization, Confidence Penalty,
Distillation Temperature Annealing, Progressive Layer Freezing,
Training Metrics Aggregator.

Entropy Regularization: максимизация/минимизация энтропии.
Ref: Pereyra et al., "Regularizing Neural Networks by Penalizing
     Confident Output Distributions" (2017)

Confidence Penalty: штраф за overconfident predictions.
Ref: Pereyra et al. (2017)

Distillation Temperature Annealing: schedule для T в distillation.
Ref: Common practice in knowledge distillation.

Progressive Layer Freezing: постепенная заморозка слоёв.
Ref: Howard & Ruder, "Universal Language Model Fine-tuning" (2018)

Training Metrics Aggregator: сбор и агрегация метрик.
Ref: Standard training infrastructure.
"""

import math
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Entropy Regularization ====================

class EntropyRegularization:
    """
    Регуляризация через энтропию выходного распределения.

    H(p) = -Σ p(x) log p(x)

    Максимизация энтропии → менее уверенные предсказания.
    Минимизация энтропии → более уверенные.

    Args:
        weight: вес регуляризации
        mode: 'maximize' (больше энтропия) или 'minimize'
        target_entropy: целевая энтропия (None = max/min)
    """
    def __init__(self, weight=0.1, mode='maximize', target_entropy=None):
        self.weight = weight
        self.mode = mode
        self.target_entropy = target_entropy
        self._history = []

    def compute(self, logits):
        """
        Вычисляет entropy regularization loss.

        Args:
            logits: (B, C) или (B, T, C)

        Returns:
            dict: {loss, entropy, max_entropy}
        """
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        max_entropy = math.log(logits.size(-1))

        if self.target_entropy is not None:
            # Penalize deviation from target
            loss = self.weight * (entropy - self.target_entropy) ** 2
        elif self.mode == 'maximize':
            loss = -self.weight * entropy  # Negative = maximize
        else:
            loss = self.weight * entropy  # Positive = minimize

        self._history.append(entropy.item())

        return {
            'loss': loss,
            'entropy': entropy.item(),
            'max_entropy': max_entropy,
            'normalized_entropy': entropy.item() / max_entropy,
        }

    def get_stats(self):
        if not self._history:
            return {'avg_entropy': 0}
        return {
            'avg_entropy': sum(self._history[-100:]) / len(self._history[-100:]),
            'min_entropy': min(self._history[-100:]),
            'max_entropy': max(self._history[-100:]),
        }


# ==================== Confidence Penalty ====================

class ConfidencePenalty:
    """
    Штраф за overconfident predictions.

    Penalizes когда max(p) слишком высок.
    Альтернатива label smoothing.

    Args:
        weight: вес штрафа
        threshold: порог confidence (штраф только выше)
        method: 'entropy' или 'max_prob'
    """
    def __init__(self, weight=0.1, threshold=0.9, method='entropy'):
        self.weight = weight
        self.threshold = threshold
        self.method = method
        self._stats = {'n_penalized': 0, 'n_total': 0}

    def compute(self, logits):
        """
        Вычисляет confidence penalty.

        Args:
            logits: (B, C) или (B, T, C)

        Returns:
            dict: {loss, avg_confidence, pct_overconfident}
        """
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))

        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values

        self._stats['n_total'] += max_probs.numel()

        if self.method == 'entropy':
            log_probs = F.log_softmax(logits, dim=-1)
            neg_entropy = (probs * log_probs).sum(dim=-1)  # Negative entropy
            # Penalty on confident samples
            mask = max_probs > self.threshold
            if mask.any():
                loss = self.weight * neg_entropy[mask].mean()
                self._stats['n_penalized'] += mask.sum().item()
            else:
                loss = torch.tensor(0.0, device=logits.device)
        else:  # max_prob
            over = torch.clamp(max_probs - self.threshold, min=0)
            loss = self.weight * over.mean()
            self._stats['n_penalized'] += (max_probs > self.threshold).sum().item()

        return {
            'loss': loss,
            'avg_confidence': max_probs.mean().item(),
            'pct_overconfident': (max_probs > self.threshold).float().mean().item(),
        }

    def get_stats(self):
        total = max(self._stats['n_total'], 1)
        return {
            'penalized_rate': self._stats['n_penalized'] / total,
        }


# ==================== Distillation Temperature Annealing ====================

class DistillationTemperatureAnnealing:
    """
    Постепенное изменение температуры для Knowledge Distillation.

    Начинает с высокой T (мягкие распределения) и
    снижает к 1.0 (жёсткие).

    Args:
        initial_temp: начальная температура
        final_temp: конечная температура
        total_steps: общее число шагов
        schedule: 'linear', 'cosine', 'exponential'
    """
    def __init__(self, initial_temp=10.0, final_temp=1.0,
                 total_steps=10000, schedule='cosine'):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_steps = total_steps
        self.schedule = schedule
        self._step = 0

    def step(self):
        """
        Обновляет шаг и возвращает текущую температуру.

        Returns:
            float: текущая температура
        """
        self._step += 1
        return self.get_temperature()

    def get_temperature(self):
        """Текущая температура."""
        progress = min(self._step / max(self.total_steps, 1), 1.0)

        if self.schedule == 'linear':
            t = self.initial_temp + progress * (self.final_temp - self.initial_temp)
        elif self.schedule == 'cosine':
            t = self.final_temp + (self.initial_temp - self.final_temp) * \
                (1 + math.cos(math.pi * progress)) / 2
        elif self.schedule == 'exponential':
            ratio = self.final_temp / max(self.initial_temp, 1e-10)
            t = self.initial_temp * (ratio ** progress)
        else:
            t = self.initial_temp

        return t

    def get_info(self):
        return {
            'step': self._step,
            'current_temp': self.get_temperature(),
            'schedule': self.schedule,
        }


# ==================== Progressive Layer Freezing ====================

class ProgressiveLayerFreezing:
    """
    Постепенная заморозка слоёв при fine-tuning.

    Начинает с заморозки глубоких (ранних) слоёв,
    постепенно размораживает.
    Или наоборот: замораживает уже обученные слои.

    Args:
        n_layers: число слоёв
        mode: 'unfreeze' (постепенная разморозка) или 'freeze' (заморозка)
        steps_per_layer: шагов на каждый слой
    """
    def __init__(self, n_layers=6, mode='unfreeze', steps_per_layer=1000):
        self.n_layers = n_layers
        self.mode = mode
        self.steps_per_layer = steps_per_layer
        self._step = 0
        self._frozen_layers = set()

    def step(self, model):
        """
        Обновляет заморозку.

        Args:
            model: nn.Module

        Returns:
            dict: {frozen_layers, unfrozen_layers, step}
        """
        self._step += 1
        layers_to_process = self._step // max(self.steps_per_layer, 1)
        layers_to_process = min(layers_to_process, self.n_layers)

        if self.mode == 'unfreeze':
            # Start all frozen, unfreeze from top (last layer first)
            frozen = set(range(self.n_layers - layers_to_process))
        else:
            # Start all unfrozen, freeze from bottom (first layer first)
            frozen = set(range(layers_to_process))

        # Apply freezing
        import re
        for name, p in model.named_parameters():
            layer_idx = self._get_layer_index(name)
            if layer_idx is not None:
                p.requires_grad = layer_idx not in frozen

        self._frozen_layers = frozen

        return {
            'frozen_layers': sorted(frozen),
            'unfrozen_layers': sorted(set(range(self.n_layers)) - frozen),
            'step': self._step,
        }

    def _get_layer_index(self, name):
        import re
        patterns = [r'layers?\.(\d+)', r'blocks?\.(\d+)',
                    r'h\.(\d+)', r'transformer\.(\d+)']
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None

    def get_info(self):
        return {
            'step': self._step,
            'mode': self.mode,
            'frozen_layers': sorted(self._frozen_layers),
            'n_frozen': len(self._frozen_layers),
        }


# ==================== Training Metrics Aggregator ====================

class TrainingMetricsAggregator:
    """
    Агрегация метрик обучения.

    Собирает метрики по окнам, вычисляет
    moving averages, отслеживает тренды.

    Args:
        window_size: размер скользящего окна
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        self._metrics = defaultdict(list)
        self._step = 0
        self._start_time = time.time()

    def update(self, **metrics):
        """
        Добавляет метрики.

        Args:
            **metrics: name=value пары

        Returns:
            dict: текущие средние
        """
        self._step += 1
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._metrics[name].append(value)
            # Keep only last N*2 for trend computation
            if len(self._metrics[name]) > self.window_size * 2:
                self._metrics[name] = self._metrics[name][-self.window_size * 2:]

        return self.get_averages()

    def get_averages(self):
        """Moving averages для всех метрик."""
        result = {}
        for name, values in self._metrics.items():
            window = values[-self.window_size:]
            result[name] = sum(window) / len(window)
        return result

    def get_trends(self):
        """
        Тренды: сравнение текущего и предыдущего окна.

        Returns:
            dict: {metric: {'current', 'previous', 'change', 'improving'}}
        """
        trends = {}
        for name, values in self._metrics.items():
            if len(values) < self.window_size * 2:
                continue
            current = values[-self.window_size:]
            previous = values[-self.window_size * 2:-self.window_size]
            curr_avg = sum(current) / len(current)
            prev_avg = sum(previous) / len(previous)
            change = curr_avg - prev_avg

            # For 'loss'-like metrics, decrease is improving
            improving = change < 0 if 'loss' in name.lower() else change > 0

            trends[name] = {
                'current': curr_avg,
                'previous': prev_avg,
                'change': change,
                'improving': improving,
            }
        return trends

    def get_summary(self):
        """Полная сводка."""
        elapsed = time.time() - self._start_time
        return {
            'step': self._step,
            'elapsed_seconds': elapsed,
            'steps_per_second': self._step / max(elapsed, 1e-6),
            'averages': self.get_averages(),
            'trends': self.get_trends(),
        }

    def reset(self):
        """Сброс всех метрик."""
        self._metrics.clear()
        self._step = 0
        self._start_time = time.time()
