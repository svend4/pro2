"""
v32 утилиты: Gradient Histogram Tracker, Cosine Annealing with Warm Restarts,
Multi-Scale Loss, Parameter Norm Monitor, Data Mixing Scheduler.

Gradient Histogram: статистика распределения градиентов по слоям.

Cosine Annealing with Warm Restarts (SGDR).
Ref: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017)

Multi-Scale Loss: комбинирование loss на разных масштабах.

Parameter Norm Monitor: отслеживание роста норм весов.

Data Mixing Scheduler: динамические пропорции датасетов.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque


# ==================== Gradient Histogram Tracker ====================

class GradientHistogramTracker:
    """
    Отслеживание статистик градиентов по слоям.

    Собирает mean, std, min, max, sparsity, l2_norm
    для градиентов каждого параметра.

    Args:
        window_size: размер окна для скользящих средних
    """
    def __init__(self, window_size=50):
        self.window_size = window_size
        self._stats = defaultdict(lambda: deque(maxlen=window_size))

    def record(self, model):
        """
        Записывает статистики градиентов.

        Args:
            model: nn.Module

        Returns:
            dict: {param_name: {mean, std, min, max, l2_norm, sparsity}}
        """
        snapshot = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.data
                stats = {
                    'mean': g.mean().item(),
                    'std': g.std().item() if g.numel() > 1 else 0.0,
                    'min': g.min().item(),
                    'max': g.max().item(),
                    'l2_norm': g.norm(2).item(),
                    'sparsity': (g == 0).float().mean().item(),
                    'abs_mean': g.abs().mean().item(),
                }
                self._stats[name].append(stats)
                snapshot[name] = stats
        return snapshot

    def get_summary(self, name):
        """
        Сводка по параметру за окно.

        Args:
            name: имя параметра

        Returns:
            dict или None
        """
        history = self._stats.get(name)
        if not history:
            return None

        keys = ['mean', 'std', 'l2_norm', 'abs_mean', 'sparsity']
        summary = {}
        for key in keys:
            values = [s[key] for s in history]
            summary[f'{key}_avg'] = sum(values) / len(values)
            summary[f'{key}_latest'] = values[-1]
        return summary

    def get_layer_ranking(self, metric='l2_norm'):
        """
        Ранжирование слоёв по метрике.

        Args:
            metric: какую метрику использовать

        Returns:
            list[tuple]: [(name, value), ...] отсортировано по убыванию
        """
        ranking = []
        for name, history in self._stats.items():
            if history:
                val = history[-1].get(metric, 0)
                ranking.append((name, val))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def detect_vanishing(self, threshold=1e-7):
        """Обнаруживает слои с затухающими градиентами."""
        vanishing = []
        for name, history in self._stats.items():
            if history and history[-1]['abs_mean'] < threshold:
                vanishing.append(name)
        return vanishing

    def detect_exploding(self, threshold=100.0):
        """Обнаруживает слои с взрывающимися градиентами."""
        exploding = []
        for name, history in self._stats.items():
            if history and history[-1]['l2_norm'] > threshold:
                exploding.append(name)
        return exploding

    def reset(self):
        self._stats.clear()


# ==================== Cosine Annealing with Warm Restarts ====================

class CosineAnnealingWarmRestarts:
    """
    SGDR: Cosine Annealing с тёплыми перезапусками.

    LR следует косинусному расписанию, периодически перезапускаясь.
    Период может увеличиваться с каждым перезапуском.

    Args:
        optimizer: torch optimizer
        T_0: начальный период (шагов)
        T_mult: множитель периода после каждого перезапуска
        eta_min: минимальный LR
        eta_max: максимальный LR (None = из optimizer)
    """
    def __init__(self, optimizer, T_0=10, T_mult=2, eta_min=1e-6, eta_max=None):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max or optimizer.param_groups[0]['lr']
        self._step = 0
        self._cycle = 0
        self._cycle_step = 0
        self._current_T = T_0

    def step(self):
        """Один шаг расписания."""
        self._step += 1
        self._cycle_step += 1

        # Check for restart
        if self._cycle_step >= self._current_T:
            self._cycle += 1
            self._cycle_step = 0
            self._current_T = int(self.T_0 * (self.T_mult ** self._cycle))

        # Cosine annealing within current cycle
        progress = self._cycle_step / max(self._current_T, 1)
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
             (1 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        return lr

    def get_lr(self):
        """Текущий LR."""
        progress = self._cycle_step / max(self._current_T, 1)
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
               (1 + math.cos(math.pi * progress))

    @property
    def cycle(self):
        return self._cycle

    @property
    def current_step(self):
        return self._step

    def get_info(self):
        """Информация о текущем состоянии."""
        return {
            'step': self._step,
            'cycle': self._cycle,
            'cycle_step': self._cycle_step,
            'current_T': self._current_T,
            'lr': self.get_lr(),
        }


# ==================== Multi-Scale Loss ====================

class MultiScaleLoss(nn.Module):
    """
    Loss на нескольких масштабах последовательности.

    Комбинирует token-level, chunk-level и sequence-level loss.

    Args:
        vocab_size: размер словаря
        chunk_sizes: список размеров чанков
        weights: веса для каждого масштаба (token + chunks)
    """
    def __init__(self, vocab_size, chunk_sizes=(4, 8), weights=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.chunk_sizes = chunk_sizes
        n_scales = 1 + len(chunk_sizes)  # token + chunks
        if weights is None:
            weights = [1.0 / n_scales] * n_scales
        self.weights = weights

    def forward(self, logits, targets):
        """
        Вычисляет multi-scale loss.

        Args:
            logits: (B, T, V)
            targets: (B, T)

        Returns:
            dict: {total_loss, token_loss, chunk_losses}
        """
        B, T, V = logits.shape

        # Token-level loss
        token_loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction='mean'
        )

        losses = [token_loss]
        chunk_losses = {}

        # Chunk-level losses
        for idx, chunk_size in enumerate(self.chunk_sizes):
            if T < chunk_size:
                chunk_losses[chunk_size] = token_loss
                losses.append(token_loss)
                continue

            n_chunks = T // chunk_size
            truncated_T = n_chunks * chunk_size

            # Average logits within chunks
            chunk_logits = logits[:, :truncated_T].reshape(B, n_chunks, chunk_size, V)
            chunk_logits_mean = chunk_logits.mean(dim=2)  # (B, n_chunks, V)

            # Mode target within chunks (most frequent)
            chunk_targets = targets[:, :truncated_T].reshape(B, n_chunks, chunk_size)
            # Use last token as chunk target
            chunk_target = chunk_targets[:, :, -1]  # (B, n_chunks)

            cl = F.cross_entropy(
                chunk_logits_mean.reshape(-1, V),
                chunk_target.reshape(-1),
                reduction='mean'
            )
            chunk_losses[chunk_size] = cl
            losses.append(cl)

        # Weighted sum
        total = sum(w * l for w, l in zip(self.weights, losses))

        return {
            'total_loss': total,
            'token_loss': token_loss,
            'chunk_losses': chunk_losses,
        }


# ==================== Parameter Norm Monitor ====================

class ParamNormMonitor:
    """
    Мониторинг норм параметров.

    Отслеживает рост/уменьшение норм весов для раннего
    обнаружения нестабильности.

    Args:
        window_size: окно для трендов
    """
    def __init__(self, window_size=50):
        self.window_size = window_size
        self._history = defaultdict(lambda: deque(maxlen=window_size))
        self._step = 0

    def record(self, model):
        """
        Записывает нормы параметров.

        Args:
            model: nn.Module

        Returns:
            dict: {name: norm}
        """
        self._step += 1
        norms = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                norm = p.data.norm(2).item()
                self._history[name].append(norm)
                norms[name] = norm
        return norms

    def get_total_norm(self, model):
        """Общая норма всех параметров."""
        total = 0.0
        with torch.no_grad():
            for p in model.parameters():
                total += p.data.norm(2).item() ** 2
        return math.sqrt(total)

    def get_trends(self):
        """
        Тренды норм для каждого параметра.

        Returns:
            dict: {name: 'growing'|'shrinking'|'stable'}
        """
        trends = {}
        for name, history in self._history.items():
            values = list(history)
            if len(values) < 4:
                trends[name] = 'insufficient_data'
                continue
            mid = len(values) // 2
            first = sum(values[:mid]) / mid
            second = sum(values[mid:]) / (len(values) - mid)
            if second > first * 1.1:
                trends[name] = 'growing'
            elif second < first * 0.9:
                trends[name] = 'shrinking'
            else:
                trends[name] = 'stable'
        return trends

    def detect_anomalies(self, threshold=10.0):
        """
        Обнаруживает параметры с аномальным ростом нормы.

        Args:
            threshold: множитель для считания аномалией

        Returns:
            list[dict]: аномалии
        """
        anomalies = []
        for name, history in self._history.items():
            values = list(history)
            if len(values) < 2:
                continue
            if values[-1] > values[0] * threshold:
                anomalies.append({
                    'name': name,
                    'initial_norm': values[0],
                    'current_norm': values[-1],
                    'growth_factor': values[-1] / max(values[0], 1e-8),
                })
        return anomalies

    def get_summary(self):
        """Сводка норм."""
        summary = {}
        for name, history in self._history.items():
            values = list(history)
            if values:
                summary[name] = {
                    'current': values[-1],
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                }
        return summary

    def reset(self):
        self._history.clear()
        self._step = 0


# ==================== Data Mixing Scheduler ====================

class DataMixingScheduler:
    """
    Динамические пропорции смешивания датасетов.

    Управляет пропорциями нескольких источников данных
    на протяжении обучения.

    Стратегии:
    - constant: фиксированные пропорции
    - linear_shift: линейный сдвиг от initial к final
    - loss_based: адаптация на основе loss каждого источника
    - temperature: softmax по loss с температурой

    Args:
        n_sources: число источников данных
        strategy: стратегия смешивания
        initial_weights: начальные пропорции
        final_weights: конечные пропорции (для linear_shift)
        total_steps: общее число шагов
        temperature: температура для temperature стратегии
    """
    def __init__(self, n_sources, strategy='constant',
                 initial_weights=None, final_weights=None,
                 total_steps=1000, temperature=1.0):
        self.n_sources = n_sources
        self.strategy = strategy
        self.total_steps = max(total_steps, 1)
        self.temperature = temperature

        if initial_weights is None:
            initial_weights = [1.0 / n_sources] * n_sources
        self.initial_weights = initial_weights

        if final_weights is None:
            final_weights = initial_weights
        self.final_weights = final_weights

        self._current_weights = list(initial_weights)
        self._source_losses = defaultdict(lambda: deque(maxlen=50))
        self._step = 0

    def step(self):
        """Продвинуть шаг."""
        self._step += 1
        self._update_weights()

    def report_loss(self, source_idx, loss_value):
        """
        Сообщить loss для источника.

        Args:
            source_idx: индекс источника
            loss_value: значение loss
        """
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        self._source_losses[source_idx].append(loss_value)

    def _update_weights(self):
        """Пересчитать пропорции."""
        if self.strategy == 'constant':
            return

        elif self.strategy == 'linear_shift':
            progress = min(self._step / self.total_steps, 1.0)
            for i in range(self.n_sources):
                self._current_weights[i] = (
                    self.initial_weights[i] +
                    (self.final_weights[i] - self.initial_weights[i]) * progress
                )

        elif self.strategy == 'loss_based':
            # Higher loss → more weight (focus on harder sources)
            avg_losses = []
            for i in range(self.n_sources):
                losses = list(self._source_losses.get(i, []))
                if losses:
                    avg_losses.append(sum(losses) / len(losses))
                else:
                    avg_losses.append(1.0)
            total = sum(avg_losses)
            if total > 0:
                self._current_weights = [l / total for l in avg_losses]

        elif self.strategy == 'temperature':
            avg_losses = []
            for i in range(self.n_sources):
                losses = list(self._source_losses.get(i, []))
                if losses:
                    avg_losses.append(sum(losses) / len(losses))
                else:
                    avg_losses.append(1.0)
            # Softmax with temperature
            scaled = [l / self.temperature for l in avg_losses]
            max_s = max(scaled)
            exps = [math.exp(s - max_s) for s in scaled]
            total = sum(exps)
            self._current_weights = [e / total for e in exps]

        # Normalize
        total = sum(self._current_weights)
        if total > 0:
            self._current_weights = [w / total for w in self._current_weights]

    def get_weights(self):
        """Текущие пропорции."""
        return list(self._current_weights)

    def sample_source(self):
        """
        Сэмплирует источник по текущим пропорциям.

        Returns:
            int: индекс источника
        """
        r = torch.rand(1).item()
        cumulative = 0.0
        for i, w in enumerate(self._current_weights):
            cumulative += w
            if r < cumulative:
                return i
        return self.n_sources - 1

    def get_info(self):
        """Информация о текущем состоянии."""
        return {
            'step': self._step,
            'strategy': self.strategy,
            'weights': list(self._current_weights),
            'source_losses': {
                i: list(losses)[-5:] if losses else []
                for i, losses in self._source_losses.items()
            },
        }

    def reset(self):
        self._current_weights = list(self.initial_weights)
        self._source_losses.clear()
        self._step = 0
