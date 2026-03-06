"""
v35 утилиты: Curriculum Learning, Gradient Noise, Dynamic Batch Size,
Parameter Efficiency Analyzer, Training Stability Monitor.

Curriculum Learning: постепенное усложнение данных.
Ref: Bengio et al., "Curriculum Learning" (2009)

Gradient Noise: шум в градиентах для escape из saddle points.
Ref: Neelakantan et al., "Adding Gradient Noise Improves Learning" (2015)

Dynamic Batch Size: масштабирование batch size по ходу обучения.
Ref: Smith et al., "Don't Decay the Learning Rate, Increase the Batch Size" (2018)

Parameter Efficiency: анализ вклада параметров.

Training Stability Monitor: отслеживание стабильности.
"""

import math
import torch
import torch.nn as nn
from collections import deque


# ==================== Curriculum Learning Scheduler ====================

class CurriculumScheduler:
    """
    Планировщик curriculum learning.

    Постепенно увеличивает сложность данных:
    - длину последовательности
    - сложность примеров (по метрике)
    - долю используемого датасета

    Args:
        total_steps: общее число шагов
        strategy: 'linear', 'sqrt', 'step'
        warmup_fraction: доля шагов разогрева (данные простые)
    """
    def __init__(self, total_steps=10000, strategy='linear', warmup_fraction=0.2):
        self.total_steps = total_steps
        self.strategy = strategy
        self.warmup_fraction = warmup_fraction
        self._current_step = 0

    def get_difficulty(self):
        """
        Возвращает текущую сложность [0, 1].

        Returns:
            float: difficulty от 0 (простое) до 1 (всё)
        """
        progress = min(self._current_step / max(self.total_steps, 1), 1.0)

        if progress < self.warmup_fraction:
            # В warmup — линейно от 0 до начального уровня
            warmup_progress = progress / self.warmup_fraction
            base = 0.3  # начальная сложность
            return base * warmup_progress

        # После warmup
        post_warmup = (progress - self.warmup_fraction) / (1 - self.warmup_fraction)

        if self.strategy == 'linear':
            difficulty = 0.3 + 0.7 * post_warmup
        elif self.strategy == 'sqrt':
            difficulty = 0.3 + 0.7 * math.sqrt(post_warmup)
        elif self.strategy == 'step':
            # 3 ступени
            if post_warmup < 0.33:
                difficulty = 0.4
            elif post_warmup < 0.66:
                difficulty = 0.7
            else:
                difficulty = 1.0
        else:
            difficulty = post_warmup

        return min(difficulty, 1.0)

    def get_max_seq_len(self, full_seq_len):
        """
        Максимальная длина последовательности на текущем этапе.

        Args:
            full_seq_len: полная длина

        Returns:
            int: текущая максимальная длина
        """
        difficulty = self.get_difficulty()
        min_len = max(4, full_seq_len // 4)
        return max(min_len, int(full_seq_len * difficulty))

    def get_data_fraction(self):
        """
        Доля датасета для использования.

        Returns:
            float: [0.1, 1.0]
        """
        difficulty = self.get_difficulty()
        return max(0.1, difficulty)

    def filter_by_difficulty(self, losses, threshold_percentile=None):
        """
        Фильтрует примеры по сложности.

        Args:
            losses: list/tensor losses для примеров
            threshold_percentile: если None, используется difficulty

        Returns:
            list[int]: индексы выбранных примеров
        """
        if isinstance(losses, torch.Tensor):
            losses = losses.tolist()

        difficulty = self.get_difficulty()
        if threshold_percentile is None:
            threshold_percentile = difficulty

        sorted_indices = sorted(range(len(losses)), key=lambda i: losses[i])
        n_select = max(1, int(len(sorted_indices) * threshold_percentile))
        return sorted_indices[:n_select]

    def step(self):
        """Увеличить счётчик шагов."""
        self._current_step += 1

    @property
    def current_step(self):
        return self._current_step

    def get_info(self):
        return {
            'step': self._current_step,
            'difficulty': self.get_difficulty(),
            'strategy': self.strategy,
            'progress': min(self._current_step / max(self.total_steps, 1), 1.0),
        }


# ==================== Gradient Noise Injection ====================

class GradientNoiseInjector:
    """
    Добавление шума в градиенты.

    Шум с variance = eta / (1 + t)^gamma помогает
    выбраться из seddle points и улучшает генерализацию.

    Args:
        eta: начальная variance шума
        gamma: скорость затухания (0.55 рекомендовано)
        noise_type: 'gaussian' или 'uniform'
    """
    def __init__(self, eta=0.1, gamma=0.55, noise_type='gaussian'):
        self.eta = eta
        self.gamma = gamma
        self.noise_type = noise_type
        self._step = 0

    def get_noise_variance(self):
        """Текущая variance шума."""
        return self.eta / (1 + self._step) ** self.gamma

    def inject(self, model):
        """
        Добавляет шум к градиентам модели.

        Args:
            model: nn.Module (должен иметь .grad)

        Returns:
            dict: {variance, n_params_noised}
        """
        variance = self.get_noise_variance()
        std = math.sqrt(variance)
        n_noised = 0

        for p in model.parameters():
            if p.grad is not None:
                if self.noise_type == 'gaussian':
                    noise = torch.randn_like(p.grad) * std
                else:  # uniform
                    noise = (torch.rand_like(p.grad) - 0.5) * 2 * std
                p.grad.add_(noise)
                n_noised += 1

        self._step += 1

        return {
            'variance': variance,
            'std': std,
            'n_params_noised': n_noised,
        }

    def get_info(self):
        return {
            'step': self._step,
            'current_variance': self.get_noise_variance(),
            'eta': self.eta,
            'gamma': self.gamma,
        }

    def reset(self):
        self._step = 0


# ==================== Dynamic Batch Size Scaler ====================

class DynamicBatchSizeScaler:
    """
    Автоматическое масштабирование batch size.

    Увеличивает batch size при стабильном loss,
    уменьшает при нестабильности. LR корректируется
    пропорционально (linear scaling rule).

    Args:
        initial_batch_size: начальный batch size
        min_batch_size: минимальный
        max_batch_size: максимальный
        scale_factor: множитель при scale up/down
        stability_window: окно для оценки стабильности
    """
    def __init__(self, initial_batch_size=32, min_batch_size=8,
                 max_batch_size=256, scale_factor=2, stability_window=50):
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.scale_factor = scale_factor
        self.stability_window = stability_window
        self._loss_history = deque(maxlen=stability_window)
        self._scale_history = []

    def update(self, loss_value):
        """
        Обновляет историю и решает о масштабировании.

        Args:
            loss_value: текущий loss

        Returns:
            dict: {batch_size, lr_multiplier, action, stability}
        """
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()

        self._loss_history.append(loss_value)

        action = 'hold'
        if len(self._loss_history) >= self.stability_window:
            stability = self._compute_stability()

            if stability < 0.05 and self.current_batch_size < self.max_batch_size:
                # Stable — scale up
                new_bs = min(self.current_batch_size * self.scale_factor, self.max_batch_size)
                if new_bs != self.current_batch_size:
                    self.current_batch_size = new_bs
                    action = 'scale_up'
                    self._loss_history.clear()
            elif stability > 0.3 and self.current_batch_size > self.min_batch_size:
                # Unstable — scale down
                new_bs = max(self.current_batch_size // self.scale_factor, self.min_batch_size)
                if new_bs != self.current_batch_size:
                    self.current_batch_size = new_bs
                    action = 'scale_down'
                    self._loss_history.clear()
        else:
            stability = None

        lr_multiplier = self.current_batch_size / self.initial_batch_size

        if action != 'hold':
            self._scale_history.append({
                'action': action,
                'batch_size': self.current_batch_size,
                'stability': stability,
            })

        return {
            'batch_size': self.current_batch_size,
            'lr_multiplier': lr_multiplier,
            'action': action,
            'stability': stability,
        }

    def _compute_stability(self):
        """Вычисляет коэф. вариации loss."""
        losses = list(self._loss_history)
        if len(losses) < 2:
            return 0.0
        mean = sum(losses) / len(losses)
        if mean == 0:
            return 0.0
        var = sum((x - mean) ** 2 for x in losses) / len(losses)
        return math.sqrt(var) / abs(mean)

    def get_info(self):
        return {
            'current_batch_size': self.current_batch_size,
            'lr_multiplier': self.current_batch_size / self.initial_batch_size,
            'scale_history': self._scale_history[-5:],
        }


# ==================== Parameter Efficiency Analyzer ====================

class ParameterEfficiencyAnalyzer:
    """
    Анализатор эффективности параметров модели.

    Определяет:
    - мёртвые нейроны (всегда 0)
    - дублирующиеся веса
    - параметры с малым gradient flow
    - общую утилизацию параметров

    Args:
        threshold: порог для "малого" параметра
    """
    def __init__(self, threshold=1e-6):
        self.threshold = threshold

    def analyze(self, model):
        """
        Полный анализ параметров.

        Args:
            model: nn.Module

        Returns:
            dict: {total_params, active_params, dead_params,
                   utilization, layer_stats}
        """
        total_params = 0
        dead_params = 0
        near_zero_params = 0
        layer_stats = []

        for name, p in model.named_parameters():
            n = p.numel()
            total_params += n

            dead = (p.data.abs() < self.threshold).sum().item()
            dead_params += dead
            near_zero = (p.data.abs() < self.threshold * 100).sum().item()
            near_zero_params += near_zero

            layer_stats.append({
                'name': name,
                'shape': list(p.shape),
                'n_params': n,
                'dead_fraction': dead / max(n, 1),
                'near_zero_fraction': near_zero / max(n, 1),
                'mean_abs': p.data.abs().mean().item(),
                'std': p.data.std().item(),
                'max_abs': p.data.abs().max().item(),
            })

        active_params = total_params - dead_params
        utilization = active_params / max(total_params, 1)

        return {
            'total_params': total_params,
            'active_params': active_params,
            'dead_params': dead_params,
            'near_zero_params': near_zero_params,
            'utilization': utilization,
            'layer_stats': layer_stats,
        }

    def find_redundant_layers(self, model, similarity_threshold=0.95):
        """
        Находит слои с похожими весами.

        Args:
            model: nn.Module
            similarity_threshold: порог cosine similarity

        Returns:
            list[tuple]: пары похожих слоёв
        """
        weight_vectors = {}
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                weight_vectors[name] = p.data.flatten()

        redundant_pairs = []
        names = list(weight_vectors.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                v1 = weight_vectors[names[i]]
                v2 = weight_vectors[names[j]]
                if v1.shape != v2.shape:
                    continue
                cos_sim = torch.nn.functional.cosine_similarity(
                    v1.unsqueeze(0), v2.unsqueeze(0)
                ).item()
                if cos_sim > similarity_threshold:
                    redundant_pairs.append((names[i], names[j], cos_sim))

        return redundant_pairs

    def gradient_flow(self, model):
        """
        Анализ gradient flow.

        Args:
            model: nn.Module (после backward)

        Returns:
            list[dict]: gradient stats per layer
        """
        flow = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                flow.append({
                    'name': name,
                    'grad_mean': p.grad.data.abs().mean().item(),
                    'grad_std': p.grad.data.std().item(),
                    'grad_max': p.grad.data.abs().max().item(),
                    'has_grad': True,
                })
            else:
                flow.append({'name': name, 'has_grad': False})
        return flow


# ==================== Training Stability Monitor ====================

class TrainingStabilityMonitor:
    """
    Мониторинг стабильности обучения.

    Отслеживает:
    - loss spikes
    - gradient explosions
    - weight divergence
    - learning rate effectiveness

    Args:
        spike_threshold: множитель для обнаружения spike
        window_size: окно для статистик
    """
    def __init__(self, spike_threshold=3.0, window_size=100):
        self.spike_threshold = spike_threshold
        self.window_size = window_size
        self._loss_history = deque(maxlen=window_size)
        self._grad_norm_history = deque(maxlen=window_size)
        self._weight_norm_history = deque(maxlen=window_size)
        self._events = []

    def update(self, loss, grad_norm=None, weight_norm=None):
        """
        Обновляет мониторинг.

        Args:
            loss: текущий loss
            grad_norm: норма градиента
            weight_norm: норма весов

        Returns:
            dict: {stable, events, metrics}
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        events = []

        # Check loss spike
        if len(self._loss_history) >= 10:
            mean_loss = sum(self._loss_history) / len(self._loss_history)
            if loss > mean_loss * self.spike_threshold:
                events.append({
                    'type': 'loss_spike',
                    'value': loss,
                    'mean': mean_loss,
                    'ratio': loss / max(mean_loss, 1e-8),
                })

        self._loss_history.append(loss)

        # Check gradient explosion
        if grad_norm is not None:
            if len(self._grad_norm_history) >= 10:
                mean_gn = sum(self._grad_norm_history) / len(self._grad_norm_history)
                if grad_norm > mean_gn * self.spike_threshold:
                    events.append({
                        'type': 'gradient_explosion',
                        'value': grad_norm,
                        'mean': mean_gn,
                    })
            self._grad_norm_history.append(grad_norm)

        # Check weight divergence
        if weight_norm is not None:
            if len(self._weight_norm_history) >= 10:
                prev = self._weight_norm_history[-1]
                change = abs(weight_norm - prev) / max(prev, 1e-8)
                if change > 0.5:
                    events.append({
                        'type': 'weight_divergence',
                        'change_ratio': change,
                    })
            self._weight_norm_history.append(weight_norm)

        self._events.extend(events)
        stable = len(events) == 0

        return {
            'stable': stable,
            'events': events,
            'metrics': self._compute_metrics(),
        }

    def _compute_metrics(self):
        """Метрики стабильности."""
        metrics = {}
        if self._loss_history:
            losses = list(self._loss_history)
            metrics['loss_mean'] = sum(losses) / len(losses)
            metrics['loss_std'] = math.sqrt(
                sum((x - metrics['loss_mean'])**2 for x in losses) / len(losses)
            ) if len(losses) > 1 else 0.0
            metrics['loss_cv'] = metrics['loss_std'] / max(metrics['loss_mean'], 1e-8)

        if self._grad_norm_history:
            gns = list(self._grad_norm_history)
            metrics['grad_norm_mean'] = sum(gns) / len(gns)

        return metrics

    def get_stability_score(self):
        """
        Оценка стабильности [0, 1]. 1 = полностью стабильно.

        Returns:
            float
        """
        if len(self._loss_history) < 10:
            return 1.0

        metrics = self._compute_metrics()
        cv = metrics.get('loss_cv', 0)
        # Lower CV = more stable
        score = max(0, 1.0 - cv * 5)
        return min(score, 1.0)

    def get_recent_events(self, n=10):
        return self._events[-n:]

    def get_summary(self):
        return {
            'stability_score': self.get_stability_score(),
            'total_events': len(self._events),
            'recent_events': self.get_recent_events(5),
            'metrics': self._compute_metrics(),
        }

    def reset(self):
        self._loss_history.clear()
        self._grad_norm_history.clear()
        self._weight_norm_history.clear()
        self._events.clear()
