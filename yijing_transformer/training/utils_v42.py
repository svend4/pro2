"""
v42 утилиты: Cosine Annealing Warm Restarts, Gradient Accumulation,
Model EMA, Curriculum Learning, Training Stability Monitor.

Cosine Annealing Warm Restarts (SGDR):
Ref: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent
with Warm Restarts" (2017)

Model EMA: сглаживание весов для лучшей генерализации.
Ref: Polyak & Juditsky, "Acceleration of Stochastic Approximation" (1992)

Curriculum Learning: постепенное усложнение данных.
Ref: Bengio et al., "Curriculum Learning" (2009)
"""

import math
import copy
import torch
import torch.nn as nn
from collections import deque


# ==================== Cosine Annealing with Warm Restarts ====================

class CosineAnnealingWarmRestarts:
    """
    SGDR: Cosine annealing с тёплыми перезапусками.

    LR следует косинусной кривой с периодическими перезапусками,
    каждый цикл может увеличиваться (T_mult).

    Args:
        base_lr: начальный LR
        T_0: длина первого цикла (шаги)
        T_mult: множитель периода при каждом рестарте
        eta_min: минимальный LR
    """
    def __init__(self, base_lr=0.001, T_0=100, T_mult=2, eta_min=1e-6):
        self.base_lr = base_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._step = 0
        self._cycle = 0
        self._cycle_step = 0
        self._current_T = T_0
        self._lr_history = []

    def step(self):
        """
        Один шаг scheduler.

        Returns:
            float: текущий LR
        """
        self._step += 1
        self._cycle_step += 1

        # Check restart
        if self._cycle_step >= self._current_T:
            self._cycle += 1
            self._cycle_step = 0
            self._current_T = int(self._current_T * self.T_mult)

        # Cosine annealing within cycle
        progress = self._cycle_step / max(self._current_T, 1)
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * progress)) / 2

        self._lr_history.append(lr)
        return lr

    def apply(self, optimizer):
        """
        Применяет LR к оптимизатору.

        Args:
            optimizer: torch optimizer

        Returns:
            float: новый LR
        """
        lr = self.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_info(self):
        return {
            'current_lr': self._lr_history[-1] if self._lr_history else self.base_lr,
            'cycle': self._cycle,
            'cycle_step': self._cycle_step,
            'current_T': self._current_T,
            'total_steps': self._step,
        }


# ==================== Gradient Accumulation Manager ====================

class GradientAccumulationManager:
    """
    Управление gradient accumulation.

    Эмулирует большие batch size через накопление
    градиентов за несколько micro-batch шагов.

    Args:
        accumulation_steps: число шагов накопления
        normalize: делить градиенты на accumulation_steps
    """
    def __init__(self, accumulation_steps=4, normalize=True):
        self.accumulation_steps = accumulation_steps
        self.normalize = normalize
        self._current_step = 0
        self._accumulated_loss = 0.0

    def should_step(self):
        """
        Нужно ли делать optimizer.step().

        Returns:
            bool
        """
        return self._current_step % self.accumulation_steps == 0 and self._current_step > 0

    def accumulate(self, loss):
        """
        Накапливает loss и делает backward.

        Args:
            loss: tensor loss

        Returns:
            dict: {should_step, accumulated_loss, micro_step}
        """
        self._current_step += 1

        if self.normalize:
            scaled_loss = loss / self.accumulation_steps
        else:
            scaled_loss = loss

        scaled_loss.backward()
        self._accumulated_loss += loss.item()

        should = self.should_step()
        result = {
            'should_step': should,
            'accumulated_loss': self._accumulated_loss,
            'micro_step': (self._current_step - 1) % self.accumulation_steps + 1,
        }

        if should:
            self._accumulated_loss = 0.0

        return result

    def get_effective_batch_size(self, micro_batch_size):
        """Эффективный batch size."""
        return micro_batch_size * self.accumulation_steps

    @property
    def current_micro_step(self):
        return (self._current_step - 1) % self.accumulation_steps + 1

    def get_info(self):
        return {
            'accumulation_steps': self.accumulation_steps,
            'current_micro_step': self.current_micro_step,
            'total_steps': self._current_step,
            'optimizer_steps': self._current_step // self.accumulation_steps,
        }


# ==================== Model EMA ====================

class ModelEMA:
    """
    Exponential Moving Average весов модели.

    Поддерживает shadow copy: θ_ema = decay * θ_ema + (1-decay) * θ.
    EMA модель обычно генерализует лучше.

    Args:
        model: nn.Module
        decay: коэффициент сглаживания (0.999 типично)
        warmup_steps: шагов до включения EMA
    """
    def __init__(self, model, decay=0.999, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._shadow = {}
        self._backup = {}

        # Initialize shadow
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone()

    def update(self, model):
        """
        Обновляет EMA weights.

        Args:
            model: nn.Module

        Returns:
            dict: {decay_used, step}
        """
        self._step += 1

        # Warmup: use lower decay initially
        if self._step <= self.warmup_steps:
            decay = min(self.decay, (1 + self._step) / (10 + self._step))
        else:
            decay = self.decay

        for name, param in model.named_parameters():
            if param.requires_grad and name in self._shadow:
                self._shadow[name].mul_(decay).add_(
                    param.data, alpha=1.0 - decay
                )

        return {'decay_used': decay, 'step': self._step}

    def apply_shadow(self, model):
        """
        Заменяет веса модели на EMA (для eval).

        Args:
            model: nn.Module
        """
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self._shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self._shadow[name])

    def restore(self, model):
        """
        Восстанавливает оригинальные веса.

        Args:
            model: nn.Module
        """
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def get_shadow_state(self):
        """Возвращает EMA state dict."""
        return {k: v.clone() for k, v in self._shadow.items()}

    def divergence(self, model):
        """
        Расхождение между текущими весами и EMA.

        Args:
            model: nn.Module

        Returns:
            dict: {mean_divergence, max_divergence}
        """
        divergences = []
        for name, param in model.named_parameters():
            if name in self._shadow:
                diff = (param.data - self._shadow[name]).norm().item()
                divergences.append(diff)

        if not divergences:
            return {'mean_divergence': 0.0, 'max_divergence': 0.0}

        return {
            'mean_divergence': sum(divergences) / len(divergences),
            'max_divergence': max(divergences),
        }


# ==================== Curriculum Learning Scheduler ====================

class CurriculumScheduler:
    """
    Планирование сложности данных при обучении.

    Постепенно увеличивает сложность (длину, шум и т.д.)
    от простого к сложному.

    Args:
        total_steps: общее число шагов
        strategy: 'linear', 'sqrt', 'step'
        warmup_fraction: доля шагов на прогрев
    """
    def __init__(self, total_steps=10000, strategy='linear',
                 warmup_fraction=0.1):
        self.total_steps = total_steps
        self.strategy = strategy
        self.warmup_fraction = warmup_fraction
        self._step = 0
        self._difficulty_history = []

    def step(self):
        """
        Возвращает текущий уровень сложности [0, 1].

        Returns:
            float: difficulty level
        """
        self._step += 1
        progress = min(self._step / max(self.total_steps, 1), 1.0)

        if self.strategy == 'linear':
            difficulty = progress
        elif self.strategy == 'sqrt':
            difficulty = math.sqrt(progress)
        elif self.strategy == 'step':
            # 3 ступени: easy, medium, hard
            if progress < 0.33:
                difficulty = 0.33
            elif progress < 0.66:
                difficulty = 0.66
            else:
                difficulty = 1.0
        else:
            difficulty = progress

        # Warmup: clamp to low difficulty
        warmup_end = self.warmup_fraction
        if progress < warmup_end:
            difficulty = min(difficulty, progress / warmup_end * 0.3)

        difficulty = max(0.0, min(1.0, difficulty))
        self._difficulty_history.append(difficulty)
        return difficulty

    def get_sequence_length(self, min_len=8, max_len=128):
        """
        Длина последовательности по текущей сложности.

        Args:
            min_len: минимальная длина
            max_len: максимальная длина

        Returns:
            int
        """
        d = self._difficulty_history[-1] if self._difficulty_history else 0.0
        return int(min_len + (max_len - min_len) * d)

    def get_noise_level(self, max_noise=0.1):
        """
        Уровень шума по текущей сложности.

        Returns:
            float
        """
        d = self._difficulty_history[-1] if self._difficulty_history else 0.0
        return max_noise * d

    def get_info(self):
        return {
            'step': self._step,
            'difficulty': self._difficulty_history[-1] if self._difficulty_history else 0.0,
            'strategy': self.strategy,
        }


# ==================== Training Stability Monitor ====================

class TrainingStabilityMonitor:
    """
    Мониторинг стабильности обучения.

    Отслеживает loss spikes, gradient explosions,
    NaN/Inf, и другие нестабильности.

    Args:
        window_size: размер скользящего окна
        spike_threshold: порог для обнаружения спайков (множитель std)
    """
    def __init__(self, window_size=100, spike_threshold=3.0):
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self._loss_window = deque(maxlen=window_size)
        self._grad_norm_window = deque(maxlen=window_size)
        self._events = []
        self._step = 0

    def update(self, loss, grad_norm=None, model=None):
        """
        Обновляет монитор.

        Args:
            loss: текущий loss (float или tensor)
            grad_norm: норма градиентов (опционально)
            model: nn.Module для автоматического вычисления grad_norm

        Returns:
            dict: {stable, events, stats}
        """
        self._step += 1

        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        events = []

        # Check NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            events.append({
                'type': 'nan_inf_loss',
                'step': self._step,
                'value': loss,
            })
        else:
            self._loss_window.append(loss)

        # Compute grad norm if model provided
        if grad_norm is None and model is not None:
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
            grad_norm = math.sqrt(total)

        if grad_norm is not None:
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                events.append({
                    'type': 'nan_inf_grad',
                    'step': self._step,
                    'value': grad_norm,
                })
            else:
                self._grad_norm_window.append(grad_norm)

        # Detect loss spike
        if len(self._loss_window) >= 10 and not math.isnan(loss) and not math.isinf(loss):
            window_list = list(self._loss_window)
            mean = sum(window_list[:-1]) / len(window_list[:-1])
            std = (sum((x - mean) ** 2 for x in window_list[:-1]) / len(window_list[:-1])) ** 0.5
            if std > 0 and (loss - mean) > self.spike_threshold * std:
                events.append({
                    'type': 'loss_spike',
                    'step': self._step,
                    'value': loss,
                    'mean': mean,
                    'std': std,
                })

        # Detect gradient explosion
        if len(self._grad_norm_window) >= 10 and grad_norm is not None:
            gn_list = list(self._grad_norm_window)
            gn_mean = sum(gn_list[:-1]) / len(gn_list[:-1])
            gn_std = (sum((x - gn_mean) ** 2 for x in gn_list[:-1]) / len(gn_list[:-1])) ** 0.5
            if gn_std > 0 and (grad_norm - gn_mean) > self.spike_threshold * gn_std:
                events.append({
                    'type': 'grad_spike',
                    'step': self._step,
                    'value': grad_norm,
                })

        self._events.extend(events)

        # Stats
        stats = {}
        if self._loss_window:
            lw = list(self._loss_window)
            stats['loss_mean'] = sum(lw) / len(lw)
            stats['loss_std'] = (sum((x - stats['loss_mean']) ** 2 for x in lw) / len(lw)) ** 0.5
        if self._grad_norm_window:
            gw = list(self._grad_norm_window)
            stats['grad_norm_mean'] = sum(gw) / len(gw)

        return {
            'stable': len(events) == 0,
            'events': events,
            'stats': stats,
        }

    def get_all_events(self):
        """Все зафиксированные события."""
        return self._events

    def is_training_healthy(self):
        """
        Общая оценка здоровья обучения.

        Returns:
            dict: {healthy, issues, score}
        """
        issues = []
        recent = self._events[-20:] if self._events else []

        n_spikes = sum(1 for e in recent if e['type'] == 'loss_spike')
        n_nan = sum(1 for e in recent if 'nan' in e['type'])
        n_grad = sum(1 for e in recent if e['type'] == 'grad_spike')

        if n_nan > 0:
            issues.append('nan_detected')
        if n_spikes > 3:
            issues.append('frequent_loss_spikes')
        if n_grad > 3:
            issues.append('frequent_grad_spikes')

        # Check loss trend
        if len(self._loss_window) >= 50:
            lw = list(self._loss_window)
            first = sum(lw[:25]) / 25
            second = sum(lw[25:50]) / 25
            if second > first * 1.1:
                issues.append('loss_increasing')

        score = max(0.0, 1.0 - len(issues) * 0.25)

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'score': score,
        }
