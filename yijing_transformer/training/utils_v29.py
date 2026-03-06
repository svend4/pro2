"""
v29 утилиты: EMA, Curriculum Learning, Gradient Noise,
LR Probing, Weight Decay Scheduler.

EMA: экспоненциальное скользящее среднее весов для инференса.
Ref: Polyak & Juditsky, "Acceleration of Stochastic Approximation" (1992)

Curriculum Learning: расписание сложности данных.
Ref: Bengio et al., "Curriculum Learning" (2009)

Gradient Noise: добавление шума для exploration.
Ref: Neelakantan et al., "Adding Gradient Noise Improves Learning" (2015)

LR Probing: автоматический поиск диапазона LR.
Ref: Smith, "Cyclical Learning Rates" (2017)

Weight Decay Scheduler: адаптивный weight decay.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ==================== Exponential Moving Average ====================

class EMA:
    """
    Exponential Moving Average весов модели.

    Поддерживает shadow-копию весов, которая обновляется
    через экспоненциальное сглаживание:
    shadow = decay * shadow + (1 - decay) * param

    Args:
        model: nn.Module
        decay: EMA decay (0.999 типично)
        warmup_steps: шагов до полного decay (linear warmup)
    """
    def __init__(self, model, decay=0.999, warmup_steps=0):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._step = 0
        self._shadow = {}
        self._backup = {}

        # Initialize shadow weights
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._shadow[name] = p.data.clone()

    def _get_decay(self):
        """Получить текущий decay с учётом warmup."""
        if self.warmup_steps <= 0:
            return self.decay
        ratio = min(self._step / self.warmup_steps, 1.0)
        return self.decay * ratio

    def update(self):
        """Обновить shadow weights."""
        self._step += 1
        decay = self._get_decay()

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self._shadow:
                    self._shadow[name].mul_(decay).add_(p.data, alpha=1 - decay)

    def apply_shadow(self):
        """Применить shadow weights к модели (для inference)."""
        self._backup = {}
        for name, p in self.model.named_parameters():
            if name in self._shadow:
                self._backup[name] = p.data.clone()
                p.data.copy_(self._shadow[name])

    def restore(self):
        """Восстановить оригинальные weights."""
        for name, p in self.model.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    def get_shadow_state_dict(self):
        """Получить shadow weights как state_dict."""
        return dict(self._shadow)

    def load_shadow_state_dict(self, state):
        """Загрузить shadow weights."""
        for name, tensor in state.items():
            if name in self._shadow:
                self._shadow[name].copy_(tensor)

    @property
    def step(self):
        return self._step

    @property
    def current_decay(self):
        return self._get_decay()


# ==================== Curriculum Learning Scheduler ====================

class CurriculumScheduler:
    """
    Расписание сложности данных для curriculum learning.

    Управляет порядком представления данных: от простых к сложным.

    Стратегии:
    - linear: линейное увеличение сложности
    - sqrt: корневое (быстрый старт, медленный рост)
    - step: ступенчатое (discrete stages)
    - exponential: экспоненциальное нарастание

    Args:
        total_steps: общее число шагов обучения
        strategy: стратегия расписания
        n_stages: число стадий (для step стратегии)
        min_difficulty: начальная сложность (0-1)
        max_difficulty: конечная сложность (0-1)
    """
    def __init__(self, total_steps, strategy='linear', n_stages=5,
                 min_difficulty=0.0, max_difficulty=1.0):
        self.total_steps = max(total_steps, 1)
        self.strategy = strategy
        self.n_stages = n_stages
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty

    def get_difficulty(self, step):
        """
        Текущий уровень сложности.

        Args:
            step: текущий шаг

        Returns:
            float: difficulty в [min_difficulty, max_difficulty]
        """
        progress = min(step / self.total_steps, 1.0)

        if self.strategy == 'linear':
            raw = progress
        elif self.strategy == 'sqrt':
            raw = math.sqrt(progress)
        elif self.strategy == 'step':
            stage = min(int(progress * self.n_stages), self.n_stages - 1)
            raw = (stage + 1) / self.n_stages
        elif self.strategy == 'exponential':
            raw = (math.exp(progress * 3) - 1) / (math.exp(3) - 1)
        else:
            raw = progress

        diff_range = self.max_difficulty - self.min_difficulty
        return self.min_difficulty + raw * diff_range

    def get_data_fraction(self, step):
        """
        Какую долю данных использовать (от самых простых).

        Args:
            step: текущий шаг

        Returns:
            float: fraction в (0, 1]
        """
        return max(0.01, self.get_difficulty(step))

    def get_max_seq_len(self, step, min_len=4, max_len=128):
        """
        Текущая максимальная длина последовательности.

        Args:
            step: текущий шаг
            min_len: минимальная длина
            max_len: максимальная длина

        Returns:
            int
        """
        difficulty = self.get_difficulty(step)
        return max(min_len, int(min_len + difficulty * (max_len - min_len)))

    def get_stage_info(self, step):
        """Информация о текущей стадии."""
        difficulty = self.get_difficulty(step)
        progress = min(step / self.total_steps, 1.0)
        if self.strategy == 'step':
            stage = min(int(progress * self.n_stages), self.n_stages - 1)
        else:
            stage = int(difficulty * 10)  # approximate
        return {
            'step': step,
            'progress': progress,
            'difficulty': difficulty,
            'stage': stage,
            'strategy': self.strategy,
        }


# ==================== Gradient Noise Injection ====================

class GradientNoise:
    """
    Добавление затухающего шума к градиентам.

    Noise ~ N(0, σ²), где σ = η / (1 + t)^γ

    Помогает:
    - Escape sharp minima в начале
    - Explore loss landscape
    - Regularize training

    Args:
        eta: начальная амплитуда шума
        gamma: скорость затухания (0.55 типично)
    """
    def __init__(self, eta=0.1, gamma=0.55):
        self.eta = eta
        self.gamma = gamma
        self._step = 0

    def get_noise_std(self):
        """Текущее стандартное отклонение шума."""
        return self.eta / (1 + self._step) ** self.gamma

    def add_noise(self, model):
        """
        Добавляет шум к градиентам модели.

        Args:
            model: nn.Module

        Returns:
            dict: {noise_std, n_params_noised}
        """
        self._step += 1
        std = self.get_noise_std()
        n_noised = 0

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * std
                    p.grad.add_(noise)
                    n_noised += 1

        return {'noise_std': std, 'n_params_noised': n_noised}

    def add_noise_to_params(self, parameters):
        """
        Добавляет шум к градиентам заданных параметров.

        Args:
            parameters: iterable of Parameters

        Returns:
            dict
        """
        self._step += 1
        std = self.get_noise_std()
        n_noised = 0

        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * std
                    p.grad.add_(noise)
                    n_noised += 1

        return {'noise_std': std, 'n_params_noised': n_noised}

    @property
    def step(self):
        return self._step

    def reset(self):
        self._step = 0


# ==================== Learning Rate Probing ====================

class LRProbe:
    """
    Автоматический поиск диапазона learning rate.

    Увеличивает LR от min_lr до max_lr на протяжении n_steps,
    записывает loss. Оптимальный LR — где loss падает быстрее всего.

    Args:
        model: nn.Module
        min_lr: начальный LR
        max_lr: конечный LR
        n_steps: число шагов пробинга
    """
    def __init__(self, model, min_lr=1e-7, max_lr=10.0, n_steps=100):
        self.model = model
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_steps = n_steps
        self.history = []
        self._initial_state = None

    def probe(self, data_fn, loss_fn=None):
        """
        Запускает LR probing.

        Args:
            data_fn: callable() → (input, target)
            loss_fn: callable(model, x, y) → loss (optional)

        Returns:
            dict: {best_lr, lr_history, loss_history, suggested_range}
        """
        # Save initial weights
        self._initial_state = copy.deepcopy(self.model.state_dict())

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.min_lr)

        # Multiplicative factor for geometric increase
        factor = (self.max_lr / self.min_lr) ** (1.0 / self.n_steps)

        self.history = []
        best_loss = float('inf')
        smoothed_loss = None

        for step in range(self.n_steps):
            lr = self.min_lr * (factor ** step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad()
            x, y = data_fn()

            if loss_fn is not None:
                loss = loss_fn(self.model, x, y)
            else:
                output = self.model(x, y)
                if isinstance(output, tuple) and len(output) >= 2:
                    loss = output[1]
                else:
                    loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()

            loss_val = loss.item()

            # Smoothed loss
            if smoothed_loss is None:
                smoothed_loss = loss_val
            else:
                smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss_val

            self.history.append({
                'step': step,
                'lr': lr,
                'loss': loss_val,
                'smoothed_loss': smoothed_loss,
            })

            # Stop if loss diverges
            if step > 5 and smoothed_loss > best_loss * 4:
                break

            best_loss = min(best_loss, smoothed_loss)

            loss.backward()
            optimizer.step()

        # Restore initial weights
        self.model.load_state_dict(self._initial_state)
        self._initial_state = None

        return self._analyze()

    def _analyze(self):
        """Анализирует результаты probing."""
        if len(self.history) < 3:
            return {
                'best_lr': self.min_lr,
                'lr_history': [],
                'loss_history': [],
                'suggested_range': (self.min_lr, self.max_lr),
            }

        lrs = [h['lr'] for h in self.history]
        losses = [h['smoothed_loss'] for h in self.history]

        # Find steepest descent
        best_idx = 0
        best_descent = 0
        for i in range(1, len(losses)):
            descent = losses[i - 1] - losses[i]
            if descent > best_descent:
                best_descent = descent
                best_idx = i

        best_lr = lrs[best_idx]

        # Suggested range: 1/10 to 1x of best
        suggested = (best_lr / 10, best_lr)

        return {
            'best_lr': best_lr,
            'lr_history': lrs,
            'loss_history': losses,
            'suggested_range': suggested,
        }


# ==================== Weight Decay Scheduler ====================

class WeightDecayScheduler:
    """
    Адаптивный weight decay scheduler.

    Стратегии:
    - constant: фиксированный
    - linear: линейное уменьшение
    - cosine: косинусное затухание
    - proportional: пропорционально LR (WD/LR = const)

    Args:
        optimizer: optimizer с weight_decay
        initial_wd: начальный weight decay
        final_wd: конечный weight decay
        total_steps: общее число шагов
        strategy: стратегия
    """
    def __init__(self, optimizer, initial_wd=0.01, final_wd=0.0,
                 total_steps=1000, strategy='cosine'):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.total_steps = max(total_steps, 1)
        self.strategy = strategy
        self._step = 0
        self._initial_lrs = [g.get('lr', 0.001) for g in optimizer.param_groups]

    def step(self):
        """Обновить weight decay."""
        self._step += 1
        wd = self._compute_wd()
        self._apply_wd(wd)
        return wd

    def _compute_wd(self):
        """Вычислить текущий weight decay."""
        progress = min(self._step / self.total_steps, 1.0)

        if self.strategy == 'constant':
            return self.initial_wd
        elif self.strategy == 'linear':
            return self.initial_wd + (self.final_wd - self.initial_wd) * progress
        elif self.strategy == 'cosine':
            return self.final_wd + (self.initial_wd - self.final_wd) * \
                   0.5 * (1 + math.cos(math.pi * progress))
        elif self.strategy == 'proportional':
            # Keep WD/LR ratio constant
            current_lr = self.optimizer.param_groups[0].get('lr', 0.001)
            if self._initial_lrs[0] > 0:
                ratio = self.initial_wd / self._initial_lrs[0]
                return current_lr * ratio
            return self.initial_wd
        return self.initial_wd

    def _apply_wd(self, wd):
        """Применить WD ко всем param groups."""
        for group in self.optimizer.param_groups:
            group['weight_decay'] = wd

    def get_current_wd(self):
        """Текущий weight decay."""
        return self._compute_wd()

    @property
    def current_step(self):
        return self._step
