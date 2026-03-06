"""
v23 утилиты: Lookahead, Activation Histogram, Curriculum Sampler, EMA Model, Gradient Noise.

Lookahead Optimizer: медленные/быстрые веса для стабилизации обучения.
Ref: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (2019)

Activation Histogram: трекинг распределений активаций по слоям.

Curriculum Sampler: расписание сложности обучающих данных.
Ref: Bengio et al., "Curriculum Learning" (2009)

EMA Model: экспоненциальное скользящее среднее весов для inference.
Ref: Polyak & Juditsky (1992)

Gradient Noise Injection: калиброванный шум для регуляризации.
Ref: Neelakantan et al., "Adding Gradient Noise Improves Learning" (2015)
"""

import math
import copy
import torch
import torch.nn as nn
from collections import defaultdict


# ==================== Lookahead Optimizer ====================

class Lookahead:
    """
    Lookahead Optimizer wrapper.

    Поддерживает два набора весов:
    - fast weights: обновляются inner optimizer каждый шаг
    - slow weights: обновляются каждые k шагов интерполяцией

    θ_slow ← θ_slow + α(θ_fast - θ_slow)

    Args:
        optimizer: inner optimizer (e.g., Adam)
        k: число шагов между slow updates
        alpha: интерполяция slow weights (0..1)
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0

        # Cache slow weights
        self.slow_params = []
        for group in optimizer.param_groups:
            slow = []
            for p in group['params']:
                slow.append(p.data.clone())
            self.slow_params.append(slow)

    def step(self, closure=None):
        """Inner optimizer step + periodic slow weight update."""
        loss = self.optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            self._sync_slow_weights()

        return loss

    def _sync_slow_weights(self):
        """Slow weight update: θ_slow += α(θ_fast - θ_slow)."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for p_idx, p in enumerate(group['params']):
                slow = self.slow_params[group_idx][p_idx]
                slow.add_(self.alpha * (p.data - slow))
                p.data.copy_(slow)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_params': self.slow_params,
            'step_count': self._step_count,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self.slow_params = state['slow_params']
        self._step_count = state['step_count']


# ==================== Activation Histogram ====================

class ActivationHistogram:
    """
    Трекер распределений активаций по слоям.

    Собирает гистограммы активаций для диагностики
    dead neurons, saturation, и распределения значений.

    Args:
        n_bins: число бинов гистограммы
        track_every: каждый N-й батч
    """
    def __init__(self, n_bins=50, track_every=1):
        self.n_bins = n_bins
        self.track_every = track_every
        self.histograms = defaultdict(list)
        self.stats = defaultdict(list)
        self._step = 0
        self._hooks = []

    def register(self, model, layer_names=None):
        """
        Регистрирует hooks на слоях модели.

        Args:
            model: nn.Module
            layer_names: list[str] — имена слоёв (None = все)
        """
        for name, module in model.named_modules():
            if layer_names is not None and name not in layer_names:
                continue
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.LayerNorm)):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self._step += 1
            if self._step % self.track_every != 0:
                return
            with torch.no_grad():
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                act_flat = act.float().flatten()

                # Stats
                self.stats[name].append({
                    'mean': act_flat.mean().item(),
                    'std': act_flat.std().item(),
                    'min': act_flat.min().item(),
                    'max': act_flat.max().item(),
                    'dead_frac': (act_flat == 0).float().mean().item(),
                })

                # Histogram
                hist = torch.histc(act_flat, bins=self.n_bins)
                self.histograms[name].append(hist.cpu())
        return hook_fn

    def get_stats(self, name):
        """Возвращает статистику для слоя."""
        return self.stats.get(name, [])

    def get_latest_stats(self):
        """Последняя статистика по всем слоям."""
        result = {}
        for name, stat_list in self.stats.items():
            if stat_list:
                result[name] = stat_list[-1]
        return result

    def get_dead_neuron_report(self, threshold=0.5):
        """
        Слои с высокой долей мёртвых нейронов.

        Args:
            threshold: порог dead_frac

        Returns:
            list[(name, dead_frac)]
        """
        report = []
        for name, stat_list in self.stats.items():
            if stat_list:
                dead = stat_list[-1]['dead_frac']
                if dead >= threshold:
                    report.append((name, dead))
        return sorted(report, key=lambda x: -x[1])

    def remove_hooks(self):
        """Удаляет все hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self.histograms.clear()
        self.stats.clear()
        self._step = 0


# ==================== Curriculum Sampler ====================

class CurriculumSampler:
    """
    Curriculum Learning: расписание сложности данных.

    Начинает с простых примеров, постепенно добавляя сложные.
    Сложность определяется внешней функцией.

    Args:
        difficulties: tensor/list — сложность каждого примера [0..1]
        total_steps: общее число шагов обучения
        strategy: 'linear', 'sqrt', 'step'
    """
    def __init__(self, difficulties, total_steps, strategy='linear'):
        if isinstance(difficulties, list):
            difficulties = torch.tensor(difficulties, dtype=torch.float32)
        self.difficulties = difficulties
        self.total_steps = total_steps
        self.strategy = strategy
        self.n_samples = len(difficulties)

    def get_competence(self, step):
        """
        Текущий уровень компетенции [0..1].

        Args:
            step: текущий шаг

        Returns:
            float: competence level
        """
        t = min(step / max(self.total_steps, 1), 1.0)
        if self.strategy == 'linear':
            return t
        elif self.strategy == 'sqrt':
            return math.sqrt(t)
        elif self.strategy == 'step':
            if t < 0.33:
                return 0.33
            elif t < 0.66:
                return 0.66
            else:
                return 1.0
        return t

    def sample_indices(self, step, batch_size):
        """
        Выбирает индексы примеров по текущей компетенции.

        Args:
            step: текущий шаг
            batch_size: размер батча

        Returns:
            tensor: индексы выбранных примеров
        """
        competence = self.get_competence(step)
        # Включаем примеры с difficulty <= competence
        mask = self.difficulties <= competence
        valid_indices = torch.where(mask)[0]

        if len(valid_indices) == 0:
            # Fallback: самые простые примеры
            _, sorted_idx = self.difficulties.sort()
            valid_indices = sorted_idx[:max(batch_size, 1)]

        # Random sample from valid
        perm = torch.randperm(len(valid_indices))[:batch_size]
        return valid_indices[perm]

    def get_curriculum_stats(self, step):
        """Статистика текущего curriculum."""
        competence = self.get_competence(step)
        mask = self.difficulties <= competence
        return {
            'step': step,
            'competence': competence,
            'available_samples': mask.sum().item(),
            'total_samples': self.n_samples,
            'coverage': mask.sum().item() / self.n_samples,
        }


# ==================== EMA Model ====================

class EMAModel:
    """
    Exponential Moving Average модели.

    Поддерживает EMA-копию весов для стабильного inference.
    θ_ema = decay * θ_ema + (1 - decay) * θ_model

    Args:
        model: nn.Module
        decay: EMA decay factor (0.999 типично)
        warmup_steps: число шагов до начала EMA
    """
    def __init__(self, model, decay=0.999, warmup_steps=0):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self._step = 0

    def _get_decay(self):
        """Decay с warmup."""
        if self._step < self.warmup_steps:
            return min(self.decay, (1 + self._step) / (10 + self._step))
        return self.decay

    def update(self):
        """Обновляет EMA weights."""
        self._step += 1
        decay = self._get_decay()
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(decay).add_(v, alpha=1 - decay)

    def apply_shadow(self):
        """Применяет EMA веса к модели. Возвращает backup."""
        backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)
        return backup

    def restore(self, backup):
        """Восстанавливает веса из backup."""
        self.model.load_state_dict(backup)

    def get_shadow_state(self):
        """Возвращает EMA state dict."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def state_dict(self):
        return {
            'shadow': self.shadow,
            'step': self._step,
            'decay': self.decay,
        }

    def load_state_dict(self, state):
        self.shadow = state['shadow']
        self._step = state['step']
        self.decay = state['decay']


# ==================== Gradient Noise Injection ====================

class GradientNoiseInjector:
    """
    Gradient Noise для регуляризации.

    Добавляет гауссов шум к градиентам:
    g_t = g_t + N(0, σ²), где σ² = η / (1 + t)^γ

    Args:
        eta: базовая дисперсия шума
        gamma: decay rate (0.55 рекомендуется)
    """
    def __init__(self, eta=0.01, gamma=0.55):
        self.eta = eta
        self.gamma = gamma
        self._step = 0

    def get_variance(self):
        """Текущая дисперсия шума."""
        return self.eta / (1 + self._step) ** self.gamma

    def inject(self, model):
        """
        Добавляет шум к градиентам модели.

        Args:
            model: nn.Module (после backward, до optimizer.step)

        Returns:
            dict: {variance, n_params_noised}
        """
        self._step += 1
        variance = self.get_variance()
        std = math.sqrt(variance)
        n_noised = 0

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * std
                    p.grad.add_(noise)
                    n_noised += 1

        return {
            'variance': variance,
            'std': std,
            'n_params_noised': n_noised,
            'step': self._step,
        }

    def reset(self):
        self._step = 0
