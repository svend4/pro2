"""
v44 утилиты: Gradient Noise, Lookahead Optimizer,
Layer-wise LR Decay, SWA, Warmup-Stable-Decay Schedule.

Gradient Noise: регуляризация через шум в градиентах.
Ref: Neelakantan et al., "Adding Gradient Noise Improves Learning" (2015)

Lookahead: медленные веса + быстрые обновления.
Ref: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back" (2019)

Layer-wise LR Decay: глубокие слои — меньший LR.
Ref: Clark et al., "ELECTRA" (2020); Howard & Ruder, "ULMFiT" (2018)

SWA: усреднение весов для плоских минимумов.
Ref: Izmailov et al., "Averaging Weights Leads to Wider Optima" (2018)

WSD Schedule: трёхфазный scheduler.
Ref: Zhai et al., "Scaling Vision Transformers" (2022)
"""

import math
import copy
import torch
import torch.nn as nn


# ==================== Gradient Noise Injection ====================

class GradientNoiseInjector:
    """
    Добавление гауссова шума к градиентам.

    σ(t) = η / (1 + t)^γ — убывающий шум по расписанию.

    Args:
        eta: начальная дисперсия шума
        gamma: скорость убывания
    """
    def __init__(self, eta=0.1, gamma=0.55):
        self.eta = eta
        self.gamma = gamma
        self._step = 0

    def get_noise_std(self):
        """Текущее σ шума."""
        return self.eta / (1 + self._step) ** self.gamma

    def inject(self, model):
        """
        Добавляет шум к градиентам.

        Args:
            model: nn.Module (после backward)

        Returns:
            dict: {noise_std, n_params}
        """
        self._step += 1
        std = self.get_noise_std()
        n = 0

        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * std
                p.grad.data.add_(noise)
                n += 1

        return {'noise_std': std, 'n_params': n}

    def get_info(self):
        return {
            'step': self._step,
            'current_std': self.get_noise_std(),
            'eta': self.eta,
            'gamma': self.gamma,
        }


# ==================== Lookahead Optimizer ====================

class Lookahead:
    """
    Lookahead optimizer wrapper.

    Поддерживает slow weights, обновляемые каждые k шагов:
    θ_slow = θ_slow + α * (θ_fast - θ_slow)

    Args:
        optimizer: базовый оптимизатор
        k: шагов между sync
        alpha: interpolation coefficient
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self._step = 0
        self._slow_params = []

        # Initialize slow params
        for group in optimizer.param_groups:
            slow = []
            for p in group['params']:
                slow.append(p.data.clone())
            self._slow_params.append(slow)

    def step(self, closure=None):
        """
        Шаг оптимизатора + lookahead sync.

        Returns:
            dict: {synced, step}
        """
        loss = self.optimizer.step(closure)
        self._step += 1

        synced = False
        if self._step % self.k == 0:
            self._sync()
            synced = True

        return {'synced': synced, 'step': self._step}

    def _sync(self):
        """Синхронизация slow weights."""
        for group_idx, group in enumerate(self.optimizer.param_groups):
            for p_idx, p in enumerate(group['params']):
                slow = self._slow_params[group_idx][p_idx]
                slow.add_(p.data - slow, alpha=self.alpha)
                p.data.copy_(slow)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'step': self._step,
        }

    def get_info(self):
        return {
            'k': self.k,
            'alpha': self.alpha,
            'step': self._step,
            'next_sync_in': self.k - (self._step % self.k),
        }


# ==================== Layer-wise LR Decay ====================

class LayerwiseLRDecay:
    """
    Разный LR для разных глубин модели.

    Более глубокие (ранние) слои получают меньший LR:
    lr_layer = base_lr * decay^(n_layers - layer_idx)

    Args:
        base_lr: LR для последнего слоя
        decay: множитель за слой (0.65-0.95 типично)
    """
    def __init__(self, base_lr=1e-3, decay=0.8):
        self.base_lr = base_lr
        self.decay = decay

    def get_param_groups(self, model, n_layers=None):
        """
        Создаёт param groups с разным LR.

        Args:
            model: nn.Module
            n_layers: число слоёв (auto-detect если None)

        Returns:
            list[dict]: param groups для optimizer
        """
        # Detect layers
        layer_params = {}
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            layer_idx = self._get_layer_index(name)
            if layer_idx is not None:
                if layer_idx not in layer_params:
                    layer_params[layer_idx] = []
                layer_params[layer_idx].append(param)
            else:
                other_params.append(param)

        if n_layers is None:
            n_layers = max(layer_params.keys()) + 1 if layer_params else 1

        groups = []

        # Layer-specific groups
        for idx in sorted(layer_params.keys()):
            lr = self.base_lr * (self.decay ** (n_layers - 1 - idx))
            groups.append({
                'params': layer_params[idx],
                'lr': lr,
                'layer_idx': idx,
            })

        # Other params (embeddings, head, etc.)
        if other_params:
            groups.append({
                'params': other_params,
                'lr': self.base_lr,
                'layer_idx': -1,
            })

        return groups

    def _get_layer_index(self, name):
        """Извлекает индекс слоя из имени параметра."""
        import re
        # Common patterns: layers.0, blocks.1, h.2, etc.
        patterns = [
            r'layers?\.(\d+)',
            r'blocks?\.(\d+)',
            r'h\.(\d+)',
            r'transformer\.(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None

    def get_lr_schedule(self, n_layers):
        """
        Предпросмотр LR для каждого слоя.

        Returns:
            list[dict]: {layer, lr}
        """
        return [
            {'layer': i, 'lr': self.base_lr * (self.decay ** (n_layers - 1 - i))}
            for i in range(n_layers)
        ]


# ==================== Stochastic Weight Averaging ====================

class StochasticWeightAveraging:
    """
    SWA: усреднение весов на последних эпохах.

    Собирает snapshot'ы весов и усредняет для
    более плоского минимума.

    Args:
        swa_start: шаг начала SWA
        swa_freq: частота сбора snapshot'ов
    """
    def __init__(self, swa_start=1000, swa_freq=10):
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self._step = 0
        self._n_averaged = 0
        self._avg_params = None

    def update(self, model):
        """
        Обновляет SWA если нужно.

        Args:
            model: nn.Module

        Returns:
            dict: {averaged, n_averaged, step}
        """
        self._step += 1

        if self._step < self.swa_start:
            return {'averaged': False, 'n_averaged': self._n_averaged,
                    'step': self._step}

        if (self._step - self.swa_start) % self.swa_freq != 0:
            return {'averaged': False, 'n_averaged': self._n_averaged,
                    'step': self._step}

        # Initialize or update average
        if self._avg_params is None:
            self._avg_params = {
                name: p.data.clone()
                for name, p in model.named_parameters()
            }
            self._n_averaged = 1
        else:
            self._n_averaged += 1
            for name, p in model.named_parameters():
                if name in self._avg_params:
                    self._avg_params[name].mul_(
                        (self._n_averaged - 1) / self._n_averaged
                    ).add_(p.data, alpha=1.0 / self._n_averaged)

        return {'averaged': True, 'n_averaged': self._n_averaged,
                'step': self._step}

    def apply_average(self, model):
        """
        Применяет усреднённые веса к модели.

        Args:
            model: nn.Module

        Returns:
            bool: True если применено
        """
        if self._avg_params is None:
            return False

        for name, p in model.named_parameters():
            if name in self._avg_params:
                p.data.copy_(self._avg_params[name])
        return True

    def get_info(self):
        return {
            'step': self._step,
            'swa_active': self._step >= self.swa_start,
            'n_averaged': self._n_averaged,
        }


# ==================== Warmup-Stable-Decay Schedule ====================

class WarmupStableDecaySchedule:
    """
    Трёхфазный LR scheduler: warmup → stable → decay.

    Фаза 1 (warmup): линейный рост от 0 до base_lr
    Фаза 2 (stable): постоянный base_lr
    Фаза 3 (decay): косинусное/линейное снижение до min_lr

    Args:
        base_lr: пиковый LR
        total_steps: общее число шагов
        warmup_fraction: доля шагов на warmup
        decay_fraction: доля шагов на decay
        min_lr: минимальный LR
        decay_type: 'cosine' или 'linear'
    """
    def __init__(self, base_lr=1e-3, total_steps=10000,
                 warmup_fraction=0.1, decay_fraction=0.3,
                 min_lr=1e-6, decay_type='cosine'):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.decay_steps = int(total_steps * decay_fraction)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.min_lr = min_lr
        self.decay_type = decay_type
        self._step = 0
        self._lr_history = []

    def step(self):
        """
        Один шаг scheduler.

        Returns:
            float: текущий LR
        """
        self._step += 1

        if self._step <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * self._step / max(self.warmup_steps, 1)
        elif self._step <= self.warmup_steps + self.stable_steps:
            # Stable phase
            lr = self.base_lr
        else:
            # Decay phase
            decay_progress = (self._step - self.warmup_steps - self.stable_steps) / max(self.decay_steps, 1)
            decay_progress = min(decay_progress, 1.0)

            if self.decay_type == 'cosine':
                lr = self.min_lr + (self.base_lr - self.min_lr) * \
                     (1 + math.cos(math.pi * decay_progress)) / 2
            else:  # linear
                lr = self.base_lr - (self.base_lr - self.min_lr) * decay_progress

        self._lr_history.append(lr)
        return lr

    def apply(self, optimizer):
        """
        Применяет LR к оптимизатору.

        Returns:
            float: текущий LR
        """
        lr = self.step()
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_phase(self):
        """Текущая фаза."""
        if self._step <= self.warmup_steps:
            return 'warmup'
        elif self._step <= self.warmup_steps + self.stable_steps:
            return 'stable'
        else:
            return 'decay'

    def get_info(self):
        return {
            'step': self._step,
            'phase': self.get_phase(),
            'current_lr': self._lr_history[-1] if self._lr_history else self.base_lr,
            'warmup_steps': self.warmup_steps,
            'stable_steps': self.stable_steps,
            'decay_steps': self.decay_steps,
        }
