"""
v24 утилиты: LLRD, Gradient Accumulation Scheduler, Attention Entropy,
Weight Decay Scheduler, Loss Spike Detector.

LLRD: layer-wise learning rate decay для fine-tuning.
Ref: Clark et al., "ELECTRA" (2020); Howard & Ruder, "ULMFiT" (2018)

Gradient Accumulation Scheduler: динамические accumulation steps.

Attention Entropy Monitor: отслеживание распределения attention.

Weight Decay Scheduler: динамический weight decay.

Loss Spike Detector: обнаружение нестабильностей при обучении.
"""

import math
import torch
import torch.nn as nn
from collections import deque


# ==================== Layer-wise Learning Rate Decay ====================

class LLRD:
    """
    Layer-wise Learning Rate Decay.

    Нижние слои получают меньший LR: lr_layer = base_lr * decay^(n-i),
    где n — общее число слоёв, i — индекс слоя.
    Полезно при fine-tuning: нижние слои уже хорошо обучены.

    Args:
        model: nn.Module с атрибутом layers/blocks
        base_lr: базовый LR (для верхнего слоя)
        decay: множитель на слой (0.65-0.95)
        n_layers: число слоёв (auto-detect если None)
    """
    def __init__(self, model, base_lr=1e-4, decay=0.8, n_layers=None):
        self.model = model
        self.base_lr = base_lr
        self.decay = decay
        self.n_layers = n_layers or self._detect_layers()

    def _detect_layers(self):
        """Определяет число слоёв в модели."""
        for attr in ['layers', 'blocks', 'transformer_blocks']:
            if hasattr(self.model, attr):
                return len(getattr(self.model, attr))
        # Count Sequential children
        count = 0
        for name, _ in self.model.named_children():
            count += 1
        return max(count, 1)

    def get_layer_lr(self, layer_idx):
        """LR для конкретного слоя."""
        depth_from_top = self.n_layers - 1 - layer_idx
        return self.base_lr * (self.decay ** depth_from_top)

    def get_param_groups(self):
        """
        Формирует param_groups для optimizer.

        Returns:
            list[dict]: [{params, lr, layer_name}, ...]
        """
        groups = []

        # Embedding и прочие non-layer параметры → минимальный LR
        non_layer_params = []
        layer_params = {i: [] for i in range(self.n_layers)}
        layer_found = False

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            assigned = False
            for attr in ['layers', 'blocks', 'transformer_blocks', 'core.layers']:
                for i in range(self.n_layers):
                    patterns = [
                        f'{attr}.{i}.',
                        f'{attr}[{i}].',
                    ]
                    if any(p in name for p in patterns):
                        layer_params[i].append(param)
                        assigned = True
                        layer_found = True
                        break
                if assigned:
                    break

            if not assigned:
                non_layer_params.append(param)

        # Non-layer params get lowest LR
        if non_layer_params:
            lowest_lr = self.base_lr * (self.decay ** self.n_layers)
            groups.append({
                'params': non_layer_params,
                'lr': lowest_lr,
                'layer_name': 'non_layer',
            })

        # Layer params
        for i in range(self.n_layers):
            if layer_params[i]:
                groups.append({
                    'params': layer_params[i],
                    'lr': self.get_layer_lr(i),
                    'layer_name': f'layer_{i}',
                })

        return groups

    def get_lr_schedule(self):
        """Возвращает LR для каждого слоя (для логирования)."""
        return {i: self.get_layer_lr(i) for i in range(self.n_layers)}


# ==================== Gradient Accumulation Scheduler ====================

class GradAccumulationScheduler:
    """
    Динамический Gradient Accumulation.

    Увеличивает effective batch size по мере обучения,
    начиная с малого accumulation и увеличивая.

    Args:
        start_accum: начальное число accumulation steps
        max_accum: максимальное число
        warmup_steps: шагов до max_accum
        strategy: 'linear', 'step', 'exponential'
    """
    def __init__(self, start_accum=1, max_accum=8, warmup_steps=1000, strategy='linear'):
        self.start_accum = start_accum
        self.max_accum = max_accum
        self.warmup_steps = warmup_steps
        self.strategy = strategy

    def get_accum_steps(self, step):
        """
        Число accumulation steps на данном шаге.

        Returns:
            int: accumulation steps
        """
        if step >= self.warmup_steps:
            return self.max_accum

        t = step / max(self.warmup_steps, 1)

        if self.strategy == 'linear':
            raw = self.start_accum + t * (self.max_accum - self.start_accum)
        elif self.strategy == 'exponential':
            log_start = math.log(max(self.start_accum, 1))
            log_max = math.log(max(self.max_accum, 1))
            raw = math.exp(log_start + t * (log_max - log_start))
        elif self.strategy == 'step':
            # Двоичные шаги: 1 → 2 → 4 → 8
            n_doublings = math.log2(self.max_accum / max(self.start_accum, 1))
            current_doubling = int(t * n_doublings)
            raw = self.start_accum * (2 ** current_doubling)
        else:
            raw = self.max_accum

        # Round to nearest power of 2
        result = max(1, min(int(raw), self.max_accum))
        # Snap to power of 2
        power = round(math.log2(max(result, 1)))
        return min(2 ** power, self.max_accum)

    def should_step(self, micro_step, global_step):
        """
        Должен ли optimizer сделать шаг?

        Args:
            micro_step: текущий micro-step внутри accumulation
            global_step: глобальный шаг обучения

        Returns:
            bool
        """
        accum = self.get_accum_steps(global_step)
        return (micro_step + 1) % accum == 0

    def get_scale(self, global_step):
        """Loss scale = 1/accum_steps."""
        return 1.0 / self.get_accum_steps(global_step)

    def get_stats(self, step):
        accum = self.get_accum_steps(step)
        return {
            'step': step,
            'accum_steps': accum,
            'loss_scale': self.get_scale(step),
            'effective_batch_multiplier': accum,
        }


# ==================== Attention Entropy Monitor ====================

class AttentionEntropyMonitor:
    """
    Мониторинг энтропии attention распределений.

    Высокая энтропия → uniform attention (модель не фокусируется).
    Низкая энтропия → sharp attention (возможно, деградация).

    Args:
        n_layers: число слоёв
        n_heads: число голов
    """
    def __init__(self, n_layers=None, n_heads=None):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.history = []
        self._hooks = []

    def compute_entropy(self, attn_weights):
        """
        Вычисляет энтропию attention weights.

        Args:
            attn_weights: (B, H, T, T) — attention probabilities

        Returns:
            dict: {per_head: (H,), mean, min, max}
        """
        # Clamp to avoid log(0)
        attn = attn_weights.clamp(min=1e-8)
        # Entropy: -sum(p * log(p)) per position, averaged
        entropy = -(attn * attn.log()).sum(dim=-1)  # (B, H, T)
        per_head = entropy.mean(dim=(0, 2))  # (H,)

        return {
            'per_head': per_head.detach().cpu(),
            'mean': per_head.mean().item(),
            'min': per_head.min().item(),
            'max': per_head.max().item(),
        }

    def update(self, attn_weights, layer_idx=0):
        """
        Записывает энтропию для слоя.

        Args:
            attn_weights: (B, H, T, T)
            layer_idx: индекс слоя
        """
        entropy_info = self.compute_entropy(attn_weights)
        entropy_info['layer'] = layer_idx
        self.history.append(entropy_info)

    def get_summary(self):
        """Последняя статистика по всем записанным слоям."""
        if not self.history:
            return {}
        latest = {}
        for entry in self.history:
            latest[entry['layer']] = {
                'mean': entry['mean'],
                'min': entry['min'],
                'max': entry['max'],
            }
        return latest

    def detect_collapse(self, threshold=0.1):
        """
        Обнаруживает коллапс attention (слишком низкая энтропия).

        Args:
            threshold: порог энтропии

        Returns:
            list[(layer, head_idx, entropy)]
        """
        collapsed = []
        for entry in self.history[-20:]:  # last 20 entries
            per_head = entry['per_head']
            for h, e in enumerate(per_head):
                if e.item() < threshold:
                    collapsed.append((entry['layer'], h, e.item()))
        return collapsed

    def detect_uniformity(self, seq_len=32):
        """
        Обнаруживает слишком uniform attention.

        Args:
            seq_len: длина последовательности для порога

        Returns:
            list[(layer, head_idx, entropy)]
        """
        max_entropy = math.log(seq_len)
        uniform = []
        for entry in self.history[-20:]:
            per_head = entry['per_head']
            for h, e in enumerate(per_head):
                if e.item() > 0.95 * max_entropy:
                    uniform.append((entry['layer'], h, e.item()))
        return uniform

    def reset(self):
        self.history.clear()


# ==================== Weight Decay Scheduler ====================

class WeightDecayScheduler:
    """
    Динамический Weight Decay.

    Меняет weight decay в процессе обучения.
    Стратегии: constant, linear warmup, cosine.

    Args:
        optimizer: optimizer с param_groups
        base_wd: базовый weight decay
        total_steps: общее число шагов
        strategy: 'constant', 'linear', 'cosine'
        warmup_steps: шагов до base_wd
    """
    def __init__(self, optimizer, base_wd=0.01, total_steps=10000,
                 strategy='constant', warmup_steps=0):
        self.optimizer = optimizer
        self.base_wd = base_wd
        self.total_steps = total_steps
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self._step = 0

    def get_wd(self, step=None):
        """Текущий weight decay."""
        if step is None:
            step = self._step
        t = min(step / max(self.total_steps, 1), 1.0)

        if self.strategy == 'constant':
            wd = self.base_wd
        elif self.strategy == 'linear':
            if step < self.warmup_steps:
                wd = self.base_wd * step / max(self.warmup_steps, 1)
            else:
                wd = self.base_wd
        elif self.strategy == 'cosine':
            wd = self.base_wd * 0.5 * (1 + math.cos(math.pi * t))
        else:
            wd = self.base_wd

        return wd

    def step(self):
        """Обновляет weight decay в optimizer."""
        self._step += 1
        wd = self.get_wd()
        for group in self.optimizer.param_groups:
            group['weight_decay'] = wd
        return wd

    def get_schedule(self, n_points=10):
        """Возвращает расписание WD для визуализации."""
        points = []
        for i in range(n_points + 1):
            step = int(i * self.total_steps / n_points)
            points.append((step, self.get_wd(step)))
        return points


# ==================== Loss Spike Detector ====================

class LossSpikeDetector:
    """
    Детектор скачков loss при обучении.

    Отслеживает EMA loss и обнаруживает аномальные скачки.
    Может рекомендовать действия: skip batch, reduce LR, rollback.

    Args:
        window_size: размер окна для статистик
        spike_threshold: порог (в стд. откл.) для спайка
        ema_decay: decay для EMA loss
    """
    def __init__(self, window_size=100, spike_threshold=3.0, ema_decay=0.99):
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.ema_decay = ema_decay
        self.loss_history = deque(maxlen=window_size)
        self.ema_loss = None
        self.ema_var = None
        self.spike_log = []
        self._step = 0

    def update(self, loss_value):
        """
        Обновляет трекер и проверяет на спайк.

        Args:
            loss_value: текущий loss (float)

        Returns:
            dict: {is_spike, loss, ema_loss, z_score, action}
        """
        self._step += 1
        self.loss_history.append(loss_value)

        # Initialize EMA
        if self.ema_loss is None:
            self.ema_loss = loss_value
            self.ema_var = 0.0
            return {
                'is_spike': False,
                'loss': loss_value,
                'ema_loss': self.ema_loss,
                'z_score': 0.0,
                'action': 'none',
            }

        # Update EMA
        diff = loss_value - self.ema_loss
        self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * loss_value
        self.ema_var = self.ema_decay * self.ema_var + (1 - self.ema_decay) * diff ** 2
        std = max(math.sqrt(self.ema_var), 1e-8)

        # Z-score
        z_score = diff / std

        is_spike = abs(z_score) > self.spike_threshold
        action = 'none'
        if is_spike:
            if z_score > 5.0:
                action = 'rollback'
            elif z_score > self.spike_threshold:
                action = 'skip_batch'
            self.spike_log.append({
                'step': self._step,
                'loss': loss_value,
                'z_score': z_score,
                'action': action,
            })

        return {
            'is_spike': is_spike,
            'loss': loss_value,
            'ema_loss': self.ema_loss,
            'z_score': z_score,
            'action': action,
        }

    def get_stats(self):
        """Статистика трекера."""
        if not self.loss_history:
            return {}
        losses = list(self.loss_history)
        return {
            'n_steps': self._step,
            'current_ema': self.ema_loss,
            'n_spikes': len(self.spike_log),
            'recent_mean': sum(losses) / len(losses),
            'recent_min': min(losses),
            'recent_max': max(losses),
        }

    def get_spike_log(self):
        return list(self.spike_log)

    def is_diverging(self, n_recent=10):
        """Проверяет, расходится ли обучение."""
        if len(self.loss_history) < n_recent:
            return False
        recent = list(self.loss_history)[-n_recent:]
        return all(recent[i] > recent[i - 1] for i in range(1, len(recent)))

    def reset(self):
        self.loss_history.clear()
        self.ema_loss = None
        self.ema_var = None
        self.spike_log.clear()
        self._step = 0
