"""
v53 утилиты: Stochastic Depth, Token Mixing MLP, Loss Balancer,
Warmup-Stable-Decay Schedule, Gradient Checkpointing Scheduler.

Stochastic Depth: случайное отключение слоёв при обучении.
Ref: Huang et al., "Deep Networks with Stochastic Depth" (2016)

Token Mixing MLP: MLP-based token mixing (альтернатива attention).
Ref: Tolstikhin et al., "MLP-Mixer" (2021)

Loss Balancer: автоматическая балансировка multi-task losses.
Ref: Kendall et al., "Multi-Task Learning Using Uncertainty" (2018)

Warmup-Stable-Decay Schedule: WSD scheduler для LLM pre-training.
Ref: Zhai et al., "Scaling Vision Transformers" (2022)

Gradient Checkpointing Scheduler: автоматический выбор слоёв для checkpointing.
Ref: Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Stochastic Depth ====================

class StochasticDepth(nn.Module):
    """
    Stochastic Depth: случайно пропускает целые слои при обучении.

    С вероятностью drop_prob слой заменяется на identity.
    При eval — взвешенный output.

    Args:
        drop_prob: вероятность пропуска слоя
        mode: 'row' (per-sample) или 'batch' (весь batch)
    """
    def __init__(self, drop_prob=0.1, mode='row'):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        if mode not in ('row', 'batch'):
            raise ValueError(f"mode must be 'row' or 'batch', got {mode}")
        self.drop_prob = drop_prob
        self.mode = mode

    def forward(self, x, residual=None):
        """
        Args:
            x: output слоя (будет dropped)
            residual: residual connection (если None, возвращает 0 при drop)
        """
        if not self.training or self.drop_prob == 0.0:
            return x if residual is None else x + residual

        if self.mode == 'batch':
            keep_prob = 1.0 - self.drop_prob
            random_tensor = torch.rand(1, device=x.device)
            binary_mask = (random_tensor < keep_prob).float()
            output = x * binary_mask / keep_prob
        else:
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.dim() - 1)
            random_tensor = torch.rand(shape, device=x.device)
            binary_mask = (random_tensor < keep_prob).float()
            output = x * binary_mask / keep_prob

        if residual is not None:
            output = output + residual
        return output

    def get_info(self):
        return {
            'drop_prob': self.drop_prob,
            'mode': self.mode,
            'keep_prob': 1.0 - self.drop_prob,
        }


# ==================== Token Mixing MLP ====================

class TokenMixingMLP(nn.Module):
    """
    Token Mixing через MLP (MLP-Mixer стиль).

    Транспонирует seq и feature dims, применяет MLP вдоль token dimension.
    Альтернатива self-attention для фиксированных seq_len.

    Args:
        seq_len: длина последовательности
        hidden_mult: множитель для скрытого размера
        dropout: dropout rate
    """
    def __init__(self, seq_len, hidden_mult=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(seq_len * hidden_mult)
        self.norm = nn.LayerNorm(seq_len)
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, seq_len)
        self.drop = nn.Dropout(dropout)
        self.seq_len = seq_len

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Transpose: mix along token dimension
        residual = x
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_t = self.norm(x_t)
        x_t = self.fc1(x_t)
        x_t = self.act(x_t)
        x_t = self.drop(x_t)
        x_t = self.fc2(x_t)
        x_t = self.drop(x_t)
        x_t = x_t.transpose(1, 2)  # back to (batch, seq_len, d_model)
        return residual + x_t

    def get_info(self):
        return {
            'seq_len': self.seq_len,
            'params': sum(p.numel() for p in self.parameters()),
        }


# ==================== Loss Balancer ====================

class LossBalancer(nn.Module):
    """
    Автоматическая балансировка multi-task losses через learnable weights.

    Использует homoscedastic uncertainty для оценки весов.
    loss_i_weighted = (1 / (2 * sigma_i²)) * loss_i + log(sigma_i)

    Args:
        n_tasks: количество задач
        initial_weight: начальный лог-вес
    """
    def __init__(self, n_tasks, initial_weight=0.0):
        super().__init__()
        self.log_vars = nn.Parameter(
            torch.full((n_tasks,), initial_weight)
        )
        self.n_tasks = n_tasks

    def forward(self, losses):
        """
        Args:
            losses: list/tuple из n_tasks скалярных losses
        Returns:
            weighted_total: взвешенная сумма
        """
        if len(losses) != self.n_tasks:
            raise ValueError(
                f"Expected {self.n_tasks} losses, got {len(losses)}"
            )
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    def get_weights(self):
        """Возвращает текущие веса (precision) для каждой задачи."""
        with torch.no_grad():
            precisions = torch.exp(-self.log_vars).cpu().tolist()
        return {f'task_{i}': w for i, w in enumerate(precisions)}

    def get_info(self):
        weights = self.get_weights()
        return {
            'n_tasks': self.n_tasks,
            'weights': weights,
            'log_vars': self.log_vars.detach().cpu().tolist(),
        }


# ==================== Warmup-Stable-Decay Schedule ====================

class WarmupStableDecayScheduler:
    """
    WSD (Warmup-Stable-Decay) Learning Rate Scheduler.

    Три фазы:
    1. Warmup: линейный рост от 0 до base_lr
    2. Stable: постоянный base_lr
    3. Decay: cosine decay до min_lr

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: шаги warmup
        stable_steps: шаги стабильной фазы
        decay_steps: шаги decay
        base_lr: базовый learning rate
        min_lr: минимальный LR (конец decay)
    """
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps,
                 base_lr=1e-3, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.current_step = 0

    def get_lr(self, step=None):
        if step is None:
            step = self.current_step
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step + 1) / self.warmup_steps
        elif step < self.warmup_steps + self.stable_steps:
            # Stable phase
            return self.base_lr
        else:
            # Cosine decay
            decay_step = step - self.warmup_steps - self.stable_steps
            decay_step = min(decay_step, self.decay_steps)
            progress = decay_step / max(self.decay_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr

    def get_phase(self, step=None):
        if step is None:
            step = self.current_step
        if step < self.warmup_steps:
            return 'warmup'
        elif step < self.warmup_steps + self.stable_steps:
            return 'stable'
        else:
            return 'decay'

    def get_info(self):
        return {
            'current_step': self.current_step,
            'current_lr': self.get_lr(),
            'phase': self.get_phase(),
            'warmup_steps': self.warmup_steps,
            'stable_steps': self.stable_steps,
            'decay_steps': self.decay_steps,
            'total_steps': self.total_steps,
        }


# ==================== Gradient Checkpointing Scheduler ====================

class GradientCheckpointingScheduler:
    """
    Автоматический планировщик gradient checkpointing.

    Определяет, какие слои checkpoint'ить для оптимального
    баланса memory/compute. Стратегии:
    - 'uniform': каждый k-й слой
    - 'sqrt': checkpoint sqrt(N) слоёв (оптимально по памяти)
    - 'memory_budget': вписаться в бюджет памяти

    Args:
        n_layers: общее число слоёв
        strategy: стратегия выбора
    """
    def __init__(self, n_layers, strategy='sqrt'):
        if strategy not in ('uniform', 'sqrt', 'memory_budget'):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.n_layers = n_layers
        self.strategy = strategy

    def get_checkpoint_layers(self, interval=None, memory_budget_ratio=None):
        """
        Возвращает set индексов слоёв для checkpointing.

        Args:
            interval: интервал для 'uniform' стратегии
            memory_budget_ratio: доля памяти для 'memory_budget' (0-1)
        """
        if self.strategy == 'uniform':
            if interval is None:
                interval = 2
            return {i for i in range(0, self.n_layers, interval)}

        elif self.strategy == 'sqrt':
            n_checkpoints = max(1, int(math.sqrt(self.n_layers)))
            step = self.n_layers / n_checkpoints
            return {int(i * step) for i in range(n_checkpoints)}

        elif self.strategy == 'memory_budget':
            if memory_budget_ratio is None:
                memory_budget_ratio = 0.5
            # Чем меньше бюджет, тем больше checkpoint'ов
            n_checkpoints = max(
                1,
                int(self.n_layers * (1.0 - memory_budget_ratio))
            )
            step = self.n_layers / n_checkpoints
            return {int(i * step) for i in range(n_checkpoints)}

    def should_checkpoint(self, layer_idx, **kwargs):
        """Проверяет, нужно ли checkpoint'ить данный слой."""
        return layer_idx in self.get_checkpoint_layers(**kwargs)

    def estimate_memory_savings(self):
        """Оценка экономии памяти."""
        checkpoint_layers = self.get_checkpoint_layers()
        n_checkpointed = len(checkpoint_layers)
        # Без checkpointing: O(N) activations
        # С checkpointing: O(sqrt(N)) activations + recompute cost
        ratio = n_checkpointed / max(self.n_layers, 1)
        return {
            'n_layers': self.n_layers,
            'n_checkpointed': n_checkpointed,
            'checkpoint_ratio': ratio,
            'estimated_memory_saved': f"{ratio * 100:.1f}%",
            'recompute_overhead': f"{ratio * 100:.1f}%",
        }

    def get_info(self):
        layers = self.get_checkpoint_layers()
        return {
            'strategy': self.strategy,
            'n_layers': self.n_layers,
            'checkpoint_layers': sorted(layers),
            'memory_savings': self.estimate_memory_savings(),
        }
