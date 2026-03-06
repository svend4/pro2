"""
v22 утилиты: Matryoshka Embeddings, GC Profiler, Reptile, Token Frequency, SWA.

Matryoshka Embeddings: вложенные представления на разных размерностях.
Ref: Kusupati et al., "Matryoshka Representation Learning" (2022)

Gradient Checkpointing Profiler: анализ memory/compute tradeoff.

Reptile Meta-Learning: простой meta-learning через weight interpolation.
Ref: Nichol et al., "On First-Order Meta-Learning Algorithms" (2018)

Token Frequency Tracker: мониторинг распределения токенов при обучении.

SWA (Stochastic Weight Averaging): усреднение весов для плоских минимумов.
Ref: Izmailov et al., "Averaging Weights Leads to Wider Optima" (2018)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter


# ==================== Matryoshka Embeddings ====================

class MatryoshkaHead(nn.Module):
    """
    Matryoshka Representation Learning.

    Обучает модель так, чтобы первые d' < d измерений embedding
    были полезны сами по себе. Позволяет гибко выбирать размерность
    при inference (trade-off quality/speed).

    Args:
        d_model: полная размерность
        vocab_size: размер словаря
        dims: list[int] — вложенные размерности (e.g., [64, 128, 256, 512])
    """
    def __init__(self, d_model, vocab_size, dims=None):
        super().__init__()
        self.d_model = d_model
        if dims is None:
            # Степени двойки до d_model
            dims = []
            d = 64
            while d <= d_model:
                dims.append(d)
                d *= 2
            if d_model not in dims:
                dims.append(d_model)
        self.dims = sorted(dims)

        # Отдельная head для каждой размерности
        self.heads = nn.ModuleDict({
            str(d): nn.Linear(d, vocab_size, bias=False)
            for d in self.dims
        })

    def forward(self, hidden, dim=None):
        """
        Args:
            hidden: (B, T, D) — full hidden states
            dim: конкретная размерность (None = все)

        Returns:
            dict[int, Tensor] или Tensor: logits для каждой/одной размерности
        """
        if dim is not None:
            return self.heads[str(dim)](hidden[:, :, :dim])

        results = {}
        for d in self.dims:
            results[d] = self.heads[str(d)](hidden[:, :, :d])
        return results

    def compute_loss(self, hidden, targets):
        """
        Matryoshka loss: средний CE по всем размерностям.

        Args:
            hidden: (B, T, D)
            targets: (B, T)

        Returns:
            total_loss, per_dim_losses: dict[int, float]
        """
        total = 0.0
        per_dim = {}
        for d in self.dims:
            logits = self.heads[str(d)](hidden[:, :, :d])
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
            total = total + loss
            per_dim[d] = loss.item()

        total = total / len(self.dims)
        return total, per_dim


# ==================== Gradient Checkpointing Profiler ====================

class GCProfiler:
    """
    Gradient Checkpointing Profiler.

    Измеряет memory/compute tradeoff при разных стратегиях
    gradient checkpointing: none, every-N, selective.

    Args:
        model: nn.Module
    """
    def __init__(self, model):
        self.model = model
        self.results = []

    def profile_layer(self, layer, x, name='layer'):
        """
        Профилирует один слой: memory и compute.

        Args:
            layer: nn.Module
            x: входной tensor
            name: имя для логирования

        Returns:
            dict: {name, forward_mem, backward_mem, activation_size}
        """
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        # Measure activation size
        x_clone = x.detach().requires_grad_(True)
        with torch.no_grad():
            out = layer(x_clone) if not isinstance(layer(x_clone), tuple) else layer(x_clone)

        if isinstance(out, tuple):
            act_size = sum(o.numel() * o.element_size() for o in out if isinstance(o, torch.Tensor))
        else:
            act_size = out.numel() * out.element_size()

        result = {
            'name': name,
            'activation_bytes': act_size,
            'activation_mb': act_size / (1024 * 1024),
            'input_shape': list(x.shape),
            'param_count': sum(p.numel() for p in layer.parameters()),
        }
        self.results.append(result)
        return result

    def estimate_savings(self, n_layers, activation_mb_per_layer):
        """
        Оценивает экономию памяти при checkpointing.

        Args:
            n_layers: число слоёв
            activation_mb_per_layer: MB на активации одного слоя

        Returns:
            dict: стратегии и их потребление памяти
        """
        no_ckpt = n_layers * activation_mb_per_layer
        full_ckpt = 2 * activation_mb_per_layer  # только 2 слоя в памяти
        sqrt_ckpt = math.sqrt(n_layers) * activation_mb_per_layer

        return {
            'no_checkpoint_mb': no_ckpt,
            'full_checkpoint_mb': full_ckpt,
            'sqrt_checkpoint_mb': sqrt_ckpt,
            'full_savings_pct': (1 - full_ckpt / no_ckpt) * 100,
            'sqrt_savings_pct': (1 - sqrt_ckpt / no_ckpt) * 100,
            'full_recompute_overhead': '~33%',
            'n_layers': n_layers,
        }

    def get_report(self):
        """Возвращает все результаты профилирования."""
        return list(self.results)

    def reset(self):
        self.results = []


# ==================== Reptile Meta-Learning ====================

class Reptile:
    """
    Reptile meta-learning: простой first-order meta-learning.

    Алгоритм:
    1. Для каждой задачи: сделать K шагов SGD → получить θ'
    2. Meta-update: θ ← θ + ε(θ' - θ)

    Проще MAML, не требует second-order gradients.

    Args:
        model: nn.Module
        inner_lr: learning rate для inner loop
        meta_lr: learning rate для meta update
        inner_steps: шагов inner loop на задачу
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.1, inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

    def inner_loop(self, task_loss_fn):
        """
        Inner loop: K шагов на одной задаче.

        Args:
            task_loss_fn: callable(model) → loss

        Returns:
            dict: {inner_losses, final_state}
        """
        # Clone model
        fast_model = copy.deepcopy(self.model)
        opt = torch.optim.SGD(fast_model.parameters(), lr=self.inner_lr)

        losses = []
        for _ in range(self.inner_steps):
            opt.zero_grad()
            loss = task_loss_fn(fast_model)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        return {
            'inner_losses': losses,
            'final_state': fast_model.state_dict(),
        }

    def meta_step(self, task_loss_fns):
        """
        Meta step: inner loop на каждой задаче, затем meta update.

        Args:
            task_loss_fns: list[callable(model) → loss]

        Returns:
            dict: {avg_final_loss, n_tasks}
        """
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Accumulate updates from all tasks
        accumulated_diff = {k: torch.zeros_like(v) for k, v in original_state.items()}
        avg_loss = 0.0

        for task_fn in task_loss_fns:
            result = self.inner_loop(task_fn)
            final_state = result['final_state']
            avg_loss += result['inner_losses'][-1]

            for k in accumulated_diff:
                accumulated_diff[k] += (final_state[k] - original_state[k])

        n_tasks = len(task_loss_fns)

        # Meta update: θ ← θ + meta_lr * avg(θ' - θ)
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                v.add_(self.meta_lr / n_tasks * accumulated_diff[k])

        return {
            'avg_final_loss': avg_loss / n_tasks,
            'n_tasks': n_tasks,
        }


# ==================== Token Frequency Tracker ====================

class TokenFrequencyTracker:
    """
    Трекер частоты токенов при обучении.

    Мониторит распределение токенов в входных и целевых
    последовательностях. Полезно для анализа data balance.

    Args:
        vocab_size: размер словаря
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.input_counts = Counter()
        self.target_counts = Counter()
        self.total_input_tokens = 0
        self.total_target_tokens = 0
        self._n_batches = 0

    def update(self, input_ids, target_ids=None):
        """
        Обновляет счётчики.

        Args:
            input_ids: (B, T) tensor
            target_ids: (B, T) tensor (опционально)
        """
        self._n_batches += 1

        # Input counts
        ids = input_ids.flatten().tolist()
        self.input_counts.update(ids)
        self.total_input_tokens += len(ids)

        if target_ids is not None:
            tids = target_ids.flatten().tolist()
            self.target_counts.update(tids)
            self.total_target_tokens += len(tids)

    def get_coverage(self):
        """Доля словаря, встреченная хотя бы раз."""
        seen = len(self.input_counts)
        return seen / self.vocab_size

    def get_top_k(self, k=10, source='input'):
        """
        Top-k самых частых токенов.

        Returns:
            list[(token_id, count, frequency)]
        """
        counts = self.input_counts if source == 'input' else self.target_counts
        total = self.total_input_tokens if source == 'input' else self.total_target_tokens
        total = max(1, total)

        return [
            (tok, cnt, cnt / total)
            for tok, cnt in counts.most_common(k)
        ]

    def get_entropy(self, source='input'):
        """Entropy распределения токенов (bits)."""
        counts = self.input_counts if source == 'input' else self.target_counts
        total = self.total_input_tokens if source == 'input' else self.total_target_tokens
        if total == 0:
            return 0.0

        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_stats(self):
        """Сводная статистика."""
        return {
            'n_batches': self._n_batches,
            'total_input_tokens': self.total_input_tokens,
            'total_target_tokens': self.total_target_tokens,
            'unique_input_tokens': len(self.input_counts),
            'unique_target_tokens': len(self.target_counts),
            'coverage': self.get_coverage(),
            'input_entropy': self.get_entropy('input'),
        }

    def reset(self):
        self.input_counts.clear()
        self.target_counts.clear()
        self.total_input_tokens = 0
        self.total_target_tokens = 0
        self._n_batches = 0


# ==================== Stochastic Weight Averaging (SWA) ====================

class SWA:
    """
    Stochastic Weight Averaging.

    Усредняет веса модели по нескольким checkpoint-ам
    при циклическом LR, находя более плоские минимумы.

    Args:
        model: nn.Module
        swa_start: шаг начала SWA
        swa_freq: частота обновления SWA
    """
    def __init__(self, model, swa_start=0, swa_freq=1):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_state = None
        self.n_averaged = 0
        self._step = 0

    def step(self):
        """Один шаг SWA."""
        self._step += 1
        if self._step < self.swa_start:
            return False

        if (self._step - self.swa_start) % self.swa_freq != 0:
            return False

        self._update_average()
        return True

    def _update_average(self):
        """Обновляет running average."""
        if self.swa_state is None:
            self.swa_state = {
                k: v.clone().float()
                for k, v in self.model.state_dict().items()
            }
            self.n_averaged = 1
        else:
            self.n_averaged += 1
            for k, v in self.model.state_dict().items():
                self.swa_state[k] += (v.float() - self.swa_state[k]) / self.n_averaged

    def apply_average(self):
        """Применяет усреднённые веса к модели."""
        if self.swa_state is None:
            return
        state = {k: v.to(dtype=self.model.state_dict()[k].dtype)
                 for k, v in self.swa_state.items()}
        self.model.load_state_dict(state)

    def get_swa_state(self):
        """Возвращает SWA state dict."""
        return self.swa_state

    def update_bn(self, data_loader_fn, n_batches=10):
        """
        Обновляет BatchNorm статистику после SWA.

        Args:
            data_loader_fn: callable() → (input_ids,)
            n_batches: число батчей для пересчёта
        """
        self.model.train()
        with torch.no_grad():
            for _ in range(n_batches):
                x = data_loader_fn()
                if isinstance(x, tuple):
                    self.model(x[0])
                else:
                    self.model(x)

    def state_dict(self):
        return {
            'swa_state': self.swa_state,
            'n_averaged': self.n_averaged,
            'step': self._step,
        }

    def load_state_dict(self, state):
        self.swa_state = state['swa_state']
        self.n_averaged = state['n_averaged']
        self._step = state['step']
