"""
v21 утилиты: FSDP Simulator, Grokking Detector, GradEMA, Mixture of Tokenizations, AGC.

FSDP Sharding Simulator: оценка экономии памяти при Fully Sharded Data Parallel.
Ref: Zhao et al., "PyTorch FSDP" (2023)

Grokking Detector: обнаружение delayed generalization по train/val метрикам.
Ref: Power et al., "Grokking: Generalization Beyond Overfitting" (2022)

GradEMA: экспоненциальное скользящее среднее градиентов для сглаживания.

Mixture of Tokenizations: ансамбль разных гранулярностей токенизации.
Ref: Limisiewicz et al., "Mixture of Tokenizers" (2024)

AGC (Adaptive Gradient Clipping): unit-wise clipping по норме весов.
Ref: Brock et al., "NFNet / AGC" (2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ==================== FSDP Sharding Simulator ====================

class FSDPSimulator:
    """
    Симулятор FSDP для оценки экономии памяти.

    Вычисляет, сколько памяти нужно при разном числе GPU
    с полным sharding параметров, градиентов и optimizer state.

    Args:
        model: nn.Module
    """
    def __init__(self, model):
        self.model = model
        self._param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        self._n_params = sum(p.numel() for p in model.parameters())

    def estimate_memory(self, n_gpus=1, optimizer_states=2,
                         activation_bytes=0, dtype_bytes=4):
        """
        Оценка памяти per GPU.

        Args:
            n_gpus: число GPU
            optimizer_states: множитель состояния optimizer (Adam = 2)
            activation_bytes: байт на активации (0 = не учитывать)
            dtype_bytes: байт на параметр

        Returns:
            dict: {params_mb, grads_mb, optimizer_mb, total_mb, savings_pct}
        """
        param_bytes = self._n_params * dtype_bytes
        grad_bytes = param_bytes  # gradients same size as params

        # FSDP: sharding across GPUs
        shard_param = param_bytes / n_gpus
        shard_grad = grad_bytes / n_gpus
        shard_opt = param_bytes * optimizer_states / n_gpus

        total = shard_param + shard_grad + shard_opt + activation_bytes
        no_shard = param_bytes + grad_bytes + param_bytes * optimizer_states + activation_bytes
        savings = (1 - total / no_shard) * 100 if no_shard > 0 else 0

        mb = 1024 * 1024
        return {
            'params_mb': shard_param / mb,
            'grads_mb': shard_grad / mb,
            'optimizer_mb': shard_opt / mb,
            'activation_mb': activation_bytes / mb,
            'total_mb': total / mb,
            'no_shard_mb': no_shard / mb,
            'savings_pct': savings,
            'n_gpus': n_gpus,
        }

    def scaling_report(self, gpu_counts=(1, 2, 4, 8)):
        """
        Отчёт о масштабировании для разного числа GPU.

        Returns:
            list[dict]: estimate_memory для каждого n_gpus
        """
        return [self.estimate_memory(n) for n in gpu_counts]

    @property
    def total_params(self):
        return self._n_params

    @property
    def total_param_mb(self):
        return self._param_bytes / (1024 * 1024)


# ==================== Grokking Detector ====================

class GrokkingDetector:
    """
    Детектор гроккинга: delayed generalization после overfitting.

    Отслеживает train/val loss и обнаруживает паттерн гроккинга:
    1. Train loss падает → low
    2. Val loss стагнирует → high
    3. Через много шагов val loss резко падает → grokking!

    Args:
        patience: шагов ожидания перед сигналом
        train_threshold: порог train loss для "overfit"
        gap_threshold: минимальный gap train/val для grokking candidate
    """
    def __init__(self, patience=100, train_threshold=0.1, gap_threshold=0.5):
        self.patience = patience
        self.train_threshold = train_threshold
        self.gap_threshold = gap_threshold
        self.train_losses = []
        self.val_losses = []
        self._overfit_start = None
        self._grokking_detected = False
        self._grokking_step = None

    def update(self, train_loss, val_loss):
        """
        Обновляет детектор.

        Args:
            train_loss: текущий train loss
            val_loss: текущий val loss

        Returns:
            dict: {phase, gap, overfit_steps, grokking_detected}
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        step = len(self.train_losses)

        gap = val_loss - train_loss

        # Определяем фазу
        phase = 'learning'
        overfit_steps = 0

        if train_loss < self.train_threshold and gap > self.gap_threshold:
            if self._overfit_start is None:
                self._overfit_start = step
            overfit_steps = step - self._overfit_start
            phase = 'overfitting'
        elif train_loss < self.train_threshold and gap <= self.gap_threshold:
            if self._overfit_start is not None:
                # Was overfitting, now generalizing → grokking!
                if not self._grokking_detected:
                    self._grokking_detected = True
                    self._grokking_step = step
                phase = 'grokking'
            else:
                phase = 'generalizing'
            self._overfit_start = None
        else:
            self._overfit_start = None

        return {
            'phase': phase,
            'gap': gap,
            'overfit_steps': overfit_steps,
            'grokking_detected': self._grokking_detected,
            'grokking_step': self._grokking_step,
            'step': step,
        }

    def is_grokking(self):
        """Был ли обнаружен гроккинг."""
        return self._grokking_detected

    def get_summary(self):
        """Сводка."""
        return {
            'total_steps': len(self.train_losses),
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'grokking_detected': self._grokking_detected,
            'grokking_step': self._grokking_step,
            'min_train_loss': min(self.train_losses) if self.train_losses else None,
            'min_val_loss': min(self.val_losses) if self.val_losses else None,
        }

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self._overfit_start = None
        self._grokking_detected = False
        self._grokking_step = None


# ==================== GradEMA ====================

class GradEMA:
    """
    Exponential Moving Average градиентов.

    Сглаживает шумные градиенты через EMA, уменьшая variance.
    Может использоваться как альтернатива momentum.

    Args:
        model: nn.Module
        decay: EMA decay (0.9-0.99)
    """
    def __init__(self, model, decay=0.95):
        self.model = model
        self.decay = decay
        self.shadow_grads = {}
        self._initialized = False

    def update(self):
        """Обновляет EMA градиентов и заменяет .grad."""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            if name not in self.shadow_grads:
                self.shadow_grads[name] = p.grad.clone()
            else:
                self.shadow_grads[name].mul_(self.decay).add_(
                    p.grad, alpha=1 - self.decay
                )

            # Заменяем gradient на EMA
            p.grad.copy_(self.shadow_grads[name])

        self._initialized = True

    def get_grad_stats(self):
        """Статистика EMA градиентов."""
        if not self.shadow_grads:
            return {'n_params': 0, 'avg_norm': 0.0}

        norms = [g.norm().item() for g in self.shadow_grads.values()]
        return {
            'n_params': len(norms),
            'avg_norm': sum(norms) / len(norms),
            'max_norm': max(norms),
            'min_norm': min(norms),
        }

    def reset(self):
        self.shadow_grads = {}
        self._initialized = False


# ==================== Mixture of Tokenizations ====================

class MixtureOfTokenizations:
    """
    Mixture of Tokenizations: ансамбль разных токенизаций.

    Создаёт несколько вариантов tokenization одного текста
    (char, subword, byte) и объединяет результаты.

    Args:
        tokenizers: list[tokenizer] — каждый с encode/decode
        weights: list[float] — веса для каждого tokenizer (None = равные)
    """
    def __init__(self, tokenizers, weights=None):
        self.tokenizers = tokenizers
        self.n = len(tokenizers)
        if weights is None:
            self.weights = [1.0 / self.n] * self.n
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def encode_all(self, text):
        """
        Encode через все tokenizers.

        Returns:
            list[list[int]]: ids от каждого tokenizer
        """
        return [tok.encode(text) for tok in self.tokenizers]

    def select_by_length(self, text, prefer='shortest'):
        """
        Выбирает tokenization по длине.

        Args:
            text: входной текст
            prefer: 'shortest', 'longest', 'median'

        Returns:
            (ids, tokenizer_idx)
        """
        all_ids = self.encode_all(text)
        lengths = [len(ids) for ids in all_ids]

        if prefer == 'shortest':
            idx = lengths.index(min(lengths))
        elif prefer == 'longest':
            idx = lengths.index(max(lengths))
        else:  # median
            sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
            idx = sorted_idx[len(sorted_idx) // 2]

        return all_ids[idx], idx

    def encode_stochastic(self, text):
        """
        Случайный выбор tokenizer по весам.

        Returns:
            (ids, tokenizer_idx)
        """
        import random
        idx = random.choices(range(self.n), weights=self.weights, k=1)[0]
        return self.tokenizers[idx].encode(text), idx

    def get_compression_stats(self, text):
        """
        Статистика сжатия для каждого tokenizer.

        Returns:
            list[dict]: {tokenizer_idx, n_tokens, compression_ratio}
        """
        char_len = len(text)
        stats = []
        for i, tok in enumerate(self.tokenizers):
            ids = tok.encode(text)
            stats.append({
                'tokenizer_idx': i,
                'n_tokens': len(ids),
                'compression_ratio': char_len / max(1, len(ids)),
            })
        return stats


# ==================== Adaptive Gradient Clipping (AGC) ====================

class AGC:
    """
    Adaptive Gradient Clipping (AGC).

    Clipping per-unit (per-neuron) по соотношению
    ||grad|| / ||weight||. Более стабильно, чем global clipping.

    Args:
        model: nn.Module
        clip_factor: максимальное соотношение grad/weight norm
        eps: epsilon для стабильности
    """
    def __init__(self, model, clip_factor=0.01, eps=1e-3):
        self.model = model
        self.clip_factor = clip_factor
        self.eps = eps
        self._clip_stats = defaultdict(list)

    @torch.no_grad()
    def clip(self):
        """
        Применяет AGC ко всем параметрам.

        Returns:
            dict: {n_clipped, n_total, clip_ratio}
        """
        n_clipped = 0
        n_total = 0

        for name, p in self.model.named_parameters():
            if p.grad is None or p.dim() < 2:
                continue

            n_total += 1

            # Per-unit norms (each row = one unit/neuron)
            w_norm = p.norm(dim=tuple(range(1, p.dim())), keepdim=True)
            g_norm = p.grad.norm(dim=tuple(range(1, p.dim())), keepdim=True)

            # Clip factor: ||g|| ≤ clip_factor * ||w||
            max_norm = self.clip_factor * w_norm.clamp(min=self.eps)
            clip_mask = g_norm > max_norm

            if clip_mask.any():
                n_clipped += 1
                scale = max_norm / g_norm.clamp(min=1e-10)
                scale = torch.where(clip_mask, scale, torch.ones_like(scale))
                p.grad.mul_(scale)

            self._clip_stats[name].append(clip_mask.float().mean().item())

        return {
            'n_clipped': n_clipped,
            'n_total': n_total,
            'clip_ratio': n_clipped / max(1, n_total),
        }

    def get_stats(self):
        """Статистика clipping per layer."""
        stats = {}
        for name, ratios in self._clip_stats.items():
            stats[name] = {
                'avg_clip_ratio': sum(ratios) / len(ratios),
                'max_clip_ratio': max(ratios),
                'n_steps': len(ratios),
            }
        return stats

    def reset_stats(self):
        self._clip_stats.clear()
