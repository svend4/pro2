"""
v25 утилиты: Gradient Centralization, Token Mixing MLP, LR Finder,
Parameter Efficiency Tracker, Batch Size Finder.

Gradient Centralization: центрирование градиентов для ускорения сходимости.
Ref: Yong et al., "Gradient Centralization" (2020)

Token Mixing MLP: лёгкая альтернатива attention через MLP-mixing.
Ref: Tolstikhin et al., "MLP-Mixer" (2021)

LR Finder: автоматический поиск оптимального learning rate.
Ref: Smith, "Cyclical Learning Rates" (2017)

Parameter Efficiency Tracker: анализ params/FLOPs/memory.

Batch Size Finder: поиск оптимального batch size для GPU.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ==================== Gradient Centralization ====================

class GradientCentralization:
    """
    Gradient Centralization.

    Центрирует градиенты (вычитает среднее по output-dim),
    что улучшает условие оптимизации и ускоряет сходимость.

    Применяется только к весам с ndim >= 2 (не bias, не LayerNorm).

    Args:
        model: nn.Module
        gc_conv_only: применять только к conv/linear (True) или ко всем (False)
    """
    def __init__(self, model, gc_conv_only=False):
        self.model = model
        self.gc_conv_only = gc_conv_only

    def centralize(self):
        """
        Центрирует градиенты in-place.

        Returns:
            dict: {n_centralized, n_skipped}
        """
        n_centralized = 0
        n_skipped = 0

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue

                if p.grad.ndim < 2:
                    n_skipped += 1
                    continue

                if self.gc_conv_only and 'weight' not in name:
                    n_skipped += 1
                    continue

                # Centralize: subtract mean over all dims except first
                dims = list(range(1, p.grad.ndim))
                p.grad.sub_(p.grad.mean(dim=dims, keepdim=True))
                n_centralized += 1

        return {
            'n_centralized': n_centralized,
            'n_skipped': n_skipped,
        }


# ==================== Token Mixing MLP ====================

class TokenMixingMLP(nn.Module):
    """
    Token Mixing MLP (MLP-Mixer style).

    Заменяет attention: миксирует информацию между позициями
    через MLP по token dimension.

    Args:
        seq_len: максимальная длина последовательности
        d_model: размерность модели
        expansion: expansion factor для hidden dim
        dropout: dropout rate
    """
    def __init__(self, seq_len, d_model, expansion=2, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        hidden = int(seq_len * expansion)

        self.norm = nn.LayerNorm(d_model)
        # Token mixing: (B, D, T) → MLP → (B, D, T)
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D)

        Returns:
            (B, T, D)
        """
        residual = x
        x = self.norm(x)
        # Transpose to (B, D, T) for token mixing
        x = x.transpose(1, 2)  # (B, D, T)
        T = x.size(-1)
        if T < self.seq_len:
            # Pad
            x = F.pad(x, (0, self.seq_len - T))
        x = self.token_mix(x)
        if T < self.seq_len:
            x = x[:, :, :T]
        x = x.transpose(1, 2)  # (B, T, D)
        return residual + x


# ==================== Learning Rate Finder ====================

class LRFinder:
    """
    Learning Rate Range Test.

    Постепенно увеличивает LR от min_lr до max_lr,
    записывая loss. Оптимальный LR — где loss падает быстрее всего.

    Args:
        model: nn.Module
        optimizer: optimizer
        min_lr: начальный LR
        max_lr: максимальный LR
        n_steps: число шагов теста
    """
    def __init__(self, model, optimizer, min_lr=1e-7, max_lr=1.0, n_steps=100):
        self.model = model
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_steps = n_steps
        self.history = []
        self._initial_state = None

    def _save_state(self):
        self._initial_state = {
            'model': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
        }

    def _restore_state(self):
        if self._initial_state:
            self.model.load_state_dict(self._initial_state['model'])
            self.optimizer.load_state_dict(self._initial_state['optimizer'])

    def run(self, loss_fn, smooth_factor=0.05):
        """
        Запускает LR range test.

        Args:
            loss_fn: callable() → loss (одна итерация)
            smooth_factor: smoothing для EMA loss

        Returns:
            list[dict]: [{lr, loss, smoothed_loss}, ...]
        """
        self._save_state()
        self.history = []

        mult = (self.max_lr / self.min_lr) ** (1.0 / max(self.n_steps - 1, 1))
        lr = self.min_lr
        best_loss = float('inf')
        smoothed = None

        for step in range(self.n_steps):
            # Set LR
            for group in self.optimizer.param_groups:
                group['lr'] = lr

            # Forward + backward
            self.optimizer.zero_grad()
            loss = loss_fn()
            loss_val = loss.item()
            loss.backward()
            self.optimizer.step()

            # Smoothed loss
            if smoothed is None:
                smoothed = loss_val
            else:
                smoothed = smoothed * (1 - smooth_factor) + loss_val * smooth_factor

            self.history.append({
                'lr': lr,
                'loss': loss_val,
                'smoothed_loss': smoothed,
                'step': step,
            })

            # Stop if loss explodes
            if smoothed > 4 * best_loss and step > 10:
                break
            best_loss = min(best_loss, smoothed)

            lr *= mult

        self._restore_state()
        return self.history

    def suggest_lr(self):
        """
        Рекомендует LR на основе результатов.

        Returns:
            dict: {suggested_lr, min_loss_lr, steepest_lr}
        """
        if not self.history:
            return None

        # Min loss LR
        min_entry = min(self.history, key=lambda x: x['smoothed_loss'])

        # Steepest descent (max negative gradient)
        steepest_lr = min_entry['lr']
        max_gradient = 0
        for i in range(1, len(self.history)):
            grad = (self.history[i - 1]['smoothed_loss'] - self.history[i]['smoothed_loss'])
            if grad > max_gradient:
                max_gradient = grad
                steepest_lr = self.history[i]['lr']

        # Suggestion: 1 order below steepest
        suggested = steepest_lr / 10.0

        return {
            'suggested_lr': suggested,
            'min_loss_lr': min_entry['lr'],
            'steepest_lr': steepest_lr,
        }


# ==================== Parameter Efficiency Tracker ====================

class ParamEfficiencyTracker:
    """
    Трекер эффективности параметров модели.

    Анализирует params/FLOPs/memory для каждого слоя.
    Полезно для оптимизации архитектуры.

    Args:
        model: nn.Module
    """
    def __init__(self, model):
        self.model = model
        self._analysis = None

    def analyze(self):
        """
        Полный анализ модели.

        Returns:
            dict: {total_params, trainable_params, layers: [...]}
        """
        total = 0
        trainable = 0
        layers = []

        for name, module in self.model.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            train_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

            if params > 0:
                mem_bytes = sum(
                    p.numel() * p.element_size()
                    for p in module.parameters(recurse=False)
                )
                layers.append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params,
                    'trainable': train_params,
                    'memory_mb': mem_bytes / (1024 * 1024),
                    'frozen_pct': (1 - train_params / max(params, 1)) * 100,
                })

            total += params
            trainable += train_params

        self._analysis = {
            'total_params': total,
            'trainable_params': trainable,
            'frozen_params': total - trainable,
            'trainable_pct': trainable / max(total, 1) * 100,
            'total_memory_mb': sum(l['memory_mb'] for l in layers),
            'n_layers': len(layers),
            'layers': layers,
        }
        return self._analysis

    def get_top_layers(self, k=5, by='params'):
        """Top-k слоёв по params или memory."""
        if self._analysis is None:
            self.analyze()
        return sorted(self._analysis['layers'], key=lambda x: -x[by])[:k]

    def get_efficiency_ratio(self):
        """Отношение trainable/total params."""
        if self._analysis is None:
            self.analyze()
        return self._analysis['trainable_pct']

    def estimate_flops(self, input_shape):
        """
        Грубая оценка FLOPs для Linear и Conv слоёв.

        Args:
            input_shape: (B, T, D) или (B, C, H, W)

        Returns:
            dict: {total_flops, per_layer: [...]}
        """
        total_flops = 0
        per_layer = []

        for name, module in self.model.named_modules():
            flops = 0
            if isinstance(module, nn.Linear):
                # FLOPs ≈ 2 * in * out * batch_tokens
                batch_tokens = 1
                for d in input_shape[:-1]:
                    batch_tokens *= d
                flops = 2 * module.in_features * module.out_features * batch_tokens
            elif isinstance(module, nn.Conv1d):
                batch_tokens = input_shape[0] * input_shape[-1] if len(input_shape) > 2 else input_shape[0]
                flops = 2 * module.in_channels * module.out_channels * module.kernel_size[0] * batch_tokens

            if flops > 0:
                per_layer.append({
                    'name': name,
                    'flops': flops,
                    'gflops': flops / 1e9,
                })
                total_flops += flops

        return {
            'total_flops': total_flops,
            'total_gflops': total_flops / 1e9,
            'per_layer': per_layer,
        }


# ==================== Batch Size Finder ====================

class BatchSizeFinder:
    """
    Поиск оптимального batch size.

    Пробует batch sizes от min до max (степени двойки),
    измеряя память и throughput.

    Args:
        model: nn.Module
        max_batch_size: максимальный batch size
        seq_len: длина последовательности
        vocab_size: размер словаря
    """
    def __init__(self, model, max_batch_size=256, seq_len=128, vocab_size=2048):
        self.model = model
        self.max_batch_size = max_batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.results = []

    def _try_batch_size(self, batch_size):
        """Пробует один batch size."""
        try:
            x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))

            # Move to same device as model
            device = next(self.model.parameters()).device
            x = x.to(device)

            self.model.train()
            with torch.no_grad():
                output = self.model(x)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

            # Estimate memory per sample
            param_mem = sum(p.numel() * p.element_size() for p in self.model.parameters())
            act_mem_estimate = logits.numel() * logits.element_size()

            return {
                'batch_size': batch_size,
                'success': True,
                'param_memory_mb': param_mem / (1024 * 1024),
                'activation_memory_mb': act_mem_estimate / (1024 * 1024),
                'output_shape': list(logits.shape),
            }
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return {
                    'batch_size': batch_size,
                    'success': False,
                    'error': str(e)[:100],
                }
            raise

    def find(self):
        """
        Ищет максимальный batch size (степени двойки).

        Returns:
            dict: {max_batch_size, results: [...]}
        """
        self.results = []
        bs = 1
        max_working = 0

        while bs <= self.max_batch_size:
            result = self._try_batch_size(bs)
            self.results.append(result)

            if result['success']:
                max_working = bs
                bs *= 2
            else:
                break

        return {
            'max_batch_size': max_working,
            'recommended_batch_size': max(1, max_working // 2),  # 50% margin
            'results': self.results,
        }

    def estimate_throughput(self, batch_size, n_steps=5):
        """
        Оценивает throughput (tokens/sec) для batch size.

        Args:
            batch_size: размер батча
            n_steps: число шагов для замера

        Returns:
            dict: {tokens_per_sec, samples_per_sec}
        """
        import time
        device = next(self.model.parameters()).device
        self.model.eval()

        times = []
        with torch.no_grad():
            for _ in range(n_steps):
                x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len)).to(device)
                start = time.perf_counter()
                self.model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

        avg_time = sum(times) / len(times)
        tokens = batch_size * self.seq_len

        return {
            'batch_size': batch_size,
            'avg_time_ms': avg_time * 1000,
            'tokens_per_sec': tokens / avg_time,
            'samples_per_sec': batch_size / avg_time,
        }
