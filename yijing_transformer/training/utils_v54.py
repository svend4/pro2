"""
v54 утилиты: Quantization-Aware Training, Structured Pruning Scheduler,
Online Distillation Loss, Model FLOPs Profiler, Activation Memory Estimator.

Quantization-Aware Training (QAT): fake-quantization при обучении.
Ref: Jacob et al., "Quantization and Training of Neural Networks" (2018)

Structured Pruning Scheduler: постепенное прореживание нейронов/головок.
Ref: Zhu & Gupta, "To Prune, or Not to Prune" (2018)

Online Distillation Loss: distillation без отдельной teacher модели.
Ref: Zhang et al., "Deep Mutual Learning" (2018)

Model FLOPs Profiler: оценка вычислительной стоимости модели.
Ref: Standard profiling technique.

Activation Memory Estimator: оценка памяти для активаций.
Ref: Korthikanti et al., "Reducing Activation Recomputation" (2022)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Quantization-Aware Training ====================

class FakeQuantize(nn.Module):
    """
    Fake Quantization для QAT.

    Имитирует квантизацию (round + clamp) в forward pass,
    но сохраняет float градиенты через straight-through estimator.

    Args:
        bits: число бит квантизации
        symmetric: симметричная квантизация
        per_channel: per-channel или per-tensor
    """
    def __init__(self, bits=8, symmetric=True, per_channel=False):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        if symmetric:
            self.qmin = -(2 ** (bits - 1))
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        # EMA для scale/zero_point
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.ema_decay = 0.999

    def update_stats(self, x):
        """Обновление min/max статистик через EMA."""
        with torch.no_grad():
            cur_min = x.min().item()
            cur_max = x.max().item()
            if self.min_val.item() == float('inf'):
                self.min_val.fill_(cur_min)
                self.max_val.fill_(cur_max)
            else:
                self.min_val.mul_(self.ema_decay).add_(
                    cur_min * (1 - self.ema_decay)
                )
                self.max_val.mul_(self.ema_decay).add_(
                    cur_max * (1 - self.ema_decay)
                )

    def compute_scale_zp(self):
        """Вычисление scale и zero_point."""
        if self.symmetric:
            abs_max = max(abs(self.min_val.item()), abs(self.max_val.item()))
            abs_max = max(abs_max, 1e-8)
            scale = abs_max / self.qmax
            zero_point = 0.0
        else:
            val_range = max(
                self.max_val.item() - self.min_val.item(), 1e-8
            )
            scale = val_range / (self.qmax - self.qmin)
            zero_point = self.qmin - self.min_val.item() / scale
            zero_point = round(max(self.qmin, min(self.qmax, zero_point)))
        return scale, zero_point

    def forward(self, x):
        if self.training:
            self.update_stats(x)

        scale, zero_point = self.compute_scale_zp()

        # Fake quantize: quantize then dequantize
        x_q = torch.clamp(
            torch.round(x / scale + zero_point),
            self.qmin, self.qmax
        )
        x_dq = (x_q - zero_point) * scale

        # Straight-through estimator
        if self.training:
            return x + (x_dq - x).detach()
        return x_dq

    def get_info(self):
        scale, zp = self.compute_scale_zp()
        return {
            'bits': self.bits,
            'symmetric': self.symmetric,
            'range': [self.qmin, self.qmax],
            'scale': scale,
            'zero_point': zp,
        }


# ==================== Structured Pruning Scheduler ====================

class StructuredPruningScheduler:
    """
    Планировщик структурного прореживания.

    Постепенно увеличивает sparsity от 0 до target по кубической кривой.
    Может прореживать attention heads или FFN нейроны.

    Args:
        total_steps: общее число шагов обучения
        target_sparsity: целевая разреженность (0-1)
        warmup_fraction: доля шагов без pruning
        cooldown_fraction: доля шагов с фиксированной sparsity
    """
    def __init__(self, total_steps, target_sparsity=0.5,
                 warmup_fraction=0.1, cooldown_fraction=0.1):
        self.total_steps = total_steps
        self.target_sparsity = target_sparsity
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.cooldown_steps = int(total_steps * cooldown_fraction)
        self.pruning_steps = total_steps - self.warmup_steps - self.cooldown_steps
        self.current_step = 0

    def get_sparsity(self, step=None):
        """Текущая целевая sparsity."""
        if step is None:
            step = self.current_step

        if step < self.warmup_steps:
            return 0.0
        elif step >= self.total_steps - self.cooldown_steps:
            return self.target_sparsity
        else:
            # Cubic schedule
            progress = (step - self.warmup_steps) / max(self.pruning_steps, 1)
            progress = min(progress, 1.0)
            return self.target_sparsity * (1 - (1 - progress) ** 3)

    def step(self):
        sparsity = self.get_sparsity()
        self.current_step += 1
        return sparsity

    def compute_mask(self, weight, sparsity=None):
        """
        Создаёт structured mask для весов.

        Прореживает целые выходные нейроны (строки).
        """
        if sparsity is None:
            sparsity = self.get_sparsity()
        if sparsity == 0.0:
            return torch.ones(weight.shape[0], device=weight.device)

        # L2 norm по выходным нейронам
        importance = weight.data.norm(2, dim=tuple(range(1, weight.dim())))
        n_prune = int(weight.shape[0] * sparsity)
        n_prune = min(n_prune, weight.shape[0] - 1)

        if n_prune == 0:
            return torch.ones(weight.shape[0], device=weight.device)

        threshold = importance.kthvalue(n_prune).values
        mask = (importance > threshold).float()
        return mask

    def get_info(self):
        return {
            'current_step': self.current_step,
            'current_sparsity': self.get_sparsity(),
            'target_sparsity': self.target_sparsity,
            'phase': ('warmup' if self.current_step < self.warmup_steps
                      else 'cooldown' if self.current_step >= self.total_steps - self.cooldown_steps
                      else 'pruning'),
        }


# ==================== Online Distillation Loss ====================

class OnlineDistillationLoss(nn.Module):
    """
    Online (Mutual) Distillation Loss.

    Два peer модели учат друг друга через KL-divergence.
    Не требует отдельной teacher модели.

    Args:
        temperature: температура для softmax
        alpha: баланс между CE и KD loss
    """
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits_1, logits_2, targets):
        """
        Args:
            logits_1: логиты модели 1
            logits_2: логиты модели 2
            targets: ground truth labels
        Returns:
            loss_1, loss_2: losses для каждой модели
        """
        # CE losses
        ce_1 = F.cross_entropy(logits_1, targets)
        ce_2 = F.cross_entropy(logits_2, targets)

        # KD losses (each learns from the other)
        T = self.temperature
        soft_1 = F.log_softmax(logits_1 / T, dim=-1)
        soft_2 = F.log_softmax(logits_2 / T, dim=-1)
        prob_1 = F.softmax(logits_1.detach() / T, dim=-1)
        prob_2 = F.softmax(logits_2.detach() / T, dim=-1)

        kd_1 = F.kl_div(soft_1, prob_2, reduction='batchmean') * (T ** 2)
        kd_2 = F.kl_div(soft_2, prob_1, reduction='batchmean') * (T ** 2)

        # Combined
        loss_1 = (1 - self.alpha) * ce_1 + self.alpha * kd_1
        loss_2 = (1 - self.alpha) * ce_2 + self.alpha * kd_2

        return loss_1, loss_2

    def get_info(self):
        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
        }


# ==================== Model FLOPs Profiler ====================

class ModelFLOPsProfiler:
    """
    Оценка FLOPs для transformer-моделей.

    Подсчитывает FLOPs для linear, attention, LayerNorm, embedding слоёв.

    Args:
        model: PyTorch модель
    """
    def __init__(self, model):
        self.model = model
        self._flops = {}

    def _count_linear(self, module, name, input_shape):
        """FLOPs для nn.Linear: 2 * in * out (multiply-add)."""
        batch = input_shape[0] if len(input_shape) > 0 else 1
        seq = input_shape[1] if len(input_shape) > 2 else 1
        flops = 2 * module.in_features * module.out_features * batch * seq
        if module.bias is not None:
            flops += module.out_features * batch * seq
        self._flops[name] = flops

    def _count_layernorm(self, module, name, input_shape):
        """FLOPs для LayerNorm: ~5 * normalized_shape."""
        numel = 1
        for s in module.normalized_shape:
            numel *= s
        batch = input_shape[0] if len(input_shape) > 0 else 1
        seq = input_shape[1] if len(input_shape) > 2 else 1
        self._flops[name] = 5 * numel * batch * seq

    def _count_embedding(self, module, name, input_shape):
        """FLOPs для Embedding: lookup only, negligible."""
        self._flops[name] = 0

    def estimate(self, input_shape=(1, 512)):
        """
        Оценивает FLOPs модели.

        Args:
            input_shape: (batch_size, seq_len)
        Returns:
            dict с деталями
        """
        self._flops = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._count_linear(module, name, input_shape)
            elif isinstance(module, nn.LayerNorm):
                self._count_layernorm(module, name, input_shape)
            elif isinstance(module, nn.Embedding):
                self._count_embedding(module, name, input_shape)

        total = sum(self._flops.values())
        return {
            'total_flops': total,
            'total_gflops': total / 1e9,
            'per_layer': dict(sorted(
                self._flops.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # top-10
        }

    def get_info(self):
        params = sum(p.numel() for p in self.model.parameters())
        return {
            'total_params': params,
            'total_params_m': params / 1e6,
            'trainable_params': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }


# ==================== Activation Memory Estimator ====================

class ActivationMemoryEstimator:
    """
    Оценка памяти для активаций transformer-модели.

    Учитывает: attention scores, FFN activations, residuals, norms.

    Args:
        n_layers: число transformer-слоёв
        d_model: размерность модели
        n_heads: число attention heads
        d_ff: размерность FFN (default: 4 * d_model)
        dtype_bytes: байт на элемент (fp32=4, fp16/bf16=2)
    """
    def __init__(self, n_layers, d_model, n_heads, d_ff=None, dtype_bytes=2):
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff or 4 * d_model
        self.dtype_bytes = dtype_bytes

    def estimate(self, batch_size, seq_len):
        """
        Оценка памяти активаций.

        Returns:
            dict с деталями (в байтах и MB)
        """
        B, S = batch_size, seq_len

        # Per-layer activations
        # Attention: Q, K, V projections
        qkv_mem = 3 * B * S * self.d_model * self.dtype_bytes
        # Attention scores: (B, n_heads, S, S)
        attn_scores_mem = B * self.n_heads * S * S * self.dtype_bytes
        # Attention output
        attn_out_mem = B * S * self.d_model * self.dtype_bytes
        # FFN: two linear layers
        ffn_mem = B * S * self.d_ff * self.dtype_bytes  # after first linear
        ffn_mem += B * S * self.d_model * self.dtype_bytes  # after second
        # LayerNorm inputs (need to save for backward)
        norm_mem = 2 * B * S * self.d_model * self.dtype_bytes  # 2 norms
        # Residuals
        residual_mem = 2 * B * S * self.d_model * self.dtype_bytes

        per_layer = (qkv_mem + attn_scores_mem + attn_out_mem +
                     ffn_mem + norm_mem + residual_mem)
        total = per_layer * self.n_layers

        # Embedding
        embed_mem = B * S * self.d_model * self.dtype_bytes

        total += embed_mem

        return {
            'per_layer_bytes': per_layer,
            'per_layer_mb': per_layer / (1024 ** 2),
            'total_bytes': total,
            'total_mb': total / (1024 ** 2),
            'total_gb': total / (1024 ** 3),
            'breakdown': {
                'qkv': qkv_mem * self.n_layers,
                'attn_scores': attn_scores_mem * self.n_layers,
                'attn_output': attn_out_mem * self.n_layers,
                'ffn': ffn_mem * self.n_layers,
                'norms': norm_mem * self.n_layers,
                'residuals': residual_mem * self.n_layers,
                'embedding': embed_mem,
            },
        }

    def compare_with_checkpointing(self, batch_size, seq_len):
        """Сравнение памяти с и без gradient checkpointing."""
        normal = self.estimate(batch_size, seq_len)
        # С checkpointing: сохраняем только входы слоёв, пересчитываем остальное
        checkpoint_layers = max(1, int(math.sqrt(self.n_layers)))
        savings_ratio = 1.0 - (checkpoint_layers / self.n_layers)
        checkpointed_total = normal['total_bytes'] * (1 - savings_ratio * 0.7)

        return {
            'without_checkpointing_mb': normal['total_mb'],
            'with_checkpointing_mb': checkpointed_total / (1024 ** 2),
            'memory_saved_mb': (normal['total_bytes'] - checkpointed_total) / (1024 ** 2),
            'savings_ratio': 1 - checkpointed_total / max(normal['total_bytes'], 1),
        }

    def get_info(self):
        return {
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dtype_bytes': self.dtype_bytes,
        }
