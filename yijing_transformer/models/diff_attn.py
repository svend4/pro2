"""
Differential Attention и KV-cache quantization.

Differential Attention вычисляет attention как разность двух softmax карт,
подавляя шум и улучшая выделение сигнала:
    attn = softmax(Q1 K1^T / √d) - λ * softmax(Q2 K2^T / √d)

Ref: Ye et al., "Differential Transformer" (2024)

KV-cache quantization: хранит KV-cache в INT8 для экономии памяти
при генерации (2x сжатие без заметной потери качества).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialAttention(nn.Module):
    """
    Differential Attention: разность двух attention карт.

    Каждая голова разделяется на две «полуголовы»:
    - Q1, K1 → attn_positive
    - Q2, K2 → attn_negative
    - output = (attn_positive - λ * attn_negative) @ V

    λ — обучаемый скаляр, инициализируется ≈ 0.8.
    Это подавляет «шумные» attention паттерны, оставляя сигнал.

    Args:
        d_model: размерность модели
        n_heads: число голов (каждая голова = 2 полуголовы)
        head_dim: размерность головы
        dropout: dropout rate
    """
    def __init__(self, d_model, n_heads, head_dim=None, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or (d_model // n_heads)
        self.half_head_dim = self.head_dim // 2
        self.scale = 1.0 / math.sqrt(self.half_head_dim)

        # Проекции: Q и K разделены на две части
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Lambda: обучаемый коэффициент подавления (per head)
        # Инициализируем exp(lambda_init) ≈ 0.8
        self.lambda_init = nn.Parameter(torch.full((n_heads,), -0.2))

    def forward(self, x, causal_mask=None):
        """
        Args:
            x: (B, T, D)
            causal_mask: (T, T) или None (auto-создаётся)

        Returns:
            output: (B, T, D)
        """
        B, T, D = x.shape
        H = self.n_heads
        hd = self.head_dim
        hhd = self.half_head_dim

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)

        # Разделяем Q и K на две половины
        q1, q2 = q[..., :hhd], q[..., hhd:]  # (B, H, T, hhd)
        k1, k2 = k[..., :hhd], k[..., hhd:]

        # Attention scores для обеих половин
        scores1 = (q1 @ k1.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        scores2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        # Causal mask
        if causal_mask is None:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores1 = scores1.masked_fill(mask == 0, float('-inf'))
        scores2 = scores2.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)

        # Differential: attn = attn1 - lambda * attn2
        lam = torch.sigmoid(self.lambda_init).view(1, H, 1, 1)  # (1, H, 1, 1)
        diff_attn = attn1 - lam * attn2
        diff_attn = self.dropout(diff_attn)

        # Weighted sum
        out = diff_attn @ v  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, H * hd)
        return self.out_proj(out)


# ==================== KV-cache Quantization ====================

class QuantizedKVCache:
    """
    INT8 KV-cache для экономии памяти при генерации.

    Хранит K и V в INT8 с per-channel scales.
    При чтении дequantizes обратно в float для attention.
    Экономия: 2x по памяти для KV-cache.

    Использование:
        cache = QuantizedKVCache()
        cache.update(k_new, v_new)
        k_full, v_full = cache.get()
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.k_quantized = None
        self.v_quantized = None
        self.k_scales = None
        self.v_scales = None
        self.k_float = None  # fallback для disabled
        self.v_float = None

    @staticmethod
    def _quantize_per_token(x):
        """
        Квантизует тензор в INT8 с per-token scale.

        Args:
            x: (..., D) float tensor

        Returns:
            x_int8: (..., D) int8 tensor
            scales: (..., 1) float tensor
        """
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = amax / 127.0
        x_int8 = (x / scales).round().clamp(-128, 127).to(torch.int8)
        return x_int8, scales

    @staticmethod
    def _dequantize(x_int8, scales):
        """Dequantize INT8 → float."""
        return x_int8.float() * scales

    def update(self, k_new, v_new):
        """
        Добавляет новые K, V к кэшу.

        Args:
            k_new: (B, H, T_new, D) — новые ключи
            v_new: (B, H, T_new, D) — новые значения
        """
        if not self.enabled:
            if self.k_float is None:
                self.k_float = k_new
                self.v_float = v_new
            else:
                self.k_float = torch.cat([self.k_float, k_new], dim=2)
                self.v_float = torch.cat([self.v_float, v_new], dim=2)
            return

        k_q, k_s = self._quantize_per_token(k_new)
        v_q, v_s = self._quantize_per_token(v_new)

        if self.k_quantized is None:
            self.k_quantized = k_q
            self.v_quantized = v_q
            self.k_scales = k_s
            self.v_scales = v_s
        else:
            self.k_quantized = torch.cat([self.k_quantized, k_q], dim=2)
            self.v_quantized = torch.cat([self.v_quantized, v_q], dim=2)
            self.k_scales = torch.cat([self.k_scales, k_s], dim=2)
            self.v_scales = torch.cat([self.v_scales, v_s], dim=2)

    def get(self):
        """
        Возвращает dequantized K, V.

        Returns:
            k: (B, H, T_total, D) float
            v: (B, H, T_total, D) float
        """
        if not self.enabled:
            return self.k_float, self.v_float

        if self.k_quantized is None:
            return None, None

        k = self._dequantize(self.k_quantized, self.k_scales)
        v = self._dequantize(self.v_quantized, self.v_scales)
        return k, v

    @property
    def seq_len(self):
        """Текущая длина кэша."""
        if not self.enabled:
            return 0 if self.k_float is None else self.k_float.shape[2]
        return 0 if self.k_quantized is None else self.k_quantized.shape[2]

    def memory_bytes(self):
        """Оценка занятой памяти в байтах."""
        if not self.enabled:
            if self.k_float is None:
                return 0
            return (self.k_float.numel() + self.v_float.numel()) * 4  # float32

        if self.k_quantized is None:
            return 0
        # INT8 + scales (float32)
        int8_bytes = (self.k_quantized.numel() + self.v_quantized.numel()) * 1
        scale_bytes = (self.k_scales.numel() + self.v_scales.numel()) * 4
        return int8_bytes + scale_bytes

    def reset(self):
        """Очищает кэш."""
        self.k_quantized = None
        self.v_quantized = None
        self.k_scales = None
        self.v_scales = None
        self.k_float = None
        self.v_float = None
