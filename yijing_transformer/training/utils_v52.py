"""
v52 утилиты: Sliding Window Attention, ALiBi Positional Bias,
Rotary Position Embedding (RoPE), Flash Attention Approximation,
Multi-Query Attention Helper.

Sliding Window: локальный attention с фиксированным окном.
Ref: Beltagy et al., "Longformer" (2020)

ALiBi: линейный bias вместо positional embeddings.
Ref: Press et al., "Train Short, Test Long" (2022)

RoPE: вращательные позиционные embeddings.
Ref: Su et al., "RoFormer" (2021)

Flash Attention Approximation: memory-efficient attention.
Ref: Dao et al., "FlashAttention" (2022)

Multi-Query Attention: shared KV heads.
Ref: Shazeer, "Fast Transformer Decoding" (2019);
     Ainslie et al., "GQA" (2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Sliding Window Attention ====================

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention для длинных последовательностей.

    Каждый токен attend только к window_size ближайших.
    O(n*w) вместо O(n²).

    Args:
        window_size: размер окна (в одну сторону)
        causal: причинная маска
    """
    def __init__(self, window_size=256, causal=True):
        super().__init__()
        self.window_size = window_size
        self.causal = causal

    def create_mask(self, seq_len, device='cpu'):
        """
        Создаёт sliding window mask.

        Args:
            seq_len: длина последовательности
            device: устройство

        Returns:
            Tensor: (1, 1, T, T) маска (True = attend)
        """
        # Position indices
        rows = torch.arange(seq_len, device=device).unsqueeze(1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)

        # Window mask
        mask = (cols - rows).abs() <= self.window_size

        if self.causal:
            causal_mask = cols <= rows
            mask = mask & causal_mask

        return mask.unsqueeze(0).unsqueeze(0).float()

    def apply_mask(self, attention_scores, seq_len):
        """
        Применяет sliding window mask к attention scores.

        Args:
            attention_scores: (B, H, T, T)
            seq_len: длина последовательности

        Returns:
            Tensor: masked attention scores
        """
        mask = self.create_mask(seq_len, device=attention_scores.device)
        return attention_scores.masked_fill(mask == 0, -1e9)


# ==================== ALiBi Positional Bias ====================

class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases.

    Добавляет -m * |i-j| к attention scores.
    Не требует positional embeddings, экстраполирует.

    Args:
        n_heads: число голов
    """
    def __init__(self, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        # Geometric slopes: 2^(-8/n), 2^(-16/n), ...
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)

    @staticmethod
    def _get_slopes(n_heads):
        """Вычисляет slopes для ALiBi."""
        def get_slopes_power(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power(n_heads))

        # Non-power-of-2: interpolate
        closest_power = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power(closest_power)
        extra = get_slopes_power(2 * closest_power)
        extra = extra[0::2][:n_heads - closest_power]
        return torch.tensor(slopes + extra)

    def get_bias(self, seq_len, device='cpu'):
        """
        Вычисляет ALiBi bias matrix.

        Args:
            seq_len: длина последовательности

        Returns:
            Tensor: (1, H, T, T) bias
        """
        positions = torch.arange(seq_len, device=device)
        # |i - j| distance matrix
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        relative_pos = relative_pos.abs().float()

        # Apply slopes: (H, 1, 1) * (T, T) → (H, T, T)
        slopes = self.slopes.to(device)
        bias = -slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos.unsqueeze(0)

        return bias.unsqueeze(0)  # (1, H, T, T)

    def forward(self, attention_scores):
        """
        Добавляет ALiBi bias к attention scores.

        Args:
            attention_scores: (B, H, T, T)

        Returns:
            Tensor: (B, H, T, T) biased scores
        """
        T = attention_scores.size(-1)
        bias = self.get_bias(T, device=attention_scores.device)
        return attention_scores + bias


# ==================== Rotary Position Embedding ====================

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embedding.

    Кодирует позицию через вращение в паре измерений.
    Позволяет relative position через dot product.

    Args:
        dim: размерность (должна быть чётной)
        max_seq_len: максимальная длина
        base: база для частот (10000 стандартно)
    """
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        """
        Применяет RoPE к тензору.

        Args:
            x: (B, H, T, D) или (B, T, D)

        Returns:
            Tensor: rotated x
        """
        if x.dim() == 4:
            T = x.size(2)
        else:
            T = x.size(1)

        if T > self.max_seq_len:
            self._build_cache(T)
            self.max_seq_len = T

        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]

        return self._apply_rotary(x, cos, sin)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        """Применяет вращение."""
        if x.dim() == 4:
            # (B, H, T, D)
            cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            # (B, T, D)
            cos = cos.unsqueeze(0)  # (1, T, D)
            sin = sin.unsqueeze(0)

        # Split into pairs and rotate
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]

        cos = cos[..., :d//2]
        sin = sin[..., :d//2]

        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)

        return rotated


# ==================== Flash Attention Approximation ====================

class FlashAttentionApprox(nn.Module):
    """
    Memory-efficient attention computation.

    Chunked attention: разбивает Q на блоки, вычисляет
    attention по частям для экономии памяти.
    Не тру Flash Attention, но экономит память.

    Args:
        chunk_size: размер блока query
        causal: причинная маска
        dropout: dropout rate
    """
    def __init__(self, chunk_size=256, causal=True, dropout=0.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.causal = causal
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        """
        Chunked attention computation.

        Args:
            q: (B, H, T_q, D)
            k: (B, H, T_k, D)
            v: (B, H, T_k, D)
            mask: optional (B, 1, T_q, T_k)

        Returns:
            Tensor: (B, H, T_q, D)
        """
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        scale = 1.0 / math.sqrt(D)

        # If small enough, use standard attention
        if T_q <= self.chunk_size:
            return self._standard_attention(q, k, v, scale, mask)

        # Chunked computation
        outputs = []
        for start in range(0, T_q, self.chunk_size):
            end = min(start + self.chunk_size, T_q)
            q_chunk = q[:, :, start:end, :]

            # Attention scores for chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

            if self.causal:
                # Causal mask for this chunk
                q_pos = torch.arange(start, end, device=q.device)
                k_pos = torch.arange(T_k, device=q.device)
                causal_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

            if mask is not None:
                chunk_mask = mask[:, :, start:end, :]
                scores = scores.masked_fill(chunk_mask == 0, -1e9)

            attn = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                attn = F.dropout(attn, p=self.dropout)

            output = torch.matmul(attn, v)
            outputs.append(output)

        return torch.cat(outputs, dim=2)

    def _standard_attention(self, q, k, v, scale, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.causal:
            T_q, T_k = q.size(2), k.size(2)
            causal_mask = torch.tril(torch.ones(T_q, T_k, device=q.device)).bool()
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        return torch.matmul(attn, v)


# ==================== Multi-Query Attention Helper ====================

class MultiQueryAttentionHelper:
    """
    Утилиты для Multi-Query (MQA) и Grouped-Query (GQA) Attention.

    MQA: все головы share одни K,V → меньше памяти.
    GQA: группы голов share K,V → компромисс.

    Args:
        n_heads: число query heads
        n_kv_heads: число KV heads (1 = MQA, n_heads = MHA)
    """
    def __init__(self, n_heads=8, n_kv_heads=1):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads

    def expand_kv(self, kv):
        """
        Расширяет KV heads для GQA.

        Args:
            kv: (B, n_kv_heads, T, D)

        Returns:
            Tensor: (B, n_heads, T, D)
        """
        if self.n_kv_heads == self.n_heads:
            return kv

        B, _, T, D = kv.shape
        # Repeat each KV head n_groups times
        kv = kv.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        return kv.reshape(B, self.n_heads, T, D)

    def compute_attention(self, q, k, v, mask=None, causal=True):
        """
        GQA attention computation.

        Args:
            q: (B, n_heads, T, D)
            k: (B, n_kv_heads, T, D)
            v: (B, n_kv_heads, T, D)
            mask: optional mask
            causal: причинная маска

        Returns:
            Tensor: (B, n_heads, T, D)
        """
        # Expand KV
        k = self.expand_kv(k)
        v = self.expand_kv(v)

        # Standard attention
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal:
            T = q.size(2)
            causal_mask = torch.tril(torch.ones(T, T, device=q.device)).bool()
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def get_memory_savings(self, seq_len, head_dim):
        """
        Оценка экономии памяти.

        Returns:
            dict: {mha_bytes, gqa_bytes, savings_ratio}
        """
        mha_kv = 2 * self.n_heads * seq_len * head_dim
        gqa_kv = 2 * self.n_kv_heads * seq_len * head_dim
        return {
            'mha_params': mha_kv,
            'gqa_params': gqa_kv,
            'savings_ratio': 1.0 - gqa_kv / max(mha_kv, 1),
        }

    def get_info(self):
        return {
            'n_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads,
            'n_groups': self.n_groups,
            'type': 'MQA' if self.n_kv_heads == 1 else
                    'MHA' if self.n_kv_heads == self.n_heads else 'GQA',
        }
