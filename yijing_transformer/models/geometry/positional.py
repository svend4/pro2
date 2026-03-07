"""
Позиционные кодирования: RoPE, ALiBi, FourLevelPE, CubicPE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """RoPE: вращение пар измерений в зависимости от позиции."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0,
                 scaling: str = None, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling = scaling
        self.scaling_factor = scaling_factor
        if scaling == 'ntk' and scaling_factor > 1.0:
            base = base * (scaling_factor ** (dim / (dim - 2)))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        if self.scaling == 'linear' and self.scaling_factor > 1.0:
            t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Поворот половины измерений для RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Применение RoPE к тензору x: (B, H, T, D)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class ALiBi(nn.Module):
    """Attention with Linear Biases."""
    def __init__(self, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
        self._build_cache(max_seq_len)

    @staticmethod
    def _get_slopes(n_heads):
        def _get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(torch.tensor(float(n)).log2().floor().item() - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        if n_heads & (n_heads - 1) == 0:
            slopes = _get_slopes_power_of_2(n_heads)
        else:
            closest_power = 2 ** int(torch.tensor(float(n_heads)).log2().floor().item())
            slopes = _get_slopes_power_of_2(closest_power)
            extra = _get_slopes_power_of_2(2 * closest_power)
            slopes = slopes + extra[0::2][:n_heads - closest_power]
        return torch.tensor(slopes, dtype=torch.float32)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = distances.abs().float()
        self.register_buffer('_distances', distances, persistent=False)

    def forward(self, seq_len, offset=0):
        total_len = seq_len + offset
        if total_len > self._distances.shape[0]:
            self._build_cache(total_len)
        distances = self._distances[offset:offset + seq_len, :total_len]
        bias = -self.slopes.view(1, -1, 1, 1) * distances.unsqueeze(0).unsqueeze(0)
        return bias


class FourLevelPositionalEncoding(nn.Module):
    """4-уровневое позиционное кодирование Андреева."""
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.line_emb = nn.Embedding(6, d_model)
        self.trigram_emb = nn.Embedding(2, d_model)
        self.hexagram_emb = nn.Embedding(64, d_model)
        self.seq_emb = nn.Embedding(max_seq_len, d_model)
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, seq_len: int, device=None):
        if device is None:
            device = self.line_emb.weight.device
        pos = torch.arange(seq_len, device=device)
        line_idx = pos % 6
        trigram_idx = (pos % 6) // 3
        hex_idx = (pos // 6) % 64
        seq_idx = pos
        pe = (self.line_emb(line_idx)
              + self.trigram_emb(trigram_idx)
              + self.hexagram_emb(hex_idx)
              + self.seq_emb(seq_idx))
        return self.scale * pe


class CubicPositionalEncoding(nn.Module):
    """3D позиционное кодирование Касаткина.

    Каждый токен получает координату (x, y, z) в кубе 4×4×4 вместо
    линейной позиции. Координаты определяются через Касаткинское
    отображение hex → Z³.

    Для последовательностей длиннее 64: позиция циклически отображается
    в куб, плюс добавляется residual линейное смещение.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        # 3D coordinate embeddings (4 значения на ось)
        self.x_emb = nn.Embedding(4, d_model)
        self.y_emb = nn.Embedding(4, d_model)
        self.z_emb = nn.Embedding(4, d_model)
        # Residual linear position (for sequences > 64)
        self.linear_emb = nn.Embedding(max_seq_len, d_model)
        # Learnable scale to start small
        self.scale = nn.Parameter(torch.tensor(0.1))
        # Pre-compute cubic coordinates for 64 positions
        self._build_coordinate_cache()

    def _build_coordinate_cache(self):
        """Build mapping: position mod 64 → (x, y, z)."""
        coords = torch.zeros(64, 3, dtype=torch.long)
        for i in range(64):
            # 6-bit decomposition: i = b5*32 + b4*16 + b3*8 + b2*4 + b1*2 + b0
            bits = [(i >> bit) & 1 for bit in range(6)]
            # Pair mapping: (b0,b1)→x, (b2,b3)→y, (b4,b5)→z
            coords[i, 0] = bits[0] * 2 + bits[1]
            coords[i, 1] = bits[2] * 2 + bits[3]
            coords[i, 2] = bits[4] * 2 + bits[5]
        self.register_buffer('cubic_coords', coords)

    def forward(self, seq_len: int, device=None):
        if device is None:
            device = self.x_emb.weight.device

        pos = torch.arange(seq_len, device=device)
        cubic_idx = pos % 64
        coords = self.cubic_coords[cubic_idx]  # (T, 3)

        pe_3d = (self.x_emb(coords[:, 0])
                 + self.y_emb(coords[:, 1])
                 + self.z_emb(coords[:, 2]))  # (T, d_model)

        pe_linear = self.linear_emb(pos)  # (T, d_model)

        return self.scale * (pe_3d + 0.1 * pe_linear)
