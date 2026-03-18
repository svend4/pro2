"""
Attention паттерны и модули: Triangular, Palace, Quadrant, Recursive,
Weaving, Heisenberg, FlowerOfLife, Mobius, Privileged, CubeDiagonal,
Bidirectional, DualMode, Hexagram, Geometric.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    get_trigrams, palace_attention_mask, triangular_distance_matrix,
)


class TriangularAttentionBias(nn.Module):
    """Attention bias на основе треугольных расстояний Андреева."""
    def __init__(self, max_seq_len: int = 512, P: int = 64):
        super().__init__()
        self.P = P
        self.scale = nn.Parameter(torch.tensor(-0.1))
        dist_matrix = triangular_distance_matrix(P)
        full_dist = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                full_dist[i, j] = dist_matrix[i % P, j % P]
        self.register_buffer('dist_matrix', full_dist)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.scale * self.dist_matrix[:seq_len, :seq_len]


class PalaceAttention(nn.Module):
    """Block-sparse attention по 8 дворцам (Склярова)."""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        mask = palace_attention_mask(64)
        self.register_buffer('palace_mask', mask)
        self.inter_palace_weight = nn.Parameter(torch.tensor(0.1))
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def get_mask(self, seq_len: int) -> torch.Tensor:
        if seq_len <= 64:
            base_mask = self.palace_mask[:seq_len, :seq_len]
        else:
            base_mask = torch.ones(seq_len, seq_len, device=self.palace_mask.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    base_mask[i, j] = self.palace_mask[i % 64, j % 64]
        intra = (base_mask == 1.0).float()
        inter = (base_mask < 1.0).float()
        return intra + torch.sigmoid(self.inter_palace_weight) * inter

    def forward(self, x, mask=None):
        B, T, D = x.shape
        scale = self.head_dim ** -0.5
        palace_mask = self.get_mask(T)
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn * palace_mask.unsqueeze(0).unsqueeze(0)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(B, T, D)


class QuadrantAttention(nn.Module):
    """4-квадрантный attention по Беляеву."""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.quadrant_weights = nn.Parameter(torch.tensor([1.0, -0.5, -0.5, 0.25]))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H = self.n_heads
        d = self.head_dim
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        scale = (d // 2) ** -0.5
        w = self.quadrant_weights.softmax(dim=0)
        attn_pp = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        attn_pn = torch.matmul(q1, k2.transpose(-2, -1)) * scale
        attn_np = torch.matmul(q2, k1.transpose(-2, -1)) * scale
        attn_nn = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        attn = w[0] * attn_pp + w[1] * attn_pn + w[2] * attn_np + w[3] * attn_nn
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


class RecursiveCubeAttention(nn.Module):
    """Рекурсивный attention «куб из кубов» (Беляев)."""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.intra_q = nn.Linear(d_model, d_model, bias=False)
        self.intra_k = nn.Linear(d_model, d_model, bias=False)
        self.intra_v = nn.Linear(d_model, d_model, bias=False)
        self.inter_q = nn.Linear(d_model, d_model, bias=False)
        self.inter_k = nn.Linear(d_model, d_model, bias=False)
        self.inter_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.cube_size = 8

    def forward(self, x, mask=None):
        B, T, D = x.shape
        scale = self.head_dim ** -0.5
        n_cubes = (T + self.cube_size - 1) // self.cube_size
        pad_len = n_cubes * self.cube_size - T
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        x_cubes = x_padded.reshape(B, n_cubes, self.cube_size, D)
        q1 = self.intra_q(x_cubes)
        k1 = self.intra_k(x_cubes)
        v1 = self.intra_v(x_cubes)
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.matmul(attn1, v1)
        cube_reps = out1.mean(dim=2)
        q2 = self.inter_q(cube_reps)
        k2 = self.inter_k(cube_reps)
        v2 = self.inter_v(cube_reps)
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        attn2 = F.softmax(attn2, dim=-1)
        inter_out = torch.matmul(attn2, v2)
        inter_broadcast = inter_out.unsqueeze(2).expand_as(out1)
        combined = out1 + inter_broadcast
        combined = combined.reshape(B, n_cubes * self.cube_size, D)
        if pad_len > 0:
            combined = combined[:, :T, :]
        return self.out_proj(combined)


class WeavingLoomArchitecture(nn.Module):
    """4-уровневая иерархия «ткацкий станок» Беляева."""
    def __init__(self, d_model: int, max_level: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_level = min(max_level, 4)
        self.level1_gate = nn.Linear(d_model, 1, bias=True)
        if max_level >= 2:
            self.level2_q = nn.Linear(d_model, d_model, bias=False)
            self.level2_k = nn.Linear(d_model, d_model, bias=False)
            self.level2_v = nn.Linear(d_model, d_model, bias=False)
            trigrams = get_trigrams()
            tri_dist = torch.cdist(trigrams, trigrams)
            self.register_buffer('level2_bias', -tri_dist * 0.1)
        if max_level >= 3:
            self.level3_q = nn.Linear(d_model, d_model, bias=False)
            self.level3_k = nn.Linear(d_model, d_model, bias=False)
            self.level3_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        head_dim = D // max(1, getattr(self, 'n_heads', 1))
        scale = head_dim ** -0.5
        gate = torch.sigmoid(self.level1_gate(x))
        out = x * gate
        if self.max_level < 2:
            return self.out_proj(out)
        group_size = min(8, T)
        n_groups = max(1, (T + group_size - 1) // group_size)
        pad_len = n_groups * group_size - T
        if pad_len > 0:
            out_padded = F.pad(out, (0, 0, 0, pad_len))
        else:
            out_padded = out
        out_groups = out_padded.reshape(B, n_groups, group_size, D)
        q2 = self.level2_q(out_groups)
        k2 = self.level2_k(out_groups)
        v2 = self.level2_v(out_groups)
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        if group_size <= 8:
            attn2 = attn2 + self.level2_bias[:group_size, :group_size]
        attn2 = F.softmax(attn2, dim=-1)
        out2 = torch.matmul(attn2, v2)
        out2 = out2.reshape(B, n_groups * group_size, D)
        if pad_len > 0:
            out2 = out2[:, :T, :]
        if self.max_level < 3:
            return self.out_proj(out2)
        q3 = self.level3_q(out2)
        k3 = self.level3_k(out2)
        v3 = self.level3_v(out2)
        attn3 = torch.matmul(q3, k3.transpose(-2, -1)) * scale
        if mask is not None:
            attn3 = attn3.masked_fill(mask == 0, float('-inf'))
        attn3 = F.softmax(attn3, dim=-1)
        attn3 = attn3.nan_to_num(0.0)
        out3 = torch.matmul(attn3, v3)
        return self.out_proj(out3)


class BidirectionalTriangularAttention(nn.Module):
    """Двунаправленный треугольный attention (Андреев)."""
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.direction_bias = nn.Parameter(torch.tensor(0.0))
        tri_lower = torch.tril(torch.ones(max_seq_len, max_seq_len))
        tri_upper = torch.triu(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('tri_lower', tri_lower)
        self.register_buffer('tri_upper', tri_upper)

    def get_mask(self, seq_len: int) -> torch.Tensor:
        alpha = torch.sigmoid(self.direction_bias)
        lower = self.tri_lower[:seq_len, :seq_len]
        upper = self.tri_upper[:seq_len, :seq_len]
        return alpha * lower + (1 - alpha) * upper


class CubeDiagonalAttention(nn.Module):
    """Attention по 4 типам диагоналей куба (Касаткин)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        self.diag_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.25, 2.0]))

    def get_bias(self, x):
        z = self.proj_to_3d(x)
        z_sign = torch.sign(z)
        hamming = (z_sign.unsqueeze(2) != z_sign.unsqueeze(1)).float().sum(dim=-1)
        bias = torch.zeros_like(hamming)
        for d_type in range(4):
            mask = (hamming == d_type).float()
            bias = bias + self.diag_weights[min(d_type, 3)] * mask
        return bias


class HeisenbergAttention(nn.Module):
    """Attention из принципа Гейзенберга (Беляев)."""
    def __init__(self, d_model: int, min_temp: float = 0.1, max_temp: float = 5.0):
        super().__init__()
        self.d_model = d_model
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.hbar_half = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q_uncertainty = q.norm(dim=-1, keepdim=True)
        temperature = self.hbar_half / (q_uncertainty + 1e-8)
        temperature = temperature.clamp(self.min_temp, self.max_temp)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        return torch.matmul(attn, v)


class FlowerOfLifeGAT(nn.Module):
    """Цветок Жизни как GAT-граф (Беляев)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = 7
        adj = torch.zeros(7, 7)
        for i in range(1, 7):
            adj[0, i] = 1
            adj[i, 0] = 1
        for i in range(1, 7):
            j = (i % 6) + 1
            adj[i, j] = 1
            adj[j, i] = 1
        self.register_buffer('adjacency', adj)
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Linear(2 * d_model, 1, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        node_tokens = []
        for i in range(self.n_nodes):
            if i < T:
                indices = torch.arange(i, T, self.n_nodes, device=x.device)
                node_tokens.append(x[:, indices].mean(dim=1))
            else:
                node_tokens.append(torch.zeros(B, D, device=x.device))
        nodes = torch.stack(node_tokens, dim=1)
        h = self.W(nodes)
        h_i = h.unsqueeze(2).expand(-1, -1, 7, -1)
        h_j = h.unsqueeze(1).expand(-1, 7, -1, -1)
        e = self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        e = e.masked_fill(self.adjacency == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)
        alpha = alpha.nan_to_num(0.0)
        out_nodes = torch.matmul(alpha, h)
        out_nodes = self.out_proj(out_nodes)
        result = x.clone()
        for i in range(min(self.n_nodes, T)):
            indices = torch.arange(i, T, self.n_nodes, device=x.device)
            result[:, indices] = result[:, indices] + out_nodes[:, i:i+1]
        return result


class MobiusAttentionPattern(nn.Module):
    """Attention-паттерн с топологией ленты Мёбиуса (Беляев)."""
    def __init__(self, max_seq_len: int = 512):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        bias = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                half = max_seq_len // 2
                if i < half and j >= half:
                    mirror_j = max_seq_len - 1 - j
                    dist = abs(i - mirror_j)
                elif i >= half and j < half:
                    mirror_i = max_seq_len - 1 - i
                    dist = abs(mirror_i - j)
                else:
                    dist = abs(i - j)
                bias[i, j] = -dist
        self.register_buffer('mobius_bias', bias)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.scale * self.mobius_bias[:seq_len, :seq_len]


class CubicAttentionBias(nn.Module):
    """Attention bias на основе 3D расстояний Касаткина.

    bias[i][j] = -alpha * ||kasatkin(i mod 64) - kasatkin(j mod 64)||²

    Позиции в кубе 4×4×4 определяют «близость» токенов:
    - Токены в одной ячейке куба имеют bias=0
    - Дальние в кубе получают отрицательный bias (ослабленный attention)
    """
    def __init__(self, max_seq_len: int = 512):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        # Build 64×64 distance matrix
        coords = torch.zeros(64, 3, dtype=torch.float32)
        for i in range(64):
            bits = [(i >> bit) & 1 for bit in range(6)]
            coords[i, 0] = bits[0] * 2 + bits[1]
            coords[i, 1] = bits[2] * 2 + bits[3]
            coords[i, 2] = bits[4] * 2 + bits[5]
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (64, 64, 3)
        dist_sq = (diff ** 2).sum(dim=-1)  # (64, 64)
        # Normalize so max distance = 1
        dist_sq = dist_sq / dist_sq.max().clamp(min=1e-6)
        self.register_buffer('dist_matrix', dist_sq)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns (T, T) bias matrix."""
        pos = torch.arange(seq_len, device=self.dist_matrix.device) % 64
        bias = -self.scale * self.dist_matrix[pos][:, pos]  # (T, T)
        return bias


class PrivilegedAxisAttention(nn.Module):
    """Attention с привилегированной осью (Касаткин)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        self.axis_scale = nn.Parameter(torch.tensor(0.1))
        axis = torch.ones(3) / math.sqrt(3)
        self.register_buffer('axis', axis)

    def get_bias(self, x):
        z = self.proj_to_3d(x)
        axis_proj = (z * self.axis).sum(dim=-1)
        bias = self.axis_scale * axis_proj.unsqueeze(-1) * axis_proj.unsqueeze(-2)
        return bias


class DualModeHead(nn.Module):
    """Мезонный/барионный dual-mode attention head (Беляев)."""
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.mode = nn.Parameter(torch.tensor(0.5))
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        meson_attn = -torch.matmul(q, k.transpose(-2, -1)) * self.scale
        k_perm = torch.roll(k, shifts=1, dims=-1)
        baryon_attn = torch.matmul(q, k_perm.transpose(-2, -1)) * self.scale
        alpha = torch.sigmoid(self.mode)
        attn = alpha * meson_attn + (1 - alpha) * baryon_attn
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class StructuralDefectLayer(nn.Module):
    """Структурный дефект 16→12: геометрический bottleneck (Беляев)."""
    def __init__(self, d_model: int, input_size: int = 16, output_size: int = 12):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.anchors = nn.Parameter(torch.randn(output_size, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(self.anchors.unsqueeze(0).expand(B, -1, -1))
        k = self.k_proj(x)
        scale = D ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, x)


class HexagramAttentionPattern(nn.Module):
    """Гексаграммный паттерн attention: 64 фиксированных паттерна."""
    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))
        patterns = torch.zeros(6, block_size, block_size)
        for i in range(block_size):
            for j in range(block_size):
                dist = abs(i - j)
                patterns[0, i, j] = 1.0 if dist <= 2 else -1.0
                patterns[1, i, j] = 1.0 if dist <= 8 else -1.0
                patterns[2, i, j] = 1.0 if dist <= 32 else -1.0
                patterns[3, i, j] = 1.0 if dist % 2 == 0 else -1.0
                patterns[4, i, j] = 1.0 if dist % 4 == 0 else -1.0
                patterns[5, i, j] = 1.0 if min(i, j) <= 4 else -1.0
        self.register_buffer('patterns', patterns)

    def forward(self, x, T):
        x_mean = x.mean(dim=1)
        z6 = self.proj_to_6d(x_mean)
        hex_weights = torch.tanh(z6)
        patterns = self.patterns[:, :T, :T]
        bias = torch.einsum('bk,kij->bij', hex_weights, patterns)
        return self.scale * bias.unsqueeze(1)


class GeometricAttention(nn.Module):
    """Attention, полностью основанный на геометрии триграмм."""
    def __init__(self, cfg):
        super().__init__()
        from .positional import RotaryEmbedding
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.use_rope = cfg.use_rope
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * 3, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * 3, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        trigrams = get_trigrams()
        trigrams_norm = F.normalize(trigrams, p=2, dim=1)
        self.register_buffer('head_dirs', trigrams_norm[:cfg.n_heads])
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(3.0)))
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                dim=4, max_seq_len=cfg.block_size, base=cfg.rope_base,
            )

    def forward(self, x):
        B, T, C = x.shape
        q3 = self.q_proj(x).reshape(B, T, self.n_heads, 3).transpose(1, 2)
        k3 = self.k_proj(x).reshape(B, T, self.n_heads, 3).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q_on_dir = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)
        k_on_dir = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
        dot_scores = (q3 @ k3.transpose(-2, -1)) * self.scale
        geo_bias = q_on_dir.unsqueeze(-1) * k_on_dir.unsqueeze(-2)
        scores = dot_scores + geo_bias
        causal = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class MagicSquareInitializer:
    """Инициализация attention-весов магическим квадратом."""
    @staticmethod
    def loshu_3x3() -> torch.Tensor:
        return torch.tensor([[2, 7, 6], [9, 5, 1], [4, 3, 8]], dtype=torch.float32)

    @staticmethod
    def magic_4x4() -> torch.Tensor:
        return torch.tensor([
            [16, 3, 2, 13], [5, 10, 11, 8],
            [9, 6, 7, 12], [4, 15, 14, 1]
        ], dtype=torch.float32)

    @staticmethod
    def from_hermann_packing(k: int) -> torch.Tensor:
        from .core import hermann_packing
        n = 2 ** k
        field = hermann_packing(2 * k)
        return field.reshape(n, n).float()

    @staticmethod
    def init_attention_weights(weight: torch.Tensor, n_heads: int = 8):
        H, T, _ = weight.shape if weight.dim() == 3 else (1, *weight.shape)
        if T <= 4:
            ms = MagicSquareInitializer.magic_4x4()[:T, :T]
        else:
            ms = MagicSquareInitializer.loshu_3x3()
            ms = F.interpolate(ms.unsqueeze(0).unsqueeze(0), size=(T, T),
                               mode='bilinear', align_corners=False).squeeze()
        ms = ms / ms.sum()
        with torch.no_grad():
            if weight.dim() == 3:
                for h in range(H):
                    weight[h] = ms
            else:
                weight.copy_(ms)


class TriangularCurriculumScheduler:
    """Curriculum learning по треугольным числам (Андреев)."""
    def __init__(self, max_level: int = 11):
        self.max_level = max_level
        self.levels = [k * (k + 1) // 2 for k in range(1, max_level + 1)]

    def get_seq_len(self, epoch: int, total_epochs: int) -> int:
        progress = min(epoch / max(total_epochs, 1), 1.0)
        level = int(progress * self.max_level)
        level = max(0, min(level, len(self.levels) - 1))
        return self.levels[level]
