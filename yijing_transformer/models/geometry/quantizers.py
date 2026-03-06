"""
Квантизаторы: YiJing, E8, Factored, FourState, Antipodal,
Hierarchical, Deformable, Gumbel, Grouped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    get_trigrams, get_hexagrams, generate_hypercube,
    generate_e8_roots, generate_four_state_codebook, antipodal_index,
)


class YiJingQuantizer(nn.Module):
    """Наивная квантизация к 64 гексаграммам (вершинам гиперкуба {-1,+1}⁶)."""
    def __init__(self, temp=0.3):
        super().__init__()
        self.temp = temp
        hexagrams = get_hexagrams()
        self.register_buffer('codebook', hexagrams)

    def forward(self, x):
        dists_sq = torch.cdist(x, self.codebook.unsqueeze(0).expand(x.shape[0], -1, -1)
                                if x.dim() == 3 else self.codebook) ** 2
        weights = F.softmax(-dists_sq / self.temp, dim=-1)
        quantized = weights @ self.codebook
        return x + (quantized - x).detach()

    def hard_quantize(self, x):
        return torch.sign(x)


class E8Quantizer(nn.Module):
    """Квантизация к 240 корням решётки E8 в R⁸."""
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        e8 = generate_e8_roots()
        self.register_buffer('codebook', e8)
        self.register_buffer('codebook_norm_sq', (e8 ** 2).sum(dim=1))

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        cross = x @ self.codebook.T
        dists_sq = x_norm_sq - 2 * cross + self.codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        quantized = weights @ self.codebook
        if self.adaptive_temp:
            return quantized
        return x + (quantized - x).detach()

    def hard_quantize(self, x):
        dists = torch.cdist(x.reshape(-1, 8), self.codebook)
        idx = dists.argmin(dim=-1)
        return self.codebook[idx].reshape(x.shape)


class FactoredYiJingQuantizer(nn.Module):
    """Факторизованная квантизация: раздельно для верхней и нижней триграммы.
    Сложность: 2 × softmax(8) = O(16) вместо softmax(64) = O(64)."""
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        trigrams = get_trigrams()
        self.register_buffer('trigrams', trigrams)
        self.register_buffer('trigrams_norm_sq', (trigrams ** 2).sum(dim=1))

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        upper, lower = x[..., :3], x[..., 3:]
        upper_q = self._soft_quantize(upper)
        lower_q = self._soft_quantize(lower)
        quantized = torch.cat([upper_q, lower_q], dim=-1)
        if self.adaptive_temp:
            return quantized
        else:
            return x + (quantized - x).detach()

    def _soft_quantize(self, z):
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
        cross = z @ self.trigrams.T
        dists_sq = z_norm_sq - 2 * cross + self.trigrams_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        return weights @ self.trigrams

    def hard_quantize(self, x):
        return torch.sign(x)


class FourStateQuantizer(nn.Module):
    """Квантизация к 4096 состояниям: {-1, -0.5, +0.5, +1}⁶.
    Факторизация: 6 независимых подпространств по 4 точки."""
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        states = torch.tensor([-1.0, -0.5, 0.5, 1.0])
        self.register_buffer('states', states)

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        quantized_dims = []
        for d in range(x.shape[-1]):
            xi = x[..., d:d+1]
            dists_sq = (xi - self.states) ** 2
            weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
            q = (weights * self.states).sum(dim=-1, keepdim=True)
            quantized_dims.append(q)
        quantized = torch.cat(quantized_dims, dim=-1)
        if self.adaptive_temp:
            return quantized
        return x + (quantized - x).detach()


class AntipodalQuantizer(nn.Module):
    """Квантизатор с антиподальным weight tying.
    Только 32 свободных эмбеддинга — антиподы определены автоматически."""
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        hexagrams = get_hexagrams()
        self.register_buffer('codebook', hexagrams)
        self.register_buffer('antipod_idx', antipodal_index())

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        cross = x @ self.codebook.T
        codebook_norm_sq = (self.codebook ** 2).sum(dim=1)
        dists_sq = x_norm_sq - 2 * cross + codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        quantized = weights @ self.codebook
        if self.adaptive_temp:
            return quantized
        return x + (quantized - x).detach()

    def antipodal_loss(self, x):
        """Штраф: ||Q(x) + Q(-x)|| → 0."""
        q_x = self.forward(x)
        q_neg_x = self.forward(-x)
        return (q_x + q_neg_x).pow(2).mean()


class HierarchicalQuantizer(nn.Module):
    """Иерархическая квантизация: разбивает вход на группы (Product Quantization)."""
    def __init__(self, total_dim: int, group_dim: int = 2,
                 temp: float = 0.3, adaptive_temp: bool = False):
        super().__init__()
        assert total_dim % group_dim == 0
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim

        codebook = generate_hypercube(group_dim)
        self.register_buffer('codebook', codebook)
        self.register_buffer('codebook_norm_sq', (codebook ** 2).sum(dim=1))

        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)
        quantized_groups = self._soft_quantize_batch(groups)
        quantized = quantized_groups.reshape(*shape, self.total_dim)
        if self.adaptive_temp:
            return quantized
        else:
            return x + (quantized - x).detach()

    def _soft_quantize_batch(self, z):
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
        cross = z @ self.codebook.T
        dists_sq = z_norm_sq - 2 * cross + self.codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        return weights @ self.codebook

    def hard_quantize(self, x):
        return torch.sign(x)

    def codebook_info(self):
        return {
            'total_dim': self.total_dim,
            'group_dim': self.group_dim,
            'n_groups': self.n_groups,
            'n_codewords_per_group': self.n_codewords,
            'total_codewords': self.n_codewords ** self.n_groups,
            'softmax_ops': self.n_groups * self.n_codewords,
        }


class DeformableQuantizer(nn.Module):
    """Деформируемый кодбук: base_hypercube + learnable_delta."""
    def __init__(self, total_dim: int, group_dim: int = 3,
                 temp: float = 0.3, deform_scale: float = 0.0):
        super().__init__()
        assert total_dim % group_dim == 0
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim

        base = generate_hypercube(group_dim)
        self.register_buffer('base_codebook', base)
        self.delta = nn.Parameter(torch.zeros_like(base))
        self.deform_scale = nn.Parameter(torch.tensor(deform_scale))
        self.temp = temp

    @property
    def codebook(self):
        return self.base_codebook + self.deform_scale * self.delta

    def forward(self, x):
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)
        cb = self.codebook
        cb_norm_sq = (cb * cb).sum(dim=1)
        z_norm_sq = (groups * groups).sum(dim=-1, keepdim=True)
        cross = groups @ cb.T
        dists_sq = z_norm_sq - 2 * cross + cb_norm_sq
        weights = F.softmax(-dists_sq / self.temp, dim=-1)
        quantized_groups = weights @ cb
        quantized = quantized_groups.reshape(*shape, self.total_dim)
        return x + (quantized - x).detach()

    def deformation_stats(self):
        delta_norm = self.delta.norm().item()
        scale = self.deform_scale.item()
        return {
            'delta_norm': delta_norm,
            'deform_scale': scale,
            'effective_shift': delta_norm * abs(scale),
        }


class GumbelQuantizer(nn.Module):
    """Gumbel-Softmax квантизация к вершинам гиперкуба."""
    def __init__(self, total_dim: int, group_dim: int = 3,
                 temp: float = 1.0, hard: bool = False,
                 commitment_weight: float = 0.25):
        super().__init__()
        assert total_dim % group_dim == 0
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim
        self.hard = hard
        self.commitment_weight = commitment_weight
        self.log_temp = nn.Parameter(torch.tensor(temp).log())

        codebook = generate_hypercube(group_dim)
        self.register_buffer('codebook', codebook)
        self.register_buffer('codebook_norm_sq', (codebook ** 2).sum(dim=1))
        self._commitment_loss = torch.tensor(0.0)

    @property
    def current_temp(self):
        return self.log_temp.exp().clamp(min=0.05, max=5.0)

    def forward(self, x):
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)
        z_norm_sq = (groups * groups).sum(dim=-1, keepdim=True)
        cross = groups @ self.codebook.T
        dists_sq = z_norm_sq - 2 * cross + self.codebook_norm_sq
        logits = -dists_sq

        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.current_temp, hard=self.hard)
        else:
            idx = logits.argmax(dim=-1)
            weights = F.one_hot(idx, self.n_codewords).float()

        quantized_groups = weights @ self.codebook
        quantized = quantized_groups.reshape(*shape, self.total_dim)

        if self.training and self.commitment_weight > 0:
            self._commitment_loss = (
                (x.detach() - quantized).pow(2).mean()
                + self.commitment_weight * (x - quantized.detach()).pow(2).mean()
            )
        else:
            self._commitment_loss = torch.tensor(0.0, device=x.device)
        return quantized

    def get_commitment_loss(self):
        return self._commitment_loss


class GroupedQuantizer(nn.Module):
    """Grouped (per-channel) quantization с обучаемыми scales."""
    def __init__(self, d_model, group_size=128, n_bits=8, symmetric=True):
        super().__init__()
        self.d_model = d_model
        self.group_size = min(group_size, d_model)
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.n_groups = (d_model + self.group_size - 1) // self.group_size

        if symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1

        self.scales = nn.Parameter(torch.ones(self.n_groups))
        if not symmetric:
            self.zero_points = nn.Parameter(torch.zeros(self.n_groups))

    def _reshape_to_groups(self, x):
        *batch, d = x.shape
        pad = (self.group_size - d % self.group_size) % self.group_size
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x.view(*batch, self.n_groups, self.group_size)

    def _unreshape(self, x, orig_d):
        *batch, ng, gs = x.shape
        return x.reshape(*batch, ng * gs)[..., :orig_d]

    def quantize(self, x):
        orig_d = x.shape[-1]
        x_g = self._reshape_to_groups(x)
        scales = self.scales.abs().clamp(min=1e-8)
        scales = scales.view(*([1] * (x_g.dim() - 2)), self.n_groups, 1)

        if self.symmetric:
            x_scaled = x_g / scales
            x_quant = x_scaled.round().clamp(self.qmin, self.qmax)
            x_dequant = x_quant * scales
        else:
            zp = self.zero_points.round()
            zp = zp.view(*([1] * (x_g.dim() - 2)), self.n_groups, 1)
            x_scaled = x_g / scales + zp
            x_quant = x_scaled.round().clamp(self.qmin, self.qmax)
            x_dequant = (x_quant - zp) * scales

        result = x_g + (x_dequant - x_g).detach()
        return self._unreshape(result, orig_d)

    def forward(self, x):
        return self.quantize(x)

    def calibrate(self, x):
        with torch.no_grad():
            x_g = self._reshape_to_groups(x)
            if self.symmetric:
                amax = x_g.abs().amax(dim=-1).mean(dim=tuple(range(x_g.dim() - 2)))
                self.scales.data = amax / self.qmax
            else:
                vmin = x_g.amin(dim=-1).mean(dim=tuple(range(x_g.dim() - 2)))
                vmax = x_g.amax(dim=-1).mean(dim=tuple(range(x_g.dim() - 2)))
                self.scales.data = (vmax - vmin) / (self.qmax - self.qmin)
                self.zero_points.data = (self.qmin - vmin / self.scales.data).round()

    def extra_repr(self):
        return (f"d_model={self.d_model}, groups={self.n_groups}, "
                f"group_size={self.group_size}, bits={self.n_bits}, "
                f"symmetric={self.symmetric}")
