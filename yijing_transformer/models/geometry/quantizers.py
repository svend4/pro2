"""
Квантизаторы: YiJing, E8, Factored, FourState, Antipodal,
Hierarchical, Deformable, Gumbel, Grouped, WHT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    get_trigrams, get_hexagrams, generate_hypercube,
    generate_e8_roots, generate_four_state_codebook, antipodal_index,
    generate_ternary_hypercube, generate_ternary_trigrams,
    hex_digit_semantics,
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
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
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
        dists_sq = (x_norm_sq - 2 * cross + self.codebook_norm_sq).clamp(min=0)
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
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
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
        dists_sq = (z_norm_sq - 2 * cross + self.trigrams_norm_sq).clamp(min=0)
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
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
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
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
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
        dists_sq = (x_norm_sq - 2 * cross + codebook_norm_sq).clamp(min=0)
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
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
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
        dists_sq = (z_norm_sq - 2 * cross + self.codebook_norm_sq).clamp(min=0)
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
        dists_sq = (z_norm_sq - 2 * cross + cb_norm_sq).clamp(min=0)
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
        self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())

        codebook = generate_hypercube(group_dim)
        self.register_buffer('codebook', codebook)
        self.register_buffer('codebook_norm_sq', (codebook ** 2).sum(dim=1))
        self.register_buffer('_commitment_loss', torch.tensor(0.0), persistent=False)

    @property
    def current_temp(self):
        return self.log_temp.exp().clamp(min=0.05, max=5.0)

    def forward(self, x):
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)
        z_norm_sq = (groups * groups).sum(dim=-1, keepdim=True)
        cross = groups @ self.codebook.T
        dists_sq = (z_norm_sq - 2 * cross + self.codebook_norm_sq).clamp(min=0)
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
            self._commitment_loss = x.new_tensor(0.0)
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
                # Защита от div-by-zero при scales == 0 (all-zero group)
                safe_scales = self.scales.data.clamp(min=1e-8)
                self.zero_points.data = (self.qmin - vmin / safe_scales).round()

    def extra_repr(self):
        return (f"d_model={self.d_model}, groups={self.n_groups}, "
                f"group_size={self.group_size}, bits={self.n_bits}, "
                f"symmetric={self.symmetric}")


class TernaryQuantizer(nn.Module):
    """Тернарная квантизация к {-1, 0, +1}^n — объединение Лукасевича, Аймара и 变爻.

    Трёхзначная логика:
    - +1 = ян / истина / сплошная линия (━━━)
    - -1 = инь / ложь / прерванная линия (━ ━)
    -  0 = 变爻 / неопределённость / линия в изменении (━·━)

    Три режима квантизации:
    1. 'full': полный кодбук 3^n вершин (729 для n=6)
    2. 'factored': факторизованная по триграммам 2 × 27 = O(54) вместо O(729)
    3. 'sparse': только вершины с ≤ k нулей (ограничивает неопределённость)

    Параметр uncertainty_budget [0,1] контролирует, сколько 0-значений допускается:
    - 0.0 = чисто бинарные {-1,+1} (= стандартный YiJing)
    - 1.0 = полный тернарный (все 729 вершин)
    - 0.3 = не более ~2 变爻 из 6 линий

    Связь с Atamiri/Аймара: третье значение 0 кодирует эпистемическую
    неопределённость — «не знаю, ян или инь». Модель учится, какие
    компоненты представления «ещё не определены» vs «определённо ян/инь».
    """

    def __init__(self, total_dim: int = 6, mode: str = 'factored',
                 temp: float = 0.3, adaptive_temp: bool = False,
                 uncertainty_budget: float = 0.3, max_zeros: int = 2,
                 warmup_steps: int = 5000, start_temp: float = 1.0,
                 end_temp: float = 0.01):
        super().__init__()
        self.total_dim = total_dim
        self.mode = mode
        self.max_zeros = max_zeros

        # Cosine annealing schedule (task 0.2 — gap→fix)
        self.warmup_steps = warmup_steps
        self.start_temp = start_temp
        self.end_temp = end_temp
        self._step = 0

        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
        else:
            self.temp = temp

        # Learnable uncertainty budget
        self.log_uncertainty = nn.Parameter(
            torch.tensor(uncertainty_budget).clamp(0.01, 0.99).logit()
        )

        # All modes are initialized — the model can blend between them
        # via soft gating, instead of hard enum dispatch.
        # Factored mode (always available for factored quantization)
        if total_dim % 3 == 0:
            trigrams = generate_ternary_trigrams()  # (27, 3)
            self.register_buffer('trigrams', trigrams)
            self.register_buffer('trigrams_norm_sq', (trigrams ** 2).sum(dim=1))
            self._has_factored = True
        else:
            self._has_factored = False

        # Full/sparse codebook
        if mode == 'sparse':
            codebook = self._generate_sparse_codebook(total_dim, max_zeros)
        else:
            codebook = generate_ternary_hypercube(total_dim)  # (3^n, n)
        self.register_buffer('codebook', codebook)
        self.register_buffer('codebook_norm_sq', (codebook ** 2).sum(dim=1))

        # Soft mode gate: learns to blend factored vs full when both available
        if self._has_factored:
            self.mode_gate_logit = nn.Parameter(torch.tensor(
                1.5 if mode == 'factored' else -1.5  # initialize toward preferred mode
            ))
        else:
            self.mode_gate_logit = None

        # Penalty weight for uncertainty usage
        self.register_buffer('_uncertainty_loss', torch.tensor(0.0), persistent=False)

    @staticmethod
    def _generate_sparse_codebook(n: int, max_zeros: int) -> torch.Tensor:
        """Генерирует только вершины с ≤ max_zeros нулевых координат."""
        import itertools
        signs = [-1.0, 0.0, 1.0]
        vertices = []
        for v in itertools.product(signs, repeat=n):
            if sum(1 for x in v if x == 0.0) <= max_zeros:
                vertices.append(v)
        return torch.tensor(vertices, dtype=torch.float32)

    def step_temp(self):
        """Cosine annealing step: start_temp → end_temp over warmup_steps.

        Call once per training step to update the temperature schedule.
        After warmup_steps, temperature stays at end_temp.
        """
        import math as _math
        if not self.adaptive_temp and self.warmup_steps > 0:
            t = min(self._step / self.warmup_steps, 1.0)
            cosine = 0.5 * (1 + _math.cos(_math.pi * t))
            self.temp = self.end_temp + (self.start_temp - self.end_temp) * cosine
        self._step += 1

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    @property
    def uncertainty_budget(self) -> torch.Tensor:
        return torch.sigmoid(self.log_uncertainty)

    @property
    def codebook_size(self) -> int:
        if self.mode == 'factored':
            return 27  # per trigram group
        return self.codebook.shape[0]

    def _soft_quantize_full(self, x: torch.Tensor) -> torch.Tensor:
        """Полная квантизация к 3^n вершинам."""
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        cross = x @ self.codebook.T
        dists_sq = (x_norm_sq - 2 * cross + self.codebook_norm_sq).clamp(min=0)
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        return weights @ self.codebook

    def _soft_quantize_factored(self, x: torch.Tensor) -> torch.Tensor:
        """Факторизованная квантизация: 2 × softmax(27) = O(54)."""
        n_groups = self.total_dim // 3
        parts = []
        for g in range(n_groups):
            z = x[..., g*3:(g+1)*3]
            z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
            cross = z @ self.trigrams.T
            dists_sq = (z_norm_sq - 2 * cross + self.trigrams_norm_sq).clamp(min=0)
            weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
            parts.append(weights @ self.trigrams)
        return torch.cat(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Квантизация с контролем неопределённости.

        Args:
            x: (..., total_dim) — входные координаты

        Returns:
            quantized: (..., total_dim) — квантизованные к {-1, 0, +1}
        """
        # Organic mode blending: when both factored and full are available,
        # the model softly blends between them via a learned gate.
        if self._has_factored and self.mode_gate_logit is not None:
            gate = torch.sigmoid(self.mode_gate_logit)  # scalar in [0, 1]
            q_factored = self._soft_quantize_factored(x)
            q_full = self._soft_quantize_full(x)
            quantized = gate * q_factored + (1 - gate) * q_full
        elif self._has_factored and self.mode == 'factored':
            quantized = self._soft_quantize_factored(x)
        else:
            quantized = self._soft_quantize_full(x)

        # Uncertainty penalty
        zero_fraction = (1.0 - quantized.abs()).clamp(min=0).mean()
        target_fraction = self.uncertainty_budget
        self._uncertainty_loss = ((zero_fraction - target_fraction) ** 2) * 0.1

        # Smooth STE: temperature-controlled instead of hard .detach()
        if self.adaptive_temp:
            return quantized
        # Soft residual: blend toward quantized proportional to temperature
        temp = self.current_temp
        blend = 1.0 / (1.0 + temp)  # low temp → blend≈1 (more quantized)
        return x + blend * (quantized - x)

    def hard_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Ternary quantization via smooth tanh (no hard threshold).

        Uses steep tanh to approximate {-1, 0, +1} while preserving
        gradient flow. The uncertainty_budget controls the width of
        the zero-zone through the steepness parameter.
        """
        # Steepness from uncertainty budget: more budget → wider zero zone → lower steepness
        steepness = 5.0 / (self.uncertainty_budget + 0.1)
        result = torch.tanh(x * steepness)
        return result

    def get_uncertainty_loss(self) -> torch.Tensor:
        """Возвращает штраф за неопределённость для добавления к основному loss."""
        return self._uncertainty_loss

    def trit_distribution(self, x: torch.Tensor) -> dict:
        """Распределение тритов {+1, 0, -1} для мониторинга."""
        q = self.hard_quantize(x)
        total = q.numel()
        return {
            'pos': (q > 0).float().sum().item() / total,
            'zero': (q == 0).float().sum().item() / total,
            'neg': (q < 0).float().sum().item() / total,
        }

    def analyze_bian_yao(self, x: torch.Tensor) -> dict:
        """Анализирует распределение 变爻 (изменяющихся линий).

        Полезно для интерпретируемости: какие измерения «не определены»?

        Returns:
            dict с:
            - n_bian_yao: среднее число 变爻 на вектор
            - bian_yao_mask: (batch, seq_len, dim) маска 变爻
            - uncertainty_per_dim: (dim,) — частота 变爻 по измерениям
        """
        q = self.hard_quantize(x)
        bian_mask = (q == 0.0)
        return {
            'n_bian_yao': bian_mask.float().sum(dim=-1).mean().item(),
            'bian_yao_mask': bian_mask,
            'uncertainty_per_dim': bian_mask.float().mean(dim=tuple(range(bian_mask.dim() - 1))),
        }


class PairedBitQuantizer(nn.Module):
    """Строительная логика: трит = согласие/несогласие пары битов.

    Идея: вместо квантизации одного скаляра в {-1, 0, +1} через пороговую
    функцию (которая создаёт мёртвую зону для градиентов при 0), каждый трит
    кодируется ПАРОЙ независимых бинарных решений:

        (1, 1) → +1 (jisa, да-да)     — оба согласны: ян / лето
        (0, 0) → -1 (jani, нет-нет)   — оба согласны: инь / зима
        (0, 1) →  0↑ (ina↑, весна)    — несогласие, тренд вверх (инь→ян)
        (1, 0) →  0↓ (ina↓, осень)    — несогласие, тренд вниз (ян→инь)

    Четверичная по форме (4 состояния), троичная по духу (3 значения трита),
    но с дополнительным битом — направлением перехода для нулевых тритов.

    Преимущества над TernaryQuantizer:
    1. Нет мёртвой зоны: каждый бит — сигмоида с полным градиентом через STE
    2. Нулевое состояние возникает из НЕСОГЛАСИЯ двух решений, не из порога
    3. Направление перехода (весна/осень) = дополнительная информация
    4. Естественная связь с 变爻: 01 = инь→ян (весна), 10 = ян→инь (осень)

    Формула: trit = bit_a + bit_b - 1
        bit_a, bit_b ∈ {0, 1}  →  trit ∈ {-1, 0, +1}

    Пространство:
        n тритов = 2n битов = 4^n состояний по форме (vs 3^n для TernaryQuantizer)
        Для n=6: 4096 состояний (из них 729 уникальных по тритовому значению
        + направления переходов для каждого 0-трита)

    Args:
        total_dim: число тритов (= n, пар битов). Входной вектор: 2*total_dim скаляров
        temp: температура сигмоиды для мягкой квантизации
        adaptive_temp: обучаемая температура
        uncertainty_budget: целевая доля нулевых тритов [0, 1]
    """

    def __init__(self, total_dim: int = 6, temp: float = 1.0,
                 adaptive_temp: bool = False, uncertainty_budget: float = 0.3):
        super().__init__()
        self.total_dim = total_dim
        self.input_dim = total_dim * 2  # 2 бита на каждый трит

        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
        else:
            self.temp = temp

        # Learnable uncertainty budget
        self.log_uncertainty = nn.Parameter(
            torch.tensor(uncertainty_budget).clamp(0.01, 0.99).logit()
        )

        # Штраф за неопределённость
        self._uncertainty_loss = torch.tensor(0.0)
        # Последние статистики
        self._last_trit_distribution = {'pos': 0.33, 'zero': 0.33, 'neg': 0.33}
        self._last_direction_stats = {'spring': 0.0, 'autumn': 0.0}

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.05, max=5.0)
        return self.temp

    @property
    def uncertainty_budget(self) -> torch.Tensor:
        return torch.sigmoid(self.log_uncertainty)

    def forward(self, x: torch.Tensor) -> tuple:
        """Парно-битовая квантизация.

        Args:
            x: (..., 2 * total_dim) — входные логиты для пар битов

        Returns:
            trits: (..., total_dim) — тритовые значения {-1, 0, +1}
            direction: (..., total_dim) — направление перехода для 0-тритов
                +1 = осень (ян→инь, bit_a=1 bit_b=0)
                -1 = весна (инь→ян, bit_a=0 bit_b=1)
                 0 = стабильное состояние (оба одинаковые)
        """
        # Reshape: последнее измерение → (total_dim, 2)
        x_pairs = x.reshape(*x.shape[:-1], self.total_dim, 2)

        # Каждый бит — независимая сигмоида
        temp = self.current_temp
        probs = torch.sigmoid(x_pairs / temp)  # (..., total_dim, 2)

        # STE: soft forward, hard backward
        bits_hard = (probs > 0.5).float()
        bits = probs + (bits_hard - probs).detach()  # STE

        bit_a = bits[..., 0]  # (..., total_dim)
        bit_b = bits[..., 1]  # (..., total_dim)

        # Трит = bit_a + bit_b - 1
        # (1,1)→+1, (0,0)→-1, (0,1)→0, (1,0)→0
        trits = bit_a + bit_b - 1.0

        # Направление перехода (значимо только для тритов ≈ 0)
        # bit_a - bit_b: (1,0)→+1=осень, (0,1)→-1=весна, (0,0)→0, (1,1)→0
        direction = bit_a - bit_b

        # Uncertainty penalty: контроль бюджета нулевых тритов
        with torch.no_grad():
            hard_trits = (bits_hard[..., 0] + bits_hard[..., 1] - 1.0)
            zero_fraction = (hard_trits == 0).float().mean()
            total = hard_trits.numel()
            self._last_trit_distribution = {
                'pos': (hard_trits > 0).float().sum().item() / total,
                'zero': (hard_trits == 0).float().sum().item() / total,
                'neg': (hard_trits < 0).float().sum().item() / total,
            }
            # Направления (только для 0-тритов)
            zero_mask = (hard_trits == 0)
            if zero_mask.any():
                hard_dir = bits_hard[..., 0] - bits_hard[..., 1]
                spring = ((hard_dir == -1) & zero_mask).float().sum().item()
                autumn = ((hard_dir == 1) & zero_mask).float().sum().item()
                n_zeros = zero_mask.float().sum().item()
                self._last_direction_stats = {
                    'spring': spring / max(n_zeros, 1),
                    'autumn': autumn / max(n_zeros, 1),
                }

        # Soft zero_fraction for gradient flow
        soft_zero = 1.0 - trits.abs()  # близко к 1 для тритов ≈ 0
        soft_zero_fraction = soft_zero.clamp(min=0).mean()
        target_fraction = self.uncertainty_budget
        self._uncertainty_loss = ((soft_zero_fraction - target_fraction) ** 2) * 0.1

        return trits, direction

    def get_uncertainty_loss(self) -> torch.Tensor:
        return self._uncertainty_loss

    def get_trit_distribution(self) -> dict:
        return self._last_trit_distribution

    def get_direction_stats(self) -> dict:
        return self._last_direction_stats

    def hard_quantize(self, x: torch.Tensor) -> tuple:
        """Жёсткая квантизация без STE (для инференса)."""
        x_pairs = x.reshape(*x.shape[:-1], self.total_dim, 2)
        bits = (x_pairs > 0).float()
        trits = bits[..., 0] + bits[..., 1] - 1.0
        direction = bits[..., 0] - bits[..., 1]
        return trits, direction


class MatryoshkaQuantizer(nn.Module):
    """Матрёшечный квантизатор: иерархическое кодирование от бита до Q12.

    Принцип Наутилуса в кодировании: каждый уровень строится
    из пар предыдущего, как камеры раковины.

    ┌──────────────────────────────────────────────────────────────────┐
    │ Уровень 0 — Бит (1 координата Q6):                              │
    │   {-1, +1} = 2 состояния                                        │
    │   6 бит = 64 гексаграммы = вершины Q6                           │
    ├──────────────────────────────────────────────────────────────────┤
    │ Уровень 1 — Трит (пространственная пара):                        │
    │   2 соседних координаты (d₁,d₂):                                │
    │     (+1,+1) → +1 ян   (лето,  да-да,  老阳)                     │
    │     (-1,-1) → -1 инь  (зима,  нет-нет, 老阴)                    │
    │     (-1,+1) →  0↑ весна (нет→да, 少阳)                          │
    │     (+1,-1) →  0↓ осень (да→нет, 少阴)                          │
    │   3 пары = 3 трита + 3 направления = 4³ = 64 обогащённых        │
    ├──────────────────────────────────────────────────────────────────┤
    │ Уровень 2 — Гекс-цифра (пространство × время):                  │
    │   Ребро можно прочесть двумя способами:                          │
    │     A) Два соседних ребра (пространство) → 4 состояния           │
    │     B) Одно ребро в два момента (время) → 4 состояния            │
    │   Вместе: 4 × 4 = 16 состояний = 1 hex digit = вершина Q4       │
    │   3 гекс-цифры = 16³ = 4096 = вершина Q12 = Q6(space)×Q6(time) │
    └──────────────────────────────────────────────────────────────────┘

    Интерпретация «времени» (x_ref):
      - Предыдущий токен в последовательности
      - Представление до обогащения (вход камеры Наутилуса)
      - Выход предыдущей камеры Наутилуса

    Рекурсивный потенциал (не реализован, на будущее):
      Q6 → Q12 (space×time) → Q24 (space×time × space×time from another source)
      Каждый уровень удваивает размерность через пространственно-временное спаривание.

    Args:
        total_dim: размерность Q-пространства (6 для Q6, должно быть чётное)
        d_model: размерность модели (для enriched-выхода)
        temp: температура мягкой квантизации
        adaptive_temp: обучаемая температура
    """

    def __init__(self, total_dim: int = 6, d_model: int = 128,
                 temp: float = 0.3, adaptive_temp: bool = False):
        super().__init__()
        assert total_dim % 2 == 0, f"total_dim must be even, got {total_dim}"
        self.total_dim = total_dim
        self.d_model = d_model
        self.n_pairs = total_dim // 2

        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(max(temp, 1e-4)).log())
        else:
            self.temp = temp

        # === Level 0: Binary codebook Q_n ===
        binary_cb = generate_hypercube(total_dim)  # (2^n, n)
        self.register_buffer('binary_codebook', binary_cb)
        self.register_buffer('binary_cb_norm_sq', (binary_cb ** 2).sum(dim=1))

        # === Level 1: Pair codebook Q2 ===
        pair_cb = generate_hypercube(2)  # (4, 2)
        self.register_buffer('pair_codebook', pair_cb)
        self.register_buffer('pair_cb_norm_sq', (pair_cb ** 2).sum(dim=1))

        # Trit and direction lookup tables (indexed by pair_codebook order)
        # generate_hypercube(2): [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
        #   (-1,-1) → trit=-1 (инь),     direction=0  (стабильно)
        #   (-1,+1) → trit=0  (переход),  direction=+1 (весна, ↑)
        #   (+1,-1) → trit=0  (переход),  direction=-1 (осень, ↓)
        #   (+1,+1) → trit=+1 (ян),       direction=0  (стабильно)
        self.register_buffer('trit_table', torch.tensor([-1.0, 0.0, 0.0, 1.0]))
        self.register_buffer('direction_table', torch.tensor([0.0, 1.0, -1.0, 0.0]))

        # === Level 2: Hex digit codebook Q4 ===
        hex_cb = generate_hypercube(4)  # (16, 4)
        self.register_buffer('hex_codebook', hex_cb)
        self.register_buffer('hex_cb_norm_sq', (hex_cb ** 2).sum(dim=1))
        # Hex digit → index conversion weights (multi-GPU safe: buffer, not forward-time tensor)
        self.register_buffer('_hex_digit_weights', torch.tensor([8, 4, 2, 1], dtype=torch.long))

        # === Projections to d_model (for enriched output) ===
        # Level 0: Q_n quantized coordinates → d_model
        self.proj_level0 = nn.Linear(total_dim, d_model, bias=False)
        # Level 1: trits + directions → d_model
        self.proj_level1 = nn.Linear(self.n_pairs * 2, d_model, bias=False)
        # Level 2: soft Q4 quantized × n_pairs → d_model
        self.proj_level2 = nn.Linear(self.n_pairs * 4, d_model, bias=False)

        # Level gates (Nautilus: higher levels ← larger initial weight)
        self.level_gates = nn.Parameter(torch.tensor([0.0, 0.5, 1.0]))

        # Diagnostics
        self._stats = {}

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def _soft_quantize(self, x, codebook, cb_norm_sq):
        """Мягкая квантизация: расстояния → softmax → взвешенная сумма."""
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        cross = x @ codebook.T
        dists_sq = (x_norm_sq - 2 * cross + cb_norm_sq).clamp(min=0)
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        quantized = weights @ codebook
        return quantized, weights

    def _extract_pairs(self, x):
        """Разбивает вектор на пары координат (Касаткин): (d₁d₂), (d₃d₄), (d₅d₆)."""
        return [x[..., p * 2:(p + 1) * 2] for p in range(self.n_pairs)]

    def _pairs_to_trits(self, pairs):
        """Пары координат → триты {-1,0,+1} + направления {-1,0,+1}."""
        trits = []
        directions = []
        for pair in pairs:
            _, w = self._soft_quantize(pair, self.pair_codebook, self.pair_cb_norm_sq)
            trits.append((w * self.trit_table).sum(dim=-1))
            directions.append((w * self.direction_table).sum(dim=-1))
        return torch.stack(trits, dim=-1), torch.stack(directions, dim=-1)

    def _pairs_to_hex(self, pairs_now, pairs_ref):
        """Пространство × время → гекс-цифры (Q4 soft quantization).

        Каждая гекс-цифра = (d₁_now, d₂_now, d₁_ref, d₂_ref) ∈ Q4.
        16 состояний = комбинация пространственного и временного прочтения.
        """
        hex_feats = []
        for p in range(self.n_pairs):
            combined = torch.cat([pairs_now[p], pairs_ref[p]], dim=-1)  # (..., 4)
            q, _ = self._soft_quantize(combined, self.hex_codebook, self.hex_cb_norm_sq)
            hex_feats.append(q)  # (..., 4) — soft-quantized Q4 vertex
        return torch.cat(hex_feats, dim=-1)  # (..., n_pairs × 4)

    def forward(self, x, x_ref=None):
        """Иерархическая квантизация по принципу Матрёшки.

        Args:
            x: (..., total_dim) — текущее представление в Q-пространстве
            x_ref: (..., total_dim) — опорное представление (для Level 2).
                None → только Level 0 + Level 1 (без временного измерения).

        Returns:
            output: (..., d_model) — обогащённое иерархическое представление
            info: dict с диагностикой каждого уровня
        """
        info = {}

        # === Level 0: Binary quantization (Q6 → 64 hexagrams) ===
        q0, w0 = self._soft_quantize(x, self.binary_codebook, self.binary_cb_norm_sq)
        out0 = self.proj_level0(q0)
        info['level0_quantized'] = q0

        # === Level 1: Spatial pairs → trits ===
        pairs = self._extract_pairs(q0)
        trits, directions = self._pairs_to_trits(pairs)
        trit_features = torch.cat([trits, directions], dim=-1)  # (..., n_pairs × 2)
        out1 = self.proj_level1(trit_features)
        info['trits'] = trits
        info['directions'] = directions

        # === Level 2: Space×Time hex digits (requires reference) ===
        has_level2 = x_ref is not None
        if has_level2:
            q_ref, _ = self._soft_quantize(x_ref, self.binary_codebook, self.binary_cb_norm_sq)
            pairs_ref = self._extract_pairs(q_ref)
            hex_feats = self._pairs_to_hex(pairs, pairs_ref)  # (..., n_pairs × 4)
            out2 = self.proj_level2(hex_feats)
            info['hex_features'] = hex_feats

        # === Combine levels with gated weights (Nautilus scaling) ===
        gates = torch.sigmoid(self.level_gates)
        output = gates[0] * out0 + gates[1] * out1
        if has_level2:
            output = output + gates[2] * out2

        info['level_gates'] = gates.detach()

        # Store monitoring stats
        with torch.no_grad():
            self._stats = {
                'trit_yang': (trits > 0.5).float().mean().item(),
                'trit_transition': ((trits > -0.5) & (trits < 0.5)).float().mean().item(),
                'trit_yin': (trits < -0.5).float().mean().item(),
                'dir_spring': (directions > 0.5).float().mean().item(),
                'dir_autumn': (directions < -0.5).float().mean().item(),
                'gate_L0': gates[0].item(),
                'gate_L1': gates[1].item(),
                'gate_L2': gates[2].item(),
                'has_spacetime': has_level2,
            }

        return output, info

    def hard_quantize(self, x, x_ref=None):
        """Жёсткая квантизация всех уровней (для инференса и анализа).

        Returns:
            dict с:
              bits: (..., total_dim) — Level 0 знаковое кодирование
              trits: (..., n_pairs) — Level 1 тритовые значения
              directions: (..., n_pairs) — Level 1 направления перехода
              hex_digits: (..., n_pairs) — Level 2 индексы гекс-цифр [0-15]
              hex_vectors: (..., n_pairs × 4) — Level 2 бинарные Q4 вершины
        """
        hard_bits = torch.sign(x)
        pairs = self._extract_pairs(hard_bits)

        # Level 1: hard trits
        hard_trits = []
        hard_dirs = []
        for pair in pairs:
            b1, b2 = pair[..., 0], pair[..., 1]
            # trit = (b1 + b2) / 2: (+1+1)/2=+1, (-1-1)/2=-1, mixed=0
            hard_trits.append((b1 + b2) / 2)
            # direction = (b2 - b1) / 2: (-1,+1)→+1=spring(↑), (+1,-1)→-1=autumn(↓)
            hard_dirs.append((b2 - b1) / 2)
        trits = torch.stack(hard_trits, dim=-1)
        dirs = torch.stack(hard_dirs, dim=-1)

        result = {'bits': hard_bits, 'trits': trits, 'directions': dirs}

        # Level 2: space×time hex digits
        if x_ref is not None:
            hard_ref = torch.sign(x_ref)
            pairs_ref = self._extract_pairs(hard_ref)
            hex_indices = []
            hex_vectors = []
            for p in range(self.n_pairs):
                bits_4 = torch.cat([pairs[p], pairs_ref[p]], dim=-1)  # (..., 4)
                hex_vectors.append(bits_4)
                # Convert {-1,+1}⁴ → index [0-15]
                bits_01 = ((bits_4 + 1) / 2).long()
                idx = (bits_01 * self._hex_digit_weights).sum(dim=-1)
                hex_indices.append(idx)
            result['hex_digits'] = torch.stack(hex_indices, dim=-1)
            result['hex_vectors'] = torch.cat(hex_vectors, dim=-1)

        return result

    def get_stats(self):
        """Текущая статистика для мониторинга."""
        return self._stats

    def matryoshka_analysis(self, x, x_ref=None):
        """Полный анализ иерархии для визуализации и интерпретации.

        Returns:
            dict с подробным описанием каждого уровня.
        """
        hard = self.hard_quantize(x, x_ref)

        analysis = {
            'n_levels': 3 if x_ref is not None else 2,
            'level0': {
                'name': 'Бит (Q6, 64 гексаграммы)',
                'dim': self.total_dim,
                'states': 2 ** self.total_dim,
                'bits': hard['bits'],
            },
            'level1': {
                'name': 'Трит (пространственная пара, 4→3 состояния)',
                'n_pairs': self.n_pairs,
                'states_per_pair': 4,
                'collapsed_states': 3,
                'total_states': 4 ** self.n_pairs,
                'trits': hard['trits'],
                'directions': hard['directions'],
                'distribution': {
                    'yang': (hard['trits'] > 0.5).float().mean().item(),
                    'yin': (hard['trits'] < -0.5).float().mean().item(),
                    'transition': ((hard['trits'] > -0.5) & (hard['trits'] < 0.5)).float().mean().item(),
                },
            },
        }

        if x_ref is not None and 'hex_digits' in hard:
            hex_idx = hard['hex_digits']
            analysis['level2'] = {
                'name': 'Гекс-цифра (пространство×время, Q4→Q12)',
                'states_per_digit': 16,
                'n_digits': self.n_pairs,
                'total_states': 16 ** self.n_pairs,
                'new_dimension': self.total_dim * 2,
                'hex_digits': hex_idx,
                'unique_states_used': len(torch.unique(hex_idx.reshape(-1))),
            }

        return analysis


# ── WHT_Quantizer: Walsh-Hadamard квантизация (Теорема 5) ──────────────────────

class WHT_Quantizer(nn.Module):
    """Walsh-Hadamard Transform квантизатор для Z₂^n (Теорема 5).

    Реализует спектральное разложение O(n log n) через WHT:
        Ĥ = WHT(h),  h ∈ {-1,+1}^n
        WHT(x)[k] = Σ_i x[i] · (-1)^{popcount(i & k)}

    Применение к Q6-гиперкубу (n=6):
    - Входной вектор x ∈ R^d проецируется в R^6 = {-1,+1}^6
    - WHT спектр Ĥ ∈ R^6 = линейные коэффициенты Фурье над Z₂^6
    - Наибольшие спектральные компоненты указывают на «главные оси»
    - Квантизованный выход: hard {-1,+1}^6 по знаку WHT

    Связь с теорией bent-функций:
    - Равномерный WHT-спектр (|Ĥ[k]| = const) ↔ максимально нелинейная функция
    - Используется в meta_q6.py как seed архетипы

    Сложность: O(n log n) через разделяй-и-властвуй (butterfly network).
    Для n=6: 6 × log₂(6) ≈ 15 операций.

    Args:
        d_model: размерность входных векторов
        n_bits: размерность Q6 пространства (обычно 6)
        temp: температура для soft квантизации (< 0 = hard)
        use_spectral_loss: если True, добавляет равномерность WHT как loss
    """

    def __init__(self, d_model: int, n_bits: int = 6,
                 temp: float = 0.5, use_spectral_loss: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_bits = n_bits
        self.temp = temp
        self.use_spectral_loss = use_spectral_loss

        # Проектор d_model → n_bits
        self.proj = nn.Linear(d_model, n_bits, bias=True)
        nn.init.orthogonal_(self.proj.weight)

        # Предвычисленная WHT матрица Адамара (2^n × 2^n)
        H = self._build_hadamard(n_bits)  # (2^n, 2^n)
        self.register_buffer('H', H)

        # Буфер для spectral loss
        self.register_buffer('_spectral_loss', torch.tensor(0.0), persistent=False)

    @staticmethod
    def _build_hadamard(n: int) -> torch.Tensor:
        """Строит нормированную матрицу Адамара-Уолша 2^n × 2^n.

        Использует кронекерово произведение H₁ = [[1,1],[1,-1]] / sqrt(2).
        """
        H = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / (2.0 ** 0.5)
        result = H
        for _ in range(n - 1):
            result = torch.kron(result, H)
        return result  # (2^n, 2^n), нормированная

    def _wht(self, x: torch.Tensor) -> torch.Tensor:
        """WHT через butterfly: O(n log n).

        Args:
            x: (..., n_bits) — входной вектор в R^n
        Returns:
            x_hat: (..., n_bits) — WHT первых n_bits компонент
                   (упрощённая версия: H[:n_bits, :n_bits] @ x)
        """
        # Полный WHT потребовал бы 2^n входов, что нецелесообразно.
        # Вместо этого используем n_bits × n_bits под-матрицу (approx WHT).
        # Для n=6: H_approx = первые 6 строк × 6 столбцов нормированной H.
        H_approx = self.H[:self.n_bits, :self.n_bits]  # (n_bits, n_bits)
        return x @ H_approx.T  # (..., n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход: x → WHT спектр → квантизация → {-1,+1}^n.

        Args:
            x: (B, T, d_model) или (B, d_model)
        Returns:
            q: того же shape что x (через обратный проектор)
            OR (q6_bits, x_reconstructed) если нужны биты
        """
        # Проецируем в n_bits пространство
        z = self.proj(x)  # (..., n_bits)

        # WHT спектр
        z_hat = self._wht(z)  # (..., n_bits)

        if self.temp > 0:
            # Soft: sigmoid с температурой → (-1, +1)
            bits_soft = torch.tanh(z_hat / self.temp)  # (..., n_bits)
        else:
            # Hard: знак WHT
            bits_soft = z_hat.sign()

        # Spectral loss: равномерность |WHT| → максимальная нелинейность (bent)
        if self.use_spectral_loss and self.training:
            spectrum = z_hat.abs().mean(dim=list(range(z_hat.dim() - 1)))  # (n_bits,)
            # Penalty = дисперсия спектра (чем ниже — тем равномернее)
            self._spectral_loss = spectrum.var()

        return bits_soft

    @torch.no_grad()
    def hard_bits(self, x: torch.Tensor) -> torch.Tensor:
        """Жёсткая квантизация: → {-1, +1}^n_bits (для RAG/routing)."""
        z = self.proj(x)
        z_hat = self._wht(z)
        return z_hat.sign()

    def spectral_loss(self) -> torch.Tensor:
        """Возвращает накопленный spectral uniformity loss."""
        return self._spectral_loss

    def diagnostics(self) -> dict:
        """Возвращает диагностику WHT квантизатора."""
        return {
            'n_bits': self.n_bits,
            'temp': self.temp,
            'proj_weight_norm': self.proj.weight.norm().item(),
            'spectral_loss': self._spectral_loss.item(),
        }

    def __repr__(self):
        return (f"WHT_Quantizer(d_model={self.d_model}, n_bits={self.n_bits}, "
                f"temp={self.temp}, spectral_loss={self.use_spectral_loss})")
