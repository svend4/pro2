"""
EXPERIMENTAL — не используется в production моделях.
Все 9 классов используются только в scripts/run_all_extensions.py (benchmark harness).

Расширения YiJing-Transformer: фазы A2–E14.

A2: Multi-Head Geometric Attention (отдельный кодбук на голову)
A3: Hierarchical Z₂¹² quantization (12D гиперкуб)
A4: Geometric MoE (маршрутизация через гексаграммы)
B5: Reed-Muller error correction codes
B6: Walsh-Hadamard spectral analysis
B7: Rate-distortion analysis
C8: Fair comparison utilities
C9: Hypercube weight quantization (PTQ)
D11: Gate dynamics tracker
D12: Attention + hexagram activation maps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from typing import Optional, Dict, List, Tuple

from .geometry import (
    get_trigrams, generate_hexagrams, generate_hypercube,
    RotaryEmbedding, apply_rotary_emb,
    FactoredYiJingQuantizer,
)


# ========================== A2: MULTI-HEAD GEOMETRIC ATTENTION ==========================

class MultiHeadGeometricAttention(nn.Module):
    """
    A2: Каждая голова attention имеет собственный кодбук триграмм.

    Стандартный GeometricAttention использует общие 8 триграммных направлений
    для всех голов. Здесь каждая голова получает:
    - Собственный обучаемый кодбук {-1,+1}^k (k — dim кодбука)
    - Независимую температуру квантизации
    - Свой bias от геометрической структуры

    Это позволяет разным головам специализироваться на разных
    геометрических паттернах.
    """

    def __init__(self, cfg, codebook_dim=3):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.codebook_dim = codebook_dim

        # Проекции
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        # Per-head кодбук: каждая голова → свой набор вершин гиперкуба
        # Проекция head_dim → codebook_dim (shared across heads)
        self.q_geo = nn.Linear(self.head_dim, codebook_dim, bias=False)
        self.k_geo = nn.Linear(self.head_dim, codebook_dim, bias=False)

        # Обучаемые кодбуки: начинаем с триграмм, можно отклоняться
        n_codes = 2 ** codebook_dim
        base_codes = generate_hypercube(codebook_dim)  # (2^k, k)
        # Каждая голова получает свою копию
        self.codebooks = nn.Parameter(
            base_codes.unsqueeze(0).expand(self.n_heads, -1, -1).clone()
        )  # (H, 2^k, k)

        # Per-head температура
        self.temps = nn.Parameter(torch.full((self.n_heads,), 0.3).log())

        # Масштаб геометрического bias
        self.geo_scale = nn.Parameter(torch.tensor(0.1))
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # RoPE
        if cfg.use_rope:
            self.rotary = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=cfg.block_size,
                base=cfg.rope_base,
            )
        else:
            self.rotary = None

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_heads

        # Стандартные проекции
        q = self.q_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)

        # RoPE
        if self.rotary is not None:
            cos, sin = self.rotary(T)
            cos, sin = cos.to(x.device), sin.to(x.device)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Стандартные dot-product scores
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Геометрический bias: проецируем Q/K в codebook space per-head
        # q_flat: (B, T, H*head_dim) → (B, T, H*codebook_dim)
        q_flat = q.transpose(1, 2).reshape(B * T, H, self.head_dim)
        k_flat = k.transpose(1, 2).reshape(B * T, H, self.head_dim)

        # Проецируем каждую голову независимо: (B*T*H, head_dim) → (B*T*H, k)
        q_geo_all = self.q_geo(q_flat.reshape(B * T * H, self.head_dim))
        q_geo_all = q_geo_all.reshape(B * T, H, self.codebook_dim)
        k_geo_all = self.k_geo(k_flat.reshape(B * T * H, self.head_dim))
        k_geo_all = k_geo_all.reshape(B * T, H, self.codebook_dim)

        # Soft-quantize к кодбуку каждой головы
        temps = self.temps.exp().clamp(min=0.01, max=5.0)  # (H,)
        # dists: (B*T, H, 2^k)
        codebooks = self.codebooks  # (H, 2^k, k)
        q_expanded = q_geo_all.unsqueeze(2)  # (B*T, H, 1, k)
        cb_expanded = codebooks.unsqueeze(0)  # (1, H, 2^k, k)
        q_dists = ((q_expanded - cb_expanded) ** 2).sum(-1)  # (B*T, H, 2^k)
        k_dists = ((k_geo_all.unsqueeze(2) - cb_expanded) ** 2).sum(-1)

        # Soft assignments
        q_weights = F.softmax(-q_dists / temps.unsqueeze(0).unsqueeze(-1), dim=-1)  # (B*T, H, 2^k)
        k_weights = F.softmax(-k_dists / temps.unsqueeze(0).unsqueeze(-1), dim=-1)

        # Геометрический bias: скалярное произведение soft assignments
        # shape: (B, T, H, 2^k) → (B, H, T, 2^k)
        q_w = q_weights.reshape(B, T, H, -1).permute(0, 2, 1, 3)  # (B, H, T, 2^k)
        k_w = k_weights.reshape(B, T, H, -1).permute(0, 2, 1, 3)

        # Bias = q_weights @ k_weights^T → мера геометрической близости
        geo_bias = q_w @ k_w.transpose(-2, -1)  # (B, H, T, T)

        scores = scores + self.geo_scale * geo_bias

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ========================== A3: HIERARCHICAL Z₂¹² QUANTIZATION ==========================

class HierarchicalZ2_12Quantizer(nn.Module):
    """
    A3: Прогрессивная квантизация: триграммы → гексаграммы → Z₂¹².

    Три уровня:
    1. Level 1: Z₂³ (8 точек, триграммы) — грубая квантизация
    2. Level 2: Z₂⁶ (64 точки, гексаграммы) — средняя
    3. Level 3: Z₂¹² (4096 точек) — тонкая

    Каждый уровень уточняет предыдущий (residual quantization).
    """

    def __init__(self, d_model, temp=0.3, adaptive_temp=True):
        super().__init__()
        self.d_model = d_model

        # Level 1: Z₂³
        trigrams = generate_hypercube(3)  # (8, 3)
        self.register_buffer('codes_l1', trigrams)
        self.proj_to_l1 = nn.Linear(d_model, 3, bias=False)
        self.proj_from_l1 = nn.Linear(3, d_model, bias=False)

        # Level 2: Z₂⁶ = Z₂³ ⊗ Z₂³ (тензорная факторизация)
        hexagrams = generate_hypercube(6)  # (64, 6)
        self.register_buffer('codes_l2', hexagrams)
        self.proj_to_l2 = nn.Linear(d_model, 6, bias=False)
        self.proj_from_l2 = nn.Linear(6, d_model, bias=False)

        # Level 3: Z₂¹² = Z₂⁶ ⊗ Z₂⁶ (факторизация: два softmax по 64)
        # НЕ храним все 4096 точек — используем тензорную факторизацию
        self.proj_to_l3_upper = nn.Linear(d_model, 6, bias=False)
        self.proj_to_l3_lower = nn.Linear(d_model, 6, bias=False)
        self.proj_from_l3 = nn.Linear(12, d_model, bias=False)

        # Температуры
        if adaptive_temp:
            self.temp_l1 = nn.Parameter(torch.tensor(temp).log())
            self.temp_l2 = nn.Parameter(torch.tensor(temp * 0.7).log())
            self.temp_l3 = nn.Parameter(torch.tensor(temp * 0.5).log())
        else:
            self.register_buffer('temp_l1', torch.tensor(temp).log())
            self.register_buffer('temp_l2', torch.tensor(temp * 0.7).log())
            self.register_buffer('temp_l3', torch.tensor(temp * 0.5).log())

        # Масштабы для каждого уровня
        self.scale_l1 = nn.Parameter(torch.tensor(0.1))
        self.scale_l2 = nn.Parameter(torch.tensor(0.05))
        self.scale_l3 = nn.Parameter(torch.tensor(0.02))

        # Gating: сколько от каждого уровня
        self.level_gate = nn.Linear(d_model, 3, bias=True)

    def _quantize_to_codes(self, z, codes, temp_log):
        """Soft quantization к набору кодов."""
        temp = temp_log.exp().clamp(min=0.01, max=5.0)
        z_flat = z.reshape(-1, z.size(-1))  # (N, dim)
        dists = torch.cdist(z_flat.unsqueeze(0), codes.unsqueeze(0)).squeeze(0)  # (N, n_codes)
        weights = F.softmax(-dists / temp, dim=-1)
        quantized = weights @ codes  # (N, dim)
        return quantized.reshape_as(z)

    def forward(self, x):
        """
        x: (B, T, D) → hierarchical quantized residuals → (B, T, D)
        """
        # Gate: определяет вклад каждого уровня
        gate = F.softmax(self.level_gate(x.detach()), dim=-1)  # (B, T, 3)

        # Level 1: грубая квантизация (8 точек)
        z1 = self.proj_to_l1(x)
        q1 = self._quantize_to_codes(z1, self.codes_l1, self.temp_l1)
        out_l1 = self.proj_from_l1(q1)

        # Level 2: средняя (64 точки), на residual
        z2 = self.proj_to_l2(x)
        q2 = self._quantize_to_codes(z2, self.codes_l2, self.temp_l2)
        out_l2 = self.proj_from_l2(q2)

        # Level 3: тонкая (4096 = 64×64), тензорно-факторизованная
        z3_upper = self.proj_to_l3_upper(x)
        z3_lower = self.proj_to_l3_lower(x)
        q3_upper = self._quantize_to_codes(z3_upper, self.codes_l2, self.temp_l3)
        q3_lower = self._quantize_to_codes(z3_lower, self.codes_l2, self.temp_l3)
        q3_full = torch.cat([q3_upper, q3_lower], dim=-1)  # (B, T, 12)
        out_l3 = self.proj_from_l3(q3_full)

        # Взвешенная сумма уровней
        out = (gate[..., 0:1] * self.scale_l1 * out_l1 +
               gate[..., 1:2] * self.scale_l2 * out_l2 +
               gate[..., 2:3] * self.scale_l3 * out_l3)

        return x + out, {
            'gate_l1': gate[..., 0].mean().item(),
            'gate_l2': gate[..., 1].mean().item(),
            'gate_l3': gate[..., 2].mean().item(),
        }


# ========================== A4: GEOMETRIC MOE ==========================

class HexagramMoE(nn.Module):
    """
    A4: Mixture of Experts с маршрутизацией через гексаграммы.

    64 гексаграммы → 64 эксперта (или K < 64 с группировкой).
    Маршрутизация: проекция в Z₂⁶, soft-assignment к гексаграммам,
    top-k отбор экспертов.

    Тензорная факторизация: вместо 64 независимых экспертов,
    эксперт = (верхняя триграмма FFN) ∘ (нижняя триграмма FFN).
    """

    def __init__(self, d_model, n_experts=64, top_k=4, ffn_mult=4.0,
                 use_factored=True, temp=0.3):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_factored = use_factored

        # Router: проекция в Z₂⁶
        hexagrams = generate_hexagrams()  # (64, 6)
        self.register_buffer('hexagrams', hexagrams)
        self.router_proj = nn.Linear(d_model, 6, bias=False)
        self.router_temp = nn.Parameter(torch.tensor(temp).log())

        ffn_hidden = int(d_model * ffn_mult)
        # Ensure even division for non-factored experts
        if not (use_factored and n_experts == 64):
            expert_dim = ffn_hidden // n_experts
            ffn_hidden = expert_dim * n_experts  # round down to exact multiple

        if use_factored and n_experts == 64:
            # Факторизованные эксперты: 8 верхних + 8 нижних триграммных FFN
            self.upper_w1 = nn.Parameter(torch.randn(8, d_model, ffn_hidden // 8) * 0.02)
            self.upper_w2 = nn.Parameter(torch.randn(8, ffn_hidden // 8, d_model) * 0.02)
            self.lower_w1 = nn.Parameter(torch.randn(8, d_model, ffn_hidden // 8) * 0.02)
            self.lower_w2 = nn.Parameter(torch.randn(8, ffn_hidden // 8, d_model) * 0.02)
        else:
            # Независимые эксперты (для n_experts != 64)
            self.expert_w1 = nn.Parameter(
                torch.randn(n_experts, d_model, ffn_hidden // n_experts) * 0.02
            )
            self.expert_w2 = nn.Parameter(
                torch.randn(n_experts, ffn_hidden // n_experts, d_model) * 0.02
            )

        # Load balancing loss coefficient
        self.balance_coeff = 0.01

    def _route(self, x):
        """Маршрутизация через геометрическую близость к гексаграммам."""
        z = self.router_proj(x)  # (B, T, 6)
        temp = self.router_temp.exp().clamp(min=0.01, max=5.0)

        # Расстояния до гексаграмм
        z_flat = z.reshape(-1, 6)
        dists = torch.cdist(z_flat.unsqueeze(0), self.hexagrams.unsqueeze(0)).squeeze(0)
        router_logits = -dists / temp  # (B*T, 64)

        # Top-k
        topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B*T, top_k)

        return topk_weights, topk_idx, router_logits

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        weights, indices, router_logits = self._route(x)  # (B*T, top_k)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)

        if self.use_factored and self.n_experts == 64:
            # Факторизованные: эксперт[i] = upper[i//8] ∘ lower[i%8]
            for k_idx in range(self.top_k):
                expert_ids = indices[:, k_idx]  # (B*T,)
                w = weights[:, k_idx:k_idx + 1]  # (B*T, 1)
                upper_ids = expert_ids // 8
                lower_ids = expert_ids % 8

                # Batch computation: group by expert
                for uid in range(8):
                    for lid in range(8):
                        mask = (upper_ids == uid) & (lower_ids == lid)
                        if mask.any():
                            x_sel = x_flat[mask]  # (n, D)
                            # upper path
                            h = F.silu(x_sel @ self.upper_w1[uid])
                            h = h @ self.upper_w2[uid]
                            # lower path
                            h2 = F.silu(x_sel @ self.lower_w1[lid])
                            h2 = h2 @ self.lower_w2[lid]
                            # Combine
                            expert_out = h + h2
                            output[mask] += w[mask] * expert_out
        else:
            for k_idx in range(self.top_k):
                expert_ids = indices[:, k_idx]
                w = weights[:, k_idx:k_idx + 1]
                for eid in range(self.n_experts):
                    mask = (expert_ids == eid)
                    if mask.any():
                        x_sel = x_flat[mask]
                        h = F.silu(x_sel @ self.expert_w1[eid])
                        h = h @ self.expert_w2[eid]
                        output[mask] += w[mask] * h

        output = output.reshape(B, T, D)

        # Load balancing loss
        router_probs = F.softmax(router_logits, dim=-1).mean(0)  # (n_experts,)
        uniform = torch.ones_like(router_probs) / self.n_experts
        balance_loss = self.balance_coeff * F.kl_div(
            router_probs.log(), uniform, reduction='sum'
        )

        return output, balance_loss


# ========================== B5: REED-MULLER ERROR CORRECTION ==========================

class ReedMullerCodebook(nn.Module):
    """
    B5: Reed-Muller коды на Z₂⁶.

    RM(r, m) — код порядка r длины 2^m с минимальным расстоянием 2^(m-r).
    RM(1, 6): первый порядок на Z₂⁶ — 128 кодовых слов, d_min = 32.

    Используется для robustness: квантизация к кодовым словам RM(1,6)
    даёт устойчивость к шуму в представлениях.
    """

    def __init__(self, d_model, order=1, length_exp=6, temp=0.3):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.m = length_exp

        # Генерируем кодовые слова RM(order, m) в {-1, +1}
        codewords = self._generate_rm_codewords(order, length_exp)
        self.register_buffer('codewords', codewords)
        self.n_codes = codewords.size(0)
        self.code_len = codewords.size(1)

        self.proj_to = nn.Linear(d_model, self.code_len, bias=False)
        self.proj_from = nn.Linear(self.code_len, d_model, bias=False)
        self.temp = nn.Parameter(torch.tensor(temp).log())
        self.scale = nn.Parameter(torch.tensor(0.05))

    @staticmethod
    def _generate_rm_codewords(order, m):
        """Генерирует кодовые слова RM(order, m) в {-1, +1}^{2^m}."""
        n = 2 ** m
        # Генераторные строки RM(1, m): строка всех единиц + m строк координат
        # Для RM(r, m): все мономы степени <= r

        # Матрица переменных: x_i для i = 0..m-1
        variables = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append((j >> i) & 1)
            variables.append(row)

        # Генерируем все мономы степени <= order
        import itertools
        generators = []
        # Степень 0: вектор всех 1
        generators.append([1] * n)
        # Степени 1..order
        for deg in range(1, order + 1):
            for combo in itertools.combinations(range(m), deg):
                mono = [1] * n
                for idx in combo:
                    for j in range(n):
                        mono[j] *= variables[idx][j]
                generators.append(mono)

        # Все линейные комбинации (в GF(2)) генераторов = кодовые слова
        k = len(generators)
        codewords = []
        for bits in range(2 ** k):
            cw = [0] * n
            for g_idx in range(k):
                if (bits >> g_idx) & 1:
                    for j in range(n):
                        cw[j] ^= generators[g_idx][j]
            # Преобразуем 0/1 → -1/+1
            cw_pm = [1 - 2 * b for b in cw]
            codewords.append(cw_pm)

        return torch.tensor(codewords, dtype=torch.float32)

    def forward(self, x):
        """Квантизация к RM кодовым словам + residual."""
        z = self.proj_to(x)  # (B, T, code_len)
        temp = self.temp.exp().clamp(min=0.01, max=5.0)

        z_flat = z.reshape(-1, self.code_len)
        dists = torch.cdist(z_flat.unsqueeze(0), self.codewords.unsqueeze(0)).squeeze(0)
        weights = F.softmax(-dists / temp, dim=-1)
        quantized = weights @ self.codewords
        quantized = quantized.reshape_as(z)

        return x + self.scale * self.proj_from(quantized), {
            'n_codewords': self.n_codes,
            'code_length': self.code_len,
            'min_distance': 2 ** (self.m - self.order),
        }

    def measure_robustness(self, x, noise_std=0.1, n_trials=10):
        """Измеряет robustness квантизации к шуму."""
        with torch.no_grad():
            z = self.proj_to(x)
            z_flat = z.reshape(-1, self.code_len)

            # Без шума
            dists_clean = torch.cdist(z_flat.unsqueeze(0), self.codewords.unsqueeze(0)).squeeze(0)
            clean_idx = dists_clean.argmin(dim=-1)

            # С шумом
            flips = 0
            for _ in range(n_trials):
                z_noisy = z_flat + torch.randn_like(z_flat) * noise_std
                dists_noisy = torch.cdist(z_noisy.unsqueeze(0), self.codewords.unsqueeze(0)).squeeze(0)
                noisy_idx = dists_noisy.argmin(dim=-1)
                flips += (clean_idx != noisy_idx).float().mean().item()

            return flips / n_trials  # flip rate


# ========================== B6: WALSH-HADAMARD SPECTRAL ANALYSIS ==========================

class WalshHadamardAnalyzer:
    """
    B6: Walsh-Hadamard спектральный анализ представлений.

    Разлагает представления по базису Уолша-Адамара (характеры Z₂ⁿ).
    Показывает, какие «частоты» доминируют в выученных представлениях.
    """

    @staticmethod
    def hadamard_matrix(n):
        """Генерирует матрицу Адамара H_n (порядка 2^n)."""
        H = torch.tensor([[1.0]])
        for _ in range(n):
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        return H

    @staticmethod
    def fast_walsh_hadamard(x):
        """Быстрое преобразование Уолша-Адамара (in-place butterfly)."""
        n = x.size(-1)
        assert n & (n - 1) == 0, "Length must be a power of 2"
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    a = x[..., j].clone()
                    b = x[..., j + h].clone()
                    x[..., j] = a + b
                    x[..., j + h] = a - b
            h *= 2
        return x / math.sqrt(n)

    @classmethod
    def analyze_representations(cls, representations, n_bits=6):
        """
        Анализирует спектр представлений на гиперкубе Z₂ⁿ.

        Args:
            representations: (N, d) — N представлений (N должно быть 2^n)
            n_bits: порядок гиперкуба

        Returns:
            spectrum: (N, d) — коэффициенты Уолша-Адамара
            energy_by_order: dict[int, float] — энергия по порядкам (0=const, 1=linear, ...)
        """
        N, d = representations.shape
        assert N == 2 ** n_bits, f"Expected {2**n_bits} points, got {N}"

        # WHT каждого столбца
        H = cls.hadamard_matrix(n_bits).to(representations.device)  # (N, N)
        spectrum = H @ representations  # (N, d)

        # Энергия по порядкам: порядок = popcount индекса
        energy_by_order = {}
        for idx in range(N):
            order = bin(idx).count('1')
            energy = (spectrum[idx] ** 2).sum().item()
            energy_by_order[order] = energy_by_order.get(order, 0.0) + energy

        total_energy = sum(energy_by_order.values())
        if total_energy > 0:
            energy_by_order = {k: v / total_energy for k, v in energy_by_order.items()}

        return spectrum, energy_by_order

    @classmethod
    def analyze_model_codebook(cls, model):
        """Анализирует спектр кодбука модели (если есть квантизатор)."""
        results = {}
        for name, module in model.named_modules():
            if hasattr(module, 'codebooks') and isinstance(module.codebooks, nn.Parameter):
                cb = module.codebooks.data  # (H, n_codes, dim)
                for h_idx in range(cb.size(0)):
                    codes = cb[h_idx]
                    n_codes = codes.size(0)
                    n_bits = int(math.log2(n_codes))
                    if 2 ** n_bits == n_codes:
                        spectrum, energy = cls.analyze_representations(codes, n_bits)
                        results[f"{name}.head_{h_idx}"] = energy
        return results


# ========================== B7: RATE-DISTORTION ANALYSIS ==========================

class RateDistortionAnalyzer:
    """
    B7: Rate-distortion анализ квантизации к различным кодбукам.

    Сравнивает:
    1. Z₂⁶ гиперкуб (64 точки)
    2. E8 решётка (240 точек)
    3. Обучаемый кодбук (N точек)
    4. Random codebook (N точек)

    Метрики:
    - Rate: log₂(n_codes) бит
    - Distortion: MSE квантизации
    - R-D кривая
    """

    @staticmethod
    @torch.no_grad()
    def compute_rd_point(data, codebook, temp=0.1):
        """
        Вычисляет одну точку R-D кривой.

        Args:
            data: (N, d) — данные для квантизации
            codebook: (K, d) — кодбук

        Returns:
            rate: log₂(K) бит
            distortion: среднее MSE
            usage: энтропия использования кодбука
        """
        K = codebook.size(0)
        rate = math.log2(K)

        # Квантизация
        dists = torch.cdist(data, codebook)  # (N, K)

        # Hard quantization
        min_idx = dists.argmin(dim=-1)  # (N,)
        quantized = codebook[min_idx]
        distortion = ((data - quantized) ** 2).mean().item()

        # Usage entropy
        counts = torch.bincount(min_idx, minlength=K).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()

        return {
            'rate': rate,
            'distortion': distortion,
            'entropy': entropy,
            'utilization': (counts > 0).float().mean().item(),
        }

    @classmethod
    @torch.no_grad()
    def compare_codebooks(cls, data, codebooks_dict):
        """
        Сравнивает несколько кодбуков по R-D.

        Args:
            data: (N, d) — данные
            codebooks_dict: {name: (K, d)} — словарь кодбуков

        Returns:
            results: {name: {rate, distortion, entropy, utilization}}
        """
        results = {}
        for name, codebook in codebooks_dict.items():
            # Если данные и кодбук разной размерности, проецируем
            if data.size(-1) != codebook.size(-1):
                # Простая проекция через PCA-like
                d_data = data.size(-1)
                d_code = codebook.size(-1)
                if d_data > d_code:
                    # Проецируем данные вниз
                    proj = torch.randn(d_data, d_code, device=data.device)
                    proj = torch.linalg.qr(proj)[0]
                    data_proj = data @ proj
                    results[name] = cls.compute_rd_point(data_proj, codebook)
                else:
                    # Pad данные нулями
                    data_pad = F.pad(data, (0, d_code - d_data))
                    results[name] = cls.compute_rd_point(data_pad, codebook)
            else:
                results[name] = cls.compute_rd_point(data, codebook)
        return results


# ========================== C8: FAIR COMPARISON UTILITIES ==========================

def compute_model_params(cfg, model_class):
    """Подсчитывает параметры модели без создания (приблизительно)."""
    model = model_class(cfg)
    total, geo = model.count_parameters()
    del model
    return total, geo


def find_fair_config(target_params, model_class, base_cfg_kwargs,
                     search_dims=[64, 96, 128, 160, 192, 256]):
    """
    C8: Находит конфигурацию с заданным числом параметров.

    Перебирает d_model, чтобы приблизиться к target_params.
    """
    from config.config import YiJingConfig
    best_cfg = None
    best_diff = float('inf')

    for d in search_dims:
        n_heads = max(1, d // 32)
        cfg = YiJingConfig(d_model=d, n_heads=n_heads, **base_cfg_kwargs)
        try:
            total, _ = compute_model_params(cfg, model_class)
        except Exception:
            continue
        diff = abs(total - target_params)
        if diff < best_diff:
            best_diff = diff
            best_cfg = cfg
            best_params = total

    return best_cfg, best_params


# ========================== C9: HYPERCUBE WEIGHT QUANTIZATION (PTQ) ==========================

class HypercubeWeightQuantizer:
    """
    C9: Post-Training Quantization весов через гиперкуб.

    Вместо стандартной uniform quantization (INT8/INT4),
    квантизуем веса к ближайшим вершинам {-1, +1}^k с масштабированием.

    Для каждой группы весов:
    1. Нормализуем
    2. Проецируем в Z₂^k
    3. Квантизуем к ближайшей вершине
    4. Сохраняем scale factor

    Это даёт лучшее приближение для весов с определённой структурой.
    """

    @staticmethod
    @torch.no_grad()
    def quantize_weight(weight, group_size=64, n_bits=3):
        """
        Квантизует матрицу весов гиперкубной квантизацией.

        Args:
            weight: (out_dim, in_dim) — матрица весов
            group_size: размер группы для квантизации
            n_bits: dim гиперкуба (2^n_bits кодовых слов)

        Returns:
            quantized: квантизованные веса
            codes: индексы кодовых слов
            scales: масштабные множители
        """
        codebook = generate_hypercube(n_bits)  # (2^n, n)

        orig_shape = weight.shape
        w = weight.reshape(-1, group_size)  # (n_groups, group_size)

        # PCA-like сжатие до n_bits измерений
        # Используем SVD первых n_bits компонент
        n_groups = w.size(0)
        scales = w.norm(dim=-1, keepdim=True) / math.sqrt(group_size)
        w_norm = w / (scales + 1e-8)

        # Проецируем в n_bits-мерное пространство
        if group_size >= n_bits:
            # Используем случайную проекцию (хешинг)
            torch.manual_seed(42)
            proj = torch.randn(group_size, n_bits, device=weight.device) / math.sqrt(group_size)
            z = w_norm @ proj  # (n_groups, n_bits)

            # Квантизуем к ближайшей вершине
            dists = torch.cdist(z, codebook.to(weight.device))  # (n_groups, 2^n)
            code_idx = dists.argmin(dim=-1)  # (n_groups,)
            quantized_z = codebook.to(weight.device)[code_idx]

            # Восстанавливаем
            proj_pinv = torch.linalg.pinv(proj)  # (n_bits, group_size)
            w_quantized = (quantized_z @ proj_pinv) * scales
        else:
            w_quantized = w.clone()
            code_idx = torch.zeros(n_groups, dtype=torch.long)

        w_quantized = w_quantized.reshape(orig_shape)
        mse = ((weight - w_quantized) ** 2).mean().item()

        return w_quantized, code_idx, scales.squeeze(-1), mse

    @classmethod
    @torch.no_grad()
    def quantize_model(cls, model, group_size=64, n_bits=3, skip_embed=True):
        """Квантизует все Linear слои модели."""
        results = {}
        total_mse = 0.0
        total_params = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if skip_embed and ('tok_emb' in name or 'head' in name):
                    continue
                w = module.weight.data
                q_w, codes, scales, mse = cls.quantize_weight(w, group_size, n_bits)
                module.weight.data = q_w
                results[name] = {
                    'shape': list(w.shape),
                    'mse': mse,
                    'n_groups': codes.size(0),
                }
                total_mse += mse * w.numel()
                total_params += w.numel()

        avg_mse = total_mse / max(total_params, 1)
        return results, avg_mse


# ========================== D11: GATE DYNAMICS TRACKER ==========================

class GateDynamicsTracker:
    """
    D11: Трекер динамики гейтов по шагам обучения.

    Записывает значения гейтов (vanilla vs geometry) на каждом шаге,
    позволяет визуализировать эволюцию.
    """

    def __init__(self):
        self.history = {}  # {layer_name: {step: gate_value}}
        self.steps = []

    def record(self, model, step):
        """Записывает текущие значения гейтов."""
        self.steps.append(step)
        if hasattr(model, 'get_gate_summary'):
            summary = model.get_gate_summary()
            for layer_name, gates in summary.items():
                for gate_name, gate_info in gates.items():
                    if isinstance(gate_info, dict) and 'gate_mean' in gate_info:
                        key = f"{layer_name}/{gate_name}"
                        if key not in self.history:
                            self.history[key] = []
                        self.history[key].append(gate_info['gate_mean'])

    def get_summary(self):
        """Возвращает сводку динамики."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = {
                    'start': values[0],
                    'end': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'trend': 'increasing' if values[-1] > values[0] else 'decreasing',
                }
        return summary

    def to_json(self):
        """Сериализует для сохранения."""
        return {
            'steps': self.steps,
            'history': self.history,
            'summary': self.get_summary(),
        }


# ========================== D12: ATTENTION + HEXAGRAM MAPS ==========================

class HexagramActivationTracker:
    """
    D12: Трекер активаций гексаграмм по типам входных данных.

    Записывает, какие гексаграммы активируются на разных типах текста.
    """

    def __init__(self):
        self.activations = {}  # {data_type: {hexagram_idx: count}}

    @torch.no_grad()
    def record_activations(self, model, x, data_type='unknown'):
        """
        Записывает гексаграммные активации для батча.

        Ищет квантизаторы в модели и записывает nearest-neighbor assignments.
        """
        if data_type not in self.activations:
            self.activations[data_type] = {}

        for name, module in model.named_modules():
            if hasattr(module, 'codebook') or hasattr(module, 'hexagrams'):
                # Получаем проекцию
                if hasattr(module, 'router_proj'):
                    z = module.router_proj(x.float() if x.dtype != torch.float else x)
                elif hasattr(module, 'proj_to'):
                    z = module.proj_to(x.float() if x.dtype != torch.float else x)
                else:
                    continue

                # Ближайшие кодовые слова
                codes = getattr(module, 'hexagrams', getattr(module, 'codebook', None))
                if codes is None:
                    continue

                z_flat = z.reshape(-1, z.size(-1))
                if z_flat.size(-1) != codes.size(-1):
                    continue
                dists = torch.cdist(z_flat.unsqueeze(0), codes.unsqueeze(0)).squeeze(0)
                nearest = dists.argmin(dim=-1)

                for idx in nearest.tolist():
                    self.activations[data_type][idx] = \
                        self.activations[data_type].get(idx, 0) + 1

    def get_summary(self):
        """Возвращает сводку активаций."""
        summary = {}
        for dtype, counts in self.activations.items():
            total = sum(counts.values())
            n_active = len(counts)
            top_5 = sorted(counts.items(), key=lambda x: -x[1])[:5]
            summary[dtype] = {
                'total_activations': total,
                'unique_hexagrams': n_active,
                'top_5': top_5,
                'utilization': n_active / 64 if total > 0 else 0,
            }
        return summary
