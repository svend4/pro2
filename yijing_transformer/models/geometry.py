"""
И-Цзин геометрия: триграммы, гексаграммы, квантизаторы и MoE.

Математическая основа:
- 8 триграмм = вершины куба {-1, +1}³ = группа Z₂³
- 64 гексаграммы = вершины гиперкуба {-1, +1}⁶ = Z₂⁶
- Тензорная факторизация: гексаграмма = верхняя_триграмма ⊗ нижняя_триграмма
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


# ==================== ГЕНЕРАЦИЯ ГЕОМЕТРИИ ====================

def generate_trigrams() -> torch.Tensor:
    """8 триграмм — все вершины куба {-1, +1}³."""
    signs = [-1.0, 1.0]
    trigrams = torch.tensor(
        list(itertools.product(signs, repeat=3)), dtype=torch.float32
    )
    return trigrams  # (8, 3)


def generate_hexagrams() -> torch.Tensor:
    """64 гексаграммы — все вершины гиперкуба {-1, +1}⁶."""
    signs = [-1.0, 1.0]
    hexagrams = torch.tensor(
        list(itertools.product(signs, repeat=6)), dtype=torch.float32
    )
    return hexagrams  # (64, 6)


def verify_yijing_properties(trigrams: torch.Tensor, hexagrams: torch.Tensor):
    """Проверка математических свойств."""
    assert trigrams.shape == (8, 3), f"Триграммы: ожидалось (8,3), получено {trigrams.shape}"
    assert hexagrams.shape == (64, 6), f"Гексаграммы: ожидалось (64,6), получено {hexagrams.shape}"

    # Все координаты ±1
    assert torch.all((trigrams == 1.0) | (trigrams == -1.0)), "Триграммы содержат не ±1"
    assert torch.all((hexagrams == 1.0) | (hexagrams == -1.0)), "Гексаграммы содержат не ±1"

    # Нормы
    tri_norms = torch.norm(trigrams, dim=1)
    assert torch.allclose(tri_norms, torch.tensor(3.0).sqrt()), "Неверные нормы триграмм"
    hex_norms = torch.norm(hexagrams, dim=1)
    assert torch.allclose(hex_norms, torch.tensor(6.0).sqrt()), "Неверные нормы гексаграмм"

    # Уникальность
    assert torch.unique(trigrams, dim=0).shape[0] == 8, "Дублированные триграммы"
    assert torch.unique(hexagrams, dim=0).shape[0] == 64, "Дублированные гексаграммы"

    # Центрирование
    assert torch.norm(trigrams.sum(dim=0)).item() < 1e-6, "Триграммы не центрированы"
    assert torch.norm(hexagrams.sum(dim=0)).item() < 1e-6, "Гексаграммы не центрированы"


# Кэш
_TRIGRAMS = generate_trigrams()
_HEXAGRAMS = generate_hexagrams()
verify_yijing_properties(_TRIGRAMS, _HEXAGRAMS)


def get_trigrams() -> torch.Tensor:
    return _TRIGRAMS


def get_hexagrams() -> torch.Tensor:
    return _HEXAGRAMS


# ==================== КВАНТИЗАТОРЫ ====================

class YiJingQuantizer(nn.Module):
    """
    Наивная квантизация к 64 гексаграммам (вершинам гиперкуба {-1,+1}⁶).
    Аналог E8Quantizer, но с 64 точками в 6D вместо 240 в 8D.
    """
    def __init__(self, temp=0.3):
        super().__init__()
        self.temp = temp
        hexagrams = get_hexagrams()
        self.register_buffer('codebook', hexagrams)  # (64, 6)

    def forward(self, x):
        # x: (..., 6)
        dists_sq = torch.cdist(x, self.codebook.unsqueeze(0).expand(x.shape[0], -1, -1)
                                if x.dim() == 3 else self.codebook) ** 2
        weights = F.softmax(-dists_sq / self.temp, dim=-1)
        quantized = weights @ self.codebook
        return x + (quantized - x).detach()  # STE

    def hard_quantize(self, x):
        """sign(x) — тривиальная квантизация к ближайшей вершине гиперкуба."""
        return torch.sign(x)


class FactoredYiJingQuantizer(nn.Module):
    """
    Факторизованная квантизация: раздельно для верхней и нижней триграммы.

    Сложность: 2 × softmax(8) = O(16) вместо softmax(64) = O(64).
    Это ключевое преимущество тензорной структуры И-Цзин.
    """
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        trigrams = get_trigrams()
        self.register_buffer('trigrams', trigrams)  # (8, 3)
        self.register_buffer('trigrams_norm_sq', (trigrams ** 2).sum(dim=1))  # (8,)

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        # x: (..., 6) → разделяем на верхнюю и нижнюю триграмму
        upper, lower = x[..., :3], x[..., 3:]

        upper_q = self._soft_quantize(upper)
        lower_q = self._soft_quantize(lower)

        quantized = torch.cat([upper_q, lower_q], dim=-1)

        if self.adaptive_temp:
            # Для адаптивной температуры: soft quantized output напрямую
            # (градиент к temp течёт через softmax weights)
            return quantized
        else:
            return x + (quantized - x).detach()  # STE

    def _soft_quantize(self, z):
        # z: (..., 3)
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
        cross = z @ self.trigrams.T
        dists_sq = z_norm_sq - 2 * cross + self.trigrams_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        return weights @ self.trigrams

    def hard_quantize(self, x):
        return torch.sign(x)


# ==================== ТРАНСФОРМАЦИЯ ГЕКСАГРАММ (变卦) ====================

class BianGuaTransform(nn.Module):
    """
    变卦 (Трансформация гексаграмм) — покоординатное отражение в {-1,+1}⁶.

    В И-Цзин переход между гексаграммами происходит через «изменение линий».
    В бинарном пространстве это XOR, в непрерывном — обучаемое отражение.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.proj_from_6d = nn.Linear(6, d_model, bias=False)
        self.change_logits = nn.Parameter(torch.zeros(6))
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        z = self.proj_to_6d(x)
        change_prob = torch.sigmoid(self.change_logits)
        # При prob=0: z не меняется. При prob=1: z → -z (инверсия линии).
        z_transformed = z * (1 - 2 * change_prob)
        return x + self.scale * self.proj_from_6d(z_transformed)


# ==================== ROTARY POSITION EMBEDDINGS ====================

class RotaryEmbedding(nn.Module):
    """RoPE: вращение пар измерений в зависимости от позиции."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
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
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ==================== SwiGLU FFN ====================

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward: более эффективный, чем GELU FFN (LLaMA-style)."""
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ==================== MoE НА ТРИГРАММАХ ====================

class TrigramMoE(nn.Module):
    """
    Mixture of Experts, где каждый эксперт соответствует триграмме.

    Router проецирует вход в 3D (пространство триграмм),
    затем выбирает top-k ближайших триграмм как экспертов.
    Это геометрически мотивированный MoE.
    """
    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model

        if ffn_hidden is None:
            ffn_hidden = 4 * d_model

        # Router: проецируем в пространство триграмм
        trigrams = get_trigrams()[:n_experts]
        self.register_buffer('trigram_dirs', F.normalize(trigrams, p=2, dim=1))
        self.router_proj = nn.Linear(d_model, 3, bias=False)

        # Эксперты
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_hidden, bias=False),
                nn.GELU(),
                nn.Linear(ffn_hidden, d_model, bias=False),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

        # Load balancing loss coefficient
        self.aux_loss_coeff = 0.01

    def forward(self, x):
        """
        x: (B, T, D) → (B, T, D)
        Возвращает также aux_loss для балансировки нагрузки.
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)

        # Router scores через проекцию в 3D пространство триграмм
        z3 = self.router_proj(x_flat)  # (B*T, 3)
        router_logits = z3 @ self.trigram_dirs.T  # (B*T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Применение экспертов
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*T,)
            expert_weights = top_k_probs[:, k]    # (B*T,)

            for e in range(self.n_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        # Aux loss для балансировки (GShard-style)
        tokens_per_expert = router_probs.mean(dim=0)  # (n_experts,)
        uniform = torch.ones_like(tokens_per_expert) / self.n_experts
        aux_loss = self.aux_loss_coeff * F.mse_loss(tokens_per_expert, uniform)

        return output.view(B, T, D), aux_loss
