"""
И-Цзин геометрия: триграммы, гексаграммы, октограммы, квантизаторы и MoE.

Иерархия гиперкубов:
- 4 биграммы    = {-1, +1}² = Z₂²
- 8 триграмм    = {-1, +1}³ = Z₂³
- 16 тетраграмм = {-1, +1}⁴ = Z₂⁴
- 64 гексаграмм = {-1, +1}⁶ = Z₂⁶
- 256 октограмм = {-1, +1}⁸ = Z₂⁸  (vs 240 корней E8 в R⁸!)

Тензорные факторизации:
- Гексаграмма = верхняя_триграмма ⊗ нижняя_триграмма (8×8)
- Октограмма  = 4 биграммы (4×4×4×4) или 2 тетраграммы (16×16)
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


def generate_hypercube(n: int) -> torch.Tensor:
    """2^n точек — все вершины гиперкуба {-1, +1}^n."""
    signs = [-1.0, 1.0]
    points = torch.tensor(
        list(itertools.product(signs, repeat=n)), dtype=torch.float32
    )
    return points  # (2^n, n)


def generate_octograms() -> torch.Tensor:
    """256 октограмм — все вершины гиперкуба {-1, +1}⁸.
    Сравнимо с E8 (240 корней в R⁸), но с тензорной факторизацией."""
    return generate_hypercube(8)  # (256, 8)


def generate_tetragrams() -> torch.Tensor:
    """16 тетраграмм — {-1, +1}⁴."""
    return generate_hypercube(4)  # (16, 4)


def generate_e8_roots() -> torch.Tensor:
    """
    240 корней решётки E8 в R⁸.

    Два типа корней:
    1) 112 векторов: все перестановки (±1, ±1, 0, 0, 0, 0, 0, 0)
    2) 128 векторов: (±½, ±½, ..., ±½) с чётным числом минусов

    Все корни имеют норму √2. Сравни с октограммами {-1,+1}⁸ (256 точек, норма √8).
    """
    roots = []

    # Тип 1: 112 корней — пары ±1 во всех позициях
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [-1.0, 1.0]:
                for sj in [-1.0, 1.0]:
                    v = [0.0] * 8
                    v[i] = si
                    v[j] = sj
                    roots.append(v)

    # Тип 2: 128 корней — (±½)⁸ с чётным числом минусов
    for signs in itertools.product([-0.5, 0.5], repeat=8):
        n_neg = sum(1 for s in signs if s < 0)
        if n_neg % 2 == 0:
            roots.append(list(signs))

    e8 = torch.tensor(roots, dtype=torch.float32)
    assert e8.shape == (240, 8), f"E8: ожидалось (240, 8), получено {e8.shape}"
    # Проверка: все нормы = √2
    norms = torch.norm(e8, dim=1)
    assert torch.allclose(norms, torch.tensor(2.0).sqrt(), atol=1e-6), "E8: неверные нормы"
    return e8


def compare_e8_vs_hypercube():
    """
    Сравнение E8 решётки с гиперкубами по ключевым метрикам.

    Возвращает dict с метриками: число точек, размерность, нормы,
    минимальное/среднее расстояние, packing density.
    """
    configs = {
        'E8 (240 roots)': generate_e8_roots(),
        'Hexagrams {-1,+1}⁶': generate_hexagrams(),
        'Octograms {-1,+1}⁸': generate_octograms(),
    }
    results = {}
    for name, points in configs.items():
        dists = torch.cdist(points, points)
        mask = ~torch.eye(len(points), dtype=torch.bool)
        d_vals = dists[mask]
        results[name] = {
            'n_points': len(points),
            'dim': points.shape[1],
            'norm': points[0].norm().item(),
            'min_dist': d_vals.min().item(),
            'mean_dist': d_vals.mean().item(),
            'max_dist': d_vals.max().item(),
            'bits': torch.tensor(float(len(points))).log2().item(),
        }
    return results


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


class E8Quantizer(nn.Module):
    """
    Квантизация к 240 корням решётки E8 в R⁸.

    Для прямого сравнения с гиперкубными квантизаторами.
    Сложность: softmax(240) — без факторизации (brute force).
    """
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        e8 = generate_e8_roots()
        self.register_buffer('codebook', e8)  # (240, 8)
        self.register_buffer('codebook_norm_sq', (e8 ** 2).sum(dim=1))

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        # x: (..., 8)
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        cross = x @ self.codebook.T
        dists_sq = x_norm_sq - 2 * cross + self.codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        quantized = weights @ self.codebook
        if self.adaptive_temp:
            return quantized
        return x + (quantized - x).detach()

    def hard_quantize(self, x):
        """Ближайший корень E8."""
        dists = torch.cdist(x.reshape(-1, 8), self.codebook)
        idx = dists.argmin(dim=-1)
        return self.codebook[idx].reshape(x.shape)


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


# ==================== ИЕРАРХИЧЕСКАЯ КВАНТИЗАЦИЯ ====================

class HierarchicalQuantizer(nn.Module):
    """
    Иерархическая квантизация: разбивает вход на группы по k координат
    и квантизует каждую группу к {-1,+1}^k (Product Quantization).

    Примеры конфигураций для 8D входа:
    - n_groups=4, group_dim=2: 4 биграммы (4×4×4×4 = 256 точек)
    - n_groups=2, group_dim=4: 2 тетраграммы (16×16 = 256 точек)
    - n_groups=1, group_dim=8: 1 октограмма (256 точек, без факторизации)

    Для 6D (гексаграммы): n_groups=2, group_dim=3 — стандартная факторизация.
    """
    def __init__(self, total_dim: int, group_dim: int = 2,
                 temp: float = 0.3, adaptive_temp: bool = False):
        super().__init__()
        assert total_dim % group_dim == 0, \
            f"total_dim ({total_dim}) must be divisible by group_dim ({group_dim})"
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim  # число вершин гиперкуба per group

        # Кодбук для одной группы: все вершины {-1,+1}^group_dim
        codebook = generate_hypercube(group_dim)  # (2^k, k)
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
        # x: (..., total_dim)
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)  # (..., G, K)

        quantized_groups = self._soft_quantize_batch(groups)
        quantized = quantized_groups.reshape(*shape, self.total_dim)

        if self.adaptive_temp:
            return quantized
        else:
            return x + (quantized - x).detach()

    def _soft_quantize_batch(self, z):
        # z: (..., G, K)
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)  # (..., G, 1)
        cross = z @ self.codebook.T  # (..., G, 2^K)
        dists_sq = z_norm_sq - 2 * cross + self.codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)  # (..., G, 2^K)
        return weights @ self.codebook  # (..., G, K)

    def hard_quantize(self, x):
        return torch.sign(x)

    def codebook_info(self):
        """Информация о кодбуке для логирования."""
        return {
            'total_dim': self.total_dim,
            'group_dim': self.group_dim,
            'n_groups': self.n_groups,
            'n_codewords_per_group': self.n_codewords,
            'total_codewords': self.n_codewords ** self.n_groups,
            'softmax_ops': self.n_groups * self.n_codewords,
        }


class DeformableQuantizer(nn.Module):
    """
    Деформируемый кодбук: начинаем с идеального гиперкуба {-1,+1}^n,
    но позволяем модели «деформировать» кодовые слова.

    codebook = base_hypercube + learnable_delta

    Это мост между:
    - Фиксированным кодбуком (YiJing, E8) — хорошая инициализация
    - Полностью обучаемым кодбуком (VQ-VAE) — максимальная гибкость
    """
    def __init__(self, total_dim: int, group_dim: int = 3,
                 temp: float = 0.3, deform_scale: float = 0.0):
        super().__init__()
        assert total_dim % group_dim == 0
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim

        base = generate_hypercube(group_dim)
        self.register_buffer('base_codebook', base)  # (2^k, k)

        # Обучаемая деформация (инициализирована нулями)
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
        """Статистика деформации для мониторинга."""
        delta_norm = self.delta.norm().item()
        scale = self.deform_scale.item()
        effective_shift = delta_norm * abs(scale)
        return {
            'delta_norm': delta_norm,
            'deform_scale': scale,
            'effective_shift': effective_shift,
        }


class GumbelQuantizer(nn.Module):
    """
    Gumbel-Softmax квантизация к вершинам гиперкуба.

    Вместо soft attention по расстояниям используем Gumbel-Softmax
    для дискретного выбора ближайшей вершины. При обучении —
    дифференцируемое приближение, при инференсе — hard argmax.

    Поддерживает commitment loss: ||x - sg(quantized)||² + β·||sg(x) - quantized||²
    """
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

        # Лог-температура (обучаемая)
        self.log_temp = nn.Parameter(torch.tensor(temp).log())

        codebook = generate_hypercube(group_dim)
        self.register_buffer('codebook', codebook)
        self.register_buffer('codebook_norm_sq', (codebook ** 2).sum(dim=1))

        # Для commitment loss
        self._commitment_loss = torch.tensor(0.0)

    @property
    def current_temp(self):
        return self.log_temp.exp().clamp(min=0.05, max=5.0)

    def forward(self, x):
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)  # (..., G, K)

        # Расстояния до кодовых слов
        z_norm_sq = (groups * groups).sum(dim=-1, keepdim=True)
        cross = groups @ self.codebook.T
        dists_sq = z_norm_sq - 2 * cross + self.codebook_norm_sq
        logits = -dists_sq  # (..., G, 2^K)

        # Gumbel-Softmax
        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.current_temp, hard=self.hard)
        else:
            # Hard argmax при инференсе
            idx = logits.argmax(dim=-1)  # (..., G)
            weights = F.one_hot(idx, self.n_codewords).float()

        quantized_groups = weights @ self.codebook  # (..., G, K)
        quantized = quantized_groups.reshape(*shape, self.total_dim)

        # Commitment loss
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


# ==================== ГЕКСАГРАММНЫЙ ATTENTION PATTERN ====================

class HexagramAttentionPattern(nn.Module):
    """
    Гексаграммный паттерн attention: 64 фиксированных паттерна
    внимания, каждый соответствует гексаграмме.

    Каждая из 6 линий гексаграммы контролирует один аспект:
    - Линии 1-3 (нижняя триграмма): локальные паттерны (ближние связи)
    - Линии 4-6 (верхняя триграмма): глобальные паттерны (дальние связи)

    +1 = «усиливать связь», -1 = «ослаблять связь»
    """
    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))  # начинаем отключённым

        # 6 базовых паттернов attention (один на линию)
        # Линии 1-3: окна разного размера (локальные)
        # Линии 4-6: шаги разного размера (глобальные/периодические)
        patterns = torch.zeros(6, block_size, block_size)
        for i in range(block_size):
            for j in range(i + 1):  # causal
                dist = i - j
                # Линия 1: ближайший сосед (окно 2)
                patterns[0, i, j] = 1.0 if dist <= 2 else -1.0
                # Линия 2: среднее окно (окно 8)
                patterns[1, i, j] = 1.0 if dist <= 8 else -1.0
                # Линия 3: широкое окно (окно 32)
                patterns[2, i, j] = 1.0 if dist <= 32 else -1.0
                # Линия 4: чётные позиции
                patterns[3, i, j] = 1.0 if dist % 2 == 0 else -1.0
                # Линия 5: каждые 4
                patterns[4, i, j] = 1.0 if dist % 4 == 0 else -1.0
                # Линия 6: начало последовательности
                patterns[5, i, j] = 1.0 if j <= 4 else -1.0

        self.register_buffer('patterns', patterns)  # (6, T, T)

    def forward(self, x, T):
        """
        Возвращает attention bias (B, 1, T, T).
        x: (B, T, D) → проецируем в 6D → взвешиваем паттерны.
        """
        # Средний вектор последовательности → 6D координата
        x_mean = x.mean(dim=1)  # (B, D)
        z6 = self.proj_to_6d(x_mean)  # (B, 6)
        hex_weights = torch.tanh(z6)  # (B, 6) в [-1, 1]

        # Взвешенная комбинация 6 паттернов
        patterns = self.patterns[:, :T, :T]  # (6, T, T)
        bias = torch.einsum('bk,kij->bij', hex_weights, patterns)  # (B, T, T)

        return self.scale * bias.unsqueeze(1)  # (B, 1, T, T)
