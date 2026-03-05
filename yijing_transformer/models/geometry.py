"""
И-Цзин геометрия: триграммы, гексаграммы и квантизаторы.

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
    def __init__(self, temp=0.3):
        super().__init__()
        self.temp = temp
        trigrams = get_trigrams()
        self.register_buffer('trigrams', trigrams)  # (8, 3)
        self.register_buffer('trigrams_norm_sq', (trigrams ** 2).sum(dim=1))  # (8,)

    def forward(self, x):
        # x: (..., 6) → разделяем на верхнюю и нижнюю триграмму
        upper, lower = x[..., :3], x[..., 3:]

        upper_q = self._soft_quantize(upper)
        lower_q = self._soft_quantize(lower)

        quantized = torch.cat([upper_q, lower_q], dim=-1)
        return x + (quantized - x).detach()  # STE

    def _soft_quantize(self, z):
        # z: (..., 3)
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
        cross = z @ self.trigrams.T
        dists_sq = z_norm_sq - 2 * cross + self.trigrams_norm_sq
        weights = F.softmax(-dists_sq / self.temp, dim=-1)
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
