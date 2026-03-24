"""
Единый слой интеграции 6 теоретических источников (PLAN-v51).

Источники:
1. Склярова  — дворцовая кластеризация (8 дворцов по 8 гексаграмм)
2. Фомюк     — антиподальная структура (i, 63-i)
3. Андреев   — треугольная матрица расстояний
4. Касаткин  — 3D диофантовы координаты (4×4×4 куб)
5. Герман    — упаковка P=2^k без коллизий
6. Беляев    — рычажный баланс, комплементарность
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= Конфигурация =========================

@dataclass
class SixSourceConfig:
    """Флаги включения каждого из 6 источников."""
    palace: bool = True       # Склярова
    antipodal: bool = True    # Фомюк
    triangular: bool = True   # Андреев
    kasatkin: bool = True     # Касаткин
    hermann: bool = True      # Герман
    belyaev: bool = True      # Беляев


# ========================= Подмодули ============================

class PalaceSource(nn.Module):
    """Склярова: назначаем токены в 8 дворцов, intra-palace attention."""

    def __init__(self, d_model: int, n_palaces: int = 8):
        super().__init__()
        self.n_palaces = n_palaces
        self.palace_proj = nn.Linear(d_model, n_palaces, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D): дворцово-взвешенный сигнал."""
        B, T, D = x.shape
        # Мягкое назначение во дворцы: (B, T, 8)
        palace_logits = self.palace_proj(x)
        palace_weights = F.softmax(palace_logits, dim=-1)
        # Средние по дворцам: (B, 8, D)
        v = self.value_proj(x)
        palace_means = torch.einsum('btp,btd->bpd', palace_weights, v)
        palace_means = palace_means / (palace_weights.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-8)
        # Обратная проекция: каждый токен получает взвешенную сумму дворцов
        out = torch.einsum('btp,bpd->btd', palace_weights, palace_means)
        return self.out_proj(out)


class AntipodalSource(nn.Module):
    """Фомюк: антиподальная регуляризация — отталкивание от антипода."""

    def __init__(self, d_model: int):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D): антиподальный сигнал."""
        # Антипод = отрицание проекции
        h = self.proj(x)
        antipode = -h
        # Отталкиваем: добавляем разность (h - antipode) = 2h, масштабированную
        return self.scale * (h - antipode)


class TriangularSource(nn.Module):
    """Андреев: треугольный bias на основе позиций."""

    def __init__(self, d_model: int, max_period: int = 64):
        super().__init__()
        self.max_period = max_period
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Linear(d_model, d_model, bias=False)
        # Предвычисляем треугольные позиции
        tri_pos = torch.zeros(max_period)
        for n in range(max_period):
            tri_pos[n] = (n * (n - 1) // 2) % max_period
        self.register_buffer('tri_pos', tri_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D): треугольный позиционный сигнал."""
        B, T, D = x.shape
        # Треугольные расстояния между позициями
        pos = torch.arange(T, device=x.device) % self.max_period
        tp = self.tri_pos[pos]  # (T,)
        dist = (tp.unsqueeze(0) - tp.unsqueeze(1)).abs()  # (T, T)
        # Softmax bias → взвешенная сумма
        bias = -self.scale * dist
        weights = F.softmax(bias, dim=-1)  # (T, T)
        h = self.proj(x)
        return torch.matmul(weights.unsqueeze(0), h)


class KasatkinSource(nn.Module):
    """Касаткин: проекция в 3D диофантовы координаты (4×4×4 куб)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.to_3d = nn.Linear(d_model, 3, bias=False)
        self.from_3d = nn.Linear(3, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D): 3D bottleneck-сигнал."""
        coords = self.to_3d(x)
        # Квантизация к ближайшей точке куба 4×4×4 (значения 0,1,2,3)
        quantized = torch.clamp(torch.round(torch.sigmoid(coords) * 3), 0, 3) / 3.0
        # STE: straight-through estimator
        quantized = coords + (quantized - coords).detach()
        return self.from_3d(quantized)


class HermannSource(nn.Module):
    """Герман: проекция к ближайшей валидной упакованной вершине (P=2^k)."""

    def __init__(self, d_model: int, k: int = 3):
        super().__init__()
        n = 2 ** k  # 8 вершин
        self.proj = nn.Linear(d_model, k, bias=False)
        self.unproj = nn.Linear(k, d_model, bias=False)
        # Кодбук: все вершины {-1, +1}^k
        signs = torch.tensor([-1.0, 1.0])
        import itertools
        codebook = torch.tensor(list(itertools.product(signs.tolist(), repeat=k)))
        self.register_buffer('codebook', codebook)  # (2^k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D): snap к ближайшей вершине гиперкуба."""
        z = self.proj(x)  # (B, T, k)
        # Ближайшая вершина через sign (STE)
        snapped = torch.sign(z)
        snapped = z + (snapped - z).detach()
        return self.unproj(snapped)


class BelyaevSource(nn.Module):
    """Беляев: рычажный баланс — комплементарные активации дают константу."""

    def __init__(self, d_model: int):
        super().__init__()
        self.target_sum = nn.Parameter(torch.tensor(1.0))
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D): балансирующий сигнал."""
        h = self.proj(x)
        # Комплементарная пара: h и (target - h)
        complement = self.target_sum - h
        # Средний сигнал = (h + complement) / 2 = target_sum / 2 (константа)
        # Отклонение от баланса используем как сигнал
        balance_error = h - complement  # = 2h - target_sum
        return 0.5 * balance_error


# ========================= Единый слой ==========================

class SixSourceLayer(nn.Module):
    """Единый слой, объединяющий 6 теоретических источников через learnable gating.

    Каждый источник вычисляет свой сигнал параллельно.
    Softmax-гейт определяет вклад каждого.
    """

    def __init__(self, d_model: int, config: SixSourceConfig | None = None):
        super().__init__()
        if config is None:
            config = SixSourceConfig()
        self.config = config

        # Создаём только включённые источники
        self.sources = nn.ModuleDict()
        if config.palace:
            self.sources['palace'] = PalaceSource(d_model)
        if config.antipodal:
            self.sources['antipodal'] = AntipodalSource(d_model)
        if config.triangular:
            self.sources['triangular'] = TriangularSource(d_model)
        if config.kasatkin:
            self.sources['kasatkin'] = KasatkinSource(d_model)
        if config.hermann:
            self.sources['hermann'] = HermannSource(d_model)
        if config.belyaev:
            self.sources['belyaev'] = BelyaevSource(d_model)

        n_sources = len(self.sources)
        self.n_sources = n_sources

        # Learnable gate: (D) → (n_sources) через глобальный пулинг
        self.gate_proj = nn.Linear(d_model, n_sources, bias=True)
        self.ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, T, D)

        Returns:
            (B, T, D) выход + aux_info dict с весами гейта
        """
        B, T, D = x.shape
        h = self.ln(x)

        # Вычисляем все сигналы параллельно
        source_names = list(self.sources.keys())
        signals = [self.sources[name](h) for name in source_names]  # каждый (B, T, D)

        # Гейт на основе среднего по последовательности
        x_mean = h.mean(dim=1)  # (B, D)
        gate_logits = self.gate_proj(x_mean)  # (B, n_sources)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, n_sources)

        # Взвешенная сумма сигналов
        stacked = torch.stack(signals, dim=2)  # (B, T, n_sources, D)
        gw = gate_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, n_sources, 1)
        combined = (stacked * gw).sum(dim=2)  # (B, T, D)

        out = x + self.out_proj(combined)

        # Вспомогательная информация
        aux_info = {
            f'gate_{name}': gate_weights[:, i].mean().item()
            for i, name in enumerate(source_names)
        }

        return out, aux_info
