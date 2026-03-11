"""
v63: NautilusHierarchy — «Каждый сверчок знай свой шесток»

Вместо удаления «вредных» модулей — иерархическое упорядочивание
по семантическому масштабу, как камеры раковины Наутилуса.

Каждый геометрический модуль получает:
  1. Свой масштаб (receptive field / scope)
  2. Свой вес (начальный, обучаемый)
  3. Своё время активации (progressive curriculum)
  4. Свою функцию в иерархии (что именно он обогащает)

Камеры (от мелкого к крупному):
  Chamber 1 — CubeDiagonal:    микро-геометрия ориентации (3D sign patterns)
  Chamber 2 — PrivilegedAxis:  выделение главного направления
  Chamber 3 — DualEmbedding:   инь-ян парность (6D↔3D duality)
  Chamber 4 — D4Equivariant:   малая группа симметрий (D₄ триграмм)
  Chamber 5 — Palace:          блочная структура (8 дворцов)
  Chamber 6 — Heisenberg:      неопределённость в потоке
  Chamber 7 — FlowerOfLife:    глобальная геометрия связей (GAT)

Принцип граммофонной трубы: сигнал проходит от узкой камеры
к широкой, каждая добавляет свой уровень обогащения.
Мелкие камеры — тонкая настройка, крупные — грубое обогащение.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    CubeDiagonalAttention,
    PrivilegedAxisAttention,
    HeisenbergAttention,
    FlowerOfLifeGAT,
    PalaceAttention,
)
from .equivariant import (
    D4EquivariantLayer,
    DualEmbedding,
)


class NautilusChamber(nn.Module):
    """Одна камера Наутилуса — обёртка над геометрическим модулем.

    Добавляет:
    - обучаемый масштаб (начинается малым для мелких камер)
    - gate (контент-зависимая активация)
    - scope projection (ограничение receptive field)
    """

    def __init__(
        self,
        module: nn.Module,
        d_model: int,
        chamber_idx: int,
        n_chambers: int,
        module_type: str,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.module = module
        self.module_type = module_type
        self.chamber_idx = chamber_idx

        # Обучаемый масштаб — мелкие камеры стартуют с меньшим весом
        # Геометрическая прогрессия: chamber 0 → init_scale, chamber 6 → init_scale * 4
        progression = 1.0 + 3.0 * (chamber_idx / max(n_chambers - 1, 1))
        self.scale = nn.Parameter(torch.tensor(init_scale * progression))

        # Content-dependent gate: «нужна ли эта камера для данного токена?»
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.0)  # sigmoid(0) = 0.5

        # Диагностика
        self._last_gate_mean = 0.5
        self._last_scale = init_scale

    def forward(self, x: torch.Tensor, curriculum_mask: float = 1.0) -> torch.Tensor:
        """Применяет камеру с gate и масштабом.

        Args:
            x: (B, T, D) входные embeddings
            curriculum_mask: 0..1, прогрессивная активация (0 = выключен)

        Returns:
            enrichment delta (B, T, D) — добавка к x
        """
        if curriculum_mask < 1e-6:
            return torch.zeros_like(x)

        # Применяем геометрический модуль
        if self.module_type == 'cube_diagonal':
            # CubeDiagonalAttention возвращает bias, конвертируем в enrichment
            bias = self.module.get_bias(x)  # (B, T, T)
            weights = F.softmax(bias, dim=-1)
            enrichment = torch.bmm(weights, x) - x  # delta
        elif self.module_type == 'privileged_axis':
            # PrivilegedAxisAttention возвращает bias
            bias = self.module.get_bias(x)  # (B, T, T)
            weights = F.softmax(bias, dim=-1)
            enrichment = torch.bmm(weights, x) - x  # delta
        elif self.module_type == 'dual_embedding':
            enrichment = self.module(x) - x  # DualEmbedding возвращает x + scale*...
        elif self.module_type == 'd4_equivariant':
            enrichment = self.module(x) - x  # D4 возвращает x + scale*...
        elif self.module_type == 'palace':
            enrichment = self.module(x)  # PalaceAttention возвращает attention output
        elif self.module_type == 'heisenberg':
            enrichment = self.module(x) - x  # delta
        elif self.module_type == 'flower_gat':
            enrichment = self.module(x) - x  # FlowerGAT возвращает x + node_enrichment
        else:
            enrichment = self.module(x) - x

        # Content-dependent gate
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, 1)

        # Итого: scale * gate * curriculum * enrichment
        result = self.scale * gate * curriculum_mask * enrichment

        # Диагностика
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_scale = self.scale.item()

        return result

    def get_stats(self) -> dict:
        return {
            'gate_mean': round(self._last_gate_mean, 4),
            'scale': round(self._last_scale, 4),
            'chamber': self.chamber_idx,
            'type': self.module_type,
        }


class NautilusScheduler:
    """Прогрессивная активация камер — мелкие раньше, крупные позже.

    При training step=0 активна только камера 0.
    К step=warmup_steps активны все камеры.
    Каждая камера «раскрывается» по косинусному расписанию.
    """

    def __init__(self, n_chambers: int, warmup_steps: int = 2000):
        self.n_chambers = n_chambers
        self.warmup_steps = warmup_steps

    def get_masks(self, step: int) -> list:
        """Возвращает список масок [0..1] для каждой камеры."""
        masks = []
        for i in range(self.n_chambers):
            # Камера i начинает раскрываться на step = warmup * i / n_chambers
            # и полностью раскрыта на step = warmup * (i + 1) / n_chambers
            start = self.warmup_steps * i / self.n_chambers
            end = self.warmup_steps * (i + 1) / self.n_chambers
            if step <= start:
                masks.append(0.0)
            elif step >= end:
                masks.append(1.0)
            else:
                # Косинусный ramp-up
                progress = (step - start) / (end - start)
                masks.append(0.5 * (1.0 - math.cos(math.pi * progress)))
        return masks


class NautilusHierarchy(nn.Module):
    """Наутилус-иерархия: 7 камер геометрических модулей.

    Вместо плоского конкурентного применения — последовательное
    обогащение от мелкого масштаба к крупному.

    Сигнал проходит через камеры как звук через граммофонную трубу:
    каждая камера усиливает на своём уровне.
    """

    # Канонический порядок камер (от мелкого к крупному)
    CHAMBER_ORDER = [
        ('cube_diagonal',    'CubeDiagonal — микро-геометрия ориентации'),
        ('privileged_axis',  'PrivilegedAxis — главное направление'),
        ('dual_embedding',   'DualEmbedding — инь-ян парность'),
        ('d4_equivariant',   'D4Equivariant — симметрия триграммы'),
        ('palace',           'Palace — 8 дворцов'),
        ('heisenberg',       'Heisenberg — неопределённость потока'),
        ('flower_gat',       'FlowerOfLife — глобальная геометрия'),
    ]

    def __init__(
        self,
        d_model: int,
        init_scale: float = 0.01,
        warmup_steps: int = 2000,
        mode: str = 'sequential',
        enabled_chambers: list = None,
    ):
        """
        Args:
            d_model: размерность модели
            init_scale: начальный масштаб для камер
            warmup_steps: шагов для полной активации всех камер
            mode: 'sequential' (каскад) или 'parallel' (все параллельно + merge)
            enabled_chambers: список включённых камер (None = все)
        """
        super().__init__()
        self.d_model = d_model
        self.mode = mode

        # Определяем какие камеры включены
        if enabled_chambers is None:
            enabled_chambers = [name for name, _ in self.CHAMBER_ORDER]

        # Создаём модули
        self.chamber_names = []
        chambers = []
        for idx, (name, desc) in enumerate(self.CHAMBER_ORDER):
            if name not in enabled_chambers:
                continue
            module = self._build_module(name, d_model)
            chamber = NautilusChamber(
                module=module,
                d_model=d_model,
                chamber_idx=idx,
                n_chambers=len(self.CHAMBER_ORDER),
                module_type=name,
                init_scale=init_scale,
            )
            chambers.append(chamber)
            self.chamber_names.append(name)

        self.chambers = nn.ModuleList(chambers)
        self.scheduler = NautilusScheduler(
            n_chambers=len(self.chambers),
            warmup_steps=warmup_steps,
        )

        # Для параллельного режима: обучаемое объединение
        if mode == 'parallel':
            self.merge_proj = nn.Linear(
                d_model * len(self.chambers), d_model, bias=False
            )
            nn.init.zeros_(self.merge_proj.weight)

        # Layer norm перед и после (стабилизация)
        self.ln_pre = nn.LayerNorm(d_model)
        self.ln_post = nn.LayerNorm(d_model)

        # Глобальный residual gate
        self.residual_gate = nn.Parameter(torch.tensor(0.1))

        # Текущий шаг (обновляется из training loop)
        self._current_step = 0

    @staticmethod
    def _build_module(name: str, d_model: int) -> nn.Module:
        if name == 'cube_diagonal':
            return CubeDiagonalAttention(d_model)
        elif name == 'privileged_axis':
            return PrivilegedAxisAttention(d_model)
        elif name == 'dual_embedding':
            return DualEmbedding(d_model)
        elif name == 'd4_equivariant':
            return D4EquivariantLayer(d_model)
        elif name == 'palace':
            return PalaceAttention(d_model)
        elif name == 'heisenberg':
            return HeisenbergAttention(d_model)
        elif name == 'flower_gat':
            return FlowerOfLifeGAT(d_model)
        else:
            raise ValueError(f"Unknown chamber type: {name}")

    def set_step(self, step: int):
        """Устанавливает текущий шаг обучения для curriculum."""
        self._current_step = step

    def forward(self, x: torch.Tensor) -> tuple:
        """Применяет Наутилус-иерархию.

        Args:
            x: (B, T, D) token embeddings

        Returns:
            (enriched_x, nautilus_info)
        """
        masks = self.scheduler.get_masks(self._current_step)
        h = self.ln_pre(x)

        if self.mode == 'sequential':
            # Каскадный режим: каждая камера обогащает результат предыдущей
            enrichment = torch.zeros_like(h)
            for i, chamber in enumerate(self.chambers):
                mask = masks[i] if i < len(masks) else 1.0
                delta = chamber(h + enrichment, curriculum_mask=mask)
                enrichment = enrichment + delta

        elif self.mode == 'parallel':
            # Параллельный режим: все камеры независимо, затем merge
            deltas = []
            for i, chamber in enumerate(self.chambers):
                mask = masks[i] if i < len(masks) else 1.0
                delta = chamber(h, curriculum_mask=mask)
                deltas.append(delta)
            # Конкатенация + проекция
            concatenated = torch.cat(deltas, dim=-1)  # (B, T, D*N)
            enrichment = self.merge_proj(concatenated)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Post-norm + residual gate
        enrichment = self.ln_post(enrichment)
        result = x + self.residual_gate * enrichment

        # Собираем диагностику
        info = {
            'step': self._current_step,
            'mode': self.mode,
            'residual_gate': self.residual_gate.item(),
            'masks': masks,
            'chambers': {
                name: chamber.get_stats()
                for name, chamber in zip(self.chamber_names, self.chambers)
            },
        }

        return result, info

    def get_nautilus_stats(self) -> dict:
        """Агрегированная статистика для логирования."""
        stats = {}
        for name, chamber in zip(self.chamber_names, self.chambers):
            s = chamber.get_stats()
            stats[f'nautilus/{name}/gate'] = s['gate_mean']
            stats[f'nautilus/{name}/scale'] = s['scale']
        stats['nautilus/residual_gate'] = round(self.residual_gate.item(), 4)
        masks = self.scheduler.get_masks(self._current_step)
        for i, name in enumerate(self.chamber_names):
            m = masks[i] if i < len(masks) else 1.0
            stats[f'nautilus/{name}/curriculum'] = round(m, 4)
        return stats
