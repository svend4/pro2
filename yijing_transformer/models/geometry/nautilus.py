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

        # Learnable scale per chamber — initialized with soft progression
        # but free to adapt during training (no fixed geometric ratio).
        # softplus ensures positive scale without hard clamp.
        progression = chamber_idx / max(n_chambers - 1, 1)
        init_value = math.log(math.exp(init_scale * (1.0 + 3.0 * progression)) - 1.0 + 1e-6)
        self._scale_param = nn.Parameter(torch.tensor(init_value))
        # Learnable progression bias — replaces the hard 3.0 multiplier
        self._progression_logit = nn.Parameter(torch.tensor(0.0))

        # Content-dependent gate: «нужна ли эта камера для данного токена?»
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        # Neutral init: sigmoid(0) = 0.5, letting the model learn the preference
        nn.init.zeros_(self.gate_proj.bias)

        # Диагностика
        self._last_gate_mean = 0.5
        self._last_scale = init_scale

    @property
    def scale(self):
        """Positive scale via softplus — learnable without hard constraints."""
        return F.softplus(self._scale_param)

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

        # Apply geometric module organically — unified enrichment extraction.
        # Instead of hard if/elif dispatch on string type, all modules
        # are handled through two generic protocols:
        #   1. Bias-producing modules (have get_bias) → softmax attention delta
        #   2. Direct modules (return transformed x) → subtract identity for delta
        # No string matching needed — the module's interface decides.
        if hasattr(self.module, 'get_bias'):
            # Bias-producing modules (CubeDiagonal, PrivilegedAxis, etc.)
            bias = self.module.get_bias(x)  # (B, T, T)
            T = x.shape[1]
            causal = torch.tril(torch.ones(T, T, device=x.device))
            bias = bias.masked_fill(causal.unsqueeze(0) == 0, float('-inf'))
            weights = F.softmax(bias, dim=-1)
            weights = weights.nan_to_num(0.0)
            enrichment = torch.bmm(weights, x) - x  # delta
        else:
            # Direct modules — try forward, extract delta
            module_out = self.module(x)
            if module_out.shape == x.shape:
                enrichment = module_out - x  # delta from residual-style modules
            else:
                enrichment = module_out  # raw output (e.g. PalaceAttention)

        # Content-dependent gate
        gate = torch.sigmoid(self.gate_proj(x))  # (B, T, 1)

        # Итого: scale * gate * curriculum * enrichment
        result = self.scale * gate * curriculum_mask * enrichment

        # Диагностика
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_scale = self.scale.item() if isinstance(self.scale, torch.Tensor) else self.scale

        return result

    def get_stats(self) -> dict:
        return {
            'gate_mean': round(self._last_gate_mean, 4),
            'scale': round(self._last_scale, 4),
            'chamber': self.chamber_idx,
            'type': self.module_type,
        }


class NautilusScheduler:
    """Smooth progressive chamber activation — no hard phase boundaries.

    Each chamber has a sigmoid activation curve centered at its natural
    activation point. All transitions are smooth — no if/elif step checks.

    Chamber i activates around step = warmup * (i + 0.5) / n_chambers
    with smooth sigmoid transition (no discrete jump).
    """

    def __init__(self, n_chambers: int, warmup_steps: int = 2000):
        self.n_chambers = n_chambers
        self.warmup_steps = max(warmup_steps, 1)

    def get_masks(self, step: int) -> list:
        """Returns smooth [0..1] masks for each chamber via sigmoid."""
        masks = []
        for i in range(self.n_chambers):
            # Center of activation for chamber i
            center = self.warmup_steps * (i + 0.5) / self.n_chambers
            # Steepness: scales with warmup duration
            slope = 8.0 * self.n_chambers / self.warmup_steps
            # Smooth sigmoid — no hard boundaries
            mask = 1.0 / (1.0 + math.exp(-slope * (step - center)))
            masks.append(mask)
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

        # Always create merge_proj for soft mode blending
        self.merge_proj = nn.Linear(
            d_model * len(self.chambers), d_model, bias=False
        )
        nn.init.zeros_(self.merge_proj.weight)

        # Learnable mode gate: sigmoid → 0=sequential, 1=parallel
        init_mode = 0.0 if mode == 'sequential' else 0.0
        self._mode_logit = nn.Parameter(torch.tensor(init_mode))

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

        # Soft mode blending: both paths computed, gate decides the mix
        mode_gate = torch.sigmoid(self._mode_logit)  # 0=sequential, 1=parallel

        # Sequential path: cascading enrichment
        seq_enrichment = torch.zeros_like(h)
        deltas = []
        for i, chamber in enumerate(self.chambers):
            mask = masks[i] if i < len(masks) else 1.0
            delta_seq = chamber(h + seq_enrichment, curriculum_mask=mask)
            seq_enrichment = seq_enrichment + delta_seq
            delta_par = chamber(h, curriculum_mask=mask)
            deltas.append(delta_par)

        # Parallel path: independent chambers + merge
        concatenated = torch.cat(deltas, dim=-1)  # (B, T, D*N)
        par_enrichment = self.merge_proj(concatenated)

        # Blend: learned gate mixes sequential and parallel
        enrichment = (1 - mode_gate) * seq_enrichment + mode_gate * par_enrichment

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


class MatryoshkaNautilus(nn.Module):
    """v66: NautilusHierarchy + MatryoshkaQuantizer — пространственно-временное
    кодирование между камерами.

    Идея: каждая камера Наутилуса обогащает сигнал. «До камеры» и «после камеры»
    — это естественная пара (пространство × время) для MatryoshkaQuantizer.

    Level 2 (гекс-цифры) кодирует *что именно изменила каждая камера*,
    давая модели эксплицитный сигнал об иерархическом обогащении.

    Поток:
        h = ln_pre(x)
        for each chamber:
            state_before = h + accumulated_enrichment
            delta = chamber(state_before)
            state_after = state_before + delta
            # Matryoshka: q_after vs q_before → spacetime encoding
            q_before = to_q(state_before)
            q_after  = to_q(state_after)
            m_out = matryoshka(q_after, x_ref=q_before)
            accumulated_enrichment += delta + gate * m_out
        result = x + residual_gate * ln_post(accumulated_enrichment)

    Результат: Matryoshka Level 2 получает реальный временной сигнал
    (разница между состояниями до/после камеры), а не тождественный.
    """

    def __init__(
        self,
        d_model: int,
        q_dim: int = 6,
        matryoshka_temp: float = 0.3,
        init_scale: float = 0.01,
        warmup_steps: int = 2000,
        mode: str = 'sequential',
        enabled_chambers: list = None,
    ):
        super().__init__()

        # Reuse NautilusHierarchy for chambers, scheduler, norms
        self.nautilus = NautilusHierarchy(
            d_model=d_model,
            init_scale=init_scale,
            warmup_steps=warmup_steps,
            mode=mode,
            enabled_chambers=enabled_chambers,
        )

        # MatryoshkaQuantizer для межкамерного кодирования
        from .quantizers import MatryoshkaQuantizer
        self.matryoshka = MatryoshkaQuantizer(
            total_dim=q_dim, d_model=d_model, temp=matryoshka_temp,
        )

        # Projection d_model → Q-space
        self.to_q = nn.Linear(d_model, q_dim, bias=False)

        # Gate for matryoshka enrichment (per-chamber)
        n_chambers = len(self.nautilus.chambers)
        self.matryoshka_gates = nn.Parameter(torch.zeros(n_chambers))

        # Diagnostics
        self._matryoshka_infos = []

    def set_step(self, step: int):
        """Proxy to NautilusHierarchy.set_step."""
        self.nautilus.set_step(step)

    def forward(self, x: torch.Tensor) -> tuple:
        """Применяет Наутилус-иерархию с межкамерным MatryoshkaQuantizer.

        Args:
            x: (B, T, D) token embeddings

        Returns:
            (enriched_x, info_dict)
        """
        masks = self.nautilus.scheduler.get_masks(self.nautilus._current_step)
        h = self.nautilus.ln_pre(x)

        if self.nautilus.mode == 'sequential':
            enrichment = torch.zeros_like(h)
            self._matryoshka_infos = []

            for i, chamber in enumerate(self.nautilus.chambers):
                mask = masks[i] if i < len(masks) else 1.0

                state_before = h + enrichment
                delta = chamber(state_before, curriculum_mask=mask)
                enrichment = enrichment + delta
                state_after = h + enrichment

                # Matryoshka: encode what this chamber changed
                # Detach при train чтобы Matryoshka gradient не корруптил Nautilus cascade
                q_before = self.to_q(state_before.detach() if self.training else state_before)
                q_after = self.to_q(state_after)
                m_out, m_info = self.matryoshka(q_after, x_ref=q_before)

                # Gated matryoshka enrichment, modulated by curriculum
                m_gate = torch.sigmoid(self.matryoshka_gates[i])
                enrichment = enrichment + m_gate * mask * m_out
                self._matryoshka_infos.append(m_info)

        elif self.nautilus.mode == 'parallel':
            # Parallel: каждая камера + её matryoshka delta
            deltas = []
            self._matryoshka_infos = []
            for i, chamber in enumerate(self.nautilus.chambers):
                mask = masks[i] if i < len(masks) else 1.0
                delta = chamber(h, curriculum_mask=mask)

                # Matryoshka: encode chamber enrichment
                q_before = self.to_q(h)
                q_after = self.to_q(h + delta)
                m_out, m_info = self.matryoshka(q_after, x_ref=q_before)

                m_gate = torch.sigmoid(self.matryoshka_gates[i])
                deltas.append(delta + m_gate * mask * m_out)
                self._matryoshka_infos.append(m_info)

            concatenated = torch.cat(deltas, dim=-1)
            enrichment = self.nautilus.merge_proj(concatenated)

        else:
            raise ValueError(f"Unknown mode: {self.nautilus.mode}")

        # Post-norm + residual gate
        enrichment = self.nautilus.ln_post(enrichment)
        result = x + self.nautilus.residual_gate * enrichment

        # Collect info
        info = {
            'step': self.nautilus._current_step,
            'mode': self.nautilus.mode,
            'residual_gate': self.nautilus.residual_gate.item(),
            'masks': masks,
            'chambers': {
                name: chamber.get_stats()
                for name, chamber in zip(
                    self.nautilus.chamber_names, self.nautilus.chambers
                )
            },
            'matryoshka': {
                'n_chambers': len(self.nautilus.chambers),
                'gates': [
                    round(torch.sigmoid(self.matryoshka_gates[i]).item(), 4)
                    for i in range(len(self.nautilus.chambers))
                ],
                'quantizer_stats': self.matryoshka.get_stats(),
            },
        }

        return result, info

    def get_stats(self) -> dict:
        """Aggregated stats for logging."""
        stats = self.nautilus.get_nautilus_stats()
        # Add matryoshka-specific stats
        for i, name in enumerate(self.nautilus.chamber_names):
            g = torch.sigmoid(self.matryoshka_gates[i]).item()
            stats[f'matryoshka/{name}/gate'] = round(g, 4)
        mq_stats = self.matryoshka.get_stats()
        for k, v in mq_stats.items():
            stats[f'matryoshka/quantizer/{k}'] = v
        return stats


class PostCascadeMatryoshkaNautilus(nn.Module):
    """v67: Матрёшка ПОСЛЕ полного каскада Наутилуса.

    Урок v66: вставка MatryoshkaQuantizer между камерами Наутилуса добавляет
    шум через случайную проекцию to_q, разрушая каскадное обогащение
    (PPL 3.07 vs 1.01 у чистого Наутилуса).

    Решение: Наутилус работает без помех (каскад камер как обычно),
    MatryoshkaQuantizer применяется ПОСЛЕ — кодирует суммарное изменение.

    Поток:
        x_enriched, nautilus_info = nautilus(x)    # полный каскад без вмешательства
        q_before = to_q(x)                         # вход = «прошлое»
        q_after  = to_q(x_enriched)                # выход = «настоящее»
        m_out, m_info = matryoshka(q_after, x_ref=q_before)  # Level 2 = spacetime
        result = x_enriched + m_gate * m_out       # дополнительное обогащение

    Level 2 кодирует «что изменил весь Наутилус» — от микро-геометрии
    CubeDiagonal до глобальной FlowerOfLife.
    """

    def __init__(
        self,
        d_model: int,
        q_dim: int = 6,
        matryoshka_temp: float = 0.3,
        init_scale: float = 0.01,
        warmup_steps: int = 2000,
        mode: str = 'sequential',
        enabled_chambers: list = None,
    ):
        super().__init__()

        # NautilusHierarchy runs its full cascade unmodified
        self.nautilus = NautilusHierarchy(
            d_model=d_model,
            init_scale=init_scale,
            warmup_steps=warmup_steps,
            mode=mode,
            enabled_chambers=enabled_chambers,
        )

        # MatryoshkaQuantizer applied after cascade
        from .quantizers import MatryoshkaQuantizer
        self.matryoshka = MatryoshkaQuantizer(
            total_dim=q_dim, d_model=d_model, temp=matryoshka_temp,
        )

        # Projection d_model → Q-space
        self.to_q = nn.Linear(d_model, q_dim, bias=False)

        # Single gate for post-cascade matryoshka enrichment
        self.matryoshka_gate = nn.Parameter(torch.tensor(0.0))

    def set_step(self, step: int):
        self.nautilus.set_step(step)

    def forward(self, x: torch.Tensor) -> tuple:
        """Наутилус-каскад + пост-каскадная Матрёшка.

        Args:
            x: (B, T, D) token embeddings

        Returns:
            (enriched_x, info_dict)
        """
        # 1. Full Nautilus cascade (uninterrupted)
        x_enriched, nautilus_info = self.nautilus(x)

        # 2. Matryoshka: encode what the entire Nautilus changed
        q_before = self.to_q(x)           # input = «past» (space)
        q_after = self.to_q(x_enriched)   # output = «present» (time)
        m_out, m_info = self.matryoshka(q_after, x_ref=q_before)

        # 3. Gated addition
        m_gate = torch.sigmoid(self.matryoshka_gate)
        result = x_enriched + m_gate * m_out

        # Collect info
        info = nautilus_info.copy()
        info['matryoshka'] = {
            'mode': 'post_cascade',
            'gate': round(m_gate.item(), 4),
            'quantizer_stats': self.matryoshka.get_stats(),
        }

        return result, info

    def get_stats(self) -> dict:
        stats = self.nautilus.get_nautilus_stats()
        stats['matryoshka/gate'] = round(
            torch.sigmoid(self.matryoshka_gate).item(), 4
        )
        mq_stats = self.matryoshka.get_stats()
        for k, v in mq_stats.items():
            stats[f'matryoshka/quantizer/{k}'] = v
        return stats
