"""
PseudoRAG-мост между Q4 (16 архетипов) и Q6 (64 гексаграммы).

Реализует формальное вложение Q4 ⊂ Q6 из PSEUDORAG_YIJING_BRIDGE.md:
    φ: Q4 → Q6
    φ(b₁, b₂, b₃, b₄) = (2b₁-1, 2b₂-1, 2b₃-1, 2b₄-1, 0, 0)

16 PseudoRAG-архетипов вкладываются в «экваториальные» точки Q6
(нулевые последние 2 координаты). Каждый Q4-архетип порождает
кластер из 4 гексаграмм (фиксированы b₁–b₄, свободны b₅–b₆).

Модуль содержит:
- PseudoRAGProjection: проекция D → Q4 logits → Q6 весов через кластерное расширение
- PseudoRAGDistillationLoss: KL-дивергенция после агрегации 64→16 по кластерам
- q4_to_q6_embedding(): фиксированное вложение (16, 6)
- q6_cluster_membership(): бинарная матрица (64, 16) принадлежности к кластерам

Интегрируется через use_pseudo_rag=True в YiJingConfig.

Ref: PSEUDORAG_YIJING_BRIDGE.md — «PseudoRAG ↔ YiJing-Transformer: Формальная связь»
Ref: daten22/pseudorag (archetypes.py, query_expander.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Q4 архетипы PseudoRAG ====================

Q4_ARCHETYPES = {
    # code: (index, b1_Materiality, b2_Dynamics, b3_Scale, b4_Structure, name_ru)
    # b=1 → +1, b=0 → -1 в Q6-координатах
    # Materiality: M=1, A=0  |  Dynamics: S=0, D=1  |  Scale: E=0, C=1  |  Structure: O=0, F=1
    'MSEO': (0,  1, 0, 0, 0, 'Кристалл'),
    'MSEF': (1,  1, 0, 0, 1, 'Песок'),
    'MSCO': (2,  1, 0, 1, 0, 'Здание'),
    'MSCF': (3,  1, 0, 1, 1, 'Лес'),
    'MDEO': (4,  1, 1, 0, 0, 'Механизм'),
    'MDEF': (5,  1, 1, 0, 1, 'Организм'),
    'MDCO': (6,  1, 1, 1, 0, 'Машина'),
    'MDCF': (7,  1, 1, 1, 1, 'Город'),
    'ASEO': (8,  0, 0, 0, 0, 'Аксиома'),
    'ASEF': (9,  0, 0, 0, 1, 'Архетип'),
    'ASCO': (10, 0, 0, 1, 0, 'Теория'),
    'ASCF': (11, 0, 0, 1, 1, 'Культура'),
    'ADEO': (12, 0, 1, 0, 0, 'Алгоритм'),
    'ADEF': (13, 0, 1, 0, 1, 'Интуиция'),
    'ADCO': (14, 0, 1, 1, 0, 'Программа'),
    'ADCF': (15, 0, 1, 1, 1, 'Общество'),
}


# ==================== Вспомогательные функции ====================

def q4_to_q6_embedding() -> torch.Tensor:
    """Строит фиксированное вложение φ: Q4 → Q6 (16, 6).

    φ(b₁, b₂, b₃, b₄) = (2b₁-1, 2b₂-1, 2b₃-1, 2b₄-1, 0, 0)

    Каждый Q4-архетип отображается в «экваториальную» точку Q6
    с нулевыми последними двумя координатами.

    Returns:
        Tensor (16, 6) — значения ∈ {-1, +1, 0}
    """
    embed = torch.zeros(16, 6)
    for code, (idx, b1, b2, b3, b4, _name) in Q4_ARCHETYPES.items():
        embed[idx, 0] = 2 * b1 - 1  # Materiality
        embed[idx, 1] = 2 * b2 - 1  # Dynamics
        embed[idx, 2] = 2 * b3 - 1  # Scale
        embed[idx, 3] = 2 * b4 - 1  # Structure
        # embed[idx, 4] = 0  (по умолчанию)
        # embed[idx, 5] = 0  (по умолчанию)
    return embed


def q6_cluster_membership() -> torch.Tensor:
    """Бинарная матрица принадлежности гексаграмм к Q4-кластерам (64, 16).

    Каждая гексаграмма Q6 = (b₁, b₂, b₃, b₄, b₅, b₆) ∈ {0,1}⁶
    принадлежит Q4-кластеру, определяемому первыми 4 битами (b₁, b₂, b₃, b₄).
    Каждый Q4-архетип отвечает ровно за 4 гексаграммы (2² вариантов b₅, b₆).

    Returns:
        Tensor (64, 16), бинарная: membership[hex_idx, q4_idx] = 1.0
            если гексаграмма hex_idx входит в кластер q4_idx
    """
    membership = torch.zeros(64, 16)
    for hex_idx in range(64):
        # Разложим hex_idx в 6 бит (b₁ — старший)
        b1 = (hex_idx >> 5) & 1
        b2 = (hex_idx >> 4) & 1
        b3 = (hex_idx >> 3) & 1
        b4 = (hex_idx >> 2) & 1
        # Q4-кластер определяется первыми 4 битами
        q4_idx = (b1 << 3) | (b2 << 2) | (b3 << 1) | b4
        membership[hex_idx, q4_idx] = 1.0
    return membership


# ==================== PseudoRAGProjection ====================

class PseudoRAGProjection(nn.Module):
    """Проекция D → Q4 logits → Q6 весов через кластерное расширение.

    Двухэтапная процедура:
    1. Линейная проекция hidden → Q4 logits (16 архетипов)
    2. Расширение Q4 → Q6 через фиксированную кластерную матрицу + обучаемую
       тонкую дифференциацию последних 2 координат

    Архитектура:
        x ∈ R^D → proj_q4 → logits ∈ R^16 → softmax → q4_weights ∈ Δ^16
        q4_weights → cluster_expand (16→64) → q6_coarse ∈ R^64
        x → refine_proj → Δ_fine ∈ R^64
        q6_weights = softmax(q6_coarse + Δ_fine)

    Args:
        d_model: размерность модели
        n_q4: число Q4-архетипов (16)
        n_q6: число Q6-гексаграмм (64)
    """

    def __init__(self, d_model: int, n_q4: int = 16, n_q6: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_q4 = n_q4
        self.n_q6 = n_q6

        # Фиксированное вложение Q4 → Q6 (16, 6)
        self.register_buffer('q4_embed', q4_to_q6_embedding())

        # Фиксированная кластерная принадлежность (64, 16)
        self.register_buffer('cluster_membership', q6_cluster_membership())

        # Транспонированная для расширения: (16, 64)
        # cluster_expand[q4, hex] = 1/4 если hex ∈ кластер(q4), иначе 0
        # Нормировка: каждый Q4 → равномерно по своим 4 гексаграммам
        expand = self.cluster_membership.t()  # (16, 64)
        expand = expand / expand.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.register_buffer('cluster_expand', expand)

        # Линейная проекция hidden → Q4 logits
        self.proj_q4 = nn.Linear(d_model, n_q4, bias=False)

        # Обучаемая тонкая дифференциация: hidden → Q6 refinement logits
        # Моделирует вклад последних 2 координат (b₅, b₆)
        self.refine_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model // 4, n_q6, bias=False),
        )

        # Обучаемые «мягкие» последние 2 координаты для каждого Q4-архетипа
        # Каждый Q4 → 4 вектора (b₅, b₆) варианта, обучаемые
        # Форма: (16, 4, 2) — для 4 вариантов (00, 01, 10, 11) × 2 координаты
        self.fine_coords = nn.Parameter(torch.randn(n_q4, 4, 2) * 0.1)

        # Масштаб смешивания грубого и тонкого сигналов
        self.blend_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Проекция hidden → Q6 весов.

        Args:
            x: (B, T, D) — hidden representations

        Returns:
            q6_weights: (B, T, 64) — нормализованные веса по 64 гексаграммам
        """
        B, T, D = x.shape

        # Шаг 1: Q4 logits и softmax
        q4_logits = self.proj_q4(x)  # (B, T, 16)
        q4_weights = F.softmax(q4_logits, dim=-1)  # (B, T, 16)

        # Шаг 2: Грубое расширение Q4 → Q6 через кластерную матрицу
        # q4_weights @ cluster_expand = (B, T, 16) @ (16, 64) → (B, T, 64)
        q6_coarse = torch.matmul(q4_weights, self.cluster_expand)

        # Шаг 3: Тонкая дифференциация (обучаемая)
        q6_fine = self.refine_proj(x)  # (B, T, 64)

        # Шаг 4: Комбинация грубого и тонкого сигнала
        q6_logits = q6_coarse + self.blend_scale * q6_fine
        q6_weights = F.softmax(q6_logits, dim=-1)  # (B, T, 64)

        return q6_weights

    def get_q4_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает Q4-логиты (для distillation loss).

        Args:
            x: (B, T, D)

        Returns:
            q4_logits: (B, T, 16)
        """
        return self.proj_q4(x)

    def get_full_q6_embedding(self) -> torch.Tensor:
        """Строит полное вложение Q6 (64, 6): фиксированные b₁–b₄ + обучаемые b₅–b₆.

        Returns:
            Tensor (64, 6) — координаты всех 64 гексаграмм
        """
        embed = torch.zeros(64, 6, device=self.q4_embed.device)
        for hex_idx in range(64):
            b1 = (hex_idx >> 5) & 1
            b2 = (hex_idx >> 4) & 1
            b3 = (hex_idx >> 3) & 1
            b4 = (hex_idx >> 2) & 1
            b5 = (hex_idx >> 1) & 1
            b6 = hex_idx & 1
            q4_idx = (b1 << 3) | (b2 << 2) | (b3 << 1) | b4
            variant_idx = (b5 << 1) | b6  # 0..3

            embed[hex_idx, :4] = self.q4_embed[q4_idx, :4]
            embed[hex_idx, 4:6] = self.fine_coords[q4_idx, variant_idx]
        return embed


# ==================== PseudoRAGDistillationLoss ====================

class PseudoRAGDistillationLoss(nn.Module):
    """Дистилляция Q4-учитель → Q6-ученик через агрегацию кластеров.

    Для каждого Q4-архетипа суммируем вероятности 4 входящих Q6-гексаграмм,
    получая распределение студента на Q4, затем считаем KL-дивергенцию
    с распределением учителя.

    KL(P_teacher || P_student_aggregated)

    Args:
        temperature: температура для сглаживания (по умолчанию 2.0)
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
        # Кластерная принадлежность (64, 16)
        self.register_buffer('cluster_membership', q6_cluster_membership())

    def forward(
        self,
        q6_logits: torch.Tensor,
        q4_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Вычисляет distillation loss.

        Args:
            q6_logits: (B, T, 64) — логиты Q6-студента
            q4_targets: (B, T, 16) — софт-мишени Q4-учителя (распределение)

        Returns:
            loss: скаляр — KL-дивергенция
        """
        # Софтмакс по Q6 с температурой
        q6_probs = F.softmax(q6_logits / self.temperature, dim=-1)  # (B, T, 64)

        # Агрегация 64 → 16: суммируем вероятности по кластерам
        # q6_probs @ cluster_membership = (B, T, 64) @ (64, 16) → (B, T, 16)
        q4_student = torch.matmul(q6_probs, self.cluster_membership)

        # Лог-вероятности студента (clamped для стабильности)
        q4_student_log = torch.log(q4_student.clamp(min=1e-8))

        # Софт-мишени учителя (уже нормализованы)
        q4_teacher = F.softmax(q4_targets / self.temperature, dim=-1)

        # KL(teacher || student) = Σ teacher * log(teacher / student)
        loss = F.kl_div(
            q4_student_log,
            q4_teacher,
            reduction='batchmean',
            log_target=False,
        )

        # Масштабирование по T² (стандарт для distillation)
        loss = loss * (self.temperature ** 2)

        return loss
