"""
hierarchical_moe.py — Иерархическая Mixture-of-Experts архитектура.

Полная 4-уровневая схема:
    Input
      │
      ▼
    [MultiScaleGlobalRouter]  ← Matryoshka Q2→Q3→Q6 → группа (ABSTRACT/DYNAMIC/CONCRETE)
      │
      ├─► [GroupRouter A]  ──► MicroExpert_Theory
      │                    ──► MicroExpert_Models
      │
      ├─► [GroupRouter B]  ──► MicroExpert_Self
      │                    ──► MicroExpert_Training
      │
      ├─► [GroupRouter C]  ──► MicroExpert_Scripts
      │                    ──► MicroExpert_Data
      │
      ├─► [BridgeExperts]   ← синтез межгрупповых связей
      │    ├── Bridge_ABSTRACT↔DYNAMIC
      │    ├── Bridge_DYNAMIC↔CONCRETE
      │    └── Bridge_ABSTRACT↔CONCRETE
      │
      └─► [Q6ExpertBank]   ← 4-й уровень: 64 Q6-эксперта (векторизованный)
           (активируется только для топ-1 группы, use_hex_tier=True)

Иерархия роутинга (Matryoshka):
    Q2 (4 вершины)  → выбор группы (ABSTRACT / DYNAMIC / CONCRETE)
    Q3 (8 вершин)   → уточнение доменов внутри группы
    Q6 (64 вершины) → точный Q6-сигнал для HexagramMoE 4-го уровня

Обучение поэтапно:
    Stage 1: только MicroExperts (по одному на кластер, остальные заморожены)
    Stage 2: GroupRouters + MicroExperts
    Stage 3: MultiScaleGlobalRouter + всё вместе
    Stage 4: BridgeExperts (межгрупповые мосты)
    Stage 5: JointFinetune (всё вместе)
    Stage 6: Q6ExpertBank (4-й уровень, опционально)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

# Кластеры репозитория → домены Q6
CLUSTER_TO_DOMAIN: Dict[str, str] = {
    "Theory":   "NOOS",    # ䷿ Сознание — философия/теория
    "Models":   "COSMO",   # ䷛ Пустота  — структуры/модели
    "Self":     "AERO",    # ䷅ Ветер    — саморефлексия/метод
    "Training": "PYRO",    # ䷬ Огонь    — обучение/трансформация
    "Scripts":  "GEO",     # ䷁ Земля    — код/инструменты
    "Data":     "HYDRO",   # ䷒ Вода     — данные/поток
}

# Группы доменов (для иерархии роутеров)
DOMAIN_GROUPS: Dict[str, List[str]] = {
    "ABSTRACT": ["NOOS", "COSMO"],     # абстрактное мышление
    "DYNAMIC":  ["AERO", "PYRO"],      # динамические процессы
    "CONCRETE": ["GEO",  "HYDRO"],     # конкретные структуры
}

# Обратный индекс: домен → группа
DOMAIN_TO_GROUP: Dict[str, str] = {
    d: g for g, domains in DOMAIN_GROUPS.items() for d in domains
}

# Q6 якоря доменов (индексы гексаграмм 0..63)
#
# hexlearn/hexopt: k-medoids + SA-оптимизация якорей (sep-score 12 → 21, +75%).
# Спектральный принцип (hexgraph): якоря соответствуют собственным значениям Q6:
#   ABSTRACT (λ=+4): вершины с 5+ битами (верхняя полусфера)
#   DYNAMIC  (λ= 0): вершины с ровно 3 битами (экватор, null-space)
#   CONCRETE (λ=-4): вершины с 0-1 битом (нижняя полусфера)
#
# Предыдущие (sep-score=12):
#   GEO=0, HYDRO=18(2бит), PYRO=45(4бит), AERO=6(2бит), COSMO=27(4бит), NOOS=63
# Оптимизированные (sep-score=21):
#   GEO=0(0бит), HYDRO=8(1бит), PYRO=21(3бит), AERO=19(3бит), COSMO=62(5бит), NOOS=63(6бит)
DOMAIN_Q6_IDX: Dict[str, int] = {
    "GEO":   0,   # 000000 (0 бит) — Земля, чистая конкретность (Yin×6)
    "HYDRO": 8,   # 001000 (1 бит) — Вода, текучий поток
    "PYRO":  21,  # 010101 (3 бита) — Огонь, трансформация (истинный экватор)
    "AERO":  19,  # 010011 (3 бита) — Ветер, саморефлексия (истинный экватор)
    "COSMO": 62,  # 111110 (5 бит) — Пустота, структуры (верхняя полусфера)
    "NOOS":  63,  # 111111 (6 бит) — Сознание, чистая абстракция (Yang×6)
}

# Примечание: BRIDGE_PAIRS удалены — в топологии восьмёрки используется
# единственная точка пересечения BidirBridgeExpert(ABSTRACT, CONCRETE),
# а не три симметричных моста. DYNAMIC — не рядовая группа, а crossing node.


def _make_hexagrams() -> torch.Tensor:
    import itertools
    verts = list(itertools.product([-1.0, 1.0], repeat=6))
    return torch.tensor(verts, dtype=torch.float32)  # (64, 6)


def _make_sub_hypercube(dims: int) -> torch.Tensor:
    import itertools
    verts = list(itertools.product([-1.0, 1.0], repeat=dims))
    return torch.tensor(verts, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Q6ExpertBank — 4-й уровень: 64 Q6-эксперта (векторизованный bank)
# ──────────────────────────────────────────────────────────────────────────────

class Q6ExpertBank(nn.Module):
    """Векторизованный банк 64 экспертов — по одному на каждую гексаграмму.

    Эксперт i вычисляет: FFN_i(x) = W_out[i] · SiLU(W_in[i] · x + b_in[i])
    Роутинг через hex_weights (B, T, 64) из MultiScaleGlobalRouter.
    """

    def __init__(self, d_model: int, d_ff: int, top_k: int = 4):
        super().__init__()
        self.top_k = top_k
        # Векторизованные банки: (64, d_ff, d_model)
        self.W_in   = nn.Parameter(torch.randn(64, d_ff, d_model) * (d_model ** -0.5))
        self.b_in   = nn.Parameter(torch.zeros(64, d_ff))
        self.W_out  = nn.Parameter(torch.randn(64, d_model, d_ff) * (d_ff ** -0.5))
        self.b_out  = nn.Parameter(torch.zeros(64, d_model))
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                hex_weights: torch.Tensor) -> torch.Tensor:
        """
        x:           (B, T, d_model)
        hex_weights: (B, T, 64)  — из MultiScaleGlobalRouter
        Returns:     (B, T, d_model)
        """
        B, T, D = x.shape
        x_n = self.norm(x)

        top_w, top_idx = hex_weights.topk(self.top_k, dim=-1)     # (B, T, K)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)  # ренорм

        x_flat  = x_n.view(B * T, D)
        output  = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            idx_k  = top_idx[:, :, k].reshape(B * T)
            w_k    = top_w[:, :, k].reshape(B * T, 1)
            W_in_k = self.W_in[idx_k]    # (B*T, d_ff, D)
            b_in_k = self.b_in[idx_k]    # (B*T, d_ff)
            W_out_k = self.W_out[idx_k]  # (B*T, D, d_ff)
            b_out_k = self.b_out[idx_k]  # (B*T, D)

            h    = torch.bmm(W_in_k, x_flat.unsqueeze(-1)).squeeze(-1) + b_in_k
            h    = F.silu(h)
            out_k = torch.bmm(W_out_k, h.unsqueeze(-1)).squeeze(-1) + b_out_k
            output = output + w_k * out_k

        return output.view(B, T, D)

    def load_balance_loss(self, hex_weights: torch.Tensor) -> torch.Tensor:
        """Вспомогательный loss для равномерной нагрузки экспертов."""
        mean_w = hex_weights.mean(dim=[0, 1])             # (64,)
        return (mean_w * torch.log(mean_w + 1e-8)).sum()


# ──────────────────────────────────────────────────────────────────────────────
# MicroExpert — специализированный FFN на один кластер
# ──────────────────────────────────────────────────────────────────────────────

class MicroExpert(nn.Module):
    """SwiGLU FFN — один микро-эксперт на кластер."""

    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        d_ff = d_model * expansion
        self.gate_proj  = nn.Linear(d_model, d_ff, bias=False)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.out_proj   = nn.Linear(d_ff, d_model, bias=False)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate  = F.silu(self.gate_proj(x))
        value = self.value_proj(x)
        return self.out_proj(gate * value)


# ──────────────────────────────────────────────────────────────────────────────
# BridgeExpert — эксперт-мост между двумя группами
# ──────────────────────────────────────────────────────────────────────────────

class BidirBridgeExpert(nn.Module):
    """Двунаправленный эксперт-мост между двумя петлями восьмёрки.

    Реализует подлинное двунаправленное обогащение из bidir_train.py:
        ВПЕРЁД  A→B: group_A запрашивает group_B (специализация → обобщение)
        НАЗАД   B→A: group_B запрашивает group_A (обобщение → специализация)
        Синтез: alpha * fwd + (1-alpha) * bwd

    Causal mask гарантирует авторегрессионность в обоих направлениях.
    Обучаемый alpha = сигмоид(log_alpha) управляет балансом направлений.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Два независимых cross-attention: A→B и B→A
        self.attn_fwd = nn.MultiheadAttention(d_model, num_heads=4,
                                               batch_first=True, bias=False)
        self.attn_bwd = nn.MultiheadAttention(d_model, num_heads=4,
                                               batch_first=True, bias=False)
        # Нормы для каждого направления
        self.norm_a   = nn.LayerNorm(d_model)
        self.norm_b   = nn.LayerNorm(d_model)
        # SwiGLU синтез после смешивания
        self.ffn_gate = nn.Linear(d_model, d_model * 2, bias=False)
        self.ffn_val  = nn.Linear(d_model, d_model * 2, bias=False)
        self.ffn_out  = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm_ffn = nn.LayerNorm(d_model)
        # Обучаемые параметры
        self.log_alpha = nn.Parameter(torch.zeros(1))   # баланс fwd/bwd, ≈0.5 изначально
        self.gate      = nn.Parameter(torch.tensor(-2.0))  # общий выходной gate

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x_a: torch.Tensor,
                x_b: torch.Tensor) -> torch.Tensor:
        """
        x_a: (B, T, d) — петля A (ABSTRACT или левая петля)
        x_b: (B, T, d) — петля B (CONCRETE или правая петля)
        returns: (B, T, d) — двунаправленный синтез (точка пересечения)
        """
        T    = x_a.shape[1]
        mask = self._causal_mask(T, x_a.device)
        na, nb = self.norm_a(x_a), self.norm_b(x_b)

        # A→B: ABSTRACT запрашивает CONCRETE (специализация → обобщение)
        fwd, _ = self.attn_fwd(na, nb, nb, attn_mask=mask)
        # B→A: CONCRETE запрашивает ABSTRACT (обобщение → специализация)
        bwd, _ = self.attn_bwd(nb, na, na, attn_mask=mask)

        alpha = torch.sigmoid(self.log_alpha)          # (1,) ∈ (0, 1)
        # Crossing point: взвешенный синтез обоих направлений
        crossed = alpha * (x_a + fwd) + (1.0 - alpha) * (x_b + bwd)

        h = self.norm_ffn(crossed)
        out = crossed + self.ffn_out(F.silu(self.ffn_gate(h)) * self.ffn_val(h))
        return out * torch.sigmoid(self.gate)


# ──────────────────────────────────────────────────────────────────────────────
# GroupRouter — роутер внутри одной группы доменов
# ──────────────────────────────────────────────────────────────────────────────

class GroupRouter(nn.Module):
    """Роутер внутри группы: Q6-сигнал → веса микро-экспертов группы.

    Anti-circle (по Крюкову): если один эксперт доминирует streak_limit
    последовательных шагов подряд — штраф в lb_loss, вынуждая переключение.
    Аналог флага «один бит не флипается > 4 раз подряд → форс-смена».
    """

    def __init__(self, d_model: int, expert_names: List[str],
                 streak_limit: int = 4, anticircle_weight: float = 0.1):
        super().__init__()
        n = len(expert_names)
        self.expert_names     = expert_names
        self.streak_limit     = streak_limit
        self.anticircle_weight = anticircle_weight

        self.proj = nn.Linear(d_model, n, bias=True)
        self.norm = nn.LayerNorm(d_model)
        # Скользящее среднее нагрузки по экспертам (EMA, не обучаемый)
        self.register_buffer('_ema_load', torch.ones(n) / n)

    def forward(self, x: torch.Tensor, top_k: int = 2
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights:  (B, T, n)  — sparse top-k веса
            indices:  (B, T, k)  — top-k индексы
            lb_loss:  scalar     — load-balance + anti-circle loss
        """
        logits  = self.proj(self.norm(x))                      # (B, T, n)
        weights = F.softmax(logits, dim=-1)                    # (B, T, n)

        k = min(top_k, weights.shape[-1])
        topk_vals, indices = torch.topk(weights, k, dim=-1)
        sparse_w = torch.zeros_like(weights).scatter_(-1, indices, topk_vals)
        sparse_w = sparse_w / (sparse_w.sum(dim=-1, keepdim=True) + 1e-8)

        # ── Load-balance: МИНИМИЗИРУЕМ отрицательную энтропию = МАКСИМИЗИРУЕМ баланс
        mean_w  = weights.mean(dim=(0, 1))                     # (n,)
        lb_loss = (mean_w * torch.log(mean_w + 1e-8)).sum()    # Σ p·log(p) < 0

        # ── Anti-circle: штраф за доминирование одного эксперта ─────────────
        # Обновляем EMA нагрузки
        if self.training:
            with torch.no_grad():
                self._ema_load.mul_(0.95).add_(mean_w.detach() * 0.05)

        # Доминирование: штраф если макс. нагрузка превышает допуск над uniform.
        # Порог = uniform + margin, где margin масштабируется с n_experts:
        #   n=2: threshold = 0.5 + 0.15 = 0.65 (срабатывает при >65% на одном эксперте)
        #   n=6: threshold = 0.167 + 0.10 = 0.267
        # Предыдущая формула uniform*(1+streak/n) давала threshold=1.5 для n=2
        # → relu(max_load - 1.5) = 0 всегда → anti-circle никогда не работал.
        max_load = self._ema_load.max()
        n_exp    = max(len(self.expert_names), 1)
        uniform  = 1.0 / n_exp
        margin   = min(0.15, (1.0 - uniform) * 0.3)  # 30% от доступного диапазона, max 0.15
        threshold = uniform + margin
        anticircle_penalty = F.relu(max_load - threshold)
        lb_loss = lb_loss + self.anticircle_weight * anticircle_penalty

        return sparse_w, indices, lb_loss


# ──────────────────────────────────────────────────────────────────────────────
# GlobalRouter (legacy, оставлен для совместимости)
# ──────────────────────────────────────────────────────────────────────────────

class GlobalRouter(nn.Module):
    """Q6-based роутер верхнего уровня (одномасштабный).
    Оставлен для обратной совместимости. Новый код использует MultiScaleGlobalRouter.
    """

    def __init__(self, d_model: int, group_names: List[str]):
        super().__init__()
        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)
        self.group_names = group_names
        n_groups = len(group_names)

        self.q6_proj  = nn.Linear(d_model, 6, bias=False)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.norm     = nn.LayerNorm(d_model)

        group_anchors = []
        for g in group_names:
            domain_vecs = torch.stack([
                hexagrams[DOMAIN_Q6_IDX[d]] for d in DOMAIN_GROUPS[g]
            ])
            group_anchors.append(domain_vecs.mean(0))
        # hexgeom (Voronoi/Hamming): нормализуем якоря до единичной нормы.
        # DYNAMIC якорь имеет norm≈1.414 vs ABSTRACT/CONCRETE norm≈2.0 —
        # без нормализации DYNAMIC систематически проигрывает на 29%.
        anch = torch.stack(group_anchors)
        anch = F.normalize(anch, dim=-1)
        self.register_buffer('group_anchors', anch)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.1, 5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h        = self.norm(x)
        soft_bits = torch.tanh(self.q6_proj(h))
        sim       = soft_bits @ self.hexagrams.T
        hex_w     = F.softmax(sim / self.temperature, dim=-1)
        soft_hex  = hex_w @ self.hexagrams
        # hexgeom: нормализуем якоря on-the-fly — работает и при загрузке чекпоинта
        anchors_norm  = F.normalize(self.group_anchors, dim=-1)
        group_scores  = soft_hex @ anchors_norm.T
        group_weights = F.softmax(group_scores, dim=-1)
        mean_w  = group_weights.mean(dim=(0, 1))
        lb_loss = (mean_w * torch.log(mean_w + 1e-8)).sum()
        return group_weights, lb_loss


# ──────────────────────────────────────────────────────────────────────────────
# MultiScaleGlobalRouter — Matryoshka Q2→Q3→Q6 роутер
# ──────────────────────────────────────────────────────────────────────────────

class MultiScaleGlobalRouter(nn.Module):
    """Matryoshka роутер: Q2→Q3→Q6 три масштаба, обучаемое смешивание.

    Иерархия:
        Q2 (4 вершины, 2 бита)  → грубый выбор группы (3 группы из 4 Q2-вершин)
        Q3 (8 вершин,  3 бита)  → средний масштаб, уточнение доменов
        Q6 (64 вершины, 6 бит)  → точный Q6-сигнал + hex_weights для HexMoE

    Смешивание:
        group_weights = softmax(w2·score_q2 + w3·score_q3 + w6·score_q6)
        hex_weights   = softmax(Q6-сходство / T)  — для Q6ExpertBank

    Q2→группа маппинг:
        Q2 вершина (-1,-1) → ABSTRACT  (отрицание×2 = углублённость)
        Q2 вершина (-1,+1) → DYNAMIC   (контраст = движение)
        Q2 вершина (+1,-1) → CONCRETE  (контраст = осязаемость)
        Q2 вершина (+1,+1) → среднее ABSTRACT+CONCRETE (оба позитивны)
    """

    # Маппинг Q2-вершин на группы (вершина 0..3 → group_idx)
    #
    # hexgraph (спектральный): вершины Q2 в кодировке {-1,+1} где +1=Yang=1-бит:
    #   (-1,-1) = 0 Yang-битов → нижняя полусфера Q2 → CONCRETE (Kun/Земля)
    #   (-1,+1) = 1 Yang-бит  → экватор Q2 (λ=0)    → DYNAMIC
    #   (+1,-1) = 1 Yang-бит  → экватор Q2 (λ=0)    → DYNAMIC
    #   (+1,+1) = 2 Yang-бита → верхняя полусфера Q2 → ABSTRACT (Qian/Небо)
    #
    # Старый маппинг [0,1,2,0]: (-1,-1)→ABSTRACT, (+1,+1)→ABSTRACT — семантически
    # неверен: Kun(000000) = Земля = CONCRETE, а не ABSTRACT. Ошибка давала 2:1:1
    # дисбаланс в пользу ABSTRACT на уровне Q2.
    #
    # Новый маппинг [2,1,1,0]: 1:2:1 — DYNAMIC получает 50% Q2-голосов, что
    # компенсирует его структурное недопредставление на Q6-уровне.
    _Q2_TO_GROUP: List[int] = [2, 1, 1, 0]  # (−−)→CON, (−+)→DYN, (+−)→DYN, (++)→ABS

    def __init__(self, d_model: int, group_names: List[str]):
        super().__init__()
        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)          # (64, 6)
        self.register_buffer('q2_verts', _make_sub_hypercube(2))  # (4, 2)
        self.register_buffer('q3_verts', _make_sub_hypercube(3))  # (8, 3)
        self.group_names = group_names
        n_groups = len(group_names)

        # Масштаб-специфичные проекции
        self.norm     = nn.LayerNorm(d_model)
        self.proj_q2  = nn.Linear(d_model, 2, bias=False)
        self.proj_q3  = nn.Linear(d_model, 3, bias=False)
        self.proj_q6  = nn.Linear(d_model, 6, bias=False)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.5)))

        # Обучаемые веса смешивания масштабов (log-domain для устойчивости)
        self.log_scale_mix = nn.Parameter(torch.zeros(3))   # [w_Q2, w_Q3, w_Q6]

        # Q2 → group matrix (фиксированный маппинг, не обучаемый)
        q2_to_group = torch.zeros(4, n_groups)
        for v_idx, g_idx in enumerate(self._Q2_TO_GROUP):
            if g_idx < n_groups:
                q2_to_group[v_idx, g_idx] = 1.0
        self.register_buffer('q2_to_group', q2_to_group)    # (4, n_groups)

        # Q3 якоря: обучаемый маппинг 8 Q3-вершин → n_groups
        # Инициализируем статическим приближением, затем дообучаем
        q3_init = self._build_q3_anchors(n_groups)
        self.q3_to_group = nn.Parameter(q3_init)             # (8, n_groups), обучаемый

        # Q6 якоря для групп (среднее по доменам группы)
        # hexgeom: нормализуем до единичной нормы — иначе DYNAMIC anchor
        # (norm≈1.414) проигрывает ABSTRACT/CONCRETE (norm≈2.0) на 29%.
        group_anchors = []
        for g in group_names:
            vecs = torch.stack([hexagrams[DOMAIN_Q6_IDX[d]] for d in DOMAIN_GROUPS[g]])
            group_anchors.append(vecs.mean(0))
        anch = torch.stack(group_anchors)                   # (n_groups, 6)
        anch = F.normalize(anch, dim=-1)                    # единичная норма
        self.register_buffer('group_anchors', anch)         # (n_groups, 6)

        # ── WHT спектральные маски полусфер Q6 (Этап 7) ─────────────────────────
        # Q6 гиперкуб: λ_k = 6-2k; полусферы по числу Yang-битов:
        #   ABSTRACT  (λ=+4): bitcount ≥ 5  →  7 вершин  (Qian-сторона)
        #   DYNAMIC   (λ= 0): bitcount = 3  → 20 вершин  (экватор, λ=0)
        #   CONCRETE  (λ=-4): bitcount ≤ 1  →  7 вершин  (Kun-сторона)
        # Нормировка / mask.sum() убирает структурный перевес DYNAMIC (20 vs 7).
        _bits = torch.tensor([bin(i).count('1') for i in range(64)],
                             dtype=torch.float32)
        self.register_buffer('wht_abstract', (_bits >= 5).float())   # (64,)
        self.register_buffer('wht_dynamic',  (_bits == 3).float())   # (64,)
        self.register_buffer('wht_concrete', (_bits <= 1).float())   # (64,)
        # Обучаемый скаляр смешения cosine↔WHT; init=-2 → sigmoid≈0.12 (WHT слаб
        # при старте, позволяет загрузить существующий чекпоинт без потери качества)
        self.log_wht_mix = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _build_q3_anchors(n_groups: int) -> torch.Tensor:
        """Q3 (8 вершин, 3 бита) → soft group matrix (8, n_groups).

        hexgraph (спектральный): Q3 — это куб с собственными значениями λ_k=3-2k:
          k=0 (λ=+3): 0 Yang-битов (−−−) → чистый CONCRETE (Kun)
          k=1 (λ=+1): 1 Yang-бит          → DYNAMIC тяготение
          k=2 (λ=−1): 2 Yang-бита         → ABSTRACT тяготение
          k=3 (λ=−3): 3 Yang-бита (+ + +) → чистый ABSTRACT (Qian)

        Маппинг по числу Yang-битов (+1 в {-1,+1} кодировке):
          0 битов → CONCRETE (уверенно)
          1 бит   → DYNAMIC (уверенно, 2 бита от одного из экватора Q3)
          2 бита  → ABSTRACT тяготение (смешанно, с DYNAMIC)
          3 бита  → ABSTRACT (уверенно)

        Это заменяет старую эвристику (bit0/bit1 знаки) на спектрально-обоснованную.
        """
        q3_map = torch.zeros(8, n_groups)
        for i in range(8):
            # i в {-1,+1}^3: +1 = Yang = "1 бит", -1 = Yin = "0 бит"
            # verts = list(product([-1,1], repeat=3)), поэтому:
            # i=0 (-1,-1,-1): 0 Yang = CONCRETE
            # i=1 (-1,-1,+1): 1 Yang = DYNAMIC
            # i=2 (-1,+1,-1): 1 Yang = DYNAMIC
            # i=3 (-1,+1,+1): 2 Yang = ABSTRACT-тяготение
            # i=4 (+1,-1,-1): 1 Yang = DYNAMIC
            # i=5 (+1,-1,+1): 2 Yang = ABSTRACT-тяготение
            # i=6 (+1,+1,-1): 2 Yang = ABSTRACT-тяготение
            # i=7 (+1,+1,+1): 3 Yang = ABSTRACT
            yang_count = bin(i).count('1')  # число Yang-битов в бинарном представлении
            if yang_count == 0:
                # (---) Kun = чистый CONCRETE
                q3_map[i, 2 % n_groups] = 1.0
            elif yang_count == 1:
                # 1 Yang = экватор Q3 → DYNAMIC
                q3_map[i, 1 % n_groups] = 1.0
            elif yang_count == 2:
                # 2 Yang = слабый ABSTRACT с примесью DYNAMIC
                q3_map[i, 0 % n_groups] = 0.6
                q3_map[i, 1 % n_groups] = 0.4
            else:
                # (+++) Qian = чистый ABSTRACT
                q3_map[i, 0 % n_groups] = 1.0
        # Нормализуем строки
        row_sum = q3_map.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return q3_map / row_sum

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.1, 5.0)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            group_weights: (B, T, n_groups)  — итоговые веса групп
            hex_weights:   (B, T, 64)         — Q6 hex веса для Q6ExpertBank
            lb_loss:       scalar
        """
        h = self.norm(x)                                               # (B, T, d)
        T_inv = self.temperature

        # ── Q2 масштаб ──────────────────────────────────────────────────────
        soft_q2  = torch.tanh(self.proj_q2(h))                        # (B, T, 2)
        sim_q2   = soft_q2 @ self.q2_verts.T                          # (B, T, 4)
        w_q2     = F.softmax(sim_q2 / T_inv, dim=-1)                  # (B, T, 4)
        score_q2 = w_q2 @ self.q2_to_group                            # (B, T, n_groups)

        # ── Q3 масштаб ──────────────────────────────────────────────────────
        soft_q3  = torch.tanh(self.proj_q3(h))                        # (B, T, 3)
        sim_q3   = soft_q3 @ self.q3_verts.T                          # (B, T, 8)
        w_q3     = F.softmax(sim_q3 / T_inv, dim=-1)                  # (B, T, 8)
        # q3_to_group — обучаемый параметр; нормализуем строки на лету
        q3_map   = F.softmax(self.q3_to_group, dim=-1)                 # (8, n_groups)
        score_q3 = w_q3 @ q3_map                                       # (B, T, n_groups)

        # ── Q6 масштаб ──────────────────────────────────────────────────────
        soft_q6  = torch.tanh(self.proj_q6(h))                        # (B, T, 6)
        sim_q6   = soft_q6 @ self.hexagrams.T                         # (B, T, 64)
        hex_weights = F.softmax(sim_q6 / T_inv, dim=-1)               # (B, T, 64)
        soft_hex    = hex_weights @ self.hexagrams                     # (B, T, 6)
        # hexgeom: нормализуем on-the-fly (устойчиво к чекпоинтам)
        anchors_norm  = F.normalize(self.group_anchors, dim=-1)        # (n_groups, 6)
        score_q6_cos  = soft_hex @ anchors_norm.T                      # (B, T, n_groups)

        # ── WHT спектральный роутинг (Этап 7) ───────────────────────────────────
        # Для каждой полусферы суммируем вероятности hex_weights и нормируем
        # на размер полусферы → при равномерном hex_weights даёт 1/64 для всех,
        # устраняя структурный перевес DYNAMIC (20 вершин vs 7 у ABSTRACT/CONCRETE).
        score_q6_wht = torch.stack([
            (hex_weights * self.wht_abstract).sum(-1) / self.wht_abstract.sum(),
            (hex_weights * self.wht_dynamic ).sum(-1) / self.wht_dynamic.sum(),
            (hex_weights * self.wht_concrete).sum(-1) / self.wht_concrete.sum(),
        ], dim=-1)                                                      # (B, T, n_groups)
        wht_alpha = torch.sigmoid(self.log_wht_mix)                    # ∈ (0,1)
        score_q6  = (1.0 - wht_alpha) * score_q6_cos + wht_alpha * score_q6_wht

        # ── Смешивание масштабов ─────────────────────────────────────────────
        scale_mix    = F.softmax(self.log_scale_mix, dim=0)            # (3,)
        mixed_scores = (scale_mix[0] * score_q2 +
                        scale_mix[1] * score_q3 +
                        scale_mix[2] * score_q6)
        group_weights = F.softmax(mixed_scores, dim=-1)                # (B, T, n_groups)

        # ── Load-balance loss ────────────────────────────────────────────────
        mean_gw  = group_weights.mean(dim=(0, 1))
        mean_hex = hex_weights.mean(dim=(0, 1))
        lb_loss  = ((mean_gw  * torch.log(mean_gw  + 1e-8)).sum() +
                    (mean_hex * torch.log(mean_hex + 1e-8)).sum() * 0.1)

        return group_weights, hex_weights, lb_loss


# ──────────────────────────────────────────────────────────────────────────────
# HierarchicalMoEFFN — главный компонент, заменяет _ffn в Variant3Block
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HMoEConfig:
    d_model:            int   = 128
    expert_expansion:   int   = 2      # expansion factor внутри MicroExpert
    top_k_experts:      int   = 2      # top-k внутри группы
    lb_loss_weight:     float = 0.01   # вес load-balancing loss
    bridge_weight:      float = 0.5    # вес BridgeExpert выходов
    use_multiscale:     bool  = True   # MultiScaleGlobalRouter vs GlobalRouter
    use_hex_tier:       bool  = False  # 4-й уровень: Q6ExpertBank (64 эксперта)
    hex_tier_top_k:     int   = 4      # top-k для Q6ExpertBank
    hex_tier_weight:    float = 0.3    # вес 4-го уровня в финальном выходе
    hex_tier_d_ff_mult: int   = 1      # d_ff = d_model * hex_tier_d_ff_mult
    # ── Совместный 4-уровневый топологический loss (схема Крюкова) ────────────
    # Уровень 1 (Формула / Математика): LCI → π  →  |w_A - w_B|² → 0
    lambda_lci:         float = 0.1
    # Уровень 2 (Архетип / Физика): ни одна группа не доминирует выше порога
    lambda_balance:     float = 0.05
    # Уровень 3 (Алгоритм / Химия): DYNAMIC ≥ 0.20 (точка пересечения жива)
    lambda_dynamic:     float = 0.1
    # Порог доминирования для lambda_balance (по умолчанию 40%)
    balance_threshold:  float = 0.40
    # hexphys (Ising): температурная регуляризация роутера к T_c(Q6)=3.0
    # T_c = z·J/2 = 6·1/2 = 3.0 (mean-field для 6-мерного гиперкуба)
    # Штраф: (log_T - log_T_c)² если T < T_c (упорядоченная фаза → доминирование)
    lambda_temp_reg:    float = 0.0
    ising_T_c:          float = 3.0   # критическая температура Q6


class HierarchicalMoEFFN(nn.Module):
    """Иерархический MoE FFN с 4-уровневой архитектурой.

    Поток (полный, use_hex_tier=True):
        Топология восьмёрки (Крюков / Алгоритм Скарабея):

        x → GlobalRouter → group_weights
          ↓
        Петля A: ABSTRACT эксперты (биты 0-2, NOOS+COSMO)
        Петля B: CONCRETE эксперты (биты 3-5, GEO+HYDRO)
          ↓
        BidirBridgeExpert (точка пересечения, экватор Q6):
          A→B fwd + B→A bwd → x_crossing
          ↓
        DYNAMIC эксперты получают x_crossing (AERO+PYRO)
          ↓
        combined = A + crossing + DYNAMIC(crossing) + B
        out = out_proj(out_norm(combined))
    """

    def __init__(self, cfg: HMoEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        group_names = list(DOMAIN_GROUPS.keys())  # ["ABSTRACT", "DYNAMIC", "CONCRETE"]

        # Уровень 1: Global router (MultiScale или legacy)
        if cfg.use_multiscale:
            self.global_router = MultiScaleGlobalRouter(d, group_names)
        else:
            self.global_router = GlobalRouter(d, group_names)

        # Уровень 2-3: MicroExperts + GroupRouters
        self.micro_experts = nn.ModuleDict({
            cluster: MicroExpert(d, cfg.expert_expansion)
            for cluster in CLUSTER_TO_DOMAIN.keys()
        })
        self.group_to_clusters: Dict[str, List[str]] = {g: [] for g in group_names}
        for cluster, domain in CLUSTER_TO_DOMAIN.items():
            self.group_to_clusters[DOMAIN_TO_GROUP[domain]].append(cluster)

        self.group_routers = nn.ModuleDict({
            g: GroupRouter(d, clusters)
            for g, clusters in self.group_to_clusters.items()
        })

        # ── Топология восьмёрки (Крюков / Алгоритм Скарабея) ─────────────────
        # Один BidirBridgeExpert — точка пересечения петли A и петли B.
        # Заменяет три симметричных BridgeExperts: DYNAMIC становится
        # не рядовой группой, а медиатором между ABSTRACT и CONCRETE.
        self.crossing = BidirBridgeExpert(d)

        # Уровень 4: Q6ExpertBank (64 эксперта, опционально)
        if cfg.use_hex_tier:
            d_ff = d * cfg.hex_tier_d_ff_mult
            self.hex_tier = Q6ExpertBank(d, d_ff, top_k=cfg.hex_tier_top_k)
        else:
            self.hex_tier = None

        # Финальная проекция
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d, bias=False)

    def _topology_loss(self, group_weights: torch.Tensor) -> torch.Tensor:
        """Совместный 4-уровневый топологический loss (схема Крюкова).

        group_weights: (B, T, 3) — [ABSTRACT, DYNAMIC, CONCRETE]

        Уровень 1 (Формула / Математика):
            LCI = (1 - |w_A - w_B|) * π → π
            ↔  |w_A - w_B|² → 0

        Уровень 2 (Архетип / Физика):
            Ни одна группа не доминирует выше balance_threshold.
            Штраф: relu(w - threshold)² для каждой группы.

        Уровень 3 (Алгоритм / Химия):
            DYNAMIC (точка пересечения) не схлопывается.
            Штраф: relu(0.20 - w_X) — если DYNAMIC < 20%.
        """
        all_zero = (self.cfg.lambda_lci == 0.0 and
                    self.cfg.lambda_balance == 0.0 and
                    self.cfg.lambda_dynamic == 0.0 and
                    self.cfg.lambda_temp_reg == 0.0)
        if all_zero:
            return torch.tensor(0.0, device=group_weights.device)

        gw = group_weights.mean(dim=(0, 1))     # (3,) — средние веса групп
        w_a, w_x, w_b = gw[0], gw[1], gw[2]    # ABSTRACT, DYNAMIC, CONCRETE

        # Уровень 1: |w_A - w_B|² — баланс петель восьмёрки (LCI → π)
        lci_loss = (w_a - w_b).pow(2)

        # Уровень 2: штраф за доминирование любой группы
        thr = self.cfg.balance_threshold
        balance_loss = (F.relu(w_a - thr).pow(2) +
                        F.relu(w_b - thr).pow(2) +
                        F.relu(w_x - thr).pow(2))

        # Уровень 3: DYNAMIC не схлопывается (≥ 20%)
        dynamic_loss = F.relu(0.20 - w_x)

        # hexphys (Ising T_c): роутер не должен уходить в упорядоченную фазу.
        # T_c = 3.0 (mean-field Q6: z·J/2 = 6·1/2).
        # Штраф только когда T < T_c (упорядоченная фаза → доминирование группы).
        temp_reg_loss = torch.tensor(0.0, device=group_weights.device)
        if self.cfg.lambda_temp_reg > 0.0:
            router = self.global_router
            if hasattr(router, 'log_temp'):
                T_now = router.log_temp.exp()
                T_c   = torch.tensor(self.cfg.ising_T_c, device=group_weights.device)
                # Только если T < T_c (в упорядоченной фазе)
                temp_reg_loss = F.relu(T_c - T_now).pow(2)

        return (self.cfg.lambda_lci      * lci_loss      +
                self.cfg.lambda_balance  * balance_loss   +
                self.cfg.lambda_dynamic  * dynamic_loss   +
                self.cfg.lambda_temp_reg * temp_reg_loss)

    def _run_group(self, x: torch.Tensor, group: str,
                   group_weights: torch.Tensor, g_idx: int
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Вычислить выход одной группы экспертов, её lb_loss и expert_weights.

        Returns:
            (weighted_out, lb_loss, expert_weights_mean)
            expert_weights_mean: (n_clusters_in_group,) — средние веса экспертов
        """
        clusters = self.group_to_clusters[group]
        expert_weights, _, lb = self.group_routers[group](
            x, top_k=self.cfg.top_k_experts)
        expert_outs = torch.stack(
            [self.micro_experts[c](x) for c in clusters], dim=-1
        )                                                      # (B, T, d, n)
        group_out = (expert_outs * expert_weights.unsqueeze(-2)).sum(-1)
        gw = group_weights[..., g_idx:g_idx+1]
        return gw * group_out, lb, expert_weights.mean(dim=(0, 1))  # (n,)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Поток по топологии восьмёрки (Алгоритм Скарабея / Крюков):

            x → GlobalRouter → group_weights
              ↓
            Петля A  : ABSTRACT эксперты (биты 0-2, верхняя полусфера Q6)
              ↓
            Петля B  : CONCRETE эксперты (биты 3-5, нижняя полусфера Q6)
              ↓
            Пересечение: BidirBridgeExpert(A, B)
              fwd A→B: специализация → обобщение
              bwd B→A: обобщение → специализация
              → x_crossing (обогащённый двунаправленный сигнал)
              ↓
            DYNAMIC эксперты получают x_crossing (не исходный x!)
              эксперты экватора обрабатывают уже скрещённый сигнал
              ↓
            Финальный синтез: A + crossing + DYNAMIC(crossing) + B
        """
        B, T, d = x.shape
        info: Dict[str, torch.Tensor] = {}
        total_lb_loss = torch.tensor(0.0, device=x.device)

        # ── 1. GlobalRouter ───────────────────────────────────────────────────
        if self.cfg.use_multiscale:
            group_weights, hex_weights, lb_global = self.global_router(x)
            info['hex_weights'] = hex_weights.detach()
        else:
            group_weights, lb_global = self.global_router(x)
            hex_weights = None
        total_lb_loss = total_lb_loss + lb_global
        info['group_weights'] = group_weights.detach()

        # ── 1b. Топологический loss уровней 1-3 (добавляется ДО масштабирования
        #        lb_loss_weight, чтобы иметь прямой контроль через lambda_*)
        topo_loss = self._topology_loss(group_weights)

        # ── 2. Петля A: ABSTRACT (верхняя полусфера, биты 0-2) ───────────────
        out_a, lb_a, ew_abs = self._run_group(x, "ABSTRACT", group_weights, 0)
        total_lb_loss = total_lb_loss + lb_a
        info['lb_ABSTRACT'] = lb_a.detach()

        # ── 3. Петля B: CONCRETE (нижняя полусфера, биты 3-5) ────────────────
        out_b, lb_b, ew_con = self._run_group(x, "CONCRETE", group_weights, 2)
        total_lb_loss = total_lb_loss + lb_b
        info['lb_CONCRETE'] = lb_b.detach()

        # ── 4. Точка пересечения: двунаправленный синтез A ↔ B ───────────────
        #   fwd: ABSTRACT → запрашивает → CONCRETE  (специализация→обобщение)
        #   bwd: CONCRETE → запрашивает → ABSTRACT  (обобщение→специализация)
        #   alpha = sigmoid(log_alpha) управляет балансом fwd/bwd
        x_crossing = self.crossing(out_a, out_b)              # (B, T, d)
        info['crossing_alpha'] = torch.sigmoid(
            self.crossing.log_alpha).detach()

        # ── 5. DYNAMIC: эксперты экватора получают x_crossing ────────────────
        #   Аналог «точки пересечения» в _figure8_walk:
        #   hexagrams with exactly 3 bits set → AERO, PYRO
        out_d, lb_d, ew_dyn = self._run_group(
            x_crossing, "DYNAMIC", group_weights, 1)
        total_lb_loss = total_lb_loss + lb_d
        info['lb_DYNAMIC'] = lb_d.detach()

        # ── 5b. 6-domain routing entropy (Этап 9) ────────────────────────────
        # domain_w_i = group_w_g * expert_w_{g,i} — совместная вероятность
        # порядок: [NOOS, COSMO, AERO, PYRO, GEO, HYDRO]
        # (= [ABS×ew_abs[0], ABS×ew_abs[1], DYN×ew_dyn[0], DYN×ew_dyn[1],
        #    CON×ew_con[0],   CON×ew_con[1]])
        gw_mean = group_weights.mean(dim=(0, 1))               # (3,)
        domain_w = torch.cat([
            gw_mean[0] * ew_abs,   # ABSTRACT: NOOS, COSMO
            gw_mean[1] * ew_dyn,   # DYNAMIC:  AERO, PYRO
            gw_mean[2] * ew_con,   # CONCRETE: GEO,  HYDRO
        ])                                                      # (6,)
        domain_w = domain_w / (domain_w.sum() + 1e-8)          # нормировка
        # Entropy в БИТАХ (log2) для совместимости с C_etd ≈ 2.61 бит/шаг
        _ln2 = torch.tensor(0.6931471805599453, device=x.device)  # ln(2)
        routing_entropy = -(domain_w * torch.log(domain_w + 1e-8)).sum() / _ln2
        # eff = H / log2(6) ∈ [0,1]; eff=1 → равномерный роутинг
        routing_eff = routing_entropy / (torch.log(
            torch.tensor(6.0, device=x.device)) / _ln2 + 1e-8)
        info['domain_weights']   = domain_w.detach()           # (6,)
        info['routing_entropy']  = routing_entropy.detach()    # скаляр (бит)
        info['routing_eff']      = routing_eff.detach()        # ∈[0,1]

        # ── 6. Финальный синтез: A + crossing + DYNAMIC + B ──────────────────
        combined = out_a + x_crossing + out_d + out_b

        # ── 7. Q6ExpertBank (4-й уровень, опционально) ───────────────────────
        if self.hex_tier is not None and hex_weights is not None:
            hex_out = self.hex_tier(x, hex_weights)
            lb_hex  = self.hex_tier.load_balance_loss(hex_weights)
            total_lb_loss = total_lb_loss + lb_hex * 0.1
            combined = combined + self.cfg.hex_tier_weight * hex_out
            info['hex_tier_active'] = torch.tensor(1.0, device=x.device)

        out = self.out_proj(self.out_norm(combined))
        # lb_loss: load-balance (масштабированный) + topology (прямой)
        info['lb_loss'] = total_lb_loss * self.cfg.lb_loss_weight + topo_loss
        info['topo_loss'] = topo_loss.detach()
        return out, info


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты для поэтапного обучения
# ──────────────────────────────────────────────────────────────────────────────

TRAINING_STAGES = {
    1: {
        "name": "MicroExperts",
        "description": "Обучение микро-экспертов по кластерам (остальное заморожено)",
        "unfreeze": ["micro_experts"],
        "freeze":   ["global_router", "group_routers", "crossing", "out_proj"],
    },
    2: {
        "name": "GroupRouters",
        "description": "Обучение групповых роутеров + продолжение экспертов",
        "unfreeze": ["micro_experts", "group_routers"],
        "freeze":   ["global_router", "crossing"],
    },
    3: {
        "name": "GlobalRouter",
        "description": "Обучение глобального роутера + всё вместе",
        "unfreeze": ["micro_experts", "group_routers", "global_router", "out_proj"],
        "freeze":   ["crossing"],
    },
    4: {
        "name": "Crossing",
        "description": "Обучение точки пересечения (BidirBridgeExpert) + GlobalRouter",
        "unfreeze": ["crossing", "global_router"],
        "freeze":   [],  # micro_experts и routers можно заморозить для экономии
    },
    5: {
        "name": "JointFinetune",
        "description": "Совместная финальная настройка всех компонентов",
        "unfreeze": [],  # всё разморожено
        "freeze":   [],
    },
    6: {
        "name": "HexTier",
        "description": "4-й уровень: Q6ExpertBank (64 Q6-эксперта)",
        "unfreeze": ["hex_tier", "global_router"],
        "freeze":   ["micro_experts", "group_routers"],
    },
}


def set_moe_stage(moe: HierarchicalMoEFFN, stage: int) -> List[str]:
    """Устанавливает этап обучения: замораживает/размораживает нужные части.

    Returns: список размороженных модулей.
    """
    if stage not in TRAINING_STAGES:
        raise ValueError(f"Stage {stage} не существует. Доступны: {list(TRAINING_STAGES)}")

    cfg = TRAINING_STAGES[stage]
    unfrozen = []

    if not cfg["freeze"] and not cfg["unfreeze"]:
        # Stage 5: всё разморожено
        for name, param in moe.named_parameters():
            param.requires_grad = True
            unfrozen.append(name)
        return unfrozen

    # Сначала замораживаем всё
    for param in moe.parameters():
        param.requires_grad = False

    # Размораживаем нужные модули
    for module_name in cfg["unfreeze"]:
        module = getattr(moe, module_name, None)
        if module is None:
            continue
        for name, param in module.named_parameters():
            param.requires_grad = True
            unfrozen.append(f"{module_name}.{name}")

    return unfrozen


def get_stage_info(stage: int) -> str:
    if stage not in TRAINING_STAGES:
        return f"Unknown stage {stage}"
    s = TRAINING_STAGES[stage]
    return (f"Stage {stage}: {s['name']}\n"
            f"  {s['description']}\n"
            f"  Unfreeze: {s['unfreeze']}\n"
            f"  Freeze:   {s['freeze']}")
