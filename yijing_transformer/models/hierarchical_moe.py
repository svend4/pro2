"""
hierarchical_moe.py — Иерархическая Mixture-of-Experts архитектура.

Схема:
    Input
      │
      ▼
    [Global Router]   ← Q6-сигнал → группа экспертов (NOOS, AERO, GEO...)
      │
      ├─► [Group Router A]  ──► MicroExpert_Theory
      │                     ──► MicroExpert_Self
      │                     ──► MicroExpert_Models
      │
      ├─► [Group Router B]  ──► MicroExpert_Scripts
      │                     ──► MicroExpert_Math
      │
      └─► [Bridge Experts]  ← соединяют группы (межгрупповые мосты)
           ├── Bridge_NOOS↔AERO
           ├── Bridge_GEO↔HYDRO
           └── Bridge_Universal

Обучение поэтапно:
    Stage 1: только MicroExperts (по одному на кластер, остальные заморожены)
    Stage 2: GroupRouters + MicroExperts
    Stage 3: GlobalRouter + всё вместе
    Stage 4: BridgeExperts (межгрупповые мосты)
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
DOMAIN_Q6_IDX: Dict[str, int] = {
    "GEO":   0,
    "HYDRO": 18,
    "PYRO":  45,
    "AERO":  6,
    "COSMO": 27,
    "NOOS":  63,
}

# Мосты между группами
BRIDGE_PAIRS: List[Tuple[str, str]] = [
    ("ABSTRACT", "DYNAMIC"),
    ("DYNAMIC",  "CONCRETE"),
    ("ABSTRACT", "CONCRETE"),
]


def _make_hexagrams() -> torch.Tensor:
    import itertools
    verts = list(itertools.product([-1.0, 1.0], repeat=6))
    return torch.tensor(verts, dtype=torch.float32)  # (64, 6)


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

class BridgeExpert(nn.Module):
    """Эксперт-мост: принимает конкатенацию двух групп, синтезирует связь."""

    def __init__(self, d_model: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4,
                                                 batch_first=True, bias=False)
        self.ffn_gate   = nn.Linear(d_model, d_model * 2, bias=False)
        self.ffn_val    = nn.Linear(d_model, d_model * 2, bias=False)
        self.ffn_out    = nn.Linear(d_model * 2, d_model, bias=False)
        self.norm_q     = nn.LayerNorm(d_model)
        self.norm_kv    = nn.LayerNorm(d_model)
        self.norm_ffn   = nn.LayerNorm(d_model)
        self.gate       = nn.Parameter(torch.tensor(-2.0))  # starts ~0.12

    def forward(self, x_query: torch.Tensor,
                x_key: torch.Tensor) -> torch.Tensor:
        """
        x_query: (B, T, d) — выход группы A
        x_key:   (B, T, d) — выход группы B
        returns: (B, T, d) — синтез моста
        """
        q  = self.norm_q(x_query)
        kv = self.norm_kv(x_key)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x = x_query + attn_out

        h = self.norm_ffn(x)
        x = x + self.ffn_out(F.silu(self.ffn_gate(h)) * self.ffn_val(h))
        return x * torch.sigmoid(self.gate)


# ──────────────────────────────────────────────────────────────────────────────
# GroupRouter — роутер внутри одной группы доменов
# ──────────────────────────────────────────────────────────────────────────────

class GroupRouter(nn.Module):
    """Роутер внутри группы: Q6-сигнал → веса микро-экспертов группы."""

    def __init__(self, d_model: int, expert_names: List[str]):
        super().__init__()
        n = len(expert_names)
        self.expert_names = expert_names
        self.proj  = nn.Linear(d_model, n, bias=True)
        self.norm  = nn.LayerNorm(d_model)
        # load-balance буфер
        self.register_buffer('_load_counts', torch.zeros(n))

    def forward(self, x: torch.Tensor, top_k: int = 2
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights:  (B, T, n)  — softmax веса по экспертам
            indices:  (B, T, k)  — top-k индексы
            lb_loss:  scalar     — load-balancing loss
        """
        logits  = self.proj(self.norm(x))                      # (B, T, n)
        weights = F.softmax(logits, dim=-1)                    # (B, T, n)

        k = min(top_k, weights.shape[-1])
        topk_vals, indices = torch.topk(weights, k, dim=-1)   # (B, T, k)
        sparse_w = torch.zeros_like(weights).scatter_(-1, indices, topk_vals)
        sparse_w = sparse_w / (sparse_w.sum(dim=-1, keepdim=True) + 1e-8)

        # Load-balancing loss (следим чтобы эксперты использовались равномерно)
        mean_w = weights.mean(dim=(0, 1))                      # (n,)
        lb_loss = (mean_w * torch.log(mean_w + 1e-8)).sum().neg()  # энтропия

        return sparse_w, indices, lb_loss


# ──────────────────────────────────────────────────────────────────────────────
# GlobalRouter — роутер верхнего уровня (группа → group_weights)
# ──────────────────────────────────────────────────────────────────────────────

class GlobalRouter(nn.Module):
    """Q6-based роутер верхнего уровня: определяет веса по группам.

    Использует гексаграммные якоря доменов для мягкого routing.
    """

    def __init__(self, d_model: int, group_names: List[str]):
        super().__init__()
        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)
        self.group_names = group_names
        n_groups = len(group_names)

        self.q6_proj  = nn.Linear(d_model, 6, bias=False)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.group_proj = nn.Linear(6, n_groups, bias=True)
        self.norm = nn.LayerNorm(d_model)

        # Q6 якоря для каждой группы (среднее по доменам группы)
        group_anchors = []
        for g in group_names:
            domain_vecs = torch.stack([
                hexagrams[DOMAIN_Q6_IDX[d]] for d in DOMAIN_GROUPS[g]
            ])
            group_anchors.append(domain_vecs.mean(0))
        self.register_buffer('group_anchors',
                             torch.stack(group_anchors))  # (n_groups, 6)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.1, 5.0)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            group_weights: (B, T, n_groups) — softmax веса групп
            lb_loss:       scalar
        """
        h = self.norm(x)
        soft_bits = torch.tanh(self.q6_proj(h))                        # (B, T, 6)

        # Мягкое Q6 присваивание через сходство с гексаграммами
        sim = soft_bits @ self.hexagrams.T                              # (B, T, 64)
        hex_w = F.softmax(sim / self.temperature, dim=-1)               # (B, T, 64)
        soft_hex = hex_w @ self.hexagrams                               # (B, T, 6)

        # Близость к якорям групп
        group_scores  = soft_hex @ self.group_anchors.T                 # (B, T, n_groups)
        group_weights = F.softmax(group_scores, dim=-1)

        mean_w = group_weights.mean(dim=(0, 1))
        lb_loss = (mean_w * torch.log(mean_w + 1e-8)).sum().neg()

        return group_weights, lb_loss


# ──────────────────────────────────────────────────────────────────────────────
# HierarchicalMoEFFN — главный компонент, заменяет _ffn в Variant3Block
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HMoEConfig:
    d_model:        int   = 128
    expert_expansion: int = 2      # expansion factor внутри MicroExpert
    top_k_experts:  int   = 2      # top-k внутри группы
    lb_loss_weight: float = 0.01   # вес load-balancing loss
    bridge_weight:  float = 0.5    # вес BridgeExpert выходов


class HierarchicalMoEFFN(nn.Module):
    """Иерархический MoE FFN.

    Заменяет единый _ffn(x) в Variant3Block.

    Поток:
        x → GlobalRouter → group_weights (3 группы)
          → для каждой группы: GroupRouter → top-k MicroExperts → взвешенная сумма
          → группы взвешенно суммируются по group_weights
          → BridgeExperts синтезируют межгрупповые связи
          → финальный выход = group_out + bridge_contribution
    """

    def __init__(self, cfg: HMoEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        group_names = list(DOMAIN_GROUPS.keys())  # ["ABSTRACT", "DYNAMIC", "CONCRETE"]

        # Global router
        self.global_router = GlobalRouter(d, group_names)

        # MicroExperts: один на каждый кластер/домен
        self.micro_experts = nn.ModuleDict({
            cluster: MicroExpert(d, cfg.expert_expansion)
            for cluster in CLUSTER_TO_DOMAIN.keys()
        })

        # Отображение: группа → список имён кластеров
        self.group_to_clusters: Dict[str, List[str]] = {g: [] for g in group_names}
        for cluster, domain in CLUSTER_TO_DOMAIN.items():
            group = DOMAIN_TO_GROUP[domain]
            self.group_to_clusters[group].append(cluster)

        # Group routers: один на группу
        self.group_routers = nn.ModuleDict({
            g: GroupRouter(d, clusters)
            for g, clusters in self.group_to_clusters.items()
        })

        # Bridge experts: один на каждую пару групп
        self.bridge_experts = nn.ModuleDict({
            f"{a}_{b}": BridgeExpert(d)
            for a, b in BRIDGE_PAIRS
        })

        # Финальная проекция
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            out:  (B, T, d_model)
            info: dict с lb_losses и routing stats
        """
        B, T, d = x.shape
        group_names = list(DOMAIN_GROUPS.keys())
        info: Dict[str, torch.Tensor] = {}
        total_lb_loss = torch.tensor(0.0, device=x.device)

        # ── 1. Global routing ────────────────────────────────────────────────
        group_weights, lb_global = self.global_router(x)   # (B, T, 3)
        total_lb_loss = total_lb_loss + lb_global
        info['group_weights'] = group_weights.detach()

        # ── 2. Group-level routing + MicroExperts ────────────────────────────
        group_outputs: Dict[str, torch.Tensor] = {}

        for g_idx, group in enumerate(group_names):
            clusters = self.group_to_clusters[group]
            g_router = self.group_routers[group]

            # Роутер внутри группы
            expert_weights, _, lb_group = g_router(x, top_k=self.cfg.top_k_experts)
            total_lb_loss = total_lb_loss + lb_group
            info[f'lb_{group}'] = lb_group.detach()

            # Взвешенная сумма микро-экспертов группы
            group_out = torch.zeros(B, T, d, device=x.device)
            for e_idx, cluster in enumerate(clusters):
                w = expert_weights[..., e_idx:e_idx+1]           # (B, T, 1)
                expert_out = self.micro_experts[cluster](x)       # (B, T, d)
                group_out = group_out + w * expert_out

            # Масштабируем выход группы глобальным весом
            gw = group_weights[..., g_idx:g_idx+1]               # (B, T, 1)
            group_outputs[group] = gw * group_out

        # ── 3. Bridge experts ────────────────────────────────────────────────
        bridge_contribution = torch.zeros(B, T, d, device=x.device)
        bw = self.cfg.bridge_weight / len(BRIDGE_PAIRS)

        for a, b in BRIDGE_PAIRS:
            bridge = self.bridge_experts[f"{a}_{b}"]
            if a in group_outputs and b in group_outputs:
                bridge_out = bridge(group_outputs[a], group_outputs[b])
                bridge_contribution = bridge_contribution + bw * bridge_out

        # ── 4. Финальный выход ───────────────────────────────────────────────
        combined = sum(group_outputs.values()) + bridge_contribution
        out = self.out_proj(self.out_norm(combined))

        info['lb_loss'] = total_lb_loss * self.cfg.lb_loss_weight
        return out, info


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты для поэтапного обучения
# ──────────────────────────────────────────────────────────────────────────────

TRAINING_STAGES = {
    1: {
        "name": "MicroExperts",
        "description": "Обучение микро-экспертов по кластерам (остальное заморожено)",
        "unfreeze": ["micro_experts"],
        "freeze":   ["global_router", "group_routers", "bridge_experts", "out_proj"],
    },
    2: {
        "name": "GroupRouters",
        "description": "Обучение групповых роутеров + продолжение экспертов",
        "unfreeze": ["micro_experts", "group_routers"],
        "freeze":   ["global_router", "bridge_experts"],
    },
    3: {
        "name": "GlobalRouter",
        "description": "Обучение глобального роутера + всё вместе",
        "unfreeze": ["micro_experts", "group_routers", "global_router", "out_proj"],
        "freeze":   ["bridge_experts"],
    },
    4: {
        "name": "BridgeExperts",
        "description": "Обучение мостов между группами",
        "unfreeze": ["bridge_experts", "global_router"],
        "freeze":   [],  # micro_experts и routers можно заморозить для экономии
    },
    5: {
        "name": "JointFinetune",
        "description": "Совместная финальная настройка всех компонентов",
        "unfreeze": [],  # всё разморожено
        "freeze":   [],
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
