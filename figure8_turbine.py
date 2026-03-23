#!/usr/bin/env python3
"""
figure8_turbine.py — Turbine-interchange self-training loop for HMoE.

Расширяет фигуру-8 (клеверный лист) до архитектуры ТУРБИННОЙ РАЗВЯЗКИ:

  Принципы дорожной турбинной развязки → архитектура:
  ┌─────────────────────────────────────────────────────────┐
  │  Турбина (вид сверху)         HMoE-архитектура          │
  │  ─────────────────────        ─────────────────────      │
  │  Спиральные дуги              Эксперты по кругу          │
  │  Нет левых поворотов          Нет резких разворотов      │
  │  Приоритет = загрузка         TSP-порядок по LCI         │
  │  Pазгонные полосы             Residual connections        │
  │  V/C ratio (пропускная ёмк.)  gate_k / LCI_k            │
  │  Kirchhoff (ток в узле=0)     Σ(gate_k × LCI_k) ≈ π    │
  └─────────────────────────────────────────────────────────┘

4 роли экспертов (станции турбины):
  ABSTRACT  — "highway mainline"  (высокоскоростной, zoom-out)
  DYNAMIC   — "interchange hub"   (ось турбины, BidirBridge)
  CONCRETE  — "local roads"       (детализация, zoom-in)
  META      — "roundabout"        (агрегатор + Kirchhoff-узел)

Динамический multi-TSP маршрут:
  Вместо фиксированного A→X→B→X порядка — nearest-neighbor TSP
  по LCI-расстоянию: сначала посещаем эксперта, который больше
  всего отклонился от резонанса (π). Аналог: разгрузить перегруженную
  полосу сначала.

Условие Кирхгофа (Kirchhoff LCI):
  На каждом узле сумма "токов" (gate × LCI) ≈ π.
  Это превращает LCI-резонанс из эвристики в физически обоснованное
  условие минимальной энергии системы.

Внутренние семантические петли:
  После каждого шага эксперта: если LCI ухудшился (ушёл дальше от π),
  токен рециркулирует через тот же эксперт 1 раз (как U-turn в кольце).

Usage:
  python figure8_turbine.py --checkpoint hmoe_self_trained_v5.pt
  python figure8_turbine.py --fast
  python figure8_turbine.py --cycles 6 --steps_per_loop 30 --temperature 1.4
  python figure8_turbine.py --no-tsp          # фиксированный порядок (как v7)
  python figure8_turbine.py --no-recirculate  # без внутренних петель
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

# Prevent thread contention between numpy/BLAS and PyTorch (fixes ~87-min hang).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# meta_q6: temperature annealing из svend4/meta
try:
    from meta_q6 import metropolis_temperature
    _META_Q6_AVAILABLE = True
except ImportError:
    _META_Q6_AVAILABLE = False

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    HMoEConfig,
    HierarchicalMoEFFN,
    CLUSTER_TO_DOMAIN,
    DOMAIN_GROUPS,
    DOMAIN_TO_GROUP,
    set_moe_stage,
)

# Переиспользуем утилиты из self_train_hmoe
from self_train_hmoe import (
    lci_from_routing,
    lci_from_embeddings,
    micro_train,
    quality_filter,
    RagBuffer,
    _generate,
    _ids_to_text,
    _encode,
    _hex_prompt,
    _get_emb,
    _get_moes,
    _freeze_all_except,
    MODEL_CFG,
    HMOE_CFG,
    _ODD_SERIES,
    _LCI_EPSILON,
)

# ── Константы турбины ──────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4 станции турбины (роли экспертов)
_TURBINE_EXPERTS = ["ABSTRACT", "DYNAMIC", "CONCRETE", "META"]

_TURBINE_ROLES = {
    "ABSTRACT": "highway mainline  (zoom-out, обобщение)",
    "DYNAMIC":  "interchange hub   (ось, BidirBridge)",
    "CONCRETE": "local roads       (zoom-in, специализация)",
    "META":     "roundabout        (агрегатор, Kirchhoff-узел)",
}

# META использует DYNAMIC-группу но размораживает все группы (агрегация)
_EXPERT_FREEZE_MAP: Dict[str, List[str]] = {
    "ABSTRACT": ["ABSTRACT"],
    "DYNAMIC":  ["DYNAMIC"],
    "CONCRETE": ["CONCRETE"],
    "META":     ["ABSTRACT", "DYNAMIC", "CONCRETE"],  # все разморожены
}

# Вес META в gate-сумме (фиктивный, нет отдельной группы)
_META_GATE_WEIGHT = 1 / 3   # усреднение по трём группам

# Kirchhoff: отклонение Σ(gate_k × LCI_k) от π считается "энергией"
_KIRCHHOFF_EPSILON = 0.4


# ── Kirchhoff LCI ─────────────────────────────────────────────────────────────

def kirchhoff_balance(
    model: Variant3GPT,
    ids: torch.Tensor,
    expert_lci_map: Dict[str, float],
) -> Tuple[float, float]:
    """
    Вычислить Kirchhoff LCI: Σ(gate_k × LCI_k).

    Аналог закона Кирхгофа: ток в узле = Σ(проводимость × напряжение).
    Здесь: gate_k = "ток", LCI_k = "напряжение", π = целевой потенциал.

    Returns:
        kirchhoff_val  — Σ(gate_k × LCI_k)
        kirchhoff_dev  — |kirchhoff_val - π|
    """
    _, gw = lci_from_routing(model, ids)

    groups = list(DOMAIN_GROUPS.keys())  # ABSTRACT, DYNAMIC, CONCRETE
    total = 0.0
    gate_sum = 0.0

    for expert in _TURBINE_EXPERTS:
        lci_k = expert_lci_map.get(expert, math.pi)
        if expert == "META":
            gate_k = _META_GATE_WEIGHT
        else:
            gate_k = gw.get(expert, 1.0 / len(groups))
        total += gate_k * lci_k
        gate_sum += gate_k

    # Нормировать на сумму gate (аналог нормировки тока)
    if gate_sum > 1e-8:
        total /= gate_sum

    return total, abs(total - math.pi)


# ── Динамический TSP-порядок экспертов ────────────────────────────────────────

def tsp_expert_order(
    model: Variant3GPT,
    ids: torch.Tensor,
    expert_lci_history: Dict[str, float],
) -> List[str]:
    """
    Nearest-neighbor TSP: найти оптимальный порядок обхода 4 экспертов.

    Принцип турбины: нет левых поворотов, нет резких разворотов.
    Стоимость перехода expert_i → expert_j:
        cost(i→j) = |LCI_j - π|  (насколько эксперт j далёк от резонанса)
    Алгоритм: жадный nearest-neighbor, старт с самого отклонённого.

    Реализует принцип "разгрузки перегруженной полосы":
    сначала посещаем эксперта с наибольшим отклонением от π.
    """
    _, gw = lci_from_routing(model, ids)
    lci_r, _ = lci_from_routing(model, ids)

    # Оценка LCI каждого эксперта по его группе: если группа доминирует → LCI отклоняется
    expert_cost: Dict[str, float] = {}
    for expert in _TURBINE_EXPERTS:
        if expert == "META":
            # META: среднее отклонение всех групп
            hist_val = expert_lci_history.get("META", math.pi)
            expert_cost["META"] = abs(hist_val - math.pi)
        else:
            gate = gw.get(expert, 1.0 / 3)
            hist_val = expert_lci_history.get(expert, math.pi)
            # Взвесить: чем выше gate (перегружен) и дальше от π → тем приоритетнее
            deviation = abs(hist_val - math.pi)
            overload = max(gate - 1.0 / 3, 0.0)   # превышение над равной долей
            expert_cost[expert] = deviation + overload * math.pi

    # Жадный nearest-neighbor: старт с наибольшей стоимостью
    unvisited = list(_TURBINE_EXPERTS)
    unvisited.sort(key=lambda e: expert_cost[e], reverse=True)  # перегруженные первыми
    order = [unvisited.pop(0)]  # первый = самый отклонённый

    # Последующие: ближайший (минимальная стоимость) из оставшихся
    while unvisited:
        current = order[-1]
        # Избегаем резких разворотов: не возвращаемся к предыдущему
        prev = order[-2] if len(order) >= 2 else None
        candidates = [e for e in unvisited if e != prev]
        if not candidates:
            candidates = unvisited
        next_expert = min(candidates, key=lambda e: expert_cost[e])
        unvisited.remove(next_expert)
        order.append(next_expert)

    return order


# ── LCI-loss шаг (Kirchhoff как явный штраф) ─────────────────────────────────

def _lci_loss_step(model: Variant3GPT, ids: torch.Tensor, lr: float) -> float:
    """
    Один шаг градиентного спуска по Kirchhoff-отклонению:
      loss = |routing_LCI - π|
    Получает group_weights напрямую из GlobalRouter (с живым графом).
    """
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return 0.0
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    inp = ids[:, :-1] if ids.shape[1] > 1 else ids

    # Собрать group_weights с живым графом через GlobalRouter напрямую
    group_weights_list = []
    try:
        x = model.tok_emb(inp)
        for block in model.blocks:
            # Запустить global_router с grad
            moe = getattr(block, 'hmoe', None)
            if moe is not None and hasattr(moe, 'global_router'):
                result = moe.global_router(x)
                gw = result[0] if isinstance(result, tuple) else result
                if not torch.isnan(gw).any():
                    if gw.dim() == 3:
                        gw = gw.mean(dim=(0, 1))
                    elif gw.dim() == 2:
                        gw = gw.mean(dim=0)
                    group_weights_list.append(gw)
            x = block(x)
    except Exception:
        return 0.0

    if not group_weights_list:
        return 0.0

    avg_gw = torch.stack(group_weights_list).mean(0)
    groups = list(DOMAIN_GROUPS.keys())
    gw_dict = {g: avg_gw[i] for i, g in enumerate(groups)}

    w_a = gw_dict.get("ABSTRACT", avg_gw[0])
    w_b = gw_dict.get("CONCRETE", avg_gw[2] if len(avg_gw) > 2 else avg_gw[0])
    w_total = avg_gw.sum() + 1e-8
    imbalance = torch.abs(w_a / w_total - w_b / w_total)
    lci_loss = imbalance * math.pi   # цель: imbalance→0 → LCI→π → loss→0

    opt.zero_grad()
    lci_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 0.5)
    opt.step()
    return lci_loss.item()


def _tsp_2opt(order: List[str], cost_fn) -> List[str]:
    """
    2-opt улучшение TSP маршрута.
    Переставляет пары рёбер пока есть улучшение.
    cost_fn(expert) → float (меньше = лучше).
    """
    def route_cost(route):
        return sum(cost_fn(e) for e in route)

    best = list(order)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_cost(new_route) < route_cost(best) - 1e-6:
                    best = new_route
                    improved = True
    return best


# ── Одна фаза эксперта ────────────────────────────────────────────────────────

def run_expert_phase(
    model: Variant3GPT,
    ids: torch.Tensor,
    expert: str,
    n_steps: int,
    temperature: float,
    block_size: int,
    train_lr: float,
    do_train: bool,
    rag: RagBuffer,
    do_recirculate: bool = True,
    lci_loss_lambda: float = 0.0,
) -> Tuple[torch.Tensor, float, float, int]:
    """
    Выполнить одну фазу (станцию) турбины для данного эксперта.

    Включает внутреннюю семантическую мини-петлю:
    если после шага LCI ухудшился (дальше от π) — рециркулировать 1 раз.
    Аналог: автомобиль в кольце объезжает ещё один круг, если нужный съезд занят.

    Returns:
        new_ids        — токены после фазы
        lci_emb        — embedding LCI за фазу
        lci_r          — routing LCI в конце фазы
        n_generated    — число сгенерированных текстов
    """
    # Настроить заморозку для данного эксперта
    freeze_groups = _EXPERT_FREEZE_MAP[expert]
    for moe in _get_moes(model):
        _freeze_all_except(moe, freeze_groups)
    if expert == "META":
        # Разморозить BidirBridge (точка X)
        set_moe_stage(
            model.blocks[0].hmoe if hasattr(model.blocks[0], 'hmoe') else None, 4
        )

    start_ids = ids.clone()
    current_ids = ids.clone()
    n_generated = 0

    # Измерить LCI до начала фазы
    lci_r_before, _ = lci_from_routing(model, current_ids)

    for step in range(n_steps):
        gen_ids = _generate(model, current_ids, block_size, temperature, n_tokens=8)
        gen_text = _ids_to_text(gen_ids)

        # Внутренняя семантическая мини-петля (recirculation)
        if do_recirculate:
            lci_r_after, _ = lci_from_routing(model, gen_ids)
            if abs(lci_r_after - math.pi) > abs(lci_r_before - math.pi) + 0.05:
                # LCI ухудшился — рециркулировать: генерировать ещё раз с текущего
                gen_ids = _generate(model, current_ids, block_size,
                                    temperature * 0.9, n_tokens=8)
                gen_text = _ids_to_text(gen_ids)
            lci_r_before = lci_r_after

        if do_train and quality_filter(gen_text):
            lr_scale = 0.5 if expert == "CONCRETE" else (0.3 if expert == "META" else 1.0)
            micro_train(model, gen_ids, lr=train_lr * lr_scale, n_steps=2)
            # LCI-loss: явный штраф Kirchhoff после micro_train
            if lci_loss_lambda > 0.0:
                _lci_loss_step(model, gen_ids, train_lr * lr_scale * lci_loss_lambda)
            rag.add(gen_text, _get_emb(model, gen_ids))
            n_generated += 1

        current_ids = gen_ids

    # Финальные метрики фазы
    lci_emb = lci_from_embeddings(model, start_ids, current_ids)
    lci_r_final, _ = lci_from_routing(model, current_ids)

    return current_ids, lci_emb, lci_r_final, n_generated


# ── Главный цикл турбины ──────────────────────────────────────────────────────

def turbine_figure8(
    model: Variant3GPT,
    seed_texts: List[str],
    block_size: int = MODEL_CFG["block_size"] - 1,
    n_cycles: int = 4,
    steps_per_expert: int = 20,
    temperature: float = 1.4,
    temp_decay: float = 0.0,
    train_lr: float = 1e-5,
    do_train: bool = True,
    use_tsp: bool = True,
    use_tsp_2opt: bool = False,
    do_recirculate: bool = True,
    lci_loss_lambda: float = 0.0,
) -> List[Dict]:
    """
    Турбинное само-обучение HMoE.

    Каждый цикл:
      1. Вычислить LCI-расстояния для 4 экспертов
      2. Если use_tsp=True: TSP-порядок (перегруженные первыми)
         Иначе: фиксированный A→X→B→META
      3. Прогнать все 4 эксперта по спирали
      4. Проверить условие Кирхгофа: Σ(gate_k × LCI_k) ≈ π
      5. Обновить RAG из накопленных текстов
    """
    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ ТУРБИНА + HMoE")
    print(f"{'═' * 72}")
    print(f"  Циклов              : {n_cycles}")
    print(f"  Шагов/эксперт       : {steps_per_expert}")
    t_mode = f"Metropolis decay={temp_decay:.2f}" if temp_decay > 0 else "фиксированная"
    print(f"  Температура         : {temperature:.2f}  ({t_mode})")
    tsp_mode = ("2-opt" if use_tsp_2opt else "greedy") if use_tsp else "НЕТ (A→X→B→META)"
    print(f"  TSP-маршрутизация   : {tsp_mode}")
    print(f"  Рециркуляция        : {'ДА (внутренние мини-петли)' if do_recirculate else 'НЕТ'}")
    print(f"  LCI-loss λ          : {lci_loss_lambda:.3f}  {'(активен)' if lci_loss_lambda > 0 else '(выкл)'}")
    print()
    print(f"  Станции турбины:")
    for e, role in _TURBINE_ROLES.items():
        groups = _EXPERT_FREEZE_MAP[e]
        print(f"    {e:10s} : {role}  [группы: {', '.join(groups)}]")
    print()

    # RAG-буфер
    rag = RagBuffer(max_size=400)
    model.eval()
    for text in seed_texts[:50]:
        ids = _encode(text, block_size)
        emb = _get_emb(model, ids)
        rag.add(text, emb)

    print(f"  RAG-буфер           : {len(rag)} текстов")

    # История LCI по экспертам (для TSP-метрики)
    expert_lci_history: Dict[str, float] = {e: math.pi for e in _TURBINE_EXPERTS}

    # Стартовый промпт
    start_hex = random.randint(0, 63)
    current_ids = _hex_prompt(start_hex, block_size)

    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        cycle_start_time = time.perf_counter()

        # Metropolis temperature annealing
        if temp_decay > 0:
            if _META_Q6_AVAILABLE:
                cur_temp = metropolis_temperature(cycle - 1, n_cycles, temperature, T_min=0.5, decay=temp_decay)
            else:
                cur_temp = max(0.5, temperature * (temp_decay ** (cycle - 1)))
        else:
            cur_temp = temperature

        # ── Точка X: измерить текущее состояние ─────────────────────────────
        lci_r0, gw0 = lci_from_routing(model, current_ids)

        resonance_mark = (
            "✓ РЕЗОНАНС" if abs(lci_r0 - math.pi) < _LCI_EPSILON else
            f"δ={lci_r0 - math.pi:+.3f}"
        )
        print(f"\n  Цикл {cycle}/{n_cycles}  T={cur_temp:.3f}  "
              f"routing_LCI={lci_r0:.3f}  {resonance_mark}")
        print(f"    Веса: A={gw0.get('ABSTRACT',0):.3f}  "
              f"X={gw0.get('DYNAMIC',0):.3f}  "
              f"B={gw0.get('CONCRETE',0):.3f}")

        # ── TSP-порядок экспертов ────────────────────────────────────────────
        if use_tsp:
            expert_order = tsp_expert_order(model, current_ids, expert_lci_history)
            if use_tsp_2opt:
                _, gw_tmp = lci_from_routing(model, current_ids)
                cost_fn = lambda e: abs(expert_lci_history.get(e, math.pi) - math.pi)
                expert_order = _tsp_2opt(expert_order, cost_fn)
        else:
            expert_order = ["ABSTRACT", "DYNAMIC", "CONCRETE", "META"]

        print(f"    Маршрут: {' → '.join(expert_order)}")

        # ── Прогнать все 4 станции турбины ───────────────────────────────────
        cycle_lci_emb: List[float] = []
        cycle_lci_r:   List[float] = []
        cycle_generated = 0
        expert_lci_this_cycle: Dict[str, float] = {}

        for expert in expert_order:
            # Если RAG полон — обогатить промпт
            if len(rag) > 10:
                cur_emb = _get_emb(model, current_ids)
                retrieved = rag.retrieve(cur_emb, top_k=1)
                if retrieved:
                    current_ids = _encode(retrieved[0], block_size)

            new_ids, lci_emb, lci_r, n_gen = run_expert_phase(
                model        = model,
                ids          = current_ids,
                expert       = expert,
                n_steps      = steps_per_expert,
                temperature  = cur_temp,
                block_size   = block_size,
                train_lr     = train_lr,
                do_train     = do_train,
                rag          = rag,
                do_recirculate   = do_recirculate,
                lci_loss_lambda  = lci_loss_lambda,
            )

            current_ids = new_ids
            cycle_lci_emb.append(lci_emb)
            cycle_lci_r.append(lci_r)
            cycle_generated += n_gen
            expert_lci_this_cycle[expert] = lci_r

            res = "✓" if abs(lci_r - math.pi) < _LCI_EPSILON else "✗"
            print(f"    [{expert:10s}] emb_LCI={lci_emb:.3f}  "
                  f"routing_LCI={lci_r:.3f}  {res}  gen={n_gen}")

        # Обновить историю LCI
        expert_lci_history.update(expert_lci_this_cycle)

        # ── Kirchhoff-проверка ────────────────────────────────────────────────
        k_val, k_dev = kirchhoff_balance(model, current_ids, expert_lci_this_cycle)
        k_mark = "✓ KIRCHHOFF" if k_dev < _KIRCHHOFF_EPSILON else f"KΔ={k_dev:+.3f}"

        # ── Итог цикла ────────────────────────────────────────────────────────
        avg_lci_emb = sum(cycle_lci_emb) / len(cycle_lci_emb)
        avg_lci_r   = sum(cycle_lci_r)   / len(cycle_lci_r)
        n_resonant  = sum(1 for l in cycle_lci_r if abs(l - math.pi) < _LCI_EPSILON)
        elapsed     = time.perf_counter() - cycle_start_time

        print(f"    → avg_LCI_emb={avg_lci_emb:.3f}  avg_LCI_r={avg_lci_r:.3f}  "
              f"резонанс={n_resonant}/{len(_TURBINE_EXPERTS)}  "
              f"{k_mark}  gen={cycle_generated}  t={elapsed:.1f}s")

        log.append({
            "cycle":          cycle,
            "expert_order":   expert_order,
            "temperature":    round(temperature, 3),
            "lci_r0":         round(lci_r0, 4),
            "avg_lci_emb":    round(avg_lci_emb, 4),
            "avg_lci_r":      round(avg_lci_r, 4),
            "kirchhoff_val":  round(k_val, 4),
            "kirchhoff_dev":  round(k_dev, 4),
            "n_resonant":     n_resonant,
            "n_generated":    cycle_generated,
            "expert_lci":     {e: round(v, 4) for e, v in expert_lci_this_cycle.items()},
            "elapsed_s":      round(elapsed, 2),
        })

    # ── Финальный итог ────────────────────────────────────────────────────────
    n_res = sum(1 for r in log if r["n_resonant"] >= 3)
    n_k   = sum(1 for r in log if r["kirchhoff_dev"] < _KIRCHHOFF_EPSILON)
    avg_l = sum(r["avg_lci_emb"] for r in log) / len(log) if log else 0.0

    print(f"\n{'─' * 72}")
    print(f"  ИТОГ ТУРБИНЫ:")
    print(f"    Резонансных циклов (≥3/4 экспертов): {n_res}/{n_cycles}")
    print(f"    Kirchhoff-сбалансированных циклов:   {n_k}/{n_cycles}")
    print(f"    avg_LCI_emb = {avg_l:.3f}  (цель = π = {math.pi:.3f})")
    print(f"    RAG-буфер:   {len(rag)} текстов")
    print(f"    Финальный Kirchhoff: {'✓' if n_k > 0 else '✗'}")

    return log


# ── Загрузка модели ───────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str) -> Variant3GPT:
    cfg = Variant3Config(**MODEL_CFG)
    model = Variant3GPT(cfg)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Загружен чекпоинт: {checkpoint_path}")
    else:
        print(f"  [!] Чекпоинт не найден: {checkpoint_path} — используем случайные веса")
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Модель: {n_params/1e6:.2f}M параметров")
    return model


_ROOT = os.path.dirname(os.path.abspath(__file__))

def _load_seed_texts(no_corpus: bool, block_size: int) -> List[str]:
    texts: List[str] = []
    if not no_corpus:
        try:
            from repo_corpus_loader import RepoCorpusLoader
            loader = RepoCorpusLoader(_ROOT)
            for cluster in CLUSTER_TO_DOMAIN.keys():
                try:
                    for item in loader.load_cluster(cluster):
                        t = item if isinstance(item, str) else item.get("text", "")
                        if len(t) > 10:
                            texts.append(t)
                except Exception:
                    pass
            print(f"  Корпус загружен: {len(texts)} текстов")
        except Exception as e:
            print(f"  [!] Корпус не загружен: {e}")

    if not texts:
        texts = [
            "def forward(self, x): return self.linear(x)",
            "import torch; x = torch.randn(4, 128)",
            "for i, batch in enumerate(dataloader): optimizer.step()",
            "loss.backward(); optimizer.zero_grad(); scheduler.step()",
            "self.attn = nn.MultiheadAttention(d_model, n_heads)",
            "x = x + self.crossing(out_a, out_b)",
            "The hexagram represents the intersection of abstract and concrete.",
            "Kryukov figure-8: abstract loop A, concrete loop B, crossing DYNAMIC.",
            "consciousness emerges from recursive self-reference in Q6 space",
        ] * 5
        print(f"  Синтетических текстов: {len(texts)}")

    # Добавить синтетические тексты из гексаграмм
    for h in range(64):
        ids = _hex_prompt(h, block_size)
        texts.append(_ids_to_text(ids))
    print(f"  Итого seed текстов: {len(texts)}")
    return texts


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turbine-interchange самообучение HMoE (архитектура турбинной развязки)"
    )
    parser.add_argument("--checkpoint",      type=str, default="hmoe_curriculum.pt")
    parser.add_argument("--fast",            action="store_true",
                        help="2 цикла, 5 шагов/эксперт (тест)")
    parser.add_argument("--cycles",          type=int, default=4)
    parser.add_argument("--steps_per_expert",type=int, default=20,
                        help="Шагов генерации на 1 эксперта (default: 20)")
    parser.add_argument("--temperature",     type=float, default=1.4)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--no-train",        action="store_true")
    parser.add_argument("--no-corpus",       action="store_true")
    parser.add_argument("--no-tsp",          action="store_true",
                        help="Фиксированный порядок A→X→B→META (без TSP)")
    parser.add_argument("--no-recirculate",  action="store_true",
                        help="Без внутренних семантических мини-петель")
    parser.add_argument("--tsp-2opt",        action="store_true",
                        help="2-opt улучшение TSP маршрута после greedy")
    parser.add_argument("--lci-loss",        type=float, default=0.0,
                        help="λ для LCI-loss (Kirchhoff штраф в micro_train, default 0=выкл)")
    parser.add_argument("--temp-decay",      type=float, default=0.0,
                        dest="temp_decay", metavar="γ",
                        help="[meta_q6] Metropolis temperature decay (0=выкл, рек. 0.85). "
                             "T(c)=max(0.5, T0*γ^c). Устраняет осцилляции ±0.05.")
    parser.add_argument("--save",            type=str, default="hmoe_turbine.pt")
    args = parser.parse_args()

    if args.fast:
        args.cycles = 2
        args.steps_per_expert = 5

    block_size = MODEL_CFG["block_size"] - 1

    print(f"\n{'═' * 72}")
    print(f"  ТУРБИННАЯ РАЗВЯЗКА HMoE")
    print(f"{'═' * 72}")

    model      = _load_model(args.checkpoint)
    seed_texts = _load_seed_texts(args.no_corpus, block_size)

    import json

    t0  = time.perf_counter()
    log = turbine_figure8(
        model            = model,
        seed_texts       = seed_texts,
        block_size       = block_size,
        n_cycles         = args.cycles,
        steps_per_expert = args.steps_per_expert,
        temperature      = args.temperature,
        temp_decay       = args.temp_decay,
        train_lr         = args.lr,
        do_train         = not args.no_train,
        use_tsp          = not args.no_tsp,
        use_tsp_2opt     = args.tsp_2opt,
        do_recirculate   = not args.no_recirculate,
        lci_loss_lambda  = args.lci_loss,
    )
    elapsed = time.perf_counter() - t0

    # Сохранить
    ckpt_out = {
        "model_state":    model.state_dict(),
        "turbine_log":    log,
        "elapsed_sec":    round(elapsed, 2),
        "n_cycles":       args.cycles,
        "steps_per_expert": args.steps_per_expert,
        "temperature":    args.temperature,
        "use_tsp":        not args.no_tsp,
        "do_recirculate": not args.no_recirculate,
    }
    torch.save(ckpt_out, args.save)
    print(f"\n  Сохранено: {args.save}  (elapsed={elapsed:.1f}s)")

    log_path = args.save.replace(".pt", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  Лог:       {log_path}")


if __name__ == "__main__":
    main()
