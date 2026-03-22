#!/usr/bin/env python3
"""
nautilus_4agent.py — 4-агентный Наутилус: по одному агенту на каждую орбиту Aut(Q6).

Геометрия: orбиты под B₆ = S₆ ⋉ (Z₂)⁶ (Aut(Q6), 46080 элементов).
Q6 = {-1,+1}^6 имеет 7 орбит по весу Хэмминга (0..6).
Агенты специализируются на орбитах, шаги пропорциональны размеру орбиты.
Координация происходит в META (центральный узел) — как у multi-salesman в DYNAMIC.

  Агент-М  (META,     орбиты 0,6 — полюса)      — координатор, агрегатор
  Агент-А  (ABSTRACT, орбиты 4,5 — Yang-сторона) — абстрактный полюс
  Агент-Х  (DYNAMIC,  орбита  3  — экватор)      — динамический мост
  Агент-В  (CONCRETE, орбиты 1,2 — Yin-сторона)  — конкретный полюс

Шаги = размер орбиты / 2 (масштабирование к 100 суммарных шагов):
  Орбиты 0,6: 1+1=2 вершины  → 10 шагов
  Орбиты 4,5: 15+6=21 вершин → 21 шаг
  Орбита  3:  20 вершин      → 20 шагов
  Орбиты 1,2: 6+15=21 вершин → 21 шаг  (~ 72 итого, нормируем до 100)

hexsym: каждый агент видит только "свои" вершины Q6 → архитектурная дифференциация.

Один цикл = 100 шагов:
  1. Каждый агент делает шаги в своём кольце (параллельно, модель одна)
  2. Все встречаются в META: load_balance + обмен контекстом через shared RAG
  3. Kirchhoff-проверка: LCI всех агентов ≈ π

Отличие от nautilus_clover:
  clover    = 1 агент обходит все кольца последовательно
  4agent    = 4 агента в своих кольцах + синхронизация в META (как мульти-TSP)

Usage:
  python nautilus_4agent.py --checkpoint hmoe_self_trained_v5.pt
  python nautilus_4agent.py --fast
  python nautilus_4agent.py --cycles 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    DOMAIN_GROUPS, set_moe_stage,
)
from self_train_hmoe import (
    lci_from_routing, lci_from_embeddings, micro_train, quality_filter,
    RagBuffer, _generate, _ids_to_text, _encode, _hex_prompt,
    _get_emb, _get_q6_vertex, _get_moes, _freeze_all_except, MODEL_CFG, _LCI_EPSILON,
)
from nautilus_clover import _lci_loss_step

# meta_q6: интеграция с svend4/meta (bent seeds, temperature annealing)
try:
    from meta_q6 import bent_seed_texts, metropolis_temperature, cosine_temperature
    _META_Q6_AVAILABLE = True
except ImportError:
    _META_Q6_AVAILABLE = False

DEVICE = "cpu"
_ROOT  = os.path.dirname(os.path.abspath(__file__))

# ── Aut(Q6) орбиты по весу Хэмминга (hexsym) ─────────────────────────────────
# B₆ = S₆ ⋉ (Z₂)⁶ действует на Q6 = {-1,+1}^6 перестановками + флипом знаков.
# Орбиты = классы по числу +1 битов (Hamming weight в {0,1}^6 кодировке).
# Размеры: C(6,k) для k=0..6: 1, 6, 15, 20, 15, 6, 1  (итого 64 вершины)
_AUT_Q6_ORBITS: Dict[int, List[int]] = {
    k: [v for v in range(64) if bin(v).count('1') == k]
    for k in range(7)
}

# ── Кольца — orbits-based (hexsym) ────────────────────────────────────────────
# Шаги пропорциональны размеру орбиты, масштабированы до ~100 суммарных.
# META    = орбиты 0,6 (полюса Kun/Qian): 1+1=2  → 10 шагов (координатор)
# ABSTRACT= орбиты 4,5 (Yang-сторона):   15+6=21 → 26 шагов
# DYNAMIC = орбита  3  (экватор Q6):     20      → 25 шагов
# CONCRETE= орбиты 1,2 (Yin-сторона):    6+15=21 → 26 шагов
# Seed-гексаграммы берутся из своей орбиты (архитектурная дифференциация).
RINGS = [
    {
        "name": "META",
        "steps": 10,
        "groups": ["ABSTRACT", "DYNAMIC", "CONCRETE"],
        "ratio": 1,
        "orbits": [0, 6],  # полюса: Kun (000000) + Qian (111111)
        "orbit_verts": _AUT_Q6_ORBITS[0] + _AUT_Q6_ORBITS[6],
    },
    {
        "name": "ABSTRACT",
        "steps": 26,
        "groups": ["ABSTRACT"],
        "ratio": 2,
        "orbits": [4, 5],  # Yang-сторона: k=4 (15 вершин) + k=5 (6 вершин)
        "orbit_verts": _AUT_Q6_ORBITS[4] + _AUT_Q6_ORBITS[5],
    },
    {
        "name": "DYNAMIC",
        "steps": 25,
        "groups": ["DYNAMIC"],
        "ratio": 3,
        "orbits": [3],     # Экватор Q6: k=3 (20 вершин, λ=0)
        "orbit_verts": _AUT_Q6_ORBITS[3],
    },
    {
        "name": "CONCRETE",
        "steps": 26,
        "groups": ["CONCRETE"],
        "ratio": 4,
        "orbits": [1, 2],  # Yin-сторона: k=1 (6 вершин) + k=2 (15 вершин)
        "orbit_verts": _AUT_Q6_ORBITS[1] + _AUT_Q6_ORBITS[2],
    },
]
_RING_BY_NAME = {r["name"]: r for r in RINGS}
_TOTAL_STEPS  = sum(r["steps"] for r in RINGS)   # 87 → масштабируется через step_scale

# Агенты: home-кольцо, seed из своей орбиты (hexsym дифференциация)
_AGENTS = [
    {"name": "Агент-М (meta/poles)",    "home": "META",     "orbit_verts": _AUT_Q6_ORBITS[0] + _AUT_Q6_ORBITS[6]},
    {"name": "Агент-А (abstract/yang)", "home": "ABSTRACT", "orbit_verts": _AUT_Q6_ORBITS[4] + _AUT_Q6_ORBITS[5]},
    {"name": "Агент-Х (dynamic/eq)",    "home": "DYNAMIC",  "orbit_verts": _AUT_Q6_ORBITS[3]},
    {"name": "Агент-В (concrete/yin)",  "home": "CONCRETE", "orbit_verts": _AUT_Q6_ORBITS[1] + _AUT_Q6_ORBITS[2]},
]

_META_FREEZE = ["ABSTRACT", "DYNAMIC", "CONCRETE"]


# ── Утилиты ───────────────────────────────────────────────────────────────────

def _freeze_for(model: Variant3GPT, ring_name: str) -> None:
    ring = _RING_BY_NAME[ring_name]
    for moe in _get_moes(model):
        _freeze_all_except(moe, ring["groups"])
    if ring_name in ("META", "DYNAMIC"):
        set_moe_stage(getattr(model.blocks[0], "hmoe", None), 4)


# ── Фаза одного кольца ────────────────────────────────────────────────────────

def _run_ring(
    model: Variant3GPT,
    ring_name: str,
    ids: torch.Tensor,
    rag: RagBuffer,
    steps: int,
    train_lr: float,
    temperature: float,
    do_train: bool,
    block_size: int,
    lci_loss_lambda: float = 0.0,
) -> Tuple[torch.Tensor, float, float, int]:
    """
    Тренировка агента в своём кольце за steps шагов.
    lci_loss_lambda > 0 → добавляем Kirchhoff-штраф к каждому шагу.
    Returns: (new_ids, lci_r, lci_emb, n_gen)
    """
    _freeze_for(model, ring_name)
    start_ids = ids.clone()
    n_gen = 0

    for _ in range(steps):
        gen_ids  = _generate(model, ids, block_size, temperature, n_tokens=8)
        gen_text = _ids_to_text(gen_ids)
        if do_train and quality_filter(gen_text):
            micro_train(model, gen_ids, lr=train_lr, n_steps=1)
            rag.add(gen_text, _get_emb(model, gen_ids),
                    q6_vert=_get_q6_vertex(model, gen_ids))
            n_gen += 1
            if lci_loss_lambda > 0:
                _lci_loss_step(model, gen_ids, lr=train_lr * lci_loss_lambda)
        ids = gen_ids

    lci_r   = lci_from_routing(model, ids)[0]
    lci_emb = lci_from_embeddings(model, start_ids, ids)
    return ids, lci_r, lci_emb, n_gen


# ── Координация в META (load balance + обмен) ─────────────────────────────────

def _meta_coordination(
    model: Variant3GPT,
    agent_ids: List[torch.Tensor],
    agent_lcis: List[float],
    agent_rags: List[RagBuffer],
    rag_shared: RagBuffer,
    block_size: int,
    cross_pollinate: bool = False,
) -> Tuple[float, float]:
    """
    Координация в META: все 4 агента встречаются.

    Если cross_pollinate=True:
      - Ранжируем агентов по |LCI - π| (лучший = ближайший к π)
      - Лучший агент делится контекстом (ids + RAG) со слабыми
      - Слабые агенты дополнительно обогащаются из RAG лучшего

    Returns: (avg_lci, load_balance)
    """
    # Все агенты вносят контекст в shared RAG
    for ids in agent_ids:
        txt = _ids_to_text(ids)
        if quality_filter(txt):
            rag_shared.add(txt, _get_emb(model, ids),
                           q6_vert=_get_q6_vertex(model, ids))

    if not agent_lcis:
        return math.pi, 1.0

    avg_lci = sum(agent_lcis) / len(agent_lcis)
    max_lci = max(agent_lcis)
    min_lci = min(agent_lcis)
    balance = 1.0 - (max_lci - min_lci) / (math.pi + 1e-8)
    balance = max(0.0, min(1.0, balance))

    if cross_pollinate and len(agent_lcis) > 1:
        # Лучший агент = ближайший LCI к π
        best_i = min(range(len(agent_lcis)), key=lambda i: abs(agent_lcis[i] - math.pi))
        best_emb = _get_emb(model, agent_ids[best_i])

        for i, (ids, lci) in enumerate(zip(agent_ids, agent_lcis)):
            if i == best_i:
                continue
            # Слабый агент тянется к ближайшему тексту из RAG лучшего
            gap = abs(lci - math.pi) - abs(agent_lcis[best_i] - math.pi)
            if gap > 0.05 and len(agent_rags[best_i]) > 2:
                near = agent_rags[best_i].retrieve(_get_emb(model, ids), top_k=1,
                                                   query_q6=_get_q6_vertex(model, ids))
                if near:
                    agent_ids[i] = _encode(near[0], block_size)
                    # Добавляем лучший контекст и в RAG слабого
                    agent_rags[i].add(near[0], best_emb,
                                  q6_vert=_get_q6_vertex(model, agent_ids[best_i]))

    return avg_lci, balance


# ── Главный цикл ──────────────────────────────────────────────────────────────

def nautilus_4agent_cycle(
    model: Variant3GPT,
    agent_ids: List[torch.Tensor],
    agent_rags: List[RagBuffer],
    rag_shared: RagBuffer,
    train_lr: float,
    temperature: float,
    do_train: bool,
    step_scale: float,
    block_size: int,
    lci_loss_lambda: float = 0.0,
    cross_pollinate: bool = False,
) -> Tuple[List[torch.Tensor], Dict]:
    """
    Один цикл 4-агентного Наутилуса (100 × step_scale шагов).

    Каждый агент тренируется в своём кольце пропорциональное число шагов,
    затем все синхронизируются в META.
    """
    t0      = time.perf_counter()
    n_gen   = 0
    lci_log = {}

    for i, ring in enumerate(RINGS):
        name      = ring["name"]
        steps     = max(1, int(ring["steps"] * step_scale))
        agent     = _AGENTS[i]

        # Обогатить ids агента из shared RAG перед его фазой
        if len(rag_shared) > 3:
            near = rag_shared.retrieve(_get_emb(model, agent_ids[i]), top_k=1,
                                       query_q6=_get_q6_vertex(model, agent_ids[i]))
            if near:
                agent_ids[i] = _encode(near[0], block_size)

        new_ids, lci_r, lci_emb, ng = _run_ring(
            model, name, agent_ids[i], agent_rags[i],
            steps, train_lr, temperature, do_train, block_size,
            lci_loss_lambda=lci_loss_lambda,
        )
        agent_ids[i] = new_ids
        n_gen       += ng

        mark   = "✓" if abs(lci_r - math.pi) < _LCI_EPSILON else "✗"
        print(f"    [{name:10}] ×{steps:2d}шаг  "
              f"lci_r={lci_r:.3f}  emb={lci_emb:.3f}  {mark}  {agent['name']}")

        lci_log[name] = {"lci_r": round(lci_r, 4), "lci_emb": round(lci_emb, 4),
                         "steps": steps, "gen": ng}

    # ── Координация в META ────────────────────────────────────────────────────
    agent_lcis = [lci_log[r["name"]]["lci_r"] for r in RINGS]
    avg_lci, balance = _meta_coordination(
        model, agent_ids, agent_lcis, agent_rags, rag_shared,
        block_size, cross_pollinate=cross_pollinate,
    )
    kirchhoff_val = avg_lci
    kirchhoff_ok  = abs(kirchhoff_val - math.pi) < _LCI_EPSILON
    cp_s = " [cross✓]" if cross_pollinate else ""

    print(f"    Координация META: avg_LCI={avg_lci:.3f}  balance={balance:.3f}"
          f"  {'✓ KIRCHHOFF' if kirchhoff_ok else '✗'}{cp_s}")

    # ── Итоговый LCI (Forward = META агент) ──────────────────────────────────
    lci_r_final, _ = lci_from_routing(model, agent_ids[0])
    elapsed = time.perf_counter() - t0

    n_resonant = sum(1 for r in RINGS if abs(lci_log[r["name"]]["lci_r"] - math.pi) < _LCI_EPSILON)

    return agent_ids, {
        "rings":         lci_log,
        "avg_lci_all":   round(avg_lci, 4),
        "lci_r_final":   round(lci_r_final, 4),
        "kirchhoff":     kirchhoff_ok,
        "kirchhoff_val": round(kirchhoff_val, 4),
        "load_balance":  round(balance, 4),
        "n_resonant":    n_resonant,
        "n_generated":   n_gen,
        "elapsed_s":     round(elapsed, 2),
        "resonant":      n_resonant >= 3,
    }


# ── Основная функция ──────────────────────────────────────────────────────────

def _adaptive_scale(cycle: int, n_cycles: int) -> float:
    """Warmup schedule: 0.3 → ramp → 1.0 по параболе."""
    t = (cycle - 1) / max(1, n_cycles - 1)   # 0.0 .. 1.0
    return 0.3 + 0.7 * (t ** 0.7)            # начало=0.3, конец=1.0


def nautilus_4agent(
    model: Variant3GPT,
    seed_texts: List[str],
    n_cycles: int = 4,
    step_scale: float = 1.0,
    adaptive: bool = False,
    temperature: float = 1.4,
    temp_decay: float = 0.0,
    train_lr: float = 1e-5,
    do_train: bool = True,
    lci_loss_lambda: float = 0.0,
    cross_pollinate: bool = False,
    block_size: int = MODEL_CFG["block_size"] - 1,
) -> List[Dict]:
    """
    temp_decay > 0: Metropolis temperature annealing.
      T(c) = max(0.5, T0 * temp_decay^c)
      Рекомендуемые значения: 0.85 (агрессивно), 0.92 (мягко).
      Устраняет осцилляции ±0.05 в поздних циклах.
    """

    total_per_cycle = int(_TOTAL_STEPS * step_scale)
    mode_s = "адаптивный 0.3→1.0" if adaptive else f"фикс {step_scale:.2f}"
    t_mode = f"Metropolis decay={temp_decay:.2f}" if temp_decay > 0 else f"фикс {temperature:.2f}"

    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ 4-АГЕНТНЫЙ НАУТИЛУС + HMoE")
    print(f"{'═' * 72}")
    print(f"  Циклов              : {n_cycles}")
    print(f"  Шагов/цикл          : {total_per_cycle}  (={_TOTAL_STEPS}×{step_scale:.2f})  [{mode_s}]")
    print(f"  Температура         : {temperature:.2f}  [{t_mode}]")
    print(f"\n  Агенты (кольца Пифагорейской тетрактиды 1:2:3:4):")
    for i, (ag, ring) in enumerate(zip(_AGENTS, RINGS)):
        steps = max(1, int(ring["steps"] * step_scale))
        print(f"    {ag['name']:28}  кольцо={ring['name']:10}  {steps:3d}шаг ({ring['steps']*step_scale/total_per_cycle*100:.0f}%)")
    print()

    # ── Инициализация RAG ──────────────────────────────────────────────────────
    rag_shared = RagBuffer(max_size=500)
    agent_rags = [RagBuffer(max_size=200) for _ in _AGENTS]

    model.eval()
    for text in seed_texts[:50]:
        ids = _encode(text, block_size)
        emb = _get_emb(model, ids)
        rag_shared.add(text, emb)
        for rag in agent_rags:
            rag.add(text, emb)

    print(f"  Shared RAG          : {len(rag_shared)} текстов")

    # ── Инициализация агентов (hexsym: seed из своей орбиты Aut(Q6)) ───────────
    agent_ids: List[torch.Tensor] = []
    for ag in _AGENTS:
        # hexsym: seed-гексаграмма из орбиты агента — архитектурная дифференциация
        orbit_verts = ag["orbit_verts"]
        hex_id = random.choice(orbit_verts)
        agent_ids.append(_hex_prompt(hex_id, block_size))
        lci0, _ = lci_from_routing(model, agent_ids[-1])
        orbit_label = f"orbits={_RING_BY_NAME[ag['home']]['orbits']}"
        print(f"    {ag['name']:32}  start LCI={lci0:.3f}  {orbit_label}")

    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        lci_r0, _ = lci_from_routing(model, agent_ids[0])
        res_mark = "✓ РЕЗОНАНС" if abs(lci_r0 - math.pi) < _LCI_EPSILON else f"δ={lci_r0 - math.pi:+.3f}"
        cur_scale = _adaptive_scale(cycle, n_cycles) if adaptive else step_scale

        # Metropolis temperature annealing (meta_q6 или встроенный)
        if temp_decay > 0:
            if _META_Q6_AVAILABLE:
                cur_temp = metropolis_temperature(cycle - 1, n_cycles, temperature, T_min=0.5, decay=temp_decay)
            else:
                cur_temp = max(0.5, temperature * (temp_decay ** (cycle - 1)))
        else:
            cur_temp = temperature

        print(f"\n  Цикл {cycle}/{n_cycles}  LCI_Агент-М={lci_r0:.3f}  {res_mark}"
              f"  scale={cur_scale:.2f}  T={cur_temp:.3f}")

        agent_ids, result = nautilus_4agent_cycle(
            model, agent_ids, agent_rags, rag_shared,
            train_lr=train_lr, temperature=cur_temp,
            do_train=do_train, step_scale=cur_scale, block_size=block_size,
            lci_loss_lambda=lci_loss_lambda,
            cross_pollinate=cross_pollinate,
        )
        result["step_scale"] = round(cur_scale, 3)
        result["temperature"] = round(cur_temp, 3)

        k_s = "✓ KIRCHHOFF" if result["kirchhoff"] else f"✗ {result['n_resonant']}/{len(RINGS)}"
        print(f"    → avg_LCI={result['avg_lci_all']:.3f}  balance={result['load_balance']:.3f}"
              f"  резонанс={result['n_resonant']}/{len(RINGS)}  {k_s}"
              f"  gen={result['n_generated']}  t={result['elapsed_s']:.1f}s")

        log.append({"cycle": cycle, "lci_r0": round(lci_r0, 4), **result})

    # ── Итог ──────────────────────────────────────────────────────────────────
    n_kirchhoff  = sum(1 for r in log if r["kirchhoff"])
    n_resonant_c = sum(1 for r in log if r["resonant"])
    avg_lci      = sum(r["avg_lci_all"] for r in log) / len(log) if log else 0.0
    avg_bal      = sum(r["load_balance"] for r in log) / len(log) if log else 0.0

    print(f"\n{'─' * 72}")
    print(f"  ИТОГ 4-АГЕНТНОГО НАУТИЛУСА:")
    print(f"    Резонансных циклов (≥3/4):        {n_resonant_c}/{n_cycles}")
    print(f"    Kirchhoff-сбалансированных:       {n_kirchhoff}/{n_cycles}")
    print(f"    avg_LCI                : {avg_lci:.3f}  (цель π={math.pi:.3f})")
    print(f"    avg_load_balance       : {avg_bal:.3f}  (1.0 = идеал)")
    print(f"    Shared RAG             : {len(rag_shared)} текстов")

    return log


# ── Загрузка ──────────────────────────────────────────────────────────────────

def _load_model(path: str) -> Variant3GPT:
    cfg = Variant3Config(**MODEL_CFG)
    m   = Variant3GPT(cfg)
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck.get("model_state", ck), strict=False)
        print(f"  Загружен: {path}")
    else:
        print(f"  [!] Не найден: {path} — случайные веса")
    m.to(DEVICE)
    print(f"  Модель: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M параметров")
    return m


def _load_seeds(block_size: int, use_bent: bool = False) -> List[str]:
    """
    Загрузить seed-тексты для инициализации RAG.

    use_bent=True: использовать bent-функции из meta_q6 вместо ручных строк.
    Bent-функции — математически оптимальные архетипы с гарантированным
    разнообразием (nl=28, равномерный WHT-спектр), устраняют diversity collapse.
    """
    if use_bent and _META_Q6_AVAILABLE:
        texts = bent_seed_texts(n=20, block_size=block_size)
        print(f"  [meta_q6] Bent seeds: {len(texts)} архетипов (nl=28, WHT равномерный)")
    else:
        texts = [
            "def forward(self, x): return self.linear(x)",
            "loss.backward(); optimizer.zero_grad(); scheduler.step()",
            "x = x + self.crossing(out_a, out_b)",
            "The hexagram represents the intersection of abstract and concrete.",
            "consciousness emerges from recursive self-reference in Q6 space",
            "The Pythagorean tetractys: 1+2+3+4=10, the decade of nature",
            "spiral growth: each ring proportionally larger than the previous",
            "nautilus shell: logarithmic spiral, self-similar at every scale",
            "four agents, four rings, one coordination point at META",
            "load balance: all agents converge to the same LCI = π",
        ] * 5
    for h in range(64):
        texts.append(_ids_to_text(_hex_prompt(h, block_size)))
    return texts


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="4-агентный Наутилус: специализированные агенты по кольцам"
    )
    parser.add_argument("--checkpoint",  type=str,   default="hmoe_self_trained_v5.pt")
    parser.add_argument("--fast",        action="store_true",
                        help="быстрый тест: 2 цикла, 0.3× шагов")
    parser.add_argument("--cycles",      type=int,   default=4)
    parser.add_argument("--step-scale",  type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--adaptive",    action="store_true",
                        help="адаптивный step_scale: warmup 0.3→1.0")
    parser.add_argument("--lci-loss",    type=float, default=0.0,
                        dest="lci_loss", metavar="λ",
                        help="Kirchhoff-штраф λ (0=выкл, рекомендуется 0.05)")
    parser.add_argument("--cross-pollinate", action="store_true",
                        dest="cross_pollinate",
                        help="META: лучший агент обучает слабых")
    parser.add_argument("--no-train",    action="store_true")
    # meta_q6 интеграция
    parser.add_argument("--bent-seeds",  action="store_true",
                        dest="bent_seeds",
                        help="[meta_q6] Bent-функции Q6 как seed-архетипы RAG (nl=28)")
    parser.add_argument("--temp-decay",  type=float, default=0.0,
                        dest="temp_decay", metavar="γ",
                        help="[meta_q6] Metropolis temperature decay (0=выкл, рек. 0.85). "
                             "T(c)=max(0.5, T0*γ^c). Устраняет осцилляции ±0.05.")
    parser.add_argument("--save",        type=str,   default="hmoe_nautilus_4agent_v1.pt")
    args = parser.parse_args()

    block_size = MODEL_CFG["block_size"] - 1

    if args.fast:
        args.cycles    = 2
        args.step_scale = 0.3

    print(f"\n{'═' * 72}")
    print(f"  4-АГЕНТНЫЙ НАУТИЛУС HMoE")
    print(f"{'═' * 72}")
    model      = _load_model(args.checkpoint)
    seed_texts = _load_seeds(block_size, use_bent=args.bent_seeds)
    print(f"  Seed текстов: {len(seed_texts)}")

    log = nautilus_4agent(
        model,
        seed_texts=seed_texts,
        n_cycles=args.cycles,
        step_scale=args.step_scale,
        adaptive=args.adaptive,
        temperature=args.temperature,
        temp_decay=args.temp_decay,
        train_lr=args.lr,
        do_train=not args.no_train,
        lci_loss_lambda=args.lci_loss,
        cross_pollinate=args.cross_pollinate,
        block_size=block_size,
    )

    torch.save(
        {"model_state": model.state_dict(), "log": log, "config": MODEL_CFG},
        args.save,
    )
    log_path = args.save.replace(".pt", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n  Сохранено: {args.save}")
    print(f"  Лог:       {log_path}")


if __name__ == "__main__":
    main()
