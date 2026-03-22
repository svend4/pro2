#!/usr/bin/env python3
"""
nautilus_4agent.py — 4-агентный Наутилус: по одному агенту на каждое кольцо.

Геометрия: 4 кольца Пифагорейской тетрактиды (10:20:30:40 шагов).
Каждый агент живёт в своём кольце и специализируется на нём.
Координация происходит в META (центральный узел) — как у multi-salesman в DYNAMIC.

  Агент-М  (META,     10 шагов) — координатор, агрегатор
  Агент-А  (ABSTRACT, 20 шагов) — абстрактный полюс
  Агент-Х  (DYNAMIC,  30 шагов) — динамический мост
  Агент-В  (CONCRETE, 40 шагов) — конкретный полюс

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
    _get_emb, _get_moes, _freeze_all_except, MODEL_CFG, _LCI_EPSILON,
)
from nautilus_clover import _lci_loss_step

DEVICE = "cpu"
_ROOT  = os.path.dirname(os.path.abspath(__file__))

# ── Кольца (Пифагорейская тетрактида 1:2:3:4) ────────────────────────────────
RINGS = [
    {"name": "META",     "steps": 10, "groups": ["ABSTRACT", "DYNAMIC", "CONCRETE"], "ratio": 1},
    {"name": "ABSTRACT", "steps": 20, "groups": ["ABSTRACT"],                        "ratio": 2},
    {"name": "DYNAMIC",  "steps": 30, "groups": ["DYNAMIC"],                         "ratio": 3},
    {"name": "CONCRETE", "steps": 40, "groups": ["CONCRETE"],                        "ratio": 4},
]
_RING_BY_NAME = {r["name"]: r for r in RINGS}
_TOTAL_STEPS  = sum(r["steps"] for r in RINGS)   # 100

# Агенты: home-кольцо, seed-диапазон гексаграмм (аналог multi-salesman)
_AGENTS = [
    {"name": "Агент-М (meta)",      "home": "META",     "seed_range": (0,  15)},
    {"name": "Агент-А (abstract)",  "home": "ABSTRACT", "seed_range": (16, 31)},
    {"name": "Агент-Х (dynamic)",   "home": "DYNAMIC",  "seed_range": (32, 47)},
    {"name": "Агент-В (concrete)",  "home": "CONCRETE", "seed_range": (48, 63)},
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
            rag.add(gen_text, _get_emb(model, gen_ids))
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
    rag_shared: RagBuffer,
    block_size: int,
) -> Tuple[float, float]:
    """
    Koordinacja: все 4 агента встречаются в META.
    - Каждый добавляет свой лучший контекст в shared RAG
    - Считаем load_balance (аналог multi-salesman)
    - Возвращаем avg_lci и balance

    Returns: (avg_lci, load_balance)
    """
    # Каждый агент вносит свой текст в shared RAG
    for ids in agent_ids:
        txt = _ids_to_text(ids)
        if quality_filter(txt):
            rag_shared.add(txt, _get_emb(model, ids))

    # Load balance: насколько равномерно распределены LCI агентов
    if not agent_lcis:
        return math.pi, 1.0
    avg_lci  = sum(agent_lcis) / len(agent_lcis)
    max_lci  = max(agent_lcis)
    min_lci  = min(agent_lcis)
    balance  = 1.0 - (max_lci - min_lci) / (math.pi + 1e-8)
    balance  = max(0.0, min(1.0, balance))
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
            near = rag_shared.retrieve(_get_emb(model, agent_ids[i]), top_k=1)
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
        model, agent_ids, agent_lcis, rag_shared, block_size
    )
    kirchhoff_val = avg_lci
    kirchhoff_ok  = abs(kirchhoff_val - math.pi) < _LCI_EPSILON

    print(f"    Координация META: avg_LCI={avg_lci:.3f}  balance={balance:.3f}"
          f"  {'✓ KIRCHHOFF' if kirchhoff_ok else '✗'}")

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
    train_lr: float = 1e-5,
    do_train: bool = True,
    lci_loss_lambda: float = 0.0,
    block_size: int = MODEL_CFG["block_size"] - 1,
) -> List[Dict]:

    total_per_cycle = int(_TOTAL_STEPS * step_scale)
    mode_s = "адаптивный 0.3→1.0" if adaptive else f"фикс {step_scale:.2f}"

    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ 4-АГЕНТНЫЙ НАУТИЛУС + HMoE")
    print(f"{'═' * 72}")
    print(f"  Циклов              : {n_cycles}")
    print(f"  Шагов/цикл          : {total_per_cycle}  (={_TOTAL_STEPS}×{step_scale:.2f})  [{mode_s}]")
    print(f"  Температура         : {temperature:.2f}")
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

    # ── Инициализация агентов ──────────────────────────────────────────────────
    agent_ids: List[torch.Tensor] = []
    for ag in _AGENTS:
        lo, hi = ag["seed_range"]
        hex_id  = random.randint(lo, hi)
        agent_ids.append(_hex_prompt(hex_id, block_size))
        lci0, _ = lci_from_routing(model, agent_ids[-1])
        print(f"    {ag['name']:28}  start LCI={lci0:.3f}")

    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        lci_r0, _ = lci_from_routing(model, agent_ids[0])
        res_mark = "✓ РЕЗОНАНС" if abs(lci_r0 - math.pi) < _LCI_EPSILON else f"δ={lci_r0 - math.pi:+.3f}"
        cur_scale = _adaptive_scale(cycle, n_cycles) if adaptive else step_scale
        print(f"\n  Цикл {cycle}/{n_cycles}  LCI_Агент-М={lci_r0:.3f}  {res_mark}  scale={cur_scale:.2f}")

        agent_ids, result = nautilus_4agent_cycle(
            model, agent_ids, agent_rags, rag_shared,
            train_lr=train_lr, temperature=temperature,
            do_train=do_train, step_scale=cur_scale, block_size=block_size,
            lci_loss_lambda=lci_loss_lambda,
        )
        result["step_scale"] = round(cur_scale, 3)

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


def _load_seeds(block_size: int) -> List[str]:
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
    parser.add_argument("--no-train",    action="store_true")
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
    seed_texts = _load_seeds(block_size)
    print(f"  Seed текстов: {len(seed_texts)}")

    log = nautilus_4agent(
        model,
        seed_texts=seed_texts,
        n_cycles=args.cycles,
        step_scale=args.step_scale,
        adaptive=args.adaptive,
        temperature=args.temperature,
        train_lr=args.lr,
        do_train=not args.no_train,
        lci_loss_lambda=args.lci_loss,
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
