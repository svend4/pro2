#!/usr/bin/env python3
"""
multi_salesman.py — Мульти-коммивояжёр для HMoE само-обучения.

Концепция (из data7/multi_agent_coordinator.py):
  Классическая TSP: 1 агент, N городов, minimize Σ distance
  Мульти-TSP:      K агентов, каждый ведёт свой маршрут, координация в точке X

  Здесь:
    Агент (коммивояжёр) = независимый поток токенов
    Город               = эксперт HMoE (ABSTRACT / DYNAMIC / CONCRETE)
    Расстояние          = |LCI_эксперта - π| (семантическое, не географическое)
    Точка встречи       = DYNAMIC (обмен контекстом через RAG)
    Координация         = context-switching cost из data7

  Каждый агент специализируется на своём seed-тексте, но обменивается
  знаниями через общий RAG в точке DYNAMIC.

  Динамическая TSP:
    Города (концепции) появляются из генерации на каждом шаге.
    Агент адаптирует маршрут по мере появления новых токенов.

  Масштабирование (данные data7):
    - 2 агента: dissertation ↔ encyclopedia (bidirectional)
    - 3 агента: добавляется meta-агент (агрегатор)
    - N агентов: каждый на своём тематическом кластере

Usage:
  python multi_salesman.py --checkpoint hmoe_self_trained_v5.pt
  python multi_salesman.py --fast
  python multi_salesman.py --agents 3 --cycles 6 --steps 15
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    HMoEConfig, CLUSTER_TO_DOMAIN, DOMAIN_GROUPS, DOMAIN_TO_GROUP, set_moe_stage,
)
from self_train_hmoe import (
    lci_from_routing, lci_from_embeddings, micro_train, quality_filter,
    RagBuffer, _generate, _ids_to_text, _encode, _hex_prompt,
    _get_emb, _get_moes, _freeze_all_except, MODEL_CFG, _LCI_EPSILON,
)

DEVICE = "cpu"
_ROOT  = os.path.dirname(os.path.abspath(__file__))

# Специализации агентов (из data7: разные роли торговых представителей)
_AGENT_SPECS = [
    {"name": "Агент-А (абстрактный)",  "home": "ABSTRACT", "seed_range": (0,  21)},
    {"name": "Агент-X (динамический)", "home": "DYNAMIC",  "seed_range": (22, 42)},
    {"name": "Агент-B (конкретный)",   "home": "CONCRETE", "seed_range": (43, 63)},
]

_EXPERT_ORDER_BASE = ["ABSTRACT", "DYNAMIC", "CONCRETE"]
_META_FREEZE = ["ABSTRACT", "DYNAMIC", "CONCRETE"]


def _freeze_for(model, expert: str):
    groups = _META_FREEZE if expert == "META" else [expert]
    for moe in _get_moes(model):
        _freeze_all_except(moe, groups)
    if expert in ("DYNAMIC", "META"):
        set_moe_stage(
            model.blocks[0].hmoe if hasattr(model.blocks[0], 'hmoe') else None, 4
        )


def _context_switch_cost(lci_from: float, lci_to: float) -> float:
    """
    Стоимость переключения контекста между экспертами.
    Аналог context_switching_cost из data7/multi_agent_coordinator.py.
    |LCI_from - LCI_to| = семантическое расстояние.
    """
    return abs(lci_from - lci_to)


def _greedy_tsp_route(
    start_lci: float,
    expert_lci: Dict[str, float],
    unvisited: List[str],
) -> List[str]:
    """
    Nearest-neighbor TSP для одного агента.
    Стоимость = context_switching_cost между текущим LCI и целевым экспертом.
    """
    route = []
    current_lci = start_lci
    remaining = list(unvisited)

    while remaining:
        costs = {e: _context_switch_cost(current_lci, expert_lci.get(e, math.pi))
                 for e in remaining}
        next_e = min(costs, key=costs.get)
        route.append(next_e)
        current_lci = expert_lci.get(next_e, math.pi)
        remaining.remove(next_e)

    return route


class Agent:
    """Один коммивояжёр — независимый поток токенов."""

    def __init__(self, spec: Dict, block_size: int, rag: RagBuffer):
        self.name       = spec["name"]
        self.home       = spec["home"]
        self.block_size = block_size
        self.rag        = rag
        # Стартовый промпт из специализированного диапазона
        h = random.randint(*spec["seed_range"])
        self.ids = _hex_prompt(h, block_size)
        self.lci_history: List[float] = []
        self.expert_lci: Dict[str, float] = {e: math.pi for e in _EXPERT_ORDER_BASE}
        self.n_generated = 0

    def compute_route(self) -> List[str]:
        """Рассчитать оптимальный маршрут по экспертам (TSP)."""
        lci_r, _ = lci_from_routing_safe(self.ids)
        return _greedy_tsp_route(lci_r, self.expert_lci, list(_EXPERT_ORDER_BASE))

    def run_expert(
        self,
        model: Variant3GPT,
        expert: str,
        n_steps: int,
        temperature: float,
        train_lr: float,
        do_train: bool,
    ) -> float:
        """Прогнать агента через одного эксперта. Вернуть routing_LCI."""
        _freeze_for(model, expert)

        for _ in range(n_steps):
            gen_ids = _generate(model, self.ids, self.block_size, temperature, n_tokens=8)
            gen_text = _ids_to_text(gen_ids)

            if do_train and quality_filter(gen_text):
                lr_scale = 0.5 if expert == "CONCRETE" else 1.0
                micro_train(model, gen_ids, lr=train_lr * lr_scale, n_steps=2)
                emb = _get_emb(model, gen_ids)
                self.rag.add(gen_text, emb)
                self.n_generated += 1

            self.ids = gen_ids

        lci_r, _ = lci_from_routing(model, self.ids)
        self.expert_lci[expert] = lci_r
        self.lci_history.append(lci_r)
        return lci_r

    def enrich_from_rag(self):
        """Обогатить контекст из RAG (аналог разведки рынка торговым агентом)."""
        if len(self.rag) > 5:
            emb = _get_emb_safe(self.ids)
            retrieved = self.rag.retrieve(emb, top_k=1)
            if retrieved:
                self.ids = _encode(retrieved[0], self.block_size)


# Вспомогательные обёртки без model (для Agent)
def lci_from_routing_safe(ids):
    """Заглушка: вернуть π (без модели), реальный вызов в цикле."""
    return math.pi, {}


def _get_emb_safe(ids):
    """Заглушка: возвращает нулевой эмбеддинг."""
    return torch.zeros(128)


def _coordinate_at_dynamic(
    model: Variant3GPT,
    agents: List[Agent],
    block_size: int,
    temperature: float,
    train_lr: float,
    do_train: bool,
    rag: RagBuffer,
) -> Tuple[float, float]:
    """
    Координация агентов в точке DYNAMIC.

    Принцип из data7/multi_agent_coordinator.py:
      - Каждый агент оставляет своё знание в общем RAG
      - Вычисляется load_balance_score = 1 - coefficient_of_variation (LCI)
      - Агент с наибольшим отклонением от π получает контекст от ближайшего к π

    Returns:
        load_balance  — 0..1 (1 = все агенты равно близки к π)
        kirchhoff_val — Σ(lci_k) / N  (среднее LCI в точке встречи)
    """
    _freeze_for(model, "DYNAMIC")

    lci_values: List[float] = []
    for agent in agents:
        # Агент делится контекстом в точке DYNAMIC
        agent.enrich_from_rag()
        lci_r = agent.run_expert(model, "DYNAMIC", n_steps=3,
                                  temperature=temperature, train_lr=train_lr,
                                  do_train=do_train)
        lci_values.append(lci_r)

    # Load balance score (из data7: 1 - коэффициент вариации)
    if len(lci_values) > 1:
        mean_lci = sum(lci_values) / len(lci_values)
        variance = sum((l - mean_lci) ** 2 for l in lci_values) / len(lci_values)
        std_lci  = math.sqrt(variance)
        cv = std_lci / (mean_lci + 1e-8)
        load_balance = max(0.0, 1.0 - cv)
    else:
        load_balance = 1.0

    kirchhoff_val = sum(lci_values) / len(lci_values) if lci_values else math.pi

    # Перераспределить контекст: агент далеко от π получает RAG от агента близко к π
    sorted_agents = sorted(agents, key=lambda a: abs(a.lci_history[-1] - math.pi) if a.lci_history else 0)
    best_agent = sorted_agents[0]   # ближайший к π
    worst_agent = sorted_agents[-1] # самый далёкий

    if best_agent is not worst_agent and len(rag) > 0:
        best_emb = _get_emb(model, best_agent.ids)
        retrieved = rag.retrieve(best_emb, top_k=1)
        if retrieved:
            worst_agent.ids = _encode(retrieved[0], block_size)

    return load_balance, kirchhoff_val


def multi_salesman(
    model: Variant3GPT,
    seed_texts: List[str],
    block_size: int = MODEL_CFG["block_size"] - 1,
    n_cycles: int = 4,
    n_agents: int = 3,
    steps_per_expert: int = 10,
    temperature: float = 1.4,
    train_lr: float = 1e-5,
    do_train: bool = True,
) -> List[Dict]:
    """
    Мульти-коммивояжёр само-обучение HMoE.

    Каждый цикл:
      1. Каждый агент вычисляет свой TSP-маршрут (по своему LCI-состоянию)
      2. Агенты параллельно (последовательно в CPU) проходят свои маршруты
      3. Координация в точке DYNAMIC: обмен контекстом, load balancing
      4. Kirchhoff-проверка по всем агентам
    """
    n_agents = min(n_agents, len(_AGENT_SPECS))
    specs = _AGENT_SPECS[:n_agents]

    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ МУЛЬТИ-КОММИВОЯЖЁР + HMoE")
    print(f"{'═' * 72}")
    print(f"  Циклов         : {n_cycles}")
    print(f"  Агентов        : {n_agents}")
    print(f"  Шагов/эксперт  : {steps_per_expert}")
    print(f"  Температура    : {temperature:.2f}")
    for s in specs:
        print(f"    {s['name']:30s}  home={s['home']}  seed={s['seed_range']}")
    print()

    # Общий RAG для всех агентов (обмен знаниями)
    rag = RagBuffer(max_size=400)
    model.eval()
    for text in seed_texts[:50]:
        ids = _encode(text, block_size)
        rag.add(text, _get_emb(model, ids))
    print(f"  Общий RAG: {len(rag)} текстов")

    # Создать агентов
    agents = [Agent(spec, block_size, rag) for spec in specs]

    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        cycle_t = time.perf_counter()

        # Измерить начальное LCI для каждого агента
        lci_r_starts: List[float] = []
        for agent in agents:
            lci_r, _ = lci_from_routing(model, agent.ids)
            lci_r_starts.append(lci_r)

        print(f"\n  Цикл {cycle}/{n_cycles}  "
              f"LCI_starts=[{', '.join(f'{l:.3f}' for l in lci_r_starts)}]")

        # ── Каждый агент проходит свой TSP-маршрут ───────────────────────
        agent_results: List[Dict] = []
        for agent in agents:
            # Обновить expert_lci реальными значениями
            for expert in _EXPERT_ORDER_BASE:
                agent.expert_lci[expert] = agent.lci_history[-1] if agent.lci_history else math.pi

            route = agent.compute_route()
            # Убрать DYNAMIC из маршрута (он будет в точке координации)
            route_no_dyn = [e for e in route if e != "DYNAMIC"]

            agent_lci: List[float] = []
            for expert in route_no_dyn:
                agent.enrich_from_rag()
                lci_r = agent.run_expert(
                    model, expert, steps_per_expert, temperature, train_lr, do_train
                )
                agent_lci.append(lci_r)

            avg_lci = sum(agent_lci) / len(agent_lci) if agent_lci else math.pi
            print(f"    {agent.name}: маршрут={route_no_dyn}  "
                  f"LCI=[{', '.join(f'{l:.3f}' for l in agent_lci)}]  "
                  f"avg={avg_lci:.3f}  gen={agent.n_generated}")
            agent_results.append({
                "agent": agent.name,
                "route": route_no_dyn,
                "lci":   [round(l, 4) for l in agent_lci],
                "avg_lci": round(avg_lci, 4),
            })

        # ── Координация в точке DYNAMIC ───────────────────────────────────
        load_balance, kirchhoff_val = _coordinate_at_dynamic(
            model, agents, block_size, temperature, train_lr, do_train, rag
        )
        k_mark = "✓ KIRCHHOFF" if abs(kirchhoff_val - math.pi) < 0.5 else f"KΔ={kirchhoff_val-math.pi:+.3f}"
        lb_mark = f"balance={load_balance:.3f}"
        print(f"    Координация DYNAMIC: {k_mark}  {lb_mark}  "
              f"kirchhoff={kirchhoff_val:.3f}")

        elapsed = time.perf_counter() - cycle_t
        all_lci = [r["avg_lci"] for r in agent_results]
        avg_all = sum(all_lci) / len(all_lci) if all_lci else 0.0
        resonant = abs(kirchhoff_val - math.pi) < _LCI_EPSILON and load_balance > 0.7
        total_gen = sum(a.n_generated for a in agents)

        print(f"    → avg_LCI_all={avg_all:.3f}  load_balance={load_balance:.3f}  "
              f"{'✓ РЕЗОНАНС' if resonant else '✗'}  gen={total_gen}  t={elapsed:.1f}s")

        log.append({
            "cycle":          cycle,
            "lci_r_starts":   [round(l, 4) for l in lci_r_starts],
            "agent_results":  agent_results,
            "kirchhoff_val":  round(kirchhoff_val, 4),
            "load_balance":   round(load_balance, 4),
            "avg_lci_all":    round(avg_all, 4),
            "resonant":       resonant,
            "n_generated":    total_gen,
            "elapsed_s":      round(elapsed, 2),
        })

    n_res = sum(1 for r in log if r["resonant"])
    avg_lb = sum(r["load_balance"] for r in log) / len(log) if log else 0.0
    avg_l  = sum(r["avg_lci_all"] for r in log) / len(log) if log else 0.0
    print(f"\n{'─' * 72}")
    print(f"  ИТОГ МУЛЬТИ-КОММИВОЯЖЁРА:")
    print(f"    Резонансных циклов: {n_res}/{n_cycles}")
    print(f"    avg_load_balance:   {avg_lb:.3f}  (1.0 = идеальный баланс)")
    print(f"    avg_LCI:            {avg_l:.3f}  (цель π={math.pi:.3f})")
    print(f"    RAG-буфер:          {len(rag)} текстов")
    return log


def _load_model(path: str) -> Variant3GPT:
    cfg = Variant3Config(**MODEL_CFG)
    m = Variant3GPT(cfg)
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck.get("model_state", ck), strict=False)
        print(f"  Загружен: {path}")
    else:
        print(f"  [!] Не найден: {path} — случайные веса")
    m.to(DEVICE)
    print(f"  Модель: {sum(p.numel() for p in m.parameters())/1e6:.2f}M параметров")
    return m


def _load_seeds(block_size: int) -> List[str]:
    texts = [
        "def forward(self, x): return self.linear(x)",
        "loss.backward(); optimizer.zero_grad(); scheduler.step()",
        "x = x + self.crossing(out_a, out_b)",
        "The hexagram represents the intersection of abstract and concrete.",
        "consciousness emerges from recursive self-reference in Q6 space",
    ] * 9
    for h in range(64):
        texts.append(_ids_to_text(_hex_prompt(h, block_size)))
    return texts


def main():
    parser = argparse.ArgumentParser(description="Мульти-коммивояжёр самообучение HMoE")
    parser.add_argument("--checkpoint",  type=str,   default="hmoe_curriculum.pt")
    parser.add_argument("--fast",        action="store_true", help="2 цикла, 3 шага, 2 агента")
    parser.add_argument("--cycles",      type=int,   default=4)
    parser.add_argument("--agents",      type=int,   default=3, help="число агентов (1-3)")
    parser.add_argument("--steps",       type=int,   default=10)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--no-train",    action="store_true")
    parser.add_argument("--save",        type=str,   default="hmoe_multisalesman.pt")
    args = parser.parse_args()

    if args.fast:
        args.cycles = 2
        args.steps = 3
        args.agents = 2

    block_size = MODEL_CFG["block_size"] - 1
    print(f"\n{'═' * 72}")
    print(f"  МУЛЬТИ-КОММИВОЯЖЁР HMoE")
    print(f"{'═' * 72}")

    model = _load_model(args.checkpoint)
    seeds = _load_seeds(block_size)
    print(f"  Seed текстов: {len(seeds)}")

    import json
    t0 = time.perf_counter()
    log = multi_salesman(
        model            = model,
        seed_texts       = seeds,
        block_size       = block_size,
        n_cycles         = args.cycles,
        n_agents         = args.agents,
        steps_per_expert = args.steps,
        temperature      = args.temperature,
        train_lr         = args.lr,
        do_train         = not args.no_train,
    )
    elapsed = time.perf_counter() - t0

    torch.save({"model_state": model.state_dict(), "multisalesman_log": log,
                "elapsed_sec": round(elapsed, 2)}, args.save)
    print(f"\n  Сохранено: {args.save}  (elapsed={elapsed:.1f}s)")

    log_path = args.save.replace(".pt", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  Лог: {log_path}")


if __name__ == "__main__":
    main()
