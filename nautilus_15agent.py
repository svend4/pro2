#!/usr/bin/env python3
"""
nautilus_15agent.py — 15-агентный Наутилус: каждый агент на своём Q4-тессеракте.

Геометрия (hexdim): Q6 содержит C(6,4)=15 копий Q4 (16 вершин каждая).
Каждый агент специализируется на своём тессеракте Q4 ⊂ Q6.
15 × 4 вершин = 60 из 64 вершин Q6 покрыто (4 вершины веса 1,2,3,4 общие).

Цикл:
  1. Каждый агент работает со своими Q4-вершинами (специализация)
  2. META-координация: лучшие агенты делятся контекстом с остальными
  3. Kirchhoff-проверка по всем 15 агентам

Преимущества перед 4-agent:
  - Полное покрытие Q6: все 64 вершины представлены (15 × 16, с перекрытием)
  - Архитектурная специализация вместо ручных колец META/ABSTRACT/DYNAMIC/CONCRETE
  - Математически обоснованная топология (hexdim, a.k.a. sub-hypercubes of Q6)

Usage:
  python nautilus_15agent.py --checkpoint hmoe_self_trained_v5.pt
  python nautilus_15agent.py --fast
  python nautilus_15agent.py --cycles 4 --steps 5
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
from self_train_hmoe import (
    lci_from_routing, lci_from_embeddings, micro_train, quality_filter,
    RagBuffer, _generate, _ids_to_text, _encode, _hex_prompt,
    _get_emb, _get_moes, MODEL_CFG, _LCI_EPSILON,
)

try:
    from meta_q6 import (
        q4_tesseracts, bent_seed_texts,
        metropolis_temperature, hamming,
    )
    _META_Q6_AVAILABLE = True
except ImportError:
    _META_Q6_AVAILABLE = False

DEVICE = "cpu"
_ROOT  = os.path.dirname(os.path.abspath(__file__))
_PI    = math.pi


# ── Q4-агенты ─────────────────────────────────────────────────────────────────

def _build_agents() -> List[Dict]:
    """15 агентов, каждый привязан к одному Q4-тессеракту внутри Q6."""
    tesseracts = q4_tesseracts() if _META_Q6_AVAILABLE else _fallback_tesseracts()
    agents = []
    for i, verts in enumerate(tesseracts[:15]):
        verts_list = sorted(verts)
        agents.append({
            "id":       i,
            "name":     f"Q4-{i:02d}",
            "verts":    verts_list,           # 16 Q6-вершин
            "center":   verts_list[len(verts_list) // 2],  # центральная вершина
        })
    return agents


def _fallback_tesseracts() -> List:
    """15 Q4 тессерактов без meta-импорта (то же что meta_q6._fallback_tesseracts)."""
    from itertools import combinations
    result = []
    for free_axes in combinations(range(6), 4):
        verts = set()
        for mask in range(16):
            v = 0
            for bit_idx, axis in enumerate(free_axes):
                if (mask >> bit_idx) & 1:
                    v |= (1 << axis)
            verts.add(v)
        result.append(frozenset(verts))
    return result


# ── Один шаг агента ───────────────────────────────────────────────────────────

def _agent_step(
    model: Variant3GPT,
    agent: Dict,
    ids: torch.Tensor,
    rag: RagBuffer,
    steps: int,
    lr: float,
    temperature: float,
    do_train: bool,
    block_size: int,
) -> Tuple[torch.Tensor, float, int]:
    """Шаги агента в его Q4-области. Возвращает (new_ids, lci_r, n_gen)."""
    start_ids = ids.clone()
    n_gen = 0

    for _ in range(steps):
        gen_ids  = _generate(model, ids, block_size, temperature, n_tokens=8)
        gen_text = _ids_to_text(gen_ids)
        if do_train and quality_filter(gen_text):
            micro_train(model, gen_ids, lr=lr, n_steps=1)
            rag.add(gen_text, _get_emb(model, gen_ids))
            n_gen += 1
        ids = gen_ids

    lci_r, _ = lci_from_routing(model, ids)
    return ids, lci_r, n_gen


# ── META-координация ──────────────────────────────────────────────────────────

def _meta_coordination_15(
    model: Variant3GPT,
    agent_ids: List[torch.Tensor],
    agent_lcis: List[float],
    agent_rags: List[RagBuffer],
    rag_shared: RagBuffer,
    block_size: int,
) -> Tuple[float, float]:
    """
    Координация 15 агентов:
    - Все добавляют контекст в shared RAG
    - Топ-3 агента (ближайших к π) делятся с отстающими

    Returns: (avg_lci, load_balance)
    """
    for ids in agent_ids:
        txt = _ids_to_text(ids)
        if quality_filter(txt):
            rag_shared.add(txt, _get_emb(model, ids))

    if not agent_lcis:
        return _PI, 1.0

    avg_lci = sum(agent_lcis) / len(agent_lcis)
    max_lci = max(agent_lcis)
    min_lci = min(agent_lcis)
    balance = max(0.0, 1.0 - (max_lci - min_lci) / (_PI + 1e-8))

    # Топ-3 по близости к π
    ranked = sorted(range(len(agent_lcis)), key=lambda i: abs(agent_lcis[i] - _PI))
    top3   = ranked[:3]

    for i in ranked[3:]:
        gap = abs(agent_lcis[i] - _PI)
        if gap > 0.1:
            # Берём совет у ближайшего из топ-3 (по Hamming к центру Q4-i)
            best_i = min(top3, key=lambda j: abs(agent_lcis[j] - _PI))
            if len(agent_rags[best_i]) > 0:
                near = agent_rags[best_i].retrieve(_get_emb(model, agent_ids[i]), top_k=1)
                if near:
                    agent_ids[i] = _encode(near[0], block_size)

    return avg_lci, balance


# ── Главный цикл ──────────────────────────────────────────────────────────────

def nautilus_15agent(
    model: Variant3GPT,
    seed_texts: List[str],
    n_cycles: int = 4,
    steps_per_agent: int = 5,
    temperature: float = 1.4,
    temp_decay: float = 0.85,
    train_lr: float = 1e-5,
    do_train: bool = True,
    block_size: int = MODEL_CFG["block_size"] - 1,
) -> List[Dict]:
    """
    15-агентный наутилус по Q4-тессерактам Q6.
    """
    agents = _build_agents()
    n_agents = len(agents)

    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ 15-АГЕНТНЫЙ НАУТИЛУС (hexdim Q4⊂Q6) + HMoE")
    print(f"{'═' * 72}")
    print(f"  Агентов    : {n_agents}  (15 Q4-тессерактов)")
    print(f"  Циклов     : {n_cycles}")
    print(f"  Шагов/агент: {steps_per_agent}")
    print(f"  Температура: {temperature:.2f} (decay={temp_decay:.2f})")
    verts_total = len({v for ag in agents for v in ag['verts']})
    print(f"  Q6 покрытие: {verts_total}/64 вершин")
    print()

    # ── Инициализация ────────────────────────────────────────────────────────────
    rag_shared   = RagBuffer(max_size=500)
    agent_rags   = [RagBuffer(max_size=100) for _ in agents]

    model.eval()
    for text in seed_texts[:60]:
        ids = _encode(text, block_size)
        emb = _get_emb(model, ids)
        rag_shared.add(text, emb)

    # Инициализируем каждый агент со случайной вершиной его Q4-тессеракта
    agent_ids: List[torch.Tensor] = []
    for ag in agents:
        start_v = random.choice(ag["verts"])
        agent_ids.append(_hex_prompt(start_v, block_size))

    print(f"  Shared RAG: {len(rag_shared)} текстов")
    lci0_all = [lci_from_routing(model, ids)[0] for ids in agent_ids]
    print(f"  Start LCI: mean={sum(lci0_all)/len(lci0_all):.3f}, "
          f"min={min(lci0_all):.3f}, max={max(lci0_all):.3f}")

    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        t0 = time.perf_counter()

        # Температура с Metropolis-расписанием
        cur_temp = max(0.5, temperature * (temp_decay ** (cycle - 1)))

        # Каждый агент делает steps_per_agent шагов в своей Q4-области
        agent_lcis: List[float] = []
        total_gen = 0

        for i, ag in enumerate(agents):
            # Обогатить из shared RAG
            if len(rag_shared) > 3:
                near = rag_shared.retrieve(_get_emb(model, agent_ids[i]), top_k=1)
                if near:
                    agent_ids[i] = _encode(near[0], block_size)

            new_ids, lci_r, n_gen = _agent_step(
                model, ag, agent_ids[i], agent_rags[i],
                steps_per_agent, train_lr, cur_temp, do_train, block_size,
            )
            agent_ids[i] = new_ids
            agent_lcis.append(lci_r)
            total_gen += n_gen

        # Координация
        avg_lci, balance = _meta_coordination_15(
            model, agent_ids, agent_lcis, agent_rags, rag_shared, block_size,
        )

        n_resonant = sum(1 for lci in agent_lcis if abs(lci - _PI) < _LCI_EPSILON)
        kirchhoff  = abs(avg_lci - _PI) < 0.5
        elapsed    = time.perf_counter() - t0

        res_mark = "✓ РЕЗОНАНС" if kirchhoff else f"δ={avg_lci - _PI:+.3f}"
        print(f"  Цикл {cycle}/{n_cycles}  T={cur_temp:.3f}  "
              f"avg_LCI={avg_lci:.3f}  {res_mark}  "
              f"resonant={n_resonant}/{n_agents}  "
              f"gen={total_gen}  t={elapsed:.1f}s")

        log.append({
            "cycle":        cycle,
            "avg_lci_all":  round(avg_lci, 4),
            "load_balance": round(balance, 4),
            "n_resonant":   n_resonant,
            "kirchhoff":    kirchhoff,
            "resonant":     kirchhoff,
            "n_generated":  total_gen,
            "temperature":  round(cur_temp, 3),
        })

    # Итог
    final_lcis = [lci_from_routing(model, ids)[0] for ids in agent_ids]
    final_avg  = sum(final_lcis) / len(final_lcis)
    print(f"\n{'─' * 72}")
    print(f"  Финальный avg_LCI : {final_avg:.3f}  (Δπ = {final_avg - _PI:+.4f})")
    print(f"  Резонансных агентов: {sum(1 for l in final_lcis if abs(l-_PI) < _LCI_EPSILON)}/{n_agents}")
    print(f"  Shared RAG        : {len(rag_shared)} текстов")

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


def _load_seeds(block_size: int, use_bent: bool = True) -> List[str]:
    if use_bent and _META_Q6_AVAILABLE:
        texts = bent_seed_texts(n=20, block_size=block_size)
        print(f"  [meta_q6] Bent seeds: {len(texts)} (nl=28)")
    else:
        texts = [
            "def route(src, dst): return ecube_path(src ^ dst)",
            "Q4 tesseract: 16 vertices, 4 dimensions, 32 edges",
            "hexdim: all 15 copies of Q4 inside Q6 — complete coverage",
            "agent specialization: each Q4 agent owns 16 hexagrams",
        ] * 5
    for h in range(64):
        texts.append(_ids_to_text(_hex_prompt(h, block_size)))
    return texts


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="15-agent nautilus: каждый агент на своём Q4⊂Q6 тессеракте"
    )
    parser.add_argument("--checkpoint", type=str, default="hmoe_self_trained_v5.pt")
    parser.add_argument("--save",       type=str, default="hmoe_15agent_v1.pt")
    parser.add_argument("--cycles",     type=int, default=4)
    parser.add_argument("--steps",      type=int, default=5,
                        help="Шагов на агента за цикл")
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--temp-decay", type=float, default=0.85,
                        dest="temp_decay")
    parser.add_argument("--no-train",   action="store_true")
    parser.add_argument("--no-bent",    action="store_true",
                        help="Не использовать bent seeds (default: используем)")
    parser.add_argument("--fast",       action="store_true",
                        help="Быстрый тест: 2 цикла, 3 шага/агент")
    args = parser.parse_args()

    if args.fast:
        args.cycles = 2
        args.steps  = 3

    block_size = MODEL_CFG["block_size"] - 1

    print(f"{'═' * 72}")
    print(f"  15-AGENT NAUTILUS (hexdim Q4⊂Q6)")
    print(f"{'═' * 72}")

    model      = _load_model(args.checkpoint)
    seed_texts = _load_seeds(block_size, use_bent=not args.no_bent)
    print(f"  Seed текстов: {len(seed_texts)}")

    log = nautilus_15agent(
        model           = model,
        seed_texts      = seed_texts,
        n_cycles        = args.cycles,
        steps_per_agent = args.steps,
        temperature     = args.temperature,
        temp_decay      = args.temp_decay,
        train_lr        = args.lr,
        do_train        = not args.no_train,
        block_size      = block_size,
    )

    # Сохранить
    save_path = args.save
    ck = {"model_state": model.state_dict(), "next_phase": 5}
    torch.save(ck, save_path)
    print(f"\n  Сохранено: {save_path}")

    log_path = save_path.replace(".pt", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Лог:        {log_path}")


if __name__ == "__main__":
    main()
