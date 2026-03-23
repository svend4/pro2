#!/usr/bin/env python3
"""
roundabout.py — Кольцевая развязка (roundabout) для HMoE само-обучения.

Принцип кольца (roundabout):
  - Нет светофоров: непрерывный круговой поток токенов
  - Въезд по требованию: новый эксперт включается когда LCI отклоняется
  - Выезд по критерию: токен покидает кольцо когда |LCI - π| < ε
  - Приоритет у тех кто уже в кольце (momentum)
  - Максимальное число оборотов = защита от бесконечного кольца

Маршрут: ABSTRACT → DYNAMIC → CONCRETE → META → (обратно в ABSTRACT если нужно)
Выход: когда routing_LCI ∈ [π - ε, π + ε] И emb_LCI растёт (движение вперёд)

Ключевое отличие от turbine:
  turbine   = фиксированное число шагов на каждого эксперта
  roundabout = адаптивное: делает обороты пока не достигнет резонанса

Usage:
  python roundabout.py --checkpoint hmoe_self_trained_v5.pt
  python roundabout.py --fast
  python roundabout.py --max-laps 6 --lap-steps 10
"""

from __future__ import annotations

import argparse
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ROUNDABOUT_RING = ["ABSTRACT", "DYNAMIC", "CONCRETE", "META"]
_META_FREEZE = ["ABSTRACT", "DYNAMIC", "CONCRETE"]

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _freeze_for(model, expert: str):
    groups = _META_FREEZE if expert == "META" else [expert]
    for moe in _get_moes(model):
        _freeze_all_except(moe, groups)
    if expert in ("META", "DYNAMIC"):
        set_moe_stage(
            model.blocks[0].hmoe if hasattr(model.blocks[0], 'hmoe') else None, 4
        )


def roundabout_loop(
    model: Variant3GPT,
    seed_texts: List[str],
    block_size: int = MODEL_CFG["block_size"] - 1,
    n_cycles: int = 4,
    lap_steps: int = 10,
    max_laps: int = 4,
    temperature: float = 1.4,
    train_lr: float = 1e-5,
    do_train: bool = True,
) -> List[Dict]:
    """
    Кольцевая само-обучение HMoE.

    Один цикл = серия оборотов кольца.
    Каждый оборот: ABSTRACT → DYNAMIC → CONCRETE → META.
    Выход из кольца при достижении резонанса (LCI ≈ π) или max_laps.
    """
    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ КОЛЬЦЕВАЯ РАЗВЯЗКА (Roundabout) + HMoE")
    print(f"{'═' * 72}")
    print(f"  Циклов        : {n_cycles}")
    print(f"  Шагов/эксперт : {lap_steps}  (на один оборот кольца)")
    print(f"  Макс. оборотов: {max_laps}")
    print(f"  Температура   : {temperature:.2f}")
    print(f"  Кольцо        : {' → '.join(_ROUNDABOUT_RING)} → ↺")
    print()

    rag = RagBuffer(max_size=300)
    model.eval()
    for text in seed_texts[:50]:
        ids = _encode(text, block_size)
        rag.add(text, _get_emb(model, ids))
    print(f"  RAG-буфер: {len(rag)} текстов")

    current_ids = _hex_prompt(random.randint(0, 63), block_size)
    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        cycle_t = time.perf_counter()
        lci_r0, gw0 = lci_from_routing(model, current_ids)
        res_mark = "✓ РЕЗОНАНС" if abs(lci_r0 - math.pi) < _LCI_EPSILON else f"δ={lci_r0-math.pi:+.3f}"
        print(f"\n  Цикл {cycle}/{n_cycles}  routing_LCI={lci_r0:.3f}  {res_mark}")

        lap = 0
        total_gen = 0
        laps_log: List[Dict] = []
        exited_early = False

        while lap < max_laps:
            lap += 1
            lap_lci_r: List[float] = []

            for expert in _ROUNDABOUT_RING:
                # Обогатить промпт из RAG
                if len(rag) > 5:
                    cur_emb = _get_emb(model, current_ids)
                    retrieved = rag.retrieve(cur_emb, top_k=1)
                    if retrieved:
                        current_ids = _encode(retrieved[0], block_size)

                _freeze_for(model, expert)
                start_ids = current_ids.clone()

                for _ in range(lap_steps):
                    gen_ids = _generate(model, current_ids, block_size, temperature, n_tokens=8)
                    gen_text = _ids_to_text(gen_ids)
                    if do_train and quality_filter(gen_text):
                        micro_train(model, gen_ids, lr=train_lr, n_steps=1)
                        rag.add(gen_text, _get_emb(model, gen_ids))
                        total_gen += 1
                    current_ids = gen_ids

                lci_r_e, _ = lci_from_routing(model, current_ids)
                lap_lci_r.append(lci_r_e)

            avg_lap_lci = sum(lap_lci_r) / len(lap_lci_r)
            resonant = abs(avg_lap_lci - math.pi) < _LCI_EPSILON
            lap_mark = "✓ выход" if resonant else f"продолжаем (δ={avg_lap_lci-math.pi:+.3f})"
            print(f"    Оборот {lap}/{max_laps}  avg_routing_LCI={avg_lap_lci:.3f}  {lap_mark}")

            laps_log.append({"lap": lap, "avg_lci_r": round(avg_lap_lci, 4), "resonant": resonant})

            if resonant:
                exited_early = True
                break

        lci_r_end, gw_end = lci_from_routing(model, current_ids)
        elapsed = time.perf_counter() - cycle_t
        print(f"    → final_LCI={lci_r_end:.3f}  gen={total_gen}  "
              f"оборотов={lap}  {'✓ ранний выход' if exited_early else '✗ достигнут лимит'}  t={elapsed:.1f}s")

        log.append({
            "cycle":        cycle,
            "lci_r0":       round(lci_r0, 4),
            "lci_r_end":    round(lci_r_end, 4),
            "n_laps":       lap,
            "exited_early": exited_early,
            "n_generated":  total_gen,
            "laps":         laps_log,
            "elapsed_s":    round(elapsed, 2),
        })

    n_early = sum(1 for r in log if r["exited_early"])
    avg_lci = sum(r["lci_r_end"] for r in log) / len(log) if log else 0.0
    avg_laps = sum(r["n_laps"] for r in log) / len(log) if log else 0.0
    print(f"\n{'─' * 72}")
    print(f"  ИТОГ КОЛЬЦА:")
    print(f"    Ранних выходов:  {n_early}/{n_cycles}")
    print(f"    avg_LCI_r_end:   {avg_lci:.3f}  (цель π={math.pi:.3f})")
    print(f"    avg оборотов:    {avg_laps:.1f} / {max_laps}")
    print(f"    RAG-буфер:       {len(rag)} текстов")
    return log


def _load_model(path: str) -> Variant3GPT:
    cfg = Variant3Config(**MODEL_CFG)
    m = Variant3GPT(cfg)
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=True)
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
    parser = argparse.ArgumentParser(description="Roundabout самообучение HMoE")
    parser.add_argument("--checkpoint",   type=str,   default="hmoe_curriculum.pt")
    parser.add_argument("--fast",         action="store_true", help="2 цикла, 3 шага, 2 оборота")
    parser.add_argument("--cycles",       type=int,   default=4)
    parser.add_argument("--lap-steps",    type=int,   default=10, help="шагов на эксперта за оборот")
    parser.add_argument("--max-laps",     type=int,   default=4,  help="макс. оборотов кольца")
    parser.add_argument("--temperature",  type=float, default=1.4)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--no-train",     action="store_true")
    parser.add_argument("--save",         type=str,   default="hmoe_roundabout.pt")
    args = parser.parse_args()

    if args.fast:
        args.cycles = 2
        args.lap_steps = 3
        args.max_laps = 2

    block_size = MODEL_CFG["block_size"] - 1
    print(f"\n{'═' * 72}")
    print(f"  КОЛЬЦЕВАЯ РАЗВЯЗКА HMoE")
    print(f"{'═' * 72}")

    model = _load_model(args.checkpoint)
    seeds = _load_seeds(block_size)
    print(f"  Seed текстов: {len(seeds)}")

    import json
    t0 = time.perf_counter()
    log = roundabout_loop(
        model        = model,
        seed_texts   = seeds,
        block_size   = block_size,
        n_cycles     = args.cycles,
        lap_steps    = args.lap_steps,
        max_laps     = args.max_laps,
        temperature  = args.temperature,
        train_lr     = args.lr,
        do_train     = not args.no_train,
    )
    elapsed = time.perf_counter() - t0

    torch.save({"model_state": model.state_dict(), "roundabout_log": log,
                "elapsed_sec": round(elapsed, 2)}, args.save)
    print(f"\n  Сохранено: {args.save}  (elapsed={elapsed:.1f}s)")

    log_path = args.save.replace(".pt", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  Лог: {log_path}")


if __name__ == "__main__":
    main()
