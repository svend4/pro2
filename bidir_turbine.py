#!/usr/bin/env python3
"""
bidir_turbine.py — Двунаправленная турбина для HMoE само-обучения.

Принцип:
  Два потока токенов движутся навстречу друг другу:
    Поток A (zoom-out): ABSTRACT → DYNAMIC → CONCRETE
    Поток B (zoom-in):  CONCRETE → DYNAMIC → ABSTRACT
  Встреча в точке DYNAMIC = точка X (крест восьмёрки).
  На встрече: обмен контекстом через RAG + Kirchhoff-проверка.

Аналог из data7/knowledge_transformer.py:
  Dissertation → (decompose) → Encyclopedia → (synthesize) → Dissertation
  Здесь:
    abstract_text → (zoom-in) → concrete_text → (zoom-out) → abstract_text

Дополнительно — Multi-TSP:
  Каждый поток ведёт своего "коммивояжёра" по экспертам.
  Встреча = объединение маршрутов в точке DYNAMIC.

Usage:
  python bidir_turbine.py --checkpoint hmoe_self_trained_v5.pt
  python bidir_turbine.py --fast
  python bidir_turbine.py --cycles 6 --steps 15
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
_ROOT  = os.path.dirname(os.path.abspath(__file__))

# Потоки турбины
_STREAM_A = ["ABSTRACT", "DYNAMIC", "CONCRETE"]   # zoom-out → центр → zoom-in
_STREAM_B = ["CONCRETE", "DYNAMIC", "ABSTRACT"]   # zoom-in  → центр → zoom-out
_META_FREEZE = ["ABSTRACT", "DYNAMIC", "CONCRETE"]

# Kirchhoff порог для точки встречи
_MEET_KIRCHHOFF_EPS = 0.5


def _freeze_for(model, expert: str):
    groups = _META_FREEZE if expert == "META" else [expert]
    for moe in _get_moes(model):
        _freeze_all_except(moe, groups)
    if expert in ("DYNAMIC", "META"):
        set_moe_stage(
            model.blocks[0].hmoe if hasattr(model.blocks[0], 'hmoe') else None, 4
        )


def _run_stream(
    model: Variant3GPT,
    start_ids: torch.Tensor,
    stream: List[str],
    n_steps: int,
    temperature: float,
    block_size: int,
    train_lr: float,
    do_train: bool,
    rag: RagBuffer,
    stream_name: str,
) -> Tuple[torch.Tensor, List[float], int]:
    """Прогнать один поток (A или B) через список экспертов."""
    current_ids = start_ids.clone()
    lci_r_list: List[float] = []
    total_gen = 0

    for expert in stream:
        _freeze_for(model, expert)
        for _ in range(n_steps):
            gen_ids = _generate(model, current_ids, block_size, temperature, n_tokens=8)
            gen_text = _ids_to_text(gen_ids)
            if do_train and quality_filter(gen_text):
                lr_scale = 0.5 if expert == "CONCRETE" else 1.0
                micro_train(model, gen_ids, lr=train_lr * lr_scale, n_steps=2)
                rag.add(gen_text, _get_emb(model, gen_ids))
                total_gen += 1
            current_ids = gen_ids

        lci_r, _ = lci_from_routing(model, current_ids)
        lci_r_list.append(lci_r)

    return current_ids, lci_r_list, total_gen


def _meet_at_dynamic(
    model: Variant3GPT,
    ids_a: torch.Tensor,
    ids_b: torch.Tensor,
    block_size: int,
    temperature: float,
    train_lr: float,
    do_train: bool,
    rag: RagBuffer,
) -> Tuple[torch.Tensor, float, float]:
    """
    Точка встречи двух потоков в DYNAMIC (точка X).

    Смешивает контексты: объединяет через RAG, обучает BidirBridge.
    Возвращает объединённый контекст + Kirchhoff метрику.
    """
    _freeze_for(model, "DYNAMIC")

    # Получить эмбеддинги обоих потоков
    emb_a = _get_emb(model, ids_a)
    emb_b = _get_emb(model, ids_b)

    # Cosine similarity между потоками = "угол встречи"
    cos_ab = F.cosine_similarity(
        F.normalize(emb_a.unsqueeze(0).float(), dim=-1),
        F.normalize(emb_b.unsqueeze(0).float(), dim=-1)
    ).item()
    meet_angle = math.acos(max(-1.0, min(1.0, cos_ab)))

    # Сохранить оба контекста в RAG
    rag.add(_ids_to_text(ids_a), emb_a)
    rag.add(_ids_to_text(ids_b), emb_b)

    # Промпт = среднее (конкатенация коротких фрагментов из обоих)
    text_a = _ids_to_text(ids_a)[:16]
    text_b = _ids_to_text(ids_b)[:16]
    merged_text = text_a + " " + text_b
    merged_ids = _encode(merged_text, block_size)

    # Несколько шагов через DYNAMIC
    for _ in range(5):
        gen_ids = _generate(model, merged_ids, block_size, temperature, n_tokens=8)
        gen_text = _ids_to_text(gen_ids)
        if do_train and quality_filter(gen_text):
            micro_train(model, gen_ids, lr=train_lr, n_steps=2)
            rag.add(gen_text, _get_emb(model, gen_ids))
        merged_ids = gen_ids

    lci_r, _ = lci_from_routing(model, merged_ids)
    kirchhoff_dev = abs(lci_r - math.pi)

    return merged_ids, lci_r, meet_angle


def bidir_turbine(
    model: Variant3GPT,
    seed_texts: List[str],
    block_size: int = MODEL_CFG["block_size"] - 1,
    n_cycles: int = 4,
    steps_per_expert: int = 10,
    temperature: float = 1.4,
    train_lr: float = 1e-5,
    do_train: bool = True,
) -> List[Dict]:
    """
    Двунаправленное само-обучение HMoE.

    Каждый цикл:
      1. Поток A: ABSTRACT → DYNAMIC → CONCRETE  (zoom-out)
      2. Поток B: CONCRETE → DYNAMIC → ABSTRACT  (zoom-in)
      3. Встреча в DYNAMIC: обмен контекстом + Kirchhoff
      4. Объединённый контекст → следующий цикл
    """
    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ ДВУНАПРАВЛЕННАЯ ТУРБИНА + HMoE")
    print(f"{'═' * 72}")
    print(f"  Циклов        : {n_cycles}")
    print(f"  Шагов/эксперт : {steps_per_expert}")
    print(f"  Температура   : {temperature:.2f}")
    print(f"  Поток A (↓)   : {' → '.join(_STREAM_A)}  (zoom-out)")
    print(f"  Поток B (↑)   : {' → '.join(_STREAM_B)}  (zoom-in)")
    print(f"  Встреча       : DYNAMIC (точка X, BidirBridge)")
    print()

    rag = RagBuffer(max_size=400)
    model.eval()
    for text in seed_texts[:50]:
        ids = _encode(text, block_size)
        rag.add(text, _get_emb(model, ids))
    print(f"  RAG-буфер: {len(rag)} текстов")

    # Два независимых стартовых промпта
    ids_a = _hex_prompt(random.randint(0,  31), block_size)   # первая половина гексаграмм
    ids_b = _hex_prompt(random.randint(32, 63), block_size)   # вторая половина

    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        cycle_t = time.perf_counter()
        lci_r0, gw0 = lci_from_routing(model, ids_a)
        print(f"\n  Цикл {cycle}/{n_cycles}  routing_LCI(A)={lci_r0:.3f}")

        # ── Поток A: zoom-out ─────────────────────────────────────────────
        end_a, lci_a_list, gen_a = _run_stream(
            model, ids_a, _STREAM_A, steps_per_expert,
            temperature, block_size, train_lr, do_train, rag, "A"
        )
        print(f"    Поток A: LCI={[f'{l:.3f}' for l in lci_a_list]}  gen={gen_a}")

        # ── Поток B: zoom-in ──────────────────────────────────────────────
        end_b, lci_b_list, gen_b = _run_stream(
            model, ids_b, _STREAM_B, steps_per_expert,
            temperature, block_size, train_lr, do_train, rag, "B"
        )
        print(f"    Поток B: LCI={[f'{l:.3f}' for l in lci_b_list]}  gen={gen_b}")

        # ── Встреча в DYNAMIC ─────────────────────────────────────────────
        merged_ids, lci_meet, meet_angle = _meet_at_dynamic(
            model, end_a, end_b, block_size, temperature, train_lr, do_train, rag
        )
        k_mark = "✓ KIRCHHOFF" if abs(lci_meet - math.pi) < _MEET_KIRCHHOFF_EPS else f"KΔ={lci_meet-math.pi:+.3f}"
        print(f"    Встреча: LCI={lci_meet:.3f}  угол={math.degrees(meet_angle):.1f}°  {k_mark}")

        # Следующий цикл: A начинает от конца B и наоборот (перекрёстный старт)
        ids_a = merged_ids.clone()
        ids_b = merged_ids.clone()

        elapsed = time.perf_counter() - cycle_t
        avg_lci = (sum(lci_a_list) + sum(lci_b_list) + lci_meet) / (len(lci_a_list) + len(lci_b_list) + 1)
        resonant = abs(lci_meet - math.pi) < _LCI_EPSILON
        print(f"    → avg_LCI={avg_lci:.3f}  gen={gen_a+gen_b}  "
              f"{'✓ РЕЗОНАНС' if resonant else '✗'}  t={elapsed:.1f}s")

        log.append({
            "cycle":       cycle,
            "lci_r0":      round(lci_r0, 4),
            "lci_a":       [round(l, 4) for l in lci_a_list],
            "lci_b":       [round(l, 4) for l in lci_b_list],
            "lci_meet":    round(lci_meet, 4),
            "meet_angle_deg": round(math.degrees(meet_angle), 2),
            "avg_lci":     round(avg_lci, 4),
            "resonant":    resonant,
            "n_generated": gen_a + gen_b,
            "elapsed_s":   round(elapsed, 2),
        })

    n_res = sum(1 for r in log if r["resonant"])
    avg_l = sum(r["avg_lci"] for r in log) / len(log) if log else 0.0
    print(f"\n{'─' * 72}")
    print(f"  ИТОГ BIDIR:")
    print(f"    Резонансных: {n_res}/{n_cycles}")
    print(f"    avg_LCI:     {avg_l:.3f}  (цель π={math.pi:.3f})")
    print(f"    RAG-буфер:   {len(rag)} текстов")
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
    parser = argparse.ArgumentParser(description="Двунаправленная турбина HMoE")
    parser.add_argument("--checkpoint",  type=str,   default="hmoe_curriculum.pt")
    parser.add_argument("--fast",        action="store_true", help="2 цикла, 3 шага/эксперт")
    parser.add_argument("--cycles",      type=int,   default=4)
    parser.add_argument("--steps",       type=int,   default=10, help="шагов на эксперта")
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--no-train",    action="store_true")
    parser.add_argument("--save",        type=str,   default="hmoe_bidir.pt")
    args = parser.parse_args()

    if args.fast:
        args.cycles = 2
        args.steps = 3

    block_size = MODEL_CFG["block_size"] - 1
    print(f"\n{'═' * 72}")
    print(f"  ДВУНАПРАВЛЕННАЯ ТУРБИНА HMoE")
    print(f"{'═' * 72}")

    model = _load_model(args.checkpoint)
    seeds = _load_seeds(block_size)
    print(f"  Seed текстов: {len(seeds)}")

    import json
    t0 = time.perf_counter()
    log = bidir_turbine(
        model            = model,
        seed_texts       = seeds,
        block_size       = block_size,
        n_cycles         = args.cycles,
        steps_per_expert = args.steps,
        temperature      = args.temperature,
        train_lr         = args.lr,
        do_train         = not args.no_train,
    )
    elapsed = time.perf_counter() - t0

    torch.save({"model_state": model.state_dict(), "bidir_log": log,
                "elapsed_sec": round(elapsed, 2)}, args.save)
    print(f"\n  Сохранено: {args.save}  (elapsed={elapsed:.1f}s)")

    log_path = args.save.replace(".pt", "_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  Лог: {log_path}")


if __name__ == "__main__":
    main()
