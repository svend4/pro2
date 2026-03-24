#!/usr/bin/env python3
"""
test_lci_attractor.py — Тест аттрактора LCI ≈ π.

In-process: загружает модель один раз, запускает 5 независимых fast-прогонов
с разными random seeds (разные начальные hex-промпты).
Строит распределение финальных LCI.

Usage:
  python test_lci_attractor.py
  python test_lci_attractor.py --runs 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import HMoEConfig, HierarchicalMoEFFN
from self_train_hmoe import (
    lci_from_routing, lci_from_embeddings,
    _generate, _ids_to_text, _encode, _hex_prompt,
    MODEL_CFG, _LCI_EPSILON,
)
try:
    from self_train_hmoe import HMOE_CFG
except ImportError:
    from yijing_transformer.models.hierarchical_moe import HMoEConfig
    HMOE_CFG = HMoEConfig()

_PI   = math.pi
_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_seed(model: Variant3GPT, seed: int, n_samples: int,
             block_size: int, **_) -> Dict:
    """Один прогон с данным seed.

    Измеряет LCI на n_samples случайных hex-промптах.
    Нет генерации — быстро.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Сэмплируем случайные начальные вершины Q6
    hexes = [random.randint(0, 63) for _ in range(n_samples)]
    lcis  = []
    for h in hexes:
        ids = _hex_prompt(h, block_size)
        lci_r, _ = lci_from_routing(model, ids)
        lcis.append(lci_r)

    mean_lci = sum(lcis) / len(lcis)
    return {
        "seed":       seed,
        "hex_sample": hexes[:4],
        "lci_per_hex": [round(l, 4) for l in lcis],
        "final_lci":   round(mean_lci, 4),
        "delta_pi":    round(mean_lci - _PI, 4),
        "std":         round((sum((x - mean_lci)**2 for x in lcis) / len(lcis)) ** 0.5, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",       type=int,   default=5)
    parser.add_argument("--cycles",     type=int,   default=20,
                        help="число случайных hex-промптов на seed")
    parser.add_argument("--checkpoint", default="hmoe_self_trained_v5.pt")
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--output",     default="experiments/lci_attractor_results.json")
    args = parser.parse_args()

    # Загружаем модель один раз
    print(f"{'═' * 60}")
    print(f"  ТЕСТ АТТРАКТОРА LCI ≈ π")
    print(f"{'═' * 60}")
    print(f"  Прогонов   : {args.runs}  (разные random seed)")
    print(f"  Сэмплов/run: {args.cycles}  (случайных hex-промптов, без генерации)")
    print(f"  Чекпоинт   : {args.checkpoint}")
    print()

    cfg   = Variant3Config(**MODEL_CFG)
    model = Variant3GPT(cfg)
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            block.hmoe = HierarchicalMoEFFN(HMOE_CFG)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        print(f"  Загружен: {args.checkpoint}")
    else:
        print(f"  ⚠️ Чекпоинт не найден, используем случайные веса")

    model.eval()
    block_size = MODEL_CFG["block_size"] - 1
    seeds = [42, 137, 271, 314, 999][:args.runs]

    results = []
    for i, seed in enumerate(seeds, 1):
        t0 = time.perf_counter()
        r = run_seed(model, seed, args.cycles, block_size)
        r["elapsed_s"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        print(f"  seed={seed:4d}  LCI={r['final_lci']:.4f}  Δπ={r['delta_pi']:+.4f}"
              f"  std={r['std']:.4f}  t={r['elapsed_s']}s")

    # Анализ
    lcis = [r["final_lci"] for r in results]
    mean = sum(lcis) / len(lcis)
    var  = sum((x - mean) ** 2 for x in lcis) / len(lcis)
    std  = var ** 0.5
    close = sum(1 for l in lcis if abs(l - _PI) < 0.1)

    print()
    print(f"{'─' * 60}")
    print(f"  ИТОГ")
    print(f"  LCI финальные: {[f'{l:.4f}' for l in lcis]}")
    print(f"  mean={mean:.4f}  std={std:.4f}  Δπ={mean - _PI:+.4f}")
    print(f"  Близко к π (|Δ|<0.1): {close}/{len(lcis)}")
    attractor_verdict = (
        "ДА — π является аттрактором" if std < 0.15 and abs(mean - _PI) < 0.15
        else "НЕТ — π не является явным аттрактором"
    )
    print(f"  Вердикт: {attractor_verdict}")

    # Сохраняем
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary = {
        "runs": results,
        "mean_lci":   round(mean, 4),
        "std_lci":    round(std, 4),
        "delta_pi":   round(mean - _PI, 4),
        "close_to_pi": close,
        "n_runs":     len(lcis),
        "verdict":    attractor_verdict,
        "pi":         _PI,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Результаты: {args.output}")


if __name__ == "__main__":
    main()
