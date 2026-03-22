#!/usr/bin/env python3
"""
bench_all.py — Бенчмарк всех вариантов само-обучения HMoE.

Запускает все варианты в режиме --fast (минимальные прогоны),
собирает метрики и выводит сравнительную таблицу.

Варианты (от простых к сложным):
  1. figure-8 baseline        — оригинальная петля (A + B)
  2. figure-8 fixed-T         — baseline + фиксированная температура (v7)
  3. turbine greedy-TSP       — 4 эксперта, динамический TSP-маршрут
  4. turbine 2opt-TSP         — turbine + 2-opt улучшение маршрута
  5. turbine LCI-loss         — turbine + явный Kirchhoff штраф в loss
  6. turbine no-tsp           — turbine с фиксированным порядком
  7. roundabout               — кольцо с адаптивным числом оборотов
  8. bidir-turbine            — два встречных потока, встреча в DYNAMIC
  9. multi-salesman (2)       — 2 агента с общим RAG
 10. multi-salesman (3)       — 3 агента с load balancing

Метрики сравнения:
  - avg_LCI_r        — routing LCI (цель: π ≈ 3.14)
  - resonance_rate   — % циклов в резонансе
  - kirchhoff_ok     — % циклов с Kirchhoff-балансом
  - gen_per_cycle    — генерация текстов за цикл (производительность)
  - elapsed_s        — время в секундах
  - score            — взвешенная оценка (главная метрика)

Usage:
  python bench_all.py --checkpoint hmoe_self_trained_v5.pt
  python bench_all.py --checkpoint hmoe_self_trained_v5.pt --variants 1,2,3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PI   = math.pi


# ── Конфигурации вариантов ────────────────────────────────────────────────────

VARIANTS = [
    {
        "id":     1,
        "name":   "figure-8 baseline",
        "script": "self_train_hmoe.py",
        "args":   "--fast --cycles 2 --steps_per_loop 5",
        "log_key": "figure8_log",
        "type":   "figure8",
    },
    {
        "id":     2,
        "name":   "figure-8 fixed-T",
        "script": "self_train_hmoe.py",
        "args":   "--fast --cycles 2 --steps_per_loop 5 --fixed-temperature",
        "log_key": "figure8_log",
        "type":   "figure8",
    },
    {
        "id":     3,
        "name":   "turbine greedy-TSP",
        "script": "figure8_turbine.py",
        "args":   "--fast --cycles 2 --steps_per_expert 5",
        "log_key": "turbine_log",
        "type":   "turbine",
    },
    {
        "id":     4,
        "name":   "turbine 2opt-TSP",
        "script": "figure8_turbine.py",
        "args":   "--fast --cycles 2 --steps_per_expert 5 --tsp-2opt",
        "log_key": "turbine_log",
        "type":   "turbine",
    },
    {
        "id":     5,
        "name":   "turbine LCI-loss λ=0.1",
        "script": "figure8_turbine.py",
        "args":   "--fast --cycles 2 --steps_per_expert 5 --lci-loss 0.1",
        "log_key": "turbine_log",
        "type":   "turbine",
    },
    {
        "id":     6,
        "name":   "turbine no-TSP",
        "script": "figure8_turbine.py",
        "args":   "--fast --cycles 2 --steps_per_expert 5 --no-tsp",
        "log_key": "turbine_log",
        "type":   "turbine",
    },
    {
        "id":     7,
        "name":   "roundabout",
        "script": "roundabout.py",
        "args":   "--fast",
        "log_key": "roundabout_log",
        "type":   "roundabout",
    },
    {
        "id":     8,
        "name":   "bidir-turbine",
        "script": "bidir_turbine.py",
        "args":   "--fast",
        "log_key": "bidir_log",
        "type":   "bidir",
    },
    {
        "id":     9,
        "name":   "multi-salesman (2 агента)",
        "script": "multi_salesman.py",
        "args":   "--fast --agents 2",
        "log_key": "multisalesman_log",
        "type":   "multisalesman",
    },
    {
        "id":    10,
        "name":  "multi-salesman (3 агента)",
        "script": "multi_salesman.py",
        "args":  "--fast --agents 3",
        "log_key": "multisalesman_log",
        "type":  "multisalesman",
    },
    {
        "id":    11,
        "name":  "nautilus-clover (bidir)",
        "script": "nautilus_clover.py",
        "args":  "--fast",
        "log_key": "nautilus_log",
        "type":  "nautilus",
    },
    {
        "id":    12,
        "name":  "nautilus-clover + LCI-loss",
        "script": "nautilus_clover.py",
        "args":  "--fast --lci-loss 0.1",
        "log_key": "nautilus_log",
        "type":  "nautilus",
    },
]


# ── Парсинг метрик из логов ────────────────────────────────────────────────────

def extract_metrics(log: List[Dict], variant_type: str) -> Dict:
    """Извлечь унифицированные метрики из лога любого варианта."""
    if not log:
        return {"avg_lci_r": 0.0, "resonance_rate": 0.0, "kirchhoff_ok": 0.0,
                "gen_per_cycle": 0.0, "score": 0.0}

    n = len(log)

    if variant_type == "figure8":
        avg_lci_r      = sum(r.get("avg_lci_r", r.get("lci_a_r", 0)) for r in log) / n
        # resonance: проверяем routing_LCI (как у остальных вариантов)
        resonance_rate = sum(
            1 for r in log
            if abs(r.get("avg_lci_r", r.get("lci_a_r", 0)) - _PI) < 0.5
        ) / n
        kirchhoff_ok   = resonance_rate
        # поля генерации: texts_a/texts_b (не gen_a/gen_b)
        gen_per_cycle  = sum(
            r.get("texts_a", r.get("gen_a", 0)) + r.get("texts_b", r.get("gen_b", 0))
            for r in log
        ) / n

    elif variant_type == "turbine":
        avg_lci_r      = sum(r.get("avg_lci_r", r.get("lci_r0", _PI)) for r in log) / n
        resonance_rate = sum(1 for r in log if r.get("n_resonant", 0) >= 3) / n
        kirchhoff_ok   = sum(1 for r in log if r.get("kirchhoff_dev", 99) < 0.5) / n
        gen_per_cycle  = sum(r.get("n_generated", 0) for r in log) / n

    elif variant_type == "roundabout":
        avg_lci_r      = sum(r.get("lci_r_end", _PI) for r in log) / n
        resonance_rate = sum(1 for r in log if abs(r.get("lci_r_end", 0) - _PI) < 0.5) / n
        kirchhoff_ok   = resonance_rate
        early          = sum(1 for r in log if r.get("exited_early", False)) / n
        gen_per_cycle  = sum(r.get("n_generated", 0) for r in log) / n
        # Добавить бонус за ранние выходы
        kirchhoff_ok   = (kirchhoff_ok + early) / 2

    elif variant_type == "bidir":
        avg_lci_r      = sum(r.get("lci_meet", _PI) for r in log) / n
        resonance_rate = sum(1 for r in log if r.get("resonant", False)) / n
        kirchhoff_ok   = resonance_rate
        gen_per_cycle  = sum(r.get("n_generated", 0) for r in log) / n

    elif variant_type == "multisalesman":
        avg_lci_r      = sum(r.get("avg_lci_all", _PI) for r in log) / n
        resonance_rate = sum(1 for r in log if r.get("resonant", False)) / n
        kirchhoff_ok   = sum(1 for r in log if r.get("kirchhoff_val", 99) and
                             abs(r.get("kirchhoff_val", 99) - _PI) < 0.5) / n
        gen_per_cycle  = sum(r.get("n_generated", 0) for r in log) / n

    elif variant_type == "nautilus":
        # nautilus_clover log: [{lci_r_final, lci_emb_final, n_resonant, kirchhoff, n_generated, rings}, ...]
        avg_lci_r      = sum(r.get("lci_r_final", _PI) for r in log) / n
        n_rings        = 4  # META + ABSTRACT + DYNAMIC + CONCRETE
        resonance_rate = sum(r.get("n_resonant", 0) / n_rings for r in log) / n
        kirchhoff_ok   = sum(1 for r in log if r.get("kirchhoff", False)) / n
        gen_per_cycle  = sum(r.get("n_generated", 0) for r in log) / n

    else:
        avg_lci_r, resonance_rate, kirchhoff_ok, gen_per_cycle = _PI, 0.0, 0.0, 0.0

    # Взвешенная оценка (score):
    #   lci_proximity: насколько avg_LCI_r близко к π (0..1)
    #   resonance:     доля резонансных циклов
    #   kirchhoff:     доля Kirchhoff-сбалансированных
    lci_proximity  = max(0.0, 1.0 - abs(avg_lci_r - _PI) / _PI)
    score = (0.40 * lci_proximity +
             0.35 * resonance_rate +
             0.25 * kirchhoff_ok)

    return {
        "avg_lci_r":      round(avg_lci_r,      3),
        "resonance_rate": round(resonance_rate,  3),
        "kirchhoff_ok":   round(kirchhoff_ok,    3),
        "gen_per_cycle":  round(gen_per_cycle,   1),
        "score":          round(score,           3),
    }


# ── Запуск одного варианта ────────────────────────────────────────────────────

def run_variant(variant: Dict, checkpoint: str, only_ids: Optional[List[int]]) -> Optional[Dict]:
    if only_ids and variant["id"] not in only_ids:
        return None

    vid = variant["id"]
    name = variant["name"]
    script = os.path.join(_ROOT, variant["script"])
    save_path = os.path.join(_ROOT, f"bench_v{vid}.pt")
    log_path  = os.path.join(_ROOT, f"bench_v{vid}_log.json")

    cmd = [
        sys.executable, script,
        "--checkpoint", checkpoint,
        "--save", save_path,
    ] + variant["args"].split()

    print(f"\n{'─' * 72}")
    print(f"  [{vid:2d}] {name}")
    print(f"       cmd: python {variant['script']} {variant['args']}")
    print(f"{'─' * 72}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=False, text=True,
            cwd=_ROOT, timeout=600
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"  [!] Ошибка (exit code {result.returncode})")
            return {"id": vid, "name": name, "error": True, "elapsed_s": round(elapsed, 1)}
    except subprocess.TimeoutExpired:
        print(f"  [!] Таймаут (>600s)")
        return {"id": vid, "name": name, "error": True, "elapsed_s": 600}
    except Exception as e:
        print(f"  [!] Исключение: {e}")
        return {"id": vid, "name": name, "error": True, "elapsed_s": 0}

    elapsed = time.perf_counter() - t0

    # Загрузить лог
    log_file = save_path.replace(".pt", "_log.json")
    log = []
    if os.path.exists(log_file):
        with open(log_file, encoding="utf-8") as f:
            log = json.load(f)

    metrics = extract_metrics(log, variant["type"])
    metrics["id"]        = vid
    metrics["name"]      = name
    metrics["elapsed_s"] = round(elapsed, 1)
    metrics["error"]     = False

    print(f"  → score={metrics['score']:.3f}  avg_LCI_r={metrics['avg_lci_r']:.3f}  "
          f"resonance={metrics['resonance_rate']:.0%}  kirchhoff={metrics['kirchhoff_ok']:.0%}  "
          f"gen/cycle={metrics['gen_per_cycle']:.0f}  t={elapsed:.1f}s")

    return metrics


# ── Финальная таблица ─────────────────────────────────────────────────────────

def print_table(results: List[Dict]):
    print(f"\n{'═' * 90}")
    print(f"  СРАВНИТЕЛЬНАЯ ТАБЛИЦА ВСЕХ ВАРИАНТОВ")
    print(f"{'═' * 90}")
    header = (f"  {'#':>2}  {'Вариант':<28}  {'Score':>6}  {'LCI_r':>6}  "
              f"{'Res%':>5}  {'Kirch%':>6}  {'Gen/c':>5}  {'t(s)':>6}")
    print(header)
    print(f"  {'─'*2}  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*5}  {'─'*6}")

    valid = [r for r in results if not r.get("error")]
    valid_sorted = sorted(valid, key=lambda r: r["score"], reverse=True)

    for rank, r in enumerate(valid_sorted, 1):
        marker = " ★" if rank == 1 else ("  " if rank > 3 else "  ")
        print(f"  {r['id']:>2}  {r['name']:<28}  {r['score']:>6.3f}  {r['avg_lci_r']:>6.3f}  "
              f"{r['resonance_rate']:>5.0%}  {r['kirchhoff_ok']:>6.0%}  "
              f"{r['gen_per_cycle']:>5.0f}  {r['elapsed_s']:>6.1f}{marker}")

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n  Ошибки: {', '.join(str(r['id']) for r in errors)}")

    if valid_sorted:
        best = valid_sorted[0]
        print(f"\n  ★ ЛУЧШИЙ: [{best['id']}] {best['name']}  score={best['score']:.3f}")
        print(f"\n  Рекомендации для дальнейшего развития:")
        for r in valid_sorted[:3]:
            print(f"    [{r['id']}] {r['name']}  — score={r['score']:.3f}")

    print(f"{'═' * 90}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Бенчмарк всех вариантов само-обучения HMoE")
    parser.add_argument("--checkpoint", type=str, default="hmoe_self_trained_v5.pt",
                        help="Базовый чекпоинт для всех вариантов")
    parser.add_argument("--variants",   type=str, default="",
                        help="Через запятую: ID вариантов для запуска (пусто = все)")
    args = parser.parse_args()

    only_ids = None
    if args.variants:
        only_ids = [int(x.strip()) for x in args.variants.split(",") if x.strip()]

    print(f"\n{'═' * 90}")
    print(f"  BENCH_ALL — Сравнительный бенчмарк вариантов само-обучения HMoE")
    print(f"{'═' * 90}")
    print(f"  Чекпоинт: {args.checkpoint}")
    print(f"  Режим:    --fast (минимальные прогоны для сравнения)")
    if only_ids:
        print(f"  Варианты: {only_ids}")
    else:
        print(f"  Варианты: все ({len(VARIANTS)})")

    results = []
    total_t = time.perf_counter()

    for variant in VARIANTS:
        r = run_variant(variant, args.checkpoint, only_ids)
        if r is not None:
            results.append(r)

    total_elapsed = time.perf_counter() - total_t

    print_table(results)
    print(f"\n  Общее время: {total_elapsed:.1f}s")

    # Сохранить результаты
    out_path = os.path.join(_ROOT, "bench_all_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Результаты: {out_path}")


if __name__ == "__main__":
    main()
