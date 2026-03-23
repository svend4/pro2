#!/usr/bin/env python3
"""
bench_stability.py — Проверка воспроизводимости bench_all.

Запускает bench_all N раз на одном чекпоинте и считает mean/std score.
Если std > 0.002 — результаты нестабильны, A/B-выводы ненадёжны.

Usage:
  python bench_stability.py --checkpoint hmoe_self_trained_v5.pt
  python bench_stability.py --checkpoint hmoe_self_trained_v5.pt --runs 3 --variants 5,8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List

_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_once(checkpoint: str, variants: str) -> List[Dict]:
    """Запустить bench_all один раз и вернуть список результатов."""
    cmd = [
        sys.executable, os.path.join(_ROOT, "bench_all.py"),
        "--checkpoint", checkpoint,
    ]
    if variants:
        cmd += ["--variants", variants]

    result = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)

    out_path = os.path.join(_ROOT, "bench_all_results.json")
    if not os.path.exists(out_path):
        return []
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Stability benchmark: N × bench_all → mean/std score"
    )
    parser.add_argument("--checkpoint", type=str, default="hmoe_self_trained_v5.pt")
    parser.add_argument("--runs",       type=int, default=5,
                        help="Число запусков (рекоменд. 5)")
    parser.add_argument("--variants",   type=str, default="5,6,8",
                        help="Варианты bench_all (быстрые: 5,6,8)")
    args = parser.parse_args()

    print(f"\n{'═' * 72}")
    print(f"  BENCH STABILITY  ({args.runs} запусков)")
    print(f"  Чекпоинт: {args.checkpoint}")
    print(f"  Варианты:  {args.variants}")
    print(f"{'═' * 72}")

    all_scores: Dict[str, List[float]] = {}   # name → [score1, score2, ...]
    all_lcis:   Dict[str, List[float]] = {}

    total_t = time.perf_counter()
    for run_i in range(1, args.runs + 1):
        print(f"\n  ── Запуск {run_i}/{args.runs} ──")
        t0 = time.perf_counter()
        results = run_once(args.checkpoint, args.variants)
        elapsed = time.perf_counter() - t0

        for r in results:
            if r.get("error"):
                continue
            name = r["name"]
            all_scores.setdefault(name, []).append(r["score"])
            all_lcis.setdefault(name, []).append(r["avg_lci_r"])

        brief = ", ".join(
            f"{r['name'][:12]}={r['score']:.3f}"
            for r in results if not r.get("error")
        )
        print(f"  t={elapsed:.1f}s  {brief}")

    total_elapsed = time.perf_counter() - total_t

    # ── Итоговая таблица ──────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  ИТОГ  (n={args.runs}, общее время={total_elapsed:.1f}s)")
    print(f"{'═' * 72}")
    print(f"  {'Вариант':<28}  {'mean':>6}  {'std':>6}  {'cv%':>5}  {'stable':>7}  {'LCI_mean':>9}")
    print(f"  {'─'*28}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*9}")

    _PI = math.pi
    any_unstable = False
    for name in sorted(all_scores.keys()):
        sc = all_scores[name]
        lc = all_lcis.get(name, [])
        mean_s  = sum(sc) / len(sc)
        std_s   = math.sqrt(sum((x - mean_s)**2 for x in sc) / max(1, len(sc) - 1)) if len(sc) > 1 else 0.0
        cv      = 100 * std_s / max(mean_s, 1e-9)
        stable  = "OK" if std_s <= 0.002 else "НЕСТАБ!"
        if std_s > 0.002:
            any_unstable = True
        mean_lci = sum(lc) / len(lc) if lc else 0.0
        print(f"  {name:<28}  {mean_s:>6.3f}  {std_s:>6.4f}  {cv:>5.1f}  {stable:>7}  {mean_lci:>9.3f}")

    print(f"{'═' * 72}")
    if any_unstable:
        print("  ⚠  Есть нестабильные варианты (std > 0.002). A/B-выводы ненадёжны.")
    else:
        print("  ✓  Все варианты стабильны (std ≤ 0.002).")

    # Сохранить сырые данные
    out = {
        "checkpoint": args.checkpoint,
        "runs":       args.runs,
        "variants":   args.variants,
        "scores":     all_scores,
        "lcis":       all_lcis,
    }
    out_path = os.path.join(_ROOT, "bench_stability_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Сырые данные: {out_path}")


if __name__ == "__main__":
    main()
