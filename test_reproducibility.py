#!/usr/bin/env python3
"""
test_reproducibility.py — Тест воспроизводимости bench_all.

Запускает variant 5 (turbine LCI-loss λ=0.1) N раз,
собирает score и LCI, вычисляет std.

Usage:
  python test_reproducibility.py
  python test_reproducibility.py --runs 5 --variant 5
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
_PI   = math.pi

# Конфигурации из bench_all.py
VARIANT_CONFIGS = {
    5: {
        "name":   "turbine LCI-loss λ=0.1",
        "script": "figure8_turbine.py",
        "args":   "--fast --cycles 2 --lci-loss 0.1",
        "type":   "turbine",
    },
    8: {
        "name":   "bidir-turbine",
        "script": "bidir_turbine.py",
        "args":   "--fast --cycles 2",
        "type":   "bidir",
    },
}


def run_once(variant_id: int, checkpoint: str, run_idx: int) -> Dict:
    """Один прогон варианта. Возвращает метрики."""
    vc       = VARIANT_CONFIGS[variant_id]
    save_pt  = os.path.join(_ROOT, f"bench_v{variant_id}.pt")
    log_path = save_pt.replace(".pt", "_log.json")

    cmd = [
        sys.executable,
        os.path.join(_ROOT, vc["script"]),
        "--checkpoint", checkpoint,
        "--save", save_pt,
    ] + vc["args"].split()

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=_ROOT, timeout=300,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        return {"run": run_idx, "error": True, "elapsed_s": round(elapsed, 1)}

    # Читаем лог
    log = []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)

    if not log:
        return {"run": run_idx, "error": True, "elapsed_s": round(elapsed, 1)}

    n = len(log)
    r0 = log[0]
    vtype = vc["type"]

    if vtype == "turbine":
        avg_lci_r      = sum(r.get("avg_lci_r", _PI) for r in log) / n
        resonance_rate = sum(1 for r in log if r.get("resonant", False)) / n
        kirchhoff_ok   = sum(1 for r in log if r.get("kirchhoff_ok", False)) / n
        early          = sum(1 for r in log if r.get("exited_early", False)) / n
        kirchhoff_ok   = (kirchhoff_ok + early) / 2
        gen_per_cycle  = sum(r.get("n_generated", 0) for r in log) / n
    elif vtype == "bidir":
        avg_lci_r      = sum(r.get("lci_meet", _PI) for r in log) / n
        resonance_rate = sum(1 for r in log if r.get("resonant", False)) / n
        kirchhoff_ok   = resonance_rate
        gen_per_cycle  = sum(r.get("n_generated", 0) for r in log) / n
    else:
        avg_lci_r, resonance_rate, kirchhoff_ok, gen_per_cycle = _PI, 0, 0, 0

    lci_proximity = max(0.0, 1.0 - abs(avg_lci_r - _PI) / _PI)
    score = 0.40 * lci_proximity + 0.35 * resonance_rate + 0.25 * kirchhoff_ok

    return {
        "run":            run_idx,
        "avg_lci_r":      round(avg_lci_r, 4),
        "resonance_rate": round(resonance_rate, 4),
        "kirchhoff_ok":   round(kirchhoff_ok, 4),
        "gen_per_cycle":  round(gen_per_cycle, 1),
        "score":          round(score, 4),
        "elapsed_s":      round(elapsed, 1),
        "error":          False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",       type=int, default=5)
    parser.add_argument("--variant",    type=int, default=5, choices=[5, 8])
    parser.add_argument("--checkpoint", default="hmoe_self_trained_v5.pt")
    parser.add_argument("--output",     default="experiments/reproducibility_results.json")
    args = parser.parse_args()

    vc = VARIANT_CONFIGS[args.variant]
    print(f"{'═' * 60}")
    print(f"  ТЕСТ ВОСПРОИЗВОДИМОСТИ")
    print(f"{'═' * 60}")
    print(f"  Вариант    : [{args.variant}] {vc['name']}")
    print(f"  Прогонов   : {args.runs}")
    print(f"  Чекпоинт   : {args.checkpoint}")
    print()

    results = []
    for i in range(1, args.runs + 1):
        print(f"  [{i}/{args.runs}] ...", end="", flush=True)
        r = run_once(args.variant, args.checkpoint, i)
        results.append(r)
        if r.get("error"):
            print(f"  ОШИБКА  t={r['elapsed_s']}s")
        else:
            print(f"  score={r['score']:.4f}  LCI={r['avg_lci_r']:.4f}  t={r['elapsed_s']}s")

    # Анализ
    valid = [r for r in results if not r.get("error")]
    if valid:
        scores = [r["score"] for r in valid]
        lcis   = [r["avg_lci_r"] for r in valid]
        mean_s = sum(scores) / len(scores)
        mean_l = sum(lcis) / len(lcis)
        std_s  = (sum((x - mean_s)**2 for x in scores) / len(scores)) ** 0.5
        std_l  = (sum((x - mean_l)**2 for x in lcis)   / len(lcis))   ** 0.5

        print()
        print(f"{'─' * 60}")
        print(f"  ИТОГ ({len(valid)}/{args.runs} успешно)")
        print(f"  scores : {[f'{s:.4f}' for s in scores]}")
        print(f"  mean_score={mean_s:.4f}  std={std_s:.4f}")
        print(f"  LCI    : {[f'{l:.4f}' for l in lcis]}")
        print(f"  mean_LCI  ={mean_l:.4f}  std={std_l:.4f}")
        stable = "ДА" if std_s < 0.002 else "НЕТ"
        print(f"  Воспроизводим (std<0.002): {stable}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary = {
        "variant_id":   args.variant,
        "variant_name": vc["name"],
        "runs":         results,
        "mean_score":   round(mean_s, 4) if valid else None,
        "std_score":    round(std_s, 4)  if valid else None,
        "mean_lci":     round(mean_l, 4) if valid else None,
        "std_lci":      round(std_l, 4)  if valid else None,
        "stable":       std_s < 0.002    if valid else None,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Результаты: {args.output}")


if __name__ == "__main__":
    main()
