#!/usr/bin/env python3
"""
pipeline.py — автоматический трёхфазный curriculum-пайплайн.

Фаза 1: N × nautilus-4agent (step_scale=0.4, 8 циклов каждый)
          — разнообразное обучение по кольцам, накопление LCI
Фаза 2: turbine-LCI-loss (8 циклов)
          — стабилизация и Kirchhoff-балансировка
Фаза 3: оценка через bench_all на топ-вариантах

Оптимальные параметры (по экспериментам):
  nautilus_passes = 2     (3-й проход даёт насыщение/деградацию)
  step_scale      = 0.4   (100 шагов/цикл избыточно)
  turbine_cycles  = 8
  turbine_spe     = 8     (steps_per_expert)

Usage:
  python pipeline.py --checkpoint hmoe_self_trained_v5.pt
  python pipeline.py --checkpoint hmoe_self_trained_v5.pt --passes 2 --fast
  python pipeline.py --checkpoint hmoe_self_trained_v5.pt --passes 3  # для исследования
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

# Prevent thread contention between numpy/BLAS and PyTorch (fixes ~87-min hang).
os.environ.setdefault("OMP_NUM_THREADS", "1")

_ROOT = os.path.dirname(os.path.abspath(__file__))

_PI = math.pi


# ── Запуск подпроцесса с логированием ─────────────────────────────────────────

def run_phase(cmd: List[str], phase_name: str) -> subprocess.CompletedProcess:
    """Запустить команду, выводить в реальном времени, вернуть результат."""
    print(f"\n{'─' * 72}")
    print(f"  ФАЗА: {phase_name}")
    print(f"  Команда: {' '.join(cmd)}")
    print(f"{'─' * 72}")
    t0 = time.perf_counter()

    result = subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=False,   # вывод прямо в консоль
    )

    elapsed = time.perf_counter() - t0
    status = "✓ OK" if result.returncode == 0 else f"✗ ОШИБКА (код={result.returncode})"
    print(f"\n  {status}  [{phase_name}]  t={elapsed:.1f}s")

    if result.returncode != 0:
        print(f"  [!] Фаза завершилась с ошибкой. Прерываем пайплайн.")
        sys.exit(result.returncode)

    return result


# ── Чтение avg_LCI из лога ────────────────────────────────────────────────────

def read_avg_lci(log_path: str) -> float:
    """Извлечь avg_LCI из JSON-лога (nautilus или turbine формат)."""
    if not os.path.exists(log_path):
        return 0.0
    try:
        with open(log_path) as f:
            log = json.load(f)
        if not log:
            return 0.0
        r0 = log[0]
        if "avg_lci_all" in r0:           # nautilus_4agent
            return sum(r.get("avg_lci_all", 0) for r in log) / len(log)
        elif "avg_lci_r" in r0:           # turbine / figure8_turbine
            return sum(r.get("avg_lci_r", 0) for r in log) / len(log)
        elif "avg_LCI_r" in r0:           # старый формат
            return sum(r.get("avg_LCI_r", 0) for r in log) / len(log)
        elif "lci_r_final" in r0:         # nautilus_clover
            return sum(r.get("lci_r_final", 0) for r in log) / len(log)
        elif "avg_lci" in r0:             # multi_salesman / bidir
            return sum(r.get("avg_lci", 0) for r in log) / len(log)
    except Exception as e:
        print(f"  [warn] read_avg_lci({log_path}): {e}")
    return 0.0


# ── Главный пайплайн ──────────────────────────────────────────────────────────

def run_pipeline(
    start_checkpoint: str,
    n_nautilus_passes: int,
    nautilus_cycles: int,
    nautilus_step_scale: float,
    turbine_cycles: int,
    turbine_spe: int,
    turbine_lci_loss: float,
    fast: bool,
    output_dir: str,
    adaptive_lr: bool = True,
    reset_rag_pass: int = 3,
    lr_threshold: float = 2.8,
) -> Dict:
    """
    Запустить полный curriculum-пайплайн и вернуть итоговые метрики.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_id   = f"pipeline_{int(time.time())}"
    history  = []          # [{phase, checkpoint, avg_lci}]

    if not os.path.isfile(start_checkpoint):
        raise FileNotFoundError(
            f"start_checkpoint не найден: {start_checkpoint!r}\n"
            "Убедитесь, что файл существует перед запуском пайплайна."
        )
    current_ckpt = start_checkpoint
    print(f"\n{'═' * 72}")
    print(f"  CURRICULUM PIPELINE")
    print(f"{'═' * 72}")
    print(f"  Старт             : {start_checkpoint}")
    print(f"  Nautilus проходов : {n_nautilus_passes}  (по {nautilus_cycles} цикл × {nautilus_step_scale:.2f}×)")
    print(f"  Turbine финал     : {turbine_cycles} цикл, {turbine_spe} шагов, LCI-loss={turbine_lci_loss}")
    print(f"  Fast-mode         : {fast}")
    print(f"  Адаптивный LR     : {adaptive_lr}  (снижает lr при LCI>3.0)")
    print(f"  RAG reset pass    : {reset_rag_pass}  (bent seeds с прохода {reset_rag_pass})")
    print(f"  Сохранение в      : {output_dir}/")

    # ── Фаза 1: N проходов nautilus-4agent ────────────────────────────────────
    for pass_i in range(1, n_nautilus_passes + 1):
        out_ckpt = os.path.join(output_dir, f"{run_id}_n4a_pass{pass_i}.pt")
        out_log  = out_ckpt.replace(".pt", "_log.json")

        cmd = [
            sys.executable, os.path.join(_ROOT, "nautilus_4agent.py"),
            "--checkpoint", current_ckpt,
            "--save", out_ckpt,
        ]
        if fast:
            cmd += ["--fast"]
        else:
            cmd += ["--cycles", str(nautilus_cycles),
                    "--step-scale", str(nautilus_step_scale)]

        # RAG reset: с прохода reset_rag_pass используем bent seeds (математически
        # оптимальные архетипы, cosine diversity 33% лучше обычных seed).
        # Устраняет saturation из-за diversity collapse в поздних проходах.
        if pass_i >= reset_rag_pass:
            cmd += ["--bent-seeds"]
            print(f"  [RAG reset] Проход {pass_i}: используем bent seeds (meta_q6)")

        # Адаптивный LR: снижаем lr при LCI > 2.8 начиная со 2-го прохода.
        # Порог снижен 3.0 → 2.8: устраняет saturation на 3-м проходе
        # (гипотеза: lr=1e-5 вызывает forgetting когда модель уже хорошая).
        if adaptive_lr and pass_i > 1 and history:
            prev_lci = history[-1].get("avg_lci", 0.0)
            if prev_lci > lr_threshold and not fast:
                cmd += ["--lr", "5e-6"]
                print(f"  [adaptive LR] LCI={prev_lci:.3f} > {lr_threshold} → lr=5e-6")

        run_phase(cmd, f"Nautilus-4agent проход {pass_i}/{n_nautilus_passes}")

        avg_lci = read_avg_lci(out_log)
        history.append({"phase": f"nautilus_pass_{pass_i}",
                         "checkpoint": out_ckpt, "avg_lci": avg_lci})
        print(f"  → avg_LCI после прохода {pass_i}: {avg_lci:.3f}")
        current_ckpt = out_ckpt

    # ── Фаза 2: turbine-LCI-loss ──────────────────────────────────────────────
    out_ckpt_t = os.path.join(output_dir, f"{run_id}_turbine.pt")
    out_log_t  = out_ckpt_t.replace(".pt", "_log.json")

    cmd_t = [
        sys.executable, os.path.join(_ROOT, "figure8_turbine.py"),
        "--checkpoint", current_ckpt,
        "--save", out_ckpt_t,
        "--lci-loss", str(turbine_lci_loss),
    ]
    if fast:
        cmd_t += ["--fast"]
    else:
        cmd_t += ["--cycles", str(turbine_cycles),
                   "--steps_per_expert", str(turbine_spe)]

    run_phase(cmd_t, "Turbine-LCI-loss (стабилизация)")
    avg_lci_t = read_avg_lci(out_log_t)
    history.append({"phase": "turbine_lci",
                     "checkpoint": out_ckpt_t, "avg_lci": avg_lci_t})
    current_ckpt = out_ckpt_t

    # ── Фаза 3: bench на финальном чекпоинте ─────────────────────────────────
    cmd_b = [
        sys.executable, os.path.join(_ROOT, "bench_all.py"),
        "--checkpoint", current_ckpt,
        "--variants", "5,8,3,6,7,9,10,13,14",
    ]
    run_phase(cmd_b, "Финальный бенчмарк")

    # ── Итог ──────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  ИТОГ ПАЙПЛАЙНА")
    print(f"{'═' * 72}")
    for step in history:
        gap = abs(step["avg_lci"] - _PI)
        bar = "█" * int((1 - gap / _PI) * 20)
        print(f"  {step['phase']:25}  LCI={step['avg_lci']:.3f}  Δπ={gap:+.3f}  {bar}")
    print(f"  Финальный чекпоинт: {current_ckpt}")

    result = {
        "run_id":     run_id,
        "start":      start_checkpoint,
        "final":      current_ckpt,
        "history":    history,
        "n_passes":   n_nautilus_passes,
    }
    result_path = os.path.join(output_dir, f"{run_id}_summary.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Сохранено: {result_path}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum pipeline: N×nautilus-4agent → turbine-LCI-loss → bench"
    )
    parser.add_argument("--checkpoint",    type=str, required=True,
                        help="Стартовый чекпоинт")
    parser.add_argument("--passes",        type=int, default=2,
                        help="Число проходов nautilus-4agent (рекоменд. 2)")
    parser.add_argument("--nautilus-cycles", type=int, default=8,
                        dest="nautilus_cycles")
    parser.add_argument("--step-scale",    type=float, default=0.4,
                        dest="step_scale")
    parser.add_argument("--turbine-cycles", type=int, default=8,
                        dest="turbine_cycles")
    parser.add_argument("--turbine-spe",   type=int, default=8,
                        dest="turbine_spe",
                        help="Turbine steps_per_expert")
    parser.add_argument("--lci-loss",      type=float, default=0.1,
                        dest="lci_loss")
    parser.add_argument("--fast",          action="store_true",
                        help="Быстрый тест (2 цикла, 0.3× шаг)")
    parser.add_argument("--output-dir",    type=str, default="pipeline_runs",
                        dest="output_dir")
    parser.add_argument("--no-adaptive-lr", action="store_true",
                        dest="no_adaptive_lr",
                        help="Отключить адаптивный LR (по умолчанию включён)")
    parser.add_argument("--reset-rag-pass", type=int, default=3,
                        dest="reset_rag_pass",
                        help="С какого прохода сбрасывать RAG через bent seeds (default=3)")
    parser.add_argument("--lr-threshold",  type=float, default=2.8,
                        dest="lr_threshold",
                        help="LCI-порог для снижения lr до 5e-6 (default=2.8, было 3.0)")
    args = parser.parse_args()

    run_pipeline(
        start_checkpoint    = args.checkpoint,
        n_nautilus_passes   = args.passes,
        nautilus_cycles     = args.nautilus_cycles,
        nautilus_step_scale = args.step_scale,
        turbine_cycles      = args.turbine_cycles,
        turbine_spe         = args.turbine_spe,
        turbine_lci_loss    = args.lci_loss,
        fast                = args.fast,
        output_dir          = args.output_dir,
        adaptive_lr         = not args.no_adaptive_lr,
        reset_rag_pass      = args.reset_rag_pass,
        lr_threshold        = args.lr_threshold,
    )


if __name__ == "__main__":
    main()
