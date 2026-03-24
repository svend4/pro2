#!/usr/bin/env python3
"""
e2_concept_evolution.py — Трекинг изменения Q6-координат концептов во времени.

Позволяет понять КАК модель «учится» — как Q6-позиции ключевых концептов
смещаются по мере обучения через кластеры и циклы.

Функциональность:
  1. SNAPSHOT  — сохранить текущие Q6-координаты набора концептов
  2. COMPARE   — сравнить два снапшота (до/после обучения)
  3. TIMELINE  — построить траекторию движения концепта в Q6-пространстве
  4. STABILITY — найти концепты с наиболее стабильными/нестабильными Q6
  5. CLUSTER_DRIFT — выяснить какие кластеры обучения сильнее всего сдвигают Q6

Интеграция с обучением:
  - Вызывается в train_e2_clusters.py после каждого кластера
  - Читает train_e2_clusters_log.json для корреляции с PPL

Usage:
  python e2_concept_evolution.py --snapshot "before_cycle2"
  python e2_concept_evolution.py --compare before_cycle1 after_cycle1
  python e2_concept_evolution.py --timeline "гексаграмма"
  python e2_concept_evolution.py --stability
  python e2_concept_evolution.py --auto     # полный автоматический анализ
  python e2_concept_evolution.py --track-training  # снапшот перед+после обучения

Горизонтальные связи (↔):
  ↔ train_e2_clusters.py  — основной потребитель (снапшоты после кластеров)
  ↔ train_e2_joint.py     — joint training снапшоты
  ↔ e2_inference.py       — использует E2Inference для embed
  ↔ graph_health.py       — передаёт Q6-данные в KnowledgeGraph
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from yijing_transformer.constants import HEX_NAMES
_HEX_NAMES = HEX_NAMES

EVOLUTION_DIR  = _ROOT / "q6_evolution"
EVOLUTION_DIR.mkdir(exist_ok=True)
EVOLUTION_INDEX = EVOLUTION_DIR / "index.json"

# Эталонные концепты для отслеживания
TRACKED_CONCEPTS = [
    # Теоретические
    "гексаграмма", "Q6 пространство", "архетипы", "трансформация знаний",
    "философия", "онтология",
    # Модели
    "HierarchicalE2", "Variant3GPT", "NautilusPortal", "ArchetypalInterlingua",
    "NautilusHierarchy", "ConvergenceBridge",
    # Тренировка
    "corpus loader", "bidir training", "self training", "benchmark",
    # Домены
    "кристалл", "вода", "огонь", "космос", "знание", "метод",
    # Коды
    "utils_v52", "train_e2", "checkpoint",
]

def _hex_name(idx: int) -> str:
    return _HEX_NAMES[idx] if 0 <= idx < 64 else f"#{idx}"


def _hamming(a: list, b: list) -> int:
    return sum(x != y for x, y in zip(a, b))


def _load_index() -> dict:
    if EVOLUTION_INDEX.exists():
        return json.loads(EVOLUTION_INDEX.read_text(encoding="utf-8"))
    return {"snapshots": []}


def _save_index(idx: dict) -> None:
    EVOLUTION_INDEX.write_text(
        json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка модели
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(checkpoint: Optional[str] = None):
    """Загружает HierarchicalE2 из лучшего доступного checkpoint."""
    import torch
    from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config

    cfg = E2Config(
        vocab_size=256, d_model=128, block_size=32,
        n_core=4, n_heads=4, dropout=0.0,
        hamming_lambda=0.15, uncertainty_budget=0.25, ffn_mult=4,
        n_archetypes=64, il_use_ternary=True,
        nautilus_warmup=200, nautilus_mode="sequential", nautilus_chambers=None,
        conv_window=4, conv_stride=2, grammar_rows=8, grammar_cols=8,
    )
    model = HierarchicalE2(cfg)

    candidates = [
        checkpoint,
        str(_ROOT / "checkpoint_e2_improved.pt"),
        str(_ROOT / "checkpoint_e2_joint.pt"),
        str(_ROOT / "checkpoint_e2_clusters.pt"),
        str(_ROOT / "checkpoint_e2.pt"),
    ]
    used = None
    for ckpt in candidates:
        if ckpt and Path(ckpt).exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            used = ckpt
            break

    model.eval()
    return model, used


# ══════════════════════════════════════════════════════════════════════════════
# Снапшоты
# ══════════════════════════════════════════════════════════════════════════════

def take_snapshot(
    label: str,
    concepts: list[str] = TRACKED_CONCEPTS,
    checkpoint: Optional[str] = None,
    metadata: dict = None,
) -> dict:
    """
    Делает снапшот Q6-координат для набора концептов.
    Сохраняет в q6_evolution/{label}.json и обновляет индекс.
    """
    model, ckpt_used = _load_model(checkpoint)
    print(f"  Снапшот «{label}»  ({len(concepts)} концептов)  "
          f"checkpoint: {Path(ckpt_used).name if ckpt_used else '?'}")

    q6_map: dict[str, dict] = {}
    for concept in concepts:
        r = model.embed_text(concept)
        q6_map[concept] = {
            "q6":      r["q6"],
            "hex_idx": r["hex_idx"],
            "hex_name": _hex_name(r["hex_idx"]),
        }

    snapshot = {
        "label":      label,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": ckpt_used,
        "concepts":   q6_map,
        **(metadata or {}),
    }

    snap_path = EVOLUTION_DIR / f"{label}.json"
    snap_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Обновляем индекс
    idx = _load_index()
    # Удалить если уже есть
    idx["snapshots"] = [s for s in idx["snapshots"] if s["label"] != label]
    idx["snapshots"].append({
        "label":     label,
        "timestamp": snapshot["timestamp"],
        "checkpoint": ckpt_used,
        "n_concepts": len(q6_map),
    })
    _save_index(idx)
    print(f"  Сохранён: {snap_path.relative_to(_ROOT)}")
    return snapshot


def load_snapshot(label: str) -> dict:
    """Загружает снапшот по метке."""
    snap_path = EVOLUTION_DIR / f"{label}.json"
    if not snap_path.exists():
        raise FileNotFoundError(f"Снапшот «{label}» не найден в {EVOLUTION_DIR}")
    return json.loads(snap_path.read_text(encoding="utf-8"))


def list_snapshots() -> list[dict]:
    """Список всех снапшотов."""
    return _load_index().get("snapshots", [])


# ══════════════════════════════════════════════════════════════════════════════
# Сравнение снапшотов
# ══════════════════════════════════════════════════════════════════════════════

def compare_snapshots(label_a: str, label_b: str) -> dict:
    """
    Сравнивает два снапшота: показывает изменения Q6-координат.
    Возвращает список концептов с их дрейфом.
    """
    a = load_snapshot(label_a)
    b = load_snapshot(label_b)

    concepts_a = a["concepts"]
    concepts_b = b["concepts"]

    diffs = []
    for concept in set(concepts_a) & set(concepts_b):
        q6a = concepts_a[concept]["q6"]
        q6b = concepts_b[concept]["q6"]
        dist = _hamming(q6a, q6b)
        moved = dist > 0
        diffs.append({
            "concept":   concept,
            "q6_before": q6a,
            "q6_after":  q6b,
            "hamming":   dist,
            "hex_before": concepts_a[concept]["hex_idx"],
            "hex_after":  concepts_b[concept]["hex_idx"],
            "moved":     moved,
        })

    diffs.sort(key=lambda x: -x["hamming"])
    moved_count  = sum(1 for d in diffs if d["moved"])
    avg_drift    = sum(d["hamming"] for d in diffs) / max(len(diffs), 1)
    stable_count = len(diffs) - moved_count

    return {
        "snapshot_a":    label_a,
        "snapshot_b":    label_b,
        "timestamp_a":   a.get("timestamp"),
        "timestamp_b":   b.get("timestamp"),
        "total":         len(diffs),
        "moved":         moved_count,
        "stable":        stable_count,
        "avg_drift":     round(avg_drift, 3),
        "stability_pct": round(stable_count / max(len(diffs), 1) * 100, 1),
        "diffs":         diffs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Временная шкала (timeline)
# ══════════════════════════════════════════════════════════════════════════════

def concept_timeline(concept: str) -> dict:
    """
    Показывает траекторию Q6-координат одного концепта по всем снапшотам.
    """
    snaps = list_snapshots()
    timeline = []
    for snap_info in sorted(snaps, key=lambda s: s["timestamp"]):
        snap = load_snapshot(snap_info["label"])
        if concept in snap["concepts"]:
            entry = snap["concepts"][concept]
            timeline.append({
                "label":    snap_info["label"],
                "timestamp": snap_info["timestamp"],
                "q6":       entry["q6"],
                "hex_idx":  entry["hex_idx"],
                "hex_name": entry["hex_name"],
                "q6_str":   "".join(map(str, entry["q6"])),
            })

    # Вычислить шаги дрейфа
    for i in range(1, len(timeline)):
        timeline[i]["step_drift"] = _hamming(
            timeline[i-1]["q6"], timeline[i]["q6"]
        )
    if timeline:
        timeline[0]["step_drift"] = 0

    total_drift = sum(t.get("step_drift", 0) for t in timeline)
    # Итоговый дрейф (первый → последний)
    final_drift = _hamming(timeline[0]["q6"], timeline[-1]["q6"]) if len(timeline) > 1 else 0

    return {
        "concept":      concept,
        "n_snapshots":  len(timeline),
        "total_drift":  total_drift,
        "final_drift":  final_drift,
        "trajectory":   timeline,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Стабильность
# ══════════════════════════════════════════════════════════════════════════════

def stability_report(concepts: list[str] = TRACKED_CONCEPTS) -> dict:
    """
    Анализирует стабильность Q6-координат каждого концепта по всем снапшотам.
    Стабильные концепты = хорошо выученные.
    Нестабильные = модель всё ещё учится / конфликтующее знание.
    """
    snaps = list_snapshots()
    if len(snaps) < 2:
        return {"error": "Нужно минимум 2 снапшота для анализа стабильности"}

    by_concept: dict[str, list] = {c: [] for c in concepts}

    for snap_info in sorted(snaps, key=lambda s: s["timestamp"]):
        snap = load_snapshot(snap_info["label"])
        for concept in concepts:
            if concept in snap["concepts"]:
                by_concept[concept].append(snap["concepts"][concept]["q6"])

    results = []
    for concept, q6_history in by_concept.items():
        if len(q6_history) < 2:
            continue
        # Считаем среднее Хэмминг-расстояние между последовательными снапшотами
        drifts = [_hamming(q6_history[i], q6_history[i+1])
                  for i in range(len(q6_history)-1)]
        avg_drift = sum(drifts) / len(drifts)
        total_drift = _hamming(q6_history[0], q6_history[-1])

        # Самая частая Q6-координата = "домашняя"
        from collections import Counter
        q6_counts = Counter(tuple(q) for q in q6_history)
        home_q6 = list(q6_counts.most_common(1)[0][0])
        home_hex = sum(b << i for i, b in enumerate(home_q6))
        stability = 1 - avg_drift / 6  # 0..1

        results.append({
            "concept":     concept,
            "n_snapshots": len(q6_history),
            "avg_drift":   round(avg_drift, 3),
            "total_drift": total_drift,
            "stability":   round(stability, 3),
            "home_q6":     home_q6,
            "home_hex":    home_hex,
            "home_hex_name": _hex_name(home_hex),
            "last_q6":     q6_history[-1],
            "last_hex":    sum(b << i for i, b in enumerate(q6_history[-1])),
        })

    results.sort(key=lambda x: -x["stability"])

    return {
        "n_snapshots":    len(snaps),
        "n_concepts":     len(results),
        "avg_stability":  round(sum(r["stability"] for r in results) /
                               max(len(results), 1), 3),
        "most_stable":    [r for r in results if r["stability"] >= 0.8][:5],
        "least_stable":   [r for r in results[::-1] if r["stability"] < 0.8][:5],
        "all":            results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Автоматический анализ
# ══════════════════════════════════════════════════════════════════════════════

def auto_analysis() -> None:
    """Полный автоматический анализ эволюции Q6."""
    snaps = list_snapshots()
    print(f"\n  Снапшотов: {len(snaps)}")
    for s in snaps:
        print(f"    {s['label']:<25}  {s['timestamp']}  ({s['n_concepts']} концептов)")

    if len(snaps) < 2:
        print("\n  Нужно минимум 2 снапшота. Создайте их через:")
        print("    python e2_concept_evolution.py --snapshot before_training")
        print("    ... обучение ...")
        print("    python e2_concept_evolution.py --snapshot after_training")
        return

    # Сравниваем первый и последний
    first = snaps[0]["label"]
    last  = snaps[-1]["label"]
    print(f"\n  Сравниваю: «{first}» → «{last}»")
    comp = compare_snapshots(first, last)

    print(f"\n  {'─'*60}")
    print(f"  СРАВНЕНИЕ Q6-ПРОСТРАНСТВА")
    print(f"  {'─'*60}")
    print(f"  Концептов проверено: {comp['total']}")
    print(f"  Сдвинулись:  {comp['moved']} ({100-comp['stability_pct']:.0f}%)")
    print(f"  Стабильны:   {comp['stable']} ({comp['stability_pct']:.0f}%)")
    print(f"  Средний дрейф: {comp['avg_drift']:.2f} бит")

    print(f"\n  Топ-10 по дрейфу:")
    print(f"  {'Концепт':<28} {'Δ':>3}  {'До':^12}  {'После':^12}")
    print("  " + "─" * 60)
    for d in comp["diffs"][:10]:
        q6b = "".join(map(str, d["q6_before"]))
        q6a = "".join(map(str, d["q6_after"]))
        marker = "→" if d["moved"] else "="
        print(f"  {d['concept']:<28} {d['hamming']:>3} {marker}"
              f"  [{q6b}]#{d['hex_before']:<2}  [{q6a}]#{d['hex_after']:<2}")

    # Стабильность
    print(f"\n  {'─'*60}")
    print(f"  СТАБИЛЬНОСТЬ КОНЦЕПТОВ")
    print(f"  {'─'*60}")
    stab = stability_report()
    if "error" not in stab:
        print(f"  Средняя стабильность: {stab['avg_stability']*100:.1f}%")
        print(f"\n  Наиболее стабильные (выучены):")
        for r in stab["most_stable"]:
            q6s = "".join(map(str, r["home_q6"]))
            print(f"    {r['concept']:<28}  стаб={r['stability']*100:.0f}%  "
                  f"Q6=[{q6s}] «{r['home_hex_name']}»")
        print(f"\n  Наименее стабильные (ещё учатся):")
        for r in stab["least_stable"]:
            q6s = "".join(map(str, r["home_q6"]))
            print(f"    {r['concept']:<28}  стаб={r['stability']*100:.0f}%  "
                  f"дрейф={r['avg_drift']:.2f}")


def track_training(
    before_label: str,
    after_label: str,
    checkpoint_before: Optional[str] = None,
    checkpoint_after:  Optional[str] = None,
) -> None:
    """Снапшоты до и после обучения + немедленное сравнение."""
    print(f"\n  Снапшот ДО: «{before_label}»")
    snap_a = take_snapshot(before_label, checkpoint=checkpoint_before)

    yield_marker = "Нажмите Enter после обучения..."
    print(f"\n  {yield_marker}")
    # В автоматическом режиме просто берём текущий checkpoint
    snap_b = take_snapshot(after_label, checkpoint=checkpoint_after)

    comp = compare_snapshots(before_label, after_label)
    print(f"\n  Результат: сдвинулось {comp['moved']}/{comp['total']} концептов  "
          f"ср.дрейф={comp['avg_drift']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="E2 Concept Evolution Tracker")
    parser.add_argument("--snapshot",   default="", help="Сделать снапшот с меткой")
    parser.add_argument("--compare",    nargs=2, metavar=("A", "B"),
                        help="Сравнить два снапшота")
    parser.add_argument("--timeline",   default="", help="Траектория одного концепта")
    parser.add_argument("--stability",  action="store_true", help="Отчёт стабильности")
    parser.add_argument("--auto",       action="store_true", help="Полный авто-анализ")
    parser.add_argument("--list",       action="store_true", help="Список снапшотов")
    parser.add_argument("--checkpoint", default="", help="Путь к .pt файлу")
    parser.add_argument("--concepts",   nargs="+", default=None,
                        help="Список концептов (по умолчанию — стандартный набор)")
    parser.add_argument("--json",       action="store_true")
    args = parser.parse_args()

    concepts = args.concepts or TRACKED_CONCEPTS

    if args.list:
        snaps = list_snapshots()
        print(f"\n  Снапшотов: {len(snaps)}")
        for s in snaps:
            print(f"  {s['label']:<25}  {s['timestamp']}  "
                  f"ckpt={Path(s.get('checkpoint') or '?').name}")
        return

    if args.snapshot:
        snap = take_snapshot(args.snapshot, concepts=concepts,
                             checkpoint=args.checkpoint or None)
        if args.json:
            print(json.dumps({k: v for k, v in snap.items() if k != "concepts"},
                             ensure_ascii=False, indent=2))
        else:
            print(f"\n  Q6 снапшот «{args.snapshot}»  ({len(snap['concepts'])} концептов)")
            print(f"  {'Концепт':<28}  Q6       Гексаграмма")
            print("  " + "─" * 50)
            for concept, data in list(snap["concepts"].items())[:15]:
                q6s = "".join(map(str, data["q6"]))
                print(f"  {concept:<28}  [{q6s}]  #{data['hex_idx']:<3} {data['hex_name']}")
        return

    if args.compare:
        comp = compare_snapshots(args.compare[0], args.compare[1])
        if args.json:
            print(json.dumps(comp, ensure_ascii=False, indent=2))
        else:
            print(f"\n  Сравнение: «{comp['snapshot_a']}» → «{comp['snapshot_b']}»")
            print(f"  Стабильность: {comp['stability_pct']}%  "
                  f"ср.дрейф={comp['avg_drift']:.2f}")
            print(f"\n  {'Концепт':<28} {'Δ':>3}  {'До':^12}  {'После':^12}")
            print("  " + "─" * 56)
            for d in comp["diffs"][:15]:
                q6b = "".join(map(str, d["q6_before"]))
                q6a = "".join(map(str, d["q6_after"]))
                print(f"  {d['concept']:<28} {d['hamming']:>3}"
                      f"  [{q6b}]#{d['hex_before']:<2}  [{q6a}]#{d['hex_after']:<2}")
        return

    if args.timeline:
        tl = concept_timeline(args.timeline)
        if args.json:
            print(json.dumps(tl, ensure_ascii=False, indent=2))
        else:
            print(f"\n  Траектория «{tl['concept']}»  ({tl['n_snapshots']} точек)")
            print(f"  Итоговый дрейф: {tl['final_drift']} бит  "
                  f"суммарный: {tl['total_drift']}")
            for p in tl["trajectory"]:
                print(f"  {p['label']:<25}  [{p['q6_str']}]  "
                      f"#{p['hex_idx']:<3} {p['hex_name']:<20}  "
                      f"Δ={p.get('step_drift', 0)}")
        return

    if args.stability:
        stab = stability_report(concepts)
        if args.json:
            print(json.dumps(stab, ensure_ascii=False, indent=2))
        else:
            if "error" in stab:
                print(f"  ⚠️  {stab['error']}")
            else:
                print(f"\n  Стабильность концептов ({stab['n_snapshots']} снапшотов):")
                print(f"  Средняя: {stab['avg_stability']*100:.1f}%")
                print(f"\n  {'Концепт':<28}  {'Стаб.%':>7}  {'Дрейф':>6}  Гексаграмма")
                print("  " + "─" * 60)
                for r in stab["all"]:
                    q6s = "".join(map(str, r["home_q6"]))
                    print(f"  {r['concept']:<28}  {r['stability']*100:>6.1f}%"
                          f"  {r['avg_drift']:>6.2f}  [{q6s}] «{r['home_hex_name']}»")
        return

    if args.auto or not any([args.snapshot, args.compare, args.timeline,
                              args.stability, args.list]):
        auto_analysis()


if __name__ == "__main__":
    import json
    main()
