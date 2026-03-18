#!/usr/bin/env python3
"""
q6_graph_updater.py — Обновление KnowledgeGraph из Q6-результатов HierarchicalE2.

Мост между e2_inference.py и graph_health.py:
  E2Inference.embed(text) → Q6-координаты концептов
      ↓
  q6_graph_updater.build_graph(corpus)
      ↓
  graph_health_state.json  →  graph_health.py  → метрики CD/VT/CR/DB

Принцип построения рёбер:
  Два концепта связаны если Хэмминг(Q6_a, Q6_b) <= edge_threshold (по умолч. 2).
  Тип ребра:
    «same_hex»   — одинаковая гексаграмма (Хэмминг=0)
    «neighbor»   — Хэмминг=1 (соседние вершины гиперкуба)
    «close»      — Хэмминг=2

  Дополнительно строятся вертикальные рёбра:
    «alpha_link» — концепты из разных α-уровней (ext α vs repo α)
    «domain_link»— концепты одного домена (GEO, NOOS, ...)

Usage:
  python q6_graph_updater.py                     # обновить граф и метрики
  python q6_graph_updater.py --n 100             # использовать 100 концептов
  python q6_graph_updater.py --threshold 1       # только Хэмминг ≤ 1
  python q6_graph_updater.py --health            # только показать метрики
  python q6_graph_updater.py --evolve            # добавить данные из q6_evolution/

Горизонтальные связи (↔):
  ↔ e2_inference.py    — источник Q6-координат
  ↔ graph_health.py    — потребитель graph_health_state.json
  ↔ e2_concept_evolution.py — эволюция Q6 по снапшотам
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

STATE_FILE = _ROOT / "graph_health_state.json"


def _hamming(a: list, b: list) -> int:
    return sum(x != y for x, y in zip(a, b))


def _popcount(q6: list) -> int:
    return sum(q6)


_DOMAINS = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]


def _q6_to_domain(q6: list) -> str:
    for i in reversed(range(6)):
        if q6[i]:
            return _DOMAINS[i]
    return "GEO"


# ══════════════════════════════════════════════════════════════════════════════

def build_q6_graph(
    max_concepts: int = 200,
    edge_threshold: int = 2,
    include_evolution: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Строит граф концептов из Q6-эмбеддингов E2Inference.

    Returns:
        graph_state dict совместимый с GraphHealthCalculator.load_from_file()
    """
    from e2_inference import E2Inference

    if verbose:
        print("  Загружаю E2 модель...")
    engine = E2Inference(load_corpus=True)

    if verbose:
        print(f"  Корпус: {len(engine._corpus)} текстов")

    # ── Собираем концепты ──────────────────────────────────────────────────
    items = engine._corpus[:max_concepts]

    # Добавляем концепты из снапшотов эволюции
    if include_evolution:
        evo_dir = _ROOT / "q6_evolution"
        if evo_dir.exists():
            for snap_file in sorted(evo_dir.glob("*.json")):
                if snap_file.name == "index.json":
                    continue
                try:
                    snap = json.loads(snap_file.read_text(encoding="utf-8"))
                    for concept, data in snap.get("concepts", {}).items():
                        items.append({
                            "text":    concept,
                            "q6":      data["q6"],
                            "hex_idx": data["hex_idx"],
                            "source":  f"evo/{snap['label']}",
                            "domain":  _q6_to_domain(data["q6"]),
                            "alpha":   0,
                        })
                except Exception:
                    pass
            if verbose:
                print(f"  + Снапшоты эволюции: {len(items) - max_concepts} концептов")

    if verbose:
        print(f"  Строю граф из {len(items)} концептов...")

    # ── Концепты (вершины) ────────────────────────────────────────────────
    concepts_dict: dict[str, dict] = {}
    for i, item in enumerate(items):
        key = item["text"][:60]
        if key in concepts_dict:
            continue
        q6 = item["q6"]
        concepts_dict[key] = {
            "domain":   item.get("domain", _q6_to_domain(q6)),
            "depth":    _popcount(q6),          # 0..6 = «высота» в Q6
            "hex_idx":  item.get("hex_idx", 0),
            "alpha":    item.get("alpha", 0),
            "source":   item.get("source", "?"),
            "q6":       q6,
        }

    concept_keys = list(concepts_dict.keys())
    n = len(concept_keys)

    # ── Рёбра ────────────────────────────────────────────────────────────
    edges: list[dict] = []
    edge_set: set = set()

    def _add_edge(src: str, dst: str, etype: str, weight: float) -> None:
        key = (min(src, dst), max(src, dst))
        if key not in edge_set:
            edge_set.add(key)
            edges.append({"src": src, "dst": dst, "type": etype, "weight": weight})

    # Рёбра по Q6-близости (O(n²) но ограничено max_concepts)
    q6_list = [concepts_dict[k]["q6"] for k in concept_keys]

    for i in range(n):
        for j in range(i + 1, n):
            dist = _hamming(q6_list[i], q6_list[j])
            if dist <= edge_threshold:
                if dist == 0:
                    etype, weight = "same_hex", 1.0
                elif dist == 1:
                    etype, weight = "neighbor", 0.8
                else:
                    etype, weight = "close", 0.5
                _add_edge(concept_keys[i], concept_keys[j], etype, weight)

    # Дополнительные рёбра по домену (слабые связи внутри домена)
    by_domain: dict[str, list] = defaultdict(list)
    for k, c in concepts_dict.items():
        by_domain[c["domain"]].append(k)

    for domain, members in by_domain.items():
        for i in range(min(len(members), 10)):
            for j in range(i + 1, min(len(members), 10)):
                _add_edge(members[i], members[j], "domain_link", 0.3)

    # Рёбра по α-уровням (вертикальные)
    by_alpha: dict[int, list] = defaultdict(list)
    for k, c in concepts_dict.items():
        by_alpha[c["alpha"]].append(k)

    alpha_levels = sorted(by_alpha.keys())
    for lvl_idx in range(len(alpha_levels) - 1):
        lo = by_alpha[alpha_levels[lvl_idx]]
        hi = by_alpha[alpha_levels[lvl_idx + 1]]
        for k1 in lo[:5]:
            for k2 in hi[:5]:
                _add_edge(k1, k2, "alpha_link", 0.4)

    if verbose:
        print(f"  Вершин: {n}  Рёбер: {len(edges)}")

    # ── Псевдо-циклы (из логов обучения) ────────────────────────────────
    cycles = []
    for log_file in [
        _ROOT / "train_e2_clusters_log.json",
        _ROOT / "train_e2_joint_log.json",
        _ROOT / "train_e2_log.json",
    ]:
        if log_file.exists():
            try:
                log = json.loads(log_file.read_text(encoding="utf-8"))
                for i, entry in enumerate(log):
                    if entry:
                        cycles.append({
                            "cycle":      i,
                            "n_concepts": n,
                            "n_edges":    len(edges),
                            "source":     log_file.stem,
                            "ppl_after":  entry.get("ppl_after", 0),
                        })
            except Exception:
                pass

    # Если нет циклов — добавляем фиктивный
    if not cycles:
        cycles = [{"cycle": 0, "n_concepts": n, "n_edges": len(edges)}]

    # ── Убираем q6 из concepts (graph_health не ожидает) ─────────────────
    concepts_clean = {}
    for k, v in concepts_dict.items():
        concepts_clean[k] = {kk: vv for kk, vv in v.items() if kk != "q6"}

    return {
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "concepts":   concepts_clean,
        "edges":      edges,
        "cycles":     cycles,
        "meta": {
            "source":         "q6_graph_updater",
            "edge_threshold": edge_threshold,
            "checkpoint":     engine.checkpoint_used,
        },
    }


def update_graph_health(
    max_concepts: int = 200,
    edge_threshold: int = 2,
    include_evolution: bool = False,
    verbose: bool = True,
) -> dict:
    """Обновляет graph_health_state.json и возвращает метрики."""
    from graph_health import GraphHealthCalculator

    state = build_q6_graph(max_concepts, edge_threshold, include_evolution, verbose)

    # Сохраняем state
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"  💾 {STATE_FILE.name} обновлён")

    # Вычисляем метрики через GraphHealthCalculator
    calc = GraphHealthCalculator(STATE_FILE)
    calc.load_from_file()
    metrics = calc.compute_all()
    issues  = calc.diagnose(metrics)
    calc.save_to_history(metrics)

    return {"metrics": metrics, "issues": issues, "state": state}


# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Q6 Graph Updater → graph_health.py")
    parser.add_argument("--n",         type=int, default=200,
                        help="Максимум концептов")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Хэмминг-порог для ребра (0-3)")
    parser.add_argument("--health",    action="store_true",
                        help="Только показать текущие метрики (без перестройки)")
    parser.add_argument("--evolve",    action="store_true",
                        help="Включить снапшоты эволюции в граф")
    parser.add_argument("--json",      action="store_true")
    args = parser.parse_args()

    if args.health and STATE_FILE.exists():
        from graph_health import GraphHealthCalculator
        calc = GraphHealthCalculator()
        calc.load_from_file()
        metrics = calc.compute_all()
        issues  = calc.diagnose(metrics)
        if args.json:
            print(json.dumps({"metrics": metrics, "issues": issues},
                             ensure_ascii=False, indent=2))
        else:
            print(GraphHealthCalculator.render_report(metrics, issues))
        return

    print("\n  Q6 GRAPH UPDATER — Обновление KnowledgeGraph из Q6-эмбеддингов")
    print("  " + "─" * 60)

    result = update_graph_health(
        max_concepts=args.n,
        edge_threshold=args.threshold,
        include_evolution=args.evolve,
        verbose=True,
    )

    m = result["metrics"]
    issues = result["issues"]

    if args.json:
        print(json.dumps({"metrics": m, "n_issues": len(issues)},
                         ensure_ascii=False, indent=2))
    else:
        from graph_health import GraphHealthCalculator
        print("\n" + GraphHealthCalculator.render_report(m, issues))

        if not issues:
            print("  ✅ Все метрики в норме")
        else:
            for issue in issues:
                print(f"  ⚠️  {issue['message']}")

        print(f"\n  Граф: {m['n_concepts']} вершин  {m['n_edges']} рёбер  "
              f"{m['n_cycles']} циклов")


if __name__ == "__main__":
    main()
