#!/usr/bin/env python3
"""
graph_health.py — Метрики здоровья для графа знаний pro2.

Адаптация info1/tools/health_metrics.py для pro2:
  info1 считает: markdown-документы + wikilinks (← →)
  pro2 считает: концепты Q6 + рёбра KnowledgeGraph

Метрики:
  CD  (Connectivity Density)    — плотность рёбер графа; цель: 10-25%
  VT  (Vertical Traceability)   — связность Q6-уровней (dim=0..5); цель: ≥50%
  CR  (Convergence Rate)        — скорость роста концептов по циклам; цель: 0.7-1.5
  DB  (Directional Balance)     — баланс ⇑⇓ vs ↔ рёбер; цель: <30%

Источник данных: graph_health_state.json (сохраняется bidir_train.py)
Или: передаётся объект KnowledgeGraph напрямую.

Usage:
  python graph_health.py                    # читает graph_health_state.json
  python graph_health.py --json             # JSON-вывод
  python graph_health.py --check            # exit 1 если метрики вне нормы
  python graph_health.py --save             # сохранить в graph_health_history.json

Источник методологии: info1/tools/health_metrics.py (адаптировано)
Уровень абстракции: α=-4 (исполняемый код)

Вертикальные связи (⇑):
  ⇑ PASSPORT.md (α=0) — метрики в разделе "Текущее состояние"
  ⇑ bidir_train.py — генерирует граф для измерения

Горизонтальные связи (↔):
  ↔ graph_self_improvement.py — использует метрики для рекомендаций
  ↔ corpus_loader.py — загружает корпус, который граф индексирует
"""

import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional


# ── пути ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_STATE_FILE   = _ROOT / "graph_health_state.json"
_HISTORY_FILE = _ROOT / "graph_health_history.json"

# ── пороги (аналог info1 thresholds) ──────────────────────────────────────────
THRESHOLDS = {
    "cd": {"min": 10.0, "max": 25.0, "unit": "%",
           "description": "Плотность рёбер графа"},
    "vt": {"min": 50.0, "max": 85.0, "unit": "%",
           "description": "Вертикальная трассируемость Q6-уровней"},
    "cr": {"min": 0.7,  "max": 1.5,  "unit":  "",
           "description": "Скорость роста концептов"},
    "db": {"min": 0.0,  "max": 30.0, "unit": "%",
           "description": "Дисбаланс ⇑⇓ vs ↔"},
}


# ══════════════════════════════════════════════════════════════════════════════
class GraphHealthCalculator:
    """
    Аналог info1.HealthMetricsCalculator — но для KnowledgeGraph.

    Принимает либо путь к graph_health_state.json,
    либо объект KnowledgeGraph из bidir_train.py напрямую.
    """

    def __init__(self, state_path: Optional[Path] = None):
        self.state_path = state_path or _STATE_FILE
        self.concepts: dict = {}   # name → {domain, depth, hex_idx, pagerank}
        self.edges: list = []      # [{src, dst, type, weight}]
        self.cycles: list = []     # [{cycle, n_concepts, n_edges}]
        self._loaded = False

    # ── загрузка ──────────────────────────────────────────────────────────────

    def load_from_file(self) -> bool:
        if not self.state_path.exists():
            return False
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.concepts = data.get("concepts", {})
        self.edges    = data.get("edges", [])
        self.cycles   = data.get("cycles", [])
        self._loaded  = True
        return True

    def load_from_graph(self, graph) -> None:
        """Принимает объект KnowledgeGraph из bidir_train.py напрямую."""
        for name, c in graph.concepts.items():
            self.concepts[name] = {
                "domain":   c.domain,
                "depth":    c.depth,
                "hex_idx":  c.hex_idx,
                "pagerank": c.pagerank,
            }
        for src, neighbours in graph.adj.items():
            for dst, attrs in neighbours.items():
                self.edges.append({
                    "src":    src,
                    "dst":    dst,
                    "type":   attrs.get("type", "related_to"),
                    "weight": attrs.get("weight", 1.0),
                })
        self._loaded = True

    # ── метрики ───────────────────────────────────────────────────────────────

    def cd(self) -> float:
        """
        CD = |E| / (|V| * (|V|-1)) * 100%
        Аналог info1.CD = ссылок / (документов * (документов-1))
        """
        n = len(self.concepts)
        if n < 2:
            return 0.0
        return len(self.edges) / (n * (n - 1)) * 100.0

    def vt(self) -> float:
        """
        VT = % рёбер соединяющих разные Q6-уровни (dim-глубину).

        Q6-уровень концепта = число бит=1 в его hex_idx (popcount).
        Рёбро «вертикальное» если |level(src) - level(dst)| >= 1.
        Аналог info1.VT = % связей между разными α-уровнями.
        """
        if not self.edges:
            return 0.0

        def level(name: str) -> int:
            hex_idx = self.concepts.get(name, {}).get("hex_idx", -1)
            if hex_idx < 0:
                return 3  # по умолчанию средний
            return bin(hex_idx).count("1")  # popcount = «высота» в Q6

        vertical = sum(
            1 for e in self.edges
            if abs(level(e["src"]) - level(e["dst"])) >= 1
        )
        return vertical / len(self.edges) * 100.0

    def cr(self) -> float:
        """
        CR = среднее (концептов_в_цикле_N / концептов_в_цикле_N-1).
        Аналог info1.CR = средний прирост документов по циклам.
        Если циклов <2 — используем текущий размер как прокси.
        """
        if len(self.cycles) < 2:
            # прокси: 1 концепт за "цикл" → CR=1.0
            return 1.0
        rates = []
        for i in range(1, len(self.cycles)):
            prev = self.cycles[i-1].get("n_concepts", 1)
            curr = self.cycles[i].get("n_concepts", 1)
            if prev > 0:
                rates.append(curr / prev)
        return sum(rates) / len(rates) if rates else 1.0

    def db(self) -> float:
        """
        DB = |horizontal_edges - vertical_edges| / total_edges * 100%
        Horizontal = рёбра внутри одного домена (↔)
        Vertical   = рёбра между доменами (⇑⇓)
        Аналог info1.DB = дисбаланс ↔ vs ⇑⇓ связей.
        """
        if not self.edges:
            return 0.0
        domains = {n: d.get("domain", "") for n, d in self.concepts.items()}
        horiz = sum(
            1 for e in self.edges
            if domains.get(e["src"]) == domains.get(e["dst"])
        )
        vert = len(self.edges) - horiz
        return abs(horiz - vert) / len(self.edges) * 100.0

    def compute_all(self) -> dict:
        return {
            "timestamp":  datetime.now().isoformat(),
            "n_concepts": len(self.concepts),
            "n_edges":    len(self.edges),
            "n_cycles":   len(self.cycles),
            "CD":  round(self.cd(), 2),
            "VT":  round(self.vt(), 2),
            "CR":  round(self.cr(), 2),
            "DB":  round(self.db(), 2),
        }

    # ── диагностика ───────────────────────────────────────────────────────────

    def diagnose(self, metrics: dict) -> list[dict]:
        """
        Аналог info1.SelfImprovementAnalyzer.generate_recommendations().
        Возвращает список {metric, status, message, priority}.
        """
        issues = []
        for key, thr in THRESHOLDS.items():
            val = metrics.get(key.upper(), 0.0)
            lo, hi = thr["min"], thr["max"]
            unit = thr["unit"]
            if val < lo:
                issues.append({
                    "metric":   key.upper(),
                    "value":    val,
                    "status":   "LOW",
                    "priority": "HIGH",
                    "message":  f"{key.upper()}={val}{unit} < {lo}{unit} — {thr['description']}",
                })
            elif val > hi:
                issues.append({
                    "metric":   key.upper(),
                    "value":    val,
                    "status":   "HIGH",
                    "priority": "HIGH",
                    "message":  f"{key.upper()}={val}{unit} > {hi}{unit} — {thr['description']}",
                })
        return issues

    # ── история ───────────────────────────────────────────────────────────────

    def save_to_history(self, metrics: dict) -> None:
        history = []
        if _HISTORY_FILE.exists():
            history = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
        history.append(metrics)
        _HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── отчёт ─────────────────────────────────────────────────────────────────

    @staticmethod
    def render_report(metrics: dict, issues: list) -> str:
        def _status(key: str) -> str:
            thr = THRESHOLDS.get(key.lower(), {})
            val = metrics.get(key, 0.0)
            if val < thr.get("min", 0) or val > thr.get("max", 9999):
                return "⚠️ "
            return "✅"

        lines = [
            "=" * 66,
            "GRAPH HEALTH METRICS — pro2/bidir_train.py KnowledgeGraph",
            "=" * 66,
            f"  Концептов: {metrics['n_concepts']}   "
            f"Рёбер: {metrics['n_edges']}   "
            f"Циклов: {metrics['n_cycles']}",
            f"  {metrics['timestamp']}",
            "",
            f"  CD: {metrics['CD']:6.2f}%  {_status('CD')}  "
            f"(цель: 10–25%)   — {THRESHOLDS['cd']['description']}",
            f"  VT: {metrics['VT']:6.2f}%  {_status('VT')}  "
            f"(цель: ≥50%)     — {THRESHOLDS['vt']['description']}",
            f"  CR: {metrics['CR']:6.2f}   {_status('CR')}  "
            f"(цель: 0.7–1.5)  — {THRESHOLDS['cr']['description']}",
            f"  DB: {metrics['DB']:6.2f}%  {_status('DB')}  "
            f"(цель: <30%)     — {THRESHOLDS['db']['description']}",
        ]
        if issues:
            lines += ["", "ПРОБЛЕМЫ:"]
            for iss in issues:
                lines.append(f"  [{iss['priority']}] {iss['message']}")
        else:
            lines += ["", "✅ Все метрики в норме"]
        lines.append("=" * 66)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Graph Health Metrics for pro2")
    parser.add_argument("--state",  default=str(_STATE_FILE),
                        help="Путь к graph_health_state.json")
    parser.add_argument("--json",   action="store_true", help="JSON-вывод")
    parser.add_argument("--check",  action="store_true",
                        help="Exit 1 если метрики вне нормы")
    parser.add_argument("--save",   action="store_true",
                        help="Сохранить в graph_health_history.json")
    args = parser.parse_args()

    calc = GraphHealthCalculator(Path(args.state))
    if not calc.load_from_file():
        print(f"❌ Файл не найден: {args.state}")
        print("   Запустите bidir_train.py --save-state чтобы создать его.")
        raise SystemExit(1)

    metrics = calc.compute_all()
    issues  = calc.diagnose(metrics)

    if args.save:
        calc.save_to_history(metrics)
        print(f"💾 Сохранено в {_HISTORY_FILE}")

    if args.json:
        print(json.dumps({"metrics": metrics, "issues": issues},
                         ensure_ascii=False, indent=2))
    else:
        print(calc.render_report(metrics, issues))

    if args.check and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
