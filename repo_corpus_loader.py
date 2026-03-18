#!/usr/bin/env python3
"""
repo_corpus_loader.py — Загрузчик ВСЕХ файлов самого репозитория pro2 в 7 кластеров.

Разбивает файлы самого репо (не _clones) на 7 кластеров по типу и домену:

  Кластер 1 «Theory»   (NOOS, α=+4): MD-документы с теорией
  Кластер 2 «Models»   (PYRO, α=+2): Python-модели (models/*.py, geometry/*.py)
  Кластер 3 «Training» (AERO, α= 0): utils_v12→v52, training/*.py
  Кластер 4 «Benchmarks» (GEO, α=−2): JSON-результаты бенчмарков
  Кластер 5 «Scripts»  (HYDRO, α=−4): scripts/*.py, yijing_transformer/scripts/*.py
  Кластер 6 «Portal»   (COSMO, α=+2): nautilus/*.py, adapters, passports, portal docs
  Кластер 7 «Self»     (METHOD, α= 0): корневые .py, логи обучения, отчёты

Используется в train_e2_clusters.py для многократного применения E2-схемы.

Usage:
  from repo_corpus_loader import RepoCorpusLoader
  loader = RepoCorpusLoader()
  cluster = loader.get_cluster("Theory")
  print(loader.report())
"""

import os
import re
import json
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).parent


# ── Кластеры ──────────────────────────────────────────────────────────────────

CLUSTER_DEFS = {
    "Theory": {
        "domain":      "NOOS",
        "alpha":       4,
        "e2_phase":    5,        # начинать с PhiloLevel (α=+4)
        "description": "Теоретические MD-документы: онтология, концепции, анализ",
        "globs": [
            "*.md",
            "yijing_transformer/*.md",
        ],
        "exclude_patterns": ["README", "PASSPORT", "PORTAL", "_clones", ".git",
                             ".pytest_cache"],
    },
    "Models": {
        "domain":      "PYRO",
        "alpha":       2,
        "e2_phase":    4,        # TheoryLevel (α=+2)
        "description": "Python-модели: variant3, hierarchical_e2, nautilus_yijing, geometry/*",
        "globs": [
            "yijing_transformer/models/*.py",
            "yijing_transformer/models/geometry/*.py",
        ],
        "exclude_patterns": ["__pycache__", "_clones"],
    },
    "Training": {
        "domain":      "AERO",
        "alpha":       0,
        "e2_phase":    3,        # MethodLevel (α=0)
        "description": "Утилиты обучения: utils_v12→v52, training/*.py, data_utils/*.py",
        "globs": [
            "yijing_transformer/training/*.py",
            "yijing_transformer/data_utils/*.py",
            "yijing_transformer/inference/*.py",
        ],
        "exclude_patterns": ["__pycache__", "_clones"],
    },
    "Benchmarks": {
        "domain":      "GEO",
        "alpha":       -2,
        "e2_phase":    2,        # CoreLevel (α=−2)
        "description": "JSON-результаты бенчмарков v53→v69, логи, ablation",
        "globs": [
            "yijing_transformer/*.json",
            "*.json",
        ],
        "exclude_patterns": ["_clones", ".git", "manifest.json", "nautilus.json",
                             "checkpoint"],
    },
    "Scripts": {
        "domain":      "HYDRO",
        "alpha":       -4,
        "e2_phase":    1,        # GlyphLevel (α=−4)
        "description": "Эксперименты и скрипты: yijing_transformer/scripts/*.py, scripts/*.py",
        "globs": [
            "yijing_transformer/scripts/*.py",
            "scripts/*.py",
            "yijing_transformer/notebooks/*.py",
        ],
        "exclude_patterns": ["__pycache__", "_clones"],
    },
    "Portal": {
        "domain":      "COSMO",
        "alpha":       2,
        "e2_phase":    4,        # TheoryLevel (α=+2)
        "description": "Интеграционный портал: nautilus/*.py, adapters, PASSPORT, PORTAL-PROTOCOL",
        "globs": [
            "nautilus/*.py",
            "nautilus/adapters/*.py",
            "passports/*.json",
            "PASSPORT.md",
            "PORTAL-PROTOCOL.md",
            "portal.py",
        ],
        "exclude_patterns": ["__pycache__", "_clones"],
    },
    "Self": {
        "domain":      "METHOD",
        "alpha":       0,
        "e2_phase":    3,        # MethodLevel (α=0)
        "description": "Самознание: корневые .py, логи обучения, отчёты, config",
        "globs": [
            "*.py",
            "yijing_transformer/config/*.py",
            "REPORT*.md",
            "PLAN*.md",
            "*_log.json",
        ],
        "exclude_patterns": ["__pycache__", "_clones", "train_e2_clusters.py",
                             "repo_corpus_loader.py"],
    },
}


# ── Утилиты очистки текста ────────────────────────────────────────────────────

def _clean(text: str, max_len: int = 512) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            text = text[end + 3:]
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = text.strip()
    return text[:max_len]


def _file_to_text(path: Path) -> str | None:
    """Читает файл и возвращает очищенный текст (None если слишком короткий)."""
    try:
        if path.suffix == ".json":
            raw = path.read_text(encoding="utf-8", errors="ignore")
            # Для JSON: краткое представление ключей/значений
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    # Первые 10 ключей и значения
                    lines = []
                    for k, v in list(data.items())[:10]:
                        if isinstance(v, (str, int, float)):
                            lines.append(f"{k}: {v}")
                        elif isinstance(v, list) and v:
                            lines.append(f"{k}: [{len(v)} items]")
                    text = path.name + "\n" + "\n".join(lines)
                elif isinstance(data, list) and data:
                    text = path.name + f"\n[{len(data)} records]"
                    if isinstance(data[0], dict):
                        for k in list(data[0].keys())[:5]:
                            text += f"\n  key: {k}"
                else:
                    text = path.name
            except Exception:
                text = raw[:512]
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")

        text = _clean(text)
        return text if len(text) >= 20 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
class RepoCorpusLoader:
    """
    Загружает файлы самого репозитория pro2 в 7 тематических кластеров.

    Аналог corpus_loader.CorpusLoader — но для ВНУТРЕННИХ файлов проекта,
    не для внешних клонов.  Применяется в train_e2_clusters.py для
    многократного прогона E2-схемы по разным кластерам.
    """

    def __init__(self, root: Path = _ROOT):
        self.root = root

    def _collect_cluster(self, name: str, defn: dict) -> list[dict]:
        """Собирает файлы одного кластера."""
        seen: set[Path] = set()
        items: list[dict] = []

        for pattern in defn["globs"]:
            for path in sorted(self.root.glob(pattern)):
                if path in seen or not path.is_file():
                    continue
                # Исключения
                path_str = str(path)
                if any(ex in path_str for ex in defn["exclude_patterns"]):
                    continue
                if any(p in path_str for p in ["__pycache__", ".git", "node_modules"]):
                    continue

                text = _file_to_text(path)
                if text is None:
                    continue

                seen.add(path)
                items.append({
                    "text":    text,
                    "domain":  defn["domain"],
                    "alpha":   defn["alpha"],
                    "cluster": name,
                    "path":    str(path.relative_to(self.root)),
                    "e2_phase": defn["e2_phase"],
                })

        return items

    def get_cluster(self, name: str) -> list[dict]:
        """Возвращает тексты одного кластера."""
        if name not in CLUSTER_DEFS:
            raise ValueError(f"Неизвестный кластер: {name}. Доступны: {list(CLUSTER_DEFS)}")
        return self._collect_cluster(name, CLUSTER_DEFS[name])

    def get_all_clusters(self) -> dict[str, list[dict]]:
        """Возвращает все кластеры как dict {name: [items]}."""
        result = {}
        for name, defn in CLUSTER_DEFS.items():
            result[name] = self._collect_cluster(name, defn)
        return result

    def as_flat_corpus(self, clusters: list[str] | None = None) -> list[dict]:
        """Плоский список всех текстов из указанных (или всех) кластеров."""
        names = clusters or list(CLUSTER_DEFS.keys())
        items = []
        for name in names:
            items.extend(self.get_cluster(name))
        return items

    def stats(self) -> dict:
        """Статистика по кластерам (без загрузки текстов в память)."""
        result = {}
        for name, defn in CLUSTER_DEFS.items():
            count = 0
            for pattern in defn["globs"]:
                for path in self.root.glob(pattern):
                    path_str = str(path)
                    if any(ex in path_str for ex in defn["exclude_patterns"]):
                        continue
                    if any(p in path_str for p in ["__pycache__", ".git"]):
                        continue
                    if path.is_file():
                        count += 1
            result[name] = {
                "files":       count,
                "domain":      defn["domain"],
                "alpha":       defn["alpha"],
                "e2_phase":    defn["e2_phase"],
                "description": defn["description"],
            }
        return result

    def report(self) -> str:
        """Отчёт о доступных кластерах."""
        lines = ["REPO CORPUS CLUSTERS — pro2", "=" * 60]
        total = 0
        for name, info in self.stats().items():
            n = info["files"]
            total += n
            lines.append(
                f"  {name:<12} {n:>4} файлов  "
                f"[{info['domain']:<7} α={info['alpha']:+d}  фаза={info['e2_phase']}]  "
                f"{info['description']}"
            )
        lines.append(f"  {'TOTAL':<12} {total:>4} файлов")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Repo Corpus Loader для pro2")
    parser.add_argument("--stats",   action="store_true", help="Показать статистику")
    parser.add_argument("--cluster", default="", help="Показать тексты из кластера")
    parser.add_argument("--n",       type=int, default=3, help="Кол-во примеров")
    args = parser.parse_args()

    loader = RepoCorpusLoader()

    if args.stats or not args.cluster:
        print(loader.report())

    if args.cluster:
        items = loader.get_cluster(args.cluster)
        print(f"\nКластер «{args.cluster}»: {len(items)} текстов")
        for item in items[:args.n]:
            print(f"\n  [{item['domain']}|α={item['alpha']}] {item['path']}")
            print(f"  {item['text'][:150]}")
            print("  ---")
