#!/usr/bin/env python3
"""
corpus_loader.py — Универсальный загрузчик корпуса для pro2.

Делает pro2 независимым: собирает тренировочные тексты из ВСЕХ
доступных источников без необходимости запрашивать другие репо в runtime.

Источники (по убыванию приоритета):
  1. data/svend4_corpus/_clones/data7/    ← теория K₀→K₁→K₂
  2. data/svend4_corpus/_clones/info1/    ← методология ⇑⇓↔
  3. data/svend4_corpus/_clones/meta/     ← Q6/hexagram система
  4. data/svend4_corpus/_clones/data2/    ← данные ЕТД Крюкова
  5. data/svend4_corpus/infosystems/      ← инфосистемы
  6. data/svend4_corpus/ai_agents/        ← AI агенты
  7. data/svend4_corpus/knowledge/        ← база знаний
  8. data/svend4_corpus/meta/             ← мета-документы

Адаптировано из info1/tools/knowledge_extractor.py:
  Вместо "извлечь паттерны из истории метрик" →
  "загрузить тексты из всех клонов как тренировочные примеры".

Usage:
  from corpus_loader import CorpusLoader
  loader = CorpusLoader()
  texts = loader.load(domains=["info1", "data7", "meta"])
  for text, meta in texts:
      # text = строка для обучения
      # meta = {"source": repo, "domain": str, "alpha": int, "path": str}

  # Или прямо в bidir_train.py:
  loader = CorpusLoader()
  corpus = loader.as_training_corpus(max_per_source=200)

Горизонтальные связи (↔):
  ↔ bidir_train.py — основной потребитель корпуса
  ↔ self_train_v2.py — использует corpus для domain triplet loss
  ↔ graph_health.py — граф строится из загруженных концептов
  ↔ nautilus_inference.py — корпус для RAG-ответов
"""

import os
import re
import json
from pathlib import Path
from typing import Iterator


_ROOT   = Path(__file__).parent
_CLONES = _ROOT / "data" / "svend4_corpus" / "_clones"
_CORPUS = _ROOT / "data" / "svend4_corpus"

# Домены (аналог DOMAINS в variant3.py)
_DOMAIN_KEYWORDS = {
    "GEO":    ["геология", "порода", "минерал", "кристалл", "земля", "geo", "rock", "mineral"],
    "HYDRO":  ["вода", "жидкость", "поток", "гидро", "water", "fluid", "flow", "hydro"],
    "PYRO":   ["огонь", "тепло", "энергия", "горение", "fire", "heat", "energy", "pyro"],
    "AERO":   ["воздух", "атмосфера", "газ", "ветер", "air", "gas", "wind", "aero"],
    "COSMO":  ["космос", "звезда", "вселенная", "орбита", "space", "star", "universe"],
    "NOOS":   ["знание", "концепт", "теория", "методология", "philosophy", "knowledge", "concept"],
    "METHOD": ["методология", "метод", "подход", "система", "method", "approach", "system"],
    "YIJING": ["гексаграмма", "и-цзин", "q6", "hexagram", "yijing", "trigram", "триграмма"],
}


def _guess_domain(text: str, path_str: str = "") -> str:
    """Угадывает домен по содержимому и пути файла."""
    combined = (text[:500] + " " + path_str).lower()
    scores = {d: 0 for d in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "NOOS"


def _guess_alpha(path: Path) -> int:
    """Угадывает уровень абстракции α по пути файла. Аналог info1 α-уровней."""
    p = str(path).lower()
    if any(x in p for x in ["философия", "philosophy", "онтология", "ontology"]):
        return 4
    if any(x in p for x in ["концепция", "concept", "теория", "theory"]):
        return 3
    if any(x in p for x in ["методология", "methodology", "принцип"]):
        return 2
    if any(x in p for x in ["методы", "methods", "patterns", "паттерн"]):
        return 1
    if any(x in p for x in ["readme", "overview", "summary"]):
        return 0
    if any(x in p for x in ["template", "шаблон", "пример"]):
        return -1
    if any(x in p for x in ["spec", "спецификация", "config"]):
        return -2
    if any(x in p for x in ["tools", "инструмент", "скрипт", "script"]):
        return -3
    if path.suffix in {".py", ".sh", ".json"}:
        return -4
    return 0


def _clean(text: str, max_len: int = 512) -> str:
    """Минимальная очистка: убираем HTML, frontmatter, лишние пробелы."""
    # YAML frontmatter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            text = text[end+3:]
    # HTML-теги
    text = re.sub(r"<[^>]+>", " ", text)
    # Множественные пробелы
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = text.strip()
    return text[:max_len] if len(text) > max_len else text


# ══════════════════════════════════════════════════════════════════════════════
class CorpusLoader:
    """
    Собирает тренировочные тексты из всех доступных источников.

    Аналог info1.KnowledgeExtractor — но вместо "извлечь паттерны из истории"
    делает "загрузить тексты из всех клонов".
    """

    SOURCES = {
        "data7": {
            "path": _CLONES / "data7",
            "priority": 1,
            "description": "Теория трансформации знаний K₀→K₁→K₂",
        },
        "info1": {
            "path": _CLONES / "info1",
            "priority": 2,
            "description": "Методология параллельного развития ⇑⇓↔",
        },
        "meta": {
            "path": _CORPUS / "meta",
            "priority": 3,
            "description": "Q6/Hexagram математическая система",
        },
        "data2": {
            "path": _CORPUS / "data2",
            "priority": 4,
            "description": "ЕТД Крюкова / Скарабей",
        },
        "infosystems": {
            "path": _CORPUS / "infosystems",
            "priority": 5,
            "description": "Информационные системы",
        },
        "ai_agents": {
            "path": _CORPUS / "ai_agents",
            "priority": 6,
            "description": "AI агенты",
        },
        "knowledge": {
            "path": _CORPUS / "knowledge",
            "priority": 7,
            "description": "База знаний",
        },
        "meta_corpus": {
            "path": _CORPUS / "meta",
            "priority": 8,
            "description": "Мета-документы",
        },
    }

    _TEXT_EXTS = {".md", ".txt", ".rst"}
    _CODE_EXTS = {".py", ".js", ".ts"}

    def __init__(self, include_code: bool = False):
        self.include_code = include_code

    def _iter_source(self, name: str, src: dict) -> Iterator[tuple[str, dict]]:
        """Итерирует файлы одного источника."""
        base: Path = src["path"]
        if not base.exists():
            return

        exts = self._TEXT_EXTS | (self._CODE_EXTS if self.include_code else set())

        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in exts:
                continue
            if any(p in str(path) for p in [".git", "__pycache__", "node_modules"]):
                continue
            try:
                raw  = path.read_text(encoding="utf-8", errors="ignore")
                text = _clean(raw)
                if len(text) < 30:
                    continue
                yield text, {
                    "source":   name,
                    "domain":   _guess_domain(text, str(path)),
                    "alpha":    _guess_alpha(path),
                    "path":     str(path.relative_to(_ROOT)),
                    "priority": src["priority"],
                }
            except Exception:
                continue

    def load(self,
             domains: list[str] | None = None,
             max_alpha: int = 4,
             min_alpha: int = -4) -> list[tuple[str, dict]]:
        """
        Загружает все доступные тексты.

        Args:
            domains:   список источников (None = все)
            max_alpha: максимальный уровень α (включительно)
            min_alpha: минимальный уровень α (включительно)

        Returns:
            список (text, metadata) отсортированный по приоритету источника
        """
        results = []
        sources = {k: v for k, v in self.SOURCES.items()
                   if domains is None or k in domains}
        for name, src in sorted(sources.items(), key=lambda x: x[1]["priority"]):
            for text, meta in self._iter_source(name, src):
                if min_alpha <= meta["alpha"] <= max_alpha:
                    results.append((text, meta))
        return results

    def as_training_corpus(self,
                           max_per_source: int = 300,
                           shuffle: bool = True) -> list[dict]:
        """
        Возвращает список {"text": str, "domain": str, "alpha": int}
        готовый для bidir_train.py / self_train_v2.py.

        Лимитирует max_per_source текстов из каждого источника.
        """
        import random
        by_source: dict[str, list] = {}
        for text, meta in self.load():
            src = meta["source"]
            if src not in by_source:
                by_source[src] = []
            by_source[src].append({
                "text":   text,
                "domain": meta["domain"],
                "alpha":  meta["alpha"],
                "source": src,
            })

        corpus = []
        for src, items in by_source.items():
            if len(items) > max_per_source:
                items = random.sample(items, max_per_source)
            corpus.extend(items)

        if shuffle:
            random.shuffle(corpus)
        return corpus

    def stats(self) -> dict:
        """Статистика по источникам (не загружает всё в память)."""
        result = {}
        for name, src in self.SOURCES.items():
            base: Path = src["path"]
            if not base.exists():
                result[name] = {"exists": False, "files": 0}
                continue
            count = sum(
                1 for p in base.rglob("*")
                if p.suffix.lower() in self._TEXT_EXTS
                and not any(x in str(p) for x in [".git", "__pycache__"])
            )
            result[name] = {
                "exists":      True,
                "files":       count,
                "description": src["description"],
                "priority":    src["priority"],
            }
        return result

    def availability_report(self) -> str:
        """Краткий отчёт: какие источники доступны."""
        lines = ["CORPUS AVAILABILITY — pro2", "=" * 50]
        total = 0
        for name, info in sorted(self.stats().items(), key=lambda x: x[1].get("priority", 99)):
            if info["exists"]:
                n = info["files"]
                total += n
                lines.append(f"  ✅ {name:<14} {n:>5} файлов — {info.get('description','')}")
            else:
                lines.append(f"  ❌ {name:<14}  н/д  — {self.SOURCES[name]['description']}")
        lines.append(f"  {'TOTAL':<14} {total:>5} файлов")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corpus Loader для pro2")
    parser.add_argument("--stats",  action="store_true", help="Показать статистику")
    parser.add_argument("--sample", type=int, default=0,
                        help="Показать N примеров текстов")
    parser.add_argument("--source", default=None,
                        help="Фильтр по источнику (data7, info1, meta, ...)")
    parser.add_argument("--json",   action="store_true", help="JSON-вывод")
    args = parser.parse_args()

    loader = CorpusLoader()

    if args.stats:
        print(loader.availability_report())

    if args.sample > 0:
        domains = [args.source] if args.source else None
        items   = loader.load(domains=domains)
        import random
        sample  = random.sample(items, min(args.sample, len(items)))
        if args.json:
            print(json.dumps([{"text": t[:200], "meta": m} for t, m in sample],
                              ensure_ascii=False, indent=2))
        else:
            for text, meta in sample:
                print(f"\n[{meta['source']}|{meta['domain']}|α={meta['alpha']}]")
                print(text[:200])
                print("---")
