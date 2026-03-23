#!/usr/bin/env python3
"""
e2_inference.py — Полный inference API для HierarchicalE2.

Возможности:
  1. embed(text)          → Q6-координата + гексаграмма + домен
  2. find_similar(text, n)→ топ-N похожих концептов по Q6 (Хэмминг-расстояние)
  3. concept_map()        → Q6-карта всего корпуса (64 ячейки гиперкуба)
  4. domain_report()      → какие домены активны в Q6-пространстве
  5. generate(prefix)     → продолжение текста (авторегрессивно)
  6. cross_repo_align()   → сравнение Q6 между внешним и внутренним корпусами

Загрузка модели (приоритет):
  checkpoint_e2_joint.pt > checkpoint_e2_clusters.pt > checkpoint_e2.pt

Usage:
  python e2_inference.py --embed "кристалл"
  python e2_inference.py --similar "гексаграмма" --n 5
  python e2_inference.py --map                    # Q6-карта
  python e2_inference.py --domain-report
  python e2_inference.py --generate "трансформация"
  python e2_inference.py --cross-align
  python e2_inference.py --query "nautilus portal"   # всё сразу

Горизонтальные связи (↔):
  ↔ hierarchical_e2.py     — модель
  ↔ corpus_loader.py       — внешние тексты
  ↔ repo_corpus_loader.py  — внутренние кластеры
  ↔ nautilus_inference.py  — Q6-embed через Variant3GPT (аналог)
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Optional
from collections import defaultdict

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from yijing_transformer.constants import HEX_NAMES
_HEX_NAMES = HEX_NAMES

# ── Имена гексаграмм (по Вильгельму) ─────────────────────────────────────────
_DOMAINS = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]


def _hex_name(idx: int) -> str:
    return _HEX_NAMES[idx] if 0 <= idx < 64 else f"#{idx}"


def _q6_to_domain(q6: list[int]) -> str:
    for i in reversed(range(6)):
        if q6[i]:
            return _DOMAINS[i]
    return _DOMAINS[0]


def _hamming(a: list[int], b: list[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


# ══════════════════════════════════════════════════════════════════════════════
class E2Inference:
    """
    Inference engine для HierarchicalE2.
    Автоматически загружает лучший доступный checkpoint.
    """

    def __init__(self, checkpoint: Optional[str] = None, load_corpus: bool = True):
        import torch
        from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config

        self._torch = torch

        cfg = E2Config(
            vocab_size=256, d_model=128, block_size=32,
            n_core=4, n_heads=4, dropout=0.0,
            hamming_lambda=0.15, uncertainty_budget=0.25, ffn_mult=4,
            n_archetypes=64, il_use_ternary=True,
            nautilus_warmup=200, nautilus_mode="sequential", nautilus_chambers=None,
            conv_window=4, conv_stride=2, grammar_rows=8, grammar_cols=8,
        )
        self.model = HierarchicalE2(cfg)
        self.cfg   = cfg

        # Загружаем лучший checkpoint
        candidates = [
            checkpoint,
            str(_ROOT / "checkpoint_e2_joint.pt"),
            str(_ROOT / "checkpoint_e2_clusters.pt"),
            str(_ROOT / "checkpoint_e2.pt"),
        ]
        self.checkpoint_used = None
        for ckpt in candidates:
            if ckpt and Path(ckpt).exists():
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state, strict=False)
                self.checkpoint_used = ckpt
                break

        self.model.eval()

        # Корпус для поиска похожих
        self._corpus: list[dict] = []  # {"text", "q6", "hex_idx", "source", "domain"}
        if load_corpus:
            self._build_corpus_index()

    def _build_corpus_index(self, max_per_source: int = 50) -> None:
        """Строит Q6-индекс корпуса для поиска похожих."""
        items: list[dict] = []

        # Внешний корпус
        try:
            from corpus_loader import CorpusLoader
            ext = CorpusLoader().as_training_corpus(max_per_source=max_per_source)
            for it in ext:
                if len(it["text"]) >= 8:
                    items.append({"text": it["text"][:120],
                                  "source": it["source"],
                                  "domain": it["domain"],
                                  "alpha": it["alpha"]})
        except Exception:
            pass

        # Внутренние кластеры
        try:
            from repo_corpus_loader import RepoCorpusLoader
            int_items = RepoCorpusLoader().as_flat_corpus()
            for it in int_items:
                if len(it["text"]) >= 8:
                    items.append({"text": it["text"][:120],
                                  "source": f"repo/{it['cluster']}",
                                  "domain": it["domain"],
                                  "alpha": it["alpha"]})
        except Exception:
            pass

        # Вычисляем Q6 для каждого элемента
        self._corpus = []
        for it in items:
            try:
                emb = self.embed(it["text"])
                self._corpus.append({
                    "text":    it["text"][:80],
                    "source":  it["source"],
                    "domain":  it.get("domain", "?"),
                    "alpha":   it.get("alpha", 0),
                    "q6":      emb["q6"],
                    "hex_idx": emb["hex_idx"],
                })
            except Exception:
                pass

    # ── Основные методы ──────────────────────────────────────────────────────

    def embed(self, text: str) -> dict:
        """Q6-эмбеддинг текста."""
        r = self.model.embed_text(text)
        return {
            "text":     text[:60],
            "q6":       r["q6"],
            "hex_idx":  r["hex_idx"],
            "hex_name": _hex_name(r["hex_idx"]),
            "domain":   _q6_to_domain(r["q6"]),
            "q6_str":   "".join(map(str, r["q6"])),
        }

    def find_similar(self, text: str, n: int = 5) -> list[dict]:
        """Топ-N похожих элементов корпуса по Хэмминг-расстоянию в Q6."""
        query_emb = self.embed(text)
        q6 = query_emb["q6"]

        scored = []
        for item in self._corpus:
            dist = _hamming(q6, item["q6"])
            scored.append({**item, "hamming": dist})

        scored.sort(key=lambda x: x["hamming"])
        return scored[:n]

    def concept_map(self) -> dict:
        """
        Q6-карта: для каждой из 64 гексаграмм — список концептов.
        Показывает как знание распределено в Q6-гиперкубе.
        """
        cells: dict[int, list] = defaultdict(list)
        for item in self._corpus:
            cells[item["hex_idx"]].append({
                "text":   item["text"][:50],
                "source": item["source"],
                "domain": item["domain"],
            })

        result = {}
        for idx in range(64):
            contents = cells.get(idx, [])
            result[idx] = {
                "hex_name":  _hex_name(idx),
                "count":     len(contents),
                "domains":   list({it["domain"] for it in contents}),
                "sources":   list({it["source"] for it in contents}),
                "examples":  contents[:3],
            }
        return result

    def domain_report(self) -> dict:
        """Распределение доменов в Q6-пространстве."""
        by_domain: dict[str, list] = defaultdict(list)
        for item in self._corpus:
            by_domain[item["domain"]].append(item["hex_idx"])

        result = {}
        for domain, hex_ids in sorted(by_domain.items()):
            # Найти наиболее занятые ячейки
            from collections import Counter
            counter = Counter(hex_ids)
            top_cells = counter.most_common(3)
            result[domain] = {
                "count":      len(hex_ids),
                "unique_hex": len(set(hex_ids)),
                "top_hex":    [{"idx": idx, "name": _hex_name(idx), "count": cnt}
                               for idx, cnt in top_cells],
                "hex_density": round(len(set(hex_ids)) / 64 * 100, 1),
            }
        return result

    def generate(self, prefix: str, max_new_tokens: int = 30,
                 temperature: float = 0.8) -> str:
        """Авторегрессивная генерация продолжения текста."""
        import torch
        vocab_size = self.cfg.vocab_size
        block_size = self.cfg.block_size

        ids = [min(b, vocab_size - 1) for b in prefix.encode("utf-8")][-block_size:]
        ids = torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)

        generated = list(prefix.encode("utf-8"))

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = ids[:, -block_size:]
                logits, _, _ = self.model(inp)
                next_logits = logits[0, -1] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                generated.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

        try:
            return bytes(generated[:len(prefix.encode())+max_new_tokens]).decode(
                "utf-8", errors="replace")
        except Exception:
            return prefix

    def cross_repo_align(self) -> dict:
        """
        Сравнивает Q6-распределение внешнего корпуса vs внутренних кластеров.
        Показывает зоны пересечения и расхождения знания.
        """
        ext_q6s: list[list[int]] = []
        int_q6s: list[list[int]] = []

        for item in self._corpus:
            if item["source"].startswith("repo/"):
                int_q6s.append(item["q6"])
            else:
                ext_q6s.append(item["q6"])

        if not ext_q6s or not int_q6s:
            return {"error": "Недостаточно данных для сравнения"}

        # Средний Q6 (центроид)
        def centroid(q6_list: list[list[int]]) -> list[float]:
            return [sum(q6[i] for q6 in q6_list) / len(q6_list)
                    for i in range(6)]

        ext_c = centroid(ext_q6s)
        int_c = centroid(int_q6s)

        # Расстояние между центроидами
        dist = sum(abs(a - b) for a, b in zip(ext_c, int_c))

        # Пересечение ячеек гиперкуба
        from collections import Counter
        ext_cells = Counter(tuple(q6) for q6 in ext_q6s)
        int_cells = Counter(tuple(q6) for q6 in int_q6s)
        overlap = len(set(ext_cells) & set(int_cells))
        total   = len(set(ext_cells) | set(int_cells))

        return {
            "external_texts":  len(ext_q6s),
            "internal_texts":  len(int_q6s),
            "ext_centroid":    [round(x, 3) for x in ext_c],
            "int_centroid":    [round(x, 3) for x in int_c],
            "centroid_dist":   round(dist, 3),
            "alignment":       round((1 - dist / 6) * 100, 1),
            "shared_hex_cells": overlap,
            "total_hex_cells":  total,
            "coverage_overlap": round(overlap / max(total, 1) * 100, 1),
            "interpretation":   (
                "Хорошая интеграция" if dist < 1.5 else
                "Частичная интеграция" if dist < 3.0 else
                "Домены разошлись — рекомендуется joint-training"
            ),
        }

    def query(self, text: str) -> dict:
        """Полный запрос: embed + similar + генерация."""
        emb     = self.embed(text)
        similar = self.find_similar(text, n=5)
        gen     = self.generate(text, max_new_tokens=20)
        return {
            "query":       text,
            "embed":       emb,
            "similar":     similar,
            "generated":   gen,
            "checkpoint":  self.checkpoint_used,
            "corpus_size": len(self._corpus),
        }


# ══════════════════════════════════════════════════════════════════════════════

def _print_embed(r: dict) -> None:
    print(f"\n  Q6-эмбеддинг: «{r['text']}»")
    print(f"  Q6  = [{r['q6_str']}]  →  гексаграмма #{r['hex_idx']} «{r['hex_name']}»")
    print(f"  Домен: {r['domain']}")


def _print_similar(items: list[dict], query: str) -> None:
    print(f"\n  Похожие на «{query}»:")
    for i, it in enumerate(items, 1):
        q6s = "".join(map(str, it["q6"]))
        print(f"  {i}. [{it['source']:<20}] d={it['hamming']}  Q6=[{q6s}]")
        print(f"     {it['text'][:80]}")


def _print_map(cmap: dict, top_n: int = 10) -> None:
    non_empty = [(idx, info) for idx, info in cmap.items() if info["count"] > 0]
    non_empty.sort(key=lambda x: -x[1]["count"])
    print(f"\n  Q6-карта (топ {top_n} ячеек из 64):")
    print(f"  {'#':<4} {'Гексаграмма':<22} {'Текстов':>8}  Домены")
    print("  " + "─" * 56)
    for idx, info in non_empty[:top_n]:
        domains = ", ".join(info["domains"][:3])
        print(f"  #{idx:<3} {info['hex_name']:<22} {info['count']:>8}  {domains}")
    total = sum(info["count"] for _, info in non_empty)
    print(f"\n  Заполнено ячеек: {len(non_empty)}/64  Всего текстов: {total}")


def _print_domain_report(report: dict) -> None:
    print(f"\n  {'Домен':<10} {'Текстов':>8} {'Ячеек':>7} {'Плотн.%':>8}  Топ гексаграмма")
    print("  " + "─" * 60)
    for domain, info in sorted(report.items()):
        top = info["top_hex"][0] if info["top_hex"] else {}
        top_str = f"#{top.get('idx','?')} {top.get('name','?')}" if top else ""
        print(f"  {domain:<10} {info['count']:>8} {info['unique_hex']:>7} "
              f"{info['hex_density']:>7.1f}%  {top_str}")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="E2 Inference API")
    parser.add_argument("--embed",         default="", help="Q6-эмбеддинг текста")
    parser.add_argument("--similar",       default="", help="Найти похожие концепты")
    parser.add_argument("--n",             type=int, default=5)
    parser.add_argument("--map",           action="store_true", help="Q6-карта корпуса")
    parser.add_argument("--domain-report", action="store_true", help="Отчёт по доменам")
    parser.add_argument("--generate",      default="", help="Генерация текста")
    parser.add_argument("--cross-align",   action="store_true",
                        help="Сравнение внешнего и внутреннего корпусов")
    parser.add_argument("--query",         default="", help="Полный запрос")
    parser.add_argument("--checkpoint",    default="", help="Путь к .pt файлу")
    parser.add_argument("--json",          action="store_true", help="JSON-вывод")
    parser.add_argument("--no-corpus",     action="store_true",
                        help="Не загружать корпус (быстрее для --embed)")
    args = parser.parse_args()

    load_corpus = not args.no_corpus and (
        args.similar or args.map or args.domain_report or
        args.cross_align or args.query
    )

    print("\n  Загружаю E2 модель...", end=" ", flush=True)
    engine = E2Inference(
        checkpoint=args.checkpoint or None,
        load_corpus=load_corpus,
    )
    ckpt_name = Path(engine.checkpoint_used).name if engine.checkpoint_used else "случайные веса"
    print(f"✅ {ckpt_name}  корпус={len(engine._corpus)} текстов")

    if args.embed:
        r = engine.embed(args.embed)
        if args.json:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        else:
            _print_embed(r)

    if args.similar:
        results = engine.find_similar(args.similar, n=args.n)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            _print_similar(results, args.similar)

    if args.map:
        cmap = engine.concept_map()
        if args.json:
            print(json.dumps(cmap, ensure_ascii=False, indent=2))
        else:
            _print_map(cmap)

    if args.domain_report:
        report = engine.domain_report()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            _print_domain_report(report)

    if args.generate:
        text = engine.generate(args.generate)
        print(f"\n  Генерация: «{args.generate}»\n  → {text}")

    if args.cross_align:
        result = engine.cross_repo_align()
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\n  Выравнивание внешний ↔ внутренний:")
            for k, v in result.items():
                print(f"    {k}: {v}")

    if args.query:
        result = engine.query(args.query)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _print_embed(result["embed"])
            _print_similar(result["similar"], args.query)
            print(f"\n  Генерация: {result['generated'][:100]}")


if __name__ == "__main__":
    main()
