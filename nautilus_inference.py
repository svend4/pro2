#!/usr/bin/env python3
"""
nautilus_inference.py — Micro-model inference через Nautilus Portal.

Показывает КАК микро-модель (Variant3GPT) работает МЕЖДУ репозиториями:

  Репо A (info1) ──запрос──┐
  Репо B (data7) ──запрос──┤
  Репо C (meta)  ──запрос──┤→ NautilusPortal → CorpusLoader → Variant3GPT
  Репо D (data2) ──запрос──┘         ↓
                                  ответ + Q6-координата + источник

Принцип работы:
  1. NautilusPortal.query(concept) → собирает контекст из всех адаптеров
  2. CorpusLoader.load(source=...)  → загружает релевантные тексты
  3. Variant3GPT.generate(context)  → генерирует ответ (или эмбеддинг)
  4. Результат содержит: текст + Q6[b0..b5] + источник-репо + α-уровень

Режимы работы:
  A. STANDALONE  — модель загружена из checkpoint, работает без других репо
  B. PORTAL      — модель запрашивает контекст через NautilusPortal
  C. FEDERATED   — каждый репо обучает свой фрагмент, модель объединяет

Usage:
  python nautilus_inference.py --query "кристалл" --mode portal
  python nautilus_inference.py --query "гексаграмма" --mode standalone
  python nautilus_inference.py --embed "трансформация знаний"  # Q6-эмбеддинг

Горизонтальные связи (↔):
  ↔ nautilus/portal.py — источник контекста
  ↔ corpus_loader.py   — загрузка текстов для RAG
  ↔ graph_health.py    — метрики после инференса
  ↔ bidir_train.py     — модель откуда берётся checkpoint
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from yijing_transformer.constants import HEX_NAMES
_HEX_NAMES = HEX_NAMES


# ══════════════════════════════════════════════════════════════════════════════
# Режим A: Standalone (модель загружена из checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(checkpoint: Optional[Path] = None):
    """Загружает Variant3GPT из checkpoint или создаёт новую (для demo)."""
    try:
        import torch
        from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT

        cfg = Variant3Config(
            vocab_size=256, block_size=32, d_model=128,
            n_heads=4, n_layers=4, ffn_mult=4,
            hamming_lambda=0.15, uncertainty_budget=0.25,
            dropout=0.0, use_domain_routing=True,
        )
        model = Variant3GPT(cfg)

        ckpt_path = checkpoint or (_ROOT / "checkpoint_bidir_v2.pt")
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            model.eval()
            return model, cfg, True  # loaded=True
        else:
            model.eval()
            return model, cfg, False  # loaded=False (random weights)
    except ImportError as e:
        return None, None, False


# ══════════════════════════════════════════════════════════════════════════════
# Q6-эмбеддинг текста (без генерации)
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str, model, cfg) -> dict:
    """
    Прогоняет text через модель и возвращает Q6-координату.
    Аналог: meta.Q6Encoder.encode(text) но полностью внутри pro2.
    """
    import torch

    tokens = torch.tensor(
        [min(ord(c), cfg.vocab_size - 1) for c in text[:cfg.block_size]],
        dtype=torch.long,
    ).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        # Берём hidden state из первого forward-прохода
        logits, _ = model(tokens)
        # Средний hidden (прокси для эмбеддинга)
        hidden = logits.mean(dim=1).squeeze(0)  # [vocab_size]
        # Проецируем в 6D бинарный вектор (Q6)
        q6_raw = hidden[:6]
        q6 = (q6_raw > 0).int().tolist()
        hex_idx = sum(b << i for i, b in enumerate(q6))

    return {
        "text":     text[:60],
        "q6":       q6,
        "hex_idx":  hex_idx,
        "hex_name": _hex_name(hex_idx),
        "domain":   _q6_to_domain(q6),
    }


def _hex_name(idx: int) -> str:
    return _HEX_NAMES[idx] if 0 <= idx < 64 else f"#{idx}"


_DOMAINS = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]


def _q6_to_domain(q6: list[int]) -> str:
    """Активная линия гексаграммы = домен (линия с наивысшим битом)."""
    for i in reversed(range(6)):
        if q6[i]:
            return _DOMAINS[i]
    return _DOMAINS[0]


# ══════════════════════════════════════════════════════════════════════════════
# Режим B: Portal (запрашивает контекст у всех репо через NautilusPortal)
# ══════════════════════════════════════════════════════════════════════════════

def portal_query(concept: str) -> dict:
    """
    Использует NautilusPortal для сбора контекста из всех репо,
    затем встраивает каждый концепт в Q6-пространство.

    Это и есть "работа модели между несколькими репозиториями":
    - info1  даёт: методологический контекст (α-уровни)
    - meta   даёт: математический контекст (hexagram ID)
    - data7  даёт: теоретический контекст (K₀→K₁→K₂)
    - data2  даёт: прикладной контекст (ЕТД/Скарабей)
    - pro2   даёт: семантический эмбеддинг (Q6)
    """
    # Добавляем nautilus/ в путь поиска
    nautilus_dir = _ROOT / "nautilus"
    sys.path.insert(0, str(nautilus_dir))

    try:
        from portal import NautilusPortal, render_text
        portal = NautilusPortal()
        result = portal.query(concept)

        model, cfg, loaded = _load_model()

        entries_with_q6 = []
        for e in result.entries:
            entry_data = {
                "id":     e.id,
                "title":  e.title,
                "repo":   e.id.split(":")[0],
                "source": e.source,
                "content_preview": e.content[:120],
                "links":  e.links[:3],
            }
            if model and cfg:
                emb = embed_text(e.content, model, cfg)
                entry_data["q6"]      = emb["q6"]
                entry_data["hex_idx"] = emb["hex_idx"]
                entry_data["hex_name"] = emb["hex_name"]
                entry_data["domain"]  = emb["domain"]
            entries_with_q6.append(entry_data)

        return {
            "query":       concept,
            "mode":        "portal",
            "model_loaded": loaded,
            "repos_queried": list({e["repo"] for e in entries_with_q6}),
            "entries":     entries_with_q6,
            "cross_links": result.cross_links,
            "consensus":   result.consensus,
        }
    finally:
        if str(nautilus_dir) in sys.path:
            sys.path.remove(str(nautilus_dir))


# ══════════════════════════════════════════════════════════════════════════════
# Режим C: Federated — как модель функционирует между несколькими репо
# ══════════════════════════════════════════════════════════════════════════════

def federated_info() -> dict:
    """
    Описывает схему федеративного обучения/инференса.
    Каждый репо обучает свой фрагмент → pro2 объединяет через Q6.
    """
    return {
        "architecture": "Federated Q6 Micro-Model",
        "principle": (
            "Каждый репо остаётся независимым и обучает "
            "СВОЮ часть знания на СВОИХ данных. "
            "pro2 (Variant3GPT) — координирующая модель, "
            "которая знает Q6-пространство и может "
            "отображать знание из ЛЮБОГО репо в Q6."
        ),
        "repos": {
            "info1": {
                "trains": "α-уровни → Q6 dim=5..0 (абстракция→конкретность)",
                "exports": "gradient на α-embedding",
                "receives": "Q6-координата для каждого α-документа",
            },
            "meta": {
                "trains": "гексаграмма_N → Q6[bin(N)] (тождество!)",
                "exports": "CA-правило N → Q6-bits",
                "receives": "Q6-позиция каждого CA-правила в semantic space",
            },
            "data7": {
                "trains": "K₀→K₁→K₂ трансформация → bidir_train forward/backward",
                "exports": "refinement loop proposals",
                "receives": "Q6-центральность каждого концепта в графе",
            },
            "data2": {
                "trains": "Скарабей-паттерн → Q6 topology",
                "exports": "topology graph",
                "receives": "Q6-neighbourhood для каждого узла топологии",
            },
            "pro2": {
                "role": "КООРДИНАТОР",
                "trains": "ВСЁ ВМЕСТЕ через bidir_train.py",
                "exports": "Обученные Q6-эмбеддинги для всех концептов всех репо",
                "receives": "Тексты и концепты от всех репо через corpus_loader.py",
            },
        },
        "shared_space": "Q6 = {-1,+1}⁶ гиперкуб = 64 гексаграммы",
        "convergence": (
            "Обучение сходится когда: "
            "каждый репо может генерировать тексты, "
            "которые Variant3GPT распознаёт как "
            "«достаточно центральные» в Q6."
        ),
        "runtime_flow": [
            "1. Запрос приходит (из любого репо или прямо)",
            "2. corpus_loader.py ищет релевантные тексты из всех клонов",
            "3. Variant3GPT.embed(context) → Q6[b0..b5]",
            "4. NautilusPortal находит ближайшие концепты (Хэмминг-расстояние)",
            "5. Ответ: текст + Q6-координата + источник-репо + α-уровень",
            "6. graph_health.py обновляет метрики CD/VT/CR/DB",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Nautilus Inference — micro-model across repos"
    )
    parser.add_argument("--query",    "-q", default="", help="Концепт для запроса")
    parser.add_argument("--embed",    "-e", default="", help="Текст для Q6-эмбеддинга")
    parser.add_argument("--mode",     "-m", default="portal",
                        choices=["standalone", "portal", "federated"],
                        help="Режим работы")
    parser.add_argument("--json",     action="store_true", help="JSON-вывод")
    parser.add_argument("--checkpoint", default="", help="Путь к .pt файлу")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None

    # ── Embed mode ─────────────────────────────────────────────────────────
    if args.embed:
        model, cfg, loaded = _load_model(ckpt)
        if model is None:
            print("❌ Не удалось загрузить модель (torch не установлен?)")
            return
        result = embed_text(args.embed, model, cfg)
        result["model_loaded"] = loaded
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"\n📐 Q6-эмбеддинг: '{args.embed}'")
            print(f"  Q6 = {result['q6']}  (hex #{result['hex_idx']})")
            print(f"  Гексаграмма: {result['hex_name']}")
            print(f"  Домен: {result['domain']}")
            print(f"  Модель загружена: {'да ✅' if loaded else 'нет (случайные веса) ⚠️'}")
        return

    # ── Federated info ──────────────────────────────────────────────────────
    if args.mode == "federated":
        info = federated_info()
        if args.json:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print(f"\n⬡ FEDERATED Q6 MICRO-MODEL")
            print("=" * 60)
            print(info["principle"])
            print("\nСхема репозиториев:")
            for repo, details in info["repos"].items():
                print(f"\n  [{repo.upper()}]")
                for k, v in details.items():
                    print(f"    {k}: {v}")
            print("\nПоток инференса:")
            for step in info["runtime_flow"]:
                print(f"  {step}")
        return

    # ── Portal / Standalone mode ────────────────────────────────────────────
    query = args.query or "знание"

    if args.mode == "portal":
        result = portal_query(query)
    else:
        # Standalone: только Q6-embed без портала
        model, cfg, loaded = _load_model(ckpt)
        result = {
            "query":       query,
            "mode":        "standalone",
            "model_loaded": loaded,
            "embed":       embed_text(query, model, cfg) if model else {},
        }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n⬡ NAUTILUS INFERENCE — '{result['query']}'")
        print("=" * 60)
        if "repos_queried" in result:
            print(f"Репо: {', '.join(result.get('repos_queried', []))}")
        if "entries" in result:
            for e in result["entries"][:5]:
                q6_str = str(e.get("q6", "?"))
                name   = e.get("hex_name", "")
                print(f"\n  [{e['repo'].upper()}] {e['title']}")
                print(f"    {e['content_preview'][:100]}")
                if "q6" in e:
                    print(f"    Q6={q6_str} → гексаграмма '{name}' [{e.get('domain','')}]")
        if "consensus" in result:
            c = result["consensus"]
            print(f"\nКонсенсус: {c.get('coverage',0)*100:.0f}% репо отвечают")
        print(f"Модель загружена: {'да ✅' if result.get('model_loaded') else 'нет ⚠️'}")


if __name__ == "__main__":
    main()
