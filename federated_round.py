#!/usr/bin/env python3
"""
federated_round.py — Федеративный раунд: синхронизация checkpoint_e2_joint.pt
                     с другими репозиториями через NautilusPortal.

Принцип федеративного обучения:
  1. EXPORT    — взять Q6-эмбеддинги из нашей обученной модели
  2. BROADCAST — передать их в NautilusPortal (как «знание pro2»)
  3. IMPORT    — собрать знание других репо (info1, meta, data7, data2)
  4. ALIGN     — выровнять Q6-пространства через Хэмминг-близость
  5. INTEGRATE — дообучить модель на импортированных концептах
  6. REPORT    — сравнить Q6-метрики до и после раунда

Режимы:
  LOCAL   — только локальная часть (без сети, используя _clones)
  PORTAL  — через NautilusPortal.query() (требует nautilus/portal.py)
  DRY_RUN — только анализ, без обучения

Usage:
  python federated_round.py                   # LOCAL режим (по умолчанию)
  python federated_round.py --mode portal     # через NautilusPortal
  python federated_round.py --mode dry-run    # только анализ
  python federated_round.py --fast            # быстро (20 шагов интеграции)
  python federated_round.py --concepts "гексаграмма,кристалл,трансформация"

Горизонтальные связи (↔):
  ↔ nautilus/portal.py        — NautilusPortal для межрепо запросов
  ↔ corpus_loader.py          — данные из _clones (info1, data7, meta, data2)
  ↔ e2_inference.py           — E2Inference для embed/compare
  ↔ e2_concept_evolution.py   — снапшоты до/после раунда
  ↔ q6_graph_updater.py       — обновление графа после интеграции
"""

import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path

import torch

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
from corpus_loader import CorpusLoader
from e2_concept_evolution import take_snapshot, compare_snapshots, TRACKED_CONCEPTS

# ── Конфигурация ──────────────────────────────────────────────────────────────

E2_CFG = E2Config(
    vocab_size=256, d_model=128, block_size=32,
    n_core=4, n_heads=4, dropout=0.05,
    hamming_lambda=0.15, uncertainty_budget=0.25, ffn_mult=4,
    n_archetypes=64, il_use_ternary=True,
    nautilus_warmup=200, nautilus_mode="sequential", nautilus_chambers=None,
    conv_window=4, conv_stride=2, grammar_rows=8, grammar_cols=8,
)

JOINT_CHECKPOINT   = _ROOT / "checkpoint_e2_joint.pt"
CLUSTERS_CHECKPOINT = _ROOT / "checkpoint_e2_clusters.pt"
E2_CHECKPOINT      = _ROOT / "checkpoint_e2.pt"
FED_CHECKPOINT     = _ROOT / "checkpoint_e2_federated.pt"
FED_LOG            = _ROOT / "federated_round_log.json"

# Репо-партнёры
REPO_PARTNERS = {
    "info1":  {"alpha": 3, "domain": "NOOS",  "description": "Методология α-уровней"},
    "data7":  {"alpha": 4, "domain": "NOOS",  "description": "Теория K₀→K₁→K₂"},
    "meta":   {"alpha": 3, "domain": "YIJING","description": "Q6/hexagram система"},
    "data2":  {"alpha": 2, "domain": "GEO",   "description": "ЕТД Крюкова"},
}

# Фокусные концепты для федерации
FED_CONCEPTS = [
    "гексаграмма", "Q6 координата", "трансформация знаний",
    "архетипы", "nautilus", "информация", "кристалл",
    "методология", "α-уровень", "convergence",
]


def _hamming(a: list, b: list) -> int:
    return sum(x != y for x, y in zip(a, b))


def encode(text: str, vocab_size: int = 256, block_size: int = 32) -> torch.Tensor:
    ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
    return torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)


def perplexity(model: HierarchicalE2, texts: list[str], n: int = 20) -> float:
    model.eval()
    ppls = []
    for text in random.sample(texts, min(n, len(texts))):
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            _, loss, _ = model(tokens[:, :-1], targets=tokens[:, 1:])
        if loss is not None and not torch.isnan(loss):
            ppls.append(math.exp(min(loss.item(), 10)))
    return sum(ppls) / len(ppls) if ppls else float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# Шаг 1: EXPORT — Q6-карта текущей модели
# ══════════════════════════════════════════════════════════════════════════════

def export_q6_knowledge(model: HierarchicalE2, concepts: list[str]) -> dict:
    """Экспортирует Q6-знание нашей модели для передачи партнёрам."""
    model.eval()
    q6_map = {}
    for concept in concepts:
        r = model.embed_text(concept)
        q6_map[concept] = {
            "q6":       r["q6"],
            "hex_idx":  r["hex_idx"],
            "source":   "pro2",
        }
    return {
        "repo":      "pro2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "concepts":  q6_map,
        "n_concepts": len(q6_map),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Шаг 2/3: IMPORT — Сбор знания из партнёров
# ══════════════════════════════════════════════════════════════════════════════

def import_local(concepts: list[str], verbose: bool = True) -> dict:
    """
    Локальный импорт: ищет концепты в _clones через corpus_loader.
    Не требует сети — работает с локальными клонами.
    """
    loader = CorpusLoader()
    partner_data: dict[str, list[str]] = {}

    for partner, info in REPO_PARTNERS.items():
        items = loader.load(domains=[partner])
        texts = []
        for text, meta in items:
            # Фильтруем: берём тексты, содержащие хотя бы один фокусный концепт
            if any(c.lower() in text.lower() for c in concepts):
                texts.append(text)
        if texts:
            partner_data[partner] = texts[:50]
            if verbose:
                print(f"  {partner:<10}: {len(texts)} → взято {len(partner_data[partner])}")
        else:
            if verbose:
                print(f"  {partner:<10}: 0 текстов (клон отсутствует или не содержит концептов)")

    return partner_data


def import_portal(concepts: list[str], verbose: bool = True) -> dict:
    """
    Портальный импорт: через NautilusPortal.query().
    Требует nautilus/portal.py.
    """
    nautilus_dir = _ROOT / "nautilus"
    sys.path.insert(0, str(nautilus_dir))
    partner_data: dict[str, list[str]] = {}

    try:
        from portal import NautilusPortal
        portal = NautilusPortal()

        for concept in concepts[:5]:  # ограничиваем запросы
            try:
                result = portal.query(concept)
                for entry in result.entries:
                    repo = entry.id.split(":")[0]
                    if repo not in partner_data:
                        partner_data[repo] = []
                    partner_data[repo].append(entry.content[:200])
                if verbose:
                    repos = list({entry.id.split(":")[0] for entry in result.entries})
                    print(f"  «{concept}»: ответили {repos}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  portal.query({concept}): {e}")

    except ImportError as e:
        if verbose:
            print(f"  ⚠️  NautilusPortal недоступен: {e}")
        # Fallback на локальный импорт
        return import_local(concepts, verbose)
    finally:
        if str(nautilus_dir) in sys.path:
            sys.path.remove(str(nautilus_dir))

    return partner_data


# ══════════════════════════════════════════════════════════════════════════════
# Шаг 4: ALIGN — Анализ выравнивания Q6 с партнёрами
# ══════════════════════════════════════════════════════════════════════════════

def align_analysis(
    model: HierarchicalE2,
    partner_data: dict[str, list[str]],
    our_q6: dict,
) -> dict:
    """
    Анализирует насколько Q6-пространство pro2 «выровнено» с каждым партнёром.
    Измеряет: среднее Хэмминг-расстояние между нашими концептами
    и тем, как партнёрские тексты проецируются в Q6.
    """
    alignment: dict[str, dict] = {}

    for partner, texts in partner_data.items():
        if not texts:
            continue
        # Embed партнёрских текстов
        partner_q6s = []
        for text in texts[:20]:
            r = model.embed_text(text[:60])
            partner_q6s.append(r["q6"])

        # Наши Q6
        our_q6_list = [v["q6"] for v in our_q6["concepts"].values()]

        # Среднее Хэмминг-расстояние
        if partner_q6s and our_q6_list:
            total_dist = 0
            count = 0
            for pq6 in partner_q6s[:10]:
                for oq6 in our_q6_list[:10]:
                    total_dist += _hamming(pq6, oq6)
                    count += 1
            avg_dist = total_dist / count if count else 6

            alignment[partner] = {
                "n_texts":    len(texts),
                "avg_hamming": round(avg_dist, 3),
                "alignment_pct": round((1 - avg_dist / 6) * 100, 1),
                "status": (
                    "ALIGNED" if avg_dist < 2 else
                    "PARTIAL" if avg_dist < 4 else
                    "DIVERGED"
                ),
            }

    return alignment


# ══════════════════════════════════════════════════════════════════════════════
# Шаг 5: INTEGRATE — Дообучение на импортированных данных
# ══════════════════════════════════════════════════════════════════════════════

def integrate_partner_data(
    model:        HierarchicalE2,
    partner_data: dict[str, list[str]],
    steps_per_partner: int = 50,
    fast: bool = False,
) -> dict:
    """Дообучает модель на текстах партнёрских репо."""
    if fast:
        steps_per_partner = 15

    results = {}
    partner_phase = {
        "info1": 4,   # α=+3 → TheoryLevel
        "data7": 5,   # α=+4 → PhiloLevel
        "meta":  4,   # α=+3 → TheoryLevel
        "data2": 3,   # α=+2 → MethodLevel
    }

    for partner, texts in partner_data.items():
        if not texts:
            continue

        phase = partner_phase.get(partner, 3)
        lr    = 5e-5  # осторожный LR для федеративной интеграции
        print(f"\n  Интегрирую «{partner}»  (фаза={phase}, шагов={steps_per_partner})")

        ppl_before = perplexity(model, texts)
        model.set_training_phase(phase)
        model.train()

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4,
        )

        losses = []
        for step in range(1, steps_per_partner + 1):
            text   = random.choice(texts)
            tokens = encode(text)
            if tokens.shape[1] < 2:
                continue
            _, loss, _ = model(tokens[:, :-1], targets=tokens[:, 1:])
            if loss is None or torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(loss.item())

        ppl_after = perplexity(model, texts)
        delta = (ppl_before - ppl_after) / ppl_before * 100 if ppl_before else 0
        avg_loss = sum(losses[-10:]) / max(1, len(losses[-10:]))
        sign = "✅" if delta > 0 else "⚠️ "
        print(f"  PPL: {ppl_before:.2f} → {ppl_after:.2f}  Δ={delta:+.1f}%  {sign}")

        results[partner] = {
            "ppl_before": round(ppl_before, 2),
            "ppl_after":  round(ppl_after, 2),
            "ppl_delta":  round(delta, 2),
            "final_loss": round(avg_loss, 4),
            "n_texts":    len(texts),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция федеративного раунда
# ══════════════════════════════════════════════════════════════════════════════

def federated_round(
    mode:     str = "local",
    concepts: list[str] = FED_CONCEPTS,
    fast:     bool = False,
    dry_run:  bool = False,
    verbose:  bool = True,
) -> dict:
    """
    Выполняет один федеративный раунд:
      export → import → align → integrate → snapshot
    """
    print(f"\n{'═'*66}")
    print(f"  ФЕДЕРАТИВНЫЙ РАУНД  [{mode.upper()}]")
    print(f"{'═'*66}")

    # ── Загрузка модели ───────────────────────────────────────────────────
    model = HierarchicalE2(E2_CFG)
    for ckpt in [FED_CHECKPOINT, JOINT_CHECKPOINT, CLUSTERS_CHECKPOINT, E2_CHECKPOINT]:
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"  ✅ Модель: {ckpt.name}")
            break

    # ── Снапшот ДО ───────────────────────────────────────────────────────
    snap_label_before = f"fed_before_{int(time.time())}"
    snap_before = take_snapshot(snap_label_before, concepts=TRACKED_CONCEPTS)
    print(f"  📸 Снапшот ДО: «{snap_label_before}»")

    # ── EXPORT ───────────────────────────────────────────────────────────
    print(f"\n  EXPORT: Q6-знание pro2 ({len(concepts)} концептов)")
    our_q6 = export_q6_knowledge(model, concepts)
    for concept, data in list(our_q6["concepts"].items())[:5]:
        q6s = "".join(map(str, data["q6"]))
        print(f"    «{concept:<25}»  Q6=[{q6s}]  →#{data['hex_idx']}")

    # ── IMPORT ───────────────────────────────────────────────────────────
    print(f"\n  IMPORT: данные от партнёров [{mode}]")
    if mode == "portal":
        partner_data = import_portal(concepts, verbose)
    else:
        partner_data = import_local(concepts, verbose)

    total_imported = sum(len(v) for v in partner_data.values())
    print(f"  Импортировано: {total_imported} текстов из {len(partner_data)} партнёров")

    # ── ALIGN ────────────────────────────────────────────────────────────
    print(f"\n  ALIGN: анализ выравнивания Q6")
    alignment = align_analysis(model, partner_data, our_q6)
    for partner, info in alignment.items():
        status_icon = {"ALIGNED": "✅", "PARTIAL": "⚡", "DIVERGED": "❌"}.get(
            info["status"], "?")
        print(f"  {partner:<10}  Хэмминг={info['avg_hamming']:.2f}  "
              f"align={info['alignment_pct']:.1f}%  {status_icon} {info['status']}")

    if dry_run:
        print("\n  [DRY-RUN] Интеграция пропущена")
        return {
            "mode": mode, "dry_run": True,
            "our_q6": our_q6, "alignment": alignment,
            "partner_data_sizes": {k: len(v) for k, v in partner_data.items()},
        }

    # ── INTEGRATE ────────────────────────────────────────────────────────
    print(f"\n  INTEGRATE: дообучение на партнёрских данных")
    if not partner_data:
        print("  ⚠️  Нет данных для интеграции — партнёры недоступны")
        integration_results = {}
    else:
        integration_results = integrate_partner_data(
            model, partner_data, fast=fast,
        )

    # ── Сохраняем checkpoint ─────────────────────────────────────────────
    torch.save(model.state_dict(), FED_CHECKPOINT)
    print(f"\n  💾 {FED_CHECKPOINT.name} сохранён")

    # ── Снапшот ПОСЛЕ ────────────────────────────────────────────────────
    snap_label_after = f"fed_after_{int(time.time())}"
    snap_after = take_snapshot(snap_label_after, concepts=TRACKED_CONCEPTS)
    print(f"  📸 Снапшот ПОСЛЕ: «{snap_label_after}»")

    # ── Сравнение снапшотов ───────────────────────────────────────────────
    comparison = compare_snapshots(snap_label_before, snap_label_after)
    moved  = comparison["moved"]
    total  = comparison["total"]
    drift  = comparison["avg_drift"]
    print(f"\n  Q6-изменения: {moved}/{total} концептов сдвинулись  "
          f"ср.дрейф={drift:.2f}")

    # ── Обновляем граф ────────────────────────────────────────────────────
    print("\n  Обновляю граф знаний...")
    try:
        from q6_graph_updater import update_graph_health
        gh = update_graph_health(max_concepts=150, verbose=False)
        m = gh["metrics"]
        print(f"  Граф: {m['n_concepts']} вершин  {m['n_edges']} рёбер  "
              f"CD={m['CD']:.1f}%  VT={m['VT']:.1f}%")
    except Exception as e:
        print(f"  ⚠️  Граф: {e}")
        m = {}

    return {
        "mode":               mode,
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "our_q6":             our_q6,
        "alignment":          alignment,
        "integration":        integration_results,
        "q6_drift": {
            "moved":    moved,
            "total":    total,
            "avg_drift": drift,
        },
        "graph_metrics":      m,
        "snapshot_before":    snap_label_before,
        "snapshot_after":     snap_label_after,
    }


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Federated Round для HierarchicalE2")
    parser.add_argument("--mode",     default="local",
                        choices=["local", "portal", "dry-run"],
                        help="Режим федерации")
    parser.add_argument("--fast",     action="store_true",
                        help="Быстрый режим (15 шагов интеграции)")
    parser.add_argument("--concepts", default="",
                        help="Концепты через запятую")
    parser.add_argument("--rounds",   type=int, default=1,
                        help="Количество раундов")
    parser.add_argument("--json",     action="store_true")
    args = parser.parse_args()

    concepts = ([c.strip() for c in args.concepts.split(",") if c.strip()]
                or FED_CONCEPTS)
    dry_run  = (args.mode == "dry-run")
    mode     = "local" if dry_run else args.mode

    all_results = []
    for round_n in range(1, args.rounds + 1):
        if args.rounds > 1:
            print(f"\n{'━'*66}\n  РАУНД {round_n}/{args.rounds}\n{'━'*66}")

        result = federated_round(
            mode=mode, concepts=concepts,
            fast=args.fast, dry_run=dry_run,
        )
        all_results.append(result)

        # Сохраняем лог после каждого раунда
        FED_LOG.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n{'═'*66}")
    print("  ИТОГИ ФЕДЕРАТИВНОГО РАУНДА")
    print(f"{'═'*66}")

    for i, r in enumerate(all_results, 1):
        print(f"\n  Раунд {i}:")
        for partner, info in r.get("alignment", {}).items():
            print(f"    {partner:<10}  align={info['alignment_pct']:.1f}%  "
                  f"{info['status']}")
        drift = r.get("q6_drift", {})
        if drift:
            print(f"    Q6 дрейф: {drift['moved']}/{drift['total']} концептов  "
                  f"ср.={drift['avg_drift']:.2f}")
        for partner, res in r.get("integration", {}).items():
            print(f"    {partner:<10}  PPL Δ={res['ppl_delta']:+.1f}%")

    print(f"\n  📄 Лог: {FED_LOG.name}")

    if args.json:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
