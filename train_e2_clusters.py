#!/usr/bin/env python3
"""
train_e2_clusters.py — Многократное применение E2-схемы ко всем файлам репо.

Применяет HierarchicalE2 поочерёдно ко всем 7 кластерам репозитория:

  Кластер 1 «Scripts»    (HYDRO, α=−4) → фаза 1 (GlyphLevel)
  Кластер 2 «Benchmarks» (GEO,   α=−2) → фаза 2 (CoreLevel)
  Кластер 3 «Training»   (AERO,  α= 0) → фаза 3 (MethodLevel)
  Кластер 4 «Self»       (METHOD,α= 0) → фаза 3 (MethodLevel)
  Кластер 5 «Models»     (PYRO,  α=+2) → фаза 4 (TheoryLevel)
  Кластер 6 «Portal»     (COSMO, α=+2) → фаза 4 (TheoryLevel)
  Кластер 7 «Theory»     (NOOS,  α=+4) → фаза 5 (PhiloLevel)

Схема обучения:
  - Каждый кластер обучает только соответствующие α-уровни (set_training_phase)
  - Checkpoint сохраняется после каждого кластера
  - Можно запустить несколько циклов (--cycles N) для углублённой интеграции
  - Между циклами — случайное перемешивание кластеров для генерализации

Usage:
  python train_e2_clusters.py                  # полный прогон всех кластеров
  python train_e2_clusters.py --cluster Theory # только один кластер
  python train_e2_clusters.py --fast           # 20 шагов на кластер (демо)
  python train_e2_clusters.py --cycles 3       # 3 цикла по всем кластерам
  python train_e2_clusters.py --resume         # продолжить с checkpoint_e2_clusters.pt

Горизонтальные связи (↔):
  ↔ repo_corpus_loader.py    — источник кластеров
  ↔ train_e2.py              — исходная E2-схема (по внешним данным)
  ↔ hierarchical_e2.py       — модель HierarchicalE2
  ↔ checkpoint_e2_clusters.pt — выходной checkpoint
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
from repo_corpus_loader import RepoCorpusLoader, CLUSTER_DEFS

# ── Конфигурация ──────────────────────────────────────────────────────────────

torch.manual_seed(42)
random.seed(42)

_ROOT   = Path(__file__).parent
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

E2_CFG = E2Config(
    vocab_size=256,
    d_model=128,
    block_size=32,
    n_core=4,
    n_heads=4,
    dropout=0.05,
    hamming_lambda=0.15,
    uncertainty_budget=0.25,
    ffn_mult=4,
    n_archetypes=64,
    il_use_ternary=True,
    nautilus_warmup=200,
    nautilus_mode="sequential",
    nautilus_chambers=None,
    conv_window=4,
    conv_stride=2,
    grammar_rows=8,
    grammar_cols=8,
)

# Шагов на кластер (полное обучение)
CLUSTER_STEPS = {
    "Scripts":    120,   # много файлов, α=−4 — быстрая кластеризация
    "Benchmarks": 100,   # JSON-данные, числовые паттерны
    "Training":   150,   # утилиты обучения — богатый словарь
    "Self":       120,   # корневые файлы — смешанный α
    "Models":     150,   # модели — сложный синтаксис Python
    "Portal":     100,   # портальные файлы — интеграционные паттерны
    "Theory":     130,   # теоретические MD — высокий α
}

CLUSTER_LR = {
    "Scripts":    3e-4,
    "Benchmarks": 2e-4,
    "Training":   2e-4,
    "Self":       2e-4,
    "Models":     1e-4,
    "Portal":     1e-4,
    "Theory":     5e-5,
}

# Порядок обучения: снизу вверх по α (E2-схема)
CLUSTER_ORDER = [
    "Scripts",     # α=−4 → фаза 1
    "Benchmarks",  # α=−2 → фаза 2
    "Training",    # α= 0 → фаза 3
    "Self",        # α= 0 → фаза 3
    "Models",      # α=+2 → фаза 4
    "Portal",      # α=+2 → фаза 4
    "Theory",      # α=+4 → фаза 5
]

V3_CHECKPOINT      = _ROOT / "checkpoint_bidir_v2.pt"
E2_BASE_CHECKPOINT = _ROOT / "checkpoint_e2.pt"
CLUSTERS_CHECKPOINT = _ROOT / "checkpoint_e2_clusters.pt"
CLUSTERS_LOG       = _ROOT / "train_e2_clusters_log.json"


# ══════════════════════════════════════════════════════════════════════════════

def encode(text: str, vocab_size: int = 256, block_size: int = 32) -> torch.Tensor:
    ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
    if not ids:
        ids = [32]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def perplexity(model: HierarchicalE2, texts: list[str], n: int = 25) -> float:
    model.eval()
    ppls = []
    sample = random.sample(texts, min(n, len(texts)))
    for text in sample:
        tokens = encode(text, E2_CFG.vocab_size, E2_CFG.block_size)
        if tokens.shape[1] < 2:
            continue
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        with torch.no_grad():
            _, loss, _ = model(inp, targets=tgt)
        if loss is not None and not torch.isnan(loss):
            ppls.append(math.exp(min(loss.item(), 10)))
    return sum(ppls) / len(ppls) if ppls else float("inf")


# ══════════════════════════════════════════════════════════════════════════════

def train_cluster(
    model:    HierarchicalE2,
    cluster:  str,
    texts:    list[str],
    steps:    int,
    lr:       float,
    e2_phase: int,
    log:      list,
    cycle:    int = 1,
) -> dict:
    """Обучает модель на одном кластере через соответствующую E2-фазу."""
    phase_names = {
        1: "GlyphLevel   α=−4",
        2: "CoreLevel    α=−2",
        3: "MethodLevel  α= 0",
        4: "TheoryLevel  α=+2",
        5: "PhiloLevel   α=+4",
    }

    defn = CLUSTER_DEFS[cluster]
    print(f"\n{'─'*66}")
    print(f"  КЛАСТЕР «{cluster}»  [{defn['domain']} α={defn['alpha']:+d}]  "
          f"цикл {cycle}")
    print(f"  E2-фаза {e2_phase}: {phase_names[e2_phase]}")
    print(f"  Текстов: {len(texts)}  Шагов: {steps}  LR: {lr}")
    print(f"{'─'*66}")

    if not texts:
        print("  ⚠️  Кластер пуст — пропускаем")
        return {}

    model.set_training_phase(e2_phase)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    ppl_before = perplexity(model, texts)
    print(f"  PPL до: {ppl_before:.2f}")

    losses   = []
    t0       = time.time()
    log_rows = []

    for step in range(1, steps + 1):
        text   = random.choice(texts)
        tokens = encode(text, E2_CFG.vocab_size, E2_CFG.block_size)
        if tokens.shape[1] < 2:
            continue

        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]

        _, loss, info = model(inp, targets=tgt)

        if loss is None or torch.isnan(loss):
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        scheduler.step()
        losses.append(loss.item())

        if step % max(1, steps // 5) == 0 or step == steps:
            avg_l = sum(losses[-20:]) / max(1, len(losses[-20:]))
            elapsed = time.time() - t0
            q6 = info["q6_coords"][0, 0].detach()
            q6_bin = (q6 > 0).int().tolist()
            hex_i = sum(b << i for i, b in enumerate(q6_bin))
            print(f"  шаг {step:>4}/{steps}  loss={avg_l:.4f}  "
                  f"Q6={''.join(map(str,q6_bin))} →#{hex_i:<2}  ({elapsed:.0f}s)")
            log_rows.append({"step": step, "loss": avg_l, "hex_idx": hex_i})

    ppl_after = perplexity(model, texts)
    delta = (ppl_before - ppl_after) / ppl_before * 100 if ppl_before < 1e6 else 0
    avg_final = sum(losses[-20:]) / max(1, len(losses[-20:]))

    sign = "✅" if delta > 0 else "⚠️ "
    print(f"\n  PPL после: {ppl_after:.2f}  (Δ{delta:+.1f}%)  {sign}")
    print(f"  Финальный loss: {avg_final:.4f}")

    result = {
        "cycle":      cycle,
        "cluster":    cluster,
        "domain":     defn["domain"],
        "alpha":      defn["alpha"],
        "e2_phase":   e2_phase,
        "texts":      len(texts),
        "steps":      steps,
        "lr":         lr,
        "ppl_before": round(ppl_before, 2),
        "ppl_after":  round(ppl_after, 2),
        "ppl_delta":  round(delta, 2),
        "final_loss": round(avg_final, 4),
        "log":        log_rows,
    }
    log.append(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════

def print_summary(log: list) -> None:
    if not log:
        return
    print(f"\n{'═'*70}")
    print("  ИТОГИ — E2 КЛАСТЕРНОЕ ОБУЧЕНИЕ")
    print(f"{'═'*70}")
    print(f"  {'Цикл':<5} {'Кластер':<13} {'Домен':<8} {'PPL до':>8} "
          f"{'PPL после':>10} {'Δ':>8}")
    print("  " + "─"*60)
    for r in log:
        if not r:
            continue
        sign = "⬇️ " if r["ppl_delta"] > 0 else "⬆️ "
        print(f"  {r['cycle']:<5} {r['cluster']:<13} {r['domain']:<8} "
              f"{r['ppl_before']:>8.2f} {r['ppl_after']:>10.2f} "
              f"  {sign}{abs(r['ppl_delta']):.1f}%")

    # Прогресс по циклам
    cycles_done = sorted(set(r["cycle"] for r in log if r))
    if len(cycles_done) > 1:
        print(f"\n  Прогресс по циклам:")
        for cyc in cycles_done:
            cyc_rows = [r for r in log if r and r["cycle"] == cyc]
            if cyc_rows:
                avg_delta = sum(r["ppl_delta"] for r in cyc_rows) / len(cyc_rows)
                print(f"    Цикл {cyc}: средний Δ={avg_delta:+.1f}%  "
                      f"({len(cyc_rows)} кластеров)")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E2-кластерное обучение на файлах репозитория pro2"
    )
    parser.add_argument("--cluster", default="",
                        help="Обучать только один кластер (Theory/Models/...)")
    parser.add_argument("--steps",   type=int, default=0,
                        help="Переопределить число шагов для всех кластеров")
    parser.add_argument("--fast",    action="store_true",
                        help="Быстрый режим: 20 шагов на кластер")
    parser.add_argument("--cycles",  type=int, default=1,
                        help="Количество полных циклов по всем кластерам")
    parser.add_argument("--resume",  action="store_true",
                        help="Загрузить checkpoint_e2_clusters.pt если есть")
    parser.add_argument("--no-v3",   action="store_true",
                        help="Не загружать веса Variant3 для CoreLevel")
    parser.add_argument("--shuffle", action="store_true",
                        help="Перемешивать порядок кластеров между циклами")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  E2-КЛАСТЕРНОЕ ОБУЧЕНИЕ — Применение E2 ко ВСЕМ файлам pro2")
    print("═" * 70)

    # ── Загрузка кластеров ────────────────────────────────────────────────────
    print("\n  Загружаю кластеры репозитория...")
    loader = RepoCorpusLoader()
    print(loader.report())

    all_clusters: dict[str, list[str]] = {}
    for name in CLUSTER_ORDER:
        items = loader.get_cluster(name)
        texts = [it["text"] for it in items if len(it["text"]) >= 8]
        all_clusters[name] = texts
        print(f"  {name:<13}: {len(texts)} текстов загружено")

    # ── Создание модели ───────────────────────────────────────────────────────
    print("\n  Создаю модель HierarchicalE2...")
    model = HierarchicalE2(E2_CFG)

    # Загрузка CoreLevel из Variant3
    if not args.no_v3 and V3_CHECKPOINT.exists():
        loaded = model.load_core_from_v3(str(V3_CHECKPOINT))
        if loaded:
            print(f"  ✅ CoreLevel инициализирован из {V3_CHECKPOINT.name}")
        else:
            print("  ⚠️  CoreLevel — случайные веса")
    else:
        print("  ⚠️  CoreLevel — случайные веса")

    # Загрузка из E2-checkpoint (приоритет: clusters > base e2)
    if args.resume:
        ckpt = CLUSTERS_CHECKPOINT if CLUSTERS_CHECKPOINT.exists() else E2_BASE_CHECKPOINT
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"  ✅ Возобновлено из {ckpt.name}")

    print(f"\n{model.describe()}")

    # ── Определение кластеров для обучения ───────────────────────────────────
    clusters_to_run = [args.cluster] if args.cluster else CLUSTER_ORDER

    log: list = []

    # ── Циклы обучения ────────────────────────────────────────────────────────
    for cycle in range(1, args.cycles + 1):
        if args.cycles > 1:
            print(f"\n{'━'*70}")
            print(f"  ЦИКЛ {cycle}/{args.cycles}")
            print(f"{'━'*70}")

        order = list(clusters_to_run)
        if args.shuffle and cycle > 1:
            random.shuffle(order)

        for cluster_name in order:
            if cluster_name not in all_clusters:
                print(f"  ⚠️  Кластер «{cluster_name}» не найден — пропускаем")
                continue

            texts = all_clusters[cluster_name]
            if not texts:
                print(f"  ⚠️  Кластер «{cluster_name}» пуст — пропускаем")
                continue

            defn     = CLUSTER_DEFS[cluster_name]
            e2_phase = defn["e2_phase"]
            steps    = args.steps or (20 if args.fast else CLUSTER_STEPS[cluster_name])
            lr       = CLUSTER_LR[cluster_name]

            train_cluster(
                model, cluster_name, texts,
                steps, lr, e2_phase, log, cycle=cycle,
            )

            # Checkpoint после каждого кластера
            torch.save(model.state_dict(), CLUSTERS_CHECKPOINT)
            print(f"  💾 {CLUSTERS_CHECKPOINT.name} сохранён")

    # ── Сохранение лога ───────────────────────────────────────────────────────
    CLUSTERS_LOG.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  📄 Лог: {CLUSTERS_LOG.name}")

    # ── Итоговый отчёт ────────────────────────────────────────────────────────
    print_summary(log)

    # ── Быстрый тест Q6-эмбеддинга ───────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  ТЕСТ РЕПОЗИТОРНЫХ КОНЦЕПТОВ (Q6-эмбеддинг)")
    print(f"{'═'*70}")
    model.eval()
    test_concepts = [
        "Variant3GPT",
        "HierarchicalE2",
        "NautilusPortal",
        "utils_v52",
        "benchmark_v69",
        "archetypal interlingua",
        "corpus loader",
    ]
    for concept in test_concepts:
        r = model.embed_text(concept)
        q6s = "".join(map(str, r["q6"]))
        print(f"  «{concept:<28}»  Q6=[{q6s}] → гексаграмма #{r['hex_idx']}")


if __name__ == "__main__":
    main()
