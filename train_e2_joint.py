#!/usr/bin/env python3
"""
train_e2_joint.py — Объединённое E2-обучение: внешний корпус + внутренние кластеры.

Стратегия: каждый обучающий шаг берёт текст из СМЕШАННОГО потока:
  - 40% внешний корпус (corpus_loader: data7, info1, meta, data2, ...)
  - 60% внутренние кластеры (repo_corpus_loader: Theory, Models, Training, ...)

Распределение по α-уровням:
  Внешний:   α=-4..+4 → E2-фазы 1..5
  Scripts:   α=-4     → фаза 1 (GlyphLevel)
  Benchmarks:α=-2     → фаза 2 (CoreLevel)
  Training:  α= 0     → фаза 3 (MethodLevel)
  Models:    α=+2     → фаза 4 (TheoryLevel)
  Theory:    α=+4     → фаза 5 (PhiloLevel)

Режимы:
  UNIFIED  — все источники вместе, фаза 5 (все параметры активны)
  CASCADE  — снизу вверх по α: сначала фаза 1, потом 2, ..., 5
  DOMAIN   — поочерёдно по доменам (GEO→HYDRO→PYRO→AERO→COSMO→NOOS)

Usage:
  python train_e2_joint.py                    # CASCADE режим, 1 цикл
  python train_e2_joint.py --mode unified     # всё вместе, 400 шагов
  python train_e2_joint.py --mode domain      # по доменам
  python train_e2_joint.py --cycles 3         # 3 прохода CASCADE
  python train_e2_joint.py --fast             # 30 шагов (демо)
  python train_e2_joint.py --resume           # из checkpoint_e2_joint.pt
"""

import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
from corpus_loader import CorpusLoader
from repo_corpus_loader import RepoCorpusLoader, CLUSTER_DEFS

# ── Конфигурация ──────────────────────────────────────────────────────────────

torch.manual_seed(42)
random.seed(42)

_ROOT  = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

E2_CFG = E2Config(
    vocab_size=256, d_model=128, block_size=32,
    n_core=4, n_heads=4, dropout=0.05,
    hamming_lambda=0.15, uncertainty_budget=0.25, ffn_mult=4,
    n_archetypes=64, il_use_ternary=True,
    nautilus_warmup=200, nautilus_mode="sequential", nautilus_chambers=None,
    conv_window=4, conv_stride=2, grammar_rows=8, grammar_cols=8,
)

V3_CHECKPOINT      = _ROOT / "checkpoint_bidir_v2.pt"
E2_CHECKPOINT      = _ROOT / "checkpoint_e2.pt"
CLUSTERS_CHECKPOINT = _ROOT / "checkpoint_e2_clusters.pt"
JOINT_CHECKPOINT   = _ROOT / "checkpoint_e2_joint.pt"
JOINT_LOG          = _ROOT / "train_e2_joint_log.json"

# α → E2-фаза
ALPHA_TO_PHASE = {-4: 1, -3: 1, -2: 2, -1: 2, 0: 3, 1: 3, 2: 4, 3: 4, 4: 5}

DOMAIN_ORDER = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS", "METHOD", "YIJING"]

# Шагов на фазу в CASCADE-режиме
CASCADE_STEPS = {1: 150, 2: 150, 3: 200, 4: 150, 5: 100}
CASCADE_LR    = {1: 3e-4, 2: 2e-4, 3: 2e-4, 4: 1e-4, 5: 5e-5}


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка данных
# ══════════════════════════════════════════════════════════════════════════════

class JointDataset:
    """
    Объединяет внешний корпус и внутренние кластеры в один поток.
    Каждый элемент: {"text", "domain", "alpha", "source"}
    """

    def __init__(self, ext_ratio: float = 0.4, max_ext: int = 150, max_int: int = 300):
        self.ext_ratio = ext_ratio
        print("  Загружаю внешний корпус...")
        ext_items = CorpusLoader().as_training_corpus(max_per_source=max_ext)
        self.external = [it for it in ext_items if len(it["text"]) >= 8]
        print(f"    {len(self.external)} текстов из внешних репо")

        print("  Загружаю внутренние кластеры...")
        loader = RepoCorpusLoader()
        int_items = loader.as_flat_corpus()
        self.internal = [it for it in int_items if len(it["text"]) >= 8]
        print(f"    {len(self.internal)} текстов из внутренних кластеров")

        # Группировка по α
        self.by_alpha: dict[int, list] = defaultdict(list)
        for it in self.external + self.internal:
            self.by_alpha[it["alpha"]].append(it)

        # Группировка по домену
        self.by_domain: dict[str, list] = defaultdict(list)
        for it in self.external + self.internal:
            self.by_domain[it["domain"]].append(it)

        print(f"  Итого: {len(self.external) + len(self.internal)} текстов")
        print(f"  Распределение по α: " +
              "  ".join(f"α={a:+d}:{len(v)}" for a, v in sorted(self.by_alpha.items())))

    def sample(self, n: int = 1) -> list[dict]:
        """Случайная выборка с учётом соотношения ext/int."""
        result = []
        for _ in range(n):
            if random.random() < self.ext_ratio and self.external:
                result.append(random.choice(self.external))
            elif self.internal:
                result.append(random.choice(self.internal))
        return result

    def sample_alpha(self, alpha: int, n: int = 1) -> list[dict]:
        """Выборка только для конкретного α-уровня."""
        pool = self.by_alpha.get(alpha, [])
        if not pool:
            # Ближайший α
            for da in range(1, 5):
                pool = self.by_alpha.get(alpha + da, self.by_alpha.get(alpha - da, []))
                if pool:
                    break
        return random.choices(pool, k=n) if pool else []

    def sample_domain(self, domain: str, n: int = 1) -> list[dict]:
        """Выборка только для конкретного домена."""
        pool = self.by_domain.get(domain, [])
        return random.choices(pool, k=n) if pool else []

    def all_texts(self, alpha: int | None = None, domain: str | None = None) -> list[str]:
        """Все тексты (с фильтрацией по α или домену)."""
        pool = self.external + self.internal
        if alpha is not None:
            pool = [it for it in pool if it["alpha"] == alpha]
        if domain is not None:
            pool = [it for it in pool if it["domain"] == domain]
        return [it["text"] for it in pool]


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты
# ══════════════════════════════════════════════════════════════════════════════

def encode(text: str, vocab_size: int = 256, block_size: int = 32) -> torch.Tensor:
    ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
    return torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)


def perplexity(model: HierarchicalE2, texts: list[str], n: int = 30) -> float:
    model.eval()
    ppls = []
    for text in random.sample(texts, min(n, len(texts))):
        tokens = encode(text, E2_CFG.vocab_size, E2_CFG.block_size)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            _, loss, _ = model(tokens[:, :-1], targets=tokens[:, 1:])
        if loss is not None and not torch.isnan(loss):
            ppls.append(math.exp(min(loss.item(), 10)))
    return sum(ppls) / len(ppls) if ppls else float("inf")


def train_loop(
    model: HierarchicalE2,
    items: list[dict],
    steps: int,
    lr: float,
    phase: int,
    label: str,
) -> dict:
    """Базовый обучающий цикл."""
    texts = [it["text"] for it in items if len(it["text"]) >= 8]
    if not texts:
        return {}

    model.set_training_phase(phase)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    ppl_before = perplexity(model, texts)
    print(f"\n  [{label}]  фаза={phase}  текстов={len(texts)}  "
          f"PPL до={ppl_before:.2f}")

    losses = []
    t0 = time.time()

    for step in range(1, steps + 1):
        text   = random.choice(texts)
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue

        _, loss, info = model(tokens[:, :-1], targets=tokens[:, 1:])
        if loss is None or torch.isnan(loss):
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        scheduler.step()
        losses.append(loss.item())

        if step % max(1, steps // 4) == 0 or step == steps:
            avg_l = sum(losses[-20:]) / len(losses[-20:])
            q6 = info["q6_coords"][0, 0].detach()
            q6_bin = (q6 > 0).int().tolist()
            print(f"    шаг {step:>4}/{steps}  loss={avg_l:.4f}  "
                  f"Q6={''.join(map(str,q6_bin))}  ({time.time()-t0:.0f}s)")

    ppl_after  = perplexity(model, texts)
    delta = (ppl_before - ppl_after) / ppl_before * 100 if ppl_before < 1e6 else 0
    avg_final  = sum(losses[-20:]) / max(1, len(losses[-20:]))
    sign = "✅" if delta > 0 else "⚠️ "
    print(f"  [{label}]  PPL после={ppl_after:.2f}  Δ={delta:+.1f}%  {sign}")

    return {
        "label": label, "phase": phase, "texts": len(texts), "steps": steps,
        "ppl_before": round(ppl_before, 2), "ppl_after": round(ppl_after, 2),
        "ppl_delta": round(delta, 2), "final_loss": round(avg_final, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Режимы обучения
# ══════════════════════════════════════════════════════════════════════════════

def mode_cascade(model: HierarchicalE2, ds: JointDataset,
                 steps_mult: float, log: list, cycle: int) -> None:
    """Снизу вверх по α-уровням, смешивая внешние и внутренние данные."""
    print(f"\n  ── CASCADE (цикл {cycle}) ──────────────────────────────")
    for phase in range(1, 6):
        # Находим α для этой фазы
        alpha = {1: -4, 2: -2, 3: 0, 4: 2, 5: 4}[phase]
        items = ds.sample_alpha(alpha, n=200)
        if not items:
            continue
        steps = int(CASCADE_STEPS[phase] * steps_mult)
        lr    = CASCADE_LR[phase]
        r = train_loop(model, items, steps, lr, phase,
                       f"cascade/phase{phase}/α={alpha:+d}")
        if r:
            r["mode"] = "cascade"
            r["cycle"] = cycle
            log.append(r)


def mode_unified(model: HierarchicalE2, ds: JointDataset,
                 steps: int, log: list) -> None:
    """Все источники вместе, полная активация (фаза 5)."""
    print(f"\n  ── UNIFIED ({steps} шагов) ──────────────────────────────")
    items = ds.sample(n=steps)
    r = train_loop(model, items, steps, lr=1e-4, phase=5,
                   label="unified/all-sources")
    if r:
        r["mode"] = "unified"
        log.append(r)


def mode_domain(model: HierarchicalE2, ds: JointDataset,
                steps_per_domain: int, log: list, cycle: int) -> None:
    """По доменам: GEO→HYDRO→PYRO→AERO→COSMO→NOOS."""
    print(f"\n  ── DOMAIN (цикл {cycle}) ──────────────────────────────")
    domain_phase = {
        "GEO": 2, "HYDRO": 1, "PYRO": 4, "AERO": 3,
        "COSMO": 4, "NOOS": 5, "METHOD": 3, "YIJING": 5,
    }
    for domain in DOMAIN_ORDER:
        items = ds.sample_domain(domain, n=100)
        if not items:
            continue
        phase = domain_phase.get(domain, 3)
        r = train_loop(model, items, steps_per_domain, lr=1e-4, phase=phase,
                       label=f"domain/{domain}")
        if r:
            r["mode"] = "domain"
            r["cycle"] = cycle
            log.append(r)


# ══════════════════════════════════════════════════════════════════════════════

def print_summary(log: list) -> None:
    if not log:
        return
    print(f"\n{'═'*70}")
    print("  ИТОГИ — E2 ОБЪЕДИНЁННОЕ ОБУЧЕНИЕ")
    print(f"{'═'*70}")
    improvements = [r["ppl_delta"] for r in log if r and "ppl_delta" in r]
    if improvements:
        avg_imp = sum(improvements) / len(improvements)
        best    = max(improvements)
        print(f"  Пройдено шагов: {len(log)} блоков")
        print(f"  Средний Δ PPL: {avg_imp:+.1f}%   Лучший: {best:+.1f}%")

    # По режимам
    modes = {}
    for r in log:
        if not r:
            continue
        m = r.get("mode", "?")
        if m not in modes:
            modes[m] = []
        modes[m].append(r["ppl_delta"])
    for m, deltas in modes.items():
        avg = sum(deltas) / len(deltas)
        print(f"  Режим {m:<10}: {len(deltas)} блоков  ср.Δ={avg:+.1f}%")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Joint E2 Training")
    parser.add_argument("--mode",    default="cascade",
                        choices=["cascade", "unified", "domain", "all"],
                        help="Режим обучения")
    parser.add_argument("--cycles",  type=int, default=1)
    parser.add_argument("--steps",   type=int, default=0,
                        help="Шагов для unified-режима")
    parser.add_argument("--fast",    action="store_true",
                        help="Быстрый режим (0.15× шагов)")
    parser.add_argument("--resume",  action="store_true",
                        help="Загрузить checkpoint_e2_joint.pt (или clusters/e2)")
    parser.add_argument("--no-v3",   action="store_true")
    args = parser.parse_args()

    print("\n" + "═"*70)
    print("  E2 JOINT TRAINING — Объединённое обучение pro2")
    print("═"*70)

    # ── Данные ───────────────────────────────────────────────────────────────
    ds = JointDataset(ext_ratio=0.4, max_ext=100 if args.fast else 200)

    # ── Модель ───────────────────────────────────────────────────────────────
    print("\n  Создаю модель...")
    model = HierarchicalE2(E2_CFG)

    if not args.no_v3 and V3_CHECKPOINT.exists():
        loaded = model.load_core_from_v3(str(V3_CHECKPOINT))
        if loaded:
            print(f"  ✅ CoreLevel из {V3_CHECKPOINT.name}")

    if args.resume:
        # Приоритет: joint > clusters > e2
        for ckpt in [JOINT_CHECKPOINT, CLUSTERS_CHECKPOINT, E2_CHECKPOINT]:
            if ckpt.exists():
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=False)
                print(f"  ✅ Возобновлено из {ckpt.name}")
                break

    steps_mult = 0.15 if args.fast else 1.0

    log: list = []

    # ── Обучение ─────────────────────────────────────────────────────────────
    modes_to_run = (
        ["cascade", "unified", "domain"] if args.mode == "all"
        else [args.mode]
    )

    for cycle in range(1, args.cycles + 1):
        if args.cycles > 1:
            print(f"\n{'━'*70}\n  ЦИКЛ {cycle}/{args.cycles}\n{'━'*70}")

        for mode in modes_to_run:
            if mode == "cascade":
                mode_cascade(model, ds, steps_mult, log, cycle)
            elif mode == "unified":
                n = args.steps or int(400 * steps_mult)
                mode_unified(model, ds, n, log)
            elif mode == "domain":
                n = int(60 * steps_mult)
                mode_domain(model, ds, n, log, cycle)

        torch.save(model.state_dict(), JOINT_CHECKPOINT)
        print(f"\n  💾 {JOINT_CHECKPOINT.name} сохранён")

    # ── Лог ──────────────────────────────────────────────────────────────────
    JOINT_LOG.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  📄 Лог: {JOINT_LOG.name}")

    print_summary(log)

    # ── Q6-тест ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  Q6-ТЕСТ ОБЪЕДИНЁННОГО ЗНАНИЯ")
    print(f"{'═'*70}")
    model.eval()
    test_items = (
        ("кристалл", "GEO"),
        ("гексаграмма", "YIJING"),
        ("трансформация знаний", "NOOS"),
        ("HierarchicalE2", "MODEL"),
        ("nautilus portal", "COSMO"),
        ("utils_v52", "TRAINING"),
        ("corpus loader", "METHOD"),
        ("benchmark_v69", "GEO"),
    )
    for text, domain in test_items:
        r = model.embed_text(text)
        q6s = "".join(map(str, r["q6"]))
        print(f"  «{text:<28}» [{domain:<9}]  Q6=[{q6s}] →#{r['hex_idx']}")


if __name__ == "__main__":
    main()
