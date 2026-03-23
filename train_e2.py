#!/usr/bin/env python3
"""
train_e2.py — Обучение HierarchicalE2 снизу вверх по α-уровням.

Стратегия: каждый уровень активируется только когда предыдущий стабилизировался.
  Фаза 1 (α=−4): GlyphLevel         — Q6-проекция + кластеризация
  Фаза 2 (α=−2): CoreLevel          — Variant3 блоки (загрузка из checkpoint)
  Фаза 3 (α= 0): MethodLevel        — ArchetypalInterlingua
  Фаза 4 (α=+2): TheoryLevel        — NautilusHierarchy
  Фаза 5 (α=+4): PhiloLevel         — ConvergenceBridge + MatrixGrammar

Данные: corpus_loader.py (2448 файлов из 8 репозиториев)

Usage:
  python train_e2.py                    # полное обучение все фазы
  python train_e2.py --phase 1          # только фаза 1
  python train_e2.py --phase 1 --steps 100  # фаза 1, 100 шагов
  python train_e2.py --fast             # 30 шагов на фазу (демо)
  python train_e2.py --resume           # продолжить с последнего checkpoint
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
from corpus_loader import CorpusLoader

# ── Конфигурация ──────────────────────────────────────────────────────────────

torch.manual_seed(42)
random.seed(42)

_ROOT  = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры модели
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
    nautilus_chambers=None,   # все 7 камер
    conv_window=4,
    conv_stride=2,
    grammar_rows=8,
    grammar_cols=8,
)

# Шагов на фазу (полное обучение)
PHASE_STEPS = {
    1: 200,   # α=−4 быстро — простая кластеризация
    2: 300,   # α=−2 больше — ядро самое важное
    3: 250,   # α= 0 интерлингва
    4: 200,   # α=+2 наутилус
    5: 150,   # α=+4 философия
}

PHASE_LR = {
    1: 3e-4,
    2: 1e-4,  # меньше — ядро уже предобучено
    3: 2e-4,
    4: 1e-4,
    5: 5e-5,  # самый верхний уровень — осторожно
}

PHASE_NAMES = {
    1: "GlyphLevel   α=−4",
    2: "CoreLevel    α=−2",
    3: "MethodLevel  α= 0",
    4: "TheoryLevel  α=+2",
    5: "PhiloLevel   α=+4",
}

V3_CHECKPOINT = _ROOT / "checkpoint_bidir_v2.pt"
E2_CHECKPOINT = _ROOT / "checkpoint_e2.pt"
E2_LOG        = _ROOT / "train_e2_log.json"


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка корпуса
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(max_per_source: int = 150) -> list[str]:
    """Загружает тексты из всех доступных репозиториев."""
    loader = CorpusLoader()
    items  = loader.as_training_corpus(max_per_source=max_per_source)
    texts  = [it["text"] for it in items if len(it["text"]) >= 8]
    random.shuffle(texts)
    print(f"  Корпус: {len(texts)} текстов из {len(set(it['source'] for it in items))} источников")
    return texts


def encode(text: str, vocab_size: int = 256, block_size: int = 32) -> torch.Tensor:
    """Байт-кодирование текста → токены [1, T]."""
    ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
    if not ids:
        ids = [32]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def perplexity(model: HierarchicalE2, texts: list[str],
               n: int = 30) -> float:
    """Средняя перплексия на выборке текстов."""
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
# Обучение одной фазы
# ══════════════════════════════════════════════════════════════════════════════

def train_phase(
    model:  HierarchicalE2,
    phase:  int,
    texts:  list[str],
    steps:  int,
    lr:     float,
    log:    list,
) -> dict:
    """Обучает одну фазу. Возвращает dict с результатами."""
    print(f"\n{'═'*66}")
    print(f"  ФАЗА {phase}: {PHASE_NAMES[phase]}")
    print(f"{'═'*66}")

    model.set_training_phase(phase)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    ppl_before = perplexity(model, texts)
    print(f"  PPL до фазы {phase}: {ppl_before:.2f}")
    print(f"  Шагов: {steps}  LR: {lr}\n")

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

        if step % max(1, steps // 10) == 0 or step == steps:
            avg_l = sum(losses[-20:]) / max(1, len(losses[-20:]))
            elapsed = time.time() - t0
            # Q6 диагностика
            q6 = info["q6_coords"][0, 0].detach()
            q6_bin = (q6 > 0).int().tolist()
            hex_i  = sum(b << i for i, b in enumerate(q6_bin))
            print(f"  шаг {step:>4}/{steps}  loss={avg_l:.4f}  "
                  f"Q6={''.join(map(str,q6_bin))} →#{hex_i:<2}  "
                  f"({elapsed:.0f}s)")
            log_rows.append({
                "step": step, "loss": avg_l, "hex_idx": hex_i,
            })

    ppl_after = perplexity(model, texts)
    delta = (ppl_before - ppl_after) / ppl_before * 100 if ppl_before < 1e6 else 0
    avg_final = sum(losses[-30:]) / max(1, len(losses[-30:]))

    result = {
        "phase":      phase,
        "name":       PHASE_NAMES[phase],
        "steps":      steps,
        "lr":         lr,
        "ppl_before": round(ppl_before, 2),
        "ppl_after":  round(ppl_after, 2),
        "ppl_delta":  round(delta, 2),
        "final_loss": round(avg_final, 4),
        "log":        log_rows,
    }
    log.append(result)

    sign = "✅" if delta > 0 else "⚠️ "
    print(f"\n  PPL после: {ppl_after:.2f}  (Δ{delta:+.1f}%)  {sign}")
    print(f"  Финальный loss: {avg_final:.4f}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Сводный отчёт
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(log: list, model: HierarchicalE2) -> None:
    print(f"\n{'═'*66}")
    print("  ИТОГИ ОБУЧЕНИЯ HierarchicalE2 (Вариант Е2)")
    print(f"{'═'*66}")
    print(f"  {'Фаза':<5} {'Уровень':<22} {'PPL до':>8} {'PPL после':>10} {'Δ':>8}")
    print("  " + "─"*56)
    for r in log:
        sign = "⬇️ " if r["ppl_delta"] > 0 else "⬆️ "
        print(f"  {r['phase']:<5} {r['name']:<22} "
              f"{r['ppl_before']:>8.2f} {r['ppl_after']:>10.2f} "
              f"  {sign}{abs(r['ppl_delta']):.1f}%")
    if log:
        total = (log[0]["ppl_before"] - log[-1]["ppl_after"]) / log[0]["ppl_before"] * 100
        print(f"\n  Суммарное улучшение PPL: {total:+.1f}%  "
              f"({log[0]['ppl_before']:.1f} → {log[-1]['ppl_after']:.1f})")
    print(f"\n{model.describe()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Обучение HierarchicalE2")
    parser.add_argument("--phase",  type=int, default=0,
                        help="Запустить только фазу N (0 = все фазы)")
    parser.add_argument("--steps",  type=int, default=0,
                        help="Переопределить число шагов")
    parser.add_argument("--fast",   action="store_true",
                        help="Быстрый режим: 30 шагов на фазу")
    parser.add_argument("--resume", action="store_true",
                        help="Загрузить checkpoint_e2.pt если есть")
    parser.add_argument("--no-v3",  action="store_true",
                        help="Не загружать веса Variant3 для CoreLevel")
    args = parser.parse_args()

    print("\n" + "═"*66)
    print("  HIERARCHICAL E2 — Вариант Е2 (5 уровней α)")
    print("═"*66)

    # ── Загрузка корпуса ─────────────────────────────────────────────────────
    print("\n  Загружаю корпус...")
    texts = load_corpus(max_per_source=100 if args.fast else 200)
    if not texts:
        print("  ❌ Корпус пуст — запустите corpus_loader.py --stats")
        return

    # ── Создание модели ──────────────────────────────────────────────────────
    print("\n  Создаю модель...")
    model = HierarchicalE2(E2_CFG)

    # Загрузка CoreLevel из Variant3 checkpoint
    if not args.no_v3 and V3_CHECKPOINT.exists():
        loaded = model.load_core_from_v3(str(V3_CHECKPOINT))
        if loaded:
            print(f"  ✅ CoreLevel инициализирован из {V3_CHECKPOINT.name}")
        else:
            print("  ⚠️  CoreLevel — случайные веса")
    else:
        print("  ⚠️  CoreLevel — случайные веса (--no-v3 или checkpoint не найден)")

    # Загрузка E2 checkpoint (resume)
    if args.resume and E2_CHECKPOINT.exists():
        state = torch.load(E2_CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"  ✅ Возобновлено из {E2_CHECKPOINT.name}")

    print(f"\n{model.describe()}")

    # ── Определение фаз для обучения ────────────────────────────────────────
    phases_to_run = [args.phase] if args.phase > 0 else list(range(1, 6))

    log: list = []

    # ── Обучение ─────────────────────────────────────────────────────────────
    for phase in phases_to_run:
        n_steps = args.steps or (30 if args.fast else PHASE_STEPS[phase])
        lr      = PHASE_LR[phase]

        train_phase(model, phase, texts, n_steps, lr, log)

        # Сохраняем checkpoint после каждой фазы
        torch.save(model.state_dict(), E2_CHECKPOINT)
        print(f"  💾 checkpoint_e2.pt сохранён")

    # ── Сохранение лога ──────────────────────────────────────────────────────
    E2_LOG.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  📄 Лог: {E2_LOG.name}")

    # ── Итоговый отчёт ───────────────────────────────────────────────────────
    print_summary(log, model)

    # ── Тест нескольких запросов ─────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print("  ТЕСТ ЗАПРОСОВ (Q6-эмбеддинг)")
    print(f"{'═'*66}")
    model.eval()
    test_queries = [
        "кристалл", "знание", "гексаграмма",
        "трансформация", "поток воды", "философия",
    ]
    for q in test_queries:
        r   = model.embed_text(q)
        q6s = "".join(map(str, r["q6"]))
        print(f"  «{q:<22}»  Q6=[{q6s}] → гексаграмма #{r['hex_idx']}")


if __name__ == "__main__":
    main()
