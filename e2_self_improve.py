#!/usr/bin/env python3
"""
e2_self_improve.py — Петля самоулучшения HierarchicalE2.

Принцип: модель сама анализирует свои слабые места и целенаправленно улучшается.

Алгоритм:
  1. ДИАГНОСТИКА  — измерить PPL по каждому кластеру/домену
  2. ОТБОР        — выбрать кластеры с наибольшим PPL (слабые места)
  3. ЦЕЛЕВОЕ ОБУЧЕНИЕ — усиленный прогон на слабых кластерах
  4. ВЕРИФИКАЦИЯ  — проверить, улучшился ли PPL
  5. ПОВТОР       — до достижения целевого PPL или max итераций

Дополнительно:
  SYNTHESIS — после улучшения всех слабых мест: синтетический проход
              по всем кластерам для закрепления интеграции

Usage:
  python e2_self_improve.py                    # автоматическое улучшение
  python e2_self_improve.py --iters 5          # 5 итераций диагностики
  python e2_self_improve.py --target-ppl 80    # цель: PPL < 80
  python e2_self_improve.py --fast             # быстрый режим (демо)
  python e2_self_improve.py --diagnose-only    # только диагностика

Горизонтальные связи (↔):
  ↔ train_e2_joint.py     — базовое обучение (перед самоулучшением)
  ↔ repo_corpus_loader.py — внутренние кластеры
  ↔ corpus_loader.py      — внешний корпус
  ↔ e2_inference.py       — проверка Q6-качества после улучшения
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
from corpus_loader import CorpusLoader
from repo_corpus_loader import RepoCorpusLoader, CLUSTER_DEFS

# ── Q4⊂Q6 инициализация RAG (задача 0.6) ─────────────────────────────────────
# 16 PseudoRAG архетипов Q4 ↔ 64 вершины Q6 (avg_hamming=2.56 → PARTIAL)
# Если avg_hamming < 2.5, организуем тексты по Q4-кластерам для RAG init

_Q4_Q6_RESULT_PATH = Path(__file__).parent / "experiments" / "q4_q6_result.json"

# 16 Q4-архетипов: каждый = 4-битная маска, вложена в Q6 через расширение
# Маппинг Q4-индекс (0-15) → Q6-вершина (0-63): первые 4 бита из Q4, биты 4-5 = 0
_Q4_TO_Q6 = {i: i for i in range(16)}  # прямое вложение в первые 16 вершин Q6

# Семантические метки для 16 Q4-архетипов (из validate_q4_q6.py)
_Q4_LABELS = [
    "structure", "pattern", "logic", "sequence",
    "balance", "hierarchy", "transformation", "flow",
    "duality", "cycle", "emergence", "boundary",
    "resonance", "synthesis", "recursion", "wholeness",
]


def load_q4_q6_hamming() -> float:
    """Загружает avg_hamming из предыдущего запуска validate_q4_q6.py."""
    try:
        data = json.loads(_Q4_Q6_RESULT_PATH.read_text())
        # Поддержка обоих форматов результата
        return float(data.get("global_avg_hamming", data.get("avg_hamming", 99.0)))
    except Exception:
        return 99.0  # недоступен → не активируем Q4 init


def q4_cluster_init(texts: list[str], n_per_cluster: int = 10) -> dict[str, list[str]]:
    """Организует тексты по Q4-архетипам на основе ключевых слов.

    Каждый Q4-архетип имеет семантическую метку из _Q4_LABELS.
    Тексты сортируются по совпадению с меткой (простой keyword match).

    Returns:
        dict label → list[text] — по n_per_cluster текстов на архетип
    """
    clusters: dict[str, list[str]] = {label: [] for label in _Q4_LABELS}
    for text in texts:
        text_lower = text.lower()
        best_label = "wholeness"  # fallback
        best_score = 0
        for label in _Q4_LABELS:
            score = sum(1 for kw in label.split("_") if kw in text_lower)
            if score > best_score:
                best_score = score
                best_label = label
        clusters[best_label].append(text)

    # Обрезаем до n_per_cluster
    return {k: v[:n_per_cluster] for k, v in clusters.items() if v}


# ── Конфигурация ──────────────────────────────────────────────────────────────

torch.manual_seed(42)
random.seed(42)

_ROOT = Path(__file__).parent

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
IMPROVE_CHECKPOINT = _ROOT / "checkpoint_e2_improved.pt"
IMPROVE_LOG        = _ROOT / "e2_self_improve_log.json"

# Порог «слабого места»: кластер считается слабым если его PPL > медианы × WEAK_RATIO
WEAK_RATIO = 1.20


# ══════════════════════════════════════════════════════════════════════════════

def encode(text: str, vocab_size: int = 256, block_size: int = 32) -> torch.Tensor:
    ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
    return torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)


def measure_ppl(model: HierarchicalE2, texts: list[str], n: int = 30) -> float:
    """Измеряет PPL модели на выборке текстов."""
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


def targeted_train(
    model:   HierarchicalE2,
    texts:   list[str],
    steps:   int,
    lr:      float,
    phase:   int,
    label:   str,
) -> float:
    """Целевое обучение на конкретном кластере. Возвращает финальный loss."""
    model.set_training_phase(phase)
    model.train()

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    losses = []
    for step in range(1, steps + 1):
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
        scheduler.step()
        losses.append(loss.item())

    return sum(losses[-10:]) / max(1, len(losses[-10:]))


# ══════════════════════════════════════════════════════════════════════════════
# Диагностика
# ══════════════════════════════════════════════════════════════════════════════

class SelfDiagnostics:
    """Диагностирует слабые места модели по кластерам."""

    def __init__(self, fast: bool = False):
        self.fast = fast
        self._data: dict[str, list[str]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Загружает все кластеры (внешние + внутренние)."""
        print("  Загружаю данные для диагностики...")

        # Внешние источники
        try:
            ext = CorpusLoader().as_training_corpus(
                max_per_source=30 if self.fast else 100)
            by_src: dict[str, list] = {}
            for it in ext:
                s = it["source"]
                by_src.setdefault(s, [])
                by_src[s].append(it["text"])
            for src, texts in by_src.items():
                self._data[f"ext/{src}"] = texts
            print(f"    Внешний: {sum(len(v) for v in by_src.values())} текстов, "
                  f"{len(by_src)} источников")
        except Exception as e:
            print(f"    ⚠️  Внешний корпус: {e}")

        # Внутренние кластеры
        try:
            loader = RepoCorpusLoader()
            for name in CLUSTER_DEFS:
                items = loader.get_cluster(name)
                texts = [it["text"] for it in items if len(it["text"]) >= 8]
                if texts:
                    self._data[f"repo/{name}"] = texts
            total_int = sum(len(v) for k, v in self._data.items()
                           if k.startswith("repo/"))
            print(f"    Внутренний: {total_int} текстов, "
                  f"{sum(1 for k in self._data if k.startswith('repo/'))} кластеров")
        except Exception as e:
            print(f"    ⚠️  Внутренние кластеры: {e}")

        # ── Q4⊂Q6 инициализация (задача 0.6) ────────────────────────────────
        # Если avg_hamming < 2.5 → активируем Q4-кластерную организацию данных
        avg_hamming = load_q4_q6_hamming()
        if avg_hamming < 2.5:
            try:
                all_texts = [t for texts in self._data.values() for t in texts]
                q4_clusters = q4_cluster_init(all_texts,
                                              n_per_cluster=15 if self.fast else 40)
                n_added = 0
                for label, texts in q4_clusters.items():
                    key = f"q4/{label}"
                    if key not in self._data:
                        self._data[key] = texts
                        n_added += len(texts)
                print(f"    Q4⊂Q6: avg_hamming={avg_hamming:.2f} < 2.5 → "
                      f"добавлено {n_added} текстов по {len(q4_clusters)} архетипам")
            except Exception as e:
                print(f"    ⚠️  Q4⊂Q6 init: {e}")
        else:
            print(f"    Q4⊂Q6: avg_hamming={avg_hamming:.2f} ≥ 2.5 → Q4 init пропущен")

    def diagnose(self, model: HierarchicalE2,
                 n_per_cluster: int = 20) -> dict[str, float]:
        """Измеряет PPL для каждого кластера."""
        print(f"\n  {'─'*60}")
        print(f"  ДИАГНОСТИКА ({len(self._data)} кластеров × {n_per_cluster} текстов)")
        print(f"  {'─'*60}")

        ppls: dict[str, float] = {}
        for label, texts in sorted(self._data.items()):
            if not texts:
                continue
            ppl = measure_ppl(model, texts, n=n_per_cluster)
            ppls[label] = ppl
            bar = "█" * min(20, int(ppl / 20))
            print(f"  {label:<25}  PPL={ppl:>8.2f}  {bar}")

        return ppls

    def find_weak(self, ppls: dict[str, float],
                  ratio: float = WEAK_RATIO) -> list[str]:
        """Находит кластеры с PPL > медиана × ratio."""
        if not ppls:
            return []
        vals = sorted(ppls.values())
        median = vals[len(vals) // 2]
        threshold = median * ratio
        weak = [label for label, ppl in ppls.items() if ppl > threshold]
        weak.sort(key=lambda l: -ppls[l])
        print(f"\n  Медиана PPL: {median:.2f}  Порог: {threshold:.2f}")
        print(f"  Слабых мест: {len(weak)}/{len(ppls)}")
        for label in weak:
            print(f"    ⚠️  {label:<25}  PPL={ppls[label]:.2f}")
        return weak

    def get_texts(self, label: str) -> list[str]:
        return self._data.get(label, [])

    def get_phase(self, label: str) -> int:
        """Определяет E2-фазу для кластера по метке."""
        if "repo/" in label:
            cluster = label.split("/")[-1]
            return CLUSTER_DEFS.get(cluster, {}).get("e2_phase", 3)
        # Внешние: по α-уровню источника (приблизительно)
        src = label.split("/")[-1]
        alpha_map = {"data7": 4, "info1": 3, "meta": 3,
                     "data2": 2, "infosystems": 1, "ai_agents": 2,
                     "knowledge": 3, "meta_corpus": 3}
        alpha = alpha_map.get(src, 0)
        return {-4:1,-2:1,0:3,1:3,2:3,3:4,4:5}.get(alpha, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Петля самоулучшения
# ══════════════════════════════════════════════════════════════════════════════

def self_improve_loop(
    model:       HierarchicalE2,
    diag:        SelfDiagnostics,
    max_iters:   int,
    target_ppl:  float,
    fast:        bool,
    log:         list,
) -> None:
    steps_per_weak = 20 if fast else 80
    lr_weak        = 2e-4

    global_ppl_history = []

    for iteration in range(1, max_iters + 1):
        print(f"\n{'━'*66}")
        print(f"  ИТЕРАЦИЯ САМОУЛУЧШЕНИЯ {iteration}/{max_iters}")
        print(f"{'━'*66}")

        # ── Диагностика ──────────────────────────────────────────────────────
        n_diag = 15 if fast else 25
        ppls   = diag.diagnose(model, n_per_cluster=n_diag)
        if not ppls:
            break

        avg_ppl = sum(ppls.values()) / len(ppls)
        global_ppl_history.append(avg_ppl)
        print(f"\n  Средний PPL: {avg_ppl:.2f}")

        if avg_ppl <= target_ppl:
            print(f"  🎯 Цель достигнута! PPL={avg_ppl:.2f} ≤ {target_ppl}")
            break

        # ── Выбор слабых мест ────────────────────────────────────────────────
        weak_labels = diag.find_weak(ppls)
        if not weak_labels:
            print("  Слабых мест не найдено — все кластеры сбалансированы ✅")
            break

        # ── Целевое обучение на слабых кластерах ────────────────────────────
        iter_results = []
        for label in weak_labels[:4]:   # не более 4 слабых мест за итерацию
            texts = diag.get_texts(label)
            if not texts:
                continue
            phase = diag.get_phase(label)

            ppl_before = ppls[label]
            print(f"\n  Улучшаю «{label}» (фаза={phase}, шагов={steps_per_weak})...")
            final_loss = targeted_train(model, texts, steps_per_weak,
                                        lr_weak, phase, label)
            ppl_after  = measure_ppl(model, texts, n=15)
            delta = (ppl_before - ppl_after) / ppl_before * 100 if ppl_before else 0
            sign  = "✅" if delta > 0 else "⚠️ "
            print(f"  PPL: {ppl_before:.2f} → {ppl_after:.2f}  Δ={delta:+.1f}%  {sign}")

            iter_results.append({
                "label":      label,
                "phase":      phase,
                "ppl_before": round(ppl_before, 2),
                "ppl_after":  round(ppl_after, 2),
                "ppl_delta":  round(delta, 2),
                "final_loss": round(final_loss, 4),
            })

        log.append({
            "iteration":       iteration,
            "avg_ppl":         round(avg_ppl, 2),
            "target_ppl":      target_ppl,
            "weak_clusters":   weak_labels[:4],
            "improvements":    iter_results,
        })

        # ── Сохраняем прогресс ───────────────────────────────────────────────
        torch.save(model.state_dict(), IMPROVE_CHECKPOINT)
        print(f"\n  💾 {IMPROVE_CHECKPOINT.name} сохранён")

    # ── Синтетический закрепляющий проход ────────────────────────────────────
    print(f"\n{'─'*66}")
    print("  SYNTHESIS — финальный закрепляющий проход")
    print(f"{'─'*66}")
    model.set_training_phase(5)  # все параметры
    model.train()

    all_texts = []
    for texts in diag._data.values():
        all_texts.extend(random.sample(texts, min(5, len(texts))))
    random.shuffle(all_texts)

    synth_steps = 30 if fast else 100
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=1e-4,
    )
    losses = []
    for step in range(1, synth_steps + 1):
        text   = random.choice(all_texts)
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

    if losses:
        avg = sum(losses[-10:]) / len(losses[-10:])
        print(f"  Synthesis loss: {avg:.4f}")

    torch.save(model.state_dict(), IMPROVE_CHECKPOINT)
    print(f"  💾 Финальный checkpoint: {IMPROVE_CHECKPOINT.name}")

    # ── Итоговый отчёт ───────────────────────────────────────────────────────
    if global_ppl_history:
        delta_total = (global_ppl_history[0] - global_ppl_history[-1])
        delta_pct   = delta_total / global_ppl_history[0] * 100
        print(f"\n{'═'*66}")
        print(f"  ИТОГ САМОУЛУЧШЕНИЯ")
        print(f"  PPL: {global_ppl_history[0]:.2f} → {global_ppl_history[-1]:.2f}  "
              f"(Δ{delta_pct:+.1f}%)")
        print(f"  Итераций пройдено: {len(global_ppl_history)}")


# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E2 Self-Improvement Loop")
    parser.add_argument("--iters",         type=int, default=3,
                        help="Максимум итераций диагностики")
    parser.add_argument("--target-ppl",    type=float, default=90.0,
                        help="Целевой PPL (остановиться при достижении)")
    parser.add_argument("--fast",          action="store_true",
                        help="Быстрый режим: меньше шагов")
    parser.add_argument("--diagnose-only", action="store_true",
                        help="Только диагностика, без обучения")
    parser.add_argument("--resume",        action="store_true",
                        help="Загрузить checkpoint_e2_improved.pt")
    parser.add_argument("--no-v3",         action="store_true")
    args = parser.parse_args()

    print("\n" + "═"*66)
    print("  E2 SELF-IMPROVEMENT — Петля самоулучшения HierarchicalE2")
    print("═"*66)

    # ── Данные ───────────────────────────────────────────────────────────────
    diag = SelfDiagnostics(fast=args.fast)

    # ── Модель ───────────────────────────────────────────────────────────────
    print("\n  Загружаю модель...")
    model = HierarchicalE2(E2_CFG)

    from yijing_transformer.models.variant3 import Variant3Config
    if not args.no_v3 and V3_CHECKPOINT.exists():
        loaded = model.load_core_from_v3(str(V3_CHECKPOINT))
        if loaded:
            print(f"  ✅ CoreLevel из {V3_CHECKPOINT.name}")

    # Загружаем лучший доступный checkpoint
    if args.resume:
        candidates = [IMPROVE_CHECKPOINT, JOINT_CHECKPOINT, CLUSTERS_CHECKPOINT,
                      E2_CHECKPOINT]
    else:
        candidates = [JOINT_CHECKPOINT, CLUSTERS_CHECKPOINT, E2_CHECKPOINT]

    for ckpt in candidates:
        if ckpt.exists():
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"  ✅ Загружен: {ckpt.name}")
            break

    print(f"\n{model.describe()}")

    log: list = []

    if args.diagnose_only:
        ppls = diag.diagnose(model, n_per_cluster=25)
        weak = diag.find_weak(ppls)
        avg  = sum(ppls.values()) / len(ppls) if ppls else 0
        print(f"\n  Средний PPL: {avg:.2f}")
        print(f"  Слабые места ({len(weak)}): {weak[:5]}")
        return

    # ── Петля самоулучшения ───────────────────────────────────────────────────
    self_improve_loop(
        model, diag,
        max_iters=args.iters,
        target_ppl=args.target_ppl,
        fast=args.fast,
        log=log,
    )

    # ── Сохранение лога ──────────────────────────────────────────────────────
    IMPROVE_LOG.write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  📄 Лог: {IMPROVE_LOG.name}")

    # ── Q6-тест после улучшения ───────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print("  Q6-ТЕСТ ПОСЛЕ САМОУЛУЧШЕНИЯ")
    print(f"{'═'*66}")
    model.eval()
    for concept in ["кристалл", "гексаграмма", "трансформация", "HierarchicalE2"]:
        r = model.embed_text(concept)
        q6s = "".join(map(str, r["q6"]))
        print(f"  «{concept:<28}»  Q6=[{q6s}] →#{r['hex_idx']}")


V3_CHECKPOINT = _ROOT / "checkpoint_bidir_v2.pt"

if __name__ == "__main__":
    main()
