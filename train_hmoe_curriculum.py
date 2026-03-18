#!/usr/bin/env python3
"""
train_hmoe_curriculum.py — Обучение HierarchicalMoE от простого к сложному.

Учебный план (curriculum) по α-уровням:

  Фаза 1  α=−4,−2  КОНКРЕТНОЕ   Scripts, Data           → CONCRETE micro_experts
  Фаза 2  α= 0     ДИНАМИЧЕСКОЕ Self, Training           → crossing + DYNAMIC micro_experts
  Фаза 3  α=+2,+4  АБСТРАКТНОЕ  Models, Theory, Portal  → ABSTRACT micro_experts
  Фаза 4  любой    РОУТЕРЫ      все кластеры            → group_routers + global_router
  Фаза 5  любой    СОВМЕСТНАЯ   все кластеры            → всё вместе (fine-tune)

Логика:
  - Каждая фаза разблокирует только нужные части модели.
  - Данные для обучения выбираются по α-уровню кластеров (RepoCorpusLoader).
  - Переход к следующей фазе — после N шагов (или --steps_per_phase).

Usage:
  python train_hmoe_curriculum.py
  python train_hmoe_curriculum.py --fast                  # 50 шагов на фазу
  python train_hmoe_curriculum.py --steps_per_phase 500
  python train_hmoe_curriculum.py --phase 2               # только фаза 2
  python train_hmoe_curriculum.py --resume checkpoint.pt  # продолжить с чекпоинта
  python train_hmoe_curriculum.py --no-corpus             # без RepoCorpusLoader
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    HMoEConfig,
    HierarchicalMoEFFN,
    CLUSTER_TO_DOMAIN,
    DOMAIN_GROUPS,
    DOMAIN_TO_GROUP,
    TRAINING_STAGES,
    set_moe_stage,
    get_stage_info,
)

# ── Константы ─────────────────────────────────────────────────────────────────

torch.manual_seed(42)
random.seed(42)

_ROOT  = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"

# Модель (малый размер для демонстрации)
MODEL_CFG = dict(
    vocab_size      = 256,
    block_size      = 64,
    d_model         = 128,
    n_heads         = 4,
    n_layers        = 4,
    ffn_mult        = 4,
    hamming_lambda  = 0.15,
    uncertainty_budget = 0.25,
    dropout         = 0.1,
    use_domain_routing = False,
    use_hierarchical_moe = True,
)

HMOE_CFG = HMoEConfig(
    d_model       = 128,
    use_multiscale = True,
    use_hex_tier  = False,
)

# ── Учебный план ──────────────────────────────────────────────────────────────

# Кластеры по α-уровням
ALPHA_MAP: Dict[str, int] = {
    "Theory":     4,
    "Models":     2,
    "Portal":     2,
    "Self":       0,
    "Training":   0,
    "Benchmarks": -2,
    "Scripts":    -4,
    "Data":       -4,
}

# Группа (петля восьмёрки) по кластеру
CLUSTER_GROUP: Dict[str, str] = {
    cluster: DOMAIN_TO_GROUP[domain]
    for cluster, domain in CLUSTER_TO_DOMAIN.items()
}

# Описание фаз curriculum
@dataclass
class CurriculumPhase:
    phase:       int
    name:        str
    description: str
    alpha_range: Tuple[int, int]   # включительно
    clusters:    List[str]          # кластеры для обучения (или "all")
    group:       Optional[str]      # какую группу экспертов тренировать (None = all)
    hmoe_stage:  int                # TRAINING_STAGES ключ
    lr_scale:    float = 1.0        # масштаб LR относительно base_lr


CURRICULUM: List[CurriculumPhase] = [
    CurriculumPhase(
        phase       = 1,
        name        = "Конкретное (GlyphLevel)",
        description = "α=−4,−2 → CONCRETE micro_experts: Scripts, Data, Benchmarks",
        alpha_range = (-4, -2),
        clusters    = ["Scripts", "Data", "Benchmarks"],
        group       = "CONCRETE",
        hmoe_stage  = 1,         # только micro_experts
        lr_scale    = 1.0,
    ),
    CurriculumPhase(
        phase       = 2,
        name        = "Динамическое (MethodLevel)",
        description = "α= 0  → crossing + DYNAMIC micro_experts: Self, Training",
        alpha_range = (0, 0),
        clusters    = ["Self", "Training"],
        group       = "DYNAMIC",
        hmoe_stage  = 4,         # crossing + global_router
        lr_scale    = 0.8,
    ),
    CurriculumPhase(
        phase       = 3,
        name        = "Абстрактное (TheoryLevel)",
        description = "α=+2,+4 → ABSTRACT micro_experts: Models, Theory, Portal",
        alpha_range = (2, 4),
        clusters    = ["Models", "Theory", "Portal"],
        group       = "ABSTRACT",
        hmoe_stage  = 1,         # только micro_experts
        lr_scale    = 0.6,
    ),
    CurriculumPhase(
        phase       = 4,
        name        = "Сборка роутеров",
        description = "Все кластеры → group_routers + global_router",
        alpha_range = (-4, 4),
        clusters    = [],        # all
        group       = None,
        hmoe_stage  = 3,         # global_router + все вместе
        lr_scale    = 0.4,
    ),
    CurriculumPhase(
        phase       = 5,
        name        = "Совместная настройка",
        description = "Все кластеры → fine-tune всей системы",
        alpha_range = (-4, 4),
        clusters    = [],        # all
        group       = None,
        hmoe_stage  = 5,         # всё разморожено
        lr_scale    = 0.2,
    ),
]


# ── Утилиты ───────────────────────────────────────────────────────────────────

def encode(text: str, block_size: int = 63) -> torch.Tensor:
    """Кодирует текст как byte-токены."""
    ids = [min(b, 255) for b in text.encode("utf-8")][:block_size]
    if not ids:
        ids = [32]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, T)


def load_corpus(root: str, clusters: List[str]) -> List[str]:
    """Загружает тексты из указанных кластеров через RepoCorpusLoader."""
    try:
        from repo_corpus_loader import RepoCorpusLoader
        loader = RepoCorpusLoader(root)
        texts: List[str] = []
        for cluster_name in clusters:
            try:
                items = loader.load_cluster(cluster_name)
                for item in items:
                    t = item if isinstance(item, str) else item.get("text", "")
                    if len(t) > 10:
                        texts.append(t)
            except Exception:
                pass
        return texts
    except ImportError:
        return []


def make_batch(texts: List[str], block_size: int = 63) -> Tuple[torch.Tensor, torch.Tensor]:
    """Случайный батч из списка текстов."""
    text = random.choice(texts)
    tokens = encode(text, block_size)
    if tokens.shape[1] < 2:
        tokens = torch.randint(0, 256, (1, block_size))
    inp = tokens[:, :-1]
    tgt = tokens[:, 1:]
    return inp, tgt


def freeze_group_experts(moe: HierarchicalMoEFFN, keep_group: str) -> None:
    """Замораживает micro_experts всех групп, кроме keep_group."""
    for cluster, domain in CLUSTER_TO_DOMAIN.items():
        g = DOMAIN_TO_GROUP[domain]
        expert = moe.micro_experts[cluster] if cluster in moe.micro_experts else None
        if expert is None:
            continue
        requires_grad = (g == keep_group)
        for p in expert.parameters():
            p.requires_grad = requires_grad


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_all_moe(model: Variant3GPT) -> List[HierarchicalMoEFFN]:
    moes = []
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            moes.append(block.hmoe)
    return moes


def perplexity(model: nn.Module, texts: List[str], n: int = 20) -> float:
    model.eval()
    ppls = []
    sample = random.sample(texts, min(n, len(texts)))
    for text in sample:
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        with torch.no_grad():
            _, loss, _ = model(inp, targets=tgt)
        if loss is not None and not torch.isnan(loss):
            ppls.append(min(math.exp(loss.item()), 1e5))
    return sum(ppls) / len(ppls) if ppls else float("inf")


def routing_entropy(model: nn.Module, texts: List[str], n: int = 10) -> float:
    """Средняя энтропия маршрутизации (выше = разнообразнее)."""
    model.eval()
    entropies = []
    for text in random.sample(texts, min(n, len(texts))):
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            model(tokens[:, :-1])
        for block in model.blocks:
            info = getattr(block, '_last_moe_info', None)
            if info and 'group_weights' in info:
                gw = info['group_weights'][0].mean(0)
                ent = -(gw * torch.log(gw + 1e-8)).sum().item()
                entropies.append(ent)
                break
    return sum(entropies) / len(entropies) if entropies else float("nan")


# ── Фаза обучения ─────────────────────────────────────────────────────────────

def run_phase(
    model:           Variant3GPT,
    phase_cfg:       CurriculumPhase,
    texts_for_phase: List[str],
    texts_all:       List[str],
    steps:           int,
    base_lr:         float,
    log_every:       int,
    phase_num:       int,
    total_phases:    int,
) -> Dict[str, float]:
    """Обучение одной фазы.

    Returns: словарь метрик {'loss', 'ppl', 'ent'}.
    """
    moes = get_all_moe(model)

    # 1) Установить TRAINING_STAGE для каждого MoE блока
    for moe in moes:
        unfrozen = set_moe_stage(moe, phase_cfg.hmoe_stage)

    # 2) Дополнительно: если у фазы задана конкретная группа — заморозить остальные
    if phase_cfg.group is not None:
        for moe in moes:
            freeze_group_experts(moe, keep_group=phase_cfg.group)

    # 3) Оптимизатор только по trainable параметрам
    lr = base_lr * phase_cfg.lr_scale
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    print(f"\n{'─' * 72}")
    print(f"  ФАЗА {phase_num}/{total_phases}: {phase_cfg.name}")
    print(f"  {phase_cfg.description}")
    print(f"  HMoE stage: {phase_cfg.hmoe_stage} ({TRAINING_STAGES[phase_cfg.hmoe_stage]['name']})")
    print(f"  Trainable params: {n_trainable:,}  |  LR: {lr:.2e}  |  Steps: {steps}")
    print(f"  Кластеры: {phase_cfg.clusters if phase_cfg.clusters else 'все'}")
    print(f"  Группа экспертов: {phase_cfg.group or 'все'}")

    if not texts_for_phase:
        print("  ⚠️  Нет текстов для этой фазы — пропускаем")
        return {"loss": float("nan"), "ppl": float("nan"), "ent": float("nan")}

    model.train()
    running_loss = 0.0
    running_aux  = 0.0
    losses = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        inp, tgt = make_batch(texts_for_phase, block_size=63)

        logits, loss, aux_loss = model(inp, targets=tgt)

        if loss is None:
            continue

        total_loss = loss
        if aux_loss is not None and not torch.isnan(aux_loss):
            total_loss = total_loss + aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        if aux_loss is not None and not torch.isnan(aux_loss):
            running_aux += aux_loss.item()
        losses.append(loss.item())

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            avg_aux  = running_aux  / log_every
            elapsed  = time.perf_counter() - t0
            ppl_est  = math.exp(min(avg_loss, 10))
            print(f"    step {step:>5d}/{steps}  loss={avg_loss:.4f}  "
                  f"aux={avg_aux:.5f}  ppl≈{ppl_est:.1f}  "
                  f"({elapsed:.1f}s elapsed)")
            running_loss = 0.0
            running_aux  = 0.0

    # Финальные метрики
    eval_texts = texts_all if texts_all else texts_for_phase
    final_ppl = perplexity(model, eval_texts, n=20)
    final_ent = routing_entropy(model, eval_texts, n=10)
    avg_train_loss = sum(losses) / len(losses) if losses else float("nan")

    print(f"  → Фаза {phase_num} завершена: "
          f"loss={avg_train_loss:.4f}  PPL={final_ppl:.2f}  ent={final_ent:.3f}")

    return {"loss": avg_train_loss, "ppl": final_ppl, "ent": final_ent}


# ── Главная функция ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum training HierarchicalMoE от простого к сложному"
    )
    parser.add_argument("--fast",           action="store_true",
                        help="50 шагов на фазу (для тестирования)")
    parser.add_argument("--steps_per_phase", type=int, default=200,
                        help="Количество шагов на фазу (default: 200)")
    parser.add_argument("--phase",          type=int, default=None,
                        help="Запустить только конкретную фазу (1-5)")
    parser.add_argument("--lr",             type=float, default=3e-4,
                        help="Базовый learning rate (default: 3e-4)")
    parser.add_argument("--resume",         type=str, default=None,
                        help="Путь к чекпоинту для продолжения обучения")
    parser.add_argument("--save",           type=str, default="hmoe_curriculum.pt",
                        help="Куда сохранять чекпоинт (default: hmoe_curriculum.pt)")
    parser.add_argument("--no-corpus",      action="store_true",
                        help="Не использовать RepoCorpusLoader (только синтетические тексты)")
    parser.add_argument("--force",          action="store_true",
                        help="Принудительно запустить фазу даже если уже пройдена")
    parser.add_argument("--log_every",      type=int, default=25,
                        help="Логировать каждые N шагов (default: 25)")
    args = parser.parse_args()

    steps = 50 if args.fast else args.steps_per_phase
    log_every = min(args.log_every, steps)

    print("\n" + "═" * 72)
    print("  CURRICULUM TRAINING: от простого к сложному")
    print("  Архитектура: HierarchicalMoE (фигура-восьмёрка по Крюкову)")
    print("═" * 72)

    # ── Создать/загрузить модель ────────────────────────────────────────────
    cfg = Variant3Config(**MODEL_CFG)
    model = Variant3GPT(cfg)

    # Заменить HMoE конфиг во всех блоках
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            block.hmoe = HierarchicalMoEFFN(HMOE_CFG)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n  Модель: {total_p/1e6:.2f}M параметров")

    start_phase = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        start_phase = ckpt.get("next_phase", 1)
        print(f"  Чекпоинт загружен: {args.resume}  (продолжаем с фазы {start_phase})")

    # ── Загрузить корпус ────────────────────────────────────────────────────
    all_cluster_names = list(CLUSTER_TO_DOMAIN.keys())
    corpus_by_cluster: Dict[str, List[str]] = {}

    if not args.no_corpus:
        print("\n  Загружаем корпус репозитория...")
        for cluster in all_cluster_names:
            texts = load_corpus(_ROOT, [cluster])
            corpus_by_cluster[cluster] = texts
            if texts:
                print(f"    {cluster:12s}: {len(texts):4d} текстов  (α={ALPHA_MAP.get(cluster, '?')})")
            else:
                print(f"    {cluster:12s}: —  (файлы не найдены)")
    else:
        print("  Режим --no-corpus: используем синтетические тексты")

    # Синтетические тексты как запасной вариант
    SYNTHETIC: Dict[str, List[str]] = {
        "Scripts":   ["def forward(self, x): return x", "import os; sys.exit(0)",
                      "for i in range(10): print(i)"] * 5,
        "Data":      ["{'key': 'value', 'count': 42}", "[1, 2, 3, 4, 5]",
                      "null values found in column"] * 5,
        "Benchmarks":["benchmark result: 0.95 accuracy", "loss=0.34 ppl=1.4",
                      "forward_ms=12.3 params=1.2M"] * 5,
        "Training":  ["optimizer.step()", "loss.backward()", "scheduler.step(loss)"] * 5,
        "Self":      ["self.attention = nn.MultiheadAttention(d)", "x = self.norm(x + attn_out)",
                      "return output, aux_loss"] * 5,
        "Models":    ["class HierarchicalMoEFFN(nn.Module):", "def _run_group(self, x, group):",
                      "crossing = BidirBridgeExpert(d_model)"] * 5,
        "Theory":    ["The I Ching hexagram represents binary states",
                      "Kryukov figure-8 topology: abstract and concrete loops",
                      "consciousness emerges from recursive self-reference"] * 5,
        "Portal":    ["nautilus_adapter.connect()", "portal.register_domain(domain)",
                      "federated_round.aggregate()"] * 5,
    }

    for cluster in all_cluster_names:
        if not corpus_by_cluster.get(cluster):
            corpus_by_cluster[cluster] = SYNTHETIC.get(cluster, SYNTHETIC["Self"])

    # Все тексты вместе (для eval)
    texts_all = [t for ts in corpus_by_cluster.values() for t in ts]

    # ── Curriculum ─────────────────────────────────────────────────────────
    phases_to_run = CURRICULUM
    if args.phase is not None:
        phases_to_run = [p for p in CURRICULUM if p.phase == args.phase]
        if not phases_to_run:
            print(f"  ❌ Фаза {args.phase} не найдена (доступны: 1-5)")
            sys.exit(1)

    results: Dict[int, Dict[str, float]] = {}

    for phase_cfg in phases_to_run:
        if phase_cfg.phase < start_phase and not args.force:
            print(f"  Пропускаем фазу {phase_cfg.phase} (уже пройдена)")
            continue

        # Собрать тексты для этой фазы
        if phase_cfg.clusters:
            phase_clusters = phase_cfg.clusters
        else:
            phase_clusters = all_cluster_names

        texts_for_phase: List[str] = []
        for c in phase_clusters:
            texts_for_phase.extend(corpus_by_cluster.get(c, []))

        metrics = run_phase(
            model          = model,
            phase_cfg      = phase_cfg,
            texts_for_phase= texts_for_phase,
            texts_all      = texts_all,
            steps          = steps,
            base_lr        = args.lr,
            log_every      = log_every,
            phase_num      = phase_cfg.phase,
            total_phases   = len(CURRICULUM),
        )
        results[phase_cfg.phase] = metrics

        # Сохранить чекпоинт после каждой фазы
        ckpt = {
            "model_state": model.state_dict(),
            "next_phase":  phase_cfg.phase + 1,
            "metrics":     results,
            "phase_name":  phase_cfg.name,
        }
        torch.save(ckpt, args.save)
        print(f"  💾 Чекпоинт сохранён: {args.save}")

    # ── Итоговая таблица ───────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  ИТОГИ CURRICULUM TRAINING")
    print("═" * 72)
    print(f"  {'Фаза':>5s}  {'Название':30s}  {'Loss':>7s}  {'PPL':>7s}  {'Ent':>6s}")
    print("  " + "─" * 62)
    for phase_cfg in phases_to_run:
        if phase_cfg.phase not in results:
            continue
        m = results[phase_cfg.phase]
        print(f"  {phase_cfg.phase:>5d}  {phase_cfg.name:30s}  "
              f"{m['loss']:>7.4f}  {m['ppl']:>7.2f}  {m['ent']:>6.3f}")

    print("\n  ✅ Curriculum обучение завершено")
    if args.phase is None:
        print(f"  💾 Финальная модель: {args.save}")
    print()


if __name__ == "__main__":
    main()
