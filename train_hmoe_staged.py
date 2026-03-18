#!/usr/bin/env python3
"""
train_hmoe_staged.py — Поэтапное обучение Hierarchical MoE архитектуры.

Этапы обучения:
  Stage 1 (MicroExperts)  — каждый эксперт обучается на своём кластере отдельно
  Stage 2 (GroupRouters)  — роутеры внутри групп + эксперты
  Stage 3 (GlobalRouter)  — глобальный роутер + все компоненты
  Stage 4 (BridgeExperts) — мосты между группами (межкластерный синтез)
  Stage 5 (JointFinetune) — совместная тонкая настройка всего

Иерархия маршрутизации:
  GlobalRouter → GroupRouter[ABSTRACT/DYNAMIC/CONCRETE] → MicroExpert[cluster]
                                                        └→ BridgeExpert[A↔B]

Usage:
  python train_hmoe_staged.py                   # все 5 этапов
  python train_hmoe_staged.py --stage 1         # только Stage 1
  python train_hmoe_staged.py --stage 1 2       # Stage 1 и 2
  python train_hmoe_staged.py --fast            # 20 шагов на кластер (демо)
  python train_hmoe_staged.py --resume          # продолжить с checkpoint_hmoe.pt

Горизонтальные связи:
  ↔ yijing_transformer/models/variant3.py       — базовая модель
  ↔ yijing_transformer/models/hierarchical_moe.py — HMoE компоненты
  ↔ repo_corpus_loader.py                       — источник кластеров
  ↔ checkpoint_hmoe.pt                          — выходной checkpoint
"""

import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    HierarchicalMoEFFN, HMoEConfig,
    CLUSTER_TO_DOMAIN, DOMAIN_TO_GROUP, DOMAIN_GROUPS,
    TRAINING_STAGES, set_moe_stage, get_stage_info,
)
from repo_corpus_loader import RepoCorpusLoader, CLUSTER_DEFS

# ── Конфигурация ───────────────────────────────────────────────────────────────

torch.manual_seed(42)
random.seed(42)

_ROOT  = Path(__file__).parent
DEVICE = "cpu"

V3_CFG = Variant3Config(
    vocab_size=256,
    block_size=32,
    d_model=128,
    n_heads=4,
    n_layers=4,
    ffn_mult=4,
    hamming_lambda=0.15,
    uncertainty_budget=0.25,
    dropout=0.05,
    use_domain_routing=True,
    use_hierarchical_moe=True,   # ← ключевой флаг
)

HMOE_CFG = HMoEConfig(
    d_model=128,
    expert_expansion=2,
    top_k_experts=2,
    lb_loss_weight=0.01,
    bridge_weight=0.5,
)

# Шагов на кластер по этапам
STEPS_PER_CLUSTER = {
    1: {"Scripts": 80, "Benchmarks": 60, "Training": 100,
        "Self": 80, "Models": 100, "Theory": 90},
    2: 60,   # все кластеры по N шагов
    3: 40,
    4: 50,
    5: 30,
}

LR_PER_STAGE = {1: 3e-4, 2: 2e-4, 3: 1e-4, 4: 2e-4, 5: 5e-5}

BASE_CHECKPOINT  = _ROOT / "checkpoint_e2_clusters.pt"
HMOE_CHECKPOINT  = _ROOT / "checkpoint_hmoe.pt"
HMOE_LOG         = _ROOT / "train_hmoe_log.json"


# ── Утилиты ───────────────────────────────────────────────────────────────────

def encode(text: str, vocab_size: int = 256, block_size: int = 32) -> torch.Tensor:
    ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
    return torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)


def perplexity(model: Variant3GPT, texts: List[str], n: int = 20) -> float:
    model.eval()
    ppls = []
    for text in random.sample(texts, min(n, len(texts))):
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        with torch.no_grad():
            _, loss, _ = model(inp, targets=tgt)
        if loss is not None and not torch.isnan(loss):
            ppls.append(math.exp(min(loss.item(), 10)))
    return sum(ppls) / len(ppls) if ppls else float("inf")


def collect_moe_lb_loss(model: Variant3GPT) -> torch.Tensor:
    """Суммирует lb_loss из всех блоков с HMoE."""
    total = None
    for block in model.blocks:
        info = getattr(block, '_last_moe_info', None)
        if info and 'lb_loss' in info:
            if total is None:
                total = info['lb_loss']
            else:
                total = total + info['lb_loss']
    if total is None:
        total = torch.tensor(0.0, device=next(model.parameters()).device)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: MicroExperts — каждый эксперт на своём кластере
# ══════════════════════════════════════════════════════════════════════════════

def train_stage1_micro_experts(
    model: Variant3GPT,
    clusters: Dict[str, List[str]],
    steps_cfg: Dict,
    lr: float,
    fast: bool,
    log: list,
) -> None:
    print("\n" + "═" * 66)
    print("  STAGE 1: MicroExperts — специализация по кластерам")
    print("═" * 66)

    # Для каждого блока с HMoE устанавливаем Stage 1
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            unfrozen = set_moe_stage(block.hmoe, stage=1)
            print(f"  Разморожено {len(unfrozen)} параметров (MicroExperts)")
            break

    # Тренируем каждый кластер → его эксперт
    for cluster, texts in clusters.items():
        if not texts:
            continue

        group = DOMAIN_TO_GROUP.get(CLUSTER_TO_DOMAIN.get(cluster, ""), None)
        steps = 20 if fast else (
            steps_cfg.get(cluster, 60) if isinstance(steps_cfg, dict) else steps_cfg
        )

        print(f"\n  ── Кластер «{cluster}» → группа {group}, {steps} шагов ──")
        ppl_before = perplexity(model, texts)
        print(f"  PPL до: {ppl_before:.2f}")

        # Оптимизируем только эксперт этого кластера
        expert_params = []
        for block in model.blocks:
            if hasattr(block, 'hmoe') and cluster in block.hmoe.micro_experts:
                expert_params += list(block.hmoe.micro_experts[cluster].parameters())

        if not expert_params:
            print(f"  ⚠️  Эксперт «{cluster}» не найден — пропуск")
            continue

        opt = torch.optim.AdamW(expert_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        model.train()
        t0 = time.time()
        for step in range(1, steps + 1):
            text   = random.choice(texts)
            tokens = encode(text)
            if tokens.shape[1] < 2:
                continue
            inp, tgt = tokens[:, :-1], tokens[:, 1:]
            _, loss, _ = model(inp, targets=tgt)
            if loss is None or torch.isnan(loss):
                continue

            lb = collect_moe_lb_loss(model)
            total_loss = loss + lb

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(expert_params, 1.0)
            opt.step()
            scheduler.step()

            if step % (steps // 5 or 1) == 0:
                print(f"  шаг {step:4d}/{steps}  loss={loss.item():.4f}  "
                      f"lb={lb.item():.4f}  ({int(time.time()-t0)}s)")

        ppl_after = perplexity(model, texts)
        delta = (ppl_before - ppl_after) / ppl_before * 100
        symbol = "⬇️" if ppl_after < ppl_before else "⬆️"
        print(f"  PPL после: {ppl_after:.2f}  ({symbol} {abs(delta):.1f}%)")

        log.append({"stage": 1, "cluster": cluster, "group": group,
                    "ppl_before": round(ppl_before, 2),
                    "ppl_after":  round(ppl_after, 2),
                    "delta_pct":  round(delta, 1)})


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: GroupRouters
# ══════════════════════════════════════════════════════════════════════════════

def train_stage_generic(
    model: Variant3GPT,
    clusters: Dict[str, List[str]],
    stage: int,
    steps: int,
    lr: float,
    fast: bool,
    log: list,
) -> None:
    info = TRAINING_STAGES[stage]
    print(f"\n{'═' * 66}")
    print(f"  STAGE {stage}: {info['name']}")
    print(f"  {info['description']}")
    print("═" * 66)

    # Устанавливаем этап для всех блоков
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            unfrozen = set_moe_stage(block.hmoe, stage=stage)

    # Собираем все обучаемые параметры HMoE
    trainable = [p for block in model.blocks
                 if hasattr(block, 'hmoe')
                 for p in block.hmoe.parameters() if p.requires_grad]

    if not trainable:
        print("  ⚠️  Нет обучаемых параметров — пропуск")
        return

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    actual_steps = 20 if fast else steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=actual_steps)

    all_texts = [t for texts in clusters.values() for t in texts]
    ppl_before = perplexity(model, all_texts)
    print(f"  PPL до (все кластеры): {ppl_before:.2f}")
    print(f"  Обучаемых параметров: {sum(p.numel() for p in trainable):,}")

    model.train()
    t0 = time.time()
    for step in range(1, actual_steps + 1):
        text   = random.choice(all_texts)
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        _, loss, _ = model(inp, targets=tgt)
        if loss is None or torch.isnan(loss):
            continue

        lb = collect_moe_lb_loss(model)
        total_loss = loss + lb

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        scheduler.step()

        if step % max(actual_steps // 5, 1) == 0:
            print(f"  шаг {step:4d}/{actual_steps}  loss={loss.item():.4f}  "
                  f"lb={lb.item():.4f}  ({int(time.time()-t0)}s)")

    ppl_after = perplexity(model, all_texts)
    delta = (ppl_before - ppl_after) / ppl_before * 100
    symbol = "⬇️" if ppl_after < ppl_before else "⬆️"
    print(f"  PPL после: {ppl_after:.2f}  ({symbol} {abs(delta):.1f}%)")

    log.append({"stage": stage, "cluster": "all",
                "ppl_before": round(ppl_before, 2),
                "ppl_after":  round(ppl_after, 2),
                "delta_pct":  round(delta, 1)})


# ══════════════════════════════════════════════════════════════════════════════
# Главная функция
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Поэтапное обучение Hierarchical MoE")
    parser.add_argument("--stage", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5],
                        help="Этапы для запуска (1-5)")
    parser.add_argument("--fast", action="store_true",
                        help="Быстрый режим: 20 шагов")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить с checkpoint_hmoe.pt")
    args = parser.parse_args()

    print("\n  Загружаю кластеры репозитория...")
    loader = RepoCorpusLoader(_ROOT)
    clusters_raw = loader.load_all_clusters()

    # Преобразуем в dict[cluster_name → list[str]]
    clusters: Dict[str, List[str]] = {}
    for name, items in clusters_raw.items():
        clusters[name] = [item if isinstance(item, str) else item.get("text", "")
                          for item in items]

    print(f"  Кластеров: {len(clusters)}")
    for name, texts in clusters.items():
        print(f"    {name:15s}: {len(texts)} текстов")

    # ── Модель ────────────────────────────────────────────────────────────────
    print("\n  Инициализирую Variant3GPT с HierarchicalMoEFFN...")
    model = Variant3GPT(V3_CFG)

    # Загрузка базового checkpoint
    ckpt_path = HMOE_CHECKPOINT if (args.resume and HMOE_CHECKPOINT.exists()) \
                else BASE_CHECKPOINT
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE)
        # Загружаем только совместимые ключи
        model_state = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_state and v.shape == model_state[k].shape}
        model_state.update(compatible)
        model.load_state_dict(model_state, strict=False)
        print(f"  Загружен: {ckpt_path.name}  "
              f"({len(compatible)}/{len(model_state)} ключей)")
    else:
        print("  Новая модель (нет checkpoint)")

    total_params = sum(p.numel() for p in model.parameters())
    hmoe_params  = sum(p.numel() for block in model.blocks
                       if hasattr(block, 'hmoe')
                       for p in block.hmoe.parameters())
    print(f"  Всего параметров: {total_params:,}")
    print(f"  HMoE параметров:  {hmoe_params:,} ({hmoe_params/total_params*100:.1f}%)")

    # ── Обучение по этапам ────────────────────────────────────────────────────
    log = []

    for stage in sorted(args.stage):
        if stage not in TRAINING_STAGES:
            print(f"  ⚠️  Stage {stage} не существует — пропуск")
            continue

        print(f"\n{get_stage_info(stage)}")

        lr = LR_PER_STAGE[stage]
        steps_cfg = STEPS_PER_CLUSTER.get(stage, 50)

        if stage == 1:
            train_stage1_micro_experts(
                model, clusters,
                steps_cfg=steps_cfg if isinstance(steps_cfg, dict) else {},
                lr=lr, fast=args.fast, log=log,
            )
        else:
            steps = 20 if args.fast else (
                steps_cfg if isinstance(steps_cfg, int) else 50
            )
            train_stage_generic(
                model, clusters, stage=stage,
                steps=steps, lr=lr, fast=args.fast, log=log,
            )

        # Сохраняем checkpoint после каждого этапа
        torch.save(model.state_dict(), HMOE_CHECKPOINT)
        print(f"\n  💾 {HMOE_CHECKPOINT.name} сохранён (после Stage {stage})")

    # ── Итоги ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print("  ИТОГИ — HMoE ПОЭТАПНОЕ ОБУЧЕНИЕ")
    print("═" * 66)
    print(f"  {'Stage':6s}  {'Кластер':15s}  {'PPL до':>8s}  {'PPL после':>9s}  {'Δ':>8s}")
    print(f"  {'─'*6}  {'─'*15}  {'─'*8}  {'─'*9}  {'─'*8}")
    for row in log:
        symbol = "⬇️" if row['delta_pct'] > 0 else "⬆️"
        print(f"  {row['stage']:6d}  {row['cluster']:15s}  "
              f"{row['ppl_before']:8.2f}  {row['ppl_after']:9.2f}  "
              f"{symbol} {abs(row['delta_pct']):.1f}%")

    HMOE_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n  📄 Лог: {HMOE_LOG.name}")
    print(f"  💾 Checkpoint: {HMOE_CHECKPOINT.name}")

    # ── Тест роутинга ────────────────────────────────────────────────────────
    print("\n" + "═" * 66)
    print("  ТЕСТ ИЕРАРХИЧЕСКОГО РОУТИНГА")
    print("═" * 66)
    model.eval()
    test_concepts = ["Theory", "Models", "Self", "Training", "Scripts"]
    group_names = list(DOMAIN_GROUPS.keys())

    for concept in test_concepts:
        texts = clusters.get(concept, [])
        if not texts:
            continue
        sample = random.choice(texts[:10])
        tokens = encode(sample[:20])
        if tokens.shape[1] < 2:
            continue
        inp = tokens[:, :-1]
        with torch.no_grad():
            model(inp)

        # Читаем роутинг из последнего блока с HMoE
        for block in reversed(model.blocks):
            info = getattr(block, '_last_moe_info', None)
            if info and 'group_weights' in info:
                gw = info['group_weights'][0].mean(0)  # (n_groups,)
                top_group = group_names[gw.argmax().item()]
                weights_str = "  ".join(
                    f"{g[:3]}={gw[i]:.2f}" for i, g in enumerate(group_names)
                )
                print(f"  «{concept:10s}» → {top_group:8s}  [{weights_str}]")
                break

    print("\n  ✅ Обучение завершено")


if __name__ == "__main__":
    main()
