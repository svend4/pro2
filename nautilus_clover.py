#!/usr/bin/env python3
"""
nautilus_clover.py — Наутилус-Клевер: 4 кольца Пифагорейской тетрактиды + HMoE

Геометрия (четырёхлистный клевер / Наутилус):

  Пифагорейская тетрактида: 1+2+3+4 = 10
  4 кольца-лепестка разного размера:

      ┌─────────────────────────────────────────────────┐
      │  Кольцо 1  META      10 шагов  (10%)  ← центр  │
      │  Кольцо 2  ABSTRACT  20 шагов  (20%)            │
      │  Кольцо 3  DYNAMIC   30 шагов  (30%)            │
      │  Кольцо 4  CONCRETE  40 шагов  (40%)  ← самый  │
      │                                          большой│
      │  Итого = 100 шагов = 100% = спираль             │
      └─────────────────────────────────────────────────┘

  META — центральный узел (все лепестки проходят через него).
  Каждый лепесток образует замкнутое кольцо с META в точке пересечения.
  Переход между кольцами = восьмёрка (figure-8) через META.

Режимы движения:
  ○  Внутри кольца  — roundabout (ранний выход при LCI ≈ π)
  ∞  Через META      — figure-8 (bidir обмен встречных потоков)
  🌀 Наутилус        — META→ABSTRACT→DYNAMIC→CONCRETE (шаги растут)

Два агента (из multi-salesman):
  ▶ Forward: META → ABSTRACT → DYNAMIC → CONCRETE (наружу)
  ◀ Reverse: CONCRETE → DYNAMIC → ABSTRACT → META  (внутрь)
  Встреча и обмен — в META (центральный узел клевера).

LCI-loss (исправлен: работает с живым вычислительным графом).

Комбинирует лучшее из бенчмарка:
  multi-salesman  (0.989) → два агента + координация в DYNAMIC/META
  roundabout      (0.975) → ранний выход при резонансе
  bidir-turbine   (0.972) → встречные потоки
  turbine LCI-loss        → градиентный толчок gating → π (исправлен)

Usage:
  python nautilus_clover.py --checkpoint hmoe_self_trained_v5.pt
  python nautilus_clover.py --fast
  python nautilus_clover.py --cycles 8 --no-bidir
  python nautilus_clover.py --cycles 8 --lci-loss 0.1
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    HMoEConfig, CLUSTER_TO_DOMAIN, DOMAIN_GROUPS, DOMAIN_TO_GROUP, set_moe_stage,
)
from self_train_hmoe import (
    lci_from_routing, lci_from_embeddings, micro_train, quality_filter,
    RagBuffer, _generate, _ids_to_text, _encode, _hex_prompt,
    _get_emb, _get_moes, _freeze_all_except, MODEL_CFG, _LCI_EPSILON,
)

DEVICE = "cpu"
_ROOT  = os.path.dirname(os.path.abspath(__file__))

# ── Пифагорейская тетрактида: 4 кольца 10:20:30:40 шагов ─────────────────────
#   1 + 2 + 3 + 4 = 10  →  10 + 20 + 30 + 40 = 100 = 100%
RINGS = [
    {"name": "META",     "steps": 10, "groups": ["ABSTRACT", "DYNAMIC", "CONCRETE"], "ratio": 1},
    {"name": "ABSTRACT", "steps": 20, "groups": ["ABSTRACT"],                        "ratio": 2},
    {"name": "DYNAMIC",  "steps": 30, "groups": ["DYNAMIC"],                         "ratio": 3},
    {"name": "CONCRETE", "steps": 40, "groups": ["CONCRETE"],                        "ratio": 4},
]
_RING_BY_NAME = {r["name"]: r for r in RINGS}
_TOTAL_STEPS  = sum(r["steps"] for r in RINGS)   # 100

# Наутилус-порядок (малое → большое): META(10)→ABSTRACT(20)→DYNAMIC(30)→CONCRETE(40)
_SPIRAL_FWD = ["META", "ABSTRACT", "DYNAMIC", "CONCRETE"]
# Анти-Наутилус (встречный агент): CONCRETE→DYNAMIC→ABSTRACT→META
_SPIRAL_REV = ["CONCRETE", "DYNAMIC", "ABSTRACT", "META"]

_META_FREEZE = ["ABSTRACT", "DYNAMIC", "CONCRETE"]

# Описания лепестков для вывода
_RING_DESC = {
    "META":     "центральный узел / точка пересечения",
    "ABSTRACT": "абстрактный лепесток  (обобщение)",
    "DYNAMIC":  "динамический лепесток (мост)",
    "CONCRETE": "конкретный лепесток   (специализация)",
}


# ── Утилиты ───────────────────────────────────────────────────────────────────

def _freeze_for(model: Variant3GPT, ring_name: str) -> None:
    ring = _RING_BY_NAME[ring_name]
    for moe in _get_moes(model):
        _freeze_all_except(moe, ring["groups"])
    if ring_name in ("META", "DYNAMIC"):
        block0 = model.blocks[0]
        set_moe_stage(getattr(block0, "hmoe", None), 4)


def _lci_loss_step(model: Variant3GPT, ids: torch.Tensor, lr: float) -> float:
    """
    Один шаг Kirchhoff-штрафа: двигаем gating к балансу ABSTRACT/CONCRETE.
    Исправленная версия: group_weights берём напрямую из GlobalRouter
    (с живым вычислительным графом, не из _last_moe_info.detach).
    """
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return 0.0
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    inp = ids[:, :-1] if ids.shape[1] > 1 else ids
    gw_list: List[torch.Tensor] = []
    try:
        x = model.tok_emb(inp)
        for block in model.blocks:
            moe = getattr(block, "hmoe", None)
            if moe is not None and hasattr(moe, "global_router"):
                res = moe.global_router(x)
                gw  = res[0] if isinstance(res, tuple) else res
                if not torch.isnan(gw).any():
                    gw_list.append(
                        gw.mean(dim=(0, 1)) if gw.dim() == 3 else gw.mean(dim=0)
                    )
            x = block(x)
    except Exception:
        return 0.0

    if not gw_list:
        return 0.0

    avg_gw = torch.stack(gw_list).mean(0)
    groups = list(DOMAIN_GROUPS.keys())
    n = len(groups)
    idx_a = groups.index("ABSTRACT") if "ABSTRACT" in groups else 0
    idx_b = groups.index("CONCRETE") if "CONCRETE" in groups else min(2, n - 1)

    w_total  = avg_gw.sum() + 1e-8
    imbalance = torch.abs(avg_gw[idx_a] / w_total - avg_gw[idx_b] / w_total)
    loss = imbalance * math.pi   # цель: imbalance→0 ↔ LCI→π

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 0.5)
    opt.step()
    return float(loss.item())


# ── Фаза одного кольца ────────────────────────────────────────────────────────

def _run_ring(
    model: Variant3GPT,
    ring_name: str,
    ids: torch.Tensor,
    rag: RagBuffer,
    steps: int,
    train_lr: float,
    temperature: float,
    do_train: bool,
    lci_loss_lambda: float = 0.0,
    early_exit: bool = True,
    block_size: int = MODEL_CFG["block_size"] - 1,
) -> Tuple[torch.Tensor, float, float, int, bool]:
    """
    Тренировка внутри одного кольца за steps шагов.
    Roundabout-стиль: ранний выход при достижении резонанса (early_exit=True).

    Returns: (new_ids, lci_r, lci_emb, n_generated, exited_early)
    """
    _freeze_for(model, ring_name)
    start_ids = ids.clone()
    n_gen = 0
    exited = False

    for step in range(steps):
        gen_ids  = _generate(model, ids, block_size, temperature, n_tokens=8)
        gen_text = _ids_to_text(gen_ids)

        if do_train and quality_filter(gen_text):
            micro_train(model, gen_ids, lr=train_lr, n_steps=1)
            rag.add(gen_text, _get_emb(model, gen_ids))
            n_gen += 1

        if lci_loss_lambda > 0 and do_train:
            _lci_loss_step(model, gen_ids, train_lr * lci_loss_lambda)

        ids = gen_ids

        # Roundabout: ранний выход при резонансе
        if early_exit and (step + 1) % max(1, steps // 3) == 0:
            lci_r_check, _ = lci_from_routing(model, ids)
            if abs(lci_r_check - math.pi) < _LCI_EPSILON:
                exited = True
                break

    lci_r   = lci_from_routing(model, ids)[0]
    lci_emb = lci_from_embeddings(model, start_ids, ids)
    return ids, lci_r, lci_emb, n_gen, exited


# ── Точка пересечения (figure-8 / bidir crossing) ─────────────────────────────

def _bidir_crossing(
    model: Variant3GPT,
    ids_fwd: torch.Tensor,
    ids_rev: torch.Tensor,
    rag_fwd: RagBuffer,
    rag_rev: RagBuffer,
    rag_shared: RagBuffer,
    steps: int,
    train_lr: float,
    temperature: float,
    do_train: bool,
    block_size: int = MODEL_CFG["block_size"] - 1,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Точка пересечения (восьмёрка через META) с раздельными RAG-буферами.

    Каждый агент хранит свой контекст (rag_fwd / rag_rev).
    В точке crossing оба обмениваются 1 текстом через rag_shared —
    не полным RAG, иначе они коллапсируют в одну точку.

    Returns: (ids_fwd, ids_rev, lci_fwd, lci_rev)
    """
    _freeze_for(model, "META")
    half = max(1, steps // 2)

    # Обмен: каждый агент кладёт свой лучший текст в shared-RAG соперника
    txt_f0 = _ids_to_text(ids_fwd)
    txt_r0 = _ids_to_text(ids_rev)
    if quality_filter(txt_f0):
        rag_rev.add(txt_f0, _get_emb(model, ids_fwd))     # fwd → rev видит
        rag_shared.add(txt_f0, _get_emb(model, ids_fwd))
    if quality_filter(txt_r0):
        rag_fwd.add(txt_r0, _get_emb(model, ids_rev))     # rev → fwd видит
        rag_shared.add(txt_r0, _get_emb(model, ids_rev))

    for _ in range(half):
        # Forward: тренируется, тянется к своему RAG (не к RAG реверса)
        gen_f = _generate(model, ids_fwd, block_size, temperature, n_tokens=8)
        txt_f = _ids_to_text(gen_f)
        if do_train and quality_filter(txt_f):
            micro_train(model, gen_f, lr=train_lr, n_steps=1)
            rag_fwd.add(txt_f, _get_emb(model, gen_f))
            rag_shared.add(txt_f, _get_emb(model, gen_f))
        ids_fwd = gen_f

        # Reverse: тренируется, тянется к своему RAG
        gen_r = _generate(model, ids_rev, block_size, temperature, n_tokens=8)
        txt_r = _ids_to_text(gen_r)
        if do_train and quality_filter(txt_r):
            micro_train(model, gen_r, lr=train_lr, n_steps=1)
            rag_rev.add(txt_r, _get_emb(model, gen_r))
            rag_shared.add(txt_r, _get_emb(model, gen_r))
        ids_rev = gen_r

        # Каждый тянется к своему собственному RAG — дивергенция сохраняется
        if len(rag_fwd) > 3:
            near_f = rag_fwd.retrieve(_get_emb(model, ids_fwd), top_k=1)
            if near_f:
                ids_fwd = _encode(near_f[0], block_size)
        if len(rag_rev) > 3:
            near_r = rag_rev.retrieve(_get_emb(model, ids_rev), top_k=1)
            if near_r:
                ids_rev = _encode(near_r[0], block_size)

    lci_fwd = lci_from_routing(model, ids_fwd)[0]
    lci_rev = lci_from_routing(model, ids_rev)[0]
    return ids_fwd, ids_rev, lci_fwd, lci_rev


# ── Главный цикл: спираль Наутилуса ───────────────────────────────────────────

def nautilus_cycle(
    model: Variant3GPT,
    ids_fwd: torch.Tensor,
    ids_rev: Optional[torch.Tensor],
    rag: RagBuffer,
    rag_fwd: Optional[RagBuffer],
    rag_rev: Optional[RagBuffer],
    train_lr: float,
    temperature: float,
    do_train: bool,
    bidir: bool,
    lci_loss_lambda: float,
    step_scale: float = 1.0,
    block_size: int = MODEL_CFG["block_size"] - 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
    """
    Один цикл Наутилус-Клевера.

    Forward agent: META(10) → ABSTRACT(20) → DYNAMIC(30) → CONCRETE(40)
    Reverse agent: CONCRETE(40) → DYNAMIC(30) → ABSTRACT(20) → META(10)

    Переходы через META = figure-8 crossing с раздельными RAG-буферами.
    Дивергенция сохраняется: каждый агент живёт в своём контексте.
    """
    t0          = time.perf_counter()
    n_gen       = 0
    ring_log: Dict[str, Dict] = {}
    ids_fwd_start = ids_fwd.clone()

    if ids_rev is None and bidir:
        ids_rev = _hex_prompt(random.randint(0, 63), block_size)

    # Раздельные RAG: если не переданы — создать новые (однократное использование)
    _rag_fwd    = rag_fwd    if rag_fwd    is not None else RagBuffer(max_size=200)
    _rag_rev    = rag_rev    if rag_rev    is not None else RagBuffer(max_size=200)

    for ring in RINGS:
        name  = ring["name"]
        steps = max(1, int(ring["steps"] * step_scale))

        if bidir and name == "META":
            # ─ Точка пересечения: figure-8, раздельные RAG ─
            ids_fwd, ids_rev, lci_f, lci_r2 = _bidir_crossing(
                model, ids_fwd, ids_rev,
                _rag_fwd, _rag_rev, rag,   # shared = общий RAG для train
                steps, train_lr, temperature, do_train, block_size,
            )
            diff = abs(lci_f - lci_r2)
            ring_log[name] = {
                "steps": steps, "lci_r_fwd": round(lci_f, 4),
                "lci_r_rev": round(lci_r2, 4),
                "avg_lci": round((lci_f + lci_r2) / 2, 4),
                "divergence": round(diff, 4),
                "mode": "bidir_crossing",
            }
            mark = "✓" if abs((lci_f + lci_r2) / 2 - math.pi) < _LCI_EPSILON else "✗"
            print(f"    [{'META':10}] ×{steps:2d}шаг  lci_fwd={lci_f:.3f}  lci_rev={lci_r2:.3f}"
                  f"  Δ={diff:.3f}  bidir {mark}")
        else:
            # ─ Внутри лепестка: roundabout, используем общий RAG ─
            ids_fwd, lci_r, lci_emb, ng, early = _run_ring(
                model, name, ids_fwd, rag, steps,
                train_lr, temperature, do_train, lci_loss_lambda,
                early_exit=True, block_size=block_size,
            )
            n_gen += ng
            ring_log[name] = {
                "steps": steps, "lci_r": round(lci_r, 4),
                "lci_emb": round(lci_emb, 4),
                "early_exit": early, "gen": ng, "mode": "roundabout",
            }
            mark  = "✓" if abs(lci_r - math.pi) < _LCI_EPSILON else "✗"
            early_s = " ←ранний" if early else ""
            print(f"    [{name:10}] ×{steps:2d}шаг  "
                  f"lci_r={lci_r:.3f}  emb={lci_emb:.3f}  {mark}{early_s}")

    # Reverse agent: анти-Наутилус в своём RAG (без META, уже пройден)
    if bidir and ids_rev is not None:
        for ring_name in _SPIRAL_REV[:-1]:
            ring = _RING_BY_NAME[ring_name]
            rev_steps = max(1, int(ring["steps"] * step_scale * 0.5))
            ids_rev, _, _, ng, _ = _run_ring(
                model, ring_name, ids_rev, _rag_rev, rev_steps,
                train_lr * 0.7, temperature, do_train, 0.0,
                early_exit=True, block_size=block_size,
            )
            n_gen += ng

    # ── Итоговые метрики цикла ────────────────────────────────────────────────
    lci_r_final, _ = lci_from_routing(model, ids_fwd)
    lci_e_final    = lci_from_embeddings(model, ids_fwd_start, ids_fwd)
    elapsed        = time.perf_counter() - t0

    all_lci = []
    for v in ring_log.values():
        if "lci_r" in v:
            all_lci.append(v["lci_r"])
        elif "avg_lci" in v:
            all_lci.append(v["avg_lci"])
    n_resonant = sum(1 for lci in all_lci if abs(lci - math.pi) < _LCI_EPSILON)
    kirchhoff  = n_resonant == len(RINGS)

    result = {
        "rings":         ring_log,
        "lci_r_final":   round(lci_r_final, 4),
        "lci_emb_final": round(lci_e_final, 4),
        "n_resonant":    n_resonant,
        "kirchhoff":     kirchhoff,
        "n_generated":   n_gen,
        "elapsed_s":     round(elapsed, 2),
    }
    return ids_fwd, ids_rev, result


# ── Основная функция ───────────────────────────────────────────────────────────

def nautilus_clover(
    model: Variant3GPT,
    seed_texts: List[str],
    n_cycles: int = 4,
    step_scale: float = 1.0,
    temperature: float = 1.4,
    train_lr: float = 1e-5,
    do_train: bool = True,
    bidir: bool = True,
    lci_loss_lambda: float = 0.0,
    block_size: int = MODEL_CFG["block_size"] - 1,
) -> List[Dict]:

    total_steps_per_cycle = int(_TOTAL_STEPS * step_scale)
    agent_mode = "bidir (Forward ▶ + Reverse ◀)" if bidir else "single (Forward ▶)"

    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ НАУТИЛУС-КЛЕВЕР + HMoE")
    print(f"{'═' * 72}")
    print(f"  Циклов              : {n_cycles}")
    print(f"  Шагов/цикл          : {total_steps_per_cycle}  (={_TOTAL_STEPS}×{step_scale:.2f})")
    print(f"  Режим агентов       : {agent_mode}")
    print(f"  LCI-loss λ          : {lci_loss_lambda:.3f}  {'(активен)' if lci_loss_lambda > 0 else '(выкл)'}")
    print(f"  Температура         : {temperature:.2f}")
    print(f"\n  Кольца Пифагорейской тетрактиды (1:2:3:4):")
    for r in RINGS:
        desc = _RING_DESC[r["name"]]
        steps = max(1, int(r["steps"] * step_scale))
        print(f"    Кольцо {r['ratio']}  {r['name']:10} {steps:3d} шагов ({r['steps']*step_scale/total_steps_per_cycle*100:.0f}%)  — {desc}")
    print(f"    {'─'*50}")
    print(f"    Итого:             {total_steps_per_cycle} шагов = 100%")
    print()

    # Общий RAG (обучение + seed), раздельные RAG для bidir-агентов
    rag     = RagBuffer(max_size=500)
    rag_fwd = RagBuffer(max_size=200) if bidir else None
    rag_rev = RagBuffer(max_size=200) if bidir else None

    model.eval()
    for text in seed_texts[:50]:
        ids = _encode(text, block_size)
        emb = _get_emb(model, ids)
        rag.add(text, emb)
        if bidir:
            rag_fwd.add(text, emb)
            rag_rev.add(text, emb)
    print(f"  RAG-буфер           : {len(rag)} текстов  "
          f"({'раздельные fwd/rev' if bidir else 'единый'})")

    ids_fwd = _hex_prompt(random.randint(0, 63), block_size)
    ids_rev = _hex_prompt(random.randint(32, 63), block_size) if bidir else None

    log: List[Dict] = []

    for cycle in range(1, n_cycles + 1):
        lci_r0, _ = lci_from_routing(model, ids_fwd)
        res_mark   = "✓ РЕЗОНАНС" if abs(lci_r0 - math.pi) < _LCI_EPSILON else f"δ={lci_r0 - math.pi:+.3f}"
        print(f"\n  Цикл {cycle}/{n_cycles}  routing_LCI={lci_r0:.3f}  {res_mark}")

        ids_fwd, ids_rev, result = nautilus_cycle(
            model, ids_fwd, ids_rev, rag, rag_fwd, rag_rev,
            train_lr=train_lr,
            temperature=temperature,
            do_train=do_train,
            bidir=bidir,
            lci_loss_lambda=lci_loss_lambda,
            step_scale=step_scale,
            block_size=block_size,
        )

        kirchhoff_s = "✓ KIRCHHOFF" if result["kirchhoff"] else f"✗ {result['n_resonant']}/{len(RINGS)}"
        print(f"    → lci_r={result['lci_r_final']:.3f}  emb={result['lci_emb_final']:.3f}"
              f"  резонанс={result['n_resonant']}/{len(RINGS)}  {kirchhoff_s}"
              f"  gen={result['n_generated']}  t={result['elapsed_s']:.1f}s")

        log.append({"cycle": cycle, "lci_r0": round(lci_r0, 4), **result})

    # ── Итог ──────────────────────────────────────────────────────────────────
    n_kirchhoff = sum(1 for r in log if r["kirchhoff"])
    avg_lci     = sum(r["lci_r_final"] for r in log) / len(log) if log else 0.0
    avg_emb     = sum(r["lci_emb_final"] for r in log) / len(log) if log else 0.0

    print(f"\n{'─' * 72}")
    print(f"  ИТОГ НАУТИЛУС-КЛЕВЕРА:")
    print(f"    Kirchhoff-сбалансированных циклов: {n_kirchhoff}/{n_cycles}")
    print(f"    avg_LCI_routing : {avg_lci:.3f}  (цель π={math.pi:.3f})")
    print(f"    avg_LCI_emb     : {avg_emb:.3f}  (цель π)")
    print(f"    RAG-буфер       : {len(rag)} текстов")

    return log


# ── Загрузка модели и seed-текстов ────────────────────────────────────────────

def _load_model(path: str) -> Variant3GPT:
    cfg = Variant3Config(**MODEL_CFG)
    m   = Variant3GPT(cfg)
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=True)
        m.load_state_dict(ck.get("model_state", ck), strict=False)
        print(f"  Загружен: {path}")
    else:
        print(f"  [!] Чекпоинт не найден: {path} — случайные веса")
    m.to(DEVICE)
    print(f"  Модель: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M параметров")
    return m


def _load_seeds(block_size: int) -> List[str]:
    texts = [
        "def forward(self, x): return self.linear(x)",
        "loss.backward(); optimizer.zero_grad(); scheduler.step()",
        "x = x + self.crossing(out_a, out_b)",
        "The hexagram represents the intersection of abstract and concrete.",
        "consciousness emerges from recursive self-reference in Q6 space",
        "The Pythagorean tetractys: 1+2+3+4=10, the decade of nature",
        "spiral growth: each ring doubles the previous circumference",
        "nautilus shell: logarithmic spiral, golden ratio in nature",
        "φ = (1 + √5) / 2 ≈ 1.618, growth proportional to itself",
        "clover leaf: four petals sharing one central node",
    ] * 5
    for h in range(64):
        texts.append(_ids_to_text(_hex_prompt(h, block_size)))
    return texts


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Наутилус-Клевер: само-обучение HMoE по геометрии 4-лепесткового клевера"
    )
    parser.add_argument("--checkpoint",  type=str,   default="hmoe_self_trained_v5.pt")
    parser.add_argument("--fast",        action="store_true",
                        help="быстрый тест: 2 цикла, 0.3× шагов")
    parser.add_argument("--cycles",      type=int,   default=4,
                        help="число Наутилус-циклов (каждый = 100 шагов × scale)")
    parser.add_argument("--step-scale",  type=float, default=1.0,
                        help="множитель шагов (0.5 = 50 шагов/цикл, 1.0 = 100)")
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--no-train",    action="store_true")
    parser.add_argument("--no-bidir",    action="store_true",
                        help="только Forward агент (без встречного)")
    parser.add_argument("--lci-loss",    type=float, default=0.0,
                        help="λ для Kirchhoff LCI-loss (0 = выкл)")
    parser.add_argument("--save",        type=str,   default="hmoe_nautilus_clover_v1.pt")
    args = parser.parse_args()

    block_size = MODEL_CFG["block_size"] - 1

    if args.fast:
        args.cycles    = 2
        args.step_scale = 0.3   # 30 шагов/цикл вместо 100

    print(f"\n{'═' * 72}")
    print(f"  НАУТИЛУС-КЛЕВЕР HMoE")
    print(f"{'═' * 72}")
    model      = _load_model(args.checkpoint)
    seed_texts = _load_seeds(block_size)
    print(f"  Seed текстов: {len(seed_texts)}")

    log = nautilus_clover(
        model,
        seed_texts=seed_texts,
        n_cycles=args.cycles,
        step_scale=args.step_scale,
        temperature=args.temperature,
        train_lr=args.lr,
        do_train=not args.no_train,
        bidir=not args.no_bidir,
        lci_loss_lambda=args.lci_loss,
        block_size=block_size,
    )

    # ── Сохранение ──────────────────────────────────────────────────────────
    save_path = args.save
    torch.save(
        {"model_state": model.state_dict(), "log": log, "config": MODEL_CFG},
        save_path,
    )
    log_path = save_path.replace(".pt", "_log.json")
    import json
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n  Сохранено: {save_path}")
    print(f"  Лог:       {log_path}")


if __name__ == "__main__":
    main()
