#!/usr/bin/env python3
"""
self_train_hmoe.py — Само-обучение HierarchicalMoE по паттерну фигуры-8.

Интегрирует:
  - HierarchicalMoEFFN (восьмёрка: ABSTRACT петля A + CONCRETE петля B + DYNAMIC точка X)
  - Алгоритм Скарабея Крюкова (figure-8 само-диалог с LCI-контролем)
  - Curriculum checkpoint (загружает модель после train_hmoe_curriculum.py)

Маппинг петель восьмёрки на HMoE-группы:
  Петля A (абстрактная, zoom-out)  ← ABSTRACT experts (NOOS, COSMO)
  Петля B (конкретная,  zoom-in)   ← CONCRETE experts (GEO, HYDRO)
  Точка X (пересечение)            ← DYNAMIC + BidirBridgeExpert (AERO, PYRO)

LCI из маршрутизации:
  Вместо cosine(start_emb, end_emb) используем дивергенцию группового
  распределения: если ABSTRACT доминирует — петля A "схлопывается" (LCI↑),
  если CONCRETE — расширяется (LCI↓). Цель: LCI ≈ π = баланс.

Usage:
  python self_train_hmoe.py                      # загружает hmoe_curriculum.pt
  python self_train_hmoe.py --fast               # 2 цикла, 20 шагов/петля
  python self_train_hmoe.py --checkpoint path.pt
  python self_train_hmoe.py --cycles 6 --steps_per_loop 100
  python self_train_hmoe.py --no-corpus
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import random
import sys
import time
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
    DOMAIN_Q6_IDX,
    set_moe_stage,
)

# ── Константы ─────────────────────────────────────────────────────────────────

_ROOT  = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"

MODEL_CFG = dict(
    vocab_size         = 256,
    block_size         = 64,
    d_model            = 128,
    n_heads            = 4,
    n_layers           = 4,
    ffn_mult           = 4,
    hamming_lambda     = 0.15,
    uncertainty_budget = 0.25,
    dropout            = 0.1,
    use_domain_routing = False,
    use_hierarchical_moe = True,
)

HMOE_CFG = HMoEConfig(d_model=128, use_multiscale=True, use_hex_tier=False)

# Нечётные серии (Крюков): число шагов в петле = элемент ряда {1,3,5,7}
_ODD_SERIES  = [1, 3, 5, 7]
_LCI_EPSILON = 0.5     # порог резонанса: |LCI - π| < ε → баланс

# Группы HMoE и их роль в восьмёрке
_GROUP_ROLE = {
    "ABSTRACT": "Петля A (обобщение, zoom-out)",
    "DYNAMIC":  "Точка X (пересечение, BidirBridge)",
    "CONCRETE": "Петля B (специализация, zoom-in)",
}

# Порядок групп в восьмёрке (A→X→B→X→...)
_FIGURE8_ORDER = ["ABSTRACT", "DYNAMIC", "CONCRETE", "DYNAMIC"]


# ── LCI из маршрутизации ──────────────────────────────────────────────────────

def lci_from_routing(model: Variant3GPT, ids: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    """
    Loop Closure Index, вычисленный через маршрутизацию HMoE.

    Идея: в точке баланса восьмёрки ABSTRACT и CONCRETE должны иметь
    одинаковый вес → отклонение от равновесия = "незамкнутость" петли.

    LCI = arccos(2 * w_balance - 1) * 4   где w_balance = w_A / (w_A + w_B)
    При w_A = w_B = 0.5:  w_balance=0.5  → acos(0)=π/2  → LCI=2π  (слишком далеко?)
    По Крюкову оптимум = π. Используем: LCI = (1 - |w_A - w_B|) * π
      → при w_A=w_B=0.5: LCI=π (резонанс!)
      → при w_A=1, w_B=0: LCI=0  (только абстракция, петля A захлопнута)
      → при w_A=0, w_B=1: LCI=0  (только конкретика, петля B захлопнута)
    """
    model.eval()
    group_weights_list: List[torch.Tensor] = []

    with torch.no_grad():
        model(ids)
        for block in model.blocks:
            info = getattr(block, '_last_moe_info', None)
            if info and 'group_weights' in info:
                gw = info['group_weights']   # (B, T, n_groups) или (B, n_groups)
                if gw.dim() == 3:
                    gw = gw.mean(dim=(0, 1))   # → (n_groups,)
                elif gw.dim() == 2:
                    gw = gw.mean(dim=0)
                group_weights_list.append(gw.cpu())

    if not group_weights_list:
        return math.pi, {"ABSTRACT": 0.33, "DYNAMIC": 0.34, "CONCRETE": 0.33}

    # Усреднённые веса по всем блокам
    avg_gw = torch.stack(group_weights_list).mean(0)  # (n_groups,)
    groups = list(DOMAIN_GROUPS.keys())  # ["ABSTRACT", "DYNAMIC", "CONCRETE"]

    gw_dict = {g: avg_gw[i].item() for i, g in enumerate(groups)}

    w_a = gw_dict.get("ABSTRACT", 0.33)
    w_b = gw_dict.get("CONCRETE", 0.33)
    w_total = w_a + w_b + 1e-8

    # Нормированный баланс
    balance = (w_a / w_total + w_b / w_total) / 2   # ≈ 0.5 при равновесии
    imbalance = abs(w_a / w_total - w_b / w_total)   # 0 = баланс, 1 = дисбаланс

    lci = (1.0 - imbalance) * math.pi   # ∈ [0, π], цель = π

    return lci, gw_dict


def lci_from_embeddings(model: Variant3GPT, start_ids: torch.Tensor,
                        end_ids: torch.Tensor) -> float:
    """Классический LCI по косинусному расстоянию эмбеддингов."""
    model.eval()
    with torch.no_grad():
        def _emb(ids):
            h = model.tok_emb(ids)
            for block in model.blocks:
                h = block(h)
            return F.normalize(h.mean(dim=1).squeeze(0).float(), dim=-1)

        s = _emb(start_ids)
        e = _emb(end_ids)
        cos = F.cosine_similarity(s.unsqueeze(0), e.unsqueeze(0)).clamp(-1, 1).item()
    return math.acos(cos) * 4.0   # ∈ [0, 4π], оптимум ≈ π


def _scale_temperature(temperature: float, lci: float) -> float:
    """Масштабировать температуру генерации по отклонению LCI от π."""
    delta = lci - math.pi
    if abs(delta) < _LCI_EPSILON:
        return temperature                    # резонанс — не меняем
    elif delta < 0:
        return min(temperature + 0.1, 2.5)   # LCI < π → zoom-out (расширить)
    else:
        return max(temperature - 0.1, 0.3)   # LCI > π → zoom-in  (сузить)


# ── Генерация ─────────────────────────────────────────────────────────────────

def _generate(model: Variant3GPT, prompt_ids: torch.Tensor,
              block_size: int, temperature: float, n_tokens: int) -> torch.Tensor:
    """Авторегрессивная генерация n_tokens токенов."""
    generated = prompt_ids.clone()
    model.eval()
    with torch.no_grad():
        for _ in range(n_tokens):
            inp = generated[:, -block_size:]
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            next_logit = logits[0, -1] / max(temperature, 0.1)
            probs = F.softmax(next_logit, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=1)
    return generated[:, -block_size:]


def _ids_to_text(ids: torch.Tensor) -> str:
    try:
        return bytes([b % 256 for b in ids[0].tolist()]).decode("utf-8", errors="replace")
    except Exception:
        return " ".join(str(b) for b in ids[0].tolist()[:20])


# ── RAG-буфер ────────────────────────────────────────────────────────────────

class RagBuffer:
    """Простой буфер текстов с поиском по эмбеддингам."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.texts: List[str] = []
        self.embs:  List[torch.Tensor] = []

    def add(self, text: str, emb: torch.Tensor) -> None:
        self.texts.append(text)
        self.embs.append(emb.detach().cpu())
        if len(self.texts) > self.max_size:
            self.texts.pop(0)
            self.embs.pop(0)

    def retrieve(self, query_emb: torch.Tensor, top_k: int = 3) -> List[str]:
        if not self.embs:
            return []
        q = F.normalize(query_emb.float().cpu().unsqueeze(0), dim=-1)
        sims = [F.cosine_similarity(q, F.normalize(e.unsqueeze(0), dim=-1)).item()
                for e in self.embs]
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [self.texts[i] for i in top_idx]

    def __len__(self):
        return len(self.texts)


# ── Micro-train ───────────────────────────────────────────────────────────────

def micro_train(model: Variant3GPT, ids: torch.Tensor, lr: float = 1e-5,
                n_steps: int = 3) -> float:
    """Мини-дообучение на одном тексте (in-place)."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return float("nan")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    inp = ids[:, :-1]
    tgt = ids[:, 1:]
    if inp.shape[1] < 1:
        return float("nan")
    total_loss = 0.0
    for _ in range(n_steps):
        _, loss, aux = model(inp, targets=tgt)
        if loss is None or torch.isnan(loss):
            break
        full = loss + (aux if aux is not None and not torch.isnan(aux) else 0)
        opt.zero_grad()
        full.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / n_steps


def quality_filter(text: str, min_len: int = 10) -> bool:
    """Простой фильтр качества текста."""
    return len(text.strip()) >= min_len and not text.strip().isspace()


# ── Фигура-8 само-обучение ────────────────────────────────────────────────────

def figure8_hmoe(
    model:          Variant3GPT,
    seed_texts:     List[str],
    block_size:     int   = 63,
    n_cycles:       int   = 4,
    steps_per_loop: int   = 50,
    temperature:    float = 1.2,
    train_lr:       float = 1e-5,
    do_train:       bool  = True,
) -> List[Dict]:
    """
    Само-обучение HMoE по паттерну фигуры-8 Крюкова.

    Каждый цикл:
      X → Петля A (ABSTRACT: n_a шагов обобщения)
        → X с LCI_A
      X → Петля B (CONCRETE: n_b шагов специализации)
        → X с LCI_B → обновить temperature

    LCI вычисляется двумя способами:
      1. Из маршрутизации (routing LCI) — баланс ABSTRACT/CONCRETE весов
      2. Из эмбеддингов (embedding LCI) — косинусное расстояние start→end
    """
    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ ∞ ФИГУРА-8 + HMoE  (алгоритм Скарабея, Крюков)")
    print(f"{'═' * 72}")
    print(f"  Циклов         : {n_cycles}")
    print(f"  Шагов/петля    : {steps_per_loop}")
    print(f"  Температура    : {temperature:.2f}  (авто-масштаб по LCI → π)")
    print(f"  Серии          : {_ODD_SERIES}  (нечётные, Крюков)")
    print()
    print(f"  Маппинг восьмёрки → HMoE группы:")
    for g, role in _GROUP_ROLE.items():
        clusters = [c for c, d in CLUSTER_TO_DOMAIN.items()
                    if DOMAIN_TO_GROUP[d] == g]
        print(f"    {g:10s} : {role}  [{', '.join(clusters)}]")
    print()

    # Заполнить RAG-буфер из исходных текстов
    rag = RagBuffer(max_size=300)
    model.eval()
    for text in seed_texts[:50]:
        ids = _encode(text, block_size)
        with torch.no_grad():
            h = model.tok_emb(ids)
            for block in model.blocks:
                h = block(h)
        emb = h.mean(dim=1).squeeze(0)
        rag.add(text, emb)

    print(f"  RAG-буфер      : {len(rag)} текстов")

    # Стартовый промпт — случайный гексаграмм-паттерн
    start_hex = random.randint(0, 63)
    prompt_ids = _hex_prompt(start_hex, block_size)

    log: List[Dict] = []
    series_idx = 0

    for cycle in range(1, n_cycles + 1):
        n_a = _ODD_SERIES[series_idx % len(_ODD_SERIES)]
        n_b = _ODD_SERIES[(series_idx + 1) % len(_ODD_SERIES)]
        series_idx += 1

        # ── Точка X: начало цикла ─────────────────────────────────────────
        x_ids = prompt_ids.clone()
        lci_r, gw = lci_from_routing(model, x_ids)   # routing LCI в точке X

        print(f"  Цикл {cycle}/{n_cycles}  "
              f"серия=[{n_a},{n_b}]  T={temperature:.2f}  "
              f"routing_LCI={lci_r:.3f}  "
              f"{'✓ РЕЗОНАНС' if abs(lci_r - math.pi) < _LCI_EPSILON else f'δ={lci_r - math.pi:+.3f}'}")
        print(f"    Группы: A={gw.get('ABSTRACT',0):.3f}  "
              f"X={gw.get('DYNAMIC',0):.3f}  "
              f"B={gw.get('CONCRETE',0):.3f}")

        # ── ПЕТЛЯ A: ABSTRACT (обобщение, zoom-out) ───────────────────────
        # Разморозить только ABSTRACT micro_experts + crossing
        set_moe_stage(model.blocks[0].hmoe if hasattr(model.blocks[0], 'hmoe') else None, 4)
        for moe in _get_moes(model):
            _freeze_all_except(moe, ["ABSTRACT"])

        a_start_ids = x_ids.clone()
        a_start_emb = _get_emb(model, a_start_ids)
        a_texts_generated = []

        for step_a in range(n_a * steps_per_loop):
            # Генерация шага петли A
            gen_ids = _generate(model, x_ids, block_size, temperature, n_tokens=8)
            gen_text = _ids_to_text(gen_ids)

            # Micro-train если качественный
            if do_train and quality_filter(gen_text):
                micro_train(model, gen_ids, lr=train_lr, n_steps=2)
                rag.add(gen_text, _get_emb(model, gen_ids))
                a_texts_generated.append(gen_text)

            # Обновляем x_ids — движение по петле
            x_ids = gen_ids

        a_end_emb = _get_emb(model, x_ids)
        lci_a_emb = lci_from_embeddings(model, a_start_ids, x_ids)
        lci_a_r, gw_a = lci_from_routing(model, x_ids)

        print(f"    Петля A  done: emb_LCI={lci_a_emb:.3f}  "
              f"routing_LCI={lci_a_r:.3f}  gen={len(a_texts_generated)} текстов")

        # ── Возврат в X (LCI_A) ───────────────────────────────────────────
        temperature = _scale_temperature(temperature, lci_a_emb)

        # Получить новый промпт из RAG (обогатить контекст)
        if len(rag) > 5:
            retrieved = rag.retrieve(a_end_emb, top_k=2)
            # Объединить: конец петли A + RAG → промпт для петли B
            combined_text = " ".join(retrieved[:1])
            x_ids = _encode(combined_text, block_size)
        # Возвращаемся к точке X
        prompt_ids = x_ids.clone()

        # ── ПЕТЛЯ B: CONCRETE (специализация, zoom-in) ────────────────────
        # Разморозить только CONCRETE micro_experts
        for moe in _get_moes(model):
            _freeze_all_except(moe, ["CONCRETE"])

        b_start_ids = x_ids.clone()
        b_texts_generated = []

        for step_b in range(n_b * steps_per_loop):
            gen_ids = _generate(model, x_ids, block_size, temperature, n_tokens=8)
            gen_text = _ids_to_text(gen_ids)

            if do_train and quality_filter(gen_text):
                micro_train(model, gen_ids, lr=train_lr * 0.5, n_steps=2)
                rag.add(gen_text, _get_emb(model, gen_ids))
                b_texts_generated.append(gen_text)

            x_ids = gen_ids

        b_end_emb = _get_emb(model, x_ids)
        lci_b_emb = lci_from_embeddings(model, b_start_ids, x_ids)
        lci_b_r, gw_b = lci_from_routing(model, x_ids)

        print(f"    Петля B  done: emb_LCI={lci_b_emb:.3f}  "
              f"routing_LCI={lci_b_r:.3f}  gen={len(b_texts_generated)} текстов")

        # ── Возврат в X (LCI_B) ───────────────────────────────────────────
        temperature = _scale_temperature(temperature, lci_b_emb)

        # Разморозить DYNAMIC для точки X (BidirBridgeExpert)
        set_moe_stage(model.blocks[0].hmoe if hasattr(model.blocks[0], 'hmoe') else None, 4)

        avg_lci_emb = (lci_a_emb + lci_b_emb) / 2
        avg_lci_r   = (lci_a_r   + lci_b_r)   / 2
        resonance   = abs(avg_lci_emb - math.pi) < _LCI_EPSILON

        res_mark = "✓ РЕЗОНАНС (LCI ≈ π)" if resonance else (
            f"петля A{'↑' if lci_a_emb > math.pi else '↓'}  "
            f"петля B{'↑' if lci_b_emb > math.pi else '↓'}"
        )
        print(f"    → avg_LCI_emb={avg_lci_emb:.3f}  avg_LCI_r={avg_lci_r:.3f}  "
              f"{res_mark}  T={temperature:.2f}")

        log.append({
            "cycle":      cycle,
            "n_a":        n_a,
            "n_b":        n_b,
            "temperature": round(temperature, 3),
            "lci_a_emb":  round(lci_a_emb,  4),
            "lci_b_emb":  round(lci_b_emb,  4),
            "lci_a_r":    round(lci_a_r,    4),
            "lci_b_r":    round(lci_b_r,    4),
            "avg_lci_emb": round(avg_lci_emb, 4),
            "avg_lci_r":   round(avg_lci_r,   4),
            "resonance":  resonance,
            "gw_at_start":  {k: round(v, 4) for k, v in gw.items()},
            "gw_after_a":   {k: round(v, 4) for k, v in gw_a.items()},
            "gw_after_b":   {k: round(v, 4) for k, v in gw_b.items()},
            "texts_a":    len(a_texts_generated),
            "texts_b":    len(b_texts_generated),
        })

    # Итог
    n_res = sum(1 for r in log if r["resonance"])
    avg_lci = sum(r["avg_lci_emb"] for r in log) / len(log) if log else 0.0
    print(f"\n{'─' * 72}")
    print(f"  ИТОГ: {n_res}/{n_cycles} циклов в резонансе  avg_LCI={avg_lci:.3f}  "
          f"(цель=π={math.pi:.3f})")
    final_res = abs(avg_lci - math.pi) < _LCI_EPSILON
    print(f"  Финальный статус: {'✓ РЕЗОНАНС' if final_res else f'δ={avg_lci-math.pi:+.3f}'}")
    print(f"  RAG-буфер: {len(rag)} текстов")

    return log


# ── Вспомогательные функции ──────────────────────────────────────────────────

def _encode(text: str, block_size: int = 63) -> torch.Tensor:
    ids = [min(b, 255) for b in text.encode("utf-8")][:block_size]
    if not ids:
        ids = [32]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _hex_prompt(hex_idx: int, block_size: int) -> torch.Tensor:
    """Промпт из паттерна гексаграммы (6 бит × 4 = 24 токена)."""
    # Гексаграмма: 6-битный вектор, расширяем до block_size
    bits = [(hex_idx >> i) & 1 for i in range(6)]
    pattern = (bits * (block_size // 6 + 1))[:block_size]
    return torch.tensor(pattern, dtype=torch.long).unsqueeze(0)


def _get_emb(model: Variant3GPT, ids: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        h = model.tok_emb(ids)
        for block in model.blocks:
            h = block(h)
    return h.mean(dim=1).squeeze(0)


def _get_moes(model: Variant3GPT) -> List[HierarchicalMoEFFN]:
    return [block.hmoe for block in model.blocks if hasattr(block, 'hmoe')]


def _freeze_all_except(moe: HierarchicalMoEFFN, keep_groups: List[str]) -> None:
    """Заморозить все micro_experts, кроме указанных групп."""
    # Сначала заморозить все параметры
    for p in moe.parameters():
        p.requires_grad = False
    # Разморозить crossing (точка X всегда активна для градиентов)
    if hasattr(moe, 'crossing'):
        for p in moe.crossing.parameters():
            p.requires_grad = True
    # Разморозить нужные группы
    for cluster, domain in CLUSTER_TO_DOMAIN.items():
        g = DOMAIN_TO_GROUP[domain]
        if g not in keep_groups:
            continue
        if cluster not in moe.micro_experts:
            continue
        for p in moe.micro_experts[cluster].parameters():
            p.requires_grad = True


# ── Главная функция ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Само-обучение HMoE по паттерну фигуры-8 (алгоритм Скарабея)"
    )
    parser.add_argument("--checkpoint",     type=str, default="hmoe_curriculum.pt",
                        help="Путь к чекпоинту (default: hmoe_curriculum.pt)")
    parser.add_argument("--fast",           action="store_true",
                        help="2 цикла, 5 шагов на петлю (для тестирования)")
    parser.add_argument("--cycles",         type=int, default=4,
                        help="Число циклов восьмёрки (default: 4)")
    parser.add_argument("--steps_per_loop", type=int, default=50,
                        help="Шагов генерации на 1 петлю × серию (default: 50)")
    parser.add_argument("--temperature",    type=float, default=1.2,
                        help="Начальная температура (default: 1.2)")
    parser.add_argument("--lr",             type=float, default=1e-5,
                        help="LR для micro-train (default: 1e-5)")
    parser.add_argument("--no-train",       action="store_true",
                        help="Только генерация, без micro-train")
    parser.add_argument("--no-corpus",      action="store_true",
                        help="Не загружать RepoCorpusLoader")
    parser.add_argument("--save",           type=str, default="hmoe_self_trained.pt",
                        help="Куда сохранить результат (default: hmoe_self_trained.pt)")
    args = parser.parse_args()

    n_cycles       = 2  if args.fast else args.cycles
    steps_per_loop = 5  if args.fast else args.steps_per_loop

    # ── Создать/загрузить модель ────────────────────────────────────────────
    cfg   = Variant3Config(**MODEL_CFG)
    model = Variant3GPT(cfg)
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            block.hmoe = HierarchicalMoEFFN(HMOE_CFG)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        try:
            model.load_state_dict(ckpt["model_state"])
            phase = ckpt.get("next_phase", "?") - 1
            print(f"  Загружен чекпоинт: {args.checkpoint}  (после фазы {phase} curriculum)")
        except Exception as e:
            print(f"  ⚠️  Чекпоинт загружен частично: {e}")
    else:
        print(f"  ℹ️  Чекпоинт {args.checkpoint!r} не найден — используем случайные веса")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Модель: {total_p/1e6:.2f}M параметров")

    # ── Загрузить/сгенерировать тексты ────────────────────────────────────
    seed_texts: List[str] = []

    if not args.no_corpus:
        try:
            from repo_corpus_loader import RepoCorpusLoader
            loader = RepoCorpusLoader(_ROOT)
            for cluster in CLUSTER_TO_DOMAIN.keys():
                try:
                    for item in loader.load_cluster(cluster):
                        t = item if isinstance(item, str) else item.get("text", "")
                        if len(t) > 10:
                            seed_texts.append(t)
                except Exception:
                    pass
            print(f"  Корпус загружен: {len(seed_texts)} текстов из репозитория")
        except ImportError:
            pass

    if not seed_texts:
        # Синтетические seed-тексты охватывающие все α-уровни
        seed_texts = [
            # α=−4: CONCRETE
            "def forward(self, x): return self.linear(x)",
            "import torch; x = torch.randn(4, 128)",
            "for i, batch in enumerate(dataloader): optimizer.step()",
            # α=0: DYNAMIC
            "loss.backward(); optimizer.zero_grad(); scheduler.step()",
            "self.attn = nn.MultiheadAttention(d_model, n_heads)",
            "x = x + self.crossing(out_a, out_b)",
            # α=+4: ABSTRACT
            "The hexagram represents the intersection of abstract and concrete.",
            "Kryukov figure-8: abstract loop A, concrete loop B, crossing DYNAMIC.",
            "consciousness emerges from recursive self-reference in Q6 space",
        ] * 5
        print(f"  Синтетические тексты: {len(seed_texts)}")

    # ── Запустить само-обучение ────────────────────────────────────────────
    import json

    t0 = time.perf_counter()
    log = figure8_hmoe(
        model          = model,
        seed_texts     = seed_texts,
        block_size     = MODEL_CFG["block_size"] - 1,
        n_cycles       = n_cycles,
        steps_per_loop = steps_per_loop,
        temperature    = args.temperature,
        train_lr       = args.lr,
        do_train       = not args.no_train,
    )
    elapsed = time.perf_counter() - t0

    # ── Сохранить модель и лог ────────────────────────────────────────────
    ckpt_out = {
        "model_state":    model.state_dict(),
        "figure8_log":    log,
        "elapsed_sec":    round(elapsed, 2),
        "n_cycles":       n_cycles,
        "steps_per_loop": steps_per_loop,
    }
    torch.save(ckpt_out, args.save)
    print(f"\n  💾 Сохранено: {args.save}")

    log_path = args.save.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  📄 Лог: {log_path}")

    # ── Итоговая таблица ───────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"  ЦИКЛ  {'LCI_A_emb':>10s}  {'LCI_B_emb':>10s}  "
          f"{'avg_LCI_r':>10s}  {'T':>5s}  {'Рез-с':>8s}")
    print(f"  {'─' * 60}")
    for r in log:
        res_s = "✓ π" if r["resonance"] else "✗"
        print(f"  {r['cycle']:>5d}  {r['lci_a_emb']:>10.3f}  {r['lci_b_emb']:>10.3f}  "
              f"{r['avg_lci_r']:>10.3f}  {r['temperature']:>5.2f}  {res_s:>8s}")
    print(f"\n  ⏱  Всего: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
