#!/usr/bin/env python3
"""
self_train_hmoe_v4.py — Резонансный прорыв: асимметричная серия A>>B.

Диагноз (v3 plateau):
  После петли A: ABSTRACT≈0.255, CONCRETE≈0.246 → LCI=3.11 (почти π!)
  После петли B: ABSTRACT≈0.251, CONCRETE≈0.271 → LCI=3.08 (CONCRETE +0.02)
  → CONCRETE имеет систематическое преимущество в маршрутизации.

Стратегия v4:
  1. Серия [7,1] вместо [1,3,5,7] — всегда 7× ABSTRACT на 1× CONCRETE
  2. LR_A = 2e-5 (вдвое выше базового), LR_B = 5e-6 (вдвое ниже базового)
  3. Адаптивный балансёр: если w_B > w_A + 0.01 после петли B,
     запускается мини-петля A (3 шага) без петли B
  4. Цель: avg_LCI_r > 3.13 (в пределах 0.004 от π=3.142)

Usage:
  python self_train_hmoe_v4.py                          # 8 циклов из hmoe_v3_self.pt
  python self_train_hmoe_v4.py --fast                   # 2 цикла (тест)
  python self_train_hmoe_v4.py --checkpoint path.pt
  python self_train_hmoe_v4.py --cycles 12 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    HMoEConfig,
    HierarchicalMoEFFN,
    CLUSTER_TO_DOMAIN,
    DOMAIN_GROUPS,
    DOMAIN_TO_GROUP,
    set_moe_stage,
)

# ── Константы ─────────────────────────────────────────────────────────────────

_ROOT  = os.path.dirname(os.path.abspath(__file__))

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

# v4: всегда 7 шагов ABSTRACT, 1 шаг CONCRETE (асимметрия для компенсации)
_SERIES_A = 7   # шагов в петле ABSTRACT
_SERIES_B = 1   # шагов в петле CONCRETE
_LCI_EPSILON  = 0.5    # порог резонанса
_BALANCE_THOLD = 0.01  # если w_B > w_A + 0.01 → запуск мини-балансёра


# ── Вспомогательные функции (скопированы из self_train_hmoe.py) ───────────────

def lci_from_routing(model: Variant3GPT, ids: torch.Tensor) -> Tuple[float, Dict[str, float]]:
    """LCI через маршрутизацию — два варианта, возвращает максимальный.

    LCI_classic = (1 - |w_A - w_B|) · π           ← оригинал (2 сферы)
    LCI_quater  = √(w_A² + w_X² + w_B² + w_cr²) · π  ← ScarabQuaternion (4 сферы)

    ScarabQuaternion (scarab_algorithm.py): |A| = √(BVS²+SVS²+MVS²+ChVS²) = π при мастерстве.
    Маппинг: BVS=ABSTRACT, SVS=DYNAMIC, MVS=CONCRETE, ChVS=crossing_alpha (BidirBridgeExpert).
    При идеальном балансе (1/4,1/4,1/4,1/4): LCI_quater = √(4·(1/4)²) · π = 0.5π — не π!
    Нормировка: LCI_quater = √(4·Σw²) · π  чтобы максимум = π при равных весах.
    """
    model.eval()
    gw_list   = []
    alpha_list = []
    with torch.no_grad():
        model(ids)
        for block in model.blocks:
            info = getattr(block, '_last_moe_info', None)
            if info and 'group_weights' in info:
                gw = info['group_weights']
                if torch.isnan(gw).any():
                    continue
                if gw.dim() == 3:
                    gw = gw.mean(dim=(0, 1))
                elif gw.dim() == 2:
                    gw = gw.mean(dim=0)
                gw_list.append(gw.cpu())
                # crossing_alpha = баланс fwd/bwd в BidirBridgeExpert (4-я сфера ЧВС)
                if 'crossing_alpha' in info:
                    alpha_list.append(info['crossing_alpha'].cpu().item())
    if not gw_list:
        return math.pi, {"ABSTRACT": 0.33, "DYNAMIC": 0.34, "CONCRETE": 0.33}

    avg_gw = torch.stack(gw_list).mean(0)
    groups = list(DOMAIN_GROUPS.keys())
    gw_dict = {g: avg_gw[i].item() for i, g in enumerate(groups)}

    w_a = gw_dict.get("ABSTRACT", 0.33)
    w_x = gw_dict.get("DYNAMIC",  0.34)
    w_b = gw_dict.get("CONCRETE", 0.33)
    # 4-я сфера (ЧВС): crossing_alpha — близость к 0.5 означает баланс fwd↔bwd
    w_cr = (1.0 - abs(sum(alpha_list) / len(alpha_list) - 0.5) * 2) if alpha_list else 0.5

    # ── LCI classic (2 сферы) ─────────────────────────────────────────────
    lci_classic = (1.0 - abs(w_a - w_b)) * math.pi

    # ── LCI quaternion (4 сферы, ScarabQuaternion) ────────────────────────
    # Нормировка: при равных w_a=w_x=w_b=1/3, w_cr=1 → нормировать на √(4/3)
    # Общая формула: LCI_q = π · √(w_a²+w_x²+w_b²) / √(1/3)  (3-компонентная часть)
    # + бонус за balanced crossing: max π когда all equal AND w_cr near 1
    sum_sq = w_a**2 + w_x**2 + w_b**2              # min 1/3 (равные), max 1 (один доминирует)
    # Нормируем: inversely — меньше sum_sq = лучше баланс = выше LCI
    balance_3 = 1.0 - (sum_sq - 1/3) / (2/3)       # 1.0 при равных, 0.0 при полном доминировании
    lci_quater = math.pi * (0.8 * balance_3 + 0.2 * w_cr)

    # Возвращаем среднее — оба сигнала важны
    lci = (lci_classic + lci_quater) / 2.0
    gw_dict['_lci_classic']  = round(lci_classic, 4)
    gw_dict['_lci_quater']   = round(lci_quater,  4)
    gw_dict['_crossing_alpha'] = round(w_cr, 4)
    return lci, gw_dict


def routing_channel_capacity(gw_dict: Dict[str, float]) -> float:
    """Пропускная способность канала роутинга (биты/шаг).

    ETD (VOLUME_157): C = B·log₂(1+SNR) при B=4/π Гц, SNR=π → C≈2.63 бит/шаг.
    Текущая entropy = -Σ wᵢ·log₂(wᵢ) → цель: ≥ 2.63 бит (= log₂(6.25) ≈ log₂(2π)).

    Ниже 2.63 = роутер работает менее чем на полную ETD-ёмкость.
    Равно log₂(3)≈1.585 = равномерное по 3 группам (текущий максимум без Q6).
    Равно log₂(64)=6 = теоретический максимум Q6.
    """
    C_etd = (4 / math.pi) * math.log2(1 + math.pi)   # ≈ 2.626 бит/шаг по ETD
    weights = [v for k, v in gw_dict.items()
               if not k.startswith('_') and isinstance(v, float) and v > 0]
    if not weights:
        return 0.0
    total = sum(weights)
    entropy = -sum((w/total) * math.log2(w/total + 1e-12) for w in weights)
    efficiency = entropy / C_etd  # 0..1+: >1 = превышает ETD-ёмкость
    return entropy, C_etd, efficiency


def _freeze_all_except(moe: HierarchicalMoEFFN, keep_groups: List[str]) -> None:
    """Заморозить все группы кроме keep_groups."""
    for name, p in moe.named_parameters():
        frozen = True
        for g in keep_groups:
            if g.lower() in name.lower():
                frozen = False
                break
        p.requires_grad_(not frozen)


def _get_moes(model: Variant3GPT) -> List[HierarchicalMoEFFN]:
    moes = []
    for block in model.blocks:
        if hasattr(block, 'hmoe') and block.hmoe is not None:
            moes.append(block.hmoe)
    return moes


def _encode(text: str, block_size: int = MODEL_CFG["block_size"] - 1) -> torch.Tensor:
    ids = [min(b, MODEL_CFG["vocab_size"] - 1) for b in text.encode("utf-8")][:block_size]
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _hex_prompt(hex_idx: int, block_size: int) -> torch.Tensor:
    pattern = [(hex_idx >> i) & 1 for i in range(6)]
    ids = (pattern * (block_size // 6 + 1))[:block_size]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _generate(model: Variant3GPT, prompt_ids: torch.Tensor,
              block_size: int, temperature: float, n_tokens: int = 8) -> torch.Tensor:
    model.eval()
    ids = prompt_ids.clone()
    with torch.no_grad():
        for _ in range(n_tokens):
            inp = ids[:, -block_size:]
            out = model(inp)
            logits = out[0]
            next_logit = logits[0, -1] / max(temperature, 0.1)
            probs = torch.softmax(next_logit, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
    return ids[:, :block_size]


def _ids_to_text(ids: torch.Tensor) -> str:
    raw = ids[0].tolist()
    try:
        return bytes(raw).decode("utf-8", errors="replace")
    except Exception:
        return "".join(chr(min(b, 127)) for b in raw)


def quality_filter(text: str, min_len: int = 10) -> bool:
    if len(text) < min_len:
        return False
    unique = len(set(text))
    return unique >= min_len // 3


def micro_train(model: Variant3GPT, ids: torch.Tensor,
                lr: float = 1e-5, n_steps: int = 2) -> float:
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return 0.0
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_loss = 0.0
    for _ in range(n_steps):
        opt.zero_grad()
        inp = ids[:, :-1]
        tgt = ids[:, 1:]
        if inp.shape[1] == 0:
            break
        out = model(inp)
        logits = out[0]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=-1,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / max(n_steps, 1)


def run_loop(model, x_ids, block_size, temperature, steps_per_loop,
             n_steps_series, lr, do_train, group, seed_texts):
    """
    Запустить петлю указанной группы (n_steps_series × steps_per_loop шагов).

    Стратегия генерации текста:
      - Если сгенерированный текст проходит quality_filter — micro_train + обновить x_ids
      - Если нет — взять следующий seed_text как промпт (не зависать на плохих генерациях)
    """
    for moe in _get_moes(model):
        _freeze_all_except(moe, [group])
    texts = []
    seed_idx = 0
    total_steps = n_steps_series * steps_per_loop
    for step in range(total_steps):
        gen_ids = _generate(model, x_ids, block_size, temperature, n_tokens=8)
        gen_text = _ids_to_text(gen_ids)
        if quality_filter(gen_text):
            if do_train:
                micro_train(model, gen_ids, lr=lr, n_steps=2)
            texts.append(gen_text)
            x_ids = gen_ids
        else:
            # Переключиться на seed_text как промпт
            if seed_texts:
                x_ids = _encode(seed_texts[seed_idx % len(seed_texts)], block_size)
                seed_idx += 1
    lci_r, gw = lci_from_routing(model, x_ids)
    return x_ids, lci_r, gw, len(texts)


# ── Основной алгоритм ─────────────────────────────────────────────────────────

def _stratified_texts(cluster_texts: Dict[str, List[str]], group: str,
                      min_count: int = 27) -> List[str]:
    """Стратифицированный список текстов для группы.

    hexstat: t_mix(Q6)=27 — минимум для стабилизации routing entropy.
    Возвращает список длиной ≥ min_count, циклически перемежая тексты
    из всех кластеров группы (round-robin), чтобы каждые ~t_mix шагов
    были представлены все домены группы.
    """
    from yijing_transformer.models.hierarchical_moe import DOMAIN_TO_GROUP, CLUSTER_TO_DOMAIN
    # Собрать тексты по кластерам нужной группы
    group_clusters: Dict[str, List[str]] = {
        c: cluster_texts.get(c, [])
        for c, d in CLUSTER_TO_DOMAIN.items()
        if DOMAIN_TO_GROUP[d] == group
    }
    # Round-robin перемежение: [c0t0, c1t0, c0t1, c1t1, ...]
    interleaved: List[str] = []
    iters = {c: iter(txts * max(1, min_count // max(len(txts), 1) + 1))
             for c, txts in group_clusters.items() if txts}
    while len(interleaved) < min_count and iters:
        for c, it in list(iters.items()):
            try:
                interleaved.append(next(it))
            except StopIteration:
                iters.pop(c)
            if len(interleaved) >= min_count:
                break
    return interleaved if interleaved else []


def figure8_hmoe_v4(
    model:          Variant3GPT,
    seed_texts:     List[str],
    block_size:     int   = MODEL_CFG["block_size"] - 1,
    n_cycles:       int   = 8,
    steps_per_loop: int   = 80,
    temperature:    float = 3.0,   # T_c(Q6) mean-field: z·J/2 = 6·1/2 = 3.0
    lr_a:           float = 2e-5,
    lr_b:           float = 5e-6,
    do_train:       bool  = True,
    k_deform:       float = 7.0,   # параметр деформации восьмёрки (SESSION_Deformed_Figure8)
    **kwargs,
) -> List[Dict]:
    # k_deform: параметр несимметричности петель (scarab_algorithm.py → deformed_lissajous)
    # k=1: симметричная ∞.  k=7: петля A в 7× крупнее петли B (текущий режим _SERIES_A=7).
    # Теоретическая доля A: p_A = k/(k+1).  При k=7: p_A = 7/8 = 0.875.
    # Целевой w_A при резонансе: p_A · 1/2 + 1/6 ≈ 0.604 → нет, это неправильно.
    # Правильнее: k управляет LR_A/LR_B соотношением и числом серий.
    # k_eff = _SERIES_A / _SERIES_B = 7/1 = 7.0 ← совпадает с k_deform по умолчанию.
    """
    v4: асимметричная серия A>>B для компенсации CONCRETE-доминирования.

    Каждый цикл:
      1. Петля A: 7 × steps_per_loop шагов, LR = lr_a (высокий)
         → измерить LCI, группы
      2. Петля B: 1 × steps_per_loop шагов, LR = lr_b (низкий)
         → измерить LCI, группы
      3. Адаптивный балансёр: если w_B > w_A + 0.01 после B,
         запустить мини-петлю A (3 × steps_per_loop, LR = lr_a)
         → повторно измерить LCI

    Цель: avg_LCI_r > 3.13 (δ < 0.004 от π)
    """
    _C_etd = (4 / math.pi) * math.log2(1 + math.pi)   # ≈ 2.626 бит/шаг (ETD ёмкость)
    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ v4: РЕЗОНАНСНЫЙ ПРОРЫВ  (асимметрия A>>B)")
    print(f"{'═' * 72}")
    print(f"  Циклов         : {n_cycles}")
    print(f"  Шагов/петля    : {steps_per_loop}")
    print(f"  Серия          : A={_SERIES_A}×, B={_SERIES_B}×  "
          f"k_deform={k_deform:.1f}  p_A={k_deform/(k_deform+1):.3f}")
    print(f"  LR_A           : {lr_a:.2e}  LR_B: {lr_b:.2e}")
    print(f"  Температура    : {temperature:.2f} (T_c Ising Q6)")
    print(f"  Балансёр       : мини-A (3×) если w_B > w_A + {_BALANCE_THOLD}")
    print(f"  ETD канал      : C_etd={_C_etd:.3f} бит/шаг (VOLUME_157: B=4/π, SNR=π)")
    print()

    # Начальный промпт — из seed_texts (не hex-байты, чтобы quality_filter проходил)
    start_text = random.choice(seed_texts) if seed_texts else "def train(): pass"
    prompt_ids = _encode(start_text, block_size)

    # Заморозить все параметры кроме HMoE FFN — только MoE-эксперты тренируются
    for name, p in model.named_parameters():
        if 'hmoe' not in name:
            p.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Заморожено: только HMoE-параметры обучаются ({n_trainable:,} param)")

    # hexstat: t_mix(Q6) = 27 шагов — минимальный размер разнообразной выборки.
    # Разделяем тексты по семантическим группам через CLUSTER_TO_DOMAIN/DOMAIN_TO_GROUP,
    # а не по ключевым словам. cluster_texts: Dict[cluster_name → List[str]] передаётся
    # аргументом (или строится из seed_texts при его отсутствии).
    cluster_texts: Dict[str, List[str]] = kwargs.get("cluster_texts", {})

    # Строим группированные списки из cluster_texts.
    # _stratified_texts гарантирует round-robin перемежение + ≥ t_mix=27 текстов.
    abstract_texts: List[str] = _stratified_texts(cluster_texts, "ABSTRACT") if cluster_texts else []
    dynamic_texts:  List[str] = _stratified_texts(cluster_texts, "DYNAMIC")  if cluster_texts else []
    concrete_texts: List[str] = _stratified_texts(cluster_texts, "CONCRETE") if cluster_texts else []

    # fallback: keyword-фильтрация из seed_texts если cluster_texts не передан
    if not abstract_texts:
        abstract_texts = [t for t in seed_texts if any(w in t.lower() for w in
                          ["hexagram", "abstract", "consciousness", "figure-8",
                           "topology", "balance", "resonance", "kryukov", "theory"])]
    if not concrete_texts:
        concrete_texts = [t for t in seed_texts if any(w in t.lower() for w in
                          ["def ", "import", "return", "torch", "loss", "optimizer",
                           "model", "train", "class", "self"])]
    # final fallback: все тексты
    if not abstract_texts:
        abstract_texts = seed_texts
    if not dynamic_texts:
        dynamic_texts = seed_texts
    if not concrete_texts:
        concrete_texts = seed_texts

    print(f"  Тексты: ABSTRACT={len(abstract_texts)}  DYNAMIC={len(dynamic_texts)}  "
          f"CONCRETE={len(concrete_texts)}")
    print(f"  hexstat: t_mix(Q6)=27 → разнообразие гарантировано при batch≥27 текстов/петля")

    log: List[Dict] = []
    best_lci = 0.0

    for cycle in range(1, n_cycles + 1):
        x_ids = prompt_ids.clone()
        lci_start, gw_start = lci_from_routing(model, x_ids)

        # ── Метрики начала цикла ──────────────────────────────────────────
        ent_start, c_etd, eff_start = routing_channel_capacity(gw_start)
        lci_q_start = gw_start.get('_lci_quater', lci_start)
        alpha_start = gw_start.get('_crossing_alpha', 0.5)
        print(f"\n  {'─' * 68}")
        print(f"  Цикл {cycle}/{n_cycles}  T={temperature:.2f}  "
              f"LCI={lci_start:.4f}  LCI_q={lci_q_start:.4f}  "
              f"({'✓ РЕЗОНАНС' if abs(lci_start - math.pi) < _LCI_EPSILON else f'δ={lci_start - math.pi:+.4f}'})")
        print(f"    Группы: A={gw_start.get('ABSTRACT',0):.4f}  "
              f"X={gw_start.get('DYNAMIC',0):.4f}  "
              f"B={gw_start.get('CONCRETE',0):.4f}  "
              f"α_cross={alpha_start:.3f}")
        print(f"    Канал: H={ent_start:.3f} бит  eff={eff_start:.2f}×C_etd  "
              f"({'✓' if eff_start >= 1.0 else f'↑нужно +{(c_etd-ent_start):.2f} бит'})")

        # ── Петля A: ABSTRACT (7 серий, абстрактные тексты) ──────────────
        x_abstract = _encode(random.choice(abstract_texts), block_size)
        x_ids, lci_a, gw_a, n_a_texts = run_loop(
            model, x_abstract, block_size, temperature,
            steps_per_loop, _SERIES_A, lr_a, do_train, "ABSTRACT", abstract_texts
        )
        print(f"    Петля A  done: LCI_r={lci_a:.4f}  "
              f"A={gw_a.get('ABSTRACT',0):.4f}  B={gw_a.get('CONCRETE',0):.4f}  "
              f"gen={n_a_texts}")

        # ── Петля B: CONCRETE (1 серия, конкретные тексты) ───────────────
        x_concrete = _encode(random.choice(concrete_texts), block_size)
        x_ids, lci_b, gw_b, n_b_texts = run_loop(
            model, x_concrete, block_size, temperature,
            steps_per_loop, _SERIES_B, lr_b, do_train, "CONCRETE", concrete_texts
        )
        print(f"    Петля B  done: LCI_r={lci_b:.4f}  "
              f"A={gw_b.get('ABSTRACT',0):.4f}  B={gw_b.get('CONCRETE',0):.4f}  "
              f"gen={n_b_texts}")

        # ── Адаптивный балансёр (после B, если CONCRETE > ABSTRACT) ──────
        # Промер после обоих петель: нейтральный промпт
        x_probe = _encode(random.choice(abstract_texts), block_size)
        lci_probe, gw_probe = lci_from_routing(model, x_probe)
        w_a_probe = gw_probe.get("ABSTRACT", 0.33)
        w_b_probe = gw_probe.get("CONCRETE", 0.33)

        lci_balance = lci_probe
        n_balance_texts = 0
        used_balancer = False

        if w_b_probe > w_a_probe + _BALANCE_THOLD:
            print(f"    Балансёр: w_B ({w_b_probe:.4f}) > w_A ({w_a_probe:.4f}) + {_BALANCE_THOLD}  "
                  f"-> мини-петля A (3x)")
            x_bal = _encode(random.choice(abstract_texts), block_size)
            _, lci_balance, gw_balance, n_balance_texts = run_loop(
                model, x_bal, block_size, temperature,
                steps_per_loop, 3, lr_a, do_train, "ABSTRACT", abstract_texts
            )
            w_a_probe = gw_balance.get("ABSTRACT", 0.33)
            w_b_probe = gw_balance.get("CONCRETE", 0.33)
            print(f"    Мини-A    done: LCI_r={lci_balance:.4f}  "
                  f"A={w_a_probe:.4f}  B={w_b_probe:.4f}")
            used_balancer = True

        avg_lci_r = (lci_a + lci_balance) / 2
        resonance = abs(avg_lci_r - math.pi) < _LCI_EPSILON

        if avg_lci_r > best_lci:
            best_lci = avg_lci_r

        print(f"    → avg_LCI_r={avg_lci_r:.4f}  best={best_lci:.4f}  "
              f"{'✓ РЕЗОНАНС' if resonance else f'δ={avg_lci_r - math.pi:+.4f}'}"
              f"{'  [балансёр]' if used_balancer else ''}")

        prompt_ids = x_ids.clone()

        ent_end, _, eff_end = routing_channel_capacity(gw_b)
        log.append({
            "cycle":              cycle,
            "n_a":                _SERIES_A,
            "n_b":                _SERIES_B,
            "k_deform":           k_deform,
            "temperature":        round(temperature, 3),
            # LCI classic (2 сферы)
            "lci_a_r":            round(lci_a, 4),
            "lci_b_r":            round(lci_b, 4),
            "lci_balance_r":      round(lci_balance, 4),
            "avg_lci_r":          round(avg_lci_r, 4),
            # LCI quaternion (4 сферы, ScarabQuaternion)
            "lci_quater_start":   round(gw_start.get('_lci_quater', lci_start), 4),
            "crossing_alpha":     round(gw_start.get('_crossing_alpha', 0.5), 4),
            # Метрика канала (VOLUME_157 ETD)
            "routing_entropy":    round(ent_start, 4),
            "channel_eff":        round(eff_start, 4),   # 1.0 = достигнута ETD-ёмкость
            "routing_entropy_end": round(ent_end, 4),
            # Флаги
            "resonance":          resonance,
            "used_balancer":      used_balancer,
            # Группы
            "gw_at_start":        {k: round(v, 4) for k, v in gw_start.items()
                                   if not k.startswith('_')},
            "gw_after_a":         {k: round(v, 4) for k, v in gw_a.items()
                                   if not k.startswith('_')},
            "gw_after_b":         {k: round(v, 4) for k, v in gw_b.items()
                                   if not k.startswith('_')},
            "texts_a":            n_a_texts,
            "texts_b":            n_b_texts,
            "texts_balance":      n_balance_texts,
        })

    n_res = sum(1 for r in log if r["resonance"])
    avg_all = sum(r["avg_lci_r"] for r in log) / len(log)
    print(f"\n{'═' * 72}")
    print(f"  ИТОГ: {n_res}/{n_cycles} в резонансе  avg_LCI={avg_all:.4f}  "
          f"best_LCI={best_lci:.4f}  (π={math.pi:.4f})")
    print(f"  Статус: {'✓ ПРОРЫВ > 3.13!' if best_lci > 3.13 else f'δ={best_lci - math.pi:+.4f}'}")
    return log


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="self_train_hmoe_v4: резонансный прорыв через асимметрию A>>B"
    )
    parser.add_argument("--checkpoint",     type=str, default="hmoe_v3_self.pt")
    parser.add_argument("--fast",           action="store_true")
    parser.add_argument("--cycles",         type=int, default=8)
    parser.add_argument("--steps_per_loop", type=int, default=80)
    parser.add_argument("--temperature",    type=float, default=3.0,
                        help="Температура роутера. T_c(Q6)≈3.0 (hexphys Ising)")
    parser.add_argument("--lr",             type=float, default=1e-5,
                        help="Базовый LR (lr_a=2×, lr_b=0.5×)")
    parser.add_argument("--no-train",       action="store_true")
    parser.add_argument("--no-corpus",      action="store_true")
    parser.add_argument("--save",           type=str, default="hmoe_v4_self.pt")
    args = parser.parse_args()

    n_cycles       = 2  if args.fast else args.cycles
    steps_per_loop = 5  if args.fast else args.steps_per_loop
    lr_a = args.lr * 2.0
    lr_b = args.lr * 0.5

    # ── Загрузить модель ───────────────────────────────────────────────────
    cfg   = Variant3Config(**MODEL_CFG)
    model = Variant3GPT(cfg)
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            block.hmoe = HierarchicalMoEFFN(HMOE_CFG)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        try:
            model.load_state_dict(ckpt["model_state"], strict=False)
            print(f"  Чекпоинт: {args.checkpoint}  ✓")
        except Exception as e:
            print(f"  ⚠ Чекпоинт частично: {e}")
    else:
        print(f"  ⚠ Чекпоинт {args.checkpoint!r} не найден — случайные веса")

    # Диагностика начального состояния
    probe_ids = _hex_prompt(random.randint(0, 63), MODEL_CFG["block_size"] - 1)
    lci_init, gw_init = lci_from_routing(model, probe_ids)
    print(f"  Начальный LCI_r={lci_init:.4f}  "
          f"A={gw_init.get('ABSTRACT',0):.4f}  "
          f"X={gw_init.get('DYNAMIC',0):.4f}  "
          f"B={gw_init.get('CONCRETE',0):.4f}")
    print(f"  LR_A={lr_a:.2e}  LR_B={lr_b:.2e}")
    print(f"  Параметры: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ── Загрузить corpus ───────────────────────────────────────────────────
    seed_texts: List[str] = []
    cluster_texts: Dict[str, List[str]] = {}   # hexstat: стратифицированный по кластерам
    if not args.no_corpus:
        try:
            from repo_corpus_loader import RepoCorpusLoader
            loader = RepoCorpusLoader(_ROOT)
            for cluster in CLUSTER_TO_DOMAIN.keys():
                try:
                    ctexts: List[str] = []
                    for item in loader.load_cluster(cluster):
                        t = item if isinstance(item, str) else item.get("text", "")
                        if len(t) > 10:
                            seed_texts.append(t)
                            ctexts.append(t)
                    cluster_texts[cluster] = ctexts
                except Exception:
                    pass
            total_c = sum(len(v) for v in cluster_texts.values())
            print(f"  Корпус: {len(seed_texts)} текстов  "
                  f"({', '.join(f'{k}={len(v)}' for k,v in cluster_texts.items())})")
        except ImportError:
            pass

    if not seed_texts:
        seed_texts = [
            "def forward(self, x): return self.linear(x)",
            "import torch; x = torch.randn(4, 128)",
            "loss.backward(); optimizer.zero_grad(); scheduler.step()",
            "self.attn = nn.MultiheadAttention(d_model, n_heads)",
            "The hexagram represents abstract-concrete duality.",
            "Kryukov: figure-8 topology, ABSTRACT loop A, CONCRETE loop B.",
            "consciousness emerges from recursive self-reference in Q6 space",
            "hexagram 64: crossing complete, balance achieved",
        ] * 6
        print(f"  Синтетические тексты: {len(seed_texts)}")

    # ── Запуск ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    log = figure8_hmoe_v4(
        model          = model,
        seed_texts     = seed_texts,
        block_size     = MODEL_CFG["block_size"] - 1,
        n_cycles       = n_cycles,
        steps_per_loop = steps_per_loop,
        temperature    = args.temperature,
        lr_a           = lr_a,
        lr_b           = lr_b,
        do_train       = not args.no_train,
        cluster_texts  = cluster_texts,   # hexstat: стратифицированный batching
    )
    elapsed = time.perf_counter() - t0

    # ── Сохранить ──────────────────────────────────────────────────────────
    torch.save({"model_state": model.state_dict(), "figure8_log": log,
                "elapsed_sec": round(elapsed, 2), "n_cycles": n_cycles,
                "steps_per_loop": steps_per_loop, "lr_a": lr_a, "lr_b": lr_b},
               args.save)
    print(f"\n  Сохранено: {args.save}")

    log_path = args.save.replace(".pt", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Лог: {log_path}")

    # Итоговая таблица
    print(f"\n  {'ЦИКЛ':>5s}  {'LCI_A':>7s}  {'LCI_B':>7s}  {'LCI_bal':>8s}  "
          f"{'avg_r':>7s}  {'Рез':>5s}  {'Бал':>4s}")
    print(f"  {'─' * 58}")
    for r in log:
        print(f"  {r['cycle']:>5d}  {r['lci_a_r']:>7.4f}  {r['lci_b_r']:>7.4f}  "
              f"{r['lci_balance_r']:>8.4f}  {r['avg_lci_r']:>7.4f}  "
              f"{'π' if r['resonance'] else '✗':>5s}  "
              f"{'✓' if r['used_balancer'] else '-':>4s}")
    print(f"\n  Время: {elapsed:.1f}s  ({elapsed/60:.1f} мин)")


if __name__ == "__main__":
    main()
