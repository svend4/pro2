#!/usr/bin/env python3
"""
eval_hmoe.py — Диагностика HierarchicalMoE с рекомендациями.

Анализирует чекпоинт и выдаёт конкретные рекомендации по следующим шагам:
  • Баланс восьмёрки (routing LCI, ABSTRACT/CONCRETE/DYNAMIC веса)
  • Здоровье маршрутизации (entropy, CV экспертов, anti-circle статус)
  • Точка пересечения (crossing alpha, lb_loss по группам)
  • Что делать дальше (следующая команда, какая фаза, какой параметр)

Usage:
  python eval_hmoe.py                            # анализ hmoe_curriculum.pt
  python eval_hmoe.py --checkpoint hmoe_self_trained.pt
  python eval_hmoe.py --log hmoe_self_trained_log.json  # + анализ LCI-лога
  python eval_hmoe.py --fast                     # меньше текстов для оценки
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    TRAINING_STAGES,
    get_stage_info,
)

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
    dropout            = 0.0,   # eval mode
    use_domain_routing = False,
    use_hierarchical_moe = True,
)

HMOE_CFG = HMoEConfig(d_model=128, use_multiscale=True, use_hex_tier=False)

_LCI_TARGET  = math.pi
_LCI_EPS     = 0.5
_ENT_TARGET  = math.log(3)    # идеальная энтропия для 3 групп = ln(3) ≈ 1.099
_CV_WARN     = 0.5            # CV > 0.5 → неравномерная нагрузка экспертов
_ALPHA_LOW   = 0.3            # crossing_alpha < 0.3 → слабое обобщение
_ALPHA_HIGH  = 0.7            # crossing_alpha > 0.7 → слабая специализация

_GROUPS = ["ABSTRACT", "DYNAMIC", "CONCRETE"]
_GROUP_IDEAL = {"ABSTRACT": 0.333, "DYNAMIC": 0.334, "CONCRETE": 0.333}


# ── Метрики ───────────────────────────────────────────────────────────────────

@dataclass
class DiagResult:
    # Маршрутизация
    gw_mean:         Dict[str, float]   # средние group_weights
    routing_entropy: float              # энтропия распределения групп
    routing_lci:     float              # (1 - |w_A - w_B|) × π
    crossing_alpha:  float              # sigmoid(log_alpha) в BidirBridge
    # Нагрузка экспертов
    lb_by_group:     Dict[str, float]   # lb_loss по группам
    expert_cv:       Dict[str, float]   # CV нагрузки внутри каждой группы
    ema_load:        Dict[str, float]   # EMA-нагрузка на кластеры
    # Качество
    ppl:             float
    avg_loss:        float
    # Фигура-8 (из лога, если есть)
    lci_trend:       Optional[List[float]] = None   # avg_lci_emb по циклам
    n_resonance:     int = 0


def _encode(text: str, block_size: int = 63) -> torch.Tensor:
    ids = [min(b, 255) for b in text.encode("utf-8")][:block_size]
    return torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)


def _collect_metrics(model: Variant3GPT, texts: List[str]) -> DiagResult:
    """Прогон через тексты, сбор всех диагностических метрик."""
    model.eval()

    all_gw: List[torch.Tensor]     = []
    all_lb_a: List[float]          = []
    all_lb_b: List[float]          = []
    all_lb_d: List[float]          = []
    all_alpha: List[float]         = []
    all_losses: List[float]        = []
    cluster_loads: Dict[str, List[float]] = {c: [] for c in CLUSTER_TO_DOMAIN}

    with torch.no_grad():
        for text in texts:
            ids = _encode(text)
            if ids.shape[1] < 2:
                continue
            inp, tgt = ids[:, :-1], ids[:, 1:]
            logits, loss, _ = model(inp, targets=tgt)

            if loss is not None and not torch.isnan(loss):
                all_losses.append(loss.item())

            # Собрать info из всех блоков
            for block in model.blocks:
                info = getattr(block, '_last_moe_info', None)
                if info is None:
                    continue

                gw = info.get('group_weights')
                if gw is not None:
                    all_gw.append(gw.mean(dim=(0, 1)).cpu())

                for key, lst in [('lb_ABSTRACT', all_lb_a),
                                  ('lb_CONCRETE', all_lb_b),
                                  ('lb_DYNAMIC',  all_lb_d)]:
                    v = info.get(key)
                    if v is not None:
                        lst.append(v.item())

                alpha = info.get('crossing_alpha')
                if alpha is not None:
                    all_alpha.append(alpha.item())

                # EMA нагрузка на кластеры через GroupRouter
                if hasattr(block, 'hmoe'):
                    moe = block.hmoe
                    for g, gr in moe.group_routers.items():
                        ema = gr._ema_load.cpu()
                        for i, cluster in enumerate(moe.group_to_clusters[g]):
                            if i < len(ema):
                                cluster_loads[cluster].append(ema[i].item())

    # Усреднить group_weights
    if all_gw:
        avg_gw = torch.stack(all_gw).mean(0)
        gw_mean = {g: avg_gw[i].item() for i, g in enumerate(_GROUPS)}
    else:
        gw_mean = {g: 1/3 for g in _GROUPS}

    # Энтропия маршрутизации
    gw_t = torch.tensor([gw_mean[g] for g in _GROUPS])
    entropy = -(gw_t * torch.log(gw_t + 1e-8)).sum().item()

    # Routing LCI
    w_a = gw_mean.get("ABSTRACT", 0.33)
    w_b = gw_mean.get("CONCRETE", 0.33)
    imbalance = abs(w_a - w_b) / (w_a + w_b + 1e-8)
    routing_lci = (1.0 - imbalance) * math.pi

    # Load balance по группам
    lb_by_group = {
        "ABSTRACT": sum(all_lb_a) / len(all_lb_a) if all_lb_a else float("nan"),
        "CONCRETE": sum(all_lb_b) / len(all_lb_b) if all_lb_b else float("nan"),
        "DYNAMIC":  sum(all_lb_d) / len(all_lb_d) if all_lb_d else float("nan"),
    }

    # Crossing alpha
    crossing_alpha = sum(all_alpha) / len(all_alpha) if all_alpha else float("nan")

    # CV нагрузки по группам (через EMA)
    expert_cv: Dict[str, float] = {}
    for g, clusters in DOMAIN_GROUPS.items():
        loads = []
        for c in clusters:
            ls = cluster_loads.get(c, [])
            if ls:
                loads.append(sum(ls) / len(ls))
        if len(loads) >= 2:
            mean_l = sum(loads) / len(loads)
            std_l  = (sum((x - mean_l) ** 2 for x in loads) / len(loads)) ** 0.5
            expert_cv[g] = std_l / (mean_l + 1e-8)
        else:
            expert_cv[g] = 0.0

    # EMA нагрузка (средняя по кластеру)
    ema_load = {
        c: sum(vs) / len(vs) if vs else float("nan")
        for c, vs in cluster_loads.items()
    }

    # PPL
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else float("nan")
    ppl = math.exp(min(avg_loss, 10)) if not math.isnan(avg_loss) else float("nan")

    return DiagResult(
        gw_mean         = gw_mean,
        routing_entropy = entropy,
        routing_lci     = routing_lci,
        crossing_alpha  = crossing_alpha,
        lb_by_group     = lb_by_group,
        expert_cv       = expert_cv,
        ema_load        = ema_load,
        ppl             = ppl,
        avg_loss        = avg_loss,
    )


def _analyze_lci_log(log: List[Dict]) -> Tuple[List[float], int]:
    """Анализ JSON-лога self_train_hmoe.py."""
    trend = [r.get("avg_lci_emb", math.nan) for r in log]
    n_res = sum(1 for r in log if r.get("resonance", False))
    return trend, n_res


# ── ASCII визуализация ────────────────────────────────────────────────────────

def _bar(value: float, total: float = 1.0, width: int = 20) -> str:
    filled = int(round(value / max(total, 1e-8) * width))
    return "█" * filled + "░" * (width - filled)


def _trend_spark(values: List[float]) -> str:
    """Простой sparkline из ASCII-символов."""
    if not values:
        return ""
    levels = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn or 1.0
    return "".join(levels[int((v - mn) / rng * (len(levels) - 1))] for v in values)


# ── Рекомендации ─────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    priority: int          # 1=критично, 2=важно, 3=информация
    category: str
    finding:  str
    action:   str
    command:  Optional[str] = None


def _make_recommendations(d: DiagResult, ckpt_path: str) -> List[Recommendation]:
    recs: List[Recommendation] = []

    # ── 1. Баланс восьмёрки ───────────────────────────────────────────────
    w_a = d.gw_mean.get("ABSTRACT", 0)
    w_b = d.gw_mean.get("CONCRETE", 0)
    w_d = d.gw_mean.get("DYNAMIC",  0)

    if abs(w_a - w_b) > 0.15:
        dominant = "ABSTRACT" if w_a > w_b else "CONCRETE"
        weak     = "CONCRETE" if w_a > w_b else "ABSTRACT"
        clust    = ["Models", "Theory"] if weak == "ABSTRACT" else ["Scripts", "Data"]
        phase    = 3 if weak == "ABSTRACT" else 1
        recs.append(Recommendation(
            priority = 1,
            category = "Баланс восьмёрки",
            finding  = f"{dominant} доминирует ({d.gw_mean[dominant]:.3f} vs "
                       f"{d.gw_mean[weak]:.3f}) — петли несбалансированы",
            action   = f"Дообучить {weak} micro_experts на кластерах {clust}",
            command  = f"python train_hmoe_curriculum.py --phase {phase} "
                       f"--steps_per_phase 300 --resume {ckpt_path}",
        ))

    if w_d < 0.15:
        recs.append(Recommendation(
            priority = 2,
            category = "Точка пересечения",
            finding  = f"DYNAMIC слабо задействован (вес={w_d:.3f} < 0.15) "
                       "— точка X не работает как мост",
            action   = "Дообучить crossing (BidirBridgeExpert) + DYNAMIC experts",
            command  = f"python train_hmoe_curriculum.py --phase 2 "
                       f"--steps_per_phase 200 --resume {ckpt_path}",
        ))

    # ── 2. LCI резонанс ───────────────────────────────────────────────────
    lci = d.routing_lci
    delta = lci - _LCI_TARGET
    if abs(delta) > _LCI_EPS:
        if delta < 0:
            recs.append(Recommendation(
                priority = 2,
                category = "LCI резонанс",
                finding  = f"routing_LCI={lci:.3f} < π ({delta:+.3f}) — "
                           "петли слишком схожи, нет расхождения",
                action   = "Запустить само-обучение с высокой температурой (zoom-out)",
                command  = "python self_train_hmoe.py --cycles 4 --temperature 1.6 "
                           f"--checkpoint {ckpt_path}",
            ))
        else:
            recs.append(Recommendation(
                priority = 2,
                category = "LCI резонанс",
                finding  = f"routing_LCI={lci:.3f} > π ({delta:+.3f}) — "
                           "петли расходятся слишком сильно",
                action   = "Запустить само-обучение с низкой температурой (zoom-in)",
                command  = "python self_train_hmoe.py --cycles 4 --temperature 0.8 "
                           f"--checkpoint {ckpt_path}",
            ))
    else:
        recs.append(Recommendation(
            priority = 3,
            category = "LCI резонанс",
            finding  = f"routing_LCI={lci:.3f} ≈ π — восьмёрка в резонансе ✓",
            action   = "Можно переходить к финальной настройке (Phase 5)",
            command  = f"python train_hmoe_curriculum.py --phase 5 "
                       f"--steps_per_phase 500 --resume {ckpt_path}",
        ))

    # ── 3. Entropy ────────────────────────────────────────────────────────
    ent_delta = d.routing_entropy - _ENT_TARGET
    if d.routing_entropy < _ENT_TARGET * 0.7:
        recs.append(Recommendation(
            priority = 1,
            category = "Routing Entropy",
            finding  = f"Entropy={d.routing_entropy:.3f} низкая (цель≥{_ENT_TARGET*0.7:.3f}) "
                       "— GlobalRouter схлопнулся в одну группу",
            action   = "Переобучить GlobalRouter с усиленным lb_loss",
            command  = f"python train_hmoe_curriculum.py --phase 4 "
                       f"--steps_per_phase 300 --resume {ckpt_path}",
        ))

    # ── 4. Crossing alpha ─────────────────────────────────────────────────
    alpha = d.crossing_alpha
    if not math.isnan(alpha):
        if alpha < _ALPHA_LOW:
            recs.append(Recommendation(
                priority = 2,
                category = "BidirBridge α",
                finding  = f"crossing_alpha={alpha:.3f} < {_ALPHA_LOW} — "
                           "обобщение (fwd A→B) подавлено, перекос в специализацию",
                action   = "Добавить тексты с высоким α-уровнем (Theory, Models)",
                command  = f"python train_hmoe_curriculum.py --phase 3 "
                           f"--steps_per_phase 200 --resume {ckpt_path}",
            ))
        elif alpha > _ALPHA_HIGH:
            recs.append(Recommendation(
                priority = 2,
                category = "BidirBridge α",
                finding  = f"crossing_alpha={alpha:.3f} > {_ALPHA_HIGH} — "
                           "специализация (bwd B→A) подавлена, перекос в обобщение",
                action   = "Добавить тексты с низким α-уровнем (Scripts, Data)",
                command  = f"python train_hmoe_curriculum.py --phase 1 "
                           f"--steps_per_phase 200 --resume {ckpt_path}",
            ))

    # ── 5. Expert CV ──────────────────────────────────────────────────────
    for group, cv in d.expert_cv.items():
        if cv > _CV_WARN:
            clusters = list(DOMAIN_GROUPS.get(group, []))
            recs.append(Recommendation(
                priority = 2,
                category = f"Expert CV ({group})",
                finding  = f"CV={cv:.3f} > {_CV_WARN} в группе {group} — "
                           "один эксперт доминирует (anti-circle не справляется)",
                action   = f"Снизить streak_limit или увеличить anticircle_weight "
                           f"в GroupRouter для {group}",
                command  = None,
            ))

    # ── 6. PPL ────────────────────────────────────────────────────────────
    if not math.isnan(d.ppl):
        if d.ppl > 50:
            recs.append(Recommendation(
                priority = 1,
                category = "PPL",
                finding  = f"PPL={d.ppl:.1f} высокий — модель слабо обучена",
                action   = "Запустить полный curriculum с большим числом шагов",
                command  = "python train_hmoe_curriculum.py --steps_per_phase 1000",
            ))
        elif d.ppl > 20:
            recs.append(Recommendation(
                priority = 2,
                category = "PPL",
                finding  = f"PPL={d.ppl:.1f} — есть куда улучшать",
                action   = "Запустить Phase 5 (joint fine-tune)",
                command  = f"python train_hmoe_curriculum.py --phase 5 "
                           f"--steps_per_phase 500 --resume {ckpt_path}",
            ))
        else:
            recs.append(Recommendation(
                priority = 3,
                category = "PPL",
                finding  = f"PPL={d.ppl:.1f} — хороший результат ✓",
                action   = "Переходить к само-обучению / inference",
                command  = f"python self_train_hmoe.py --cycles 6 "
                           f"--checkpoint {ckpt_path}",
            ))

    # Сортировать: критичные первыми
    recs.sort(key=lambda r: r.priority)
    return recs


# ── Печать ────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def _print_report(d: DiagResult, recs: List[Recommendation],
                  ckpt_path: str, log: Optional[List[Dict]] = None) -> None:

    print(f"\n{'═' * 72}")
    print(f"  ДИАГНОСТИКА HierarchicalMoE")
    print(f"  Чекпоинт: {ckpt_path}")
    print(f"{'═' * 72}")

    # ── Раздел 1: Баланс восьмёрки ────────────────────────────────────────
    _print_section("1. БАЛАНС ВОСЬМЁРКИ (figure-8 topology)")

    lci = d.routing_lci
    lci_status = ("✓ РЕЗОНАНС" if abs(lci - _LCI_TARGET) < _LCI_EPS
                  else f"δ={lci - _LCI_TARGET:+.3f}")
    print(f"  routing_LCI = {lci:.3f}  (цель=π={_LCI_TARGET:.3f})  {lci_status}")
    print()

    for g in _GROUPS:
        w = d.gw_mean.get(g, 0)
        role = {"ABSTRACT": "Петля A ↑ обобщение  ",
                "DYNAMIC":  "Точка X ∞ пересечение",
                "CONCRETE": "Петля B ↓ специализ.  "}[g]
        ideal = _GROUP_IDEAL[g]
        delta = w - ideal
        flag = "  ✓" if abs(delta) < 0.1 else f"  {'↑' if delta > 0 else '↓'}{abs(delta):.2f}"
        print(f"  {g:10s} [{role}]  {_bar(w)} {w:.3f}{flag}")

    # ── Раздел 2: Crossing alpha ──────────────────────────────────────────
    _print_section("2. ТОЧКА ПЕРЕСЕЧЕНИЯ (BidirBridgeExpert)")

    alpha = d.crossing_alpha
    if not math.isnan(alpha):
        fwd_share = alpha
        bwd_share = 1.0 - alpha
        alpha_status = ("← баланс ✓" if _ALPHA_LOW <= alpha <= _ALPHA_HIGH
                        else ("← перекос в обобщение" if alpha > _ALPHA_HIGH
                              else "← перекос в специализацию"))
        print(f"  crossing_alpha (sigmoid) = {alpha:.4f}  {alpha_status}")
        print(f"  fwd A→B (специализация→обобщение) : {_bar(fwd_share)} {fwd_share:.3f}")
        print(f"  bwd B→A (обобщение→специализация) : {_bar(bwd_share)} {bwd_share:.3f}")

    # ── Раздел 3: Энтропия и нагрузка ────────────────────────────────────
    _print_section("3. МАРШРУТИЗАЦИЯ И НАГРУЗКА ЭКСПЕРТОВ")

    ent_status = ("✓" if d.routing_entropy >= _ENT_TARGET * 0.85 else "⚠ низкая")
    print(f"  Routing entropy = {d.routing_entropy:.4f}  "
          f"(цель≈{_ENT_TARGET:.3f} = ln(3))  {ent_status}")
    print()

    for g in _GROUPS:
        lb  = d.lb_by_group.get(g, float("nan"))
        cv  = d.expert_cv.get(g, 0.0)
        cv_flag = "✓" if cv <= _CV_WARN else f"⚠ CV={cv:.2f}"
        # Кластеры для группы (cluster names, not domain names)
        clusters = [c for c, d_ in CLUSTER_TO_DOMAIN.items()
                    if DOMAIN_TO_GROUP[d_] == g]
        print(f"  {g:10s}  lb_loss={lb:+.5f}  CV={cv:.3f} {cv_flag}")
        # EMA нагрузка по кластерам
        for c in clusters:
            v = d.ema_load.get(c, float("nan"))
            bar = _bar(v, total=1.0, width=12) if not math.isnan(v) else "?"
            clust_flag = ("✓" if not math.isnan(v) and 0.2 < v < 0.8
                          else ("↑ перегруз" if not math.isnan(v) and v >= 0.8
                                else "↓ простой"))
            print(f"    {c:12s} [{bar}] {v:.3f}  {clust_flag}")

    # ── Раздел 4: PPL ─────────────────────────────────────────────────────
    _print_section("4. КАЧЕСТВО МОДЕЛИ")

    ppl_bar = _bar(min(d.ppl / 100, 1.0) if not math.isnan(d.ppl) else 0, width=20)
    ppl_note = ("отлично" if d.ppl < 15 else
                "хорошо"  if d.ppl < 30 else
                "средне"  if d.ppl < 60 else "слабо")
    if not math.isnan(d.ppl):
        print(f"  PPL  = {d.ppl:7.2f}  [{ppl_bar}]  {ppl_note}")
        print(f"  loss = {d.avg_loss:.4f}")

    # ── Раздел 5: LCI-лог (если есть) ────────────────────────────────────
    if log:
        _print_section("5. ИСТОРИЯ САМО-ОБУЧЕНИЯ (figure-8 log)")
        trend, n_res = _analyze_lci_log(log)
        spark = _trend_spark(trend)
        print(f"  Циклов: {len(log)}  Резонансов: {n_res}/{len(log)}")
        print(f"  avg_LCI тренд:  {spark}  [{' → '.join(f'{v:.2f}' for v in trend)}]")
        if len(trend) >= 2:
            direction = "↑ растёт" if trend[-1] > trend[0] else "↓ падает"
            target_note = ("→ приближается к π ✓" if trend[-1] > trend[0] and trend[-1] < _LCI_TARGET
                          else "→ превышает π" if trend[-1] > _LCI_TARGET
                          else "→ ещё не достигло π")
            print(f"  Динамика: {direction}  {target_note}")

        # Последний цикл — веса групп
        last = log[-1]
        print(f"\n  Последний цикл: серия=[{last['n_a']},{last['n_b']}]  "
              f"T={last['temperature']}")
        for phase, key in [("до петли A",  "gw_at_start"),
                           ("после петли A", "gw_after_a"),
                           ("после петли B", "gw_after_b")]:
            gw = last.get(key, {})
            vals = "  ".join(f"{g}={gw.get(g, 0):.3f}" for g in _GROUPS)
            print(f"    {phase:16s}: {vals}")

    # ── Раздел 6: Рекомендации ────────────────────────────────────────────
    _print_section("6. РЕКОМЕНДАЦИИ")

    prio_emoji = {1: "🔴", 2: "🟡", 3: "🟢"}
    prio_label = {1: "КРИТИЧНО", 2: "ВАЖНО   ", 3: "ИНФО    "}

    for i, rec in enumerate(recs, 1):
        em = prio_emoji.get(rec.priority, "•")
        lb = prio_label.get(rec.priority, "")
        print(f"\n  [{em} {lb}] {rec.category}")
        print(f"  Найдено  : {rec.finding}")
        print(f"  Действие : {rec.action}")
        if rec.command:
            print(f"  Команда  : {rec.command}")

    # ── Итог: следующий шаг ───────────────────────────────────────────────
    _print_section("СЛЕДУЮЩИЙ ШАГ")
    critical = [r for r in recs if r.priority == 1]
    important = [r for r in recs if r.priority == 2]
    first = (critical or important or recs)
    if first:
        top = first[0]
        print(f"  {top.action}")
        if top.command:
            print(f"\n  $ {top.command}")
    print()


# ── Главная функция ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Диагностика HierarchicalMoE")
    parser.add_argument("--checkpoint", type=str, default="hmoe_curriculum.pt")
    parser.add_argument("--log",        type=str, default=None,
                        help="JSON-лог от self_train_hmoe.py")
    parser.add_argument("--fast",       action="store_true",
                        help="Меньше текстов (быстрая диагностика)")
    parser.add_argument("--no-corpus",  action="store_true")
    args = parser.parse_args()

    # ── Загрузить модель ──────────────────────────────────────────────────
    cfg   = Variant3Config(**MODEL_CFG)
    model = Variant3GPT(cfg)
    for block in model.blocks:
        if hasattr(block, 'hmoe'):
            block.hmoe = HierarchicalMoEFFN(HMOE_CFG)

    ckpt_path = args.checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        key = "model_state" if "model_state" in ckpt else None
        if key:
            try:
                model.load_state_dict(ckpt[key])
                phase_done = ckpt.get("next_phase", "?")
                print(f"  Загружен: {ckpt_path}  (фаза {phase_done})")
            except Exception as e:
                print(f"  ⚠️  Частичная загрузка: {e}")
    else:
        print(f"  ⚠️  Чекпоинт не найден: {ckpt_path} — используем случайные веса")

    # ── Загрузить тексты ──────────────────────────────────────────────────
    texts: List[str] = []
    if not args.no_corpus:
        try:
            from repo_corpus_loader import RepoCorpusLoader
            loader = RepoCorpusLoader(_ROOT)
            for cluster in CLUSTER_TO_DOMAIN.keys():
                try:
                    for item in loader.load_cluster(cluster):
                        t = item if isinstance(item, str) else item.get("text", "")
                        if len(t) > 10:
                            texts.append(t)
                except Exception:
                    pass
        except ImportError:
            pass

    if not texts:
        texts = [
            "def forward(self, x): return self.linear(x)",
            "import torch; x = torch.randn(4, 128)",
            "loss.backward(); optimizer.zero_grad()",
            "self.attn = nn.MultiheadAttention(d_model, n_heads)",
            "x = x + self.crossing(out_a, out_b)",
            "The hexagram represents the intersection of abstract and concrete.",
            "Kryukov figure-8: abstract loop A, concrete loop B, crossing DYNAMIC.",
            "consciousness emerges from recursive self-reference in Q6 space",
            "for i, batch in enumerate(dataloader): optimizer.step()",
            "class HierarchicalMoEFFN(nn.Module): ...",
        ] * 5

    n_eval = 20 if args.fast else min(len(texts), 60)
    eval_texts = random.sample(texts, n_eval)
    print(f"  Текстов для оценки: {n_eval}")

    # ── Загрузить LCI-лог ─────────────────────────────────────────────────
    log_data = None
    log_path = args.log
    if log_path is None:
        # Попробовать найти автоматически
        auto = ckpt_path.replace(".pt", "_log.json")
        if os.path.exists(auto):
            log_path = auto
        elif os.path.exists("hmoe_self_trained_log.json"):
            log_path = "hmoe_self_trained_log.json"

    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            log_data = json.load(f)
        print(f"  LCI-лог: {log_path}  ({len(log_data)} циклов)")

    # ── Собрать метрики ───────────────────────────────────────────────────
    print("  Собираем метрики...")
    diag = _collect_metrics(model, eval_texts)

    if log_data:
        trend, n_res = _analyze_lci_log(log_data)
        diag.lci_trend   = trend
        diag.n_resonance = n_res

    # ── Сформировать рекомендации ─────────────────────────────────────────
    recs = _make_recommendations(diag, ckpt_path)

    # ── Напечатать отчёт ──────────────────────────────────────────────────
    _print_report(diag, recs, ckpt_path, log=log_data)


if __name__ == "__main__":
    main()
