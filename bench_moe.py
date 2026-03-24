#!/usr/bin/env python3
"""
bench_moe.py — Сравнительный бенчмарк трёх архитектур FFN.

Сравнивает три варианта FFN в Variant3Block:
  A. SwiGLU FFN (оригинальный, use_hierarchical_moe=False)
  B. HierarchicalMoEFFN с MultiScaleGlobalRouter (use_multiscale=True)
  C. HierarchicalMoEFFN + Q6ExpertBank (use_hex_tier=True)

Метрики:
  - Параметры модели
  - Память (param bytes)
  - Forward time (ms/batch)
  - Perplexity на репо-корпусе
  - Routing diversity (энтропия group_weights)
  - Expert utilization (коэффициент вариации нагрузки)

Usage:
  python bench_moe.py
  python bench_moe.py --fast        # 10 примеров вместо 50
  python bench_moe.py --no-corpus   # только speed/memory, без PPL
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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import (
    HMoEConfig, DOMAIN_GROUPS,
)

# ── Конфигурация ───────────────────────────────────────────────────────────────

torch.manual_seed(0)
random.seed(0)

_ROOT  = __file__ if os.path.isabs(__file__) else os.path.abspath(__file__)
_ROOT  = os.path.dirname(_ROOT)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_CFG = dict(
    vocab_size=256,
    block_size=32,
    d_model=128,
    n_heads=4,
    n_layers=4,
    ffn_mult=4,
    hamming_lambda=0.15,
    uncertainty_budget=0.25,
    dropout=0.0,
    use_domain_routing=False,   # выключаем NautilusRouter для чистоты
)

VARIANTS: Dict[str, dict] = {
    "A_SwiGLU": {
        "desc": "Оригинальный SwiGLU FFN",
        "cfg_override": {"use_hierarchical_moe": False},
        "hmoe_cfg": None,
    },
    "B_HMoE_MultiScale": {
        "desc": "HierarchicalMoE + MultiScaleGlobalRouter",
        "cfg_override": {"use_hierarchical_moe": True},
        "hmoe_cfg": HMoEConfig(d_model=128, use_multiscale=True, use_hex_tier=False),
    },
    "C_HMoE_HexTier": {
        "desc": "HierarchicalMoE + MultiScale + Q6ExpertBank (4-й уровень)",
        "cfg_override": {"use_hierarchical_moe": True},
        "hmoe_cfg": HMoEConfig(d_model=128, use_multiscale=True, use_hex_tier=True,
                               hex_tier_top_k=4, hex_tier_weight=0.3),
    },
}


# ── Утилиты ───────────────────────────────────────────────────────────────────

def make_model(variant_key: str) -> Variant3GPT:
    v = VARIANTS[variant_key]
    cfg_kwargs = {**BASE_CFG, **v["cfg_override"]}
    cfg = Variant3Config(**cfg_kwargs)
    model = Variant3GPT(cfg)

    # Если задан кастомный HMoEConfig — заменяем в блоках
    if v["hmoe_cfg"] is not None:
        from yijing_transformer.models.hierarchical_moe import HierarchicalMoEFFN
        for block in model.blocks:
            if hasattr(block, 'hmoe'):
                block.hmoe = HierarchicalMoEFFN(v["hmoe_cfg"])

    return model


def count_params(model: nn.Module) -> Tuple[int, int]:
    """Возвращает (total, trainable)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def forward_time_ms(model: nn.Module, n_runs: int = 50) -> float:
    model.eval()
    tokens = torch.randint(0, 256, (1, 31))
    # Прогрев
    for _ in range(5):
        with torch.no_grad():
            model(tokens)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(tokens)
    return (time.perf_counter() - t0) / n_runs * 1000


def encode(text: str, block_size: int = 31) -> torch.Tensor:
    ids = [min(b, 255) for b in text.encode("utf-8")][:block_size]
    return torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)


def perplexity_on_texts(model: nn.Module, texts: List[str]) -> float:
    model.eval()
    ppls = []
    for text in texts:
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        inp, tgt = tokens[:, :-1], tokens[:, 1:]
        with torch.no_grad():
            _, loss, _ = model(inp, targets=tgt)
        if loss is not None and not torch.isnan(loss):
            ppls.append(math.exp(min(loss.item(), 10)))
    return sum(ppls) / len(ppls) if ppls else float("inf")


def routing_entropy(model: nn.Module, texts: List[str]) -> Tuple[float, float]:
    """Возвращает (mean_group_entropy, expert_cv).

    group_entropy: средняя энтропия group_weights (выше = разнообразнее)
    expert_cv:     коэффициент вариации нагрузки по группам (ниже = равномернее)
    """
    model.eval()
    all_gw: List[torch.Tensor] = []
    for text in texts[:20]:
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            model(tokens[:, :-1])
        for block in model.blocks:
            info = getattr(block, '_last_moe_info', None)
            if info and 'group_weights' in info:
                gw = info['group_weights'][0].mean(0)  # (n_groups,)
                all_gw.append(gw)
                break

    if not all_gw:
        return float('nan'), float('nan')

    stacked = torch.stack(all_gw)           # (N, n_groups)
    ent_per_sample = -(stacked * torch.log(stacked + 1e-8)).sum(dim=-1)
    mean_ent = ent_per_sample.mean().item()

    mean_load = stacked.mean(dim=0)         # (n_groups,)
    cv = (mean_load.std() / (mean_load.mean() + 1e-8)).item()

    return mean_ent, cv


# ── Нагрузочный тест маршрутизации ───────────────────────────────────────────

def routing_analysis(model: nn.Module, texts: List[str]) -> Dict[str, float]:
    """Анализ паттернов маршрутизации для HMoE вариантов."""
    model.eval()
    group_names = list(DOMAIN_GROUPS.keys())
    per_group_load: Dict[str, float] = {g: 0.0 for g in group_names}
    n_samples = 0

    for text in texts[:30]:
        tokens = encode(text)
        if tokens.shape[1] < 2:
            continue
        with torch.no_grad():
            model(tokens[:, :-1])
        for block in model.blocks:
            info = getattr(block, '_last_moe_info', None)
            if info and 'group_weights' in info:
                gw = info['group_weights'][0].mean(0)
                for i, g in enumerate(group_names):
                    per_group_load[g] += gw[i].item()
                n_samples += 1
                break

    if n_samples > 0:
        for g in per_group_load:
            per_group_load[g] /= n_samples

    return per_group_load


# ── Главная функция ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",       action="store_true",
                        help="10 примеров вместо 50")
    parser.add_argument("--no-corpus",  action="store_true",
                        help="Только speed/memory, без PPL")
    args = parser.parse_args()

    n_ppl_texts = 10 if args.fast else 50

    # Загрузка корпуса (опционально)
    texts: List[str] = []
    if not args.no_corpus:
        try:
            from repo_corpus_loader import RepoCorpusLoader
            loader = RepoCorpusLoader(_ROOT)
            clusters = loader.load_all_clusters()
            for items in clusters.values():
                for item in items:
                    t = item if isinstance(item, str) else item.get("text", "")
                    if len(t) > 10:
                        texts.append(t)
            texts = random.sample(texts, min(n_ppl_texts, len(texts)))
            print(f"  Корпус: {len(texts)} текстов из репо")
        except Exception as e:
            print(f"  ⚠️  Корпус недоступен: {e}")
            texts = []
    else:
        print("  Режим --no-corpus: пропускаем PPL")

    # Синтетические тексты если корпус пуст
    if not texts:
        texts = [
            "def forward(self, x):",
            "class Model(nn.Module):",
            "import torch.nn as nn",
            "Theory of consciousness",
            "Training loop with Adam",
        ] * 10

    # ── Бенчмарк ─────────────────────────────────────────────────────────────
    n_runs = 20 if args.fast else 50

    results = {}
    print("\n" + "═" * 72)
    print("  БЕНЧМАРК: SwiGLU vs HierarchicalMoE vs HMoE+Q6ExpertBank")
    print("═" * 72)

    for name, spec in VARIANTS.items():
        print(f"\n  [{name}] {spec['desc']}")
        model = make_model(name)

        total_p, train_p = count_params(model)
        mem = param_mb(model)
        fwd_ms = forward_time_ms(model, n_runs=n_runs)

        ppl = perplexity_on_texts(model, texts) if texts else float('nan')
        ent, cv = routing_entropy(model, texts)
        load_dist = routing_analysis(model, texts)

        results[name] = {
            "desc":        spec["desc"],
            "params_M":    total_p / 1e6,
            "mem_MB":      mem,
            "fwd_ms":      fwd_ms,
            "ppl":         ppl,
            "route_ent":   ent,
            "route_cv":    cv,
            "load_dist":   load_dist,
        }

        print(f"    Параметры:   {total_p/1e6:.2f}M  ({mem:.1f} MB)")
        print(f"    Forward:     {fwd_ms:.2f} ms/batch")
        print(f"    PPL:         {ppl:.2f}")
        if not math.isnan(ent):
            print(f"    Route ent:   {ent:.3f} (↑ = разнообразнее)")
            print(f"    Route CV:    {cv:.3f} (↓ = равномернее)")
            print(f"    Нагрузка групп: " + "  ".join(
                f"{g[:3]}={v:.3f}" for g, v in load_dist.items()))

    # ── Сводная таблица ───────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  ИТОГОВАЯ ТАБЛИЦА")
    print("═" * 72)
    header = f"  {'Вариант':25s} {'Params':>8s} {'MB':>6s} {'ms':>6s} {'PPL':>8s} {'Ent':>6s} {'CV':>5s}"
    print(header)
    print("  " + "─" * 70)
    base_ppl = results["A_SwiGLU"]["ppl"]
    base_ms  = results["A_SwiGLU"]["fwd_ms"]
    for name, r in results.items():
        ppl_delta = ""
        if not math.isnan(r["ppl"]) and not math.isnan(base_ppl):
            d = r["ppl"] - base_ppl
            ppl_delta = f" ({'+' if d >= 0 else ''}{d:.1f})"
        ent_str = f"{r['route_ent']:.3f}" if not math.isnan(r.get('route_ent', float('nan'))) else "  —  "
        cv_str  = f"{r['route_cv']:.3f}"  if not math.isnan(r.get('route_cv',  float('nan'))) else "  —"
        ppl_str = f"{r['ppl']:.1f}{ppl_delta}" if not math.isnan(r["ppl"]) else "  —"
        print(f"  {name:25s} {r['params_M']:>7.2f}M {r['mem_MB']:>5.1f}MB "
              f"{r['fwd_ms']:>5.1f}ms {ppl_str:>14s} {ent_str:>6s} {cv_str:>5s}")

    # ── Анализ trade-offs ─────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  АНАЛИЗ TRADE-OFFS")
    print("═" * 72)

    r_a = results["A_SwiGLU"]
    r_b = results.get("B_HMoE_MultiScale", {})
    r_c = results.get("C_HMoE_HexTier", {})

    if r_b:
        overhead_ms  = r_b["fwd_ms"]  - r_a["fwd_ms"]
        overhead_pct = overhead_ms / r_a["fwd_ms"] * 100
        param_delta  = r_b["params_M"] - r_a["params_M"]
        print(f"\n  B vs A (MultiScale HMoE):")
        print(f"    Доп. параметры: +{param_delta:.2f}M")
        print(f"    Доп. latency:   +{overhead_ms:.2f}ms (+{overhead_pct:.0f}%)")
        if not math.isnan(r_b.get("ppl", float("nan"))):
            ppl_d = r_b["ppl"] - r_a["ppl"]
            print(f"    PPL delta:      {'+' if ppl_d >= 0 else ''}{ppl_d:.1f}")

    if r_c:
        overhead_ms  = r_c["fwd_ms"]  - r_a["fwd_ms"]
        overhead_pct = overhead_ms / r_a["fwd_ms"] * 100
        param_delta  = r_c["params_M"] - r_a["params_M"]
        print(f"\n  C vs A (HMoE + Q6ExpertBank):")
        print(f"    Доп. параметры: +{param_delta:.2f}M")
        print(f"    Доп. latency:   +{overhead_ms:.2f}ms (+{overhead_pct:.0f}%)")
        if not math.isnan(r_c.get("ppl", float("nan"))):
            ppl_d = r_c["ppl"] - r_a["ppl"]
            print(f"    PPL delta:      {'+' if ppl_d >= 0 else ''}{ppl_d:.1f}")

    print("\n  ✅ Бенчмарк завершён")


if __name__ == "__main__":
    main()
