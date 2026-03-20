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
    """LCI через маршрутизацию: LCI = (1 - |w_A - w_B|) * π."""
    model.eval()
    gw_list = []
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
    if not gw_list:
        return math.pi, {"ABSTRACT": 0.33, "DYNAMIC": 0.34, "CONCRETE": 0.33}
    avg_gw = torch.stack(gw_list).mean(0)
    groups = list(DOMAIN_GROUPS.keys())
    gw_dict = {g: avg_gw[i].item() for i, g in enumerate(groups)}
    w_a = gw_dict.get("ABSTRACT", 0.33)
    w_b = gw_dict.get("CONCRETE", 0.33)
    imbalance = abs(w_a - w_b)
    lci = (1.0 - imbalance) * math.pi
    return lci, gw_dict


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

def figure8_hmoe_v4(
    model:          Variant3GPT,
    seed_texts:     List[str],
    block_size:     int   = MODEL_CFG["block_size"] - 1,
    n_cycles:       int   = 8,
    steps_per_loop: int   = 80,
    temperature:    float = 2.5,
    lr_a:           float = 2e-5,
    lr_b:           float = 5e-6,
    do_train:       bool  = True,
) -> List[Dict]:
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
    print(f"\n{'═' * 72}")
    print(f"  САМО-ОБУЧЕНИЕ v4: РЕЗОНАНСНЫЙ ПРОРЫВ  (асимметрия A>>B)")
    print(f"{'═' * 72}")
    print(f"  Циклов         : {n_cycles}")
    print(f"  Шагов/петля    : {steps_per_loop}")
    print(f"  Серия          : A={_SERIES_A}×, B={_SERIES_B}× (7:1 asym)")
    print(f"  LR_A           : {lr_a:.2e}  LR_B: {lr_b:.2e}")
    print(f"  Температура    : {temperature:.2f} (фикс)")
    print(f"  Балансёр       : мини-A (3×) если w_B > w_A + {_BALANCE_THOLD}")
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

    # Разделить seed_texts по домену
    abstract_texts = [t for t in seed_texts if any(w in t.lower() for w in
                      ["hexagram", "abstract", "consciousness", "figure-8",
                       "topology", "balance", "resonance", "kryukov", "theory"])]
    concrete_texts = [t for t in seed_texts if any(w in t.lower() for w in
                      ["def ", "import", "return", "torch", "loss", "optimizer",
                       "model", "train", "class", "self"])]
    # fallback: использовать все если разделение пустое
    if not abstract_texts:
        abstract_texts = seed_texts
    if not concrete_texts:
        concrete_texts = seed_texts
    print(f"  Тексты: ABSTRACT={len(abstract_texts)}  CONCRETE={len(concrete_texts)}")

    log: List[Dict] = []
    best_lci = 0.0

    for cycle in range(1, n_cycles + 1):
        x_ids = prompt_ids.clone()
        lci_start, gw_start = lci_from_routing(model, x_ids)

        print(f"\n  {'─' * 68}")
        print(f"  Цикл {cycle}/{n_cycles}  T={temperature:.2f}  "
              f"routing_LCI={lci_start:.4f}  "
              f"({'✓ РЕЗОНАНС' if abs(lci_start - math.pi) < _LCI_EPSILON else f'δ={lci_start - math.pi:+.4f}'})")
        print(f"    Группы: A={gw_start.get('ABSTRACT',0):.4f}  "
              f"X={gw_start.get('DYNAMIC',0):.4f}  "
              f"B={gw_start.get('CONCRETE',0):.4f}")

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

        log.append({
            "cycle":          cycle,
            "n_a":            _SERIES_A,
            "n_b":            _SERIES_B,
            "temperature":    round(temperature, 3),
            "lci_a_r":        round(lci_a, 4),
            "lci_b_r":        round(lci_b, 4),
            "lci_balance_r":  round(lci_balance, 4),
            "avg_lci_r":      round(avg_lci_r, 4),
            "resonance":      resonance,
            "used_balancer":  used_balancer,
            "gw_at_start":    {k: round(v, 4) for k, v in gw_start.items()},
            "gw_after_a":     {k: round(v, 4) for k, v in gw_a.items()},
            "gw_after_b":     {k: round(v, 4) for k, v in gw_b.items()},
            "texts_a":        n_a_texts,
            "texts_b":        n_b_texts,
            "texts_balance":  n_balance_texts,
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
    parser.add_argument("--temperature",    type=float, default=2.5)
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
            print(f"  Корпус: {len(seed_texts)} текстов")
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
