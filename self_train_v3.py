#!/usr/bin/env python3
"""
self_train_v3.py — Самообучение по Алгоритму Скарабея (Крюков, data2).

Три изменения по сравнению с v2:

  1. STAGE 0 — Scarab Topology (восьмёрка вместо случайного обхода)
     make_biangua_batch()  →  make_figure8_biangua_batch()
     Q6-пространство обходится как ФИЗИЧЕСКОЕ пространство в алгоритме Скарабея:
       Петля A: предпочитает флипы бит 0,1,2 (верхняя полусфера Q6)
       Петля B: предпочитает флипы бит 3,4,5 (нижняя полусфера Q6)
       Точки пересечения: гексаграммы с ровно 3 битами (экватор Q6)
       Anti-circle: бит не флипается > 4 раз подряд
       Anti-line:   одно направление > 3 шагов → форс-поворот

  2. STAGE 3 — figure8_dialog() вместо stage3_self_dialog()
     Само-диалог с LCI-контролем и авто-масштабом температуры.

  3. СВЯЗЬ data2 ↔ pro2
     Алгоритм Скарабея из data2/scarab_algorithm.py:
       "Deformed figure-8 in 3D physical space → 6-bit Q6 symbol"
     Этот файл: то же самое в 6D семантическом пространстве Q6.
     Один паттерн — два уровня реализации.

Все лоссы v2 сохранены: L_lm + α·L_domain + β·L_quality + γ·L_gate.
"""

import os, sys, math, random, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ── Импорт из v2 (лоссы, данные, метрики) ────────────────────────────────────
from self_train_v2 import (
    CFG, DEVICE, DOMAIN_CORPUS, BAD_TEXTS,
    ALPHA_DOMAIN, BETA_QUALITY, GAMMA_GATE,
    hexagrams, biangua,
    hname, HEX_NAMES,
    text_to_ids, ids_pad,
    get_hw_dw, get_hidden, get_metrics, get_gate_counts,
    domain_triplet_loss, quality_contrastive_loss, gate_entropy_loss,
    make_text_batch, all_good_texts,
    stage1, stage2, final_eval,
    evaluator, qfilter,
)

# ── Импорт figure8_dialog из self_train.py ───────────────────────────────────
from self_train import (
    figure8_dialog,
    _ODD_SERIES, _LCI_EPSILON, _lci,
    _generate, _ids_to_text, _emb_of,
    compute_text_quality, RAGBuffer, build_rag_buffer,
    QUALITY_CORPUS_RAW,
)

from yijing_transformer.models.variant3 import Variant3GPT, DOMAINS, DOMAIN_ANCHORS


# ═══════════════════════════════════════════════════════════════════════════
# SCARAB BIANGUA WALK — фигура-8 в Q6-пространстве
# ═══════════════════════════════════════════════════════════════════════════

# Q6: 6 измерений, каждое — один бит гексаграммы
_UPPER_BITS = [0, 1, 2]   # Петля A: «верхняя полусфера» (биты 0–2)
_LOWER_BITS = [3, 4, 5]   # Петля B: «нижняя полусфера» (биты 3–5)


def _figure8_walk(start: int, length: int) -> list:
    """
    Figure-8 обход biangua-графа Q6 (Scarab Algorithm).

    Структура одного цикла восьмёрки:
    ───────────────────────────────────────────────────────────────
    [Точка пересечения]   гексаграмма с ровно 3 установленными битами
         ↓ n_a шагов (нечётное)
    [Петля A — верхняя]   флипы бит 0,1,2 (битовое пространство 0–2)
         ↓
    [Точка пересечения]   возврат на «экватор» Q6
         ↓ n_b шагов (следующее нечётное)
    [Петля B — нижняя]    флипы бит 3,4,5 (битовое пространство 3–5)
         ↓
    [Точка пересечения]
    ───────────────────────────────────────────────────────────────

    Anti-circle: один бит не флипается > 4 раз подряд → форс-смена.
    Anti-line:   один бит не флипается > 3 раз подряд → предпочесть другой.
    """
    seq = [start]
    cur = start
    flip_hist = collections.deque(maxlen=5)   # история флипнутых бит

    # Серии с отскоком: 1→3→5→7→5→3→1→3→...
    series_seq = []
    idx, direction = 0, +1
    while len(series_seq) < 8:
        series_seq.append(_ODD_SERIES[idx])
        idx += direction
        if idx >= len(_ODD_SERIES) - 1:
            direction = -1
        elif idx <= 0:
            direction = +1

    for s_idx, n_steps in enumerate(series_seq):
        preferred = _UPPER_BITS if s_idx % 2 == 0 else _LOWER_BITS
        other     = _LOWER_BITS if s_idx % 2 == 0 else _UPPER_BITS

        for _ in range(n_steps):
            # Anti-circle: последние 4 флипа одинаковы → перейти на другое измерение
            if len(flip_hist) >= 4 and len(set(list(flip_hist)[-4:])) == 1:
                candidates = list(other)
            else:
                candidates = list(preferred)
                # Anti-line: один бит 3 раза подряд → убираем его из кандидатов
                if (len(flip_hist) >= 3
                        and flip_hist[-1] == flip_hist[-2] == flip_hist[-3]):
                    exclude = flip_hist[-1]
                    filtered = [b for b in candidates if b != exclude]
                    if filtered:
                        candidates = filtered

            bit = random.choice(candidates)
            cur = cur ^ (1 << bit)
            cur = cur & 0x3F   # держим в [0, 63]
            flip_hist.append(bit)
            seq.append(cur)

            if len(seq) >= length:
                return seq[:length]

        # Точка пересечения: направляемся к «экватору» (ровно 3 бита)
        bits_set = bin(cur).count('1')
        if bits_set < 3:
            zero_bits = [i for i in range(6) if not (cur >> i & 1)]
            if zero_bits:
                bit = random.choice(zero_bits)
                cur = cur ^ (1 << bit)
                flip_hist.append(bit)
                seq.append(cur)
        elif bits_set > 3:
            one_bits = [i for i in range(6) if cur >> i & 1]
            if one_bits:
                bit = random.choice(one_bits)
                cur = cur ^ (1 << bit)
                flip_hist.append(bit)
                seq.append(cur)

        if len(seq) >= length:
            return seq[:length]

    # Padding если не хватило
    while len(seq) < length:
        seq.append(seq[-1])
    return seq[:length]


def make_figure8_biangua_batch(batch_size: int, block_size: int):
    """
    Батч для Stage 0 по паттерну фигуры-8 (Scarab Algorithm).

    Вместо случайного обхода biangua-графа (v1/v2):
      случайный сосед → следующий сосед → ...  (круг, деградирует)

    Здесь: structured figure-8 с anti-circle, anti-line, нечётными сериями.

    Мост data2 ↔ pro2:
      data2 scarab_algorithm.py  → figure-8 в 3D физическом пространстве
      Этот файл                  → figure-8 в 6D семантическом Q6-пространстве
    """
    xs, ys = [], []
    for _ in range(batch_size):
        start = random.randint(0, 63)
        walk = _figure8_walk(start, block_size + 1)
        x = torch.tensor(walk[:-1], dtype=torch.long)
        y = torch.tensor(walk[1:],  dtype=torch.long)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 0 — Scarab Topology (восьмёрка в Q6)
# ═══════════════════════════════════════════════════════════════════════════

def stage0_scarab(model, steps=500, batch=16, lr=3e-4, log_every=25):
    """
    Stage 0 с figure-8 biangua traversal.

    Сравнение с v2:
      v2: make_biangua_batch()          — случайный обход (круг)
      v3: make_figure8_biangua_batch()  — восьмёрка (Scarab)

    Восьмёрка учит модель:
    - различать «верхнюю полусферу» Q6 (биты 0–2) и «нижнюю» (биты 3–5)
    - переходить через «экватор» (гексаграммы с 3 битами)
    - не застревать в одном направлении (anti-circle)
    - не идти прямо (anti-line)

    В итоге — топология Q6 выучивается структурированно, а не хаотично.
    """
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 0: SCARAB TOPOLOGY (фигура-8 в Q6)")
    print(f"{'═'*72}")
    print(f"  Данные  : biangua-граф Q6 обход по Алгоритму Скарабея")
    print(f"  Петля A : биты 0,1,2 (верхняя полусфера Q6)")
    print(f"  Петля B : биты 3,4,5 (нижняя полусфера Q6)")
    print(f"  Пересеч.: гексаграммы с ровно 3 битами (экватор Q6)")
    print(f"  Лоссы   : L_lm + {GAMMA_GATE}·L_gate_entropy")
    print(f"  Связь   : data2/scarab_algorithm.py ↔ pro2/Q6 (один паттерн)")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    bs = model.cfg.block_size
    history_ent = collections.deque(maxlen=20)
    # Дополнительно: LCI серий — мера качества восьмёрки
    lci_series = collections.deque(maxlen=10)
    log = []

    for step in range(1, steps + 1):
        model.train()
        x, y = make_figure8_biangua_batch(batch, bs)

        out = model(x, y)
        lm_loss = out[1]
        if lm_loss is None:
            logits = out[0]
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        gate_l = gate_entropy_loss(model, x[:4])
        loss = lm_loss + GAMMA_GATE * gate_l

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            xv, _ = make_figure8_biangua_batch(4, bs)
            with torch.no_grad():
                metrics = get_metrics(model, xv)
                yang, zero, yin = get_gate_counts(model, xv)
                hw, _ = get_hw_dw(model, xv[:1])
                dh = hw.argmax(-1)[0].item() if hw is not None else 0

                # LCI серии: сходство начала и конца walk
                emb_start = _emb_of(model, xv[:1, :bs//4])
                emb_end   = _emb_of(model, xv[:1, -bs//4:])
                lci_val   = _lci(emb_start, emb_end)
                lci_series.append(lci_val)

            ent = metrics.get("hex_entropy", 0)
            history_ent.append(ent)
            coh = metrics.get("domain_coherence", 0)
            res = "≈π" if abs(lci_val - math.pi) < _LCI_EPSILON else (
                  "↑"  if lci_val < math.pi else "↓")

            print(f"  s{step:4d}  L={loss.item():.4f}  "
                  f"ent={ent:.3f}  coh={coh:.3f}  "
                  f"LCI={lci_val:.3f}({res})  "
                  f"▲{yang*100:.0f}%◼{zero*100:.0f}%▽{yin*100:.0f}%  "
                  f"[{dh:2d}]{hname(dh)}")
            log.append({"step": step, "loss": loss.item(),
                        "lci": lci_val, "yang": yang, "zero": zero, **metrics})

        # Стоп: энтропия стабилизировалась
        if len(history_ent) >= 15:
            ent_std = torch.tensor(list(history_ent)).std().item()
            if ent_std < 0.015:
                avg_lci = sum(lci_series) / len(lci_series) if lci_series else 0
                print(f"\n  ✓ Стадия 0 завершена: std(ent)={ent_std:.4f}  "
                      f"avg_LCI={avg_lci:.3f} (цель=π={math.pi:.3f})")
                break

    return log


# ═══════════════════════════════════════════════════════════════════════════
# MAIN v3
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  SELF-TRAIN V3 — SCARAB ALGORITHM (Крюков, data2 → pro2)")
    print("=" * 72)
    print(f"\n  Архитектура : {CFG.n_layers}×{CFG.d_model}d, {CFG.n_heads} голов")
    print(f"  Изменения v3 vs v2:")
    print(f"    Stage 0: случайный обход → figure-8 Scarab Topology")
    print(f"    Stage 3: stage3_self_dialog → figure8_dialog (LCI-контроль)")
    print(f"  data2 bridge: scarab_algorithm.py → Q6-восьмёрка в pro2")
    print(f"\n  Дополнительные лоссы:")
    print(f"    L_domain  (α={ALPHA_DOMAIN})  — triplet по доменам")
    print(f"    L_quality (β={BETA_QUALITY})  — контрастив хор./пл.")
    print(f"    L_gate    (γ={GAMMA_GATE})  — энтропия TernaryGate")

    torch.manual_seed(42)
    random.seed(42)

    model = Variant3GPT(CFG).to(DEVICE)
    print(f"  Параметры   : {model.count_parameters():,}")

    # STAGE 0 — Scarab Topology (figure-8 в Q6)
    log0 = stage0_scarab(model, steps=500, batch=16, lr=3e-4, log_every=25)

    # STAGE 1 — разведение доменов (из v2, без изменений)
    log1 = stage1(model, DOMAIN_CORPUS, steps=500, batch=8, lr=1e-4, log_every=25)

    # STAGE 2 — обучение QFilter (из v2, без изменений)
    log2 = stage2(model, DOMAIN_CORPUS, BAD_TEXTS, steps=400, batch=8, lr=5e-5, log_every=20)

    # STAGE 3 — figure8_dialog (вместо stage3_self_dialog)
    # Строим RAG-буфер из качественного корпуса
    print(f"\n{'─'*72}")
    print(f"  Строим RAG-буфер из {len(QUALITY_CORPUS_RAW)} текстов для figure8_dialog...")

    from self_train import STAGE_PARAMS
    sp1 = STAGE_PARAMS[1]
    rag_buf = build_rag_buffer(model, QUALITY_CORPUS_RAW, sp1, CFG.block_size)

    # Дополняем буфер текстами из DOMAIN_CORPUS
    good_texts = all_good_texts(DOMAIN_CORPUS)
    for text in good_texts:
        from self_train import compute_text_quality
        score, emb = compute_text_quality(model, text, CFG.block_size)
        rag_buf.add(text, emb, score)
    print(f"  RAG-буфер после пополнения доменными текстами: {len(rag_buf)} текстов")

    lci_log = figure8_dialog(
        model, rag_buf, CFG.block_size,
        n_cycles=4, temperature=1.1, do_train=True
    )

    # Финальная оценка (из v2)
    final_eval(model)

    # Сохраняем чекпоинт
    torch.save({
        "model_state":   model.state_dict(),
        "qfilter_state": qfilter.state_dict(),
        "config":        CFG.__dict__,
        "version":       "v3",
        "final_rag_buf": len(rag_buf),
    }, "checkpoint_v3.pt")
    print(f"\n  Чекпоинт сохранён: checkpoint_v3.pt")

    # Лог
    avg_lci = (sum((x["lci_a"] + x["lci_b"]) / 2 for x in lci_log) / len(lci_log)
               if lci_log else 0.0)
    resonance = abs(avg_lci - math.pi) < _LCI_EPSILON

    summary = {
        "version": "v3",
        "stage0_steps":  len(log0),
        "stage1_steps":  len(log1),
        "stage2_steps":  len(log2),
        "figure8_cycles": len(lci_log),
        "figure8_avg_lci": round(avg_lci, 4),
        "figure8_resonance": resonance,
        "final_rag_buf": len(rag_buf),
        "data2_bridge": "scarab_algorithm.py → figure8_biangua (Q6)",
    }
    with open("self_train_v3_log.json", "w") as f:
        json.dump({"summary": summary, "stage0": log0, "stage1": log1,
                   "stage2": log2, "figure8_lci": lci_log}, f, indent=2)
    print(f"  Лог сохранён  : self_train_v3_log.json")

    res_mark = "✓ РЕЗОНАНС (LCI ≈ π)" if resonance else f"δ={avg_lci-math.pi:+.3f}"
    print(f"\n  РЕАЛИЗОВАНО:")
    print(f"  Stage 0: biangua-граф обходится по восьмёрке (Scarab Topology)")
    print(f"    Петля A: Q6 биты 0,1,2  ↔  Петля B: Q6 биты 3,4,5")
    print(f"    Anti-circle + Anti-line + нечётные серии {{1,3,5,7}}")
    print(f"  Stage 3: figure8_dialog — LCI={avg_lci:.3f}  {res_mark}")
    print(f"  Мост: data2/scarab_algorithm.py → pro2/self_train_v3.py")
    print(f"    Один паттерн: физическое пространство ↔ семантическое Q6")


if __name__ == "__main__":
    main()
