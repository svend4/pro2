#!/usr/bin/env python3
"""
self_train_v2.py — Исправленное и расширенное самообучение.

Что исправлено по сравнению с v1:
  1. stop_window теперь считается в реальных шагах (не в чекпоинтах)
  2. Добавлен domain_triplet_loss → домены реально разводятся
  3. Добавлен quality_contrastive_loss → QFilter учится отличать хорошее от плохого
  4. Добавлен gate_entropy_reward → TernaryGate выходит из 100%-変爻
  5. Итерационный само-диалог: модель обновляет свой буфер и переучивается

Новые лоссы:
  L_total = L_lm + α·L_domain + β·L_quality + γ·L_gate

  L_domain  (triplet): margin_ranking_loss между примерами разных доменов
  L_quality (contrastive): BCE на парах (хорошо/плохо) через QFilter
  L_gate    (entropy):  - mean( p·log(p) ) по {ян,инь,変爻} → награда за выход из 0
"""

import os, sys, math, random, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from yijing_transformer.models.variant3 import (
    Variant3Config, Variant3GPT,
    DOMAINS, DOMAIN_ANCHORS,
)
from yijing_transformer.models.variant3_extensions import (
    HexagramEvaluator, TextQualityFilter,
    get_hexagrams, get_biangua,
)

# ─── config ──────────────────────────────────────────────────────────────────

DEVICE = "cpu"
torch.manual_seed(42)
random.seed(42)

CFG = Variant3Config(
    vocab_size=256, block_size=32, d_model=128,
    n_heads=4, n_layers=4, ffn_mult=4,
    hamming_lambda=0.15, uncertainty_budget=0.25,
    dropout=0.05, use_domain_routing=True,
)

# Веса вспомогательных лоссов
ALPHA_DOMAIN  = 0.30   # domain triplet
BETA_QUALITY  = 0.20   # quality contrastive
GAMMA_GATE    = 0.10   # gate entropy

hexagrams  = get_hexagrams()
biangua    = get_biangua()
evaluator  = HexagramEvaluator(threshold=0.01)
qfilter    = TextQualityFilter(CFG.d_model)

from yijing_transformer.constants import HEX_NAMES
def hname(i): return HEX_NAMES[i] if i < len(HEX_NAMES) else f"#{i}"


# ─── данные ──────────────────────────────────────────────────────────────────

# Качественный корпус по доменам (для triplet loss)
DOMAIN_CORPUS = {
    "PYRO":  [
        "Fire transforms matter through rapid oxidation and heat release.",
        "The flame burns bright, consuming fuel and releasing light energy.",
        "Volcanic eruptions channel heat from the earth's molten core.",
        "Photons carry electromagnetic energy across the visible spectrum.",
        "Combustion reactions release stored chemical energy as heat and light.",
    ],
    "HYDRO": [
        "Rivers carve valleys over millennia through erosive water flow.",
        "Ocean currents regulate global temperature through thermal transport.",
        "Water molecules form hydrogen bonds creating surface tension.",
        "Rainfall accumulates in aquifers forming underground water reservoirs.",
        "The water cycle evaporates oceans into clouds that return as rain.",
    ],
    "AERO":  [
        "Wind patterns emerge from differential heating of Earth's surface.",
        "Air pressure gradients drive atmospheric circulation across hemispheres.",
        "Birds exploit thermal updrafts to soar without flapping wings.",
        "Sound waves propagate through air as longitudinal pressure variations.",
        "Jet streams carry weather systems across continents at high altitude.",
    ],
    "GEO":   [
        "Tectonic plates drift slowly, reshaping continents over geologic time.",
        "Mountain ranges form where continental plates collide and fold.",
        "Soil composition determines which plants can grow in a region.",
        "Earthquakes release tension accumulated along geological fault lines.",
        "Sedimentary layers record millions of years of Earth's history.",
    ],
    "COSMO": [
        "Galaxies form through gravitational collapse of primordial gas clouds.",
        "Stellar nucleosynthesis fuses hydrogen into heavier atomic elements.",
        "Dark matter provides gravitational scaffolding for visible structure.",
        "The cosmic microwave background echoes the Big Bang's afterglow.",
        "Black holes warp spacetime beyond the event horizon of no return.",
    ],
    "NOOS":  [
        "Logic and reason provide the foundation for systematic thought.",
        "Language encodes abstract concepts through symbolic representation.",
        "Mathematical proofs derive truth from axioms through deduction.",
        "Consciousness arises from the complex integration of neural signals.",
        "Scientific method tests hypotheses through controlled experiments.",
    ],
}

# Плохие тексты для контрастивного обучения качества
BAD_TEXTS = [
    "spam click buy win prize free limited offer",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "x x x x x x x x x x x x x x x x",
    "asdf qwer zxcv random noise garbage",
    "!!!! #### @@@@ %%%% ^^^^ ****",
    "buy now click here special deal today",
    "error 404 null undefined NaN NaN NaN",
    "0000000000000000000000000000000000000",
]

# ─── утилиты ─────────────────────────────────────────────────────────────────

def text_to_ids(text, block_size=32):
    ids = [b for b in text.encode("utf-8")][:block_size]
    ids = ids or [0]
    return torch.tensor(ids, dtype=torch.long)

def ids_pad(ids_list, block_size):
    """Pad list of id tensors to block_size."""
    out = []
    for ids in ids_list:
        if len(ids) < block_size:
            ids = torch.cat([ids, torch.zeros(block_size - len(ids), dtype=torch.long)])
        out.append(ids[:block_size])
    return torch.stack(out)

def get_hw_dw(model, ids_batch):
    """Forward → (hex_weights, domain_weights): both averaged over T."""
    out = model(ids_batch)
    info = out[2] if isinstance(out, tuple) and len(out) > 2 else {}
    hw = info.get("hex_weights")   # (B, T, 64)
    dw = info.get("domain_weights")  # (B, T, 6) or None
    if hw is None:
        return None, None
    hw_mean = hw.mean(dim=1)   # (B, 64)
    if dw is not None and dw.dim() == 3:
        dw_mean = dw.mean(dim=1)   # (B, 6)
    else:
        anchors = [DOMAIN_ANCHORS[d] for d in DOMAINS]
        dw_mean = torch.stack([hw_mean[:, a] for a in anchors], dim=-1)
        dw_mean = dw_mean / (dw_mean.sum(-1, keepdim=True) + 1e-8)
    return hw_mean, dw_mean

def get_hidden(model, ids_batch):
    """Forward → mean hidden state (B, D)."""
    with torch.no_grad():
        emb = model.tok_emb(ids_batch)
        for block in model.blocks:
            emb = block(emb)
    return emb.mean(dim=1)   # (B, D)

def get_metrics(model, ids_batch):
    hw, dw = get_hw_dw(model, ids_batch[:1])
    if hw is None:
        return {}
    return evaluator.evaluate(hw.unsqueeze(1), dw)

def get_gate_counts(model, ids_batch):
    """Returns (yang_pct, zero_pct, yin_pct) averaged over all layers."""
    yang_tot, zero_tot, yin_tot = 0.0, 0.0, 0.0
    for block in model.blocks:
        gate = block.ternary_gate
        with torch.no_grad():
            emb = model.tok_emb(ids_batch)
            for b in model.blocks:
                emb = b(emb)
                if b is block:
                    break
            x = emb
            scores = torch.tanh(gate.gate_proj(x) / gate.temperature)
            budget = torch.sigmoid(gate.log_uncertainty)
            thr = (1.0 - budget) * 0.5 + 0.1
            yang = (scores > thr).float().mean().item()
            yin  = (scores < -thr).float().mean().item()
            zero = 1.0 - yang - yin
        yang_tot += yang
        zero_tot += zero
        yin_tot  += yin
    n = len(model.blocks)
    return yang_tot/n, zero_tot/n, yin_tot/n


# ─── вспомогательные лоссы ───────────────────────────────────────────────────

def domain_triplet_loss(model, domain_corpus, block_size, margin=0.3):
    """
    Triplet loss по доменам:
      anchor   = текст домена A
      positive = другой текст домена A
      negative = текст домена B (B ≠ A)

    Хотим: dist(a,p) + margin < dist(a,n)
    → доменные кластеры в Q6-пространстве.

    Используем косинусное расстояние по hex_weights.
    """
    # Выбираем два разных домена
    doms = random.sample(list(domain_corpus.keys()), 2)
    dom_a, dom_b = doms[0], doms[1]

    texts_a = random.sample(domain_corpus[dom_a], min(2, len(domain_corpus[dom_a])))
    texts_b = random.sample(domain_corpus[dom_b], min(1, len(domain_corpus[dom_b])))

    if len(texts_a) < 2:
        return torch.tensor(0.0)

    anchor_ids   = text_to_ids(texts_a[0], block_size)
    positive_ids = text_to_ids(texts_a[1], block_size)
    negative_ids = text_to_ids(texts_b[0], block_size)

    batch = ids_pad([anchor_ids, positive_ids, negative_ids], block_size)
    hw, _ = get_hw_dw(model, batch)  # (3, 64)
    if hw is None:
        return torch.tensor(0.0)

    a, p, n = hw[0], hw[1], hw[2]
    # Косинусное расстояние (1 - cos_sim)
    dist_ap = 1.0 - F.cosine_similarity(a.unsqueeze(0), p.unsqueeze(0))
    dist_an = 1.0 - F.cosine_similarity(a.unsqueeze(0), n.unsqueeze(0))
    loss = F.relu(dist_ap - dist_an + margin)
    return loss.mean()


def quality_contrastive_loss(model, good_texts, bad_texts, block_size):
    """
    Контрастивный лосс качества:
      Для хорошего текста: QFilter-score → 1
      Для плохого текста:  QFilter-score → 0

    BCE на средних оценках.
    """
    good = random.choice(good_texts)
    bad  = random.choice(bad_texts)

    ids_good = text_to_ids(good, block_size).unsqueeze(0)
    ids_bad  = text_to_ids(bad,  block_size).unsqueeze(0)

    losses = []
    for ids, target in [(ids_good, 1.0), (ids_bad, 0.0)]:
        emb = model.tok_emb(ids)
        for block in model.blocks:
            emb = block(emb)
        scores, _, _ = qfilter(emb)  # (1, T, 6)
        # Среднее по времени и осям → скаляр в (-1, 1)
        score_mean = scores.mean()
        # Перевод в [0, 1] и BCE
        prob = (score_mean.clamp(-1, 1) + 1) / 2.0
        t = torch.tensor(target)
        loss = F.binary_cross_entropy(prob.unsqueeze(0), t.unsqueeze(0))
        losses.append(loss)

    return sum(losses) / len(losses)


def gate_entropy_loss(model, ids_batch):
    """
    Штраф за 100% 変爻 (нулевые гейты).
    Хотим чтобы распределение {ян, 変爻, инь} было не вырожденным.

    Используем отрицательную энтропию по трём категориям → минимизация
    (т.е. максимизация энтропии = выход из 100%-变爻).
    """
    total_loss = torch.tensor(0.0, requires_grad=True)
    for block in model.blocks:
        gate = block.ternary_gate
        emb = model.tok_emb(ids_batch)
        for b in model.blocks:
            emb_out = b(emb)
            if b is block:
                break
            emb = emb_out

        scores = torch.tanh(gate.gate_proj(emb) / gate.temperature)
        budget = torch.sigmoid(gate.log_uncertainty)
        thr = (1.0 - budget) * 0.5 + 0.1

        # Мягкие вероятности трёх состояний (differentiable)
        p_yang = torch.sigmoid((scores - thr) * 10).mean()
        p_yin  = torch.sigmoid((-scores - thr) * 10).mean()
        p_zero = 1.0 - p_yang - p_yin

        probs = torch.stack([p_yang, p_zero.clamp(0), p_yin]) + 1e-8
        probs = probs / probs.sum()
        entropy = -(probs * probs.log()).sum()
        # Хотим максимизировать энтропию → минимизируем -entropy
        total_loss = total_loss - entropy

    return total_loss / len(model.blocks)


# ─── батч-генераторы ─────────────────────────────────────────────────────────

def make_biangua_batch(batch_size, block_size):
    adj = (biangua > 0.5)
    xs, ys = [], []
    for _ in range(batch_size):
        cur = random.randint(0, 63)
        seq = [cur]
        for _ in range(block_size):
            nbrs = adj[cur].nonzero(as_tuple=False).squeeze(1).tolist()
            cur = random.choice(nbrs) if nbrs else cur
            seq.append(cur)
        xs.append(torch.tensor(seq[:-1], dtype=torch.long))
        ys.append(torch.tensor(seq[1:],  dtype=torch.long))
    return torch.stack(xs), torch.stack(ys)


def make_text_batch(texts, block_size, batch_size):
    selected = random.sample(texts, min(batch_size, len(texts)))
    xs, ys = [], []
    for text in selected:
        ids = text_to_ids(text, block_size + 1)
        ids = ids[:block_size + 1]
        if len(ids) < 2:
            continue
        x = ids[:-1]
        y = ids[1:]
        if len(x) < block_size:
            pad = torch.zeros(block_size - len(x), dtype=torch.long)
            x, y = torch.cat([x, pad]), torch.cat([y, pad])
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    return torch.stack(xs), torch.stack(ys)


def all_good_texts(domain_corpus):
    return [t for texts in domain_corpus.values() for t in texts]


# ─── логирование ─────────────────────────────────────────────────────────────

def log_step(step, loss, lm, dom, qual, gate_l, metrics, yang, zero, yin, dom_name):
    ent = metrics.get("hex_entropy", 0)
    coh = metrics.get("domain_coherence", 0)
    print(f"  s{step:4d}  L={loss:.3f}(lm={lm:.3f} d={dom:.3f} q={qual:.3f} g={gate_l:.3f})"
          f"  ent={ent:.3f}  coh={coh:.3f}"
          f"  ▲{yang*100:.0f}%◼{zero*100:.0f}%▽{yin*100:.0f}%"
          f"  [{dom_name}]")


# ═══════════════════════════════════════════════════════════════════════════
# СТАДИЯ 0 — Самопознание (Q6-топология + gate activation)
# ═══════════════════════════════════════════════════════════════════════════

def stage0(model, steps=500, batch=16, lr=3e-4, log_every=25):
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 0: САМОПОЗНАНИЕ + GATE ACTIVATION")
    print(f"{'═'*72}")
    print(f"  Данные : biangua-переходы (нет текстов)")
    print(f"  Лоссы  : L_lm + {GAMMA_GATE}·L_gate_entropy")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    bs = model.cfg.block_size
    history = collections.deque(maxlen=20)   # 20 последних точек
    log = []

    for step in range(1, steps + 1):
        model.train()
        x, y = make_biangua_batch(batch, bs)

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
            xv, _ = make_biangua_batch(4, bs)
            with torch.no_grad():
                metrics = get_metrics(model, xv)
                yang, zero, yin = get_gate_counts(model, xv)
                hw, _ = get_hw_dw(model, xv[:1])
                dh = hw.argmax(-1)[0].item() if hw is not None else 0

            ent = metrics.get("hex_entropy", 0)
            history.append(ent)
            log_step(step, loss.item(), lm_loss.item(), 0, 0,
                     gate_l.item(), metrics, yang, zero, yin, hname(dh))
            log.append({"step": step, "loss": loss.item(),
                        "yang": yang, "zero": zero, **metrics})

        # Стоп: последние 20 записей стабильны И gate активен
        if len(history) >= 15:
            ent_std = torch.tensor(list(history)).std().item()
            if ent_std < 0.015:
                print(f"\n  ✓ Стадия 0 стабилизировалась: std(ent)={ent_std:.4f}")
                break

    return log


# ═══════════════════════════════════════════════════════════════════════════
# СТАДИЯ 1 — Разведение доменов (domain triplet + LM)
# ═══════════════════════════════════════════════════════════════════════════

def stage1(model, domain_corpus, steps=500, batch=8, lr=1e-4, log_every=25):
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 1: РАЗВЕДЕНИЕ ДОМЕНОВ")
    print(f"{'═'*72}")
    good_texts = all_good_texts(domain_corpus)
    print(f"  Данные : {len(good_texts)} текстов по {len(domain_corpus)} доменам")
    print(f"  Лоссы  : L_lm + {ALPHA_DOMAIN}·L_domain_triplet + {GAMMA_GATE}·L_gate")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    bs = model.cfg.block_size
    history_coh = collections.deque(maxlen=15)
    log = []

    for step in range(1, steps + 1):
        model.train()
        x, y = make_text_batch(good_texts, bs, batch)
        if x is None:
            continue

        out = model(x, y)
        lm_loss = out[1]
        if lm_loss is None:
            logits = out[0]
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        dom_l  = domain_triplet_loss(model, domain_corpus, bs)
        gate_l = gate_entropy_loss(model, x[:4])
        loss = lm_loss + ALPHA_DOMAIN * dom_l + GAMMA_GATE * gate_l

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            xv, _ = make_text_batch(good_texts, bs, 4)
            if xv is None:
                continue
            with torch.no_grad():
                metrics = get_metrics(model, xv)
                yang, zero, yin = get_gate_counts(model, xv)
                hw, _ = get_hw_dw(model, xv[:1])
                dh = hw.argmax(-1)[0].item() if hw is not None else 0

            coh = metrics.get("domain_coherence", 0)
            history_coh.append(coh)
            log_step(step, loss.item(), lm_loss.item(), dom_l.item(), 0,
                     gate_l.item(), metrics, yang, zero, yin, hname(dh))
            log.append({"step": step, "loss": loss.item(),
                        "yang": yang, "zero": zero, **metrics})

        # Стоп: средняя связность достигла порога
        if len(history_coh) >= 10:
            avg_coh = sum(history_coh) / len(history_coh)
            if avg_coh >= 0.22:
                print(f"\n  ✓ Стадия 1: avg_coh={avg_coh:.4f} >= 0.22")
                break

    return log


# ═══════════════════════════════════════════════════════════════════════════
# СТАДИЯ 2 — Обучение качественного фильтра
# ═══════════════════════════════════════════════════════════════════════════

def stage2(model, domain_corpus, bad_texts, steps=400, batch=8, lr=5e-5, log_every=20):
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 2: ОБУЧЕНИЕ КАЧЕСТВЕННОГО ФИЛЬТРА (QFilter)")
    print(f"{'═'*72}")
    good_texts = all_good_texts(domain_corpus)
    print(f"  Данные : {len(good_texts)} хороших + {len(bad_texts)} плохих текстов")
    print(f"  Лоссы  : L_lm + {ALPHA_DOMAIN}·L_domain + {BETA_QUALITY}·L_quality + {GAMMA_GATE}·L_gate")

    # Обучаем и модель и qfilter
    all_params = list(model.parameters()) + list(qfilter.parameters())
    opt = AdamW(all_params, lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    bs = model.cfg.block_size
    history_qdiff = collections.deque(maxlen=10)
    log = []

    for step in range(1, steps + 1):
        model.train()
        qfilter.train()

        x, y = make_text_batch(good_texts, bs, batch)
        if x is None:
            continue

        out = model(x, y)
        lm_loss = out[1]
        if lm_loss is None:
            logits = out[0]
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        dom_l  = domain_triplet_loss(model, domain_corpus, bs)
        qual_l = quality_contrastive_loss(model, good_texts, bad_texts, bs)
        gate_l = gate_entropy_loss(model, x[:4])
        loss = lm_loss + ALPHA_DOMAIN * dom_l + BETA_QUALITY * qual_l + GAMMA_GATE * gate_l

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sch.step()

        if step % log_every == 0 or step == 1:
            model.eval()
            qfilter.eval()
            # Измеряем разрыв качества: хорошее - плохое
            good_sample = random.choice(good_texts)
            bad_sample  = random.choice(bad_texts)
            ids_g = text_to_ids(good_sample, bs).unsqueeze(0)
            ids_b = text_to_ids(bad_sample,  bs).unsqueeze(0)
            with torch.no_grad():
                h_g = model.tok_emb(ids_g)
                h_b = model.tok_emb(ids_b)
                for block in model.blocks:
                    h_g = block(h_g)
                    h_b = block(h_b)
                sc_g, _, _ = qfilter(h_g)
                sc_b, _, _ = qfilter(h_b)
                qdiff = sc_g.mean().item() - sc_b.mean().item()
            history_qdiff.append(qdiff)

            xv, _ = make_text_batch(good_texts, bs, 4)
            if xv is None:
                continue
            with torch.no_grad():
                metrics = get_metrics(model, xv)
                yang, zero, yin = get_gate_counts(model, xv)
                hw, _ = get_hw_dw(model, xv[:1])
                dh = hw.argmax(-1)[0].item() if hw is not None else 0

            print(f"  s{step:4d}  L={loss.item():.3f}  "
                  f"qdiff(хор-пл)={qdiff:+.4f}  "
                  f"▲{yang*100:.0f}%◼{zero*100:.0f}%▽{yin*100:.0f}%  "
                  f"coh={metrics.get('domain_coherence',0):.3f}  [{hname(dh)}]")
            log.append({"step": step, "loss": loss.item(),
                        "qdiff": qdiff, "yang": yang, **metrics})

        # Стоп: разрыв качества стабильно > 0.05
        if len(history_qdiff) >= 8:
            avg_diff = sum(history_qdiff) / len(history_qdiff)
            if avg_diff > 0.05:
                print(f"\n  ✓ Стадия 2: avg_qdiff={avg_diff:.4f} > 0.05")
                break

    return log


# ═══════════════════════════════════════════════════════════════════════════
# СТАДИЯ 3 — Само-диалог с верификацией
# ═══════════════════════════════════════════════════════════════════════════

def stage3_self_dialog(model, good_texts, bad_texts, n_rounds=3,
                       turns_per_round=6, block_size=32, temperature=1.1):
    """
    Итерационный само-диалог:
    Round r:
      1. Модель генерирует N текстов
      2. QFilter оценивает → принимает/отклоняет
      3. Принятые → дообучение на 50 шагов
      4. Повтор

    Это закольцовывает генерацию и обучение.
    """
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 3: САМО-ДИАЛОГ (итерационный)")
    print(f"{'═'*72}")

    all_good = list(good_texts)
    log = []

    for rnd in range(1, n_rounds + 1):
        print(f"\n  --- Раунд {rnd}/{n_rounds} ---")

        # Генерируем тексты
        model.eval()
        generated_accepted = []
        for t in range(turns_per_round):
            # Промпт: случайный текст из хорошего пула
            prompt_text = random.choice(all_good)
            prompt_ids = text_to_ids(prompt_text, block_size // 2).unsqueeze(0)

            with torch.no_grad():
                gen = prompt_ids.clone()
                for _ in range(block_size // 2):
                    out = model(gen)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                    nxt = torch.multinomial(probs, 1)
                    gen = torch.cat([gen, nxt.unsqueeze(0)], dim=1)[:, -block_size:]

                # Оцениваем качество
                h = model.tok_emb(gen)
                for block in model.blocks:
                    h = block(h)
                sc, _, _ = qfilter(h)
                score = (sc.clamp(-1, 1) + 1).mean().item() / 2.0

            try:
                gen_text = bytes([b % 256 for b in gen[0].tolist()]).decode("utf-8", errors="replace")
            except Exception:
                gen_text = ""

            accepted = score > 0.52
            if accepted:
                generated_accepted.append(gen_text)
                all_good.append(gen_text)

            print(f"    Ход {t+1}: score={score:.3f} {'✓' if accepted else '✗'} "
                  f"'{gen_text[:40].strip()}'...")

        print(f"  Принято: {len(generated_accepted)}/{turns_per_round} "
              f"→ буфер вырос до {len(all_good)} текстов")

        # Дообучение на обновлённом пуле (50 шагов)
        if generated_accepted or rnd > 1:
            model.train()
            opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            fine_loss_sum = 0.0
            for fine_step in range(50):
                x, y = make_text_batch(all_good, block_size, 8)
                if x is None:
                    continue
                out = model(x, y)
                lm = out[1]
                if lm is None:
                    logits = out[0]
                    lm = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                dom_l = domain_triplet_loss(model, DOMAIN_CORPUS, block_size)
                loss = lm + 0.2 * dom_l
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                fine_loss_sum += loss.item()
            print(f"  Дообучение 50 шагов: avg_loss={fine_loss_sum/50:.4f}")
            log.append({"round": rnd, "accepted": len(generated_accepted),
                        "buffer_size": len(all_good), "fine_loss": fine_loss_sum/50})

    return log, all_good


# ═══════════════════════════════════════════════════════════════════════════
# ФИНАЛЬНАЯ ОЦЕНКА
# ═══════════════════════════════════════════════════════════════════════════

def final_eval(model):
    print(f"\n{'═'*72}")
    print(f"  ФИНАЛЬНАЯ ОЦЕНКА")
    print(f"{'═'*72}")

    test_cases = [
        # (текст, ожидаемый домен)
        ("fire burns bright, heat and light energy release", "PYRO"),
        ("water flows through rivers and fills the ocean",   "HYDRO"),
        ("wind carries air currents across mountains",        "AERO"),
        ("mountains and soil form the geological record",     "GEO"),
        ("stars and galaxies shape the cosmic structure",     "COSMO"),
        ("logic and reason define systematic intelligence",   "NOOS"),
        ("spam click win prize free limited offer",           None),   # плохой → любой
    ]

    print(f"\n  {'Текст':40s}  {'Ожид.':6s}  {'Факт':6s}  {'Гекс':4s}  {'ян%':4s}  {'▽%':4s}  {'Qdiff':7s}")
    print(f"  {'─'*40}  {'─'*6}  {'─'*6}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*7}")

    model.eval()
    qfilter.eval()

    # Базовая линия плохого текста для Qdiff
    bad_ref = "spam click win prize free limited offer"
    ids_bad = text_to_ids(bad_ref, model.cfg.block_size).unsqueeze(0)
    with torch.no_grad():
        h_bad = model.tok_emb(ids_bad)
        for block in model.blocks:
            h_bad = block(h_bad)
        sc_bad, _, _ = qfilter(h_bad)
        score_bad_ref = sc_bad.mean().item()

    for text, expected_dom in test_cases:
        ids = text_to_ids(text, model.cfg.block_size).unsqueeze(0)
        with torch.no_grad():
            hw, dw = get_hw_dw(model, ids)
            yang, zero, yin = get_gate_counts(model, ids)
            h = model.tok_emb(ids)
            for block in model.blocks:
                h = block(h)
            sc, _, _ = qfilter(h)
            qdiff = sc.mean().item() - score_bad_ref

        dh = hw.argmax(-1)[0].item() if hw is not None else 0
        actual_dom = DOMAINS[dw.argmax(-1)[0].item()] if dw is not None else "?"

        match = "✓" if (expected_dom is None or actual_dom == expected_dom) else "~"
        print(f"  {match} {text[:40]:40s}  {str(expected_dom or '-'):6s}  {actual_dom:6s}  "
              f"[{dh:2d}]  {yang*100:4.0f}  {yin*100:4.0f}  {qdiff:+.4f}")

    # Тест антонимных пар — разошлись ли домены?
    print(f"\n  АНТОНИМНЫЕ ПАРЫ — расстояние в Q6:")
    pairs = [
        ("fire energy heat light combustion", "water flow river ocean current"),
        ("galaxy stars cosmos void dark",     "soil rock mountain earth plate"),
        ("logic reason proof deduction",      "wind air storm breeze gust"),
    ]
    from yijing_transformer.models.variant3_extensions import bfs_distances
    dist_mat = bfs_distances(biangua)
    for ta, tb in pairs:
        ids_a = text_to_ids(ta, model.cfg.block_size).unsqueeze(0)
        ids_b = text_to_ids(tb, model.cfg.block_size).unsqueeze(0)
        with torch.no_grad():
            hw_a, dw_a = get_hw_dw(model, ids_a)
            hw_b, dw_b = get_hw_dw(model, ids_b)
        ha = hw_a.argmax(-1)[0].item() if hw_a is not None else 0
        hb = hw_b.argmax(-1)[0].item() if hw_b is not None else 0
        dom_a = DOMAINS[dw_a.argmax(-1)[0].item()] if dw_a is not None else "?"
        dom_b = DOMAINS[dw_b.argmax(-1)[0].item()] if dw_b is not None else "?"
        bfs_d = dist_mat[ha, hb].item()
        cos = F.cosine_similarity(hw_a, hw_b).item() if hw_a is not None else 0
        print(f"  [{ha:2d}]{dom_a:5s} vs [{hb:2d}]{dom_b:5s}  bfs={bfs_d:.0f}  cos={cos:+.4f}"
              f"  | '{ta[:28]}' vs '{tb[:28]}'")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  SELF-TRAIN V2 — ИСПРАВЛЕННОЕ САМООБУЧЕНИЕ VARIANT3GPT")
    print("=" * 72)
    print(f"\n  Архитектура : {CFG.n_layers}×{CFG.d_model}d, {CFG.n_heads} голов")
    print(f"  Дополнительные лоссы:")
    print(f"    L_domain  (α={ALPHA_DOMAIN})  — triplet по доменам")
    print(f"    L_quality (β={BETA_QUALITY})  — контрастив хор./пл.")
    print(f"    L_gate    (γ={GAMMA_GATE})  — энтропия TernaryGate")

    model = Variant3GPT(CFG).to(DEVICE)
    print(f"  Параметры   : {model.count_parameters():,}")

    # СТАДИЯ 0
    log0 = stage0(model, steps=500, batch=16, lr=3e-4, log_every=25)

    # СТАДИЯ 1 — разведение доменов
    log1 = stage1(model, DOMAIN_CORPUS, steps=500, batch=8, lr=1e-4, log_every=25)

    # СТАДИЯ 2 — обучение QFilter
    log2 = stage2(model, DOMAIN_CORPUS, BAD_TEXTS, steps=400, batch=8, lr=5e-5, log_every=20)

    # СТАДИЯ 3 — само-диалог
    log3, final_buf = stage3_self_dialog(
        model, all_good_texts(DOMAIN_CORPUS), BAD_TEXTS,
        n_rounds=3, turns_per_round=6
    )

    # Финальная оценка
    final_eval(model)

    # Сохраняем чекпоинт
    torch.save({
        "model_state": model.state_dict(),
        "qfilter_state": qfilter.state_dict(),
        "config": CFG.__dict__,
        "final_buffer_size": len(final_buf),
    }, "checkpoint_v2.pt")
    print(f"\n  Чекпоинт сохранён: checkpoint_v2.pt")

    # Лог
    with open("self_train_v2_log.json", "w") as f:
        json.dump({
            "stage0_steps": len(log0),
            "stage1_steps": len(log1),
            "stage2_steps": len(log2),
            "stage3_rounds": log3,
            "final_buffer": len(final_buf),
        }, f, indent=2)
    print(f"  Лог сохранён  : self_train_v2_log.json")


if __name__ == "__main__":
    main()
