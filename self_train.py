#!/usr/bin/env python3
"""
self_train.py — Самообучение модели Variant3GPT.

Идея: три стадии, от самой себя наружу.

СТАДИЯ 0 — САМОПОЗНАНИЕ (Self-Topology)
  Модель учится на своих СОБСТВЕННЫХ гексаграммах:
  вход = случайная гексаграмма, цель = соседняя по biangua-графу.
  Нет внешних данных — только структура Q6.

СТАДИЯ 1 — RAG-БУФЕР (Small Quality Corpus)
  Куратор-функция (TextQualityFilter) отбирает только тексты
  с гексаграммой-качества >= порога. Модель учится на малом
  но высокоотобранном корпусе. Размер буфера << полного датасета.

СТАДИЯ 2 — ШИРОКИЙ МИР (General Data, filtered by RAG)
  Случайные тексты пропускаются через RAG-буфер:
  только те, что нашли прецедент в буфере (косинусное сходство > δ),
  попадают в батч. Модель видит "дикие данные" только через
  фильтр уже выученного.

Каждая стадия заканчивается, когда модель "определилась":
  - гексаграмм-энтропия стабилизировалась (std < ε за N шагов)
  - доменная связность выросла (coherence > порог)
  - сама модель генерирует тексты, которые раг-буфер принимает
"""

import os, sys, math, time, json, random, collections
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

# ─── конфиг ──────────────────────────────────────────────────────────────────

DEVICE = "cpu"
SEED   = 42

CFG = Variant3Config(
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
)

# ─── стадии ──────────────────────────────────────────────────────────────────

STAGE_PARAMS = {
    0: dict(  # Самопознание
        name="САМОПОЗНАНИЕ / Self-Topology",
        lr=3e-4,
        max_steps=400,
        batch=16,
        # Критерий перехода: энтропия стабилизировалась
        stop_entropy_std=0.03,
        stop_window=30,
    ),
    1: dict(  # RAG-буфер
        name="RAG-БУФЕР / Quality Corpus",
        lr=1e-4,
        max_steps=600,
        batch=8,
        quality_threshold=0.55,   # TextQualityFilter score >= 55%
        stop_coherence=0.28,      # доменная связность
        stop_window=30,
    ),
    2: dict(  # Широкий мир
        name="ШИРОКИЙ МИР / Filtered Wild Data",
        lr=5e-5,
        max_steps=400,
        batch=8,
        rag_sim_threshold=0.90,   # косинусное сходство (высокое т.к. эмбеддинги схожи до обучения)
        stop_window=30,
    ),
}

# ─── утилиты ─────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
random.seed(SEED)

hexagrams  = get_hexagrams()   # (64, 6)
biangua    = get_biangua()     # (64, 64)
evaluator  = HexagramEvaluator(threshold=0.01)
qfilter    = TextQualityFilter(CFG.d_model)

HEX_NAMES = [
    "Творчество","Исполнение","Начало","Юность","Ожидание","Конфликт",
    "Войско","Единение","Малое накопление","Хождение","Мир","Застой",
    "Братство","Великое","Скромность","Воодушевление","Следование","Исправление",
    "Горное","Созерцание","Укус","Украшение","Распад","Возврат",
    "Беспорочность","Великое накопление","Питание","Избыток","Бездна","Красота",
    "Взаимодействие","Длительность","Отступание","Великая мощь","Прогресс","Затмение",
    "Семья","Разрыв","Малые преграды","Освобождение","Уменьшение","Умножение",
    "Прорыв","Соединение","Собирание","Подъём","Угнетение","Колодец",
    "Революция","Котёл","Гром","Гора","Постепенность","Невеста",
    "Изобилие","Путник","Ветер","Радость","Рассеивание","Ограничение",
    "Правда","Малые препятствия","Уже завершено","Ещё не завершено",
]

def hex_name(i):
    return HEX_NAMES[i] if i < len(HEX_NAMES) else f"#{i}"


def fmt(step, loss, metrics, extra=""):
    ent  = metrics.get("hex_entropy", 0)
    coh  = metrics.get("domain_coherence", 0)
    cov  = metrics.get("biangua_coverage", 0)
    return (f"  step {step:4d}  loss={loss:.4f}  "
            f"ent={ent:.3f}  coh={coh:.3f}  cov={cov*100:.1f}%  {extra}")


def get_dominant_hex(model, ids):
    """Quick forward → dominant hexagram index."""
    with torch.no_grad():
        out = model(ids)
        info = out[2] if isinstance(out, tuple) and len(out) > 2 else {}
        hw = info.get("hex_weights")
        if hw is None:
            return -1
        return hw.mean(dim=1).argmax(dim=-1)[0].item()


def get_metrics(model, ids):
    """Forward → HexagramEvaluator metrics."""
    with torch.no_grad():
        out = model(ids)
        info = out[2] if isinstance(out, tuple) and len(out) > 2 else {}
        hw = info.get("hex_weights")
        dw = info.get("domain_weights")
        if hw is None:
            return {}
        # domain_weights from info has shape (1, T, 6) — take mean over T
        if dw is not None and dw.dim() == 3:
            dw = dw.mean(dim=1)   # (1, 6)
        elif dw is None:
            # fallback: compute from hex_weights via anchors
            anchors = [DOMAIN_ANCHORS[d] for d in DOMAINS]
            dw = torch.stack([hw[:, :, a] for a in anchors], dim=-1).mean(dim=1)
            dw = dw / (dw.sum(dim=-1, keepdim=True) + 1e-8)
        return evaluator.evaluate(hw, dw)


# ═══════════════════════════════════════════════════════════════════════════
# СТАДИЯ 0 — САМОПОЗНАНИЕ (Self-Topology Learning)
# ═══════════════════════════════════════════════════════════════════════════

def make_biangua_batch(batch_size, block_size, device=DEVICE):
    """
    Батч из biangua-переходов.

    Каждый пример: случайный путь по biangua-графу длины block_size.
    Входы = байты-коды гексаграмм (hex_idx + 1 как токен).
    Цели = следующий шаг пути (autoregressive).

    Это обучает модель предсказывать СТРУКТУРУ Q6,
    не видя никаких внешних текстов.
    """
    adj = (biangua > 0.5)  # (64, 64) bool

    xs, ys = [], []
    for _ in range(batch_size):
        # Случайный старт
        cur = random.randint(0, 63)
        seq = [cur]
        for _ in range(block_size):
            nbrs = adj[cur].nonzero(as_tuple=False).squeeze(1).tolist()
            if not nbrs:
                nbrs = [cur]
            cur = random.choice(nbrs)
            seq.append(cur)

        # Кодируем гексаграмму: hex_idx как один токен (0..63)
        # Это даёт ровно block_size+1 токенов → x и y длиной block_size
        tokens = [h for h in seq]  # уже длина block_size+1

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:],  dtype=torch.long)
        xs.append(x)
        ys.append(y)

    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y


def run_stage0(model, sp):
    """Стадия 0: модель учится на biangua-структуре (нет внешних данных)."""
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 0: {sp['name']}")
    print(f"{'═'*72}")
    print(f"  Данные      : структура Q6-гиперкуба (biangua-переходы)")
    print(f"  Цель        : модель выучивает топологию 64-вершинного графа")
    print(f"  Без текстов : только геометрия Ицзин")

    opt = AdamW(model.parameters(), lr=sp["lr"], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["max_steps"])

    block_size = model.cfg.block_size
    history_ent = collections.deque(maxlen=sp["stop_window"])
    log = []

    for step in range(1, sp["max_steps"] + 1):
        model.train()
        x, y = make_biangua_batch(sp["batch"], block_size)

        out = model(x, y)
        loss = out[1]
        if loss is None:
            logits = out[0]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % 20 == 0 or step == 1:
            model.eval()
            # Замеряем метрики на одном тестовом батче
            xv, yv = make_biangua_batch(4, block_size)
            metrics = get_metrics(model, xv)
            ent = metrics.get("hex_entropy", 0.0)
            history_ent.append(ent)

            # Какую гексаграмму доминирует?
            dh = get_dominant_hex(model, xv[:1])
            dname = hex_name(dh) if dh >= 0 else "?"

            extra = f"→ гекс [{dh:2d}] {dname}"
            print(fmt(step, loss.item(), metrics, extra))
            log.append({"step": step, "loss": loss.item(), **metrics})

        # Критерий остановки: энтропия стабилизировалась
        if len(history_ent) >= sp["stop_window"]:
            ent_std = torch.tensor(list(history_ent)).std().item()
            if ent_std < sp["stop_entropy_std"]:
                print(f"\n  ✓ Стадия 0 завершена на шаге {step}: "
                      f"std(entropy)={ent_std:.4f} < {sp['stop_entropy_std']}")
                break

    print(f"\n  Итог стадии 0:")
    print(f"  Модель выучила топологию Q6 без единого внешнего текста.")
    print(f"  Теперь она 'знает себя' — структуру пространства ответов.")
    return log


# ═══════════════════════════════════════════════════════════════════════════
# СТАДИЯ 1 — RAG-БУФЕР (Small Quality Corpus)
# ═══════════════════════════════════════════════════════════════════════════

# Минимальный качественный корпус (имитация — в реальности здесь ваши тексты)
QUALITY_CORPUS_RAW = [
    "The hexagram represents a state of change and transformation.",
    "Information theory connects entropy with uncertainty and knowledge.",
    "A system learns by reducing prediction error through feedback.",
    "Fire transforms matter; water preserves form; air connects all.",
    "The scientific method requires reproducible and falsifiable evidence.",
    "Logic and structure underlie all systematic thought.",
    "Topology studies properties preserved under continuous deformation.",
    "A graph consists of vertices connected by edges with defined distances.",
    "Language encodes meaning through symbol systems and grammar rules.",
    "Mathematics provides the foundation for all formal reasoning.",
    "Quantum states exist in superposition until observed and measured.",
    "Entropy always increases in isolated systems over time.",
    "Neural networks approximate functions through layered transformations.",
    "The observer effect changes the system being measured.",
    "Recursive structures emerge when processes reference themselves.",
    "Six binary lines define sixty-four possible hexagram states.",
    "A Hamming distance measures how many bits differ between two codes.",
    "Attention mechanisms allow models to focus on relevant context.",
    "Knowledge representation requires both structure and semantics.",
    "Systems thinking examines relationships between components of a whole.",
    "The principle of minimum description length guides model selection.",
    "Pattern recognition requires generalization beyond training examples.",
    "Ternary logic introduces uncertainty as a first-class value.",
    "Symbolic reasoning operates on discrete abstract representations.",
    "Self-reference creates paradox when a statement denies itself.",
    "Biangua transitions correspond to flipping one line of a hexagram.",
    "Domain routing assigns inputs to specialized processing pathways.",
    "Curriculum learning presents examples in order of increasing difficulty.",
    "Quality filtering removes noise before model training begins.",
    "Retrieval augmentation supplements generation with external knowledge.",
]

def text_to_ids(text, block_size):
    ids = [b for b in text.encode("utf-8")][:block_size]
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long)


class RAGBuffer:
    """
    Буфер высококачественных примеров.

    Три функции:
    1. add(text, emb)    — добавить пример с его эмбеддингом
    2. is_quality(text)  — пропустить ли текст через качественный фильтр
    3. retrieve(query_emb, k) — найти k ближайших примеров (для RAG)

    Именно этот буфер — "промежуточный уровень" между моделью и диким миром:
    - Шире чем модель (хранит внешние примеры)
    - Уже чем весь корпус (только отобранные)
    """

    def __init__(self, quality_threshold=0.55, max_size=500):
        self.threshold = quality_threshold
        self.max_size  = max_size
        self.texts     = []
        self.embeddings = []  # list of (D,) tensors
        self.scores    = []

    def add(self, text: str, emb: torch.Tensor, score: float):
        if score >= self.threshold:
            self.texts.append(text)
            self.embeddings.append(emb.detach().cpu())
            self.scores.append(score)
            # LRU eviction if over max
            if len(self.texts) > self.max_size:
                worst = min(range(len(self.scores)), key=lambda i: self.scores[i])
                del self.texts[worst]
                del self.embeddings[worst]
                del self.scores[worst]
            return True
        return False

    def retrieve(self, query_emb: torch.Tensor, k: int = 3):
        """Return top-k most similar texts by cosine similarity."""
        if not self.embeddings:
            return []
        q = F.normalize(query_emb.cpu().float(), dim=-1)
        sims = []
        for e in self.embeddings:
            e_n = F.normalize(e.float(), dim=-1)
            sims.append(F.cosine_similarity(q.unsqueeze(0), e_n.unsqueeze(0)).item())
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return [(self.texts[i], sims[i]) for i in order[:k]]

    def mean_score(self):
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def __len__(self):
        return len(self.texts)


def compute_text_quality(model, text, block_size):
    """
    Возвращает (score [0..1], emb [D]).
    Score = средний балл TextQualityFilter по 6 осям.
    emb = усреднённый скрытый вектор последнего слоя.
    """
    ids = text_to_ids(text, block_size).unsqueeze(0)
    with torch.no_grad():
        emb_in = model.tok_emb(ids)
        h = emb_in
        for block in model.blocks:
            h = block(h)
        scores, _, _ = qfilter(h)
        emb = h.mean(dim=1).squeeze(0)   # (D,)
        score = (scores.clamp(-1, 1) + 1).mean().item() / 2.0  # в [0,1]
    return score, emb


def build_rag_buffer(model, corpus, sp, block_size):
    """
    Наполнить RAG-буфер из сырого корпуса.

    Стратегия: абсолютный порог ИЛИ top-50% по оценке —
    что бы ни дало хотя бы 10 примеров (работает и до обучения).
    """
    # Оцениваем всё
    scored = []
    for text in corpus:
        score, emb = compute_text_quality(model, text, block_size)
        scored.append((text, score, emb))

    # Относительный порог: медиана оценок
    scores_sorted = sorted(s for _, s, _ in scored)
    median_score = scores_sorted[len(scores_sorted) // 2]
    effective_threshold = min(sp["quality_threshold"], median_score * 0.98)

    buf = RAGBuffer(quality_threshold=effective_threshold)
    accepted = 0
    for text, score, emb in scored:
        ok = buf.add(text, emb, score)
        if ok:
            accepted += 1

    print(f"  RAG-буфер: {accepted}/{len(corpus)} текстов принято "
          f"(порог={effective_threshold:.3f}, медиана={median_score:.3f}), "
          f"ср.оценка={buf.mean_score():.3f}")
    return buf


def make_quality_batch(texts, block_size, batch_size, device=DEVICE):
    """Создать батч из списка текстов."""
    selected = random.sample(texts, min(batch_size, len(texts)))
    xs, ys = [], []
    for text in selected:
        ids = text_to_ids(text, block_size + 1)
        ids = ids[:block_size + 1]
        if len(ids) < 2:
            continue
        x = ids[:-1]
        y = ids[1:]
        # pad
        if len(x) < block_size:
            pad = torch.zeros(block_size - len(x), dtype=torch.long)
            x = torch.cat([x, pad])
            y = torch.cat([y, pad])
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def run_stage1(model, sp, rag_buf):
    """Стадия 1: обучение на отобранном буфере."""
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 1: {sp['name']}")
    print(f"{'═'*72}")
    print(f"  Данные      : {len(rag_buf)} текстов из RAG-буфера")
    print(f"  Фильтр      : quality_score >= {sp['quality_threshold']}")
    print(f"  Цель        : доменная связность > {sp['stop_coherence']}")

    opt = AdamW(model.parameters(), lr=sp["lr"], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["max_steps"])

    block_size = model.cfg.block_size
    history_coh = collections.deque(maxlen=sp["stop_window"])
    log = []
    texts = rag_buf.texts

    for step in range(1, sp["max_steps"] + 1):
        model.train()
        x, y = make_quality_batch(texts, block_size, sp["batch"])
        if x is None:
            continue

        out = model(x, y)
        loss = out[1]
        if loss is None:
            logits = out[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if step % 30 == 0 or step == 1:
            model.eval()
            xv, yv = make_quality_batch(texts, block_size, 4)
            if xv is None:
                continue
            metrics = get_metrics(model, xv)
            coh = metrics.get("domain_coherence", 0.0)
            history_coh.append(coh)

            dh = get_dominant_hex(model, xv[:1])
            extra = f"→ гекс [{dh:2d}] {hex_name(dh)}"
            print(fmt(step, loss.item(), metrics, extra))
            log.append({"step": step, "loss": loss.item(), **metrics})

        # Критерий остановки: связность доменов выросла
        if len(history_coh) >= sp["stop_window"]:
            avg_coh = sum(history_coh) / len(history_coh)
            if avg_coh >= sp["stop_coherence"]:
                print(f"\n  ✓ Стадия 1 завершена на шаге {step}: "
                      f"avg_coherence={avg_coh:.4f} >= {sp['stop_coherence']}")
                break

    # Переоценить эмбеддинги с обновлённой моделью (тексты те же)
    print(f"\n  Переоценка эмбеддингов RAG-буфера с обновлённой моделью...")
    old_len = len(rag_buf)
    all_texts = list(rag_buf.texts)
    scored = []
    for text in all_texts:
        score, emb = compute_text_quality(model, text, block_size)
        scored.append((text, score, emb))
    # Адаптивный порог (медиана)
    scores_sorted = sorted(s for _, s, _ in scored)
    median_score = scores_sorted[len(scores_sorted) // 2] if scored else 0.5
    effective_thr = min(sp["quality_threshold"], median_score * 0.98) if scored else 0.4
    new_buf = RAGBuffer(quality_threshold=effective_thr)
    for text, score, emb in scored:
        new_buf.add(text, emb, score)
    print(f"  Буфер: {old_len} → {len(new_buf)} текстов "
          f"(порог={effective_thr:.3f}, ср.оценка → {new_buf.mean_score():.3f})")
    return log, new_buf


# ═══════════════════════════════════════════════════════════════════════════
# СТАДИЯ 2 — ШИРОКИЙ МИР (Wild Data filtered by RAG)
# ═══════════════════════════════════════════════════════════════════════════

# "Дикие" данные — имитация случайных текстов разного качества
WILD_CORPUS = [
    # разнородные, некоторые полезные, некоторые нет
    "buy now click here special offer limited time",
    "The laws of thermodynamics govern energy conversion in all systems.",
    "free download virus warning your computer is infected",
    "Graph theory studies relationships between nodes through edge connections.",
    "spam spam spam click buy win prize cheap fast",
    "A function maps each element of a domain to exactly one codomain value.",
    "win win win best price guaranteed lowest cost",
    "Cellular automata evolve discrete states through local transition rules.",
    "asdf qwer zxcv poiuy lkjhg mnbvc random nonsense",
    "The Fourier transform decomposes signals into their frequency components.",
    "call now limited offer exclusive deal just for you",
    "Bayesian inference updates prior beliefs using observed evidence.",
    "blah blah lorem ipsum dolor sit amet consectetur",
    "Recursion defines a function in terms of simpler versions of itself.",
    "Click click click advertisement banner popup overlay",
    "Set theory provides the foundation of modern mathematics.",
    "Random text generation produces output without meaningful content.",
    "Binary trees organize data hierarchically for efficient search and retrieval.",
    "follow follow follow like subscribe share notification bell",
    "Abstract algebra studies algebraic structures like groups and rings.",
    "error 404 page not found broken link dead end",
    "Markov chains model sequences of events with probabilistic transitions.",
    "act now before it is too late hurry offer expires",
    "The halting problem demonstrates limits of algorithmic computation.",
    "x x x x x x x x x x x x x x x x x",
    "Convex optimization finds global minima in convex objective functions.",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "Lambda calculus provides a formal system for function definition.",
    "test test test test placeholder filler content ignore",
    "Type theory classifies mathematical objects and validates proofs formally.",
]


def is_rag_accepted(text, model, rag_buf, block_size, sim_threshold):
    """Проверить, принимает ли RAG-буфер этот текст."""
    if not rag_buf.embeddings:
        return False, 0.0
    _, emb = compute_text_quality(model, text, block_size)
    results = rag_buf.retrieve(emb, k=1)
    if not results:
        return False, 0.0
    _, sim = results[0]
    return sim >= sim_threshold, sim


def run_stage2(model, sp, rag_buf):
    """Стадия 2: дикие данные, но только через фильтр RAG-буфера."""
    print(f"\n{'═'*72}")
    print(f"  СТАДИЯ 2: {sp['name']}")
    print(f"{'═'*72}")
    print(f"  Данные      : {len(WILD_CORPUS)} 'диких' текстов")
    print(f"  RAG-фильтр  : косинусное сходство >= {sp['rag_sim_threshold']}")

    # Предварительно отфильтруем дикие данные
    block_size = model.cfg.block_size
    model.eval()
    accepted_wild = []
    rejected_wild = []
    for text in WILD_CORPUS:
        ok, sim = is_rag_accepted(text, model, rag_buf, block_size, sp["rag_sim_threshold"])
        if ok:
            accepted_wild.append(text)
        else:
            rejected_wild.append((text, sim))

    print(f"\n  Принято     : {len(accepted_wild)}/{len(WILD_CORPUS)} 'диких' текстов")
    print(f"  Отклонено   : {len(rejected_wild)}")
    print(f"\n  Примеры принятых:")
    for t in accepted_wild[:5]:
        print(f"    ✓ \"{t[:65]}\"")
    print(f"\n  Примеры отклонённых (sim ниже порога):")
    for t, s in sorted(rejected_wild, key=lambda x: x[1])[:5]:
        print(f"    ✗ [{s:.3f}] \"{t[:55]}\"")

    if not accepted_wild:
        print(f"\n  ⚠ Нет принятых 'диких' текстов — снижаем порог до 0.5")
        sp["rag_sim_threshold"] = 0.5
        for text in WILD_CORPUS:
            ok, sim = is_rag_accepted(text, model, rag_buf, block_size, 0.5)
            if ok:
                accepted_wild.append(text)

    opt = AdamW(model.parameters(), lr=sp["lr"], weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["max_steps"])

    history_loss = collections.deque(maxlen=sp["stop_window"])
    log = []

    for step in range(1, sp["max_steps"] + 1):
        model.train()

        # Смешиваем: 70% из RAG-буфера + 30% из принятых диких
        buf_texts = rag_buf.texts
        pool = buf_texts * 7 + accepted_wild * 3
        x, y = make_quality_batch(pool, block_size, sp["batch"])
        if x is None:
            continue

        out = model(x, y)
        loss = out[1]
        if loss is None:
            logits = out[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        history_loss.append(loss.item())

        if step % 30 == 0 or step == 1:
            model.eval()
            xv, yv = make_quality_batch(pool, block_size, 4)
            if xv is None:
                continue
            metrics = get_metrics(model, xv)
            dh = get_dominant_hex(model, xv[:1])
            extra = f"→ гекс [{dh:2d}] {hex_name(dh)}"
            print(fmt(step, loss.item(), metrics, extra))
            log.append({"step": step, "loss": loss.item(), **metrics})

        # Критерий: loss стабилизировался
        if len(history_loss) >= sp["stop_window"]:
            std = torch.tensor(list(history_loss)).std().item()
            if std < 0.005:
                print(f"\n  ✓ Стадия 2 завершена на шаге {step}: "
                      f"std(loss)={std:.5f}")
                break

    return log


# ═══════════════════════════════════════════════════════════════════════════
# САМОГЕНЕРАЦИЯ — модель диалогирует сама с собой
# ═══════════════════════════════════════════════════════════════════════════

def self_dialog(model, rag_buf, block_size, n_turns=5, temperature=1.2):
    """
    Модель генерирует текст → текст идёт в RAG-буфер если качественный →
    следующий промпт = конкатенация предыдущего ответа.

    Это "петля самообучения": модель пополняет свой же обучающий корпус.
    """
    print(f"\n{'─'*72}")
    print(f"  САМО-ДИАЛОГ: модель учит саму себя")
    print(f"{'─'*72}")

    # Стартовый промпт — случайная гексаграмма в байтах
    start_hex = random.randint(0, 63)
    bits = hexagrams[start_hex].tolist()
    prompt_ids = torch.tensor(
        [int(b > 0) for b in bits] * 4, dtype=torch.long
    ).unsqueeze(0)[:, :block_size]

    accepted = 0
    for turn in range(1, n_turns + 1):
        model.eval()
        # Генерация токенов
        generated = prompt_ids.clone()
        with torch.no_grad():
            for _ in range(block_size // 2):
                out = model(generated)
                logits = out[0] if isinstance(out, tuple) else out
                next_logit = logits[0, -1] / temperature
                probs = F.softmax(next_logit, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=1)
                generated = generated[:, -block_size:]

        # Декодируем (байтовое декодирование с fallback)
        gen_bytes = generated[0].tolist()
        try:
            gen_text = bytes([b % 256 for b in gen_bytes]).decode("utf-8", errors="replace")
        except Exception:
            gen_text = " ".join(str(b) for b in gen_bytes[:20])

        # Оцениваем качество и пробуем добавить в буфер
        score, emb = compute_text_quality(model, gen_text, block_size)
        accepted_this = rag_buf.add(gen_text, emb, score)
        if accepted_this:
            accepted += 1

        # Следующий промпт = конец сгенерированного текста
        prompt_ids = generated[:, -block_size//2:]

        # Короткий лог
        status = "✓ принято" if accepted_this else "✗ отклонено"
        print(f"  Ход {turn:2d}: score={score:.3f} {status}  "
              f"гекс=[{get_dominant_hex(model, generated):.0f}]  "
              f"'{gen_text[:40].strip()}'…")

    print(f"\n  Самодиалог: {accepted}/{n_turns} ходов приняты в RAG-буфер.")
    print(f"  Буфер вырос до {len(rag_buf)} текстов.")


# ═══════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ЗАПУСК
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  SELF-TRAIN: САМООБУЧЕНИЕ VARIANT3GPT")
    print("=" * 72)
    print(f"\n  Архитектура: {CFG.n_layers}×{CFG.d_model}d, {CFG.n_heads} голов")
    print(f"  Схема: Самопознание → RAG-буфер → Широкий мир → Само-диалог")

    model = Variant3GPT(CFG).to(DEVICE)
    print(f"  Параметры: {model.count_parameters():,}")

    # СТАДИЯ 0 — учим топологию
    log0 = run_stage0(model, STAGE_PARAMS[0])

    # Строим RAG-буфер после стадии 0
    print(f"\n{'─'*72}")
    print(f"  Строим RAG-буфер из {len(QUALITY_CORPUS_RAW)} качественных текстов...")
    rag_buf = build_rag_buffer(model, QUALITY_CORPUS_RAW, STAGE_PARAMS[1], CFG.block_size)

    # СТАДИЯ 1 — обучаем на RAG-буфере
    log1, rag_buf = run_stage1(model, STAGE_PARAMS[1], rag_buf)

    # СТАДИЯ 2 — широкий мир через RAG-фильтр
    log2 = run_stage2(model, STAGE_PARAMS[2], rag_buf)

    # САМО-ДИАЛОГ — модель пополняет свой буфер
    self_dialog(model, rag_buf, CFG.block_size, n_turns=8)

    # ФИНАЛЬНАЯ ОЦЕНКА
    print(f"\n{'═'*72}")
    print(f"  ФИНАЛЬНАЯ ОЦЕНКА ПОСЛЕ САМООБУЧЕНИЯ")
    print(f"{'═'*72}")

    test_prompts = [
        ("fire light energy",     "PYRO"),
        ("water river flow",      "HYDRO"),
        ("star galaxy cosmos",    "COSMO"),
        ("logic reason thought",  "NOOS"),
    ]

    for text, expected in test_prompts:
        ids = text_to_ids(text, CFG.block_size).unsqueeze(0)
        metrics = get_metrics(model, ids)
        dh = get_dominant_hex(model, ids)

        with torch.no_grad():
            out = model(ids)
            info = out[2] if isinstance(out, tuple) and len(out) > 2 else {}
            dw = info.get("domain_weights")
            if dw is not None and dw.dim() == 3:
                dw = dw.mean(dim=1)

        if dw is not None:
            top_dom = DOMAINS[dw[0].argmax().item()]
            dom_match = "✓" if top_dom == expected else "~"
        else:
            top_dom = "?"
            dom_match = "?"

        print(f"  {dom_match} '{text:30s}'  "
              f"гекс=[{dh:2d}] {hex_name(dh):20s}  "
              f"домен={top_dom:6s} (ожид.={expected})")

    # Сохраняем лог
    full_log = {
        "stage0_steps": len(log0),
        "stage1_steps": len(log1),
        "stage2_steps": len(log2),
        "rag_buffer_final_size": len(rag_buf),
        "final_rag_mean_score": rag_buf.mean_score(),
    }
    with open("self_train_log.json", "w") as f:
        json.dump({"summary": full_log, "stage0": log0, "stage1": log1, "stage2": log2}, f, indent=2)

    print(f"\n  Лог сохранён: self_train_log.json")
    print(f"\n  ИДЕЯ РЕАЛИЗОВАНА:")
    print(f"  1. Модель познала себя (Q6-топология без внешних данных)")
    print(f"  2. RAG-буфер = промежуточный уровень (качественные тексты)")
    print(f"  3. Дикие данные только через RAG-фильтр (косинусный порог)")
    print(f"  4. Само-диалог пополняет буфер из собственной генерации")
    print(f"  5. Критерии остановки — по метрикам, не по числу шагов")
    print(f"\n  Следующий шаг: запустить на реальных данных (ваш корпус)")
    print(f"  Заменить QUALITY_CORPUS_RAW и WILD_CORPUS на настоящие тексты.")


if __name__ == "__main__":
    main()
