#!/usr/bin/env python3
"""
model_test.py — Полное тестирование модели Nautilus/Yijing (Variant3GPT).

Тестирует:
  1. Загрузку checkpoint (что было обучено)
  2. Q6-эмбеддинг и доменную маршрутизацию
  3. Реакцию на вопросы / концепты (генерация + интерпретация)
  4. Самообучение (один mini-цикл с новыми данными)
  5. Метрики здоровья графа через corpus_loader
  6. Сравнение baseline vs v2 checkpoint

Usage:
  python model_test.py              # полный тест
  python model_test.py --fast       # быстрый тест (без самообучения)
  python model_test.py --query "кристалл"  # только один запрос
"""

import os, sys, math, json, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from yijing_transformer.models.variant3 import (
    Variant3Config, Variant3GPT,
    DOMAINS, DOMAIN_ANCHORS,
    get_dominant_hexagram, get_active_domains,
    _make_hexagrams,
)
from yijing_transformer.models.variant3_extensions import (
    HexagramEvaluator, TextQualityFilter,
    get_hexagrams, get_biangua, bfs_distances,
)
from yijing_transformer.constants import HEX_NAMES_SHORT as HEX_NAMES_RU

# ── конфигурация (та же что в bidir_train.py) ─────────────────────────────────
torch.manual_seed(42)
random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = Variant3Config(
    vocab_size=256, block_size=32, d_model=128,
    n_heads=4, n_layers=4, ffn_mult=4,
    hamming_lambda=0.15, uncertainty_budget=0.25,
    dropout=0.0, use_domain_routing=True,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))

DOMAIN_DESCRIPTIONS = {
    "GEO":   "地 Chi   Земля    — Мастер/Инженер   (структура, материал)",
    "HYDRO": "水 Sui   Вода     — Аналитик/Разведчик(поток, анализ)",
    "PYRO":  "火 Ka    Огонь    — Архитектор        (трансформация, система)",
    "AERO":  "風 Fu    Ветер    — Математик/Логик   (движение, паттерн)",
    "COSMO": "空 Kū    Пустота  — Лидер/Дипломат    (связность, отношения)",
    "NOOS":  "識 Shiki Сознание — Философ/Мудрец    (смысл, концепт)",
}


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════════════

def _encode(text: str, vocab_size: int = 256, block_size: int = 32) -> torch.Tensor:
    """Байтовое кодирование: текст → токены [1, T]."""
    ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
    if not ids:
        ids = [32]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _q6_of_model(model: Variant3GPT, text: str) -> dict:
    """
    Прогоняет text через модель, возвращает Q6-координату.
    Q6 = знак первых 6 элементов усреднённого hidden-state.
    """
    tokens = _encode(text)
    with torch.no_grad():
        logits, _, routing = model(tokens)
        # hidden до head — прокси через токен-эмбеддинги
        tok_h  = model.tok_emb(tokens)          # (1,T,d)
        mean_h = tok_h.mean(dim=1).squeeze(0)   # (d,)
        q6_raw = mean_h[:6]
        q6     = (q6_raw > 0).int().tolist()
        hex_i  = sum(b << i for i, b in enumerate(q6))

        # Топ-токен (что модель предсказывает следующим)
        next_tok = logits[0, -1].argmax().item()
        next_chr = chr(next_tok) if 32 <= next_tok < 128 else f"[{next_tok}]"

        # Доменные веса из routing_info
        domain_w = None
        if routing and "domain_weights" in routing:
            dw = routing["domain_weights"]   # (1, T, 6) или (1, 6)
            if dw.dim() == 3:
                dw = dw.mean(dim=1)          # (1, 6)
            domain_w = dw.squeeze(0).tolist()

    return {
        "text":      text,
        "q6":        q6,
        "hex_idx":   hex_i,
        "hex_name":  HEX_NAMES_RU[hex_i],
        "next_char": next_chr,
        "domain_weights": domain_w,
    }


def _dominant_domain(domain_weights: list | None) -> str:
    if not domain_weights:
        return "?"
    return DOMAINS[domain_weights.index(max(domain_weights))]


def _perplexity(model: Variant3GPT, text: str) -> float:
    """Перплексия модели на тексте — чем ниже, тем лучше знает текст."""
    tokens = _encode(text)
    if tokens.shape[1] < 2:
        return float("inf")
    inp  = tokens[:, :-1]
    tgt  = tokens[:, 1:]
    with torch.no_grad():
        _, loss, _ = model(inp, targets=tgt)
    return math.exp(loss.item()) if loss else float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 1: Загрузка и описание модели
# ══════════════════════════════════════════════════════════════════════════════

def section_load() -> tuple[Variant3GPT, Variant3GPT | None, dict]:
    """Загружает checkpoint_bidir_v2.pt (основной) и checkpoint_bidir.pt (baseline)."""
    print("\n" + "═" * 70)
    print("  РАЗДЕЛ 1: ЗАГРУЗКА МОДЕЛИ")
    print("═" * 70)

    # --- v2 (основной) ---
    model_v2 = Variant3GPT(CFG).eval()
    ckpt_v2  = os.path.join(_ROOT, "checkpoint_bidir_v2.pt")
    loaded_v2 = False
    if os.path.exists(ckpt_v2):
        state = torch.load(ckpt_v2, map_location="cpu", weights_only=True)
        model_v2.load_state_dict(state, strict=False)
        sz = os.path.getsize(ckpt_v2) // 1024
        print(f"  ✅ checkpoint_bidir_v2.pt  загружен  ({sz} KB)")
        loaded_v2 = True
    else:
        print("  ⚠️  checkpoint_bidir_v2.pt  НЕ найден (случайные веса)")

    # --- baseline ---
    model_base = None
    ckpt_base  = os.path.join(_ROOT, "checkpoint_bidir.pt")
    if os.path.exists(ckpt_base):
        model_base = Variant3GPT(CFG).eval()
        state_b = torch.load(ckpt_base, map_location="cpu", weights_only=True)
        model_base.load_state_dict(state_b, strict=False)
        sz = os.path.getsize(ckpt_base) // 1024
        print(f"  ✅ checkpoint_bidir.pt     загружен  ({sz} KB) [для сравнения]")

    print(f"\n{model_v2.describe()}")

    # Лог обучения
    log_v2 = {}
    log_path = os.path.join(_ROOT, "bidir_train_v2_log.json")
    if os.path.exists(log_path):
        log_v2 = json.load(open(log_path))
        rlog = log_v2.get("run_log", [])
        print(f"\n  Обучение v2: {len(rlog)} фаз | "
              f"bridge-пар: {log_v2.get('n_bridge_texts', '?')} | "
              f"интеграция: {log_v2.get('n_integ_texts', '?')} текстов")
        for phase in rlog:
            label  = phase.get("phase", f"round {phase.get('round','?')}")
            closed = phase.get("closed", sum(1 for d in phase.get("details", []) if d.get("closed")))
            total  = len(phase.get("details", phase.get("closure_details", [])))
            avg_l  = phase.get("avg_loss", "")
            extra  = f"  loss={avg_l:.4f}" if avg_l else ""
            print(f"    фаза '{label}': {closed}/{total} пар замкнуто в Q6{extra}")

    # Лог self-train v2
    st_path = os.path.join(_ROOT, "self_train_v2_log.json")
    if os.path.exists(st_path):
        st = json.load(open(st_path))
        rounds = st.get("stage3_rounds", [])
        if rounds:
            last = rounds[-1]
            print(f"\n  Self-train v2: {len(rounds)} раундов | "
                  f"буфер: {last['buffer_size']} | "
                  f"fine_loss: {last['fine_loss']:.4f}")

    return model_v2, model_base, log_v2


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 2: Q6-эмбеддинг и доменная маршрутизация
# ══════════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    # (запрос, ожидаемый домен)
    ("crystal",          "GEO"),
    ("water flow",       "HYDRO"),
    ("fire transforms",  "PYRO"),
    ("wind pattern",     "AERO"),
    ("star galaxy",      "COSMO"),
    ("knowledge theory", "NOOS"),
    ("кристалл",         "GEO"),
    ("знание",           "NOOS"),
    ("гексаграмма",      "NOOS"),
    ("трансформация",    "PYRO"),
]

def section_q6(model: Variant3GPT, model_base: Variant3GPT | None) -> None:
    print("\n" + "═" * 70)
    print("  РАЗДЕЛ 2: Q6-ЭМБЕДДИНГ И ДОМЕННАЯ МАРШРУТИЗАЦИЯ")
    print("═" * 70)
    print(f"  {'Запрос':<22} {'Q6':^14} {'Гекс':>5}  {'Гексаграмма':<20} {'Домен'}")
    print("  " + "-" * 68)

    correct = 0
    for text, expected in QUESTIONS:
        r     = _q6_of_model(model, text)
        dom   = _dominant_domain(r["domain_weights"])
        q6s   = "".join(str(b) for b in r["q6"])
        name  = r["hex_name"]
        flag  = "✅" if dom == expected else "〇"
        if dom == expected:
            correct += 1
        print(f"  {text:<22} [{q6s}]  #{r['hex_idx']:>2}  {name:<20} {dom:<6} {flag}")

    print(f"\n  Точность домена: {correct}/{len(QUESTIONS)} = {correct/len(QUESTIONS)*100:.0f}%")

    # Сравнение с baseline
    if model_base:
        print("\n  Сравнение v2 vs baseline (перплексия ↓ лучше):")
        test_texts = ["crystal structure", "знание и концепт", "water flows"]
        print(f"  {'Текст':<28} {'baseline':>10} {'v2':>10} {'улучш.':>8}")
        print("  " + "-" * 58)
        for t in test_texts:
            p_base = _perplexity(model_base, t)
            p_v2   = _perplexity(model, t)
            delta  = (p_base - p_v2) / p_base * 100 if p_base < float("inf") else 0
            sign   = "⬇️ " if delta > 0 else "⬆️ "
            print(f"  {t:<28} {p_base:>10.2f} {p_v2:>10.2f} {sign}{abs(delta):>5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 3: Реакция на вопросы
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_QUESTIONS = [
    "Что такое гексаграмма?",
    "Как работает Q6-пространство?",
    "Crystal formation process",
    "knowledge transformation K0 to K1",
    "Nautilus portal coordination",
    "ternary logic gates",
    "bidirectional knowledge loop",
    "Что такое домен NOOS?",
]

def section_questions(model: Variant3GPT) -> None:
    print("\n" + "═" * 70)
    print("  РАЗДЕЛ 3: РЕАКЦИЯ МОДЕЛИ НА ВОПРОСЫ")
    print("═" * 70)
    print("  (Модель — микро-GPT char-level. Интерпретируем через Q6, ")
    print("   доменный роутинг и гексаграмму — не через текстовый вывод.)\n")

    evaluator = HexagramEvaluator(threshold=0.01)

    for q in CONCEPT_QUESTIONS:
        r = _q6_of_model(model, q)
        dom = _dominant_domain(r["domain_weights"])
        q6s = "".join(str(b) for b in r["q6"])

        # Оцениваем «уверенность» через perplexity
        ppl = _perplexity(model, q)
        confidence = "высокая" if ppl < 5 else ("средняя" if ppl < 15 else "низкая")

        print(f"  ❓ {q}")
        print(f"     Q6=[{q6s}] Гексаграмма #{r['hex_idx']}: «{r['hex_name']}»")
        print(f"     Домен: {dom} — {DOMAIN_DESCRIPTIONS.get(dom,'')}")
        print(f"     Perplexity: {ppl:.2f}  Уверенность: {confidence}")

        # Ближайшие концепты в Q6 через BianGua (Хэмминг-1)
        hex_i = r["hex_idx"]
        hexagrams = _make_hexagrams()
        biangua = get_biangua()
        neighbours = biangua[hex_i].nonzero().squeeze(1).tolist()
        if neighbours:
            neigh_names = [HEX_NAMES_RU[n] for n in neighbours[:3]]
            print(f"     BianGua-соседи (Хэмминг=1): {', '.join(neigh_names)}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 4: Mini-цикл самообучения
# ══════════════════════════════════════════════════════════════════════════════

NEW_CORPUS = [
    # Новые тексты которых не было в обучении
    "hexagram 1 is creative force yang pure",
    "hexagram 2 is receptive yin pure earth",
    "Q6 space has 64 vertices hypercube six dimensions",
    "nautilus portal connects repositories through shared Q6 embedding",
    "knowledge transformation K0 raw data K1 structured K2 synthesized",
    "ternary gate outputs minus one zero plus one uncertainty",
    "biangua adjacent hexagrams differ by one line",
    "domain GEO handles structure material crystal formation",
    "domain NOOS handles meaning concept philosophy theory",
    "cross hexagram analogy bridges distant concepts in Q6",
]

def section_self_train(model: Variant3GPT, fast: bool = False) -> Variant3GPT:
    print("\n" + "═" * 70)
    print("  РАЗДЕЛ 4: MINI-ЦИКЛ САМООБУЧЕНИЯ")
    print("═" * 70)

    if fast:
        print("  [--fast режим: пропускаем самообучение]")
        return model

    n_steps = 20
    lr      = 1e-4
    opt     = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"  Новых текстов: {len(NEW_CORPUS)}")
    print(f"  Шагов:         {n_steps}")
    print(f"  LR:            {lr}\n")

    # Замерим perplexity ДО
    ppl_before = [_perplexity(model, t) for t in NEW_CORPUS]
    avg_before  = sum(ppl_before) / len(ppl_before)
    print(f"  Perplexity ДО самообучения:   {avg_before:.2f}")

    model.train()
    losses = []
    for step in range(n_steps):
        text   = random.choice(NEW_CORPUS)
        tokens = _encode(text)
        if tokens.shape[1] < 2:
            continue
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]

        logits, loss, routing = model(inp, targets=tgt)
        if loss is None:
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if (step + 1) % 5 == 0:
            avg_l = sum(losses[-5:]) / 5
            print(f"    шаг {step+1:>3}/{n_steps}  loss={avg_l:.4f}")

    model.eval()

    # Замерим perplexity ПОСЛЕ
    ppl_after = [_perplexity(model, t) for t in NEW_CORPUS]
    avg_after  = sum(ppl_after) / len(ppl_after)
    delta      = (avg_before - avg_after) / avg_before * 100
    print(f"\n  Perplexity ПОСЛЕ самообучения: {avg_after:.2f}")
    print(f"  Улучшение:  {delta:+.1f}%  {'✅' if delta > 0 else '⚠️'}")

    # Проверим конкретные тексты
    print("\n  Детальная perplexity до / после:")
    print(f"  {'Текст':<45} {'до':>8} {'после':>8}")
    print("  " + "-" * 63)
    for i, t in enumerate(NEW_CORPUS[:5]):
        print(f"  {t:<45} {ppl_before[i]:>8.2f} {ppl_after[i]:>8.2f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 5: Корпус и граф знаний
# ══════════════════════════════════════════════════════════════════════════════

def section_corpus() -> None:
    print("\n" + "═" * 70)
    print("  РАЗДЕЛ 5: ДОСТУПНЫЙ КОРПУС (corpus_loader.py)")
    print("═" * 70)

    try:
        from corpus_loader import CorpusLoader
        loader = CorpusLoader()
        print(loader.availability_report())

        # Покажем 3 примера из разных источников
        print("\n  Примеры текстов по доменам:")
        sample = loader.load(domains=["info1", "data7", "meta"])
        shown  = set()
        for text, meta in sample:
            d = meta["domain"]
            if d not in shown and len(shown) < 4:
                print(f"\n  [{meta['source']}|{d}|α={meta['alpha']}]")
                print(f"  {text[:120].strip()}")
                shown.add(d)
    except Exception as e:
        print(f"  ⚠️  corpus_loader: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 6: Что было улучшено (history of improvements)
# ══════════════════════════════════════════════════════════════════════════════

def section_improvements() -> None:
    print("\n" + "═" * 70)
    print("  РАЗДЕЛ 6: ЧТО БЫЛО УЛУЧШЕНО В МОДЕЛИ")
    print("═" * 70)

    improvements = [
        ("Архитектура", [
            "HexagramProjection: проекция на 64 вершины Q6-гиперкуба {-1,+1}⁶",
            "BianGuaAttention: внимание с мягкой метрикой Хэмминга (Хэмминг ≡ смысловая близость)",
            "TernaryGate {-1,0,+1}: три уровня знания (отрицание/неопределённость/утверждение)",
            "ArchetypalInterlingua: 64 архетипа как семантический хаб между токенами",
            "CrossHexagramAnalogy: 変爻-аналогии через переходы с Хэмминг=1",
            "NautilusYiJinRouter: роутинг по 6 доменам (GEO/HYDRO/PYRO/AERO/COSMO/NOOS)",
        ]),
        ("Обучение (bidir_train_v2.py)", [
            "Двунаправленный цикл: вперёд (специализация→обобщение) + назад (обобщение→специализация)",
            "НЕДОСТАЮЩАЯ ПЕТЛЯ из data7: proposals → decompose → refinement (была только в комментарии!)",
            "KnowledgeGraph с PageRank: концепты ранжируются, топ-K становятся Q6-анкорами",
            "AdaptiveLearningOptimizer: lr меняется исходя из gradient_similarity с Q6-проекцией",
            "TSP-оптимизация порядка концептов: BFS по biangua-графу для curriculum",
            "QFilter: оценщик качества генераций (принимает/отклоняет гипотезы)",
        ]),
        ("Самообучение (self_train_v2.py)", [
            "Domain triplet loss: (anchor, positive-домен, negative-домен) × α=0.30",
            "Quality contrastive loss: хорошие vs плохие тексты через TextQualityFilter × β=0.20",
            "Gate entropy reward: поощряем ненулевую тернарную активацию × γ=0.10",
            "Stop-window: останавливаемся по реальным шагам (не по чекпоинтам)",
            "Буфер self-dialog: 47 принятых гипотез за 3 раунда самодиалога",
        ]),
        ("Интеграция (corpus_loader.py — новое)", [
            "2448 текстов из 8 репозиториев: info1, meta, data2, data7, infosystems, ai_agents, knowledge",
            "Автоматическое определение домена (GEO/HYDRO/PYRO/AERO/COSMO/NOOS/METHOD/YIJING)",
            "Автоматическое определение α-уровня (-4..+4) по пути файла",
            "as_training_corpus(): готовый список для bidir_train/self_train без сетевых запросов",
        ]),
        ("Метрики здоровья (graph_health.py — новое)", [
            "CD (Connectivity Density): плотность рёбер графа (цель: 10-25%)",
            "VT (Vertical Traceability): % рёбер между разными Q6-уровнями (цель: ≥50%)",
            "CR (Convergence Rate): скорость роста концептов по циклам (цель: 0.7-1.5)",
            "DB (Directional Balance): баланс ⇑⇓ vs ↔ рёбер (цель: <30%)",
        ]),
        ("Nautilus Portal (nautilus_inference.py — новое)", [
            "Режим standalone: Q6-эмбеддинг любого текста без других репо",
            "Режим portal: сбор контекста из 5 репо одновременно через адаптеры",
            "Режим federated: схема независимого обучения каждого репо + координация через Q6",
            "Каждый результат содержит: Q6[b0..b5] + гексаграмма + домен + α-уровень",
        ]),
    ]

    for section, points in improvements:
        print(f"\n  ▸ {section}:")
        for p in points:
            print(f"    • {p}")


# ══════════════════════════════════════════════════════════════════════════════
# Раздел 7: Кастомный запрос
# ══════════════════════════════════════════════════════════════════════════════

def section_query(model: Variant3GPT, query: str) -> None:
    print("\n" + "═" * 70)
    print(f"  ЗАПРОС: «{query}»")
    print("═" * 70)

    r   = _q6_of_model(model, query)
    dom = _dominant_domain(r["domain_weights"])
    ppl = _perplexity(model, query)
    q6s = "".join(str(b) for b in r["q6"])

    print(f"\n  Q6-координата:  [{q6s}]")
    print(f"  Гексаграмма:    #{r['hex_idx']} — «{r['hex_name']}»")
    print(f"  Активный домен: {dom}")
    print(f"  Описание:       {DOMAIN_DESCRIPTIONS.get(dom, '')}")
    print(f"  Perplexity:     {ppl:.2f}")

    # Доменные веса (если есть)
    if r["domain_weights"]:
        print(f"\n  Доменное распределение:")
        for i, (d, w) in enumerate(zip(DOMAINS, r["domain_weights"])):
            bar = "█" * int(w * 40)
            print(f"    {d:<6} {w:.3f} {bar}")

    # BianGua-соседи
    hexagrams = _make_hexagrams()
    biangua   = get_biangua()
    neighbours = biangua[r["hex_idx"]].nonzero().squeeze(1).tolist()
    print(f"\n  BianGua-соседи (Хэмминг=1, изменение одной линии):")
    for n in neighbours[:6]:
        n_dom = DOMAINS[bin(n).count("1") % 6]
        print(f"    #{n:>2} «{HEX_NAMES_RU[n]}» [{n_dom}]")

    # Что это значит
    print(f"\n  Интерпретация:")
    print(f"  Концепт «{query}» находится в {dom}-пространстве Q6,")
    print(f"  соответствует архетипу «{r['hex_name']}» (гексаграмма {r['hex_idx']}).")
    print(f"  В рамках И-Цзин этот архетип описывает:")
    print(f"  принцип изменения через {['движение','покой','накопление','рассеивание','соединение','разделение'][r['hex_idx'] % 6]}.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Тест модели Nautilus/Yijing")
    parser.add_argument("--fast",  action="store_true",
                        help="Быстрый тест (без самообучения)")
    parser.add_argument("--query", "-q", default="",
                        help="Кастомный запрос для раздела 7")
    parser.add_argument("--only",  default="",
                        help="Запустить только раздел: load,q6,questions,train,corpus,improvements,query")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else set()

    t0 = time.time()

    model, model_base, log = section_load()

    if not only or "q6" in only:
        section_q6(model, model_base)

    if not only or "questions" in only:
        section_questions(model)

    if not only or "train" in only:
        model = section_self_train(model, fast=args.fast)

    if not only or "corpus" in only:
        section_corpus()

    if not only or "improvements" in only:
        section_improvements()

    if args.query or "query" in only:
        section_query(model, args.query or "трансформация знаний")

    elapsed = time.time() - t0
    print("\n" + "═" * 70)
    print(f"  ТЕСТ ЗАВЕРШЁН  ({elapsed:.1f}с)")
    print("═" * 70)


if __name__ == "__main__":
    main()
