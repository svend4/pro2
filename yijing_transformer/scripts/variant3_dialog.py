#!/usr/bin/env python3
"""
Variant 3 — Интерактивный диалог с архетипической моделью.

Режимы:
  1. INSPECT  — введи текст → узнай архетип, домены, 変爻-путь
  2. COMPARE  — сравни два текста в Q6-пространстве
  3. HEXAGRAM — исследуй гексаграмму (соседи, путь до другой, семантика)
  4. TRAIN    — быстрый цикл обучения на собственном тексте
  5. PROBE    — зонд: какой слой что "думает" о токене
  6. IDEAS    — список идей для дальнейшего развития

Запуск:
    python yijing_transformer/scripts/variant3_dialog.py
    python yijing_transformer/scripts/variant3_dialog.py --mode inspect --text "Hello"
"""

import sys
import os
import math
import argparse
import textwrap
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from yijing_transformer.models.variant3 import (
    _make_hexagrams,
    _make_biangua_matrix,
    hamming_distance_soft,
    HexagramProjection,
    NautilusYiJinRouter,
    Variant3Config,
    Variant3GPT,
    DOMAINS,
    DOMAIN_ANCHORS,
    get_dominant_hexagram,
    get_active_domains,
    biangua_path,
)


# ─── Константы ────────────────────────────────────────────────────────────────

HEXAGRAM_SYMBOLS = [
    "☷", "☶", "☵", "☴", "☳", "☲", "☱", "☰",
]

HEXAGRAM_NAMES_SHORT = [
    "坤Kūn", "艮Gèn", "坎Kǎn", "巽Xùn",
    "震Zhèn", "離Lí",  "兌Duì", "乾Qián",
]

DOMAIN_EMOJI = {
    "GEO":   "🌍", "HYDRO": "🌊", "PYRO":  "🔥",
    "AERO":  "💨", "COSMO": "✨", "NOOS":  "🧠",
}

DOMAIN_MEANING = {
    "GEO":   "Земля/Код — Мастер, Инженер",
    "HYDRO": "Вода/Анализ — Разведчик, Аналитик",
    "PYRO":  "Огонь/Система — Архитектор",
    "AERO":  "Ветер/Математика — Логик",
    "COSMO": "Пустота/Человек — Дипломат, Лидер",
    "NOOS":  "Сознание/Информация — Философ",
}

HEXAGRAMS = _make_hexagrams()     # (64, 6)
BIANGUA   = _make_biangua_matrix(HEXAGRAMS)  # (64, 64)


# ─── Визуализация ─────────────────────────────────────────────────────────────

def hex_to_lines(hex_idx: int) -> str:
    """Отображает гексаграмму как 6 линий И-Цзин."""
    v = HEXAGRAMS[hex_idx].tolist()
    lines = []
    for b in reversed(v):  # И-Цзин: линия 1 снизу, линия 6 сверху
        lines.append("━━━━━━━" if b > 0 else "━━━  ━━━")
    return "\n".join(lines)


def hex_to_bits(hex_idx: int) -> str:
    """Отображает бинарный код гексаграммы."""
    v = HEXAGRAMS[hex_idx].tolist()
    return "".join("+" if b > 0 else "-" for b in v)


def domain_bar(weights: List[float], width: int = 20) -> str:
    """Рисует горизонтальный бар доменных весов."""
    bars = []
    for j, (name, w) in enumerate(zip(DOMAINS, weights)):
        filled = int(w * width)
        bar = "█" * filled + "░" * (width - filled)
        emoji = DOMAIN_EMOJI[name]
        active = "●" if w > 0.5 else "○"
        bars.append(f"  {active} {emoji} {name:5s} [{bar}] {w:.2f}")
    return "\n".join(bars)


def hexagram_display(hex_idx: int, show_lines: bool = True) -> str:
    """Полное отображение гексаграммы."""
    bits = hex_to_bits(hex_idx)
    symbol = HEXAGRAM_SYMBOLS[hex_idx % 8]
    lines = [f"  [{hex_idx:2d}] {symbol}  {bits}"]
    if show_lines:
        for line in hex_to_lines(hex_idx).split("\n"):
            lines.append(f"       {line}")
    return "\n".join(lines)


def color(text: str, code: str) -> str:
    """ANSI-цвет (работает в большинстве терминалов)."""
    codes = {
        "bold":    "\033[1m",
        "dim":     "\033[2m",
        "cyan":    "\033[96m",
        "yellow":  "\033[93m",
        "green":   "\033[92m",
        "red":     "\033[91m",
        "blue":    "\033[94m",
        "magenta": "\033[95m",
        "reset":   "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def print_separator(title: str = "", width: int = 64):
    if title:
        pad = (width - len(title) - 2) // 2
        print(color("─" * pad + f" {title} " + "─" * pad, "dim"))
    else:
        print(color("─" * width, "dim"))


# ─── Модель и инференс ────────────────────────────────────────────────────────

def build_model(d_model: int = 128, n_layers: int = 3) -> Variant3GPT:
    """Создаёт модель Варианта 3."""
    cfg = Variant3Config(
        vocab_size=256,    # байтовый токенайзер
        block_size=256,
        d_model=d_model,
        n_heads=max(2, d_model // 32),
        n_layers=n_layers,
        ffn_mult=4,
        hamming_lambda=0.1,
        uncertainty_budget=0.3,
        use_domain_routing=True,
    )
    model = Variant3GPT(cfg)
    model.eval()
    return model


def text_to_tokens(text: str) -> torch.Tensor:
    """Байтовый токенайзер: текст → (1, T) LongTensor."""
    ids = [b for b in text.encode('utf-8')][:255]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def tokens_to_text(tokens: torch.Tensor) -> str:
    """Обратное преобразование: (T,) → строка."""
    try:
        return bytes(tokens.tolist()).decode('utf-8', errors='replace')
    except Exception:
        return "?"


def analyze_text(model: Variant3GPT, text: str) -> Dict:
    """Прогоняет текст через модель и собирает аналитику."""
    tokens = text_to_tokens(text)
    with torch.no_grad():
        logits, _, routing_info = model(tokens)

    hw = routing_info['hex_weights'][0]      # (T, 64)
    dw = routing_info['domain_weights'][0]   # (T, 6)
    sh = routing_info['soft_hex'][0]         # (T, 6)

    dominant  = get_dominant_hexagram(hw.unsqueeze(0))[0]  # (T,)
    active    = get_active_domains(routing_info['domain_weights'], threshold=0.5)[0]
    probs     = F.softmax(logits[0], dim=-1)  # (T, 256)

    return {
        'tokens':    tokens[0],
        'logits':    logits[0],
        'probs':     probs,
        'hex_weights': hw,
        'domain_weights': dw,
        'soft_hex':  sh,
        'dominant':  dominant,
        'active_domains': active,
        'text':      text,
    }


# ─── Режимы диалога ───────────────────────────────────────────────────────────

def mode_inspect(model: Variant3GPT, text: str):
    """Инспектирует текст: каждый символ → архетип и домены."""
    print_separator("INSPECT")
    info = analyze_text(model, text)
    T = len(info['tokens'])

    print(f"\n  {color('Текст:', 'bold')} {repr(text)}")
    print(f"  {color('Токенов:', 'bold')} {T}\n")

    # Таблица: символ → гексаграмма → домены
    print(color("  Позиция  Символ  Hex  Биты    Домены (активные)", "bold"))
    print_separator()
    for t in range(min(T, 32)):
        tok   = info['tokens'][t].item()
        char  = chr(tok) if 32 <= tok < 128 else f"0x{tok:02x}"
        hex_i = info['dominant'][t].item()
        bits  = hex_to_bits(hex_i)
        doms  = ", ".join(info['active_domains'][t]) or "—"
        sym   = HEXAGRAM_SYMBOLS[hex_i % 8]

        print(f"  [{t:3d}]     {char:6s}  {hex_i:2d}{sym}  {bits}  {doms}")

    if T > 32:
        print(f"  ... ещё {T - 32} позиций")

    # Итоговая статистика
    print_separator()
    unique_hex = info['dominant'].unique()
    print(f"\n  {color('Уникальных архетипов:', 'cyan')} {unique_hex.numel()} из 64")

    # Средний вектор доменов
    avg_dw = info['domain_weights'].mean(dim=0)
    print(f"\n  {color('Средние веса доменов:', 'cyan')}")
    print(domain_bar(avg_dw.tolist()))

    # Топ-3 гексаграммы
    hex_counts = torch.zeros(64)
    for h in info['dominant'].tolist():
        hex_counts[h] += 1
    top3 = hex_counts.argsort(descending=True)[:3]
    print(f"\n  {color('Топ-3 гексаграммы:', 'cyan')}")
    for h in top3.tolist():
        if hex_counts[h] > 0:
            print(f"    [{h:2d}] {HEXAGRAM_SYMBOLS[h%8]} {hex_to_bits(h)}  "
                  f"встречается {int(hex_counts[h])}× "
                  f"({hex_counts[h]/T*100:.1f}%)")

    # 変爻-путь от первого к последнему архетипу
    h_first = info['dominant'][0].item()
    h_last  = info['dominant'][-1].item()
    if h_first != h_last:
        path = biangua_path(h_first, h_last)
        print(f"\n  {color('変爻-путь', 'cyan')} [{h_first}]→[{h_last}]: "
              f"{len(path)-1} изменений")
        path_str = " → ".join(
            f"{h}({HEXAGRAM_SYMBOLS[h%8]})" for h in path
        )
        for chunk in textwrap.wrap(path_str, width=60):
            print(f"    {chunk}")


def mode_compare(model: Variant3GPT, text1: str, text2: str):
    """Сравнивает два текста в Q6-пространстве."""
    print_separator("COMPARE")
    info1 = analyze_text(model, text1)
    info2 = analyze_text(model, text2)

    # Средние hex_weights и domain_weights
    hw1 = info1['hex_weights'].mean(dim=0)   # (64,)
    hw2 = info2['hex_weights'].mean(dim=0)
    dw1 = info1['domain_weights'].mean(dim=0) # (6,)
    dw2 = info2['domain_weights'].mean(dim=0)
    sh1 = info1['soft_hex'].mean(dim=0)      # (6,)
    sh2 = info2['soft_hex'].mean(dim=0)

    # Косинусное сходство в Q6
    cos_hex = F.cosine_similarity(hw1.unsqueeze(0), hw2.unsqueeze(0)).item()
    cos_dom = F.cosine_similarity(dw1.unsqueeze(0), dw2.unsqueeze(0)).item()

    # Хэмминг-расстояние между средними soft_hex
    h_dist = hamming_distance_soft(sh1, sh2).item()

    print(f"\n  {color('Текст A:', 'cyan')} {repr(text1[:50])}")
    print(f"  {color('Текст B:', 'cyan')} {repr(text2[:50])}")
    print_separator()

    print(f"\n  {color('Косинусное сходство (hex):', 'yellow')}  {cos_hex:.4f}")
    print(f"  {color('Косинусное сходство (dom):', 'yellow')}  {cos_dom:.4f}")
    print(f"  {color('Хэмминг-расстояние (soft):', 'yellow')}  {h_dist:.3f} / 6.0")
    print()

    # Доминирующий архетип для каждого
    dom_h1 = int(info1['dominant'].mode().values.item())
    dom_h2 = int(info2['dominant'].mode().values.item())
    path   = biangua_path(dom_h1, dom_h2) if dom_h1 != dom_h2 else [dom_h1]

    print(f"  {color('Доминирующий архетип A:', 'green')} "
          f"[{dom_h1:2d}] {HEXAGRAM_SYMBOLS[dom_h1%8]} {hex_to_bits(dom_h1)}")
    print(f"  {color('Доминирующий архетип B:', 'green')} "
          f"[{dom_h2:2d}] {HEXAGRAM_SYMBOLS[dom_h2%8]} {hex_to_bits(dom_h2)}")

    if dom_h1 != dom_h2:
        print(f"  {color('変爻-расстояние:', 'green')} {len(path)-1} шагов")

    print(f"\n  {color('Домены A:', 'magenta')}")
    print(domain_bar(dw1.tolist()))
    print(f"\n  {color('Домены B:', 'magenta')}")
    print(domain_bar(dw2.tolist()))

    # Дифференциальный анализ: какие домены различаются больше всего?
    diffs = [(DOMAINS[j], abs(dw1[j].item() - dw2[j].item())) for j in range(6)]
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  {color('Наибольшие различия по доменам:', 'yellow')}")
    for name, diff in diffs[:3]:
        emoji = DOMAIN_EMOJI[name]
        print(f"    {emoji} {name:5s}: Δ={diff:.3f}  ({DOMAIN_MEANING[name]})")


def mode_hexagram(hex_idx: int):
    """Исследует гексаграмму: соседи, антипод, путь."""
    print_separator(f"HEXAGRAM [{hex_idx}]")

    if not 0 <= hex_idx < 64:
        print(f"  Ошибка: индекс должен быть 0..63, получено {hex_idx}")
        return

    # Отображение
    print(f"\n{hexagram_display(hex_idx)}")
    bits = hex_to_bits(hex_idx)
    v    = HEXAGRAMS[hex_idx].tolist()

    # Семантика: какие домены "активны" (линия = ян)
    print(f"\n  {color('Активные линии (ян = +):', 'cyan')}")
    for j, (name, b) in enumerate(zip(DOMAINS, v)):
        state = "ян ━━━━━━━" if b > 0 else "инь ━━━  ━━━"
        emoji = DOMAIN_EMOJI[name]
        print(f"    Линия {j+1}: {state}  {emoji} {name} — {DOMAIN_MEANING[name]}")

    # 6 Хэмминг-1 соседей (変爻)
    neighbors = BIANGUA[hex_idx].nonzero(as_tuple=False).squeeze(-1).tolist()
    print(f"\n  {color('変爻-соседи (Хэмминг-1):', 'yellow')}")
    for nb in sorted(neighbors):
        nb_bits  = hex_to_bits(nb)
        sym      = HEXAGRAM_SYMBOLS[nb % 8]
        # Какая линия изменилась?
        changed  = [j for j in range(6) if v[j] != HEXAGRAMS[nb][j].item()]
        line_str = f"Линия {changed[0]+1}: {DOMAINS[changed[0]]}" if changed else "?"
        print(f"    [{nb:2d}] {sym} {nb_bits}  ← {line_str}")

    # Антипод
    antipodal_idx = int((HEXAGRAMS[hex_idx] == -HEXAGRAMS).all(dim=1).nonzero()[0].item())
    print(f"\n  {color('Антипод (Хэмминг-6):', 'red')} "
          f"[{antipodal_idx}] {HEXAGRAM_SYMBOLS[antipodal_idx%8]} {hex_to_bits(antipodal_idx)}")

    # Путь до антипода
    path = biangua_path(hex_idx, antipodal_idx)
    print(f"\n  {color('変爻-путь до антипода:', 'red')} {len(path)-1} шагов")
    for step, h in enumerate(path):
        sym  = HEXAGRAM_SYMBOLS[h % 8]
        b    = hex_to_bits(h)
        mark = "← START" if step == 0 else ("← END" if step == len(path)-1 else "")
        print(f"    Шаг {step}: [{h:2d}] {sym} {b}  {mark}")


def mode_train(model: Variant3GPT, text: str, steps: int = 50):
    """Быстрое обучение на пользовательском тексте."""
    print_separator("TRAIN")
    print(f"\n  Текст: {repr(text[:80])}")
    print(f"  Шагов: {steps}")
    print()

    tokens  = text_to_tokens(text)
    T       = tokens.shape[1]
    if T < 2:
        print("  Слишком короткий текст (нужно хотя бы 2 символа)")
        return

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    targets   = tokens.roll(-1, dims=-1)

    for step in range(steps):
        optimizer.zero_grad()
        _, loss, _ = model(tokens, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == steps - 1:
            bar_len  = 20
            progress = int((step + 1) / steps * bar_len)
            bar      = "█" * progress + "░" * (bar_len - progress)
            print(f"  Шаг {step+1:3d}/{steps} [{bar}] loss={loss.item():.4f}",
                  end="\r" if step < steps - 1 else "\n")

    model.eval()
    print(f"\n  {color('Обучение завершено!', 'green')}")

    # Генерация
    print(f"\n  {color('Генерация (seed = первые 8 символов):', 'cyan')}")
    seed_len = min(8, T)
    seed     = tokens[:, :seed_len].clone()
    generated = seed[0].tolist()

    with torch.no_grad():
        for _ in range(40):
            inp = torch.tensor(generated[-min(128, len(generated)):]).unsqueeze(0)
            logits, _, _ = model(inp)
            probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_tok)

    gen_text = tokens_to_text(torch.tensor(generated[seed_len:]))
    seed_text = tokens_to_text(torch.tensor(generated[:seed_len]))
    print(f"  {color(seed_text, 'dim')}{color(gen_text, 'yellow')}")


def mode_probe(model: Variant3GPT, text: str):
    """Зонд: анализирует внутреннее состояние послойно."""
    print_separator("PROBE")
    tokens = text_to_tokens(text)
    T = tokens.shape[1]

    print(f"\n  Текст: {repr(text[:50])}")
    print(f"  Слоёв: {len(model.blocks)}\n")

    # Регистрируем хуки для перехвата промежуточных активаций
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                layer_outputs[layer_idx] = out.detach()
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        h = block.register_forward_hook(make_hook(i))
        handles.append(h)

    with torch.no_grad():
        logits, _, routing_info = model(tokens)

    for h in handles:
        h.remove()

    # Анализ по слоям
    print(color("  Слой  L2-норма  Энтропия  Топ-гексаграмма", "bold"))
    print_separator()

    hex_proj = HexagramProjection(d_model=model.cfg.d_model)
    # Используем проекцию из первого блока
    if model.blocks:
        hex_proj = model.blocks[0].hex_proj

    for layer_idx, act in sorted(layer_outputs.items()):
        # act: (1, T, d)
        l2_norm   = act.norm(dim=-1).mean().item()
        # Проецируем через общую hex_proj для сравнения
        with torch.no_grad():
            _, hw = hex_proj(act)
        entropy   = -(hw * (hw + 1e-8).log()).sum(dim=-1).mean().item()
        dominant  = hw[0].argmax(dim=-1).mode().values.item()
        sym       = HEXAGRAM_SYMBOLS[int(dominant) % 8]
        bits      = hex_to_bits(int(dominant))

        print(f"  [{layer_idx}]    {l2_norm:7.3f}   {entropy:7.3f}   "
              f"[{int(dominant):2d}]{sym} {bits}")

    # Анализ после роутера
    if routing_info:
        hw = routing_info['hex_weights'][0]
        dw = routing_info['domain_weights'][0]
        avg_ent  = -(hw * (hw + 1e-8).log()).sum(dim=-1).mean().item()
        avg_dom  = dw.mean(dim=0)

        print_separator()
        print(f"\n  {color('После NautilusYiJinRouter:', 'yellow')}")
        print(f"  Энтропия hex_weights: {avg_ent:.3f} "
              f"(меньше = более острые архетипы)")
        print(f"\n  Средние веса доменов:")
        print(domain_bar(avg_dom.tolist()))

    # Перплексия
    targets = tokens.roll(-1, dims=-1)
    with torch.no_grad():
        _, loss, _ = model(tokens, targets)
    ppl = math.exp(loss.item()) if loss is not None else float('inf')
    print(f"\n  {color(f'Перплексия: {ppl:.2f}', 'cyan')}")


def mode_ideas():
    """Выводит список идей для дальнейшего развития архитектуры."""
    print_separator("IDEAS & ROADMAP")
    ideas = [
        (
            "1. HexagramPositionalEncoding",
            "Позиционное кодирование через Q6: pos t → гексаграмма(t mod 64)\n"
            "   Вместо синусоид — переходы по変爻-графу (цикл длиной 64).\n"
            "   Преимущество: периодичность с смысловой структурой (8 дворцов).",
        ),
        (
            "2. SixLineAttention (головы = линии)",
            "6 голов attention = 6 линий гексаграммы = 6 доменов.\n"
            "   Голова j обрабатывает только компоненту домена j в пространстве.\n"
            "   → Интерпретируемость каждой головы = один домен знания.",
        ),
        (
            "3. BianGuaOptimizer",
            "Оптимизатор на основе変爻-переходов:\n"
            "   Шаг = Хэмминг-1 изменение в квантованном весовом пространстве.\n"
            "   Аналог LION/Sophia, но с геометрическим prior (Q6-структура).",
        ),
        (
            "4. TernaryKVCache",
            "KV-cache с тернарными ключами {-1,0,+1}^d:\n"
            "   Хранение: 2 бита вместо float32 (экономия 16×).\n"
            "   Поиск: popcount вместо dot-product (ускорение на FPGA/CPU).",
        ),
        (
            "5. HexagramTokenizer",
            "Токенайзер на основе Q6: каждый токен = гексаграмма.\n"
            "   64 базовых токена × структурированные паттерны.\n"
            "   Аналог BPE, но слияния = Variable爻 (изменение одной линии).",
        ),
        (
            "6. CrossDomainRAG",
            "Retrieval-Augmented Generation с доменным индексом.\n"
            "   Запрос → Q6-вектор → поиск ближайших по Хэммингу документов.\n"
            "   Каждый документ имеет 'гексаграммную подпись' из 6 битов.",
        ),
        (
            "7. HexagramEval (оценка моделей)",
            "Метрика качества генерации через Q6-разнообразие:\n"
            "   - hex_entropy: насколько равномерно используются архетипы?\n"
            "   - biangua_path_coverage: все ли 変爻-пути покрыты в тексте?\n"
            "   - domain_coherence: один текст → один доминирующий домен?",
        ),
        (
            "8. MultiScaleQ6 (Matryoshka + Q6)",
            "Иерархия: Q2 (4 биграммы) → Q3 (8 триграмм) → Q6 (64 гексаграммы).\n"
            "   Matryoshka: первые 4 измерения = грубый архетип, все 6 = точный.\n"
            "   Поддерживает многоуровневый поиск (coarse-to-fine retrieval).",
        ),
        (
            "9. AdaptiveHammingLambda (расписание λ)",
            "Curriculum для топологического bias:\n"
            "   λ: 0 → max (нагрев) → 0 (отжиг) → steady.\n"
            "   На старте стандартное attention, затем включается геометрия.",
        ),
        (
            "10. HexagramMoE (64 эксперта = 64 гексаграммы)",
            "Mixture-of-Experts с 64 экспертами, по одному на архетип.\n"
            "   Роутинг через hex_weights (уже вычислен в HexagramProjection).\n"
            "   O(1) роутинг без дополнительной сети — используем Q6-геометрию.",
        ),
    ]

    for title, desc in ideas:
        print(f"\n  {color(title, 'bold')}")
        for line in desc.split("\n"):
            print(f"  {color(line, 'dim')}")

    print(f"\n  {color('Вопросы для исследования:', 'yellow')}")
    questions = [
        "Выучивает ли модель на реальных данных значимую Q6-кластеризацию?",
        "Совпадают ли доминирующие домены с семантикой текста?",
        "Является ли biangua-граф 'шагами рассуждения' в chain-of-thought?",
        "Даёт ли TernaryGate интерпретируемые 'паузы неопределённости'?",
        "Как мера Хэмминг-энтропии коррелирует с перплексией?",
    ]
    for q in questions:
        print(f"    ❓ {q}")


def mode_concepts():
    """
    Анализ концептуального текста: антонимы, конвейер, стек Forth, свой/чужой.
    Показывает, как каждая идея из текста ChatGPT интегрируется в Q6-архитектуру.
    """
    print_separator("КОНЦЕПТУАЛЬНАЯ КАРТА: ТЕКСТ → АРХИТЕКТУРА Q6")

    concept_map = [
        (
            "АНТОНИМНЫЕ ПАРЫ {+/-} ↔ Q6-оси",
            "В тексте: таблица из 6 пар (постоянное/переменное, ближнее/дальнее, …)\n"
            "В архитектуре: каждый бит гексаграммы {-1,+1} = одна ось антонимии.\n"
            "64 = 2^6 = все возможные комбинации 6 бинарных оппозиций.\n"
            "Реализация: BinaryOppositionTable — 6 линий × (positive_pole / negative_pole).\n"
            "Антонимы? ДА — противоположные полюса одной и той же оси (Хэмминг-1 = одна 変爻).",
        ),
        (
            "ПРИНЦИП КОНВЕЙЕРА ↔ Variant3Block (6 стадий)",
            "В тексте: конвейер как последовательные стадии обработки.\n"
            "В архитектуре: ConveyorVariant3Block — ровно 6 именованных стадий:\n"
            "  1. Q6_LOCALISE    — где в гиперкубе находится токен?\n"
            "  2. TOPO_ATTEND    — топологическое внимание (BianGuaAttention)\n"
            "  3. TERNARY_GATE   — фильтр неопределённости {-1, 0, +1}\n"
            "  4. INTERLINGUA    — хаб-и-спицы (ArchetypalInterlingua)\n"
            "  5. BIANGUA_ANALOGY — аналогия через 変爻 (CrossHexagramAnalogy)\n"
            "  6. SWIGLU_FFN     — нелинейный синтез (SwiGLU)\n"
            "Число стадий = число линий гексаграммы = 6. Это НЕ случайно.",
        ),
        (
            "СТЕК FORTH (LIFO) ↔ CrossHexagramAnalogy (変爻-цепочка)",
            "В тексте: стек Forth — последний вошёл, первый вышел; операция = смена вершины.\n"
            "В архитектуре: biangua_path = BFS по Хэмминг-1 графу Q6.\n"
            "  Каждый шаг пути = одна 変爻 (смена одной линии) = 'pop & push' одного бита.\n"
            "Дополнительно: CrossHexagramAnalogy 'читает' соседей (аналог peek в стеке).\n"
            "HexagramTokenizer: BPE-слияния только по Хэмминг-1 парам = merge-стек.",
        ),
        (
            "СВОЙ / ЧУЖОЙ ↔ SvoyChuzhoiGate + TernaryGate",
            "В тексте: свой/чужой — трёхзначная логика (близкий/нейтральный/дальний).\n"
            "В архитектуре двойное соответствие:\n"
            "  TernaryGate:      {-1, 0, +1} по уверенности в активации.\n"
            "  SvoyChuzhoiGate:  {+1, 0, -1} по расстоянию до прототипа в Q6.\n"
            "Точное соответствие: свой=+1 (ян), нейтральный=0 (変爻), чужой=-1 (инь).\n"
            "Антоним? ДА — свой и чужой — крайние точки Q6-расстояния.",
        ),
        (
            "МЕТОД ИСКЛЮЧЕНИЯ ↔ BinaryExclusionClassifier",
            "В тексте: метод исключения — принять только если все критерии пройдены.\n"
            "В архитектуре: BinaryExclusionClassifier — 6 осей, AND-логика.\n"
            "  accept = axis_1 AND axis_2 AND ... AND axis_6\n"
            "  Любой ноль (False) → отказ (исключение).\n"
            "Применение: маршрутизация велосипеда (6 критериев безопасности),\n"
            "  фильтрация текста (6 осей качества в TextQualityFilter).",
        ),
        (
            "ВЕЛОСИПЕДНЫЙ ИИ / БЕЗОПАСНОСТЬ ↔ NautilusYiJinRouter + BinaryExclusionClassifier",
            "В тексте: ИИ для велосипедиста — выбор безопасного маршрута.\n"
            "В архитектуре: NautilusYiJinRouter распределяет токены по 6 доменам\n"
            "  (GEO=маршрут, HYDRO=погода, PYRO=скорость, AERO=ветер, COSMO=время, NOOS=план).\n"
            "  BinaryExclusionClassifier проверяет все 6 критериев безопасности.\n"
            "Метафора: 6 линий гексаграммы = 6 факторов безопасности пути.",
        ),
        (
            "ИИ КОРМУШКИ ДЛЯ ДОМАШНИХ ЖИВОТНЫХ ↔ SvoyChuzhoiGate (прототип = владелец)",
            "В тексте: ИИ учится распознавать 'своих' питомцев vs посторонних.\n"
            "В архитектуре: SvoyChuzhoiGate хранит n_prototypes 'прототипов'.\n"
            "  Короткое расстояние → 'свой' (+1) → открыть кормушку.\n"
            "  Далёкое расстояние  → 'чужой' (-1) → не открывать.\n"
            "  Средняя зона → нейтральный (0) → ждать.",
        ),
        (
            "ФИЛЬТР КАЧЕСТВА ТЕКСТА ↔ TextQualityFilter (6-битная гексаграмма)",
            "В тексте: оценка качества текста как многомерная фильтрация.\n"
            "В архитектуре: TextQualityFilter — 6 осей качества:\n"
            "  is_factual, is_coherent, is_relevant, is_clear, is_complete, is_safe\n"
            "Каждая ось = один бит → вместе 6 битов = индекс гексаграммы (0..63).\n"
            "  63 = 111111 = 'идеальный текст' (все критерии выполнены)\n"
            "   0 = 000000 = 'спам/мусор' (все критерии провалены)\n"
            "Это идеальный аналог 64 гексаграмм Ицзин как классификатора.",
        ),
        (
            "МАТРИЦА АНТОНИМОВ ↔ Q6-гиперкуб как таблица антонимов",
            "В тексте: таблица антонимных пар — строки=пары, столбцы=полюса.\n"
            "В архитектуре: Q6 = {-1,+1}^6 — ПОЛНАЯ таблица антонимов.\n"
            "  Каждая вершина гиперкуба = одна строка в таблице антонимов.\n"
            "  Соседи (Хэмминг-1) = строки, отличающиеся ровно в одной позиции.\n"
            "  BinaryOppositionTable.interpret() возвращает текстовую версию таблицы.",
        ),
        (
            "6 ДОМЕНОВ NAUTILUS ↔ 6 ЛИНИЙ ГЕКСАГРАММЫ",
            "В тексте: любой предмет описывается через 6 измерений.\n"
            "В архитектуре: NautilusYiJinRouter → 6 доменов = 6 линий:\n"
            "  GEO=Земля   HYDRO=Вода   PYRO=Огонь\n"
            "  AERO=Воздух COSMO=Космос NOOS=Разум\n"
            "  SixLineAttention: голова i специализируется на домене i.\n"
            "  MultiScaleQ6: Q2→Q3→Q6 = 2→3→6 линий = нарастающая детализация.",
        ),
    ]

    for title, desc in concept_map:
        print(f"\n  {color(title, 'bold')}")
        for line in desc.split("\n"):
            print(f"  {color(line, 'dim')}")

    print(f"\n  {color('═' * 60, 'cyan')}")
    print(f"  {color('ВЫВОД:', 'yellow')}")
    summary = [
        "Все 6 концептов из текста ChatGPT прямо воплощены в Q6-архитектуре.",
        "Антонимы = противоположные полюса битов {-1,+1} (Хэмминг-1 от центра).",
        "Конвейер = 6 стадий Variant3Block (одна на каждую линию гексаграммы).",
        "Forth-стек = 変爻-граф с BFS-навигацией (шаг = pop+push одного бита).",
        "Свой/чужой = TernaryGate / SvoyChuzhoiGate (оба трёхзначны: +1/0/-1).",
        "Метод исключения = AND по 6 осям в BinaryExclusionClassifier.",
        "Фильтр текста = 6-битная гексаграмма качества (0=спам, 63=отлично).",
    ]
    for line in summary:
        print(f"  {color('✓', 'green')} {line}")


# ─── Главный REPL ─────────────────────────────────────────────────────────────

HELP_TEXT = """
Команды:
  inspect <текст>              — инспектировать архетипы и домены текста
  compare <текст1> | <текст2>  — сравнить два текста в Q6
  hexagram <число 0..63>       — исследовать гексаграмму
  train <текст>                — быстро обучить модель на тексте
  probe <текст>                — посмотреть внутренние слои
  ideas                        — 10 идей для развития архитектуры
  concepts                     — карта концептов (антонимы/конвейер/Forth/свой-чужой)
  help                         — эта справка
  quit / exit / q              — выход
"""


def repl(model: Variant3GPT):
    """Основной цикл REPL."""
    print(color("\n" + "═" * 64, "cyan"))
    print(color("  VARIANT 3 — ДИАЛОГ С АРХЕТИПИЧЕСКОЙ МОДЕЛЬЮ", "bold"))
    print(color("  Архетипы • Q6 • 変爻 • NautilusYiJin", "dim"))
    print(color("═" * 64, "cyan"))
    print(color(f"\n  Модель: {model.count_parameters():,} параметров", "dim"))
    print(color(f"  Домены: {' / '.join(DOMAINS)}", "dim"))
    print(color("\n  Введите 'help' для справки, 'ideas' для идей.\n", "dim"))

    while True:
        try:
            raw = input(color("  ❯ ", "green")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  До свидания! 再見 Zàijiàn!")
            break

        if not raw:
            continue

        cmd_parts = raw.split(None, 1)
        cmd  = cmd_parts[0].lower()
        rest = cmd_parts[1] if len(cmd_parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("  До свидания! 再見 Zàijiàn!")
            break
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd == "ideas":
            mode_ideas()
        elif cmd == "concepts":
            mode_concepts()
        elif cmd == "hexagram":
            try:
                idx = int(rest.strip())
                mode_hexagram(idx)
            except ValueError:
                print(f"  Ожидалось число 0..63, получено: {repr(rest)}")
        elif cmd == "inspect":
            if not rest:
                print("  Укажите текст: inspect <текст>")
            else:
                mode_inspect(model, rest)
        elif cmd == "compare":
            if "|" not in rest:
                print("  Формат: compare <текст A> | <текст B>")
            else:
                parts = rest.split("|", 1)
                mode_compare(model, parts[0].strip(), parts[1].strip())
        elif cmd == "train":
            if not rest:
                print("  Укажите текст: train <текст>")
            else:
                mode_train(model, rest)
        elif cmd == "probe":
            if not rest:
                print("  Укажите текст: probe <текст>")
            else:
                mode_probe(model, rest)
        else:
            # Попытка автоопределить команду (если введён просто текст → inspect)
            if raw[0].isalpha() and len(raw) > 3:
                print(color(f"  (inspect {repr(raw)})", "dim"))
                mode_inspect(model, raw)
            else:
                print(f"  Неизвестная команда: {repr(cmd)}. Введите 'help'.")

        print()


# ─── CLI-аргументы ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Variant 3 — интерактивный диалог с архетипической моделью"
    )
    parser.add_argument("--mode",   default="repl",
                        choices=["repl", "inspect", "compare", "hexagram",
                                 "train", "probe", "ideas", "concepts"])
    parser.add_argument("--text",   default="", help="Текст для анализа")
    parser.add_argument("--text2",  default="", help="Второй текст (для compare)")
    parser.add_argument("--hex",    type=int, default=0,
                        help="Индекс гексаграммы (для mode hexagram)")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--steps",  type=int, default=50,
                        help="Шагов обучения (для mode train)")
    args = parser.parse_args()

    print(color("  Инициализация модели Variant 3...", "dim"), end="\r")
    model = build_model(d_model=args.d_model, n_layers=args.n_layers)
    print(color("  Модель готова.                   ", "dim"))

    if args.mode == "repl":
        repl(model)
    elif args.mode == "inspect":
        text = args.text or "Hello, World!"
        mode_inspect(model, text)
    elif args.mode == "compare":
        t1 = args.text  or "mathematics and logic"
        t2 = args.text2 or "poetry and emotion"
        mode_compare(model, t1, t2)
    elif args.mode == "hexagram":
        mode_hexagram(args.hex)
    elif args.mode == "train":
        text = args.text or "The quick brown fox jumps over the lazy dog."
        mode_train(model, text, steps=args.steps)
    elif args.mode == "probe":
        text = args.text or "Hello!"
        mode_probe(model, text)
    elif args.mode == "ideas":
        mode_ideas()
    elif args.mode == "concepts":
        mode_concepts()


if __name__ == "__main__":
    main()
