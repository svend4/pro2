#!/usr/bin/env python3
"""
test_polyglot_dialogue.py — Тестовый диалог с PolyglotQuartet

Создаёт модель, обучает на небольшом корпусе, затем задаёт вопросы
и слушает ответы всех четырёх музыкантов.

Запуск:
    python scripts/test_polyglot_dialogue.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from yijing_transformer.models.polyglot import (
    build_polyglot, VOCABS, PolyglotQuartet,
)
from yijing_transformer.models.polyglot_translation import CrossTranslator
from yijing_transformer.models.polyglot_curriculum import DifficultyEstimator


# ═══════════════════════════════════════════════════════════════
# Простой символьный токенизатор
# ═══════════════════════════════════════════════════════════════

class CharTokenizer:
    """Побуквенный токенизатор: каждый символ → свой ID."""

    def __init__(self, texts):
        chars = sorted(set(''.join(texts)))
        self.ch2id = {ch: i + 2 for i, ch in enumerate(chars)}  # 0=pad, 1=unk
        self.id2ch = {i: ch for ch, i in self.ch2id.items()}
        self.vocab_size = len(chars) + 2

    def encode(self, text):
        return [self.ch2id.get(ch, 1) for ch in text]

    def decode(self, ids):
        return ''.join(self.id2ch.get(i, '?') for i in ids if i > 1)


# ═══════════════════════════════════════════════════════════════
# Мини-корпус для обучения
# ═══════════════════════════════════════════════════════════════

CORPUS = [
    # Физика
    "Энергия равна массе умноженной на скорость света в квадрате.",
    "Свет — это электромагнитная волна, которая распространяется в вакууме.",
    "Гравитация притягивает все тела друг к другу с силой обратно пропорциональной квадрату расстояния.",
    "Атом состоит из ядра и электронов, вращающихся вокруг него.",
    "Энтропия замкнутой системы никогда не убывает — второй закон термодинамики.",

    # Математика
    "Число пи — это отношение длины окружности к её диаметру.",
    "Сумма углов треугольника равна ста восьмидесяти градусам.",
    "Простое число делится только на единицу и на самого себя.",
    "Множество всех множеств не содержит само себя — парадокс Рассела.",
    "Бесконечность — не число, а концепция безграничного продолжения.",

    # Философия
    "Я мыслю, следовательно я существую — сказал Декарт.",
    "Мудрость начинается с признания собственного незнания.",
    "Всё течёт, всё меняется — говорил Гераклит.",
    "Человек — это мост между животным и сверхчеловеком.",
    "Свобода — это осознанная необходимость.",

    # Мифология
    "Феникс возрождается из собственного пепла каждые пятьсот лет.",
    "Мировое дерево соединяет три мира: верхний, средний и нижний.",
    "Дракон охраняет сокровища знания в глубине горы.",
    "Лабиринт Минотавра символизирует путь к центру самого себя.",
    "Орфей спустился в подземный мир, чтобы вернуть Эвридику.",

    # Музыка
    "Музыка — это математика, обращённая в звук.",
    "Гармония возникает когда различные голоса звучат как одно целое.",
    "Ритм — это пульс музыки, он задаёт движение и энергию.",
    "Каждая нота — это частота вибрации воздуха.",
    "Тишина между нотами так же важна, как сами ноты.",
]


def train_mini(model, tokenizer, corpus, steps=300, lr=3e-3):
    """Быстрое обучение на мини-корпусе."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Готовим данные
    block_size = 64
    all_ids = []
    for text in corpus:
        ids = tokenizer.encode(text)
        if len(ids) > block_size:
            ids = ids[:block_size]
        all_ids.append(ids)

    model.train()
    losses = []
    t0 = time.time()

    for step in range(steps):
        # Случайный батч
        batch_idx = [torch.randint(len(all_ids), (1,)).item() for _ in range(4)]
        xs, ys = [], []
        for bi in batch_idx:
            ids = all_ids[bi]
            # Дополняем до block_size
            padded = ids + [0] * (block_size - len(ids))
            xs.append(padded[:block_size])
            ys.append(padded[1:block_size] + [0])

        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)

        logits, loss, info = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0 or step == steps - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            ce = info.get('ce_loss', 0)
            ros = info.get('rosetta_loss', 0)
            spec = info.get('spec_loss', 0)
            print(
                f"  шаг {step:4d}/{steps} | "
                f"loss={avg:.3f} (CE={ce:.3f} Ros={ros:.4f} Spec={spec:.3f}) | "
                f"{elapsed:.1f}с"
            )

    return losses


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_len=80, temperature=0.8):
    """Генерация текста авторегрессивно."""
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_len):
        # Обрезаем до block_size
        idx_cond = idx[:, -64:]
        logits, _, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k фильтрация
        top_k = 40
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        # Останавливаемся на точке
        ch = tokenizer.id2ch.get(next_id.item(), '')
        if ch in '.!?':
            break

    return tokenizer.decode(idx[0].tolist())


@torch.no_grad()
def ask_musicians(model, tokenizer, prompt):
    """Каждый музыкант отвечает на своём языке."""
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    # Получаем специализированные выходы
    logits, _, info = model(idx)

    results = {}

    for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']:
        spec_logits = info['spec_logits'][name]
        token_ids = spec_logits.argmax(dim=-1)[0].tolist()

        if name in VOCABS:
            decoded = VOCABS[name].decode_str(token_ids)
        else:
            # Лингвист — обычный текст
            decoded = tokenizer.decode(token_ids)

        results[name] = decoded

    return results


@torch.no_grad()
def test_cross_translation(model, tokenizer, prompt):
    """Тест кросс-перевода между музыкантами."""
    translator = CrossTranslator(d_model=model.cfg.d_model)

    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    # Получаем скрытые состояния
    B, T = idx.shape
    tok = model.tok_emb(idx)
    pos = model.pos_emb[:, :T, :]
    shared = model.emb_drop(tok + pos)
    attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    hiddens = {}
    for name in model._musician_order:
        _, hidden = model.musicians[name](shared, attn_mask=attn_mask)
        hiddens[name] = hidden

    # Цикловая согласованность
    cycle_err = translator.cycle_loss(hiddens)

    # Переводы формалист → все остальные
    translations = {}
    for tgt in ['archetypist', 'algorithmist', 'linguist']:
        logits = translator.translate('formalist', tgt, hiddens['formalist'])
        token_ids = logits.argmax(dim=-1)[0].tolist()
        if tgt in VOCABS:
            decoded = VOCABS[tgt].decode_str(token_ids)
        else:
            decoded = tokenizer.decode(token_ids)
        translations[tgt] = decoded

    return translations, cycle_err.item()


def run_difficulty_analysis():
    """Анализ сложности фраз из корпуса."""
    estimator = DifficultyEstimator()
    print("\n" + "=" * 70)
    print("  АНАЛИЗ СЛОЖНОСТИ КОРПУСА")
    print("=" * 70)

    scored = [(estimator.estimate(t), t) for t in CORPUS]
    scored.sort(key=lambda x: x[0])

    print("\n  Самые простые:")
    for score, text in scored[:5]:
        print(f"    [{score:.3f}] {text[:60]}...")

    print("\n  Самые сложные:")
    for score, text in scored[-5:]:
        print(f"    [{score:.3f}] {text[:60]}...")

    avg = sum(s for s, _ in scored) / len(scored)
    print(f"\n  Средняя сложность: {avg:.3f}")


# ═══════════════════════════════════════════════════════════════
# Главный диалог
# ═══════════════════════════════════════════════════════════════

DISPLAY_NAMES = {
    'formalist': 'ФОРМАЛИСТ (формулы)',
    'archetypist': 'АРХЕТИПИСТ (архетипы)',
    'algorithmist': 'АЛГОРИТМИСТ (графы)',
    'linguist': 'ЛИНГВИСТ (текст)',
}

QUESTIONS = [
    "Что такое энергия",
    "Музыка это",
    "Свет и тьма",
    "Число пи равно",
    "Мудрость начинается",
    "Феникс возрождается",
    "Гармония возникает",
    "Атом состоит из",
    "Всё течёт",
    "Свобода это",
]


def main():
    print("=" * 70)
    print("  POLYGLOT QUARTET — ТЕСТОВЫЙ ДИАЛОГ")
    print("  Четыре музыканта, четыре языка, одна истина")
    print("=" * 70)

    # ── 1. Анализ сложности ──
    run_difficulty_analysis()

    # ── 2. Создание модели ──
    print("\n" + "=" * 70)
    print("  СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 70)

    tokenizer = CharTokenizer(CORPUS)
    print(f"\n  Словарь: {tokenizer.vocab_size} символов")

    model = build_polyglot(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=2,
        block_size=64,
    )

    params = model.count_parameters()
    print(f"  Параметры модели:")
    for k, v in params.items():
        print(f"    {k:20s}: {v:>10,}")

    # ── 3. Обучение ──
    print(f"\n  Обучение на {len(CORPUS)} фразах...")
    losses = train_mini(model, tokenizer, CORPUS, steps=300, lr=3e-3)

    # ── 4. Диалог: задаём вопросы ──
    print("\n" + "=" * 70)
    print("  ДИАЛОГ С ЧЕТЫРЬМЯ МУЗЫКАНТАМИ")
    print("=" * 70)

    for q in QUESTIONS:
        print(f"\n{'─' * 60}")
        print(f"  ВОПРОС: «{q}»")
        print(f"{'─' * 60}")

        # Генерация продолжения (лингвист, авторегрессивно)
        continuation = generate_text(model, tokenizer, q, max_len=80)
        print(f"\n  ПРОДОЛЖЕНИЕ (авторегрессия):")
        print(f"    {continuation}")

        # Ответы всех музыкантов
        responses = ask_musicians(model, tokenizer, q)
        print(f"\n  МУЗЫКАНТЫ (параллельный отклик):")
        for name, response in responses.items():
            display = DISPLAY_NAMES[name]
            # Обрезаем длинные ответы
            if len(response) > 120:
                response = response[:120] + "..."
            print(f"    {display}:")
            print(f"      {response}")

    # ── 5. Кросс-перевод ──
    print("\n" + "=" * 70)
    print("  КРОСС-ПЕРЕВОД: ФОРМАЛИСТ → ОСТАЛЬНЫЕ")
    print("=" * 70)

    for prompt in ["Энергия равна массе", "Музыка это математика", "Всё течёт"]:
        print(f"\n  Промпт: «{prompt}»")
        translations, cycle_err = test_cross_translation(model, tokenizer, prompt)
        for tgt, text in translations.items():
            display = DISPLAY_NAMES[tgt]
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"    → {display}: {text}")
        print(f"    Цикловая ошибка: {cycle_err:.6f}")

    # ── 6. Итоговая статистика ──
    print("\n" + "=" * 70)
    print("  ИТОГО")
    print("=" * 70)
    print(f"  Начальный loss: {losses[0]:.3f}")
    print(f"  Финальный loss: {losses[-1]:.3f}")
    print(f"  Снижение: {(1 - losses[-1]/losses[0])*100:.1f}%")
    print(f"  Модель: {params['total']:,} параметров")
    print(f"  Корпус: {len(CORPUS)} фраз, {sum(len(t) for t in CORPUS)} символов")
    print("=" * 70)


if __name__ == '__main__':
    main()
