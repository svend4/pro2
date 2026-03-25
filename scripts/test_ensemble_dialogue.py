#!/usr/bin/env python3
"""
test_ensemble_dialogue.py — Тестовый диалог двух моделей вместе

NautilusMoME (сумеречный язык, неологизмы) + PolyglotQuartet (4 языка)
работают как единый ансамбль. Все 4 режима:
  ① Дистилляция  ② Квинтет  ③ Общая память  ④ Переключение

Запуск:
    python scripts/test_ensemble_dialogue.py
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'yijing_transformer', 'scripts'))

import torch
import torch.nn.functional as F

from yijing_transformer.models.polyglot import build_polyglot, VOCABS
from yijing_transformer.models.polyglot_ensemble import (
    PolyglotEnsemble, EnsembleConfig, build_ensemble, TwilightMusician,
)

# NautilusMoME из скрипта
from train_nautilus_mome import NautilusMoME


# ═══════════════════════════════════════════════════════════════
# BPE-токенизатор (sentencepiece)
# ═══════════════════════════════════════════════════════════════

TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'yijing_transformer', 'bpe_tokenizer.model',
)

def load_bpe_tokenizer():
    """Загрузить обученный BPE-токенизатор."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_PATH)
        print(f"  BPE-токенизатор загружен: {sp.get_piece_size()} токенов")
        return sp
    except ImportError:
        print("  sentencepiece не установлен — используем CharTokenizer")
        return None
    except Exception as e:
        print(f"  Ошибка загрузки BPE: {e} — используем CharTokenizer")
        return None


class CharTokenizer:
    """Простой символьный токенизатор (fallback)."""
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [min(ord(ch) + 2, self.vocab_size - 1) for ch in text]

    def decode(self, ids):
        return ''.join(chr(max(i - 2, 0)) for i in ids if i > 1)

    def get_piece_size(self):
        return self.vocab_size


class TokenizerWrapper:
    """Обёртка для унификации интерфейса BPE и Char токенизаторов."""
    def __init__(self, sp=None, vocab_size=4096):
        self.sp = sp
        self.fallback = CharTokenizer(vocab_size)
        self.vocab_size = sp.get_piece_size() if sp else vocab_size

    def encode(self, text):
        if self.sp:
            return self.sp.encode(text)
        return self.fallback.encode(text)

    def decode(self, ids):
        if self.sp:
            return self.sp.decode(ids)
        return self.fallback.decode(ids)


# ═══════════════════════════════════════════════════════════════
# Загрузка моделей
# ═══════════════════════════════════════════════════════════════

MOME_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'yijing_transformer', 'train_mome_checkpoint.pt',
)

def load_nautilus():
    """Загрузить обученную NautilusMoME."""
    print("\n  Загрузка NautilusMoME...")
    ckpt = torch.load(MOME_CHECKPOINT, map_location='cpu', weights_only=False)
    args = ckpt.get('args', {})

    model = NautilusMoME(
        vocab_size=args.get('vocab_size', 4096),
        d_model=args.get('d_model', 128),
        n_layers=args.get('n_layers', 4),
        n_heads=args.get('n_heads', 4),
        block_size=args.get('block_size', 256),
        d_expert=args.get('d_expert', 128),
        n_experts=args.get('n_experts', 6),
        top_k=args.get('top_k', 2),
    )

    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  NautilusMoME: d={model.d_model}, vocab={model.vocab_size}, "
          f"experts={list(model.experts.keys())}")
    print(f"  Параметры: {n_params:,}, обучена {ckpt.get('step', '?')} шагов")
    return model


def create_quartet(vocab_size=4096):
    """Создать PolyglotQuartet (новую, необученную)."""
    print("\n  Создание PolyglotQuartet...")
    model = build_polyglot(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        block_size=256,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  PolyglotQuartet: d=128, vocab={vocab_size}")
    print(f"  Параметры: {n_params:,}")
    return model


# ═══════════════════════════════════════════════════════════════
# Мини-обучение ансамбля
# ═══════════════════════════════════════════════════════════════

CORPUS = [
    # Физика
    "Энергия равна массе умноженной на скорость света в квадрате.",
    "Свет — это электромагнитная волна, которая распространяется в вакууме.",
    "Гравитация притягивает все тела друг к другу.",
    "Атом состоит из ядра и электронов.",
    "Энтропия замкнутой системы никогда не убывает.",
    # Математика
    "Число пи — это отношение длины окружности к её диаметру.",
    "Сумма углов треугольника равна ста восьмидесяти градусам.",
    "Простое число делится только на единицу и на самого себя.",
    "Бесконечность — не число, а концепция безграничного продолжения.",
    # Философия
    "Я мыслю, следовательно я существую — сказал Декарт.",
    "Мудрость начинается с признания собственного незнания.",
    "Всё течёт, всё меняется — говорил Гераклит.",
    "Свобода — это осознанная необходимость.",
    # Мифология
    "Феникс возрождается из собственного пепла каждые пятьсот лет.",
    "Мировое дерево соединяет три мира: верхний, средний и нижний.",
    "Дракон охраняет сокровища знания в глубине горы.",
    # Музыка
    "Музыка — это математика, обращённая в звук.",
    "Гармония возникает когда различные голоса звучат как одно целое.",
    "Каждая нота — это частота вибрации воздуха.",
    "Тишина между нотами так же важна, как сами ноты.",
]


def train_ensemble(ensemble, tokenizer, corpus, steps=200, lr=2e-3, mode='quintet'):
    """Быстрое обучение ансамбля на мини-корпусе."""
    # Только обучаемые параметры
    trainable = [p for p in ensemble.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    block_size = 64
    all_ids = []
    for text in corpus:
        ids = tokenizer.encode(text)
        if len(ids) < 4:
            continue
        all_ids.append(ids)

    ensemble.train()
    losses = []
    t0 = time.time()

    for step in range(steps):
        # Случайный батч
        batch_idx = [random.randint(0, len(all_ids) - 1) for _ in range(4)]
        xs, ys = [], []
        for bi in batch_idx:
            ids = all_ids[bi]
            padded = (ids + [0] * block_size)[:block_size + 1]
            xs.append(padded[:block_size])
            ys.append(padded[1:block_size + 1])

        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)

        logits, loss, info = ensemble(x, targets=y, mode=mode)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0 or step == steps - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            extra = ''
            if 'kl_loss' in info:
                extra = f" KL={info['kl_loss']:.4f}"
            if 'twilight' in info:
                tw = info['twilight']
                extra += f" tw_vol={tw.get('volume', 0):.3f}"
            if 'router' in info:
                r = info['router']
                extra += f" n_w={r['nautilus_weight']:.3f} q_w={r['quartet_weight']:.3f}"
            if 'memory' in info:
                m = info['memory']
                extra += f" mem_g={m.get('gate_quartet', 0):.3f}"
            print(f"  шаг {step:4d}/{steps} | loss={avg:.3f}{extra} | {elapsed:.1f}с")

    return losses


# ═══════════════════════════════════════════════════════════════
# Генерация и диалог
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(ensemble, tokenizer, prompt, max_len=80, temperature=0.8, mode=None):
    """Авторегрессивная генерация через ансамбль."""
    ensemble.eval()
    mode = mode or ensemble.config.mode
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_len):
        idx_cond = idx[:, -64:]
        logits, _, _ = ensemble(idx_cond, mode=mode)
        logits = logits[:, -1, :] / temperature

        # Top-k
        top_k = 40
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        decoded_ch = tokenizer.decode([next_id.item()])
        if decoded_ch and decoded_ch[-1] in '.!?':
            break

    return tokenizer.decode(idx[0].tolist())


@torch.no_grad()
def generate_nautilus_solo(nautilus, tokenizer, prompt, max_len=80, temperature=0.8):
    """Генерация только через NautilusMoME (для сравнения)."""
    nautilus.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_len):
        idx_cond = idx[:, -256:]
        logits, _, info = nautilus(idx_cond)
        logits = logits[:, -1, :] / temperature

        top_k = 40
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        decoded_ch = tokenizer.decode([next_id.item()])
        if decoded_ch and decoded_ch[-1] in '.!?':
            break

    twilight_info = info.get('twilight', {})
    return tokenizer.decode(idx[0].tolist()), twilight_info


@torch.no_grad()
def ask_quartet_musicians(ensemble, tokenizer, prompt):
    """Ответы 4 музыкантов квартета на своих языках."""
    ensemble.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    # Прогоняем через квартет напрямую
    logits, _, info = ensemble.quartet(idx)
    results = {}

    if 'spec_logits' in info:
        for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']:
            spec = info['spec_logits'][name]
            token_ids = spec.argmax(dim=-1)[0].tolist()
            if name in VOCABS:
                decoded = VOCABS[name].decode_str(token_ids)
            else:
                decoded = tokenizer.decode(token_ids)
            results[name] = decoded

    return results


# ═══════════════════════════════════════════════════════════════
# Вопросы для диалога
# ═══════════════════════════════════════════════════════════════

QUESTIONS = [
    "Что такое энергия",
    "Музыка это математика",
    "Свет и тьма",
    "Мудрость начинается",
    "Феникс возрождается",
    "Всё течёт всё меняется",
    "Атом состоит из",
    "Свобода это",
    "Гармония различных голосов",
    "Бесконечность",
]

DISPLAY_NAMES = {
    'formalist': 'ФОРМАЛИСТ (формулы)',
    'archetypist': 'АРХЕТИПИСТ (архетипы)',
    'algorithmist': 'АЛГОРИТМИСТ (графы)',
    'linguist': 'ЛИНГВИСТ (текст)',
}


# ═══════════════════════════════════════════════════════════════
# Главный скрипт
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("  АНСАМБЛЬ: NautilusMoME + PolyglotQuartet")
    print("  Сумеречный язык встречает Четыре Языка Истины")
    print("=" * 75)

    # ── 1. Загрузка ──
    sp = load_bpe_tokenizer()
    tokenizer = TokenizerWrapper(sp, vocab_size=4096)

    nautilus = load_nautilus()
    quartet = create_quartet(vocab_size=4096)

    # ── 2. NautilusMoME соло (до ансамбля) ──
    print("\n" + "=" * 75)
    print("  ЭТАП 1: NautilusMoME СОЛО (сумеречный язык)")
    print("=" * 75)

    for q in QUESTIONS[:5]:
        text, tw_info = generate_nautilus_solo(nautilus, tokenizer, q)
        tw_str = tw_info.get('twilight_strength', 0)
        blend = tw_info.get('blend_ratio', 0)
        print(f"\n  Q: «{q}»")
        print(f"  A: {text}")
        print(f"     [twilight={tw_str:.3f}, blend={blend:.3f}]")

    # ── 3. Обучение ансамбля в каждом режиме ──
    MODES = ['distill', 'quintet', 'memory', 'router']
    ensemble_results = {}

    for mode in MODES:
        print("\n" + "=" * 75)
        print(f"  ЭТАП 2: ОБУЧЕНИЕ АНСАМБЛЯ — режим '{mode.upper()}'")
        print("=" * 75)

        # Пересоздаём квартет для чистого сравнения
        quartet_fresh = create_quartet(vocab_size=4096)
        ensemble = build_ensemble(quartet_fresh, nautilus, mode=mode)

        losses = train_ensemble(
            ensemble, tokenizer, CORPUS,
            steps=150, lr=2e-3, mode=mode,
        )

        # ── Генерация ──
        print(f"\n  --- Ответы ансамбля ({mode}) ---")
        mode_results = []
        for q in QUESTIONS[:5]:
            text = generate(ensemble, tokenizer, q, mode=mode)
            print(f"\n  Q: «{q}»")
            print(f"  A: {text}")
            mode_results.append((q, text))

        ensemble_results[mode] = {
            'losses': losses,
            'results': mode_results,
        }

    # ── 4. Квинтет: музыканты + сумеречник ──
    print("\n" + "=" * 75)
    print("  ЭТАП 3: КВИНТЕТ — ВСЕ 5 МУЗЫКАНТОВ")
    print("=" * 75)

    # Берём квинтет-ансамбль (пересоздаём для полного теста)
    quartet_quintet = create_quartet(vocab_size=4096)
    ensemble_q = build_ensemble(quartet_quintet, nautilus, mode='quintet')
    train_ensemble(ensemble_q, tokenizer, CORPUS, steps=200, lr=2e-3, mode='quintet')

    print("\n  --- Полный отклик всех 5 музыкантов ---")
    for q in QUESTIONS:
        print(f"\n{'─' * 65}")
        print(f"  ВОПРОС: «{q}»")
        print(f"{'─' * 65}")

        # Авторегрессивная генерация через квинтет
        text = generate(ensemble_q, tokenizer, q, mode='quintet')
        print(f"\n  КВИНТЕТ (генерация): {text}")

        # NautilusMoME соло (для сравнения)
        nautilus_text, tw_info = generate_nautilus_solo(nautilus, tokenizer, q)
        print(f"  NAUTILUS (соло):     {nautilus_text}")
        print(f"     [twilight={tw_info.get('twilight_strength', 0):.3f}]")

        # 4 музыканта квартета
        musicians = ask_quartet_musicians(ensemble_q, tokenizer, q)
        if musicians:
            print(f"\n  Музыканты квартета:")
            for name, response in musicians.items():
                display = DISPLAY_NAMES[name]
                if len(response) > 100:
                    response = response[:100] + "..."
                print(f"    {display}: {response}")

    # ── 5. Итоговое сравнение ──
    print("\n" + "=" * 75)
    print("  СРАВНЕНИЕ РЕЖИМОВ")
    print("=" * 75)

    for mode in MODES:
        data = ensemble_results[mode]
        losses = data['losses']
        start_loss = losses[0]
        end_loss = sum(losses[-10:]) / 10
        reduction = (1 - end_loss / start_loss) * 100
        print(f"\n  {mode.upper():10s}: loss {start_loss:.3f} → {end_loss:.3f} "
              f"(снижение {reduction:.1f}%)")
        for q, a in data['results'][:3]:
            # Показываем первые 80 символов ответа
            short_a = a[:80] + "..." if len(a) > 80 else a
            print(f"    «{q}» → {short_a}")

    print("\n" + "=" * 75)
    print("  ГОТОВО")
    print("=" * 75)


if __name__ == '__main__':
    main()
