#!/usr/bin/env python3
"""
test_grand_orchestrator.py — Тест Гранд-Оркестратора

Загружает все доступные модели и тестирует 3 режима оркестровки.
"""

import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'yijing_transformer', 'scripts'))

import torch
import torch.nn.functional as F

from yijing_transformer.models.polyglot import build_polyglot
from yijing_transformer.models.grand_orchestrator import (
    GrandOrchestrator, OrchestraConfig, build_grand_orchestrator,
)


# ═══════════════════════════════════════════════════════════════
# Загрузка моделей
# ═══════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def try_load_nautilus():
    """NautilusMoME — сумеречный язык."""
    try:
        from train_nautilus_mome import NautilusMoME
        ckpt_path = os.path.join(ROOT, 'yijing_transformer', 'train_mome_checkpoint.pt')
        if not os.path.exists(ckpt_path):
            return None, "чекпоинт не найден"
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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
        n_p = sum(p.numel() for p in model.parameters())
        return model, f"загружена, {n_p:,} п., {ckpt.get('step', '?')} шагов"
    except Exception as e:
        return None, f"ошибка: {e}"


def try_load_variant3():
    """Variant3GPT — тернарные гейты."""
    try:
        from yijing_transformer.models.variant3 import Variant3GPT, Variant3Config
        cfg = Variant3Config(vocab_size=4096, d_model=128, n_layers=2, n_heads=4, block_size=256)
        model = Variant3GPT(cfg)
        n_p = sum(p.numel() for p in model.parameters())
        return model, f"создана (новая), {n_p:,} п."
    except Exception as e:
        return None, f"ошибка: {e}"


def try_load_hierarchical_e2():
    """HierarchicalE2 — 5 уровней абстракции."""
    try:
        from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
        cfg = E2Config(vocab_size=4096, d_model=128, block_size=256, n_core=2, n_heads=4)
        model = HierarchicalE2(cfg)
        n_p = sum(p.numel() for p in model.parameters())
        return model, f"создана (новая), {n_p:,} п."
    except Exception as e:
        return None, f"ошибка: {e}"


def try_load_nautilus_yijing():
    """NautilusYiJing — MoME + геометрия."""
    try:
        from yijing_transformer.models.nautilus_yijing import NautilusYiJing, NautilusYiJingConfig
        cfg = NautilusYiJingConfig(vocab_size=4096, d_model=128, block_size=256, n_layers=4)
        model = NautilusYiJing(cfg)
        n_p = sum(p.numel() for p in model.parameters())
        return model, f"создана (новая), {n_p:,} п."
    except Exception as e:
        return None, f"ошибка: {e}"


def try_load_yijing_gpt():
    """YiJingGPT — основная геометрическая модель."""
    try:
        from yijing_transformer.models.model import YiJingGPT
        # Пробуем загрузить из чекпоинта
        ckpt_path = os.path.join(ROOT, 'yijing_transformer', 'train_real_data_checkpoint.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Создаём с параметрами из чекпоинта
            sd = ckpt.get('model', ckpt)
            if 'tok_emb.weight' in sd:
                v, d = sd['tok_emb.weight'].shape
            else:
                v, d = 4096, 128
            # Создаём конфиг как SimpleNamespace
            class Cfg:
                pass
            cfg = Cfg()
            cfg.vocab_size = v
            cfg.d_model = d
            cfg.block_size = 256
            cfg.n_layers = 4
            cfg.n_heads = max(d // 64, 2)
            cfg.ffn_mult = 4
            cfg.dropout = 0.05
            cfg.use_rope = True
            cfg.weight_tying = True
            cfg.label_smoothing = 0.0
            # Disable optional features for simplicity
            for attr in ['use_four_level_pe', 'use_cubic_pe', 'use_bidirectional_tri',
                         'use_convergence_bridge', 'use_matrix_grammar', 'use_abriale',
                         'use_nautilus', 'use_pseudo_rag', 'use_diff_attn',
                         'use_expert_choice', 'use_six_sources']:
                setattr(cfg, attr, False)

            model = YiJingGPT(cfg)
            model.load_state_dict(sd, strict=False)
            model.eval()
            n_p = sum(p.numel() for p in model.parameters())
            return model, f"загружена из чекпоинта, {n_p:,} п."
        else:
            return None, "чекпоинт не найден"
    except Exception as e:
        return None, f"ошибка: {e}"


# ═══════════════════════════════════════════════════════════════
# Токенизатор
# ═══════════════════════════════════════════════════════════════

def load_tokenizer():
    try:
        import sentencepiece as spm
        path = os.path.join(ROOT, 'yijing_transformer', 'bpe_tokenizer.model')
        sp = spm.SentencePieceProcessor()
        sp.load(path)
        return sp
    except Exception:
        return None

class Tokenizer:
    def __init__(self, sp=None):
        self.sp = sp
    def encode(self, text):
        if self.sp:
            return self.sp.encode(text)
        return [min(ord(ch) + 2, 4095) for ch in text]
    def decode(self, ids):
        if self.sp:
            return self.sp.decode(ids)
        return ''.join(chr(max(i - 2, 0)) for i in ids if i > 1)


# ═══════════════════════════════════════════════════════════════
# Корпус и обучение
# ═══════════════════════════════════════════════════════════════

CORPUS = [
    "Энергия равна массе умноженной на скорость света в квадрате.",
    "Свет — это электромагнитная волна в вакууме.",
    "Гравитация притягивает все тела друг к другу.",
    "Атом состоит из ядра и электронов.",
    "Число пи — отношение длины окружности к диаметру.",
    "Простое число делится только на единицу и на себя.",
    "Я мыслю следовательно существую — Декарт.",
    "Мудрость начинается с признания незнания.",
    "Всё течёт всё меняется — Гераклит.",
    "Свобода — осознанная необходимость.",
    "Феникс возрождается из пепла каждые пятьсот лет.",
    "Мировое дерево соединяет три мира.",
    "Музыка — математика обращённая в звук.",
    "Гармония возникает когда голоса звучат как одно целое.",
    "Тишина между нотами важна как сами ноты.",
]

QUESTIONS = [
    "Что такое энергия",
    "Музыка это",
    "Свет и тьма",
    "Мудрость начинается",
    "Феникс возрождается",
    "Всё течёт",
    "Свобода это",
    "Бесконечность",
]


def train_orchestrator(orch, tokenizer, steps=150, lr=2e-3):
    """Обучение оркестра на мини-корпусе."""
    trainable = [p for p in orch.parameters() if p.requires_grad]
    if not trainable:
        print("  Нет обучаемых параметров!")
        return []
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    block_size = 64
    all_ids = [tokenizer.encode(t) for t in CORPUS if len(tokenizer.encode(t)) > 3]

    orch.train()
    losses = []
    t0 = time.time()

    for step in range(steps):
        batch_idx = [random.randint(0, len(all_ids) - 1) for _ in range(4)]
        xs, ys = [], []
        for bi in batch_idx:
            ids = all_ids[bi]
            padded = (ids + [0] * block_size)[:block_size + 1]
            xs.append(padded[:block_size])
            ys.append(padded[1:block_size + 1])

        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)

        logits, loss, info = orch(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0 or step == steps - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = time.time() - t0
            extra = ""
            if 'router' in info:
                ri = info['router']
                weights_str = " ".join(f"{k}={v:.2f}" for k, v in ri.items()
                                       if k.startswith('w_'))
                extra = f" [{weights_str}]"
            print(f"  шаг {step:4d}/{steps} | loss={avg:.3f}{extra} | {elapsed:.1f}с")

    return losses


@torch.no_grad()
def generate_text(orch, tokenizer, prompt, max_len=60, temperature=0.8, mode=None):
    orch.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_len):
        idx_cond = idx[:, -64:]
        logits, _, _ = orch(idx_cond, mode=mode)
        logits = logits[:, -1, :] / temperature

        top_k = 40
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        ch = tokenizer.decode([next_id.item()])
        if ch and ch[-1] in '.!?':
            break

    return tokenizer.decode(idx[0].tolist())


# ═══════════════════════════════════════════════════════════════
# Главный скрипт
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("  ГРАНД-ОРКЕСТРАТОР: ВСЕ МОДЕЛИ ВМЕСТЕ")
    print("=" * 75)

    tokenizer = Tokenizer(load_tokenizer())
    print(f"  Токенизатор: {'BPE' if tokenizer.sp else 'char-level'}")

    # ── 1. Загрузка всех моделей ──
    print("\n" + "=" * 75)
    print("  ЗАГРУЗКА МОДЕЛЕЙ")
    print("=" * 75)

    models = {}
    loaders = [
        ('nautilus_mome',    try_load_nautilus),
        ('quartet',          lambda: (build_polyglot(4096, 128, 2, block_size=256), "создана, новая")),
        ('variant3',         try_load_variant3),
        ('yijing',           try_load_yijing_gpt),
        ('hierarchical_e2',  try_load_hierarchical_e2),
        ('nautilus_yijing',  try_load_nautilus_yijing),
    ]

    for name, loader in loaders:
        model, status = loader()
        if model is not None:
            models[name] = model
            print(f"  ✓ {name:20s} — {status}")
        else:
            print(f"  ✗ {name:20s} — {status}")

    if len(models) < 2:
        print("\n  Недостаточно моделей для оркестра!")
        return

    print(f"\n  Загружено {len(models)} из {len(loaders)} моделей")

    # ── 2. Тест всех 3 режимов ──
    MODES = ['blend', 'cascade', 'expert']
    results = {}

    for mode in MODES:
        print("\n" + "=" * 75)
        print(f"  РЕЖИМ: {mode.upper()}")
        print("=" * 75)

        orch = build_grand_orchestrator(
            models, mode=mode, vocab_size=4096, d_model=128,
            freeze=True, expert_top_k=3,
        )

        # Обучение
        losses = train_orchestrator(orch, tokenizer, steps=150, lr=2e-3)

        # Генерация
        print(f"\n  --- Ответы ({mode}) ---")
        mode_answers = []
        for q in QUESTIONS:
            text = generate_text(orch, tokenizer, q, mode=mode)
            short = text[:100] + "..." if len(text) > 100 else text
            print(f"  Q: «{q}»")
            print(f"  A: {short}")
            print()
            mode_answers.append((q, text))

        results[mode] = {
            'losses': losses,
            'answers': mode_answers,
        }

    # ── 3. Итоговое сравнение ──
    print("\n" + "=" * 75)
    print("  СРАВНЕНИЕ РЕЖИМОВ")
    print("=" * 75)

    for mode in MODES:
        data = results[mode]
        L = data['losses']
        if L:
            start, end = L[0], sum(L[-10:]) / min(10, len(L))
            pct = (1 - end / start) * 100
            print(f"\n  {mode.upper():10s}: loss {start:.3f} → {end:.3f} ({pct:.1f}%)")
        else:
            print(f"\n  {mode.upper():10s}: без обучения")

        for q, a in data['answers'][:3]:
            short_a = a[:80] + "..." if len(a) > 80 else a
            print(f"    «{q}» → {short_a}")

    # ── 4. Лучший режим: полный диалог ──
    best_mode = min(results, key=lambda m: results[m]['losses'][-1] if results[m]['losses'] else 999)
    print(f"\n{'=' * 75}")
    print(f"  ЛУЧШИЙ РЕЖИМ: {best_mode.upper()}")
    print(f"{'=' * 75}")

    orch_best = build_grand_orchestrator(
        models, mode=best_mode, vocab_size=4096, d_model=128,
        freeze=True, expert_top_k=3,
    )
    train_orchestrator(orch_best, tokenizer, steps=200, lr=2e-3)

    print(f"\n  --- Полный диалог ({best_mode}) ---")
    for q in QUESTIONS:
        text = generate_text(orch_best, tokenizer, q, mode=best_mode)
        print(f"\n  Q: «{q}»")
        print(f"  A: {text}")

    print(f"\n{'=' * 75}")
    print(f"  ГРАНД-ОРКЕСТРАТОР: {len(models)} моделей, 3 режима, готов")
    print(f"{'=' * 75}")


if __name__ == '__main__':
    main()
