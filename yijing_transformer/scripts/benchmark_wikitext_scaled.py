#!/usr/bin/env python3
"""
Бенчмарк обучения на масштабе d_model=256 с синтетическими данными (WikiText-паттерны).

Сравнивает 5 конфигураций YiJing-Transformer:
  - vanilla:        без геометрии (базовый трансформер)
  - geometry_basic:  BianGua + Heisenberg Attention
  - geometry_full:   + FlowerGAT + Triangular Bias
  - expert_choice:   Expert Choice маршрутизация (8 экспертов)
  - pseudo_rag:      PseudoRAG Q4→Q6 проекционный мост

Все конфигурации: d_model=256, n_layers=6, block_size=128, 500 шагов обучения.
Cosine LR с warmup (50 шагов), оценка каждые 100 шагов.

Использование:
    python scripts/benchmark_wikitext_scaled.py

Результаты сохраняются в benchmark_wikitext_scaled_results.json.
"""

import sys
import os
import math
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT


# ── Пути ──────────────────────────────────────────────────────

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmark_wikitext_scaled_results.json'
)


# ── Воспроизводимость ────────────────────────────────────────

def set_seed(seed=42):
    """Фиксация seed для воспроизводимости."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Данные ────────────────────────────────────────────────────

def load_wikitext_data(block_size=128, seed=42):
    """
    Загрузка данных: WikiTextDataset (если доступен) или синтетические данные.

    Синтетические данные имитируют распределение WikiText:
    предложения из частотных английских слов, byte-level кодирование.

    Возвращает: (train_data, val_data, vocab_size)
    """
    # Попытка загрузить реальный WikiText
    try:
        from yijing_transformer.data_utils.wikitext_dataset import WikiTextDataset
        print("  Попытка загрузки WikiText через HuggingFace...")
        train_ds, val_ds, _ = WikiTextDataset.from_huggingface(
            name='wikitext-2-raw-v1',
            block_size=block_size,
            vocab_size=2048,
            tokenizer_type='byte',
        )
        print(f"  WikiText загружен: train={train_ds.n_tokens:,}, val={val_ds.n_tokens:,} токенов")
        return train_ds.data, val_ds.data, train_ds.vocab_size
    except Exception as e:
        print(f"  WikiText недоступен ({e}), используем синтетические данные")

    # Синтетические данные, имитирующие WikiText-паттерны
    random.seed(seed)
    words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "it", "was",
        "for", "on", "are", "with", "as", "his", "they", "be", "at", "one",
        "have", "this", "from", "by", "not", "but", "what", "all", "were", "when",
        "we", "there", "can", "an", "your", "which", "their", "said", "each", "she",
        "do", "how", "will", "up", "other", "about", "out", "many", "then", "them",
        "would", "like", "so", "these", "her", "long", "make", "thing", "see", "him",
        "two", "has", "look", "more", "day", "could", "go", "come", "did", "my",
        "no", "most", "who", "over", "know", "than", "call", "first", "people", "may",
        "down", "been", "now", "find", "any", "new", "work", "part", "take", "get",
        "place", "made", "after", "back", "only", "use", "where", "good", "very", "still",
        "also", "into", "year", "just", "because", "some", "help", "time", "much", "world",
    ]

    lines = []
    for _ in range(80000):
        n = random.randint(5, 30)
        line = ' '.join(random.choice(words) for _ in range(n))
        lines.append(line + '.')
    full_text = '\n'.join(lines)

    # Разделение 90% / 10%
    split_idx = int(len(full_text) * 0.9)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]

    train_data = torch.tensor(list(train_text.encode('utf-8')), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode('utf-8')), dtype=torch.long)
    vocab_size = 256  # byte-level

    print(f"  Синтетические данные (WikiText-паттерны): "
          f"train={len(train_data):,}, val={len(val_data):,} токенов (byte-level)")
    return train_data, val_data, vocab_size


def get_batch(data, block_size, batch_size):
    """Случайный батч из последовательности токенов."""
    n = len(data) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ── Конфигурации для бенчмарка ───────────────────────────────

BENCHMARK_CONFIGS = {
    'vanilla': dict(
        # Базовый трансформер без геометрии
        use_bian_gua=False,
        use_hex_moe=False,
        hex_strength=0.0,
    ),
    'geometry_basic': dict(
        # Базовая геометрия: BianGua + Heisenberg Attention
        use_bian_gua=True,
        use_heisenberg_attention=True,
        hex_strength=0.05,
    ),
    'geometry_full': dict(
        # Полная геометрия: + FlowerGAT + Triangular Bias
        use_bian_gua=True,
        use_heisenberg_attention=True,
        use_flower_gat=True,
        use_triangular_bias=True,
        hex_strength=0.05,
    ),
    'expert_choice': dict(
        # Expert Choice маршрутизация
        use_bian_gua=False,
        use_hex_moe=False,
        use_expert_choice=True,
        n_experts=8,
        hex_strength=0.0,
    ),
    'pseudo_rag': dict(
        # PseudoRAG Q4→Q6 проекционный мост
        use_bian_gua=False,
        use_hex_moe=False,
        use_pseudo_rag=True,
        hex_strength=0.0,
    ),
}

CONFIG_ORDER = ['vanilla', 'geometry_basic', 'geometry_full', 'expert_choice', 'pseudo_rag']


def make_config(overrides, vocab_size):
    """Создание YiJingConfig с d_model=256, n_layers=6, block_size=128."""
    base = dict(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        block_size=128,
        dropout=0.05,
        use_rope=True,
        use_swiglu=True,
        temp=0.3,
    )
    return YiJingConfig(**{**base, **overrides})


# ── Оценка ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, n_eval=20):
    """
    Вычисление перплексии на валидационных данных.

    Перплексия = exp(средний CE loss) по n_eval батчам.
    """
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, block_size, batch_size)
        logits, loss, _ = model(x, targets=y)
        if isinstance(loss, torch.Tensor):
            losses.append(loss.item())
        else:
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item())
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(min(avg_loss, 20))  # ограничение чтобы не взорвалось
    return avg_loss, ppl


# ── Обучение ──────────────────────────────────────────────────

def train_and_eval(cfg, name, train_data, val_data,
                   n_steps=500, batch_size=8, lr=3e-4, warmup_steps=50,
                   eval_every=100):
    """
    Полный цикл обучения одной конфигурации.

    Cosine LR schedule с линейным warmup.
    Оценка каждые eval_every шагов.

    Возвращает словарь с метриками:
        train_ppl, val_ppl, train_time, n_params, history
    """
    set_seed(42)
    model = YiJingGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 65}")
    print(f"  {name} ({n_params:,} параметров)")
    print(f"{'=' * 65}")

    history = []
    t0 = time.time()
    best_val_loss = float('inf')
    last_train_loss = float('inf')

    for step in range(1, n_steps + 1):
        model.train()

        # Cosine LR с линейным warmup
        if step < warmup_steps:
            cur_lr = lr * step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
            cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, cfg.block_size, batch_size)
        logits, loss, _ = model(x, targets=y)
        if not isinstance(loss, torch.Tensor):
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        last_train_loss = loss.item()

        if step % eval_every == 0 or step == 1:
            val_loss, val_ppl = evaluate(model, val_data, cfg.block_size, batch_size)
            best_val_loss = min(best_val_loss, val_loss)
            train_ppl = math.exp(min(last_train_loss, 20))
            print(f"  Шаг {step:5d}: train_loss={last_train_loss:.4f} "
                  f"train_ppl={train_ppl:.1f} val_loss={val_loss:.4f} "
                  f"val_ppl={val_ppl:.1f} lr={cur_lr:.6f}")
            history.append({
                'step': step,
                'train_loss': last_train_loss,
                'train_ppl': train_ppl,
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            })

    train_time = time.time() - t0

    # Финальные метрики
    final_train_ppl = math.exp(min(last_train_loss, 20))
    final_val_loss, final_val_ppl = evaluate(model, val_data, cfg.block_size, batch_size)
    best_val_ppl = math.exp(min(best_val_loss, 20))

    print(f"\n  ИТОГО: train_ppl={final_train_ppl:.1f} val_ppl={final_val_ppl:.1f} "
          f"best_val_ppl={best_val_ppl:.1f} время={train_time:.1f}с")

    return {
        'name': name,
        'n_params': n_params,
        'train_ppl': final_train_ppl,
        'val_ppl': final_val_ppl,
        'best_val_ppl': best_val_ppl,
        'train_time': train_time,
        'history': history,
    }


# ── Вывод результатов ─────────────────────────────────────────

def print_results_table(results):
    """Печать итоговой таблицы сравнения конфигураций."""
    print("\n" + "=" * 90)
    print("  ИТОГОВАЯ ТАБЛИЦА: Бенчмарк WikiText-Scaled (d_model=256, n_layers=6)")
    print("=" * 90)
    print(f"  {'Конфиг':<20} {'Параметры':>12} {'Train PPL':>12} "
          f"{'Val PPL':>12} {'Best Val PPL':>14} {'Время (с)':>10}")
    print("  " + "-" * 86)

    vanilla_ppl = None
    for r in results:
        if r['name'] == 'vanilla':
            vanilla_ppl = r['best_val_ppl']
            break

    for r in results:
        delta_str = ""
        if vanilla_ppl is not None and r['name'] != 'vanilla':
            delta = r['best_val_ppl'] - vanilla_ppl
            sign = "+" if delta >= 0 else ""
            delta_str = f"  ({sign}{delta:.2f})"
        print(f"  {r['name']:<20} {r['n_params']:>12,} {r['train_ppl']:>12.2f} "
              f"{r['val_ppl']:>12.2f} {r['best_val_ppl']:>14.2f} "
              f"{r['train_time']:>9.1f}s{delta_str}")

    print("  " + "-" * 86)

    # Лучшая конфигурация
    best = min(results, key=lambda r: r['best_val_ppl'])
    print(f"\n  Лучшая конфигурация: {best['name']} "
          f"(best_val_ppl={best['best_val_ppl']:.2f})")


def save_results(results):
    """Сохранение результатов в JSON."""
    output = {
        'benchmark': 'wikitext_scaled',
        'settings': {
            'd_model': 256,
            'n_layers': 6,
            'n_heads': 8,
            'block_size': 128,
            'n_steps': 500,
            'batch_size': 8,
            'lr': 3e-4,
            'warmup_steps': 50,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': {},
    }
    for r in results:
        output['results'][r['name']] = {
            'n_params': r['n_params'],
            'train_ppl': r['train_ppl'],
            'val_ppl': r['val_ppl'],
            'best_val_ppl': r['best_val_ppl'],
            'train_time': r['train_time'],
            'history': r['history'],
        }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Результаты сохранены: {RESULTS_PATH}")


# ── Точка входа ───────────────────────────────────────────────

def main():
    """Запуск полного бенчмарка: 5 конфигураций на синтетических WikiText-данных."""
    print("=" * 65)
    print("  Бенчмарк WikiText-Scaled (d_model=256, n_layers=6)")
    print("  5 конфигураций, 500 шагов, batch_size=8, lr=3e-4")
    print("=" * 65)

    # Загрузка данных
    print("\nЗагрузка данных...")
    train_data, val_data, vocab_size = load_wikitext_data(block_size=128)

    # Обучение всех конфигураций
    all_results = []
    for i, name in enumerate(CONFIG_ORDER, 1):
        print(f"\n{'#' * 65}")
        print(f"  [{i}/{len(CONFIG_ORDER)}] Конфигурация: {name}")
        print(f"{'#' * 65}")

        overrides = BENCHMARK_CONFIGS[name]
        cfg = make_config(overrides, vocab_size)
        result = train_and_eval(cfg, name, train_data, val_data,
                                n_steps=500, batch_size=8, lr=3e-4,
                                warmup_steps=50, eval_every=100)
        all_results.append(result)

    # Итоги
    print_results_table(all_results)
    save_results(all_results)

    print("\nБенчмарк завершён.")


if __name__ == '__main__':
    main()
