#!/usr/bin/env python3
"""
train_on_info_corpus.py — Тренировка PolyglotQuartet на корпусе svend4/info

Загружает все 39 документов из https://github.com/svend4/info
и тренирует Полиглот-Квартет на этом богатом русскоязычном корпусе:
  - Числовые системы и символизм
  - Природные системы (элементы, цвета, времена года)
  - Фундаментальные законы физики
  - Философские концепции
  - Мифология народов мира
  - Музыкальные системы
  - Медицина и здоровье
  - Технологии и компьютеры
  ... и многое другое.

Запуск:
    python -m yijing_transformer.scripts.train_on_info_corpus
"""

import sys
import os
import time
import math
import urllib.request
import urllib.parse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from yijing_transformer.models.polyglot import (
    build_polyglot, PolyglotQuartet, VOCABS,
    FORMULA_VOCAB, ARCHETYPE_VOCAB, GRAPH_VOCAB,
)
from yijing_transformer.tokenizer.char_tokenizer import CharTokenizer


# ═══════════════════════════════════════════════════════════════
# Corpus Loader — загрузка из GitHub
# ═══════════════════════════════════════════════════════════════

# Все файлы из svend4/info repo
INFO_FILES = [
    # 01 — Числовые системы
    "01-числовые-системы/01-числа-от-1-до-10.md",
    "01-числовые-системы/02-числа-от-11-до-20.md",
    "01-числовые-системы/03-числа-до-100.md",
    "01-числовые-системы/04-числа-до-1000.md",
    # 02 — Природные системы
    "02-природные-системы/01-цвета-радуги.md",
    "02-природные-системы/02-четыре-элемента.md",
    "02-природные-системы/03-четыре-времени-года.md",
    # 03 — Временные системы
    "03-временные-системы/01-двенадцать-месяцев.md",
    "03-временные-системы/02-семь-дней-недели.md",
    # 04 — Эпохи
    "04-эпохи-и-периоды/01-средневековье.md",
    "04-эпохи-и-периоды/02-двадцатый-век.md",
    "04-эпохи-и-периоды/03-современность-21-век.md",
    "04-эпохи-и-периоды/04-фантастика-будущее.md",
    # 05 — Концепции и законы
    "05-концепции-и-законы/01-фундаментальные-законы.md",
    "05-концепции-и-законы/02-философские-концепции.md",
    # 06 — Классификации
    "06-классификации/01-классификация-по-количеству.md",
    "06-классификации/02-системы-в-музыке.md",
    "06-классификации/03-географические-системы.md",
    "06-классификации/04-системы-измерения.md",
    # 07 — Разное
    "07-разное/01-мировые-религии.md",
    "07-разное/02-астрономические-системы.md",
    "07-разное/03-биологическая-классификация.md",
    "07-разное/04-системы-письменности.md",
    "07-разное/05-языковые-семьи.md",
    "07-разное/06-политические-системы.md",
    "07-разное/07-экономические-системы.md",
    "07-разное/08-художественные-направления.md",
    "07-разное/09-архитектурные-стили.md",
    "07-разное/10-литературные-жанры.md",
    "07-разное/11-музыкальные-жанры.md",
    "07-разное/12-мировая-мифология.md",
    # 08 — Наука
    "08-наука-и-открытия/01-великие-открытия.md",
    "08-наука-и-открытия/02-великие-ученые.md",
    "08-наука-и-открытия/03-научные-дисциплины.md",
    # 09 — Медицина
    "09-медицина-и-здоровье/01-системы-организма.md",
    "09-медицина-и-здоровье/02-здоровье-и-профилактика.md",
    "09-медицина-и-здоровье/03-распространённые-заболевания.md",
    # 10 — Технологии
    "10-технологии-и-компьютеры/01-история-компьютеров.md",
    "10-технологии-и-компьютеры/02-языки-программирования.md",
]

REPO_BASE = "https://raw.githubusercontent.com/svend4/info/main/"
CACHE_DIR = Path("/home/user/pro2/data/info_corpus")


def download_corpus(max_retries: int = 3) -> str:
    """Скачивает все файлы из svend4/info и объединяет в один корпус."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = CACHE_DIR / "combined_corpus.txt"

    # Проверяем кеш
    if combined_path.exists() and combined_path.stat().st_size > 10000:
        print(f"  Используем кешированный корпус: {combined_path}")
        return combined_path.read_text(encoding='utf-8')

    all_texts = []
    downloaded = 0
    failed = 0

    for filename in INFO_FILES:
        url = REPO_BASE + urllib.parse.quote(filename)
        cache_file = CACHE_DIR / filename.replace('/', '_')

        # Проверяем локальный кеш
        if cache_file.exists() and cache_file.stat().st_size > 100:
            text = cache_file.read_text(encoding='utf-8')
            all_texts.append(text)
            downloaded += 1
            continue

        # Скачиваем
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (PolyglotQuartet Training)'
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode('utf-8')
                    cache_file.write_text(text, encoding='utf-8')
                    all_texts.append(text)
                    downloaded += 1
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  ОШИБКА: {filename}: {e}")
                    failed += 1
                else:
                    time.sleep(2 ** attempt)

    corpus = "\n\n---\n\n".join(all_texts)

    # Сохраняем объединённый корпус
    combined_path.write_text(corpus, encoding='utf-8')

    print(f"  Скачано: {downloaded}/{len(INFO_FILES)} файлов, ошибок: {failed}")
    print(f"  Размер корпуса: {len(corpus):,} символов")

    return corpus


def clean_corpus(text: str) -> str:
    """Минимальная очистка markdown-разметки для тренировки."""
    import re
    # Убираем ссылки и эмодзи-маркеры, оставляем текст
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url) → text
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)     # images
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)  # code blocks
    # Оставляем markdown заголовки (они несут информацию)
    # Убираем горизонтальные линии
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # Убираем множественные пустые строки
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_polyglot(model, tokenizer, text, n_steps=1000, lr=3e-4,
                   block_size=256, batch_size=16, device='cpu',
                   warmup_steps=100, log_every=100):
    """Тренировка полиглот-квартета на корпусе info."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    n = len(data)
    print(f"  Данные: {n:,} токенов, block_size={block_size}, batch_size={batch_size}")
    print(f"  Шагов: {n_steps}, lr={lr}, warmup={warmup_steps}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup + cosine decay
    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    losses = []
    t0 = time.time()
    best_loss = float('inf')

    for step in range(n_steps):
        model.set_step(step)

        ix = torch.randint(0, n - block_size - 1, (batch_size,), device=device)
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        logits, loss, info = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % log_every == 0 or step == n_steps - 1:
            avg = sum(losses[-log_every:]) / len(losses[-log_every:])
            ppl = math.exp(min(avg, 10))
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            ce = info.get('ce_loss', 0)
            ros = info.get('rosetta_loss', 0)
            sp = info.get('spec_loss', 0)

            marker = ' *' if avg < best_loss else ''
            if avg < best_loss:
                best_loss = avg

            print(f"  step {step:5d}/{n_steps} | loss {avg:.3f} | ppl {ppl:.1f} | "
                  f"CE={ce:.3f} Ros={ros:.4f} Spec={sp:.3f} | "
                  f"lr={lr_now:.2e} | {elapsed:.0f}s{marker}")

            if info.get('rounds') and len(info['rounds']) > 0:
                blend = info['rounds'][-1].get('blend', [])
                if blend:
                    names = ['ФОРМ', 'АРХЕ', 'АЛГО', 'ЛИНГ']
                    blend_str = ' '.join(f'{n}={w:.2f}' for n, w in zip(names, blend))
                    vols = [info['rounds'][-1].get(m, {}).get('volume', 0)
                            for m in ['formalist', 'archetypist', 'algorithmist', 'linguist']]
                    vol_str = ' '.join(f'{n}={v:.2f}' for n, v in zip(names, vols))
                    print(f"          blend: {blend_str}  vol: {vol_str}")

    return losses


# ═══════════════════════════════════════════════════════════════
# Evaluation & Demo
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def rosetta_stone(model, tokenizer, prompt, device='cpu'):
    """Розеттский камень: один промпт → четыре языка."""
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    T = x.shape[1]
    if T > model.cfg.block_size:
        x = x[:, :model.cfg.block_size]

    logits, _, info = model(x)

    results = {}

    # Лингвист
    text_ids = logits[0].argmax(dim=-1).tolist()
    results['linguist'] = tokenizer.decode(text_ids)

    # Формалист
    fv = VOCABS['formalist']
    fids = info['spec_logits']['formalist'][0].argmax(dim=-1).tolist()
    results['formalist'] = fv.decode_str(fids)

    # Архетипист
    av = VOCABS['archetypist']
    aids = info['spec_logits']['archetypist'][0].argmax(dim=-1).tolist()
    results['archetypist'] = av.decode_str(aids)

    # Алгоритмист
    gv = VOCABS['algorithmist']
    gids = info['spec_logits']['algorithmist'][0].argmax(dim=-1).tolist()
    results['algorithmist'] = gv.decode_str(gids)

    return results


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new=200,
                  temperature=0.8, top_k=50, device='cpu'):
    """Генерация текста от полного ансамбля."""
    model.eval()
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    block_size = model.cfg.block_size

    for _ in range(max_new):
        x_crop = x[:, -block_size:]
        logits, _, _ = model(x_crop)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = logits.topk(top_k)
            logits[logits < v[:, -1:]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


@torch.no_grad()
def analyze_specialization(model, tokenizer, texts, device='cpu'):
    """Анализ специализации музыкантов на разных текстах."""
    model.eval()

    musician_names = ['formalist', 'archetypist', 'algorithmist', 'linguist']
    vocabs = {
        'formalist': VOCABS['formalist'],
        'archetypist': VOCABS['archetypist'],
        'algorithmist': VOCABS['algorithmist'],
    }

    for text in texts:
        ids = tokenizer.encode(text[:model.cfg.block_size])
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _, _, info = model(x)

        print(f"\n  «{text[:70]}...»")

        for name in musician_names:
            spec_logits = info['spec_logits'][name][0]
            probs = F.softmax(spec_logits, dim=-1)

            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()

            avg_probs = probs.mean(dim=0)
            top5_vals, top5_idx = avg_probs.topk(5)

            if name in vocabs:
                top5_tokens = vocabs[name].decode(top5_idx.tolist())
            else:
                top5_tokens = [tokenizer.decode([i]) for i in top5_idx.tolist()]

            top5_str = ', '.join(
                f'"{t}"({v:.3f})' for t, v in zip(top5_tokens, top5_vals.tolist())
            )

            label = {'formalist': 'ФОРМ', 'archetypist': 'АРХЕ',
                      'algorithmist': 'АЛГО', 'linguist': 'ЛИНГ'}[name]
            print(f"    {label}: H={entropy:.2f}  [{top5_str}]")


@torch.no_grad()
def rosetta_matrix(model, tokenizer, text, device='cpu'):
    """Матрица косинусных сходств между музыкантами."""
    model.eval()
    ids = tokenizer.encode(text[:model.cfg.block_size])
    x = torch.tensor([ids], dtype=torch.long, device=device)

    tok = model.tok_emb(x)
    pos = model.pos_emb[:, :x.shape[1], :]
    shared = model.emb_drop(tok + pos)
    attn_mask = torch.triu(
        torch.ones(x.shape[1], x.shape[1], device=device, dtype=torch.bool),
        diagonal=1
    )
    hiddens = []
    for name in model._musician_order:
        _, h = model.musicians[name](shared, attn_mask=attn_mask)
        hiddens.append(h)
    _, rosetta_info = model.rosetta(hiddens)
    return rosetta_info['cosine_sims']


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # ── 1. Загрузка корпуса ──
    print("=" * 70)
    print("  ЗАГРУЗКА КОРПУСА: svend4/info")
    print("=" * 70)
    raw_corpus = download_corpus()
    corpus = clean_corpus(raw_corpus)
    print(f"  Очищенный корпус: {len(corpus):,} символов")
    print()

    # ── 2. Токенизация ──
    tokenizer = CharTokenizer.from_text(corpus)
    vocab_size = tokenizer.get_piece_size()
    print(f"  Словарь: {vocab_size} символов")

    # Покажем уникальные символы
    sample_chars = corpus[:5000]
    unique = sorted(set(sample_chars))
    print(f"  Уникальных символов (выборка): {len(unique)}")
    print()

    # ── 3. Модель ──
    print("=" * 70)
    print("  СОЗДАНИЕ МОДЕЛИ: PolyglotQuartet")
    print("=" * 70)

    model = build_polyglot(
        vocab_size=vocab_size,
        d_model=192,          # побольше для богатого корпуса
        n_layers=3,           # 3 слоя у каждого музыканта
        n_heads=3,
        n_micro_experts=6,
        micro_dim=48,
        block_size=384,       # длинный контекст
        rehearsal_rounds=2,
        rosetta_weight=0.1,
        spec_weight=0.05,
        dropout=0.1,
    ).to(device)

    params = model.count_parameters()
    print("  Параметры:")
    for k, v in params.items():
        print(f"    {k}: {v:,}")
    print()

    # ── 4. Тренировка ──
    print("=" * 70)
    print("  ТРЕНИРОВКА НА КОРПУСЕ INFO")
    print("=" * 70)
    losses = train_polyglot(
        model, tokenizer, corpus,
        n_steps=800,
        lr=3e-4,
        block_size=384,
        batch_size=12,
        device=device,
        warmup_steps=80,
        log_every=50,
    )
    print()

    # ── 5. РОЗЕТТСКИЙ КАМЕНЬ ──
    print("=" * 70)
    print("  РОЗЕТТСКИЙ КАМЕНЬ: Четыре Языка Одной Истины")
    print("=" * 70)

    # Промпты из разных областей корпуса
    prompts = [
        # Физика
        "Энергия равна массе умноженной на квадрат скорости света",
        # Философия
        "Сократ говорил я знаю что ничего не знаю",
        # Мифология
        "Зевс царь богов громовержец правит на Олимпе",
        # Музыка
        "Семь нот диатонической гаммы до ре ми фа соль ля си",
        # Элементы
        "Четыре стихии Огонь Вода Земля Воздух образуют мир",
        # Медицина
        "Сердце бьётся сто тысяч раз в день перекачивая кровь",
        # Технологии
        "Первый компьютер ENIAC весил тридцать тонн",
    ]

    musician_labels = {
        'formalist':   '① ФОРМАЛИСТ  (формулы)',
        'archetypist': '② АРХЕТИПИСТ (архетипы)',
        'algorithmist':'③ АЛГОРИТМИСТ (графы)',
        'linguist':    '④ ЛИНГВИСТ   (слова)',
    }

    for prompt in prompts:
        print(f"\n{'─' * 70}")
        print(f"  ПРОМПТ: «{prompt[:65]}»")
        print(f"{'─' * 70}")

        results = rosetta_stone(model, tokenizer, prompt, device)

        for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']:
            label = musician_labels[name]
            output = results[name][:250]
            print(f"\n  {label}:")
            print(f"    {output}")

    # ── 6. Генерация текста ──
    print()
    print("=" * 70)
    print("  ГЕНЕРАЦИЯ ТЕКСТА (полный ансамбль)")
    print("=" * 70)

    gen_prompts = [
        "Закон всемирного тяготения гласит что ",
        "В греческой мифологии ",
        "Четыре элемента ",
        "Первый закон термодинамики ",
        "Число семь символизирует ",
        "Алгоритм сортировки ",
    ]

    for prompt in gen_prompts:
        text = generate_text(model, tokenizer, prompt,
                             max_new=200, temperature=0.8, device=device)
        print(f"\n  «{prompt}...»")
        # Показываем до первого \n\n или 300 символов
        output = text[:350]
        if '\n\n' in output[len(prompt):]:
            cut = output.index('\n\n', len(prompt))
            output = output[:cut]
        print(f"  → {output}")

    # ── 7. Анализ специализации ──
    print()
    print("=" * 70)
    print("  АНАЛИЗ СПЕЦИАЛИЗАЦИИ МУЗЫКАНТОВ")
    print("=" * 70)

    analysis_texts = [
        "Формула Эйнштейна E равно mc квадрат связывает энергию и массу",
        "Дракон символизирует восток и весну в китайской мифологии",
        "Функция преобразования проецирует вектор через слои нейронной сети",
        "Слово есть мост между мыслью и миром создающий реальность",
        "Число четыре означает стабильность четыре стихии четыре стороны света",
        "Бетховен написал девять симфоний каждая уникальна до ре ми фа соль",
    ]
    analyze_specialization(model, tokenizer, analysis_texts, device)

    # ── 8. Rosetta Bridge ──
    print()
    print("=" * 70)
    print("  ROSETTA BRIDGE — Матрицы Согласия")
    print("=" * 70)

    test_texts = [
        ("Физика", "Энергия равна массе умноженной на квадрат скорости света"),
        ("Мифология", "Один отдал свой глаз за мудрость Тор защищает людей"),
        ("Музыка", "Аккорд состоит из трёх нот тоника терция квинта"),
        ("Философия", "Кант писал о категорическом императиве и вещи в себе"),
    ]

    names = ['ФОРМ', 'АРХЕ', 'АЛГО', 'ЛИНГ']

    for label, text in test_texts:
        sims = rosetta_matrix(model, tokenizer, text, device)
        print(f"\n  [{label}]: «{text[:60]}»")
        print(f"  {'':8s} {'ФОРМ':>8s} {'АРХЕ':>8s} {'АЛГО':>8s} {'ЛИНГ':>8s}")
        for i, ni in enumerate(names):
            row = f"  {ni:8s}"
            for j, nj in enumerate(names):
                if i == j:
                    row += f" {'1.000':>8s}"
                else:
                    key = f'{min(i,j)}-{max(i,j)}'
                    val = sims.get(key, 0)
                    row += f" {val:8.3f}"
            print(row)

    # ── 9. Итоги ──
    print()
    print("=" * 70)
    print("  ИТОГИ ТРЕНИРОВКИ НА КОРПУСЕ INFO")
    print("=" * 70)

    final_loss = sum(losses[-50:]) / len(losses[-50:])
    initial_loss = sum(losses[:10]) / len(losses[:10])
    print(f"""
  Корпус: svend4/info ({len(corpus):,} символов, {vocab_size} уникальных)
  Модель: PolyglotQuartet ({params['total']:,} параметров)

  Тренировка:
    Начальный loss: {initial_loss:.3f} (ppl {math.exp(min(initial_loss, 10)):.1f})
    Финальный loss:  {final_loss:.3f} (ppl {math.exp(min(final_loss, 10)):.1f})
    Улучшение:       {((initial_loss - final_loss) / initial_loss * 100):.1f}%

  Четыре Музыканта:
    ① ФОРМАЛИСТ  — видит формулы: E=mc², F=ma, ΔU=Q-W
    ② АРХЕТИПИСТ — видит архетипы: ОГОНЬ→ЭНЕРГИЯ, ВОДА→ВРЕМЯ
    ③ АЛГОРИТМИСТ — видит графы: INPUT→TRANSFORM→OUTPUT
    ④ ЛИНГВИСТ   — видит слова: естественный текст

  Розеттский Мост обеспечивает семантическое согласие
  между четырьмя разными представлениями одного явления.
""")


if __name__ == '__main__':
    main()
