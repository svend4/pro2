"""
Полный тренировочный скрипт для PolyglotQuartet.

Три фазы обучения:
  ① Претрейн с учебным планом (curriculum) — от простого к сложному
  ② Кросс-перевод между четырьмя музыкантами (cycle consistency)
  ③ Дообучение с учителем (supervised fine-tuning) — заморозка формалиста и архетиписта

Использование:
    python scripts/train_polyglot_full.py
    python scripts/train_polyglot_full.py --phase pretrain --pretrain-steps 5000
    python scripts/train_polyglot_full.py --corpus-dir data/svend4_corpus --device cuda
"""

import os
import sys
import argparse
import glob as glob_mod
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Добавляем корень проекта в sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from yijing_transformer.models.polyglot import build_polyglot, PolyglotQuartet
from yijing_transformer.models.polyglot_curriculum import (
    CurriculumTrainer,
    CurriculumScheduler,
    build_curriculum_trainer,
)
from yijing_transformer.models.polyglot_translation import (
    CrossTranslator,
    MUSICIAN_NAMES,
)
from yijing_transformer.models.polyglot_supervised import (
    SupervisedTrainer,
    SupervisedConfig,
    build_supervised_trainer,
)


# ═══════════════════════════════════════════════════════════════
# Символьный токенизатор (ord / chr)
# ═══════════════════════════════════════════════════════════════

class CharTokenizer:
    """Простой посимвольный токенизатор: символ -> ord(), обратно -> chr().

    Нулевой индекс зарезервирован под <pad>.
    Все коды сдвинуты на +1, чтобы 0 оставался паддингом.
    """

    def __init__(self, vocab_size: int = 4096):
        # Максимальный допустимый код символа
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Кодируем строку в список целых чисел (ord + 1, с ограничением)."""
        ids = []
        for ch in text:
            code = ord(ch) + 1  # сдвиг: 0 зарезервирован под <pad>
            if code >= self.vocab_size:
                code = 1  # <unk> — выход за пределы словаря
            ids.append(code)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Декодируем список целых обратно в строку."""
        chars = []
        for code in ids:
            if code <= 0:
                continue  # пропускаем <pad>
            try:
                chars.append(chr(code - 1))
            except (ValueError, OverflowError):
                chars.append('?')
        return ''.join(chars)


# ═══════════════════════════════════════════════════════════════
# Загрузка корпуса
# ═══════════════════════════════════════════════════════════════

# Текстовые расширения — совпадают с fetch_svend4_corpus.py
_TEXT_EXTENSIONS = {".md", ".txt", ".skill", ".rst"}


def load_corpus(corpus_dir: str) -> str:
    """Загружаем все текстовые файлы из директории корпуса svend4.

    Рекурсивно обходит corpus_dir, читает файлы с расширениями
    .md, .txt, .skill, .rst и объединяет их в одну строку.

    Args:
        corpus_dir: путь к директории с корпусом (например data/svend4_corpus)

    Returns:
        объединённый текст всех файлов корпуса
    """
    if not os.path.isdir(corpus_dir):
        print(f"[ОШИБКА] Директория корпуса не найдена: {corpus_dir}")
        print("  Запустите сначала: python scripts/fetch_svend4_corpus.py")
        sys.exit(1)

    # Собираем все текстовые файлы рекурсивно
    all_texts = []
    file_count = 0
    total_bytes = 0

    for root, dirs, files in os.walk(corpus_dir):
        # Пропускаем служебные директории
        dirs[:] = [d for d in dirs if d not in {
            ".git", "node_modules", "__pycache__", ".github", "_clones"
        }]
        for fname in files:
            _, ext = os.path.splitext(fname.lower())
            if ext not in _TEXT_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if content.strip():
                    all_texts.append(content)
                    file_count += 1
                    total_bytes += len(content.encode("utf-8"))
            except Exception:
                pass

    if not all_texts:
        print(f"[ОШИБКА] В {corpus_dir} не найдено текстовых файлов")
        sys.exit(1)

    # Объединяем все тексты с разделителем
    corpus = "\n\n".join(all_texts)
    print(f"  Корпус загружен: {file_count} файлов, "
          f"{total_bytes // 1024} KB, "
          f"{len(corpus)} символов")
    return corpus


# ═══════════════════════════════════════════════════════════════
# Фаза 1: Претрейн с учебным планом (Curriculum)
# ═══════════════════════════════════════════════════════════════

def phase1_pretrain(
    model: PolyglotQuartet,
    corpus: str,
    args: argparse.Namespace,
) -> PolyglotQuartet:
    """Предварительное обучение модели с учебным планом.

    Используем CurriculumTrainer: подаём примеры от простых к сложным.
    Три ступени — лёгкое, среднее, экспертное.

    Args:
        model: модель PolyglotQuartet
        corpus: объединённый текст корпуса
        args: аргументы командной строки

    Returns:
        обученная модель
    """
    print("\n" + "=" * 70)
    print("  ФАЗА 1: ПРЕТРЕЙН С УЧЕБНЫМ ПЛАНОМ (CURRICULUM)")
    print("=" * 70)

    block_size = model.cfg.block_size
    device = args.device
    model = model.to(device)

    # Создаём токенизатор
    tokenizer = CharTokenizer(vocab_size=model.cfg.vocab_size)

    # Создаём тренер с учебным планом из корпуса
    print("\n  Подготовка учебного плана...")
    trainer = build_curriculum_trainer(
        corpus=corpus,
        block_size=block_size,
        device=device,
        log_every=100,
    )

    # Настраиваем оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )

    # Запускаем обучение
    total_steps = args.pretrain_steps
    print(f"\n  Запуск обучения: {total_steps} шагов, "
          f"lr={args.lr}, batch_size={args.batch_size}")
    print(f"  Устройство: {device}")
    print()

    stats = trainer.train_epoch(
        model=model,
        optimizer=optimizer,
        total_steps=total_steps,
        current_step=0,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=args.batch_size,
        grad_clip=1.0,
    )

    # Итоговая статистика
    if stats:
        final_loss = stats[-1].loss
        print(f"\n  Фаза 1 завершена: финальный loss = {final_loss:.4f}")
    else:
        print("\n  Фаза 1 завершена (без статистики)")

    # Сохраняем чекпоинт после фазы 1
    _save_checkpoint(model, args.save_path, phase="pretrain")

    return model


# ═══════════════════════════════════════════════════════════════
# Фаза 2: Кросс-перевод между музыкантами
# ═══════════════════════════════════════════════════════════════

def phase2_translate(
    model: PolyglotQuartet,
    args: argparse.Namespace,
) -> PolyglotQuartet:
    """Обучение голов перевода между четырьмя музыкантами.

    Используем CrossTranslator: тренируем 12 направлений перевода
    (4 музыканта x 3 направления) с помощью cycle_loss —
    перевод по кругу должен вернуть исходное представление.

    Args:
        model: предобученная модель PolyglotQuartet
        args: аргументы командной строки

    Returns:
        модель (без изменений — переводчик обучается отдельно)
    """
    print("\n" + "=" * 70)
    print("  ФАЗА 2: КРОСС-ПЕРЕВОД МЕЖДУ МУЗЫКАНТАМИ")
    print("=" * 70)

    device = args.device
    model = model.to(device)
    model.eval()  # модель заморожена, обучаем только переводчик

    # Создаём кросс-переводчик
    d_model = model.cfg.d_model
    translator = CrossTranslator(
        d_model=d_model,
        linguist_vocab_size=model.cfg.vocab_size,
    ).to(device)

    # Оптимизатор для переводчика
    optimizer = torch.optim.AdamW(
        translator.parameters(),
        lr=args.lr * 0.5,  # пониженная скорость для стабильности
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )

    total_steps = args.translate_steps
    block_size = model.cfg.block_size
    tokenizer = CharTokenizer(vocab_size=model.cfg.vocab_size)

    print(f"\n  Запуск обучения переводчика: {total_steps} шагов")
    print(f"  Устройство: {device}")
    print()

    translator.train()

    for step in range(total_steps):
        # Генерируем случайный вход для извлечения скрытых состояний
        # Используем случайные токены как «стимул» для музыкантов
        rand_ids = torch.randint(
            1, model.cfg.vocab_size,
            (args.batch_size, block_size),
            device=device,
        )

        # Извлекаем скрытые состояния всех музыкантов (без градиентов по модели)
        with torch.no_grad():
            hiddens_dict = _extract_musician_hiddens(model, rand_ids)

        # Для cycle_loss нужны скрытые состояния с градиентами по переводчику
        # Отсоединяем от графа модели, но позволяем переводчику строить свой граф
        hiddens_detached = {
            name: h.detach().requires_grad_(False)
            for name, h in hiddens_dict.items()
        }

        # Вычисляем цикловую согласованность
        loss = translator.cycle_loss(hiddens_detached)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
        optimizer.step()

        # Логирование
        if step % 100 == 0 or step == total_steps - 1:
            print(f"  шаг {step:5d}/{total_steps} | cycle_loss = {loss.item():.6f}")

    print(f"\n  Фаза 2 завершена: финальный cycle_loss = {loss.item():.6f}")

    # Сохраняем переводчик вместе с моделью
    _save_checkpoint(model, args.save_path, phase="translate", translator=translator)

    return model


def _extract_musician_hiddens(
    model: PolyglotQuartet,
    idx: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Извлекаем скрытые состояния всех четырёх музыкантов.

    Прогоняем вход через все раунды репетиций и возвращаем
    скрытые состояния последнего раунда.

    Args:
        model: модель PolyglotQuartet
        idx: (B, T) — входные токены

    Returns:
        словарь {musician_name: (B, T, d_model)}
    """
    B, T = idx.shape
    device = idx.device

    tok = model.tok_emb(idx)
    pos = model.pos_emb[:, :T, :]
    shared_state = model.emb_drop(tok + pos)

    attn_mask = torch.triu(
        torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
    )

    hiddens_dict = {}
    for r, conductor in enumerate(model.conductors):
        contributions = []
        for name in model._musician_order:
            musician = model.musicians[name]
            contrib, hidden = musician(shared_state, attn_mask=attn_mask)
            contributions.append(contrib)
            # Запоминаем скрытые состояния последнего раунда
            if r == len(model.conductors) - 1:
                hiddens_dict[name] = hidden

        orchestrated, _ = conductor(shared_state, contributions)
        gate = torch.sigmoid(model.rehearsal_gates[r])
        shared_state = shared_state + gate * orchestrated

    return hiddens_dict


# ═══════════════════════════════════════════════════════════════
# Фаза 3: Дообучение с учителем (Supervised Fine-Tuning)
# ═══════════════════════════════════════════════════════════════

def phase3_finetune(
    model: PolyglotQuartet,
    corpus: str,
    args: argparse.Namespace,
) -> PolyglotQuartet:
    """Дообучение модели на размеченных данных.

    Замораживаем формалиста и архетиписта — они уже выучили
    свои представления. Обучаем алгоритмиста и лингвиста
    на аннотированных парах (вход -> эталонный выход).

    Args:
        model: предобученная модель PolyglotQuartet
        corpus: текст корпуса для генерации пар
        args: аргументы командной строки

    Returns:
        дообученная модель
    """
    print("\n" + "=" * 70)
    print("  ФАЗА 3: ДООБУЧЕНИЕ С УЧИТЕЛЕМ (SUPERVISED FINE-TUNING)")
    print("=" * 70)

    device = args.device
    model = model.to(device)

    block_size = model.cfg.block_size
    tokenizer = CharTokenizer(vocab_size=model.cfg.vocab_size)

    # Конфигурация: режим 'annotated', замораживаем формалиста и архетиписта
    config = SupervisedConfig(
        mode='annotated',
        learning_rate=args.lr * 0.1,  # пониженная скорость для дообучения
        warmup_steps=50,
        max_steps=args.finetune_steps,
        batch_size=args.batch_size,
        freeze_embeddings=True,
        freeze_musicians=['formalist', 'archetypist'],
        log_every=50,
    )

    # Создаём тренер дообучения
    trainer = SupervisedTrainer(student=model, config=config)

    # Выводим статистику замороженных параметров
    summary = trainer.freezer.summary()
    print(f"\n  Параметры: всего={summary['total']:,}, "
          f"обучаемых={summary['trainable']:,}, "
          f"заморожено={summary['frozen']:,} ({summary['frozen_pct']}%)")
    print(f"  Замороженные музыканты: формалист, архетипист")
    print(f"  Обучаемые музыканты: алгоритмист, лингвист")

    # Подготавливаем данные — создаём генератор батчей из корпуса
    total_steps = args.finetune_steps
    print(f"\n  Запуск дообучения: {total_steps} шагов")
    print()

    # Генератор батчей: для аннотированного режима создаём пары вход->цель
    dataloader = _build_annotated_dataloader(
        corpus=corpus,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=args.batch_size,
        num_batches=total_steps,
        device=device,
    )

    # Обучаем одну эпоху (все шаги)
    result = trainer.train_epoch(dataloader, mode='annotated')

    print(f"\n  Фаза 3 завершена: средний loss = {result['avg_loss']:.4f}")

    # Размораживаем все параметры после дообучения
    trainer.freezer.unfreeze_all()

    # Сохраняем финальный чекпоинт
    _save_checkpoint(model, args.save_path, phase="finetune")

    return model


def _build_annotated_dataloader(
    corpus: str,
    tokenizer: 'CharTokenizer',
    block_size: int,
    batch_size: int,
    num_batches: int,
    device: str,
) -> List[Dict[str, torch.Tensor]]:
    """Создаём список батчей для аннотированного обучения.

    Для каждого батча выбираем случайные фрагменты корпуса,
    кодируем их и создаём пары (вход, цель) со сдвигом на 1 токен.

    Args:
        corpus: текст корпуса
        tokenizer: посимвольный токенизатор
        block_size: длина последовательности
        batch_size: размер батча
        num_batches: количество батчей
        device: устройство для тензоров

    Returns:
        список словарей {'input_ids': ..., 'targets': ...}
    """
    import random

    corpus_len = len(corpus)
    batches = []

    for _ in range(num_batches):
        input_ids_list = []
        targets_list = []

        for _ in range(batch_size):
            # Случайный фрагмент корпуса
            max_start = max(0, corpus_len - block_size - 1)
            start = random.randint(0, max_start) if max_start > 0 else 0
            fragment = corpus[start:start + block_size + 1]

            # Кодируем
            ids = tokenizer.encode(fragment)

            # Дополняем или обрезаем до block_size + 1
            if len(ids) > block_size + 1:
                ids = ids[:block_size + 1]
            elif len(ids) < block_size + 1:
                ids = ids + [0] * (block_size + 1 - len(ids))

            input_ids_list.append(ids[:block_size])
            targets_list.append(ids[1:block_size + 1])

        batch = {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long, device=device),
            'targets': torch.tensor(targets_list, dtype=torch.long, device=device),
        }
        batches.append(batch)

    return batches


# ═══════════════════════════════════════════════════════════════
# Сохранение чекпоинтов
# ═══════════════════════════════════════════════════════════════

def _save_checkpoint(
    model: PolyglotQuartet,
    save_path: str,
    phase: str,
    translator: Optional[CrossTranslator] = None,
) -> None:
    """Сохраняем чекпоинт модели после завершения фазы.

    Args:
        model: модель PolyglotQuartet
        save_path: путь для сохранения (например checkpoints/polyglot_full.pt)
        phase: название завершённой фазы
        translator: кросс-переводчик (сохраняется при наличии)
    """
    # Создаём директорию если не существует
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Формируем имя файла с указанием фазы
    base, ext = os.path.splitext(save_path)
    phase_path = f"{base}_{phase}{ext}"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.cfg,
        'phase': phase,
    }

    if translator is not None:
        checkpoint['translator_state_dict'] = translator.state_dict()

    torch.save(checkpoint, phase_path)
    print(f"  Чекпоинт сохранён: {phase_path}")

    # Также сохраняем как «последний» чекпоинт
    torch.save(checkpoint, save_path)
    print(f"  Последний чекпоинт: {save_path}")


# ═══════════════════════════════════════════════════════════════
# Главная функция
# ═══════════════════════════════════════════════════════════════

def main():
    """Точка входа: парсинг аргументов и запуск фаз обучения."""
    parser = argparse.ArgumentParser(
        description="Полный тренировочный скрипт для PolyglotQuartet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Пути и устройства
    parser.add_argument(
        "--corpus-dir", default="data/svend4_corpus",
        help="Директория с корпусом svend4",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Устройство для обучения (cpu / cuda / mps)",
    )
    parser.add_argument(
        "--save-path", default="checkpoints/polyglot_full.pt",
        help="Путь для сохранения чекпоинтов",
    )

    # Шаги обучения
    parser.add_argument(
        "--pretrain-steps", type=int, default=3000,
        help="Количество шагов претрейна (фаза 1)",
    )
    parser.add_argument(
        "--translate-steps", type=int, default=1000,
        help="Количество шагов кросс-перевода (фаза 2)",
    )
    parser.add_argument(
        "--finetune-steps", type=int, default=500,
        help="Количество шагов дообучения (фаза 3)",
    )

    # Гиперпараметры
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Базовая скорость обучения",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Размер батча",
    )
    parser.add_argument(
        "--d-model", type=int, default=128,
        help="Размерность модели (d_model)",
    )

    # Выбор фазы
    parser.add_argument(
        "--phase", default="all",
        choices=["all", "pretrain", "translate", "finetune"],
        help="Какую фазу запустить (all — все три последовательно)",
    )

    args = parser.parse_args()

    # ── Заголовок ──
    print("=" * 70)
    print("  ПОЛНОЕ ОБУЧЕНИЕ POLYGLOT QUARTET")
    print("=" * 70)
    print(f"  Корпус:           {args.corpus_dir}")
    print(f"  Устройство:       {args.device}")
    print(f"  d_model:          {args.d_model}")
    print(f"  Скорость обучения:{args.lr}")
    print(f"  Размер батча:     {args.batch_size}")
    print(f"  Фаза:             {args.phase}")
    print(f"  Чекпоинт:         {args.save_path}")
    print()

    # ── Загружаем корпус ──
    need_corpus = args.phase in ("all", "pretrain", "finetune")
    corpus = None
    if need_corpus:
        print("  Загрузка корпуса...")
        corpus = load_corpus(args.corpus_dir)

    # ── Строим модель ──
    print("\n  Создание модели PolyglotQuartet...")
    model = build_polyglot(
        vocab_size=4096,
        d_model=args.d_model,
        n_layers=2,
    )

    # Выводим количество параметров
    param_counts = model.count_parameters()
    print(f"  Параметры модели: {param_counts['total']:,} всего")
    for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']:
        print(f"    {name}: {param_counts[name]:,}")
    print(f"    дирижёры:    {param_counts['conductors']:,}")
    print(f"    розетта:     {param_counts['rosetta']:,}")
    print(f"    эмбеддинги:  {param_counts['embeddings']:,}")

    # ── Запуск фаз ──
    run_pretrain = args.phase in ("all", "pretrain")
    run_translate = args.phase in ("all", "translate")
    run_finetune = args.phase in ("all", "finetune")

    if run_pretrain:
        model = phase1_pretrain(model, corpus, args)

    if run_translate:
        model = phase2_translate(model, args)

    if run_finetune:
        model = phase3_finetune(model, corpus, args)

    # ── Итого ──
    print("\n" + "=" * 70)
    print("  ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)

    phases_done = []
    if run_pretrain:
        phases_done.append(f"претрейн ({args.pretrain_steps} шагов)")
    if run_translate:
        phases_done.append(f"кросс-перевод ({args.translate_steps} шагов)")
    if run_finetune:
        phases_done.append(f"дообучение ({args.finetune_steps} шагов)")

    print(f"  Выполненные фазы: {', '.join(phases_done)}")
    print(f"  Финальный чекпоинт: {args.save_path}")
    print()


if __name__ == "__main__":
    main()
