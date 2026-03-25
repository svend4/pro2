"""
polyglot_curriculum.py — Учебный План (Curriculum Training) для PolyglotQuartet

Обучение от простого к сложному, как восхождение по ступеням:

  Ступень ①  (0–25%)  → Лёгкое: короткие предложения, частые слова, числа 1–10
  Ступень ②  (25–75%) → Среднее: формулы, мифы, философские тезисы
  Ступень ③  (75–100%)→ Экспертное: сложные формулы, редкие термины, плотный текст

Научная основа:
  - Curriculum Learning (Bengio et al., 2009)
  - Самоорганизованное обучение: модель осваивает простое прежде сложного
  - Улучшает стабильность и качество обучения на малых корпусах

Корпус svend4/info — русскоязычный, охватывает:
  физику, математику, философию, мифологию, музыку,
  биологию, астрономию, числовые системы и многое другое.
"""

import math
import random
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# DifficultyEstimator — Оценка Сложности Текста
# ═══════════════════════════════════════════════════════════════

# Частые русские символы (кириллица, пробел, знаки препинания)
_COMMON_CHARS = set(
    'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    ' .,!?;:-–—()«»\n\t0123456789'
)

# Маркеры доменной сложности: наличие этих паттернов повышает сложность
_COMPLEX_PATTERNS = [
    re.compile(r'[α-ωΑ-Ω]'),              # Греческие буквы
    re.compile(r'[=+\-×÷^√∫∂∑∞≈≠≤≥]'),    # Математические операторы
    re.compile(r'\b[A-Z]{2,}\b'),           # Аббревиатуры (ENIAC, NASA)
    re.compile(r'\d{4,}'),                  # Длинные числа
    re.compile(r'[²³⁴⁵⁶⁷⁸⁹]'),            # Верхние индексы
]

# Простые русские слова (маркер лёгкости)
_SIMPLE_WORDS = {
    'это', 'и', 'в', 'на', 'не', 'что', 'он', 'она', 'они',
    'как', 'но', 'из', 'по', 'для', 'от', 'до', 'так', 'все',
    'его', 'её', 'их', 'был', 'есть', 'быть', 'мир', 'год',
    'раз', 'два', 'три', 'один', 'тот', 'свой', 'весь', 'этот',
}


class DifficultyEstimator:
    """Оценщик сложности текста.

    Вычисляет сложность от 0.0 (легко) до 1.0 (трудно)
    по четырём факторам:
      1. Лексическая сложность — доля редких символов/слов
      2. Длина предложений — средняя длина предложения в символах
      3. Информационная плотность — отношение уникальных токенов к общему числу
      4. Доменная сложность — наличие формул, греческих букв, аббревиатур
    """

    def __init__(
        self,
        vocab_weight: float = 0.3,
        length_weight: float = 0.2,
        density_weight: float = 0.2,
        domain_weight: float = 0.3,
    ):
        """
        Args:
            vocab_weight: вес фактора лексической сложности
            length_weight: вес фактора длины предложений
            density_weight: вес фактора информационной плотности
            domain_weight: вес фактора доменной сложности
        """
        self.vocab_weight = vocab_weight
        self.length_weight = length_weight
        self.density_weight = density_weight
        self.domain_weight = domain_weight

        # Сумма весов для нормализации
        total = vocab_weight + length_weight + density_weight + domain_weight
        self._norm = total if total > 0 else 1.0

    def _vocab_complexity(self, text: str) -> float:
        """Доля редких (не общеупотребительных) символов в тексте."""
        if not text:
            return 0.0
        rare = sum(1 for ch in text if ch not in _COMMON_CHARS)
        return min(rare / max(len(text), 1), 1.0)

    def _sentence_length_score(self, text: str) -> float:
        """Средняя длина предложений, нормализованная в [0, 1].

        Короткие предложения (~20 символов) → 0.0
        Длинные предложения (~200+ символов) → 1.0
        """
        # Разбиваем по терминаторам предложений
        sentences = re.split(r'[.!?。\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        avg_len = sum(len(s) for s in sentences) / len(sentences)
        # Линейная шкала: 20 символов → 0.0, 200 символов → 1.0
        return max(0.0, min((avg_len - 20.0) / 180.0, 1.0))

    def _info_density(self, text: str) -> float:
        """Информационная плотность: уникальные токены / все токены.

        Высокая плотность (много уникальных слов) → сложнее.
        """
        words = text.lower().split()
        if not words:
            return 0.0
        unique = len(set(words))
        total = len(words)
        ratio = unique / total  # от 0 до 1
        # Если >80% слов уникальны → сложный текст
        # Если <30% уникальны → простой (повторяющийся)
        return max(0.0, min((ratio - 0.3) / 0.5, 1.0))

    def _domain_complexity(self, text: str) -> float:
        """Доменная сложность: наличие формул, спецсимволов, терминов."""
        if not text:
            return 0.0
        score = 0.0
        # Каждый сложный паттерн добавляет баллы
        for pattern in _COMPLEX_PATTERNS:
            matches = len(pattern.findall(text))
            score += min(matches / max(len(text) / 100, 1), 1.0)
        score /= len(_COMPLEX_PATTERNS)

        # Доля простых слов — вычитаем
        words = text.lower().split()
        if words:
            simple_ratio = sum(1 for w in words if w in _SIMPLE_WORDS) / len(words)
            score = score * (1.0 - simple_ratio * 0.5)

        return max(0.0, min(score, 1.0))

    def estimate(self, text: str) -> float:
        """Оценка сложности одного текста.

        Args:
            text: входной текст (русский)

        Returns:
            сложность от 0.0 (легко) до 1.0 (трудно)
        """
        if not text or not text.strip():
            return 0.0

        v = self._vocab_complexity(text)
        l = self._sentence_length_score(text)
        d = self._info_density(text)
        c = self._domain_complexity(text)

        score = (
            self.vocab_weight * v
            + self.length_weight * l
            + self.density_weight * d
            + self.domain_weight * c
        ) / self._norm

        return max(0.0, min(score, 1.0))

    def estimate_batch(self, texts: List[str]) -> List[float]:
        """Оценка сложности для списка текстов.

        Args:
            texts: список текстов

        Returns:
            список оценок сложности [0.0 .. 1.0]
        """
        return [self.estimate(t) for t in texts]


# ═══════════════════════════════════════════════════════════════
# CurriculumScheduler — Расписание Обучения
# ═══════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """Расписание сложности: от простого к сложному.

    Три ступени обучения:
      ① Лёгкое   (0–25% тренировки)   → сложность 0.0–0.3
      ② Среднее  (25–75% тренировки)  → сложность 0.3–0.7
      ③ Экспертное (75–100% тренировки) → сложность 0.7–1.0

    Переходы между ступенями плавные (линейная интерполяция).
    """

    def __init__(
        self,
        easy_end: float = 0.25,
        medium_end: float = 0.75,
        easy_range: Tuple[float, float] = (0.0, 0.3),
        medium_range: Tuple[float, float] = (0.3, 0.7),
        expert_range: Tuple[float, float] = (0.7, 1.0),
    ):
        """
        Args:
            easy_end: доля тренировки, когда заканчивается лёгкая ступень
            medium_end: доля тренировки, когда заканчивается средняя ступень
            easy_range: диапазон сложности на лёгкой ступени (min, max)
            medium_range: диапазон сложности на средней ступени (min, max)
            expert_range: диапазон сложности на экспертной ступени (min, max)
        """
        self.easy_end = easy_end
        self.medium_end = medium_end
        self.easy_range = easy_range
        self.medium_range = medium_range
        self.expert_range = expert_range

        # Внутреннее состояние
        self._current_step = 0
        self._total_steps = 1
        self._progress = 0.0

    def step(self, current_step: int, total_steps: int) -> None:
        """Обновить внутреннее состояние планировщика.

        Args:
            current_step: текущий шаг тренировки
            total_steps: общее число шагов тренировки
        """
        self._current_step = current_step
        self._total_steps = max(total_steps, 1)
        self._progress = current_step / self._total_steps

    @property
    def progress(self) -> float:
        """Текущий прогресс тренировки от 0.0 до 1.0."""
        return self._progress

    @property
    def stage_name(self) -> str:
        """Название текущей ступени."""
        if self._progress < self.easy_end:
            return 'лёгкое'
        elif self._progress < self.medium_end:
            return 'среднее'
        else:
            return 'экспертное'

    def get_difficulty_range(self, progress: float) -> Tuple[float, float]:
        """Диапазон допустимой сложности для данного прогресса.

        Линейная интерполяция между ступенями обеспечивает
        плавный переход (нет резких скачков сложности).

        Args:
            progress: прогресс тренировки от 0.0 до 1.0

        Returns:
            (min_difficulty, max_difficulty) — допустимый диапазон
        """
        progress = max(0.0, min(progress, 1.0))

        if progress <= self.easy_end:
            # Ступень ①: Лёгкое
            # Линейно расширяем от самого лёгкого к границе easy_range
            t = progress / max(self.easy_end, 1e-8)
            min_d = self.easy_range[0]
            max_d = self.easy_range[0] + t * (self.easy_range[1] - self.easy_range[0])
            return (min_d, max_d)

        elif progress <= self.medium_end:
            # Ступень ②: Среднее
            t = (progress - self.easy_end) / max(self.medium_end - self.easy_end, 1e-8)
            min_d = self.easy_range[1] + t * (self.medium_range[0] - self.easy_range[1])
            max_d = self.easy_range[1] + t * (self.medium_range[1] - self.easy_range[1])
            return (min_d, max_d)

        else:
            # Ступень ③: Экспертное
            t = (progress - self.medium_end) / max(1.0 - self.medium_end, 1e-8)
            min_d = self.medium_range[0] + t * (self.expert_range[0] - self.medium_range[0])
            max_d = self.medium_range[1] + t * (self.expert_range[1] - self.medium_range[1])
            return (min_d, max_d)


# ═══════════════════════════════════════════════════════════════
# CurriculumDataLoader — Загрузчик с фильтрацией по сложности
# ═══════════════════════════════════════════════════════════════

class CurriculumDataLoader:
    """Загрузчик данных, фильтрующий примеры по сложности.

    При инициализации оценивает сложность всех фрагментов текста
    и индексирует их. При запросе батча возвращает фрагменты,
    попадающие в заданный диапазон сложности.

    Если примеров недостаточно — диапазон расширяется на 0.1,
    пока не наберётся достаточно.
    """

    def __init__(
        self,
        texts: List[str],
        estimator: Optional[DifficultyEstimator] = None,
        expand_step: float = 0.1,
    ):
        """
        Args:
            texts: список текстовых фрагментов
            estimator: оценщик сложности (создаётся по умолчанию)
            expand_step: шаг расширения диапазона при нехватке примеров
        """
        self.texts = list(texts)
        self.estimator = estimator or DifficultyEstimator()
        self.expand_step = expand_step

        # Предвычисляем сложность для всех текстов
        self.difficulties = self.estimator.estimate_batch(self.texts)

        # Сортированные индексы по сложности (для эффективного поиска)
        self._sorted_indices = sorted(
            range(len(self.texts)),
            key=lambda i: self.difficulties[i]
        )

    @property
    def size(self) -> int:
        """Количество текстов в загрузчике."""
        return len(self.texts)

    def difficulty_stats(self) -> Dict[str, float]:
        """Статистика сложности корпуса."""
        if not self.difficulties:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
        d = self.difficulties
        mean = sum(d) / len(d)
        std = math.sqrt(sum((x - mean) ** 2 for x in d) / max(len(d), 1))
        return {
            'min': min(d),
            'max': max(d),
            'mean': mean,
            'std': std,
        }

    def _indices_in_range(self, min_d: float, max_d: float) -> List[int]:
        """Индексы текстов, попадающих в диапазон сложности [min_d, max_d]."""
        return [
            i for i in self._sorted_indices
            if min_d <= self.difficulties[i] <= max_d
        ]

    def get_batch(
        self,
        difficulty_range: Tuple[float, float],
        batch_size: int,
    ) -> Tuple[List[str], List[float]]:
        """Получить батч текстов заданной сложности.

        Если текстов в указанном диапазоне недостаточно, диапазон
        расширяется на expand_step с каждой стороны, пока не
        наберётся batch_size примеров или не будет покрыт весь корпус.

        Args:
            difficulty_range: (min_difficulty, max_difficulty)
            batch_size: размер батча

        Returns:
            (тексты, сложности) — батч текстов и их оценки сложности
        """
        min_d, max_d = difficulty_range
        candidates = self._indices_in_range(min_d, max_d)

        # Расширяем диапазон при нехватке
        expand = 0.0
        while len(candidates) < batch_size and expand < 1.0:
            expand += self.expand_step
            expanded_min = max(0.0, min_d - expand)
            expanded_max = min(1.0, max_d + expand)
            candidates = self._indices_in_range(expanded_min, expanded_max)

        # Если всё ещё мало — берём все
        if len(candidates) < batch_size:
            candidates = list(range(len(self.texts)))

        # Случайная выборка из кандидатов
        selected = random.sample(candidates, min(batch_size, len(candidates)))

        batch_texts = [self.texts[i] for i in selected]
        batch_diffs = [self.difficulties[i] for i in selected]
        return batch_texts, batch_diffs


# ═══════════════════════════════════════════════════════════════
# CurriculumTrainer — Тренировочный цикл с учебным планом
# ═══════════════════════════════════════════════════════════════

@dataclass
class CurriculumStats:
    """Статистика одного батча в рамках учебного плана."""
    step: int = 0
    progress: float = 0.0
    stage: str = ''
    difficulty_range: Tuple[float, float] = (0.0, 1.0)
    batch_avg_difficulty: float = 0.0
    batch_min_difficulty: float = 0.0
    batch_max_difficulty: float = 0.0
    loss: float = 0.0
    ce_loss: float = 0.0
    rosetta_loss: float = 0.0
    spec_loss: float = 0.0


class CurriculumTrainer:
    """Тренер с учебным планом для PolyglotQuartet.

    Интегрирует CurriculumScheduler и CurriculumDataLoader
    в единый тренировочный цикл, обеспечивая подачу примеров
    от простых к сложным в соответствии с расписанием.

    Поддерживает корпус svend4/info — русскоязычный корпус,
    охватывающий физику, математику, философию, мифологию,
    музыку, биологию, астрономию и числовые системы.
    """

    def __init__(
        self,
        scheduler: Optional[CurriculumScheduler] = None,
        data_loader: Optional[CurriculumDataLoader] = None,
        estimator: Optional[DifficultyEstimator] = None,
        log_every: int = 50,
        device: str = 'cpu',
    ):
        """
        Args:
            scheduler: планировщик сложности (создаётся по умолчанию)
            data_loader: загрузчик данных (должен быть предоставлен перед train_epoch)
            estimator: оценщик сложности (создаётся по умолчанию)
            log_every: частота логирования (в шагах)
            device: устройство для тензоров ('cpu' или 'cuda')
        """
        self.scheduler = scheduler or CurriculumScheduler()
        self.data_loader = data_loader
        self.estimator = estimator or DifficultyEstimator()
        self.log_every = log_every
        self.device = device
        self.history: List[CurriculumStats] = []

    @classmethod
    def from_corpus(
        cls,
        corpus: str,
        block_size: int = 256,
        stride: Optional[int] = None,
        **kwargs,
    ) -> 'CurriculumTrainer':
        """Создаёт тренер из сплошного корпуса текста.

        Разбивает корпус на фрагменты длиной block_size
        с шагом stride и оценивает сложность каждого фрагмента.

        Args:
            corpus: полный текст корпуса
            block_size: размер фрагмента в символах
            stride: шаг между фрагментами (по умолчанию block_size // 2)
            **kwargs: дополнительные аргументы для CurriculumTrainer

        Returns:
            готовый к работе CurriculumTrainer
        """
        stride = stride or block_size // 2
        estimator = kwargs.pop('estimator', None) or DifficultyEstimator()

        # Нарезаем корпус на перекрывающиеся фрагменты
        fragments = []
        for start in range(0, len(corpus) - block_size + 1, stride):
            fragment = corpus[start:start + block_size]
            if fragment.strip():  # пропускаем пустые
                fragments.append(fragment)

        if not fragments:
            fragments = [corpus]

        data_loader = CurriculumDataLoader(
            texts=fragments,
            estimator=estimator,
        )

        return cls(
            data_loader=data_loader,
            estimator=estimator,
            **kwargs,
        )

    def _encode_batch(
        self,
        texts: List[str],
        tokenizer,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Кодирует батч текстов в тензоры входа и целей.

        Args:
            texts: список текстов
            tokenizer: токенизатор с методом encode()
            block_size: максимальная длина последовательности

        Returns:
            (x, y) — тензоры входа и целей, shape (B, T)
        """
        xs, ys = [], []
        for text in texts:
            ids = tokenizer.encode(text)
            # Обрезаем или дополняем до block_size + 1
            if len(ids) > block_size + 1:
                ids = ids[:block_size + 1]
            elif len(ids) < block_size + 1:
                # Дополняем нулями (pad)
                ids = ids + [0] * (block_size + 1 - len(ids))
            xs.append(ids[:block_size])
            ys.append(ids[1:block_size + 1])

        x = torch.tensor(xs, dtype=torch.long, device=self.device)
        y = torch.tensor(ys, dtype=torch.long, device=self.device)
        return x, y

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        current_step: int,
        tokenizer=None,
        block_size: int = 256,
        batch_size: int = 16,
        grad_clip: float = 1.0,
        scheduler_lr: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> List[CurriculumStats]:
        """Один «эпохальный» цикл тренировки с учебным планом.

        На каждом шаге:
          1. Планировщик определяет диапазон сложности
          2. Загрузчик выбирает батч подходящей сложности
          3. Модель обучается на этом батче
          4. Логируется статистика сложности

        Args:
            model: модель PolyglotQuartet
            optimizer: оптимизатор
            total_steps: общее число шагов за всю тренировку
            current_step: текущий глобальный шаг (начало этого вызова)
            tokenizer: токенизатор с методом encode()
            block_size: размер контекстного окна
            batch_size: размер батча
            grad_clip: максимальная норма градиентов
            scheduler_lr: планировщик скорости обучения (опционально)

        Returns:
            список CurriculumStats за все шаги этого вызова
        """
        if self.data_loader is None:
            raise RuntimeError(
                "CurriculumDataLoader не задан. "
                "Используйте CurriculumTrainer.from_corpus() или передайте data_loader."
            )
        if tokenizer is None:
            raise RuntimeError(
                "Токенизатор не задан. "
                "Передайте tokenizer в метод train_epoch()."
            )

        model.train()
        epoch_stats = []
        steps_in_epoch = total_steps - current_step

        for local_step in range(steps_in_epoch):
            global_step = current_step + local_step

            # ── Шаг 1: Определяем диапазон сложности ──
            self.scheduler.step(global_step, total_steps)
            progress = self.scheduler.progress
            diff_range = self.scheduler.get_difficulty_range(progress)

            # ── Шаг 2: Получаем батч подходящей сложности ──
            batch_texts, batch_diffs = self.data_loader.get_batch(
                difficulty_range=diff_range,
                batch_size=batch_size,
            )

            # ── Шаг 3: Кодируем и обучаем ──
            x, y = self._encode_batch(batch_texts, tokenizer, block_size)

            # Прямой проход через PolyglotQuartet
            if hasattr(model, 'set_step'):
                model.set_step(global_step)

            logits, loss, info = model(x, targets=y)

            # Обратный проход
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if scheduler_lr is not None:
                scheduler_lr.step()

            # ── Шаг 4: Собираем статистику ──
            stats = CurriculumStats(
                step=global_step,
                progress=progress,
                stage=self.scheduler.stage_name,
                difficulty_range=diff_range,
                batch_avg_difficulty=sum(batch_diffs) / max(len(batch_diffs), 1),
                batch_min_difficulty=min(batch_diffs) if batch_diffs else 0.0,
                batch_max_difficulty=max(batch_diffs) if batch_diffs else 0.0,
                loss=loss.item(),
                ce_loss=info.get('ce_loss', 0.0),
                rosetta_loss=info.get('rosetta_loss', 0.0),
                spec_loss=info.get('spec_loss', 0.0),
            )
            epoch_stats.append(stats)
            self.history.append(stats)

            # ── Логирование ──
            if global_step % self.log_every == 0 or local_step == steps_in_epoch - 1:
                _log_curriculum_step(stats)

        return epoch_stats


# ═══════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def _log_curriculum_step(stats: CurriculumStats) -> None:
    """Вывод статистики одного шага учебного плана."""
    d_lo, d_hi = stats.difficulty_range
    print(
        f"  шаг {stats.step:5d} | "
        f"ступень: {stats.stage:10s} | "
        f"сложность [{d_lo:.2f}–{d_hi:.2f}] "
        f"факт={stats.batch_avg_difficulty:.2f} | "
        f"loss {stats.loss:.3f} "
        f"(CE={stats.ce_loss:.3f} "
        f"Ros={stats.rosetta_loss:.4f} "
        f"Spec={stats.spec_loss:.3f})"
    )


def split_corpus_to_fragments(
    corpus: str,
    block_size: int = 256,
    stride: Optional[int] = None,
) -> List[str]:
    """Разбивает корпус на перекрывающиеся фрагменты.

    Используется для предобработки корпуса svend4/info
    перед созданием CurriculumDataLoader.

    Args:
        corpus: полный текст
        block_size: длина каждого фрагмента в символах
        stride: шаг (по умолчанию block_size // 2, т.е. 50% перекрытие)

    Returns:
        список текстовых фрагментов
    """
    stride = stride or block_size // 2
    fragments = []
    for start in range(0, len(corpus) - block_size + 1, stride):
        fragment = corpus[start:start + block_size]
        if fragment.strip():
            fragments.append(fragment)
    if not fragments and corpus.strip():
        fragments = [corpus]
    return fragments


def build_curriculum_trainer(
    corpus: str,
    block_size: int = 256,
    device: str = 'cpu',
    log_every: int = 50,
    **kwargs,
) -> CurriculumTrainer:
    """Фабричная функция: создаёт CurriculumTrainer из корпуса.

    Удобная обёртка для типичного сценария использования
    с корпусом svend4/info.

    Args:
        corpus: текст корпуса (все документы объединённые)
        block_size: размер контекстного окна
        device: устройство ('cpu' или 'cuda')
        log_every: частота логирования
        **kwargs: дополнительные аргументы для CurriculumScheduler

    Returns:
        настроенный CurriculumTrainer
    """
    scheduler = CurriculumScheduler(
        easy_end=kwargs.get('easy_end', 0.25),
        medium_end=kwargs.get('medium_end', 0.75),
    )

    trainer = CurriculumTrainer.from_corpus(
        corpus=corpus,
        block_size=block_size,
        scheduler=scheduler,
        device=device,
        log_every=log_every,
    )

    stats = trainer.data_loader.difficulty_stats()
    print(f"  Учебный план: {trainer.data_loader.size} фрагментов")
    print(
        f"  Сложность корпуса: "
        f"min={stats['min']:.3f} "
        f"max={stats['max']:.3f} "
        f"mean={stats['mean']:.3f} "
        f"std={stats['std']:.3f}"
    )

    return trainer
