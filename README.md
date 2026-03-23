# pro2 — YiJing Transformer: Variant 3

Трансформер, который **думает гексаграммами и говорит словами**.

---

## Ключевая идея

```
64 гексаграммы И-Цзин  ≅  вершины гиперкуба Q6 = {-1,+1}⁶
                           ↓
                     Коды Рида-Маллера  (линейный код d=2)
                           ↓
                     Уолш-Адамар WHT    (O(n log n) спектр)
```

Это **математическое тождество**, а не аналогия. Три тысячелетия назад составители
Чжоу И перебрали все комбинации шести бинарных линий и получили 64 гексаграммы —
тот же объект, что вершины 6-мерного гиперкуба.

Модель использует эту структуру как встроенную систему координат: каждый токен
проецируется в Q6-пространство, каждый следующий шаг генерации — шаг по графу
Бян-Гуа (граф смежности гексаграмм, где рёбра = флип одного бита).

---

## Быстрый старт

```bash
# Установка
pip install -e .

# С дополнительными зависимостями
pip install -e ".[all]"

# Тесты (1688 тестов)
pytest yijing_transformer/tests/ -x -q
```

### Минимальный пример

```python
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models import YiJingGPT

config = YiJingConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
model = YiJingGPT(config)

import torch
x = torch.randint(0, 256, (1, 64))
logits = model(x)  # (1, 64, 256)
```

### Variant3 — архетипическая архитектура

```python
from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT

config = Variant3Config(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
model = Variant3GPT(config)
```

### HierarchicalMoE — иерархическая смесь экспертов

```python
from yijing_transformer.models.hierarchical_moe import (
    HierarchicalMoEConfig, HierarchicalMoE
)

config = HierarchicalMoEConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
model = HierarchicalMoE(config)
```

---

## Архитектура (Вариант 3)

### Ядро

| Компонент | Файл | Назначение |
|-----------|------|-----------|
| `Variant3GPT` | `yijing_transformer/models/variant3.py` | Главная языковая модель |
| `HierarchicalMoEFFN` | `yijing_transformer/models/hierarchical_moe.py` | 4-уровневая MoE: Q2→Q3→Q6 |
| `HexagramProjection` | `variant3.py:76-123` | Проекция токена в 64 архетипа Q6 |
| `BianGuaAttention` | `variant3.py:130-169` | Внимание с топологическим смещением Q6 |
| `TernaryGate` | `models/` | Тернарная логика {−1, 0, +1} (变爻) |

### Математический фундамент

- **8 триграмм** = вершины 3D-куба {-1, +1}³
- **64 гексаграммы** = вершины 6D-гиперкуба {-1, +1}⁶
- **Хэмминговое расстояние** — метрика семантической близости
- **Факторизованная квантизация**: гексаграмма = верхняя триграмма ⊗ нижняя триграмма (два 8-точечных softmax вместо одного 64-точечного, 3.75× ускорение)

### Три варианта архитектуры

| Вариант | Модуль | Описание |
|---------|--------|----------|
| **YiJingGPT** | `models/model.py` | Базовый трансформер с геометрической регуляризацией |
| **Variant3GPT** | `models/variant3.py` | Архетипо-центричная: HexagramProjection → BianGuaAttention → TernaryGate → ArchetypalInterlingua |
| **HierarchicalMoE** | `models/hierarchical_moe.py` | 4-уровневая иерархия Matryoshka Q2→Q3→Q6 с маршрутизацией по доменам |

### Дополнительные архитектуры

| Модуль | Описание |
|--------|----------|
| `models/hierarchical_e2.py` | E2 — 5-уровневая иерархическая архитектура |
| `models/nautilus_yijing.py` | NautilusYiJing — камерная организация Наутилуса с И-Цзин маршрутизацией |
| `models/baseline.py` | VanillaGPT — стандартный трансформер (базовая линия) |

### Гейтинг и маршрутизация

| Класс | Файл | Назначение |
|-------|------|-----------|
| `GatedPathSelector` | `geometry/routing.py` | Выбор геометрического vs стандартного пути |
| `AdaptiveGatedPathSelector` | `geometry/routing.py` | Контентно-зависимый гейт с температурой |
| `AbrialeBridgeMediator` | `geometry/routing.py` | Лучший результат PPL 1.24 (v59) |
| `ArchetypalInterlingua` | `geometry/routing.py` | 64-экспертный банк (hub-and-spoke) |
| `DynamicCurriculumController` | `geometry/routing.py` | Адаптирует силу геометрии при обучении |

### 6 доменов Q6

Каждая из 6 линий гексаграммы отвечает за семантический домен:

| Линия | Домен | Стихия | Роль |
|-------|-------|--------|------|
| 1 | GEO / CODE | 地 Земля | Структура, материал |
| 2 | HYDRO / RECON | 水 Вода | Анализ, поток |
| 3 | PYRO / SYSTEM | 火 Огонь | Трансформация, система |
| 4 | AERO / MATH | 風 Ветер | Движение, паттерн |
| 5 | COSMO / HUMAN | 空 Пустота | Связность, отношения |
| 6 | NOOS / INFO | 識 Сознание | Смысл, концепция |

Каждый домен закреплён за одним измерением Q6. Гексаграмма определяет комбинацию
активных доменов.

---

## Результаты бенчмарков (800 шагов, d=128)

| Конфигурация | Параметры | PPL |
|-------------|-----------|-----|
| vanilla baseline | 839K | 2.94 |
| BridgeOfModules v58 | 1.4M | 1.35 |
| **AbrialeBridgeMediator v59** | **1.9M** | **1.24** ★ |
| ArchetypalInterlingua v60 | 2.1M | 2.93 |
| BridgedInterlingua v61 | 2.2M | 2.92 |

Лучшая модель — `AbrialeBridgeMediator` (v59). Архетипная интерлингва (v60-v61)
пока не превышает baseline (открытая проблема: общий `trit_proj` не даёт доменам
разойтись).

---

## Обучение

### Быстрый старт

```bash
# Самообучение (три стадии: Q6-топология → корпус → фильтрованный мир)
python self_train.py

# С дополнительными лоссами (domain triplet + quality contrastive + gate entropy)
python self_train_v2.py

# Алгоритм Скарабея (figure-8 обход Q6)
python self_train_v3.py
```

### Варианты архитектуры обучения

| Скрипт | Метод | Особенность |
|--------|-------|-------------|
| `self_train.py` | 3 стадии: Self-Topology → RAG-буфер → Wild | Базовый |
| `self_train_v2.py` | То же + domain triplet + gate entropy loss | +3 вспомогательных лосса |
| `self_train_v3.py` | v2 + Stage0 = figure-8 обход Q6 | Алгоритм Скарабея |
| `bidir_train.py` | Два встречных потока, специализация ↔ генерализация | Bidirectional |
| `train_hmoe_curriculum.py` | Curriculum learning для HMoE | Поэтапный |
| `train_hmoe_staged.py` | Staged training HMoE | Multi-stage |
| `train_e2.py` | E2-архитектура | 5-level hierarchy |

---

## Inference

```bash
# Q6-эмбеддинг текста
python e2_inference.py --embed "трансформация"

# Найти похожие концепты по Хэмминг-расстоянию в Q6
python e2_inference.py --similar "гексаграмма" --n 5

# Q6-карта корпуса (64 ячейки гиперкуба)
python e2_inference.py --map

# Генерация текста
python e2_inference.py --generate "начало пути"

# Отчёт по доменам
python e2_inference.py --domain-report
```

---

## Чекпоинты

Все `.pt`-файлы в корне (≈19 MB каждый, формат `{"model_state": ...}`):

| Файл | Источник | Примечание |
|------|---------|-----------|
| `hmoe_curriculum.pt` | `train_hmoe_curriculum.py` | Curriculum-обучение |
| `hmoe_curriculum_fixed.pt` | То же, после патча STE | Fixed STE zero-gradient |
| `hmoe_fixed_joint.pt` | Joint-training с фиксом | Совместное обучение |
| `hmoe_fixed_self.pt` | Self-training с фиксом | Самообучение |
| `hmoe_joint_base.pt` | Joint-training, база | Отправная точка |
| `hmoe_self_trained_v{2,3,4}.pt` | `self_train_hmoe.py` поколения 2–4 | Результаты итераций |
| `hmoe_v{2,3,4}_self.pt` | Альтернативный путь self-training | Параллельный ряд |

Загрузка чекпоинта:
```python
import torch
ckpt = torch.load("hmoe_self_trained_v4.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
```

---

## Структура проекта

```
pro2/
├── self_train*.py           # Самообучение (v1/v2/v3)
├── bidir_train*.py          # Бидирекционное обучение
├── train_hmoe_*.py          # HMoE curriculum/staged
├── train_e2*.py             # E2-архитектура
├── e2_inference.py          # Inference API: embed/similar/generate/map
├── nautilus_inference.py    # Инференс NautilusYiJing
├── eval_hmoe.py             # Оценка HMoE
├── run_dialog_eval.py       # Оценка диалогов
├── model_test.py            # Комплексное тестирование
├── bench_moe.py             # Бенчмарк MoE
├── hmoe_*.pt                # Чекпоинты (≈19 MB каждый)
│
└── yijing_transformer/
    ├── config/
    │   └── config.py              # YiJingConfig — 100+ параметров
    ├── constants.py               # HEX_NAMES (64 гексаграммы Вэнь-Вана)
    │
    ├── models/
    │   ├── model.py               # YiJingGPT — основная модель (90KB)
    │   ├── variant3.py            # Variant3GPT — архетипо-центричный вариант
    │   ├── variant3_extensions.py # Расширения Variant3
    │   ├── hierarchical_moe.py    # HierarchicalMoE — иерархическая MoE
    │   ├── hierarchical_e2.py     # E2 — 5-уровневая иерархия
    │   ├── nautilus_yijing.py     # NautilusYiJing — камерная архитектура
    │   ├── baseline.py            # VanillaGPT (без геометрии)
    │   ├── extensions.py          # 13 экспериментальных расширений
    │   ├── lora.py                # LoRA-адаптация
    │   ├── diff_attn.py           # Дифференциальное внимание
    │   ├── prefix_tuning.py       # Prefix tuning + logit lens
    │   ├── speculative.py         # Спекулятивная декодировка
    │   ├── expert_choice.py       # Expert-Choice маршрутизация
    │   ├── export.py              # Экспорт ONNX / TorchScript
    │   │
    │   └── geometry/              # Геометрические модули (316KB)
    │       ├── core.py            # Генерация кодбуков, порядки, палацы
    │       ├── quantizers.py      # 12 типов квантизаторов
    │       ├── attention.py       # 15+ паттернов внимания
    │       ├── positional.py      # RoPE, ALiBi, кубические позиции
    │       ├── equivariant.py     # D4-эквивариантность, BianGua
    │       ├── ffn.py             # SwiGLU, TrigramMoE, DomainMoE
    │       ├── routing.py         # Маршрутизация, мосты, интерлингва
    │       ├── convergence.py     # GlyphComposer, MatrixGrammar
    │       ├── nautilus.py        # Камеры Наутилуса, MatryoshkaNautilus
    │       └── abriale.py         # Мосты Абриаль, BridgeOfCultures
    │
    ├── training/
    │   ├── train.py               # Главный тренировочный цикл
    │   ├── bridge.py              # TrainingBridge — интеграция утилит
    │   ├── optim.py               # Оптимизаторы
    │   ├── ema.py                 # Экспоненциальное скользящее среднее
    │   ├── distillation.py        # Дистилляция знаний
    │   └── utils_v*.py            # Утилиты тренировки (v12–v52)
    │
    ├── inference/
    │   ├── generate.py            # Генерация текста
    │   └── bridge_inference.py    # Инференс-мост
    │
    ├── data_utils/
    │   ├── text_dataset.py        # TextDataset, ShuffledBatchIterator
    │   ├── wikitext_dataset.py    # WikiText
    │   ├── streaming_dataset.py   # Стриминговая загрузка
    │   └── svend4_dataset.py      # Пользовательский корпус
    │
    ├── tokenizer/
    │   ├── char_tokenizer.py      # Посимвольная токенизация
    │   └── glyph_tokenizer.py     # Глифовая токенизация
    │
    ├── scripts/                   # CLI-скрипты
    │   ├── train_model.py         # yijing-train
    │   ├── wikitext_train.py      # yijing-wikitext
    │   ├── inference_cli.py       # Инференс CLI
    │   ├── downstream_finetune.py # Файнтюнинг
    │   └── ...                    # Эксперименты, визуализация
    │
    └── tests/                     # 1688 тестов
        ├── test_model_pytest.py           # Основная модель
        ├── test_variant3.py               # Variant3
        ├── test_variant3_extended.py      # Variant3 расширенные
        ├── test_geometry_pytest.py        # Геометрия
        ├── test_hierarchical_moe.py       # HierarchicalMoE
        ├── test_nautilus_yijing.py        # NautilusYiJing
        ├── test_abriale.py               # AbrialeLayer
        ├── test_multigpu_precision.py     # Multi-GPU + fp16/bf16
        └── ...                            # 18 файлов тестов
```

---

## Квантизаторы

12 типов квантизации для дискретизации в пространство гексаграмм:

| Квантизатор | Описание |
|-------------|----------|
| `YiJingQuantizer` | Базовый — ближайшая вершина гиперкуба |
| `FactoredYiJingQuantizer` | Факторизованный 8×8 (3.75× быстрее) |
| `HierarchicalQuantizer` | Многоуровневый Q2→Q3→Q6 |
| `MatryoshkaQuantizer` | Вложенная матрёшка прогрессий |
| `DeformableQuantizer` | Обучаемые деформации кодбука |
| `GumbelQuantizer` | Gumbel-Softmax (дифференцируемый) |
| `TernaryQuantizer` | Тернарный {-1, 0, +1} |
| `E8Quantizer` | Решётка E8 (240 корней) |
| `GroupedQuantizer` | Групповой |
| `AntipodalQuantizer` | Антиподальные пары |
| `PairedBitQuantizer` | Побитовая попарная |
| `FourStateQuantizer` | 4-состояния (старый + молодой инь/ян) |

---

## Конфигурация

`YiJingConfig` предоставляет 100+ параметров:

```python
from yijing_transformer.config import YiJingConfig

config = YiJingConfig(
    # Архитектура
    vocab_size=32000, d_model=512, n_layers=12, n_heads=8, block_size=1024,

    # И-Цзин специфика
    hex_strength=0.1,              # сила геометрической регуляризации
    quantizer_type='factored6',    # тип квантизатора
    use_bian_gua=True,             # бянь-гуа трансформация

    # Современные техники
    use_rope=True,                 # Rotary Position Embedding
    use_swiglu=True,               # SwiGLU FFN
    use_amp=True,                  # Automatic Mixed Precision
    use_lora=False,                # LoRA-адаптация

    # Обучение
    label_smoothing=0.1,
    dropout=0.1,
)
```

---

## Метрики качества

| Метрика | Описание | Цель |
|---------|----------|------|
| `avg_LCI_r` | Routing LCI (Laplacian Coherence Index) | π ≈ 3.14 |
| `resonance_rate` | % циклов в резонансе (LCI близко к π) | ≥ 50% |
| `kirchhoff_ok` | % циклов с Kirchhoff-балансом | ≥ 50% |
| `score` | 0.40×LCI + 0.35×resonance + 0.25×kirchhoff | выше = лучше |
| `ppl_val` | Perplexity на валидации | ниже = лучше |

---

## Тестирование

```bash
# Все тесты
pytest yijing_transformer/tests/ -x -q

# Конкретный модуль
pytest yijing_transformer/tests/test_variant3.py -x -q

# С подробным выводом
pytest yijing_transformer/tests/ -v --tb=short

# Тесты precision / device
pytest yijing_transformer/tests/test_multigpu_precision.py -x -q
```

**Покрытие:** 1688 тестов, 18 файлов. Покрыты все основные архитектуры, геометрические модули, тренировочные утилиты, edge cases (batch=1, seq=1), mixed precision (fp16/bf16), device transfer.

---

## CLI

```bash
# Обучение
yijing-train --config config.json

# WikiText обучение
yijing-wikitext --epochs 10

# Расширения
yijing-extensions

# Файнтюнинг
yijing-downstream --task sentiment
```

---

## Зависимости

- **Обязательные:** `torch>=2.0.0`
- **Данные:** `datasets`, `sentencepiece`
- **Визуализация:** `matplotlib`
- **Трекинг:** `wandb`, `tensorboard`
- **Разработка:** `pytest`

```bash
pip install -e ".[all]"   # Всё
pip install -e ".[data]"  # Только данные
pip install -e ".[viz]"   # Визуализация
```

Устройство определяется автоматически (CUDA → CPU) во всех скриптах.

---

## Документация

### Техническая

| Файл | Содержание |
|------|-----------|
| `docs/ARCHITECTURE.md` | Архитектурный справочник: Q6, HMoE, компоненты, конфигурации |
| `docs/TRAINING.md` | Руководство по обучению: скрипты, pipeline, диагностика |
| `docs/GEOMETRY.md` | Справочник модуля geometry/: квантизаторы, attention, routing, FFN |
| `docs/TRAINING_UTILS.md` | Тренировочные утилиты: TrainingBridge, оптимизаторы, мониторинг |

### Концептуальная

| Файл | Содержание |
|------|-----------|
| `CONCEPTUAL_STAGE.md` | Концептуальная карта: Q6, тождество, 19 теорем |
| `KNOWLEDGE_FRAMEWORK.md` | 4-уровневая система: Формула → Архетип → Алгоритм → Теорема |
| `yijing-transformer-concept.md` | Оригинальная концепция с математической формализацией |
| `GERMES_NOTATION.md` | Доцифровой прообраз системы (Germes vZvete) |
| `PORTAL-PROTOCOL.md` | Спецификация 7-портовой архитектуры |
| `PASSPORT.md` | Паспорт проекта |

---

## Лицензия

MIT
