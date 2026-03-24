# YiJing-Transformer

**Трансформерная архитектура с геометрическим индуктивным смещением на основе гексаграмм И-Цзин (Q6-гиперкуб)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Version 0.50.0](https://img.shields.io/badge/version-0.50.0-green.svg)](setup.py)

---

## Концепция

64 гексаграммы И-Цзин математически изоморфны вершинам 6-мерного гиперкуба Q6 = {-1,+1}⁶. Это не метафора — это тавтология:

- **8 триграмм** = все 8 вершин куба в R³ (группа Z₂³)
- **64 гексаграммы** = все 2⁶ = 64 вершины гиперкуба Q6

Проект исследует, как встраивание этой геометрической структуры в трансформер в качестве **заданного индуктивного смещения** (а не обучаемого) улучшает эффективность обучения и интерпретируемость модели.

```
Гексаграмма #42 (益, Приумножение):
  верхняя триграмма: ☴ Сюнь (011) = (-1, +1, +1)
  нижняя триграмма:  ☳ Чжэнь (100) = (+1, -1, -1)
  Q6-вектор: (-1, +1, +1, +1, -1, -1) ∈ {-1,+1}⁶
```

---

## Структура проекта

```
pro2/
├── yijing_transformer/        # Основной Python-пакет (v0.50.0)
│   ├── models/                # Реализации моделей
│   │   ├── geometry/          # Геометрические структуры (~9K строк)
│   │   │   ├── trigrams       # Триграммы — вершины куба {-1,+1}³
│   │   │   ├── hexagrams      # Гексаграммы — вершины Q6
│   │   │   ├── quantizers     # YiJing/Factored/E8 квантизаторы
│   │   │   └── attention      # Геометрические паттерны внимания
│   │   ├── yijing_gpt.py      # YiJingGPT (основная модель)
│   │   └── vanilla_gpt.py     # VanillaGPT (бейзлайн)
│   ├── training/              # Утилиты обучения (v12–v48)
│   ├── scripts/               # 50+ скриптов бенчмарков и экспериментов
│   ├── data_utils/            # Загрузка данных, WikiText, аугментация
│   ├── tokenizer/             # BPE токенизатор (vocab_size=4096)
│   ├── config/                # Управление конфигурацией
│   └── inference/             # Утилиты инференса
│
├── nautilus/                  # NautilusMoME — мультидоменная MoE-система
├── experiments/               # Исследовательские эксперименты
│   ├── hexlearn_router/       # Q6-маршрутизация (Хэммингово расстояние)
│   ├── solan_nautilus/        # Семантическая интеграция глифов
│   ├── palace_block_sparse/   # Разреженное palace-внимание
│   └── interlingua_fixed/     # Архетипный межъязыковой мост
│
├── pipeline.py                # 3-фазный автоматический pipeline обучения
├── bench_all.py               # Комплексный бенчмарк моделей
├── docs/                      # Дополнительная документация
├── checkpoints/               # Чекпойнты моделей
├── data/                      # Обучающий корпус (svend4_corpus)
└── passports/                 # Метаданные моделей
```

---

## Основные компоненты

### 1. YiJingGPT

GPT-архитектура с геометрическими кодбуками гексаграмм:

- Входные токены отображаются в Q6-пространство через 64 архетипных «полюса»
- Квантизация использует расстояние Хэмминга вместо евклидова
- Тензорная факторизация: квантизация к 64 точкам ≡ два softmax по 8 точкам

```python
from yijing_transformer.models import YiJingGPT

model = YiJingGPT(
    vocab_size=4096,
    d_model=128,
    n_layers=4,
    n_heads=4,
    block_size=256,
)
```

### 2. NautilusMoME (~3M параметров)

Мультидоменная языковая модель на основе **Mixture of Micro-Experts**:

```
Input → core_first → Router → [6 Experts] → NautilusBridge → core_second → Output
                                                   ↓
                                       [CrossDomainAnalogy]
                                                   ↓
                                         [ArchetypeLayer]
```

**6 специализированных экспертов:**

| Эксперт | Специализация              |
|---------|---------------------------|
| CODE    | Python, JS, React         |
| RECON   | Русский язык, документация |
| SYSTEM  | SQL, K8s, Docker          |
| MATH    | Формулы, алгоритмы        |
| HUMAN   | Общий текст               |
| INFO    | YAML, конфигурации, README |

**Параметры модели:**

| Компонент           | Параметры   |
|---------------------|-------------|
| Базовая NautilusMoME | 1,822,797  |
| CrossDomainAnalogy   | 1,083,551  |
| ArchetypeLayer       | ~120,000   |
| **Итого**            | **~3,026,000** |

Гиперпараметры: `d_model=128`, `n_layers=4`, `n_heads=4`, `block_size=256`, `vocab_size=4096`

### 3. Pipeline обучения (3 фазы)

```bash
python pipeline.py
```

| Фаза    | Модуль           | Результат              |
|---------|-----------------|------------------------|
| Фаза 1  | NautilusMoE      | Мультидоменное обучение |
| Фаза 2  | Turbine LCI-loss | Оптимизация аттрактора  |
| Фаза 3  | Benchmarking     | Оценка на всех доменах  |

---

## Установка

```bash
git clone <repo>
cd pro2

# Базовая установка
pip install -e .

# С зависимостями для данных
pip install -e ".[data]"

# Полная установка
pip install -e ".[all]"
```

**Требования:** Python 3.8+, PyTorch 2.0+

---

## Быстрый старт

### Обучение NautilusMoME

```bash
# Полный pipeline
python pipeline.py

# Только бенчмарк
python bench_all.py

# Curriculum-обучение
python train_hmoe_curriculum.py
```

### CLI-команды (после установки пакета)

```bash
# Обучение YiJingGPT
yijing-train

# Обучение на WikiText
yijing-wikitext

# Запуск всех расширений
yijing-extensions

# Downstream fine-tuning
yijing-downstream
```

### Использование API

```python
from yijing_transformer.models.geometry import generate_hexagrams, YiJingQuantizer
from yijing_transformer.inference import generate_text

# Генерация кодбука гексаграмм
hexagrams = generate_hexagrams()  # shape: (64, 6)

# Квантизация вектора к ближайшей гексаграмме
quantizer = YiJingQuantizer(d_model=128)
quantized = quantizer(embeddings)  # маппинг в Q6-пространство

# Генерация текста
output = generate_text(model, tokenizer, prompt="def hello():", max_tokens=100)
```

---

## Метрики и результаты

### Достигнутые результаты

| Конфигурация               | PPL (лучший) | Примечание                   |
|---------------------------|:------------:|------------------------------|
| Бейзлайн VanillaGPT        | 20–64        | Зависит от домена            |
| NautilusMoME v1            | 17–18        | После analogy training       |
| Best on CODE               | 5–15         | Специализированный эксперт   |
| AbrialeBridgeMediator       | **1.24**     | Лучший результат (v61)       |

### LCI-аттрактор

Экспериментально установлено: значение LCI (Loss Correlation Index) **сходится к π ≈ 3.14** как к естественному аттрактору при данной архитектуре. Превышение π невозможно — это верхняя граница, встроенная в геометрию задачи.

### Воспроизводимость

- R² > 0.998 между независимыми запусками
- Тестируется: `python test_reproducibility.py`

---

## Геометрические основания

### Сравнение: E8 vs И-Цзин (Q6) vs случайные точки

| Свойство                | E8           | Q6 (И-Цзин)               | Случайные    |
|------------------------|:------------:|:---------------------------:|:------------:|
| Размерность             | 8            | **6**                       | Любая        |
| Число точек             | 240          | **64**                      | Произвольно  |
| Группа симметрий        | Исключительная Ли | Z₂⁶ (абелева)          | Нет          |
| Тензорное разложение    | Нет          | **Есть: 8×8**               | Нет          |
| Вычислительная сложность | O(240·d)    | **O(64·d) — в 3.75x быстрее** | O(n·d)    |

**Ключевое преимущество:** Тензорная факторизация Q6 позволяет заменить softmax по 64 точкам двумя независимыми softmax по 8 точкам.

### Квантизаторы (реализованы)

| Квантизатор        | Описание                                    |
|--------------------|---------------------------------------------|
| `YiJingQuantizer`  | Базовый Q6-квантизатор                      |
| `FactoredQuantizer` | Факторизованный (верх. × нижн. триграмма)  |
| `HierarchicalQuantizer` | Иерархический (2-уровневый)           |
| `DeformableQuantizer`  | Обучаемые деформации вершин Q6           |
| `E8Quantizer`      | E8-решётка (сравнительный бейзлайн)         |

---

## Теоретические источники

Проект интегрирует результаты 6+ теоретических работ:

- **Склярова** — геометрическая теория маршрутизации
- **Фомюк** — физические основания LCI-метрики
- **Андреев** — структура информационных потоков
- **Касаткин** — теория маршрутизатора
- **Герман** — формализм тернарной нотации GERMES {+, <, −}
- **Беляев** — межмодальный перенос
- **SOLAN** — семантические глифы и символическое представление

---

## Документация

| Файл | Содержание |
|------|-----------|
| [yijing-transformer-concept.md](yijing-transformer-concept.md) | Математическая формализация |
| [IMPLEMENTATION_ARTICLE.md](IMPLEMENTATION_ARTICLE.md) | NautilusMoME v1→v2 roadmap |
| [theoretical-foundations.md](theoretical-foundations.md) | Теоретические основы |
| [archetypal-interlingua-theory.md](archetypal-interlingua-theory.md) | Архетипный межъязыковой мост |
| [GERMES_NOTATION.md](GERMES_NOTATION.md) | Тернарная нотация GERMES |
| [REPORT-v60-v61-status.md](REPORT-v60-v61-status.md) | Актуальный статус (v60-v61) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Детальная архитектура |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Пошаговое руководство |

---

## Статус проекта

**Версия:** 0.50.0 (Advanced Alpha)

**Проверено и подтверждено:**
- Геометрические основы (триграммы/гексаграммы)
- Все квантизаторы (YiJing, Factored, Hierarchical, Deformable, E8)
- MoE-маршрутизация при малом масштабе
- CrossDomainAnalogy (36 пар экспертов)
- Воспроизводимость R² > 0.998

**В работе:**
- NautilusMoME v2 (Nautilus-Sharp)
- Интеграция meta-репозитория (hexnet, hexlearn, hexsym, hexphys)
- AbrialeBridgeMediator (PPL 1.24 → улучшение)

---

## Лицензия

MIT License
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

# Тесты (1987+ тестов)
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
    └── tests/                     # 1987+ тестов
        ├── test_model_pytest.py           # Основная модель
        ├── test_variant3.py               # Variant3
        ├── test_variant3_extended.py      # Variant3 расширенные
        ├── test_geometry_pytest.py        # Геометрия
        ├── test_hierarchical_moe.py       # HierarchicalMoE
        ├── test_nautilus_yijing.py        # NautilusYiJing
        ├── test_abriale.py               # AbrialeLayer
        ├── test_multigpu_precision.py     # Multi-GPU + fp16/bf16
        ├── test_variant3_extensions.py       # Variant3 расширения
        ├── test_matryoshka_pytest.py         # Matryoshka-квантизация
        ├── test_matryoshka_nautilus_pytest.py # MatryoshkaNautilus
        ├── test_ternary_matrix_pytest.py     # Тернарная матрица
        ├── test_conditional_attention.py     # Условное внимание
        ├── test_convergence_pytest.py        # Convergence модули
        ├── test_bridge_of_modules_pytest.py  # BridgeOfModules
        ├── test_archetypal_interlingua_pytest.py # ArchetypalInterlingua
        ├── test_integration_scale.py         # Интеграционные тесты
        ├── test_generation_quality.py        # Качество генерации
        ├── test_training_utils.py            # Тренировочные утилиты
        ├── test_hierarchical_e2.py           # E2-иерархия
        └── test_nautilus_pytest.py           # Nautilus-модули
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

**Покрытие:** 1987+ тестов, 25 файлов. Покрыты все основные архитектуры, геометрические модули, тренировочные утилиты, edge cases (batch=1, seq=1), mixed precision (fp16/bf16), device transfer.

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
| `docs/API.md` | Справочник API: модели, конфиг, inference, квантизаторы, LoRA, экспорт |
| `docs/GEOMETRY.md` | Справочник модуля geometry/: квантизаторы, attention, routing, FFN |
| `docs/TRAINING_UTILS.md` | Тренировочные утилиты: TrainingBridge, оптимизаторы, мониторинг |
| `docs/IMPLEMENTATION_STATUS.md` | Аудит: что реализовано в коде, что только в теории, что не используется |
Полный указатель: [`docs/INDEX.md`](docs/INDEX.md)

| Файл | Аудитория | Содержание |
|------|-----------|-----------|
| `docs/ARCHITECTURE.md` | Разработчики | Q6, HMoE, компоненты, LCI |
| `docs/MODELS.md` | Разработчики | Все модели: когда использовать, примеры |
| `docs/TRAINING.md` | ML-инженеры | Pipeline, скрипты обучения, Алгоритм Скарабея |
| `docs/TRAINING_UTILS.md` | ML-инженеры | TrainingBridge, оптимизаторы, мониторинг |
| `docs/GEOMETRY.md` | Исследователи | geometry/: квантизаторы, attention, routing, FFN |
| `docs/INFERENCE.md` | Пользователи | E2Inference, генерация, Q6-эмбеддинг, экспорт |
| `docs/CONTRIBUTING.md` | Контрибьюторы | Тесты, стиль кода, добавление архитектур |

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
