# 4-LEVEL KNOWLEDGE FRAMEWORK: YIJING-TRANSFORMER

## Формула → Архетип → Алгоритм → Теорема

Организация проекта по 4-уровневой системе познания.
Прямая связь между **Формулой** (гиперкубическая геометрия) и **Теоремой** (языковые данные)
невозможна — между ними **золотая середина**: Архетип и Алгоритм.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    4-УРОВНЕВАЯ СИСТЕМА ПОЗНАНИЯ                        │
│                                                                         │
│   [1] ФОРМУЛА          ──────►  Математика (чистая геометрия)          │
│       ↕ качественный переход                                            │
│   [2] АРХЕТИП           ──────►  Физика (структурные паттерны)          │
│       ↕ золотая середина                                                │
│   [3] АЛГОРИТМ          ──────►  Химия (процессы комбинирования)       │
│       ↕ качественный переход                                            │
│   [4] ТЕОРЕМА           ──────►  Биология/Лингвистика (результаты)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Уровень 1: ФОРМУЛА (Математика)

**Наука**: Чистая математика — группы, поля, изоморфизмы
**Вопрос**: Что ЕСТЬ гиперкуб?
**Дисциплина**: Алгебра, комбинаторика, теория групп

### Содержание
- Группа ({-1,+1}^6, ⊙) ≅ (Z₂^6, ⊕) — XOR-изоморфизм
- 64 вершины = 64 гексаграммы И-Цзин
- Факторизация 8×8 (триграммы)
- Расстояние Хэмминга = (n - ⟨u,v⟩)/2
- Walsh-Hadamard спектральное разложение
- E8 решётка как расширение гиперкуба

### Файлы (11 модулей)
```
models/geometry/core.py           — Генерация вершин Z₂ⁿ, триграммы, гексаграммы
models/geometry/quantizers.py     — Проекция на вершины: YiJing, E8, Factored, Gumbel
models/geometry/equivariant.py    — BianGua, D4-эквивариантность, дуальность yin-yang
models/geometry/convergence.py    — GlyphComposer: иерархическая композиция
models/geometry/abriale.py        — Изотропные сети, N-арные отношения
models/geometry/positional.py     — RoPE, ALiBi, CubicPE, FourLevelPE
config/config.py                  — YiJingConfig: все гиперпараметры
```

### 19 доказанных теорем
| # | Теорема | Статус |
|---|---------|--------|
| 1 | XOR-изоморфизм | ✅ Доказана |
| 2 | Квантизатор-гомоморфизм при T→0 | ✅ Доказана |
| 3 | Хэмминг через скалярное произведение | ✅ Верифицирована |
| 4 | WHT-спектральное разложение | ✅ Верифицирована |
| ... | (ещё 15 теорем) | ✅ |

---

## Уровень 2: АРХЕТИП (Физика)

**Наука**: Физика — как математические структуры ПРОЯВЛЯЮТСЯ в нейросетях
**Вопрос**: Как геометрия ОТОБРАЖАЕТСЯ на attention/routing?
**Дисциплина**: Теория полей, резонансы, симметрии

### Содержание
- 15+ паттернов внимания из геометрии гиперкуба
- Маршрутизация по Хэмминговому расстоянию (Q6 routing)
- Экспертная иерархия (Nautilus: микро→макро)
- Факторизация триграмм → Mixture of Experts
- 7 теоретических источников как архетипы применения геометрии

### Файлы (23 модуля)
```
# Паттерны внимания (как геометрия → attention)
models/geometry/attention.py      — 15 паттернов: Triangular, Palace, Heisenberg, FlowerOfLife...
models/geometry/nautilus.py       — NautilusHierarchy: 7 камер от микро до макро
models/geometry/routing.py        — GatedPathSelector, AdaptiveGate, TaskAwareRouter
models/geometry/ffn.py            — SwiGLU, TrigramMoE, DomainMoE, GeometricFFN

# Архитектуры (как архетипы → модели)
models/model.py                   — YiJingGPT: основная архитектура
models/baseline.py                — VanillaGPT: базовая линия без геометрии
models/diff_attn.py               — DifferentialAttention
models/expert_choice.py           — Expert-Choice маршрутизация

# Иерархические структуры
models/hierarchical_moe.py        — Matryoshka Q2→Q3→Q6
models/variant3.py                — Q6-core + BianGuaAttention
models/variant3_extensions.py     — 10 расширений
models/hierarchical_e2.py         — E2: 5 иерархических уровней
models/nautilus_yijing.py         — Полная интеграция YiJing-Nautilus

# Адаптация
models/lora.py                    — LoRA
models/prefix_tuning.py           — Prefix tuning, Logit Lens
models/speculative.py             — Speculative decoding
models/extensions.py              — A2-E14: Walsh-Hadamard, Reed-Muller
```

### Ключевой вопрос уровня 2
**Почему 15 паттернов внимания, а не 1-2?**
Потому что каждый паттерн — это отдельный **архетип** того, как геометрия
может проявляться в attention. Triangular — через треугольное неравенство,
FlowerOfLife — через перекрёстные симметрии, CubeDiagonal — через диагонали
гиперкуба. Архетипы — это ВАРИАНТЫ отображения формулы в физику.

---

## Уровень 3: АЛГОРИТМ (Химия)

**Наука**: Химия — как КОМБИНИРОВАТЬ элементы, в каком порядке, с какими пропорциями
**Вопрос**: Как ТРЕНИРОВАТЬ модель с геометрическими компонентами?
**Дисциплина**: Процессы, пайплайны, рецепты

### Содержание
- Training loop с warmup, curriculum, gradient accumulation
- 41 экспериментальных оптимизатор (utils_v12..v52)
- Bridge-модули: как объединить 7 источников
- Data pipelines: tokenization, streaming, domain routing
- Temperature annealing для квантизации

### Файлы (67 модулей)
```
# Ядро обучения
training/train.py                 — Основной training loop
training/optim.py                 — Optimizer factory, LLRD
training/bridge.py                — Мост к 41 экспериментальному модулю

# Регуляризация и потери
training/regularization.py        — Token Merging, Cosine Annealing
training/distillation.py          — Knowledge distillation
training/ema.py                   — Exponential Moving Average

# Bridge-модули (комбинирование источников)
training/bridge_optimizers.py     — Sophia, LAMB, Lion, SAM
training/bridge_schedulers.py     — WSD, Curriculum, LLRD scheduling
training/bridge_regularization.py — Z-Loss, AGC, Label Smoothing
training/bridge_monitors.py       — Loss spike detection, Grokking
training/bridge_model_surgery.py  — µP, Pruning, DoRA, Merging

# Экспериментальные модули (41 файл)
training/utils_v12.py .. utils_v52.py

# Data pipelines
data_utils/text_dataset.py        — TextDataset
data_utils/streaming_dataset.py   — StreamingDataset
data_utils/svend4_dataset.py      — Svend4 multi-domain
data_utils/wikitext_dataset.py    — WikiText
data_utils/bridge_augmentation.py — BPE dropout, RAG, packing

# Токенизация
tokenizer/char_tokenizer.py       — CharTokenizer
tokenizer/glyph_tokenizer.py      — GlyphTokenizer
tokenizer/tokenizer_utils.py      — Утилиты
```

### Ключевой вопрос уровня 3
**Почему 41 экспериментальный модуль?**
Потому что комбинирование (химия) — это ПОИСК правильных пропорций:
какой optimizer + scheduler + regularization даёт лучший результат
для конкретной геометрической архитектуры. Каждый utils_vN — это
отдельный "рецепт" комбинирования.

---

## Уровень 4: ТЕОРЕМА (Биология/Лингвистика)

**Наука**: Биология/Лингвистика — ЖИВЫЕ результаты на реальных данных
**Вопрос**: Работает ли это на ЯЗЫКЕ?
**Дисциплина**: Эмпирическая проверка, бенчмарки, практические примеры

### Содержание
- Генерация текста (greedy, beam, nucleus)
- 30+ benchmark результатов (PPL, diversity, accuracy)
- Ablation studies: какие компоненты помогают
- Сравнение с vanilla baseline

### Файлы (49 модулей)
```
# Inference (применение к языку)
inference/generate.py             — Генерация текста
inference/bridge_inference.py     — AdvancedGenerator: beam, nucleus, speculative

# Экспорт
models/export.py                  — ONNX, TorchScript, model cards

# Бенчмарки (31 скрипт)
scripts/benchmark*.py             — v53..v69: PPL, diversity, convergence
scripts/ablation_*.py             — Ablation: 3 modes, 6 sources

# Эксперименты с реальными данными
scripts/train_real_data.py        — Обучение на реальных данных
scripts/train_text_demo.py        — Генерация текста
scripts/wikitext_train.py         — WikiText training
scripts/downstream_finetune.py    — Fine-tuning на downstream задачи
```

### Текущие результаты
| Архитектура | Params | PPL | Задача |
|-------------|--------|-----|--------|
| Vanilla baseline | 839K | 2.91 | Synthetic WikiText |
| AbrialeBridge (v59) | 1.9M | **1.24** | Synthetic WikiText |
| Belyaev source | — | **1.01** | Synthetic WikiText |
| Geometric (XOR) | — | **100%** | Z₂⁶ classification |

---

## ДИАГНОСТИКА: Почему Формула не соединяется с Теоремой

```
Проблема:  [1] ФОРМУЛА ─────── ✗ ─────── [4] ТЕОРЕМА
           (геометрия)                     (язык)
           gate ≈ 0.5                      PPL ≈ vanilla

Причина:   Уровни [2] и [3] недостаточно развиты

Уровень 2 (Архетип):
  ✅ 15 паттернов внимания существуют
  ⚠️  Не ясно какие 2-3 из 15 реально помогают
  ❌ Нет систематического ablation по паттернам

Уровень 3 (Алгоритм):
  ✅ 41 рецепт обучения существует
  ⚠️  Temperature annealing не доходит до сходимости (800 шагов < 2000)
  ❌ STE zero-gradient trap блокирует ternary quantization
  ❌ Нет curriculum от простых к сложным геометрическим bias

Решение:   Укрепить золотую середину

[2] → Выбрать 2-3 архетипа из 15 через ablation на реальных данных
[3] → Достаточный warmup, Gumbel-Softmax вместо STE, curriculum
```

---

## ПЛАН РЕОРГАНИЗАЦИИ

### Фаза 1: Маркировка (текущий коммит)
Каждый модуль получает маркер уровня в __init__.py

### Фаза 2: Селекция Архетипов (Level 2)
- Ablation 15 attention паттернов на реальном WikiText-2
- Выбрать top-3 по PPL improvement / compute cost
- Заморозить архитектуру

### Фаза 3: Оптимизация Алгоритмов (Level 3)
- Из 41 utils_vN выбрать production-ready рецепт
- Gumbel-Softmax для quantizer training
- Curriculum: 0-geometry → partial → full

### Фаза 4: Доказательство Теоремы (Level 4)
- Full WikiText-2/103 training при d_model≥256
- LAMBADA, HellaSwag для downstream
- Сравнение с GPT-2 small при равном compute

---

## СООТВЕТСТВИЕ С INFO3 ФРЕЙМВОРКОМ

| info3 Компонент | pro2 Компонент |
|----------------|----------------|
| FormulaAgent | `models/geometry/core.py`, `quantizers.py` |
| ArchetypeAgent | `models/geometry/attention.py`, `routing.py`, `nautilus.py` |
| AlgorithmAgent | `training/train.py`, `bridge*.py`, `data_utils/` |
| TheoremAgent | `inference/`, `scripts/benchmark*.py` |
| CoordinatorAgent | `models/model.py` (YiJingGPT — координирует все уровни) |
| InfoBroker | `models/geometry/routing.py` (маршрутизация запросов к экспертам) |
| KnowledgeSystem | Весь проект как 4-уровневая система |

### Применимость info3 паттернов

**Маршрутизация по уровням**:
- Запрос "формула" → Level 1 (quantizers)
- Запрос "как attention" → Level 2 (patterns)
- Запрос "как тренировать" → Level 3 (train.py)
- Запрос "результат PPL" → Level 4 (benchmarks)

**InfoBroker = Q6GeometricRouter**:
Router уже существует и маршрутизирует по 7 экспертам (MATH, CODE, HUMAN...),
что изоморфно 4-уровневой маршрутизации:
- MATH ≈ Level 1 (формулы)
- SYSTEM ≈ Level 2 (архитектура)
- CODE ≈ Level 3 (алгоритмы)
- HUMAN ≈ Level 4 (язык)

---

**Версия**: 1.0
**Дата**: 2026-03-19
**Формат**: Формула → Архетип → Алгоритм → Теорема
