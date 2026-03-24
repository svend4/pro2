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
