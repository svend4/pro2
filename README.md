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

## Архитектура (Вариант 3)

### Ядро

| Компонент | Файл | Назначение |
|-----------|------|-----------|
| `Variant3GPT` | `yijing_transformer/models/variant3.py` | Главная языковая модель |
| `HierarchicalMoEFFN` | `yijing_transformer/models/hierarchical_moe.py` | 4-уровневая MoE: Q2→Q3→Q6 |
| `HexagramProjection` | `variant3.py:76-123` | Проекция токена в 64 архетипа Q6 |
| `BianGuaAttention` | `variant3.py:130-169` | Внимание с топологическим смещением Q6 |
| `TernaryGate` | `models/` | Тернарная логика {−1, 0, +1} (变爻) |

### Гейтинг и маршрутизация

| Класс | Файл | Назначение |
|-------|------|-----------|
| `GatedPathSelector` | `geometry/routing.py:11` | Выбор геометрического vs стандартного пути |
| `AdaptiveGatedPathSelector` | `geometry/routing.py:37` | Контентно-зависимый гейт с температурой |
| `AbrialeBridgeMediator` | `geometry/routing.py:573` | Лучший результат PPL 1.24 (v59) |
| `ArchetypalInterlingua` | `geometry/routing.py:944` | 64-экспертный банк (hub-and-spoke) |
| `DynamicCurriculumController` | `geometry/routing.py:1818` | Адаптирует силу геометрии при обучении |

### 6 доменов Q6

```
GEO   — геология, земля      (бит 0)
HYDRO — гидрология, вода     (бит 1)
PYRO  — огонь, химия         (бит 2)
AERO  — воздух, атмосфера    (бит 3)
COSMO — космос               (бит 4)
NOOS  — разум, логика        (бит 5)
```

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
# Одна итерация pipeline (nautilus → turbine → bench)
python pipeline.py --checkpoint hmoe_self_trained_v4.pt

# Самообучение (три стадии: Q6-топология → корпус → фильтрованный мир)
python self_train.py

# С дополнительными лоссами (domain triplet + quality contrastive + gate entropy)
python self_train_v2.py

# Алгоритм Скарабея (figure-8 обход Q6)
python self_train_v3.py

# Сравнительный бенчмарк всех вариантов
python bench_all.py --checkpoint hmoe_self_trained_v4.pt
```

### Варианты архитектуры обучения

| Скрипт | Метод | Особенность |
|--------|-------|-------------|
| `self_train.py` | 3 стадии: Self-Topology → RAG-буфер → Wild | Базовый |
| `self_train_v2.py` | То же + domain triplet + gate entropy loss | +3 вспомогательных лосса |
| `self_train_v3.py` | v2 + Stage0 = figure-8 обход Q6 | Алгоритм Скарабея |
| `figure8_turbine.py` | 4 эксперта, TSP-маршрут | Greedy/2-opt TSP |
| `nautilus_4agent.py` | 4 агента по кольцам META/ABSTRACT/DYNAMIC/CONCRETE | Наутилус-4 |
| `roundabout.py` | Кольцо с адаптивным числом оборотов | Адаптивный LCI |
| `bidir_turbine.py` | Два встречных потока, встреча в DYNAMIC | Bidirectional |
| `multi_salesman.py` | 2–3 агента с общим RAG | Multi-agent |

### Pipeline (curriculum)

```
nautilus_4agent (2 прохода) → figure8_turbine (LCI-loss) → bench_all
```

```bash
python pipeline.py --checkpoint hmoe_self_trained_v4.pt --passes 2
```

Адаптивный LR: при LCI > 2.8 снижается до 5e-6. RAG-сброс через bent seeds с 3-го прохода.

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

## Структура проекта

```
pro2/
├── pipeline.py              # Curriculum pipeline: nautilus → turbine → bench
├── bench_all.py             # Сравнительный бенчмарк 14 вариантов
├── self_train*.py           # Самообучение (v1/v2/v3)
├── self_train_hmoe*.py      # HMoE-специфическое самообучение
├── figure8_turbine.py       # Figure-8 + TSP routing
├── nautilus_4agent.py       # 4-агентный Наутилус
├── roundabout.py            # Кольцевой маршрут
├── bidir_turbine.py         # Двунаправленная турбина
├── multi_salesman.py        # Multi-agent торговые представители
├── e2_inference.py          # Inference API: embed/similar/generate/map
├── eval_hmoe.py             # Оценка HMoE модели
├── hmoe_*.pt                # Чекпоинты (≈19 MB каждый)
│
└── yijing_transformer/
    ├── constants.py                    # HEX_NAMES, DOMAINS (64 гексаграммы)
    ├── models/
    │   ├── variant3.py                 # Variant3GPT (главная модель)
    │   ├── hierarchical_moe.py         # HierarchicalMoEFFN (Q2→Q3→Q6)
    │   ├── hierarchical_e2.py          # HierarchicalE2 (E2 архитектура)
    │   └── geometry/
    │       ├── routing.py              # Гейты, маршрутизаторы, curriculum (1844 LOC)
    │       ├── core.py                 # Q6-гиперкуб, кодбук
    │       └── quantizers.py          # TernaryQuantizer, FactoredQuantizer
    ├── scripts/                        # Утилиты и эксперименты
    └── training/                       # Тренировочные утилиты
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
| `DEV-STATUS-REVIEW.md` | Технический статус: PPL бенчмарки, открытые вопросы |
| `GERMES_NOTATION.md` | Доцифровой прообраз системы (Germes vZvete) |

---

## Зависимости

```
torch>=2.0.0
sentencepiece
datasets
```

Устройство определяется автоматически (CUDA → CPU) во всех скриптах.
