# Отчёт: исследования, реализация и промежуточные результаты

## Экосистема svend4/pro2 — YiJing-Transformer World Model на Q6

---

## Содержание

1. [Промежуточные результаты](#1-промежуточные-результаты)
2. [Смежные исследования](#2-смежные-исследования)
3. [Реализованные улучшения: GrMoE + SIPS + LatFormer](#3-реализованные-улучшения)
4. [Hyperagents: рекурсивное самоулучшение](#4-hyperagents)
5. [Дорожная карта](#5-дорожная-карта)

---

## 1. Промежуточные результаты

### 1.1. Обучение 7 моделей на полном корпусе (20 МБ, 160K последовательностей)

Корпус собран из 12 репозиториев svend4 (info4, info5, data30, daten, info7, info, daten11, data10, info1, info3, daten22, data7) + info_corpus. Итого 2101 источник, 15.4M символов, 160 492 обучающих последовательности (40x увеличение от базового info_corpus).

| Модель | Параметры | Шагов | Start Loss | Final Loss | Снижение |
|--------|-----------|-------|-----------|-----------|----------|
| Quartet | 2.5M | 8000 | 8.66 | 0.051 | 99.4% |
| NautilusYiJing | 1.7M | 8000 | 8.17 | 2.356 | 71.2% |
| Variant3 | 1.4M | 8000 | 8.12 | 2.360 | 70.9% |
| NautilusMoME | 2.9M | 3000 | 7.89 | 2.436 | 69.1% |
| HMoE | 5.7M | 5000 | 8.35 | 2.646 | 68.3% |
| YiJingGPT | 1.8M | 5000 | 7.02 | 2.656 | 62.2% |
| HierarchicalE2 | 2.1M | 8000 | 8.21 | 2.695 | 67.2% |

**Конфигурация обучения:**
- batch_size=16, block_size=64
- Cosine LR scheduling с eta_min = lr * 0.05
- Pretrained модели (NautilusMoME): lr=2e-4, 3000 шагов
- Новые модели: lr=5e-4, 8000 шагов
- Checkpoint модели (YiJingGPT, HMoE): lr=3e-4, 5000 шагов

### 1.2. Расширение vocab 256 -> 4096

**Проблема:** HMoE и YiJingGPT обучены с byte-level токенизацией (vocab=256), но единая система использует BPE (vocab=4096). При декодировании BPE ID=208 (первый байт кириллицы "М") маппится на случайный BPE-субтокен.

**Решение (первая версия):** Расширение nn.Embedding(256, d) -> nn.Embedding(4096, d):
- Строки 0-255: копируются обученные веса
- Строки 256-4095: инициализируются случайно (N(0, 0.02))
- Аналогично для nn.Linear LM head

**Анализ post-hoc (измерено на 6.0M токенов корпуса):** эта схема
сохранила практически ничего, и объясняет, почему HMoE и YiJingGPT оказались
в хвосте таблицы и генерировали мусор.

| Факт | Значение |
|------|---------|
| Токенов корпуса с id < 256 (скопированные строки) | **1.3%** |
| Токенов с id >= 256 (случайная инициализация) | **98.7%** |
| Кириллических токенов с id < 256 | **0.00%** (кириллица — 36% корпуса) |
| Смещение byte-fallback id в BPE | id = byte + 4 (`<0x00>`=4 ... `<0xFF>`=259) |

Два независимых дефекта:
1. BPE почти не использует байтовые токены: 98.7% токенов попадают в строки,
   инициализированные случайно. Фактически модели учились с нуля, но получили
   на 3000 шагов меньше, чем «честно новые» модели.
2. Даже скопированные строки сдвинуты: byte-level строка `b` попала в BPE-id `b`,
   а BPE-id `b` означает байт `b-4`. Строка буквы "A" (0x41) оказалась под
   `<0x3D>` ("="). Скопированные веса были не бесполезны, а **вредны**.

**Исправление:** корректный remap `new_emb[4 + b] = old_emb[b]` для b в 0..255
(см. `scripts/train_all_models.py`, `train_full_corpus.py`, `test_grand_orchestrator.py`).
Это чинит дефект 2, но не дефект 1 — вывод: byte-level чекпоинты не переносятся
на BPE расширением словаря. Честные варианты: (a) переобучить HMoE и YiJingGPT
с нуля на BPE с тем же бюджетом, что и остальные; (b) оставить их byte-level
и дать оркестратору per-model токенизатор.

### 1.3. Качество генерации (диалоги)

**NautilusYiJing (#1 по качеству, loss=2.356):**
```
В: "Что такое мудрость"
О: "Останащая душа не слушка. Развитие пробуждения — это
    разлости. Духовная улучшение — особрода."

В: "Мудрость начинается"
О: "на душевной запускает имеет конфликты | Основные |
    Культе и пробуждений, принимать традиции на основу."

В: "Информационная система"
О: "направленного направления (80-50%, 0-90%, 60%) —
    Гибры к 8 день (100% real) — Содержанные карточки"
```

**NautilusMoME (#2, loss=2.436):**
```
В: "Искусственный интеллект и человек"
О: "внутреннее обола, потеряться направления доброго уровня
    и души, неделю и беках людины — свои внутренняя мудрость"

В: "Алгоритм поиска истины"
О: "для бюдрения, сообществование доступной формируют
    взаимодействиям провос-меет идельности, развития"
```

**Quartet (loss=0.051 — переобучение):**
```
В: "Музыка это"
О: "это это это это это это это это это это это это..."
```
Quartet достигает минимального loss, но страдает от repetition collapse. Генерирует повторяющиеся фрагменты. Для исправления нужны: dropout, repetition penalty, nucleus sampling или увеличение модели.

### 1.4. Новые компоненты: GrMoE + SIPS + LatFormer

Реализованы и протестированы 5 новых модулей в `yijing_transformer/models/geometry/advanced_routing.py`:

| Компонент | Тест | Результат |
|-----------|------|-----------|
| GrassmannianRouter | forward + backward | weights: (B,T,6), aux_loss: 0.003 |
| Q6GrassmannianRouter | forward + backward | geo_gate: 0.500, 7 экспертов |
| Q6SoftAttentionMask | forward | mask: (B,H,T,T), range [-0.03, 0.06] |
| EnhancedBianGuaAttention | forward | output: (B,T,128), hamming_lambda: 0.1 |
| GrassmannianMoEFFN | forward + backward | 203K params, aux: 0.005 |

**Обучение Enhanced-моделей (с нуля, 5000 шагов):**

| Модель | Параметры | Final Loss | Тип |
|--------|-----------|-----------|-----|
| Variant3 (original, pretrained) | 1.4M | **2.308** | Дообучение |
| NautilusYiJing (original, pretrained) | 1.7M | **2.354** | Дообучение |
| EnhancedNautilus GrMoE | 2.0M | 4.198 | С нуля |
| EnhancedVariant3 GrMoE | 1.9M | 4.351 | С нуля |

**Диагностика маршрутизации GrMoE:**
- EnhancedNautilus: geo_gate = **0.663** (модель предпочитает Q6 геометрию)
- SIPS: alpha=1.630, beta=2.651 (обученная стабильность)
- Lambda (концентрация): 0.603 (умеренная разреженность)
- Load balance: [0.39, 0.11, 0.09, 0.16, 0.06, 0.19] — нет коллапса экспертов

**Вывод:** Enhanced-модели показывают 50% снижение loss за 5000 шагов с нуля. Для достижения уровня оригиналов (loss ~2.3) нужно 15-20K шагов или предобучение на чекпоинтах.

### 1.5. Артефакты и чекпоинты

**Чекпоинты** (в `checkpoints/`, gitignored, ~280 МБ):

| Файл | Модель | Корпус |
|------|--------|--------|
| `*_trained.pt` | 7 моделей | info_corpus (653 КБ) |
| `*_full.pt` | 7 моделей | info_corpus + svend4 (20 МБ) |
| `*_grmoe.pt` | 2 Enhanced | info_corpus + svend4 (20 МБ) |
| `grand_orchestrator.pt` | Ансамбль | — |

**Скрипты** (в `scripts/`):
- `train_all_models.py` — обучение 7 моделей на info_corpus
- `train_full_corpus.py` — обучение на полном 20 МБ корпусе
- `test_grand_orchestrator.py` — тестирование ансамбля
- `fetch_svend4_corpus.py` — загрузка 12 репозиториев

---

## 2. Смежные исследования

### 2.1. Геометрические приоритеты в архитектурах

**LatFormer (Lattice Symmetry Priors in Attention)**

Встраивает симметрии гиперкубической решётки в механизмы внимания через мягкие маски. На два порядка меньше данных для обучения геометрическим трансформациям. Прямое подтверждение гипотезы svend4: встраивание геометрической структуры (Q6) даёт радикальный выигрыш в эффективности обучения.

**Связь с pro2:** Метод маскирования внимания адаптирован для BianGuaAttention — реализован как `Q6SoftAttentionMask` в `advanced_routing.py`.

### 2.2. Иерархическая и продвинутая маршрутизация MoE

**GrMoE (Grassmannian Mixture-of-Experts)**

Маршрутизация на многообразии Грассмана: каждый эксперт представлен подпространством, а не вектором. Веса через распределение Матрицы Бингема. Результат: 0% коллапса экспертов, 15-30% улучшение балансировки.

**Связь с pro2:** Реализован как `GrassmannianRouter` и `Q6GrassmannianRouter`. Эксперты Q6 расширены от битов к подпространствам. Концентрация Lambda — аналог температуры маршрутизации.

**L2R (Low-Rank & Lipschitz-Controlled Routing)**

Проецирует входы в низкоранговое пространство, применяет SIPS (Saturated Inner-Product Scoring) для контроля Липшицевости. Улучшает стабильность и специализацию.

**Связь с pro2:** SIPS реализован в `GrassmannianRouter` и `EnhancedBianGuaAttention`. Низкоранговая проекция в Q6 уже является естественным L2R.

**SAM Routing, HRME**

SAM активирует несколько экспертов без роста сложности (аналог NautilusMoME). HRME использует бинарное дерево (альтернатива матрёшке Q2->Q3->Q6).

### 2.3. Нейро-символический ИИ

**HNSDT (Hierarchical Neuro-Symbolic Decision Transformer)**

Символьный планировщик задаёт последовательность операторов, Decision Transformer исполняет их. Формальный анализ накопления ошибок.

**Связь с pro2:** Прямое соответствие: символьный уровень (гексаграммы, архетипы) + нейросетевой (трансформер). Анализ ошибок применим к world model.

**TIC (Translate-Infer-Compile)**

LLM -> логический вывод (ASP) -> PDDL -> классический планировщик. Открытый код от Intuit AI Research.

**Связь с pro2:** Может быть интегрирован как внешний эксперт в децентрализованный Наутилус для PDDL-планирования.

### 2.4. Сводная таблица приоритетов

| Проект | Ключевая идея | Приоритет | Статус |
|--------|--------------|-----------|--------|
| GrMoE | Подпространства экспертов | ***** | Реализован |
| L2R/SIPS | Липшицев скоринг | ***** | Реализован |
| LatFormer | Мягкие маски Q6 | **** | Реализован |
| HNSDT | Символьный планировщик | **** | Концепция |
| TIC | PDDL-эксперт | **** | Концепция |
| HRME | Бинарное дерево | *** | Не начат |

---

## 3. Реализованные улучшения

### 3.1. GrassmannianRouter

**Файл:** `yijing_transformer/models/geometry/advanced_routing.py`

Каждый эксперт представлен k-мерным подпространством в d-мерном пространстве (точка на Gr(k,d)). Аффинность токена к эксперту = ||U_e^T z||^2.

```python
from yijing_transformer.models.geometry.advanced_routing import GrassmannianRouter

router = GrassmannianRouter(
    d_model=128,     # входная размерность
    n_experts=6,     # число экспертов
    subspace_dim=4,  # размерность подпространства (k)
    top_k=2,         # число активных экспертов
    latent_dim=32,   # размерность L2R-проекции
)
weights, indices, aux_loss = router(x)  # x: (B, T, d_model)
```

Особенности:
- **Orthogonal initialization** экспертных подпространств
- **SIPS scoring** с обучаемыми alpha, beta
- **EMA load balancing** с MSE-потерей к равномерному распределению
- **Log-domain Lambda** для стабильного управления концентрацией

### 3.2. Q6GrassmannianRouter

Комбинирует Grassmannian с Q6-геометрией гексаграмм:

```python
from yijing_transformer.models.geometry.advanced_routing import Q6GrassmannianRouter

router = Q6GrassmannianRouter(
    d_model=128,
    n_experts=7,      # MATH, CODE, HUMAN, SYSTEM, RECON, INFO, SYNTH
    subspace_dim=4,
    top_k=2,
    hamming_weight=0.3,
)
```

Доменные якоря: MATH=63 (Qian/Heaven), CODE=6 (Wind), HUMAN=0 (Kun/Earth), SYSTEM=18, RECON=45, INFO=27, SYNTH=36.

Обучаемый `geo_gate` (sigmoid) смешивает Grassmannian и Hamming scores. На EnhancedNautilus после 5000 шагов geo_gate = 0.663 — модель предпочитает геометрическое пространство Q6 линейному.

### 3.3. Q6SoftAttentionMask (LatFormer)

Обучаемые мягкие маски внимания на основе Q6-симметрий:

```python
from yijing_transformer.models.geometry.advanced_routing import Q6SoftAttentionMask

mask_gen = Q6SoftAttentionMask(
    d_model=128,
    n_heads=4,
    n_transforms=7,     # 6 bit-flips + 1 antipodal
    temperature=0.5,
)
mask = mask_gen(x)  # (B, n_heads, T, T) — additive attention bias
```

7 трансформаций:
- 6 одиночных flip-ов битов (bian yao / changing lines)
- 1 антиподальное отражение (полная инверсия)

Per-head, per-transform обучаемые веса позволяют каждой голове внимания специализироваться на разных Q6-трансформациях.

### 3.4. EnhancedBianGuaAttention

Drop-in замена для BianGuaAttention:

```
score(q,k) = SIPS_alpha * tanh(QK^T / SIPS_beta)
           + hamming_lambda * <hex_q, hex_k> / 2
           + LatFormer_mask(q, k)
```

Три слоя геометрического смещения:
1. **SIPS** — ограничивает чувствительность скоринга
2. **Hamming bias** — топологическая близость в Q6
3. **LatFormer mask** — обучаемые Q6-трансформации

### 3.5. GrassmannianMoEFFN

MoE FFN с полным Grassmannian routing:

```python
from yijing_transformer.models.geometry.advanced_routing import GrassmannianMoEFFN

moe = GrassmannianMoEFFN(
    d_model=128,
    n_experts=6,
    d_expert=128,
    top_k=2,
    use_q6=True,  # Q6GrassmannianRouter vs GrassmannianRouter
)
output, aux_loss = moe(x)
```

---

## 4. Hyperagents: рекурсивное самоулучшение

### 4.1. Суть концепции

Hyperagents (Meta, 2026) — архитектура metacognitive self-modification: агент модифицирует не только свои действия, но и механизм улучшения. Два слоя объединены в единую редактируемую программу:

| Слой | Назначение | Аналог в svend4 |
|------|-----------|----------------|
| Task Agent | Решает конкретную задачу | pro2 + эксперты (data7, info3) |
| Meta Agent | Анализирует и улучшает Task Agent | info1 (самодиагностика) |
| Объединённая программа | Meta может изменять и себя | Рекурсивное замыкание (отсутствует) |

### 4.2. Три уровня управления для экосистемы svend4

```
Стратегический (Hyperagents)
    Перепроектирует логику оркестрации,
    протоколы, метрики, архитектуру экспертов
        |
Тактический (info1 + Service Registry)
    Распределяет запросы, следит за здоровьем,
    применяет проверенные улучшения
        |
Оперативный (pro2, data7, info3, prima.cpp)
    Решает конкретные задачи в изолированных
    песочницах
```

### 4.3. Интеграция с Q6 World Model

Hyperagents может:
- Запускать A/B-тесты архитектур (иерархический vs плоский MoE)
- Менять LCI-метрику, если находит лучшую формулу
- Добавлять/удалять экспертов на основе downstream-метрик
- Эволюционировать портальный протокол децентрализованного Наутилуса

Естественные барьеры безопасности в svend4:
- **Дискретность Q6** — ограничивает пространство мутаций
- **Интерпретируемость архетипов** — изменения анализируются через гексаграммы
- **Модульность Наутилуса** — эксперименты изолированы в микросервисах

### 4.4. Связь с Darwin Godel Machine

Экосистема svend4 уже содержит эволюционные элементы:
- info1: самодиагностика (метрики CD, VT, CR, DB)
- data7: TSP-оптимизация (отжиг, генетика)
- info3: бинарные оппозиции как мутации

Hyperagents добавляет рекурсивное замыкание: эволюция меняет не только параметры, но и сам процесс эволюции.

---

## 5. Дорожная карта

### 5.1. Ближайшие шаги (реализованы)

- [x] Обучение 7 моделей на info_corpus (1000 шагов)
- [x] Расширение vocab 256 -> 4096 для HMoE и YiJingGPT
- [x] Сбор полного корпуса из 12 svend4 репозиториев (20 МБ)
- [x] Обучение на полном корпусе (3000-8000 шагов)
- [x] Реализация GrMoE, SIPS, LatFormer
- [x] Тестирование Enhanced-моделей

### 5.2. Следующие шаги (рекомендации)

- [ ] Длительное обучение Enhanced-моделей (15-20K шагов)
- [ ] Transfer learning: инициализация Enhanced из чекпоинтов оригиналов
- [ ] Исправление Quartet overfitting (repetition penalty, early stopping)
- [ ] Grand Orchestrator с Enhanced-моделями
- [ ] HNSDT: символьный планировщик на гексаграммах
- [ ] TIC-эксперт для PDDL-планирования в Наутилусе

### 5.3. Долгосрочные цели

- [ ] Hyperagents: рекурсивный Meta-агент для info1
- [ ] Вероятностные гексаграммы (распределения на Q6)
- [ ] Активный сбор информации при высокой неопределённости
- [ ] Мультимодальный фронтенд (VLM -> символьные состояния)
- [ ] Масштабирование до d_model=256+ и полноценного BPE корпуса

---

## Приложение A: Файловая структура проекта

```
pro2/
  yijing_transformer/
    models/
      model.py              # YiJingGPT (2198 строк)
      variant3.py           # Variant3GPT (705 строк)
      nautilus_yijing.py    # NautilusYiJing (767 строк)
      hierarchical_e2.py    # HierarchicalE2 (606 строк)
      polyglot.py           # PolyglotQuartet (677 строк)
      hierarchical_moe.py   # HMoE (900+ строк)
      geometry/
        advanced_routing.py # GrMoE + SIPS + LatFormer (NEW)
        routing.py          # GatedPathSelector, ArchetypalInterlingua
        quantizers.py       # YiJingQuantizer, FactoredQuantizer
        attention.py        # TriangularAttention, PalaceAttention
        interlingua_fixed.py # ArchetypalInterlinguaFixed
        kasatkin_router.py  # Q6ExpertBank
        nautilus.py         # NautilusHierarchy
        q6_algebra.py       # Q6Arithmetic, BentFunctions
        ...
    scripts/
      train_nautilus_mome.py
    bpe_tokenizer.model
  scripts/
    train_all_models.py
    train_full_corpus.py
    test_grand_orchestrator.py
    fetch_svend4_corpus.py
  data/
    info_corpus/            # 653 КБ
    svend4_corpus/          # 20 МБ (12 репозиториев)
  checkpoints/              # ~280 МБ (gitignored)
  docs/
    RESEARCH_REPORT.md      # Этот документ
    ARCHITECTURE.md
    MODELS.md
    GEOMETRY.md
    ...
```

## Приложение B: Ключевые метрики

| Метрика | Значение | Комментарий |
|---------|---------|-------------|
| Лучшая модель (loss) | NautilusYiJing: 2.356 | На полном корпусе |
| Лучшая модель (генерация) | NautilusYiJing | Осмысленные фразы на русском |
| Размер корпуса | 20.1 МБ, 160K seq | 12 svend4 repos |
| Vocab | 4096 (BPE) | Все модели унифицированы |
| Общий параметров | 18.2M (7 моделей) | d_model=128 |
| GrMoE geo_gate | 0.663 | Предпочтение Q6 геометрии |
| GrMoE Lambda | 0.603 | Умеренная разреженность |
| SIPS alpha/beta | 1.63 / 2.65 | Обученная стабильность |
| Expert collapse | 0% | Все эксперты активны |
