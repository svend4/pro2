# Portal — Интеграция репозиториев, федеративное обучение, граф знаний

> **Актуально:** 2026-03-24  
> Связанные документы: [INDEX.md](INDEX.md) · [TRAINING.md](TRAINING.md) · [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)

---

## Содержание

1. [Portal-система — архитектура](#1-portal-система--архитектура)
2. [Адаптеры репозиториев](#2-адаптеры-репозиториев)
3. [Корпус-лоадеры](#3-корпус-лоадеры)
4. [Граф знаний и метрики](#4-граф-знаний-и-метрики)
5. [E2-концепт эволюция](#5-e2-концепт-эволюция)
6. [Федеративное обучение](#6-федеративное-обучение)
7. [Meta-мосты](#7-meta-мосты)
8. [Nautilus Inference](#8-nautilus-inference)

---

## 1. Portal-система — архитектура

**Файлы:** `portal.py` (корень), `nautilus/` (движок), `nautilus/adapters/`

Portal — единая точка входа для всех репозиториев экосистемы svend4. Позволяет модели обращаться к данным из разных источников через унифицированный API.

```
portal.py (корень)
  └── NautilusPortal (nautilus/portal.py)
        └── Адаптеры (nautilus/adapters/):
              ├── meta.py      — 256 CA-правил → 64 гексаграммы
              ├── pro2.py      — Q6-концепты, граф, эмбеддинги
              ├── info1.py     — Markdown + α-уровни (-4..+4)
              ├── data2.py     — ЕТД Крюкова (томы 1-100)
              └── data7.py     — TSP, MMORPG, энциклопедии
```

### Базовые классы (portal.py)

```python
from portal import PortalEntry, PortalResult, NautilusPortal

# PortalEntry — универсальная запись
entry = PortalEntry(
    id='pro2:q6:hexagram_1',
    title='Гексаграмма 1 — Цянь',
    source='pro2',
    format_type='q6_concept',
    content='...',
)

# Запрос через portal
portal = NautilusPortal()
result = portal.query('hexagram', source='meta', limit=10)
# result.entries: List[PortalEntry]
# result.cross_links: связи между репозиториями
# result.consensus: агрегированный ответ
```

### Доступные адаптеры

| Адаптер | Источник | Формат | Репо |
|---------|---------|--------|------|
| `MetaAdapter` | `../meta` | 256 CA-правил → 64 гексаграммы | svend4/meta |
| `Pro2Adapter` | `.` | Q6-концепты, граф знаний, эмбеддинги | svend4/pro2 |
| `Info1Adapter` | `../info1` | Markdown + α-уровни | svend4/info1 |
| `Data2Adapter` | `../data2` | ЕТД Крюкова | svend4/data2 |
| `Data7Adapter` | `../data7` | Алгоритмы, энциклопедии | svend4/data7 |

### nautilus/adapters/base.py — протокол адаптера

```python
class BaseAdapter:
    def fetch(self, query: str, limit: int = 20) -> List[PortalEntry]: ...
    def describe(self) -> dict: ...           # метаданные источника
    def is_available(self) -> bool: ...       # есть ли репо рядом
```

---

## 2. Адаптеры репозиториев

### meta.py — CA-правила и гексаграммы

Преобразует 256 правил клеточного автомата (Wolfram) в 64 гексаграммы И-Цзин через Q6-отображение. Позволяет обращаться к формальным доказательствам из `svend4/meta`.

```python
from nautilus.adapters.meta import MetaAdapter
adapter = MetaAdapter(meta_root='../meta')
entries = adapter.fetch('hexagram symmetry', limit=5)
```

### data2.py — ЕТД Крюкова

Загружает тома ЕТД (Единая теория данных) Крюкова как структурированные тексты с α-уровнями. Используется в `self_train_v3.py` (Алгоритм Скарабея).

### data7.py — многодоменный корпус

Алгоритмы (TSP, MMORPG), технические энциклопедии. Основной источник для `corpus_loader.py` (категория `data7`).

---

## 3. Корпус-лоадеры

### corpus_loader.py — универсальный загрузчик

**Файл:** `corpus_loader.py`

Загружает тексты из всех `_clones`-репозиториев. Используется в `bidir_train.py`, `train_e2_joint.py`.

```python
from corpus_loader import CorpusLoader

loader = CorpusLoader(root='.')
corpus = loader.load_all(max_per_source=200)
# corpus: Dict[source_name, List[str]]
# Источники: data7, info1, meta, data2, infosystems, ai_agents, knowledge, meta
```

**8 источников, по 200 текстов = ~1600 текстов суммарно.**

### repo_corpus_loader.py — внутренний корпус

**Файл:** `repo_corpus_loader.py`

Загружает файлы самого репозитория в 7 кластеров — модель обучается на собственном коде и документации.

```python
from repo_corpus_loader import RepoCorpusLoader

loader = RepoCorpusLoader(root='.')
clusters = loader.load_clusters()
# clusters: Dict[cluster_name, List[str]]
```

**7 кластеров:**

| Кластер | Содержание |
|---------|-----------|
| `Theory` | *.md файлы, theoretical-foundations |
| `Models` | yijing_transformer/models/*.py |
| `Training` | self_train*.py, pipeline.py, *_turbine.py |
| `Benchmarks` | bench_*.py, *_log.json |
| `Scripts` | scripts/, experiments/ |
| `Portal` | portal.py, nautilus/ |
| `Self` | repo_corpus_loader.py, corpus_loader.py сами |

Используется в: `e2_self_improve.py`, `train_e2_clusters.py`, `bidir_train_v2.py`.

---

## 4. Граф знаний и метрики

### q6_graph_updater.py — построение графа

**Файл:** `q6_graph_updater.py`

Строит граф знаний из Q6-результатов `e2_inference.py`. Вершины — концепты, рёбра — Q6-близость.

```python
from q6_graph_updater import Q6GraphUpdater

updater = Q6GraphUpdater(model, inference_engine)
updater.update_from_corpus(texts)
graph = updater.get_graph()
# graph: networkx.DiGraph с атрибутами q6_coord, domain, alpha_level
```

**Типы рёбер:**

| Тип | Условие | Значение |
|-----|---------|---------|
| `same_hex` | Hamming == 0 | одна гексаграмма |
| `neighbor` | Hamming == 1 | Бян-Гуа смежность |
| `close` | Hamming ≤ 2 | кластер Q6 |
| `alpha_link` | α-уровень ±1 | вертикальная связь |
| `domain_link` | одинаковый домен | горизонтальная связь |

**Данные в q6_evolution/:** снапшоты Q6-координат концептов до/после каждого цикла обучения.

### graph_health.py — метрики здоровья

**Файл:** `graph_health.py`

Вычисляет 4 метрики здоровья графа знаний. Используется в `bidir_train.py` для критерия остановки.

```python
from graph_health import GraphHealthMonitor

monitor = GraphHealthMonitor(graph)
health = monitor.compute()
# {CD: 0.18, VT: 0.62, CR: 1.1, DB: 0.24}
```

**Метрики:**

| Метрика | Расшифровка | Норма | Проблема |
|---------|------------|-------|---------|
| **CD** | Connectivity Density — плотность рёбер | 10–25% | < 5% → редкий граф |
| **VT** | Vertical Traceability — связность α-уровней | ≥ 50% | < 30% → нет иерархии |
| **CR** | Convergence Rate — скорость роста концептов | 0.7–1.5 | > 2.0 → взрыв, < 0.3 → стагнация |
| **DB** | Directional Balance — баланс ⇑⇓ vs ↔ | < 30% | > 50% → только горизонталь |

**Данные графа в:** `nautilus.json` (основной граф Наутилуса).

---

## 5. E2-концепт эволюция

**Файл:** `e2_concept_evolution.py`

Отслеживает, как Q6-координаты конкретных концептов меняются в процессе обучения. Позволяет видеть, «учится» ли модель или забывает.

```python
from e2_concept_evolution import ConceptEvolutionTracker

tracker = ConceptEvolutionTracker(model)

# Снапшот в начале
tracker.snapshot('before_cycle_1', concepts=['hexagram', 'routing', 'Q6'])

# ... обучение ...

tracker.snapshot('after_cycle_1', concepts=['hexagram', 'routing', 'Q6'])

# Сравнение
diff = tracker.compare('before_cycle_1', 'after_cycle_1')
# diff: {concept: {hamming_shift, domain_change, alpha_drift}}
```

**Методы:**

| Метод | Назначение |
|-------|-----------|
| `snapshot(name, concepts)` | Сохранить Q6-координаты концептов |
| `compare(before, after)` | Сравнить два снапшота |
| `timeline(concept)` | График изменений концепта |
| `stability(concept)` | Насколько стабильна позиция |
| `cluster_drift()` | Смещение кластеров за период |

**Снапшоты хранятся в:** `q6_evolution/` (initial.json, cycle_N_before/after.json).

---

## 6. Федеративное обучение

**Файл:** `federated_round.py`

Синхронизирует веса модели с другими репозиториями через NautilusPortal. Цикл: EXPORT → BROADCAST → IMPORT → ALIGN → INTEGRATE → REPORT.

```bash
python federated_round.py --repos ../meta ../info1 --rounds 3
```

**Цикл одного раунда:**

```
1. EXPORT    — экспортировать веса текущей модели как Q6-концепты
2. BROADCAST — отправить в другие репо через Portal
3. IMPORT    — получить обновления от других репо
4. ALIGN     — выровнять Q6-координаты (не конфликтовать с локальными)
5. INTEGRATE — включить внешние знания в веса
6. REPORT    — записать результаты в federated_round_log.json
```

**Лог:** `federated_round_log.json`

**Требования:** репо `meta`, `info1`, `data2` должны быть клонированы рядом:
```bash
cd ..
git clone https://github.com/svend4/meta.git
git clone https://github.com/svend4/info1.git
```

---

## 7. Meta-мосты

### meta_bridge.py — интеграция с svend4/meta

**Файл:** `meta_bridge.py`  
**Требует:** `../meta` клонирован

| Класс | Назначение |
|-------|-----------|
| `HexLearnRouter` | k-NN маршрутизация на Q6 из meta/hexlearn |
| `MetropolisAnnealer` | охлаждение из meta/hexphys (T(c) = max(0.5, T₀×0.85^c)) |
| `Q4Q6Mapper` | вложение Q4⊂Q6 (15 тессерактов) |
| `FullAnalogyCrossMatrix` | полная 6×6 матрица симметрий (36 пар) |

### meta_q6.py — Q6-утилиты из meta

**Файл:** `meta_q6.py`

Используется в `pipeline.py` для bent-функций и temperature annealing.

```python
from meta_q6 import bent_seed_texts, metropolis_temperature, q4_tesseracts

# Bent-функции как seed для RAG (разнообразнее случайных)
seeds = bent_seed_texts(n=20)
# nl=28 (нелинейность), WHT равномерный

# Расписание температуры
T = metropolis_temperature(cycle=3, T0=1.4)  # → 1.011

# 15 тессерактов Q4 в Q6
tesseracts = q4_tesseracts()  # 15 × 16 вершин = 60 из 64 Q6-вершин
```

### compare_meta.py — сравнение подходов

**Файл:** `compare_meta.py`

Сравнивает символьный подход meta (K3-категории, CA-правила) с нейросетевым про2 (TextQualityFilter, Q6-эмбеддинги). Визуализирует расхождение в Q6-координатах для одних и тех же концептов.

---

## 8. Nautilus Inference

**Файл:** `nautilus_inference.py`

Три режима инференса:

| Режим | Описание | Когда использовать |
|-------|---------|-------------------|
| `STANDALONE` | Только локальная модель | нет доступа к другим репо |
| `PORTAL` | Обращение к Portal API | есть хотя бы один адаптер |
| `FEDERATED` | Полная синхронизация | все репо доступны |

```python
from nautilus_inference import NautilusInference

# Автоматический выбор режима
inference = NautilusInference(model, mode='auto')
result = inference.generate("Что такое Q6?", max_tokens=100)

# Принудительный режим
inference = NautilusInference(model, mode='STANDALONE')
```

---

## Структура папок

```
nautilus/
  ├── __init__.py
  ├── portal.py          — движок Portal
  └── adapters/
        ├── base.py      — протокол BaseAdapter
        ├── meta.py      — MetaAdapter
        ├── pro2.py      — Pro2Adapter
        ├── info1.py     — Info1Adapter
        ├── data2.py     — Data2Adapter
        └── data7.py     — Data7Adapter

q6_evolution/            — снапшоты Q6-координат
  ├── initial.json
  ├── self_cycle_1_before.json
  ├── self_cycle_1_after.json
  └── ...

passports/               — паспорта адаптеров
  ├── meta.md
  ├── info1.md
  └── data2.md
```

---

*Последнее обновление: 2026-03-24 | Ветка: `claude/repository-audit-fvlEG`*
