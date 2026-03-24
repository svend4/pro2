# Руководство по обучению

## Содержание

1. [Установка](#1-установка)
2. [Быстрый старт](#2-быстрый-старт)
3. [Скрипты обучения](#3-скрипты-обучения)
4. [Curriculum Pipeline](#4-curriculum-pipeline)
5. [Алгоритм Скарабея (Figure-8)](#5-алгоритм-скарабея-figure-8)
6. [Чекпоинты](#6-чекпоинты)
7. [Бенчмарк](#7-бенчмарк)
8. [Диагностика](#8-диагностика)

---

## 1. Установка

```bash
# Базовая установка
pip install -e .

# С поддержкой данных (WikiText, sentencepiece)
pip install -e ".[data]"

# Полная установка (wandb, tensorboard, matplotlib)
pip install -e ".[all]"
```

**Требования:**
- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- CUDA опционален — все скрипты автоматически выбирают `cuda` или `cpu`

---

## 2. Быстрый старт

### Запуск одной итерации пайплайна

```bash
python pipeline.py --checkpoint hmoe_self_trained_v4.pt
```

### Самообучение HMoE (figure-8)

```bash
python self_train_hmoe.py                          # стандартный запуск
python self_train_hmoe.py --fast                   # 2 цикла, 20 шагов (отладка)
python self_train_hmoe.py --cycles 6 --steps_per_loop 100
python self_train_hmoe.py --no-corpus              # только синтетические данные
```

### Базовое самообучение

```bash
python self_train.py       # 3 стадии: Self-Topology → RAG → Wild
python self_train_v2.py    # + domain triplet + gate entropy loss
python self_train_v3.py    # + Stage0 = figure-8 обход Q6
```

---

## 3. Скрипты обучения

| Скрипт | Метод | LCI-формула | Ключевые параметры |
|--------|-------|------------|-------------------|
| `self_train.py` | 3-стадийное самообучение | B (embedding) | `--cycles`, `--corpus` |
| `self_train_v2.py` | То же + 3 вспомог. лосса | B | `--triplet-w`, `--entropy-w` |
| `self_train_v3.py` | + Stage0 figure-8 | B | `--stage0-cycles` |
| `self_train_hmoe.py` | HMoE + figure-8 | A (routing) + B | `--cycles`, `--steps_per_loop`, `--fast` |
| `self_train_hmoe_v4.py` | HMoE v4 | A + B | — |
| `figure8_turbine.py` | 4 агента, TSP-маршрут | A | `--lci-loss`, `--cycles`, `--steps_per_expert` |
| `nautilus_4agent.py` | 4 агента по кольцам | A + B | `--cycles`, `--step-scale`, `--bent-seeds` |
| `nautilus_clover.py` | Клеверный маршрут | A | — |
| `nautilus_15agent.py` | 15 агентов (Q4⊂Q6) | A | `--cycles`, `--steps`, `--fast`, `--no-bent` |
| `roundabout.py` | Кольцо с адаптивными оборотами | A | `--cycles`, `--lap-steps`, `--max-laps`, `--fast` |
| `bidir_turbine.py` | Два встречных потока | A | — |
| `bidir_train.py` / `bidir_train_v2.py` | Bidirectional обучение | B | — |
| `multi_salesman.py` | 2–3 агента с общим RAG | A | `--n-agents` |

### Детальное описание основных скриптов

#### `self_train_hmoe.py` — Самообучение HMoE

Реализует алгоритм Скарабея в три фазы:

```
[Инициализация]
  Загрузить модель из --checkpoint
  Инициализировать RAG-буфер (bent seeds)

[Цикл × N]
  Петля A (ABSTRACT)  → заморозить CONCRETE, DYNAMIC
  Точка X (DYNAMIC)   → размороженный BidirBridgeExpert
  Петля B (CONCRETE)  → заморозить ABSTRACT, DYNAMIC

  Вычислить LCI_A (routing) и LCI_B (embedding)
  Адаптировать температуру генерации

[Финал]
  Сохранить чекпоинт + JSON-лог
```

Логируемые метрики:
- `avg_lci_r` — routing LCI (Formula A, цель = π)
- `avg_lci_emb` — embedding LCI (Formula B, цель ≈ π при диапазоне [0, 4π])
- `resonance_rate` — доля циклов в резонансе
- `kirchhoff_ok` — доля с Kirchhoff-балансом

#### `nautilus_4agent.py` — 4-агентный Наутилус

4 агента, каждый специализируется на одном кольце:

```
META (вся модель) → ABSTRACT → DYNAMIC → CONCRETE → META → ...
```

Ключевые флаги:
- `--bent-seeds` — использовать математически оптимальные seed-векторы
- `--step-scale 0.4` — масштаб числа шагов (0.4 × base = 100 шагов/цикл)
- `--fast` — режим быстрой отладки (8 циклов, минимум шагов)

#### `figure8_turbine.py` — Турбина figure-8

4 «топлива» (эксперта) с TSP-маршрутизацией:

```
Greedy/2-opt TSP → оптимальный порядок обхода экспертов
Каждый эксперт обучает свою петлю figure-8
```

#### `bidir_train.py` / `bidir_train_v2.py` — Двунаправленное обучение

Замкнутый цикл: модель обучается из графа знаний и одновременно пополняет его.

```
ВПЕРЁД: Корпус → PageRank-центры → Q6-якоря → micro_train()
НАЗАД:  модель → генерация → QualityFilter → граф → identify_gaps()
```

`v2` закрывает конкретные пробелы (текстов с низким CD/VT из `graph_health.py`). Лог: `bidir_train_log.json`, `bidir_train_v2_log.json`.

#### `roundabout.py` — Кольцо с адаптивными оборотами

Агент движется по кольцу экспертов, число оборотов определяется LCI-прогрессом — не фиксированное, а до достижения порога.

```bash
python roundabout.py --checkpoint model.pt --cycles 4 --max-laps 5 --lap-steps 20
python roundabout.py --fast                   # smoke test без checkpoint
```

Аргументы: `--cycles` (дефолт 2), `--lap-steps` (шагов за оборот, дефолт 8), `--max-laps` (дефолт 2), `--temperature`, `--lr`, `--fast`, `--no-train`.


#### `multi_salesman.py` — K агентов с общим RAG

2–3 агента, каждый обслуживает свой кластер доменов, все делят один RAG-буфер. Встреча происходит в узле DYNAMIC.

```bash
python multi_salesman.py --n-agents 2 --checkpoint model.pt
```

#### `nautilus_15agent.py` — 15-агентный Наутилус

Каждый агент обрабатывает один Q4-тессеракт (C(6,4)=15 копий Q4 в Q6). Все 15×16=240 вершинных маршрутов покрывают 60/64 Q6-вершин.

```bash
python nautilus_15agent.py --checkpoint model.pt --cycles 4 --steps 15
python nautilus_15agent.py --fast               # smoke test без checkpoint
```

Аргументы: `--cycles` (дефолт 2), `--steps` (шагов/агент/цикл, дефолт 3 в fast), `--temperature`, `--temp-decay`, `--lr`, `--fast`, `--no-bent` (отключить bent seeds), `--no-train`, `--save`.


#### `nautilus_clover.py` — 4-листный клевер

Специальная топология для синтеза между экспертами: 4 листа (ABSTRACT, DYNAMIC, CONCRETE, META) встречаются в центральном узле. Акцент на cross-leaf аналогии.

Лог-ключ: `lci_r_final`.

#### `train_e2.py` / `train_e2_clusters.py` / `train_e2_joint.py` — E2-обучение

Обучение `HierarchicalE2` по 5 фазам снизу вверх:

| Фаза | α | Уровень | Что обучается |
|------|---|---------|--------------|
| 1 | -4 | GlyphLevel | Q6-проекция, кластеризация |
| 2 | -2 | CoreLevel | Variant3 блоки |
| 3 | 0 | MethodLevel | ArchetypalInterlingua |
| 4 | +2 | TheoryLevel | NautilusHierarchy |
| 5 | +4 | PhiloLevel | ConvergenceBridge + MatrixGrammar |

```bash
# Базовое E2-обучение
python train_e2.py --checkpoint model.pt --phases 1,2,3

# По кластерам репо (7 кластеров поочерёдно)
python train_e2_clusters.py --checkpoint model.pt

# Joint: 40% внешний + 60% внутренний корпус
python train_e2_joint.py --checkpoint model.pt --external-ratio 0.4
```

Логи: `train_e2_log.json`, `train_e2_clusters_log.json`, `train_e2_joint_log.json`.

#### `train_hmoe_staged.py` — 5-этапное HMoE-обучение

Последовательное размораживание: MicroExperts → GroupRouters → GlobalRouter → BridgeExperts → JointFinetune. Каждый этап сохраняет чекпоинт.

#### `train_hmoe_curriculum.py` — Curriculum по α-уровням

6 стадий curriculum в HMoE от α=-4 до α=+4. Соответствует `set_moe_stage()` из `hierarchical_moe.py`. Лог: `train_hmoe_log.json`.

---

## 4. Curriculum Pipeline

**Файл:** `pipeline.py`

### Структура

```
Фаза 1: N × nautilus-4agent
   step_scale=0.4, 8 циклов каждый
   цель: накопление разнообразия, подъём LCI

Фаза 2: figure8_turbine + LCI-loss
   8 циклов, lci-loss=0.1
   цель: стабилизация, Kirchhoff-балансировка

Фаза 3: bench_all (финальный бенчмарк)
   14 конфигураций: варианты 5,8,3,6,7,9,10,13,14
```

### Использование

```bash
# Стандартный запуск (2 прохода Наутилуса)
python pipeline.py --checkpoint hmoe_self_trained_v4.pt

# С явным числом проходов
python pipeline.py --checkpoint hmoe_self_trained_v4.pt --passes 2

# Быстрый режим (отладка)
python pipeline.py --checkpoint hmoe_self_trained_v4.pt --fast

# Отключить адаптивный LR
python pipeline.py --checkpoint hmoe_self_trained_v4.pt --no-adaptive-lr
```

### Параметры `run_pipeline()`

| Параметр | По умолчанию | Описание |
|----------|------------|---------|
| `start_checkpoint` | обязателен | путь к `.pt` файлу (проверяется наличие) |
| `n_nautilus_passes` | 2 | число проходов Наутилуса |
| `nautilus_cycles` | 8 | циклов за один проход |
| `nautilus_step_scale` | 0.4 | масштаб шагов |
| `turbine_cycles` | 8 | циклов в фазе 2 |
| `turbine_spe` | 8 | шагов на эксперта в турбине |
| `turbine_lci_loss` | 0.1 | вес LCI-loss в турбине |
| `adaptive_lr` | True | снижать lr до 5e-6 при LCI > 2.8 |
| `reset_rag_pass` | 3 | с какого прохода использовать bent seeds |
| `lr_threshold` | 2.8 | порог LCI для снижения lr |

### Адаптивный LR

Если `LCI > lr_threshold` (по умолчанию 2.8) начиная со 2-го прохода:

```
lr: 1e-4 → 5e-6  (снижение в 20×)
```

Это предотвращает forgetting когда модель уже хорошо обучена.

### RAG reset (bent seeds)

С прохода `reset_rag_pass` (по умолчанию 3) вместо случайных seed-векторов используются «bent seeds» — математически оптимальные архетипы Q6 с максимальным cosine diversity (+33% по сравнению со случайными).

---

## 5. Алгоритм Скарабея (Figure-8)

Назван по траектории движения навозного жука, описывающего восьмёрку.

### Топология

```
ABSTRACT (петля A, zoom-out)
    ↓       ↑
DYNAMIC (точка X, пересечение)  ← BidirBridgeExpert
    ↓       ↑
CONCRETE (петля B, zoom-in)
```

### LCI-контроль

```
|w_ABSTRACT - w_CONCRETE| → 0   (петли уравновешены)
DYNAMIC ≥ 0.20               (точка пересечения жива)
```

При резонансе LCI_A → π: модель одинаково хорошо обобщает (абстракция) и специализирует (конкретика).

### Поэтапное замораживание

На каждом шаге замораживаются все группы кроме активной:

```python
# Петля A: обучаем только ABSTRACT
_freeze_all_except(model, "ABSTRACT")

# Точка X: размораживаем BidirBridgeExpert
_freeze_all_except(model, "DYNAMIC")

# Петля B: обучаем только CONCRETE
_freeze_all_except(model, "CONCRETE")
```

---

## 6. Чекпоинты

### Формат файла

```python
{
    "model_state": OrderedDict,     # веса модели (обязательно)
    "cfg": dict,                    # Variant3Config (опционально)
    "hmoe_cfg": dict,               # HMoEConfig (опционально)
    "step": int,                    # шаг обучения
    "avg_lci": float,               # LCI на момент сохранения
}
```

### Сохранение (безопасное)

```python
torch.save({"model_state": model.state_dict()}, "checkpoint.pt")
# weights_only=True при загрузке — обязательно (см. test_security.py)
```

### Загрузка

```python
import torch
ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
```

### Имеющиеся чекпоинты

| Файл | Создан | Примечание |
|------|--------|-----------|
| `hmoe_curriculum.pt` | `train_hmoe_curriculum.py` | Curriculum стадии 1-3 |
| `hmoe_curriculum_fixed.pt` | после патча STE | STE zero-gradient исправлен |
| `hmoe_fixed_joint.pt` | Joint-training | Совместное обучение |
| `hmoe_fixed_self.pt` | Self-training | Базовое самообучение |
| `hmoe_joint_base.pt` | Joint-training | Отправная точка |
| `hmoe_self_trained_v2.pt` | `self_train_hmoe.py` поколение 2 | — |
| `hmoe_self_trained_v3.pt` | — поколение 3 | — |
| `hmoe_self_trained_v4.pt` | — поколение 4 | Рекомендуемый |
| `hmoe_v2_self.pt` | Альтернативный путь | Параллельный ряд |
| `hmoe_v3_self.pt` | — | — |
| `hmoe_v4_self.pt` | — | — |

**Рекомендуемый стартовый чекпоинт:** `hmoe_self_trained_v4.pt`

---

## 7. Бенчмарк

### Все варианты

```bash
python bench_all.py --checkpoint hmoe_self_trained_v4.pt
python bench_all.py --checkpoint hmoe_self_trained_v4.pt --variants 3,5,8
```

### Результаты (800 шагов, d=128)

| Вариант | Конфигурация | PPL |
|---------|-------------|-----|
| baseline | vanilla | 2.94 |
| v58 | BridgeOfModules | 1.35 |
| **v59** | **AbrialeBridgeMediator** | **1.24 ★** |
| v60 | ArchetypalInterlingua | 2.93 |
| v61 | BridgedInterlingua | 2.92 |

### Метрики качества LCI

| Метрика | Описание | Цель |
|---------|---------|------|
| `avg_LCI_r` | Routing LCI (Formula A) | π ≈ 3.14 |
| `avg_lci_emb` | Embedding LCI (Formula B) | ≈ π (шкала [0, 4π]) |
| `resonance_rate` | % циклов с LCI близко к π | ≥ 50% |
| `kirchhoff_ok` | % циклов с Kirchhoff-балансом | ≥ 50% |
| `score` | 0.40×LCI + 0.35×res + 0.25×kirch | выше = лучше |

---

## 8. Диагностика

### Прогон тестов

```bash
pytest                          # все тесты (≈26 секунд, 1642 теста)
pytest yijing_transformer/tests/test_hierarchical_moe.py  # только HMoE (37 тестов)
pytest yijing_transformer/tests/test_variant3.py           # только Variant3 (70 тестов)
pytest tests/test_smoke.py                                 # smoke tests (9 тестов)
pytest tests/test_security.py                              # security checks
```

### Частые проблемы

#### `FileNotFoundError` при запуске pipeline

```
pipeline.py start_checkpoint не найден: 'path.pt'
```

Убедитесь, что файл чекпоинта существует перед запуском. Проверьте текущую директорию:

```bash
ls *.pt
python pipeline.py --checkpoint hmoe_self_trained_v4.pt
```

#### `[warn] _get_q6_vertex: Q6-проекция не найдена`

Модель загружена без `use_hierarchical_moe=True` или с `use_hex_tier=False`. Это нормальное предупреждение — fallback к orbit 3.

#### LCI не сходится к π

- Проверьте `resonance_rate` в логах: если < 10% — попробуйте `--fast` для быстрой диагностики
- Уменьшите `--lr` (по умолчанию 1e-4)
- Попробуйте `--bent-seeds` для лучшей инициализации RAG

#### Зависание на многоядерных CPU

Переменная `OMP_NUM_THREADS=1` устанавливается автоматически в `pipeline.py`. Если другие скрипты зависают:

```bash
OMP_NUM_THREADS=1 python self_train_hmoe.py
```

## 9. Токенизаторы

**Файл:** `yijing_transformer/tokenizer/char_tokenizer.py`

| Класс | Описание | Vocab | Использование |
|-------|---------|-------|--------------|
| `CharTokenizer` | character-level (ASCII + Unicode) | динамический | общий случай |
| `ByteTokenizer` | byte-level (256 байт) | 259 (+ PAD/BOS/EOS) | языко-независимый |
| `BPETokenizer` | Byte-Pair Encoding, без зависимостей | обучаемый | домен-специфичный |

```python
from yijing_transformer.tokenizer.char_tokenizer import CharTokenizer, ByteTokenizer, BPETokenizer

# ByteTokenizer — рекомендуется для большинства случаев
tok = ByteTokenizer()
ids = tok.encode("Hello Q6")                    # [72, 101, 108, ...]
text = tok.decode(ids)                           # "Hello Q6"
ids_special = tok.encode_with_special("Hello")  # [BOS, 72, 101, ..., EOS]

# BPETokenizer
bpe = BPETokenizer()
bpe.train(texts, vocab_size=512)
bpe.save('my_bpe.json')
bpe2 = BPETokenizer.load('my_bpe.json')
```

**GlyphTokenizer** (`tokenizer/glyph_tokenizer.py`) — геометрическая токенизация через SOLAN (76 глифов). Сравнение с CharTokenizer: `experiments/train_with_glyph.py`.

---

## 10. Бенчмарки и диагностика

### bench_moe.py — сравнение FFN-архитектур

Сравнивает 3 FFN-реализации: `SwiGLU`, `HierarchicalMoEFFN`, `Q6ExpertBank`.

```bash
python bench_moe.py --steps 200
```

Выводит: параметры, время forward/backward, PPL, routing diversity (entropy).

### bench_stability.py — воспроизводимость

Запускает `bench_all.py` N раз, проверяет std score.

```bash
python bench_stability.py --n-runs 5
```

Успех: std < 0.005.

### eval_hmoe.py — диагностика HMoE

Детальная диагностика `HierarchicalMoEFFN` с рекомендациями по устранению проблем.

```bash
python eval_hmoe.py --checkpoint hmoe_self_trained_v4.pt
```

Проверяет:
- Баланс загрузки экспертов (восьмёрка симметрична?)
- Точка пересечения figure-8 (DYNAMIC активен?)
- Routing entropy (не коллапсирует ли маршрутизация?)

### scripts/multi_domain_benchmark.py — 39-доменный бенчмарк

Универсальность Q6: проверяет PPL на 39 доменах (код, математика, история, ...).

```bash
python scripts/multi_domain_benchmark.py --checkpoint model.pt
```

### scripts/fetch_svend4_corpus.py — загрузка корпусов

Клонирует все svend4-репозитории для обучения.

```bash
python scripts/fetch_svend4_corpus.py --target ..
# Клонирует: meta, info1, data2, data7, infosystems, ai_agents, knowledge
```

---

### Мониторинг через JSON-логи

Все скрипты сохраняют JSON-лог рядом с чекпоинтом (суффикс `_log.json`):

```python
import json
with open("hmoe_self_trained_v4_log.json") as f:
    log = json.load(f)
print(log[-1])  # последний цикл
# {'cycle': 12, 'avg_lci_r': 3.21, 'avg_lci_emb': 3.08, 'resonance_rate': 0.67, ...}
```

Поддерживаемые форматы лога (читает `pipeline.read_avg_lci()`):
- `avg_lci_all` — nautilus_4agent
- `avg_lci_r` — figure8_turbine
- `avg_LCI_r` — устаревший формат
- `lci_r_final` — nautilus_clover
- `avg_lci` — multi_salesman, bidir
