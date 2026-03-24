# Документация YiJing Transformer

Справочный индекс всех технических документов.

---

## Навигация

| Документ | Аудитория | Содержание |
|----------|-----------|-----------|
| [README.md](../README.md) | Все | Обзор проекта, быстрый старт, бенчмарки |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Разработчики | Математическая основа, компоненты, конфигурации |
| [MODELS.md](MODELS.md) | Разработчики | Все модели: когда использовать, параметры, примеры |
| [TRAINING.md](TRAINING.md) | ML-инженеры | Pipeline, скрипты обучения, алгоритм Скарабея |
| [TRAINING_UTILS.md](TRAINING_UTILS.md) | ML-инженеры | TrainingBridge, оптимизаторы, мониторинг |
| [GEOMETRY.md](GEOMETRY.md) | Исследователи | Модуль geometry/: квантизаторы, attention, routing |
| [INFERENCE.md](INFERENCE.md) | Пользователи | E2Inference, генерация, Q6-эмбеддинг, экспорт |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Контрибьюторы | Тесты, стиль кода, добавление архитектур |
| **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** | **Все** | **Что реализовано и где искать — первый документ при code review** |
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | **Исследователи** | **Все эксперименты: запуск, критерии, результаты** |
| **[PORTAL.md](PORTAL.md)** | **Разработчики** | **Portal-система, адаптеры репо, федеративное обучение, граф знаний** |
| **[CHANGELOG.md](CHANGELOG.md)** | **Все** | **Хронология изменений по сессиям** |

---

## Карта концепций

```
Q6-гиперкуб {-1,+1}⁶
  └── 64 гексаграммы И-Цзин
        ├── 6 доменов (GEO/HYDRO/PYRO/AERO/COSMO/NOOS)
        ├── 3 группы (ABSTRACT/DYNAMIC/CONCRETE)
        └── Граф Бян-Гуа (Хэмминг-1 смежность)

Модели
  ├── VanillaGPT            ← baseline (честное сравнение)
  ├── LeanYiJingGPT         ← минимальный Q6-baseline
  ├── YiJingGPT (v8)        ← полная модель с геометрией
  ├── Variant3GPT           ← production (HMoE + BianGua)
  ├── HierarchicalE2        ← 5-уровневая иерархия
  └── NautilusYiJing        ← MoE + Q6 routing

Обучение
  ├── self_train*.py         ← самообучение (3 стадии)
  ├── self_train_hmoe.py     ← figure-8 (Алгоритм Скарабея)
  ├── nautilus_4agent.py     ← 4-агентный Наутилус
  ├── figure8_turbine.py     ← турбина + TSP
  └── pipeline.py            ← curriculum (фазы 1→2→3)

LCI (Loop Closure Index)
  ├── Formula A: (1 - |w_A - w_B|) × π   ∈ [0, π]
  ├── Formula B: arccos(cosine_sim) × 4   ∈ [0, 4π]
  └── Цель: LCI → π (резонанс)
```

---

## Быстрый поиск

**Хочу обучить модель с нуля:**
→ [TRAINING.md — Быстрый старт](TRAINING.md#2-быстрый-старт)

**Хочу использовать готовую модель для инференса:**
→ [INFERENCE.md](INFERENCE.md)

**Хочу понять архитектуру HierarchicalMoE:**
→ [ARCHITECTURE.md — HierarchicalMoEFFN](ARCHITECTURE.md#4-hierarchicalmoe--4-уровневая-moe)

**Хочу добавить новый attention паттерн:**
→ [GEOMETRY.md — attention.py](GEOMETRY.md#3-attentionpy--механизмы-внимания)
→ [CONTRIBUTING.md](CONTRIBUTING.md)

**Хочу применить LoRA / Speculative Decoding:**
→ [MODELS.md — Вспомогательные модули](MODELS.md#8-вспомогательные-модули)

**Хочу понять что такое LCI:**
→ [ARCHITECTURE.md — LCI](ARCHITECTURE.md#6-lci--loop-closure-index)

**Хочу добавить свой оптимизатор:**
→ [TRAINING_UTILS.md — utils_v* таблица](TRAINING_UTILS.md#12-таблица-источников)

**Хочу понять что уже реализовано (code review):**
→ [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)

**Хочу запустить эксперимент / проверку:**
→ [EXPERIMENTS.md](EXPERIMENTS.md)
→ `./run_all_checks.sh`

**Хочу узнать что изменилось в последней сессии:**
→ [CHANGELOG.md](CHANGELOG.md)

**Хочу исправить баг ArchetypalInterlingua (PPL=vanilla):**
→ [IMPLEMENTATION_STATUS.md — interlingua_fixed](IMPLEMENTATION_STATUS.md#2-archetypalinterlinguafixed)
→ `yijing_transformer/models/geometry/interlingua_fixed.py`

**Хочу добавить Q6-routing вместо softmax:**
→ [IMPLEMENTATION_STATUS.md — KasatkinRouter](IMPLEMENTATION_STATUS.md#3-kasatkinq6router)
→ `yijing_transformer/models/geometry/kasatkin_router.py`

**Хочу понять систему Portal и адаптеров:**
→ [PORTAL.md](PORTAL.md)

**Хочу загрузить данные из других репозиториев:**
→ [PORTAL.md — Корпус-лоадеры](PORTAL.md#3-корпус-лоадеры)
→ `corpus_loader.py`, `repo_corpus_loader.py`

**Хочу проверить здоровье графа знаний:**
→ [PORTAL.md — Граф знаний](PORTAL.md#4-граф-знаний-и-метрики)
→ `graph_health.py` (метрики CD, VT, CR, DB)

**Хочу синхронизировать модель с другими репо:**
→ [PORTAL.md — Федеративное обучение](PORTAL.md#6-федеративное-обучение)
→ `federated_round.py`

**Хочу понять как устроен токенизатор:**
→ [TRAINING.md §9](TRAINING.md#9-токенизаторы)
→ `tokenizer/char_tokenizer.py` (CharTokenizer, ByteTokenizer, BPETokenizer)

**Хочу запустить двунаправленное обучение:**
→ [TRAINING.md §3 — bidir_train](TRAINING.md#3-скрипты-обучения)
→ `bidir_train.py` / `bidir_train_v2.py`

**Хочу обучить E2-модель по фазам:**
→ [TRAINING.md §3 — train_e2](TRAINING.md#3-скрипты-обучения)
→ `train_e2.py` / `train_e2_clusters.py` / `train_e2_joint.py`
