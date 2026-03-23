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
