# Implementation Status — Что реализовано и где искать

> **Актуально:** 2026-03-24  
> Связанные документы: [INDEX.md](INDEX.md) · [EXPERIMENTS.md](EXPERIMENTS.md) · [GEOMETRY.md](GEOMETRY.md) · [ARCHITECTURE.md](ARCHITECTURE.md)  
> Для истории изменений: [CHANGELOG.md](CHANGELOG.md)

**Назначение этого документа:** при code review или поиске по кодовой базе — смотреть сюда первым. Каждый раздел отвечает на вопрос «это уже сделано или нет?».

---

## Сводная таблица: теория → код

| Идея / концепция | Статус | Файл | Дата |
|------------------|--------|------|------|
| Q6 = {-1,+1}^6 гиперкуб | ✅ реализовано | `models/variant3.py`, `geometry/quantizers.py` | до 2026-03 |
| 64 гексаграммы = якоря | ✅ реализовано | `geometry/interlingua_fixed.py` | 2026-03-24 |
| HierarchicalMoE (4 уровня) | ✅ реализовано | `models/hierarchical_moe.py` | до 2026-03 |
| AbrialeBridgeMediator (v59, PPL 1.24) | ✅ реализовано | `geometry/routing.py` | до 2026-03 |
| ArchetypalInterlingua **БАГ** | ✅ ИСПРАВЛЕНО | `geometry/interlingua_fixed.py` | 2026-03-24 |
| KasatkinQ6Router (3D routing) | ✅ реализовано | `geometry/kasatkin_router.py` | 2026-03-24 |
| Diversity loss для trit_proj | ✅ реализовано | `geometry/interlingua_fixed.py` | 2026-03-24 |
| Q6ExpertBank | ✅ реализовано | `geometry/kasatkin_router.py` | 2026-03-24 |
| Gumbel-Softmax вместо STE | ✅ реализовано | `geometry/interlingua_fixed.py`, `geometry/quantizer_fixed.py` | 2026-03-24 |
| Q4⊂Q6 верификация | ✅ реализовано | `experiments/validate_q4_q6.py` | 2026-03-24 |
| Ксерокс-тест (само-осознание) | ✅ реализовано | `experiments/xerox_test.py` | 2026-03-24 |
| GlyphTokenizer сравнение | ✅ реализовано | `experiments/train_with_glyph.py` | 2026-03-24 |
| e2_self_improve мониторинг | ✅ реализовано | `monitor_improve.py` | 2026-03-24 |
| Полный пайплайн проверок | ✅ реализовано | `run_all_checks.sh` | 2026-03-24 |
| Walsh-Hadamard применение | ⏳ теория | `theoretical-foundations.md` | — |
| PseudoRAG как учитель YiJing | ⏳ теория | `PSEUDORAG_YIJING_BRIDGE.md` | — |

---

## 1. Ядро архитектуры (до 2026-03-24)

### variant3.py — основная модель

**Файл:** `yijing_transformer/models/variant3.py`

| Класс / функция | Назначение |
|----------------|------------|
| `HexagramProjection` | проекция токенов в Q6 пространство |
| `BianGuaAttention` | attention с Hamming-distance bias |
| `TernaryGate` | активация {-1, 0, +1} через STE |
| `CrossHexagramAnalogy` | аналогии через Hamming-1 соседей |
| `NautilusYiJinRouter` | маршрутизация по 6 доменам |
| `Variant3Block` | полный блок (все компоненты) |
| `Variant3GPT` | полная модель |
| `hamming_distance_soft` | дифференцируемое Hamming |
| `biangua_path` | путь между гексаграммами |
| `get_dominant_hexagram` | доминирующая гексаграмма |

### hierarchical_moe.py — HierarchicalMoE

**Файл:** `yijing_transformer/models/hierarchical_moe.py`

| Класс | Назначение |
|-------|------------|
| `Q6ExpertBank` | 64 Q6 эксперта (векторизованных) |
| `MicroExpert` | лёгкий FFN на кластер |
| `BidirBridgeExpert` | двунаправленный мост |
| `GroupRouter` | routing внутри группы |
| `GlobalRouter` | глобальный routing |
| `MultiScaleGlobalRouter` | Matryoshka Q2→Q3→Q6 routing |
| `HierarchicalMoEFFN` | полная 4-уровневая MoE |
| `set_moe_stage()` | управление стадией curriculum |

---

## 2. ArchetypalInterlinguaFixed

**Файл:** `yijing_transformer/models/geometry/interlingua_fixed.py`  
**Тест:** `python yijing_transformer/models/geometry/interlingua_fixed.py`  
**Документация теста:** [EXPERIMENTS.md — Тест Interlingua](EXPERIMENTS.md#2-тест-archetypalinterlinguafixed)

### Что было неправильно (баг в оригинале)

Оригинальный `ArchetypalInterlingua` (`geometry/convergence.py`) имел один общий `trit_proj`:
```python
# БАГ: один проектор для всех N источников
self.trit_proj = nn.Linear(d_model, n_archetypes)
```

Результат: все источники давали одинаковые триты → 64 архетипа неразличимы → gate застрял на 0.49 → PPL = vanilla (2.92-2.94).

### Что исправлено

```python
# ИСПРАВЛЕНИЕ: N отдельных проекторов
self.trit_projs = nn.ModuleList([
    nn.Linear(d_model, n_archetypes, bias=True)
    for _ in range(n_sources)
])
```

Плюс три дополнительных улучшения:

1. **Diversity loss** — штраф за косинусное сходство между источниками
2. **Gumbel-Softmax** вместо STE (нет zero-gradient trap)
3. **Cosine annealing** температуры: `start_temp=1.0` → `end_temp=0.05` за `warmup_steps`

### API

```python
from yijing_transformer.models.geometry.interlingua_fixed import ArchetypalInterlinguaFixed

model = ArchetypalInterlinguaFixed(
    d_model=128,
    n_sources=7,
    n_archetypes=64,
    diversity_weight=0.01,
    warmup_steps=3000,
    start_temp=1.0,
    end_temp=0.05,
)

# Forward
output, aux_loss = model(source_outputs, core_hidden)

# Диагностика (вызывать каждые 200 шагов)
d = model.diagnostics(source_outputs)
# d['gate'], d['diversity_loss'], d['src0_pos'], ...
```

### Здоровые значения диагностики

| Метрика | Норма | Проблема |
|---------|-------|----------|
| `diversity_loss` | > 0.0 | ≤ 0.0 |
| `gate` | 0.3 – 0.7 | < 0.1 или > 0.9 |
| `src{i}_pos spread` | > 0.03 | < 0.01 |

---

## 3. KasatkinQ6Router

**Файл:** `yijing_transformer/models/geometry/kasatkin_router.py`  
**Тест:** `python yijing_transformer/models/geometry/kasatkin_router.py`  
**Документация теста:** [EXPERIMENTS.md — KasatkinRouter](EXPERIMENTS.md#3-тест-kasatkinq6router)

### Концепция

Вместо `routing = softmax(W · x)`:
```
x (B, T, d_model)
  → proj_3d (B, T, 3)       — проекция в куб Касаткина
  → cosine с 6 осями (±X, ±Y, ±Z)
  → softmax(similarity / T)
  → routing_weights (B, T, 6)
```

Каждая ось = один домен = один эксперт. Routing **объяснимо** — токен буквально «указывает» на ось.

### Классы

```python
from yijing_transformer.models.geometry.kasatkin_router import (
    KasatkinQ6Router,
    Q6ExpertBank,
    DOMAIN_NAMES,   # ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]
    DOMAIN_ALT,     # ["CODE", "RECON", "SYSTEM", "MATH", "HUMAN", "INFO"]
)

# Только routing weights
router = KasatkinQ6Router(d_model=128, routing_temperature=0.5)
weights = router(x)  # (B, T, 6)

# Человекочитаемые метки по токенам
labels = router.hex_label(x[0])  # [{domain, alt_name, coord_3d, ...}]

# Уверенность routing (цель > 0.15)
conf = router.get_routing_confidence(x)

# Готовый MoE-блок: router + 6 FFN
bank = Q6ExpertBank(d_model=128, d_ffn=512)
out = bank(x)  # (B, T, d_model)
```

### Протокол реального теста

Тест считается действительным только после:
- ≥ 3000 шагов на реальных данных
- `routing_confidence > 0.15`
- PPL не хуже LeanYiJing baseline

---

## 4. Ксерокс-тест

**Файл:** `experiments/xerox_test.py`  
**Результат:** `experiments/xerox_test_result.json`  
**Запуск:** [EXPERIMENTS.md — Ксерокс-тест](EXPERIMENTS.md#4-ксерокс-тест--само-осознание-модели)

14 тест-кейсов в 4 доменах. Модель должна набрать score ≥ 0.80 до начала масштабирования.

**Последний результат (2026-03-24, mock-режим):** 6/14 → ожидаемо, нужна реальная модель.

---

## 5. Скрипты обучения (до 2026-03-24)

**Директория:** `/` (корень проекта)

| Скрипт | Назначение | Результат |
|--------|------------|-----------|
| `self_train.py` | 3 стадии: Q6 → RAG → Wild | — |
| `self_train_v2.py` | + domain triplet + gate entropy loss | — |
| `self_train_v3.py` | + Stage0 figure-8 (Скарабей) | — |
| `self_train_hmoe.py` | HMoE-специфичный | `hmoe_self_trained_v*.pt` |
| `figure8_turbine.py` | 4 эксперта, TSP (greedy/2-opt) | LCI=2.983 |
| `nautilus_4agent.py` | 4 агента на META/ABSTRACT/DYNAMIC/CONCRETE | — |
| `pipeline.py` | curriculum: nautilus → turbine → bench | score=0.998, LCI=3.127 |

---

## 6. Чекпоинты (сохранённые модели)

| Файл | Размер | Алгоритм | PPL / LCI |
|------|--------|----------|-----------|
| `hmoe_self_trained_v4.pt` | лучший | Алгоритм Скарабея v4 | — |
| `hmoe_self_trained_v3.pt` | — | v3 | — |
| `hmoe_curriculum_fixed.pt` | — | curriculum 6-stage | — |
| `hmoe_v4_self.pt` | — | self-train v4 | — |

**Рекомендуемый для тестов:** `hmoe_self_trained_v4.pt`

---

## 7. Известные результаты бенчмарков

| Архитектура | PPL | Статус |
|------------|-----|--------|
| AbrialeBridgeMediator (v59) | **1.24** | ЛУЧШИЙ |
| BridgeOfModules (v58) | 1.35 | — |
| vanilla baseline | 2.94 | — |
| ArchetypalInterlingua (v60, без фикса) | 2.93 | ≈ vanilla |
| BridgedInterlingua (v61, без фикса) | 2.92 | ≈ vanilla |

**Вывод:** v60/v61 давали PPL = vanilla именно из-за бага в `trit_proj`. После замены на `interlingua_fixed.py` ожидается улучшение.

---

## 8. Что ещё НЕ реализовано (теория)

| Концепция | Документ | Статус |
|-----------|----------|--------|
| Walsh-Hadamard трансформ | `theoretical-foundations.md` | теория |
| PseudoRAG → учитель YiJing | `PSEUDORAG_YIJING_BRIDGE.md` | частично (Q4⊂Q6 = 2.56) |
| 19 формальных теорем в коде | `theoretical-foundations.md` | теория |
| ArchetypalInterlingua с фиксом в production | — | нужно подключить |

---

*Последнее обновление: 2026-03-24 | Ветка: `claude/repository-audit-fvlEG`*
