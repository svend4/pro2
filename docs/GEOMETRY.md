# Справочник модуля `geometry/`

**Расположение:** `yijing_transformer/models/geometry/`
**Размер:** 15 файлов, ~7 300 строк

Модуль реализует математическую инфраструктуру проекта: генерацию кодбуков, механизмы внимания, квантизаторы, маршрутизаторы и специализированные FFN. Все компоненты импортируются из единой точки входа `geometry/__init__.py`.

```python
from yijing_transformer.models.geometry import (
    get_hexagrams, BianGuaTransform, TernaryQuantizer,
    GatedPathSelector, SwiGLU, NautilusHierarchy, ...
)
```

---

## Содержание

1. [core.py — Генерация геометрии](#1-corepy--генерация-геометрии)
2. [quantizers.py — Квантизаторы](#2-quantizerspy--квантизаторы)
3. [attention.py — Механизмы внимания](#3-attentionpy--механизмы-внимания)
4. [routing.py — Маршрутизация и гейты](#4-routingpy--маршрутизация-и-гейты)
5. [ffn.py — FFN модули](#5-ffnpy--ffn-модули)
6. [equivariant.py — Эквивариантные слои](#6-equivariantpy--эквивариантные-слои)
7. [positional.py — Позиционные кодирования](#7-positionalpy--позиционные-кодирования)
8. [convergence.py — Convergence Bridge](#8-convergencepy--convergence-bridge)
9. [nautilus.py — Иерархия Наутилуса](#9-nautiluspy--иерархия-наутилуса)
10. [abriale.py — Абриале-слой](#10-abrijalepy--абриале-слой)
11. [kasatkin_router.py — Маршрутизатор Касаткина](#11-kasatkin_routerpy--маршрутизатор-касаткина)
12. [q6_algebra.py — Алгебра Z₂^6 и операции И-Цзин](#12-q6_algebrapy--алгебра-z26-и-операции-и-цзин)
13. [interlingua_fixed.py — Исправленная Интерлингва](#interlingua_fixedpy--исправленная-интерлингва)

---

## 1. `core.py` — Генерация геометрии

**587 строк.** Математические примитивы: генерация кодбуков, порядки гексаграмм, структурные функции.

### Генерация кодбуков

| Функция | Выход | Описание |
|---------|-------|---------|
| `generate_trigrams()` | `(8, 3)` | 8 триграмм = все вершины {-1,+1}³ |
| `generate_hexagrams()` | `(64, 6)` | 64 гексаграммы = все вершины {-1,+1}⁶ |
| `generate_hypercube(n)` | `(2ⁿ, n)` | Произвольный гиперкуб |
| `generate_octograms()` | `(256, 8)` | 8-мерный куб (сравнение с E8) |
| `generate_tetragrams()` | `(16, 4)` | 4-мерный куб |
| `generate_e8_roots()` | `(240, 8)` | 240 корней решётки E8 (112 перестановок + 128 полуцелых) |
| `generate_ternary_hypercube(n)` | `(3ⁿ, n)` | Тернарный гиперкуб {-1,0,+1}ⁿ |
| `get_hexagrams()` | `(64, 6)` | Кешированный вариант (рекомендуется) |
| `get_trigrams()` | `(8, 3)` | Кешированный вариант |

```python
from yijing_transformer.models.geometry import get_hexagrams, generate_e8_roots
hexagrams = get_hexagrams()   # (64, 6), float32
e8 = generate_e8_roots()      # (240, 8), float32
```

### Порядки гексаграмм

| Функция | Описание |
|---------|---------|
| `fuxi_order()` | Бинарный порядок Фуси (от 000000 к 111111) |
| `wenwang_order()` | Традиционный порядок Вэнь-вана |
| `palace_clusters()` | 8 дворцов — по 8 гексаграмм в каждом |

### Структурные функции

| Функция | Описание |
|---------|---------|
| `antipodal_pairs()` | Пары антиподальных гексаграмм (строго напротив в Q6) |
| `antipodal_index()` | Индексная карта: i → antipodal(i) |
| `triangular_distance_matrix(P)` | Треугольные расстояния (Андреев) |
| `palace_attention_mask(block_size)` | Block-sparse маска по 8 дворцам |
| `kasatkin_embedding()` | 64-векторное встраивание по Касаткину |
| `kasatkin_distance_matrix()` | Матрица расстояний (Касаткин) |
| `verify_yijing_properties(...)` | Проверка: антиподальность, уникальность, покрытие |

---

## 2. `quantizers.py` — Квантизаторы

**1 042 строки.** Все квантизаторы используют STE (Straight-Through Estimator) для обратного распространения.

| Класс | Строка | Кодбук | Описание |
|-------|--------|--------|---------|
| `YiJingQuantizer` | 18 | 64 гексаграммы {-1,+1}⁶ | Мягкая квантизация к Q6 через softmax(-dist²/T) |
| `E8Quantizer` | 37 | 240 корней E8 | Квантизация в решётке E8; опц. `adaptive_temp` |
| `FactoredYiJingQuantizer` | 72 | 8 × 8 = 64 | Факторизованный: триграмма 1 ⊗ триграмма 2 |
| `FourStateQuantizer` | 113 | 4 состояния | {Yang, Old Yang, Yin, Old Yin} = 4-состояний |
| `AntipodalQuantizer` | 146 | 32 пары | Квантизирует к паре (гексаграмма, антипод) |
| `HierarchicalQuantizer` | 184 | Q2 → Q6 | Matryoshka: сначала Q2, потом уточнение до Q6 |
| `DeformableQuantizer` | 242 | 64 + смещение | Обучаемое смещение кодбука (mutable centroids) |
| `GumbelQuantizer` | 286 | 64 | Gumbel-Softmax для straight-through при обучении |
| `GroupedQuantizer` | 340 | k × k | Группированные кодбуки (произвольное k) |
| `TernaryQuantizer` | 415 | {-1, 0, +1}⁶ | 729 тернарных вершин, ключ для BitNet-style |
| `PairedBitQuantizer` | 598 | 64 пары | Квантизует в 2 разные гексаграммы одновременно |
| `MatryoshkaQuantizer` | 750 | Q2→Q3→Q6 | Трёхуровневая иерархическая квантизация |
| `WHT_Quantizer` | 1068 | {-1,+1}^n | Walsh-Hadamard спектр → квантизация (Теорема 5); `use_spectral_loss` для регуляризации bent-подобия |

### Пример: TernaryQuantizer

```python
from yijing_transformer.models.geometry import TernaryQuantizer
q = TernaryQuantizer(d_model=128, threshold=0.25)
x_quantized, ternary_bits = q(x)  # x: (B, T, d)
# ternary_bits ∈ {-1, 0, +1}^6 — тернарная гексаграмма
```

### Пример: HierarchicalQuantizer (Matryoshka)

```python
from yijing_transformer.models.geometry import HierarchicalQuantizer
q = HierarchicalQuantizer(d_model=128)
# Уровни: Q2(4) → Q3(8) → Q6(64)
# loss содержит балансировочный компонент
out, commitment_loss = q(x)
```

---

## 3. `attention.py` — Механизмы внимания

**558 строк.** 14 типов специализированного внимания с Q6-геометрической структурой.

| Класс | Строка | Идея | Ключевая особенность |
|-------|--------|------|---------------------|
| `TriangularAttentionBias` | 17 | Андреев | Треугольные расстояния как bias |
| `PalaceAttention` | 34 | Скляров | Block-sparse по 8 дворцам (64-блочный Q6) |
| `QuadrantAttention` | 77 | — | Квадрантная структура контекста |
| `RecursiveCubeAttention` | 113 | — | Рекурсивное разделение по кубам |
| `WeavingLoomArchitecture` | 160 | — | Ткацкий станок (переплетение паттернов) |
| `BidirectionalTriangularAttention` | 221 | — | Двустороннее треугольное внимание |
| `CubeDiagonalAttention` | 239 | Касаткин | Диагональные паттерны 3D-куба |
| `HeisenbergAttention` | 257 | — | Неопределённость position ↔ momentum |
| `FlowerOfLifeGAT` | 285 | — | Graph Attention Network по Q6-топологии |
| `MobiusAttentionPattern` | 330 | — | Топологическое скручивание (Мёбиус) |
| `PrivilegedAxisAttention` | 386 | Касаткин | Выделение главной оси куба |
| `DualModeHead` | 402 | — | Инь/Ян режимы — два независимых head |
| `HexagramAttentionPattern` | 442 | — | 6-паттернов по шести линиям |
| `GeometricAttention` | 469 | — | Полностью геометрическое внимание |

### Пример: PalaceAttention

```python
from yijing_transformer.models.geometry import PalaceAttention
attn = PalaceAttention(d_model=128, n_heads=8)
y = attn(x)  # Block-sparse по 8 дворцам Q6
```

---

## 4. `routing.py` — Маршрутизация и гейты

**1 848 строк.** Архитектурный центр — все гейты, маршрутизаторы, curriculum-контроллеры.

### Гейты

| Класс | Строка | Описание |
|-------|--------|---------|
| `GatedPathSelector` | 12 | Гейт geometric vs standard FFN. `gate * x_geo + (1-gate) * x_std` |
| `AdaptiveGatedPathSelector` | 39 | Multi-head гейт с температурой; контентно-зависимый |
| `GateLogger` | 102 | Thread-safe логирование статистики gate_mean / gate_std |

### Маршрутизаторы источников

| Класс | Строка | Описание |
|-------|--------|---------|
| `TaskAwareRouter` | 77 | Strategy-aware routing — адаптируется к типу задачи |
| `GeometricSourceRouter` | 180 | Маршрутизация по геометрическому источнику данных |
| `GeometricSourceMixer` | 258 | Смешение N источников с обучаемыми весами |
| `SequentialSourceCurriculum` | 305 | Последовательный curriculum по источникам |

### Мосты (Bridge архитектуры)

| Класс | Строка | PPL v59 | Описание |
|-------|--------|---------|---------|
| `PairwiseBridge` | 348 | — | Попарные мосты между модулями |
| `LightweightBridge` | 404 | — | Облегчённый вариант (меньше параметров) |
| `BridgeOfModules` | 449 | 1.35 | Дерево мостов O(N log N) |
| `AbrialeBridgeMediator` | 577 | **1.24 ★** | Hub через 3 медиатора (лучший результат) |
| `AdaptiveBridgeOfModules` | 711 | — | Адаптивное дерево с обучаемыми весами |

### Интерлингва (Archetypal)

| Класс | Строка | Описание |
|-------|--------|---------|
| `SourceSpecializer` | 859 | Специализация по источнику через Q6-сигнатуру |
| `ArchetypalInterlingua` | 946 | 64 архетипа-посредника (hub-and-spoke) ⚠️ **баг**: единый `trit_proj` |
| `BridgedInterlingua` | 1388 | Двойная прослойка: Bridge + Interlingua |
| `DynamicCurriculumController` | 1822 | Адаптивный curriculum scheduler |

> ⚠️ **`ArchetypalInterlingua` имеет баг:** все N источников используют один `trit_proj`,
> что приводит к PPL ≈ 2.93 вместо 2.75. Для новых экспериментов используйте
> `ArchetypalInterlinguaFixed` из `geometry/interlingua_fixed.py` (активируется флагом
> `interlingua_use_fixed=True` в `model.py`).

### `interlingua_fixed.py` — Исправленная Интерлингва

| Класс | Описание |
|-------|---------|
| `ArchetypalInterlinguaFixed` | Per-source `trit_proj[i]` — исправление бага. Активируется флагом `interlingua_use_fixed=True` в конфиге модели. |

```python
from yijing_transformer.models.geometry import ArchetypalInterlinguaFixed
model = ArchetypalInterlinguaFixed(d_model=128, n_sources=3, n_archetypes=64)
output, aux_loss = model([src1, src2, src3])  # list of (B, T, d)
diag = model.diagnostics()  # keys: gate, diversity_loss, spread, src{i}_pos
```

### `AbrialeBridgeMediator` — лучший результат (PPL 1.24)

```python
from yijing_transformer.models.geometry.routing import AbrialeBridgeMediator

mediator = AbrialeBridgeMediator(
    d_model=128,
    n_sources=7,       # число геометрических источников
    n_mediators=3,     # число хабов-посредников
)
# Поток: source_outputs → 3 медиатора → ядро
out = mediator(source_outputs_list, x_core)
```

---

## 5. `ffn.py` — FFN модули

**224 строки.**

| Класс | Строка | Описание |
|-------|--------|---------|
| `SwiGLU` | 16 | LLaMA-style SwiGLU: `W2(SiLU(W1·x) ⊙ W3·x)` |
| `DomainMoE` | 63 | MoE по 6 доменам Q6 с векторизованным dispatch |
| `TrigramMoE` | 139 | MoE по 8 триграммам (Q3-уровень) |
| `GeometricFFN` | 188 | FFN с геометрическим смещением (Q6-prior) |
| `MultiScaleHypercubeLayer` | 204 | Matryoshka FFN: Q2 → Q3 → Q6 |

**Векторизованный dispatch** (`_vectorized_dispatch`): вместо Python-цикла по экспертам — батчевый gather с предвычисленными весами. Ускорение ~3× по сравнению с наивной реализацией.

```python
from yijing_transformer.models.geometry import SwiGLU, DomainMoE
ffn = SwiGLU(d_model=128, hidden=512)
moe = DomainMoE(d_model=128, top_k=2)
```

---

## 6. `equivariant.py` — Эквивариантные слои

**107 строк.**

| Класс | Строка | Описание |
|-------|--------|---------|
| `BianGuaTransform` | 10 | 变卦 — покоординатное отражение: `z → z * (1 - 2p)` |
| `GraduatedBianGuaTransform` | 26 | Градуированная: мутируют только «старые» линии (Скляров) |
| `D4EquivariantLayer` | 47 | D₄-эквивариантный слой (группа симметрий триграмм) |
| `DualEmbedding` | 84 | Инь/Ян парность — двойное 6D↔3D встраивание |

```python
from yijing_transformer.models.geometry import BianGuaTransform
layer = BianGuaTransform(d_model=128)
y = layer(x)  # x + scale * proj_back(z * (1 - 2*change_prob))
```

---

## 7. `positional.py` — Позиционные кодирования

**179 строк.**

| Класс | Строка | Описание |
|-------|--------|---------|
| `RotaryEmbedding` | — | RoPE с опциями NTK / linear scaling для длинного контекста |
| `ALibiPositionalBias` | — | ALiBi: линейный bias по расстоянию позиций |
| `FourLevelPositionalEncoding` | — | 4-уровневое PE: local + sentence + section + global |
| `CubicPositionalEncoding` | — | Позиционное кодирование через Q6-куб |

### Пример: RoPE с NTK scaling

```python
from yijing_transformer.models.geometry import RotaryEmbedding
rope = RotaryEmbedding(dim=64, max_seq_len=4096, scaling='ntk', scaling_factor=8.0)
cos, sin = rope(seq_len=512)
```

---

## 8. `convergence.py` — Convergence Bridge

**824 строки.** Реализует конвергентную иерархию: Q6-глифы (снизу вверх) ↔ токены (сверху вниз).

### Концепция

```
Q6-глифы (примитивы) → GlyphComposer → ребро → грань → сигил
                                                    ↕ (cross-attention)
Токены (сырые) → TokenAbstractor → кластеры → 64 архетипа
```

| Класс | Строка | Описание |
|-------|--------|---------|
| `GlyphComposer` | 25 | Q6-вершины → сигилы через иерархию (глиф → ребро → грань → сигил) |
| `TokenAbstractor` | 216 | Токены → 64 архетипа-кластера (self-attention + soft assignment) |
| `ConvergenceLayer` | 342 | Cross-attention слияние глифов и токенов |
| `ConvergenceBridge` | 420 | Полный мост: GlyphComposer + ConvergenceLayer + TokenAbstractor |
| `MatrixGrammar` | 523 | Матричная грамматика — структурное сопоставление шаблонов |
| `ArchetypalInterlingua` | 744 | 64 архетипа-посредника (version из convergence) |

---

## 9. `nautilus.py` — Иерархия Наутилуса

**620 строк.** Каждый геометрический модуль получает свой масштаб и порядок активации — как камеры раковины Наутилуса.

### Камеры (от мелкого масштаба к крупному)

| Камера | Модуль | Масштаб | Что добавляет |
|--------|--------|---------|--------------|
| 1 | `CubeDiagonalAttention` | микро | Ориентация в 3D-кубе |
| 2 | `PrivilegedAxisAttention` | — | Главное направление |
| 3 | `DualEmbedding` | — | Инь/Ян парность |
| 4 | `D4EquivariantLayer` | — | Симметрии D₄ |
| 5 | `PalaceAttention` | — | Блочная структура (8 дворцов) |
| 6 | `HeisenbergAttention` | — | Неопределённость потока |
| 7 | `FlowerOfLifeGAT` | макро | Глобальная геометрия |

| Класс | Строка | Описание |
|-------|--------|---------|
| `NautilusChamber` | 45 | Одна камера: модуль + вес + curriculum scheduler |
| `NautilusScheduler` | 150 | Расписание активации камер (progressive curriculum) |
| `NautilusHierarchy` | 181 | Полная иерархия (7 камер) с `gate_stats()` |
| `MatryoshkaNautilus` | 357 | Matryoshka: вложенные Наутилусы Q2→Q3→Q6 |
| `PostCascadeMatryoshkaNautilus` | 524 | Post-cascade вариант с cascade loss |

```python
from yijing_transformer.models.geometry import NautilusHierarchy
hierarchy = NautilusHierarchy(d_model=128)
y = hierarchy(x)
stats = hierarchy.gate_stats()  # {'chamber_1_gate': 0.52, ...}
```

---

## 10. `abriale.py` — Абриале-слой

**574 строки.** Событийно-управляемая N-арная обработка по мотивам системы Абриаль (Пацкин, 2003).

### Три идеи Пацкина в нейросетевой форме

| Принцип | Реализация |
|---------|-----------|
| **Изотропная сеть** | Симметричный attention: A_ij = A_ji |
| **Временные связи** | Soft commit/rollback через TransactionGate |
| **Инверсия управления** | События активируют паттерны, не наоборот |

```
Input: x (B, T, d)
  → EventGenerator         (каждый токен → вектор события)
  → IsotropicMatcher        (симметричный N-арный attention)
  → RuleBank                (банк паттернов, активируемых событиями)
  → TransactionGate         (commit удачных связей, rollback неудачных)
Output: x + scale * committed_events
```

---

## 11. `kasatkin_router.py` — Маршрутизатор Касаткина

**212 строк.** Маршрутизация 6 экспертов через 3D-проекцию куба.

### Концепция

```
x (B, T, d) → proj_3d (B, T, 3) → cosine_sim с 6 осями ±X,±Y,±Z
            → softmax → routing_weights (B, T, 6) → 6 экспертов
```

Ключевое преимущество — **визуализируемость**: attention = потоки в 3D-кубе.

```python
from yijing_transformer.models.geometry import KasatkinQ6Router
router = KasatkinQ6Router(d_model=128, n_experts=6, routing_temperature=0.3)
out, routing_weights = router(x, expert_outputs)
```

**Протокол валидации:** минимум 3 000 шагов на реальных данных. Успех: `routing_confidence > 15%` при PPL ≤ baseline.

---

## 12. `q6_algebra.py` — Алгебра Z₂^6 и операции И-Цзин

**~690 строк.** Закрывает все теоретические пробелы из THEORY_VS_PRACTICE.md.

### Классы

| Класс / Функция | Описание |
|----------------|---------|
| `Q6Arithmetic` | Группа Z₂^6: `add`, `identity`, `hamming_from_dot` (Теорема 3), `hamming_matrix`, `subgroup_from_generators`, `coset`, `partition_into_cosets` |
| `YiJingOps` | V₄ операции: `heng` 恒, `cuo_gua` 错, `zong_gua` 综, `hu_gua` 互, `cuo_zong` 错综, `v4_orbit` |
| `BentFunctions` | Bent-функции f: GF(2)^6→GF(2): `truth_tables`, `wht_spectra`, `is_bent`, `support_vectors`, `prototype_codebook` |
| `YiJingV4Layer` | `nn.Module`: обучаемая взвешенная комбинация 4 V₄-операций |
| `Q6ArithmeticLayer` | `nn.Module`: routing по Хэммингу (Теорема 3); `use_bent_init=True` — инициализация из bent seeds |
| `BentPrototypeQuantizer` | Квантизатор на bent support set (28 вершин) |
| `verify_theorem3` / `verify_v4_group` / `verify_bent_functions` / `verify_all` | Верификационные функции (all_ok=True) |

### Теорема 3 — Хэмминг через скалярное произведение

```python
from yijing_transformer.models.geometry import Q6Arithmetic
# d_H(a,b) = (6 - dot(a,b)) / 2  в {-1,+1}^6
dist = Q6Arithmetic.hamming_from_dot(a, b)  # (N,) расстояния
H    = Q6Arithmetic.hamming_matrix(codebook) # (K,K) матрица
```

### Bent-функции как prototype seeds

```python
from yijing_transformer.models.geometry import BentFunctions, Q6ArithmeticLayer
# 8 прототипов из bent support (furthest-point sampling)
proto = BentFunctions.prototype_codebook(k=8)  # (8, 6) в {-1,+1}^6
# Routing layer с bent-инициализацией
layer = Q6ArithmeticLayer(d_model=128, n_prototypes=8, use_bent_init=True)
```

> **Примечание:** bent-функции на GF(2)^6 несбалансированы: |support| = 28 или 36
> (Ŵ(0) = ±8). Плоский WHT-спектр: все |Ŵ(u)| = 8 — 20/20 проверено.
