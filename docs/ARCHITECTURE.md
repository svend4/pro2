# Архитектурный справочник: YiJing Transformer (Variant 3)

## Содержание

1. [Математическая основа](#1-математическая-основа)
2. [Иерархия компонентов](#2-иерархия-компонентов)
3. [Variant3GPT — главная модель](#3-variant3gpt--главная-модель)
4. [HierarchicalMoEFFN — 4-уровневая MoE](#4-hierarchicalmoe--4-уровневая-moe)
5. [Маршрутизация и гейты](#5-маршрутизация-и-гейты)
6. [LCI — Loop Closure Index](#6-lci--loop-closure-index)
7. [Конфигурации](#7-конфигурации)

---

## 1. Математическая основа

### Q6-гиперкуб

64 гексаграммы И-Цзин изоморфны вершинам 6-мерного гиперкуба `Q6 = {-1, +1}^6`:

```
гексаграмма  ≅  вектор (b₀, b₁, b₂, b₃, b₄, b₅) ∈ {-1, +1}^6
```

Это **математическое тождество**, а не метафора. Три тысячелетия назад составители Чжоу И перебрали все 2^6 = 64 комбинации шести бинарных черт — тот же объект, что и вершины гиперкуба.

### Граф Бян-Гуа

Граф смежности гексаграмм, где два узла соединены ребром тогда и только тогда, когда гексаграммы отличаются ровно одной чертой (Хэмминг-расстояние = 1):

```
hamming(i, j) = 1  ↔  dot(hex_i, hex_j) = 4
```

Каждый узел имеет ровно 6 соседей (изменение одной из 6 линий).

### 6 доменов = 6 измерений Q6

| Бит | Домен | Символ | Кластер репозитория |
|-----|-------|--------|-------------------|
| 0 | GEO | 地 Земля | Scripts |
| 1 | HYDRO | 水 Вода | Data |
| 2 | PYRO | 火 Огонь | Training |
| 3 | AERO | 風 Ветер | Self |
| 4 | COSMO | 空 Пустота | Models |
| 5 | NOOS | 識 Сознание | Theory |

### Q6-якоря доменов (оптимизированные через SA, sep-score=21)

| Домен | Индекс | Биты | Описание |
|-------|--------|------|---------|
| GEO | 0 | `000000` | Yin×6, чистая конкретность |
| HYDRO | 8 | `001000` | 1 Yang-бит, текучесть |
| PYRO | 21 | `010101` | 3 бита, истинный экватор |
| AERO | 19 | `010011` | 3 бита, истинный экватор |
| COSMO | 62 | `111110` | 5 Yang-бит, верхняя полусфера |
| NOOS | 63 | `111111` | Yang×6, чистая абстракция |

### 3 группы (Matryoshka Q2 → Q3 → Q6)

| Группа | Домены | Спектральная позиция | Роль в фигуре-8 |
|--------|--------|---------------------|----------------|
| ABSTRACT | NOOS, COSMO | λ = +4 (верхняя полусфера) | Петля A (zoom-out) |
| DYNAMIC | AERO, PYRO | λ = 0 (экватор) | Точка X (пересечение) |
| CONCRETE | GEO, HYDRO | λ = -4 (нижняя полусфера) | Петля B (zoom-in) |

---

## 2. Иерархия компонентов

```
Variant3GPT
├── tok_emb: Embedding(vocab_size, d_model)
├── pos_emb: Embedding(block_size, d_model)
├── blocks: ModuleList[Variant3Block × n_layers]
│   ├── attn: BianGuaAttention         ← внимание с Q6-метрикой
│   ├── hex_proj: HexagramProjection   ← проекция в 64 архетипа
│   ├── ternary: TernaryGate           ← {-1, 0, +1}-активация
│   ├── analogy: CrossHexagramAnalogy  ← аналогии через Хэмминг-1
│   ├── router: NautilusYiJinRouter    ← маршрутизация по доменам
│   └── hmoe: HierarchicalMoEFFN       ← 4-уровневая MoE (опц.)
└── lm_head: Linear(d_model, vocab_size)
```

---

## 3. Variant3GPT — главная модель

**Файл:** `yijing_transformer/models/variant3.py`

### Конфигурация (`Variant3Config`)

```python
@dataclass
class Variant3Config:
    vocab_size:           int   = 256
    block_size:           int   = 64      # максимальный контекст
    d_model:              int   = 128
    n_heads:              int   = 4
    n_layers:             int   = 4
    ffn_mult:             int   = 4       # d_ff = d_model * ffn_mult
    hamming_lambda:       float = 0.15    # сила Q6-смещения в attention
    uncertainty_budget:   float = 0.25    # порог тернарного гейта
    dropout:              float = 0.1
    use_domain_routing:   bool  = False   # NautilusYiJinRouter
    use_hierarchical_moe: bool  = True    # HierarchicalMoEFFN вместо FFN
```

### Ключевые компоненты блока

#### `HexagramProjection` (строки 76-127)

Проецирует скрытое состояние в 64-мерное пространство архетипов:

```
h (B, T, d_model)
   → proj_q6(h) → tanh → soft_hex (B, T, 6)   # soft-гексаграмма
   → soft_hex ⊗ hexagrams → hex_weights (B, T, 64)  # распределение по архетипам
```

Soft-биты ∈ (-1, +1) обеспечивают дифференцируемость. Финальное представление — взвешенная сумма всех 64 архетипов.

#### `BianGuaAttention` (строки 130-201)

Многоголовое внимание с добавлением Q6-топологии:

```
scores = QK^T / √d_k  +  hamming_lambda · hamming_bias
```

`hamming_bias[i, j]` = мягкое расстояние Хэмминга между позициями i и j в Q6-пространстве. Близкие позиции (одинаковые домены) усиливаются, далёкие — подавляются.

#### `TernaryGate` (строки 203-261)

Тернарная {-1, 0, +1} активация (变爻, изменяющаяся черта):

- `+1` — Yang (активное знание, уверенность > budget)
- ` 0` — неопределённость (зона [−budget, +budget])
- `−1` — Yin (отрицательное знание)

Прямой проход: Straight-Through Estimator (STE) для обратного распространения.

#### `CrossHexagramAnalogy` (строки 262-332)

Аналогии через смежные гексаграммы (Хэмминг-расстояние = 1):

```
для каждой позиции: найти top-k ближайших гексаграмм в Q6
   → извлечь из RAG-буфера соответствующие паттерны
   → смешать с текущим состоянием через гейт
```

#### `NautilusYiJinRouter` (строки 333-418)

6-доменный маршрутизатор. Проецирует `h` в Q6 и вычисляет близость к доменным якорям. Выходы 6 доменов взвешиваются и суммируются.

---

## 4. HierarchicalMoEFFN — 4-уровневая MoE

**Файл:** `yijing_transformer/models/hierarchical_moe.py`

### Конфигурация (`HMoEConfig`)

```python
@dataclass
class HMoEConfig:
    d_model:            int   = 128
    expert_expansion:   int   = 2      # d_ff = d_model * expert_expansion
    top_k_experts:      int   = 2      # top-k внутри GroupRouter
    lb_loss_weight:     float = 0.01   # load-balancing loss weight
    bridge_weight:      float = 0.5    # вес BidirBridgeExpert
    use_multiscale:     bool  = True   # Matryoshka Q2→Q3→Q6 роутер
    use_hex_tier:       bool  = False  # 4-й уровень: 64 Q6-эксперта
    hex_tier_top_k:     int   = 4
    hex_tier_weight:    float = 0.3
    lambda_lci:         float = 0.0    # LCI→π loss
    lambda_balance:     float = 0.0    # balance loss
    lambda_dynamic:     float = 0.0    # DYNAMIC ≥ 0.20
    lambda_temp_reg:    float = 0.0    # Ising temp регуляризация
    ising_T_c:          float = 3.0    # критическая температура Q6
```

### Архитектура прохода (forward)

```
Input x (B, T, d)
    │
    ▼
[MultiScaleGlobalRouter]          # Matryoshka: Q2→Q3→Q6
    ├── group_weights (3 группы)  # ABSTRACT / DYNAMIC / CONCRETE
    └── hex_weights (64)          # для Q6ExpertBank (4-й уровень)
    │
    ├── [GroupRouter ABSTRACT] ─► [MicroExpert NOOS] + [MicroExpert COSMO]
    │                              → x_A (Петля A)
    │
    ├── [BidirBridgeExpert]        # Точка X пересечения
    │   ├── fwd: A → B
    │   └── bwd: B → A
    │   → x_crossing
    │
    ├── [GroupRouter DYNAMIC]  ─► принимает x_crossing
    │   [MicroExpert AERO] + [MicroExpert PYRO]
    │   → x_D
    │
    ├── [GroupRouter CONCRETE] ─► [MicroExpert GEO] + [MicroExpert HYDRO]
    │                              → x_B (Петля B)
    │
    └── combined = x_A + x_crossing + x_D + x_B
           │
           ▼ (use_hex_tier=True)
       [Q6ExpertBank]   # 64 эксперта, top-k=4 маршрутизация
           │
           ▼
       out_proj + residual → y (B, T, d)
```

### Компоненты

| Класс | Строка | Описание |
|-------|--------|---------|
| `Q6ExpertBank` | 120 | Векторизованный банк 64 экспертов (SiLU FFN) |
| `MicroExpert` | 178 | SwiGLU FFN для одного кластера |
| `BidirBridgeExpert` | 200 | Двунаправленный мост ABSTRACT↔CONCRETE |
| `GroupRouter` | 264 | Top-k роутер внутри группы (3 домена) |
| `GlobalRouter` | 326 | Legacy Q6 global router |
| `MultiScaleGlobalRouter` | 398 | Matryoshka: Q2→Q3→Q6, включает Hamming prior |
| `HMoEConfig` | 792 | Конфигурационный dataclass |
| `HierarchicalMoEFFN` | 822 | Главный модуль (forward + loss) |

### Curriculum обучения (6 стадий)

| Стадия | Что обучается | Что заморожено |
|--------|---------------|----------------|
| 1 | MicroExperts (по одному на кластер) | GroupRouters, GlobalRouter |
| 2 | GroupRouters + MicroExperts | GlobalRouter |
| 3 | MultiScaleGlobalRouter + всё | — |
| 4 | BidirBridgeExpert | остальное (опц.) |
| 5 | Joint Fine-tune (всё вместе) | — |
| 6 | Q6ExpertBank (только если `use_hex_tier=True`) | — |

```python
from yijing_transformer.models.hierarchical_moe import set_moe_stage
set_moe_stage(moe_module, stage=3)
```

---

## 5. Маршрутизация и гейты

**Файл:** `yijing_transformer/models/geometry/routing.py`

| Класс | Описание | Использование |
|-------|---------|------------|
| `GatedPathSelector` | Гейт geometric vs standard FFN | `gate * x_geo + (1-gate) * x_std` |
| `AdaptiveGatedPathSelector` | Multi-head гейт с температурой | Контентно-зависимый выбор |
| `AbrialeBridgeMediator` | Лучший PPL=1.24 (v59) | Hub-and-spoke через 3 медиатора |
| `ArchetypalInterlingua` | 64 архетипа-посредника | Все модули → архетипы → ядро |
| `DynamicCurriculumController` | Адаптивный curriculum scheduler | Управление силой геометрии |
| `TaskAwareRouter` | Strategy-aware routing | Задаче-зависимый выбор эксперта |
| `GeometricSourceRouter` | Source-based routing | Маршрутизация по источнику данных |

### `MultiScaleGlobalRouter` (Matryoshka)

```
Q2 (4 вершины)  → pre-routing (ABSTRACT / DYNAMIC / CONCRETE / mixed)
   ↓
Q3 (8 вершин)   → уточнение внутри группы
   ↓
Q6 (64 вершины) → hex_weights для Q6ExpertBank
```

Hamming prior: температура роутера инициализируется близко к критической температуре Q6 (T_c = 3.0 по Изингу), что препятствует преждевременному коллапсу к одной группе.

---

## 6. LCI — Loop Closure Index

LCI измеряет «замкнутость» петли фигуры-8. Оптимум = π ≈ 3.14.

### Formula A (routing-based)

**Файл:** `self_train_hmoe.py:94-145`

```
LCI_A = (1 - |w_ABSTRACT - w_CONCRETE|) × π
```

- Диапазон: [0, π]
- w_A = w_B → LCI_A = π (резонанс, баланс)
- w_A = 1 или w_B = 1 → LCI_A = 0 (схлопнутая петля)

### Formula B (embedding cosine)

**Файл:** `self_train_hmoe.py:148-162`

```
LCI_B = arccos(cosine_similarity(emb_start, emb_end)) × 4
```

- Диапазон: [0, 4π] (**отличается от Formula A**)
- При полном совпадении: 0 (петля замкнута без изменений)
- При ортогональности: 2π

> **Важно:** Formula A ∈ [0, π], Formula B ∈ [0, 4π]. Это разные шкалы, они не взаимозаменяемы. В `self_train_hmoe.py` обе метрики логируются отдельно (`avg_lci_r` и `avg_lci_emb`).

### Температурный контроль

```
если |LCI - π| < ε:  температура не меняется (резонанс)
если LCI < π:        T += 0.1  (zoom-out, расширяем поиск)
если LCI > π:        T -= 0.1  (zoom-in, уточняем)
```

---

## 7. Конфигурации

### Минимальная (CPU, исследовательская)

```python
from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import HMoEConfig

cfg = Variant3Config(
    vocab_size=256, block_size=64, d_model=128,
    n_heads=4, n_layers=4,
    use_hierarchical_moe=True,
)
hmoe_cfg = HMoEConfig(d_model=128, use_multiscale=True, use_hex_tier=False)
model = Variant3GPT(cfg, hmoe_cfg=hmoe_cfg)
```

### Полная (с Q6ExpertBank, 4-м уровнем)

```python
hmoe_cfg = HMoEConfig(
    d_model=256,
    use_multiscale=True,
    use_hex_tier=True,    # включает 64-экспертный банк
    hex_tier_top_k=4,
    hex_tier_weight=0.3,
    lambda_lci=0.1,       # LCI→π loss
    lambda_balance=0.05,
)
```

### Загрузка чекпоинта

```python
import torch
ckpt = torch.load("hmoe_self_trained_v4.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
```

`strict=False` необходим при смене конфигурации (например, включение `use_hex_tier`).
