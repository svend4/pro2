# Архитектура YiJing-Transformer

## Оглавление

1. [Геометрическая основа](#1-геометрическая-основа)
2. [YiJingGPT](#2-yijinggpt)
3. [NautilusMoME](#3-nautilusmome)
4. [Квантизаторы](#4-квантизаторы)
5. [Обучение и Pipeline](#5-обучение-и-pipeline)
6. [Внутренние слои](#6-внутренние-слои)

---

## 1. Геометрическая основа

### Триграммы — вершины 3-мерного куба

8 триграмм (八卦) — это все 8 вершин куба {-1,+1}³:

| # | Триграмма | Вектор | Название |
|---|-----------|--------|---------|
| 0 | ☰ | (+1, +1, +1) | Цянь (Небо) |
| 1 | ☱ | (+1, +1, -1) | Дуй (Озеро) |
| 2 | ☲ | (+1, -1, +1) | Ли (Огонь) |
| 3 | ☳ | (+1, -1, -1) | Чжэнь (Гром) |
| 4 | ☴ | (-1, +1, +1) | Сюнь (Ветер) |
| 5 | ☵ | (-1, +1, -1) | Кань (Вода) |
| 6 | ☶ | (-1, -1, +1) | Гэнь (Гора) |
| 7 | ☷ | (-1, -1, -1) | Кунь (Земля) |

### Гексаграммы — вершины Q6-гиперкуба

64 гексаграммы = все 2⁶ = 64 вершины гиперкуба Q6 = {-1,+1}⁶.

Тензорная структура:
```
Гексаграмма = (верхняя триграмма) ⊗ (нижняя триграмма)
```

Это означает: квантизация к 64 гексаграммам = **два независимых softmax по 8 триграммам** (в 8x быстрее наивного подхода).

### Метрика расстояния

Для Q6 используется расстояние Хэмминга:
```
d_H(h₁, h₂) = #{i : h₁ᵢ ≠ h₂ᵢ}  — число различающихся битов
```

Вместо евклидова расстояния — логично, т.к. все точки равноудалены от центра (||h|| = √6 для всех h ∈ Q6).

---

## 2. YiJingGPT

### Обзор архитектуры

```
Токены → Embedding → [TransformerBlock × n_layers] → LM Head → Логиты
                            ↓
                    Q6-Attention (геометрические паттерны)
                            ↓
                    YiJing-FFN (квантизация в кодбуке)
```

### Параметры по умолчанию

| Параметр | Значение |
|----------|---------|
| `vocab_size` | 4096 (BPE) |
| `d_model` | 128 |
| `n_layers` | 4 |
| `n_heads` | 4 |
| `block_size` | 256 |
| `dropout` | 0.1 |

### Кодбук гексаграмм

```python
# 64 фиксированных вектора-якоря в пространстве d_model
# Инициализированы из Q6-вершин, проецированных в d_model-мерное пространство
codebook = nn.Parameter(hexagram_embeddings)  # shape: (64, d_model)
# НЕ обучается — фиксированный индуктивный bias
```

---

## 3. NautilusMoME

### Полная схема

```
Input tokens
    │
    ▼
[core_first]        ← общие начальные слои
    │
    ▼
[Router]            ← маршрутизатор (Q6-based или learned gate)
    │
    ├─→ [Expert CODE]   ← Python, JS, React
    ├─→ [Expert RECON]  ← Русский язык
    ├─→ [Expert SYSTEM] ← SQL, K8s, Docker
    ├─→ [Expert MATH]   ← Формулы
    ├─→ [Expert HUMAN]  ← Общий текст
    └─→ [Expert INFO]   ← YAML, конфигурации
    │
    ▼
[NautilusBridge]    ← иерархическое слияние (residual gate)
    │
    ▼
[CrossDomainAnalogy] ← 15 пар аналогий (ProverbCondenser + AnalogyPair)
    │
    ▼
[ArchetypeLayer]    ← 64-архетипный слой, тернарная квантизация {-1,0,+1}
    │
    ▼
[core_second]       ← общие завершающие слои
    │
    ▼
Output logits
```

### Маршрутизатор

Router использует мягкое распределение (soft routing, не top-k hard routing):
- Вычисляет сходство запроса со всеми 6 экспертами
- Применяет температурный softmax
- Все эксперты активны, но с разными весами (нет коллапса в одного эксперта)

Наблюдаемые routing-веса на тестовых доменах:

| Домен  | CODE  | RECON | SYSTEM | MATH  | INFO  |
|--------|-------|-------|--------|-------|-------|
| CODE   | 0.261–0.296 | — | — | — | — |
| Russian | — | 0.559–0.655 | — | — | — |
| SYSTEM | — | — | 0.327–0.361 | — | — |
| INFO   | — | — | — | — | 0.310–0.339 |

### NautilusBridge

Иерархическое слияние с residual gate:
```python
bridge_out = gate * expert_combined + (1 - gate) * core_hidden
```
где `gate` — обучаемый скалярный параметр, специфичный для позиции.

### CrossDomainAnalogy

15 попарных комбинаций из 6 экспертов (C(6,2) = 15).

Каждая пара включает:
- **ProverbCondenser** — сжатый паттерн аналогии из двух доменов
- **AnalogyPair** — механизм переноса знаний между доменами

Итого: 15 пар × (ProverbCondenser + AnalogyPair) = 1,083,551 параметров

### ArchetypeLayer

64 архетипных вектора (по числу гексаграмм) с тернарной квантизацией весов:
```
w ∈ {-1, 0, +1}  (вместо float32)
```
Позволяет хранить «полюса» концептуального пространства при минимальном числе параметров (~120K).

---

## 4. Квантизаторы

### YiJingQuantizer (базовый)

```python
class YiJingQuantizer(nn.Module):
    """Квантизация к ближайшей вершине Q6."""

    def forward(self, x):
        # x: (batch, seq, d_model)
        # Проецируем в 6-мерное пространство
        projected = self.proj(x)  # (batch, seq, 6)
        # Квантизуем sign()-функцией
        quantized = torch.sign(projected)  # вершина Q6
        # Straight-through estimator для обратного прохода
        return projected + (quantized - projected).detach()
```

### FactoredQuantizer

Эксплуатирует тензорную факторизацию Q6 = Q3 ⊗ Q3:
```python
# Вместо softmax(64): два softmax(8)
top_trigram = softmax(proj_upper @ trigrams.T)   # (batch, 8)
bot_trigram = softmax(proj_lower @ trigrams.T)   # (batch, 8)
# Гексаграмма = тензорное произведение
hexagram = kron(top_trigram, bot_trigram)         # (batch, 64)
```

### HierarchicalQuantizer

2-уровневая квантизация:
1. Грубая: к ближайшей триграмме
2. Тонкая: к ближайшей гексаграмме в выбранном квадранте

### DeformableQuantizer

Вершины Q6 обучаемо деформируются:
```python
# Фиксированные якоря + обучаемые смещения
vertices = base_hexagrams + self.deformations  # shape: (64, d_model)
```

### E8Quantizer

Квантизация к решётке E8 (240 корней) — сравнительный бейзлайн:
```
E8 = {x ∈ Z⁸ : Σxᵢ ≡ 0 (mod 2)}  ∪  {x ∈ (Z+½)⁸ : Σxᵢ ≡ 0 (mod 2)}
```

---

## 5. Обучение и Pipeline

### 3-фазный Pipeline

```python
# pipeline.py
phases = [
    Phase1_NautilusMoE(steps=5000),        # Мультидоменное обучение
    Phase2_TurbineLCI(steps=2000),          # LCI-аттрактор
    Phase3_Benchmark(domains=ALL_DOMAINS),   # Оценка
]
```

### Curriculum-обучение (train_hmoe_curriculum.py)

5 фаз curriculum:

| Фаза | Данные | Шаги | Цель |
|------|--------|------|------|
| 1–3  | Базовые + SYNTH | 3000 | PPL ~18–20 |
| 4    | Analogy training | 1000 | PPL 18.32 → 17.42 |
| 5    | Archetype + LCI  | 2000 | Стабилизация |

### LCI-аттрактор (Turbine)

**Loss Correlation Index** — метрика, описывающая согласованность потерь экспертов:

```
LCI = corr(L_expert_i, L_expert_j)  усреднённое по всем парам
```

Экспериментально: LCI сходится к π ≈ 3.14 как к естественному аттрактору. Turbine-потери оптимизируют обучение в направлении этого аттрактора.

### Self-training (self_train_hmoe.py)

Итеративное самообучение:
1. Модель генерирует псевдо-метки на unlabeled данных
2. Дообучается на собственных выходах
3. Повторяется до сходимости

Версии: v1–v6 (`hmoe_self_trained_v{n}.pt`)

---

## 6. Внутренние слои

### Tokenizer (BPE, vocab=4096)

```python
from yijing_transformer.tokenizer import BPETokenizer

tokenizer = BPETokenizer.from_pretrained("yijing_transformer/tokenizer/")
tokens = tokenizer.encode("def hello_world():")
```

Размер словаря 4096 выбран как 2^12 — следующая степень двойки после 2^6=64 (число гексаграмм). Это обеспечивает возможность тензорного разложения.

### Attention с геометрическими паттернами

Q6-attention заменяет стандартный dot-product attention на геометрически-осведомлённую версию:

```python
# Стандартный attention: softmax(QK^T / sqrt(d_k)) V
# Q6-attention: применяет маску связности гиперкуба
adjacency_mask = hexagram_adjacency_matrix()  # (64, 64) — соседи в Q6
attn_weights = attn_weights * adjacency_mask   # только соседние вершины
```

### Данные (svend4_corpus)

Многодоменный корпус обучения:

| Домен | Примеры |
|-------|---------|
| CODE  | Python, JS, SQL, shell-скрипты |
| RECON | Русскоязычные документы |
| SYSTEM | Docker, K8s, конфиги |
| MATH  | Математические формулы, LaTeX |
| INFO  | README, YAML, JSON, документация |

---

## Версионирование моделей

### Трек чекпойнтов

```
checkpoint_hmoe.pt                   # Базовая HierarchicalMoE
hmoe_self_trained_v{1-6}.pt         # Self-training трек
hmoe_4agent_{variant}.pt             # 4-экспертные варианты
hmoe_nautilus_{variant}.pt           # Nautilus bridge варианты
bench_v{1-14}.pt                     # Бенчмарк треки
pipeline_{phase}.pt                  # Pipeline треки
```

### Паспорта моделей

Каждый чекпойнт сопровождается JSON-паспортом в `passports/`:
```json
{
  "version": "nautilus_v1",
  "params": 3026000,
  "ppl": {"CODE": 12.4, "RECON": 15.2, "SYSTEM": 18.7},
  "lci": 2.87,
  "training_steps": 8000,
  "architecture": "NautilusMoME"
}
```
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

> Полный справочник модуля `geometry/` — см. [`GEOMETRY.md`](GEOMETRY.md)

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
