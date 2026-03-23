# Архитектура YiJing-Transformer

Технический обзор внутреннего устройства для разработчиков.

---

## Общая схема

```
Вход (tokens)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Embedding  +  Positional Encoding (RoPE / ALiBi)    │
└──────────────────────────┬───────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │     N × Transformer     │
              │         Block           │
              │                         │
              │  ┌───────────────────┐  │
              │  │   Attention       │  │
              │  │  (+ BianGua bias) │  │
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │   FFN / MoE       │  │
              │  │  (SwiGLU / Trigram│  │
              │  │   MoE / Domain)   │  │
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │  Quantizer        │  │
              │  │  (Q6 projection)  │  │
              │  └────────┬──────────┘  │
              └───────────┼─────────────┘
                          │
              ┌───────────▼──────────────┐
              │  LM Head  →  logits      │
              └──────────────────────────┘
```

---

## Три ветви архитектуры

### 1. YiJingGPT (`models/model.py`)

Базовый трансформер с геометрической регуляризацией.

```
Token → Embedding → [Attention + FFN + Quantizer]×N → LM Head
                                    ↑
                           hex_strength × Q6_loss
```

- Квантизатор (по умолчанию `FactoredYiJingQuantizer`) проецирует hidden states в Q6
- Регуляризационный лосс `hex_strength × commitment_loss` направляет представления к вершинам гиперкуба
- Гейт `GatedPathSelector` решает, сколько геометрического сигнала примешивать

**Ключевые параметры:**
- `hex_strength` — вес геометрического лосса (по умолчанию 0.01)
- `quantizer_type` — тип квантизатора ('factored6', 'hierarchical', 'deformable', ...)
- `use_bian_gua` — добавлять ли BianGua attention bias

### 2. Variant3GPT (`models/variant3.py`)

Архетипо-центричная архитектура. Каждый токен — это не просто вектор, а позиция в пространстве 64 архетипов.

```
Token → Embedding
            │
    ┌───────▼──────────┐
    │ HexagramProjection│ — проекция в Q6 (64 архетипа)
    └───────┬──────────┘
            │
    ┌───────▼──────────┐
    │ BianGuaAttention  │ — внимание с топологическим bias из графа гексаграмм
    └───────┬──────────┘
            │
    ┌───────▼──────────┐
    │   TernaryGate     │ — {−1, 0, +1}: инь, переходное, ян
    └───────┬──────────┘
            │
    ┌───────▼──────────────────┐
    │ ArchetypalInterlingua    │ — 64 эксперта (по одному на гексаграмму)
    │ или BridgedInterlingua   │ — мосты между доменами
    └───────┬──────────────────┘
            │
    ┌───────▼──────────┐
    │ CrossHexAnalogy   │ — аналогии между гексаграммами
    └───────┬──────────┘
            │
            ▼
        LM Head
```

**6 доменов** (каждый = одно измерение Q6):
```python
DOMAINS = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]
# GEO   = бит 0  — структура
# HYDRO = бит 1  — анализ
# PYRO  = бит 2  — трансформация
# AERO  = бит 3  — движение
# COSMO = бит 4  — связи
# NOOS  = бит 5  — смысл
```

### 3. HierarchicalMoE (`models/hierarchical_moe.py`)

4-уровневая иерархия «Матрёшка»: от грубых кластеров к точным гексаграммам.

```
Token → Embedding
            │
    ┌───────▼───────────┐
    │ Level 0: Q2 (4)   │ — 4 макро-кластера
    │ MultiScaleGlobal   │
    └───────┬───────────┘
            │
    ┌───────▼───────────┐
    │ Level 1: Q3 (8)   │ — 8 триграмм
    │ GroupRouter         │
    └───────┬───────────┘
            │
    ┌───────▼───────────┐
    │ Level 2: Q6 (64)  │ — 64 гексаграммы
    │ ExpertChoice       │
    └───────┬───────────┘
            │
    ┌───────▼───────────┐
    │ Level 3: Bridge   │ — мосты между доменами
    │ BridgeExperts      │
    └───────┬───────────┘
            │
            ▼
        LM Head
```

**Staged Training Protocol:**
1. **Freeze & Train Q2** — только макро-маршрутизация (50 шагов)
2. **Unfreeze Q3** — триграммная маршрутизация (100 шагов)
3. **Full Q6** — полная иерархия (до конца)

---

## Геометрические модули

### Квантизаторы (`geometry/quantizers.py`)

Все квантизаторы наследуют от `nn.Module` и реализуют:

```python
def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, dict]:
    """
    Args:
        x: (batch, seq, d_model) — входные представления
    Returns:
        quantized: (batch, seq, d_model) — квантизованные представления
        indices: (batch, seq) — индексы ближайших вершин
        info: dict — commitment_loss, usage stats, etc.
    """
```

**Факторизация** — главная оптимизация:

```
Наивно: softmax по 64 → O(64 × d_model)
Факторизовано: 2 × softmax по 8 → O(16 × d_model)  ← 3.75× быстрее

Гексаграмма = верхняя триграмма ⊗ нижняя триграмма
Индекс: i = upper × 8 + lower
```

**STE (Straight-Through Estimator):**
```python
quantized = x + (quantized_detached - x).detach()  # gradient flows through x
```

### Паттерны внимания (`geometry/attention.py`)

15+ паттернов, основанные на структуре гиперкуба:

| Паттерн | Структура | Bias |
|---------|-----------|------|
| `PalaceAttention` | 8 палацев по 8 гексаграмм | block-diagonal |
| `TriangularAttentionBias` | 6 позиций в каждой линии | triangular |
| `QuadrantAttention` | 4 квадранта Q6 | quadrant mask |
| `RecursiveCubeAttention` | Рекурсивное разбиение куба | hierarchical |
| `HeisenbergAttention` | Некоммутативная алгебра | asymmetric |
| `HexagramAttentionPattern` | Хэмминговое расстояние | distance-based |

### Маршрутизация (`geometry/routing.py`)

```
┌─────────────────────────────────────────────────────┐
│                    routing.py                        │
│                                                     │
│  Гейты:                                            │
│  ├── GatedPathSelector         (binary gate)        │
│  ├── AdaptiveGatedPathSelector (content-dependent)  │
│  └── TaskAwareRouter           (task conditioning)  │
│                                                     │
│  Мосты:                                            │
│  ├── PairwiseBridge            (попарные связи)     │
│  ├── LightweightBridge         (лёгкий мост)       │
│  ├── BridgeOfModules           (PPL 1.35)          │
│  ├── AbrialeBridgeMediator     (PPL 1.24 ★)       │
│  └── AdaptiveBridgeOfModules   (адаптивный)        │
│                                                     │
│  Интерлингва:                                      │
│  ├── ArchetypalInterlingua     (64 эксперта)       │
│  └── BridgedInterlingua        (мосты + эксперты)  │
│                                                     │
│  Curriculum:                                        │
│  ├── GeometryCurriculumScheduler (step-based)       │
│  ├── DynamicCurriculumController (loss-adaptive)    │
│  └── SequentialSourceCurriculum  (source ordering)  │
│                                                     │
│  Специализация:                                    │
│  └── SourceSpecializer         (per-domain weights) │
└─────────────────────────────────────────────────────┘
```

### Nautilus (`geometry/nautilus.py`)

7-камерная иерархия, вдохновлённая раковиной наутилуса:

```
Chamber 0 (Core)  →  Chamber 1  →  ...  →  Chamber 6 (Shell)
    │                    │                       │
  d_model/7          d_model/7              d_model/7
    │                    │                       │
  concat → MatryoshkaNautilus → full d_model output
```

- `NautilusChamber` — один уровень иерархии
- `NautilusHierarchy` — полная 7-камерная система
- `MatryoshkaNautilus` — вложенная версия (каждая камера содержит предыдущие)
- `NautilusScheduler` — постепенное включение камер при тренировке

### Convergence Bridge (`geometry/convergence.py`)

Мост между двумя представлениями — глифовым (дискретное дерево символов) и токенным (непрерывные вектора):

```
Tokens  ←──  ConvergenceBridge  ──→  Glyphs
              │                │
        TokenAbstractor   GlyphComposer
              │                │
        (token → glyph)   (glyph → token)
```

- `MatrixGrammar` — грамматика для операций над матрицами гексаграмм

---

## Тренировочный пайплайн

### Главный цикл (`training/train.py`)

```python
train(args) → model
```

1. **Инициализация:** модель, оптимизатор, scheduler из `TrainingBridge`
2. **Warmup:** линейный рост LR от 0 до `lr` за `warmup_steps` шагов
3. **Cosine decay:** LR → `lr * 0.1` к концу тренировки
4. **Gradient accumulation:** `grad_accum_steps` микро-батчей перед `optimizer.step()`
5. **Mixed precision:** `torch.amp.GradScaler` при `use_amp=True`
6. **Validation:** каждые `val_every` шагов, `estimate_val_loss()`
7. **Checkpointing:** каждые `save_every` шагов

### TrainingBridge (`training/bridge.py`)

Интегрирует утилиты из 52 версий `utils_v*.py`:

```python
bridge = TrainingBridge(model, cfg)
optimizer = bridge.create_optimizer()
scheduler = bridge.create_scheduler(optimizer)
# После каждого шага:
bridge.step(loss, step)  # regularization, monitoring, adaptation
```

### Самообучение (`self_train.py`)

3 стадии:

```
Stage 1: Self-Topology (Q6 structure)
    │  Модель учится на собственных Q6-проекциях
    │  Цель: выучить геометрию гиперкуба
    ▼
Stage 2: RAG-Buffer (curated corpus)
    │  Обучение на качественном корпусе с PseudoRAG
    │  Цель: связать геометрию с семантикой
    ▼
Stage 3: General Data (wild data)
    │  Обучение на общих данных с domain filtering
    │  Цель: генерализация
    ▼
Output: fine-tuned checkpoint
```

---

## Device / Precision Safety

Все модули поддерживают:
- **Multi-GPU:** буферы через `register_buffer()`, маски через `.new_tensor()`
- **fp16 / bf16:** `clamp(min=eps)` перед `log()`, bool-маски вместо float, dtype-safe операции
- **CPU fallback:** `GradScaler` с device-aware инициализацией

Тесты: `yijing_transformer/tests/test_multigpu_precision.py`

---

## Пресеты конфигурации

```python
from yijing_transformer.config import YiJingConfig

# ~2M параметров (для отладки)
cfg = YiJingConfig.tiny()

# ~15M параметров
cfg = YiJingConfig.small()

# ~85M параметров
cfg = YiJingConfig.medium(vocab_size=32000)

# ~300M параметров
cfg = YiJingConfig.large(vocab_size=32000)
```

---

## Потоки данных

### Тренировка

```
Corpus  →  TextDataset / StreamingDataset
                │
         ShuffledBatchIterator
                │
         (x, y) tensors  →  model(x)  →  loss  →  backward  →  step
                                              │
                                     hex_loss (geometric regularization)
```

### Inference

```
Prompt  →  CharTokenizer / SentencePiece
                │
         AdvancedGenerator
                │
         Strategy: greedy / nucleus / beam / speculative / dynamic_temp
                │
         Decoded text
```

Стратегии генерации:

| Стратегия | Скорость | Качество | Когда использовать |
|-----------|----------|----------|-------------------|
| `greedy` | Быстро | Детерминистично | Тесты, отладка |
| `nucleus` | Средне | Разнообразно | Генерация текста |
| `beam` | Медленно | Оптимально | Перевод, точные задачи |
| `speculative` | Быстро* | Как nucleus | Большие модели (*с draft model) |
| `dynamic_temp` | Средне | Адаптивно | Когда нужен баланс |
