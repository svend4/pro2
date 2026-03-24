# Справочник моделей

Все модели в `yijing_transformer/models/`. Краткий гайд: когда какую использовать.

---

## Содержание

1. [Обзор](#1-обзор)
2. [VanillaGPT — baseline](#2-vanillagpt--baseline)
3. [LeanYiJingGPT — минимальный Q6-baseline](#3-leanyijinggpt--минимальный-q6-baseline)
4. [YiJingGPT — полная модель](#4-yijinggpt--полная-модель)
5. [Variant3GPT — production-модель](#5-variant3gpt--production-модель)
6. [HierarchicalE2 — 5-уровневая иерархия](#6-hierarchicale2--5-уровневая-иерархия)
7. [NautilusYiJing — MoE + Q6 routing](#7-nautilusyijing--moe--q6-routing)
8. [Вспомогательные модули](#8-вспомогательные-модули)
9. [Variant3Extensions — research ideas](#9-variant3extensions--research-ideas)
10. [Когда какую модель использовать](#10-когда-какую-модель-использовать)

---

## 1. Обзор

| Модель | Файл | Параметры (d=128) | Статус | PPL (800 шаг) |
|--------|------|-------------------|--------|--------------|
| `VanillaGPT` | baseline.py | ~0.8M | Stable | 2.94 |
| `LeanYiJingGPT` | lean_model.py | ~0.9M | Stable | цель < 1.07 |
| `YiJingGPT` | model.py | 0.9–2M | Stable | — |
| **`Variant3GPT`** | variant3.py | **1.4M** | **Production** | — |
| `Variant3GPT + HMoE` | variant3.py | 1.9M | Production | — |
| `HierarchicalE2` | hierarchical_e2.py | 2–3M | Research | — |
| `NautilusYiJing` | nautilus_yijing.py | ~2M | Research | — |

---

## 2. `VanillaGPT` — baseline

**Файл:** `yijing_transformer/models/baseline.py`

Чистый transformer без геометрических компонентов. Используется как честное A/B-сравнение: если архитектурное улучшение не превосходит VanillaGPT, оно неэффективно.

**Что умеет:**
- RoPE + SwiGLU (как у LLaMA, честное сравнение)
- GQA (Grouped Query Attention) через `n_kv_heads`
- Те же гиперпараметры, что у YiJingGPT, но без Q6

```python
from yijing_transformer.models.baseline import VanillaGPT
from yijing_transformer.config.config import YiJingConfig

cfg = YiJingConfig.small(vocab_size=256)
model = VanillaGPT(cfg)
```

---

## 3. `LeanYiJingGPT` — минимальный Q6-baseline

**Файл:** `yijing_transformer/models/lean_model.py`

Использует только два подтверждённых геометрических компонента: `HeisenbergAttention` + `FlowerOfLifeGAT`. Устанавливает официальный геометрический baseline (PPL < 1.07).

**Принцип Шага 3:** каждый новый компонент должен превышать этот baseline. Всё, что ниже — в «карантин».

```python
from yijing_transformer.models.lean_model import LeanYiJingBlock, LeanYiJingGPT

model = LeanYiJingGPT(vocab_size=256, d_model=128, n_heads=4, n_layers=4)
```

**Когда использовать:** при добавлении нового геометрического источника — сначала сравниваем с LeanYiJingGPT.

---

## 4. `YiJingGPT` — полная модель

**Файл:** `yijing_transformer/models/model.py`

Основная производственная модель версии v8. Включает все исследованные компоненты, управляемые через `YiJingConfig`.

### `YiJingConfig` — ключевые параметры

```python
from yijing_transformer.config.config import YiJingConfig

@dataclass
class YiJingConfig:
    d_model:          int   = 512
    n_layers:         int   = 12
    n_heads:          int   = 8
    block_size:       int   = 512
    vocab_size:       int   = 2048

    # Q6-геометрия
    hex_strength:     float = 0.01   # сила геометрического вклада
    quantizer_type:   str   = 'factored6'  # тип квантизатора
    use_bian_gua:     bool  = True   # BianGuaTransform
    adaptive_temp:    bool  = True   # обучаемая температура квантизации (TernaryQuantizer)

    # Архитектура
    use_rope:         bool  = True   # RoPE
    use_swiglu:       bool  = True   # SwiGLU FFN
    n_kv_heads:       int   = None   # GQA (None = MHA)
    use_diff_attn:    bool  = False  # Differential Attention

    # Расширения
    use_convergence_bridge: bool = False  # ConvergenceBridge
    use_abriale:            bool = False  # Абриале-слой
    use_lora:               bool = False  # LoRA адаптеры
    prefix_len:             int  = 0      # Prefix Tuning
```

### Пресеты

```python
YiJingConfig.small(vocab_size=256)   # d=128, n=4, h=4, ctx=256
YiJingConfig.medium(vocab_size=256)  # d=256, n=6, h=8, ctx=512
YiJingConfig.large(vocab_size=256)   # d=512, n=12, h=8, ctx=1024, GQA
YiJingConfig.xl(vocab_size=256)      # d=1024, n=16, h=16, ctx=2048, GQA
```

### Варианты `YiJingGPT`

| Класс | Строка | Описание |
|-------|--------|---------|
| `YiJingGPT` | 876 | Полная модель (основной вариант) |
| `PureGeometricGPT` | 1593 | Только геометрические слои, без стандартного attention |
| `HybridGatedGPT` | 1677 | Гейт между geometric и standard путями |
| `AdaptiveHybridGPT` | 1909 | Адаптивный гейт с curriculum |

```python
from yijing_transformer.models.model import YiJingGPT
from yijing_transformer.config.config import YiJingConfig

cfg = YiJingConfig.small(vocab_size=256)
model = YiJingGPT(cfg)

# Генерация
logits, loss, aux = model(idx, targets)
tokens = model.generate(idx, max_new_tokens=50, temperature=1.0, top_k=40)
```

### Температурное расписание квантизаторов (`step_temp`)

Квантизаторы с `adaptive_temp=True` (TernaryQuantizer и другие) поддерживают
cosine annealing температуры: **нужно явно вызывать `.step_temp()` каждый шаг**.

```python
from yijing_transformer.models.geometry import TernaryQuantizer

q = TernaryQuantizer(
    d_model=128,
    warmup_steps=5000,   # количество шагов до достижения end_temp
    start_temp=1.0,      # начальная температура (мягкое распределение)
    end_temp=0.05,       # конечная температура (жёсткая квантизация)
)

for step, batch in enumerate(dataloader):
    loss = model(batch)
    optimizer.step()
    q.step_temp()        # ← обязательно после каждого шага

# Текущая температура:
print(q.temp)
```

Без вызова `step_temp()` температура остаётся фиксированной на `start_temp`.
Все квантизаторы модели обновляются через `_step_all_quantizers()` в `self_train_hmoe.py`.

---

## 5. `Variant3GPT` — production-модель

**Файл:** `yijing_transformer/models/variant3.py`

Главная модель проекта. Основана на архетипах как сквозной оси: каждый токен проецируется в Q6, каждый шаг — переход по графу Бян-Гуа.

> Подробно: [ARCHITECTURE.md — Variant3GPT](ARCHITECTURE.md#3-variant3gpt--главная-модель)

```python
from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import HMoEConfig

cfg = Variant3Config(
    vocab_size=256, block_size=64, d_model=128,
    n_heads=4, n_layers=4, use_hierarchical_moe=True,
)
hmoe_cfg = HMoEConfig(d_model=128, use_multiscale=True)
model = Variant3GPT(cfg, hmoe_cfg=hmoe_cfg)
```

**Уникальные компоненты:**

| Компонент | Описание |
|-----------|---------|
| `HexagramProjection` | Soft-проекция токена в 64 архетипа Q6 |
| `BianGuaAttention` | Attention с Хэмминг-смещением (ближние домены сильнее) |
| `TernaryGate {-1,0,+1}` | Трёхзначная активация (变爻) |
| `CrossHexagramAnalogy` | Аналогии через Хэмминг-1 переходы |
| `NautilusYiJinRouter` | 6-доменная маршрутизация |
| `HierarchicalMoEFFN` | 4-уровневая MoE (Q2→Q3→Q6), опционально |

---

## 6. `HierarchicalE2` — 5-уровневая иерархия

**Файл:** `yijing_transformer/models/hierarchical_e2.py`

Пять уровней абстракции, обучаемых последовательно снизу вверх. Соответствует α-уровням методологии info1.

```
α = -4  │  GlyphLevel    — Q6-глифы → ранняя кластеризация
α = -2  │  CoreLevel     — Variant3Block (Q6-внимание + тернарные ворота)
α =  0  │  MethodLevel   — ArchetypalInterlingua (64 архетипа)
α = +2  │  TheoryLevel   — NautilusHierarchy (7 камер)
α = +4  │  PhiloLevel    — ConvergenceBridge + MatrixGrammar
```

### `E2Config`

```python
from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config

cfg = E2Config(
    vocab_size=256, d_model=128, block_size=32,
    n_core=4, n_heads=4,
    n_archetypes=64, il_use_ternary=True,
)
model = HierarchicalE2(cfg)
```

### Curriculum обучения (5 фаз)

```python
# Фаза 1: только glyph_level
model.freeze_all_except('glyph_level')
# Фаза 2: добавить core_level (можно загрузить из checkpoint_bidir_v2.pt)
model.freeze_all_except('core_level')
# ... и т.д. до philo_level
```

**Inference:** `e2_inference.py` (см. [INFERENCE.md](INFERENCE.md))

---

## 7. `NautilusYiJing` — MoE + Q6 routing

**Файл:** `yijing_transformer/models/nautilus_yijing.py`

Интеграция NautilusMoME (Mixture-of-MicroExperts) с Q6-геометрией. Три варианта интеграции:

| Вариант | Компонент | Замена | Описание |
|---------|-----------|--------|---------|
| 1 | `Q6GeometricRouter` | `ExpertRouter` | Детерминированный Q6-роутинг через Хэмминг-расстояние |
| 2 | `YiJingCoreBlock` | `TransformerBlock` | Q6-attention + RoPE + BianGuaTransform + SwiGLU |
| 3 | `YiJingMicroExpert` | `MicroExpert` | Каждый эксперт — мини-YiJing с Q6-якорём |

**Уникальная особенность:** у каждого эксперта есть геометрический якорь (гексаграмма). Эксперт MATH «думает» иначе чем HUMAN на уровне Q6.

---

## 8. Вспомогательные модули

### LoRA (`lora.py`)

Low-Rank Adaptation: дообучение без изменения основных весов.

```python
from yijing_transformer.models.lora import apply_lora, freeze_non_lora, merge_lora

apply_lora(model, cfg, rank=8, alpha=16.0)  # добавляет адаптеры
freeze_non_lora(model)                       # замораживает всё кроме LoRA
# ... обучение ...
merge_lora(model)                            # вливает LoRA в основные веса

# Сохранение только LoRA-весов
from yijing_transformer.models.lora import save_lora_weights, load_lora_weights
save_lora_weights(model, "lora_adapter.pt")
```

| Функция | Описание |
|---------|---------|
| `apply_lora(model, cfg)` | Добавить LoRA адаптеры к Linear слоям |
| `freeze_non_lora(model)` | Заморозить всё кроме LoRA |
| `merge_lora(model)` | Влить LoRA в веса (необратимо) |
| `count_lora_parameters(model)` | Подсчёт trainable параметров |

### Speculative Decoding (`speculative.py`)

Ускорение генерации в 2–3× через draft-модель.

```python
from yijing_transformer.models.speculative import build_draft_model, speculative_generate

draft = build_draft_model(cfg_large)       # автоматически cfg.d_model // 2
text = speculative_generate(
    target=model_large,
    draft=draft,
    idx=prompt_tokens,
    max_new_tokens=200,
    K=4,               # число токенов draft за итерацию
)

# Измерение качества
from yijing_transformer.models.speculative import measure_acceptance_rate
rate = measure_acceptance_rate(target, draft, test_prompts)
```

### Differential Attention (`diff_attn.py`)

Подавление шума через вычитание двух softmax карт.

```python
# W ∈ YiJingConfig: use_diff_attn=True
# Или напрямую:
from yijing_transformer.models.diff_attn import DifferentialAttention
attn = DifferentialAttention(d_model=128, n_heads=4)
```

### Prefix Tuning + Logit Lens + MTP (`prefix_tuning.py`)

```python
from yijing_transformer.models.prefix_tuning import (
    PrefixTuning, LogitLens, MultiTokenPredictionHead
)

# Prefix Tuning: обучать только prefix, основная модель заморожена
prefix = PrefixTuning(n_layers=4, prefix_len=16, n_heads=4, head_dim=32)

# Logit Lens: что думает модель на промежуточных слоях
lens = LogitLens(model)
probs_by_layer = lens.probe(hidden_states)  # список (n_layers, vocab_size)

# MTP: предсказание N следующих токенов
mtp = MultiTokenPredictionHead(d_model=128, vocab_size=256, n_future=4)
```

### Expert Choice MoE (`expert_choice.py`)

Альтернатива top-k routing: эксперты сами выбирают токены.

```python
from yijing_transformer.models.expert_choice import ExpertChoiceRouter

# Каждый из 8 экспертов выбирает top-C токенов
# Идеальный load balancing без aux loss
router = ExpertChoiceRouter(d_model=128, n_experts=8, capacity_factor=1.25)
```

### Экспорт (`export.py`)

```python
from yijing_transformer.models.export import export_onnx, export_torchscript, create_model_card

export_onnx(model, "model.onnx", seq_len=128, opset_version=14)
export_torchscript(model, "model.pt", seq_len=128)
card = create_model_card(model)  # dict с метриками и конфигурацией
```

---

## 9. `Variant3Extensions` — research ideas

**Файл:** `yijing_transformer/models/variant3_extensions.py`

10 исследовательских идей + text-analysis модули.

| Класс | Идея |
|-------|------|
| `HexagramPositionalEncoding` | BFS-расстояния на Q6 как позиционное кодирование |
| `SixLineAttention` | 6 голов = 6 линий = 6 доменов (interpretable) |
| `BianGuaOptimizer` | Q6-шаг как curriculum шаг оптимизатора |
| `TernaryKVCache` | {-1,0,+1} сжатие ключей в KV-кеше (2-bit) |
| `HexagramTokenizer` | 64-токенный словарь с правилами слияния по Бян-Гуа |
| `CrossDomainRAG` | Retrieval по Q6-сигнатуре |
| `HexagramEvaluator` | hex_entropy, domain_coherence метрики |
| `MultiScaleQ6` | Matryoshka Q2→Q3→Q6 иерархия |
| `AdaptiveHammingScheduler` | λ curriculum: warmup → anneal → steady |
| `HexagramMoE` | 64 эксперта через hex_weights |
| `BinaryOppositionTable` | Антонимные оси → биты Q6 |
| `SvoyChuzhoiGate` | Q6-расстояние → свой/чужой гейт |
| `TextQualityFilter` | 6-осевой фильтр качества текста |

---

## 10. Когда какую модель использовать

```
Задача: сравнить с baseline
  └──► VanillaGPT

Задача: установить геометрический baseline
  └──► LeanYiJingGPT (только Heisenberg + FlowerOfLifeGAT)

Задача: исследование новых геометрических источников
  └──► YiJingGPT (cfg.hex_strength + нужный компонент)

Задача: production обучение (самообучение, pipeline)
  └──► Variant3GPT + HierarchicalMoEFFN

Задача: 5-уровневая философская иерархия (α=-4..+4)
  └──► HierarchicalE2

Задача: MoE с геометрическим роутингом
  └──► NautilusYiJing (вариант 1 или 3)

Задача: дообучение без изменения весов
  └──► LoRA (apply_lora) или PrefixTuning

Задача: ускорение генерации
  └──► Speculative Decoding (build_draft_model)

Задача: интерпретируемость
  └──► LogitLens + HexagramEvaluator
       SixLineAttention (domain = head)
       NautilusYiJing вариант 1 (routing_confidence)
```
