# API Reference

---

## Модели

### YiJingGPT

Базовый трансформер с геометрической регуляризацией гиперкуба.

```python
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models import YiJingGPT

config = YiJingConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=4)
model = YiJingGPT(config)

# Forward pass
logits = model(input_ids)                    # (B, T, vocab_size)
logits = model(input_ids, targets=targets)   # + loss computation
```

### Variant3GPT

Архетипо-центричная архитектура.

```python
from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT

config = Variant3Config(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_domains=6,          # 6 доменов Q6
    interlingua_dim=64,   # размер интерлингвы
)
model = Variant3GPT(config)
```

**Полезные методы:**
```python
# Доминирующая гексаграмма для входа
hex_idx = model.get_dominant_hexagram(input_ids)  # (B,)

# Активные домены
domains = model.get_active_domains(input_ids)  # (B, 6) bool
```

### HierarchicalMoE

4-уровневая иерархическая смесь экспертов.

```python
from yijing_transformer.models.hierarchical_moe import (
    HierarchicalMoEConfig, HierarchicalMoE
)

config = HierarchicalMoEConfig(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_experts=64,         # гексаграммы
    expert_capacity=1.25, # capacity factor
)
model = HierarchicalMoE(config)
```

### NautilusYiJing

7-камерная архитектура Наутилуса.

```python
from yijing_transformer.models.nautilus_yijing import NautilusYiJing

model = NautilusYiJing(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_chambers=7,
)
```

### VanillaGPT (baseline)

Стандартный трансформер без геометрии.

```python
from yijing_transformer.models import VanillaGPT

model = VanillaGPT(config)  # принимает YiJingConfig, игнорирует hex-параметры
```

---

## Конфигурация

### YiJingConfig

```python
from yijing_transformer.config import YiJingConfig
```

#### Основные поля

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `vocab_size` | int | 2048 | Размер словаря |
| `d_model` | int | 512 | Размерность модели |
| `n_layers` | int | 12 | Число слоёв |
| `n_heads` | int | 8 | Число голов внимания |
| `block_size` | int | 512 | Максимальная длина последовательности |
| `dropout` | float | 0.05 | Dropout |
| `bias` | bool | False | Использовать bias в linear layers |

#### И-Цзин специфика

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `hex_strength` | float | 0.01 | Вес геометрического лосса |
| `temp` | float | 0.3 | Температура квантизатора |
| `use_bian_gua` | bool | True | BianGua attention bias |
| `adaptive_temp` | bool | True | Адаптивная температура |
| `quantizer_type` | str | 'factored6' | Тип квантизатора |
| `quant_total_dim` | int | 6 | Размерность Q-пространства |
| `quant_group_dim` | int | 3 | Размерность группы (факторизация) |

#### Обучение

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `lr` | float | 3e-4 | Learning rate |
| `warmup_steps` | int | 2000 | Шаги warmup |
| `batch_size` | int | 8 | Размер батча |
| `grad_accum_steps` | int | 4 | Шаги gradient accumulation |
| `total_steps` | int | 50000 | Общее число шагов |
| `weight_decay` | float | 0.1 | Weight decay |
| `max_grad_norm` | float | 1.0 | Gradient clipping |
| `label_smoothing` | float | 0.0 | Label smoothing |
| `use_amp` | bool | False | Mixed precision |

#### Логирование

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `log_every` | int | 100 | Логировать каждые N шагов |
| `save_every` | int | 2000 | Сохранять чекпоинт каждые N |
| `val_every` | int | 500 | Валидация каждые N |
| `use_wandb` | bool | False | Логирование в WandB |
| `use_tensorboard` | bool | False | Логирование в TensorBoard |
| `project_name` | str | "yijing-transformer" | Имя проекта в WandB |

#### Продвинутые техники

| Поле | Тип | По умолчанию | Описание |
|------|-----|-------------|----------|
| `use_rope` | bool | False | Rotary Position Embedding |
| `use_swiglu` | bool | False | SwiGLU FFN |
| `use_flash_attn` | bool | False | Flash Attention |
| `use_lora` | bool | False | LoRA-адаптация |
| `lora_rank` | int | 8 | Ранг LoRA |
| `lora_alpha` | float | 16.0 | Alpha LoRA |
| `use_ema` | bool | False | EMA весов |
| `weight_tying` | bool | True | Связывание embedding ↔ LM head |

#### Пресеты

```python
cfg = YiJingConfig.tiny()    # d=128, layers=4, heads=4   (~2M params)
cfg = YiJingConfig.small()   # d=256, layers=6, heads=8   (~15M params)
cfg = YiJingConfig.medium()  # d=512, layers=12, heads=8  (~85M params)
cfg = YiJingConfig.large()   # d=1024, layers=24, heads=16 (~300M params)
```

---

## Квантизаторы

```python
from yijing_transformer.models.geometry import (
    YiJingQuantizer,
    FactoredYiJingQuantizer,
    HierarchicalQuantizer,
    MatryoshkaQuantizer,
    DeformableQuantizer,
    GumbelQuantizer,
    TernaryQuantizer,
    E8Quantizer,
    GroupedQuantizer,
    AntipodalQuantizer,
    PairedBitQuantizer,
    FourStateQuantizer,
)
```

**Общий интерфейс:**

```python
quantizer = FactoredYiJingQuantizer(d_model=128, total_dim=6, group_dim=3)

quantized, indices, info = quantizer(x)
# quantized: (B, T, d_model) — квантизованные представления
# indices:   (B, T)          — индексы гексаграмм [0..63]
# info:      dict             — {"commitment_loss": tensor, ...}
```

---

## Inference

### Простая генерация

```python
from yijing_transformer.inference.generate import generate

text = generate(
    model,
    sp,                         # SentencePiece tokenizer
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
)
```

### AdvancedGenerator (5 стратегий)

```python
from yijing_transformer.inference.bridge_inference import AdvancedGenerator

gen = AdvancedGenerator(model, tokenizer, device="cuda")

# Nucleus sampling (по умолчанию)
text = gen.generate("Начало пути", max_tokens=100, strategy="nucleus")

# Greedy
text = gen.generate("Начало пути", strategy="greedy")

# Beam search
text = gen.generate("Начало пути", strategy="beam", num_beams=4)

# Speculative decoding (нужна draft-модель)
text = gen.generate("Начало пути", strategy="speculative", draft_model=draft)

# Dynamic temperature (entropy-based)
text = gen.generate("Начало пути", strategy="dynamic_temp", base_temp=0.7)

# С KV-cache
text = gen.generate_with_kv_cache("Начало пути", max_tokens=200)
```

---

## Обучение

### Основной цикл

```python
from yijing_transformer.training.train import train

# Через CLI
# python -m yijing_transformer.training.train --vocab_size 256 --d_model 128

# Программно
import argparse
args = argparse.Namespace(
    vocab_size=256, d_model=128, n_layers=4, n_heads=4,
    lr=3e-4, total_steps=1000, batch_size=8,
    use_amp=False, use_wandb=False,
)
model = train(args)
```

### Ключевые функции

```python
from yijing_transformer.training.train import (
    get_lr,                    # cosine schedule с warmup
    generate_synthetic_batch,  # синтетические данные для тестов
    estimate_val_loss,         # оценка лосса на валидации
    measure_hex_contribution,  # анализ вклада геометрии по слоям
)

# Learning rate на шаге 500:
lr = get_lr(step=500, cfg=config)

# Validation loss:
val_loss = estimate_val_loss(model, config, device="cuda", num_batches=20)

# Geometric contribution per layer:
stats = measure_hex_contribution(model)
# {"layer_0": {"hex_scale": 0.3, "temperatures": [0.5, ...]}, ...}
```

### TrainingBridge

```python
from yijing_transformer.training.bridge import TrainingBridge

bridge = TrainingBridge(model, config)
optimizer = bridge.create_optimizer()    # AdamW с bridge-specific настройками
scheduler = bridge.create_scheduler(optimizer)

for step in range(total_steps):
    loss = model(x, targets=y)
    loss.backward()
    bridge.step(loss, step)  # мониторинг, регуляризация, адаптация
    optimizer.step()
    scheduler.step()
```

---

## Данные

### TextDataset

```python
from yijing_transformer.data_utils import TextDataset, ShuffledBatchIterator

dataset = TextDataset(
    text="your training text here...",
    block_size=128,
    tokenizer=char_tokenizer,
)

iterator = ShuffledBatchIterator(dataset, batch_size=8)
for x, y in iterator:
    # x: (8, 128), y: (8, 128)
    pass
```

### Streaming (для больших корпусов)

```python
from yijing_transformer.data_utils import get_batch_streaming, create_train_val_iterators

# Требует: pip install datasets
train_iter, val_iter = create_train_val_iterators(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    block_size=128,
    batch_size=8,
)
```

---

## Токенизаторы

### CharTokenizer

```python
from yijing_transformer.tokenizer.char_tokenizer import CharTokenizer

tok = CharTokenizer()
tok.fit("training text")             # построить словарь
ids = tok.encode("hello")            # [7, 4, 11, 11, 14]
text = tok.decode(ids)               # "hello"
print(tok.vocab_size)                # количество уникальных символов
```

### GlyphTokenizer

```python
from yijing_transformer.tokenizer.glyph_tokenizer import GlyphTokenizer

tok = GlyphTokenizer(max_glyph_size=4)
tok.fit("training text")
ids = tok.encode("hello world")
```

---

## Геометрия (core)

### Генерация структур

```python
from yijing_transformer.models.geometry import (
    generate_trigrams,     # 8 триграмм: (8, 3) tensor
    generate_hexagrams,    # 64 гексаграммы: (64, 6) tensor
    generate_octograms,    # 256 октограмм: (256, 8) tensor
    generate_hypercube,    # N-мерный гиперкуб
    generate_e8_roots,     # 240 корней E8: (240, 8) tensor
)

trigrams = generate_trigrams()    # tensor of {-1, +1}^3
hexagrams = generate_hexagrams() # tensor of {-1, +1}^6
```

### Порядки и структуры

```python
from yijing_transformer.models.geometry import (
    fuxi_order,              # Порядок Фу Си (натуральный бинарный)
    wenwang_order,           # Порядок Вэнь-Вана (традиционный)
    palace_clusters,         # 8 палацев по 8 гексаграмм
    antipodal_pairs,         # 32 антиподальные пары
    loshu_kernel,            # Магический квадрат Ло Шу
    triangular_distance_matrix, # Матрица Хэмминговых расстояний
)

palaces = palace_clusters()
# {0: [0, 1, 2, ...], 1: [8, 9, 10, ...], ...}

pairs = antipodal_pairs()
# [(0, 63), (1, 62), ...]  — гексаграмма и её инверсия
```

### Kasatkin embedding

```python
from yijing_transformer.models.geometry import (
    kasatkin_embedding,          # Проекция из Q6 в R³
    kasatkin_distance_matrix,    # Евклидовы расстояния в R³
    kasatkin_axis_projection,    # Проекция на оси
)

coords_3d = kasatkin_embedding()  # (64, 3) — 3D координаты гексаграмм
```

---

## LoRA

```python
from yijing_transformer.models import (
    apply_lora,          # Применить LoRA к модели
    freeze_non_lora,     # Заморозить не-LoRA параметры
    unfreeze_all,        # Разморозить всё
    merge_lora,          # Слить LoRA в основные веса
    unmerge_lora,        # Отделить LoRA обратно
    save_lora_weights,   # Сохранить только LoRA веса
    load_lora_weights,   # Загрузить LoRA веса
    count_lora_parameters, # Подсчитать параметры LoRA
)

# Применить LoRA (rank=8) к attention layers
apply_lora(model, rank=8, alpha=16.0, target_modules=["q_proj", "v_proj"])
freeze_non_lora(model)

# Тренировка...

# Сохранение
save_lora_weights(model, "lora_weights.pt")

# Слияние для inference (без overhead)
merge_lora(model)
```

---

## Экспорт

```python
from yijing_transformer.models import (
    export_onnx,         # Экспорт в ONNX
    export_torchscript,  # Экспорт в TorchScript
    create_model_card,   # Создать model card
)

# ONNX
export_onnx(model, "model.onnx", input_shape=(1, 128))

# TorchScript
export_torchscript(model, "model.pt")

# Model card
card = create_model_card(model, config)
```

---

## Speculative Decoding

```python
from yijing_transformer.models import (
    build_draft_model,       # Построить маленькую draft-модель
    speculative_generate,    # Генерация со спекулятивной декодировкой
    measure_acceptance_rate, # Измерить acceptance rate
)

draft = build_draft_model(config, n_layers=2)  # 2 слоя вместо 12

text = speculative_generate(
    model,
    draft,
    input_ids,
    max_tokens=100,
    gamma=5,  # число спекулятивных токенов за шаг
)

rate = measure_acceptance_rate(model, draft, test_data)
# ~0.7 означает 70% спекулятивных токенов принимаются
```

---

## Константы

```python
from yijing_transformer.constants import HEX_NAMES, HEX_NAMES_SHORT, hex_name

# 64 названия гексаграмм (порядок Вэнь-Вана)
print(HEX_NAMES[0])    # "Цянь (Творчество)"
print(HEX_NAMES[1])    # "Кунь (Исполнение)"

# Короткие имена
print(HEX_NAMES_SHORT[0])  # "Цянь"

# Утилита
print(hex_name(42))          # "Цянь (Творчество)"  или что соответствует idx=42
print(hex_name(42, short=True))  # "Цянь"
```
