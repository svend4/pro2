# Справочник тренировочных утилит (`training/`)

**Расположение:** `yijing_transformer/training/`

Модуль объединяет 41 экспериментальный файл (`utils_v12..v52`, ~21 000 строк) через систему «мостов» (bridge pattern). Все утилиты доступны через единую точку входа `TrainingBridge`.

---

## Содержание

1. [Архитектура мостов](#1-архитектура-мостов)
2. [TrainingBridge — единая точка входа](#2-trainingbridge--единая-точка-входа)
3. [Оптимизаторы (bridge_optimizers.py)](#3-оптимизаторы)
4. [Планировщики LR (bridge_schedulers.py)](#4-планировщики-lr)
5. [Регуляризация (bridge_regularization.py)](#5-регуляризация)
6. [Мониторинг (bridge_monitors.py)](#6-мониторинг)
7. [Хирургия модели (bridge_model_surgery.py)](#7-хирургия-модели)
8. [Оптимизатор (optim.py)](#8-optimpy--базовый-оптимизатор)
9. [EMA и ранняя остановка (ema.py)](#9-emapy)
10. [Регуляризация специфическая для Q6 (regularization.py)](#10-regularizationpy--q6-регуляризация)
11. [Дистилляция (distillation.py)](#11-distillationpy)
12. [Таблица источников utils_v12..v52](#12-таблица-источников)

---

## 1. Архитектура мостов

```
bridge.py  ←  TrainingBridge (единая точка входа)
├── bridge_optimizers.py    — Sophia, LAMB, Lion, SAM, Lookahead
├── bridge_schedulers.py    — WSD, Cosine Restarts, Curriculum, LLRD
├── bridge_regularization.py — Z-Loss, AGC, Label Smoothing, Mixup
├── bridge_monitors.py      — Loss Spike, Grokking, Gradient Flow
└── bridge_model_surgery.py — µP, Pruning, DoRA, Merging, Freezing
```

Каждый bridge-файл — **адаптер**: импортирует классы из `utils_v*.py` и предоставляет единый API без знания внутренней версионности.

---

## 2. `TrainingBridge` — единая точка входа

**Файл:** `training/bridge.py:80`

```python
from yijing_transformer.training.bridge import TrainingBridge

bridge = TrainingBridge(model, cfg)

# Создание компонентов (ленивая инициализация)
optimizer   = bridge.build_optimizer(optimizer_type='sophia')
scheduler   = bridge.build_scheduler(scheduler_type='wsd')
reg_suite   = bridge.build_regularization()
monitor     = bridge.build_monitor(verbose=True)
surgeon     = bridge.build_surgeon()

# В training loop
for step, (x, y) in enumerate(loader):
    logits, loss = model(x, y)

    bridge.before_backward(logits, y, loss)
    loss.backward()
    bridge.after_backward(model, step)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    bridge.after_step(step, loss.item(), scheduler.get_last_lr()[0])
```

### Методы `TrainingBridge`

| Метод | Описание |
|-------|---------|
| `build_optimizer(type, wrapper, **kw)` | Создать оптимизатор из реестра |
| `build_scheduler(type, **kw)` | Создать LR scheduler |
| `build_regularization()` | Создать `RegularizationSuite` |
| `build_monitor(verbose)` | Создать `TrainingMonitor` |
| `build_surgeon()` | Создать `ModelSurgeon` (pruning/merging) |
| `build_data_pipeline()` | Создать `DataPipeline` (аугментация) |
| `before_backward(logits, targets, loss)` | Z-Loss, contrastive hooks |
| `after_backward(model, step)` | AGC, gradient centralization |
| `after_step(step, loss, lr)` | Loss spike detection, LR logging |

---

## 3. Оптимизаторы

**Файл:** `training/bridge_optimizers.py`

| Оптимизатор | Источник | Описание |
|------------|---------|---------|
| `adamw` (default) | PyTorch | AdamW с LLRD из optim.py |
| `sophia` | utils_v19 | Sophia-G — second-order (Hessian-based) |
| `lamb` | utils_v18 | LAMB — Layer-wise Adaptive Moments (large batch) |
| `lookahead` | utils_v23 | Lookahead — slow/fast weights (k=5, α=0.5) |
| `sam` | utils_v27 | SAM — Sharpness-Aware Minimization |
| `lion` | utils_v45 | Lion — Evolved Sign Momentum (memory-efficient) |

```python
# Sophia с Lookahead обёрткой
optimizer = bridge.build_optimizer(
    optimizer_type='sophia',
    wrapper='lookahead',
    lr=1e-4,
    weight_decay=0.1,
)

# LAMB для большого batch
optimizer = bridge.build_optimizer(optimizer_type='lamb', lr=1e-3)
```

---

## 4. Планировщики LR

**Файл:** `training/bridge_schedulers.py`

| Тип | Источник | Описание |
|----|---------|---------|
| `cosine` | PyTorch / utils_v32 | Cosine annealing с опциональным min_lr |
| `wsd` | utils_v18, v44 | Warmup → Stable → Decay (Llama 3 style) |
| `cosine_restarts` | utils_v32 | Cosine Annealing Warm Restarts |
| `curriculum` | utils_v29, v35 | Curriculum scheduler (рост сложности данных) |
| `llrd` | utils_v14, v24 | Layerwise LR Decay (разные lr по глубине) |

### WSD (рекомендуется для production)

```
Warmup:  0 → T_w  | lr: 0 → max_lr
Stable:  T_w → T_s | lr: max_lr
Decay:   T_s → T_d | lr: max_lr → min_lr  (cosine/linear)
```

```python
scheduler = bridge.build_scheduler(
    scheduler_type='wsd',
    warmup_steps=200,
    stable_steps=800,
    decay_steps=200,
    min_lr=1e-6,
)
```

### LLRD (Layerwise LR Decay)

```python
# Каждый слой ниже × 0.8 от lr верхнего слоя
optimizer = bridge.build_optimizer(optimizer_type='adamw', llrd_factor=0.8)
```

---

## 5. Регуляризация

**Файл:** `training/bridge_regularization.py`

| Техника | Источник | Описание |
|---------|---------|---------|
| Z-Loss | utils_v15 | Стабилизация logits (PaLM style) |
| AGC | utils_v21 | Adaptive Gradient Clipping (NFNet) |
| Gradient Centralization | utils_v30 | Центрирование градиентов |
| Spectral Regularization | utils_v31 | Ограничение спектральной нормы |
| Gradient Penalty | utils_v33 | R1/R2 penalty |
| Token Dropout | utils_v34 | Случайное выбрасывание токенов |
| Weight Standardization | utils_v36 | Нормализация весов |
| Label Smoothing | utils_v43 | Сглаживание меток (ε=0.1) |
| Focal Loss | utils_v43 | Фокусная функция потерь |
| R-Drop | utils_v43 | Консистентность двух dropout-прогонов |
| Mixup | utils_v49 | Смешение последовательностей |
| Entropy Regularization | utils_v50 | Штраф за слишком уверенный softmax |

```python
reg = bridge.build_regularization()
# reg автоматически подключается через before_backward / after_backward
```

---

## 6. Мониторинг

**Файл:** `training/bridge_monitors.py`

`TrainingMonitor` автоматически обнаруживает аномалии:

| Детектор | Источник | Триггер |
|---------|---------|---------|
| Loss Spike Detector | utils_v24 | loss > mean + 3σ |
| Grokking Detector | utils_v21 | val_loss падает после плато |
| Attention Entropy Monitor | utils_v24 | entropy < threshold (collapse) |
| Gradient Flow Monitor | utils_v51 | vanishing/exploding gradients |
| Sharpness Estimator | utils_v51 | кривизна loss landscape |
| MI Estimator | utils_v31 | Mutual Information между слоями |

```python
monitor = bridge.build_monitor(verbose=True)

# В loop
alerts = monitor.step(loss, step)
for alert in alerts:
    print(f"[ALERT] {alert}")

# Периодическая диагностика
if step % 100 == 0:
    report = monitor.diagnostics()
    # {'attention_entropy': 2.1, 'gradient_norm': 0.8, 'loss_spike': False, ...}
```

---

## 7. Хирургия модели

**Файл:** `training/bridge_model_surgery.py`

| Операция | Источник | Описание |
|---------|---------|---------|
| µP (maximal update) | utils_v12 | Инициализация для переноса гиперпараметров |
| Structured Pruning | utils_v16 | Удаление голов/нейронов по важности |
| Unstructured Pruning | utils_v20 | Magnitude pruning весов |
| DoRA | utils_v20 | Directional reparametrization (DoRA/LoRA) |
| Model Merging | utils_v16, v30 | SLERP / Task Arithmetic слияние |
| Parameter Freezing | utils_v30 | Заморозка отдельных компонентов |

```python
surgeon = bridge.build_surgeon()

# Pruning до 20% sparsity
surgeon.prune(method='magnitude', sparsity=0.2)

# Применить DoRA к attention слоям
surgeon.apply_dora(target_modules=['q_proj', 'v_proj'], rank=16)
```

---

## 8. `optim.py` — Базовый оптимизатор

**Файл:** `training/optim.py`

```python
from yijing_transformer.training.optim import build_optimizer, get_cosine_schedule

optimizer = build_optimizer(model, cfg, llrd_factor=0.8)
# Создаёт AdamW с группами параметров:
# - no_decay: bias, LayerNorm, embedding
# - decay: все остальные
# - опционально LLRD: lr × llrd_factor^depth

lr = get_cosine_schedule(step, warmup_steps=100, total_steps=1000, max_lr=1e-4)
```

---

## 9. `ema.py`

**Файл:** `training/ema.py`

```python
from yijing_transformer.training.ema import EMA, EarlyStopping

# EMA: скользящее среднее весов
ema = EMA(model, decay=0.9999)
ema.update()           # после каждого шага
with ema.apply():      # временно заменить веса на EMA для eval
    val_loss = evaluate(model)

# Ранняя остановка
stopper = EarlyStopping(patience=5, min_delta=1e-4)
if stopper(val_loss):
    break
```

---

## 10. `regularization.py` — Q6-регуляризация

**Файл:** `training/regularization.py`

Специфические для проекта техники регуляризации:

| Класс | Строка | Описание |
|-------|--------|---------|
| `AntipodalRegularization` | 302 | Штраф за нарушение антиподальной симметрии Q6 |
| `HexagramAntipodalLoss` | 335 | Loss: `∑ max(0, sim(h_i, h_antipodal(i)) - margin)` |
| `TokenMerger` | 22 | Merging близких токенов (reducing redundancy) |

```python
from yijing_transformer.training.regularization import HexagramAntipodalLoss
loss_fn = HexagramAntipodalLoss(margin=0.3)
antipodal_loss = loss_fn(hex_embeddings)  # (B, T, 64)
```

---

## 11. `distillation.py`

**Файл:** `training/distillation.py`

```python
from yijing_transformer.training.distillation import (
    distillation_loss, feature_distillation_loss, DistillationTrainer
)

# KL-дивергенция учитель/ученик
loss = distillation_loss(
    student_logits, teacher_logits, targets,
    temperature=4.0, alpha=0.7,
)

# Дистилляция скрытых состояний
feat_loss = feature_distillation_loss(student_hidden, teacher_hidden)
```

---

## 12. Таблица источников

Каждый `utils_v*.py` содержит одну или несколько изолированных идей:

| Версия | Ключевые компоненты |
|--------|-------------------|
| v12 | µP init, Dynamic Temperature, Checkpoint Manager, Perplexity |
| v13 | Ring Attention, Selective Checkpointing, Weight Decay Schedule |
| v14 | Layerwise LR Decay (LLRD), Data Mixing, Model Surgery |
| v15 | Z-Loss, Gradient Accumulation Profiler, Vocab Expansion |
| v16 | Structured Pruning, Contrastive Loss, Sequence Packing, PCGrad, Merging |
| v17 | RAG Scaffold, Activation Statistics |
| v18 | WSD Scheduler, BPE Dropout, LAMB, NEFTune |
| v19 | Sophia, RMSNorm, Chunked Prefill, CAGrad, EMA Decay |
| v20 | DoRA, Sparse Attention, Gradient Vaccine, Loss Landscape |
| v21 | FSDP Simulator, Grokking Detector, GradEMA, AGC |
| v22 | Matryoshka Embeddings, Reptile, Token Frequency, SWA |
| v23 | Lookahead, Activation Histogram, Curriculum, EMA, Gradient Noise |
| v24 | LLRD, Grad Accum Scheduler, Attention Entropy, Loss Spike |
| v25 | Gradient Centralization, LR Finder, Param Efficiency, Batch Finder |
| v26 | Sparse Attention, Gradient Vaccine, Progressive Resizing, NCE Loss |
| v27 | SAM, Dynamic Temperature, Gradient Projection, Metrics Dashboard |
| v28 | Lookahead, SWA, Label Smoothing Warmup, Batch Finder |
| v29 | EMA, Curriculum, Gradient Noise, LR Probing, Weight Decay |
| v30 | Grad Centralization, AGC, Loss Spike, Param Freezing, Snapshotter |
| v31 | MI Estimator, Gradient Surgery, Spectral Reg, Token Freq Weighting |
| v32 | Gradient Histogram, Cosine Restarts, Multi-Scale Loss, Data Mixing |
| v33 | Gradient Penalty, Polyak Averaging, Loss Landscape, Adaptive Batch |
| v34 | Safe Grad Accum, LLRD, Token Dropout, Convergence Detector |
| v35 | Curriculum, Grad Noise, Dynamic Batch, Param Efficiency, Stability |
| v36–v41 | EMA, Gradient Vaccine, AGC, LR Finder, Weight Standardization + вариации |
| v42–v52 | Lion, Focal Loss, Mixup, R-Drop, Entropy Reg, Confidence Penalty, QLoRA |

> Для доступа к специфической версии можно импортировать напрямую:
> ```python
> from yijing_transformer.training.utils_v45 import Lion
> ```
> Однако рекомендуется использовать bridge API для стабильности.
