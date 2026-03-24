# Инференс — Руководство

## Быстрый старт

```python
from e2_inference import E2Inference

engine = E2Inference()
print(engine.embed("кристалл"))          # Q6-координата + гексаграмма
print(engine.generate("трансформация"))  # продолжение текста
```

---

## E2Inference API (`e2_inference.py`)

`E2Inference` — основной inference-движок для модели `HierarchicalE2`. При инициализации автоматически выбирает лучший доступный checkpoint:

```
checkpoint_e2_joint.pt  >  checkpoint_e2_clusters.pt  >  checkpoint_e2.pt
```

Если ни один файл не найден, модель работает на случайных весах (для тестирования).

### Инициализация

```python
engine = E2Inference(
    checkpoint="my_checkpoint.pt",  # опционально, иначе авто-выбор
    load_corpus=True,               # строить Q6-индекс корпуса (нужен для find_similar, concept_map и т.д.)
)
print(engine.checkpoint_used)  # какой файл загружен
```

### Режим 1: embed

Возвращает Q6-координату текста: бинарный вектор из 6 бит, индекс гексаграммы и домен.

```python
result = engine.embed("гексаграмма")
# {
#   "text": "гексаграмма",
#   "q6": [1, 0, 1, 1, 0, 0],
#   "hex_idx": 13,
#   "hex_name": "Братство",
#   "domain": "PYRO",
#   "q6_str": "101100"
# }
```

CLI:
```bash
python e2_inference.py --embed "кристалл"
```

### Режим 2: find_similar

Ищет топ-N концептов корпуса, ближайших к запросу по расстоянию Хэмминга в Q6.

```python
results = engine.find_similar("гексаграмма", n=5)
# [{"text": ..., "source": ..., "domain": ..., "q6": ..., "hamming": 1}, ...]
```

CLI:
```bash
python e2_inference.py --similar "гексаграмма" --n 5
```

### Режим 3: concept_map

Строит карту распределения всего корпуса по 64 ячейкам Q6-гиперкуба.

```python
cmap = engine.concept_map()
# {0: {"hex_name": "Творчество", "count": 12, "domains": ["GEO"], "examples": [...]}, ...}
```

CLI:
```bash
python e2_inference.py --map
```

### Режим 4: domain_report

Показывает, какие домены активны в Q6-пространстве и насколько плотно они заполнены.

```python
report = engine.domain_report()
# {"PYRO": {"count": 45, "unique_hex": 18, "hex_density": 28.1, "top_hex": [...]}, ...}
```

CLI:
```bash
python e2_inference.py --domain-report
```

### Режим 5: generate

Авторегрессивная генерация продолжения текста из prefix.

```python
text = engine.generate("трансформация", max_new_tokens=30, temperature=0.8)
```

CLI:
```bash
python e2_inference.py --generate "трансформация"
```

### Режим 6: cross_repo_align

Сравнивает Q6-распределения внешнего корпуса (`corpus_loader`) и внутренних кластеров (`repo_corpus_loader`). Находит центроиды, процент выравнивания и пересечение ячеек гиперкуба.

```python
align = engine.cross_repo_align()
# {
#   "external_texts": 120, "internal_texts": 80,
#   "ext_centroid": [0.48, 0.51, ...],
#   "int_centroid": [0.42, 0.55, ...],
#   "centroid_dist": 0.312,
#   "alignment": 94.8,
#   "shared_hex_cells": 22, "total_hex_cells": 31,
#   "coverage_overlap": 70.9,
#   "interpretation": "Хорошая интеграция"
# }
```

Интерпретация `centroid_dist`:
- `< 1.5` — Хорошая интеграция
- `1.5..3.0` — Частичная интеграция
- `> 3.0` — Домены разошлись, рекомендуется joint-training

CLI:
```bash
python e2_inference.py --cross-align
```

### Запустить всё сразу

```bash
python e2_inference.py --query "nautilus portal"
```

---

## NautilusInference (`nautilus_inference.py`)

`nautilus_inference.py` показывает, как микро-модель `Variant3GPT` работает между несколькими репозиториями. Checkpoint по умолчанию: `checkpoint_bidir_v2.pt`.

### Режим A: Standalone

Модель загружена из checkpoint, работает без внешних репо.

```bash
python nautilus_inference.py --query "кристалл" --mode standalone
python nautilus_inference.py --embed "трансформация знаний"
```

В Python:
```python
from nautilus_inference import _load_model, embed_text

model, cfg, loaded = _load_model()  # loaded=False если checkpoint не найден
result = embed_text("трансформация знаний", model, cfg)
# {"text": ..., "q6": [1,0,0,1,1,0], "hex_idx": 25, "hex_name": "Великое накопление", "domain": "COSMO"}
```

### Режим B: Portal

Запрашивает контекст у всех репозиториев через `NautilusPortal`, затем встраивает каждый концепт в Q6.

```bash
python nautilus_inference.py --query "кристалл" --mode portal
```

Каждый репо предоставляет свой контекст:
- `info1` — методологический (α-уровни)
- `meta` — математический (hexagram ID)
- `data7` — теоретический (K₀→K₁→K₂)
- `data2` — прикладной (ЕТД/Скарабей)

В ответе: текст + Q6-координата + репо-источник + α-уровень.

### Режим C: Federated

Описывает схему федеративного обучения: каждый репо обучает свой фрагмент знания, `pro2` объединяет через Q6. `pro2` (Variant3GPT) выступает координатором — знает Q6-пространство и отображает знание из любого репо в Q6.

```bash
python nautilus_inference.py --query "кристалл" --mode federated
```

Поток выполнения:
1. Запрос поступает из любого репо или напрямую
2. `corpus_loader.py` ищет релевантные тексты из всех клонов
3. `Variant3GPT.embed(context)` → Q6[b0..b5]
4. `NautilusPortal` находит ближайшие концепты (расстояние Хэмминга)
5. Ответ: текст + Q6-координата + источник-репо + α-уровень
6. `graph_health.py` обновляет метрики CD/VT/CR/DB

Флаг `--json` выдаёт машиночитаемый вывод для всех режимов.

---

## Python API — программное использование

### Загрузка HierarchicalE2

```python
import torch
from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config

cfg = E2Config(
    vocab_size=256, d_model=128, block_size=32,
    n_core=4, n_heads=4, dropout=0.0,
    hamming_lambda=0.15, uncertainty_budget=0.25, ffn_mult=4,
    n_archetypes=64, il_use_ternary=True,
    nautilus_warmup=200, nautilus_mode="sequential",
    conv_window=4, conv_stride=2, grammar_rows=8, grammar_cols=8,
)
model = HierarchicalE2(cfg)
state = torch.load("checkpoint_e2_joint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
```

### Загрузка Variant3GPT

```python
from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT

cfg = Variant3Config(
    vocab_size=256, block_size=32, d_model=128,
    n_heads=4, n_layers=4, ffn_mult=4,
    hamming_lambda=0.15, uncertainty_budget=0.25,
    dropout=0.0, use_domain_routing=True,
)
model = Variant3GPT(cfg)
state = torch.load("checkpoint_bidir_v2.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
```

### Прямой проход и Q6-эмбеддинг

```python
import torch

text = "кристалл"
tokens = torch.tensor(
    [min(ord(c), cfg.vocab_size - 1) for c in text[:cfg.block_size]],
    dtype=torch.long,
).unsqueeze(0)  # [1, T]

with torch.no_grad():
    logits, *_ = model(tokens)
    hidden = logits.mean(dim=1).squeeze(0)  # [vocab_size]
    q6_raw = hidden[:6]
    q6 = (q6_raw > 0).int().tolist()
    hex_idx = sum(b << i for i, b in enumerate(q6))
```

---

## Speculative Decoding

Файл: `yijing_transformer/models/speculative.py`

Ускоряет генерацию в **2–3 раза** за счёт маленькой «draft» модели. Draft предлагает K токенов, target верифицирует их за один forward pass.

**Алгоритм:**
1. Draft модель генерирует K токенов авторегрессивно
2. Target модель проверяет все K+1 позиций за 1 forward
3. Токены принимаются по rejection sampling: `P(accept) = min(1, p_target / p_draft)`
4. При отказе — сэмпл из adjusted distribution `clamp(p_target - p_draft, 0)`
5. Если все K приняты — бонусный токен из target

### Использование

```python
from yijing_transformer.models.speculative import build_draft_model, speculative_generate

# Draft модель строится автоматически из конфига target.
# cfg_large должен содержать поля draft_d_model и draft_n_layers.
draft = build_draft_model(cfg_large)

# Загрузить веса draft (если есть отдельный checkpoint)
# или оставить случайные (для тестирования алгоритма).

idx = torch.tensor([[32]], dtype=torch.long)  # начальный токен
result = speculative_generate(
    target_model=target,
    draft_model=draft,
    idx=idx,
    max_new_tokens=100,
    K=4,           # токенов за один «спекулятивный» шаг
    temperature=1.0,
    top_k=None,
)
```

### Оценка качества draft модели

```python
from yijing_transformer.models.speculative import measure_acceptance_rate

metrics = measure_acceptance_rate(target, draft, data_tokens, K=4, n_samples=50)
# {
#   "acceptance_rate": 0.72,
#   "tokens_per_step": 3.88,
#   "speedup_estimate": 2.4
# }
```

Acceptance rate < 0.5 означает, что draft модель слишком слабая — нет смысла использовать speculative decoding.

---

## Q6-эмбеддинг

Q6 — это бинарная координата в 6-мерном гиперкубе: вектор из 6 бит `[b0, b1, b2, b3, b4, b5]` ∈ {0,1}⁶. Всего 64 возможных состояния — по числу гексаграмм И-цзин.

### Индекс гексаграммы

```python
hex_idx = sum(bit << i for i, bit in enumerate(q6))  # 0..63
```

### Домены (6 измерений Q6)

| Бит | Домен | Смысл |
|-----|-------|-------|
| b0  | GEO   | Земное, конкретное |
| b1  | HYDRO | Текучее, адаптивное |
| b2  | PYRO  | Трансформирующее |
| b3  | AERO  | Динамическое, воздушное |
| b4  | COSMO | Космическое, масштабное |
| b5  | NOOS  | Ноосферное, абстрактное |

Активный домен определяется старшим установленным битом (функция `_q6_to_domain`). Например, `[1,0,0,0,0,0]` → GEO, `[1,0,1,0,1,0]` → COSMO.

### Расстояние Хэмминга

Семантическая близость двух концептов измеряется расстоянием Хэмминга между их Q6-координатами: количество бит, в которых они различаются (0..6). Расстояние 0 — одна гексаграмма, 6 — противоположные вершины куба.

### Интерпретация гексаграмм

Каждый `hex_idx` соответствует гексаграмме И-цзин с именем. Примеры:

| hex_idx | Имя | Q6 |
|---------|-----|----|
| 0  | Творчество | 000000 |
| 1  | Исполнение | 100000 |
| 13 | Братство   | 101100 |
| 63 | Ещё не завершено | 111111 |

Полный список из 64 гексаграмм — в `yijing_transformer/constants.py` (константа `HEX_NAMES`).

---

## Экспорт модели

Файл: `yijing_transformer/models/export.py`

### ONNX

Экспортирует модель для использования с ONNX Runtime (C++, мобильные устройства, серверный inference).

```python
from yijing_transformer.models.export import export_onnx

export_onnx(model, "model.onnx", seq_len=128, opset_version=14)
```

Входной тензор: `input_ids` — `[batch_size, seq_len]` int64.
Выходной тензор: `logits` — `[batch_size, seq_len, vocab_size]` float32.
Оба измерения динамические (batch и seq_len).

### TorchScript

Экспортирует модель через `torch.jit.trace` для встраивания в C++ приложения.

```python
from yijing_transformer.models.export import export_torchscript

export_torchscript(model, "model_traced.pt", seq_len=128)

# Загрузка и использование без Python:
# auto model = torch::jit::load("model_traced.pt");
```

### Модельная карточка

```python
from yijing_transformer.models.export import create_model_card

card = create_model_card(model, save_path="model_card.json")
# {"model_type": "YiJingGPT", "architecture": {...}, "features": {...}, "parameters": {...}}
```

> **Важно:** перед экспортом убедитесь, что модель переведена в `model.eval()`. ONNX-экспорт использует трейсинг с dummy-входом `[1, seq_len]`.
