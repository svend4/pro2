# Быстрый старт

## Установка

### 1. Клонирование репозитория

```bash
git clone <repo_url>
cd pro2
```

### 2. Установка пакета

```bash
# Минимальная установка (только PyTorch)
pip install -e .

# С данными (datasets, sentencepiece)
pip install -e ".[data]"

# С визуализацией (matplotlib)
pip install -e ".[viz]"

# С трекингом экспериментов (wandb, tensorboard)
pip install -e ".[tracking]"

# Полная установка
pip install -e ".[all]"
```

**Требования:** Python 3.8+, PyTorch 2.0+

---

## Сценарий 1: Запуск готового Pipeline

Самый простой способ запустить полное обучение:

```bash
python pipeline.py
```

Это запустит 3-фазный pipeline:
1. **NautilusMoE** — базовое мультидоменное обучение
2. **Turbine LCI** — оптимизация LCI-аттрактора
3. **Benchmark** — оценка на всех доменах

Результаты сохраняются в `pipeline_runs/` и `checkpoints/`.

---

## Сценарий 2: Curriculum-обучение NautilusMoME

```bash
python train_hmoe_curriculum.py
```

5-фазное обучение от простого к сложному. Чекпойнты: `checkpoints/hmoe_curriculum_v*.pt`.

---

## Сценарий 3: Бенчмарк существующих моделей

```bash
python bench_all.py
```

Запускает все модели из `checkpoints/` на тестовых доменах. Результаты: `bench_all_results.json`.

---

## Сценарий 4: Self-training

```bash
python self_train_hmoe.py
```

Итеративное самообучение модели на собственных выходах. Создаёт `hmoe_self_trained_v*.pt`.

---

## Использование как библиотека

### Загрузка обученной модели

```python
import torch
from nautilus.model import NautilusMoME

# Загрузка чекпойнта
checkpoint = torch.load("checkpoints/checkpoint_hmoe.pt")
model = NautilusMoME(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state"])
model.eval()
```

### YiJingGPT с нуля

```python
from yijing_transformer.models import YiJingGPT
from yijing_transformer.tokenizer import BPETokenizer

# Инициализация модели
model = YiJingGPT(
    vocab_size=4096,
    d_model=128,
    n_layers=4,
    n_heads=4,
    block_size=256,
)

# Токенизатор
tokenizer = BPETokenizer.from_pretrained("yijing_transformer/tokenizer/")

# Прямой проход
tokens = tokenizer.encode("Hello world")
input_ids = torch.tensor([tokens])
logits, loss = model(input_ids)
```

### Работа с геометрическими структурами

```python
from yijing_transformer.models.geometry import (
    generate_trigrams,
    generate_hexagrams,
    YiJingQuantizer,
    FactoredQuantizer,
)

# Генерация кодбуков
trigrams = generate_trigrams()    # (8, 3) — вершины куба {-1,+1}³
hexagrams = generate_hexagrams()  # (64, 6) — вершины Q6

# Расстояние Хэмминга между гексаграммами
h1 = hexagrams[0]   # ☰☰ (вся Небо)
h2 = hexagrams[63]  # ☷☷ (вся Земля)
hamming_dist = (h1 != h2).sum()  # = 6 (максимальное)

# Квантизация
quantizer = YiJingQuantizer(d_model=128)
x = torch.randn(2, 10, 128)    # (batch, seq, d_model)
x_q = quantizer(x)              # квантизован к ближайшей гексаграмме
```

### Генерация текста

```python
from yijing_transformer.inference import generate_text

output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="def fibonacci(n):",
    max_tokens=100,
    temperature=0.8,
    top_k=40,
)
print(output)
```

---

## CLI-команды

После `pip install -e .` доступны команды:

```bash
# Обучение YiJingGPT (базовое)
yijing-train --epochs 10 --lr 3e-4

# Обучение на WikiText-103
yijing-wikitext --dataset wikitext-103-v1

# Все расширения (MoE, self-training, etc.)
yijing-extensions

# Downstream fine-tuning
yijing-downstream --task classification --dataset <path>
```

---

## Конфигурация

Основные параметры в `yijing_transformer/config/`:

```python
# config/default.py
DEFAULT_CONFIG = {
    # Модель
    "vocab_size": 4096,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "block_size": 256,
    "dropout": 0.1,

    # Обучение
    "lr": 3e-4,
    "batch_size": 32,
    "weight_decay": 0.01,
    "warmup_steps": 200,

    # MoE
    "n_experts": 6,
    "router_type": "soft",  # "soft" | "top1" | "top2"
    "lci_lambda": 0.1,       # вес LCI-loss
}
```

---

## Структура чекпойнтов

```
checkpoints/
├── checkpoint_hmoe.pt          # Базовая модель
├── hmoe_self_trained_v1.pt     # После 1 итерации self-training
├── hmoe_self_trained_v6.pt     # После 6 итераций (лучший)
├── bench_v14.pt                # Лучший бенчмарк
└── pipeline_phase3.pt          # После полного pipeline
```

Структура файла чекпойнта:
```python
{
    "model_state": state_dict,
    "config": model_config_dict,
    "step": int,
    "ppl": float,
    "lci": float,
    "gate_stats": dict,
}
```

---

## Тесты

```bash
# Все тесты
pytest tests/

# Тест воспроизводимости
python test_reproducibility.py

# Тест LCI-аттрактора
python test_lci_attractor.py

# Тесты пакета
pytest yijing_transformer/tests/
```

---

## Мониторинг обучения

### Wandb

```bash
pip install wandb
wandb login
python pipeline.py --use_wandb
```

### TensorBoard

```bash
pip install tensorboard
python pipeline.py --use_tensorboard
tensorboard --logdir runs/
```

### JSON-логи

Каждый запуск создаёт `pipeline_runs/run_{timestamp}.json`:
```json
{
  "step": 5000,
  "loss": 2.87,
  "ppl": 17.42,
  "lci": 3.07,
  "gate_stats": {"CODE": 0.261, "RECON": 0.559, ...}
}
```

---

## Troubleshooting

### CUDA out of memory

Уменьшите `batch_size` или `block_size`:
```bash
python pipeline.py --batch_size 16 --block_size 128
```

### Медленное обучение

Убедитесь, что используется GPU:
```python
import torch
print(torch.cuda.is_available())  # должно быть True
```

### PPL не сходится

1. Проверьте LR: попробуйте `lr=1e-3` или `lr=1e-4`
2. Проверьте данные: убедитесь, что корпус загружен корректно
3. Попробуйте warm start от готового чекпойнта:
   ```bash
   python pipeline.py --resume checkpoints/checkpoint_hmoe.pt
   ```
