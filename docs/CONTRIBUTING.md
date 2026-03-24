# Руководство для контрибьюторов

## Структура репозитория

```
pro2/
├── yijing_transformer/          # Основной пакет
│   ├── models/                  # Архитектуры моделей
│   │   ├── model.py             # YiJingGPT — базовая модель
│   │   ├── hierarchical_e2.py   # HierarchicalE2 — E2-архитектура с MoE
│   │   ├── hierarchical_moe.py  # HierarchicalMoEFFN, роутеры, Q6ExpertBank
│   │   ├── variant3.py          # Variant3GPT — микро-модель для Nautilus
│   │   ├── speculative.py       # Speculative decoding (ускорение 2-3×)
│   │   ├── export.py            # ONNX и TorchScript экспорт
│   │   └── geometry/            # Квантизаторы, Q6-геометрия
│   ├── training/                # Обучающие утилиты
│   │   └── utils_v{N}.py        # Версионированные утилиты (v12..v52)
│   ├── constants.py             # HEX_NAMES, DOMAINS
│   └── tests/                   # Модульные тесты пакета
│       └── test_hierarchical_moe.py
│
├── tests/                       # Интеграционные и smoke-тесты
│   ├── test_smoke.py            # Быстрая проверка ключевых компонентов (< 30 с)
│   └── test_security.py         # Безопасность: weights_only, JSON-логи
│
├── experiments/                 # Экспериментальные скрипты и результаты
│   ├── validate_q4_q6.py        # Валидация Q4⊂Q6
│   └── xerox_test.py            # Ксерокс-тест воспроизводимости
│
├── nautilus/                    # NautilusPortal — межрепозиторный контекст
│   └── portal.py
│
├── e2_inference.py              # CLI + Python API для HierarchicalE2
├── nautilus_inference.py        # CLI + Python API для Variant3GPT
├── pipeline.py                  # Curriculum-пайплайн обучения
├── bidir_train.py               # Двунаправленное обучение (Variant3GPT)
├── train_e2.py / train_e2_joint.py / train_e2_clusters.py  # Обучение E2
├── self_train_hmoe.py           # Само-обучение HierarchicalMoE
├── bench_all.py                 # Бенчмарк всех вариантов само-обучения
├── bench_stability.py           # Тест стабильности
├── corpus_loader.py             # Внешний корпус текстов
├── repo_corpus_loader.py        # Внутренние кластеры репозитория
├── conftest.py                  # Конфигурация pytest (исключения из сборки)
├── run_all_checks.sh            # Полная проверка архитектуры
└── docs/                        # Документация
```

---

## Запуск тестов

### Быстрый smoke-тест (< 30 секунд)

Проверяет, что ключевые компоненты не падают при запуске:

```bash
pytest tests/test_smoke.py -v
```

Что проверяется:
- `HierarchicalMoEFFN` — hamming_prior не коллапсирует к 0.5
- `MultiScaleGlobalRouter` и `GlobalRouter` — формы выходов
- `meta_q6` — bent seeds, ecube_route, q4_tesseracts
- `nautilus_15agent._build_agents()` — 15 агентов, покрытие > 60 вершин
- `pipeline.run_pipeline` — сигнатура с параметрами `adaptive_lr`, `reset_rag_pass`
- `HierarchicalMoEFFN.forward` — end-to-end прямой проход
- `bench_stability` — импорт модуля

### Тесты безопасности

```bash
pytest tests/test_security.py -v
```

Что проверяется:
- Ни один `.py` файл не содержит `weights_only=False`
- `read_avg_lci()` выводит `[warn]` при некорректном JSON, не молчит
- `self_train_common` — корректные формы `hexagrams (64,6)`, `biangua (64,64)`, CFG
- Сохранение/загрузка checkpoint с `weights_only=True` работает без ошибок

### Тесты модуля HierarchicalMoE

```bash
pytest yijing_transformer/tests/test_hierarchical_moe.py -v
```

Покрывает все компоненты MoE-архитектуры: `Q6ExpertBank`, `MicroExpert`, `BidirBridgeExpert`, `GroupRouter`, `GlobalRouter`, `MultiScaleGlobalRouter`, `HierarchicalMoEFFN`.

### Полная проверка архитектуры

```bash
bash run_all_checks.sh
```

Последовательно запускает:
1. `experiments/validate_q4_q6.py` — валидация Q4⊂Q6
2. `tests/test_interlingua_fixed.py` — per-source trit_proj
3. `yijing_transformer/models/geometry/quantizer_fixed.py --test` — TernaryQuantizerFixed
4. `experiments/xerox_test.py --mock` — ксерокс-тест

Результаты сохраняются в `experiments/*.json`.

### Запустить все тесты сразу

```bash
pytest tests/ yijing_transformer/tests/ -v
```

Файл `conftest.py` в корне исключает `test_dialog.py` из авто-сборки (он требует интерактивного ввода).

---

## Добавление нового варианта архитектуры

**Шаг 1. Создать модуль в `yijing_transformer/models/`**

```python
# yijing_transformer/models/my_variant.py

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class MyVariantConfig:
    """Конфигурация MyVariant."""
    vocab_size: int = 256
    d_model: int = 128
    block_size: int = 32
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0


class MyVariant(nn.Module):
    """
    MyVariant — краткое описание идеи.

    Горизонтальные связи (↔):
      ↔ hierarchical_moe.py — использует HierarchicalMoEFFN
      ↔ self_train_common.py — общая конфигурация CFG
    """

    def __init__(self, cfg: MyVariantConfig):
        super().__init__()
        self.cfg = cfg
        # ... инициализация слоёв

    def forward(self, idx: torch.Tensor):
        # Возвращаем (logits, aux_loss, info_dict) — стандартный контракт
        ...
        return logits, aux_loss, {}
```

Соглашение: `forward` возвращает кортеж `(logits, aux_loss, info_dict)`. Это контракт, которого ожидают `bench_all.py` и `pipeline.py`.

**Шаг 2. Добавить тест**

Создайте файл в `yijing_transformer/tests/test_my_variant.py` или добавьте в существующий `test_smoke.py`:

```python
def test_my_variant_forward():
    """MyVariant: прямой проход без ошибок."""
    from yijing_transformer.models.my_variant import MyVariant, MyVariantConfig
    cfg = MyVariantConfig(d_model=64)
    model = MyVariant(cfg)
    x = torch.randint(0, 256, (1, 8))
    logits, aux, info = model(x)
    assert logits.shape == (1, 8, 256)
    assert isinstance(info, dict)
```

Запустите тест и убедитесь, что проходит:
```bash
pytest tests/test_smoke.py::test_my_variant_forward -v
```

**Шаг 3. Добавить в `bench_all.py`**

В `bench_all.py` варианты описаны в списке конфигураций. Добавьте новый пункт:

```python
# В bench_all.py — список вариантов
VARIANTS = [
    # ... существующие варианты 1-10 ...
    {
        "id": 11,
        "name": "my-variant",
        "description": "MyVariant — краткое описание",
        "script": "my_variant_train.py",  # скрипт обучения
        "fast_args": ["--fast"],
    },
]
```

Запустите бенчмарк для нового варианта:
```bash
python bench_all.py --checkpoint hmoe_self_trained_v5.pt --variants 11
```

---

## Добавление нового обучающего скрипта

**Интеграция с pipeline**

`pipeline.py` запускает обучающие скрипты через `subprocess`. Чтобы новый скрипт работал корректно:

1. Принимайте `--checkpoint` (входной), `--save` (выходной), `--fast` (быстрый режим):

```python
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="", help="Загрузить из файла")
parser.add_argument("--save",       default="my_model.pt", help="Сохранить в файл")
parser.add_argument("--fast",       action="store_true", help="Минимальный прогон для тестов")
```

2. Загружайте checkpoint с `weights_only=True`:

```python
state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=False)
```

3. Сохраняйте только `state_dict` (не весь объект модели):

```python
torch.save(model.state_dict(), args.save)
```

**Формат JSON-лога**

Каждый обучающий скрипт должен писать лог в формате JSON-массива. `pipeline.py` читает его через `read_avg_lci()`, которая ищет поле `avg_lci_all`:

```python
import json

log_entries = []

# В цикле обучения:
log_entries.append({
    "step":        step,
    "loss":        float(loss),
    "avg_lci_all": float(lci_metric),   # главная метрика для pipeline
    "lr":          float(current_lr),
    "timestamp":   time.time(),
})

# Сохранение лога:
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(log_entries, f, ensure_ascii=False, indent=2)
```

`read_avg_lci(log_path)` вернёт среднее значение `avg_lci_all` по всем записям. При отсутствии файла или битом JSON функция возвращает `0.0` и выводит `[warn]` — не бросает исключение.

---

## Правила безопасности

### weights_only=True — обязательно

`torch.load(..., weights_only=False)` — это RCE-уязвимость: произвольный Python-код может быть выполнен при загрузке файла. В кодовой базе это запрещено и проверяется автоматически:

```bash
pytest tests/test_security.py::test_no_weights_only_false -v
```

Тест обходит все `.py` файлы репозитория и падает, если находит `weights_only=False`. Всегда используйте:

```python
# Правильно:
state = torch.load("model.pt", map_location="cpu", weights_only=True)

# Неправильно (тест упадёт):
state = torch.load("model.pt", weights_only=False)
state = torch.load("model.pt")  # по умолчанию небезопасно в старых PyTorch
```

### Нет pickle произвольных объектов

Сохраняйте только `state_dict` (словарь тензоров и примитивов), а не целые объекты модели. Это единственное, что совместимо с `weights_only=True`.

```python
# Правильно:
torch.save(model.state_dict(), "model.pt")

# Неправильно:
torch.save(model, "model.pt")  # сохраняет pickle объекта
```

### JSON-логи: валидация и предупреждения

Функции чтения логов (`read_avg_lci` и аналоги) должны обрабатывать битый JSON без исключений и всегда выводить `[warn]` в stdout. Это проверяется тестами `test_security.py::test_read_avg_lci_warns_on_bad_json` и смежными.

---

## Стиль кода

### Docstrings

Каждый класс и публичная функция должны иметь docstring. Первая строка — краткое описание на русском. Для модулей — заголовок с `↔` связями:

```python
"""
my_module.py — Краткое описание модуля.

Горизонтальные связи (↔):
  ↔ hierarchical_moe.py — использует роутеры
  ↔ corpus_loader.py    — загружает данные
"""
```

### Type hints

Используйте аннотации типов для параметров и возвращаемых значений публичных функций:

```python
def embed(self, text: str) -> dict:
    """Q6-эмбеддинг текста."""
    ...

def find_similar(self, text: str, n: int = 5) -> list[dict]:
    ...
```

### Комментарии на русском

Комментарии внутри тела функций пишутся на русском. Имена переменных и параметров — на английском:

```python
# Вычисляем центроид Q6-пространства для каждого домена
def centroid(q6_list: list[list[int]]) -> list[float]:
    return [sum(q6[i] for q6 in q6_list) / len(q6_list)
            for i in range(6)]
```

### Визуальные разделители

Для разделения смысловых блоков внутри файла используются комментарии-разделители:

```python
# ── Вспомогательные функции ────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
class MainClass:
    ...
```

### Предупреждения через print

В inference-коде некритичные ошибки логируются через `print(f"  [warn] ...")`, не через `logging` и не через исключения. Это позволяет pipeline читать stdout и отслеживать предупреждения.

---

## Версионирование

### utils_v* конвенция

Обучающие утилиты в `yijing_transformer/training/` версионируются по схеме `utils_v{N}.py`. Текущая последняя версия: `utils_v52.py`.

**Как добавить новую версию:**

1. Скопируйте последний файл:
   ```bash
   cp yijing_transformer/training/utils_v52.py yijing_transformer/training/utils_v53.py
   ```

2. Внесите изменения в `utils_v53.py`. Добавьте в начало файла docstring с описанием что изменилось:
   ```python
   """
   v53 утилиты: описание нового функционала.

   Изменения относительно v52:
   - Добавлен XxxAttention (ссылка на статью)
   - Исправлен YyyBug
   """
   ```

3. Старые файлы не удаляются — они остаются для воспроизводимости экспериментов.

4. Обновите импорты в скриптах, которые должны использовать новую версию:
   ```python
   from yijing_transformer.training.utils_v53 import NewFeature
   ```

### bench_v* логи

Результаты бенчмарков сохраняются как `bench_v{N}_log.json` в корне репозитория. Не удаляйте старые логи — они нужны для сравнения динамики обучения.

### Checkpoint-файлы

Checkpoints именуются по схеме: `{model}_{version}.pt` или `{model}_{experiment}_v{N}.pt`. Например:
- `hmoe_self_trained_v5.pt`
- `checkpoint_e2_joint.pt`
- `checkpoint_bidir_v2.pt`
