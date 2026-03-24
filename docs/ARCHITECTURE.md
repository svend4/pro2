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
