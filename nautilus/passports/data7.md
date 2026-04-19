# Паспорт: svend4/data7

| Поле | Значение |
|------|----------|
| Репозиторий | svend4/data7 |
| Формат | `.data7` — теория трансформации знаний |
| Единица | Концепт + граф + цикл K₀→K∞ |
| Адаптер | `adapters/data7.py` |
| Уровень совместимости | 2 — связанный |

## Ключевые идеи

Два полюса знания: **Диссертации** (специализированное) ↔ **Энциклопедии** (обобщённое).

Цикл трансформации: `K₀ → decompose → aggregate → K₁ → synthesize → K₂ → ...`

## Недостающая петля

В `data7/knowledge_transformer.py` есть комментарий:
```
# Missing: proposals → decomposer.decompose() → refinement loop
```
Эта петля **реализована в `pro2/bidir_train.py`** (AdaptiveLearning + identify_gaps).

## Таблица аналогий data7 ↔ pro2

| data7 | pro2 |
|-------|------|
| `compute_centrality()` | `hex_weights` |
| `identify_gaps()` | `domain_triplet_loss` |
| `generate_hypotheses()` | `self_dialog stage 3` |
| TSP-оптимизация порядка | BFS по biangua-графу |

## Мосты к другим репо

| Цель | Связь |
|------|-------|
| pro2 | bidir_train реализует K₀→K∞ цикл |
| info1 | K₀ (специализация) ↔ α=-4; K∞ (обобщение) ↔ α=+4 |
