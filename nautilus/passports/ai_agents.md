# Паспорт: домен ai_agents (внутри svend4/pro2)

| Поле | Значение |
|------|----------|
| Репозиторий | svend4/pro2 (домен) |
| Формат | Агентные паттерны — стратегии, циклы, curriculum |
| Единица | Агентный паттерн |
| Адаптер | `adapters/ai_agents.py` |
| Уровень совместимости | 2 — связанный |

## Ключевые агенты

| Агент | Файл | Тип |
|-------|------|-----|
| self_train | `self_train.py` | 3-стадийное самообучение |
| bidir | `bidir_train.py` | Двунаправленный (forward + backward loop) |
| AdvancedGenerator | `inference/bridge_inference.py` | 5 стратегий генерации |
| HMoE curriculum | `train_hmoe_curriculum.py` | 5-фазное обучение экспертов |
| NautilusHierarchy | `geometry/nautilus.py` | 7-камерная иерархия |

## Паттерн: Двунаправленный агент

```
Forward:  KnowledgeGraph → PageRank → концепты → train batch
Backward: generate → filter → evaluate → update graph
```

Цикл замкнут: граф направляет обучение, обучение обогащает граф.

## Мосты к другим репо

| Цель | Связь |
|------|-------|
| pro2 | все агенты реализованы в pro2 |
| data7 | bidir реализует недостающую петлю K₀→K∞ |
| infosystems | KnowledgeGraph — общий для обоих доменов |
