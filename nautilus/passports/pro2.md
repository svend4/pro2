# Паспорт: svend4/pro2

| Поле | Значение |
|------|----------|
| Репозиторий | svend4/pro2 |
| Формат | `.pro2` — Q6-концепты, граф знаний |
| Единица | Концепт с Q6-координатой `[b0..b5]` |
| Адаптер | `adapters/pro2.py` |
| Уровень совместимости | 3 — интерактивный |

## Семантическое пространство

64 состояния = `{-1,+1}^6` = 64 вершины гиперкуба = 64 гексаграммы И-Цзин.

Метрика расстояния: Хэмминг.

## Ключевые компоненты

| Компонент | Файл | Роль |
|-----------|------|------|
| KnowledgeGraph | `bidir_train.py` | Граф концептов с PageRank |
| AdaptiveLearning | `bidir_train.py` | Обновление весов по генерации |
| AdvancedGenerator | `inference/bridge_inference.py` | 5 стратегий генерации |
| NautilusHierarchy | `geometry/nautilus.py` | 7-камерная архитектура |
| HMoE | `train_hmoe_curriculum.py` | Иерархия экспертов |

## Замкнутый цикл (bidir)

`Корпус → KnowledgeGraph → PageRank → Q6-анкоры → GPT`  
`GPT → QFilter → AdaptiveLearning → identify_gaps → новый корпус → ↺`

Реализует «недостающую петлю» из `data7/knowledge_transformer.py`.

## Мосты к другим репо

| Цель | Связь |
|------|-------|
| meta | Q6-биты `[b0..b5]` ↔ номер гексаграммы |
| info1 | глубина концепта ↔ α-уровень |
| data7 | bidir_train ↔ K₀→K₁→K₂ цикл |
| data2 | scarab_algorithm ↔ Q6-траектория |
