# CHANGELOG — Хронология изменений

> Связанные документы: [INDEX.md](INDEX.md) · [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) · [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 2026-03-24 — Session: repository-audit-fvlEG

### Новые файлы

| Файл | Назначение |
|------|------------|
| `experiments/validate_q4_q6.py` | Верификация гипотезы Q4⊂Q6 с 5 архетипами и 6 семантическими осями |
| `monitor_improve.py` | Real-time мониторинг прогресса `e2_self_improve.py` |
| `experiments/train_with_glyph.py` | Сравнение GlyphTokenizer vs CharTokenizer (контрастивное обучение) |

### Обновлённые файлы (merged best-of-both)

#### `yijing_transformer/models/geometry/interlingua_fixed.py`
- Добавлены параметры `start_temp`, `end_temp` (косинусное расписание)
- Добавлен `readout_attn` (MultiheadAttention для readout)
- `get_diagnostics()` → переименовано в `diagnostics()` (+ alias для совместимости)
- Добавлен `__main__` self-test (30 шагов, проверка spread > 0.03)

**Ключевая архитектурная идея:** N отдельных `trit_proj` вместо одного общего → diversity loss штрафует за одинаковое голосование → Gumbel-Softmax устраняет zero-gradient trap.

#### `yijing_transformer/models/geometry/kasatkin_router.py`
- Добавлен `use_learned_axes` параметр (по умолчанию False — фиксированная геометрия)
- Добавлен метод `hex_label()` — человекочитаемые метки доменов по токенам
- Добавлен класс `Q6ExpertBank` — готовый MoE блок (router + 6 FFN)
- Добавлены константы `DOMAIN_NAMES`, `DOMAIN_ALT` на уровне модуля
- Расширен `__main__` (тест Q6ExpertBank + hex_label)

#### `experiments/xerox_test.py`
- Расширено с 11 → 14 тест-кейсов (добавлены: функция потерь, Hamming, softmax, само-описание)
- Загрузчик checkpoint теперь пробует HierarchicalMoE если LeanYiJingGPT не работает
- Улучшен вывод результатов по доменам (bar chart)

#### `run_all_checks.sh`
- Расширено с 4 → 5 проверок
- Добавлен аргумент `$CHECKPOINT` (по умолчанию `hmoe_self_trained_v4.pt`)
- Добавлены переменные статуса (`$Q4_OK`, `$INTER_OK`, etc.)
- Эксперимент 5 (GlyphTokenizer) пропускается если файл не существует

### Новые JSON-результаты

| Файл | Содержание |
|------|------------|
| `experiments/q4_q6_result.json` | avg_hamming=2.56, PARTIAL |
| `experiments/xerox_test_result.json` | score=6/14 (mock-режим) |

### Smoke-тест результаты (2026-03-24)

| Модуль | Метрика | Результат |
|--------|---------|-----------|
| `interlingua_fixed` | spread_trit_pos | 0.25 → **PASS** |
| `interlingua_fixed` | gate | 0.49 → **PASS** |
| `kasatkin_router` | routing_confidence | 0.44 >> 0.15 → **PASS** |
| `validate_q4_q6` | avg_hamming | 2.56 → **PARTIAL** |
| `xerox_test (mock)` | score | 6/14 → нужна реальная модель |

### Коммиты

```
88ddec5  impl: 7 tasks — experiments, merged geometry modules, monitoring tools
67fd389  chore: add experiment result JSONs from smoke-tests
```

---

## 2026-03-22 — Session: meta-интеграция

Подробности: [FUTURE_TASKS.md](../FUTURE_TASKS.md)

| Изменение | Результат |
|-----------|-----------|
| hexnet → Q6-router в HMoE | log_hamming_mix: -1.0 → 2.0, gate std=0.138 |
| hexlearn → RAG с Hamming distance | 8 уникальных вершин из 20 входов |
| hexsym → Aut(Q6) для routing-аугментации | 7 орбит по Hamming weight |
| hexphys → Metropolis annealing | mid_v2 LCI +0.007, weak +0.080 |
| hexring → bent-функции как seed архетипы | bent cosine=0.309 vs normal 0.463 |
| CrossDomainAnalogy 15 → 36 пар | полная 6×6 матрица |
| Pipeline: pipeline.py | score=0.998, LCI=3.127 |

---

## До 2026-03-22 — Базовая архитектура

| Архитектура | PPL | Коммент |
|------------|-----|---------|
| AbrialeBridgeMediator (v59) | **1.24** | ЛУЧШИЙ результат |
| BridgeOfModules (v58) | 1.35 | — |
| BridgedInterlingua (v61) | 2.92 | баг trit_proj |
| ArchetypalInterlingua (v60) | 2.93 | баг trit_proj |
| vanilla baseline | 2.94 | — |

Реализованы: `variant3.py`, `hierarchical_moe.py`, `geometry/routing.py`, `geometry/attention.py`, `geometry/quantizers.py`, `geometry/abriale.py`, `geometry/nautilus.py`.

---

*Последнее обновление: 2026-03-24*
