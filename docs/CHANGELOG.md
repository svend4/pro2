# CHANGELOG — Хронология изменений

> Связанные документы: [INDEX.md](INDEX.md) · [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) · [EXPERIMENTS.md](EXPERIMENTS.md) · [THEORY_VS_PRACTICE.md](THEORY_VS_PRACTICE.md)

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
374566e  docs: add EXPERIMENTS, IMPLEMENTATION_STATUS, CHANGELOG; update INDEX + FUTURE_TASKS
```

### Документация (вторая часть — аудит кодовой базы)

Аудитом выявлены и задокументированы ранее не охваченные компоненты:

| Новый/обновлённый документ | Что добавлено |
|---------------------------|--------------|
| `docs/PORTAL.md` (новый) | NautilusPortal + 5 адаптеров + corpus_loader + repo_corpus_loader + graph_health + q6_graph_updater + e2_concept_evolution + federated_round + meta_bridge + meta_q6 + nautilus_inference |
| `docs/TRAINING.md` (обновлён) | §3: bidir_train, roundabout, multi_salesman, nautilus_15agent, nautilus_clover, train_e2* (3 варианта), train_hmoe_staged, train_hmoe_curriculum; §9: токенизаторы; §10: bench_moe/stability, eval_hmoe, scripts/ |
| `docs/IMPLEMENTATION_STATUS.md` (обновлён) | Сводная таблица расширена до ~40 компонентов, разбита по 4 категориям |
| `docs/INDEX.md` (обновлён) | +PORTAL.md в навигации, +8 quick-search вопросов |

```
374566e  docs: EXPERIMENTS, IMPLEMENTATION_STATUS, CHANGELOG; update INDEX + FUTURE_TASKS
9c1aa33  docs: full codebase audit — portal, training scripts, tokenizers, benchmarks
(текущий)  docs: THEORY_VS_PRACTICE — theory vs code deep analysis
```

### Документация (третья часть — теория vs практика)

Глубокий аудит 12 теоретических документов vs 50+ файлов кода. Ключевые находки:

| Находка | Теория | Код | Критичность |
|---------|--------|-----|-------------|
| Walsh-Hadamard (Теорема 5) | есть | только в `theoretical_analysis.py` | 🟡 |
| Temperature T→0 | T=0.1 за 800 шагов | T≈0.689 (не достигает) | 🔴 |
| ArchetypalInterlingua | N тrit_proj | исправлено, но не подключено | 🔴 |
| SOLAN-76 глифов | основной токенизатор | standalone, не интегрирован | 🔴 |
| Turning Point v59 | 7 источников синтез | 1 работает (Беляев), 5 удалены | документировано |
| Q4⊂Q6 в обучении | инициализация RAG | только верификация | 🟡 |

Добавлен `docs/THEORY_VS_PRACTICE.md` с оценками (теория 8.5/10, реализация 5.5/10, интеграция 3.5/10) и дорожной картой gap→fix (6 задач ПРИОРИТЕТ 0).

### Задачи 0.1–0.6 (все gap→fix закрыты)

#### `yijing_transformer/models/model.py`
- Импортирован `ArchetypalInterlinguaFixed` из `geometry.interlingua_fixed`
- При `interlingua_use_fixed=True` (по умолчанию) используется исправленная версия
- Сохранена совместимость: `interlingua_use_fixed=False` → оригинал (для ablation)
- `_build_quantizer('ternary')`: передаёт `warmup_steps/start_temp/end_temp` из конфига

#### `yijing_transformer/models/geometry/quantizers.py`
- `TernaryQuantizer`: добавлены `warmup_steps=5000`, `start_temp=1.0`, `end_temp=0.01`
- `TernaryQuantizer.step_temp()`: косинусное расписание T: 1.0→0.01 за warmup_steps
- Новый класс `WHT_Quantizer` (Теорема 5, Walsh-Hadamard):
  - O(n log n) через butterfly матрицу Адамара-Уолша
  - `hard_bits()` → {-1,+1}^n для RAG/routing
  - `use_spectral_loss` — штраф за неравномерность спектра (bent-функции)

#### `self_train_hmoe.py`
- `_CDA_ALL_PAIRS` — 36 направленных пар 6×6 (все направления включая B→A реверсы)
- `cross_domain_signal()` — сэмплирование из 36 пар вместо random orbit
- `_init_glyph_tokenizer()` + `_USE_GLYPH` + `--glyph` флаг (задача 0.3)
- `_encode()`: если `--glyph`, SOLAN символы → Q6 vertex index вместо UTF-8 bytes
- `_step_all_quantizers()` + вызов из `micro_train()`: step_temp() для всех TernaryQuantizer

#### `experiments/train_with_glyph.py`
- Переработан: cosine similarity → triplet loss (anchor/positive/negative)
- Результат: char margin=+0.45, glyph margin=+0.78, delta=+0.33 >> 0.02 → integrate=True
- Вывод: `experiments/glyph_comparison.json`

#### `e2_self_improve.py`
- `load_q4_q6_hamming()`: читает `experiments/q4_q6_result.json`
- `q4_cluster_init()`: 16 Q4-архетипов × keyword-based text clustering
- `SelfDiagnostics._load_data()`: Q4⊂Q6 init при avg_hamming < 2.5
- Текущий статус: avg_hamming=2.56 → реализовано, не активировано (порог не достигнут)

```
7257221  fix(priority-0): close 3 theory→code gaps (tasks 0.1, 0.2, 0.4, 0.5)
c1b4d6b  fix(priority-0): tasks 0.3, 0.6 + priority-1 integrations
```

### Итог сессии 2026-03-24 — все PRIORITY 0 задачи закрыты

| Задача | Файл | Результат |
|--------|------|-----------|
| 0.1: interlingua_fixed в model.py | `models/model.py` | ✅ |
| 0.2: TernaryQuantizer cosine annealing | `geometry/quantizers.py` | ✅ |
| 0.3: GlyphTokenizer интеграция | `train_with_glyph.py` + `self_train_hmoe.py` | ✅ margin+0.33 |
| 0.4: WHT_Quantizer | `geometry/quantizers.py` | ✅ |
| 0.5: CDA 36 пар | `self_train_hmoe.py` | ✅ |
| 0.6: Q4⊂Q6 RAG init | `e2_self_improve.py` | ✅ реализовано |
| Priority 1: step_temp() в цикле | `self_train_hmoe.py` | ✅ |

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
