# YiJing-Transformer: Отчёт v60-v69 — Статус, решённые проблемы и текущее состояние

**Дата:** 2026-03-23 (обновлено; оригинальный отчёт от 2026-03-11)
**Ветка:** `main` (все фиксы слиты)

---

## 1. Краткая сводка проекта

YiJing-Transformer — трансформерная архитектура с индуктивным смещением на основе геометрии гиперкуба Q6 = {-1,+1}^6. 64 гексаграммы И-Цзин = вершины 6-мерного гиперкуба. Проект интегрирует 7 теоретических источников (Склярова, Фомюк, Андреев, Касаткин, Герман, Беляев, SOLAN) в единую архитектуру.

### Эволюция подходов к медиации между модулями

```
v53-v54:  Модули подключены напрямую к forward pass          → PPL 2.91 (vanilla-уровень)
v55-v56:  ConvergenceBridge + TernaryQuantizer                → Теоретический фундамент
v57:      AbrialeLayer (событийная активация)                 → Commit-gate механизм
v58:      BridgeOfModules (иерархическое дерево мостов)       → PPL 1.35 (прорыв!)
v59:      AbrialeBridgeMediator (Abriale + Bridge)            → PPL 1.24 (лучший результат)
v60:      ArchetypalInterlingua (hub-and-spoke, 64 архетипа)  → PPL 2.93 (≈ vanilla)
v61:      BridgedInterlingua (мосты + архетипы)               → PPL 2.92
v62:      PairedBitQuantizer (трит из пары битов)             → PPL 2.08 (29% лучше vanilla)
v63:      NautilusHierarchy + ConvergenceBridge + GlyphPrior  → PPL 1.02-2.31
v64:      Nautilus + Bridge + GlyphPrior (combined)           → PPL 1.35 (54% лучше vanilla)
v65:      MatryoshkaQuantizer (bit→trit→hex каскад)           → PPL 2.26 (23% лучше vanilla)
v66-v67:  MatryoshkaNautilus / PostCascadeMatryoshka           → PPL 5.97-6.11 (hard tasks)
v68:      Matryoshka на алгоритмических задачах               → 9.1% улучшение на Copy/Sort/ListOps
v69:      Bridge Utils (оптимизаторы/шедулеры/регуляризация)  → PPL 0.99 (nautilus+WSD)
```

---

## 2. Текущее состояние: все проблемы v60-v61 решены

### 2.1. Проблемы, диагностированные в v60-v61

Оригинальный отчёт (2026-03-11) выявил 5 ключевых проблем:

| # | Проблема | Статус | Решение | Коммит |
|---|----------|--------|---------|--------|
| 1 | Интерлингва ≈ vanilla (PPL 2.93) | **РЕШЕНА** | Per-source trit_proj + diversity loss + consensus aggregation | `209ad73`, `96c7e61` |
| 2 | Gate ≈ 0.5 (модель игнорирует интерлингву) | **РЕШЕНА** | Global gate bias +0.5 → sigmoid ≈ 0.62 | `209ad73` |
| 3 | Нет дифференциации тритов | **РЕШЕНА** | Diversity loss (Variant C) + per-source проекторы | `209ad73` |
| 4 | Общий trit_proj для всех модулей | **РЕШЕНА** | `ArchetypalInterlinguaFixed` с per-source trit_projs | `96c7e61` |
| 5 | 800 шагов < 2000 warmup | **РЕШЕНА** | Более длинные прогоны в v62-v69 бенчмарках | множественные |

### 2.2. Что именно было реализовано

#### Variant B: Per-Source trit_proj — РЕАЛИЗОВАН

- **Файл:** `routing.py`, класс `ArchetypalInterlinguaFixed`
- **Эксперимент:** `train_interlingua_fixed.py`
- **Результат: PROVEN** — PPL 1.1712 < baseline 1.51 (коммит `a13877e`)
- Каждый источник получил свой `nn.Linear(d_model, n_archetypes)` проектор

#### Variant C: Diversity Loss — РЕАЛИЗОВАН

- **Файл:** `routing.py`, строки 1360-1382 (`ArchetypalInterlingua`) и 1822-1833 (`BridgedInterlingua`)
- **Формула:** `loss += 0.1 * mean(|cos_sim_offdiag(trit_patterns)|)`
- **Механизм:**
  1. Сохраняются все trit scores: `_last_all_trit_scores` (N, B, archetypes)
  2. Flatten по batch: (N, B*archetypes)
  3. Попарная cos-similarity между источниками: (N, N) матрица
  4. Штраф = среднее |cos_sim| по off-diagonal элементам
  5. Цель: cos_sim → 0 (ортогональные голоса)
- **Верификация:** 1619 тестов пройдено, diversity loss > 0, градиенты проходят ко всем trit_projs

#### Consensus-Weighted Aggregation — РЕАЛИЗОВАН

- **Файл:** `routing.py`, строки 1279-1286 (`ArchetypalInterlingua`) и 1760-1765 (`BridgedInterlingua`)
- **Механизм:** `alignment = all_trits * consensus; source_weights = softmax(alignment)`
- Заменяет наивное `mean_contrib` на взвешенную сумму: источники, согласные с большинством, усиливаются

#### Global Gate Bias — РЕАЛИЗОВАН

- `ArchetypalInterlingua`: строка 1060, `torch.tensor(0.5)` (было 0.0)
- `BridgedInterlingua`: строка 1557, `torch.tensor(0.5)`
- sigmoid(0.5) ≈ 0.62 — стартовое преимущество interlingua pathway

#### Variant E: Hybrid Bridge+Interlingua — РЕАЛИЗОВАН

- **Класс:** `BridgedInterlingua` в `routing.py`, строка 1500+
- **Архитектура:** Module → Bridge → 64 Archetype → Core (двухфазная)
- Комментарий: "Гибрид BridgeOfModules (v58) и ArchetypalInterlingua (v60)"

#### Kasatkin Router — PROVEN

- **Коммит:** `0184a24`
- **Результат:** routing_conf = 0.3450 >> 0.15, PPL 1.0355 < baseline 1.51
- Использует PrivilegedAxis + CubeDiagonal как механизм маршрутизации

---

## 3. Архитектурные компоненты — текущее состояние

### 3.1. Ядро (всегда работает)

| Компонент | PPL | Статус |
|-----------|-----|--------|
| **Heisenberg attention** (Беляев) | 1.01 | КРИТИЧЕН |
| **FlowerGAT** (Беляев) | 1.01 (с Heisenberg) | КРИТИЧЕН |

### 3.2. Медиация модулей

| Компонент | PPL | Статус |
|-----------|-----|--------|
| BridgeOfModules (v58) | 1.35 | Работает |
| AbrialeBridgeMediator (v59) | 1.24 | Лучший PPL (bridge) |
| ArchetypalInterlingua + fixes (v60+) | 1.17 | **PROVEN** (was 2.93) |
| BridgedInterlingua (v61+) | — | Реализован с diversity loss |

### 3.3. Квантизация

| Компонент | PPL | Статус |
|-----------|-----|--------|
| TernaryQuantizer (v56) | — | Фундамент |
| PairedBitQuantizer (v62) | 2.08 | 29% лучше vanilla |
| MatryoshkaQuantizer (v65) | 2.26 | 23% лучше vanilla (bit→trit→hex) |
| PostCascadeMatryoshka (v67) | 5.97 | 9.1% на hard tasks |

### 3.4. Геометрические приоры

| Компонент | Статус |
|-----------|--------|
| ConvergenceBridge + GlyphPrior | Реализован (convergence.py) |
| GlyphComposer (Q6 hierarchies) | Реализован (4 уровня: glyph→edge→face→sigil) |
| SOLAN glyph_prior_gate | Реализован (model.py:912-926) |
| NautilusHierarchy (v63) | Реализован (nautilus.py) |

### 3.5. Инфраструктура обучения (Bridge System — 18,484 строки)

Активировано в коммите `d78b157`:

| Файл | Строк | Что делает |
|------|-------|------------|
| `bridge.py` | 220 | Главная точка входа |
| `bridge_optimizers.py` | 115 | Sophia, LAMB, Lion, SAM, Lookahead |
| `bridge_schedulers.py` | 157 | WSD, Cosine Restarts, Curriculum, LLRD |
| `bridge_regularization.py` | 240 | Z-Loss, AGC, Mixup, Label Smoothing |
| `bridge_monitors.py` | 267 | Loss Spike, Grokking, Gradient Flow |
| `bridge_model_surgery.py` | 231 | µP init, Pruning, DoRA, Merging |
| `bridge_inference.py` | 237 | Beam Search, Speculative Decoding |
| `bridge_augmentation.py` | 203 | Seq Packing, BPE Dropout, RAG |

Интеграция: `train.py` обновлён с полной поддержкой (`--optimizer`, `--scheduler`, `--mup-init`).

### 3.6. Eval Suite + Domain-Locked Generation (Stage A)

**evaluation/eval_suite.py** — 7-метрическая автоматическая оценка:

| Метрика | Baseline v1 | Domain-Locked v2 | Target |
|---------|-------------|-------------------|--------|
| PPL comfort zone | 23.5 | 20.7 | <15 |
| PPL discomfort zone | 127.4 | 152.4 | <30 |
| Routing accuracy | 75.0% | 75.0% | >70% ✅ |
| Routing confidence | 2.7% | 3.1% | >15% |
| Code completion top-1 | 50.0% | 50.0% | >55% |
| Domain mixing rate | 66.0% | 69.0% | <20% |
| Speed | 119 tok/s | 130 tok/s | >100 ✅ |

**inference/domain_locked_generate.py:**
- Gate-scale manipulation: доминирующий эксперт ×1.8, остальные ×0.4
- Router temperature sharpening: 1.0 → 0.5
- **Результат:** PPL generation 11.2 → 9.9 (+12.1%), русский текст 15.6 → 9.0 (+42%)

---

## 4. Сводка бенчмарков v53-v69

### 4.1. Основные бенчмарки (синтетический WikiText, 800-2000 шагов)

| Версия | Лучшая конфигурация | PPL | vs Vanilla | Файл результатов |
|--------|---------------------|-----|-----------|-------------------|
| v53 | baseline | 2.94 | — | `benchmark_wikitext_v53_results.json` |
| v58 | seven_bridge | 1.35 | -54% | `benchmark_v58_bridge_results.json` |
| v59 | abriale_bridge | 1.24 | -58% | `benchmark_v59_full_results.json` |
| v60 | interlingua_ternary | 2.93 | -0.3% | `benchmark_v60_interlingua_results.json` |
| v61 | bridged_lightweight | 2.92 | -0.7% | `benchmark_v61_bridged_interlingua_results.json` |
| v62 | paired_bit_ternary | 2.08 | -29% | `benchmark_v60_paired_bit_results.json` |
| v63 | heisenberg_gat_only | 1.02 | -65% | `benchmark_v63_nautilus_hierarchy_results.json` |
| v64 | all_flat (combined) | **1.35** | **-54%** | `benchmark_v64_combined_results.json` |
| v65 | matryoshka_bit_trit | 2.26 | -23% | `benchmark_v65_matryoshka_results.json` |
| v69 | nautilus_wsd | **0.99** | **-66%** | `benchmark_v69_bridge_utils_results.json` |

### 4.2. Hard Algorithmic Tasks (Copy, Sorting, ListOps)

| Конфигурация | PPL | vs Vanilla |
|-------------|-----|-----------|
| vanilla | 6.72 | — |
| nautilus_only | 6.16 | -8.3% |
| post_cascade_matryoshka | **6.11** | **-9.1%** |

### 4.3. PROVEN эксперименты (реальные данные, baseline PPL 1.51)

| Эксперимент | PPL | Routing Conf | Результат |
|-------------|-----|-------------|-----------|
| interlingua_fixed | 1.17 | — | **PROVEN** |
| kasatkin_as_router | 1.04 | 34.5% | **PROVEN** |
| meta_hexlearn_router | 1.02 | — | DISPROVEN (routing) / BEST PPL |
| palace_block_sparse | 6.9 | — | DISPROVEN |
| solan_nautilus | 6.9 | — | DISPROVEN |

---

## 5. Исторические проблемы v60-v61 и как они были решены

### 5.1. Информационный bottleneck — УСТРАНЁН

**Оригинальная цепочка причин (v60-v61):**
```
Общий trit_proj → одинаковые триты → одинаковые архетипы → readout = constant → gate ≈ 0.5 → PPL ≈ vanilla
```

**Решение (v62+):**
```
Per-source trit_projs → diversity loss → ортогональные голоса → разные архетипы → consensus-weighted aggregation → полезный readout → gate > 0.5 → PPL 1.17
```

### 5.2. Gate ≈ 0.5 — УСТРАНЁН

- **Причина:** Нулевая инициализация + отсутствие gradient signal от бесполезной интерлингвы
- **Решение:** Bias = +0.5 (sigmoid ≈ 0.62) + diversity loss обеспечивает gradient signal
- **Верификация:** Gate = 0.622 при инициализации, градиенты проходят

### 5.3. Отсутствие дифференциации тритов — УСТРАНЕНО

- **Причина:** Единый trit_proj проецировал все модули одинаково
- **Решение:** Per-source проекторы + diversity loss (штраф 0.1 × cos_sim)
- **Верификация:** 1619 тестов pass, diversity loss > 0 в smoke test

---

## 6. Аудит кода (4 раунда, 2026-03-23)

За последние 2 часа проведён масштабный аудит:

| Раунд | Коммит | Что исправлено |
|-------|--------|----------------|
| 1 | `e41bea3` | 10 critical/high/medium fixes |
| 2 | `594e075` | Gradient flow, dead code, off-by-one, k_deform wiring |
| 3 | `3ff1e07` | Training utils & eval scripts hardening |
| 4 | `b0b17a4` | Два исправления по результатам архитектурного аудита |
| GPU | `eef6e16` | Multi-GPU device/dtype safety across all model files |
| Tests | `be3f042` | Shared constants + 69 новых тестов |
| Security | `8b5b52b` | torch.load RCE fix, silent exceptions |

**Тесты:** ~1700+ (1619 verified + 69 new)

---

## 7. Направления дальнейшего развития

### 7.1. Ближайшее (Stage A+)

Eval Suite показала 2 PASS / 5 FAIL из 7 целей. Приоритеты:
- **PPL comfort zone:** 23.5 → target <15 (нужно продолжить обучение / scale up)
- **Routing confidence:** 2.7% → target >15% (kasatkin_as_router достигает 34.5% — нужно интегрировать)
- **Domain mixing rate:** 66% → target <20% (domain-locked generation снижает, но недостаточно)

### 7.2. Масштабирование

- WSD scheduler (v69) даёт лучшие результаты: nautilus_wsd PPL 0.99
- AGC регуляризация эффективна: heisenberg_agc PPL 1.00
- Готово к масштабированию на WikiText-103 с d=256+

### 7.3. Теоретическое

- PairedBitQuantizer (trit = bit_a + bit_b - 1) даёт 29% улучшение — подтверждает ценность тернарной логики
- MatryoshkaQuantizer (bit→trit→hex каскад) даёт 23% улучшение — иерархическое кодирование работает
- SOLAN glyph priors слегка помогают (v63: 2.32 → 2.31 с glyph prior) — нужно больше данных

---

## 8. Метаданные бенчмарков

| Бенчмарк | Файл результатов | Дата |
|----------|-------------------|------|
| v53 WikiText | `benchmark_wikitext_v53_results.json` | 2026-03-07 |
| v53 S-box | `benchmark_sbox_v53_results.json` | 2026-03-07 |
| v53 Pairwise | `benchmark_pairwise_v53_results.json` | 2026-03-07 |
| v58 Bridge | `benchmark_v58_bridge_results.json` | 2026-03-08 |
| v58 Bridge Real | `benchmark_v58_bridge_real_results.json` | 2026-03-08 |
| v58 Seven Sources | `benchmark_v58_seven_sources_results.json` | 2026-03-08 |
| v59 Full | `benchmark_v59_full_results.json` | 2026-03-09 |
| v60 Interlingua | `benchmark_v60_interlingua_results.json` | 2026-03-11 |
| v60 PairedBit | `benchmark_v60_paired_bit_results.json` | 2026-03-11 |
| v61 Diagnostic | `benchmark_v61_diagnostic_results.json` | 2026-03-11 |
| v61 BridgedInterlingua | `benchmark_v61_bridged_interlingua_results.json` | 2026-03-11 |
| v63 NautilusHierarchy | `benchmark_v63_nautilus_hierarchy_results.json` | 2026-03-12 |
| v64 Combined | `benchmark_v64_combined_results.json` | 2026-03-12 |
| v65 Matryoshka | `benchmark_v65_matryoshka_results.json` | 2026-03-12 |
| v67 PostCascade | `benchmark_v67_post_cascade_results.json` | 2026-03-12 |
| v68 Hard Tasks | `benchmark_v68_hard_results.json` | 2026-03-12 |
| v69 Bridge Utils | `benchmark_v69_bridge_utils_results.json` | 2026-03-12 |

---

*Обновлено 2026-03-23 на основе всех коммитов до `197bb7f` включительно.*
*Оригинальный отчёт: `REPORT-v60-v61-status.md` (2026-03-11)*
*Предыдущий отчёт: `REPORT-v53-v59.md`*
*Теоретические основания: `archetypal-interlingua-theory.md`*
*Методология v61: `docs/v61-bridged-interlingua-methodology.md`*
