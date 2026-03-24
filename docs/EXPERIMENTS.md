# Experiments — Справочник всех экспериментов

> **Актуально:** 2026-03-24  
> Связанные документы: [INDEX.md](INDEX.md) · [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) · [GEOMETRY.md](GEOMETRY.md) · [TRAINING.md](TRAINING.md)

---

## Быстрый запуск всех проверок

```bash
# Один скрипт — все 5 проверок
./run_all_checks.sh

# С конкретным checkpoint
./run_all_checks.sh hmoe_self_trained_v4.pt
```

---

## 1. Валидация Q4⊂Q6 — математическая гипотеза

**Файл:** `experiments/validate_q4_q6.py`  
**Гипотеза:** тексты одного PseudoRAG-архетипа (Q4) → гексаграммы Q6 из одного кластера (Hamming ≤ 2).  
**Источник:** [PSEUDORAG_YIJING_BRIDGE.md](../PSEUDORAG_YIJING_BRIDGE.md)

```bash
python experiments/validate_q4_q6.py
python experiments/validate_q4_q6.py --model hmoe_self_trained_v4.pt
```

**Критерий успеха:**

| global_avg_hamming | Интерпретация |
|--------------------|--------------|
| ≤ 2.5 | ПОДТВЕРЖДЕНО: PseudoRAG → учитель YiJing |
| 2.5 – 3.0 | ЧАСТИЧНО: инициализация улучшит результат |
| > 3.0 | ОТКЛОНЕНО: нужна другая схема осей Q4→Q6 |

**Последний результат (2026-03-24):** `avg_hamming=2.56` → ЧАСТИЧНО  
**Результат:** `experiments/q4_q6_result.json`

**Что настраивать при провале:** массив `AXES_Q6` в начале файла — 6 пар keyword-списков, определяющих семантические оси.

---

## 2. Тест ArchetypalInterlinguaFixed

**Файл:** `yijing_transformer/models/geometry/interlingua_fixed.py`  
**Баг исходного:** один общий `trit_proj` → все источники голосуют одинаково → PPL=vanilla.  
**Фикс:** N отдельных проекторов + diversity loss + Gumbel-Softmax.  
**Подробнее:** [IMPLEMENTATION_STATUS.md — Interlingua](IMPLEMENTATION_STATUS.md#2-archetypalinterlinguafixed)

```bash
# Встроенный self-test (30 шагов обучения)
python yijing_transformer/models/geometry/interlingua_fixed.py
```

**Критерий успеха:**

| Метрика | Норма | Провал |
|---------|-------|--------|
| `diversity_loss` | > 0.0 | ≤ 0.0 (все источники одинаковы) |
| `gate` | 0.3 – 0.7 | < 0.1 или > 0.9 (gate коллапсировал) |
| `spread_trit_pos` | > 0.03 | < 0.01 (нет дифференциации) |

**Последний результат (2026-03-24):** spread=0.25, gate=0.49 → **PASS**

**Как подключить в модель:**
```python
# БЫЛО (convergence.py):
from .geometry.convergence import ArchetypalInterlingua
self.interlingua = ArchetypalInterlingua(d_model, n_sources=7)

# СТАЛО:
from .geometry.interlingua_fixed import ArchetypalInterlinguaFixed
self.interlingua = ArchetypalInterlinguaFixed(
    d_model=d_model, n_sources=7, diversity_weight=0.01, warmup_steps=3000
)
```

---

## 3. Тест KasatkinQ6Router

**Файл:** `yijing_transformer/models/geometry/kasatkin_router.py`  
**Идея:** маршрутизация 6 экспертов через 3D-проекцию в куб Касаткина вместо softmax(W·x).  
**Преимущество:** routing объяснимо — видно, какая гексаграмма активна.  
**Подробнее:** [GEOMETRY.md — Q6-маршрутизация](GEOMETRY.md) · [IMPLEMENTATION_STATUS.md — KasatkinRouter](IMPLEMENTATION_STATUS.md#3-kasatkinq6router)

```bash
python yijing_transformer/models/geometry/kasatkin_router.py
```

**6 доменов (оси куба):**

| Ось | Домен | Alt-имя |
|-----|-------|---------|
| +X | GEO | CODE |
| -X | HYDRO | RECON |
| +Y | PYRO | SYSTEM |
| -Y | AERO | MATH |
| +Z | COSMO | HUMAN |
| -Z | NOOS | INFO |

**Критерий успеха:** `routing_confidence > 0.15` (т.е. max_weight > 1/6 + 15%)  
**Последний результат (2026-03-24):** confidence=0.44 → **PASS**  
**Протокол теста:** минимум 3000 шагов на реальных данных; результат до этого недействителен.

---

## 4. Ксерокс-тест — само-осознание модели

**Файл:** `experiments/xerox_test.py`  
**Идея:** если RECON-эксперт специализируется на русском, модель должна описывать routing на русском с PPL < 50. Если PPL=143 везде — специализация иллюзорна.  
**Подробнее:** [IMPLEMENTATION_STATUS.md — Ксерокс-тест](IMPLEMENTATION_STATUS.md#4-ксерокс-тест)

```bash
# Mock (без модели) — проверяет только routing
python experiments/xerox_test.py --mock

# С реальной моделью
python experiments/xerox_test.py --checkpoint hmoe_self_trained_v4.pt

# Сохранить результат
python experiments/xerox_test.py --checkpoint model.pt --output results/xerox_step500.json
```

**14 тест-кейсов по доменам:**

| Домен | Кол-во тестов | Max PPL |
|-------|---------------|---------|
| CODE | 4 | 8 – 18 |
| RECON | 4 | 35 – 70 |
| MATH | 4 | 15 – 40 |
| SYSTEM | 2 | 20 – 25 |

**Критерии:**

| score | Вердикт |
|-------|---------|
| ≥ 0.80 | ПРОЙДЕН: модель осознаёт архитектуру |
| 0.50 – 0.80 | ЧАСТИЧНО |
| < 0.50 | ПРОВАЛЕН: нужно больше обучения |

**Последний результат (2026-03-24, mock):** 6/14 (43%) — mock PPL не репрезентативен, нужна реальная модель.  
**Результат:** `experiments/xerox_test_result.json`

**Запускать:** каждые 500 шагов обучения. Цель: score ≥ 0.80 до масштабирования.

---

## 5. GlyphTokenizer vs CharTokenizer

**Файл:** `experiments/train_with_glyph.py`  
**Вопрос:** улучшает ли геометрическая токенизация SOLAN семантические представления?  
**Метрика:** cosine similarity между семантически близкими концептами после контрастивного обучения.

```bash
python experiments/train_with_glyph.py --steps 200 --fast
python experiments/train_with_glyph.py --steps 1000
```

**Интерпретация delta (glyph − char):**

| delta | Вывод |
|-------|-------|
| > +0.02 | GlyphTokenizer улучшает представления |
| −0.02 – +0.02 | Нейтрально, нужно больше шагов |
| < −0.02 | GlyphTokenizer вреден на этих данных |

**Результат:** `experiments/glyph_comparison.json`

---

## 6. Мониторинг e2_self_improve

**Файл:** `monitor_improve.py`  
**Назначение:** real-time отображение прогресса `e2_self_improve.py` в отдельном терминале.

```bash
# Терминал 1: запустить обучение
python e2_self_improve.py --iters 5 --target-ppl 100 --no-v3

# Терминал 2: мониторинг
python monitor_improve.py --watch 10
```

Читает `e2_self_improve_log.json` и отображает PPL-график по итерациям.

---

## Матрица: что запускать когда

| Ситуация | Эксперимент |
|----------|-------------|
| Первый запуск / свежий clone | `./run_all_checks.sh` |
| После изменения `interlingua_fixed.py` | эксперимент 2 |
| После изменения routing/MoE кода | эксперимент 3 |
| После 500 шагов обучения | эксперимент 4 (ксерокс) |
| Тест новой токенизации | эксперимент 5 |
| Проверка что Q4→Q6 работает | эксперимент 1 |
| Во время долгого `e2_self_improve` | эксперимент 6 (монитор) |

---

## Файлы результатов

| JSON | Создаётся при | Ключевые поля |
|------|---------------|---------------|
| `experiments/q4_q6_result.json` | validate_q4_q6.py | `global_avg_hamming`, `hypothesis_confirmed` |
| `experiments/xerox_test_result.json` | xerox_test.py | `score`, `passed`, `total` |
| `experiments/glyph_comparison.json` | train_with_glyph.py | `char`, `glyph`, delta |
| `e2_self_improve_log.json` | e2_self_improve.py | `avg_ppl` по итерациям |

---

*Последнее обновление: 2026-03-24 | Ветка: `claude/repository-audit-fvlEG`*
