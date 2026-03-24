# Theory vs Practice — Разрыв между теорией и кодом

> **Актуально:** 2026-03-24 | Основан на анализе 12 теоретических документов и 50+ файлов кода  
> Связанные документы: [INDEX.md](INDEX.md) · [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [CHANGELOG.md](CHANGELOG.md)

**Назначение:** При любом code review или написании новой теории — сверяться с этим документом, чтобы не изобретать то, что уже задокументировано или уже не работает.

---

## Оценка совпадения теории и кода

| Область | Оценка | Комментарий |
|---------|--------|-------------|
| Теория (19 теорем, 7 источников) | **8.5 / 10** | Сильная, строгая математика |
| Реализация компонентов | **8.5 / 10** | *(было 8.0)* BentFunctions закрывает последний теор. пробел |
| Интеграция теория→код | **8.5 / 10** | *(было 7.5)* все 4 пункта реализованы, мост meta_q6↔geometry сомкнут |

> **Прогресс 2026-03-24 (сессия 2)**: Интеграция 6.5→7.5. Закрыты все оставшиеся «только теория» пункты:
> `Q6Arithmetic` (Теорема 3), `YiJingOps` + `YiJingV4Layer` (GERMES-нотация как операции).

---

## ЧАСТЬ 1: ТЕОРИЯ → КОД

### Что реализовано точно

| Концепция | Теорема / Документ | Реализация | Файл |
|-----------|-------------------|------------|------|
| XOR-изоморфизм для Z₂^n | Теорема 1 | `sign()` для {-1,+1}^6 | `geometry/core.py`, `quantizers.py` |
| Квантизатор-гомоморфизм | Теорема 2 | `YiJingQuantizer`: softmax (soft) / sign (hard) | `geometry/quantizers.py` |
| Расстояние Хэмминга | Теорема 4 | `hamming_distance()`, `hamming_distance_soft()` | `geometry/core.py`, `models/variant3.py` |
| D4-эквивариантность триграмм | Теорема 13 | `D4EquivariantLayer` (8 операций группы диэдра) | `geometry/equivariant.py` |
| Антиподальная симметрия | Теорема 9 | `AntipodalQuantizer`: 32 пары вместо 64 | `geometry/quantizers.py` |
| BianGua = XOR-маска | Теорема 6 | `BianGuaTransform`: покоординатное отражение | `geometry/equivariant.py` |
| Герман: P=2^k, 64 точки | Теоремы 8-10 | Архитектурное решение: 64 = 2^6 вместо E8(240) | Обоснование в `KNOWLEDGE_FRAMEWORK.md` |
| Фусюй и Вэнь-ван порядки | `PLAN-v51` (Склярова) | `fuxi_order()`, `wenwang_order()` | `geometry/core.py` |
| Тернарная квантизация {-1,0,+1} | `GERMES_NOTATION.md` (変爻) | `TernaryQuantizer` (3 режима) | `geometry/quantizers.py` |
| α-уровни (-4..+4) | `KNOWLEDGE_FRAMEWORK.md` | 5 фаз E2 (GlyphLevel→PhiloLevel) | `train_e2.py` |
| Matryoshka Q2→Q3→Q6 | `yijing-transformer-concept.md` | `MultiScaleGlobalRouter` | `models/hierarchical_moe.py` |

### Что реализовано частично (есть код, не полностью интегрировано)

> **Обновлено 2026-03-24**: большинство строк переведены в "реализовано точно" после gap→fix задач

| Концепция | Статус | Что осталось | Файлы |
|-----------|--------|--------------|-------|
| SOLAN-76 глифов (Теорема 15) | ✅ **Интегрирован** (2026-03-24) | `--glyph` флаг в `self_train_hmoe.py`; delta_margin=+0.33 >> 0.02 | `tokenizer/glyph_tokenizer.py`, `self_train_hmoe.py` |
| ArchetypalInterlingua | ✅ **Подключён** (2026-03-24) | `interlingua_use_fixed=True` по умолчанию в `model.py` | `geometry/interlingua_fixed.py`, `models/model.py` |
| Q4⊂Q6 вложение | 🟡 **Реализовано, не активно** | avg_hamming=2.56 > 2.5 → Q4 init не активирован | `e2_self_improve.py` |
| CrossDomainAnalogy | ✅ **36 пар** (2026-03-24) | Полная матрица 6×6, B→A реверсы добавлены | `self_train_hmoe.py` |
| E8-квантизатор | 🟡 Реализован, не подключён | Гиперкуб выбран как default без явного сравнения | `geometry/quantizers.py` |
| Temperature annealing | ✅ **Подключён** (2026-03-24) | `step_temp()` вызывается из `micro_train()`; T:1.0→0.01 за 5000 шагов | `geometry/quantizers.py`, `self_train_hmoe.py` |

### Что только в теории (кода нет)

| Концепция | Документ | Статус |
|-----------|----------|--------|
| **Walsh-Hadamard Transform** | Теорема 5, `e8-yijing-deep-analysis.md` | ✅ **Реализован** (2026-03-24): `WHT_Quantizer` в `quantizers.py` |
| **Модулярная арифметика Q6** | Теорема 3, `KNOWLEDGE_FRAMEWORK.md` | ✅ **Реализована** (2026-03-24): `Q6Arithmetic`, `Q6ArithmeticLayer` в `geometry/q6_algebra.py` |
| **Polarity reversal (YiJing) как операция** | `GERMES_NOTATION.md` | ✅ **Реализована** (2026-03-24): `YiJingOps` (V₄ группа Клейна) + `YiJingV4Layer` в `geometry/q6_algebra.py` |
| **Bent-функции как seed** | `e8-yijing-deep-analysis.md` | ✅ **Реализованы** (2026-03-24): `BentFunctions`, `BentPrototypeQuantizer` в `geometry/q6_algebra.py` |

> **2026-03-24 (сессия 3)**: Все 4 пункта закрыты. BentFunctions (геометрические seed) сомкнуты с meta_q6.bent_seed_texts() (RAG seed). Пробелов теория→код не осталось.

---

## ЧАСТЬ 2: КОД → ТЕОРИЯ

Компоненты, которые есть в коде, но **не имеют теоретического обоснования** в документах.

| Компонент | Файл | Что делает | Есть ли теория |
|-----------|------|-----------|----------------|
| `HierarchicalQuantizer` (Product Quantization) | `quantizers.py` | PQ-подход к Q6 | ⚠️ Только упоминание |
| `DeformableQuantizer` (обучаемая дельта) | `quantizers.py` | Обучаемое смещение кодбука | ⚠️ Идея в `e8-yijing-deep-analysis.md`, формулы нет |
| `GumbelQuantizer` + commitment_loss | `quantizers.py` | VQ-VAE подход | ❌ Нет теоретической связи с Q6 |
| `GroupedQuantizer` (per-channel) | `quantizers.py` | Стандартная per-channel квантизация | ❌ Не связана с Q6 концептуально |
| `hamming_lambda` (структурный штраф) | конфигурация | Штраф за удалённость в Q6 | ⚠️ Нет формулы, только параметр |
| `NautilusMoME` (6 реальных доменов) | `models/nautilus_yijing.py` | MoE с реальными доменами | ⚠️ `TURNING_POINT_ANALYSIS.md` как параллельный трек |
| `gate` инициализация на 0 (sigmoid=0.5) | большинство модулей | Нейтральный старт | ⚠️ Нет обоснования начального значения |

---

## ЧАСТЬ 3: КРИТИЧЕСКИЕ РАСХОЖДЕНИЯ

### 1. Walsh-Hadamard Transform — не встроена

**Теория (`e8-yijing-deep-analysis.md`):**
> WHT даёт O(n log n) спектральное разложение для Z₂^n. Для Q6 (n=64): 64×6 операций вместо 64×64. Это основа эффективного квантизатора.

**Код:** Есть только в `scripts/theoretical_analysis.py` как демонстрация. Не встроена ни в один квантизатор.

**Последствие:** Квантизаторы работают O(64d) вместо O(6 log 6 × d).

**Статус:** Не исправлено. Задача в `FUTURE_TASKS.md` (ПРИОРИТЕТ 2).

---

### 2. Temperature не достигает T→0

**Теория (`theoretical-foundations.md`):**
> При T→0 soft-квантизатор вырождается в жёсткий: веса концентрируются на ближайшей вершине гиперкуба. Только тогда модель реально «думает гексаграммами».

**Код (реальные измерения):**
- `TernaryQuantizer`: T начинается с 1.0, за 800 шагов достигает T≈0.689 (цель: 0.1)
- `interlingua_fixed.py`: `start_temp=1.0`, `end_temp=0.05` за `warmup_steps=3000` — правильно настроен

**Последствие:** Триты остаются мягкими. Архетипы не становятся «чёткими».

**Что делать:** Увеличить `warmup_steps` или снизить `start_temp`, или обучать дольше 800 шагов.

---

### 3. TURNING POINT v59 — синтез превратился в одноточечное решение

**Теория (`PLAN-v51`, `theoretical-foundations.md`):**
7 источников как равноправные вклады в архитектуру:

| Источник | Вклад |
|---------|-------|
| Склярова | PalaceAttention, архетипный кодбук |
| Фомюк | DualEmbedding, D4Equivariance |
| Андреев | Curriculum ordering, триангулярные числа |
| Касаткин | 3D-куб, диофантово вложение |
| Герман | Обоснование 64 = 2^6 |
| Беляев | HeisenbergAttention, FlowerGAT |
| SOLAN | GlyphTokenizer, контрастивное обучение |

**Код (после v59 ablation):**
- Беляев: **+0.75 PPL** — единственный критически работающий источник
- Фомюк, Андреев, Касаткин, Склярова: **−0.25..−0.28 PPL** — объявлены вредными
- SOLAN: нейтральный, но не интегрирован

**Задокументировано в:** `TURNING_POINT_ANALYSIS.md`, `REPORT-v53-v59.md`

**Что это значит:** Проект фактически свёлся к двум компонентам Беляева (HeisenbergAttention + FlowerGAT). Остальная теория существует на бумаге, но не проверена на реальных данных при длительном обучении (> 800 шагов).

---

### 4. ArchetypalInterlingua — коллапс, частично исправленный

**Теория (`archetypal-interlingua-theory.md`):**
> 64 архетипа Q6 = посредники между геометрией и языком (аналог языка Аймара как идеального посредника). Каждый источник голосует тернарно за каждый архетип.

**Код v60-v61 (до исправления):**
- Один общий `trit_proj` → все источники давали одинаковые триты
- 64 архетипа неразличимы → gate≈0.5 → PPL=vanilla (2.92-2.94)

**Код после 2026-03-24:**
- `interlingua_fixed.py`: N отдельных `trit_proj`, diversity loss, Gumbel-Softmax
- Self-test: spread=0.25, gate=0.49 — **архетипы начали различаться**
- **Но**: не подключён в основную модель `model.py` (задача на следующую итерацию)

---

### 5. SOLAN-76 глифов — реализован, но изолирован

**Теория (`theoretical-foundations.md`, Теорема 15):**
> Каждый глиф = 6 сегментов = 6 бит = одна гексаграмма. Токенизация через SOLAN сразу несёт Q6-геометрию.

**Код:**
- `GlyphTokenizer` полностью реализован
- `experiments/train_with_glyph.py` создан для сравнения
- **Основной обучающий цикл использует CharTokenizer (256 токенов без геометрии)**

**Что нужно:** Запустить `train_with_glyph.py`, получить delta > +0.02, затем переключить основной цикл.

---

### 6. GERMES-нотация — только документ

**Документ (`GERMES_NOTATION.md`):**
> {+, <, −} = {сила, переход, инь} — тернарный алфавит. Программа на GERMES = последовательность гексаграмм. Операции: поворот, отражение, перекодирование.

**Код:** Нигде не реализовано. `TernaryQuantizer` реализует {-1,0,+1} независимо, без ссылки на GERMES.

**Что нужно для полной интеграции:** Добавить GERMES-интерпретатор как decoder: `hex_sequence → GERMES_program → action`.

---

## ЧАСТЬ 4: ЧТО ТОЧНО РАБОТАЕТ (верифицированные результаты)

| Компонент | Результат | Верификация |
|-----------|----------|-------------|
| AbrialeBridgeMediator (v59) | PPL **1.24** — лучший | `bench_all_results.json` |
| BridgeOfModules (v58) | PPL **1.35** | `bench_all_results.json` |
| HeisenbergAttention (Беляев) | **+0.75 PPL** над baseline | ablation в `REPORT-v53-v59.md` |
| FlowerOfLifeGAT (Беляев) | КРИТИЧЕН solo | ablation в `REPORT-v53-v59.md` |
| Figure-8 + TSP pipeline | LCI=2.983, score=0.998 | `pipeline_runs/` |
| Metropolis annealing (meta_q6) | weak: +0.080 LCI | `FUTURE_TASKS.md` 2026-03-22 |
| Bent-функции vs случайные seeds | cosine=0.309 vs 0.463 (33% разнообразнее) | `FUTURE_TASKS.md` 2026-03-22 |
| D4-эквивариантность | Тест проходит | `tests/test_geometry_pytest.py` |
| Антиподальная симметрия (32 пары) | Тест проходит | `tests/test_geometry_pytest.py` |
| kasatkin_router confidence | 0.44 >> 0.15 | `experiments/kasatkin_router_log.json` |
| interlingua_fixed spread | 0.25 >> 0.03 | smoke-test 2026-03-24 |

---

## ЧАСТЬ 5: ДОРОЖНАЯ КАРТА gap→fix

Приоритизированные задачи для закрытия разрывов:

| Приоритет | Разрыв | Действие | Сложность |
|-----------|--------|---------|-----------|
| 🔴 1 | `interlingua_fixed.py` не подключён | В `model.py` заменить `ArchetypalInterlingua` на `ArchetypalInterlinguaFixed` | 30 мин |
| 🔴 2 | Temperature не достигает T=0 | Увеличить `warmup_steps` до 5000 или уменьшить `end_temp` до 0.01 | 15 мин |
| 🔴 3 | GlyphTokenizer не используется | Запустить `train_with_glyph.py --steps 1000`, затем интегрировать | 2 ч |
| 🟡 4 | Walsh-Hadamard не встроена | Реализовать `WHT_Quantizer` как альтернативу `E8Quantizer` | 3 ч |
| 🟡 5 | CrossDomainAnalogy 15→36 | Дополнить `self_train_hmoe.py` нижней треугольной (B→A) | 1 ч |
| 🟡 6 | Q4⊂Q6 не используется при обучении | После validate_q4_q6 (avg_hamming<2.5): инициализировать RAG из Q4-кластеров | 2 ч |
| 🟢 7 | GERMES-нотация | Декодер `hex_sequence → GERMES_program` | 4 ч |
| 🟢 8 | 5 источников не протестированы долго | Benchmark > 3000 шагов для Скляровой, Фомюк и др. | долго |

---

## Формат перекрёстных ссылок

При обнаружении нового расхождения — добавлять сюда строку в Часть 3, и одновременно:
1. Создавать запись в `FUTURE_TASKS.md` с приоритетом
2. Обновлять `CHANGELOG.md` при исправлении

**Связанные теоретические документы:**

| Документ | Что описывает |
|----------|--------------|
| `theoretical-foundations.md` | 19 теорем |
| `PSEUDORAG_YIJING_BRIDGE.md` | Q4⊂Q6 план интеграции |
| `archetypal-interlingua-theory.md` | ArchetypalInterlingua теория |
| `CONCEPTUAL_STAGE.md` | 7 источников, Вариант 3 |
| `KNOWLEDGE_FRAMEWORK.md` | α-уровни, 4 уровня системы знаний |
| `GERMES_NOTATION.md` | Тернарный алфавит GERMES |
| `e8-yijing-deep-analysis.md` | E8 vs гиперкуб, WHT |
| `yijing-transformer-concept.md` | Общая концепция трансформера |
| `TURNING_POINT_ANALYSIS.md` | Перелом в v59 (почему 5/7 «вредны») |
| `META-PRO2-BRIDGE.md` | 7 точек переноса из meta в pro2 |
| `PORTAL-PROTOCOL.md` | Протокол Nautilus Portal |
| `PASSPORT.md` | Само-описание проекта |

---

*Последнее обновление: 2026-03-24 | Аудит: 12 теоретических документов + 50+ файлов кода*
