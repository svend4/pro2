# YiJing-Transformer: Текущая стадия реализации

**Дата:** 2026-03-24
**Версия:** v61 (BridgedInterlingua)
**Ветка:** `claude/current-dev-stage-50w6E`

---

## 1. Общий статус проекта

YiJing-Transformer — трансформерная архитектура с индуктивным смещением на основе геометрии гиперкуба Q6 = {-1,+1}⁶. 64 гексаграммы И-Цзин = вершины 6-мерного гиперкуба → дискретная структура для кодбука, attention-масок и квантизации.

**Пройдено 13 итераций** (v49–v61), **9 PR смержено**, **12 бенчмарков** проведено.

---

## 2. Хронология версий

| Версия | Содержание | Статус |
|--------|------------|--------|
| v49–v50 | Базовая архитектура YiJingGPT, 3 режима, A/B тест | ✅ Готово |
| v51–v52 | Интеграция 7 источников (32 подступени, 41 тест, пакет `geometry/`) | ✅ Готово |
| v53 | Первый бенчмарк (WikiText, S-box, pairwise) | ✅ Готово |
| v54 | Anti-interference routing, Kasatkin 3D embedding | ✅ Готово |
| v55 | Convergence Bridge (GlyphComposer + TokenAbstractor) | ✅ Готово |
| v56 | TernaryQuantizer {-1,0,+1}⁶ + MatrixGrammar | ✅ Готово |
| v57 | AbrialeLayer (событийные N-местные связи) | ✅ Готово |
| v58 | BridgeOfModules — иерархическая cross-attention медиация | ✅ Готово |
| v59 | Финальный бенчмарк (5 архитектур + 7 ablation = 12 конфигов) | ✅ Готово |
| v60 | ArchetypalInterlingua — hub-and-spoke посредник | ✅ Готово, не работает |
| **v61** | **BridgedInterlingua — Module→Bridge→Archetype→Core** | **✅ Готово, частичный успех** |

---

## 3. Семь математических источников

| # | Автор | Модуль | Файл | Полезность |
|---|-------|--------|------|------------|
| 1 | Склярова | Palace attention (8 дворцов × 8 гексаграмм) | `geometry/skliarova.py` | ❌ Вреден (PPL +0.28) |
| 2 | Фомюк | D4Equivariant (антиподальная симметрия) | `geometry/fomyuk.py` | ❌ Вреден (PPL +0.27) |
| 3 | Андреев | PrivilegedAxis (треугольная матрица) | `geometry/kasatkin.py` | ❌ Вреден (PPL +0.27) |
| 4 | Касаткин | CubeDiagonal + DualEmbedding | `geometry/kasatkin.py` | ❌ Вреден (PPL +0.25–0.27) |
| 5 | Герман | Теория упаковок P=2^k | Теоретический | — Не реализован |
| 6 | Беляев | **Heisenberg attention + FlowerGAT** | `geometry/belyaev.py` | **✅ Критичен** |
| 7 | SOLAN | 76-глифный алфавит, Q6 Glyph Ecosystem | `tokenizer/glyph_tokenizer.py` | ⚠️ Не интегрирован |

**Вывод:** из 7 источников реально работают только модули Беляева. Остальные 5 добавляют параметры и шум.

---

## 4. Результаты бенчмарков

### 4.1. Сравнение архитектур (v59, синтетический WikiText, 800 шагов, d=128)

| Архитектура | Params | PPL ↓ | Описание |
|-------------|--------|-------|----------|
| vanilla (baseline) | 839K | 2.94 | Стандартный трансформер |
| specializer | 1.2M | 2.17 | SourceSpecializer: маршрутизация по доменам |
| adaptive_bridge | 1.6M | 1.63 | AdaptiveBridge: динамическая сложность |
| seven_bridge | 1.4M | 1.35 | BridgeOfModules: иерархическая медиация |
| **abriale_bridge** | **1.9M** | **1.24** | **AbrialeLayer + Bridge — лучший результат** |

### 4.2. Ablation: важность источников (v59)

| Убранный источник | PPL без него | Δ PPL | Вердикт |
|-------------------|-------------|-------|---------|
| Palace (Склярова) | 1.07 | -0.28 | Убрать — станет лучше |
| D4Equivariant (Фомюк) | 1.08 | -0.27 | Убрать — станет лучше |
| PrivilegedAxis (Касаткин) | 1.09 | -0.27 | Убрать — станет лучше |
| DualEmbedding (Касаткин) | 1.09 | -0.27 | Убрать — станет лучше |
| CubeDiagonal (Касаткин) | 1.10 | -0.25 | Убрать — станет лучше |
| **FlowerGAT (Беляев)** | **1.95** | **+0.59** | **Оставить — критичен** |
| **Heisenberg (Беляев)** | **2.11** | **+0.75** | **Оставить — критичен** |

### 4.3. Interlingua эксперименты (v60–v61)

| Конфигурация | Params | PPL | Проблема |
|-------------|--------|-----|----------|
| v60_interlingua | 2.1M | 2.94 | Архетипы не активируются (STE zero-gradient trap) |
| bridged_lightweight (v61) | 2.2M | 2.92 | 64 архетипа активны после фикса, но PPL ≈ baseline |
| bridged_full (v61) | 4.1M | 2.93 | Архетипы не активируются (0 active) |
| bridged_no_ternary (v61) | 2.2M | 2.93 | Без тернарной квантизации — тот же результат |
| bridged_high_uncertainty (v61) | 2.2M | 2.93 | Повышенный uncertainty budget — без эффекта |

**Диагноз v60–v61:** Interlingua-слой учится слишком медленно (gate ≈ 0.49, scale ≈ 0.07). Модель игнорирует interlingua-ветку и обучается через основной путь.

---

## 5. Архитектурные компоненты — сводка

### Работает и используется

| Компонент | Файл | Роль |
|-----------|------|------|
| Q6 кодбук {-1,+1}⁶ | `geometry/core.py` | Дискретный кодбук на 64 вершины |
| Факторизованный квантизатор 8×8 | `geometry/core.py` | Быстрая квантизация через триграммы |
| Триграммный attention bias | `model.py` | Индуктивное смещение на основе И-Цзин |
| BianGua (变卦) трансформация | `model.py` | Геометрическая трансформация гексаграмм |
| Heisenberg attention | `geometry/belyaev.py` | Рычажные весы → attention |
| FlowerGAT | `geometry/belyaev.py` | Граф-внимание на основе «цветочной» топологии |
| BridgeOfModules | `geometry/bridge.py` | Иерархическая медиация между источниками |
| AbrialeLayer | `geometry/abriale.py` | Событийные изотропные N-местные связи |

### Реализовано, но не помогает

| Компонент | Файл | Проблема |
|-----------|------|----------|
| Palace attention | `geometry/skliarova.py` | Ухудшает PPL |
| D4Equivariant | `geometry/fomyuk.py` | Ухудшает PPL |
| PrivilegedAxis / CubeDiagonal / DualEmbedding | `geometry/kasatkin.py` | Ухудшает PPL |
| ArchetypalInterlingua | `geometry/core.py` | STE gradient trap |
| BridgedInterlingua | `geometry/bridge.py` | Игнорируется моделью |

### Реализовано, но не интегрировано в training loop

| Компонент | Файл | Статус |
|-----------|------|--------|
| GlyphTokenizer (SOLAN-76) | `tokenizer/glyph_tokenizer.py` | Standalone, encode/decode работает |
| GlyphComposer (4 уровня) | `geometry/convergence.py` | Реализован, не в пайплайне |
| TokenAbstractor (64 архетипа) | `geometry/convergence.py` | Реализован, не в пайплайне |
| TernaryQuantizer | `geometry/quantizers.py` | Реализован, STE проблемы |
| MatrixGrammar | `geometry/quantizers.py` | Реализован, не тестировался в бенчмарке |

---

## 6. Нереализованные компоненты

| Компонент | Описание | Приоритет |
|-----------|----------|-----------|
| Bitmap-визуализация (8×8 grid) | Рендеринг глифов SOLAN | Низкий |
| Шрифты font3/font4 | Визуальные варианты глифов (сейчас — random bit-flip) | Низкий |
| 39-Domain Benchmark | Мультизадачная валидация Q6 экосистемы | Средний |
| Интеграция GlyphTokenizer | Замена CharTokenizer на GlyphTokenizer в обучении | Высокий |
| Lean Model (без вредных источников) | Heisenberg + FlowerGAT + Abriale + Bridge only | Высокий |
| Валидация на полном WikiText-103 | Проверка масштабирования за пределами 800 шагов | Высокий |

---

## 7. Рекомендуемые следующие шаги

### Приоритет 1: Lean Model
Убрать 5 вредных источников (Склярова, Фомюк, Касаткин×3). Оставить Heisenberg + FlowerGAT + AbrialeLayer + BridgeOfModules. Ожидаемый PPL < 1.07 при ~1M параметров.

### Приоритет 2: Валидация на реальных данных
Все бенчмарки — на синтетическом WikiText (800 шагов, d=128). Необходим переход на WikiText-103 или OpenWebText для подтверждения результатов.

### Приоритет 3: Решение по Interlingua
Два варианта:
- **a)** Заменить STE на Gumbel-Softmax или relaxed quantization для тернарных кодов
- **b)** Отказаться от Interlingua в пользу чистого BridgeOfModules (который уже работает)

### Приоритет 4: Интеграция SOLAN
Подключить GlyphTokenizer к основному training loop. Проверить, даёт ли визуальный изоморфизм Q6 преимущество над CharTokenizer.

### Приоритет 5: Масштабирование
Тесты при d=256, d=512 и на более длинных последовательностях для проверки устойчивости архитектурного преимущества.

---

*Документ сгенерирован на основе результатов бенчмарков v53–v61 и git-истории проекта.*
