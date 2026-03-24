# Статус реализации: Код vs Теория

Аудит проведён 2026-03-24. Проверено каждое утверждение из документации
против реального кода.

---

## Сводная таблица

| Компонент | В коде | Тесты | В бою | Статус |
|-----------|--------|-------|-------|--------|
| **МОДЕЛИ** | | | | |
| YiJingGPT (model.py) | ✅ forward() работает | ✅ 1049 тестов | ✅ чекпоинты | **Реализован** |
| Variant3GPT (variant3.py) | ✅ forward() работает | ✅ 80+ тестов | ✅ чекпоинты | **Реализован** |
| HierarchicalMoE (hierarchical_moe.py) | ✅ forward() работает | ✅ 40+ тестов | ✅ чекпоинты | **Реализован** |
| HierarchicalE2 (hierarchical_e2.py) | ✅ forward() работает | ✅ тесты | ✅ чекпоинты | **Реализован** |
| NautilusYiJing (nautilus_yijing.py) | ✅ forward() работает | ✅ 25 тестов | ⚠️ нет чекпоинта | **Реализован** |
| VanillaGPT (baseline.py) | ✅ forward() работает | ✅ тесты | ✅ baseline | **Реализован** |
| | | | | |
| **ГЕОМЕТРИЯ** | | | | |
| 12 квантизаторов (quantizers.py) | ✅ все 12 с forward() | ✅ тесты | ✅ используются | **Реализованы** |
| 16 паттернов внимания (attention.py) | ✅ все 16 с forward() | ✅ тесты | ⚠️ опционально | **Реализованы** |
| Маршрутизация/мосты (routing.py) | ✅ 13+ классов | ✅ тесты | ✅ в моделях | **Реализованы** |
| RoPE, ALiBi (positional.py) | ✅ реальный код | ✅ тесты | ✅ используются | **Реализованы** |
| SwiGLU, TrigramMoE (ffn.py) | ✅ реальный код | ✅ тесты | ✅ используются | **Реализованы** |
| BianGuaTransform, D4 (equivariant.py) | ✅ реальный код | ✅ тесты | ✅ в Variant3 | **Реализованы** |
| NautilusChamber/Hierarchy (nautilus.py) | ✅ реальный код | ✅ тесты | ✅ в model.py | **Реализованы** |
| GlyphComposer, Convergence (convergence.py) | ✅ реальный код | ✅ тесты | ⚠️ опционально | **Реализованы** |
| AbrialeLayer (abriale.py) | ✅ реальный код | ✅ 24 теста | ✅ PPL 1.24 | **Реализован** |
| generate_hexagrams/trigrams (core.py) | ✅ реальный код | ✅ тесты | ✅ везде | **Реализованы** |
| | | | | |
| **ОБУЧЕНИЕ** | | | | |
| Главный тренировочный цикл (train.py) | ✅ 644 строки | — | ✅ AMP, WandB, checkpoint | **Реализован** |
| self_train.py (3 стадии) | ✅ 1065 строк, все 3 стадии | — | ✅ запускается | **Реализован** |
| bidir_train.py (бидирекционное) | ✅ 1040 строк, 2 потока | — | ✅ запускается | **Реализован** |
| train_hmoe_curriculum.py (5 фаз) | ✅ 629 строк, все фазы | — | ✅ запускается | **Реализован** |
| TrainingBridge (bridge.py) | ✅ интеграция 52 utils | — | ✅ в train.py | **Реализован** |
| | | | | |
| **ИНФЕРЕНС** | | | | |
| generate() (generate.py) | ✅ top-k/top-p/repetition | — | ✅ работает | **Реализован** |
| AdvancedGenerator (5 стратегий) | ✅ greedy/nucleus/beam/spec/dyn | — | ✅ работает | **Реализован** |
| e2_inference.py (embed/similar/map) | ✅ 451 строка, CLI | — | ✅ запускается | **Реализован** |
| | | | | |
| **УТИЛИТЫ** | | | | |
| LoRA (lora.py) | ✅ apply/merge/save/load | ✅ тесты | — | **Реализован** |
| Speculative decoding (speculative.py) | ✅ draft + verify | ✅ тесты | — | **Реализован** |
| Export ONNX/TorchScript (export.py) | ✅ реальный код | ✅ тесты | — | **Реализован** |
| CharTokenizer (char_tokenizer.py) | ✅ encode/decode | ✅ тесты | ✅ в train.py | **Реализован** |
| TextDataset (text_dataset.py) | ✅ from_file/split/batch | ✅ тесты | ✅ в train.py | **Реализован** |
| WikiTextDataset (wikitext_dataset.py) | ✅ from_huggingface() | — | ⚠️ fallback на синт. | **Реализован, редко используется** |

---

## Что реализовано ТОЛЬКО в теории

| Утверждение из документации | Где заявлено | Что в коде | Вердикт |
|----------------------------|-------------|------------|---------|
| **«19 доказанных теорем»** | KNOWLEDGE_FRAMEWORK.md | Описаны 4 из 19. Остальные 15 — галочки без формулировок | ❌ **Не доказаны в коде** |
| **PseudoRAG** | CONCEPTUAL_STAGE.md | Нет класса PseudoRAG. Упоминается как концепт в validate_q4_q6.py | ❌ **Не реализован** |
| **Multi-GPU / DDP обучение** | — | 0 результатов по DistributedDataParallel/DDP. Есть только device safety (.to(device)) | ❌ **Не реализовано** |
| **7-портовая архитектура** | PORTAL-PROTOCOL.md | portal.py имеет 3–5 адаптеров, не 7 портов | ⚠️ **Частично (3/7)** |
| **Интеграция 6 источников в единый модуль** | PLAN-v51 | Компоненты каждого источника существуют по отдельности, но единого модуля интеграции нет | ⚠️ **Разрозненно** |
| **SOLAN как система внимания** | CONCEPTUAL_STAGE.md | GlyphTokenizer с SOLAN-76 существует, но как токенизатор. Не как attention mechanism. Прирост <1% (PPL 2.32→2.31) | ⚠️ **Токенизатор есть, attention нет** |
| **Бенчмарки на больших данных** | README (PPL 1.24) | Все 19 JSON-логов — на синтетических/маленьких корпусах (50–100K токенов, d=128, 800 шагов). Не WikiText-103 с d≥256 | ⚠️ **Только toy-scale** |

---

## Что реализовано в коде, но НЕ используется ни одной моделью

Эти модули существуют, протестированы, но не вызываются в forward() ни одной из 6 моделей:

| Модуль | Файл | Тесты | Используется в forward() |
|--------|------|-------|-------------------------|
| DifferentialAttention | models/diff_attn.py | ✅ 12 тестов | ❌ только в тестах |
| PrefixTuning, LogitLens | models/prefix_tuning.py | ✅ 15 тестов | ❌ только в тестах |
| DistillationTrainer | training/distillation.py | ✅ 4 теста | ❌ только в тестах |
| EMA (Exponential Moving Average) | training/ema.py | ✅ 1 тест | ❌ только в тестах |
| ExpertChoice routing | models/expert_choice.py | — | ❌ только в knowledge_system.py |
| Extensions A2–E14 (9 классов) | models/extensions.py | ✅ через run_all_extensions.py | ❌ не в основных моделях |
| 8 из 16 attention паттернов | geometry/attention.py | ✅ тесты | ⚠️ условно через конфиг, не по умолчанию |

> **Примечание:** все 43 файла `utils_v*.py` импортируются через bridge-модули (`bridge_schedulers.py`, `bridge_optimizers.py`, `bridge_regularization.py` и др.) — мёртвого кода среди них нет.

---

## Что реализовано в коде И реально работает

### Ядро (подтверждено тестами + чекпоинтами)

**6 моделей** — все имеют working forward(), обучаются, генерируют текст:

1. **YiJingGPT** — гибрид: геометрия подключается через конфиг-флаги. Без флагов = vanilla + квантизация
2. **Variant3GPT** — геометрия обязательна: HexagramProjection → BianGuaAttention → TernaryGate → Interlingua. Всё выполняется в каждом forward()
3. **HierarchicalMoE** — 4-уровневая иерархия Q2→Q3→Q6 реальна. 6 MicroExperts + BidirBridge + MultiScaleGlobalRouter — всё wired
4. **HierarchicalE2** — самый полный: 5 α-уровней (Glyph→Core→Method→Theory→Philo), staged training с freeze/unfreeze
5. **NautilusYiJing** — Q6GeometricRouter + YiJingCoreBlock + MicroExperts + SYNTH (entropy-triggered)
6. **VanillaGPT** — чистый baseline без геометрии

### Геометрическая система (64 модуля, все с forward())

- **12 квантизаторов**: от простого YiJingQuantizer до MatryoshkaQuantizer — все с STE, commitment loss, temperature
- **16 attention паттернов**: Palace, Heisenberg, FlowerOfLife, Möbius и др. — все compute real bias matrices
- **13 routing модулей**: GatedPathSelector → AbrialeBridgeMediator (PPL 1.24 ★) → BridgedInterlingua
- **Nautilus**: 7 камер с прогрессивным включением (NautilusScheduler)
- **Convergence**: GlyphComposer (иерархическая композиция) + TokenAbstractor (soft k-means на 64 кластера)

### Обучение (4 пайплайна, все запускаются)

| Пайплайн | Что делает | Данные |
|----------|-----------|--------|
| `train.py` | Cosine LR + warmup + AMP + WandB + checkpoint resume | synthetic / TinyStories / svend4 |
| `self_train.py` | Stage 0: Q6 topology → Stage 1: RAG-buffer → Stage 2: filtered wild | corpus + synthetic |
| `bidir_train.py` | Forward (KnowledgeGraph→PageRank→train) ↔ Backward (generate→filter→update graph) | KnowledgeGraph |
| `train_hmoe_curriculum.py` | 5 фаз: CONCRETE → DYNAMIC → ABSTRACT → Router → Fine-tune | RepoCorpus + synthetic |

### Бенчмарки (19 JSON-логов, все реальные)

| Конфигурация | PPL | Параметры | Данные |
|-------------|-----|-----------|--------|
| vanilla baseline | 2.94 | 839K | synthetic |
| BridgeOfModules v58 | 1.35 | 1.4M | synthetic |
| **AbrialeBridgeMediator v59** | **1.24** | 1.9M | synthetic |
| interlingua_fixed v62 | 1.17 | ~2M | WikiText (small) |
| kasatkin_router v63 | 1.04 | ~2M | WikiText (small) |
| **nautilus/adamw_wsd v69** | **1.01** | ~2M | synthetic |

> **Важно:** все бенчмарки проведены на d_model=128, 800–1000 шагов, <100K токенов. Это валидные сравнения между конфигурациями, но не демонстрация production quality.

### Тесты (1731 pass, 0 fail)

| Файл тестов | Кол-во | Что тестирует |
|-------------|--------|---------------|
| test_model_pytest.py | ~1049 | Вся core-архитектура, training, v4–v51 фичи |
| test_variant3.py + extended | ~120 | Variant3: shape, gradient, Q6 properties, convergence |
| test_geometry_pytest.py | ~100 | Все 12 квантизаторов, паттерны, BianGua |
| test_hierarchical_moe.py | ~40 | HMoE: routing, experts, load balance, stages |
| test_nautilus_yijing.py | 25 | Router, experts, bridge, analogy, full model |
| test_abriale.py | 24 | AbrialeLayer, scatter, topk |
| test_multigpu_precision.py | 20 | fp16/bf16 forward/backward, device transfer |
| test_training_utils.py | ~30 | Sophia, DynamicTemp, GradScaler |

**Тип тестов:** unit + component integration. Проверяют shapes, gradients, math constraints, loss decrease. **НЕ проверяют** качество генерации или generalisation на реальных данных.

---

## Чего НЕТ и что нужно для production

| Что отсутствует | Приоритет | Сложность |
|----------------|-----------|-----------|
| Multi-GPU / DDP обучение | Высокий | Средняя |
| Бенчмарки на WikiText-103/OpenWebText (d≥256, >10K шагов) | Высокий | Средняя |
| Тесты качества генерации (BLEU, perplexity на held-out) | Высокий | Низкая |
| Единый модуль интеграции 6 источников | Средний | Высокая |
| PseudoRAG как реализованный класс | Средний | Средняя |
| Полные формулировки 19 теорем | Низкий | Низкая |
| SOLAN как attention mechanism (не только tokenizer) | Низкий | Средняя |
| 7-портовая архитектура (сейчас 3–5) | Низкий | Средняя |

---

## Итого

**Реализовано в коде:** 6 моделей, 64 геометрических модуля, 4 тренировочных пайплайна, 5 стратегий генерации, 12 квантизаторов, LoRA, speculative decoding, ONNX export — всё с working forward() и 1731 тестом.

**Только теория:** PseudoRAG, 15 из 19 теорем, Multi-GPU/DDP, 7-портовый протокол, единая интеграция 6 источников, SOLAN-attention.

**Главный разрыв:** код работает на toy-scale (d=128, 800 шагов, <100K токенов). Масштабирование на реальные данные и размеры не проверено.
