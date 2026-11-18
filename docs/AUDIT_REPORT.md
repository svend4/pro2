# Технический аудит кодовой базы `svend4/pro2` (Wave 2)

**Дата аудита:** 2026-11-18  
**Базовая ветка:** `main` (`d6cdc75e0eb0905ea9344b2c3f8b432b92857563`, коммит `Add GNU Affero General Public License v3` от 2026-11-02)  
**Ветка аудита:** `audit/code-review` (создана от `main`)  
**Объект сверки:** коммиты `main` после PR #85, ветка `claude/current-dev-stage-50w6E` (`bda27ab0`) и 4 неслитых коммита на `claude/current-dev-stage-ulcwt`

---

## 1. Резюме аудита

1. **Состояние `main` и сверка веток:** после PR #85 (`924339e9`, 2026-04-19) на `main` вошли только PR #86 (`0f897158`, изменения исключительно в `nautilus/*`) и коммит `LICENSE` (`d6cdc75e`, GNU AGPL v3). На ветке `claude/current-dev-stage-ulcwt` остались **4 неслитых коммита** от 2026-10-09 (`c4b89752`, `73bdf8ad`, `80c7cbdb`, `aba5ec00`), затрагивающие 10 текстовых/Python/JSON/log файлов. **Файловое пересечение с post-PR#85 коммитами на `main`, `LICENSE` и `CURRENT-STAGE.md` равно нулю.**
2. **Выполненный перенос на `audit/code-review`:** все 10 непротиворечащих файлов из 4 неслитых коммитов `ulcwt` чисто перенесены на `audit/code-review` с полным сохранением PR #86 (`nautilus/*`), `LICENSE` и `CURRENT-STAGE.md` без изменений. В числе перенесённого — критическое исправление вычисления balance loss в `GrassmannianRouter` / `Q6GrassmannianRouter` (ранее шло под `torch.no_grad()` без градиента) и исправление смещения `+4` при переносе byte-level весов (0..255) в BPE byte-fallback токены (`4..259`).
3. **Научная и архитектурная честность (данные `ulcwt`):** контролируемая абляция `scripts/ablation_grmoe.py` (d=128, 3000 шагов, val=2048 seq) показала, что после починки градиента balance loss нагрузка `Q6GrassmannianRouter` выровнялась (`0.053–0.350` → `0.129–0.210`), но **линейный top-2 роутер остаётся лучше** (`val PPL = 90.6`, нагрузка `0.160–0.175` против `99.2` у `grmoe` и `96.5` у `grmoe_lat`). Перенос словаря `256 → 4096` затрагивает лишь `1.3%` токенов корпуса (и `0.00%` кириллицы).
4. **Инженерные и безопасные проблемы `main`:** отсутствует GitHub Actions CI (`.github/workflows/`), в корне закоммичены 11 бинарных файлов весов `hmoe_*.pt` (~209 МБ суммарно, несмотря на `*.pt` в `.gitignore`), дублируются файлы портала (`portal.py`, `nautilus.json`, `passports/` в корне и в `nautilus/`), в `nautilus/portal.py:render_html` присутствует Reflected XSS из-за отсутствия `html.escape()`, а метаданные `setup.py` (`MIT`) противоречат `LICENSE` (`AGPL-3.0`).

---

## 2. Сверка PR #85 / PR #86 vs `ulcwt` и `50w6E`

### 2.1. Коммиты на `main` (от PR #85 до `HEAD`)

| Коммит | Дата | Содержание | Изменённые файлы |
|---|---|---|---|
| `924339e9433ad22c3af035efcdc203062e29bca3` | 2026-04-19 | Merge PR #85 (`ulcwt`, ранняя часть) | `.gitignore`, `scripts/test_grand_orchestrator.py`, `scripts/train_all_models.py`, `scripts/train_full_corpus.py`, `yijing_transformer/models/geometry/__init__.py`, `yijing_transformer/models/geometry/advanced_routing.py` (`+1576 / -1`) |
| `0f897158e1a5f5dda9cedee158733324cf8258e4` | 2026-04-19 | Merge PR #86 (`claude/review-nautilus-project-AoYom`) | `nautilus/adapters/info1.py`, `nautilus/nautilus.json`, `nautilus/passports/{ai_agents,data2,data7,info1,infosystems,meta,pro2}.md`, `nautilus/portal.py`, `nautilus/requirements.txt` (`+366 / -9`) |
| `d6cdc75e0eb0905ea9344b2c3f8b432b92857563` | 2026-11-02 | Add GNU Affero General Public License v3 | `LICENSE` (`+661 / -0`) |

Проверка коммита `bda27ab0d671f3fbcc95ca9afb3b2fc5259acb06` (`claude/current-dev-stage-50w6E`, 2026-03-24, `+160 / -0`): добавляет `CURRENT-STAGE.md` (v61, `BridgedInterlingua`). Этот файл уже присутствует в `main` в идентичном состоянии (`SHA d2b4a464332e9860dbcf42a930fcbafc5b3e1f58`).

### 2.2. Четыре неслитых коммита на `claude/current-dev-stage-ulcwt`

| Коммит | Дата | Изменённые файлы | Статистика | Пересечение с post-PR#85 (`nautilus/*`, `LICENSE`, `CURRENT-STAGE.md`) |
|---|---|---|---|---|
| `c4b897527a20631dfe61d49a4db34480d0e53320` | 2026-10-09 | `docs/RESEARCH_REPORT.md` | `+428 / -0` | Нет |
| `73bdf8ad7f2e9e9bc68a6e644d9a42e7f4d79f67` | 2026-10-09 | `docs/RESEARCH_REPORT.md`, `scripts/test_grand_orchestrator.py`, `scripts/train_all_models.py`, `scripts/train_full_corpus.py` | `+47 / -20` | Нет |
| `80c7cbdb1b0481905835d788fafb3651e0c9961b` | 2026-10-09 | `docs/RESEARCH_REPORT.md`, `yijing_transformer/models/geometry/advanced_routing.py`, `scripts/ablation_grmoe.py`, `experiments/ablation_grmoe_v1.log`, `experiments/ablation_grmoe_v1_results.json` | `+776 / -26` | Нет |
| `aba5ec00909b7a47dec7b143d8223e2b837035d1` | 2026-10-09 | `docs/RESEARCH_REPORT.md`, `experiments/ablation_grmoe_v2.log`, `experiments/ablation_grmoe_v2_results.json` | `+256 / -3` | Нет |

### 2.3. Перенесённые изменения на `audit/code-review` и воспроизводимый `cherry-pick`

Так как конфликтов с `main` не выявлено и все 10 файлов являются читаемыми текстовыми/Python/JSON/log-файлами, они перенесены на `audit/code-review`:
- `docs/RESEARCH_REPORT.md` (539 строк);
- `scripts/test_grand_orchestrator.py`, `scripts/train_all_models.py`, `scripts/train_full_corpus.py` (смещение byte-fallback `new[4:4+v] = old`);
- `yijing_transformer/models/geometry/advanced_routing.py` (Switch-style balance loss с градиентом и итоговый расчёт EMA/balance в `Q6GrassmannianRouter`);
- `scripts/ablation_grmoe.py`, `experiments/ablation_grmoe_v1.{log,json}`, `experiments/ablation_grmoe_v2.{log,json}`.

Для сохранения исходной авторской истории коммитов в локальном git-клоне эквивалентная операция выполняется так:

```bash
git checkout -b audit/code-review main
git cherry-pick c4b897527a20631dfe61d49a4db34480d0e53320 \
                73bdf8ad7f2e9e9bc68a6e644d9a42e7f4d79f67 \
                80c7cbdb1b0481905835d788fafb3651e0c9961b \
                aba5ec00909b7a47dec7b143d8223e2b837035d1
```

---

## 3. Архитектурный обзор

### 3.1. `YiJingGPT` (`yijing_transformer/models/model.py`, 2198 строк)
- Основной трансформер с геометрическими смещениями Q6 (`{-1,+1}^6`), квантизацией (`FactoredYiJingQuantizer`, `HierarchicalQuantizer`, `TernaryQuantizer` и др.), поддержкой RoPE/ALiBi, `BianGuaTransform`, `BridgeOfModules`, `AbrialeLayer`, `NautilusHierarchy`, `SixSourceLayer`, `SOLANAttention`, `PseudoRAGProjection`, `ExpertChoiceRouter`, `DifferentialAttention`.
- **Технические наблюдения:**
  - Чрезмерная конфигурационная поверхность (более сотни bool-флагов, согласно `docs/FULL_AUDIT_REPORT.md` — 126 флагов), что вынуждает скрипты вроде `scripts/test_grand_orchestrator.py:198-277` и `scripts/train_full_corpus.py:110-136` вручную конструировать `SimpleNamespace Cfg` с 50+ полями.
  - В `yijing_transformer/models/model.py` присутствуют дублирующиеся импорты: `HexagramAttentionPattern` (строки 25 и 63) и `SixSourceLayer` (строка 78 из `.geometry` и строка 95 из `yijing_transformer.models.geometry.six_sources`).
  - Не все экспериментальные модули включены в основной `forward()` (см. также `docs/FULL_AUDIT_REPORT.md` §11).

### 3.2. `NautilusMoME` и `NautilusYiJing`
- `NautilusMoME` (`yijing_transformer/scripts/train_nautilus_mome.py`, 2946 строк): 4-слойный базовый трансформер на BPE (SentencePiece, `vocab=4096`) + 6 доменных микро-экспертов (`MATH`, `CODE`, `HUMAN`, `SYSTEM`, `RECON`, `INFO`) и `SYNTH` с гексаграммной картой `HEXAGRAM_MAP` (`q6_idx`: 63, 6, 0, 18, 45, 27, 36), `NautilusHierarchy` и модулем межэкспертных аналогий `CrossDomainAnalogy`.
- `NautilusYiJing` (`yijing_transformer/models/nautilus_yijing.py`, 767 строк): переносит схему MoME в пакет `models/` с тремя механизмами — `Q6GeometricRouter`, `YiJingCoreBlock` и `YiJingMicroExpert`.
- **Технические наблюдения:** в `nautilus_yijing.py:62-68` выполняется мутация `sys.path` на уровне импорта модуля и используется абсолютный импорт `from models.geometry...` вместо относительного `from .geometry...`, что создаёт риск двойной загрузки модулей (`models.*` vs `yijing_transformer.models.*`).

### 3.3. `HierarchicalMoE` (`HMoE`, `yijing_transformer/models/hierarchical_moe.py`, 1160 строк)
- 4-уровневая иерархия экспертов: `MultiScaleGlobalRouter` (Matryoshka `Q2 → Q3 → Q6`, выбор группы `ABSTRACT`/`DYNAMIC`/`CONCRETE`) → `GroupRouter` → 6 `MicroExpert` (`NOOS`, `COSMO`, `AERO`, `PYRO`, `GEO`, `HYDRO` с оптимизированными Q6-якорями `DOMAIN_Q6_IDX = {GEO:0, HYDRO:8, PYRO:21, AERO:19, COSMO:62, NOOS:63}`) → `BidirBridgeExpert` → векторизованный `Q6ExpertBank` (64 Q6-эксперта).
- В `scripts/test_grand_orchestrator.py:80-160` и `scripts/train_full_corpus.py:150-179` чекпоинт `hmoe_fixed_joint.pt` (`vocab=256`) монтируется внутрь `Variant3GPT` с ручным адаптером `ArchetypalInterlingua.forward` для совпадения порядка аргументов. При передаче же `HierarchicalMoEFFN` напрямую в `GrandOrchestrator` с `model_type='hmoe_component'` (`grand_orchestrator.py:135-139`) возвращаются нулевые логиты.

### 3.4. `GrandOrchestrator` (`yijing_transformer/models/grand_orchestrator.py`, 732 строки)
- Ансамблевый оркестратор для 7 разнородных моделей в трёх режимах (`blend`, `cascade`, `expert`) через `ModelWrapper`, `OrchestraRouter`, `SharedOrchestraMemory`, `CascadeConnector`.
- Проблемы в `grand_orchestrator.py`:
  - `ModelWrapper.forward` (строки 148–156): перехватывает `Exception` при запасном вызове `self.model(idx)` и молча возвращает нулевые логиты с `{'error': 'forward_failed'}` без логирования причины;
  - `OrchestraRouter.forward` (строка 228): маскирование через `-1e9` вместо `float('-inf')` или `masked_fill`, что небезопасно при обучении в FP16 (переполнение `float16`).

### 3.5. `advanced_routing.py` (GrMoE + SIPS + LatFormer)
- В перенесённой версии из `ulcwt` (`80c7cbdb`) исправлен дефект `GrassmannianRouter.forward`: в `main` (`cfc40f1d`) расчёт `expert_frac` и `aux_loss` опирался на тензор внутри `torch.no_grad()`, из-за чего балансировка нагрузки не получала градиентов. Теперь используется Switch-style loss `(expert_frac * probs.mean(dim=(0, 1))).sum() * n_experts`, а `Q6GrassmannianRouter` считает `aux_loss` и `expert_load_ema` по итоговому распределению `combined`.
- По данным контролируемой абляции (`experiments/ablation_grmoe_v1_results.json`, `ablation_grmoe_v2_results.json`) локальное упрощение GrMoE проигрывает стандартному линейному top-2 роутеру (val PPL `99.2` против `90.6`).

### 3.6. Портал `nautilus/` (PR #86)
- `nautilus/portal.py`, `nautilus/nautilus.json` (v1.1), `nautilus/adapters/*`, `nautilus/passports/*.md`, `nautilus/requirements.txt` реализуют stdlib-портал (Python 3.10+) поверх 7 адаптеров (`info1`, `pro2`, `meta`, `data2`, `data7`, `infosystems`, `ai_agents`) с поддержкой `GITHUB_TOKEN` в `nautilus/adapters/info1.py`.

---

## 4. CI, тесты и документация

1. **Отсутствие GitHub Actions CI:** каталог `.github/workflows/` в `main` отсутствует; автоматического запуска `pytest` на PR нет.
2. **Дублирование тестовых деревьев:** тесты разделены между корневым каталогом `tests/` (8 файлов: `test_geometry_core.py`, `test_geometry_new.py`, `test_grand_orchestrator.py`, `test_interlingua_fixed.py`, `test_polyglot_modules.py`, `test_security.py`, `test_smoke.py`, `test_training_smoke.py`) и `yijing_transformer/tests/` (23 файла + `__init__.py`, в т.ч. одноимённый `test_interlingua_fixed.py`). Корневой `conftest.py` содержит только `collect_ignore = ["test_dialog.py"]`. Ручной скрипт `./run_all_checks.sh` запускает 5 точечных проверок вне `pytest`.
3. **Пробел покрытия `advanced_routing.py`:** ни в `tests/`, ни в `yijing_transformer/tests/` нет модульных тестов на `GrassmannianRouter`, `Q6GrassmannianRouter`, `Q6SoftAttentionMask`, `EnhancedBianGuaAttention`, `GrassmannianMoEFFN` (включая проверку `aux_loss.requires_grad` и градиентов на `proj` / `expert_bases`).
4. **Устаревание `docs/FULL_AUDIT_REPORT.md`:** датирован `2026-03-24`, не отражает `PolyglotQuartet`, `NautilusMoME`, `GrandOrchestrator`, `advanced_routing.py`, PR #85, PR #86 и `LICENSE`.

---

## 5. Конкретные технические находки

| Приоритет | Файл / область | Суть проблемы |
|---|---|---|
| **Высокий (Security)** | `nautilus/portal.py:76-152` (`render_html`) | Прямая интерполяция `result.query`, `e.title`, `e.content[:200]`, `l`, `lnk['from']`, `lnk['to']` в HTML без `html.escape()` → Reflected XSS через параметр `?q=...` или данные из GitHub API. |
| **Высокий (Packaging / Legal)** | `LICENSE` vs `setup.py:42` & `pyproject.toml` | `LICENSE` — GNU AGPL v3 (`d6cdc75e`), а `setup.py` по-прежнему декларирует `License :: OSI Approved :: MIT License`. В `pyproject.toml` поле `license` отсутствует, а `setup.py` через `find_packages()` без `exclude` захватывает `yijing_transformer.tests` (где лежит `__init__.py`). |
| **Высокий (Гигиена git)** | Корень `main` (`hmoe_*.pt`, `hmoe_*_log.json`) | 11 бинарных файлов `.pt` по ~19 МБ (~209 МБ суммарно) и десятки `hmoe_*_log.json` отслеживаются git в корне, несмотря на наличие `*.pt` и `hmoe_*_log.json` в `.gitignore`. |
| **Средний (Портативность)** | `scripts/train_full_corpus.py:3-4,11` и `scripts/ablation_grmoe.py:12,18-22` | Жёстко прописан путь `/home/user/pro2`. В `ablation_grmoe.py` импорт `from yijing_transformer...` идёт до вычисления `ROOT` по `__file__`, из-за чего скрипт падает при запуске не из `/home/user/pro2` без `PYTHONPATH=.`. |
| **Средний (Архитектура / Импорты)** | `portal.py`, `nautilus.json`, `passports/` vs `nautilus/*` | Дублирование файлов портала в корне и в `nautilus/`. При этом `nautilus/portal.py:12` импортирует `from adapters import ...`, что работает только при запуске из каталога `nautilus/`. |
| **Средний (Сеть / CLI)** | `nautilus/portal.py:217` (`_serve`) | `HTTPServer(("", 8000), Handler)` слушает все интерфейсы `0.0.0.0:8000` без флагов `--host`/`--port` (и подавляет все логи `log_message`). |
| **Средний (Корректность обучения)** | `yijing_transformer/models/grand_orchestrator.py:148-156,228` | Немое подавление `Exception` с возвратом нулевых логитов в `ModelWrapper`; маскирование `-1e9` в `OrchestraRouter` (риск FP16 overflow). |
| **Средний (Сложность кода)** | `yijing_transformer/models/model.py`, `nautilus_yijing.py`, `hierarchical_moe.py` | Дублирующиеся импорты в `model.py:25,63,78,95`; импорт-тайм мутация `sys.path` в `nautilus_yijing.py:62-68`; локальное дублирование `_make_hexagrams()` в `hierarchical_moe.py:105` и `advanced_routing.py:25` вместо `geometry/core.py`. |
| **Низкий (Зависимости)** | `pyproject.toml`, `setup.py`, `nautilus/requirements.txt` | Несогласованность версий Python (`>=3.8` в пакете vs `3.10+` в `nautilus/requirements.txt`) и опциональных зависимостей (`dev` есть в `setup.py`, отсутствует в `pyproject.toml`). |

---

## 6. Рекомендации (план следующих шагов)

1. **Принять изменения из `audit/code-review` в `main`** (PR по `ulcwt`-фиксам + документация аудита; PR #86 `nautilus/*`, `LICENSE`, `CURRENT-STAGE.md` остаются полностью сохранными).
2. **В отдельном PR по безопасности и порталу:** экранировать весь пользовательский и внешний вывод в `nautilus/portal.py:render_html` через `html.escape()`, добавить `--host 127.0.0.1 --port 8000`, поддержать запуск `python -m nautilus.portal` из корня и устранить дублирование корневых `portal.py` / `nautilus.json` / `passports/`.
3. **Добавить GitHub Actions CI** (`.github/workflows/ci.yml`) для прогона `pytest tests/ yijing_transformer/tests/ -q` на Python 3.11+ и добавить unit-тесты для `advanced_routing.py` (включая проверку `aux_loss.requires_grad` и градиентов роутера).
4. **Синхронизировать лицензию и упаковку:** обновить `setup.py` и `pyproject.toml` под `AGPL-3.0-or-later` (или оформить решение по лицензированию), ограничить `find_packages(include=["yijing_transformer", "yijing_transformer.*"], exclude=["*.tests", "*.tests.*"])`.
5. **Очистить git-дерево и портируемость:** убрать жёсткие `/home/user/pro2` из `scripts/train_full_corpus.py` и `scripts/ablation_grmoe.py`, вывести tracked `*.pt` (~209 МБ) и рабочие `hmoe_*_log.json` из индекса git (`git rm --cached`) с публикацией весов в GitHub Releases.
