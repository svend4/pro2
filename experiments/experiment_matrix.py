"""
Официальный протокол экспериментов (Шаг 7).

Матрица экспериментов определяет:
  - порядок запуска (что сначала, что потом)
  - условия валидного вердикта (минимум 3000 шагов, реальные данные)
  - критерии успеха (PPL, routing_confidence, ксерокс-тест)

ВАЖНО: Вердикт на 800 шагах или синтетических данных статистически
недействителен. Это привело к ошибочному исключению Касаткина в v59.

Запуск матрицы:
    python experiments/experiment_matrix.py --list       # показать список
    python experiments/experiment_matrix.py --run lean   # запустить эксперимент
    python experiments/experiment_matrix.py --status     # текущий статус
"""

from __future__ import annotations
import json
import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Официальный протокол ─────────────────────────────────────────

PROTOCOL = """
╔══════════════════════════════════════════════════════════════════╗
║          ОФИЦИАЛЬНЫЙ ПРОТОКОЛ ЭКСПЕРИМЕНТА                       ║
║          (каждое условие обязательно для валидного вердикта)     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. ДАННЫЕ: минимум 25M токенов реальных данных                  ║
║     (НЕ синтетический WikiText)                                  ║
║     Рекомендуется: репозитории из NautilusMoME                   ║
║                                                                  ║
║  2. ШАГИ: минимум 3000 (НЕ 800)                                  ║
║     Вердикт на 800 шагах — статистически недействителен          ║
║                                                                  ║
║  3. TEMPERATURE: дождаться T < 0.1 перед сравнением PPL          ║
║     Логировать T на каждом шаге                                  ║
║                                                                  ║
║  4. КСЕРОКС-ТЕСТ: запускать каждые 500 шагов                     ║
║     Если не пройден к шагу 2000 — архитектурная проблема         ║
║                                                                  ║
║  5. РОЛЬ КОМПОНЕНТА: тестировать каждый источник                 ║
║     в правильной роли (не "attention bias" для Касаткина)        ║
║                                                                  ║
║  6. BASELINE: всегда сравнивать с LeanYiJing (Шаг 3)            ║
║     Вердикт "вреден" = хуже LeanYiJing при равных условиях      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─── Реестр источников ───────────────────────────────────────────

SOURCES_REGISTRY: dict = {
    # TIER 1: Доказанно работают (из ablation v59)
    'heisenberg': {
        'module': 'yijing_transformer.models.geometry.attention.HeisenbergAttention',
        'status': 'proven',
        'ppl_contribution': +0.75,
        'correct_role': 'self-attention с адаптивной температурой',
    },
    'flower_gat': {
        'module': 'yijing_transformer.models.geometry.attention.FlowerOfLifeGAT',
        'status': 'proven',
        'ppl_contribution': +0.59,
        'correct_role': 'GAT на топологии Цветка Жизни',
    },

    # TIER 2: Не протестированы в правильных условиях
    'kasatkin_3d': {
        'module': 'yijing_transformer.models.geometry.kasatkin_router.KasatkinQ6Router',
        'status': 'untested_proper',
        'correct_role': 'координатная система для Q6-router (НЕ attention bias)',
        'test_condition': '3000 шагов, реальные данные, роль=router',
        'previous_wrong_role': 'CubicAttentionBias — добавка к score (было неправильно)',
    },
    'fomyuk_d4': {
        'module': 'yijing_transformer.models.geometry.equivariant.D4Equivariant',
        'status': 'untested_proper',
        'correct_role': 'регуляризация эмбеддингов через AntipodalRegularization',
        'test_condition': 'antipodal_loss на эмбеддинги кодбука, не слой attention',
    },
    'skliarova_palace': {
        'module': 'yijing_transformer.models.geometry.attention.PalaceAttention',
        'status': 'untested_proper',
        'correct_role': 'block-sparse attention по 8 дворцам',
        'test_condition': '8×8 блочная маска вместо dense attention, 3000 шагов',
    },
    'solan_glyph': {
        'module': 'nautilus.glyph_adapter.GlyphAwareEmbedding',
        'status': 'untested_proper',
        'correct_role': 'визуальная токенизация SOLAN-76 с Хэмминг-bias',
        'test_condition': 'NautilusMoME + GlyphAwareEmbedding, ксерокс-тест > 80%',
    },
}


# ─── Матрица экспериментов ────────────────────────────────────────

@dataclass
class Experiment:
    """Описание одного эксперимента."""
    name: str
    week: int
    model: str
    min_steps: int
    sources: list
    hypothesis: str
    success_criterion: str
    fix: str = ""
    status: str = "pending"       # pending | running | done | failed
    result_ppl: Optional[float] = None
    result_notes: str = ""
    verdict: str = ""             # proven | disproven | inconclusive

    def is_valid_verdict_possible(self) -> bool:
        """Проверяет что условия для валидного вердикта соблюдены."""
        return (
            self.min_steps >= 3000
            and self.status in ('done', 'failed')
        )


EXPERIMENTS: list[Experiment] = [
    # ── Неделя 1: официальный baseline ──────────────────────────
    Experiment(
        name='lean_baseline',
        week=1,
        model='LeanYiJingGPT',
        min_steps=3000,
        sources=['heisenberg', 'flower_gat'],
        hypothesis='Два доказанных источника дают PPL < 1.07',
        success_criterion='PPL < 1.07 на реальных данных',
        status='pending',
    ),

    # ── Неделя 2: Interlingua с исправлением ────────────────────
    Experiment(
        name='interlingua_fixed',
        week=2,
        model='LeanYiJingGPT + ArchetypalInterlingua (per-source)',
        min_steps=3000,
        sources=['heisenberg', 'flower_gat', 'interlingua'],
        hypothesis='ArchetypalInterlingua с per-source trit_proj улучшает PPL',
        success_criterion='PPL < lean_baseline PPL',
        fix=(
            'per_source_trit_proj + gumbel_softmax (T=1.0→0.05) + '
            'diversity_loss (weight=0.01)'
        ),
        status='pending',
    ),

    # ── Неделя 3: Касаткин в правильной роли ────────────────────
    Experiment(
        name='kasatkin_as_router',
        week=3,
        model='LeanYiJingGPT + KasatkinQ6Router',
        min_steps=3000,
        sources=['heisenberg', 'flower_gat', 'kasatkin_3d'],
        hypothesis='Касаткин как Q6-router даёт routing_confidence > 15%',
        success_criterion='routing_confidence > 15% И PPL не хуже lean_baseline',
        fix='Касаткин как роутер (не CubicAttentionBias)',
        status='pending',
    ),

    # ── Неделя 4: SOLAN-токенизация ─────────────────────────────
    Experiment(
        name='solan_nautilus',
        week=4,
        model='NautilusMoME + GlyphAwareEmbedding',
        min_steps=3000,
        sources=['solan_glyph'],
        hypothesis='SOLAN-токены с Хэмминг-bias улучшают routing точность',
        success_criterion='ксерокс-тест > 80% И cross_domain_diversity > 0.16',
        status='pending',
    ),

    # ── Неделя 5: Склярова в правильной роли ────────────────────
    Experiment(
        name='palace_block_sparse',
        week=5,
        model='LeanYiJingGPT + PalaceAttention (block-sparse)',
        min_steps=3000,
        sources=['heisenberg', 'flower_gat', 'skliarova_palace'],
        hypothesis='PalaceAttention как block-sparse 8×8 маска улучшает PPL',
        success_criterion='PPL < lean_baseline PPL',
        fix='block-sparse 8×8 маска вместо dense attention',
        status='pending',
    ),

    # ── Неделя 6: meta bridge интеграция ────────────────────────
    Experiment(
        name='meta_hexlearn_router',
        week=6,
        model='LeanYiJingGPT + HexLearnRouter (из meta)',
        min_steps=3000,
        sources=['heisenberg', 'flower_gat', 'kasatkin_3d'],
        hypothesis='Геометрический k-NN router из meta улучшает специализацию',
        success_criterion='routing_confidence > 20% по сравнению с KasatkinQ6Router',
        status='pending',
    ),
]


# ─── CLI ─────────────────────────────────────────────────────────

def print_list():
    print(PROTOCOL)
    print("=" * 64)
    print("МАТРИЦА ЭКСПЕРИМЕНТОВ")
    print("=" * 64)
    for exp in EXPERIMENTS:
        status_icon = {'pending': '○', 'running': '●', 'done': '✓', 'failed': '✗'}.get(exp.status, '?')
        print(f"\n  {status_icon} [{exp.name}] (Неделя {exp.week})")
        print(f"     Модель: {exp.model}")
        print(f"     Шаги:   {exp.min_steps} (минимум)")
        print(f"     Успех:  {exp.success_criterion}")
        if exp.fix:
            print(f"     Фикс:   {exp.fix}")
        if exp.result_ppl is not None:
            print(f"     PPL:    {exp.result_ppl:.4f}")
        if exp.verdict:
            print(f"     Вердикт: {exp.verdict}")


def print_status():
    done = sum(1 for e in EXPERIMENTS if e.status == 'done')
    failed = sum(1 for e in EXPERIMENTS if e.status == 'failed')
    running = sum(1 for e in EXPERIMENTS if e.status == 'running')
    total = len(EXPERIMENTS)
    print(f"Статус: {done}/{total} завершено, {running} в процессе, {failed} неуспешно")

    baseline = next((e for e in EXPERIMENTS if e.name == 'lean_baseline'), None)
    if baseline and baseline.result_ppl is not None:
        print(f"Baseline PPL (LeanYiJing): {baseline.result_ppl:.4f}")


def save_state(path: str = 'experiments/state.json'):
    state = [asdict(e) for e in EXPERIMENTS]
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"Состояние сохранено: {path}")


def update_result(name: str, ppl: float, notes: str = '', verdict: str = ''):
    """Обновить результат эксперимента."""
    for exp in EXPERIMENTS:
        if exp.name == name:
            exp.result_ppl = ppl
            exp.result_notes = notes
            exp.verdict = verdict
            exp.status = 'done'
            print(f"Обновлён {name}: PPL={ppl:.4f}, вердикт={verdict}")
            return
    print(f"Эксперимент '{name}' не найден")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Матрица экспериментов YiJing')
    parser.add_argument('--list', action='store_true', help='Показать список экспериментов')
    parser.add_argument('--status', action='store_true', help='Текущий статус')
    parser.add_argument('--save', action='store_true', help='Сохранить состояние в JSON')
    parser.add_argument('--sources', action='store_true', help='Показать реестр источников')
    args = parser.parse_args()

    if args.list:
        print_list()
    elif args.status:
        print_status()
    elif args.save:
        save_state()
    elif args.sources:
        print("\n=== РЕЕСТР ИСТОЧНИКОВ ===\n")
        for name, info in SOURCES_REGISTRY.items():
            icon = '✓' if info['status'] == 'proven' else '?'
            print(f"  {icon} {name}: {info['status']}")
            print(f"      Роль: {info['correct_role']}")
            if 'ppl_contribution' in info:
                print(f"      PPL contribution: +{info['ppl_contribution']}")
            if 'test_condition' in info:
                print(f"      Условие: {info['test_condition']}")
    else:
        print_list()
