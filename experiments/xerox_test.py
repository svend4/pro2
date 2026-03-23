"""
xerox_test.py — Автоматизированный ксерокс-тест.

Ксерокс-тест: может ли модель ответить на вопросы о собственной архитектуре
в том домене, в котором она специализируется?

Если RECON-эксперт специализируется на русском языке, модель должна уметь
описать на русском что делает RECON-эксперт.

Запуск:
    python experiments/xerox_test.py --mock              # без модели
    python experiments/xerox_test.py --checkpoint path/to/model.pt
    python experiments/xerox_test.py --step 500          # показать прогресс
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Callable

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Тестовые случаи ──────────────────────────────────────────────────────────

XEROX_TESTS = [
    # (текст, ожидаемый_домен, max_PPL, описание)

    # CODE-тесты: модель должна хорошо понимать Python
    ("def forward(self, x):\n    return self.linear(x)", "CODE", 12.0,
     "Стандартная forward-функция PyTorch"),
    ("class Transformer(nn.Module):\n    def __init__(self):", "CODE", 15.0,
     "Определение класса нейросети"),
    ("import torch\nimport torch.nn as nn", "CODE", 8.0,
     "Импорты PyTorch"),

    # RECON-тесты: модель должна понимать документацию на русском
    ("Маршрутизация запроса к эксперту через роутер", "RECON", 35.0,
     "Русское описание routing"),
    ("Тернарная квантизация работает через Gumbel-Softmax", "RECON", 40.0,
     "Русское описание архитектуры"),

    # MATH-тесты: формулы и математика
    ("loss = -sum(p * log(q)) / n", "MATH", 18.0,
     "Кросс-энтропийная функция потерь"),
    ("hamming_distance = sum(a != b for a, b in zip(v1, v2))", "MATH", 15.0,
     "Расстояние Хэмминга в коде"),

    # SYSTEM-тесты: конфигурация и инфраструктура
    ("d_model: 128\nn_heads: 4\nn_layers: 4", "SYSTEM", 20.0,
     "YAML-конфигурация модели"),
    ("git commit -m 'fix: per-source trit_proj'", "SYSTEM", 25.0,
     "Git-команда"),

    # САМО-ОПИСАНИЕ
    ("YiJing-Transformer использует 64 гексаграммы как архетипы.", "RECON", 50.0,
     "Само-описание архитектуры на русском"),
    ("Q6 = {-1,+1}^6 вершины 6-мерного гиперкуба", "MATH", 30.0,
     "Математическое само-описание"),
]


# ─── PPL-вычисление ───────────────────────────────────────────────────────────

def compute_ppl_mock(text: str) -> float:
    """Mock PPL: детерминированное число на основе длины и разнообразия."""
    chars = set(text)
    base = 10 + len(text) * 0.1
    diversity_factor = len(chars) / max(len(text), 1)
    return base * (1 + diversity_factor)


def compute_ppl_with_model(text: str, model, tokenizer=None) -> float:
    """Реальный PPL через языковую модель."""
    import torch

    if tokenizer and hasattr(tokenizer, 'encode'):
        tokens = tokenizer.encode(text)
    else:
        tokens = [min(b, 255) for b in text.encode('utf-8', errors='replace')]

    if len(tokens) < 2:
        return float('inf')

    tokens_tensor = torch.tensor(tokens[:512], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits, loss = model(tokens_tensor[:, :-1], tokens_tensor[:, 1:])
        if loss is not None:
            return math.exp(min(loss.item(), 20))

        import torch.nn.functional as F
        log_probs = F.log_softmax(logits, dim=-1)
        target = tokens_tensor[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        avg_log_prob = gathered.mean().item()
        return math.exp(-avg_log_prob)


def get_routing_mock(text: str) -> dict:
    """Mock routing: возвращает домен на основе ключевых слов."""
    text_lower = text.lower()
    scores = {
        'CODE':   sum(1 for w in ['def ', 'class ', 'import', 'return', '(self', ':'] if w in text),
        'RECON':  sum(1 for w in ['маршрутизация', 'архитектура', 'модель', 'гексаграмм', 'тернарн', 'yijing'] if w in text_lower),
        'MATH':   sum(1 for w in ['loss', 'hamming', 'sum(', 'log(', 'q6', '{-1'] if w in text_lower),
        'SYSTEM': sum(1 for w in ['d_model', 'yaml', 'config', 'git', 'n_heads', 'n_layers'] if w in text_lower),
        'HUMAN':  1,
        'INFO':   sum(1 for w in ['json', 'yaml', 'config', 'settings'] if w in text_lower),
    }
    total = sum(scores.values()) or 1
    return {k: v / total for k, v in scores.items()}


# ─── Основной тест ────────────────────────────────────────────────────────────

def run_xerox_test(
    ppl_fn: Optional[Callable] = None,
    routing_fn: Optional[Callable] = None,
    verbose: bool = True,
    step: int = 0,
) -> dict:
    """
    Запускает все ксерокс-тесты и возвращает статистику.

    Args:
        ppl_fn: функция (text) → float. Если None — используется mock.
        routing_fn: функция (text) → dict. Если None — используется mock.
        step: текущий шаг обучения (для логов)
    """
    using_mock_ppl = ppl_fn is None
    using_mock_routing = routing_fn is None
    if ppl_fn is None:
        ppl_fn = compute_ppl_mock
    if routing_fn is None:
        routing_fn = get_routing_mock

    if verbose:
        print("=" * 65)
        print(f"КСЕРОКС-ТЕСТ: самоосознание модели (шаг {step})")
        if using_mock_ppl:
            print("  PPL: mock-режим (без реальной модели)")
        if using_mock_routing:
            print("  Routing: mock-режим (keyword-based)")
        print("=" * 65)

    results = []
    passed = 0

    for text, expected_domain, max_ppl, description in XEROX_TESTS:
        try:
            ppl = ppl_fn(text)
        except Exception:
            ppl = float('inf')

        try:
            routing = routing_fn(text)
            actual_domain = max(routing, key=routing.get)
        except Exception:
            actual_domain = 'UNKNOWN'
            routing = {}

        ppl_ok = ppl < max_ppl
        routing_ok = actual_domain == expected_domain
        test_passed = ppl_ok and routing_ok

        if test_passed:
            passed += 1

        result = {
            'text': text[:60] + '...' if len(text) > 60 else text,
            'description': description,
            'expected_domain': expected_domain,
            'actual_domain': actual_domain,
            'routing_ok': routing_ok,
            'ppl': round(ppl, 2),
            'max_ppl': max_ppl,
            'ppl_ok': ppl_ok,
            'passed': test_passed,
            'routing_confidence': routing.get(actual_domain, 0),
        }
        results.append(result)

        if verbose:
            status = "OK" if test_passed else "FAIL"
            routing_str = "OK" if routing_ok else f"FAIL(→{actual_domain})"
            ppl_str = f"OK({ppl:.1f})" if ppl_ok else f"FAIL({ppl:.1f}>{max_ppl})"
            print(f"[{status}] [{expected_domain}] {description}")
            print(f"    Routing: {routing_str} | PPL: {ppl_str}")

    total = len(XEROX_TESTS)
    score = passed / total

    if verbose:
        print("\n" + "=" * 65)
        print(f"ИТОГ: {passed}/{total} ({score*100:.0f}%)")
        print()

        if score >= 0.8:
            print("КСЕРОКС-ТЕСТ ПРОЙДЕН: модель осознаёт собственную архитектуру")
        elif score >= 0.5:
            print("КСЕРОКС-ТЕСТ ЧАСТИЧНО ПРОЙДЕН: есть слабые домены")
        else:
            print("КСЕРОКС-ТЕСТ ПРОВАЛЕН: нужна доработка routing или обучение")

        by_domain: dict = {}
        for r in results:
            d = r['expected_domain']
            if d not in by_domain:
                by_domain[d] = {'passed': 0, 'total': 0}
            by_domain[d]['total'] += 1
            if r['passed']:
                by_domain[d]['passed'] += 1

        print("\nРезультаты по доменам:")
        for domain, stats in by_domain.items():
            domain_score = stats['passed'] / stats['total']
            bar = "#" * int(domain_score * 10) + "." * (10 - int(domain_score * 10))
            print(f"  {domain:<8} [{bar}] {stats['passed']}/{stats['total']}")

    return {
        'step': step,
        'score': score,
        'passed': passed,
        'total': total,
        'results': results,
        'mock_mode': using_mock_ppl or using_mock_routing,
    }


# ─── Загрузка модели ──────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: Path):
    """Загружает LeanYiJingGPT из checkpoint."""
    import torch
    from yijing_transformer.models.lean_model import LeanYiJingGPT

    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'model_state' in saved:
        cfg = saved.get('config', {})
        model = LeanYiJingGPT(
            vocab_size=cfg.get('vocab_size', 256),
            d_model=cfg.get('d_model', 128),
            n_layers=cfg.get('n_layers', 4),
            n_heads=cfg.get('n_heads', 4),
            block_size=cfg.get('block_size', 256),
        )
        model.load_state_dict(saved['model_state'], strict=False)
        model.eval()
        print(f"Модель загружена из {ckpt_path.name} "
              f"(step={saved.get('step', '?')}, "
              f"ppl={saved.get('final_ppl', '?')})")
        return model

    raise ValueError(f"Неизвестный формат checkpoint: {list(saved.keys())}")


# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ксерокс-тест для YiJing/NautilusMoME')
    parser.add_argument('--checkpoint', default='', help='Путь к .pt файлу модели')
    parser.add_argument('--mock', action='store_true', help='Mock-режим без реальной модели')
    parser.add_argument('--step', type=int, default=0, help='Текущий шаг обучения')
    parser.add_argument('--output', default='', help='Сохранить результаты в JSON')
    args = parser.parse_args()

    ppl_fn = None

    if not args.mock and args.checkpoint:
        try:
            model = load_model_from_checkpoint(Path(args.checkpoint))
            ppl_fn = lambda text: compute_ppl_with_model(text, model)
        except Exception as e:
            print(f"Ошибка загрузки ({e}), используется mock")

    output = run_xerox_test(ppl_fn=ppl_fn, step=args.step)

    save_path = args.output or str(ROOT / 'experiments' / 'xerox_test_result.json')
    Path(save_path).parent.mkdir(exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены: {save_path}")
    print(f"\nСледующий шаг: запускать этот тест каждые 500 шагов обучения.")
    print(f"Цель: score >= 0.8 до начала масштабирования.")
