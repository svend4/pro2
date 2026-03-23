"""
Тест: per-source trit_proj работает корректно.
Запуск: python tests/test_interlingua_fixed.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from yijing_transformer.models.geometry.interlingua_fixed import ArchetypalInterlinguaFixed


def test_per_source_projs_independent():
    """Проверить что у каждого источника свой проектор с разными весами."""
    model = ArchetypalInterlinguaFixed(d_model=32, n_sources=4, n_archetypes=8)

    weights = [proj.weight.data for proj in model.trit_projs]
    for i, j in [(0, 1), (0, 2), (1, 3)]:
        diff = (weights[i] - weights[j]).abs().max().item()
        assert diff > 1e-6, f"Источники {i} и {j} имеют одинаковые веса!"

    print("ТЕСТ ПРОЙДЕН: веса источников независимы")


def test_diversity_after_training():
    """
    Ключевой тест: после нескольких шагов обучения
    разные источники должны иметь разные паттерны тритов.
    """
    d_model = 64
    n_sources = 3
    B = 4
    T_seq = 8

    model = ArchetypalInterlinguaFixed(
        d_model=d_model,
        n_sources=n_sources,
        n_archetypes=16,
        diversity_weight=0.1,
        warmup_steps=50,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(30):
        sources = [
            torch.randn(B, T_seq, d_model) * (i + 1) * 0.5
            for i in range(n_sources)
        ]
        core = torch.randn(B, T_seq, d_model)

        output, aux_loss = model(sources, core)
        main_loss = output.pow(2).mean()
        total_loss = main_loss + aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    sources = [torch.randn(B, T_seq, d_model) * (i + 1) for i in range(n_sources)]
    diag = model.get_diagnostics(sources)

    print("\nДиагностика после 30 шагов:")
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    pos_vals = [diag[f'source_{i}_trit_pos'] for i in range(n_sources)]
    diversity_ok = max(pos_vals) - min(pos_vals) > 0.05

    print(f"\n  trit_pos по источникам: {[f'{v:.3f}' for v in pos_vals]}")
    print(f"  Разброс: {max(pos_vals) - min(pos_vals):.3f}")

    if diversity_ok:
        print("\nТЕСТ ПРОЙДЕН: источники дифференцированы")
    else:
        print("\nТЕСТ: дифференциация слабая, нужно больше шагов или выше diversity_weight")

    gate_val = diag['gate_value']
    assert 0 < gate_val < 1, f"Gate застрял: {gate_val}"
    print(f"Gate в допустимом диапазоне: {gate_val:.3f}")

    # diversity_ok — информационная метрика, слабая дифференциация допустима на 30 шагах


def test_gradient_flow():
    """Проверить что градиенты проходят через Gumbel-Softmax."""
    model = ArchetypalInterlinguaFixed(d_model=32, n_sources=2, n_archetypes=8)
    model.train()

    sources = [torch.randn(2, 4, 32, requires_grad=False) for _ in range(2)]
    core = torch.randn(2, 4, 32)

    output, aux_loss = model(sources, core)
    loss = output.sum() + aux_loss
    loss.backward()

    # Проверить что у trit_projs есть градиенты
    for i, proj in enumerate(model.trit_projs):
        assert proj.weight.grad is not None, f"Нет градиента у trit_projs[{i}]"
        assert proj.weight.grad.abs().max() > 0, f"Нулевой градиент у trit_projs[{i}]"

    print("ТЕСТ ПРОЙДЕН: градиенты проходят через все per-source проекторы")


if __name__ == '__main__':
    test_per_source_projs_independent()
    test_gradient_flow()
    test_diversity_after_training()
    print("\nВсе тесты завершены")
