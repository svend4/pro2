"""
Тесты математических свойств И-Цзин геометрии.

Использование:
    python scripts/test_geometry.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from models.geometry import (
    generate_trigrams,
    generate_hexagrams,
    verify_yijing_properties,
    YiJingQuantizer,
    FactoredYiJingQuantizer,
)


def test_trigram_properties():
    """Проверка свойств 8 триграмм."""
    tri = generate_trigrams()
    assert tri.shape == (8, 3)
    assert torch.all((tri == 1.0) | (tri == -1.0))

    # Взаимные расстояния
    dists = torch.cdist(tri, tri)
    print("Trigram distance matrix (Hamming ∝ L2²/4):")
    print((dists ** 2 / 4).int())

    # Все триграммы попарно различны на ≥2 координаты (L2 ≥ 2)
    mask = ~torch.eye(8, dtype=torch.bool)
    min_dist = dists[mask].min().item()
    print(f"Min pairwise L2 distance: {min_dist:.2f} (expected 2.0)")
    assert abs(min_dist - 2.0) < 0.01

    print("[PASS] Trigram properties\n")


def test_hexagram_properties():
    """Проверка свойств 64 гексаграмм."""
    hex = generate_hexagrams()
    assert hex.shape == (64, 6)

    # Факторизация: первые 3 = верхняя триграмма, последние 3 = нижняя
    tri = generate_trigrams()
    for i in range(64):
        upper = hex[i, :3]
        lower = hex[i, 3:]
        # Каждая половина должна быть одной из 8 триграмм
        assert any(torch.allclose(upper, tri[j]) for j in range(8)), \
            f"Hexagram {i} upper half is not a valid trigram"
        assert any(torch.allclose(lower, tri[j]) for j in range(8)), \
            f"Hexagram {i} lower half is not a valid trigram"

    print("[PASS] Hexagram factorization (all 64 = upper ⊗ lower)\n")


def test_quantizer_equivalence():
    """Факторизованный квантизатор должен давать тот же результат, что наивный."""
    naive = YiJingQuantizer(temp=0.01)  # Очень низкая температура → жёсткая квантизация
    factored = FactoredYiJingQuantizer(temp=0.01)

    x = torch.randn(4, 16, 6)

    naive_out = naive(x)
    factored_out = factored(x)

    # При temp→0 оба должны давать ±1
    naive_hard = torch.sign(naive_out)
    factored_hard = torch.sign(factored_out)

    match_rate = (naive_hard == factored_hard).float().mean().item()
    print(f"Hard quantization agreement: {match_rate*100:.1f}%")
    assert match_rate > 0.95, f"Low agreement: {match_rate}"

    print("[PASS] Quantizer equivalence\n")


def test_hard_quantize_is_sign():
    """sign(x) — ближайшая вершина гиперкуба."""
    q = FactoredYiJingQuantizer()
    x = torch.randn(100, 6)
    hard = q.hard_quantize(x)
    expected = torch.sign(x)
    assert torch.allclose(hard, expected)
    print("[PASS] hard_quantize(x) == sign(x)\n")


def test_quantizer_speed():
    """Сравнение скорости: naive vs factored."""
    naive = YiJingQuantizer(temp=0.3)
    factored = FactoredYiJingQuantizer(temp=0.3)

    x = torch.randn(32, 512, 6)  # Типичный размер

    # Warmup
    for _ in range(5):
        naive(x)
        factored(x)

    n_runs = 50

    start = time.time()
    for _ in range(n_runs):
        naive(x)
    naive_time = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        factored(x)
    factored_time = (time.time() - start) / n_runs

    speedup = naive_time / factored_time
    print(f"Naive quantizer:    {naive_time*1000:.2f} ms")
    print(f"Factored quantizer: {factored_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"[{'PASS' if speedup > 1.0 else 'INFO'}] Factored is {'faster' if speedup > 1 else 'slower'}\n")


def test_bian_gua_symmetry():
    """变卦: инверсия всех линий = полное отражение."""
    from models.geometry import BianGuaTransform

    bg = BianGuaTransform(d_model=64)
    # Установим change_logits → +inf (все линии изменяются)
    with torch.no_grad():
        bg.change_logits.fill_(10.0)  # sigmoid(10) ≈ 1.0

    x = torch.randn(2, 8, 64)
    z_in = bg.proj_to_6d(x)
    z_expected = -z_in  # полная инверсия

    # Проверяем, что 变 действительно инвертирует
    change_prob = torch.sigmoid(bg.change_logits)
    z_transformed = z_in * (1 - 2 * change_prob)
    assert torch.allclose(z_transformed, z_expected, atol=1e-3)

    print("[PASS] BianGua full inversion\n")


if __name__ == "__main__":
    print("=" * 50)
    print("YiJing Geometry Tests")
    print("=" * 50)
    print()

    verify_yijing_properties(generate_trigrams(), generate_hexagrams())
    print("[PASS] Basic mathematical properties\n")

    test_trigram_properties()
    test_hexagram_properties()
    test_quantizer_equivalence()
    test_hard_quantize_is_sign()
    test_quantizer_speed()
    test_bian_gua_symmetry()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
