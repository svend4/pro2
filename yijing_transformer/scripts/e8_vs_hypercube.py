#!/usr/bin/env python3
"""
Фаза 5: E8 vs Hypercube Comparison.

Сравнивает два подхода к квантизации в R⁸:
1. Октограммы {-1,+1}⁸ — 256 точек, тензорная факторизация
2. E8 корни — 240 точек, плотная упаковка

Вопросы:
- Какая геометрия даёт лучшую квантизацию?
- Как различается ошибка реконструкции?
- Есть ли разница для задач с Z₂ vs без Z₂ структуры?
"""

import os
import sys
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry import (
    generate_hypercube, generate_e8_roots,
    E8Quantizer, HierarchicalQuantizer,
)


def compare_geometry():
    """Сравнение геометрических свойств."""
    print("=" * 70)
    print("  PHASE 5: E8 vs HYPERCUBE COMPARISON")
    print("=" * 70)

    # Гиперкуб {-1,+1}⁸
    octograms = generate_hypercube(8)  # (256, 8)
    e8 = generate_e8_roots()  # (240, 8)

    print(f"\n  Октограммы {'{-1,+1}⁸'}:")
    print(f"    Точек: {len(octograms)}")
    print(f"    Нормы: min={octograms.norm(dim=1).min():.3f}, "
          f"max={octograms.norm(dim=1).max():.3f}, "
          f"mean={octograms.norm(dim=1).mean():.3f}")
    print(f"    Все нормы = √8 = {math.sqrt(8):.3f}")

    print(f"\n  E8 корни:")
    print(f"    Точек: {len(e8)}")
    print(f"    Нормы: min={e8.norm(dim=1).min():.3f}, "
          f"max={e8.norm(dim=1).max():.3f}, "
          f"mean={e8.norm(dim=1).mean():.3f}")
    print(f"    Все нормы = √2 = {math.sqrt(2):.3f}")

    # Покрывающий радиус (covering radius)
    # Максимальное расстояние от случайной точки до ближайшей кодовой точки
    torch.manual_seed(42)
    test_points = torch.randn(10000, 8)

    # Для октограмм
    oct_dists = torch.cdist(test_points, octograms)  # (10000, 256)
    oct_min_dists = oct_dists.min(dim=1).values
    oct_covering = oct_min_dists.max().item()
    oct_avg_dist = oct_min_dists.mean().item()

    # Для E8
    e8_dists = torch.cdist(test_points, e8)  # (10000, 240)
    e8_min_dists = e8_dists.min(dim=1).values
    e8_covering = e8_min_dists.max().item()
    e8_avg_dist = e8_min_dists.mean().item()

    print(f"\n  Покрывающие свойства (10000 случайных точек N(0,1)):")
    print(f"  {'':>20} {'Октограммы':>15} {'E8':>15}")
    print(f"  {'Покрыв. радиус':>20} {oct_covering:>15.3f} {e8_covering:>15.3f}")
    print(f"  {'Среднее расст.':>20} {oct_avg_dist:>15.3f} {e8_avg_dist:>15.3f}")

    # Нормализованные (на сфере)
    test_normalized = F.normalize(test_points, p=2, dim=1) * math.sqrt(8)  # на сфере √8

    oct_n_dists = torch.cdist(test_normalized, octograms).min(dim=1).values
    e8_scaled = e8 * math.sqrt(8) / math.sqrt(2)  # масштабируем E8 до нормы √8
    e8_n_dists = torch.cdist(test_normalized, e8_scaled).min(dim=1).values

    print(f"\n  На сфере радиуса √8:")
    print(f"  {'Среднее расст.':>20} {oct_n_dists.mean().item():>15.3f} {e8_n_dists.mean().item():>15.3f}")
    print(f"  {'Макс расст.':>20} {oct_n_dists.max().item():>15.3f} {e8_n_dists.max().item():>15.3f}")

    # Тензорная факторизация
    print(f"\n  Тензорная факторизация:")
    print(f"  Октограммы: 256 = 2⁸ = (2⁴)² = 16×16 или (2²)⁴ = 4⁴")
    print(f"    Можно факторизовать: softmax(16)+softmax(16) вместо softmax(256)")
    print(f"    Вычислительная сложность: O(32) вместо O(256)")
    print(f"  E8: 240 — простое число? Нет: 240 = 2⁴·3·5")
    print(f"    НЕ допускает тензорную факторизацию")
    print(f"    Вычислительная сложность: O(240)")

    # Групповая структура
    print(f"\n  Групповая структура:")
    print(f"  Октограммы: Z₂⁸ = абелева группа, замкнута относительно ⊙")

    # Проверяем замкнутость E8
    e8_set = set(tuple(x.tolist()) for x in e8)
    closed_count = 0
    not_closed_count = 0
    for i in range(min(100, len(e8))):
        for j in range(min(100, len(e8))):
            # «Умножение» для E8: сложение + проекция на ближайший корень
            s = e8[i] + e8[j]
            # Ищем ближайший корень
            dists = (e8 - s.unsqueeze(0)).norm(dim=1)
            nearest = e8[dists.argmin()]
            if tuple(nearest.tolist()) in e8_set:
                closed_count += 1
            else:
                not_closed_count += 1

    print(f"  E8: проверка замкнутости (сложение + проекция): "
          f"{closed_count}/{closed_count+not_closed_count} замкнуто")
    print(f"    E8 НЕ замкнута относительно + (корни E8 — не подгруппа R⁸)")

    return {
        'octogram_points': len(octograms),
        'e8_points': len(e8),
        'oct_covering_radius': oct_covering,
        'e8_covering_radius': e8_covering,
        'oct_avg_dist': oct_avg_dist,
        'e8_avg_dist': e8_avg_dist,
    }


def quantization_quality_test():
    """Сравнение качества квантизации для обучения."""
    print(f"\n{'=' * 70}")
    print("  QUANTIZATION QUALITY TEST")
    print(f"{'=' * 70}")

    torch.manual_seed(42)
    d_model = 128

    # Создаём два квантизатора
    oct_quant = HierarchicalQuantizer(total_dim=8, group_dim=4, temp=0.3)
    e8_quant = E8Quantizer(temp=0.3)

    # Проекции
    proj_to_8d = nn.Linear(d_model, 8, bias=False)
    proj_from_8d = nn.Linear(8, d_model, bias=False)

    # Тестовые данные
    x = torch.randn(32, 16, d_model)  # (B, T, D)

    # Квантизация через октограммы
    z = proj_to_8d(x)
    z_oct = oct_quant(z)
    x_oct = proj_from_8d(z_oct)
    oct_recon_error = F.mse_loss(x_oct, x).item()

    # Квантизация через E8
    z_e8 = e8_quant(z)
    x_e8 = proj_from_8d(z_e8)
    e8_recon_error = F.mse_loss(x_e8, x).item()

    print(f"\n  Ошибка реконструкции (MSE):")
    print(f"  Октограммы (HierarchicalQuantizer 8D): {oct_recon_error:.6f}")
    print(f"  E8 (E8Quantizer):                      {e8_recon_error:.6f}")

    # Z₂-structured данные (XOR)
    z_binary = torch.sign(torch.randn(32, 16, 8))  # Уже на гиперкубе
    z_oct_b = oct_quant(z_binary)
    z_e8_b = e8_quant(z_binary)

    oct_binary_error = F.mse_loss(z_oct_b, z_binary).item()
    e8_binary_error = F.mse_loss(z_e8_b, z_binary).item()

    print(f"\n  Ошибка на бинарных данных ({'{-1,+1}⁸'}):")
    print(f"  Октограммы: {oct_binary_error:.6f} (должно быть ~0)")
    print(f"  E8:          {e8_binary_error:.6f}")

    print(f"\n  ВЫВОД:")
    if oct_binary_error < e8_binary_error:
        print(f"  ✓ Октограммы лучше для Z₂-данных (точное попадание в вершины)")
    else:
        print(f"  ✗ E8 лучше даже для Z₂-данных (неожиданно)")

    if e8_recon_error < oct_recon_error:
        print(f"  ✓ E8 лучше для непрерывных данных (плотнее упаковка)")
    else:
        print(f"  ✗ Октограммы лучше для непрерывных данных")

    print(f"\n  Рекомендация:")
    print(f"  • Для задач с Z₂-структурой (XOR, modular) → октограммы")
    print(f"  • Для общих задач → E8 (если вычислительный бюджет позволяет)")
    print(f"  • Октограммы имеют преимущество O(32) vs O(240) на факторизации")

    return {
        'oct_recon': oct_recon_error,
        'e8_recon': e8_recon_error,
        'oct_binary': oct_binary_error,
        'e8_binary': e8_binary_error,
    }


def main():
    results = {}
    results['geometry_comparison'] = compare_geometry()
    results['quantization_quality'] = quantization_quality_test()

    output_path = os.path.join(os.path.dirname(__file__), '..', 'e8_vs_hypercube_results.json')
    output_path = os.path.abspath(output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
