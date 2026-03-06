"""
YiJing-Transformer: Интерактивное исследование
===============================================

Этот скрипт можно запустить как Jupyter notebook (через jupytext)
или как обычный Python скрипт.

Содержание:
1. Визуализация геометрии триграмм и гексаграмм
2. Сравнение квантизаторов
3. Демо модели на синтетических данных
4. Ablation study
5. Анализ обученных параметров
"""

# %% [markdown]
# # YiJing-Transformer: Исследование геометрических структур

# %% Imports
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from models.geometry import (
    generate_trigrams, generate_hexagrams, generate_octograms,
    generate_tetragrams, generate_hypercube,
    FactoredYiJingQuantizer, HierarchicalQuantizer, DeformableQuantizer,
    YiJingQuantizer,
)
from config.config import YiJingConfig
from models.model import YiJingGPT
from models.baseline import VanillaGPT

# %% [markdown]
# ## 1. Иерархия гиперкубов

# %%
print("=== Иерархия гиперкубов И-Цзин ===\n")
for name, n, count in [
    ("Монограмма", 1, 2),
    ("Биграмма", 2, 4),
    ("Триграмма", 3, 8),
    ("Тетраграмма", 4, 16),
    ("Пентаграмма", 5, 32),
    ("Гексаграмма", 6, 64),
    ("Гептаграмма", 7, 128),
    ("Октограмма", 8, 256),
]:
    cube = generate_hypercube(n)
    norm = torch.norm(cube[0]).item()
    print(f"{name:.<20} {count:>5} точек в R^{n}, ||x|| = √{n} ≈ {norm:.3f}")

print(f"\nE8:                    240 точек в R^8, ||x|| = √2 ≈ {2**0.5:.3f}")

# %% [markdown]
# ## 2. Сравнение квантизаторов

# %%
print("\n=== Сравнение квантизаторов ===\n")

configs = [
    ("Factored 6D (2×tri)", FactoredYiJingQuantizer(temp=0.3), 6, "2×softmax(8)=16"),
    ("Hierarchical 6D (3×bi)", HierarchicalQuantizer(6, 2, temp=0.3), 6, "3×softmax(4)=12"),
    ("Hierarchical 6D (2×tri)", HierarchicalQuantizer(6, 3, temp=0.3), 6, "2×softmax(8)=16"),
    ("Hierarchical 8D (4×bi)", HierarchicalQuantizer(8, 2, temp=0.3), 8, "4×softmax(4)=16"),
    ("Hierarchical 8D (2×tetra)", HierarchicalQuantizer(8, 4, temp=0.3), 8, "2×softmax(16)=32"),
    ("Naive 64 (brute force)", YiJingQuantizer(temp=0.3), 6, "softmax(64)=64"),
]

x6 = torch.randn(32, 256, 6)
x8 = torch.randn(32, 256, 8)

print(f"{'Quantizer':<30} {'Dim':>5} {'Ops':>20} {'Time (ms)':>12}")
print("-" * 70)

for name, q, dim, ops in configs:
    x = x6 if dim == 6 else x8
    # Warmup
    for _ in range(3):
        q(x)
    # Benchmark
    t0 = time.time()
    for _ in range(20):
        q(x)
    elapsed = (time.time() - t0) / 20 * 1000
    print(f"{name:<30} {dim:>5} {ops:>20} {elapsed:>12.2f}")

# %% [markdown]
# ## 3. Факторизация: E8 vs И-Цзин vs Октограммы

# %%
print("\n=== Теоретическое сравнение ===\n")

import math

comparisons = [
    ("E8 (brute force)", 8, 240, "softmax(240)", 240),
    ("Гексаграммы (brute)", 6, 64, "softmax(64)", 64),
    ("Гексаграммы (2×tri)", 6, 64, "2×softmax(8)", 16),
    ("Гексаграммы (3×bi)", 6, 64, "3×softmax(4)", 12),
    ("Октограммы (brute)", 8, 256, "softmax(256)", 256),
    ("Октограммы (2×tetra)", 8, 256, "2×softmax(16)", 32),
    ("Октограммы (4×bi)", 8, 256, "4×softmax(4)", 16),
]

print(f"{'Кодбук':<25} {'Dim':>4} {'Points':>7} {'Факторизация':<20} {'Ops':>5} {'Bits':>6}")
print("-" * 72)
for name, dim, n_pts, factor, ops in comparisons:
    bits = math.log2(n_pts)
    print(f"{name:<25} {dim:>4} {n_pts:>7} {factor:<20} {ops:>5} {bits:>6.1f}")

# %% [markdown]
# ## 4. Быстрый эксперимент: overfit на 1 батче

# %%
print("\n=== Overfit test (1 batch, 50 steps) ===\n")

def overfit_test(model_cls, cfg, name, steps=50):
    model = model_cls(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))

    losses = []
    model.train()
    for step in range(steps):
        _, loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    total, hex_p = model.count_parameters()
    print(f"{name:<25} params={total:>8,}  loss: {losses[0]:.3f} → {losses[-1]:.3f}  "
          f"(Δ={losses[0]-losses[-1]:+.3f})")
    return losses

cfg_base = YiJingConfig(
    vocab_size=128, d_model=64, n_layers=2, n_heads=4, block_size=32,
    use_rope=True, use_swiglu=True,
)

# YiJing с разными квантизаторами
for qt in ['factored6', 'hierarchical', 'octogram']:
    cfg = YiJingConfig(
        vocab_size=128, d_model=64, n_layers=2, n_heads=4, block_size=32,
        use_rope=True, use_swiglu=True, use_bian_gua=True,
        quantizer_type=qt,
        quant_total_dim=8 if qt == 'octogram' else 6,
    )
    overfit_test(YiJingGPT, cfg, f"YiJing ({qt})")

# Vanilla baseline
overfit_test(VanillaGPT, cfg_base, "Vanilla")

# %% [markdown]
# ## 5. Деформируемый кодбук

# %%
print("\n=== Деформируемый кодбук ===\n")

dq = DeformableQuantizer(total_dim=6, group_dim=3, temp=0.3, deform_scale=0.0)
x = torch.randn(4, 16, 6, requires_grad=True)

# До обучения
out_before = dq(x)
print(f"До обучения:  deform_scale={dq.deform_scale.item():.4f}, "
      f"delta_norm={dq.delta.norm().item():.4f}")

# Симулируем несколько шагов обучения
optimizer = torch.optim.Adam(dq.parameters(), lr=0.01)
for _ in range(20):
    out = dq(x)
    loss = (out - torch.ones_like(out)).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

stats = dq.deformation_stats()
print(f"После 20 шагов: deform_scale={stats['deform_scale']:.4f}, "
      f"delta_norm={stats['delta_norm']:.4f}, "
      f"effective_shift={stats['effective_shift']:.4f}")

# %% [markdown]
# ## 6. Расстояния в пространствах

# %%
print("\n=== Расстояния между точками ===\n")

for name, points in [
    ("Триграммы (8 в R³)", generate_trigrams()),
    ("Тетраграммы (16 в R⁴)", generate_tetragrams()),
    ("Гексаграммы (64 в R⁶)", generate_hexagrams()),
    ("Октограммы (256 в R⁸)", generate_octograms()),
]:
    dists = torch.cdist(points, points)
    mask = ~torch.eye(len(points), dtype=torch.bool)
    d_vals = dists[mask]
    print(f"{name:<25}  min_dist={d_vals.min():.2f}  "
          f"mean_dist={d_vals.mean():.2f}  max_dist={d_vals.max():.2f}  "
          f"norm={points[0].norm():.2f}")

print("\nDone!")
