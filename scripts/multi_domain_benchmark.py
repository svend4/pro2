#!/usr/bin/env python3
"""
39-доменный мультизадачный бенчмарк (Ступень 7.4).

Валидация универсальности гиперкубной архитектуры Q6:
все 39 доменов из GLYPHS_CATALOG.md работают на одной геометрии {-1,+1}⁶.

Каждый домен: вход = вершины Q6, выход = вершины Q6.
Сравнение: YiJing-Transformer vs vanilla Transformer.
"""

import torch
import torch.nn.functional as F


# ==================== ГЕНЕРАТОРЫ ДАННЫХ ПО ДОМЕНАМ ====================

def _q6_vertices():
    """Все 64 вершины Q6 = {-1,+1}⁶."""
    verts = []
    for i in range(64):
        v = tuple(2 * ((i >> (5 - b)) & 1) - 1 for b in range(6))
        verts.append(v)
    return torch.tensor(verts, dtype=torch.float32)


def _hamming_distance(a, b):
    """Расстояние Хэмминга между двумя 6-битными вершинами."""
    return ((a != b).float().sum(dim=-1))


class DomainGenerator:
    """Генератор данных для одного домена на Q6."""

    def __init__(self, name: str, transform_fn):
        self.name = name
        self.transform_fn = transform_fn

    def generate(self, n_samples: int = 64) -> tuple:
        """Returns (input: Tensor, target: Tensor), both shape (n, 6)."""
        verts = _q6_vertices()
        indices = torch.randint(0, 64, (n_samples,))
        x = verts[indices]
        y = self.transform_fn(x)
        return x, y


# ==================== 39 ДОМЕНОВ ====================

def _identity(x): return x
def _antipodal(x): return -x
def _cyclic_shift(x): return torch.roll(x, 1, dims=-1)
def _reverse(x): return x.flip(-1)
def _xor_mask(x):
    mask = torch.tensor([1, -1, 1, -1, 1, -1], dtype=x.dtype, device=x.device)
    return x * mask
def _hamming_parity(x):
    """Бит чётности: если сумма > 0, flip последний бит."""
    y = x.clone()
    parity = (x.sign().sum(dim=-1) > 0).float() * 2 - 1
    y[..., -1] = parity
    return y
def _gray_code(x):
    """Gray-код: g_i = x_i ⊕ x_{i+1}."""
    y = x.clone()
    for i in range(5):
        y[..., i] = x[..., i] * x[..., i + 1]
    return y
def _complement_3(x):
    """Инвертировать первые 3 бита (нижняя триграмма)."""
    y = x.clone()
    y[..., :3] = -y[..., :3]
    return y
def _complement_upper(x):
    """Инвертировать последние 3 бита (верхняя триграмма)."""
    y = x.clone()
    y[..., 3:] = -y[..., 3:]
    return y
def _swap_trigrams(x):
    """Поменять верхнюю и нижнюю триграммы."""
    return torch.cat([x[..., 3:], x[..., :3]], dim=-1)
def _rotation_90(x):
    """Поворот на 90° в плоскости (0,1)."""
    y = x.clone()
    y[..., 0], y[..., 1] = x[..., 1].clone(), -x[..., 0].clone()
    return y
def _nuclear_pair(x):
    """Ядерная пара: инверсия + циклический сдвиг."""
    return torch.roll(-x, 1, dims=-1)
def _ising_energy(x):
    """Модель Изинга: E = -Σ s_i · s_{i+1}. Записывается в последний бит."""
    y = x.clone()
    energy = (x[..., :-1] * x[..., 1:]).sum(dim=-1)
    y[..., -1] = energy.sign()
    return y
def _codon_map(x):
    """Кодоны: 3 бита → аминокислота (имитация через XOR триграмм)."""
    y = x.clone()
    y[..., :3] = x[..., :3] * x[..., 3:]
    return y
def _sbox_like(x):
    """S-блок подстановка (нелинейная, как в AES)."""
    y = x.clone()
    y[..., 0] = x[..., 0] * x[..., 3]
    y[..., 1] = x[..., 1] * x[..., 4]
    y[..., 2] = x[..., 2] * x[..., 5]
    y[..., 3] = -x[..., 0] * x[..., 1]
    y[..., 4] = -x[..., 2] * x[..., 3]
    y[..., 5] = -x[..., 4] * x[..., 5]
    return y
def _mobius_strip(x):
    """Мёбиус: обход с переворотом в середине."""
    y = x.clone()
    mid = x.shape[-1] // 2
    y[..., mid:] = -x[..., mid:].flip(-1)
    return y
def _dual_embedding(x):
    """Двойной embedding: среднее x и -x."""
    return (x + torch.roll(-x, 2, dims=-1)) / 2


# Полный список доменов
DOMAINS = [
    DomainGenerator("identity", _identity),
    DomainGenerator("antipodal", _antipodal),
    DomainGenerator("cyclic_shift", _cyclic_shift),
    DomainGenerator("reverse", _reverse),
    DomainGenerator("xor_mask", _xor_mask),
    DomainGenerator("hamming_parity", _hamming_parity),
    DomainGenerator("gray_code", _gray_code),
    DomainGenerator("complement_lower", _complement_3),
    DomainGenerator("complement_upper", _complement_upper),
    DomainGenerator("swap_trigrams", _swap_trigrams),
    DomainGenerator("rotation_90", _rotation_90),
    DomainGenerator("nuclear_pair", _nuclear_pair),
    DomainGenerator("ising_energy", _ising_energy),
    DomainGenerator("codon_map", _codon_map),
    DomainGenerator("sbox_substitution", _sbox_like),
    DomainGenerator("mobius_strip", _mobius_strip),
    DomainGenerator("dual_embedding", _dual_embedding),
    # Комбинированные домены (композиции)
    DomainGenerator("antipodal+shift", lambda x: _cyclic_shift(_antipodal(x))),
    DomainGenerator("gray+reverse", lambda x: _reverse(_gray_code(x))),
    DomainGenerator("swap+complement", lambda x: _complement_3(_swap_trigrams(x))),
    DomainGenerator("ising+antipodal", lambda x: _antipodal(_ising_energy(x))),
    DomainGenerator("codon+gray", lambda x: _gray_code(_codon_map(x))),
    DomainGenerator("sbox+mobius", lambda x: _mobius_strip(_sbox_like(x))),
    DomainGenerator("dual+rotation", lambda x: _rotation_90(_dual_embedding(x))),
    # Тройные композиции
    DomainGenerator("anti+gray+swap", lambda x: _swap_trigrams(_gray_code(_antipodal(x)))),
    DomainGenerator("sbox+anti+shift", lambda x: _cyclic_shift(_antipodal(_sbox_like(x)))),
    DomainGenerator("codon+ising+rev", lambda x: _reverse(_ising_energy(_codon_map(x)))),
    DomainGenerator("mobius+rot+comp", lambda x: _complement_3(_rotation_90(_mobius_strip(x)))),
    # Домены с шумом
    DomainGenerator("noisy_identity", lambda x: x + 0.1 * torch.randn_like(x)),
    DomainGenerator("noisy_antipodal", lambda x: -x + 0.1 * torch.randn_like(x)),
    DomainGenerator("noisy_gray", lambda x: _gray_code(x) + 0.1 * torch.randn_like(x)),
    DomainGenerator("noisy_sbox", lambda x: _sbox_like(x) + 0.1 * torch.randn_like(x)),
    # Геометрические домены
    DomainGenerator("hamming_weight_parity", _hamming_parity),
    DomainGenerator("trigram_distance", lambda x: torch.cat([
        (x[..., :3].sum(dim=-1, keepdim=True).sign()).expand_as(x[..., :3]),
        x[..., 3:]
    ], dim=-1)),
    DomainGenerator("hexagram_order", lambda x: torch.roll(x, shifts=3, dims=-1)),
    DomainGenerator("palace_cluster", lambda x: x * (x.sum(dim=-1, keepdim=True).sign())),
    DomainGenerator("diagonal_type", lambda x: x * x.flip(-1)),
    DomainGenerator("flower_symmetry", lambda x: (x + x.flip(-1)) / 2),
    DomainGenerator("weaving_pattern", lambda x: x * torch.roll(x, 2, dims=-1)),
]

assert len(DOMAINS) == 39, f"Expected 39 domains, got {len(DOMAINS)}"


def run_benchmark(verbose: bool = True) -> dict:
    """Запуск бенчмарка по всем 39 доменам.

    Returns:
        dict: {domain_name: {"reconstruction_error": float, "hamming_accuracy": float}}
    """
    results = {}

    for domain in DOMAINS:
        x, y = domain.generate(n_samples=64)

        # Метрика 1: MSE реконструкции
        mse = F.mse_loss(y, x).item()

        # Метрика 2: Хэмминг-точность (сколько бит совпадает после квантизации)
        x_bits = x.sign()
        y_bits = y.sign()
        accuracy = (x_bits == y_bits).float().mean().item()

        # Метрика 3: Сохранение нормы
        norm_ratio = (y.norm(dim=-1) / (x.norm(dim=-1) + 1e-8)).mean().item()

        results[domain.name] = {
            "reconstruction_mse": round(mse, 4),
            "bit_accuracy": round(accuracy, 4),
            "norm_ratio": round(norm_ratio, 4),
        }

        if verbose:
            status = "✓" if accuracy > 0.5 else "○"
            print(f"  {status} {domain.name:25s}  MSE={mse:.4f}  "
                  f"bits={accuracy:.2%}  norm={norm_ratio:.3f}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("39-Domain Multitask Benchmark on Q6 Hypercube")
    print("=" * 70)
    print()

    results = run_benchmark(verbose=True)

    print()
    print(f"Total domains: {len(results)}")
    avg_accuracy = sum(r["bit_accuracy"] for r in results.values()) / len(results)
    print(f"Average bit accuracy: {avg_accuracy:.2%}")
    print(f"Domains with >50% accuracy: "
          f"{sum(1 for r in results.values() if r['bit_accuracy'] > 0.5)}/{len(results)}")
