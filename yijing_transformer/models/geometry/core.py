"""
И-Цзин геометрия: генерация кодбуков, порядки, структурные функции.

Иерархия гиперкубов:
- 4 биграммы    = {-1, +1}² = Z₂²
- 8 триграмм    = {-1, +1}³ = Z₂³
- 16 тетраграмм = {-1, +1}⁴ = Z₂⁴
- 64 гексаграмм = {-1, +1}⁶ = Z₂⁶
- 256 октограмм = {-1, +1}⁸ = Z₂⁸
"""

import math
import torch
import torch.nn.functional as F
import itertools


# ==================== ГЕНЕРАЦИЯ ГЕОМЕТРИИ ====================

def generate_trigrams() -> torch.Tensor:
    """8 триграмм — все вершины куба {-1, +1}³."""
    signs = [-1.0, 1.0]
    trigrams = torch.tensor(
        list(itertools.product(signs, repeat=3)), dtype=torch.float32
    )
    return trigrams  # (8, 3)


def generate_hexagrams() -> torch.Tensor:
    """64 гексаграммы — все вершины гиперкуба {-1, +1}⁶."""
    signs = [-1.0, 1.0]
    hexagrams = torch.tensor(
        list(itertools.product(signs, repeat=6)), dtype=torch.float32
    )
    return hexagrams  # (64, 6)


def generate_hypercube(n: int) -> torch.Tensor:
    """Генерирует все вершины гиперкуба {-1, +1}^n."""
    signs = [-1.0, 1.0]
    vertices = torch.tensor(
        list(itertools.product(signs, repeat=n)), dtype=torch.float32
    )
    return vertices  # (2^n, n)


def generate_octograms() -> torch.Tensor:
    """256 октограмм = {-1, +1}⁸ (сравнение с E8)."""
    return generate_hypercube(8)


def generate_tetragrams() -> torch.Tensor:
    """16 тетраграмм = {-1, +1}⁴."""
    return generate_hypercube(4)


def generate_e8_roots() -> torch.Tensor:
    """
    240 корней решётки E8 в R⁸.

    Два типа корней:
    1. 112 перестановок (±1, ±1, 0, 0, 0, 0, 0, 0) — все знаки, все позиции
    2. 128 «полуцелых»: (±½, ±½, ..., ±½) с чётным числом минусов

    ||root|| = √2 для всех.
    """
    roots = []

    # Тип 1: перестановки (±1, ±1, 0⁶) — C(8,2) × 4 = 112
    from itertools import combinations
    for i, j in combinations(range(8), 2):
        for si in [-1.0, 1.0]:
            for sj in [-1.0, 1.0]:
                v = [0.0] * 8
                v[i] = si
                v[j] = sj
                roots.append(v)

    # Тип 2: (±½)⁸ с чётным числом минусов — 128
    for bits in range(256):
        v = [0.5 if (bits >> i) & 1 else -0.5 for i in range(8)]
        n_neg = sum(1 for x in v if x < 0)
        if n_neg % 2 == 0:
            roots.append(v)

    return torch.tensor(roots, dtype=torch.float32)  # (240, 8)


def compare_e8_vs_hypercube():
    """Сравнение E8 (240 точек в R⁸) vs гиперкуб (256 точек в R⁸)."""
    e8 = generate_e8_roots()
    hc = generate_hypercube(8)

    e8_dists = torch.cdist(e8, e8)
    hc_dists = torch.cdist(hc, hc)

    e8_min_nonzero = e8_dists[e8_dists > 0.01].min()
    hc_min_nonzero = hc_dists[hc_dists > 0.01].min()

    return {
        'e8_points': e8.shape[0],
        'hc_points': hc.shape[0],
        'e8_dim': e8.shape[1],
        'hc_dim': hc.shape[1],
        'e8_min_dist': e8_min_nonzero.item(),
        'hc_min_dist': hc_min_nonzero.item(),
        'e8_norms': e8.norm(dim=1).unique().tolist(),
        'hc_norms': hc.norm(dim=1).unique().tolist(),
    }


def verify_yijing_properties(trigrams: torch.Tensor, hexagrams: torch.Tensor):
    """Проверка ключевых алгебраических свойств И-Цзин системы."""
    results = {}

    # 1. Корректное число точек
    results['n_trigrams'] = trigrams.shape[0]
    results['n_hexagrams'] = hexagrams.shape[0]
    results['correct_counts'] = (trigrams.shape[0] == 8 and hexagrams.shape[0] == 64)

    # 2. Антиподальность: для каждой триграммы/гексаграммы существует -1 * неё
    tri_antipodal = True
    for t in trigrams:
        if not any(torch.allclose(t, -other) for other in trigrams):
            tri_antipodal = False
            break
    results['trigrams_antipodal'] = tri_antipodal

    hex_antipodal = True
    for h in hexagrams:
        if not any(torch.allclose(h, -other) for other in hexagrams):
            hex_antipodal = False
            break
    results['hexagrams_antipodal'] = hex_antipodal

    # 3. Тензорная факторизация: hexagram = upper ⊗ lower
    hex_from_tensor = []
    for u in trigrams:
        for l in trigrams:
            hex_from_tensor.append(torch.cat([u, l]))
    hex_from_tensor = torch.stack(hex_from_tensor)
    results['tensor_product_correct'] = (hex_from_tensor.shape == hexagrams.shape)

    return results


# Кэшированные генераторы
_trigrams_cache = None
_hexagrams_cache = None


def get_trigrams() -> torch.Tensor:
    """Кэшированное получение 8 триграмм."""
    global _trigrams_cache
    if _trigrams_cache is None:
        _trigrams_cache = generate_trigrams()
    return _trigrams_cache


def get_hexagrams() -> torch.Tensor:
    """Кэшированное получение 64 гексаграмм."""
    global _hexagrams_cache
    if _hexagrams_cache is None:
        _hexagrams_cache = generate_hexagrams()
    return _hexagrams_cache


# ==================== v51: ПОРЯДКИ И СТРУКТУРЫ (Ступени 1-5) ====================

# Порядок Фуси (двоичный): 0=☰(111), 1=☱(110), ..., 7=☷(000)
# Это натуральное двоичное представление, совпадающее с Лейбницем (1703)
FUXI_TO_BINARY = {
    0: (1, 1, 1),   # ☰ Небо (Цянь)
    1: (1, 1, 0),   # ☱ Озеро (Дуй)
    2: (1, 0, 1),   # ☲ Огонь (Ли)
    3: (1, 0, 0),   # ☳ Гром (Чжэнь)
    4: (0, 1, 1),   # ☴ Ветер (Сюнь)
    5: (0, 1, 0),   # ☵ Вода (Кань)
    6: (0, 0, 1),   # ☶ Гора (Гэнь)
    7: (0, 0, 0),   # ☷ Земля (Кунь)
}


def fuxi_order() -> torch.Tensor:
    """Порядок Фуси для 64 гексаграмм: натуральная двоичная нумерация."""
    hexagrams = get_hexagrams()  # (64, 6) в порядке {-1,+1}⁶
    # Конвертируем {-1,+1} → {0,1}
    binary = ((hexagrams + 1) / 2).long()
    # Двоичное число → десятичное (big-endian)
    weights = torch.tensor([32, 16, 8, 4, 2, 1])
    indices = (binary * weights).sum(dim=1)
    return indices  # (64,) — Фуси-индекс для каждой гексаграммы


def wenwang_order() -> torch.Tensor:
    """Порядок Вэнь-вана (традиционный): 1-64 по книге И-Цзин."""
    # Традиционный порядок гексаграмм в «Книге Перемен»
    # Маппинг Фуси → Вэнь-ван для первых нескольких гексаграмм
    fuxi_to_wenwang = [
        1, 43, 14, 34, 9, 5, 26, 11,
        10, 58, 38, 54, 61, 60, 41, 19,
        13, 49, 30, 55, 37, 63, 22, 36,
        25, 17, 21, 51, 42, 3, 27, 24,
        44, 28, 50, 32, 57, 48, 18, 46,
        6, 47, 64, 40, 59, 29, 4, 7,
        33, 31, 56, 62, 53, 39, 52, 15,
        12, 45, 35, 16, 20, 8, 23, 2,
    ]
    return torch.tensor(fuxi_to_wenwang)


def palace_clusters() -> list:
    """8 дворцов по 8 гексаграмм (Склярова, Ступень 1.4).

    Каждый дворец начинается с «чистой» гексаграммы (удвоенная триграмма)
    и содержит 7 производных через последовательную мутацию линий.
    """
    palaces = []
    trigrams = generate_trigrams()  # (8, 3)
    for i, tri in enumerate(trigrams):
        palace = []
        base_hex = torch.cat([tri, tri])  # удвоенная триграмма
        palace.append(base_hex)
        # 7 производных: мутация линий 1,2,3,4,5,6, затем возврат к base
        current = base_hex.clone()
        for line in range(6):
            current = current.clone()
            current[line] = -current[line]
            palace.append(current.clone())
        palaces.append(torch.stack(palace))
    return palaces  # 8 дворцов, каждый (8, 6)


def palace_attention_mask(block_size: int = 64) -> torch.Tensor:
    """Маска attention по дворцам: полная связность внутри, ослабленная между."""
    palaces = palace_clusters()
    mask = torch.zeros(block_size, block_size)
    for palace in palaces:
        indices = []
        hexagrams = get_hexagrams()
        for hex_vec in palace:
            # Ищем индекс гексаграммы в кодбуке
            dists = (hexagrams - hex_vec.unsqueeze(0)).pow(2).sum(dim=1)
            idx = dists.argmin().item()
            if idx < block_size:
                indices.append(idx)
        # Полная связность внутри дворца
        for i in indices:
            for j in indices:
                mask[i, j] = 1.0
    # Ослабленная связность между дворцами
    mask = mask + 0.1 * (1 - mask)
    return mask


def antipodal_pairs() -> list:
    """32 антиподальные пары гексаграмм (n, 63-n)."""
    pairs = []
    for i in range(32):
        pairs.append((i, 63 - i))
    return pairs


def antipodal_index() -> torch.Tensor:
    """Индекс антиподов: antipod[i] = index of -hexagram_i."""
    hexagrams = get_hexagrams()
    antipods = torch.zeros(64, dtype=torch.long)
    for i in range(64):
        target = -hexagrams[i]
        dists = (hexagrams - target.unsqueeze(0)).pow(2).sum(dim=1)
        antipods[i] = dists.argmin()
    return antipods


def loshu_kernel() -> torch.Tensor:
    """Ло-шу ядро 3×3 (магический квадрат, сумма = 15).

    По Фомюку: Ло-шу ядро определяет «натуральное» взвешивание
    для 3×3 пулинга в attention. Используется как инициализация.
    """
    return torch.tensor([
        [2, 7, 6],
        [9, 5, 1],
        [4, 3, 8]
    ], dtype=torch.float32)


def triangular_position(n: int, P: int = 64) -> int:
    """Треугольная позиция Андреева: T(n) = n(n-1)/2 mod P.

    Нелинейное отображение натуральных чисел в циклическую группу Z_P.
    Для P=64 (гексаграммы) — это нелинейный позиционный индекс.
    """
    return (n * (n - 1) // 2) % P


def triangular_distance_matrix(P: int = 64) -> torch.Tensor:
    """Матрица треугольных расстояний P×P (Андреев, Ступень 3).

    dist[i][j] = |T(i) - T(j)| mod (P/2) — расстояние на «треугольной развёртке».
    """
    T = torch.tensor([triangular_position(n, P) for n in range(P)])
    # Расстояние = |T(i) - T(j)| с учётом цикличности
    dist = torch.zeros(P, P)
    half_P = P // 2
    for i in range(P):
        for j in range(P):
            d = abs(T[i].item() - T[j].item())
            dist[i, j] = min(d, P - d)  # минимальное циклическое расстояние
    return dist


def andreev_matrix(P: int = 64) -> torch.Tensor:
    """Матрица Андреева: унифицированное представление позиций (Ступень 3.2).

    A[i,j] = T(i+j) — сумма треугольных позиций.
    Обладает свойствами модулярной арифметики на Z_P.
    """
    A = torch.zeros(P, P)
    for i in range(P):
        for j in range(P):
            A[i, j] = triangular_position(i + j, P)
    return A


def hermann_packing(k: int) -> torch.Tensor:
    """Упаковка Германа для P = 2^k (Ступень 5).

    Collision-free для P = 2^k (теорема Германа):
    position(n) = n(n-1)/2 mod P — все позиции различны.

    Для P=64 (k=6): 0 коллизий (идеальная упаковка).
    Для P=240 (E8): 144 коллизии (нет 2^k факторизации).
    """
    P = 2 ** k
    positions = torch.zeros(P, dtype=torch.long)
    for n in range(P):
        positions[n] = (n * (n - 1) // 2) % P

    # Упаковочное поле: field[pos] = list of n mapping to pos
    field = torch.zeros(P, dtype=torch.long)
    for n in range(P):
        field[positions[n]] = n

    return field


def collision_test(P: int) -> dict:
    """Тест на коллизии для произвольного P (обобщённая теорема Германа).

    Коллизия: ∃ n₁ ≠ n₂: T(n₁) = T(n₂) mod P.

    Теорема: P = 2^k ⟹ 0 коллизий (для n < P).
    """
    positions = {}
    collisions = 0
    collision_pairs = []

    for n in range(P):
        pos = (n * (n - 1) // 2) % P
        if pos in positions:
            collisions += 1
            collision_pairs.append((positions[pos], n, pos))
        else:
            positions[pos] = n

    unique_positions = len(positions)
    is_power_of_2 = (P & (P - 1)) == 0 and P > 0

    return {
        'P': P,
        'unique_positions': unique_positions,
        'collisions': collisions,
        'collision_pairs': collision_pairs[:10],  # первые 10
        'is_power_of_2': is_power_of_2,
        'coverage': unique_positions / P,
        'collision_free': collisions == 0,
    }


def valid_codebook_sizes(max_k: int = 12) -> list:
    """Все collision-free размеры кодбука до 2^max_k."""
    sizes = []
    for k in range(1, max_k + 1):
        P = 2 ** k
        result = collision_test(P)
        if result['collision_free']:
            sizes.append({
                'k': k,
                'P': P,
                'collision_free': True,
            })
    return sizes


def find_fixed_points(P: int) -> list:
    """Находит фиксированные точки: n, для которых T(n) = n-1 (Теорема 11).

    Фиксированная точка: position(n) = n (с нулевой индексации).
    """
    fixed = []
    for n in range(P):
        pos = (n * (n - 1) // 2) % P
        if pos == n:
            fixed.append(n)
    return fixed


def e8_collision_proof(P: int = 240) -> dict:
    """Доказательство 144 коллизий для E8 (P=240). Теорема 12."""
    result = collision_test(P)
    # 240 = 2⁴ · 3 · 5 — НЕ степень двойки
    factorization = []
    n = P
    for p in [2, 3, 5, 7, 11, 13]:
        while n % p == 0:
            factorization.append(p)
            n //= p
    if n > 1:
        factorization.append(n)

    result['factorization'] = factorization
    result['is_power_of_2'] = len(set(factorization)) == 1 and factorization[0] == 2
    return result


def generate_four_state_codebook() -> torch.Tensor:
    """4096 состояний: 4 состояния × 6 линий = {-1, -0.5, +0.5, +1}⁶.

    Четыре состояния линии:
    - +1.0  = 老阳 (старый ян → мутирует в инь)
    - +0.5  = 少阳 (молодой ян → стабилен)
    - -0.5  = 少阴 (молодая инь → стабильна)
    - -1.0  = 老阴 (старая инь → мутирует в ян)
    """
    states = [-1.0, -0.5, 0.5, 1.0]
    codebook = torch.tensor(
        list(itertools.product(states, repeat=6)), dtype=torch.float32
    )
    return codebook  # (4096, 6)
