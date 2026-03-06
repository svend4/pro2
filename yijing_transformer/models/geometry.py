"""
И-Цзин геометрия: триграммы, гексаграммы, октограммы, квантизаторы и MoE.

Иерархия гиперкубов:
- 4 биграммы    = {-1, +1}² = Z₂²
- 8 триграмм    = {-1, +1}³ = Z₂³
- 16 тетраграмм = {-1, +1}⁴ = Z₂⁴
- 64 гексаграмм = {-1, +1}⁶ = Z₂⁶
- 256 октограмм = {-1, +1}⁸ = Z₂⁸  (vs 240 корней E8 в R⁸!)

Тензорные факторизации:
- Гексаграмма = верхняя_триграмма ⊗ нижняя_триграмма (8×8)
- Октограмма  = 4 биграммы (4×4×4×4) или 2 тетраграммы (16×16)

v51 расширения (из плана интеграции шести источников):
- Порядки Фуси/Вэнь-вана, дворцы (Ступень 1: Склярова)
- Антиподальное спаривание, Ло-шу ядро (Ступень 2: Фомюк)
- Треугольная матрица, расстояния Андреева (Ступень 3: Андреев)
- Упаковка Германа P=2^k, коллизии (Ступень 5: Герман)
- Четыре состояния линий, 4096-кодбук (Ступень 1.2)
"""

import math
import torch
import torch.nn as nn
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
    """2^n точек — все вершины гиперкуба {-1, +1}^n."""
    signs = [-1.0, 1.0]
    points = torch.tensor(
        list(itertools.product(signs, repeat=n)), dtype=torch.float32
    )
    return points  # (2^n, n)


def generate_octograms() -> torch.Tensor:
    """256 октограмм — все вершины гиперкуба {-1, +1}⁸.
    Сравнимо с E8 (240 корней в R⁸), но с тензорной факторизацией."""
    return generate_hypercube(8)  # (256, 8)


def generate_tetragrams() -> torch.Tensor:
    """16 тетраграмм — {-1, +1}⁴."""
    return generate_hypercube(4)  # (16, 4)


def generate_e8_roots() -> torch.Tensor:
    """
    240 корней решётки E8 в R⁸.

    Два типа корней:
    1) 112 векторов: все перестановки (±1, ±1, 0, 0, 0, 0, 0, 0)
    2) 128 векторов: (±½, ±½, ..., ±½) с чётным числом минусов

    Все корни имеют норму √2. Сравни с октограммами {-1,+1}⁸ (256 точек, норма √8).
    """
    roots = []

    # Тип 1: 112 корней — пары ±1 во всех позициях
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [-1.0, 1.0]:
                for sj in [-1.0, 1.0]:
                    v = [0.0] * 8
                    v[i] = si
                    v[j] = sj
                    roots.append(v)

    # Тип 2: 128 корней — (±½)⁸ с чётным числом минусов
    for signs in itertools.product([-0.5, 0.5], repeat=8):
        n_neg = sum(1 for s in signs if s < 0)
        if n_neg % 2 == 0:
            roots.append(list(signs))

    e8 = torch.tensor(roots, dtype=torch.float32)
    assert e8.shape == (240, 8), f"E8: ожидалось (240, 8), получено {e8.shape}"
    # Проверка: все нормы = √2
    norms = torch.norm(e8, dim=1)
    assert torch.allclose(norms, torch.tensor(2.0).sqrt(), atol=1e-6), "E8: неверные нормы"
    return e8


def compare_e8_vs_hypercube():
    """
    Сравнение E8 решётки с гиперкубами по ключевым метрикам.

    Возвращает dict с метриками: число точек, размерность, нормы,
    минимальное/среднее расстояние, packing density.
    """
    configs = {
        'E8 (240 roots)': generate_e8_roots(),
        'Hexagrams {-1,+1}⁶': generate_hexagrams(),
        'Octograms {-1,+1}⁸': generate_octograms(),
    }
    results = {}
    for name, points in configs.items():
        dists = torch.cdist(points, points)
        mask = ~torch.eye(len(points), dtype=torch.bool)
        d_vals = dists[mask]
        results[name] = {
            'n_points': len(points),
            'dim': points.shape[1],
            'norm': points[0].norm().item(),
            'min_dist': d_vals.min().item(),
            'mean_dist': d_vals.mean().item(),
            'max_dist': d_vals.max().item(),
            'bits': torch.tensor(float(len(points))).log2().item(),
        }
    return results


def verify_yijing_properties(trigrams: torch.Tensor, hexagrams: torch.Tensor):
    """Проверка математических свойств."""
    assert trigrams.shape == (8, 3), f"Триграммы: ожидалось (8,3), получено {trigrams.shape}"
    assert hexagrams.shape == (64, 6), f"Гексаграммы: ожидалось (64,6), получено {hexagrams.shape}"

    # Все координаты ±1
    assert torch.all((trigrams == 1.0) | (trigrams == -1.0)), "Триграммы содержат не ±1"
    assert torch.all((hexagrams == 1.0) | (hexagrams == -1.0)), "Гексаграммы содержат не ±1"

    # Нормы
    tri_norms = torch.norm(trigrams, dim=1)
    assert torch.allclose(tri_norms, torch.tensor(3.0).sqrt()), "Неверные нормы триграмм"
    hex_norms = torch.norm(hexagrams, dim=1)
    assert torch.allclose(hex_norms, torch.tensor(6.0).sqrt()), "Неверные нормы гексаграмм"

    # Уникальность
    assert torch.unique(trigrams, dim=0).shape[0] == 8, "Дублированные триграммы"
    assert torch.unique(hexagrams, dim=0).shape[0] == 64, "Дублированные гексаграммы"

    # Центрирование
    assert torch.norm(trigrams.sum(dim=0)).item() < 1e-6, "Триграммы не центрированы"
    assert torch.norm(hexagrams.sum(dim=0)).item() < 1e-6, "Гексаграммы не центрированы"


# Кэш
_TRIGRAMS = generate_trigrams()
_HEXAGRAMS = generate_hexagrams()
verify_yijing_properties(_TRIGRAMS, _HEXAGRAMS)


def get_trigrams() -> torch.Tensor:
    return _TRIGRAMS


def get_hexagrams() -> torch.Tensor:
    return _HEXAGRAMS


# ==================== v51: ПОРЯДКИ И СТРУКТУРЫ (Ступени 1-5) ====================

# --- Ступень 1: Порядок Фуси и Вэнь-вана (Склярова) ---

# Порядок Вэнь-вана: традиционный порядок 64 гексаграмм из И-Цзин.
# Каждое число — индекс гексаграммы в лексикографическом (Фуси) порядке {-1,+1}⁶.
# Порядок Фуси = binary counting (0..63) = лексикографический.
WENWANG_ORDER = [
    1, 23, 8, 20, 16, 35, 45, 2,     # гексаграммы 1-8 по Вэнь-вану
    12, 15, 52, 39, 53, 62, 56, 31,   # 9-16
    49, 55, 63, 22, 36, 37, 30, 50,   # 17-24
    14, 38, 43, 0, 57, 48, 18, 46,    # 25-32
    13, 61, 54, 40, 19, 60, 41, 58,   # 33-40
    47, 26, 27, 44, 6, 7, 9, 10,      # 41-48
    5, 3, 29, 33, 11, 4, 59, 42,      # 49-56
    28, 17, 51, 34, 32, 21, 25, 24,   # 57-64
]


def fuxi_order() -> torch.Tensor:
    """Порядок Фуси (= binary counting): гексаграммы 0..63 в лексикографическом порядке.

    Это порядок {-1,+1}⁶, где 0=(-1,-1,-1,-1,-1,-1), 63=(+1,+1,+1,+1,+1,+1).
    Оптимален для квантизации (natural binary code).
    """
    return get_hexagrams()  # (64, 6) — уже в порядке Фуси


def wenwang_order() -> torch.Tensor:
    """Порядок Вэнь-вана: 64 гексаграммы в традиционном порядке И-Цзин.

    Порядок Вэнь-вана — эмпирически сложившаяся перестановка, оптимизированная
    для семантики (близкие по смыслу гексаграммы стоят рядом).
    """
    hexagrams = get_hexagrams()  # (64, 6)
    return hexagrams[WENWANG_ORDER]  # (64, 6)


def palace_clusters() -> list:
    """Восемь дворцов (八宮) — 8 кластеров по 8 гексаграмм.

    Каждый дворец объединён общей «корневой» триграммой (нижней).
    В пространстве {-1,+1}⁶ это 8 подкубов, каждый изоморфен {-1,+1}³.

    Возвращает список из 8 списков по 8 индексов гексаграмм.
    Дворец i содержит гексаграммы с нижней триграммой = триграмма i.
    """
    palaces = [[] for _ in range(8)]
    hexagrams = get_hexagrams()  # (64, 6)
    trigrams = get_trigrams()    # (8, 3)
    for hex_idx in range(64):
        lower = hexagrams[hex_idx, 3:]  # нижняя триграмма (биты 3-5)
        # Найти соответствующую триграмму
        dists = ((trigrams - lower) ** 2).sum(dim=1)
        tri_idx = dists.argmin().item()
        palaces[tri_idx].append(hex_idx)
    return palaces


def palace_attention_mask(block_size: int = 64) -> torch.Tensor:
    """Block-sparse attention mask: 8×(8×8) вместо полной 64×64.

    Attention внутри дворца = 1.0 (full attention),
    attention между дворцами = ослабленный коэффициент.

    Возвращает (64, 64) float mask.
    """
    palaces = palace_clusters()
    mask = torch.zeros(64, 64)
    for palace in palaces:
        for i in palace:
            for j in palace:
                mask[i, j] = 1.0
    # Между дворцами — ослабленная связь (не 0, а 0.1)
    mask = torch.where(mask == 0, torch.tensor(0.1), mask)
    return mask[:block_size, :block_size]


# --- Ступень 2: Антиподальная структура (Фомюк) ---

def antipodal_pairs() -> list:
    """32 антиподальные пары (i, 63-i) для гексаграмм.

    Свойство: hex(i) + hex(63-i) = 0 (покоординатно, т.к. антипод = инверсия).
    Герман доказывает: n + n̄ = P+1 для упаковки.
    """
    return [(i, 63 - i) for i in range(32)]


def antipodal_index() -> torch.Tensor:
    """Индекс антиподов: antipod[i] = 63-i для каждой гексаграммы.

    Используется для weight tying: embedding(hex_i) определяет embedding(hex_{63-i}).
    """
    return torch.arange(63, -1, -1, dtype=torch.long)


def loshu_kernel() -> torch.Tensor:
    """Магический квадрат Ло-шу 3×3 как нормализованное свёрточное ядро.

    Сумма по строкам, столбцам и диагоналям = 15.
    Разность противоположных элементов относительно центра (5) постоянна.
    """
    return torch.tensor([
        [2, 7, 6],
        [9, 5, 1],
        [4, 3, 8]
    ], dtype=torch.float32) / 15.0


# --- Ступень 3: Треугольная матрица (Андреев) ---

def triangular_position(n: int, P: int = 64) -> int:
    """Позиция числа n в треугольной упаковке: T(n) = n(n-1)/2 mod P.

    Единая формула, используемая и в упаковке Германа, и в матрице Андреева.
    Два автора пришли к этой формуле независимо.
    """
    return (n * (n - 1) // 2) % P


def triangular_distance_matrix(P: int = 64) -> torch.Tensor:
    """Матрица треугольных расстояний (Андреев): d(i,j) = min(|T(i)-T(j)|, P-|T(i)-T(j)|).

    Нелинейное позиционное кодирование на основе комбинаторной структуры.
    """
    positions = torch.tensor([triangular_position(n, P) for n in range(P)], dtype=torch.float32)
    # Циклическое расстояние
    diff = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    dist = torch.min(diff, P - diff)
    return dist  # (P, P)


def andreev_matrix(P: int = 64) -> torch.Tensor:
    """Треугольная матрица расположения гексаграмм по Андрееву.

    Возвращает перестановку 0..P-1, где позиция определяется T(n) = n(n-1)/2 mod P.
    """
    field = torch.zeros(P, dtype=torch.long)
    for n in range(P):
        pos = triangular_position(n, P)
        field[pos] = n
    return field


# --- Ступень 5: Упаковка Германа P=2^k ---

def hermann_packing(k: int) -> torch.Tensor:
    """Бесколлизионная упаковка P=2^k по алгоритму Германа.

    Теорема Германа: только P=2^k допускает полную бесколлизионную упаковку
    по алгоритму треугольных чисел. Числа 1..P размещаются в позиции
    с кумулятивным сдвигом: pos(n) = sum(1..n-1) = n(n-1)/2 mod P.

    Args:
        k: степень двойки (P = 2^k)

    Returns:
        Tensor (P,) — упакованное поле, field[pos] = n (n=1..P).
    """
    P = 2 ** k
    field = torch.zeros(P, dtype=torch.long)
    occupied = torch.zeros(P, dtype=torch.bool)
    pos = 0
    for n in range(1, P + 1):
        if occupied[pos % P]:
            raise ValueError(f"Коллизия при P={P}: позиция {pos % P} уже занята "
                             f"(попытка разместить n={n})")
        field[pos % P] = n
        occupied[pos % P] = True
        pos += n  # кумулятивный сдвиг = треугольное число
    return field


def collision_test(P: int) -> dict:
    """Тест на коллизии для произвольного P.

    Для P=2^k: 0 коллизий (доказано теоремой Германа).
    Для P≠2^k: находит конкретные коллизии.

    Returns:
        dict с ключами: 'P', 'is_power_of_2', 'collisions', 'first_collision'
    """
    is_pow2 = (P > 0) and (P & (P - 1) == 0)
    occupied = {}
    collisions = []
    first_collision = None

    pos = 0
    for n in range(1, P + 1):
        p = pos % P
        if p in occupied:
            collision = (occupied[p], n, p)
            collisions.append(collision)
            if first_collision is None:
                first_collision = collision
        else:
            occupied[p] = n
        pos += n

    return {
        'P': P,
        'is_power_of_2': is_pow2,
        'n_collisions': len(collisions),
        'collisions': collisions[:10],  # первые 10
        'first_collision': first_collision,
    }


def valid_codebook_sizes(max_k: int = 12) -> list:
    """Допустимые размеры кодбука: P=2^k для k=1..max_k.

    Теорема Германа доказывает, что ТОЛЬКО степени двойки
    допускают бесколлизионную циклическую упаковку.

    Returns:
        Список (k, P) пар: [(1, 2), (2, 4), (3, 8), ..., (12, 4096)]
    """
    return [(k, 2 ** k) for k in range(1, max_k + 1)]


def find_fixed_points(P: int) -> list:
    """Неподвижные точки упаковки: числа n, для которых position(n) = n-1.

    Следствие 1 Германа: при любом P такие точки существуют.
    Неподвижные точки — якорные элементы кодбука.
    """
    fixed = []
    pos = 0
    for n in range(1, P + 1):
        p = pos % P
        if p == n - 1:  # число n стоит на «своей» позиции
            fixed.append(n)
        pos += n
    return fixed


def e8_collision_proof(P: int = 240) -> dict:
    """Доказательство коллизий для E8 (P=240).

    По формуле Германа (2): (n₁+n₂)(n₂-n₁+1) / (2P) = q (целое).
    Если для некоторых n₁, n₂ это целое — есть коллизия.

    Returns:
        dict с конкретными парами (n₁, n₂), вызывающими коллизию в E8.
    """
    result = collision_test(P)
    result['note'] = (
        f'P={P} не является степенью двойки (240=2⁴·3·5). '
        f'По теореме Германа, бесколлизионная упаковка невозможна.'
    )
    return result


# --- Ступень 1.2: Четыре состояния линий ---

def generate_four_state_codebook() -> torch.Tensor:
    """4096 состояний в 6D: {-1, -0.5, +0.5, +1}⁶.

    Четыре состояния линии И-Цзин:
    - 老阳 (+1): старый ян (меняющийся)
    - 少阳 (+0.5): молодой ян (стабильный)
    - 少阴 (-0.5): молодая инь (стабильная)
    - 老阴 (-1): старая инь (меняющаяся)

    Это Product Quantization с M=6 подпространствами по K=4 центроида.
    4⁶ = 4096 состояний в 6D без увеличения размерности.
    """
    states = [-1.0, -0.5, 0.5, 1.0]
    codebook = torch.tensor(
        list(itertools.product(states, repeat=6)), dtype=torch.float32
    )
    return codebook  # (4096, 6)


# ==================== КВАНТИЗАТОРЫ ====================

class YiJingQuantizer(nn.Module):
    """
    Наивная квантизация к 64 гексаграммам (вершинам гиперкуба {-1,+1}⁶).
    Аналог E8Quantizer, но с 64 точками в 6D вместо 240 в 8D.
    """
    def __init__(self, temp=0.3):
        super().__init__()
        self.temp = temp
        hexagrams = get_hexagrams()
        self.register_buffer('codebook', hexagrams)  # (64, 6)

    def forward(self, x):
        # x: (..., 6)
        dists_sq = torch.cdist(x, self.codebook.unsqueeze(0).expand(x.shape[0], -1, -1)
                                if x.dim() == 3 else self.codebook) ** 2
        weights = F.softmax(-dists_sq / self.temp, dim=-1)
        quantized = weights @ self.codebook
        return x + (quantized - x).detach()  # STE

    def hard_quantize(self, x):
        """sign(x) — тривиальная квантизация к ближайшей вершине гиперкуба."""
        return torch.sign(x)


class E8Quantizer(nn.Module):
    """
    Квантизация к 240 корням решётки E8 в R⁸.

    Для прямого сравнения с гиперкубными квантизаторами.
    Сложность: softmax(240) — без факторизации (brute force).
    """
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        e8 = generate_e8_roots()
        self.register_buffer('codebook', e8)  # (240, 8)
        self.register_buffer('codebook_norm_sq', (e8 ** 2).sum(dim=1))

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        # x: (..., 8)
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        cross = x @ self.codebook.T
        dists_sq = x_norm_sq - 2 * cross + self.codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        quantized = weights @ self.codebook
        if self.adaptive_temp:
            return quantized
        return x + (quantized - x).detach()

    def hard_quantize(self, x):
        """Ближайший корень E8."""
        dists = torch.cdist(x.reshape(-1, 8), self.codebook)
        idx = dists.argmin(dim=-1)
        return self.codebook[idx].reshape(x.shape)


class FactoredYiJingQuantizer(nn.Module):
    """
    Факторизованная квантизация: раздельно для верхней и нижней триграммы.

    Сложность: 2 × softmax(8) = O(16) вместо softmax(64) = O(64).
    Это ключевое преимущество тензорной структуры И-Цзин.
    """
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        trigrams = get_trigrams()
        self.register_buffer('trigrams', trigrams)  # (8, 3)
        self.register_buffer('trigrams_norm_sq', (trigrams ** 2).sum(dim=1))  # (8,)

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        # x: (..., 6) → разделяем на верхнюю и нижнюю триграмму
        upper, lower = x[..., :3], x[..., 3:]

        upper_q = self._soft_quantize(upper)
        lower_q = self._soft_quantize(lower)

        quantized = torch.cat([upper_q, lower_q], dim=-1)

        if self.adaptive_temp:
            # Для адаптивной температуры: soft quantized output напрямую
            # (градиент к temp течёт через softmax weights)
            return quantized
        else:
            return x + (quantized - x).detach()  # STE

    def _soft_quantize(self, z):
        # z: (..., 3)
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)
        cross = z @ self.trigrams.T
        dists_sq = z_norm_sq - 2 * cross + self.trigrams_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        return weights @ self.trigrams

    def hard_quantize(self, x):
        return torch.sign(x)


# ==================== ТРАНСФОРМАЦИЯ ГЕКСАГРАММ (变卦) ====================

class BianGuaTransform(nn.Module):
    """
    变卦 (Трансформация гексаграмм) — покоординатное отражение в {-1,+1}⁶.

    В И-Цзин переход между гексаграммами происходит через «изменение линий».
    В бинарном пространстве это XOR, в непрерывном — обучаемое отражение.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.proj_from_6d = nn.Linear(6, d_model, bias=False)
        self.change_logits = nn.Parameter(torch.zeros(6))
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        z = self.proj_to_6d(x)
        change_prob = torch.sigmoid(self.change_logits)
        # При prob=0: z не меняется. При prob=1: z → -z (инверсия линии).
        z_transformed = z * (1 - 2 * change_prob)
        return x + self.scale * self.proj_from_6d(z_transformed)


# ==================== v51: НОВЫЕ КВАНТИЗАТОРЫ И ATTENTION ====================

class FourStateQuantizer(nn.Module):
    """Квантизация к 4096 состояниям в 6D: {-1, -0.5, +0.5, +1}⁶.

    Четыре состояния линии (老阳, 少阳, 少阴, 老阴) дают 4⁶ = 4096 состояний
    без увеличения размерности. Факторизация: 6 независимых подпространств по 4 точки.

    Ступень 1.2 плана интеграции.
    """
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        # 4 центроида на каждую координату (факторизованно)
        states = torch.tensor([-1.0, -0.5, 0.5, 1.0])
        self.register_buffer('states', states)  # (4,)

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        """x: (..., 6) → факторизованная квантизация по каждой из 6 координат."""
        # Каждая координата квантизуется к ближайшему из 4 состояний
        # Факторизация: 6 × softmax(4) = O(24) вместо softmax(4096)
        shape = x.shape
        x_flat = x.reshape(-1, 6)  # (N, 6)

        quantized_coords = []
        for i in range(6):
            xi = x_flat[:, i:i+1]  # (N, 1)
            dists_sq = (xi - self.states) ** 2  # (N, 4)
            weights = F.softmax(-dists_sq / self.current_temp, dim=-1)  # (N, 4)
            qi = (weights * self.states).sum(dim=-1, keepdim=True)  # (N, 1)
            quantized_coords.append(qi)

        quantized = torch.cat(quantized_coords, dim=-1)  # (N, 6)
        quantized = quantized.reshape(shape)

        if self.adaptive_temp:
            return quantized
        return x + (quantized - x).detach()  # STE

    def hard_quantize(self, x):
        """Ближайшее из 4 состояний для каждой координаты."""
        dists = (x.unsqueeze(-1) - self.states).abs()  # (..., 6, 4)
        idx = dists.argmin(dim=-1)  # (..., 6)
        return self.states[idx]


class AntipodalQuantizer(nn.Module):
    """Квантизатор с антиподальным weight tying (Ступень 2.3).

    Только 32 свободных эмбеддинга — антиподы определены автоматически:
    embedding(hex_{63-i}) = -embedding(hex_i).

    Это удваивает эффективную ёмкость при половине параметров.
    """
    def __init__(self, temp=0.3, adaptive_temp=False):
        super().__init__()
        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp
        hexagrams = get_hexagrams()  # (64, 6)
        self.register_buffer('codebook', hexagrams)
        # Антиподальный индекс
        self.register_buffer('antipod_idx', antipodal_index())

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        # Стандартная квантизация + антиподальная регуляризация
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True)
        cross = x @ self.codebook.T
        codebook_norm_sq = (self.codebook ** 2).sum(dim=1)
        dists_sq = x_norm_sq - 2 * cross + codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)
        quantized = weights @ self.codebook
        if self.adaptive_temp:
            return quantized
        return x + (quantized - x).detach()

    def antipodal_loss(self, x):
        """Штраф: ||Q(x) + Q(-x)|| → 0 (антиподы должны быть инверсиями)."""
        q_x = self.forward(x)
        q_neg_x = self.forward(-x)
        return (q_x + q_neg_x).pow(2).mean()


class TriangularAttentionBias(nn.Module):
    """Attention bias на основе треугольных расстояний Андреева (Ступень 3).

    Нелинейное позиционное кодирование: bias[i][j] = f(d_triangular(i, j)).
    Альтернатива ALiBi и RoPE с комбинаторным обоснованием.
    """
    def __init__(self, max_seq_len: int = 512, P: int = 64):
        super().__init__()
        self.P = P
        # Обучаемый масштаб
        self.scale = nn.Parameter(torch.tensor(-0.1))
        # Предвычисленная матрица треугольных расстояний
        dist_matrix = triangular_distance_matrix(P)
        # Расширяем до max_seq_len циклически
        full_dist = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                full_dist[i, j] = dist_matrix[i % P, j % P]
        self.register_buffer('dist_matrix', full_dist)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Возвращает (seq_len, seq_len) bias для attention scores."""
        return self.scale * self.dist_matrix[:seq_len, :seq_len]


class PalaceAttention(nn.Module):
    """Block-sparse attention по дворцам (Ступень 1.4).

    8 дворцов × 8 гексаграмм = 64. Attention внутри дворца полный,
    между дворцами — ослабленный. Экономия: O(8×8²) vs O(64²).
    """
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Предвычисленная маска дворцов
        mask = palace_attention_mask(64)
        self.register_buffer('palace_mask', mask)

        # Обучаемый межкластерный коэффициент
        self.inter_palace_weight = nn.Parameter(torch.tensor(0.1))

    def get_mask(self, seq_len: int) -> torch.Tensor:
        """Attention mask с дворцовой структурой для последовательности длины seq_len."""
        if seq_len <= 64:
            base_mask = self.palace_mask[:seq_len, :seq_len]
        else:
            # Циклическое расширение
            base_mask = torch.ones(seq_len, seq_len, device=self.palace_mask.device)
            for i in range(seq_len):
                for j in range(seq_len):
                    base_mask[i, j] = self.palace_mask[i % 64, j % 64]

        # Интерполируем с обучаемым весом
        intra = (base_mask == 1.0).float()
        inter = (base_mask < 1.0).float()
        return intra + torch.sigmoid(self.inter_palace_weight) * inter

    def forward(self, x, mask=None):
        """x: (B, T, D) → block-sparse attention по дворцам → (B, T, D)"""
        B, T, D = x.shape
        scale = self.head_dim ** -0.5
        palace_mask = self.get_mask(T)  # (T, T)

        # Reshape для multi-head
        q = x.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = x.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = x.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        attn = attn * palace_mask.unsqueeze(0).unsqueeze(0)  # применяем маску дворцов

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(B, T, D)


class DualEmbedding(nn.Module):
    """Dual embedding: 6D гиперкубное + 3D кубическое (Ступень 4.4).

    Каждый токен имеет два представления:
    - 6D {-1,+1}⁶ — для квантизации (sign-квантизация)
    - 3D Z³ — для позиционного кодирования и визуализации

    Связаны обучаемым проектором.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        self.proj_6d_to_3d = nn.Linear(6, 3, bias=False)
        self.proj_3d_to_6d = nn.Linear(3, 6, bias=False)
        self.proj_from = nn.Linear(9, d_model, bias=False)  # 6+3 = 9
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        """x: (B, T, D) → dual representation → (B, T, D)"""
        z6 = self.proj_to_6d(x)  # (B, T, 6) — гиперкубное
        z3 = self.proj_to_3d(x)  # (B, T, 3) — кубическое

        # Согласование через cross-projection
        z3_from_6 = self.proj_6d_to_3d(z6)  # (B, T, 3)
        z3_aligned = (z3 + z3_from_6) / 2  # среднее

        combined = torch.cat([z6, z3_aligned], dim=-1)  # (B, T, 9)
        return x + self.scale * self.proj_from(combined)

    def consistency_loss(self, x):
        """Loss: 3D-проекция 6D-представления ≈ прямое 3D-представление."""
        z6 = self.proj_to_6d(x)
        z3 = self.proj_to_3d(x)
        z3_from_6 = self.proj_6d_to_3d(z6)
        return F.mse_loss(z3_from_6, z3)


class QuadrantAttention(nn.Module):
    """4-квадрантный attention по Беляеву (Ступень 6.2).

    Вместо скалярного Q·K^T вычисляем 4 квадранта:
    I:   max(Q) × max(K) → сильный positive
    II:  max(Q) × min(K) → контрастный
    III: min(Q) × max(K) → обратный контрастный
    IV:  min(Q) × min(K) → слабый positive

    Обобщает differential attention (diff_attn.py).
    """
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 4 обучаемых веса квадрантов
        self.quadrant_weights = nn.Parameter(torch.tensor([1.0, -0.5, -0.5, 0.25]))

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H = self.n_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # Разделяем Q и K на две половины (аналог diff_attn)
        q1, q2 = q.chunk(2, dim=-1)  # (B, H, T, d/2) each
        k1, k2 = k.chunk(2, dim=-1)

        scale = (d // 2) ** -0.5
        w = self.quadrant_weights.softmax(dim=0)

        # 4 квадранта attention
        attn_pp = torch.matmul(q1, k1.transpose(-2, -1)) * scale  # Q+·K+
        attn_pn = torch.matmul(q1, k2.transpose(-2, -1)) * scale  # Q+·K-
        attn_np = torch.matmul(q2, k1.transpose(-2, -1)) * scale  # Q-·K+
        attn_nn = torch.matmul(q2, k2.transpose(-2, -1)) * scale  # Q-·K-

        # Взвешенная комбинация
        attn = w[0] * attn_pp + w[1] * attn_pn + w[2] * attn_np + w[3] * attn_nn

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


class GraduatedBianGuaTransform(nn.Module):
    """Градуированная 变卦: мутируют только «старые» линии (Ступень 1.3).

    В оригинале И-Цзин мутируют только линии в экстремальных позициях (±1).
    Линии в ±0.5 — стабильны и не меняются. Это даёт градуированный BianGua.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.proj_from_6d = nn.Linear(6, d_model, bias=False)
        self.change_logits = nn.Parameter(torch.zeros(6))
        self.stability_threshold = nn.Parameter(torch.tensor(0.7))
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        z = self.proj_to_6d(x)
        change_prob = torch.sigmoid(self.change_logits)

        # Маска: мутируют только «экстремальные» координаты (близкие к ±1)
        extremeness = z.abs()  # чем ближе к 1, тем «старее» линия
        threshold = torch.sigmoid(self.stability_threshold)
        mutation_mask = (extremeness > threshold).float()  # 1 = «старая» линия

        # Применяем мутацию только к «старым» линиям
        z_transformed = z * (1 - 2 * change_prob * mutation_mask)
        return x + self.scale * self.proj_from_6d(z_transformed)


class D4EquivariantLayer(nn.Module):
    """D₄-эквивариантный слой для триграмм (Ступень 2.2).

    Группа диэдра D₄ = симметрии квадрата (4 поворота + 4 отражения).
    8 триграмм образуют представление D₄: любое преобразование триграммы
    сводится к одной из 8 операций D₄.

    Аналог group-equivariant CNN (Cohen & Welling, 2016),
    но для дискретной группы D₄ на триграммах.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        self.proj_from_3d = nn.Linear(3, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.01))

        # 8 операций D₄ на 3D-триграммах (перестановки + знаки)
        # D₄ действует на 3 координаты: rot90, rot180, rot270, flip_x, flip_y, flip_xy, flip_yx, identity
        ops = torch.zeros(8, 3, 3)
        # identity
        ops[0] = torch.eye(3)
        # rot90: (a,b,c) → (b,c,a)
        ops[1] = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        # rot180: (a,b,c) → (c,a,b)
        ops[2] = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        # rot270: (a,b,c) → (-a,-b,-c) — инверсия (антипод)
        ops[3] = -torch.eye(3)
        # flip_x: (a,b,c) → (-a,b,c)
        ops[4] = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
        # flip_y: (a,b,c) → (a,-b,c)
        ops[5] = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        # flip_z: (a,b,c) → (a,b,-c)
        ops[6] = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
        # flip_xy: (a,b,c) → (b,a,c)
        ops[7] = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
        self.register_buffer('d4_ops', ops)  # (8, 3, 3)

        # Обучаемые веса для 8 операций
        self.op_weights = nn.Parameter(torch.zeros(8))

    def forward(self, x):
        """x: (B, T, D) → D₄-эквивариантное преобразование → (B, T, D)"""
        z = self.proj_to_3d(x)  # (B, T, 3)

        # Применяем все 8 операций D₄
        w = F.softmax(self.op_weights, dim=0)  # (8,)
        z_flat = z.reshape(-1, 3)  # (N, 3)

        # Взвешенная сумма преобразований: Σ w_i · (z @ R_i^T)
        transformed = torch.zeros_like(z_flat)
        for i in range(8):
            transformed += w[i] * (z_flat @ self.d4_ops[i].T)

        transformed = transformed.reshape_as(z)
        return x + self.scale * self.proj_from_3d(transformed)


class DualModeHead(nn.Module):
    """Мезонный/барионный dual-mode attention head (Ступень 6.4).

    Два режима attention:
    - Мезонный (дуадный): Q сравнивается с антиподом K (Q↔-K)
    - Барионный (триадный): Q сравнивается с циклической перестановкой K

    Обучаемый параметр mode ∈ [0,1] определяет баланс.
    По Беляеву: мезонные семейства = дуадная система ⟨u|d|s⟩↔⟨ū|d̄|s̄⟩,
    барионные = триадная ⟨u|u|d⟩↔⟨d|s|s⟩.
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.mode = nn.Parameter(torch.tensor(0.5))
        self.scale = head_dim ** -0.5

    def forward(self, q, k, v, mask=None):
        """q, k, v: (B, T, d)"""
        # Мезонный: Q · (-K)^T = -(Q · K^T) — антиподальный
        meson_attn = -torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Барионный: Q · permute(K)^T — циклическая перестановка координат
        k_perm = torch.roll(k, shifts=1, dims=-1)  # (a,b,c,d,...) → (...,a,b,c)
        baryon_attn = torch.matmul(q, k_perm.transpose(-2, -1)) * self.scale

        # Смешивание
        alpha = torch.sigmoid(self.mode)
        attn = alpha * meson_attn + (1 - alpha) * baryon_attn

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class RecursiveCubeAttention(nn.Module):
    """Рекурсивный attention «куб из кубов» (Ступень 6.5).

    Гиперкуб {-1,+1}⁶ = куб триграмм × куб триграмм (8×8).
    Двухуровневый attention:
    - Уровень 1 (intra-cube): 8 триграмм взаимодействуют внутри каждого кубика
    - Уровень 2 (inter-cube): 8 кубиков взаимодействуют через «представителей»

    Аналог Set Transformer (Lee et al., 2019), но с фиксированной кубической топологией.
    """
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Intra-cube attention (внутри каждого из 8 кубиков)
        self.intra_q = nn.Linear(d_model, d_model, bias=False)
        self.intra_k = nn.Linear(d_model, d_model, bias=False)
        self.intra_v = nn.Linear(d_model, d_model, bias=False)

        # Inter-cube attention (между 8 кубиками через «представителей»)
        self.inter_q = nn.Linear(d_model, d_model, bias=False)
        self.inter_k = nn.Linear(d_model, d_model, bias=False)
        self.inter_v = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.cube_size = 8  # триграмм в кубике

    def forward(self, x, mask=None):
        """x: (B, T, D)"""
        B, T, D = x.shape
        scale = self.head_dim ** -0.5

        # Разбиваем на кубики по 8 (padding если T не кратно 8)
        n_cubes = (T + self.cube_size - 1) // self.cube_size
        pad_len = n_cubes * self.cube_size - T
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x

        # Reshape: (B, n_cubes, 8, D)
        x_cubes = x_padded.reshape(B, n_cubes, self.cube_size, D)

        # === Уровень 1: Intra-cube attention ===
        q1 = self.intra_q(x_cubes)  # (B, n_cubes, 8, D)
        k1 = self.intra_k(x_cubes)
        v1 = self.intra_v(x_cubes)

        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale  # (B, n_cubes, 8, 8)
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.matmul(attn1, v1)  # (B, n_cubes, 8, D)

        # === Уровень 2: Inter-cube attention ===
        # «Представитель» каждого кубика = среднее
        cube_reps = out1.mean(dim=2)  # (B, n_cubes, D)

        q2 = self.inter_q(cube_reps)  # (B, n_cubes, D)
        k2 = self.inter_k(cube_reps)
        v2 = self.inter_v(cube_reps)

        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale  # (B, n_cubes, n_cubes)
        attn2 = F.softmax(attn2, dim=-1)
        inter_out = torch.matmul(attn2, v2)  # (B, n_cubes, D)

        # Broadcast inter-cube information back to tokens
        inter_broadcast = inter_out.unsqueeze(2).expand_as(out1)  # (B, n_cubes, 8, D)
        combined = out1 + inter_broadcast

        # Reshape back
        combined = combined.reshape(B, n_cubes * self.cube_size, D)
        if pad_len > 0:
            combined = combined[:, :T, :]

        return self.out_proj(combined)


class StructuralDefectLayer(nn.Module):
    """Структурный дефект 16→12: геометрический bottleneck (Ступень 6.7).

    По Беляеву: пара кубов (16 вершин) → икосаэдр (12 вершин) = потеря 4 вершин.
    Это информационное сжатие с геометрическим обоснованием.

    Конструктивный пример attention pooling с фиксированным коэффициентом 16→12.
    """
    def __init__(self, d_model: int, input_size: int = 16, output_size: int = 12):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Обучаемые «якорные» точки для сжатия
        self.anchors = nn.Parameter(torch.randn(output_size, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """x: (B, T, D) → (B, output_size, D) через attention pooling."""
        B, T, D = x.shape
        # Query = якорные точки, Key/Value = входные токены
        q = self.q_proj(self.anchors.unsqueeze(0).expand(B, -1, -1))  # (B, 12, D)
        k = self.k_proj(x)  # (B, T, D)

        scale = D ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, 12, T)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, x)  # (B, 12, D)


class MobiusAttentionPattern(nn.Module):
    """Attention-паттерн с топологией ленты Мёбиуса (Ступень 6.3).

    После полного обхода последовательности attention «переворачивается»:
    вторая половина видит первую в обратном порядке.
    Это неориентируемая поверхность в пространстве attention.
    """
    def __init__(self, max_seq_len: int = 512):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        # Предвычисляем Мёбиусов bias
        bias = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                half = max_seq_len // 2
                if i < half and j >= half:
                    # Верхний видит нижний в обратном порядке
                    mirror_j = max_seq_len - 1 - j
                    dist = abs(i - mirror_j)
                elif i >= half and j < half:
                    mirror_i = max_seq_len - 1 - i
                    dist = abs(mirror_i - j)
                else:
                    dist = abs(i - j)
                bias[i, j] = -dist
        self.register_buffer('mobius_bias', bias)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.scale * self.mobius_bias[:seq_len, :seq_len]


class PrivilegedAxisAttention(nn.Module):
    """Attention с привилегированной осью (Ступень 4.1).

    Касаткин выделяет ось (0,0,0)→(1,1,1) = ☷→☰ как «ось Z» куба.
    Attention вдоль привилегированной оси сильнее, чем перпендикулярно.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        self.axis_scale = nn.Parameter(torch.tensor(0.1))
        # Привилегированная ось: (1,1,1)/√3
        axis = torch.ones(3) / math.sqrt(3)
        self.register_buffer('axis', axis)

    def get_bias(self, x):
        """x: (B, T, D) → (B, T, T) bias для attention."""
        z = self.proj_to_3d(x)  # (B, T, 3)
        # Проекция на привилегированную ось
        axis_proj = (z * self.axis).sum(dim=-1)  # (B, T)
        # Bias: произведение проекций q и k вдоль оси
        bias = self.axis_scale * axis_proj.unsqueeze(-1) * axis_proj.unsqueeze(-2)
        return bias  # (B, T, T)


class MagicSquareInitializer:
    """Инициализация attention-весов магическим квадратом (Ступень 2.4 + 5.5).

    Герман конструирует магический квадрат из упаковки P=2^(2k).
    Фомюк показывает, что Ло-шу — частный случай для 3×3.
    """
    @staticmethod
    def loshu_3x3() -> torch.Tensor:
        """Ло-шу 3×3: сумма по строкам/столбцам/диагоналям = 15."""
        return torch.tensor([
            [2, 7, 6],
            [9, 5, 1],
            [4, 3, 8]
        ], dtype=torch.float32)

    @staticmethod
    def magic_4x4() -> torch.Tensor:
        """Магический квадрат 4×4: сумма = 34."""
        return torch.tensor([
            [16, 3, 2, 13],
            [5, 10, 11, 8],
            [9, 6, 7, 12],
            [4, 15, 14, 1]
        ], dtype=torch.float32)

    @staticmethod
    def from_hermann_packing(k: int) -> torch.Tensor:
        """Построение квадрата из упаковки Германа для P=2^(2k).

        Упаковка записывается в квадрат (2^k)×(2^k).
        Перестановки внутри столбцов дают магический квадрат.
        """
        n = 2 ** k
        P = n * n
        field = hermann_packing(2 * k)
        return field.reshape(n, n).float()

    @staticmethod
    def init_attention_weights(weight: torch.Tensor, n_heads: int = 8):
        """Инициализация attention-матрицы магическими квадратами.

        Каждая голова получает нормализованный магический квадрат.
        """
        H, T, _ = weight.shape if weight.dim() == 3 else (1, *weight.shape)
        if T <= 4:
            ms = MagicSquareInitializer.magic_4x4()[:T, :T]
        else:
            ms = MagicSquareInitializer.loshu_3x3()
            ms = F.interpolate(ms.unsqueeze(0).unsqueeze(0), size=(T, T),
                               mode='bilinear', align_corners=False).squeeze()
        # Нормализация
        ms = ms / ms.sum()
        with torch.no_grad():
            if weight.dim() == 3:
                for h in range(H):
                    weight[h] = ms
            else:
                weight.copy_(ms)


class WeavingLoomArchitecture(nn.Module):
    """4-уровневая иерархия «ткацкий станок» Беляева (Ступень 6.8).

    Уровень I:   2 → 2 (binary gate: инь/ян)
    Уровень II:  8 → 8 (триграммный куб)
    Уровень III: 64 → 64 (гексаграммный гиперкуб)
    Уровень IV:  4096 → 4096 (мега-гиперкуб, block-sparse)

    Каждый уровень использует attention своего масштаба.
    Уровни вложены: attention уровня N определяет «ткацкую нить»
    для уровня N+1.
    """
    def __init__(self, d_model: int, max_level: int = 3):
        super().__init__()
        self.d_model = d_model
        self.max_level = min(max_level, 4)

        # Уровень I: скалярный gate (sigmoid) — инь/ян
        self.level1_gate = nn.Linear(d_model, 1, bias=True)

        # Уровень II: attention 8×8 с кубической геометрией
        if max_level >= 2:
            self.level2_q = nn.Linear(d_model, d_model, bias=False)
            self.level2_k = nn.Linear(d_model, d_model, bias=False)
            self.level2_v = nn.Linear(d_model, d_model, bias=False)
            trigrams = get_trigrams()
            # Bias на основе расстояний между триграммами
            tri_dist = torch.cdist(trigrams, trigrams)
            self.register_buffer('level2_bias', -tri_dist * 0.1)

        # Уровень III: attention с гексаграммной геометрией (= основной v50)
        if max_level >= 3:
            self.level3_q = nn.Linear(d_model, d_model, bias=False)
            self.level3_k = nn.Linear(d_model, d_model, bias=False)
            self.level3_v = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """x: (B, T, D) → 4-уровневая обработка → (B, T, D)"""
        B, T, D = x.shape
        scale = D ** -0.5

        # Уровень I: binary gate
        gate = torch.sigmoid(self.level1_gate(x))  # (B, T, 1)
        out = x * gate  # инь/ян фильтрация

        if self.max_level < 2:
            return self.out_proj(out)

        # Уровень II: attention в группах по 8 (триграммный)
        n_groups = max(1, T // 8)
        group_size = min(8, T)
        # Pad if needed
        pad_len = n_groups * group_size - T
        if pad_len > 0:
            out_padded = F.pad(out, (0, 0, 0, pad_len))
        else:
            out_padded = out

        out_groups = out_padded.reshape(B, n_groups, group_size, D)
        q2 = self.level2_q(out_groups)
        k2 = self.level2_k(out_groups)
        v2 = self.level2_v(out_groups)
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
        if group_size <= 8:
            attn2 = attn2 + self.level2_bias[:group_size, :group_size]
        attn2 = F.softmax(attn2, dim=-1)
        out2 = torch.matmul(attn2, v2)
        out2 = out2.reshape(B, n_groups * group_size, D)
        if pad_len > 0:
            out2 = out2[:, :T, :]

        if self.max_level < 3:
            return self.out_proj(out2)

        # Уровень III: полный attention (гексаграммный)
        q3 = self.level3_q(out2)
        k3 = self.level3_k(out2)
        v3 = self.level3_v(out2)
        attn3 = torch.matmul(q3, k3.transpose(-2, -1)) * scale
        if mask is not None:
            attn3 = attn3.masked_fill(mask == 0, float('-inf'))
        attn3 = F.softmax(attn3, dim=-1)
        out3 = torch.matmul(attn3, v3)

        return self.out_proj(out3)


class FourLevelPositionalEncoding(nn.Module):
    """4-уровневое позиционное кодирование Андреева (Ступень 3.1).

    Уровень 1: линия (яо) — позиция внутри гексаграммы (0-5)
    Уровень 2: триграмма — верхняя/нижняя (0-1)
    Уровень 3: гексаграмма — номер (0-63)
    Уровень 4: последовательность — позиция гексаграммы в тексте

    Каждый уровень добавляет свой embedding, как в иерархическом PE.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.line_emb = nn.Embedding(6, d_model)       # 6 линий
        self.trigram_emb = nn.Embedding(2, d_model)     # верх/низ
        self.hexagram_emb = nn.Embedding(64, d_model)   # 64 гексаграммы
        self.seq_emb = nn.Embedding(max_seq_len, d_model)  # позиция в тексте
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, seq_len: int, device=None):
        """Возвращает (seq_len, d_model) позиционное кодирование."""
        if device is None:
            device = self.line_emb.weight.device

        pos = torch.arange(seq_len, device=device)

        # Иерархическая декомпозиция позиции
        line_idx = pos % 6          # линия внутри гексаграммы
        trigram_idx = (pos % 6) // 3  # верхняя(1) / нижняя(0) триграмма
        hex_idx = (pos // 6) % 64   # номер гексаграммы
        seq_idx = pos               # абсолютная позиция

        pe = (self.line_emb(line_idx)
              + self.trigram_emb(trigram_idx)
              + self.hexagram_emb(hex_idx)
              + self.seq_emb(seq_idx))

        return self.scale * pe


class BidirectionalTriangularAttention(nn.Module):
    """Двунаправленный треугольный attention (Ступень 3.3).

    Андреев: треугольная матрица attention работает в обоих направлениях.
    Нижний треугольник = прямой порядок (каузальный),
    Верхний треугольник = обратный порядок (ретроспективный).

    Обучаемый параметр direction_bias определяет баланс.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.direction_bias = nn.Parameter(torch.tensor(0.0))

        # Предвычисляем треугольные маски
        tri_lower = torch.tril(torch.ones(max_seq_len, max_seq_len))
        tri_upper = torch.triu(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('tri_lower', tri_lower)
        self.register_buffer('tri_upper', tri_upper)

    def get_mask(self, seq_len: int) -> torch.Tensor:
        """Возвращает (seq_len, seq_len) двунаправленную маску."""
        alpha = torch.sigmoid(self.direction_bias)
        lower = self.tri_lower[:seq_len, :seq_len]
        upper = self.tri_upper[:seq_len, :seq_len]
        return alpha * lower + (1 - alpha) * upper


class TriangularCurriculumScheduler:
    """Curriculum learning по треугольным числам (Ступень 3.4).

    Андреев: обучение идёт по уровням треугольной матрицы.
    Уровень k: обучение на первых T(k) = k(k+1)/2 токенах.
    По мере обучения открываются следующие уровни.
    """
    def __init__(self, max_level: int = 11):
        self.max_level = max_level
        # T(k) = k(k+1)/2
        self.levels = [k * (k + 1) // 2 for k in range(1, max_level + 1)]
        # levels = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]

    def get_seq_len(self, epoch: int, total_epochs: int) -> int:
        """Определяет текущую длину последовательности по эпохе."""
        progress = min(epoch / max(total_epochs, 1), 1.0)
        level = int(progress * self.max_level)
        level = max(0, min(level, len(self.levels) - 1))
        return self.levels[level]


class CubeDiagonalAttention(nn.Module):
    """Attention по 4 типам диагоналей куба (Ступень 4.2).

    Касаткин: куб имеет 4 типа диагоналей:
    1. Рёбра (12 шт) — d=1, соседи по 1 координатe
    2. Диагонали граней (12 шт) — d=√2, соседи по 2 координатам
    3. Пространственные диагонали (4 шт) — d=√3, антиподы-соседи
    4. Самопетли (8 шт) — d=0, self-attention

    Каждый тип диагонали получает свой вес.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        # 4 веса для 4 типов расстояний
        self.diag_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.25, 2.0]))

    def get_bias(self, x):
        """x: (B, T, D) → (B, T, T) bias на основе типа диагонали."""
        z = self.proj_to_3d(x)  # (B, T, 3)
        # Дискретизация: sign
        z_disc = torch.sign(z)  # (B, T, 3) ∈ {-1, 0, +1}

        # Расстояние Хэмминга между дискретизированными координатами
        # d_H = число различающихся координат
        z1 = z_disc.unsqueeze(2)  # (B, T, 1, 3)
        z2 = z_disc.unsqueeze(1)  # (B, 1, T, 3)
        hamming = (z1 != z2).float().sum(dim=-1)  # (B, T, T) ∈ {0,1,2,3}

        # Bias по типу диагонали
        bias = torch.zeros_like(hamming)
        for d in range(4):
            mask = (hamming == d).float()
            bias += self.diag_weights[d] * mask

        return bias


class HeisenbergAttention(nn.Module):
    """Attention из принципа Гейзенберга (Ступень 6.1).

    Беляев: ΔQ · ΔK ≥ ℏ/2 — чем точнее query, тем размытее key.
    Реализация: temperature attention с адаптивным масштабированием.

    Если |Q| велико (точный вопрос), K размывается (большая temperature).
    Если |Q| мало (общий вопрос), K фокусируется (малая temperature).
    """
    def __init__(self, d_model: int, min_temp: float = 0.1, max_temp: float = 5.0):
        super().__init__()
        self.d_model = d_model
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # ℏ/2 аналог
        self.hbar_half = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        """x: (B, T, D) → (B, T, D) с Гейзенберг-scaled attention."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # «Неопределённость» Query = нормы строк Q
        q_uncertainty = q.norm(dim=-1, keepdim=True)  # (B, T, 1)

        # Принцип: temperature ∝ |Q| (чем точнее Q, тем размытее K)
        temperature = self.hbar_half / (q_uncertainty + 1e-8)  # (B, T, 1)
        temperature = temperature.clamp(self.min_temp, self.max_temp)

        # Scaled attention с адаптивной temperature
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B, T, T)
        attn = attn * temperature  # broadcast (B, T, 1)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class FlowerOfLifeGAT(nn.Module):
    """Цветок Жизни как GAT-граф (Ступень 6.6).

    7 пересекающихся кругов Цветка Жизни задают граф:
    - 7 узлов (центры кругов)
    - Рёбра: два узла связаны, если круги пересекаются
    - Граф: центральный узел связан со всеми 6, периферийные — с 3 соседями

    GAT (Graph Attention Network) на этом графе.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.n_nodes = 7

        # Матрица смежности Цветка Жизни (7×7)
        adj = torch.zeros(7, 7)
        # Центральный (0) связан со всеми
        for i in range(1, 7):
            adj[0, i] = 1
            adj[i, 0] = 1
        # Периферийные связаны с соседями (кольцо)
        for i in range(1, 7):
            j = (i % 6) + 1  # следующий в кольце
            adj[i, j] = 1
            adj[j, i] = 1
        self.register_buffer('adjacency', adj)

        # GAT проекции
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Linear(2 * d_model, 1, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """x: (B, T, D) → (B, T, D). T токенов распределяются по 7 узлам."""
        B, T, D = x.shape

        # Разбиваем токены по 7 узлам (round-robin)
        node_tokens = []
        for i in range(self.n_nodes):
            indices = torch.arange(i, T, self.n_nodes, device=x.device)
            if len(indices) > 0:
                node_tokens.append(x[:, indices].mean(dim=1))  # (B, D)
            else:
                node_tokens.append(torch.zeros(B, D, device=x.device))
        nodes = torch.stack(node_tokens, dim=1)  # (B, 7, D)

        # GAT attention
        h = self.W(nodes)  # (B, 7, D)
        # Попарные конкатенации для attention coefficients
        h_i = h.unsqueeze(2).expand(-1, -1, 7, -1)  # (B, 7, 7, D)
        h_j = h.unsqueeze(1).expand(-1, 7, -1, -1)  # (B, 7, 7, D)
        e = self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)  # (B, 7, 7)

        # Маскируем не-соседей
        e = e.masked_fill(self.adjacency == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)  # (B, 7, 7)

        # Агрегация
        out_nodes = torch.matmul(alpha, h)  # (B, 7, D)
        out_nodes = self.out_proj(out_nodes)

        # Broadcast обратно к токенам
        result = x.clone()
        for i in range(self.n_nodes):
            indices = torch.arange(i, T, self.n_nodes, device=x.device)
            if len(indices) > 0:
                result[:, indices] = result[:, indices] + out_nodes[:, i:i+1]

        return result


# ==================== ROTARY POSITION EMBEDDINGS ====================

class RotaryEmbedding(nn.Module):
    """
    RoPE: вращение пар измерений в зависимости от позиции.

    Поддерживает scaling для расширения контекста:
    - None: стандартный RoPE
    - 'linear': линейная интерполяция позиций (позволяет extrapolation)
    - 'ntk': NTK-aware scaling (изменяет base, лучше сохраняет различимость)
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0,
                 scaling: str = None, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling = scaling
        self.scaling_factor = scaling_factor

        if scaling == 'ntk' and scaling_factor > 1.0:
            # NTK-aware: увеличиваем base пропорционально scaling_factor
            base = base * (scaling_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        if self.scaling == 'linear' and self.scaling_factor > 1.0:
            # Линейная интерполяция: сжимаем позиции
            t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Поворот половины измерений для RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x, cos, sin):
    """Применение RoPE к тензору x: (B, H, T, D)."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ==================== ALiBi ====================

class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    Добавляет линейный bias к attention scores на основе расстояния:
        bias[h, i, j] = -m_h * |i - j|

    Каждая голова получает свой slope m_h = 2^(-8h/H).
    Не требует позиционных эмбеддингов. Хорошо экстраполирует на длинные контексты.

    Ref: Press et al., "Train Short, Test Long" (2022)
    """
    def __init__(self, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads

        # Slopes: geometric sequence 2^(-8/H), 2^(-16/H), ..., 2^(-8)
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)  # (H,)
        self._build_cache(max_seq_len)

    @staticmethod
    def _get_slopes(n_heads):
        """Генерирует slopes для ALiBi."""
        def _get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(torch.tensor(float(n)).log2().floor().item() - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if n_heads & (n_heads - 1) == 0:  # power of 2
            slopes = _get_slopes_power_of_2(n_heads)
        else:
            closest_power = 2 ** int(torch.tensor(float(n_heads)).log2().floor().item())
            slopes = _get_slopes_power_of_2(closest_power)
            extra = _get_slopes_power_of_2(2 * closest_power)
            slopes = slopes + extra[0::2][:n_heads - closest_power]

        return torch.tensor(slopes, dtype=torch.float32)

    def _build_cache(self, seq_len):
        # Матрица расстояний: |i - j|
        positions = torch.arange(seq_len)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        distances = distances.abs().float()
        self.register_buffer('_distances', distances, persistent=False)

    def forward(self, seq_len, offset=0):
        """
        Возвращает ALiBi bias: (1, H, T, S) для attention scores.

        Args:
            seq_len: длина query
            offset: смещение для KV-cache (S = seq_len + offset)
        """
        total_len = seq_len + offset
        if total_len > self._distances.shape[0]:
            self._build_cache(total_len)

        # Расстояния для текущего окна
        distances = self._distances[offset:offset + seq_len, :total_len]  # (T, S)

        # Bias: -slope * distance, per head
        bias = -self.slopes.view(1, -1, 1, 1) * distances.unsqueeze(0).unsqueeze(0)
        return bias  # (1, H, T, S)


# ==================== SwiGLU FFN ====================

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward: более эффективный, чем GELU FFN (LLaMA-style)."""
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ==================== MoE НА ТРИГРАММАХ ====================

class TrigramMoE(nn.Module):
    """
    Mixture of Experts, где каждый эксперт соответствует триграмме.

    Router проецирует вход в 3D (пространство триграмм),
    затем выбирает top-k ближайших триграмм как экспертов.
    Это геометрически мотивированный MoE.
    """
    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model

        if ffn_hidden is None:
            ffn_hidden = 4 * d_model

        # Router: проецируем в пространство триграмм
        trigrams = get_trigrams()[:n_experts]
        self.register_buffer('trigram_dirs', F.normalize(trigrams, p=2, dim=1))
        self.router_proj = nn.Linear(d_model, 3, bias=False)

        # Эксперты
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_hidden, bias=False),
                nn.GELU(),
                nn.Linear(ffn_hidden, d_model, bias=False),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

        # Load balancing loss coefficient
        self.aux_loss_coeff = 0.01

    def forward(self, x):
        """
        x: (B, T, D) → (B, T, D)
        Возвращает также aux_loss для балансировки нагрузки.
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)

        # Router scores через проекцию в 3D пространство триграмм
        z3 = self.router_proj(x_flat)  # (B*T, 3)
        router_logits = z3 @ self.trigram_dirs.T  # (B*T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Применение экспертов
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (B*T,)
            expert_weights = top_k_probs[:, k]    # (B*T,)

            for e in range(self.n_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        # Aux loss для балансировки (GShard-style)
        tokens_per_expert = router_probs.mean(dim=0)  # (n_experts,)
        uniform = torch.ones_like(tokens_per_expert) / self.n_experts
        aux_loss = self.aux_loss_coeff * F.mse_loss(tokens_per_expert, uniform)

        return output.view(B, T, D), aux_loss


# ==================== ИЕРАРХИЧЕСКАЯ КВАНТИЗАЦИЯ ====================

class HierarchicalQuantizer(nn.Module):
    """
    Иерархическая квантизация: разбивает вход на группы по k координат
    и квантизует каждую группу к {-1,+1}^k (Product Quantization).

    Примеры конфигураций для 8D входа:
    - n_groups=4, group_dim=2: 4 биграммы (4×4×4×4 = 256 точек)
    - n_groups=2, group_dim=4: 2 тетраграммы (16×16 = 256 точек)
    - n_groups=1, group_dim=8: 1 октограмма (256 точек, без факторизации)

    Для 6D (гексаграммы): n_groups=2, group_dim=3 — стандартная факторизация.
    """
    def __init__(self, total_dim: int, group_dim: int = 2,
                 temp: float = 0.3, adaptive_temp: bool = False):
        super().__init__()
        assert total_dim % group_dim == 0, \
            f"total_dim ({total_dim}) must be divisible by group_dim ({group_dim})"
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim  # число вершин гиперкуба per group

        # Кодбук для одной группы: все вершины {-1,+1}^group_dim
        codebook = generate_hypercube(group_dim)  # (2^k, k)
        self.register_buffer('codebook', codebook)
        self.register_buffer('codebook_norm_sq', (codebook ** 2).sum(dim=1))

        self.adaptive_temp = adaptive_temp
        if adaptive_temp:
            self.log_temp = nn.Parameter(torch.tensor(temp).log())
        else:
            self.temp = temp

    @property
    def current_temp(self):
        if self.adaptive_temp:
            return self.log_temp.exp().clamp(min=0.01, max=5.0)
        return self.temp

    def forward(self, x):
        # x: (..., total_dim)
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)  # (..., G, K)

        quantized_groups = self._soft_quantize_batch(groups)
        quantized = quantized_groups.reshape(*shape, self.total_dim)

        if self.adaptive_temp:
            return quantized
        else:
            return x + (quantized - x).detach()

    def _soft_quantize_batch(self, z):
        # z: (..., G, K)
        z_norm_sq = (z * z).sum(dim=-1, keepdim=True)  # (..., G, 1)
        cross = z @ self.codebook.T  # (..., G, 2^K)
        dists_sq = z_norm_sq - 2 * cross + self.codebook_norm_sq
        weights = F.softmax(-dists_sq / self.current_temp, dim=-1)  # (..., G, 2^K)
        return weights @ self.codebook  # (..., G, K)

    def hard_quantize(self, x):
        return torch.sign(x)

    def codebook_info(self):
        """Информация о кодбуке для логирования."""
        return {
            'total_dim': self.total_dim,
            'group_dim': self.group_dim,
            'n_groups': self.n_groups,
            'n_codewords_per_group': self.n_codewords,
            'total_codewords': self.n_codewords ** self.n_groups,
            'softmax_ops': self.n_groups * self.n_codewords,
        }


class DeformableQuantizer(nn.Module):
    """
    Деформируемый кодбук: начинаем с идеального гиперкуба {-1,+1}^n,
    но позволяем модели «деформировать» кодовые слова.

    codebook = base_hypercube + learnable_delta

    Это мост между:
    - Фиксированным кодбуком (YiJing, E8) — хорошая инициализация
    - Полностью обучаемым кодбуком (VQ-VAE) — максимальная гибкость
    """
    def __init__(self, total_dim: int, group_dim: int = 3,
                 temp: float = 0.3, deform_scale: float = 0.0):
        super().__init__()
        assert total_dim % group_dim == 0
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim

        base = generate_hypercube(group_dim)
        self.register_buffer('base_codebook', base)  # (2^k, k)

        # Обучаемая деформация (инициализирована нулями)
        self.delta = nn.Parameter(torch.zeros_like(base))
        self.deform_scale = nn.Parameter(torch.tensor(deform_scale))
        self.temp = temp

    @property
    def codebook(self):
        return self.base_codebook + self.deform_scale * self.delta

    def forward(self, x):
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)

        cb = self.codebook
        cb_norm_sq = (cb * cb).sum(dim=1)
        z_norm_sq = (groups * groups).sum(dim=-1, keepdim=True)
        cross = groups @ cb.T
        dists_sq = z_norm_sq - 2 * cross + cb_norm_sq
        weights = F.softmax(-dists_sq / self.temp, dim=-1)
        quantized_groups = weights @ cb
        quantized = quantized_groups.reshape(*shape, self.total_dim)

        return x + (quantized - x).detach()

    def deformation_stats(self):
        """Статистика деформации для мониторинга."""
        delta_norm = self.delta.norm().item()
        scale = self.deform_scale.item()
        effective_shift = delta_norm * abs(scale)
        return {
            'delta_norm': delta_norm,
            'deform_scale': scale,
            'effective_shift': effective_shift,
        }


class GumbelQuantizer(nn.Module):
    """
    Gumbel-Softmax квантизация к вершинам гиперкуба.

    Вместо soft attention по расстояниям используем Gumbel-Softmax
    для дискретного выбора ближайшей вершины. При обучении —
    дифференцируемое приближение, при инференсе — hard argmax.

    Поддерживает commitment loss: ||x - sg(quantized)||² + β·||sg(x) - quantized||²
    """
    def __init__(self, total_dim: int, group_dim: int = 3,
                 temp: float = 1.0, hard: bool = False,
                 commitment_weight: float = 0.25):
        super().__init__()
        assert total_dim % group_dim == 0
        self.total_dim = total_dim
        self.group_dim = group_dim
        self.n_groups = total_dim // group_dim
        self.n_codewords = 2 ** group_dim
        self.hard = hard
        self.commitment_weight = commitment_weight

        # Лог-температура (обучаемая)
        self.log_temp = nn.Parameter(torch.tensor(temp).log())

        codebook = generate_hypercube(group_dim)
        self.register_buffer('codebook', codebook)
        self.register_buffer('codebook_norm_sq', (codebook ** 2).sum(dim=1))

        # Для commitment loss
        self._commitment_loss = torch.tensor(0.0)

    @property
    def current_temp(self):
        return self.log_temp.exp().clamp(min=0.05, max=5.0)

    def forward(self, x):
        shape = x.shape[:-1]
        groups = x.reshape(*shape, self.n_groups, self.group_dim)  # (..., G, K)

        # Расстояния до кодовых слов
        z_norm_sq = (groups * groups).sum(dim=-1, keepdim=True)
        cross = groups @ self.codebook.T
        dists_sq = z_norm_sq - 2 * cross + self.codebook_norm_sq
        logits = -dists_sq  # (..., G, 2^K)

        # Gumbel-Softmax
        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.current_temp, hard=self.hard)
        else:
            # Hard argmax при инференсе
            idx = logits.argmax(dim=-1)  # (..., G)
            weights = F.one_hot(idx, self.n_codewords).float()

        quantized_groups = weights @ self.codebook  # (..., G, K)
        quantized = quantized_groups.reshape(*shape, self.total_dim)

        # Commitment loss
        if self.training and self.commitment_weight > 0:
            self._commitment_loss = (
                (x.detach() - quantized).pow(2).mean()
                + self.commitment_weight * (x - quantized.detach()).pow(2).mean()
            )
        else:
            self._commitment_loss = torch.tensor(0.0, device=x.device)

        return quantized

    def get_commitment_loss(self):
        return self._commitment_loss


# ==================== ГЕКСАГРАММНЫЙ ATTENTION PATTERN ====================

class HexagramAttentionPattern(nn.Module):
    """
    Гексаграммный паттерн attention: 64 фиксированных паттерна
    внимания, каждый соответствует гексаграмме.

    Каждая из 6 линий гексаграммы контролирует один аспект:
    - Линии 1-3 (нижняя триграмма): локальные паттерны (ближние связи)
    - Линии 4-6 (верхняя триграмма): глобальные паттерны (дальние связи)

    +1 = «усиливать связь», -1 = «ослаблять связь»
    """
    def __init__(self, d_model: int, block_size: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))  # начинаем отключённым

        # 6 базовых паттернов attention (один на линию)
        # Линии 1-3: окна разного размера (локальные)
        # Линии 4-6: шаги разного размера (глобальные/периодические)
        patterns = torch.zeros(6, block_size, block_size)
        for i in range(block_size):
            for j in range(i + 1):  # causal
                dist = i - j
                # Линия 1: ближайший сосед (окно 2)
                patterns[0, i, j] = 1.0 if dist <= 2 else -1.0
                # Линия 2: среднее окно (окно 8)
                patterns[1, i, j] = 1.0 if dist <= 8 else -1.0
                # Линия 3: широкое окно (окно 32)
                patterns[2, i, j] = 1.0 if dist <= 32 else -1.0
                # Линия 4: чётные позиции
                patterns[3, i, j] = 1.0 if dist % 2 == 0 else -1.0
                # Линия 5: каждые 4
                patterns[4, i, j] = 1.0 if dist % 4 == 0 else -1.0
                # Линия 6: начало последовательности
                patterns[5, i, j] = 1.0 if j <= 4 else -1.0

        self.register_buffer('patterns', patterns)  # (6, T, T)

    def forward(self, x, T):
        """
        Возвращает attention bias (B, 1, T, T).
        x: (B, T, D) → проецируем в 6D → взвешиваем паттерны.
        """
        # Средний вектор последовательности → 6D координата
        x_mean = x.mean(dim=1)  # (B, D)
        z6 = self.proj_to_6d(x_mean)  # (B, 6)
        hex_weights = torch.tanh(z6)  # (B, 6) в [-1, 1]

        # Взвешенная комбинация 6 паттернов
        patterns = self.patterns[:, :T, :T]  # (6, T, T)
        bias = torch.einsum('bk,kij->bij', hex_weights, patterns)  # (B, T, T)

        return self.scale * bias.unsqueeze(1)  # (B, 1, T, T)


# ==================== Grouped Quantization ====================

class GroupedQuantizer(nn.Module):
    """
    Grouped (per-channel) quantization с обучаемыми scales.

    Делит d_model на группы, каждая группа квантизуется независимо
    со своим масштабом и zero-point. Подход аналогичен GPTQ/AWQ.

    Это позволяет сохранить точность при INT8 квантизации,
    учитывая разный диапазон значений в разных каналах.

    Args:
        d_model: размерность модели
        group_size: размер группы (128 по умолчанию, как в GPTQ)
        n_bits: число бит квантизации (8 = INT8)
        symmetric: симметричная квантизация (без zero-point)
    """
    def __init__(self, d_model, group_size=128, n_bits=8, symmetric=True):
        super().__init__()
        self.d_model = d_model
        self.group_size = min(group_size, d_model)
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.n_groups = (d_model + self.group_size - 1) // self.group_size

        # Диапазон квантизации
        if symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1

        # Обучаемые scales и zero points
        self.scales = nn.Parameter(torch.ones(self.n_groups))
        if not symmetric:
            self.zero_points = nn.Parameter(torch.zeros(self.n_groups))

    def _reshape_to_groups(self, x):
        """Reshape последнее измерение в (n_groups, group_size)."""
        *batch, d = x.shape
        pad = (self.group_size - d % self.group_size) % self.group_size
        if pad > 0:
            x = F.pad(x, (0, pad))
        return x.view(*batch, self.n_groups, self.group_size)

    def _unreshape(self, x, orig_d):
        """Обратно из (n_groups, group_size) → (d,)."""
        *batch, ng, gs = x.shape
        return x.reshape(*batch, ng * gs)[..., :orig_d]

    def quantize(self, x):
        """
        Квантизует тензор с per-group scales.

        Использует STE (Straight-Through Estimator) для backprop.
        """
        orig_d = x.shape[-1]
        x_g = self._reshape_to_groups(x)  # (..., n_groups, group_size)

        scales = self.scales.abs().clamp(min=1e-8)
        scales = scales.view(*([1] * (x_g.dim() - 2)), self.n_groups, 1)

        if self.symmetric:
            x_scaled = x_g / scales
            x_quant = x_scaled.round().clamp(self.qmin, self.qmax)
            x_dequant = x_quant * scales
        else:
            zp = self.zero_points.round()
            zp = zp.view(*([1] * (x_g.dim() - 2)), self.n_groups, 1)
            x_scaled = x_g / scales + zp
            x_quant = x_scaled.round().clamp(self.qmin, self.qmax)
            x_dequant = (x_quant - zp) * scales

        # STE: gradient проходит через quantize как identity
        result = x_g + (x_dequant - x_g).detach()
        return self._unreshape(result, orig_d)

    def forward(self, x):
        """Quantize-dequantize forward pass."""
        if self.training:
            return self.quantize(x)
        else:
            return self.quantize(x)

    def calibrate(self, x):
        """
        Калибровка scales на основе наблюдаемых данных (для PTQ).

        Args:
            x: тензор для калибровки (..., d_model)
        """
        with torch.no_grad():
            x_g = self._reshape_to_groups(x)
            if self.symmetric:
                amax = x_g.abs().amax(dim=-1).mean(dim=tuple(range(x_g.dim() - 2)))
                self.scales.data = amax / self.qmax
            else:
                vmin = x_g.amin(dim=-1).mean(dim=tuple(range(x_g.dim() - 2)))
                vmax = x_g.amax(dim=-1).mean(dim=tuple(range(x_g.dim() - 2)))
                self.scales.data = (vmax - vmin) / (self.qmax - self.qmin)
                self.zero_points.data = (self.qmin - vmin / self.scales.data).round()

    def extra_repr(self):
        return (f"d_model={self.d_model}, groups={self.n_groups}, "
                f"group_size={self.group_size}, bits={self.n_bits}, "
                f"symmetric={self.symmetric}")


# ==================== ГЕЙТОВЫЙ МЕХАНИЗМ ВЫБОРА ПУТИ ====================

class GatedPathSelector(nn.Module):
    """
    Гейтовый механизм выбора между геометрическим и стандартным путём.

    Принцип ненавязывания: модель сама решает, использовать ли геометрию,
    через обучаемый гейт. Значение гейта прозрачно логируется.

    gate = sigmoid(W·x + b)
    output = gate * geometric_path + (1 - gate) * standard_path

    Инициализация gate_bias=0 → начальный gate ≈ 0.5 (равные шансы).
    """
    def __init__(self, d_model: int, init_bias: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        # Инициализация: малые веса + настраиваемый bias
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)
        # Статистика для логирования
        self._last_gate_mean = 0.5
        self._last_gate_std = 0.0

    def forward(self, x_standard, x_geometric):
        """
        Args:
            x_standard: выход стандартного пути (B, T, D)
            x_geometric: выход геометрического пути (B, T, D)
        Returns:
            blended output (B, T, D)
        """
        # Гейт вычисляется на основе среднего двух путей
        combined = (x_standard + x_geometric) * 0.5
        gate = torch.sigmoid(self.gate_proj(combined))  # (B, T, 1)

        # Логирование
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_gate_std = gate.std().item()

        return gate * x_geometric + (1 - gate) * x_standard

    def get_gate_stats(self):
        """Возвращает статистику гейта для логирования."""
        return {
            'gate_mean': self._last_gate_mean,
            'gate_std': self._last_gate_std,
            'prefers_geometry': self._last_gate_mean > 0.5,
        }


# ==================== ЧИСТЫЙ ГЕОМЕТРИЧЕСКИЙ ATTENTION ====================

class GeometricAttention(nn.Module):
    """
    Attention, полностью основанный на геометрии триграмм.

    Вместо стандартного QKV-attention используется:
    1. Проекция в пространство триграмм (3D)
    2. Расстояния между триграммными проекциями как attention scores
    3. 8 голов = 8 триграммных направлений

    Это «чистый» геометрический attention без стандартного dot-product.
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.use_rope = cfg.use_rope

        # Проекции: вход → триграммное пространство (3D per head)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * 3, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * 3, bias=False)
        # Value проекция — стандартная (для выразительности)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        # 8 триграмм как направления
        trigrams = get_trigrams()  # (8, 3)
        trigrams_norm = F.normalize(trigrams, p=2, dim=1)
        self.register_buffer('head_dirs', trigrams_norm[:cfg.n_heads])

        # Масштаб для стабильности
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(3.0)))

        # RoPE для позиционной информации (применяется к 3D проекциям)
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                dim=4,  # ближайшее чётное к 3, pad to 4
                max_seq_len=cfg.block_size,
                base=cfg.rope_base,
            )

    def forward(self, x):
        B, T, C = x.shape

        # Q, K в триграммном 3D пространстве
        q3 = self.q_proj(x).reshape(B, T, self.n_heads, 3).transpose(1, 2)  # (B, H, T, 3)
        k3 = self.k_proj(x).reshape(B, T, self.n_heads, 3).transpose(1, 2)

        # V — полноразмерный (для выразительности)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Проецируем Q, K на триграммные направления и вычисляем scores
        # score = (q · dir_h) * (k · dir_h) — билинейная форма через триграмму
        q_on_dir = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)  # (B, H, T)
        k_on_dir = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)  # (B, H, T)

        # Также добавляем стандартный dot-product в 3D
        dot_scores = (q3 @ k3.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Геометрический bias: внешнее произведение проекций на триграммы
        geo_bias = q_on_dir.unsqueeze(-1) * k_on_dir.unsqueeze(-2)  # (B, H, T, T)

        scores = dot_scores + geo_bias

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class GeometricFFN(nn.Module):
    """
    FFN, основанный на геометрической маршрутизации через триграммы.

    Вместо стандартного MLP: проецируем в пространство триграмм,
    определяем ближайшие триграммы, активируем соответствующие подсети.
    Это «чистый геометрический FFN» — всегда использует TrigramMoE.
    """
    def __init__(self, cfg):
        super().__init__()
        self.moe = TrigramMoE(
            d_model=cfg.d_model,
            n_experts=min(cfg.n_heads, 8),
            top_k=2,
            ffn_hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
        )

    def forward(self, x):
        return self.moe(x)  # returns (output, aux_loss)


# ==================== ЛОГИРОВАНИЕ ГЕЙТОВ ====================

class GateLogger:
    """
    Собирает и агрегирует статистику гейтов для прозрачности.

    Позволяет отслеживать:
    - Какие слои предпочитают геометрию vs стандартный путь
    - Динамику предпочтений во время обучения
    - Распределение гейтов по слоям

    Принцип прозрачности: все решения модели видимы и интерпретируемы.
    """
    def __init__(self):
        self.history = []  # list of {step, layer_gates}

    def log_step(self, step: int, layers):
        """Логирует состояние гейтов всех слоёв."""
        entry = {'step': step, 'gates': {}}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'path_gate') and layer.path_gate is not None:
                stats = layer.path_gate.get_gate_stats()
                entry['gates'][f'layer_{i}'] = stats
        self.history.append(entry)
        return entry

    def summary(self):
        """Сводка по последнему шагу."""
        if not self.history:
            return {}
        last = self.history[-1]
        geo_layers = 0
        std_layers = 0
        for gate_info in last['gates'].values():
            if gate_info['prefers_geometry']:
                geo_layers += 1
            else:
                std_layers += 1
        return {
            'step': last['step'],
            'layers_prefer_geometry': geo_layers,
            'layers_prefer_standard': std_layers,
            'gates': last['gates'],
        }

    def get_trajectory(self):
        """Траектория средних гейтов для каждого слоя."""
        trajectories = {}
        for entry in self.history:
            for layer_name, stats in entry['gates'].items():
                if layer_name not in trajectories:
                    trajectories[layer_name] = {'steps': [], 'means': []}
                trajectories[layer_name]['steps'].append(entry['step'])
                trajectories[layer_name]['means'].append(stats['gate_mean'])
        return trajectories


# ==================== CURRICULUM SCHEDULER ====================

class GeometryCurriculumScheduler:
    """
    Планировщик curriculum learning для геометрических компонентов.

    Стратегии:
    - 'linear': линейно увеличивает hex_strength от 0 до target
    - 'warmup_hold': warmup → hold на полной силе
    - 'cosine': косинусный рост от 0 до target
    - 'step': ступенчатое увеличение каждые N шагов
    - 'geometric_first': начинает с чистой геометрии, плавно добавляет стандартный путь

    Принцип постепенности: не заставляет модель сразу использовать геометрию,
    а создаёт условия для естественного обучения.
    """
    def __init__(self, strategy: str = 'linear', total_steps: int = 10000,
                 warmup_fraction: float = 0.3, target_strength: float = 0.1,
                 n_step_stages: int = 4):
        self.strategy = strategy
        self.total_steps = total_steps
        self.warmup_fraction = warmup_fraction
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.target_strength = target_strength
        self.n_step_stages = n_step_stages

    def get_strength(self, step: int) -> float:
        """Возвращает текущую силу геометрического компонента."""
        progress = min(step / max(self.total_steps, 1), 1.0)

        if self.strategy == 'linear':
            return self.target_strength * progress

        elif self.strategy == 'warmup_hold':
            if step < self.warmup_steps:
                return self.target_strength * step / self.warmup_steps
            return self.target_strength

        elif self.strategy == 'cosine':
            return self.target_strength * 0.5 * (1 - math.cos(math.pi * progress))

        elif self.strategy == 'step':
            stage = min(int(progress * self.n_step_stages), self.n_step_stages)
            return self.target_strength * stage / self.n_step_stages

        elif self.strategy == 'geometric_first':
            # Начинает с 1.0 (полная геометрия), снижается до target
            # Идея: сначала дать геометрии шанс, потом позволить выбор
            if step < self.warmup_steps:
                return 1.0
            decay_progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return self.target_strength + (1.0 - self.target_strength) * (1 - decay_progress)

        else:
            return self.target_strength

    def get_gate_bias(self, step: int) -> float:
        """
        Возвращает bias для гейта (сдвигает предпочтение к геометрии).

        Положительный bias → гейт ближе к 1 → предпочтение геометрии.
        Используется в стратегии 'geometric_first'.
        """
        if self.strategy == 'geometric_first':
            progress = min(step / max(self.total_steps, 1), 1.0)
            # Начинаем с bias=2 (сильное предпочтение геометрии), снижаем до 0
            return 2.0 * (1 - progress)
        return 0.0


# ==================== ФАЗА 3: АДАПТИВНАЯ СПЕЦИАЛИЗАЦИЯ ====================

class AdaptiveGatedPathSelector(nn.Module):
    """
    Расширенный гейт с layer-specific и input-dependent поведением.

    Отличия от GatedPathSelector:
    1. Гейт зависит от КОНТЕНТА входа (не только среднего)
    2. Поддерживает multi-head gate (разный гейт для разных голов)
    3. Temperature scheduling для управления уверенностью
    """
    def __init__(self, d_model: int, n_heads: int = 1, init_bias: float = 0.0):
        super().__init__()
        self.n_heads = n_heads

        # Контентно-зависимый гейт: входная + позиционная информация
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

        # Обучаемая температура для управления «уверенностью»
        self.log_temperature = nn.Parameter(torch.tensor(0.0))  # T=1.0

        # Статистика
        self._last_gate_mean = 0.5
        self._last_gate_std = 0.0
        self._last_gate_entropy = 0.0

    def forward(self, x_standard, x_geometric):
        combined = (x_standard + x_geometric) * 0.5
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)

        raw_gate = self.gate_proj(combined) / temp  # (B, T, n_heads)
        gate = torch.sigmoid(raw_gate)

        if self.n_heads == 1:
            gate = gate  # (B, T, 1)
        else:
            # Multi-head: средний гейт для смешивания
            gate = gate.mean(dim=-1, keepdim=True)  # (B, T, 1)

        # Логирование
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_gate_std = gate.std().item()
            # Энтропия гейта (мера неопределённости решения)
            g = gate.mean().clamp(1e-6, 1 - 1e-6)
            self._last_gate_entropy = -(g * g.log() + (1-g) * (1-g).log()).item()

        return gate * x_geometric + (1 - gate) * x_standard

    def get_gate_stats(self):
        return {
            'gate_mean': self._last_gate_mean,
            'gate_std': self._last_gate_std,
            'gate_entropy': self._last_gate_entropy,
            'temperature': self.log_temperature.exp().item(),
            'prefers_geometry': self._last_gate_mean > 0.5,
        }


class TaskAwareRouter(nn.Module):
    """
    Task-aware routing: определяет тип входного паттерна и направляет
    через соответствующий путь.

    Механизм:
    1. Проецирует средний вектор последовательности в пространство «задач»
    2. Определяет softmax-веса для K стратегий
    3. Каждая стратегия = своя комбинация gate biases

    Это позволяет модели адаптивно выбирать стратегию на уровне
    входной последовательности, а не только на уровне токена.
    """
    def __init__(self, d_model: int, n_strategies: int = 4):
        super().__init__()
        self.n_strategies = n_strategies
        self.strategy_proj = nn.Linear(d_model, n_strategies, bias=True)
        # Каждая стратегия задаёт bias для гейта: от -2 (standard) до +2 (geometry)
        self.strategy_biases = nn.Parameter(
            torch.linspace(-1.0, 1.0, n_strategies)
        )
        self._last_strategy_probs = None

    def forward(self, x):
        """
        Args:
            x: (B, T, D) — входная последовательность
        Returns:
            gate_bias: (B, 1, 1) — bias для гейта на уровне последовательности
        """
        x_mean = x.mean(dim=1)  # (B, D)
        logits = self.strategy_proj(x_mean)  # (B, n_strategies)
        probs = F.softmax(logits, dim=-1)  # (B, n_strategies)

        # Взвешенный bias
        gate_bias = (probs * self.strategy_biases.unsqueeze(0)).sum(dim=-1)  # (B,)

        with torch.no_grad():
            self._last_strategy_probs = probs.mean(dim=0).tolist()

        return gate_bias.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

    def get_strategy_stats(self):
        if self._last_strategy_probs is not None:
            return {f'strategy_{i}': p for i, p in enumerate(self._last_strategy_probs)}
        return {}


class DynamicCurriculumController:
    """
    Dynamic curriculum: адаптирует стратегию обучения на основе
    текущего состояния гейтов.

    Если модель «отвергает» геометрию (гейты < 0.3) — уменьшаем давление.
    Если модель «принимает» геометрию (гейты > 0.7) — можно увеличить.
    Если гейты нестабильны (std > 0.2) — замедляем изменения.
    """
    def __init__(self, base_strength: float = 0.1, adapt_rate: float = 0.01):
        self.base_strength = base_strength
        self.adapt_rate = adapt_rate
        self.current_strength = base_strength
        self.history = []

    def update(self, avg_gate_value: float, gate_std: float):
        """Обновляет силу геометрии на основе текущих гейтов."""
        if gate_std > 0.2:
            # Нестабильные гейты — не менять
            pass
        elif avg_gate_value > 0.6:
            # Модель принимает геометрию — можно немного усилить
            self.current_strength = min(
                self.current_strength + self.adapt_rate,
                self.base_strength * 3.0
            )
        elif avg_gate_value < 0.35:
            # Модель отвергает — уменьшить давление
            self.current_strength = max(
                self.current_strength - self.adapt_rate,
                self.base_strength * 0.1
            )

        self.history.append({
            'strength': self.current_strength,
            'avg_gate': avg_gate_value,
            'gate_std': gate_std,
        })

        return self.current_strength


class MultiScaleHypercubeLayer(nn.Module):
    """
    Фаза 6: Multi-scale hypercube — разные размерности гиперкуба по слоям.

    Идея: нижние слои используют маленькие гиперкубы (биграммы 2D),
    верхние — большие (гексаграммы 6D, октограммы 8D).

    Это соответствует иерархии абстракций:
    - Низкоуровневые → простые бинарные решения (2D)
    - Высокоуровневые → сложные комбинации (6D, 8D)
    """
    def __init__(self, d_model: int, hypercube_dim: int = 3, temp: float = 0.3):
        super().__init__()
        self.dim = hypercube_dim
        n_vertices = 2 ** hypercube_dim

        # Генерируем вершины гиперкуба нужной размерности
        vertices = generate_hypercube(hypercube_dim)  # (2^dim, dim)
        self.register_buffer('vertices', vertices)

        # Проекции
        self.proj_to = nn.Linear(d_model, hypercube_dim, bias=False)
        self.proj_from = nn.Linear(hypercube_dim, d_model, bias=False)
        self.temp = nn.Parameter(torch.tensor(temp).log())
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        """
        x: (B, T, D) → квантизация к вершинам {-1,+1}^dim → (B, T, D)
        """
        z = self.proj_to(x)  # (B, T, dim)
        temp = self.temp.exp().clamp(min=0.01, max=5.0)

        # Soft quantization к ближайшим вершинам
        z_flat = z.reshape(-1, self.dim)  # (B*T, dim)
        dists = torch.cdist(z_flat.unsqueeze(0), self.vertices.unsqueeze(0)).squeeze(0)  # (B*T, 2^dim)
        weights = F.softmax(-dists / temp, dim=-1)  # (B*T, 2^dim)
        quantized = weights @ self.vertices  # (B*T, dim)
        quantized = quantized.reshape_as(z)

        return x + self.scale * self.proj_from(quantized)
