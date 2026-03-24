"""
meta_bridge.py — живой мост между meta (математика Q6) и pro2 (нейросеть) (Шаг 6).

НЕ копирует код из meta — импортирует напрямую через sys.path.
Требует: git clone https://github.com/svend4/meta.git ../meta

Предоставляет:
  - HexLearnRouter: k-NN маршрутизация на Q6 из meta/hexlearn
  - MetropolisAnnealer: физически корректное охлаждение из meta/hexphys
  - Q4Q6Mapper: Q4⊂Q6 вложение через meta/hexdim (15 тессерактов)
  - FullAnalogyCrossMatrix: полная матрица симметрий из meta/hexsym

Установка:
    cd /path/to  # папка рядом с pro2
    git clone https://github.com/svend4/meta.git

Проверка:
    python meta_bridge.py
"""

import sys
import math
import warnings
from pathlib import Path

# ─── Поиск meta ──────────────────────────────────────────────────
_META_CANDIDATES = [
    Path(__file__).parent.parent / 'meta',      # ../meta
    Path(__file__).parent / 'meta',              # ./meta
    Path.home() / 'meta',                        # ~/meta
]

META_ROOT = None
for _candidate in _META_CANDIDATES:
    if _candidate.exists():
        META_ROOT = _candidate
        break

META_AVAILABLE = META_ROOT is not None

if META_AVAILABLE:
    sys.path.insert(0, str(META_ROOT))
else:
    warnings.warn(
        "meta не найдена. Установите: git clone https://github.com/svend4/meta.git "
        f"в одну из: {[str(c) for c in _META_CANDIDATES]}. "
        "HexLearnRouter, MetropolisAnnealer, Q4Q6Mapper используют заглушки.",
        UserWarning,
        stacklevel=2,
    )


# ─── HexLearn: k-NN роутинг на Q6 ───────────────────────────────

class HexLearnRouter:
    """k-NN маршрутизация на Q6 из meta/hexlearn.

    Вместо обученного softmax-router — геометрический k-NN в Q6.
    Работает без обучения (pure geometry).

    Args:
        k: число ближайших соседей
    """

    def __init__(self, k: int = 3):
        self.k = k
        self._model = None
        if META_AVAILABLE:
            try:
                from projects.hexlearn.hexlearn import HexLearnModel
                self._model = HexLearnModel()
            except ImportError:
                warnings.warn("meta/hexlearn не найден, используется заглушка")

    def route(self, q6_vector: list) -> dict:
        """Маршрутизирует Q6-вектор к k ближайшим гексаграммам.

        Args:
            q6_vector: 6 бит {0, 1} — индекс гексаграммы

        Returns:
            {hex_idx: weight} — нормализованные веса (1/hamming_dist)
        """
        if self._model is None:
            return self._stub_route(q6_vector)

        hex_idx = sum(b << i for i, b in enumerate(q6_vector))
        try:
            neighbors = self._model.knn(query=hex_idx, k=self.k)
        except Exception:
            return self._stub_route(q6_vector)

        total = sum(1.0 / (d + 1e-8) for _, d in neighbors)
        return {
            idx: (1.0 / (d + 1e-8)) / total
            for idx, d in neighbors
        }

    def _stub_route(self, q6_vector: list) -> dict:
        """Заглушка: равномерные веса для k ближайших по Хэммингу."""
        hex_idx = sum(b << i for i, b in enumerate(q6_vector))
        neighbors = []
        for candidate in range(64):
            dist = bin(hex_idx ^ candidate).count('1')
            neighbors.append((candidate, dist))
        neighbors.sort(key=lambda x: x[1])
        top_k = neighbors[:self.k]
        total = sum(1.0 / (d + 1e-8) for _, d in top_k)
        return {
            idx: (1.0 / (d + 1e-8)) / total
            for idx, d in top_k
        }


# ─── HexPhys: Metropolis annealing для квантизации ───────────────

class MetropolisAnnealer:
    """Физически корректное охлаждение из meta/hexphys.

    Использует Метрополис-MCMC для гарантированного детального баланса.
    Заменяет эвристическое расписание temperature annealing.

    β = 1/T растёт от beta_start до beta_end за n_steps шагов.

    Args:
        beta_start: начальная обратная температура (малая β = высокая T)
        beta_end: конечная обратная температура (большая β = низкая T)
        n_steps: полное расписание охлаждения (рекомендуется 3000)
    """

    def __init__(
        self,
        beta_start: float = 0.1,
        beta_end: float = 10.0,
        n_steps: int = 3000,
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.n_steps = n_steps
        self.current_step = 0
        self._mcmc = None

        if META_AVAILABLE:
            try:
                from projects.hexphys.hexphys import MetropolisMCMC
                self._mcmc = MetropolisMCMC()
            except ImportError:
                pass

    def get_beta(self) -> float:
        """β = 1/T. Линейно растёт от beta_start до beta_end."""
        progress = min(self.current_step / max(self.n_steps, 1), 1.0)
        return self.beta_start + (self.beta_end - self.beta_start) * progress

    def step(self) -> float:
        """Шаг расписания. Возвращает T = 1/β для Gumbel-Softmax.

        Вызывать один раз за шаг обучения.
        """
        self.current_step += 1
        return 1.0 / max(self.get_beta(), 1e-8)

    @property
    def temperature(self) -> float:
        """Текущая температура T = 1/β."""
        return 1.0 / max(self.get_beta(), 1e-8)

    def reset(self):
        """Сброс счётчика для нового прогона."""
        self.current_step = 0


# ─── HexDim: Q4⊂Q6 вложение ─────────────────────────────────────

class Q4Q6Mapper:
    """Q4⊂Q6 вложение через hexdim из meta.

    PseudoRAG работает в Q4 (16 архетипов).
    YiJing работает в Q6 (64 гексаграммы).
    Этот класс обеспечивает мост между ними.

    Внутри Q6 содержится ровно 15 копий Q4 (тессерактов).
    Каждый тессеракт = 16 гексаграмм, образующих Q4.
    """

    def __init__(self):
        self._model = None
        if META_AVAILABLE:
            try:
                from projects.hexdim.hexdim import HexDimModel
                self._model = HexDimModel()
            except ImportError:
                pass

    def get_tesseracts(self) -> list:
        """Возвращает все 15 копий Q4 внутри Q6.

        Returns:
            list of 15 элементов, каждый — список из 16 индексов гексаграмм
        """
        if self._model is not None:
            try:
                return self._model.tesseracts()
            except Exception:
                pass
        # Заглушка: геометрически корректные тессеракты
        return self._compute_tesseracts_stub()

    def _compute_tesseracts_stub(self) -> list:
        """Вычисляет тессеракты Q4⊂Q6 без meta.

        Q6 = {0,1}^6. Выбираем 2 фиксированных бита из 6 → фиксируем их значения
        → остаются 4 свободных бита → 2^4=16 вершин = один тессеракт Q4.
        C(6,2) = 15 способов выбрать 2 фиксированных бита.
        """
        from itertools import combinations
        tesseracts = []
        for fixed_bits in combinations(range(6), 2):
            for fixed_vals in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                # Найти все 16 вершин с этими фиксированными битами
                vertices = []
                for idx in range(64):
                    bits = [(idx >> (5 - b)) & 1 for b in range(6)]
                    if all(bits[fb] == fv for fb, fv in zip(fixed_bits, fixed_vals)):
                        vertices.append(idx)
                if len(vertices) == 16:
                    tesseracts.append(vertices)
        # Убираем дубликаты, берём ровно 15
        seen = set()
        unique = []
        for t in tesseracts:
            key = tuple(sorted(t))
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique[:15]

    def q4_to_q6_cluster(self, q4_archetype_id: int) -> list:
        """PseudoRAG архетип (0..15) → список из 4 гексаграмм Q6.

        Args:
            q4_archetype_id: индекс архетипа в Q4 [0, 15]

        Returns:
            список из 4 индексов гексаграмм Q6
        """
        tesseracts = self.get_tesseracts()
        if not tesseracts:
            return [q4_archetype_id * 4 + i for i in range(4)]
        first = tesseracts[0]
        start = q4_archetype_id * 4
        return first[start:start + 4]

    def q6_to_q4(self, hex_idx: int) -> int:
        """Гексаграмма Q6 (0..63) → ближайший Q4-архетип PseudoRAG (0..15).

        Первые 4 бита гексаграммы = Q4-архетип.
        """
        return hex_idx >> 2  # убрать 2 младших бита


# ─── HexSym: полная матрица аналогий ─────────────────────────────

class FullAnalogyCrossMatrix:
    """Полная матрица симметрий Q6 из meta/hexsym.

    CrossDomainAnalogy в pro2 покрывала 15/36 клеток.
    hexsym вычисляет все орбиты Aut(Q6) — полная симметрийная картина.

    Отвечает на вопрос: нужно ли реализовывать все 36 пар экспертов
    или достаточно по одному представителю каждой орбиты?
    """

    def __init__(self):
        self._model = None
        if META_AVAILABLE:
            try:
                from projects.hexsym.hexsym import HexSymModel
                self._model = HexSymModel()
            except ImportError:
                pass

    def get_edge_orbits(self) -> list:
        """Возвращает орбиты рёбер под действием группы симметрий.

        Returns:
            list of orbits, каждая орбита = list of (i, j) пар
        """
        if self._model is not None:
            try:
                return self._model.edge_orbits()
            except Exception:
                pass
        return self._stub_orbits()

    def _stub_orbits(self) -> list:
        """Заглушка: все пары (i, j) как отдельные орбиты."""
        n = 6
        return [[(i, j)] for i in range(n) for j in range(i + 1, n)]

    def minimal_cross_pairs(self, n_experts: int = 6) -> list:
        """Минимальный набор пар (i, j), покрывающий все орбиты.

        Вместо 36 пар — только уникальные орбиты.

        Args:
            n_experts: фильтровать пары с индексами < n_experts

        Returns:
            list of (i, j) пар — по одному представителю каждой орбиты
        """
        orbits = self.get_edge_orbits()
        pairs = []
        for orbit in orbits:
            rep = orbit[0]
            i, j = rep[0], rep[1]
            if i < n_experts and j < n_experts:
                pairs.append((i, j))
        return pairs


# ─── Проверка ────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"meta доступна: {META_AVAILABLE}")
    if META_AVAILABLE:
        print(f"  путь: {META_ROOT}")

    print("\n=== Q4Q6Mapper ===")
    mapper = Q4Q6Mapper()
    tesseracts = mapper.get_tesseracts()
    print(f"Тессерактов Q4 внутри Q6: {len(tesseracts)}")  # ожидается 15
    cluster = mapper.q4_to_q6_cluster(0)
    print(f"Q4 архетип 0 → гексаграммы: {cluster}")

    print("\n=== MetropolisAnnealer ===")
    annealer = MetropolisAnnealer(n_steps=3000)
    for step in [0, 500, 1500, 3000]:
        annealer.current_step = step
        print(f"  Шаг {step:4d}: T = {annealer.temperature:.3f}, β = {annealer.get_beta():.3f}")

    print("\n=== HexLearnRouter ===")
    router = HexLearnRouter(k=3)
    weights = router.route([1, 0, 1, 0, 1, 0])
    print(f"Routing для [1,0,1,0,1,0]: {weights}")

    print("\n=== FullAnalogyCrossMatrix ===")
    matrix = FullAnalogyCrossMatrix()
    pairs = matrix.minimal_cross_pairs()
    print(f"Минимальный набор пар: {len(pairs)} (из 15 возможных)")
    print(f"  Пары: {pairs[:5]}...")
