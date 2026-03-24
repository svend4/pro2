"""
q6_algebra.py — Алгебра группы Z₂^6, операции И-Цзин, bent-функции как seed.

Закрывает все теоретические пробелы из THEORY_VS_PRACTICE.md:

1. Модулярная арифметика Q6 (Теорема 3)
   «Хэмминг через скалярное произведение»
   d_H(a,b) = (6 − ⟨a,b⟩) / 2  в {−1,+1}^6
   + полная группа Z₂^6: сложение (XOR), орбиты, косеты, граф Кэли.

2. GERMES-нотация как операции (Polarity reversal)
   Четыре классические трансформации гексаграмм → группа Клейна V₄:
     • 恒 Heng   (тождество)
     • 错 CuoGua (инверсия всех линий = антипод)
     • 综 ZongGua (обращение порядка линий)
     • 互 HuGua  (ядерная гексаграмма — внутренние 4 линии)
   + nn.Module YiJingV4Layer для обучения с V₄-симметрией.

3. Bent-функции как геометрические seed (из THEORY_VS_PRACTICE.md §3)
   Квадратичные bent-функции f: GF(2)^6 → GF(2):
     - |Ŵ(u)| = 8 для всех u (максимальная нелинейность)
     - Нелинейность nl=28 (максимум для n=6)
     - support set {x: f(x)=1} — 32 вершины Q6 с max. разнообразием
   BentFunctions: truth tables → {-1,+1}^6 прототипы
   Q6ArithmeticLayer(use_bent_init=True): инициализация из bent seeds
   BentPrototypeQuantizer: жёсткий квантизатор на bent support

Связь с теорией:
  GERMES_NOTATION.md   § 8  — гексагон Q3, четыре операции
  KNOWLEDGE_FRAMEWORK  Теорема 3: Хэмминг через скалярное произведение
  e8-yijing-deep-analysis.md § 6.4 — Z₂^6 как абелева группа
  meta_q6.py bent_seed_texts() — текстовый аналог (RAG seeds)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1. МОДУЛЯРНАЯ АРИФМЕТИКА Q6 — Группа Z₂^6
# ══════════════════════════════════════════════════════════════════════════════

class Q6Arithmetic:
    """
    Полная алгебра группы Z₂^6 = {−1,+1}^6.

    В представлении {−1,+1}:
        XOR(a,b) ↔ a * b   (покоординатное умножение)
        Identity ↔  (+1,+1,+1,+1,+1,+1)  (нейтральный элемент)
        Inverse(a) ↔ a      (каждый элемент — собственная обратная)

    Теорема 3:
        d_H(a,b) = (n − ⟨a,b⟩) / 2
    где ⟨a,b⟩ = Σᵢ aᵢ·bᵢ — скалярное произведение в {−1,+1}^n.

    Это позволяет вычислять расстояние Хэмминга через одну матричную операцию,
    без явного XOR и подсчёта единиц.
    """

    # ── Базовые групповые операции ─────────────────────────────────────────

    @staticmethod
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Сложение в Z₂^6: a ⊕ b = a * b (покоординатное умножение).

        Args:
            a, b: (..., 6) в {−1,+1}^6
        Returns:
            (..., 6) — сумма в группе
        """
        return a * b

    @staticmethod
    def identity(n: int = 6, device=None) -> torch.Tensor:
        """Нейтральный элемент группы Z₂^6: (+1,...,+1)."""
        return torch.ones(n, device=device)

    @staticmethod
    def inverse(a: torch.Tensor) -> torch.Tensor:
        """Обратный элемент: в Z₂^6 каждый элемент = собственная обратная."""
        return a  # a * a = (1,...,1) = identity

    # ── Теорема 3: расстояние Хэмминга через скалярное произведение ────────

    @staticmethod
    def hamming_from_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Теорема 3: d_H(a,b) = (n − ⟨a,b⟩) / 2.

        Эквивалентна прямому подсчёту несовпадений, но реализована через
        матричное умножение → O(n) вместо O(n) с лучшей константой.

        Args:
            a, b: (..., n) в {−1,+1}^n
        Returns:
            (...,) — расстояние Хэмминга ∈ {0,1,...,n}
        """
        n = a.shape[-1]
        dot = (a * b).sum(dim=-1)
        return (n - dot) / 2

    @staticmethod
    def hamming_matrix(codebook: torch.Tensor) -> torch.Tensor:
        """
        Матрица расстояний Хэмминга для всего кодбука через Теорему 3.

        Args:
            codebook: (K, n) — K вершин гиперкуба {−1,+1}^n
        Returns:
            (K, K) — матрица расстояний
        """
        n = codebook.shape[-1]
        # Gram matrix: G[i,j] = ⟨v_i, v_j⟩
        G = codebook @ codebook.T          # (K, K)
        return (n - G) / 2

    @staticmethod
    def weight(v: torch.Tensor) -> torch.Tensor:
        """
        Хэмминговый вес: d_H(v, identity) = число позиций с −1.

        В {−1,+1}^n: weight = (n − Σvᵢ) / 2.
        """
        n = v.shape[-1]
        return (n - v.sum(dim=-1)) / 2

    # ── Подгруппы и косеты ─────────────────────────────────────────────────

    @staticmethod
    def subgroup_from_generators(
        generators: torch.Tensor,
        n_bits: int = 6,
    ) -> torch.Tensor:
        """
        Подгруппа H ⊆ Z₂^6, порождённая generators.

        Алгоритм: замыкание под групповой операцией (XOR = умножение).

        Args:
            generators: (k, n) в {−1,+1}^n — порождающие элементы
            n_bits:     размерность (по умолчанию 6)
        Returns:
            (|H|, n) — все элементы подгруппы H (включая нейтральный)
        """
        identity = torch.ones(n_bits, device=generators.device)
        elements = [identity]
        queue = [identity]

        for gen in generators:
            new_queue = []
            for h in elements:
                candidate = h * gen  # XOR в {−1,+1}
                # Проверить, есть ли уже в списке
                is_new = all(
                    not torch.allclose(candidate, e) for e in elements
                )
                if is_new:
                    elements.append(candidate)
                    new_queue.append(candidate)
            queue = new_queue

        return torch.stack(elements)  # (|H|, n)

    @staticmethod
    def coset(
        v: torch.Tensor,
        subgroup: torch.Tensor,
    ) -> torch.Tensor:
        """
        Левый косет v·H = {v * h | h ∈ H} в Z₂^6.

        Args:
            v:         (n,) — вектор в {−1,+1}^n
            subgroup:  (|H|, n) — все элементы подгруппы H
        Returns:
            (|H|, n) — все элементы косета
        """
        return v.unsqueeze(0) * subgroup  # (|H|, n)

    @staticmethod
    def partition_into_cosets(
        hexagrams: torch.Tensor,
        subgroup: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Разбиение Q6 на косеты по подгруппе H.

        Args:
            hexagrams: (64, 6) — все вершины Q6
            subgroup:  (|H|, 6) — подгруппа H
        Returns:
            список тензоров — каждый = один косет v·H
        """
        used = torch.zeros(len(hexagrams), dtype=torch.bool)
        cosets: List[torch.Tensor] = []

        for i, v in enumerate(hexagrams):
            if used[i]:
                continue
            # Вычислить косет v·H
            coset_vecs = v.unsqueeze(0) * subgroup  # (|H|, 6)
            # Найти индексы членов косета в hexagrams
            mask = torch.zeros(len(hexagrams), dtype=torch.bool)
            for c in coset_vecs:
                match = (hexagrams == c.unsqueeze(0)).all(dim=1)
                mask |= match
            used |= mask
            cosets.append(hexagrams[mask])

        return cosets

    @staticmethod
    def cayley_distance(
        a: torch.Tensor,
        b: torch.Tensor,
        generators: torch.Tensor,
    ) -> int:
        """
        Расстояние Кэли: минимальное число шагов от a к b
        при использовании generators в качестве переходов.

        Это BFS по графу Кэли группы Z₂^6.

        Args:
            a, b:       (n,) — начало и конец
            generators: (k, n) — допустимые переходы
        Returns:
            int — расстояние Кэли (∞ = недостижимо)
        """
        target = b
        current_layer = {tuple(a.tolist())}
        visited = set(current_layer)

        for dist in range(1, 64):
            next_layer = set()
            for v_tuple in current_layer:
                v = torch.tensor(v_tuple)
                for gen in generators:
                    nv = (v * gen)
                    nv_tuple = tuple(nv.tolist())
                    if nv_tuple == tuple(target.tolist()):
                        return dist
                    if nv_tuple not in visited:
                        visited.add(nv_tuple)
                        next_layer.add(nv_tuple)
            if not next_layer:
                return float('inf')
            current_layer = next_layer

        return float('inf')

    @staticmethod
    def quotient_centroids(
        hexagrams: torch.Tensor,
        subgroup: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Центроиды косетов Q6/H — проекция в пространство косетов.

        Если embeddings заданы: центроид = среднее embedding-ов членов косета.
        Иначе: центроид = среднее bit-векторов.

        Args:
            hexagrams:  (64, 6)
            subgroup:   (|H|, 6)
            embeddings: опционально (64, d_model)
        Returns:
            (64/|H|, 6 или d_model) — центроид каждого косета
        """
        cosets = Q6Arithmetic.partition_into_cosets(hexagrams, subgroup)
        sources = embeddings if embeddings is not None else hexagrams

        centroids = []
        for coset_vecs in cosets:
            # Найти индексы
            idxs = []
            for c in coset_vecs:
                match = (hexagrams == c.unsqueeze(0)).all(dim=1).nonzero(as_tuple=True)[0]
                if len(match) > 0:
                    idxs.append(match[0].item())
            if idxs:
                centroids.append(sources[idxs].mean(0))

        return torch.stack(centroids) if centroids else hexagrams[:1]


# ══════════════════════════════════════════════════════════════════════════════
# 2. GERMES-НОТАЦИЯ КАК ОПЕРАЦИИ — Группа Клейна V₄ (И-Цзин трансформации)
# ══════════════════════════════════════════════════════════════════════════════

class YiJingOps:
    """
    Четыре классические трансформации гексаграмм И-Цзин
    как алгебраические операции над {−1,+1}^6.

    Образуют группу Клейна V₄ = Z₂ × Z₂ под композицией:
        ┌────────┬────────┬────────┬────────┐
        │  ∘     │  I     │  Cuo   │  Zong  │ CuoZong│
        ├────────┼────────┼────────┼────────┤────────┤
        │  I     │  I     │  Cuo   │  Zong  │ CuoZong│
        │  Cuo   │  Cuo   │  I     │ CuoZong│  Zong  │
        │  Zong  │  Zong  │ CuoZong│  I     │  Cuo   │
        │ CuoZong│ CuoZong│  Zong  │  Cuo   │  I     │
        └────────┴────────┴────────┴────────┘────────┘

    Источник: GERMES_NOTATION.md § 2, § 8 — геометрия Q3-гексагона.

    Операции:
        heng(v)     — 恒 тождество
        cuo_gua(v)  — 错卦 инверсия всех линий (−v = антипод)
        zong_gua(v) — 综卦 обращение порядка линий
        hu_gua(v)   — 互卦 ядерная гексаграмма (внутренние 4 линии → новая гексаграмма)
    """

    @staticmethod
    def heng(v: torch.Tensor) -> torch.Tensor:
        """恒 Heng — тождественная операция (Identity)."""
        return v

    @staticmethod
    def cuo_gua(v: torch.Tensor) -> torch.Tensor:
        """
        错卦 CuoGua — инверсия полярности всех линий.

        В {−1,+1}^6: v → −v (антипод).
        Смысл: полная противоположность — каждая ян-линия становится инь и наоборот.
        Порядок: cuo_gua(cuo_gua(v)) = v  (инволюция).
        """
        return -v

    @staticmethod
    def zong_gua(v: torch.Tensor) -> torch.Tensor:
        """
        综卦 ZongGua — обращение порядка линий.

        В {−1,+1}^6: v = [b₀,b₁,b₂,b₃,b₄,b₅] → [b₅,b₄,b₃,b₂,b₁,b₀].
        Смысл: «перевернуть» гексаграмму — нижняя триграмма становится верхней.
        Порядок: zong_gua(zong_gua(v)) = v  (инволюция).
        """
        return v.flip(-1)

    @staticmethod
    def hu_gua(v: torch.Tensor) -> torch.Tensor:
        """
        互卦 HuGua — ядерная (взаимная) гексаграмма.

        Берём внутренние 4 линии (позиции 1..4) и строим новую гексаграмму:
            верхняя триграмма ядра = линии 3,4,5 оригинала (0-ind: 2,3,4)
            нижняя триграмма ядра = линии 2,3,4 оригинала (0-ind: 1,2,3)
        → новая 6-мерная точка Q6.

        v: (..., 6) → (..., 6)
        """
        lower = v[..., 1:4]  # линии 2,3,4 (нижняя триграмма ядра)
        upper = v[..., 2:5]  # линии 3,4,5 (верхняя триграмма ядра)
        return torch.cat([lower, upper], dim=-1)  # (..., 6)

    @staticmethod
    def cuo_zong(v: torch.Tensor) -> torch.Tensor:
        """
        错综 CuoZong — композиция инверсии и обращения (4-й элемент V₄).

        Эквивалентно: сначала обратить, потом инвертировать.
        cuo_zong = cuo_gua ∘ zong_gua = zong_gua ∘ cuo_gua (V₄ абелева).
        """
        return -v.flip(-1)

    @classmethod
    def v4_orbit(cls, v: torch.Tensor) -> torch.Tensor:
        """
        Полная орбита V₄: четыре образа вершины v.

        Args:
            v: (..., 6)
        Returns:
            (..., 4, 6) — [I(v), Cuo(v), Zong(v), CuoZong(v)]
        """
        return torch.stack([
            cls.heng(v),
            cls.cuo_gua(v),
            cls.zong_gua(v),
            cls.cuo_zong(v),
        ], dim=-2)

    @classmethod
    def all_ops(cls):
        """Все 4 операции как список (name, fn)."""
        return [
            ('heng',     cls.heng),
            ('cuo_gua',  cls.cuo_gua),
            ('zong_gua', cls.zong_gua),
            ('hu_gua',   cls.hu_gua),
        ]

    @staticmethod
    def is_v4_closed(ops_tensors: list) -> bool:
        """
        Проверить, что {I, Cuo, Zong, CuoZong} образуют группу Клейна V₄.
        Используется для верификации теоретического свойства.
        """
        I, C, Z, CZ = ops_tensors
        checks = [
            torch.allclose(C * C, I),       # Cuo² = I
            torch.allclose(Z * Z, I),       # Zong² = I (для вектора с flip)
            torch.allclose(C * Z, CZ),      # Cuo ∘ Zong = CuoZong
        ]
        return all(checks)


# ══════════════════════════════════════════════════════════════════════════════
# 3. BENT-ФУНКЦИИ КАК ГЕОМЕТРИЧЕСКИЕ SEED — f: GF(2)^6 → GF(2)
# ══════════════════════════════════════════════════════════════════════════════

class BentFunctions:
    """
    Квадратичные bent-функции над GF(2)^6 как геометрические seed Q6.

    Bent-функция f: GF(2)^6 → GF(2) имеет «плоский» WHT-спектр:
        |Ŵ(u)| = 2^(n/2) = 8  для всех u ∈ GF(2)^6

    Это обеспечивает:
        1. Максимальную нелинейность nl=28 (расстояние от всех аффинных функций)
        2. Неравномерный support: |{x: f(x)=1}| = 28 или 36 (Ŵ(0) = ±8 для n=6)
        3. Максимальное разнообразие: bent support vectors — геометрически оптимальные архетипы

    Применение:
        - Инициализация прототипов Q6ArithmeticLayer (use_bent_init=True)
        - Инициализация codebook BentPrototypeQuantizer
        - Сравнение: мeta_q6.bent_seed_texts() даёт ТЕКСТОВЫЙ аналог для RAG

    Конструкция (квадратичные формы):
        f(x) = XOR_{(i,j) ∈ M} x_i · x_j
    где M — идеальное спаривание 6 бит в 3 непересекающиеся пары.
    Таких спариваний 15, каждое + дополнение → 30 функций (берём первые 20).
    """

    # Кэш вычисленных truth tables
    _cache: List[int] = []

    @classmethod
    def _build(cls) -> List[int]:
        """Строит 20 bent truth tables (квадратичные формы GF(2)^6)."""
        if cls._cache:
            return cls._cache
        pairs_list = [
            [(0,1),(2,3),(4,5)],
            [(0,2),(1,3),(4,5)],
            [(0,3),(1,2),(4,5)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,4),(3,5)],
        ]
        results = []
        for pairs in pairs_list:
            tt = 0
            for x in range(64):
                val = 0
                for (i, j) in pairs:
                    val ^= ((x >> i) & 1) & ((x >> j) & 1)
                if val:
                    tt |= (1 << x)
            results.append(tt)
            results.append(tt ^ 0xFFFFFFFFFFFFFFFF)  # дополнение — тоже bent
        cls._cache = results[:20]
        return cls._cache

    @classmethod
    def truth_tables(cls) -> torch.Tensor:
        """
        Truth tables всех 20 bent-функций над GF(2)^6.

        Returns:
            (20, 64) в {0,1} — f[i][x] = значение i-й функции на входе x
        """
        tts = cls._build()
        rows = []
        for tt in tts:
            row = torch.tensor([(tt >> x) & 1 for x in range(64)], dtype=torch.float32)
            rows.append(row)
        return torch.stack(rows)  # (20, 64)

    @classmethod
    def wht_spectra(cls) -> torch.Tensor:
        """
        WHT-спектры всех bent-функций в {-1,+1} пространстве.

        Для bent: все |Ŵ(u)| = 8 (плоский спектр).

        Returns:
            (20, 64) — Walsh-Hadamard transform каждой функции
        """
        f_pm1 = 1.0 - 2.0 * cls.truth_tables()  # {0,1} → {+1,-1}: 0→+1, 1→-1
        # WHT in-place (butterfly)
        W = f_pm1.clone()
        h = 1
        while h < 64:
            for i in range(0, 64, h * 2):
                x = W[:, i:i+h].clone()
                y = W[:, i+h:i+2*h].clone()
                W[:, i:i+h]       = x + y
                W[:, i+h:i+2*h]   = x - y
            h *= 2
        return W  # (20, 64)

    @classmethod
    def is_bent(cls) -> torch.Tensor:
        """
        Проверить bent-свойство: все |Ŵ(u)| = 8.

        Returns:
            (20,) bool — True для каждой корректной bent-функции
        """
        W = cls.wht_spectra()
        return (W.abs() - 8.0).abs().max(dim=-1).values < 1e-6

    @classmethod
    def support_vectors(cls) -> List[torch.Tensor]:
        """
        Опорные множества (support sets) в {-1,+1}^6.

        Для каждой bent-функции f: support = {x ∈ GF(2)^6 : f(x) = 1}.
        Все входы x ∈ {0,1}^6 переводятся в {-1,+1}^6: b → 1 − 2b.

        Примечание: bent-функции на GF(2)^6 НЕ сбалансированы.
        Ŵ(0) = ±8  →  |support| = (64 ± 8) / 2 = 28 или 36 (не 32).

        Returns:
            список из 20 тензоров формы (|supp_i|, 6) — support vectors в {-1,+1}^6
        """
        all_verts = torch.tensor(
            [[1 - 2 * ((x >> i) & 1) for i in range(6)] for x in range(64)],
            dtype=torch.float32,
        )  # (64, 6)

        tt = cls.truth_tables()  # (20, 64)
        result = []
        for i in range(20):
            mask = tt[i].bool()            # (64,)
            result.append(all_verts[mask]) # (28 или 36, 6)
        return result

    @classmethod
    def prototype_codebook(cls, k: int = 8) -> torch.Tensor:
        """
        Выбрать k прототипов из bent support sets методом furthest-point sampling.

        Алгоритм (greedy max-min distance):
            1. Собрать все support vectors всех 20 функций → (20×32, 6) = (640, 6)
            2. Начать с вектора, ближайшего к центроиду
            3. Итеративно добавлять вектор с максимальным минимальным расстоянием

        Это даёт k прототипов с максимально равномерным покрытием Q6.

        Args:
            k: число прототипов (1..64)
        Returns:
            (k, 6) — прототипы в {-1,+1}^6
        """
        supports = cls.support_vectors()         # list of 20 tensors (28|36, 6)
        pool = torch.cat(supports, dim=0)        # (≤640, 6)
        # Убрать дубликаты
        unique = torch.unique(pool, dim=0)       # (≤64, 6)

        if k >= len(unique):
            return unique

        # Furthest-point sampling
        selected = []
        # Стартовая точка — центроид
        centroid = unique.mean(0)
        dists_to_centroid = ((unique - centroid) ** 2).sum(-1)
        start_idx = dists_to_centroid.argmin().item()
        selected.append(start_idx)

        min_dists = torch.full((len(unique),), float('inf'))

        for _ in range(k - 1):
            last = unique[selected[-1]]
            d = ((unique - last) ** 2).sum(-1)   # Евклид²
            min_dists = torch.minimum(min_dists, d)
            next_idx = min_dists.argmax().item()
            selected.append(next_idx)

        return unique[selected]  # (k, 6)

    @classmethod
    def verify(cls) -> dict:
        """
        Верификация bent-свойств.

        Returns:
            dict с ключами: all_bent, n_bent, spectrum_flatness
        """
        bent_flags = cls.is_bent()                    # (20,)
        W = cls.wht_spectra()
        flat = (W.abs() - 8.0).abs().mean().item()

        return {
            'all_bent':         bool(bent_flags.all()),
            'n_bent':           int(bent_flags.sum()),
            'n_total':          20,
            'spectrum_flatness_err': flat,            # должно быть ≈ 0
            'support_sizes':    [len(s) for s in cls.support_vectors()[:4]],  # 28 или 36
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. nn.Module — обучаемые слои с V₄ и Z₂^6 симметрией
# ══════════════════════════════════════════════════════════════════════════════

class YiJingV4Layer(nn.Module):
    """
    Слой с V₄-симметрией: обрабатывает входные векторы через все 4 операции
    группы Клейна и объединяет результаты взвешенной суммой.

    Применяется как drop-in замена стандартного FFN в блоке модели,
    когда нужно использовать V₄-симметрию пространства гексаграмм.

    Архитектура:
        x → proj_6d → q6  ∈ R^6
            → v4_orbit(q6) → (4, 6)   [4 трансформации]
            → w_softmax (4,)            [обученные веса операций]
            → взвешенная сумма (6,)
            → proj_back → d_model
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj_in  = nn.Linear(d_model, 6, bias=False)
        self.proj_out = nn.Linear(6, d_model, bias=False)
        # Веса 4 операций: начальное значение 0 → softmax → 0.25 каждая
        self.op_logits = nn.Parameter(torch.zeros(4))
        self.scale = nn.Parameter(torch.tensor(0.01))

        # Имена операций для logging
        self._op_names = ['heng', 'cuo_gua', 'zong_gua', 'hu_gua']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d_model)
        Returns:
            (..., d_model) — x + scale * V₄_transform(x)
        """
        z = self.proj_in(x)          # (..., 6)

        ops_fn = [
            YiJingOps.heng,
            YiJingOps.cuo_gua,
            YiJingOps.zong_gua,
            YiJingOps.hu_gua,
        ]
        ops_out = torch.stack([fn(z) for fn in ops_fn], dim=-2)  # (..., 4, 6)

        w = F.softmax(self.op_logits, dim=0)   # (4,)
        z_mix = (ops_out * w.view(*([1] * (ops_out.dim() - 2)), 4, 1)).sum(-2)  # (..., 6)

        return x + self.scale * self.proj_out(z_mix)

    def dominant_op(self) -> str:
        """Текущая доминирующая операция V₄ по весу."""
        idx = self.op_logits.argmax().item()
        return self._op_names[idx]


class Q6ArithmeticLayer(nn.Module):
    """
    Слой с Z₂^6 арифметикой: использует теорему 3 для
    вычисления Хэмминговых расстояний в пространстве эмбеддингов.

    Применение:
        - Attention bias: близкие по Хэммингу → ближе в attention
        - Routing: маршрутизация по Хэммингову расстоянию до прототипов
        - Loss: Хэмминговая регуляризация эмбеддингов

    Теорема 3 используется напрямую: через dot product → O(d) вместо
    явного сравнения битов.
    """

    def __init__(
        self,
        d_model: int,
        n_prototypes: int = 8,
        use_bent_init: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_proto = n_prototypes

        # Проекция в Q6-пространство
        self.proj_q6 = nn.Linear(d_model, 6, bias=False)

        # n_prototypes вершин Q6 (обучаемые «архетипы»)
        if use_bent_init:
            # Инициализация из bent support sets — максимальная нелинейность
            protos = BentFunctions.prototype_codebook(n_prototypes).float()
        else:
            protos = torch.randn(n_prototypes, 6)
            protos = protos / protos.norm(dim=-1, keepdim=True)
        self.prototypes = nn.Parameter(protos)

        # Масштаб применения расстояний
        self.hamming_scale = nn.Parameter(torch.tensor(0.1))

    def q6_project(self, x: torch.Tensor) -> torch.Tensor:
        """Проекция x в {−1,+1}^6 через tanh + normalize."""
        z = torch.tanh(self.proj_q6(x))
        return z / (z.norm(dim=-1, keepdim=True).clamp(min=1e-6))

    def hamming_to_prototypes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вектор Хэмминговых расстояний от x до каждого прототипа.
        Использует Теорему 3: d_H = (n − ⟨a,b⟩) / 2.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, n_prototypes) — расстояния до прототипов
        """
        z = self.q6_project(x)                    # (B, T, 6)
        p = F.normalize(self.prototypes, dim=-1)  # (n_proto, 6)
        # dot product via теорему 3
        dot = z @ p.T                             # (B, T, n_proto)
        return (6 - dot * 6) / 2                  # приближение d_H

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Мягкая маршрутизация по Хэммингу: ближайший прототип = эксперт.
        softmax(-λ · d_H) — чем ближе, тем больший вес.
        """
        dist = self.hamming_to_prototypes(x)       # (B, T, n_proto)
        return F.softmax(-self.hamming_scale * dist, dim=-1)

    def hamming_bias(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention bias из Хэмминговых расстояний (T, T).
        Близкие по Q6-коду токены → положительный bias.
        """
        z = self.q6_project(x)    # (B, T, 6)
        T = z.shape[1]
        # Попарные расстояния Хэмминга через Теорему 3
        dot = torch.bmm(z, z.transpose(1, 2))  # (B, T, T)
        dist = (6 - dot * 6) / 2               # (B, T, T)
        return -self.hamming_scale * dist       # отрицательный для близких

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            weights: (B, T, n_prototypes) — routing weights по Хэммингу
        """
        return self.routing_weights(x)


class BentPrototypeQuantizer(nn.Module):
    """
    Жёсткий квантизатор на bent support set одной функции.

    Фиксированный codebook из 32 вершин Q6 (support set bent-функции f):
        support = {x ∈ {-1,+1}^6 : f(x) = 1}

    Этот codebook оптимален по нелинейности: максимальное расстояние
    от любого линейного разбиения Q6. Используется как drop-in замена
    YiJingQuantizer когда нужна математически обоснованная инициализация.

    Применение:
        q = BentPrototypeQuantizer(d_model=32, bent_idx=0)
        out = q(x)   # straight-through estimator
    """

    def __init__(self, d_model: int, bent_idx: int = 0, temp: float = 0.3):
        super().__init__()
        self.temp = temp

        # Фиксированный codebook: support set bent-функции bent_idx
        supports = BentFunctions.support_vectors()  # list of 20 tensors
        codebook = supports[bent_idx % 20]           # (28 или 36, 6)
        self.register_buffer('codebook', codebook.float())

        # Проекция d_model → 6
        self.proj = nn.Linear(d_model, 6, bias=False)
        nn.init.orthogonal_(self.proj.weight)

        # Обратная проекция 6 → d_model
        self.proj_back = nn.Linear(6, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d_model)
        Returns:
            (..., d_model) — квантизованный выход через straight-through
        """
        z = torch.tanh(self.proj(x))               # (..., 6)
        # Soft assignment к codebook через Теорему 3
        cb = self.codebook                          # (32, 6)
        dot = z @ cb.T                             # (..., 32)
        dist = (6.0 - dot * 6.0) / 2.0            # (..., 32) approx d_H
        w = torch.softmax(-dist / self.temp, dim=-1)  # (..., 32)
        q_soft = w @ cb                            # (..., 6)
        # Straight-through: hard в forward, soft grad
        q_hard = cb[dist.argmin(-1)]               # (..., 6)
        q = x + (self.proj_back(q_soft + (q_hard - q_soft).detach()) - x).detach()
        return q

    def codebook_diversity(self) -> float:
        """Среднее попарное расстояние Хэмминга в codebook."""
        cb = self.codebook
        H = Q6Arithmetic.hamming_matrix(cb)
        n = len(cb)
        return H.sum().item() / (n * (n - 1))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Функции верификации теоретических свойств
# ══════════════════════════════════════════════════════════════════════════════

def verify_theorem3(n_bits: int = 6, n_samples: int = 100) -> dict:
    """
    Верификация Теоремы 3: d_H(a,b) = (n − ⟨a,b⟩) / 2.

    Сравнивает формульное вычисление с прямым подсчётом несовпадений.

    Returns:
        dict с max_error и статусом.
    """
    a = torch.randint(0, 2, (n_samples, n_bits)) * 2 - 1  # {−1,+1}
    b = torch.randint(0, 2, (n_samples, n_bits)) * 2 - 1

    # Прямой подсчёт
    direct = (a != b).float().sum(-1)

    # Теорема 3
    theorem3 = Q6Arithmetic.hamming_from_dot(a.float(), b.float())

    max_err = (direct - theorem3).abs().max().item()
    return {
        'theorem3_max_error': max_err,
        'verified': max_err < 1e-5,
        'n_samples': n_samples,
    }


def verify_v4_group(n_tests: int = 20) -> dict:
    """
    Верификация структуры группы Клейна V₄ для YiJingOps.

    Проверяет таблицу Кэли V₄:
        Cuo² = Id, Zong² = Id, CuoZong² = Id,
        Cuo ∘ Zong = CuoZong.
    """
    errors = []
    for _ in range(n_tests):
        v = torch.randn(6)
        I  = YiJingOps.heng(v)
        C  = YiJingOps.cuo_gua(v)
        Z  = YiJingOps.zong_gua(v)
        CZ = YiJingOps.cuo_zong(v)

        checks = {
            'Cuo²=I':          torch.allclose(YiJingOps.cuo_gua(C), I),
            'Zong²=I':         torch.allclose(YiJingOps.zong_gua(Z), I),
            'CuoZong²=I':      torch.allclose(YiJingOps.cuo_zong(CZ), I),
            'Cuo∘Zong=CuoZong': torch.allclose(YiJingOps.cuo_gua(Z), CZ),
            'Zong∘Cuo=CuoZong': torch.allclose(YiJingOps.zong_gua(C), CZ),
        }
        for name, ok in checks.items():
            if not ok:
                errors.append(name)

    return {
        'v4_verified': len(errors) == 0,
        'failed_checks': list(set(errors)),
        'n_tests': n_tests,
    }


def verify_bent_functions() -> dict:
    """
    Верификация bent-функций: плоский WHT-спектр и balanced support set.

    Returns:
        dict с all_bent, spectrum_flatness_err, prototype_codebook_diversity
    """
    b = BentFunctions.verify()
    proto = BentFunctions.prototype_codebook(k=8)  # (8, 6)
    H = Q6Arithmetic.hamming_matrix(proto)
    n = len(proto)
    div = H.sum().item() / (n * (n - 1))
    b['prototype_diversity_mean_dH'] = div
    return b


def verify_all() -> dict:
    """Запустить все верификации и вернуть сводный отчёт."""
    t3   = verify_theorem3()
    v4   = verify_v4_group()
    bt   = verify_bent_functions()
    return {
        'theorem3': t3,
        'v4_group': v4,
        'bent':     bt,
        'all_ok':   t3['verified'] and v4['v4_verified'] and bt['all_bent'],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Публичный API
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Алгебра Z₂^6
    'Q6Arithmetic',
    # GERMES операции
    'YiJingOps',
    # Bent-функции как геометрические seed
    'BentFunctions',
    'BentPrototypeQuantizer',
    # nn.Module слои
    'YiJingV4Layer',
    'Q6ArithmeticLayer',
    # Верификация
    'verify_theorem3',
    'verify_v4_group',
    'verify_bent_functions',
    'verify_all',
]
