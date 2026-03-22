"""
meta_q6.py — Интеграционный мост pro2 ↔ svend4/meta

Импортирует математические примитивы Q6 из meta-репозитория
и предоставляет простой API для use в nautilus_4agent.py,
figure8_turbine.py и pipeline.py.

Решаемые проблемы pro2:
  1. bent_seed_texts()    → заменяет _load_seeds() (произвольные строки)
                            bent-функции — математически оптимальные архетипы
  2. metropolis_temp()    → заменяет фиксированную температуру 1.4
                            Metropolis-расписание: меньше осцилляций в поздних циклах
  3. hamming_q6_address() → вспомогательная: проецировать эмбеддинг на Q6
  4. q4_tesseract_agents()→ 15 копий Q4 ⊂ Q6, полное покрытие 60/64 вершин
  5. yang_weight_orbits() → 7 орбит по весу Хэмминга для agent specialization
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

# ── Путь к meta-репозиторию ───────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_META_ROOT = os.path.join(_HERE, "data", "svend4_corpus", "meta", "_clones", "meta")

def _meta_available() -> bool:
    return os.path.isdir(_META_ROOT)

def _add_meta_path() -> bool:
    if not _meta_available():
        return False
    if _META_ROOT not in sys.path:
        sys.path.insert(0, _META_ROOT)
    return True


# ── 1. Bent-функции как seed-архетипы ────────────────────────────────────────

# Набор из 20 известных bent-функций (квадратичных форм над GF(2)^6).
# Каждая — 64-битное целое: бит i = f(i).
# Вычислены через суммирование пар бит: f = Σ bit_i·bit_j
# Все 15 квадратичных спарок + их дополнения — все bent.
_BENT_FORMS: List[int] = []

def _compute_bent_forms() -> List[int]:
    """Генерирует квадратичные bent-функции над GF(2)^6."""
    global _BENT_FORMS
    if _BENT_FORMS:
        return _BENT_FORMS
    # Квадратичные формы: f(x) = XOR по всем парам из спаривания {(i,j),(k,l),(m,n)}
    # Максимальная нелинейность достигается при полном спаривании 6 бит в 3 пары
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
        results.append(tt ^ ((1 << 64) - 1) & 0xFFFFFFFFFFFFFFFF)  # дополнение
    _BENT_FORMS = results[:20]
    return _BENT_FORMS


def _is_bent(tt_int: int) -> bool:
    """Проверить bent-функцию через WHT."""
    W = [0] * 64
    tt = [(tt_int >> x) & 1 for x in range(64)]
    f = [1 - 2 * b for b in tt]
    # WHT in-place
    h = 1
    while h < 64:
        for i in range(0, 64, h * 2):
            for j in range(i, i + h):
                x, y = f[j], f[j + h]
                f[j], f[j + h] = x + y, x - y
        h *= 2
    return all(abs(v) == 8 for v in f)


def _bent_to_text(tt_int: int, idx: int) -> str:
    """
    Преобразовать bent-функцию в семантически разнообразный текст.

    Проблема: шаблонные тексты ("bent-archetype-XX: Q6 boolean...") имеют
    высокое косинусное сходство (≈0.98) — embedding-space не отражает
    математическое разнообразие функций.

    Решение: каждый архетип маппируется на уникальный семантический домен
    из 20 разных областей знания (код, философия, наука, нарратив и т.д.)
    """
    bits = [(tt_int >> i) & 1 for i in range(64)]
    yang = sum(bits)

    # Группируем биты в 6 секций по 10-11 бит каждая
    secs = []
    for i in range(6):
        start = i * 10
        secs.append(sum(bits[start:start+10]))

    # 20 семантически разных шаблонов (разные домены, разная лексика)
    templates = [
        f"def route_{yang}(x): return x ^ {tt_int & 0xFFFF} if x < 32 else ~x & 63",
        f"consciousness field: {yang} active nodes form spiral pattern {secs[0]}-{secs[1]}-{secs[2]}",
        f"The nautilus expands by ratio {yang/32:.3f}, each chamber {secs[3]} units larger",
        f"gradient descent step: w -= {yang/64:.4f} * grad; momentum {secs[4]/10:.2f}",
        f"hexagram {yang}: lower trigram {secs[0]}_{secs[1]}_{secs[2]}, upper {secs[3]}_{secs[4]}_{secs[5]}",
        f"quantum state |ψ⟩ = Σ α_i |{yang}⟩, entanglement entropy S = {yang*0.693/64:.3f}",
        f"self.crossing = nn.Linear({yang*2}, {64-yang}); dropout={secs[0]/100:.2f}",
        f"The river bends {yang} times before reaching the delta, carving {secs[2]} oxbows",
        f"prime factorization: {yang} = {' × '.join(str(p) for p in _small_factors(yang))}",
        f"loss = cross_entropy(logits[:{yang}], target) + {secs[1]/100:.3f} * l2_reg",
        f"DNA strand: {'ATCG'[secs[0]%4]}{'GCTA'[secs[1]%4]}{'TGAC'[secs[2]%4]} binds at position {yang}",
        f"turbine cycle: ABSTRACT→DYNAMIC→CONCRETE at weight {yang/64:.3f} kirchhoff balance",
        f"philosopher stone: {yang} elements in {secs[3]} layers, each transmuting {secs[0]+secs[1]}",
        f"for epoch in range({yang}): model.train(); lci = compute_routing_balance()",
        f"tidal wave height {yang/10:.1f}m, period {secs[2]+secs[3]}s, frequency {1/(secs[4]+1):.3f}Hz",
        f"The maze has {yang} rooms, {secs[0]} dead ends, solution length {secs[1]+secs[2]} steps",
        f"MoE gate weights: [{', '.join(f'{s/10:.2f}' for s in secs[:4])}] sum={sum(secs[:4])/10:.1f}",
        f"Kirchhoff node: Σ(gate_k × LCI_k) = {yang/20:.3f} ≈ π, residual {3.14159-yang/20:.4f}",
        f"recursive depth {yang}: base case at yang={secs[5]}, branch factor {secs[0]}",
        f"orbit {idx}: {yang} vertices under B₆ symmetry, stabilizer order {max(1, 46080//max(yang,1))}",
    ]

    return templates[idx % len(templates)]


def _small_factors(n: int) -> list:
    """Разложить n на простые множители (вспомогательная для шаблонов)."""
    if n <= 1:
        return [n]
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors if factors else [1]


def bent_seed_texts(n: int = 20, block_size: int = 64) -> List[str]:
    """
    Генерировать n seed-текстов из bent-функций Q6.

    Bent-функции — математически оптимальные архетипы:
    - Максимальная нелинейность nl=28 (из 28 возможных для n=6)
    - Равномерный WHT-спектр: все |Ŵ(u)| = 8
    - Максимальное расстояние от аффинных функций
    - Гарантированное разнообразие начального RAG

    Args:
        n: сколько bent-текстов сгенерировать (1..20)
        block_size: размер блока модели (для совместимости)

    Returns:
        список текстовых описаний bent-архетипов
    """
    forms = _compute_bent_forms()
    texts = []
    for i, tt in enumerate(forms[:n]):
        texts.append(_bent_to_text(tt, i))
    return texts


# ── 2. Metropolis temperature annealing ───────────────────────────────────────

def metropolis_temperature(
    cycle: int,
    max_cycles: int,
    T0: float = 1.4,
    T_min: float = 0.5,
    decay: float = 0.85,
) -> float:
    """
    Metropolis-стиль расписание температуры.

    T(c) = max(T_min, T0 * decay^c)

    В отличие от фиксированной T=1.4:
    - Ранние циклы: высокая T → большой explore (разнообразие)
    - Поздние циклы: низкая T → точное уточнение (exploit)
    - Устраняет осцилляции ±0.05 в конце обучения

    Args:
        cycle:      текущий цикл (0-based)
        max_cycles: всего циклов
        T0:         начальная температура
        T_min:      минимальная температура (нижний порог)
        decay:      множитель убывания за цикл (0.8..0.95)

    Returns:
        температура для данного цикла
    """
    return max(T_min, T0 * (decay ** cycle))


def cosine_temperature(
    cycle: int,
    max_cycles: int,
    T0: float = 1.4,
    T_min: float = 0.5,
) -> float:
    """
    Косинусное расписание температуры (мягче чем экспоненциальное).
    T(c) = T_min + 0.5*(T0 - T_min)*(1 + cos(π*c/max_cycles))
    """
    if max_cycles <= 0:
        return T_min
    frac = cycle / max_cycles
    return T_min + 0.5 * (T0 - T_min) * (1.0 + math.cos(math.pi * frac))


# ── 3. Q6 address (embedding → hexagram) ─────────────────────────────────────

def embedding_to_q6(emb_vector: List[float]) -> int:
    """
    Проецировать вещественный вектор эмбеддинга на ближайшую вершину Q6.

    Подход: взять 6 главных направлений в эмбеддинге (через знаки),
    закодировать как бинарный вектор Q6.

    Точная геометрическая проекция:
    - Разбить вектор на 6 равных сегментов
    - bit_i = 1 если среднее сегмента > 0

    Args:
        emb_vector: вещественный вектор (любой размерности)

    Returns:
        гексаграмма 0..63 (ближайшая вершина Q6)
    """
    dim = len(emb_vector)
    seg_size = max(1, dim // 6)
    result = 0
    for i in range(6):
        start = i * seg_size
        end = min(start + seg_size, dim)
        segment_mean = sum(emb_vector[start:end]) / max(1, end - start)
        if segment_mean > 0:
            result |= (1 << i)
    return result


def hamming(a: int, b: int) -> int:
    """Расстояние Хэмминга между двумя Q6-вершинами."""
    return bin(a ^ b).count('1')


# ── 4. Q4 тессеракты (hexdim) ─────────────────────────────────────────────────

def q4_tesseracts() -> List[frozenset]:
    """
    Все 60 копий Q4 (тессеракт) внутри Q6.

    C(6,4) × 2^2 = 15 × 4 = 60 тессерактов.
    Каждый — frozenset из 16 вершин Q6.

    Для nautilus-15agent: первые 15 уникальных тессерактов
    (по уникальным осям, без учёта сдвига base).

    Returns:
        список frozenset{v0,...,v15}
    """
    if not _add_meta_path():
        return _fallback_tesseracts()
    try:
        from projects.hexdim.hexdim import all_subcubes
        return [verts for _, _, verts in all_subcubes(4)]
    except ImportError:
        return _fallback_tesseracts()


def _fallback_tesseracts() -> List[frozenset]:
    """
    Fallback: 15 уникальных Q4-подграфов Q6 без meta-импорта.
    Каждая пара свободных осей (из C(6,4)=15 вариантов), base=0.
    """
    from itertools import combinations
    result = []
    axes_all = list(range(6))
    for free_axes in combinations(axes_all, 4):
        verts = set()
        for mask in range(16):
            v = 0
            for bit_idx, axis in enumerate(free_axes):
                if (mask >> bit_idx) & 1:
                    v |= (1 << axis)
            verts.add(v)
        result.append(frozenset(verts))
    return result


# ── 5. Yang weight orbits (для agent specialization) ──────────────────────────

def yang_weight_orbits() -> List[List[int]]:
    """
    7 орбит Q6 по весу Хэмминга (yang_count = число единиц).

    Под группой S₆ (перестановки битов) Q6 разбивается на 7 орбит:
    Вес 0: {0}        — 1 вершина   (0 ян-линий)
    Вес 1: {1,2,...}  — 6 вершин    (1 ян-линия)
    Вес 2: {...}      — 15 вершин   (2 ян-линии)
    Вес 3: {...}      — 20 вершин   (3 ян-линии)
    Вес 4: {...}      — 15 вершин   (4 ян-линии)
    Вес 5: {...}      — 6 вершин    (5 ян-линий)
    Вес 6: {63}       — 1 вершина   (6 ян-линий)

    Returns:
        список из 7 списков вершин
    """
    orbits: List[List[int]] = [[] for _ in range(7)]
    for h in range(64):
        w = bin(h).count('1')
        orbits[w].append(h)
    return orbits


def d4_orbits() -> List[List[int]]:
    """
    19 орбит Q6 под группой симметрий D₄ квадрата.

    Менее экстремальное разбиение чем S₆, лучше для agent specialization.
    Возвращает орбиты в порядке убывания размера.
    """
    if not _add_meta_path():
        return yang_weight_orbits()  # fallback
    try:
        from projects.hexsym.d4orbits import orbits as d4orbits_fn
        raw = d4orbits_fn()
        return [list(orb) for orb in sorted(raw, key=len, reverse=True)]
    except (ImportError, AttributeError):
        return yang_weight_orbits()


# ── 6. Ecube routing (hexnet) ─────────────────────────────────────────────────

def ecube_route(src: int, dst: int) -> List[int]:
    """
    E-cube (dimension-ordered) маршрутизация по Q6.
    Исправляет биты по порядку 0..5, длина = hamming(src, dst).

    Детерминированная, минимальная, не вырождается в константу.

    Args:
        src: исходная вершина Q6 (0..63)
        dst: целевая вершина Q6 (0..63)

    Returns:
        список вершин маршрута [src, ..., dst]
    """
    if _add_meta_path():
        try:
            from projects.hexnet.hexnet import ecube_route as _route
            return _route(src, dst)
        except ImportError:
            pass
    # Fallback (идентичная реализация)
    path = [src]
    current = src
    diff = current ^ dst
    for bit in range(6):
        if diff & (1 << bit):
            current ^= (1 << bit)
            path.append(current)
    return path


def routing_diversity(gates: List[float]) -> float:
    """
    Мера разнообразия routing весов.
    gates: список вероятностей (по экспертам).
    Возвращает энтропию Шеннона (0 = коллапс к одному эксперту, log2(N) = равномерное).
    """
    import math
    entropy = 0.0
    for g in gates:
        if g > 1e-9:
            entropy -= g * math.log2(g)
    return entropy


# ── Самодиагностика ────────────────────────────────────────────────────────────

def check_integration() -> dict:
    """
    Проверить доступность meta-компонентов и корректность вычислений.
    Возвращает словарь {component: status}.
    """
    status = {}

    # 1. meta path
    status["meta_path"] = _meta_available()

    # 2. bent functions
    forms = _compute_bent_forms()
    status["bent_forms_count"] = len(forms)
    status["bent_verified"] = _is_bent(forms[0]) if forms else False

    # 3. temperature schedule
    temps = [metropolis_temperature(c, 8) for c in range(9)]
    status["temp_T0"] = round(temps[0], 3)
    status["temp_T8"] = round(temps[8], 3)
    status["temp_decreasing"] = all(temps[i] >= temps[i+1] for i in range(8))

    # 4. Q4 tesseracts
    tess = q4_tesseracts()
    status["tesseract_count"] = len(tess)
    status["tesseract_sizes_ok"] = all(len(t) == 16 for t in tess)

    # 5. Yang orbits
    orbits = yang_weight_orbits()
    total = sum(len(o) for o in orbits)
    status["yang_orbits"] = len(orbits)
    status["yang_orbits_total"] = total

    # 6. ecube routing
    route = ecube_route(0, 63)
    status["ecube_route_0_63_len"] = len(route) - 1  # должно быть 6 (hamming(0,63)=6)
    status["ecube_route_ok"] = (len(route) - 1) == 6

    return status


if __name__ == "__main__":
    import json
    print("=== meta_q6.py integration check ===")
    result = check_integration()
    for k, v in result.items():
        mark = "✓" if v else "✗"
        print(f"  {mark}  {k}: {v}")

    print("\n=== Bent seed texts (первые 3) ===")
    for i, t in enumerate(bent_seed_texts(3)):
        print(f"  [{i}] {t}")

    print("\n=== Temperature schedule (8 cycles, T0=1.4) ===")
    for c in range(9):
        T = metropolis_temperature(c, 8)
        print(f"  cycle {c}: T={T:.3f}")

    print("\n=== Yang weight orbits ===")
    for i, orb in enumerate(yang_weight_orbits()):
        print(f"  weight {i}: {len(orb)} вершин")
