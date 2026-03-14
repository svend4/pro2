"""
Data2Adapter — адаптер для svend4/data2.
Формат: тома ЕТД (Единая Теория Движения) В.В. Крюкова.
310+ томов в 5+ сериях. Ключевой алгоритм: scarab_algorithm.py.
"""

from .base import BaseAdapter, PortalEntry


# Таблица ключевых томов: vol → (название, серия, домен, α-уровень, Q6-ключ)
VOLUME_TABLE = {
    1:   ("Архетипы движения",         "I",   "methodology",   -1, "000001"),
    2:   ("Геймдизайн ЕТД",            "I",   "design",        -1, "000010"),
    3:   ("Домашняя роботика",         "I",   "robotics",      -1, "000011"),
    4:   ("Алгоритмы",                 "I",   "algorithms",    -1, "000100"),
    5:   ("Медицина",                  "I",   "medicine",      -1, "000101"),
    6:   ("ИИ / МО",                   "I",   "ai_ml",         -1, "000110"),
    7:   ("Педагогика",                "I",   "pedagogy",      -1, "000111"),
    10:  ("Музыка",                    "I",   "music",         -1, "001010"),
    15:  ("Deep Learning",             "I",   "deep_learning", -1, "001111"),
    16:  ("Квантовые вычисления",      "I",   "quantum",        0, "010000"),
    19:  ("Социальные сети",           "I",   "social",        -1, "010011"),
    20:  ("Единая теория (синтез I)",  "I",   "synthesis",      0, "010100"),
    21:  ("Клиническая медицина",      "II",  "medicine",       0, "010101"),
    22:  ("Спортивная наука",          "II",  "sports",         0, "010110"),
    30:  ("Аэрокосмос",               "II",  "aerospace",      0, "011110"),
    36:  ("Сознание",                  "II",  "consciousness",  1, "100100"),
    40:  ("Конечный синтез II",        "II",  "synthesis",      1, "101000"),
    49:  ("Кватернионы",               "III", "quaternion",     1, "110001"),
    63:  ("Теория категорий",          "III", "math",           2, "111111"),
    110: ("Теория струн",              "IV",  "physics",        3, "110100"),
    150: ("Синтез IV",                 "IV",  "synthesis",      3, "111111"),
    310: ("Финальный синтез",          "V+",  "synthesis",      3, "111111"),
}

# Карта Архетип → CA-класс → Q6-паттерн
ARCHETYPE_CA = {
    "петля":       ("II",  "000011", "Цикл, периодический паттерн"),
    "три сферы":   ("II",  "000111", "Иерархия МВС/СВС/БВС"),
    "нечётность":  ("II",  "001001", "Закон нечётности {1,3,5,7,9}"),
    "7±2":         ("II",  "000110", "Мнемоническое ограничение Миллера"),
    "шахматка":    ("II",  "010010", "Модульная тактическая сетка"),
    "резонанс":    ("IV",  "110011", "ω_МВС = ω_СВС = ω_БВС → max"),
    "lci":         ("IV",  "101010", "Loop Closure Index — метрика системы"),
    "синтез":      ("IV",  "111110", "Интеграция всех архетипов"),
    "ката":        ("I",   "000000", "Зафиксированные паттерны движения"),
}

SERIES_ALPHA = {"I": -1, "II": 0, "III": 1, "IV": 2, "V+": 3}


class Data2Adapter(BaseAdapter):
    name = "data2"
    REPO = "svend4/data2"

    def fetch(self, query: str) -> list[PortalEntry]:
        q = query.lower()
        results = []

        # 1. Поиск по архетипам
        for arch_key, (ca_class, q6, desc) in ARCHETYPE_CA.items():
            if q in arch_key or q in desc.lower():
                results.append(self._archetype_entry(arch_key, ca_class, q6, desc))

        # 2. Поиск по томам
        for vol_id, (title, series, domain, alpha, q6) in VOLUME_TABLE.items():
            if q in title.lower() or q in domain.lower() or q in series.lower():
                results.append(self._volume_entry(vol_id, title, series, domain, alpha, q6))

        # Если ничего — базовые точки входа
        if not results:
            results = self._landmark_entries()

        return results[:5]

    def _archetype_entry(self, arch: str, ca_class: str, q6: str, desc: str) -> PortalEntry:
        alpha = {"I": -3, "II": -1, "III": +1, "IV": +3}.get(ca_class, 0)
        return PortalEntry(
            id=f"data2:arch:{arch.replace(' ', '_')}",
            title=f"Архетип: {arch.title()}",
            source=self.REPO,
            format_type="archetype",
            content=desc,
            metadata={"archetype": arch, "ca_class": ca_class, "q6": q6, "alpha": alpha},
            links=[f"meta:ca_class:{ca_class}", f"pro2:q6:{q6}", f"info1:alpha:{alpha}"],
        )

    def _volume_entry(self, vol_id, title, series, domain, alpha, q6) -> PortalEntry:
        return PortalEntry(
            id=f"data2:vol:{vol_id}",
            title=f"Том {vol_id}. {title}",
            source=self.REPO,
            format_type="volume",
            content=f"Серия {series} · домен: {domain}",
            metadata={"volume": vol_id, "series": series, "domain": domain,
                      "alpha": alpha, "q6": q6},
            links=[f"info1:alpha:{alpha}", f"pro2:q6:{q6}"],
        )

    def _landmark_entries(self) -> list[PortalEntry]:
        return [
            PortalEntry(
                id="data2:etd",
                title="ЕТД — Единая Теория Движения",
                source=self.REPO,
                format_type="volume",
                content=(
                    "12 архетипов движения × 7 аксиом Крюкова. "
                    "310+ томов в 5 сериях. LCI как метрика системы."
                ),
                metadata={"type": "overview", "volumes": "1-310", "series": "I–V+"},
                links=["info1:methodology", "meta:hexagram:50", "pro2:bidir"],
            ),
            PortalEntry(
                id="data2:scarab",
                title="Scarab Algorithm (фигура-8)",
                source=self.REPO,
                format_type="volume",
                content=(
                    "scarab_algorithm.py — обход Q6-пространства по траектории фигуры-8. "
                    "Прямой мост к семантическому пространству pro2."
                ),
                metadata={"type": "algorithm", "file": "scarab_algorithm.py"},
                links=["pro2:q6", "meta:hexagram:63"],
            ),
            PortalEntry(
                id="data2:three_spheres",
                title="Три Сферы (МВС / СВС / БВС)",
                source=self.REPO,
                format_type="archetype",
                content=(
                    "МВС (малая) / СВС (средняя) / БВС (большая) вложенные сферы. "
                    "Резонанс трёх сфер → максимальная эффективность системы."
                ),
                metadata={"archetype": "три сферы", "ca_class": "II", "alpha": -1},
                links=["meta:ca_class:II", "info1:alpha:-1"],
            ),
        ]

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": "data2",
            "native_unit": "Том (Markdown с номером и доменом)",
            "series": "I (1–20), II (21–40), III (41–80), IV (81–150), V+ (150–310)",
            "total_volumes": "310+",
            "total_files": 326,
            "archetypes_12": list(ARCHETYPE_CA.keys()),
            "key_algorithm": "scarab_algorithm.py",
            "lci": "Loop Closure Index — метрика замкнутости системы",
        }
