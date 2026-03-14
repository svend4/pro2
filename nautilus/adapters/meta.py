"""
MetaAdapter — адаптер для svend4/meta.
Формат: 256 CA-правил (Вольфрам) → 64 гексаграммы И-Цзин → Q6-координаты.
"""

from .base import BaseAdapter, PortalEntry


# Полная таблица 64 гексаграмм: id → (символ, имя, описание, CA-класс)
HEXAGRAM_TABLE = {
    1:  ("乾", "Творчество",       "Чистая созидательная сила",              "I"),
    2:  ("坤", "Исполнение",       "Принимающая сила, следование",           "I"),
    3:  ("屯", "Начальная трудность","Росток пробивается сквозь землю",       "II"),
    4:  ("蒙", "Незрелость",       "Учение через опыт",                      "II"),
    5:  ("需", "Ожидание",         "Терпение перед действием",               "II"),
    6:  ("訟", "Конфликт",         "Противостояние требует осторожности",    "III"),
    7:  ("師", "Армия",            "Организованная сила под руководством",   "II"),
    8:  ("比", "Единение",         "Сближение, поиск союзников",             "II"),
    11: ("泰", "Мир",              "Гармония неба и земли",                  "II"),
    12: ("否", "Застой",           "Разделение, блокировка",                 "III"),
    14: ("大有", "Великое владение","Изобилие и ответственность",            "II"),
    23: ("剥", "Разрушение",       "Отслаивание, конец цикла",               "III"),
    24: ("復", "Возврат",          "Поворотная точка, начало нового цикла",  "IV"),
    29: ("坎", "Бездна",           "Опасность, которую нужно пройти",        "III"),
    30: ("離", "Огонь",            "Прилипание, ясность, зависимость",       "III"),
    42: ("益", "Прибавление",      "Рост через щедрость",                    "II"),
    49: ("革", "Революция",        "Преобразование в нужный момент",         "IV"),
    50: ("鼎", "Котёл",            "Трансформация через огонь",              "IV"),
    51: ("震", "Гром",             "Пробуждение через потрясение",           "III"),
    52: ("艮", "Гора",             "Неподвижность, медитация",               "I"),
    54: ("豐", "Изобилие",         "Полнота накопленной силы",               "II"),
    57: ("巽", "Ветер",            "Мягкое проникновение",                   "II"),
    58: ("兌", "Радость",          "Открытость, обмен",                      "II"),
    63: ("既濟", "После завершения","Равновесие достигнуто",                 "IV"),
    64: ("未濟", "До завершения",  "Движение к новому состоянию",            "IV"),
}

CA_CLASS_ALPHA = {"I": -3, "II": -1, "III": +1, "IV": +3}


class MetaAdapter(BaseAdapter):
    name = "meta"
    REPO = "svend4/meta"

    def fetch(self, query: str) -> list[PortalEntry]:
        q = query.lower()
        results = []
        for hex_id, (char, name, desc, ca_class) in HEXAGRAM_TABLE.items():
            if q in name.lower() or q in desc.lower() or q in char:
                results.append(self._make_entry(hex_id, char, name, desc, ca_class))
        # если ничего не нашли по запросу — вернуть несколько ключевых
        if not results:
            for hex_id in [50, 63, 24, 1]:
                char, name, desc, ca_class = HEXAGRAM_TABLE[hex_id]
                results.append(self._make_entry(hex_id, char, name, desc, ca_class))
        return results[:4]

    def _make_entry(self, hex_id, char, name, desc, ca_class) -> PortalEntry:
        ca_rule = (hex_id - 1) * 4  # примерное соответствие
        q6_bits = format(hex_id - 1, "06b")
        alpha = CA_CLASS_ALPHA.get(ca_class, 0)
        return PortalEntry(
            id=f"meta:hexagram:{hex_id}",
            title=f"[{hex_id}] {char} {name}",
            source=self.REPO,
            format_type="rule",
            content=desc,
            metadata={
                "hexagram_id": hex_id,
                "char": char,
                "ca_rule": ca_rule % 256,
                "ca_class": ca_class,
                "q6": q6_bits,
                "alpha_equiv": alpha,
            },
            links=[
                f"pro2:q6:{q6_bits}",
                f"info1:alpha:{alpha}",
            ],
        )

    def describe(self) -> dict:
        return {
            "repo": self.REPO,
            "format": "meta",
            "native_unit": "CA-правило (0..255) + гексаграмма",
            "symbolic_space": "256 правил (классы I–IV по Вольфраму)",
            "hexagram_mapping": f"64 гексаграммы ({len(HEXAGRAM_TABLE)} в таблице)",
            "q6_formula": "hex_id - 1 → bin(6)",
        }
