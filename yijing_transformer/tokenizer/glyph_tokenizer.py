"""
GlyphTokenizer — визуальный токенизатор SOLAN-76 (Ступень 7.1).

Каждый глиф SOLAN = вершина 6-мерного гиперкуба Q6 = {-1,+1}⁶.
6 визуальных сегментов: верх, низ, лево, право, диаг1, диаг2.

Поддерживает:
- encode: текст → последовательность вершин Q6
- decode: вершины Q6 → текст (nearest-neighbor в Хэмминг-метрике)
- contrastive: два шрифта (font3/font4) как позитивные пары
"""

import torch
from torch import Tensor


# 6-битная структура сегментов: (верх, низ, лево, право, диаг1, диаг2)
# Маппинг ASCII → 6-bit vertex для 76 символов SOLAN
# Первые 64 = прямое отображение на вершины Q6
# Остальные 12 = дублирующие с модификацией

# Базовые ASCII символы и их 6-битные коды
_SOLAN_MAP = {}

# Цифры 0-9: 10 символов
_DIGITS = {
    '0': (0, 0, 0, 0, 0, 0),  # пустой глиф = начало координат
    '1': (1, 0, 0, 0, 0, 0),  # только верх
    '2': (0, 1, 0, 0, 0, 0),  # только низ
    '3': (1, 1, 0, 0, 0, 0),  # верх+низ
    '4': (0, 0, 1, 0, 0, 0),  # только лево
    '5': (1, 0, 1, 0, 0, 0),  # верх+лево
    '6': (0, 1, 1, 0, 0, 0),  # низ+лево
    '7': (1, 1, 1, 0, 0, 0),  # верх+низ+лево
    '8': (0, 0, 0, 1, 0, 0),  # только право
    '9': (1, 0, 0, 1, 0, 0),  # верх+право
}

# Прописные A-Z: 26 символов (коды 10-35)
_UPPER = {}
for i, ch in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    code = i + 10  # 10..35
    bits = tuple((code >> (5 - b)) & 1 for b in range(6))
    _UPPER[ch] = bits

# Строчные a-z: 26 символов (коды 36-61)
_LOWER = {}
for i, ch in enumerate('abcdefghijklmnopqrstuvwxyz'):
    code = i + 36  # 36..61
    bits = tuple((code >> (5 - b)) & 1 for b in range(6))
    _LOWER[ch] = bits

# Спецсимволы (коды 62-63 + дублирующие)
_SPECIAL = {
    ' ': (1, 1, 1, 1, 1, 0),   # 62: пробел = почти полный
    '.': (1, 1, 1, 1, 1, 1),   # 63: точка = полный (☰)
    ',': (0, 1, 0, 0, 1, 0),   # = '2' variant
    '!': (1, 0, 0, 0, 0, 1),   # = '1' + диаг2
    '?': (0, 1, 0, 1, 0, 1),   # антипод '4'
    '-': (0, 0, 1, 1, 0, 0),   # лево+право = горизонтальная линия
    ':': (1, 1, 0, 0, 0, 0),   # = '3' variant
    ';': (1, 1, 0, 0, 0, 1),   # = '3' + диаг2
    "'": (1, 0, 0, 0, 1, 0),   # верх+диаг1
    '"': (1, 0, 0, 0, 1, 1),   # верх+обе диагонали
    '(': (0, 0, 1, 0, 1, 0),   # лево+диаг1
    ')': (0, 0, 0, 1, 0, 1),   # право+диаг2
    '/': (0, 0, 0, 0, 1, 0),   # диаг1
    '\\': (0, 0, 0, 0, 0, 1),  # диаг2
}

# Объединяем
_SOLAN_MAP.update(_DIGITS)
_SOLAN_MAP.update(_UPPER)
_SOLAN_MAP.update(_LOWER)
_SOLAN_MAP.update(_SPECIAL)


def _bits_to_vertex(bits: tuple) -> tuple:
    """(0,1,0,1,1,0) → (-1,+1,-1,+1,+1,-1)"""
    return tuple(2 * b - 1 for b in bits)


def _vertex_to_bits(vertex: tuple) -> tuple:
    """(-1,+1,-1,+1,+1,-1) → (0,1,0,1,1,0)"""
    return tuple((v + 1) // 2 for v in vertex)


class GlyphTokenizer:
    """Символ SOLAN → 6-битная вершина Q6 → {-1,+1}⁶.

    Пример:
        tok = GlyphTokenizer()
        vertices = tok.encode("Hello")  # (5, 6) tensor
        text = tok.decode(vertices)      # "Hello" (или ближайший)
    """

    def __init__(self):
        self.char_to_bits = dict(_SOLAN_MAP)
        self.bits_to_char = {}

        # Обратный маппинг (первый символ с данным кодом побеждает)
        for ch, bits in self.char_to_bits.items():
            if bits not in self.bits_to_char:
                self.bits_to_char[bits] = ch

        self.vocab_size = len(self.char_to_bits)

    def encode(self, text: str) -> Tensor:
        """Текст → последовательность вершин Q6.

        Args:
            text: входной текст

        Returns:
            Tensor shape (seq_len, 6), значения ∈ {-1, +1}
        """
        vertices = []
        for ch in text:
            bits = self.char_to_bits.get(ch)
            if bits is None:
                # Fallback: hash символа → 6 бит
                code = hash(ch) % 64
                bits = tuple((code >> (5 - b)) & 1 for b in range(6))
            vertices.append(_bits_to_vertex(bits))

        if not vertices:
            return torch.zeros(0, 6)
        return torch.tensor(vertices, dtype=torch.float32)

    def decode(self, vertices: Tensor) -> str:
        """Вершины Q6 → текст (nearest-neighbor в Хэмминг-метрике).

        Args:
            vertices: Tensor shape (seq_len, 6), значения ∈ R

        Returns:
            Декодированный текст
        """
        chars = []
        for v in vertices:
            # Квантизация: sign → {-1, +1}
            bits = tuple(int(x) for x in ((v.sign() + 1) / 2).long().tolist())
            ch = self.bits_to_char.get(bits, '?')
            chars.append(ch)
        return ''.join(chars)

    def hamming_distance(self, text1: str, text2: str) -> Tensor:
        """Расстояние Хэмминга между кодировками двух текстов (поэлементно)."""
        v1 = self.encode(text1)
        v2 = self.encode(text2)
        min_len = min(len(v1), len(v2))
        return ((v1[:min_len] != v2[:min_len]).float().sum(dim=-1))

    def contrastive_pairs(self, text: str, noise_bits: int = 1) -> tuple:
        """Генерация позитивных пар для контрастивного обучения.

        Имитирует различия font3/font4: случайный flip 1 бита.

        Returns:
            (anchor: Tensor, positive: Tensor) — оба shape (seq_len, 6)
        """
        anchor = self.encode(text)
        positive = anchor.clone()

        # Flip noise_bits случайных координат в каждом токене
        for i in range(len(positive)):
            flip_idx = torch.randperm(6)[:noise_bits]
            positive[i, flip_idx] *= -1

        return anchor, positive
