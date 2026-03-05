"""
Простой char-level токенизатор для работы без sentencepiece.

Позволяет запускать обучение и инференс без внешних зависимостей.
"""


class CharTokenizer:
    """Character-level tokenizer с поддержкой ASCII + Unicode."""

    def __init__(self, chars=None):
        if chars is None:
            # ASCII printable + newline + tab + частые Unicode символы
            chars = (
                list(range(32, 127))  # ASCII printable
                + [10, 9]  # newline, tab
                + list(range(192, 256))  # Latin extended
            )
            chars = [chr(c) for c in chars]

        self.chars = sorted(set(chars))
        self.char_to_id = {c: i + 2 for i, c in enumerate(self.chars)}  # +2 for PAD, UNK
        self.id_to_char = {i + 2: c for i, c in enumerate(self.chars)}
        self.id_to_char[0] = ''   # PAD
        self.id_to_char[1] = '?'  # UNK

    def encode(self, text: str) -> list:
        return [self.char_to_id.get(c, 1) for c in text]

    def decode(self, ids: list) -> str:
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

    def get_piece_size(self) -> int:
        return len(self.chars) + 2  # +PAD +UNK

    def bos_id(self) -> int:
        return -1

    def eos_id(self) -> int:
        return -1

    @classmethod
    def from_text(cls, text: str):
        """Создаёт токенизатор из текста (все уникальные символы)."""
        chars = sorted(set(text))
        return cls(chars)

    @classmethod
    def from_file(cls, path: str):
        """Создаёт токенизатор из текстового файла."""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return cls.from_text(text)
