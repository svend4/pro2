"""
Токенизаторы для YiJing-Transformer.

Включает:
- CharTokenizer: char-level, все уникальные символы из текста
- ByteTokenizer: byte-level, 256 фиксированных токенов + спец.токены
- Оба работают без внешних зависимостей.
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

    def save(self, path: str):
        """Сохраняет словарь в файл."""
        with open(path, 'w', encoding='utf-8') as f:
            for c in self.chars:
                f.write(f"{ord(c)}\n")

    @classmethod
    def load(cls, path: str):
        """Загружает словарь из файла."""
        with open(path, 'r', encoding='utf-8') as f:
            chars = [chr(int(line.strip())) for line in f if line.strip()]
        return cls(chars)


class ByteTokenizer:
    """
    Byte-level tokenizer: кодирует текст как последовательность байтов.

    Фиксированный словарь 256 + 3 спец.токена = 259.
    Не требует подготовки словаря, работает с любым текстом.
    Использует UTF-8 байты.

    Специальные токены:
    - 0: PAD
    - 1: BOS (begin of sequence)
    - 2: EOS (end of sequence)
    - 3-258: байты 0x00-0xFF
    """
    SPECIAL_OFFSET = 3  # PAD=0, BOS=1, EOS=2

    def __init__(self):
        pass

    def encode(self, text: str) -> list:
        """Кодирует текст в байтовые токены."""
        return [b + self.SPECIAL_OFFSET for b in text.encode('utf-8')]

    def decode(self, ids: list) -> str:
        """Декодирует токены обратно в текст."""
        bytes_list = []
        for i in ids:
            if i >= self.SPECIAL_OFFSET:
                bytes_list.append(i - self.SPECIAL_OFFSET)
            # Пропускаем спец.токены
        try:
            return bytes(bytes_list).decode('utf-8', errors='replace')
        except Exception:
            return bytes(bytes_list).decode('latin-1')

    def get_piece_size(self) -> int:
        return 256 + self.SPECIAL_OFFSET  # 259

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2

    def encode_with_special(self, text: str) -> list:
        """Кодирует с BOS и EOS."""
        return [self.bos_id()] + self.encode(text) + [self.eos_id()]
