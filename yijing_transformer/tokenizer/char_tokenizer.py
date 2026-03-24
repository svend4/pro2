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
        """BOS не поддерживается в CharTokenizer."""
        return -1

    def eos_id(self) -> int:
        """EOS не поддерживается в CharTokenizer."""
        return -1

    def encode_with_special(self, text: str) -> list:
        """Кодирует текст (без BOS/EOS — CharTokenizer их не поддерживает)."""
        return self.encode(text)

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


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.

    Обучается на корпусе текста, последовательно сливая наиболее
    частые пары байтов/токенов. Работает без внешних зависимостей.

    Использование:
        bpe = BPETokenizer()
        bpe.train("your text corpus here", vocab_size=1000)
        ids = bpe.encode("hello world")
        text = bpe.decode(ids)
    """

    # Спец-токены
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    SPECIAL_OFFSET = 4

    def __init__(self):
        self.merges = []  # [(pair_a, pair_b), ...] в порядке обучения
        self.vocab = {}   # token_id → bytes
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Инициализирует базовый vocab: 256 байтов + спец-токены."""
        self.vocab = {
            0: b'<PAD>', 1: b'<BOS>', 2: b'<EOS>', 3: b'<UNK>',
        }
        for i in range(256):
            self.vocab[i + self.SPECIAL_OFFSET] = bytes([i])
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, text, vocab_size=1000, verbose=False):
        """
        Обучает BPE на тексте.

        Args:
            text: обучающий текст
            vocab_size: целевой размер словаря
            verbose: печатать прогресс
        """
        assert vocab_size > 256 + self.SPECIAL_OFFSET, \
            f"vocab_size must be > {256 + self.SPECIAL_OFFSET}"

        # Начинаем с байтов
        tokens = [b + self.SPECIAL_OFFSET for b in text.encode('utf-8')]

        n_merges = vocab_size - (256 + self.SPECIAL_OFFSET)
        self.merges = []

        for i in range(n_merges):
            if len(tokens) < 2:
                break

            # Считаем пары
            pair_counts = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Самая частая пара
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # Нет пар с частотой >= 2

            new_id = 256 + self.SPECIAL_OFFSET + len(self.merges)
            self.merges.append(best_pair)

            # Новый токен = конкатенация байтов
            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = new_bytes
            self.inverse_vocab[new_bytes] = new_id

            # Применяем merge
            tokens = self._apply_merge(tokens, best_pair, new_id)

            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1}/{n_merges}: {best_pair} → {new_id} "
                      f"(freq={pair_counts[best_pair]}, tokens={len(tokens)})")

    @staticmethod
    def _apply_merge(tokens, pair, new_id):
        """Заменяет все вхождения pair на new_id."""
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def encode(self, text):
        """Кодирует текст в BPE токены."""
        tokens = [b + self.SPECIAL_OFFSET for b in text.encode('utf-8')]
        for i, pair in enumerate(self.merges):
            new_id = 256 + self.SPECIAL_OFFSET + i
            tokens = self._apply_merge(tokens, pair, new_id)
        return tokens

    def decode(self, ids):
        """Декодирует BPE токены в текст."""
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab and token_id >= self.SPECIAL_OFFSET:
                byte_chunks.append(self.vocab[token_id])
        try:
            return b''.join(byte_chunks).decode('utf-8', errors='replace')
        except Exception:
            return b''.join(byte_chunks).decode('latin-1')

    def encode_with_special(self, text):
        """Кодирует с BOS и EOS."""
        return [self.BOS_ID] + self.encode(text) + [self.EOS_ID]

    def get_piece_size(self):
        """Размер словаря."""
        return len(self.vocab)

    def save(self, path):
        """Сохраняет BPE модель."""
        import json
        data = {
            'merges': self.merges,
            'vocab_size': len(self.vocab),
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        """Загружает BPE модель."""
        import json
        with open(path) as f:
            data = json.load(f)
        bpe = cls()
        # Replay merges
        for pair in data['merges']:
            pair = tuple(pair)
            new_id = 256 + cls.SPECIAL_OFFSET + len(bpe.merges)
            new_bytes = bpe.vocab[pair[0]] + bpe.vocab[pair[1]]
            bpe.vocab[new_id] = new_bytes
            bpe.inverse_vocab[new_bytes] = new_id
            bpe.merges.append(pair)
        return bpe
