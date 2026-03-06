"""
Универсальный текстовый датасет для YiJing-Transformer.

Поддерживает:
1. Текстовые файлы (txt, md)
2. Директории с текстами
3. Предзагруженные строки
4. Memory-mapped файлы для больших корпусов

Работает с CharTokenizer (без внешних зависимостей).

Использование:
    dataset = TextDataset.from_file("corpus.txt", block_size=128)
    dataset = TextDataset.from_directory("texts/", block_size=128)
    x, y = dataset.get_batch(batch_size=8, device='cpu')
"""

import os
import random
import torch


class TextDataset:
    """Текстовый датасет с char-level токенизацией."""

    def __init__(self, token_ids, block_size, tokenizer=None):
        """
        Args:
            token_ids: list[int] — токенизированный текст
            block_size: размер контекстного окна
            tokenizer: опциональный токенизатор для decode
        """
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.n_tokens = len(self.data)

    @classmethod
    def from_text(cls, text, block_size, tokenizer=None):
        """Создаёт датасет из строки."""
        from tokenizer.char_tokenizer import CharTokenizer
        if tokenizer is None:
            tokenizer = CharTokenizer.from_text(text)
        token_ids = tokenizer.encode(text)
        return cls(token_ids, block_size, tokenizer)

    @classmethod
    def from_file(cls, path, block_size, tokenizer=None):
        """Создаёт датасет из текстового файла."""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return cls.from_text(text, block_size, tokenizer)

    @classmethod
    def from_directory(cls, directory, block_size, pattern='*.txt', tokenizer=None):
        """Создаёт датасет из директории с текстами."""
        import glob
        files = sorted(glob.glob(os.path.join(directory, pattern)))
        if not files:
            # Пробуем рекурсивный поиск
            files = sorted(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
        if not files:
            raise FileNotFoundError(f"No {pattern} files in {directory}")

        texts = []
        for f in files:
            with open(f, 'r', encoding='utf-8') as fp:
                texts.append(fp.read())

        combined = '\n\n'.join(texts)
        return cls.from_text(combined, block_size, tokenizer)

    def get_batch(self, batch_size, device='cpu'):
        """Возвращает случайный батч (x, y) для обучения."""
        n = self.n_tokens - self.block_size - 1
        if n <= 0:
            raise ValueError(
                f"Dataset too small ({self.n_tokens} tokens) "
                f"for block_size={self.block_size}"
            )
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
        return x, y

    def get_vocab_size(self):
        """Возвращает размер словаря."""
        if self.tokenizer is not None:
            return self.tokenizer.get_piece_size()
        return int(self.data.max().item()) + 1

    def split(self, val_fraction=0.1):
        """Разделяет датасет на train и val."""
        n_val = max(1, int(self.n_tokens * val_fraction))
        n_train = self.n_tokens - n_val

        train_ds = TextDataset(
            self.data[:n_train].tolist(),
            self.block_size,
            self.tokenizer,
        )
        val_ds = TextDataset(
            self.data[n_train:].tolist(),
            self.block_size,
            self.tokenizer,
        )
        return train_ds, val_ds

    def __len__(self):
        return max(0, self.n_tokens - self.block_size - 1)

    def __repr__(self):
        return (
            f"TextDataset(tokens={self.n_tokens:,}, "
            f"block_size={self.block_size}, "
            f"vocab_size={self.get_vocab_size()})"
        )


class ShuffledBatchIterator:
    """
    Итератор батчей с предварительной нарезкой и перемешиванием.

    Для детерминистичного обучения: нарезает текст на куски,
    перемешивает, выдаёт батчами.
    """
    def __init__(self, dataset, batch_size, device='cpu', seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.rng = random.Random(seed)

        # Нарезаем на непересекающиеся куски
        n_chunks = len(dataset) // dataset.block_size
        self.indices = list(range(n_chunks))
        self.rng.shuffle(self.indices)
        self.pos = 0

    def get_batch(self):
        """Возвращает следующий батч."""
        if self.pos + self.batch_size > len(self.indices):
            self.rng.shuffle(self.indices)
            self.pos = 0

        batch_indices = self.indices[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size

        bs = self.dataset.block_size
        x_list, y_list = [], []
        for idx in batch_indices:
            start = idx * bs
            x_list.append(self.dataset.data[start:start + bs])
            y_list.append(self.dataset.data[start + 1:start + bs + 1])

        x = torch.stack(x_list).to(self.device)
        y = torch.stack(y_list).to(self.device)
        return x, y

    @property
    def n_batches(self):
        return len(self.indices) // self.batch_size
