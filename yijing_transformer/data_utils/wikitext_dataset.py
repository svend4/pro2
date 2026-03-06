"""
WikiText-2 / WikiText-103 загрузчик для YiJing-Transformer.

Поддерживает:
1. Автоматическую загрузку через datasets (HuggingFace)
2. Ручную загрузку из текстовых файлов
3. BPE и char-level токенизацию
4. Эффективные батчи для LM задачи

Использование:
    dataset = WikiTextDataset.from_huggingface('wikitext-2-raw-v1', vocab_size=2048)
    train_ds, val_ds = dataset.split_train_val()
    x, y = train_ds.get_batch(batch_size=8, device='cpu')
"""

import os
import random
import torch


class WikiTextDataset:
    """Датасет WikiText для языкового моделирования."""

    def __init__(self, token_ids, block_size, tokenizer, vocab_size):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.n_tokens = len(self.data)

    @classmethod
    def from_huggingface(cls, name='wikitext-2-raw-v1', block_size=256,
                         vocab_size=2048, tokenizer_type='bpe',
                         cache_dir=None, verbose=True):
        """
        Загружает WikiText из HuggingFace datasets.

        Args:
            name: 'wikitext-2-raw-v1' или 'wikitext-103-raw-v1'
            block_size: размер контекстного окна
            vocab_size: размер словаря для BPE
            tokenizer_type: 'bpe', 'byte', 'char'
            cache_dir: директория кеша
            verbose: печатать прогресс
        """
        from datasets import load_dataset

        if verbose:
            print(f"  Loading {name}...")
        ds = load_dataset('wikitext', name, cache_dir=cache_dir)

        train_text = '\n'.join(
            line for line in ds['train']['text'] if line.strip()
        )
        val_text = '\n'.join(
            line for line in ds['validation']['text'] if line.strip()
        )
        test_text = '\n'.join(
            line for line in ds['test']['text'] if line.strip()
        )

        if verbose:
            print(f"  Train: {len(train_text):,} chars")
            print(f"  Val:   {len(val_text):,} chars")
            print(f"  Test:  {len(test_text):,} chars")

        # Токенизация
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tokenizer.char_tokenizer import BPETokenizer, ByteTokenizer, CharTokenizer

        if tokenizer_type == 'bpe':
            tokenizer = BPETokenizer()
            if verbose:
                print(f"  Training BPE (vocab_size={vocab_size})...")
            tokenizer.train(train_text, vocab_size=vocab_size, verbose=verbose)
            actual_vocab = tokenizer.get_piece_size()
        elif tokenizer_type == 'byte':
            tokenizer = ByteTokenizer()
            actual_vocab = tokenizer.get_piece_size()
        else:
            tokenizer = CharTokenizer.from_text(train_text)
            actual_vocab = tokenizer.get_piece_size()

        if verbose:
            print(f"  Vocab size: {actual_vocab}")
            print(f"  Tokenizing...")

        train_ids = tokenizer.encode(train_text)
        val_ids = tokenizer.encode(val_text)
        test_ids = tokenizer.encode(test_text)

        if verbose:
            print(f"  Train tokens: {len(train_ids):,}")
            print(f"  Val tokens:   {len(val_ids):,}")

        train_ds = cls(train_ids, block_size, tokenizer, actual_vocab)
        val_ds = cls(val_ids, block_size, tokenizer, actual_vocab)
        test_ds = cls(test_ids, block_size, tokenizer, actual_vocab)

        return train_ds, val_ds, test_ds

    @classmethod
    def from_text_file(cls, path, block_size=256, vocab_size=2048,
                       tokenizer_type='bpe', val_fraction=0.05, verbose=True):
        """Загружает из текстового файла."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tokenizer.char_tokenizer import BPETokenizer, ByteTokenizer, CharTokenizer

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        if verbose:
            print(f"  Loaded {len(text):,} chars from {path}")

        # Split
        split_idx = int(len(text) * (1 - val_fraction))
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        if tokenizer_type == 'bpe':
            tokenizer = BPETokenizer()
            tokenizer.train(train_text, vocab_size=vocab_size, verbose=verbose)
            actual_vocab = tokenizer.get_piece_size()
        elif tokenizer_type == 'byte':
            tokenizer = ByteTokenizer()
            actual_vocab = tokenizer.get_piece_size()
        else:
            tokenizer = CharTokenizer.from_text(train_text)
            actual_vocab = tokenizer.get_piece_size()

        train_ids = tokenizer.encode(train_text)
        val_ids = tokenizer.encode(val_text)

        return (
            cls(train_ids, block_size, tokenizer, actual_vocab),
            cls(val_ids, block_size, tokenizer, actual_vocab),
        )

    def get_batch(self, batch_size, device='cpu'):
        """Возвращает случайный батч (x, y)."""
        n = self.n_tokens - self.block_size - 1
        if n <= 0:
            raise ValueError(f"Dataset too small: {self.n_tokens} tokens for block_size={self.block_size}")
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix]).to(device)
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix]).to(device)
        return x, y

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return max(0, self.n_tokens - self.block_size - 1)

    def __repr__(self):
        return (f"WikiTextDataset(tokens={self.n_tokens:,}, "
                f"block_size={self.block_size}, vocab={self.vocab_size})")
