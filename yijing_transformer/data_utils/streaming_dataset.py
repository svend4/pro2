"""Стриминговый датасет TinyStories с буферизацией."""

import logging
import torch
import random
from collections import deque

_log = logging.getLogger(__name__)


class StreamingBatchLoader:
    """Стриминговый загрузчик с персистентным буфером.

    Буфер сохраняется между вызовами get_batch(), обеспечивая:
      - Перемешивание (random.choice из буфера)
      - Постепенную ротацию (новые примеры заменяют старые)
      - Автоматический restart при исчерпании итератора
    """

    def __init__(self, iterator, tokenizer, block_size, device,
                 pad_token_id=1, buffer_size=200, restart_fn=None):
        """
        Args:
            iterator:    итератор по датасету (next(iterator) → dict с 'text')
            tokenizer:   объект с .encode(text) → list[int]
            block_size:  размер блока (seq_len)
            device:      torch device
            pad_token_id: id для паддинга коротких последовательностей
            buffer_size: размер буфера для перемешивания
            restart_fn:  callable() → новый итератор (для перезапуска эпохи)
        """
        self._iterator = iterator
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._device = device
        self._pad_token_id = pad_token_id
        self._buffer_size = buffer_size
        self._restart_fn = restart_fn
        self._buffer: deque = deque()
        self._exhausted = False
        self._fill_buffer()

    def _fill_buffer(self):
        """Заполняет буфер до buffer_size из итератора."""
        while len(self._buffer) < self._buffer_size:
            try:
                ex = next(self._iterator)
                self._buffer.append(ex)
            except StopIteration:
                self._exhausted = True
                if not self._buffer:
                    _log.warning("Streaming iterator exhausted with empty buffer")
                break

    def _rotate_one(self):
        """Заменяет один старый пример из буфера новым."""
        if self._exhausted:
            return
        try:
            new_ex = next(self._iterator)
            self._buffer.append(new_ex)
            if len(self._buffer) > self._buffer_size:
                self._buffer.popleft()
        except StopIteration:
            self._exhausted = True

    def _maybe_restart(self):
        """Перезапускает итератор если он исчерпан и есть restart_fn."""
        if self._exhausted and self._restart_fn is not None:
            _log.info("Restarting streaming iterator (new epoch)")
            self._iterator = self._restart_fn()
            self._exhausted = False
            self._fill_buffer()

    def get_batch(self, batch_size):
        """Возвращает (X, Y) тензоры или (None, None) если данных нет.

        Returns:
            X: (batch_size, block_size) LongTensor
            Y: (batch_size, block_size) LongTensor
        """
        if not self._buffer:
            self._maybe_restart()
            if not self._buffer:
                return None, None

        x_batch, y_batch = [], []
        attempts = 0
        max_attempts = batch_size * 10  # guard against infinite loop

        while len(x_batch) < batch_size and attempts < max_attempts:
            attempts += 1
            if not self._buffer:
                break
            ex = random.choice(self._buffer)
            tokens = self._tokenizer.encode(ex['text'])
            if len(tokens) <= 1:
                continue
            if len(tokens) > self._block_size + 1:
                start = random.randint(0, len(tokens) - self._block_size - 1)
                chunk = tokens[start:start + self._block_size + 1]
            else:
                pad_len = self._block_size + 1 - len(tokens)
                chunk = tokens + [self._pad_token_id] * pad_len
            x_batch.append(chunk[:-1])
            y_batch.append(chunk[1:])

        if not x_batch:
            return None, None

        self._rotate_one()

        X = torch.tensor(x_batch, dtype=torch.long, device=self._device)
        Y = torch.tensor(y_batch, dtype=torch.long, device=self._device)
        return X, Y

    @property
    def is_exhausted(self):
        return self._exhausted and len(self._buffer) == 0


# ── Обратная совместимость ────────────────────────────────────────────────────

def get_batch_streaming(iterator, batch_size, block_size, device, tokenizer,
                        pad_token_id=1, buffer_size=200):
    """Legacy wrapper — создаёт временный loader на один вызов.

    Предпочтительно использовать StreamingBatchLoader напрямую.
    """
    loader = StreamingBatchLoader(
        iterator=iterator, tokenizer=tokenizer, block_size=block_size,
        device=device, pad_token_id=pad_token_id, buffer_size=buffer_size,
    )
    return loader.get_batch(batch_size)


def create_train_val_iterators():
    """Создаёт итераторы для TinyStories."""
    from datasets import load_dataset
    train_ds = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
    val_ds = load_dataset("roneneldan/TinyStories", streaming=True, split="validation")
    return iter(train_ds), iter(val_ds)
