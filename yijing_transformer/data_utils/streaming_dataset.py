"""Стриминговый датасет TinyStories с буферизацией."""

import torch
import random
from collections import deque
from datasets import load_dataset


def get_batch_streaming(iterator, batch_size, block_size, device, tokenizer,
                        pad_token_id=1, buffer_size=200):
    """Получает батч из стримингового датасета с буфером для перемешивания."""
    x_batch, y_batch = [], []
    buffer = deque()

    while len(buffer) < buffer_size:
        try:
            ex = next(iterator)
            buffer.append(ex)
        except StopIteration:
            break

    while len(x_batch) < batch_size:
        if not buffer:
            return None, None
        ex = random.choice(buffer)
        tokens = tokenizer.encode(ex['text'])
        if len(tokens) <= 1:
            continue
        if len(tokens) > block_size + 1:
            start = random.randint(0, len(tokens) - block_size - 1)
            chunk = tokens[start:start + block_size + 1]
        else:
            pad_len = block_size + 1 - len(tokens)
            chunk = tokens + [pad_token_id] * pad_len
        x_batch.append(chunk[:-1])
        y_batch.append(chunk[1:])

    try:
        new_ex = next(iterator)
        buffer.append(new_ex)
        buffer.popleft()
    except StopIteration:
        pass

    X = torch.tensor(x_batch, dtype=torch.long, device=device)
    Y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return X, Y


def create_train_val_iterators():
    """Создаёт итераторы для TinyStories."""
    train_ds = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
    val_ds = load_dataset("roneneldan/TinyStories", streaming=True, split="validation")
    return iter(train_ds), iter(val_ds)
