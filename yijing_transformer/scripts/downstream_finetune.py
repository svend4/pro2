#!/usr/bin/env python3
"""
C10: Downstream Task Fine-tuning Pipeline.

Проверяет, сохраняется ли преимущество геометрических моделей
на downstream задачах:
1. Text Classification (sentiment-like)
2. Sequence Copying (algorithmic)
3. Pattern Completion (n-gram prediction)

Сравнивает: Vanilla vs Hybrid vs Adaptive — pretrain → finetune → evaluate.
"""

import os
import sys
import math
import time
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import HybridGatedGPT, AdaptiveHybridGPT
from models.baseline import VanillaGPT

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 64
BLOCK_SIZE = 128
BATCH_SIZE = 8


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def get_lr(step, total_steps, base_lr):
    warmup = int(total_steps * 0.1)
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ========================== DOWNSTREAM TASKS ==========================

class CopyTask:
    """Задача копирования: вход=[A B C | ...] → выход=[A B C]."""
    def __init__(self, vocab_size=VOCAB_SIZE, seq_len=16):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.sep_token = 0

    def get_batch(self, batch_size, block_size, device):
        seqs_x, seqs_y = [], []
        for _ in range(batch_size):
            # Generate pattern to copy
            pattern = [random.randint(1, self.vocab_size - 1) for _ in range(self.seq_len)]
            # Input: pattern + separator + pattern (shifted for LM)
            full = pattern + [self.sep_token] + pattern
            # Pad/truncate to block_size
            if len(full) > block_size + 1:
                full = full[:block_size + 1]
            else:
                full = full + [self.sep_token] * (block_size + 1 - len(full))
            seqs_x.append(full[:-1])
            seqs_y.append(full[1:])

        return (torch.tensor(seqs_x, dtype=torch.long, device=device),
                torch.tensor(seqs_y, dtype=torch.long, device=device))

    @torch.no_grad()
    def evaluate_accuracy(self, model, device, n_eval=100):
        """Точность копирования паттерна."""
        model.eval()
        correct = 0
        total = 0
        for _ in range(n_eval):
            pattern = [random.randint(1, self.vocab_size - 1) for _ in range(self.seq_len)]
            prompt = pattern + [self.sep_token]
            x = torch.tensor([prompt], dtype=torch.long, device=device)

            # Generate continuation
            for _ in range(self.seq_len):
                logits = model(x)[0]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, next_token], dim=1)

            generated = x[0, len(prompt):len(prompt) + self.seq_len].tolist()
            for g, p in zip(generated, pattern):
                if g == p:
                    correct += 1
                total += 1

        model.train()
        return correct / max(total, 1)


class PatternCompletionTask:
    """Задача дополнения паттерна: A B A B A → B."""
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size

    def get_batch(self, batch_size, block_size, device):
        seqs_x, seqs_y = [], []
        for _ in range(batch_size):
            # Random pattern of length 2-5
            pat_len = random.randint(2, 5)
            pattern = [random.randint(1, self.vocab_size - 1) for _ in range(pat_len)]
            # Repeat to fill block_size
            n_repeats = (block_size + 1) // pat_len + 1
            full = (pattern * n_repeats)[:block_size + 1]
            seqs_x.append(full[:-1])
            seqs_y.append(full[1:])

        return (torch.tensor(seqs_x, dtype=torch.long, device=device),
                torch.tensor(seqs_y, dtype=torch.long, device=device))

    @torch.no_grad()
    def evaluate_accuracy(self, model, device, n_eval=100):
        model.eval()
        correct = 0
        total = 0
        for _ in range(n_eval):
            pat_len = random.randint(2, 5)
            pattern = [random.randint(1, self.vocab_size - 1) for _ in range(pat_len)]
            # Give 3 full repeats, predict 4th
            prompt = pattern * 3
            x = torch.tensor([prompt], dtype=torch.long, device=device)

            for i in range(pat_len):
                logits = model(x)[0]
                next_token = logits[:, -1, :].argmax(dim=-1).item()
                if next_token == pattern[i]:
                    correct += 1
                total += 1
                x = torch.cat([x, torch.tensor([[pattern[i]]], device=device)], dim=1)

        model.train()
        return correct / max(total, 1)


class ClassificationTask:
    """Простая задача классификации: длинные слова → класс 1, короткие → класс 0."""
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.n_classes = 2

    def get_batch(self, batch_size, block_size, device):
        seqs_x, seqs_y = [], []
        for _ in range(batch_size):
            if random.random() < 0.5:
                # Class 0: short words (2-3 tokens) separated by spaces
                words = []
                while len(words) < block_size:
                    w_len = random.randint(2, 3)
                    words.extend([random.randint(1, self.vocab_size - 1) for _ in range(w_len)])
                    words.append(0)
                label = 0
            else:
                # Class 1: long words (5-8 tokens) separated by spaces
                words = []
                while len(words) < block_size:
                    w_len = random.randint(5, 8)
                    words.extend([random.randint(1, self.vocab_size - 1) for _ in range(w_len)])
                    words.append(0)
                label = 1

            words = words[:block_size]
            # Target: all positions predict the class
            targets = [label] * block_size
            seqs_x.append(words)
            seqs_y.append(targets)

        return (torch.tensor(seqs_x, dtype=torch.long, device=device),
                torch.tensor(seqs_y, dtype=torch.long, device=device))


def pretrain_model(model, device, steps=300):
    """Pretrain на LM задаче."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()
    for step in range(1, steps + 1):
        lr = get_lr(step, steps, 1e-3)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        # Random LM data
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE), device=device)
        y = torch.cat([x[:, 1:], torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 1), device=device)], dim=1)
        result = model(x, y)
        loss = result[1] if len(result) >= 2 else result[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def finetune_on_task(model, task, device, steps=200):
    """Fine-tune на downstream задаче."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    losses = []
    for step in range(1, steps + 1):
        lr = get_lr(step, steps, 3e-4)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        xb, yb = task.get_batch(BATCH_SIZE, BLOCK_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            losses.append(loss.item())
    return losses


def run_experiment():
    device = torch.device(DEVICE)

    print("=" * 70)
    print("  C10: DOWNSTREAM TASK FINE-TUNING")
    print("=" * 70)

    tasks = {
        'copy': CopyTask(VOCAB_SIZE, seq_len=8),
        'pattern': PatternCompletionTask(VOCAB_SIZE),
    }

    model_configs = {
        'vanilla': {
            'class': VanillaGPT,
            'cfg_extra': {},
        },
        'hybrid': {
            'class': HybridGatedGPT,
            'cfg_extra': {
                'architecture_mode': 'hybrid', 'use_bian_gua': True,
                'adaptive_temp': True, 'hex_strength': 0.01,
                'curriculum_strategy_geo': 'linear',
            },
        },
        'adaptive': {
            'class': AdaptiveHybridGPT,
            'cfg_extra': {
                'architecture_mode': 'hybrid', 'use_bian_gua': True,
                'adaptive_temp': True, 'hex_strength': 0.01,
            },
        },
    }

    results = {}

    for model_name, model_info in model_configs.items():
        print(f"\n{'▓' * 70}")
        print(f"  Model: {model_name}")
        results[model_name] = {}

        cfg = YiJingConfig(
            vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4,
            block_size=BLOCK_SIZE, total_steps=500,
            use_rope=True, use_swiglu=True,
            **model_info['cfg_extra'],
        )

        for task_name, task in tasks.items():
            set_seed(SEED)
            model = model_info['class'](cfg).to(device)
            params, geo_params = model.count_parameters()

            # Pretrain
            pretrain_model(model, device, steps=300)

            # Finetune
            ft_losses = finetune_on_task(model, task, device, steps=200)

            # Evaluate
            if hasattr(task, 'evaluate_accuracy'):
                accuracy = task.evaluate_accuracy(model, device)
            else:
                accuracy = None

            results[model_name][task_name] = {
                'params': params,
                'geo_params': geo_params,
                'final_ft_loss': ft_losses[-1] if ft_losses else None,
                'accuracy': accuracy,
            }

            acc_str = f"acc={accuracy:.3f}" if accuracy else ""
            print(f"    {task_name}: loss={ft_losses[-1]:.4f} {acc_str}")
            del model

    # Summary
    print(f"\n\n{'=' * 70}")
    print("  DOWNSTREAM RESULTS")
    print(f"{'=' * 70}")

    for task_name in tasks:
        print(f"\n  Task: {task_name}")
        print(f"  {'Model':<15} {'Params':>10} {'Loss':>10} {'Accuracy':>10}")
        print(f"  {'-' * 47}")
        for model_name in model_configs:
            r = results[model_name][task_name]
            acc = f"{r['accuracy']:.3f}" if r['accuracy'] else "N/A"
            loss = f"{r['final_ft_loss']:.4f}" if r['final_ft_loss'] else "N/A"
            print(f"  {model_name:<15} {r['params']:>10,} {loss:>10} {acc:>10}")

    # Save
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'downstream_results.json'
    )
    output_path = os.path.abspath(output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results: {output_path}")


if __name__ == '__main__':
    run_experiment()
