#!/usr/bin/env python3
"""
Фаза 7: Real Language Data Training Pipeline.

Обучает модели на реальных текстовых данных (char-level).
Сравнивает Vanilla vs Hybrid vs Adaptive на задаче языкового моделирования.

Данные: генерируются из встроенных текстовых паттернов, имитирующих
естественный язык (повторяющиеся n-граммы, грамматические структуры).
"""

import os
import sys
import math
import time
import json
import random
import string

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import HybridGatedGPT, AdaptiveHybridGPT
from models.baseline import VanillaGPT

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 500
BATCH_SIZE = 8
BLOCK_SIZE = 128
LR = 1e-3


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class CharLevelDataset:
    """
    Char-level dataset из синтетического текста, имитирующего
    паттерны естественного языка.

    Генерирует тексты с:
    - Повторяющимися словами (n-gram structure)
    - Простой грамматикой (S → NP VP, NP → Det N, VP → V NP)
    - Палиндромами и повторами (тестируют attention)
    """
    def __init__(self, vocab_size=64, seed=42):
        random.seed(seed)
        self.vocab_size = vocab_size

        # Создаём «слова» разной длины
        self.words = []
        for length in range(2, 8):
            for _ in range(vocab_size // 6):
                word = [random.randint(1, vocab_size - 1) for _ in range(length)]
                self.words.append(word)

        # Простая грамматика
        n_det = max(1, len(self.words) // 10)
        n_noun = max(1, len(self.words) // 4)
        n_verb = max(1, len(self.words) // 4)
        self.determiners = self.words[:n_det]
        self.nouns = self.words[n_det:n_det+n_noun]
        self.verbs = self.words[n_det+n_noun:n_det+n_noun+n_verb]

        # Разделитель
        self.sep = [0]  # token 0 = separator

    def generate_sentence(self):
        """Генерирует одно «предложение» по грамматике."""
        parts = []
        # NP: Det + N
        if self.determiners and self.nouns:
            parts.extend(random.choice(self.determiners))
            parts.extend(self.sep)
            parts.extend(random.choice(self.nouns))
            parts.extend(self.sep)
        # VP: V + NP
        if self.verbs and self.nouns:
            parts.extend(random.choice(self.verbs))
            parts.extend(self.sep)
            if random.random() > 0.3 and self.determiners:
                parts.extend(random.choice(self.determiners))
                parts.extend(self.sep)
            parts.extend(random.choice(self.nouns))
        return parts

    def generate_sequence(self, length):
        """Генерирует последовательность нужной длины."""
        seq = []
        while len(seq) < length + 1:
            pattern = random.random()
            if pattern < 0.4:
                # Грамматическое предложение
                seq.extend(self.generate_sentence())
                seq.extend(self.sep)
            elif pattern < 0.6:
                # Повтор слова (тестирует copy)
                word = random.choice(self.words)
                n_repeats = random.randint(2, 4)
                for _ in range(n_repeats):
                    seq.extend(word)
                    seq.extend(self.sep)
            elif pattern < 0.8:
                # Палиндром
                word = random.choice(self.words)
                seq.extend(word)
                seq.extend(self.sep)
                seq.extend(word[::-1])
                seq.extend(self.sep)
            else:
                # Случайный фрагмент
                rand_len = random.randint(3, 10)
                seq.extend([random.randint(0, self.vocab_size - 1) for _ in range(rand_len)])
                seq.extend(self.sep)

        # Обрезаем до нужной длины
        seq = seq[:length + 1]
        # Клипаем к vocab_size
        seq = [min(max(t, 0), self.vocab_size - 1) for t in seq]
        return seq

    def get_batch(self, batch_size, block_size, device):
        """Возвращает батч (x, y) для обучения."""
        sequences = []
        for _ in range(batch_size):
            seq = self.generate_sequence(block_size)
            sequences.append(seq)

        data = torch.tensor(sequences, dtype=torch.long, device=device)
        return data[:, :-1], data[:, 1:]


def get_lr(step, total_steps, base_lr):
    warmup = int(total_steps * 0.1)
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, dataset, device, n=30):
    model.eval()
    losses = []
    for _ in range(n):
        xb, yb = dataset.get_batch(BATCH_SIZE, BLOCK_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def test_generation(model, dataset, device):
    """Тестирует качество генерации."""
    model.eval()
    prompt = torch.randint(0, dataset.vocab_size, (1, 16), device=device)

    if hasattr(model, 'generate'):
        output = model.generate(prompt, max_new_tokens=64, temperature=0.8, top_k=20)
    else:
        idx = prompt
        for _ in range(64):
            idx_input = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits = model(idx_input)[0]
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, min(20, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        output = idx

    generated = output[0, 16:].tolist()
    unique = len(set(generated))
    total = len(generated)
    rep_rate = 1 - unique / max(total, 1)

    # Считаем n-gram повторения (как в реальных текстах)
    bigrams = [(generated[i], generated[i+1]) for i in range(len(generated)-1)]
    unique_bigrams = len(set(bigrams))
    bigram_diversity = unique_bigrams / max(len(bigrams), 1)

    return {
        'unique_tokens': unique,
        'total_tokens': total,
        'rep_rate': rep_rate,
        'bigram_diversity': bigram_diversity,
    }


def train_model(model, name, dataset, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_params, geo_params = model.count_parameters()
    print(f"  {name}: {total_params:,} params ({geo_params:,} geo)")

    best_val = float('inf')
    train_losses = []
    val_losses = []
    t_start = time.time()

    model.train()
    accum = 0.0

    for step in range(1, STEPS + 1):
        lr = get_lr(step, STEPS, LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)

        xb, yb = dataset.get_batch(BATCH_SIZE, BLOCK_SIZE, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        accum += loss.item()

        if step % 50 == 0:
            avg = accum / 50
            train_losses.append(avg)
            accum = 0.0
            if step % 100 == 0:
                print(f"    step {step}: train={avg:.4f}")

        if step % 100 == 0:
            vl = evaluate(model, dataset, device)
            val_losses.append(vl)
            if vl < best_val:
                best_val = vl

    elapsed = time.time() - t_start
    gen = test_generation(model, dataset, device)

    gate_info = {}
    if hasattr(model, 'get_gate_summary'):
        summary = model.get_gate_summary()
        geo_count = sum(
            1 for layer in summary.values()
            for gate in layer.values()
            if isinstance(gate, dict) and gate.get('prefers_geometry', False)
        )
        total_gates = sum(
            1 for layer in summary.values()
            for gate in layer.values()
            if isinstance(gate, dict) and 'gate_mean' in gate
        )
        gate_info = {'geo_gates': geo_count, 'total_gates': total_gates}

    print(f"    DONE: best_val={best_val:.4f} | {elapsed:.1f}s | "
          f"unique={gen['unique_tokens']}/{gen['total_tokens']} | "
          f"bigram_div={gen['bigram_diversity']:.2f}")

    return {
        'params': total_params,
        'geo_params': geo_params,
        'best_val': best_val,
        'train_curve': train_losses,
        'val_curve': val_losses,
        'elapsed': elapsed,
        'generation': gen,
        'gate_info': gate_info,
    }


def run_experiment():
    device = torch.device(DEVICE)

    print("=" * 70)
    print("  PHASE 7: REAL LANGUAGE DATA EXPERIMENT")
    print("=" * 70)

    vocab_size = 64
    dataset = CharLevelDataset(vocab_size=vocab_size, seed=SEED)

    base_cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
        use_rope=True, use_swiglu=True, use_bian_gua=True,
        adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
    )

    results = {}

    # 1. Vanilla
    print(f"\n{'▓' * 70}")
    set_seed(SEED)
    m = VanillaGPT(base_cfg).to(device)
    results['vanilla'] = train_model(m, "Vanilla", dataset, device)
    del m

    # 2. Hybrid Gated
    print(f"\n{'▓' * 70}")
    hybrid_cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
        use_rope=True, use_swiglu=True, use_bian_gua=True,
        adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
        architecture_mode='hybrid', gate_init_bias=0.0,
        curriculum_strategy_geo='linear', curriculum_target_strength=0.1,
    )
    set_seed(SEED)
    m = HybridGatedGPT(hybrid_cfg).to(device)
    results['hybrid'] = train_model(m, "HybridGated", dataset, device)
    del m

    # 3. Adaptive Hybrid
    print(f"\n{'▓' * 70}")
    adaptive_cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=128, n_layers=4, n_heads=4,
        block_size=BLOCK_SIZE, batch_size=BATCH_SIZE,
        use_rope=True, use_swiglu=True, use_bian_gua=True,
        adaptive_temp=True, hex_strength=0.01, total_steps=STEPS,
        architecture_mode='hybrid', gate_init_bias=0.0,
        curriculum_strategy_geo='none', curriculum_target_strength=0.1,
    )
    set_seed(SEED)
    m = AdaptiveHybridGPT(adaptive_cfg).to(device)
    results['adaptive'] = train_model(m, "AdaptiveHybrid", dataset, device)
    del m

    # Summary
    print(f"\n\n{'=' * 70}")
    print("  LANGUAGE DATA RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<20} {'BestVal':>10} {'Unique':>8} {'BigramDiv':>10} {'GeoGates':>10}")
    print(f"  {'-' * 60}")
    for name in ['vanilla', 'hybrid', 'adaptive']:
        r = results[name]
        gg = f"{r['gate_info'].get('geo_gates', '-')}/{r['gate_info'].get('total_gates', '-')}"
        print(f"  {name:<20} {r['best_val']:>10.4f} "
              f"{r['generation']['unique_tokens']:>8} "
              f"{r['generation']['bigram_diversity']:>10.2f} {gg:>10}")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'language_data_results.json')
    output_path = os.path.abspath(output_path)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
