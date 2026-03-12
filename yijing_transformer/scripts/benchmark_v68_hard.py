#!/usr/bin/env python3
"""
v68 Hard Benchmark: усложнённые данные для честной проверки MatryoshkaQuantizer.

Проблема v65-v67: игрушечный датасет (100 слов, случайный порядок) →
Наутилус запоминает полностью (PPL→1.0) → матрёшке нечего добавлять.

Решение: три типа данных + маленькая модель (d_model=64, n_layers=2).

═══════════════════════════════════════════════════════════════════════
Датасет A — «Structured Synthetic» (усложнённая синтетика):
  - 500 слов (вместо 100)
  - Грамматическая структура: Subject Verb Object Adverb
  - Тематические кластеры: слова из одной темы чаще рядом
  - Согласование: «the cat sleeps» но не «the cat sleep»
  → PPL не может достичь 1.0 из-за разнообразия и структуры

Датасет B — «Algorithmic» (алгоритмические паттерны):
  - Задача: распознать и продолжить паттерн с дальними зависимостями
  - Типы: repeat (ABCABC), reverse (ABCCBA), arithmetic (1+2=3)
  - Пространственно-временная структура: позиция зависит от прошлого
  → Матрёшечное spacetime кодирование теоретически помогает

Датасет C — «Source Code» (собственный код проекта):
  - ~2MB Python из yijing_transformer/
  - Реальный код с реальной структурой
  - Вложенные конструкции, отступы, длинные зависимости
  → Самый сложный и реалистичный из трёх
═══════════════════════════════════════════════════════════════════════

Configs (на каждом датасете):
  1. vanilla           — чистый трансформер
  2. nautilus_only     — NautilusHierarchy
  3. post_cascade      — PostCascadeMatryoshkaNautilus

Модель: d_model=64, n_layers=2, n_heads=4 (маленькая, не запоминает)

Usage:
  python benchmark_v68_hard.py --group all
  python benchmark_v68_hard.py --dataset structured --run post_cascade
  python benchmark_v68_hard.py --summary
"""

import sys
import os
import math
import time
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from yijing_transformer.models.geometry.nautilus import (
    NautilusHierarchy,
    PostCascadeMatryoshkaNautilus,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmark_v68_hard_results.json'
)

D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
BLOCK_SIZE = 128
N_STEPS = 800
BATCH_SIZE = 16
LR = 1e-3


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


# ═══════════════════════════════════════════════════════════════
# Dataset A: Structured Synthetic
# ═══════════════════════════════════════════════════════════════

SUBJECTS = {
    'singular': [
        "the cat", "the dog", "a bird", "the fish", "a man", "the woman",
        "the child", "a teacher", "the king", "a queen", "the robot",
        "a ghost", "the river", "the mountain", "a shadow", "the fire",
        "a wolf", "the dragon", "the monk", "a knight", "the star",
        "a stone", "the wind", "the forest", "a flower", "the ocean",
        "a spider", "the snake", "the eagle", "a deer", "the bear",
    ],
    'plural': [
        "the cats", "the dogs", "some birds", "the fish", "the men",
        "the women", "the children", "some teachers", "the kings",
        "the queens", "the robots", "some ghosts", "the rivers",
        "the mountains", "the shadows", "the fires", "the wolves",
        "the dragons", "the monks", "some knights", "the stars",
        "some stones", "the winds", "the forests", "some flowers",
        "the oceans", "some spiders", "the snakes", "the eagles",
        "some deer", "the bears",
    ],
}

VERBS = {
    'singular': [
        "sees", "hears", "finds", "takes", "makes", "knows", "wants",
        "loves", "hates", "follows", "watches", "builds", "breaks",
        "opens", "closes", "carries", "drops", "catches", "throws",
        "draws", "reads", "writes", "eats", "drinks", "grows",
    ],
    'plural': [
        "see", "hear", "find", "take", "make", "know", "want",
        "love", "hate", "follow", "watch", "build", "break",
        "open", "close", "carry", "drop", "catch", "throw",
        "draw", "read", "write", "eat", "drink", "grow",
    ],
}

OBJECTS = [
    "the light", "a path", "the truth", "a door", "the water",
    "a book", "the sword", "a shield", "the gold", "a secret",
    "the mirror", "a bridge", "the tower", "a song", "the map",
    "a key", "the ring", "a pearl", "the scroll", "a lantern",
    "the crystal", "a feather", "the mask", "a coin", "the gate",
    "a rope", "the crown", "a gem", "the staff", "a horn",
]

ADVERBS = [
    "quickly", "slowly", "quietly", "loudly", "carefully",
    "suddenly", "gently", "fiercely", "wisely", "foolishly",
    "silently", "eagerly", "bravely", "sadly", "happily",
    "deeply", "freely", "blindly", "boldly", "calmly",
]

CONJUNCTIONS = ["and", "but", "then", "while", "because", "although", "so", "yet"]
PREPOSITIONS = [
    "in the dark", "at dawn", "under the sky", "near the wall",
    "by the lake", "on the hill", "through the mist", "after the rain",
    "before the storm", "during the night", "beyond the gate",
    "above the clouds", "below the earth", "within the cave",
]


def generate_structured_sentence():
    """Generate a grammatically structured sentence with agreement."""
    parts = []

    # Choose singular or plural (agreement)
    number = random.choice(['singular', 'plural'])
    subj = random.choice(SUBJECTS[number])
    verb = random.choice(VERBS[number])
    obj = random.choice(OBJECTS)

    parts.append(subj)
    parts.append(verb)
    parts.append(obj)

    # Optionally add adverb (30%)
    if random.random() < 0.3:
        parts.append(random.choice(ADVERBS))

    # Optionally add prepositional phrase (25%)
    if random.random() < 0.25:
        parts.append(random.choice(PREPOSITIONS))

    sentence = ' '.join(parts)

    # Optionally add conjunction + second clause (20%)
    if random.random() < 0.2:
        conj = random.choice(CONJUNCTIONS)
        number2 = random.choice(['singular', 'plural'])
        subj2 = random.choice(SUBJECTS[number2])
        verb2 = random.choice(VERBS[number2])
        obj2 = random.choice(OBJECTS)
        sentence += f" {conj} {subj2} {verb2} {obj2}"

    return sentence + "."


def load_structured_data(block_size=BLOCK_SIZE, seed=42):
    """Generate structured synthetic dataset."""
    random.seed(seed)
    lines = [generate_structured_sentence() for _ in range(80000)]
    full = '\n'.join(lines)
    split = int(len(full) * 0.9)
    train_text, val_text = full[:split], full[split:]
    train_data = torch.tensor(list(train_text.encode('utf-8')), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode('utf-8')), dtype=torch.long)
    print(f"  [Structured] Train: {len(train_data):,}, Val: {len(val_data):,} tokens")
    return train_data, val_data, 256


# ═══════════════════════════════════════════════════════════════
# Dataset B: Algorithmic Patterns
# ═══════════════════════════════════════════════════════════════

def generate_algorithmic_sequence():
    """Generate sequences with patterns requiring long-range dependencies."""
    task = random.choice(['repeat', 'reverse', 'arithmetic', 'mirror'])

    if task == 'repeat':
        # PATTERN: ABC|ABC|ABC (repeat pattern 2-3 times)
        length = random.randint(3, 6)
        pattern = [random.randint(ord('a'), ord('z')) for _ in range(length)]
        repeats = random.randint(2, 3)
        seq = pattern * repeats
        return ''.join(chr(c) for c in seq) + '|'

    elif task == 'reverse':
        # PATTERN: ABCDE>EDCBA (reverse after marker)
        length = random.randint(3, 8)
        forward = [random.randint(ord('a'), ord('z')) for _ in range(length)]
        backward = list(reversed(forward))
        return ''.join(chr(c) for c in forward) + '>' + ''.join(chr(c) for c in backward) + '|'

    elif task == 'arithmetic':
        # PATTERN: 12+34=46 (simple addition in string form)
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        op = random.choice(['+', '-'])
        if op == '+':
            result = a + b
        else:
            result = a - b
        return f"{a}{op}{b}={result}|"

    elif task == 'mirror':
        # PATTERN: ABCD|DCBA (palindrome around center)
        length = random.randint(3, 6)
        half = [random.randint(ord('a'), ord('z')) for _ in range(length)]
        seq = half + list(reversed(half))
        return ''.join(chr(c) for c in seq) + '|'


def load_algorithmic_data(block_size=BLOCK_SIZE, seed=42):
    """Generate algorithmic pattern dataset."""
    random.seed(seed)
    sequences = [generate_algorithmic_sequence() for _ in range(200000)]
    full = ''.join(sequences)
    split = int(len(full) * 0.9)
    train_text, val_text = full[:split], full[split:]
    train_data = torch.tensor(list(train_text.encode('utf-8')), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode('utf-8')), dtype=torch.long)
    print(f"  [Algorithmic] Train: {len(train_data):,}, Val: {len(val_data):,} tokens")
    return train_data, val_data, 256


# ═══════════════════════════════════════════════════════════════
# Dataset C: Source Code (project's own Python files)
# ═══════════════════════════════════════════════════════════════

def load_sourcecode_data(block_size=BLOCK_SIZE, seed=42):
    """Load project's own Python source as training data."""
    import glob
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py_files = glob.glob(os.path.join(project_root, '**', '*.py'), recursive=True)
    # Exclude benchmark scripts to avoid data leakage
    py_files = [f for f in py_files if 'benchmark' not in os.path.basename(f)]

    chunks = []
    for fpath in sorted(py_files):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                chunks.append(f.read())
        except (UnicodeDecodeError, IOError):
            continue

    full = '\n\n'.join(chunks)
    # Shuffle at file level for better train/val split
    random.seed(seed)
    random.shuffle(chunks)
    full = '\n\n'.join(chunks)

    split = int(len(full) * 0.9)
    train_text, val_text = full[:split], full[split:]
    train_data = torch.tensor(list(train_text.encode('utf-8')), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode('utf-8')), dtype=torch.long)
    print(f"  [SourceCode] Train: {len(train_data):,}, Val: {len(val_data):,} tokens ({len(py_files)} files)")
    return train_data, val_data, 256


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        h = self.ln1(x)
        T = h.size(1)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class BenchmarkLM(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, block_size=BLOCK_SIZE, dropout=0.05,
                 enrichment='none', enrichment_kwargs=None):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.enrichment_type = enrichment

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        n_first = n_layers // 2
        n_second = n_layers - n_first
        self.layers_first = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_first)
        ])
        self.layers_second = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_second)
        ])

        ekw = enrichment_kwargs or {}
        if enrichment == 'nautilus':
            self.enricher = NautilusHierarchy(d_model=d_model, **ekw)
        elif enrichment == 'post_cascade':
            self.enricher = PostCascadeMatryoshkaNautilus(d_model=d_model, **ekw)
        elif enrichment == 'none':
            self.enricher = None
        else:
            raise ValueError(f"Unknown enrichment: {enrichment}")

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._step = 0

    def set_step(self, step):
        self._step = step
        if self.enricher is not None and hasattr(self.enricher, 'set_step'):
            self.enricher.set_step(step)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        for layer in self.layers_first:
            x = layer(x)

        if self.enricher is not None:
            x, _ = self.enricher(x)

        for layer in self.layers_second:
            x = layer(x)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_enrichment_stats(self):
        if self.enricher is not None and hasattr(self.enricher, 'get_stats'):
            return self.enricher.get_stats()
        return {}


def get_batch(data, block_size, batch_size):
    n = len(data) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# ═══════════════════════════════════════════════════════════════
# Configs
# ═══════════════════════════════════════════════════════════════

ENRICHMENT_CONFIGS = {
    'vanilla': dict(enrichment='none'),
    'nautilus_only': dict(
        enrichment='nautilus',
        enrichment_kwargs=dict(warmup_steps=200),
    ),
    'post_cascade': dict(
        enrichment='post_cascade',
        enrichment_kwargs=dict(warmup_steps=200),
    ),
}

ENRICHMENT_ORDER = ['vanilla', 'nautilus_only', 'post_cascade']
DATASET_NAMES = ['structured', 'algorithmic', 'sourcecode']


@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, n_eval=30):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, block_size, batch_size)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))


def train_and_eval(name, dataset_name, overrides, train_data, val_data, vocab_size):
    set_seed(42)

    model = BenchmarkLM(vocab_size=vocab_size, **overrides)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"  [{dataset_name}] {name} ({n_params:,} params, d={D_MODEL}, L={N_LAYERS})")
    print(f"{'='*70}")

    history = []
    t0 = time.time()
    best_val = float('inf')

    for step in range(1, N_STEPS + 1):
        model.train()
        model.set_step(step)

        progress = step / N_STEPS
        cur_lr = LR * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        _, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0 or step == 1:
            vl, ppl = evaluate(model, val_data, BLOCK_SIZE, BATCH_SIZE)
            best_val = min(best_val, vl)

            extra = ""
            stats = model.get_enrichment_stats()
            if stats:
                rg = stats.get('nautilus/residual_gate')
                if rg is not None:
                    extra += f" rg={rg:.3f}"
                mg = stats.get('matryoshka/gate')
                if mg is not None:
                    extra += f" mg={mg:.3f}"

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.2f}{extra}")
            history.append({
                'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl,
                **{k: v for k, v in stats.items() if isinstance(v, (int, float, bool))},
            })

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.2f} best_ppl={best_ppl:.2f} time={elapsed:.1f}s")

    return {
        'name': name, 'dataset': dataset_name, 'params': n_params,
        'enrichment': overrides.get('enrichment', 'none'),
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
        'd_model': D_MODEL, 'n_layers': N_LAYERS,
    }


# ═══════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_result(key, result):
    data = load_results()
    data[key] = result
    data['_last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  >> Saved '{key}'")


def result_key(dataset, enrichment):
    return f"{dataset}/{enrichment}"


def print_summary():
    data = load_results()
    if not data:
        print("  No results yet.")
        return

    meta_keys = {'_last_updated'}
    results = {k: v for k, v in data.items() if k not in meta_keys}

    print(f"\n  Last updated: {data.get('_last_updated', '?')}")
    print(f"  Model: d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
    print("\n" + "=" * 100)
    print("  v68 HARD BENCHMARK: Complex Synthetic + Algorithmic Tasks")
    print("=" * 100)

    for ds in DATASET_NAMES:
        ds_results = {k: v for k, v in results.items() if k.startswith(f"{ds}/")}
        if not ds_results:
            print(f"\n  [{ds.upper()}] — no results yet")
            continue

        print(f"\n  [{ds.upper()}]")
        print(f"  {'Config':<20} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8} {'Delta':>8}")
        print("  " + "-" * 70)

        vanilla_key = f"{ds}/vanilla"
        vanilla_ppl = results.get(vanilla_key, {}).get('best_ppl')

        for ename in ENRICHMENT_ORDER:
            key = f"{ds}/{ename}"
            if key not in results:
                print(f"  {ename:<20} {'---':>8} {'(pending)':>10}")
                continue
            r = results[key]
            delta_str = ""
            if vanilla_ppl is not None and ename != 'vanilla':
                delta = r['best_ppl'] - vanilla_ppl
                pct = (delta / vanilla_ppl) * 100
                sign = "+" if delta >= 0 else ""
                delta_str = f"{sign}{delta:.2f} ({sign}{pct:.1f}%)"
            print(f"  {ename:<20} {r['params']:>8,} {r['best_ppl']:>10.2f} "
                  f"{r['final_ppl']:>10.2f} {r['time']:>7.1f}s {delta_str:>16}")

    # Cross-dataset comparison
    print(f"\n  Cross-Dataset Comparison:")
    for ename in ENRICHMENT_ORDER:
        ppls = []
        for ds in DATASET_NAMES:
            key = f"{ds}/{ename}"
            if key in results:
                ppls.append(f"{ds}={results[key]['best_ppl']:.2f}")
        if ppls:
            print(f"  {ename:<20} {', '.join(ppls)}")

    # Key question
    print(f"\n  KEY QUESTION: Does post_cascade beat nautilus_only on harder data?")
    for ds in DATASET_NAMES:
        naut = results.get(f"{ds}/nautilus_only", {}).get('best_ppl')
        pc = results.get(f"{ds}/post_cascade", {}).get('best_ppl')
        if naut is not None and pc is not None:
            diff = pc - naut
            if diff < -0.05:
                verdict = f"YES! matryoshka helps ({pc:.2f} < {naut:.2f})"
            elif abs(diff) < 0.05:
                verdict = f"neutral ({pc:.2f} ≈ {naut:.2f})"
            else:
                verdict = f"no ({pc:.2f} > {naut:.2f})"
            print(f"    [{ds}]: {verdict}")

    done = sum(1 for k in results if '/' in k)
    total = len(DATASET_NAMES) * len(ENRICHMENT_ORDER)
    print(f"\n  Progress: {done}/{total} configs done")


def main():
    parser = argparse.ArgumentParser(description='v68 Hard Benchmark')
    parser.add_argument('--run', type=str, help='Enrichment config name')
    parser.add_argument('--dataset', type=str, choices=DATASET_NAMES,
                        help='Dataset to use')
    parser.add_argument('--group', type=str,
                        choices=['structured', 'algorithmic', 'sourcecode', 'all'],
                        help='Run group')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.run and args.dataset:
        print("  Loading data...")
        if args.dataset == 'structured':
            train_data, val_data, vocab_size = load_structured_data()
        elif args.dataset == 'algorithmic':
            train_data, val_data, vocab_size = load_algorithmic_data()
        elif args.dataset == 'sourcecode':
            train_data, val_data, vocab_size = load_sourcecode_data()
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        overrides = ENRICHMENT_CONFIGS[args.run]
        result = train_and_eval(args.run, args.dataset, overrides,
                                train_data, val_data, vocab_size)
        save_result(result_key(args.dataset, args.run), result)
        return

    if args.group:
        datasets = DATASET_NAMES if args.group == 'all' else [args.group]
        for ds in datasets:
            print(f"\n  Loading {ds} data...")
            if ds == 'structured':
                train_data, val_data, vocab_size = load_structured_data()
            elif ds == 'algorithmic':
                train_data, val_data, vocab_size = load_algorithmic_data()
            elif ds == 'sourcecode':
                train_data, val_data, vocab_size = load_sourcecode_data()
            else:
                raise ValueError(f"Unknown dataset: {ds}")

            for ename in ENRICHMENT_ORDER:
                overrides = ENRICHMENT_CONFIGS[ename]
                result = train_and_eval(ename, ds, overrides,
                                        train_data, val_data, vocab_size)
                save_result(result_key(ds, ename), result)

        print_summary()
        return

    parser.print_help()


if __name__ == '__main__':
    main()
