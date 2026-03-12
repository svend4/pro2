#!/usr/bin/env python3
"""
v67 Post-Cascade Matryoshka Benchmark.

Урок v66: MatryoshkaQuantizer между камерами Наутилуса добавляет шум,
разрушая каскадное обогащение (PPL 3.07 vs 1.01 у чистого Наутилуса).

Решение v67: Наутилус работает без помех, MatryoshkaQuantizer применяется
ПОСЛЕ каскада. x_ref = вход (пространство), x = выход Наутилуса (время).
Level 2 кодирует «что изменил весь Наутилус».

Configs:
  1. vanilla             — чистый трансформер (baseline)
  2. nautilus_only       — NautilusHierarchy (v66 показал PPL 1.01)
  3. post_cascade_full   — PostCascadeMatryoshkaNautilus, все 7 камер
  4. post_cascade_2ch    — PostCascadeMatryoshkaNautilus, 2 камеры (heisenberg+flower_gat)
  5. inter_chamber_seq   — MatryoshkaNautilus v66 sequential (контроль: PPL ~3.07)

Гипотезы:
  H1: post_cascade_full ≈ nautilus_only (матрёшка не мешает)
  H2: post_cascade_full < nautilus_only (матрёшка помогает!)
  H3: post_cascade_full << inter_chamber_seq (post > inter)

Usage:
  python benchmark_v67_post_cascade.py --group all
  python benchmark_v67_post_cascade.py --run post_cascade_full
  python benchmark_v67_post_cascade.py --summary
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
    MatryoshkaNautilus,
    PostCascadeMatryoshkaNautilus,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmark_v67_post_cascade_results.json'
)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def load_data(block_size=128, seed=42):
    random.seed(seed)
    words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "it", "was",
        "for", "on", "are", "with", "as", "his", "they", "be", "at", "one",
        "have", "this", "from", "by", "not", "but", "what", "all", "were", "when",
        "we", "there", "can", "an", "your", "which", "their", "said", "each", "she",
        "do", "how", "will", "up", "other", "about", "out", "many", "then", "them",
        "would", "like", "so", "these", "her", "long", "make", "thing", "see", "him",
        "two", "has", "look", "more", "day", "could", "go", "come", "did", "my",
        "no", "most", "who", "over", "know", "than", "call", "first", "people", "may",
        "down", "been", "now", "find", "any", "new", "work", "part", "take", "get",
        "place", "made", "after", "back", "only", "use", "where", "good", "very", "still",
    ]
    lines = []
    for _ in range(60000):
        n = random.randint(5, 25)
        line = ' '.join(random.choice(words) for _ in range(n))
        lines.append(line + '.')
    full = '\n'.join(lines)
    split = int(len(full) * 0.9)
    train_text, val_text = full[:split], full[split:]
    train_data = torch.tensor(list(train_text.encode('utf-8')), dtype=torch.long)
    val_data = torch.tensor(list(val_text.encode('utf-8')), dtype=torch.long)
    print(f"  Train: {len(train_data):,}, Val: {len(val_data):,} tokens (byte-level)")
    return train_data, val_data, 256


def get_batch(data, block_size, batch_size):
    n = len(data) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


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
    """LM with configurable enrichment."""

    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 block_size=128, dropout=0.05,
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
        elif enrichment == 'matryoshka_nautilus':
            self.enricher = MatryoshkaNautilus(d_model=d_model, **ekw)
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


CONFIGS = {
    'vanilla': dict(enrichment='none'),
    'nautilus_only': dict(
        enrichment='nautilus',
        enrichment_kwargs=dict(warmup_steps=200),
    ),
    'post_cascade_full': dict(
        enrichment='post_cascade',
        enrichment_kwargs=dict(warmup_steps=200),
    ),
    'post_cascade_2ch': dict(
        enrichment='post_cascade',
        enrichment_kwargs=dict(
            warmup_steps=200,
            enabled_chambers=['heisenberg', 'flower_gat'],
        ),
    ),
    'inter_chamber_seq': dict(
        enrichment='matryoshka_nautilus',
        enrichment_kwargs=dict(mode='sequential', warmup_steps=200),
    ),
}

CONFIG_ORDER = ['vanilla', 'nautilus_only', 'post_cascade_full',
                'post_cascade_2ch', 'inter_chamber_seq']


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


def train_and_eval(name, overrides, train_data, val_data, vocab_size,
                   d_model=128, n_layers=4, n_heads=4, block_size=128,
                   n_steps=800, batch_size=16, lr=1e-3, eval_every=200):
    set_seed(42)

    model = BenchmarkLM(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, **overrides,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*70}")
    print(f"  {name} ({n_params:,} params)")
    print(f"{'='*70}")

    history = []
    t0 = time.time()
    best_val = float('inf')

    for step in range(1, n_steps + 1):
        model.train()
        model.set_step(step)

        progress = step / n_steps
        cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, block_size, batch_size)
        _, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            vl, ppl = evaluate(model, val_data, block_size, batch_size)
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
                # Inter-chamber avg gate
                m_gates = [v for k, v in stats.items()
                           if k.startswith('matryoshka/') and k.endswith('/gate')
                           and k != 'matryoshka/gate']
                if m_gates:
                    extra += f" mg_avg={sum(m_gates)/len(m_gates):.3f}"

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f}{extra}")
            history.append({
                'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl,
                **{k: v for k, v in stats.items() if isinstance(v, (int, float, bool))},
            })

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")

    return {
        'name': name, 'params': n_params,
        'enrichment': overrides.get('enrichment', 'none'),
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
    }


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_result(name, result):
    data = load_results()
    data[name] = result
    data['_last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_PATH, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  >> Saved '{name}' to {RESULTS_PATH}")


def run_single(name, train_data, val_data, vocab_size):
    if name not in CONFIGS:
        print(f"  ERROR: Unknown config '{name}'. Available: {', '.join(CONFIG_ORDER)}")
        return None
    result = train_and_eval(name, CONFIGS[name], train_data, val_data, vocab_size)
    save_result(name, result)
    return result


def print_summary():
    data = load_results()
    if not data:
        print("  No results yet.")
        return

    meta_keys = {'_last_updated'}
    results = {k: v for k, v in data.items() if k not in meta_keys}
    if not results:
        print("  No results yet.")
        return

    print(f"\n  Last updated: {data.get('_last_updated', '?')}")
    print("\n" + "=" * 100)
    print("  v67 POST-CASCADE MATRYOSHKA: Nautilus uninterrupted + Matryoshka after")
    print("=" * 100)
    print(f"  {'Config':<24} {'Enrichment':<24} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8} {'Delta':>8}")
    print("  " + "-" * 96)

    vanilla_ppl = results.get('vanilla', {}).get('best_ppl')

    for name in CONFIG_ORDER:
        if name not in results:
            print(f"  {name:<24} {'':>24} {'---':>8} {'(pending)':>10}")
            continue
        r = results[name]
        delta_str = ""
        if vanilla_ppl is not None and name != 'vanilla':
            delta = r['best_ppl'] - vanilla_ppl
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}"
        print(f"  {name:<24} {r.get('enrichment', 'none'):<24} {r['params']:>8,} "
              f"{r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} "
              f"{r['time']:>7.1f}s {delta_str:>8}")

    print(f"\n  Hypothesis Testing:")
    naut = results.get('nautilus_only', {}).get('best_ppl')
    pc_full = results.get('post_cascade_full', {}).get('best_ppl')
    pc_2ch = results.get('post_cascade_2ch', {}).get('best_ppl')
    inter = results.get('inter_chamber_seq', {}).get('best_ppl')

    if pc_full is not None and naut is not None:
        diff = abs(pc_full - naut)
        if pc_full < naut:
            verdict = f"CONFIRMED (matryoshka helps! {pc_full:.2f} < {naut:.2f})"
        elif diff < 0.1:
            verdict = f"CONFIRMED (≈neutral, {pc_full:.2f} ≈ {naut:.2f})"
        else:
            verdict = f"REJECTED ({pc_full:.2f} > {naut:.2f})"
        print(f"  H1 (post_cascade ≈ nautilus):          {verdict}")

    if pc_full is not None and naut is not None:
        print(f"  H2 (post_cascade < nautilus):           "
              f"{'CONFIRMED' if pc_full < naut else 'REJECTED'}"
              f"  ({pc_full:.2f} vs {naut:.2f})")

    if pc_full is not None and inter is not None:
        print(f"  H3 (post_cascade << inter_chamber):     "
              f"{'CONFIRMED' if pc_full < inter else 'REJECTED'}"
              f"  ({pc_full:.2f} vs {inter:.2f})")

    done = sum(1 for n in CONFIG_ORDER if n in results)
    print(f"\n  Progress: {done}/{len(CONFIG_ORDER)} configs done")


def main():
    parser = argparse.ArgumentParser(description='v67 Post-Cascade Matryoshka Benchmark')
    parser.add_argument('--run', type=str, help='Run single config')
    parser.add_argument('--group', type=str,
                        choices=['baselines', 'post_cascade', 'all'],
                        help='Run group of configs')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    print("  Loading data...")
    train_data, val_data, vocab_size = load_data()

    if args.run:
        run_single(args.run, train_data, val_data, vocab_size)
    elif args.group:
        if args.group == 'baselines':
            names = ['vanilla', 'nautilus_only']
        elif args.group == 'post_cascade':
            names = ['post_cascade_full', 'post_cascade_2ch']
        else:
            names = CONFIG_ORDER
        for name in names:
            run_single(name, train_data, val_data, vocab_size)
        print_summary()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
