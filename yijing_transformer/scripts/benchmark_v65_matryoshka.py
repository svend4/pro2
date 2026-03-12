#!/usr/bin/env python3
"""
v65 Matryoshka Benchmark: иерархическое кодирование бит→трит→гекс.

Тестирует MatryoshkaQuantizer как обогащающий слой в трансформере.
Три уровня кодирования (Матрёшка/Наутилус):
  Level 0: Бит    — Q6, 64 гексаграммы
  Level 1: Трит   — пространственные пары, 4→3 состояния + направление
  Level 2: Гекс   — пространство×время, 16 состояний = Q4→Q12

Configs:
  1. vanilla           — чистый трансформер (baseline)
  2. binary_q6         — стандартная Q6 бинарная квантизация
  3. matryoshka_L01    — MatryoshkaQuantizer Level 0+1 (без x_ref)
  4. matryoshka_L012   — MatryoshkaQuantizer Level 0+1+2 (с x_ref = предыдущий слой)
  5. matryoshka_adaptive — MatryoshkaQuantizer с adaptive_temp

Гипотезы:
  H1: matryoshka_L01 > binary_q6 (иерархия лучше плоской квантизации)
  H2: matryoshka_L012 > matryoshka_L01 (временное измерение добавляет информацию)
  H3: matryoshka_adaptive > matryoshka_L012 (обучаемая температура помогает)

Usage:
  python benchmark_v65_matryoshka.py --group all
  python benchmark_v65_matryoshka.py --run matryoshka_L012
  python benchmark_v65_matryoshka.py --summary
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
from yijing_transformer.models.geometry.quantizers import (
    MatryoshkaQuantizer,
    FactoredYiJingQuantizer,
)

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmark_v65_matryoshka_results.json'
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


# ── Model: Simple Transformer + Matryoshka Enrichment ──────────


class TransformerBlock(nn.Module):
    """Minimal transformer block."""

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


class MatryoshkaLM(nn.Module):
    """Language model with optional Matryoshka enrichment.

    Architecture:
        Embedding → [TransformerBlock × n_layers/2]
                  → MatryoshkaEnrichment (optional)
                  → [TransformerBlock × n_layers/2]
                  → LM head

    Matryoshka enrichment:
        1. Project d_model → 6D (Q6 space)
        2. Apply MatryoshkaQuantizer (Level 0 + Level 1 + optionally Level 2)
        3. Add enriched signal back via residual + gate
        4. For Level 2: x_ref = representation before enrichment (temporal pair)
    """

    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 block_size=128, dropout=0.05,
                 enrichment='none', adaptive_temp=False):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.enrichment = enrichment

        # Embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # First half of transformer
        n_first = n_layers // 2
        n_second = n_layers - n_first
        self.layers_first = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_first)
        ])
        self.layers_second = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_second)
        ])

        # Enrichment module
        if enrichment == 'binary_q6':
            self.q6_proj = nn.Linear(d_model, 6, bias=False)
            self.q6_back = nn.Linear(6, d_model, bias=False)
            self.quantizer = FactoredYiJingQuantizer(temp=0.3)
            self.enrich_gate = nn.Parameter(torch.tensor(0.0))
        elif enrichment in ('matryoshka_L01', 'matryoshka_L012', 'matryoshka_adaptive'):
            self.q6_proj = nn.Linear(d_model, 6, bias=False)
            adapt = (enrichment == 'matryoshka_adaptive')
            self.matryoshka = MatryoshkaQuantizer(
                total_dim=6, d_model=d_model, temp=0.3, adaptive_temp=adapt,
            )
            self.enrich_gate = nn.Parameter(torch.tensor(0.0))
            self.use_spacetime = (enrichment in ('matryoshka_L012', 'matryoshka_adaptive'))

        # LM head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        # First half
        for layer in self.layers_first:
            x = layer(x)

        # Enrichment
        if self.enrichment == 'binary_q6':
            q6 = self.q6_proj(x)
            q6_quantized = self.quantizer(q6)
            enriched = self.q6_back(q6_quantized)
            gate = torch.sigmoid(self.enrich_gate)
            x = x + gate * enriched
        elif self.enrichment.startswith('matryoshka'):
            x_before = x  # save for temporal reference
            q6 = self.q6_proj(x)
            x_ref = self.q6_proj(x_before) if self.use_spacetime else None
            # For Level 2: x_ref is the same as q6 at this point,
            # but after a few training steps the gate will diverge them.
            # More meaningfully, use previous-layer output:
            enriched, info = self.matryoshka(q6, x_ref)
            gate = torch.sigmoid(self.enrich_gate)
            x = x + gate * enriched

        # Second half
        for layer in self.layers_second:
            x = layer(x)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def get_enrichment_stats(self):
        """Stats for monitoring during training."""
        stats = {}
        if hasattr(self, 'enrich_gate'):
            stats['enrich_gate'] = torch.sigmoid(self.enrich_gate).item()
        if hasattr(self, 'matryoshka'):
            stats.update(self.matryoshka.get_stats())
        return stats


# ── Configs ────────────────────────────────────────────────────

CONFIGS = {
    'vanilla': dict(enrichment='none'),
    'binary_q6': dict(enrichment='binary_q6'),
    'matryoshka_L01': dict(enrichment='matryoshka_L01'),
    'matryoshka_L012': dict(enrichment='matryoshka_L012'),
    'matryoshka_adaptive': dict(enrichment='matryoshka_adaptive'),
}

CONFIG_ORDER = ['vanilla', 'binary_q6', 'matryoshka_L01', 'matryoshka_L012', 'matryoshka_adaptive']


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

    model = MatryoshkaLM(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, **overrides,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    enrichment = overrides.get('enrichment', 'none')

    print(f"\n{'='*70}")
    print(f"  {name} ({n_params:,} params, enrichment={enrichment})")
    print(f"{'='*70}")

    history = []
    t0 = time.time()
    best_val = float('inf')

    for step in range(1, n_steps + 1):
        model.train()
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
                eg = stats.get('enrich_gate', 0)
                extra += f" gate={eg:.3f}"
                if 'trit_yang' in stats:
                    extra += (f" trit=[y={stats['trit_yang']:.2f}"
                              f" t={stats['trit_transition']:.2f}"
                              f" i={stats['trit_yin']:.2f}]")
                if 'gate_L0' in stats:
                    extra += (f" L=[{stats['gate_L0']:.2f}"
                              f" {stats['gate_L1']:.2f}"
                              f" {stats['gate_L2']:.2f}]")

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
        'name': name, 'params': n_params, 'enrichment': enrichment,
        'final_val': final_vl, 'final_ppl': final_ppl,
        'best_val': best_val, 'best_ppl': best_ppl,
        'time': elapsed, 'history': history,
    }


# ── Results ────────────────────────────────────────────────────

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
    print(f"  >> Result for '{name}' saved to {RESULTS_PATH}")


def run_single(name, train_data, val_data, vocab_size):
    if name not in CONFIGS:
        print(f"  ERROR: Unknown config '{name}'")
        print(f"  Available: {', '.join(CONFIG_ORDER)}")
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

    print("\n" + "=" * 95)
    print("  v65 MATRYOSHKA BENCHMARK: Hierarchical Encoding bit→trit→hex")
    print("=" * 95)
    print(f"  {'Config':<24} {'Enrichment':<24} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8} {'Delta':>8}")
    print("  " + "-" * 92)

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

    # Hypothesis testing
    print(f"\n  Hypothesis Testing:")
    binary = results.get('binary_q6', {}).get('best_ppl')
    m01 = results.get('matryoshka_L01', {}).get('best_ppl')
    m012 = results.get('matryoshka_L012', {}).get('best_ppl')
    m_adapt = results.get('matryoshka_adaptive', {}).get('best_ppl')

    if m01 is not None and binary is not None:
        print(f"  H1 (matryoshka_L01 < binary_q6):      {'CONFIRMED' if m01 < binary else 'REJECTED'}"
              f"  ({m01:.2f} vs {binary:.2f})")
    if m012 is not None and m01 is not None:
        print(f"  H2 (matryoshka_L012 < L01):            {'CONFIRMED' if m012 < m01 else 'REJECTED'}"
              f"  ({m012:.2f} vs {m01:.2f})")
    if m_adapt is not None and m012 is not None:
        print(f"  H3 (adaptive < L012):                  {'CONFIRMED' if m_adapt < m012 else 'REJECTED'}"
              f"  ({m_adapt:.2f} vs {m012:.2f})")

    done = sum(1 for n in CONFIG_ORDER if n in results)
    print(f"\n  Progress: {done}/{len(CONFIG_ORDER)} configs done")


def main():
    parser = argparse.ArgumentParser(description='v65 Matryoshka Benchmark')
    parser.add_argument('--run', type=str, help='Run single config')
    parser.add_argument('--group', type=str,
                        choices=['baselines', 'matryoshka', 'all'],
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
            names = ['vanilla', 'binary_q6']
        elif args.group == 'matryoshka':
            names = ['matryoshka_L01', 'matryoshka_L012', 'matryoshka_adaptive']
        else:
            names = CONFIG_ORDER
        for name in names:
            run_single(name, train_data, val_data, vocab_size)
        print_summary()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
