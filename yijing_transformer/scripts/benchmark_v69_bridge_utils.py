#!/usr/bin/env python3
"""
v69 Bridge Utils Benchmark — Сравнение «до» и «после» подключения утилит v12..v52.

Все предыдущие бенчмарки (v63–v68) использовали голый AdamW + cosine schedule.
Теперь тестируем лучшие архитектуры с Bridge-утилитами:

Архитектуры (фиксированы):
  1. vanilla          — чистый трансформер (baseline)
  2. nautilus_only    — NautilusHierarchy (лучшая архитектура, PPL ~1.01)
  3. post_cascade     — PostCascadeMatryoshkaNautilus (PPL ~1.01, помогает на hard)

Оптимизаторы:
  A. AdamW           — baseline (уже протестирован)
  B. Sophia          — second-order (Hessian diagonal)
  C. Lion            — evolved sign momentum
  D. AdamW+Lookahead — slow/fast weights
  E. AdamW+SAM       — sharpness-aware minimization

Schedulers:
  - Cosine (baseline, уже протестирован)
  - WSD (Warmup-Stable-Decay, Llama 3 style)

Регуляризация:
  - AGC (Adaptive Gradient Clipping) вместо фиксированного clip_grad_norm
  - Z-Loss (PaLM-style logit stabilization)

Матрица: 3 архитектуры × 10 комбинаций оптимизатор/scheduler/reg

Usage:
  python benchmark_v69_bridge_utils.py --group all
  python benchmark_v69_bridge_utils.py --group quick     # только nautilus × оптимизаторы
  python benchmark_v69_bridge_utils.py --run nautilus/sophia
  python benchmark_v69_bridge_utils.py --summary
"""

import sys
import os
import math
import time
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from yijing_transformer.models.geometry.nautilus import (
    NautilusHierarchy,
    PostCascadeMatryoshkaNautilus,
)

# Bridge imports
from training.utils_v19 import Sophia
from training.utils_v45 import Lion
from training.utils_v23 import Lookahead
from training.utils_v27 import SAM
from training.utils_v21 import AGC
from training.utils_v15 import z_loss
from training.utils_v18 import WSDScheduler

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'benchmark_v69_bridge_utils_results_v2.json'
)


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def _generate_vocab(n_words, seed=123):
    """Generate a diverse vocabulary of n_words unique pseudo-words."""
    rng = random.Random(seed)
    # Start with 200 real English words for natural byte patterns
    base = [
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
        "world", "should", "house", "between", "life", "never", "before", "great", "just", "state",
        "away", "through", "number", "hand", "high", "keep", "last", "city", "tree", "cross",
        "might", "close", "seem", "light", "along", "every", "under", "name", "school", "right",
        "think", "home", "give", "water", "room", "small", "end", "group", "play", "run",
        "start", "move", "kind", "need", "point", "old", "line", "open", "head", "turn",
        "real", "leave", "help", "next", "big", "large", "man", "woman", "child", "year",
        "different", "important", "possible", "national", "political", "social", "economic", "public",
        "international", "local", "general", "major", "current", "similar", "specific", "available",
        "natural", "military", "particular", "financial", "environmental", "medical", "traditional",
        "popular", "significant", "serious", "common", "individual", "necessary", "legal",
    ]
    # Generate additional pseudo-words via syllable combination
    onsets = ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w",
              "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl",
              "sm", "sn", "sp", "st", "str", "sw", "tr", "thr", "wh", "wr", "ch", "sh", "th"]
    vowels = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "oo", "ou", "oi", "au", "ie"]
    codas = ["", "d", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x", "ng", "nt", "nd", "st", "ct"]
    vocab = set(base)
    while len(vocab) < n_words:
        n_syl = rng.choice([1, 1, 2, 2, 2, 3, 3])
        word = ""
        for _ in range(n_syl):
            word += rng.choice(onsets) + rng.choice(vowels) + rng.choice(codas)
        if len(word) >= 2 and word not in vocab:
            vocab.add(word)
    return sorted(vocab)


def load_data(block_size=128, seed=42):
    random.seed(seed)
    words = _generate_vocab(5000)
    # Sentence templates for structural variety
    templates = [
        lambda w, r: ' '.join(r.choice(w) for _ in range(r.randint(5, 30))) + '.',
        lambda w, r: r.choice(w).capitalize() + ' ' + ' '.join(r.choice(w) for _ in range(r.randint(4, 20))) + ', ' + ' '.join(r.choice(w) for _ in range(r.randint(3, 15))) + '.',
        lambda w, r: r.choice(w).capitalize() + ' ' + r.choice(w) + ' ' + r.choice(w) + '? ' + r.choice(w).capitalize() + ' ' + ' '.join(r.choice(w) for _ in range(r.randint(3, 12))) + '.',
        lambda w, r: '"' + ' '.join(r.choice(w) for _ in range(r.randint(3, 15))) + '," ' + r.choice(w) + ' ' + r.choice(w) + '.',
        lambda w, r: ' '.join(r.choice(w) for _ in range(r.randint(8, 25))) + '; ' + ' '.join(r.choice(w) for _ in range(r.randint(5, 15))) + '.',
    ]
    rng = random.Random(seed)
    lines = []
    for _ in range(500000):
        tmpl = rng.choice(templates)
        lines.append(tmpl(words, rng))
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
    """LM with configurable enrichment (reused from v67)."""

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


# ==================== Конфигурации ====================

# Архитектуры
ARCH_CONFIGS = {
    'vanilla': dict(enrichment='none'),
    'nautilus': dict(enrichment='nautilus', enrichment_kwargs=dict(warmup_steps=200)),
    'post_cascade': dict(enrichment='post_cascade', enrichment_kwargs=dict(warmup_steps=200)),
}

# Оптимизатор + scheduler + регуляризация
TRAINING_CONFIGS = {
    # Baseline: AdamW + Cosine (как в v67)
    'adamw': dict(
        optimizer='adamw', lr=1e-3, weight_decay=0.01,
        scheduler='cosine', use_agc=False, use_zloss=False,
    ),
    # Sophia
    'sophia': dict(
        optimizer='sophia', lr=1e-4, weight_decay=0.1,
        scheduler='cosine', use_agc=False, use_zloss=False,
    ),
    # Lion (рекомендован 3-10x меньший lr, 3x больший wd)
    'lion': dict(
        optimizer='lion', lr=3e-4, weight_decay=0.03,
        scheduler='cosine', use_agc=False, use_zloss=False,
    ),
    # AdamW + Lookahead
    'adamw_lookahead': dict(
        optimizer='adamw', lr=1e-3, weight_decay=0.01,
        wrapper='lookahead', lookahead_k=5, lookahead_alpha=0.5,
        scheduler='cosine', use_agc=False, use_zloss=False,
    ),
    # AdamW + SAM
    'adamw_sam': dict(
        optimizer='adamw', lr=1e-3, weight_decay=0.01,
        wrapper='sam', sam_rho=0.05,
        scheduler='cosine', use_agc=False, use_zloss=False,
    ),
    # AdamW + WSD scheduler
    'adamw_wsd': dict(
        optimizer='adamw', lr=1e-3, weight_decay=0.01,
        scheduler='wsd', use_agc=False, use_zloss=False,
    ),
    # AdamW + AGC (вместо clip_grad_norm)
    'adamw_agc': dict(
        optimizer='adamw', lr=1e-3, weight_decay=0.01,
        scheduler='cosine', use_agc=True, use_zloss=False,
    ),
    # AdamW + Z-Loss
    'adamw_zloss': dict(
        optimizer='adamw', lr=1e-3, weight_decay=0.01,
        scheduler='cosine', use_agc=False, use_zloss=True,
    ),
    # Комбо: Sophia + WSD + AGC + Z-Loss
    'sophia_full': dict(
        optimizer='sophia', lr=1e-4, weight_decay=0.1,
        scheduler='wsd', use_agc=True, use_zloss=True,
    ),
    # Комбо: Lion + Lookahead + AGC
    'lion_lookahead_agc': dict(
        optimizer='lion', lr=3e-4, weight_decay=0.03,
        wrapper='lookahead', lookahead_k=5, lookahead_alpha=0.5,
        scheduler='cosine', use_agc=True, use_zloss=False,
    ),
}


def build_full_config_name(arch_name, train_name):
    return f"{arch_name}/{train_name}"


# Quick group: только nautilus × все оптимизаторы
QUICK_COMBOS = [
    ('nautilus', 'adamw'),
    ('nautilus', 'sophia'),
    ('nautilus', 'lion'),
    ('nautilus', 'adamw_lookahead'),
    ('nautilus', 'adamw_sam'),
    ('nautilus', 'adamw_wsd'),
    ('nautilus', 'adamw_agc'),
    ('nautilus', 'adamw_zloss'),
    ('nautilus', 'sophia_full'),
    ('nautilus', 'lion_lookahead_agc'),
]

# All group: все архитектуры × ключевые конфигурации
ALL_COMBOS = []
for arch in ['vanilla', 'nautilus', 'post_cascade']:
    for train in TRAINING_CONFIGS:
        ALL_COMBOS.append((arch, train))


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


def make_optimizer(model, tcfg):
    """Create optimizer based on training config."""
    opt_type = tcfg['optimizer']
    lr = tcfg['lr']
    wd = tcfg['weight_decay']

    if opt_type == 'adamw':
        base_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == 'sophia':
        base_opt = Sophia(model.parameters(), lr=lr, weight_decay=wd, rho=0.04)
    elif opt_type == 'lion':
        base_opt = Lion(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    # Wrapper
    wrapper = tcfg.get('wrapper')
    if wrapper == 'lookahead':
        base_opt = Lookahead(
            base_opt,
            k=tcfg.get('lookahead_k', 5),
            alpha=tcfg.get('lookahead_alpha', 0.5),
        )
    elif wrapper == 'sam':
        base_opt = SAM(base_opt, rho=tcfg.get('sam_rho', 0.05))

    return base_opt


def get_lr_at_step(step, n_steps, lr, scheduler_type, warmup_frac=0.1):
    """Compute LR at given step."""
    warmup_steps = int(n_steps * warmup_frac)

    if scheduler_type == 'cosine':
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    elif scheduler_type == 'wsd':
        # Warmup-Stable-Decay: 10% warmup, 50% stable, 40% decay
        stable_end = int(n_steps * 0.6)
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        elif step < stable_end:
            return lr
        else:
            decay_progress = (step - stable_end) / max(1, n_steps - stable_end)
            return lr * 0.5 * (1 + math.cos(math.pi * decay_progress))

    return lr


def train_and_eval(config_name, arch_cfg, train_cfg, train_data, val_data, vocab_size,
                   d_model=128, n_layers=4, n_heads=4, block_size=128,
                   n_steps=2000, batch_size=16, eval_every=400):
    set_seed(42)

    model = BenchmarkLM(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, **arch_cfg,
    )
    optimizer = make_optimizer(model, train_cfg)
    is_sam = train_cfg.get('wrapper') == 'sam'

    # AGC
    agc = None
    if train_cfg.get('use_agc'):
        agc = AGC(model, clip_factor=0.01)

    n_params = sum(p.numel() for p in model.parameters())
    lr = train_cfg['lr']
    scheduler_type = train_cfg['scheduler']

    print(f"\n{'='*70}")
    print(f"  {config_name} ({n_params:,} params)")
    print(f"  opt={train_cfg['optimizer']} lr={lr} wd={train_cfg['weight_decay']}"
          f" sched={scheduler_type} agc={train_cfg.get('use_agc', False)}"
          f" zloss={train_cfg.get('use_zloss', False)}"
          f" wrapper={train_cfg.get('wrapper', 'none')}")
    print(f"{'='*70}")

    history = []
    t0 = time.time()
    best_val = float('inf')

    for step in range(1, n_steps + 1):
        model.train()
        model.set_step(step)

        # LR schedule
        cur_lr = get_lr_at_step(step, n_steps, lr, scheduler_type)
        for pg in optimizer.param_groups if not is_sam else optimizer.optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, block_size, batch_size)
        logits, loss = model(x, targets=y)

        # Z-Loss
        if train_cfg.get('use_zloss'):
            loss = loss + z_loss(logits, weight=1e-4)

        if is_sam:
            # SAM: first forward-backward with perturbation
            loss.backward()
            optimizer.first_step()

            # SAM: second forward-backward at perturbed point
            _, loss2 = model(x, targets=y)
            if train_cfg.get('use_zloss'):
                loss2 = loss2 + z_loss(logits, weight=1e-4)
            optimizer.zero_grad()
            loss2.backward()

            # AGC on second backward
            if agc is not None:
                agc.clip()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.second_step()
        else:
            optimizer.zero_grad()
            loss.backward()

            # AGC or standard clipping
            if agc is not None:
                agc.clip()
            else:
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

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.2f}"
                  f" lr={cur_lr:.6f}{extra}")
            history.append({
                'step': step, 'train': loss.item(), 'val': vl, 'ppl': ppl,
            })

    elapsed = time.time() - t0
    final_vl, final_ppl = history[-1]['val'], history[-1]['ppl']
    best_ppl = math.exp(min(best_val, 20))

    print(f"\n  FINAL: val={final_vl:.4f} ppl={final_ppl:.2f}"
          f" best_ppl={best_ppl:.2f} time={elapsed:.1f}s")

    return {
        'name': config_name,
        'arch': arch_cfg.get('enrichment', 'none'),
        'training': {k: v for k, v in train_cfg.items()},
        'params': n_params,
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


def run_single(arch_name, train_name, train_data, val_data, vocab_size):
    if arch_name not in ARCH_CONFIGS:
        print(f"  ERROR: Unknown arch '{arch_name}'. Available: {list(ARCH_CONFIGS.keys())}")
        return None
    if train_name not in TRAINING_CONFIGS:
        print(f"  ERROR: Unknown training '{train_name}'. Available: {list(TRAINING_CONFIGS.keys())}")
        return None

    config_name = build_full_config_name(arch_name, train_name)
    result = train_and_eval(
        config_name, ARCH_CONFIGS[arch_name], TRAINING_CONFIGS[train_name],
        train_data, val_data, vocab_size,
    )
    save_result(config_name, result)
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
    print(f"\n{'='*110}")
    print("  v69 BRIDGE UTILS BENCHMARK: Same architectures, different optimizers/schedulers/regularization")
    print(f"{'='*110}")

    # Group by architecture
    for arch in ['vanilla', 'nautilus', 'post_cascade']:
        arch_results = {k: v for k, v in results.items() if k.startswith(f'{arch}/')}
        if not arch_results:
            continue

        print(f"\n  --- {arch.upper()} ---")
        print(f"  {'Training Config':<30} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8}")
        print("  " + "-" * 75)

        # Sort by best_ppl
        sorted_configs = sorted(arch_results.items(), key=lambda x: x[1].get('best_ppl', 999))

        baseline_ppl = None
        for name, r in sorted_configs:
            train_name = name.split('/')[1]
            if train_name == 'adamw':
                baseline_ppl = r['best_ppl']
            delta = ""
            if baseline_ppl is not None and train_name != 'adamw':
                d = r['best_ppl'] - baseline_ppl
                delta = f" ({'+' if d >= 0 else ''}{d:.3f})"
            print(f"  {train_name:<30} {r['params']:>8,} {r['best_ppl']:>10.3f}"
                  f" {r['final_ppl']:>10.3f} {r['time']:>7.1f}s{delta}")

    # Cross-architecture comparison for each training config
    print(f"\n  --- CROSS-ARCHITECTURE (best PPL by training config) ---")
    print(f"  {'Training':<24} {'vanilla':>10} {'nautilus':>10} {'post_cascade':>13}")
    print("  " + "-" * 65)

    all_train_names = sorted(set(
        k.split('/')[1] for k in results.keys() if '/' in k
    ))
    for tn in all_train_names:
        vals = []
        for arch in ['vanilla', 'nautilus', 'post_cascade']:
            key = f'{arch}/{tn}'
            if key in results:
                vals.append(f"{results[key]['best_ppl']:>10.3f}")
            else:
                vals.append(f"{'---':>10}")
        print(f"  {tn:<24} {'  '.join(vals)}")

    done = sum(1 for k in results if '/' in k)
    total = len(ALL_COMBOS)
    print(f"\n  Progress: {done}/{total} configs done")


def main():
    parser = argparse.ArgumentParser(description='v69 Bridge Utils Benchmark')
    parser.add_argument('--run', type=str,
                        help='Run single config, e.g. "nautilus/sophia"')
    parser.add_argument('--group', type=str,
                        choices=['quick', 'all'],
                        help='Run group: quick (nautilus only) or all')
    parser.add_argument('--summary', action='store_true', help='Print summary')
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    print("  Loading data...")
    train_data, val_data, vocab_size = load_data()

    if args.run:
        parts = args.run.split('/')
        if len(parts) != 2:
            print("  ERROR: Use format 'arch/training', e.g. 'nautilus/sophia'")
            return
        run_single(parts[0], parts[1], train_data, val_data, vocab_size)
    elif args.group:
        combos = QUICK_COMBOS if args.group == 'quick' else ALL_COMBOS
        for arch, train in combos:
            config_name = build_full_config_name(arch, train)
            # Skip already done
            existing = load_results()
            if config_name in existing:
                print(f"  SKIP {config_name} (already done)")
                continue
            run_single(arch, train, train_data, val_data, vocab_size)
        print_summary()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
