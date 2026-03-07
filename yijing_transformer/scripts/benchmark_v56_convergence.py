#!/usr/bin/env python3
"""
v56 Benchmark: Convergence Bridge + TernaryQuantizer + MatrixGrammar.

Fair-parametric сравнение при одинаковом бюджете параметров:

  1. vanilla         — стандартный transformer (baseline)
  2. yijing_binary   — стандартная бинарная квантизация {-1,+1}⁶
  3. yijing_ternary  — тернарная квантизация {-1,0,+1}⁶ (变爻)
  4. matrix_grammar  — + 2D матричная грамматика (Atamiri)
  5. convergence     — + конвергентный мост глиф↔токен
  6. full_v56        — все три вместе

Данные: WikiText-2 (byte-level) или synthetic fallback.
Метрики: val perplexity, train loss, параметры, время.
"""

import sys
import os
import math
import time
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
from yijing_transformer.config import YiJingConfig
from yijing_transformer.models.model import YiJingGPT


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step, total_steps, base_lr, warmup_frac=0.1):
    warmup = int(total_steps * warmup_frac)
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_data(block_size=256, seed=42):
    """Load WikiText-2 or fall back to synthetic corpus."""
    try:
        from datasets import load_dataset
        print("  Loading WikiText-2 from HuggingFace...")
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1')
        train_text = '\n'.join(l for l in ds['train']['text'] if l.strip())
        val_text = '\n'.join(l for l in ds['validation']['text'] if l.strip())
        print(f"  Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")
    except Exception as e:
        print(f"  HuggingFace unavailable ({e}), using synthetic corpus...")
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
            "just", "should", "because", "old", "year", "before", "also", "way", "well", "much",
        ]
        lines = []
        for _ in range(80000):
            n = random.randint(5, 30)
            line = ' '.join(random.choice(words) for _ in range(n))
            if random.random() < 0.3:
                line = line.capitalize()
            lines.append(line + '.')
        full = '\n'.join(lines)
        split = int(len(full) * 0.9)
        train_text, val_text = full[:split], full[split:]
        print(f"  Synthetic: {len(train_text):,} train, {len(val_text):,} val chars")

    train_ids = list(train_text.encode('utf-8'))
    val_ids = list(val_text.encode('utf-8'))
    vocab_size = 256

    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)
    print(f"  Vocab: {vocab_size}, Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
    return train_data, val_data, vocab_size


def get_batch(data, block_size, batch_size, device):
    n = len(data) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y


CONFIGS = {
    'vanilla': dict(
        architecture_mode='standard',
        hex_strength=0.0,
    ),
    'yijing_binary': dict(
        hex_strength=0.05,
        quantizer_type='factored6',
    ),
    'yijing_ternary': dict(
        hex_strength=0.05,
        quantizer_type='ternary',
        ternary_mode='factored',
        ternary_uncertainty=0.3,
    ),
    'matrix_grammar': dict(
        hex_strength=0.05,
        quantizer_type='factored6',
        use_matrix_grammar=True,
        matrix_grammar_rows=4,
        matrix_grammar_cols=4,
        matrix_grammar_heads=4,
    ),
    'convergence': dict(
        hex_strength=0.05,
        quantizer_type='factored6',
        use_convergence_bridge=True,
        convergence_n_clusters=64,
        convergence_window_size=4,
        convergence_stride=2,
        convergence_compose_layers=1,
        convergence_n_heads=4,
    ),
    'full_v56': dict(
        hex_strength=0.05,
        quantizer_type='ternary',
        ternary_mode='factored',
        ternary_uncertainty=0.3,
        use_matrix_grammar=True,
        matrix_grammar_rows=4,
        matrix_grammar_cols=4,
        matrix_grammar_heads=4,
        use_convergence_bridge=True,
        convergence_n_clusters=64,
        convergence_window_size=4,
        convergence_stride=2,
        convergence_compose_layers=1,
        convergence_n_heads=4,
    ),
}


def make_config(name, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=256):
    base = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, dropout=0.05,
        use_rope=True, use_swiglu=True, temp=0.3,
    )
    return YiJingConfig(**{**base, **CONFIGS[name]})


@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, device, n_eval=50):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, block_size, batch_size, device)
        logits, loss, _ = model(x, targets=y)
        if isinstance(loss, torch.Tensor):
            losses.append(loss.item())
        else:
            loss_val = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss_val.item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))


def train_and_eval(cfg, name, train_data, val_data,
                   n_steps=2000, batch_size=16, lr=1e-3,
                   eval_every=200, device='cpu'):
    set_seed(42)
    model = YiJingGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {name} ({n_params:,} params)")
    print(f"{'='*60}")

    history = {'step': [], 'train_loss': [], 'val_loss': [], 'val_ppl': []}
    t0 = time.time()
    best_val = float('inf')

    # Extra diagnostics for v56 features
    ternary_info = []

    for step in range(1, n_steps + 1):
        model.train()
        cur_lr = get_lr(step, n_steps, lr)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        logits, loss, _ = model(x, targets=y)

        if not isinstance(loss, torch.Tensor):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            vl, ppl = evaluate(model, val_data, cfg.block_size, batch_size, device)
            if vl < best_val:
                best_val = vl
            history['step'].append(step)
            history['train_loss'].append(loss.item())
            history['val_loss'].append(vl)
            history['val_ppl'].append(ppl)

            # Collect ternary diagnostics
            extra = ""
            if cfg.quantizer_type == 'ternary':
                for layer in model.core.layers:
                    if hasattr(layer.quantizer, 'uncertainty_budget'):
                        ub = layer.quantizer.uncertainty_budget.item()
                        extra = f" ub={ub:.3f}"
                        break

            # Collect convergence correlation
            if getattr(cfg, 'use_convergence_bridge', False) and hasattr(model, 'convergence_bridge'):
                corr = model.convergence_bridge.token_abstractor.cluster_hexagram_correlation().item()
                extra += f" corr={corr:.3f}"

            print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f}{extra}")

    elapsed = time.time() - t0
    final_vl, final_ppl = history['val_loss'][-1], history['val_ppl'][-1]
    best_ppl = math.exp(min(best_val, 20))

    # Final diagnostics
    extra_info = {}
    if cfg.quantizer_type == 'ternary':
        for layer in model.core.layers:
            if hasattr(layer.quantizer, 'uncertainty_budget'):
                extra_info['final_uncertainty_budget'] = layer.quantizer.uncertainty_budget.item()
                # Analyze bian yao distribution
                with torch.no_grad():
                    sample_x, _ = get_batch(train_data, cfg.block_size, batch_size, device)
                    emb = model.tok_emb(sample_x)
                    z = layer.to_qd(layer.ln_hex(emb))
                    info = layer.quantizer.analyze_bian_yao(z)
                    extra_info['avg_bian_yao'] = info['n_bian_yao']
                    extra_info['uncertainty_per_dim'] = info['uncertainty_per_dim'].tolist()
                break

    if getattr(cfg, 'use_convergence_bridge', False) and hasattr(model, 'convergence_bridge'):
        extra_info['final_hex_correlation'] = model.convergence_bridge.token_abstractor.cluster_hexagram_correlation().item()
        extra_info['bridge_scale'] = model.convergence_bridge.bridge_scale.item()

    if getattr(cfg, 'use_matrix_grammar', False) and hasattr(model, 'matrix_grammar'):
        extra_info['matrix_grammar_scale'] = model.matrix_grammar.scale.item()

    print(f"\n  FINAL: val_loss={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")
    if extra_info:
        for k, v in extra_info.items():
            if isinstance(v, list):
                print(f"    {k}: [{', '.join(f'{x:.3f}' for x in v)}]")
            else:
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    return {
        'name': name,
        'params': n_params,
        'final_val_loss': final_vl,
        'final_ppl': final_ppl,
        'best_val_loss': best_val,
        'best_ppl': best_ppl,
        'elapsed': elapsed,
        'history': history,
        **extra_info,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 70)
    print("v56 Benchmark: Convergence Bridge + Ternary + MatrixGrammar")
    print("=" * 70)

    d_model = 128
    n_layers = 4
    n_heads = 4
    block_size = 256
    n_steps = 2000
    batch_size = 16
    lr = 1e-3

    print("\n[1/2] Loading data...")
    train_data, val_data, vocab_size = load_data(block_size)

    configs_to_run = ['vanilla', 'yijing_binary', 'yijing_ternary',
                      'matrix_grammar', 'convergence', 'full_v56']

    results = []
    print(f"\n[2/2] Training {len(configs_to_run)} configurations ({n_steps} steps each)...")

    for name in configs_to_run:
        cfg = make_config(name, vocab_size, d_model, n_layers, n_heads, block_size)
        result = train_and_eval(cfg, name, train_data, val_data,
                                n_steps=n_steps, batch_size=batch_size, lr=lr,
                                eval_every=200, device=device)
        results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Params':>10} {'Best PPL':>10} {'Final PPL':>10} {'Time':>8} {'Δ PPL':>8}")
    print("-" * 80)

    vanilla_ppl = results[0]['best_ppl']
    for r in results:
        delta = r['best_ppl'] - vanilla_ppl
        sign = "+" if delta > 0 else ""
        print(f"{r['name']:<20} {r['params']:>10,} {r['best_ppl']:>10.1f} {r['final_ppl']:>10.1f} {r['elapsed']:>7.1f}s {sign}{delta:>7.1f}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'benchmark_v56_results.json')
    # Remove non-serializable items
    for r in results:
        if 'uncertainty_per_dim' in r and isinstance(r.get('uncertainty_per_dim'), list):
            pass  # already list, ok
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
