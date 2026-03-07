#!/usr/bin/env python3
"""
Direction A: WikiText-2 perplexity benchmark — v53 geometry vs vanilla.

Compares YiJingGPT with different v51 source configurations on language modeling.
Uses HuggingFace WikiText-2 or synthetic fallback.

Configs tested:
  1. vanilla        — no geometry
  2. fomyuk         — D4-equivariant + antipodal (best on modadd)
  3. belyaev        — Heisenberg + FlowerOfLife + Möbius + StructuralDefect
  4. fomyuk+belyaev — best two combined (without palace attention)
"""

import sys, os, math, time, json, random
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
            # Add sentence structure variation
            if random.random() < 0.3:
                line = line.capitalize()
            lines.append(line + '.')
        full = '\n'.join(lines)
        split = int(len(full) * 0.9)
        train_text, val_text = full[:split], full[split:]
        print(f"  Synthetic: {len(train_text):,} train, {len(val_text):,} val chars")

    # Byte-level tokenization (vocab=256, no training needed)
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


def make_config(source_name, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=256):
    base = dict(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, block_size=block_size, dropout=0.05,
        use_rope=True, use_swiglu=True,
        hex_strength=0.05, temp=0.3,
    )
    source_configs = {
        'vanilla': {},
        'fomyuk': dict(
            use_d4_equivariant=True, use_antipodal_reg=True, antipodal_weight=0.01,
        ),
        'belyaev': dict(
            use_heisenberg_attention=True, use_flower_gat=True,
            use_mobius_bias=True, use_structural_defect=True,
        ),
        'fomyuk_belyaev': dict(
            use_d4_equivariant=True, use_antipodal_reg=True, antipodal_weight=0.01,
            use_heisenberg_attention=True, use_flower_gat=True,
            use_mobius_bias=True, use_structural_defect=True,
        ),
        'andreev': dict(
            use_triangular_bias=True, use_four_level_pe=True, use_bidirectional_tri=True,
        ),
    }
    return YiJingConfig(**{**base, **source_configs[source_name]})


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
            # Compute loss manually
            loss_val = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss_val.item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))  # cap perplexity for display


def train_and_eval(cfg, source_name, train_data, val_data,
                   n_steps=3000, batch_size=16, lr=1e-3,
                   eval_every=200, device='cpu'):
    set_seed(42)
    model = YiJingGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- {source_name} ({n_params:,} params) ---")

    history = {'step': [], 'train_loss': [], 'val_loss': [], 'val_ppl': []}
    t0 = time.time()
    best_val = float('inf')

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
            if step <= 600 or step % 600 == 0 or step == n_steps:
                print(f"  Step {step:5d}: train={loss.item():.4f} val={vl:.4f} ppl={ppl:.1f} lr={cur_lr:.6f}")

    elapsed = time.time() - t0
    final_vl, final_ppl = history['val_loss'][-1], history['val_ppl'][-1]
    best_ppl = math.exp(min(best_val, 20))
    print(f"  FINAL: val_loss={final_vl:.4f} ppl={final_ppl:.1f} best_ppl={best_ppl:.1f} time={elapsed:.1f}s")

    return {
        'source': source_name,
        'params': n_params,
        'final_val_loss': final_vl,
        'final_ppl': final_ppl,
        'best_val_loss': best_val,
        'best_ppl': best_ppl,
        'elapsed': elapsed,
        'ppl_at_600': history['val_ppl'][3] if len(history['val_ppl']) > 3 else None,
        'ppl_at_1200': history['val_ppl'][6] if len(history['val_ppl']) > 6 else None,
        'history': history,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("=" * 70)
    print("Direction A: WikiText-2 Perplexity Benchmark (v53)")
    print("=" * 70)

    # Config
    d_model = 128
    n_layers = 4
    n_heads = 4
    block_size = 256
    n_steps = 3000
    batch_size = 16

    train_data, val_data, vocab_size = load_data(block_size=block_size)

    sources = ['vanilla', 'fomyuk', 'belyaev', 'andreev', 'fomyuk_belyaev']

    results = {}
    for s in sources:
        cfg = make_config(s, vocab_size, d_model, n_layers, n_heads, block_size)
        results[s] = train_and_eval(
            cfg, s, train_data, val_data,
            n_steps=n_steps, batch_size=batch_size, device=device,
        )

    # Summary
    print("\n" + "=" * 70)
    print("WIKITEXT BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Source':<20} {'Params':>9} {'PPL@600':>9} {'PPL@1200':>9} "
          f"{'FinalPPL':>9} {'BestPPL':>9} {'ValLoss':>9}")
    print("-" * 75)
    for s in sources:
        r = results[s]
        p600 = f"{r['ppl_at_600']:.1f}" if r['ppl_at_600'] else "  n/a"
        p1200 = f"{r['ppl_at_1200']:.1f}" if r['ppl_at_1200'] else "  n/a"
        print(f"{s:<20} {r['params']:>9,} {p600:>9} {p1200:>9} "
              f"{r['final_ppl']:>9.1f} {r['best_ppl']:>9.1f} {r['final_val_loss']:>9.4f}")

    vanilla_ppl = results['vanilla']['best_ppl']
    print(f"\nΔ vs vanilla (best perplexity):")
    for s in sources:
        if s == 'vanilla':
            continue
        r = results[s]
        dppl = r['best_ppl'] - vanilla_ppl
        pct = (dppl / vanilla_ppl) * 100
        print(f"  {s:<20} Δppl={dppl:+.1f} ({pct:+.1f}%)")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'benchmark_wikitext_v53_results.json')
    compact = {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in results.items()}
    with open(out_path, 'w') as f:
        json.dump(compact, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
