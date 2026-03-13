#!/usr/bin/env python3
"""
Train YiJing Nautilus model on real data from svend4 repositories.

Data sources (cloned to /tmp/):
  - meta2  (18.8 MB Python)
  - meta   (10.6 MB .txt)
  - data2  (10.5 MB mixed)
  - info1  (7.1 MB MD+Python)
  - info3  (5.5 MB texts+Python)
  - info7  (3.1 MB MD+TS)
  - info4  (2.5 MB MD+skills)
  - info   (0.8 MB MD)
  - info2  (0.5 MB JS+MD)
  - info5  (0.3 MB MD)
  Total: ~64 MB text → 67M byte-level tokens

Best config from v69 benchmark:
  Architecture: nautilus (NautilusHierarchy)
  Optimizer: AdamW (lr=1e-3, wd=0.01)
  Scheduler: WSD (Warmup-Stable-Decay)
  Result on synthetic: PPL 1.010

Usage:
  python train_real_data.py
  python train_real_data.py --steps 5000
  python train_real_data.py --resume checkpoint.pt
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
from yijing_transformer.models.geometry.nautilus import NautilusHierarchy

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'train_real_data_results.json'
)
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'train_real_data_checkpoint.pt'
)

REPO_DIRS = [
    '/tmp/meta2', '/tmp/meta', '/tmp/data2', '/tmp/info1',
    '/tmp/info3', '/tmp/info7', '/tmp/info4', '/tmp/info',
    '/tmp/info2', '/tmp/info5',
]

TEXT_EXTENSIONS = {
    '.txt', '.md', '.py', '.json', '.yaml', '.yml', '.js', '.jsx',
    '.ts', '.tsx', '.html', '.htm', '.css', '.csv', '.xml', '.sql',
    '.sh', '.bat', '.cfg', '.ini', '.toml', '.skill', '.gitignore',
}


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def collect_text_from_repos():
    """Collect all text files from cloned repos."""
    all_text = []
    total_bytes = 0
    total_files = 0

    for repo_dir in REPO_DIRS:
        if not os.path.isdir(repo_dir):
            print(f"  SKIP (not found): {repo_dir}")
            continue

        repo_name = os.path.basename(repo_dir)
        repo_bytes = 0
        repo_files = 0

        for root, dirs, files in os.walk(repo_dir):
            # Skip .git
            dirs[:] = [d for d in dirs if d != '.git']

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                # Also include extensionless files (like Makefile, Dockerfile)
                if ext not in TEXT_EXTENSIONS and ext != '':
                    continue

                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                        content = fh.read()
                    if len(content.strip()) < 10:
                        continue
                    # Add file separator for context
                    all_text.append(f"### {os.path.relpath(path, repo_dir)}\n{content}\n")
                    repo_bytes += len(content)
                    repo_files += 1
                except Exception:
                    continue

        total_bytes += repo_bytes
        total_files += repo_files
        print(f"  {repo_name}: {repo_files} files, {repo_bytes:,} bytes")

    print(f"  TOTAL: {total_files} files, {total_bytes:,} bytes")
    return '\n'.join(all_text)


def load_real_data(block_size=256, val_fraction=0.1):
    """Load real data from repos, encode to bytes, split train/val."""
    print("Loading data from repositories...")
    full_text = collect_text_from_repos()

    if len(full_text) < 1000:
        raise RuntimeError("Not enough text data! Clone repos first: "
                           "cd /tmp && git clone --depth 1 https://github.com/svend4/<repo>.git")

    # Shuffle paragraphs (separated by double newline) to mix repos
    paragraphs = full_text.split('\n\n')
    random.seed(42)
    random.shuffle(paragraphs)
    full_text = '\n\n'.join(paragraphs)

    # Encode to bytes
    raw_bytes = full_text.encode('utf-8')
    data = torch.tensor(list(raw_bytes), dtype=torch.long)

    # Split
    split_idx = int(len(data) * (1 - val_fraction))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"  Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens (byte-level)")
    print(f"  Vocab: 256 (byte-level)")
    return train_data, val_data, 256


# ==================== Model (same as v69 benchmark) ====================

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


class NautilusLM(nn.Module):
    """Language model with Nautilus enrichment — best architecture from v69."""

    def __init__(self, vocab_size=256, d_model=128, n_layers=4, n_heads=4,
                 block_size=256, dropout=0.05, warmup_steps=200):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size

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

        self.enricher = NautilusHierarchy(d_model=d_model, warmup_steps=warmup_steps)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._step = 0

    def set_step(self, step):
        self._step = step
        if hasattr(self.enricher, 'set_step'):
            self.enricher.set_step(step)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        for layer in self.layers_first:
            x = layer(x)

        x, _ = self.enricher(x)

        for layer in self.layers_second:
            x = layer(x)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ==================== Training ====================

def get_batch(data, block_size, batch_size):
    n = len(data) - block_size - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, val_data, block_size, batch_size, n_eval=50):
    model.eval()
    losses = []
    for _ in range(n_eval):
        x, y = get_batch(val_data, block_size, batch_size)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    avg = sum(losses) / len(losses)
    return avg, math.exp(min(avg, 20))


def get_lr_wsd(step, n_steps, lr, warmup_frac=0.1):
    """WSD scheduler: 10% warmup, 50% stable, 40% decay."""
    warmup_steps = int(n_steps * warmup_frac)
    stable_end = int(n_steps * 0.6)

    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    elif step < stable_end:
        return lr
    else:
        decay_progress = (step - stable_end) / max(1, n_steps - stable_end)
        return lr * 0.5 * (1 + math.cos(math.pi * decay_progress))


def generate_sample(model, start_text="# ", max_len=200, temperature=0.8):
    """Generate text sample from the model."""
    model.eval()
    tokens = list(start_text.encode('utf-8'))
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_len):
            logits, _ = model(idx[:, -model.block_size:])
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)

    generated = bytes(idx[0].tolist()).decode('utf-8', errors='replace')
    return generated


def train(args):
    set_seed(42)

    # Load data
    train_data, val_data, vocab_size = load_real_data(
        block_size=args.block_size,
        val_fraction=0.1,
    )

    # Create model
    model = NautilusLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.block_size,
        warmup_steps=int(args.steps * 0.05),  # 5% warmup for enricher
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: NautilusLM ({n_params:,} params)")
    print(f"  d_model={args.d_model}, n_layers={args.n_layers}, "
          f"n_heads={args.n_heads}, block_size={args.block_size}")
    print(f"  Optimizer: AdamW (lr={args.lr}, wd={args.wd})")
    print(f"  Scheduler: WSD ({args.steps} steps)")
    print(f"  Tokens/param ratio: {len(train_data)/n_params:.1f}x")

    # Resume from checkpoint?
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt['step']
        print(f"  Resumed from step {start_step}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.resume and os.path.exists(args.resume) and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    # Training
    history = []
    t0 = time.time()
    best_val = float('inf')
    best_ppl = float('inf')

    print(f"\n{'='*70}")
    print(f"  Training on real data ({len(train_data):,} tokens)")
    print(f"{'='*70}")

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        model.set_step(step)

        # WSD LR schedule
        cur_lr = get_lr_wsd(step, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        x, y = get_batch(train_data, args.block_size, args.batch_size)
        logits, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            vl, ppl = evaluate(model, val_data, args.block_size, args.batch_size)
            if vl < best_val:
                best_val = vl
                best_ppl = ppl

            elapsed = time.time() - t0
            tokens_per_sec = (step - start_step) * args.batch_size * args.block_size / elapsed

            print(f"  Step {step:6d}: train={loss.item():.4f} val={vl:.4f} "
                  f"ppl={ppl:.2f} lr={cur_lr:.6f} "
                  f"[{elapsed:.0f}s, {tokens_per_sec:.0f} tok/s]")

            history.append({
                'step': step, 'train': loss.item(), 'val': vl,
                'ppl': ppl, 'lr': cur_lr,
            })

        # Generate sample periodically
        if step % (args.eval_every * 5) == 0:
            sample = generate_sample(model, start_text="# ", max_len=150)
            print(f"  >>> Sample: {sample[:200]}")

        # Save checkpoint periodically
        if step % (args.eval_every * 2) == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val': best_val,
                'best_ppl': best_ppl,
                'args': vars(args),
            }, CHECKPOINT_PATH)

    elapsed = time.time() - t0

    # Final eval
    final_val, final_ppl = evaluate(model, val_data, args.block_size, args.batch_size, n_eval=100)
    print(f"\n  FINAL: val={final_val:.4f} ppl={final_ppl:.2f} "
          f"best_ppl={best_ppl:.2f} time={elapsed:.1f}s")

    # Generate final samples
    print("\n  === Generated Samples ===")
    for prompt in ["# ", "def ", "import ", "Информация"]:
        sample = generate_sample(model, start_text=prompt, max_len=200)
        print(f"  [{prompt}]: {sample[:250]}")
        print()

    # Save results
    results = {
        'model': 'NautilusLM',
        'params': n_params,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'block_size': args.block_size,
        'training': {
            'optimizer': 'adamw',
            'lr': args.lr,
            'weight_decay': args.wd,
            'scheduler': 'wsd',
            'steps': args.steps,
            'batch_size': args.batch_size,
        },
        'data': {
            'train_tokens': len(train_data),
            'val_tokens': len(val_data),
            'sources': [os.path.basename(d) for d in REPO_DIRS if os.path.isdir(d)],
        },
        'final_val': final_val,
        'final_ppl': final_ppl,
        'best_val': best_val,
        'best_ppl': best_ppl,
        'time_seconds': elapsed,
        'tokens_per_param': len(train_data) / n_params,
        'history': history,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  >> Saved results to {RESULTS_PATH}")

    # Save final checkpoint
    torch.save({
        'step': args.steps,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val': best_val,
        'best_ppl': best_ppl,
        'args': vars(args),
    }, CHECKPOINT_PATH)
    print(f"  >> Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NautilusLM on real data')
    parser.add_argument('--steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--block-size', type=int, default=256, help='Context window')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--eval-every', type=int, default=500, help='Eval interval')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    args = parser.parse_args()

    train(args)
