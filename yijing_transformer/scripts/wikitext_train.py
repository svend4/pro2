#!/usr/bin/env python3
"""
A1: Масштабирование на реальные данные — WikiText-2 training pipeline.

Обучает Vanilla vs Hybrid vs Adaptive на WikiText-2 (char/BPE).
Сравнивает val perplexity, generation diversity, training dynamics.

Использование:
    python scripts/wikitext_train.py --tokenizer byte --steps 2000
    python scripts/wikitext_train.py --tokenizer bpe --vocab-size 2048 --steps 5000
"""

import os
import sys
import math
import time
import json
import argparse
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import HybridGatedGPT, AdaptiveHybridGPT
from models.baseline import VanillaGPT


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


@torch.no_grad()
def evaluate(model, dataset, device, block_size, batch_size=8, n_eval=50):
    model.eval()
    losses = []
    for _ in range(n_eval):
        xb, yb = dataset.get_batch(batch_size, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]
        losses.append(loss.item())
    model.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss, math.exp(avg_loss)  # loss, perplexity


@torch.no_grad()
def test_generation(model, tokenizer, device, block_size, prompt_text="The "):
    """Генерирует текст и считает метрики разнообразия."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt_text)[:16]
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if hasattr(model, 'generate'):
        output = model.generate(prompt, max_new_tokens=128, temperature=0.8, top_k=40)
    else:
        idx = prompt
        for _ in range(128):
            idx_input = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits = model(idx_input)[0]
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, min(40, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        output = idx

    gen_ids = output[0, len(prompt_ids):].tolist()
    gen_text = tokenizer.decode(gen_ids)

    unique = len(set(gen_ids))
    total = len(gen_ids)
    bigrams = [(gen_ids[i], gen_ids[i+1]) for i in range(len(gen_ids)-1)]
    unique_bigrams = len(set(bigrams))

    return {
        'text': gen_text[:200],
        'unique_tokens': unique,
        'total_tokens': total,
        'rep_rate': 1 - unique / max(total, 1),
        'bigram_diversity': unique_bigrams / max(len(bigrams), 1),
    }


def train_model(model, name, train_ds, val_ds, device, args):
    """Обучает одну модель."""
    total_params, geo_params = model.count_parameters()
    print(f"\n  {name}: {total_params:,} params ({geo_params:,} geo)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    best_val = float('inf')
    best_ppl = float('inf')
    train_losses = []
    val_losses = []
    t_start = time.time()

    model.train()
    accum = 0.0
    log_interval = max(1, args.steps // 20)
    val_interval = max(1, args.steps // 10)

    for step in range(1, args.steps + 1):
        lr = get_lr(step, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if hasattr(model, 'update_curriculum'):
            model.update_curriculum(step)

        xb, yb = train_ds.get_batch(args.batch_size, device)
        result = model(xb, yb)
        loss = result[1] if len(result) >= 2 else result[0]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        accum += loss.item()

        if step % log_interval == 0:
            avg = accum / log_interval
            train_losses.append(avg)
            accum = 0.0
            print(f"    step {step}/{args.steps}: train_loss={avg:.4f} lr={lr:.6f}")

        if step % val_interval == 0:
            vl, ppl = evaluate(model, val_ds, device, args.block_size, args.batch_size)
            val_losses.append(vl)
            if vl < best_val:
                best_val = vl
                best_ppl = ppl
            print(f"    → val_loss={vl:.4f} ppl={ppl:.1f} (best={best_val:.4f})")

    elapsed = time.time() - t_start

    # Generation test
    gen = test_generation(model, train_ds.tokenizer, device, args.block_size)
    print(f"    Generated: {gen['text'][:80]}...")
    print(f"    Diversity: unique={gen['unique_tokens']}/{gen['total_tokens']} "
          f"bigram_div={gen['bigram_diversity']:.3f}")

    # Gate info
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

    return {
        'params': total_params,
        'geo_params': geo_params,
        'best_val': best_val,
        'best_ppl': best_ppl,
        'train_curve': train_losses,
        'val_curve': val_losses,
        'elapsed': elapsed,
        'generation': gen,
        'gate_info': gate_info,
    }


def run(args):
    device = torch.device(args.device)
    set_seed(args.seed)

    print("=" * 70)
    print("  A1: WIKITEXT REAL DATA EXPERIMENT")
    print("=" * 70)

    # Load data
    try:
        from data_utils.wikitext_dataset import WikiTextDataset
        train_ds, val_ds, test_ds = WikiTextDataset.from_huggingface(
            name=args.dataset,
            block_size=args.block_size,
            vocab_size=args.vocab_size,
            tokenizer_type=args.tokenizer,
            verbose=True,
        )
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        print(f"  Falling back to synthetic data with real-like patterns...")

        # Fallback: generate text with realistic patterns
        from tokenizer.char_tokenizer import ByteTokenizer
        tokenizer = ByteTokenizer()
        vocab = tokenizer.get_piece_size()

        # Generate substantial synthetic corpus
        corpus_lines = []
        words = ["the", "of", "and", "to", "in", "a", "is", "that", "it", "was",
                 "for", "on", "are", "with", "as", "his", "they", "be", "at", "one",
                 "have", "this", "from", "by", "not", "but", "what", "all", "were", "when",
                 "we", "there", "can", "an", "your", "which", "their", "said", "each", "she",
                 "do", "how", "will", "up", "other", "about", "out", "many", "then", "them",
                 "would", "like", "so", "these", "her", "long", "make", "thing", "see", "him",
                 "two", "has", "look", "more", "day", "could", "go", "come", "did", "my",
                 "no", "most", "who", "over", "know", "than", "call", "first", "people", "may",
                 "down", "been", "now", "find", "any", "new", "work", "part", "take", "get",
                 "place", "made", "after", "back", "only", "use", "where", "good", "very", "still"]

        random.seed(args.seed)
        for _ in range(50000):
            n_words = random.randint(5, 25)
            line = ' '.join(random.choice(words) for _ in range(n_words))
            corpus_lines.append(line + '.')

        full_text = '\n'.join(corpus_lines)
        split = int(len(full_text) * 0.9)
        train_ids = tokenizer.encode(full_text[:split])
        val_ids = tokenizer.encode(full_text[split:])

        from data_utils.wikitext_dataset import WikiTextDataset
        train_ds = WikiTextDataset(train_ids, args.block_size, tokenizer, vocab)
        val_ds = WikiTextDataset(val_ids, args.block_size, tokenizer, vocab)
        test_ds = val_ds

    vocab_size = train_ds.get_vocab_size()
    print(f"\n  Effective vocab: {vocab_size}")
    print(f"  Train: {train_ds}")
    print(f"  Val:   {val_ds}")

    results = {}

    # 1. Vanilla baseline
    print(f"\n{'▓' * 70}")
    print("  Training: Vanilla GPT")
    base_cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, block_size=args.block_size,
        batch_size=args.batch_size, use_rope=True, use_swiglu=True,
        total_steps=args.steps,
    )
    set_seed(args.seed)
    model = VanillaGPT(base_cfg).to(device)
    results['vanilla'] = train_model(model, "Vanilla", train_ds, val_ds, device, args)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Hybrid Gated
    print(f"\n{'▓' * 70}")
    print("  Training: Hybrid Gated GPT")
    hybrid_cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, block_size=args.block_size,
        batch_size=args.batch_size, use_rope=True, use_swiglu=True,
        use_bian_gua=True, adaptive_temp=True, hex_strength=0.01,
        total_steps=args.steps,
        architecture_mode='hybrid', gate_init_bias=0.0,
        curriculum_strategy_geo='linear', curriculum_target_strength=0.1,
    )
    set_seed(args.seed)
    model = HybridGatedGPT(hybrid_cfg).to(device)
    results['hybrid'] = train_model(model, "HybridGated", train_ds, val_ds, device, args)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Adaptive Hybrid
    print(f"\n{'▓' * 70}")
    print("  Training: Adaptive Hybrid GPT")
    adaptive_cfg = YiJingConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, block_size=args.block_size,
        batch_size=args.batch_size, use_rope=True, use_swiglu=True,
        use_bian_gua=True, adaptive_temp=True, hex_strength=0.01,
        total_steps=args.steps,
        architecture_mode='hybrid', gate_init_bias=0.0,
        curriculum_strategy_geo='none',
    )
    set_seed(args.seed)
    model = AdaptiveHybridGPT(adaptive_cfg).to(device)
    results['adaptive'] = train_model(model, "AdaptiveHybrid", train_ds, val_ds, device, args)
    del model

    # Summary
    print(f"\n\n{'=' * 70}")
    print("  WIKITEXT RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<18} {'Params':>10} {'BestVal':>10} {'PPL':>8} "
          f"{'Unique':>7} {'BigramDiv':>10} {'Time':>8}")
    print(f"  {'-' * 75}")
    for name in ['vanilla', 'hybrid', 'adaptive']:
        r = results[name]
        g = r['generation']
        print(f"  {name:<18} {r['params']:>10,} {r['best_val']:>10.4f} "
              f"{r['best_ppl']:>8.1f} {g['unique_tokens']:>7} "
              f"{g['bigram_diversity']:>10.3f} {r['elapsed']:>7.1f}s")

    # Save
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'wikitext_results.json'
    )
    output_path = os.path.abspath(output_path)

    # Remove non-serializable text
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()}

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='A1: WikiText training')
    parser.add_argument('--dataset', default='wikitext-2-raw-v1')
    parser.add_argument('--tokenizer', default='byte', choices=['bpe', 'byte', 'char'])
    parser.add_argument('--vocab-size', type=int, default=2048)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
