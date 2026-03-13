#!/usr/bin/env python3
"""
NautilusMoME Self-Description — модель рассказывает о себе.

Загружает обученный чекпоинт и генерирует текст по промптам
о своей архитектуре, плюсах, минусах и возможностях.

Usage:
    python self_description.py
    python self_description.py --checkpoint /path/to/checkpoint.pt
    python self_description.py --temperature 0.9 --max_tokens 200
"""

import sys
import os
import json
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import sentencepiece as spm

from scripts.train_nautilus_mome import (
    NautilusMoME,
    CHECKPOINT_PATH,
    TOKENIZER_MODEL,
    load_tokenizer,
)


# ==================== Generation ====================

@torch.no_grad()
def generate_text(model, sp, prompt, max_tokens=200, temperature=0.8,
                  top_k=40, top_p=0.92, repetition_penalty=1.3):
    """Generate text from prompt with nucleus sampling."""
    model.eval()
    tokens = sp.encode(prompt)
    if not tokens:
        tokens = [sp.bos_id() if sp.bos_id() != -1 else 1]

    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    generated_ids = list(tokens)

    for step in range(max_tokens):
        logits, _, info = model(idx[:, -model.block_size:])
        logits = logits[0, -1, :].clone()

        # Repetition penalty — penalize recently seen tokens
        recent = generated_ids[-60:]
        for tid in set(recent):
            count = recent.count(tid)
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty * (1 + 0.1 * count)
            else:
                logits[tid] *= repetition_penalty * (1 + 0.1 * count)

        logits /= temperature

        # Top-K filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float('-inf')

        probs = F.softmax(logits, dim=-1)

        # Top-P (nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            probs[sorted_idx[mask]] = 0.0

        probs = probs / probs.sum()
        next_token = torch.multinomial(probs, 1)
        token_id = next_token.item()

        if token_id == sp.eos_id() and sp.eos_id() != -1:
            break

        generated_ids.append(token_id)
        idx = torch.cat([idx, next_token.unsqueeze(0)], dim=1)

    return sp.decode(generated_ids)


@torch.no_grad()
def analyze_model_stats(model, sp, val_text=None):
    """Compute model statistics for the self-description report."""
    stats = {}

    # Parameter counts
    total = sum(p.numel() for p in model.parameters())
    core = sum(p.numel() for n, p in model.named_parameters()
               if 'expert' not in n and 'router' not in n and 'bridge' not in n)
    experts = sum(p.numel() for n, p in model.named_parameters() if 'expert' in n)
    router = sum(p.numel() for n, p in model.named_parameters() if 'router' in n)
    bridge = sum(p.numel() for n, p in model.named_parameters() if 'bridge' in n)

    stats['total_params'] = total
    stats['core_params'] = core
    stats['expert_params'] = experts
    stats['router_params'] = router
    stats['bridge_params'] = bridge
    stats['n_experts'] = model.n_experts
    stats['d_model'] = model.d_model
    stats['vocab_size'] = model.vocab_size
    stats['block_size'] = model.block_size
    stats['expert_names'] = model.EXPERT_NAMES[:model.n_experts]

    # Expert gate scales
    gate_scales = {}
    for name, expert in model.experts.items():
        gate_scales[name] = expert.gate_scale.item()
    stats['gate_scales'] = gate_scales

    # Bridge residual gate
    stats['bridge_gate'] = model.bridge.residual_gate.item()

    return stats


@torch.no_grad()
def analyze_routing_distribution(model, sp, prompts):
    """Analyze which experts activate for different prompts."""
    model.eval()
    results = {}

    for prompt in prompts:
        tokens = sp.encode(prompt)
        if not tokens:
            continue
        idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
        _, _, info = model(idx)
        routing = info['routing'][0]  # (T, n_experts)

        # Average routing weights per expert
        avg_weights = routing.mean(dim=0)  # (n_experts,)
        expert_names = model.EXPERT_NAMES[:model.n_experts]
        results[prompt[:50]] = {
            name: f"{avg_weights[i].item():.3f}"
            for i, name in enumerate(expert_names)
        }

    return results


# ==================== Self-Description Report ====================

def generate_self_description(model, sp, temperature=0.8, max_tokens=200):
    """Generate a comprehensive self-description report."""

    print("=" * 70)
    print("  NautilusMoME — Self-Description Report")
    print("=" * 70)

    # --- 1. Model Statistics ---
    stats = analyze_model_stats(model, sp)
    print("\n" + "─" * 70)
    print("  1. WHO AM I (Model Architecture)")
    print("─" * 70)
    print(f"""
  I am NautilusMoME — a Mixture of Micro-Experts language model.

  Architecture:
    Total parameters:  {stats['total_params']:,}
    Core (Transformer): {stats['core_params']:,} params ({stats['core_params']*100//stats['total_params']}%)
    Experts (6 domains): {stats['expert_params']:,} params ({stats['expert_params']*100//stats['total_params']}%)
    Router:             {stats['router_params']:,} params
    Bridge:             {stats['bridge_params']:,} params
    d_model:            {stats['d_model']}
    Vocabulary:         {stats['vocab_size']} BPE tokens
    Context window:     {stats['block_size']} tokens

  Expert domains: {', '.join(stats['expert_names'])}
    """)

    # Gate scales show how much each expert contributes
    print("  Expert gate scales (learned importance):")
    for name, scale in sorted(stats['gate_scales'].items(), key=lambda x: -x[1]):
        bar = '█' * int(abs(scale) * 50) + '░' * max(0, 50 - int(abs(scale) * 50))
        print(f"    {name:8s}: {bar} {scale:.4f}")

    print(f"\n  Bridge residual gate: {stats['bridge_gate']:.4f}")

    # --- 2. What the model generates about itself ---
    print("\n" + "─" * 70)
    print("  2. WHAT I GENERATE (Model's Own Words)")
    print("─" * 70)

    prompts_about_self = [
        "# NautilusMoME Architecture\n\nThis model",
        "class NautilusMoME:\n    \"\"\"",
        "## Advantages\n\n- ",
        "## Limitations\n\n- ",
        "def forward(self, x):\n    # ",
        "# Expert Routing\n\nThe router selects",
        "import torch\nfrom nautilus import",
        "# What this model knows:\n# 1.",
    ]

    for prompt in prompts_about_self:
        print(f"\n  PROMPT: {repr(prompt[:60])}")
        print("  " + "·" * 60)
        result = generate_text(model, sp, prompt,
                               max_tokens=max_tokens,
                               temperature=temperature)
        # Clean and format output
        lines = result.split('\n')
        for line in lines[:15]:  # limit output lines
            print(f"  {line}")
        if len(lines) > 15:
            print(f"  ... ({len(lines) - 15} more lines)")
        print()

    # --- 3. Routing analysis ---
    print("\n" + "─" * 70)
    print("  3. EXPERT ROUTING ANALYSIS")
    print("─" * 70)

    test_prompts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "The ethical implications of artificial intelligence",
        "SELECT * FROM users WHERE created_at > '2024-01-01'",
        "docker compose up -d --build && kubectl apply -f",
        "import React, { useState, useEffect } from 'react'",
        "∑(n=1 to ∞) 1/n² = π²/6",
        "The MBTI personality types include INTJ, ENFP",
        "OCR reconstruction of damaged document fragments",
    ]

    routing = analyze_routing_distribution(model, sp, test_prompts)
    for prompt_key, weights in routing.items():
        print(f"\n  \"{prompt_key}\"")
        top_experts = sorted(weights.items(), key=lambda x: -float(x[1]))[:3]
        for name, w in top_experts:
            w_float = float(w)
            bar = '█' * int(w_float * 40)
            print(f"    → {name:8s}: {bar} {w}")

    # --- 4. Strengths & Weaknesses ---
    print("\n" + "─" * 70)
    print("  4. STRENGTHS & WEAKNESSES (Honest Assessment)")
    print("─" * 70)

    print("""
  STRENGTHS (+):
    + Modular architecture: experts can be added/removed/retrained independently
    + Sparse activation: only 2/6 experts active per token → efficient inference
    + BPE tokenizer (4096 vocab): better than byte-level for code/mixed text
    + Hierarchical bridge: NautilusBridge merges expert outputs intelligently
    + Weight tying: embedding ↔ head reduces parameter count
    + Covers 6 domains: MATH, CODE, HUMAN, SYSTEM, RECON, INFO
    + Small footprint: ~1.8M params fits on any device
    + Trained on 20 real repositories (25.9M BPE tokens)

  WEAKNESSES (−):
    − Very small model: 1.8M params limits generation quality
    − PPL ~17.9: decent for size but far from production LLMs
    − 4096 vocab: covers code well but limited for natural language
    − 512 context window: short for complex reasoning
    − No instruction tuning: generates continuations, not answers
    − Byte-fallback BPE: unknown tokens handled but poorly
    − Single-device only: no distributed training/inference
    − Expert utilization may be uneven (depends on training data balance)

  BEST SUITED FOR:
    ✓ Code completion in Python/TypeScript/React
    ✓ Understanding project structure and patterns
    ✓ Mathematical expressions and formulas
    ✓ System configuration files (Docker, K8s, CI/CD)
    ✓ Research into MoE architectures at small scale
    ✓ Educational demonstrations of expert routing
    """)

    # --- 5. Generation examples by domain ---
    print("\n" + "─" * 70)
    print("  5. DOMAIN-SPECIFIC GENERATION SAMPLES")
    print("─" * 70)

    domain_prompts = {
        'MATH': "def gcd(a, b):\n    ",
        'CODE': "const App = () => {\n  const [",
        'HUMAN': "# Behavioral Pattern Analysis\n\n",
        'SYSTEM': "services:\n  web:\n    image: ",
        'RECON': "def reconstruct_document(fragments",
        'INFO': "# Knowledge Base Entry\n\ntitle: ",
    }

    for domain, prompt in domain_prompts.items():
        print(f"\n  [{domain}] prompt: {repr(prompt[:50])}")
        result = generate_text(model, sp, prompt,
                               max_tokens=120,
                               temperature=0.7)
        lines = result.split('\n')
        for line in lines[:8]:
            print(f"    {line}")
        if len(lines) > 8:
            print(f"    ... ({len(lines) - 8} more lines)")

    # --- 6. Speed benchmark ---
    print("\n" + "─" * 70)
    print("  6. INFERENCE SPEED")
    print("─" * 70)

    tokens_to_gen = 100
    prompt = "import torch\n"
    t0 = time.time()
    _ = generate_text(model, sp, prompt, max_tokens=tokens_to_gen, temperature=0.8)
    elapsed = time.time() - t0
    print(f"\n  Generated {tokens_to_gen} tokens in {elapsed:.2f}s")
    print(f"  Speed: {tokens_to_gen/elapsed:.1f} tokens/sec")

    print("\n" + "=" * 70)
    print("  End of Self-Description Report")
    print("=" * 70)


# ==================== Main ====================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='NautilusMoME Self-Description')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', default=TOKENIZER_MODEL,
                        help='Path to sentencepiece model')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Max tokens per generation')
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    if not os.path.exists(args.tokenizer):
        print(f"ERROR: Tokenizer not found: {args.tokenizer}")
        print("Run train_nautilus_mome.py first to train the tokenizer.")
        sys.exit(1)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    print(f"  Vocab: {sp.get_piece_size()} tokens")

    # Load model from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Run train_nautilus_mome.py first to train the model.")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')

    # Reconstruct model config from checkpoint
    # Try 'config' key first, then 'args' (training script stores args)
    model_cfg = ckpt.get('config', None)
    if model_cfg is None:
        # Training script stores argparse namespace as 'args'
        saved_args = ckpt.get('args', {})
        if hasattr(saved_args, '__dict__'):
            model_cfg = vars(saved_args)
        elif isinstance(saved_args, dict):
            model_cfg = saved_args
        else:
            model_cfg = {}

    model = NautilusMoME(
        vocab_size=model_cfg.get('vocab_size', sp.get_piece_size()),
        d_model=model_cfg.get('d_model', 128),
        n_layers=model_cfg.get('n_layers', 4),
        n_heads=model_cfg.get('n_heads', 4),
        block_size=model_cfg.get('block_size', 256),
        d_expert=model_cfg.get('d_expert', 128),
        n_experts=model_cfg.get('n_experts', 6),
        top_k=model_cfg.get('top_k', 2),
    )

    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    step = ckpt.get('step', '?')
    val_loss = ckpt.get('val_loss', None)
    ppl = math.exp(min(val_loss, 20)) if val_loss else '?'
    print(f"  Loaded model from step {step}, val_loss={val_loss}, PPL={ppl:.2f}" if val_loss else f"  Loaded model from step {step}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run self-description
    generate_self_description(model, sp,
                              temperature=args.temperature,
                              max_tokens=args.max_tokens)


if __name__ == '__main__':
    main()
