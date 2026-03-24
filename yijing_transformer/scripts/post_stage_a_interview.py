#!/usr/bin/env python3
"""
Post-Stage A Interview — тестирование модели после изменений.

Сравнивает standard vs domain-locked генерацию по 10 вопросам.
Показывает: текст, routing, PPL, archetype, coherence.

Usage:
    python scripts/post_stage_a_interview.py
"""

import sys
import os
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
)
from inference.generate import generate as standard_generate
from inference.domain_locked_generate import domain_locked_generate


# ═══════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def analyze_response(model, sp, text):
    """Full analysis of a text: routing, PPL, archetype, entropy."""
    model.eval()
    tokens = sp.encode(text)
    if len(tokens) < 2:
        return {}
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    logits, _, info = model(idx)

    # PPL
    shift_logits = logits[:, :-1, :]
    targets = idx[:, 1:]
    loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), targets.reshape(-1))
    ppl = math.exp(min(loss.item(), 20))

    # Routing
    routing = info['routing'][0]  # (T, n_experts)
    avg_routing = routing.mean(dim=0)
    names = model.EXPERT_NAMES[:model.n_experts]
    routing_dict = {names[i]: avg_routing[i].item() for i in range(len(names))}

    # Routing entropy (confidence)
    routing_probs = F.softmax(routing, dim=-1)
    entropy = -(routing_probs * (routing_probs + 1e-10).log()).sum(dim=-1)
    confidence = 1.0 - entropy.mean().item() / math.log(model.n_experts)

    # Domain switches
    dominant_per_token = routing.argmax(dim=-1)
    switches = (dominant_per_token[1:] != dominant_per_token[:-1]).sum().item()
    mixing = switches / max(len(tokens) - 1, 1)

    # Archetype
    archetype_info = info.get('archetype', {})
    arch_name = archetype_info.get('top_name', '?')
    arch_code = archetype_info.get('top_archetype', '?')
    arch_prob = archetype_info.get('top_prob', 0)

    # SYNTH
    synth_info = info.get('synth', {})
    synth_active = synth_info.get('activation_frac', 0)

    # Twilight
    twilight_info = info.get('twilight', {})
    twilight_strength = twilight_info.get('twilight_strength', 0)

    return {
        'ppl': ppl,
        'routing': routing_dict,
        'top_expert': max(routing_dict, key=routing_dict.get),
        'confidence': confidence,
        'mixing': mixing,
        'archetype': f"{arch_code} ({arch_name})",
        'arch_prob': arch_prob,
        'synth_active': synth_active,
        'twilight': twilight_strength,
    }


def bar(val, width=25, max_val=0.25):
    filled = int(min(val / max_val, 1.0) * width)
    return '█' * filled + '░' * (width - filled)


def print_routing(routing_dict, prefix="    "):
    sorted_r = sorted(routing_dict.items(), key=lambda x: -x[1])
    for name, val in sorted_r:
        print(f"{prefix}{name:6s} {bar(val)} {val:.4f}")


# ═══════════════════════════════════════════════════════
# Interview Questions
# ═══════════════════════════════════════════════════════

INTERVIEW = [
    {
        "id": 1,
        "title": "Code: Python function completion",
        "prompt": "def merge_sorted_lists(list_a, list_b):\n    ",
        "expect_expert": "CODE",
        "max_tokens": 120,
    },
    {
        "id": 2,
        "title": "Code: React component",
        "prompt": "import React, { useState, useEffect } from 'react';\n\nconst UserProfile = ({ userId }) => {\n  const [user, setUser] = useState(null);\n\n  useEffect(() => {\n    ",
        "expect_expert": "CODE",
        "max_tokens": 120,
    },
    {
        "id": 3,
        "title": "Code: PyTorch module",
        "prompt": "class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.",
        "expect_expert": "CODE",
        "max_tokens": 120,
    },
    {
        "id": 4,
        "title": "System: Docker compose",
        "prompt": "# Docker Compose for microservices architecture\nversion: '3.8'\nservices:\n  api:\n    build: ./api\n    ports:\n      - ",
        "expect_expert": "SYSTEM",
        "max_tokens": 100,
    },
    {
        "id": 5,
        "title": "System: SQL query",
        "prompt": "-- Find top 10 customers by total order value\nSELECT \n    c.customer_name,\n    ",
        "expect_expert": "SYSTEM",
        "max_tokens": 80,
    },
    {
        "id": 6,
        "title": "Russian: Technical documentation",
        "prompt": "# Архитектура системы\n\nОсновной модуль обработки данных состоит из ",
        "expect_expert": "RECON",
        "max_tokens": 100,
    },
    {
        "id": 7,
        "title": "Russian: Project description",
        "prompt": "## Описание проекта\n\nЭтот проект реализует нейронную сеть с ",
        "expect_expert": "RECON",
        "max_tokens": 100,
    },
    {
        "id": 8,
        "title": "Math: Loss function",
        "prompt": "def compute_loss(predictions, targets, class_weights=None):\n    \"\"\"Weighted cross-entropy loss with label smoothing.\"\"\"\n    ",
        "expect_expert": "MATH",
        "max_tokens": 100,
    },
    {
        "id": 9,
        "title": "Mixed: README with code",
        "prompt": "# NautilusMoME\n\nA Mixture of Micro-Experts architecture for multi-domain text generation.\n\n## Quick Start\n\n```python\n",
        "expect_expert": "RECON",
        "max_tokens": 120,
    },
    {
        "id": 10,
        "title": "Free generation: continue story",
        "prompt": "The transformer model looked at the input tokens and began to think. Each expert in its mixture whispered a different",
        "expect_expert": None,
        "max_tokens": 100,
    },
]


# ═══════════════════════════════════════════════════════
# Main Interview Loop
# ═══════════════════════════════════════════════════════

def run_interview(model, sp):
    print("\n" + "═" * 70)
    print("  POST-STAGE A INTERVIEW — NautilusMoME (step 5000, 3M params)")
    print("  Standard vs Domain-Locked Generation")
    print("═" * 70)

    results = []

    for q in INTERVIEW:
        print(f"\n{'─' * 70}")
        print(f"  Q{q['id']}: {q['title']}")
        if q['expect_expert']:
            print(f"  Expected expert: {q['expect_expert']}")
        print(f"{'─' * 70}")
        print(f"  PROMPT: {q['prompt'][:80]}{'...' if len(q['prompt']) > 80 else ''}")

        # === Standard Generation ===
        t0 = time.time()
        text_std = standard_generate(model, sp, q['prompt'],
                                     max_tokens=q['max_tokens'],
                                     temperature=0.75, top_k=40, top_p=0.9)
        t_std = time.time() - t0
        analysis_std = analyze_response(model, sp, q['prompt'] + text_std)

        print(f"\n  ┌─ STANDARD (t={t_std:.2f}s)")
        for line in text_std.strip().split('\n')[:8]:
            print(f"  │ {line}")
        if len(text_std.strip().split('\n')) > 8:
            print(f"  │ ... ({len(text_std.strip().split(chr(10)))} lines total)")
        print(f"  │")
        print(f"  │ PPL={analysis_std['ppl']:.1f}  top={analysis_std['top_expert']}  "
              f"conf={analysis_std['confidence']*100:.1f}%  "
              f"mix={analysis_std['mixing']*100:.0f}%  "
              f"arch={analysis_std['archetype']}")
        print_routing(analysis_std['routing'], prefix="  │ ")
        print(f"  └─")

        # === Domain-Locked Generation ===
        t0 = time.time()
        text_dl, dl_stats = domain_locked_generate(model, sp, q['prompt'],
                                                    max_tokens=q['max_tokens'],
                                                    temperature=0.75, top_k=40, top_p=0.9)
        t_dl = time.time() - t0
        analysis_dl = analyze_response(model, sp, q['prompt'] + text_dl)

        print(f"\n  ┌─ DOMAIN-LOCKED (t={t_dl:.2f}s, dominant={dl_stats['dominant_expert']})")
        for line in text_dl.strip().split('\n')[:8]:
            print(f"  │ {line}")
        if len(text_dl.strip().split('\n')) > 8:
            print(f"  │ ... ({len(text_dl.strip().split(chr(10)))} lines total)")
        print(f"  │")
        print(f"  │ PPL={analysis_dl['ppl']:.1f}  top={analysis_dl['top_expert']}  "
              f"conf={analysis_dl['confidence']*100:.1f}%  "
              f"mix={analysis_dl['mixing']*100:.0f}%  "
              f"arch={analysis_dl['archetype']}")
        print(f"  │ lock={dl_stats['lock_activations']}  unlock={dl_stats['unlock_activations']}  "
              f"coherence={dl_stats['avg_coherence']:.2f}")
        print_routing(analysis_dl['routing'], prefix="  │ ")
        print(f"  └─")

        # === Comparison ===
        ppl_delta = analysis_std['ppl'] - analysis_dl['ppl']
        mix_delta = analysis_std['mixing'] - analysis_dl['mixing']
        correct_std = q['expect_expert'] and analysis_std['top_expert'] == q['expect_expert']
        correct_dl = q['expect_expert'] and analysis_dl['top_expert'] == q['expect_expert']

        verdict = "?"
        if analysis_dl['ppl'] < analysis_std['ppl'] and mix_delta >= 0:
            verdict = "LOCKED WINS"
        elif analysis_std['ppl'] < analysis_dl['ppl'] and mix_delta <= 0:
            verdict = "STANDARD WINS"
        elif abs(ppl_delta) < 1.0:
            verdict = "TIE"
        else:
            verdict = "MIXED"

        print(f"\n  VERDICT: {verdict}  (PPL Δ={ppl_delta:+.1f}, mixing Δ={mix_delta*100:+.0f}%)")

        results.append({
            'id': q['id'],
            'title': q['title'],
            'ppl_std': analysis_std['ppl'],
            'ppl_dl': analysis_dl['ppl'],
            'ppl_delta': ppl_delta,
            'mix_std': analysis_std['mixing'],
            'mix_dl': analysis_dl['mixing'],
            'top_std': analysis_std['top_expert'],
            'top_dl': analysis_dl['top_expert'],
            'expected': q['expect_expert'],
            'correct_std': correct_std,
            'correct_dl': correct_dl,
            'verdict': verdict,
        })

    # === Summary ===
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")

    ppls_std = [r['ppl_std'] for r in results]
    ppls_dl = [r['ppl_dl'] for r in results]
    mixes_std = [r['mix_std'] for r in results]
    mixes_dl = [r['mix_dl'] for r in results]

    print(f"\n  {'Question':<35s} {'PPL std':>8s} {'PPL lock':>8s} {'Δ':>7s} {'Mix std':>8s} {'Mix lock':>8s} {'Verdict'}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*15}")
    for r in results:
        print(f"  {r['title'][:35]:<35s} {r['ppl_std']:8.1f} {r['ppl_dl']:8.1f} {r['ppl_delta']:+7.1f} "
              f"{r['mix_std']*100:7.0f}% {r['mix_dl']*100:7.0f}% {r['verdict']}")

    print(f"\n  Avg PPL standard:      {sum(ppls_std)/len(ppls_std):.1f}")
    print(f"  Avg PPL domain-locked: {sum(ppls_dl)/len(ppls_dl):.1f}")
    print(f"  Avg mixing standard:   {sum(mixes_std)/len(mixes_std)*100:.0f}%")
    print(f"  Avg mixing locked:     {sum(mixes_dl)/len(mixes_dl)*100:.0f}%")

    locked_wins = sum(1 for r in results if r['verdict'] == 'LOCKED WINS')
    std_wins = sum(1 for r in results if r['verdict'] == 'STANDARD WINS')
    ties = sum(1 for r in results if r['verdict'] in ('TIE', 'MIXED'))
    print(f"\n  Score: Locked {locked_wins} — Standard {std_wins} — Tie/Mixed {ties}")

    routing_correct_std = sum(1 for r in results if r['correct_std'])
    routing_correct_dl = sum(1 for r in results if r['correct_dl'])
    testable = sum(1 for r in results if r['expected'])
    print(f"  Routing accuracy: standard {routing_correct_std}/{testable}, locked {routing_correct_dl}/{testable}")


def load_model():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location='cpu')
    args = ckpt.get('args', {})
    if hasattr(args, '__dict__'):
        args = vars(args)
    model = NautilusMoME(
        vocab_size=args.get('vocab_size', sp.get_piece_size()),
        d_model=args.get('d_model', 128),
        n_layers=args.get('n_layers', 4),
        n_heads=args.get('n_heads', 4),
        block_size=args.get('block_size', 256),
        d_expert=args.get('d_expert', 128),
        n_experts=args.get('n_experts', 6),
        top_k=args.get('top_k', 2),
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    step = ckpt.get('step', '?')
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: step {step}, {n_params:,} params")
    return model, sp


if __name__ == '__main__':
    model, sp = load_model()
    run_interview(model, sp)
