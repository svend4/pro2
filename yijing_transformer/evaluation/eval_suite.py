#!/usr/bin/env python3
"""
NautilusMoME Eval Suite — автоматическая оценка модели.

Измеряет все ключевые метрики из Conceptual Stage:
  1. PPL по доменам (комфорт + дискомфорт)
  2. Routing accuracy (правильность expert assignment)
  3. Coherence score (routing consistency в окне генерации)
  4. Code completion accuracy (top-1, top-5)
  5. Domain mixing rate (% токенов с routing switch)
  6. Generation quality (PPL сгенерированного текста)
  7. Speed benchmark

Usage:
    python -m evaluation.eval_suite
    python -m evaluation.eval_suite --compare domain_locked
    python -m evaluation.eval_suite --json results.json
"""

import sys
import os
import math
import time
import json
from collections import defaultdict

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
# Test Prompts
# ═══════════════════════════════════════════════════════

COMFORT_ZONE = {
    "Python pytest": "def test_calculation():\n    result = calculate(10, 20)\n    assert result == 30",
    "Python class": "class DataProcessor:\n    def __init__(self, path):\n        self.path = path",
    "React component": "const Button = ({ onClick, children }) => (\n  <button onClick={onClick}>{children}</button>",
    "Import statements": "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F",
    "Russian docs": "# Архитектура\n\nМодуль отвечает за обработку входных данных",
    "Markdown structure": "## Features\n\n- Fast inference\n- Modular design\n- Expert routing",
    "Docker/config": "services:\n  web:\n    image: nginx:latest\n    ports:\n      - 8080:80",
    "Math formula": "def softmax(x):\n    exp_x = np.exp(x - np.max(x))\n    return exp_x / exp_x.sum()",
}

DISCOMFORT_ZONE = {
    "Medical": "The patient presents with acute myocardial infarction requiring emergent percutaneous coronary intervention",
    "Legal": "Whereas the party of the first part hereby agrees to indemnify and hold harmless the party of the second part",
    "Poetry": "Shall I compare thee to a summer's day? Thou art more lovely and more temperate",
    "Chemistry": "The reaction between sodium hydroxide and hydrochloric acid produces sodium chloride and water: NaOH + HCl",
    "Philosophy": "The categorical imperative formulated by Kant states that one should act only according to that maxim",
    "News": "The Federal Reserve announced today that interest rates will remain unchanged at 5.25% following the FOMC meeting",
    "Casual": "I went to the grocery store yesterday and bought some milk, eggs, bread, and a bag of apples",
    "Nonsense": "Blorkle sniffnax quozzle the wibbledy frang, for the pompitous of zingleberry was upon them",
}

CODE_COMPLETION = [
    ("import torch.nn as ", " nn", ["nn", "Module"]),
    ("def __init__(self, ", None, None),  # any valid continuation
    ("assert isinstance(result, ", "list", ["list", "dict", "float", "int", "str", "tuple"]),
    ("if __name__ == ", '"', ['"', "'"]),
    ("from collections import ", "defaultdict", ["defaultdict", "Counter", "OrderedDict", "namedtuple"]),
    ("const [data, setData] = useState(", None, None),
    ("except Exception as e:\n    ", "raise", ["raise", "print", "logger", "log"]),
    ("return self.", None, None),
    ("for i in range(", None, None),
    ("class Config:\n    def __init__(self):\n        self.", "config", None),
]

ROUTING_TESTS = {
    "CODE": [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "const [state, setState] = useState(null); useEffect(() => {",
        "import React from 'react'; export default function App() {",
    ],
    "SYSTEM": [
        "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id",
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:",
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .",
    ],
    "RECON": [
        "# Модуль интеграции данных\n\nЭтот компонент отвечает за",
        "# Описание архитектуры\n\nСистема состоит из нескольких модулей",
        "Результаты тестирования показали, что алгоритм работает",
    ],
    "MATH": [
        "The cross-entropy loss is defined as L = -sum y_i log(p_i)",
        "def softmax(x): return np.exp(x) / np.sum(np.exp(x))",
        "The gradient of the loss function with respect to weights",
    ],
}

GENERATION_PROMPTS = [
    ("Python function", "def merge_sorted_lists(a, b):\n    "),
    ("Pytest", "@pytest.mark.parametrize('input,expected', [\n    "),
    ("React hook", "function useDebounce(value, delay) {\n  const ["),
    ("Torch model", "class TransformerBlock(nn.Module):\n    def __init__(self, d_model):\n        super().__init__()\n        self."),
    ("Russian text", "# Описание модуля\n\nЭтот компонент предназначен для "),
    ("Config file", "services:\n  api:\n    build: .\n    ports:\n      - "),
]


# ═══════════════════════════════════════════════════════
# Metric Functions
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def compute_perplexity(model, sp, text, max_len=256):
    """Compute perplexity for given text."""
    model.eval()
    tokens = sp.encode(text)
    if len(tokens) < 2:
        return float('inf')
    tokens = tokens[:max_len]
    idx = torch.tensor([tokens], dtype=torch.long)
    logits, _, _ = model(idx)
    logits = logits[:, :-1, :]
    targets = idx[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return math.exp(min(loss.item(), 20))


@torch.no_grad()
def get_top_predictions(model, sp, prompt, top_n=5):
    """Get model's top-N predicted next tokens."""
    model.eval()
    tokens = sp.encode(prompt)
    if not tokens:
        return []
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    logits, _, _ = model(idx)
    logits = logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, min(top_n, probs.size(-1)))
    return [(sp.decode([top_ids[i].item()]).strip(), top_probs[i].item()) for i in range(len(top_ids))]


@torch.no_grad()
def get_routing(model, sp, text):
    """Get average routing weights for text."""
    model.eval()
    tokens = sp.encode(text)
    if not tokens:
        return {}
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    _, _, info = model(idx)
    routing = info['routing'][0]  # (T, n_experts)
    avg = routing.mean(dim=0)
    names = model.EXPERT_NAMES[:model.n_experts]
    return {names[i]: avg[i].item() for i in range(len(names))}


@torch.no_grad()
def measure_domain_mixing(model, sp, prompt, max_tokens=100, temperature=0.7):
    """Measure domain mixing during generation (% of tokens with routing switch)."""
    model.eval()
    tokens = sp.encode(prompt)
    if not tokens:
        return 0.0, 0

    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    generated_ids = list(tokens)
    prev_dominant = None
    switches = 0
    total = 0

    for _ in range(max_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _, info = model(idx_cond)
        logits = logits[0, -1, :].clone()

        routing = info['routing'][0][-1]
        current_dominant = routing.argmax().item()

        if prev_dominant is not None and current_dominant != prev_dominant:
            switches += 1
        prev_dominant = current_dominant
        total += 1

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        token_id = next_token.item()

        if token_id == sp.eos_id() and sp.eos_id() != -1:
            break

        generated_ids.append(token_id)
        idx = torch.cat([idx, next_token.unsqueeze(0)], dim=1)

    mixing_rate = switches / max(total, 1)
    return mixing_rate, total


@torch.no_grad()
def measure_routing_confidence(model, sp, text):
    """Measure routing confidence (how decisively the router selects experts)."""
    model.eval()
    tokens = sp.encode(text)
    if not tokens:
        return 0.0
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    _, _, info = model(idx)
    routing = info['routing'][0]  # (T, n_experts)

    # Confidence = 1 - normalized_entropy
    routing_probs = F.softmax(routing, dim=-1)
    entropy = -(routing_probs * (routing_probs + 1e-10).log()).sum(dim=-1)
    max_ent = math.log(model.n_experts)
    confidence = 1.0 - entropy.mean().item() / max_ent
    return confidence


# ═══════════════════════════════════════════════════════
# Eval Runner
# ═══════════════════════════════════════════════════════

def run_eval(model, sp, use_domain_locked=False, verbose=True):
    """Run full evaluation suite. Returns metrics dict."""
    results = {}

    def log(msg):
        if verbose:
            print(msg)

    # ── 1. PPL by Domain ──
    log("\n" + "═" * 60)
    log("  1. PERPLEXITY BY DOMAIN")
    log("═" * 60)

    comfort_ppls = {}
    for name, text in COMFORT_ZONE.items():
        ppl = compute_perplexity(model, sp, text)
        comfort_ppls[name] = ppl
        status = "+" if ppl < 15 else "~" if ppl < 30 else "-"
        log(f"  [{status}] PPL={ppl:6.1f}  {name}")

    discomfort_ppls = {}
    for name, text in DISCOMFORT_ZONE.items():
        ppl = compute_perplexity(model, sp, text)
        discomfort_ppls[name] = ppl
        log(f"  [-] PPL={ppl:6.1f}  {name}")

    avg_comfort = sum(comfort_ppls.values()) / len(comfort_ppls)
    avg_discomfort = sum(discomfort_ppls.values()) / len(discomfort_ppls)
    ppl_gap = avg_discomfort / avg_comfort if avg_comfort > 0 else 0

    results['ppl_comfort'] = avg_comfort
    results['ppl_discomfort'] = avg_discomfort
    results['ppl_gap'] = ppl_gap
    results['ppl_by_domain'] = {**comfort_ppls, **discomfort_ppls}

    log(f"\n  Avg comfort:    {avg_comfort:.1f}")
    log(f"  Avg discomfort: {avg_discomfort:.1f}")
    log(f"  PPL gap:        {ppl_gap:.1f}x")

    # ── 2. Routing Accuracy ──
    log("\n" + "═" * 60)
    log("  2. ROUTING ACCURACY")
    log("═" * 60)

    correct = 0
    total = 0
    for expected_expert, prompts in ROUTING_TESTS.items():
        for prompt in prompts:
            routing = get_routing(model, sp, prompt)
            top_expert = max(routing, key=routing.get)
            is_correct = top_expert == expected_expert
            correct += int(is_correct)
            total += 1
            mark = "+" if is_correct else "-"
            log(f"  [{mark}] Expected {expected_expert:6s}, got {top_expert:6s}  ({prompt[:50]})")

    routing_accuracy = correct / total if total > 0 else 0
    results['routing_accuracy'] = routing_accuracy
    log(f"\n  Routing accuracy: {correct}/{total} = {routing_accuracy*100:.0f}%")

    # ── 3. Routing Confidence ──
    log("\n" + "═" * 60)
    log("  3. ROUTING CONFIDENCE")
    log("═" * 60)

    confidences = []
    all_texts = list(COMFORT_ZONE.values()) + [p for ps in ROUTING_TESTS.values() for p in ps]
    for text in all_texts:
        conf = measure_routing_confidence(model, sp, text)
        confidences.append(conf)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    results['routing_confidence'] = avg_confidence
    log(f"  Avg routing confidence: {avg_confidence*100:.1f}%")

    # ── 4. Code Completion ──
    log("\n" + "═" * 60)
    log("  4. CODE COMPLETION ACCURACY")
    log("═" * 60)

    top1_correct = 0
    top5_correct = 0
    total_testable = 0

    for prompt, expected, alternatives in CODE_COMPLETION:
        preds = get_top_predictions(model, sp, prompt, top_n=5)
        pred_tokens = [t for t, _ in preds]

        if expected is None:
            continue  # skip untestable

        total_testable += 1
        valid_answers = [expected] + (alternatives or [])

        t1_hit = any(pred_tokens[0].strip().startswith(a) for a in valid_answers) if pred_tokens else False
        t5_hit = any(any(pt.strip().startswith(a) for a in valid_answers) for pt in pred_tokens)

        top1_correct += int(t1_hit)
        top5_correct += int(t5_hit)

        mark = "+" if t1_hit else ("~" if t5_hit else "-")
        log(f"  [{mark}] '{prompt.strip()}' → '{pred_tokens[0] if pred_tokens else '?'}' (expected: '{expected}')")

    top1_acc = top1_correct / total_testable if total_testable > 0 else 0
    top5_acc = top5_correct / total_testable if total_testable > 0 else 0
    results['code_completion_top1'] = top1_acc
    results['code_completion_top5'] = top5_acc
    log(f"\n  Top-1: {top1_correct}/{total_testable} = {top1_acc*100:.0f}%")
    log(f"  Top-5: {top5_correct}/{total_testable} = {top5_acc*100:.0f}%")

    # ── 5. Domain Mixing ──
    log("\n" + "═" * 60)
    log("  5. DOMAIN MIXING RATE")
    log("═" * 60)

    mixing_rates = []
    for name, prompt in GENERATION_PROMPTS:
        rate, n_tokens = measure_domain_mixing(model, sp, prompt, max_tokens=80)
        mixing_rates.append(rate)
        log(f"  {rate*100:5.1f}% mixing  ({n_tokens} tokens)  {name}")

    avg_mixing = sum(mixing_rates) / len(mixing_rates) if mixing_rates else 0
    results['domain_mixing_rate'] = avg_mixing
    log(f"\n  Avg domain mixing: {avg_mixing*100:.1f}%")

    # ── 6. Generation Quality ──
    log("\n" + "═" * 60)
    log("  6. GENERATION QUALITY")
    log("═" * 60)

    gen_ppls_standard = []
    gen_ppls_locked = []

    for name, prompt in GENERATION_PROMPTS:
        # Standard generation
        text_std = standard_generate(model, sp, prompt, max_tokens=80, temperature=0.7)
        ppl_std = compute_perplexity(model, sp, prompt + text_std)
        gen_ppls_standard.append(ppl_std)

        if use_domain_locked:
            text_dl, stats = domain_locked_generate(model, sp, prompt, max_tokens=80, temperature=0.7)
            ppl_dl = compute_perplexity(model, sp, prompt + text_dl)
            gen_ppls_locked.append(ppl_dl)
            log(f"  {name:20s}  std={ppl_std:6.1f}  locked={ppl_dl:6.1f}  dominant={stats.get('dominant_expert','?')}")
        else:
            log(f"  {name:20s}  PPL={ppl_std:6.1f}")

    avg_gen_ppl = sum(gen_ppls_standard) / len(gen_ppls_standard)
    results['gen_ppl_standard'] = avg_gen_ppl

    if gen_ppls_locked:
        avg_locked_ppl = sum(gen_ppls_locked) / len(gen_ppls_locked)
        results['gen_ppl_locked'] = avg_locked_ppl
        improvement = (avg_gen_ppl - avg_locked_ppl) / avg_gen_ppl * 100
        log(f"\n  Avg standard PPL:      {avg_gen_ppl:.1f}")
        log(f"  Avg domain-locked PPL: {avg_locked_ppl:.1f}")
        log(f"  Improvement:           {improvement:+.1f}%")
    else:
        log(f"\n  Avg generation PPL: {avg_gen_ppl:.1f}")

    # ── 7. Speed Benchmark ──
    log("\n" + "═" * 60)
    log("  7. SPEED BENCHMARK")
    log("═" * 60)

    prompt = "import torch\nimport torch.nn as nn\n"
    n_tokens = 100

    # Standard
    t0 = time.time()
    _ = standard_generate(model, sp, prompt, max_tokens=n_tokens, temperature=0.7)
    t_std = time.time() - t0
    speed_std = n_tokens / t_std

    # Domain-locked
    t0 = time.time()
    _ = domain_locked_generate(model, sp, prompt, max_tokens=n_tokens, temperature=0.7)
    t_dl = time.time() - t0
    speed_dl = n_tokens / t_dl

    results['speed_standard'] = speed_std
    results['speed_locked'] = speed_dl
    log(f"  Standard:      {speed_std:.0f} tok/s ({t_std:.2f}s)")
    log(f"  Domain-locked: {speed_dl:.0f} tok/s ({t_dl:.2f}s)")

    # ── Summary ──
    log("\n" + "═" * 60)
    log("  SUMMARY")
    log("═" * 60)

    targets = {
        'ppl_comfort':        ('< 15',   lambda v: v < 15),
        'ppl_gap':            ('> 15x',  lambda v: v > 15),
        'routing_accuracy':   ('> 70%',  lambda v: v > 0.7),
        'routing_confidence': ('> 15%',  lambda v: v > 0.15),
        'code_completion_top1': ('> 55%', lambda v: v > 0.55),
        'domain_mixing_rate': ('< 20%',  lambda v: v < 0.2),
        'speed_standard':     ('> 50',   lambda v: v > 50),
    }

    for key, (target_str, check_fn) in targets.items():
        val = results.get(key, 0)
        passed = check_fn(val)
        mark = "PASS" if passed else "FAIL"

        if 'accuracy' in key or 'confidence' in key or 'rate' in key or 'completion' in key:
            val_str = f"{val*100:.1f}%"
        elif 'speed' in key:
            val_str = f"{val:.0f} tok/s"
        else:
            val_str = f"{val:.1f}"

        log(f"  [{mark}] {key:25s}  {val_str:>10s}  (target: {target_str})")

    return results


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def load_model():
    """Load model and tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)

    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location='cpu')
    model_cfg = ckpt.get('args', {})
    if hasattr(model_cfg, '__dict__'):
        model_cfg = vars(model_cfg)

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: step {step}, {n_params:,} params")

    return model, sp


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NautilusMoME Eval Suite')
    parser.add_argument('--compare', action='store_true', help='Compare standard vs domain-locked')
    parser.add_argument('--json', type=str, default=None, help='Save results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    model, sp = load_model()
    results = run_eval(model, sp, use_domain_locked=args.compare, verbose=not args.quiet)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.json}")


if __name__ == '__main__':
    main()
