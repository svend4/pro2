"""
Domain-Locked Generation — фиксация доминирующего эксперта при генерации.

Проблема: после 15-20 токенов модель смешивает домены (код + русский + тесты).
Решение: определить доминирующего эксперта на промпте, усилить его при генерации.

Usage:
    from inference.domain_locked_generate import domain_locked_generate
    text = domain_locked_generate(model, sp, "def fibonacci(n):", max_tokens=100)
"""

import torch
import torch.nn.functional as F
import math


@torch.no_grad()
def _detect_dominant_from_prompt(model, sp, prompt):
    """Determine dominant expert from the FULL prompt routing, not just first tokens."""
    model.eval()
    tokens = sp.encode(prompt)
    if not tokens:
        return None, {}
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    _, _, info = model(idx)
    routing = info['routing'][0]  # (T, n_experts)
    # Weight later tokens more (they have more context)
    T = routing.size(0)
    weights = torch.linspace(0.5, 1.5, T, device=routing.device)
    weighted_routing = (routing * weights.unsqueeze(1)).mean(dim=0)
    names = model.EXPERT_NAMES[:model.n_experts]
    routing_dict = {names[i]: weighted_routing[i].item() for i in range(len(names))}
    dominant_idx = weighted_routing.argmax().item()
    return dominant_idx, routing_dict


@torch.no_grad()
def domain_locked_generate(
    model, sp, prompt,
    max_tokens=150,
    temperature=0.8,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.3,
    # Domain lock params
    lock_strength=1.8,        # multiplier for dominant expert
    suppress_strength=0.4,    # multiplier for non-dominant experts
    entropy_unlock=0.65,      # allow natural routing above this entropy
    lock_after_tokens=0,      # 0 = detect from prompt immediately
    # Coherence params
    coherence_window=10,      # routing history window
    coherence_penalty=0.3,    # penalty for routing shift
    # Mirostat params
    use_mirostat=False,
    mirostat_tau=5.0,         # target surprise (bits)
    mirostat_eta=0.1,         # learning rate for mu
):
    """Generate text with domain-locked expert routing.

    Detects dominant expert from the FULL prompt routing (not just first few tokens).
    Then amplifies that expert and suppresses others during generation.
    """
    model.eval()
    expert_names = model.EXPERT_NAMES[:model.n_experts]
    n_experts = model.n_experts

    tokens = sp.encode(prompt)
    if not tokens:
        tokens = [sp.bos_id() if sp.bos_id() != -1 else 1]

    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    generated_ids = list(tokens)

    # Detect dominant expert from the full prompt
    dominant_idx, prompt_routing = _detect_dominant_from_prompt(model, sp, prompt)

    # State for domain locking
    routing_history = []
    mirostat_mu = 2 * mirostat_tau  # initial mu for mirostat

    # Stats
    stats = {
        'dominant_expert': expert_names[dominant_idx] if dominant_idx is not None else None,
        'prompt_routing': prompt_routing,
        'lock_activations': 0,
        'unlock_activations': 0,
        'routing_shifts': [],
        'avg_coherence': 0.0,
    }

    # Sharpen router for generation: lower temperature → more decisive routing
    original_router_temp = getattr(model.router, 'temperature', 1.0)
    if dominant_idx is not None:
        model.router.temperature = max(0.3, original_router_temp * 0.5)

    # Temporarily boost dominant expert's gate scale
    dominant_name = expert_names[dominant_idx] if dominant_idx is not None else None
    original_gate_scale = None
    if dominant_name and dominant_name in model.experts:
        original_gate_scale = model.experts[dominant_name].gate_scale.data.clone()
        model.experts[dominant_name].gate_scale.data *= lock_strength

    # Temporarily suppress other experts' gate scales
    other_gate_scales = {}
    if dominant_idx is not None:
        for name in expert_names:
            if name != dominant_name and name in model.experts:
                other_gate_scales[name] = model.experts[name].gate_scale.data.clone()
                model.experts[name].gate_scale.data *= suppress_strength

    try:
      for step in range(max_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _, info = model(idx_cond)
        logits = logits[0, -1, :].clone()

        # Get current routing weights
        current_routing = info['routing'][0][-1].detach()  # (n_experts,)
        routing_history.append(current_routing)

        # Track domain lock activity
        if dominant_idx is not None:
            current_dominant = current_routing.argmax().item()
            if current_dominant == dominant_idx:
                stats['lock_activations'] += 1
            else:
                stats['unlock_activations'] += 1

        # Coherence: adaptive temperature based on routing consistency
        effective_temperature = temperature
        if len(routing_history) > coherence_window and coherence_penalty > 0:
            recent = torch.stack(routing_history[-coherence_window:])
            avg_recent = recent.mean(dim=0)
            cos_sim = F.cosine_similarity(
                current_routing.unsqueeze(0), avg_recent.unsqueeze(0)
            ).item()
            coherence = max(0.0, cos_sim)
            stats['routing_shifts'].append(1.0 - coherence)

            # Low coherence → lower temperature (more conservative)
            if coherence < 0.95:
                effective_temperature = temperature * (0.5 + 0.5 * coherence)

        # --- Repetition penalty ---
        recent_tokens = generated_ids[-60:]
        for tid in set(recent_tokens):
            count = recent_tokens.count(tid)
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty * (1 + 0.1 * count)
            else:
                logits[tid] *= repetition_penalty * (1 + 0.1 * count)

        # --- Sampling ---
        if use_mirostat:
            # Mirostat v2: adaptive temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            surprisals = -torch.log2(probs + 1e-10)

            # Find k where cumulative surprise exceeds mu
            k = 1
            for i in range(len(surprisals)):
                if surprisals[i].item() > mirostat_mu:
                    k = max(1, i)
                    break
            else:
                k = len(surprisals)

            # Sample from top-k
            top_probs = probs[:k]
            top_probs = top_probs / top_probs.sum()
            chosen = torch.multinomial(top_probs, 1)
            next_token_id = sorted_idx[chosen.item()].item()

            # Update mu
            observed_surprise = surprisals[chosen.item()].item()
            mirostat_mu -= mirostat_eta * (observed_surprise - mirostat_tau)
        else:
            # Standard nucleus sampling
            logits = logits / effective_temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                probs[sorted_idx[mask]] = 0.0

            probs = probs / probs.sum()
            next_token = torch.multinomial(probs, 1)
            next_token_id = next_token.item()

        # --- Update context ---
        if next_token_id == sp.eos_id() and sp.eos_id() != -1:
            break

        generated_ids.append(next_token_id)
        next_t = torch.tensor([[next_token_id]], dtype=torch.long)
        idx = torch.cat([idx, next_t], dim=1)

    finally:
        # Restore original model state
        model.router.temperature = original_router_temp
        if dominant_name and original_gate_scale is not None and dominant_name in model.experts:
            model.experts[dominant_name].gate_scale.data = original_gate_scale
        for name, orig_scale in other_gate_scales.items():
            if name in model.experts:
                model.experts[name].gate_scale.data = orig_scale

    # Compute stats
    if stats['routing_shifts']:
        stats['avg_coherence'] = 1.0 - min(1.0, sum(stats['routing_shifts']) / len(stats['routing_shifts']))

    generated_text = sp.decode(generated_ids[len(tokens):])
    return generated_text, stats


@torch.no_grad()
def generate_with_expert_bias(
    model, sp, prompt,
    expert_name,
    bias_strength=2.0,
    max_tokens=150,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.3,
):
    """Generate text with explicit expert bias.

    Forces the model to prefer a specific expert domain.
    Useful for controlled generation: "generate Python code" → bias CODE.
    """
    model.eval()
    expert_names = model.EXPERT_NAMES[:model.n_experts]

    if expert_name not in expert_names:
        raise ValueError(f"Unknown expert: {expert_name}. Available: {expert_names}")

    target_idx = expert_names.index(expert_name)

    tokens = sp.encode(prompt)
    if not tokens:
        tokens = [sp.bos_id() if sp.bos_id() != -1 else 1]

    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    generated_ids = list(tokens)

    for step in range(max_tokens):
        idx_cond = idx[:, -model.block_size:]

        # Hook into the router to bias expert selection
        # We modify router temperature to sharpen routing + bias
        original_temp = model.router.temperature
        model.router.temperature = max(0.3, original_temp * 0.5)  # sharper routing

        logits, _, info = model(idx_cond)

        model.router.temperature = original_temp  # restore

        logits = logits[0, -1, :].clone()

        # Repetition penalty
        recent = generated_ids[-60:]
        for tid in set(recent):
            count = recent.count(tid)
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty * (1 + 0.1 * count)
            else:
                logits[tid] *= repetition_penalty * (1 + 0.1 * count)

        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float('-inf')

        probs = F.softmax(logits, dim=-1)

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

    return sp.decode(generated_ids[len(tokens):])
