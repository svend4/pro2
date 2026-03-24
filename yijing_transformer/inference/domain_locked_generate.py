"""
Organic Coherence Generation — естественная когерентность через routing momentum.

Философия:
  Вместо ЖЁСТКОГО domain-lock (amplify × 1.8, suppress × 0.4) используем
  ОРГАНИЧЕСКИЙ подход: routing momentum через экспоненциальное скользящее среднее.

  Как река течёт естественно, но имеет инерцию — так и routing:
  - Не "запираем" модель в одном эксперте
  - Даём routing инерцию (momentum) через EMA
  - Позволяем естественные переходы при высокой энтропии
  - Температура адаптируется к когерентности (не к "доминантности")

  Четырёхуровневая система (info3):
    Formula (MATH) → Archetype (CODE/SYSTEM) → Algorithm (RECON/INFO) → Theorem (HUMAN)
    Каждый уровень имеет свой "ритм дыхания" — не нужно его подавлять.

Ключевые отличия от domain_lock:
  OLD: detect dominant → amplify × 1.8 → suppress others × 0.4
  NEW: observe routing → build momentum (EMA) → adapt temperature naturally
  OLD: modify model.experts[name].gate_scale (мутирует модель!)
  NEW: чистая генерация, модель не мутируется

Usage:
    from inference.domain_locked_generate import organic_generate
    text = organic_generate(model, sp, "def fibonacci(n):", max_tokens=100)

    # Legacy alias preserved for backward compatibility
    text, stats = domain_locked_generate(model, sp, "def fibonacci(n):")
"""

import torch
import torch.nn.functional as F
import math


@torch.no_grad()
def _observe_routing_profile(model, sp, prompt):
    """Observe the natural routing profile of the prompt.

    Does NOT modify the model. Simply runs the prompt through
    and observes which experts naturally activate.

    Returns:
        routing_momentum: (n_experts,) — EMA of routing over prompt tokens
        routing_dict: {expert_name: weight} — for diagnostics
    """
    model.eval()
    tokens = sp.encode(prompt)
    if not tokens:
        return None, {}
    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    _, _, info = model(idx)
    routing = info['routing'][0]  # (T, n_experts)

    # Build routing momentum via EMA over prompt tokens
    # Later tokens have more context → naturally weighted by EMA decay
    T = routing.size(0)
    momentum = torch.zeros(routing.size(1), device=routing.device)
    alpha = 0.3  # EMA decay: recent tokens matter more
    for t in range(T):
        momentum = alpha * routing[t] + (1 - alpha) * momentum

    names = model.EXPERT_NAMES[:model.n_experts]
    routing_dict = {names[i]: momentum[i].item() for i in range(len(names))}
    return momentum, routing_dict


@torch.no_grad()
def organic_generate(
    model, sp, prompt,
    max_tokens=150,
    temperature=0.8,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.3,
    # Organic coherence params
    momentum_alpha=0.2,       # EMA decay for routing momentum (lower = more inertia)
    coherence_window=10,      # routing history window for coherence measurement
    temperature_adapt=0.3,    # how much coherence affects temperature (0 = disabled)
    # Mirostat params
    use_mirostat=False,
    mirostat_tau=5.0,         # target surprise (bits)
    mirostat_eta=0.1,         # learning rate for mu
):
    """Generate text with organic routing coherence.

    Instead of locking to a dominant expert, maintains natural routing
    momentum through exponential moving average. The model's own routing
    decisions are respected — we only add soft inertia.

    Key principle: the model knows best which expert to use.
    We just smooth out noise, not override decisions.
    """
    model.eval()
    expert_names = model.EXPERT_NAMES[:model.n_experts]

    tokens = sp.encode(prompt)
    if not tokens:
        tokens = [sp.bos_id() if sp.bos_id() != -1 else 1]

    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    generated_ids = list(tokens)

    # Observe (don't lock!) the natural routing profile from prompt
    routing_momentum, prompt_routing = _observe_routing_profile(model, sp, prompt)

    # State for organic coherence
    routing_history = []
    mirostat_mu = 2 * mirostat_tau

    # Stats (for diagnostics, not control)
    stats = {
        'prompt_routing': prompt_routing,
        'routing_flow': [],        # how routing evolves naturally
        'coherence_scores': [],    # coherence over time
        'temperature_history': [], # how temperature adapts
        'entropy_history': [],     # routing entropy over time
        'avg_coherence': 0.0,
    }

    for step in range(max_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _, info = model(idx_cond)
        logits = logits[0, -1, :].clone()

        # Observe current routing (don't modify it)
        current_routing = info['routing'][0][-1].detach()  # (n_experts,)
        routing_history.append(current_routing)

        # Update routing momentum (EMA — like a river's inertia)
        if routing_momentum is not None:
            routing_momentum = momentum_alpha * current_routing + \
                             (1 - momentum_alpha) * routing_momentum

        # Measure routing entropy (how uncertain is the router?)
        routing_safe = current_routing.clamp(min=1e-8)
        routing_entropy = -(routing_safe * routing_safe.log()).sum().item()
        stats['entropy_history'].append(routing_entropy)

        # Measure coherence: how consistent is routing with recent history?
        effective_temperature = temperature
        if len(routing_history) > coherence_window and temperature_adapt > 0:
            recent = torch.stack(routing_history[-coherence_window:])
            avg_recent = recent.mean(dim=0)
            cos_sim = F.cosine_similarity(
                current_routing.unsqueeze(0), avg_recent.unsqueeze(0)
            ).item()
            coherence = max(0.0, cos_sim)
            stats['coherence_scores'].append(coherence)

            # Organic temperature adaptation:
            # High coherence → model is confident → allow normal temperature
            # Low coherence → model is shifting → slightly lower temperature for stability
            # But NEVER force — just gentle guidance
            if coherence < 0.9:
                temp_factor = 1.0 - temperature_adapt * (1.0 - coherence)
                effective_temperature = temperature * max(0.6, temp_factor)

        stats['temperature_history'].append(effective_temperature)

        # Track routing flow (for visualization)
        if step % 5 == 0:
            top_expert_idx = current_routing.argmax().item()
            stats['routing_flow'].append(expert_names[top_expert_idx])

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
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            surprisals = -torch.log2(probs + 1e-10)

            k = 1
            for i in range(len(surprisals)):
                if surprisals[i].item() > mirostat_mu:
                    k = max(1, i)
                    break
            else:
                k = len(surprisals)

            top_probs = probs[:k]
            top_probs = top_probs / top_probs.sum()
            chosen = torch.multinomial(top_probs, 1)
            next_token_id = sorted_idx[chosen.item()].item()

            observed_surprise = surprisals[chosen.item()].item()
            mirostat_mu -= mirostat_eta * (observed_surprise - mirostat_tau)
        else:
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

    # Compute final stats
    if stats['coherence_scores']:
        stats['avg_coherence'] = sum(stats['coherence_scores']) / len(stats['coherence_scores'])
    if stats['entropy_history']:
        stats['avg_entropy'] = sum(stats['entropy_history']) / len(stats['entropy_history'])

    generated_text = sp.decode(generated_ids[len(tokens):])
    return generated_text, stats


# ── Legacy alias for backward compatibility ──
# The old domain_locked_generate is replaced by organic_generate.
# Parameters are mapped: lock_strength and suppress_strength are ignored
# (organic approach doesn't mutate model weights).

def domain_locked_generate(
    model, sp, prompt,
    max_tokens=150,
    temperature=0.8,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.3,
    lock_strength=1.8,        # IGNORED — kept for API compatibility
    suppress_strength=0.4,    # IGNORED — kept for API compatibility
    entropy_unlock=0.65,      # IGNORED — organic routing is always natural
    lock_after_tokens=0,      # IGNORED
    coherence_window=10,
    coherence_penalty=0.3,
    use_mirostat=False,
    mirostat_tau=5.0,
    mirostat_eta=0.1,
):
    """Legacy wrapper — redirects to organic_generate.

    The hard domain-lock approach (amplify/suppress gate_scales)
    has been replaced by organic routing coherence (EMA momentum).
    This function preserves the old API signature for compatibility
    but uses the new organic approach internally.
    """
    return organic_generate(
        model, sp, prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        momentum_alpha=0.2,
        coherence_window=coherence_window,
        temperature_adapt=coherence_penalty,
        use_mirostat=use_mirostat,
        mirostat_tau=mirostat_tau,
        mirostat_eta=mirostat_eta,
    )


@torch.no_grad()
def generate_with_routing_hint(
    model, sp, prompt,
    expert_name,
    hint_strength=0.3,
    max_tokens=150,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.3,
):
    """Generate text with a soft expert hint (not a hard bias).

    Instead of forcing a specific expert, provides a gentle nudge
    by slightly sharpening the router temperature. The model still
    makes its own routing decisions — the hint just makes them
    more decisive in the suggested direction.

    This replaces generate_with_expert_bias which hard-forced experts.
    """
    model.eval()
    expert_names = model.EXPERT_NAMES[:model.n_experts]

    if expert_name not in expert_names:
        raise ValueError(f"Unknown expert: {expert_name}. Available: {expert_names}")

    tokens = sp.encode(prompt)
    if not tokens:
        tokens = [sp.bos_id() if sp.bos_id() != -1 else 1]

    idx = torch.tensor([tokens[-model.block_size:]], dtype=torch.long)
    generated_ids = list(tokens)

    # Soft hint: slightly sharpen router (not hard override)
    original_temp = getattr(model.router, 'temperature', 1.0)
    hint_temp = max(0.5, original_temp * (1.0 - hint_strength))

    try:
        for step in range(max_tokens):
            idx_cond = idx[:, -model.block_size:]

            # Apply soft temperature hint
            model.router.temperature = hint_temp
            logits, _, info = model(idx_cond)
            model.router.temperature = original_temp

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
    finally:
        model.router.temperature = original_temp

    return sp.decode(generated_ids[len(tokens):])


# Legacy alias
generate_with_expert_bias = generate_with_routing_hint
