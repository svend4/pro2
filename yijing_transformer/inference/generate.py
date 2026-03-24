"""Генерация текста для YiJing-Transformer."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, sp, prompt="Once upon a time", max_tokens=100,
             temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2,
             repetition_window=50, device=None):
    if device is None:
        device = next(model.parameters()).device

    block_size = getattr(model, 'block_size', None) or model.cfg.block_size
    model.eval()

    prompt_ids = sp.encode(prompt)
    if not prompt_ids:
        prompt_ids = [sp.bos_id() if sp.bos_id() != -1 else 1]

    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_tokens = []

    for _ in range(max_tokens):
        idx_cond = context[:, -block_size:]
        logits, _, _ = model(idx_cond)
        logits = logits[0, -1, :].clone()

        # Repetition penalty
        past = context[0, -repetition_window:].tolist()
        for t_idx in set(past):
            logits[t_idx] -= repetition_penalty * past.count(t_idx)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-K
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            probs[probs < v[-1]] = 0.0

        # Top-P
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = False
            probs[sorted_idx[mask]] = 0.0

        probs = probs / (probs.sum() + 1e-10)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
        token_id = next_token.item()

        eos_id = sp.eos_id()
        if eos_id != -1 and token_id == eos_id:
            break

        generated_tokens.append(token_id)

    return sp.decode(generated_tokens)
