"""
Speculative Decoding для YiJing-Transformer.

Использует маленькую «draft» модель для предложения K токенов,
затем большая модель верифицирует их за один forward pass.

Ускорение ~2-3x при хорошем совпадении draft и target моделей.

Алгоритм:
1. Draft модель генерирует K токенов авторегрессивно
2. Target модель проверяет все K+1 позиций за 1 forward
3. Принимаем токены по rejection sampling
4. Если все K приняты — бонусный токен из target

Использование:
    target = YiJingGPT(cfg_large)
    draft = YiJingGPT(cfg_small)
    text = speculative_generate(target, draft, idx, max_new_tokens=100, K=4)
"""

import torch
import torch.nn.functional as F


def build_draft_model(target_cfg):
    """Создаёт draft модель на основе конфигурации target."""
    from dataclasses import replace
    from .model import YiJingGPT

    draft_d = target_cfg.draft_d_model or target_cfg.d_model // 2
    # Подбираем n_heads чтобы head_dim был целым
    draft_n_heads = max(1, draft_d // (target_cfg.d_model // target_cfg.n_heads))
    # Убедимся что d_model делится на n_heads
    while draft_d % draft_n_heads != 0 and draft_n_heads > 1:
        draft_n_heads -= 1

    draft_cfg = replace(
        target_cfg,
        d_model=draft_d,
        n_layers=target_cfg.draft_n_layers,
        n_heads=draft_n_heads,
        n_kv_heads=None,  # MHA для draft (проще)
        use_hex_moe=False,
        use_gradient_ckpt=False,
    )
    return YiJingGPT(draft_cfg)


@torch.no_grad()
def speculative_generate(
    target_model,
    draft_model,
    idx,
    max_new_tokens=100,
    K=4,
    temperature=1.0,
    top_k=None,
):
    """
    Speculative decoding: draft предлагает K токенов, target верифицирует.

    Args:
        target_model: большая (точная) модель
        draft_model: маленькая (быстрая) модель
        idx: начальная последовательность (B, T)
        max_new_tokens: сколько токенов сгенерировать
        K: число спекулятивных токенов за раз
        temperature: температура сэмплирования
        top_k: top-k фильтрация

    Returns:
        idx: расширенная последовательность
    """
    target_model.eval()
    draft_model.eval()

    block_size = target_model.cfg.block_size
    generated = 0

    while generated < max_new_tokens:
        # Шаг 1: Draft генерирует K токенов
        draft_tokens = []
        draft_probs_list = []
        draft_input = idx

        for _ in range(min(K, max_new_tokens - generated)):
            draft_cond = draft_input[:, -block_size:]
            draft_logits, _, _ = draft_model(draft_cond)
            draft_logits = draft_logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(draft_logits, min(top_k, draft_logits.size(-1)))
                draft_logits[draft_logits < v[:, [-1]]] = -float('Inf')

            draft_probs = F.softmax(draft_logits, dim=-1)
            draft_token = torch.multinomial(draft_probs, num_samples=1)

            draft_tokens.append(draft_token)
            draft_probs_list.append(draft_probs)
            draft_input = torch.cat([draft_input, draft_token], dim=1)

        n_draft = len(draft_tokens)
        if n_draft == 0:
            break

        # Собираем draft последовательность
        draft_seq = torch.cat(draft_tokens, dim=1)  # (B, n_draft)

        # Шаг 2: Target проверяет всю последовательность за один forward
        # Подаём idx + draft_tokens, получаем logits для всех позиций
        target_input = torch.cat([idx, draft_seq], dim=1)[:, -block_size:]
        target_logits, _, _ = target_model(target_input)
        target_logits = target_logits / temperature

        if top_k is not None:
            v, _ = torch.topk(target_logits, min(top_k, target_logits.size(-1)), dim=-1)
            target_logits[target_logits < v[..., [-1]]] = -float('Inf')

        # Target probs для позиций, где draft предложил токены
        # Позиции: от -n_draft-1 до -1 (inclusive)
        target_probs_all = F.softmax(target_logits, dim=-1)

        # Шаг 3: Rejection sampling
        n_accepted = 0
        for i in range(n_draft):
            target_pos = -(n_draft + 1) + i  # позиция в target output
            target_prob = target_probs_all[:, target_pos, :]  # (B, V)
            draft_prob = draft_probs_list[i]  # (B, V)
            token = draft_tokens[i]  # (B, 1)

            # P(accept) = min(1, p_target / p_draft)
            p_target = target_prob.gather(1, token).squeeze(1)  # (B,)
            p_draft = draft_prob.gather(1, token).squeeze(1)  # (B,)

            ratio = p_target / (p_draft + 1e-10)
            accept_prob = torch.clamp(ratio, max=1.0)
            r = torch.rand_like(accept_prob)

            if (r < accept_prob).all():
                n_accepted += 1
                idx = torch.cat([idx, token], dim=1)
                generated += 1
            else:
                # Отвергнут — сэмплируем из adjusted distribution
                adjusted = torch.clamp(target_prob - draft_prob, min=0)
                adjusted = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-10)
                # Fallback если adjusted пуст
                if adjusted.sum() < 1e-8:
                    adjusted = target_prob
                new_token = torch.multinomial(adjusted, num_samples=1)
                idx = torch.cat([idx, new_token], dim=1)
                generated += 1
                break

        # Если все K приняты — бонусный токен от target
        if n_accepted == n_draft:
            bonus_probs = target_probs_all[:, -1, :]  # последняя позиция
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)
            idx = torch.cat([idx, bonus_token], dim=1)
            generated += 1

    return idx


@torch.no_grad()
def measure_acceptance_rate(target_model, draft_model, data_tokens, K=4,
                            temperature=1.0, n_samples=50):
    """
    Измеряет acceptance rate для оценки качества draft модели.

    Returns:
        dict с метриками: acceptance_rate, tokens_per_step, speedup_estimate
    """
    target_model.eval()
    draft_model.eval()
    block_size = target_model.cfg.block_size

    total_accepted = 0
    total_proposed = 0

    for _ in range(n_samples):
        # Случайный промпт из данных
        start = torch.randint(0, max(1, len(data_tokens) - block_size), (1,)).item()
        idx = data_tokens[start:start + block_size // 2].unsqueeze(0)
        device = next(target_model.parameters()).device
        idx = idx.to(device)

        # Draft генерирует K токенов
        draft_input = idx
        draft_tokens = []
        draft_probs_list = []

        for _ in range(K):
            draft_cond = draft_input[:, -block_size:]
            draft_logits, _, _ = draft_model(draft_cond)
            draft_logits = draft_logits[:, -1, :] / temperature
            draft_probs = F.softmax(draft_logits, dim=-1)
            draft_token = torch.multinomial(draft_probs, num_samples=1)
            draft_tokens.append(draft_token)
            draft_probs_list.append(draft_probs)
            draft_input = torch.cat([draft_input, draft_token], dim=1)

        if not draft_tokens:
            continue

        draft_seq = torch.cat(draft_tokens, dim=1)
        target_input = torch.cat([idx, draft_seq], dim=1)[:, -block_size:]
        target_logits, _, _ = target_model(target_input)
        target_probs_all = F.softmax(target_logits / temperature, dim=-1)

        n_draft = len(draft_tokens)
        for i in range(n_draft):
            target_pos = -(n_draft + 1) + i
            target_prob = target_probs_all[:, target_pos, :]
            draft_prob = draft_probs_list[i]
            token = draft_tokens[i]

            p_target = target_prob.gather(1, token).squeeze(1)
            p_draft = draft_prob.gather(1, token).squeeze(1)
            ratio = (p_target / (p_draft + 1e-10)).clamp(max=1.0)

            total_proposed += 1
            if torch.rand(1).item() < ratio.item():
                total_accepted += 1
            else:
                break

    acceptance_rate = total_accepted / max(1, total_proposed)
    tokens_per_step = 1 + K * acceptance_rate  # ожидаемое число токенов за шаг
    return {
        'acceptance_rate': acceptance_rate,
        'tokens_per_step': tokens_per_step,
        'speedup_estimate': tokens_per_step,  # примерное ускорение
        'total_proposed': total_proposed,
        'total_accepted': total_accepted,
    }
