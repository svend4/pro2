"""
Продвинутые оптимизационные утилиты для YiJing-Transformer.

Включает:
- Layer-wise LR Decay (LLRD) — разные LR для разных слоёв
- Cosine warmup schedule с опциональным min_lr
- Отдельные группы параметров: decay vs no-decay
- Embedding warmup schedule

Использование:
    optimizer = build_optimizer(model, cfg, llrd_factor=0.8)
    for step in range(total_steps):
        lr = get_cosine_schedule(step, cfg.warmup_steps, cfg.total_steps, cfg.lr)
"""

import math
import torch


def build_optimizer(model, cfg, llrd_factor=1.0, embedding_lr_scale=1.0):
    """
    Создаёт AdamW с параметрическими группами:
    1. No decay: bias, LayerNorm, embedding
    2. Decay: всё остальное
    3. LLRD: экспоненциально убывающий lr от верхних к нижним слоям

    Args:
        model: YiJingGPT
        cfg: YiJingConfig
        llrd_factor: множитель LR между слоями (0.8 = каждый слой ниже × 0.8)
        embedding_lr_scale: масштаб LR для embedding (полезно для warmup)

    Returns:
        torch.optim.AdamW с группами параметров
    """
    no_decay = {'bias', 'ln_attn.weight', 'ln_attn.bias', 'ln_hex.weight',
                'ln_hex.bias', 'ln_ffn.weight', 'ln_ffn.bias',
                'final_norm.weight', 'final_norm.bias'}

    # Определяем слой для каждого параметра
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Определяем layer depth
        layer_idx = _get_layer_idx(name, cfg.n_layers)

        # LR множитель на основе LLRD
        if llrd_factor < 1.0 and layer_idx is not None:
            # Верхний слой (ближе к head) = полный LR
            # Нижний слой = LR × factor^(n_layers - layer_idx - 1)
            depth = cfg.n_layers - 1 - layer_idx
            lr_mult = llrd_factor ** depth
        elif 'tok_emb' in name or 'pos_emb' in name:
            lr_mult = embedding_lr_scale
        elif 'head.' in name:
            lr_mult = 1.0
        else:
            lr_mult = 1.0

        # Decay vs no-decay
        is_no_decay = any(nd in name for nd in no_decay)
        wd = 0.0 if is_no_decay else cfg.weight_decay

        group_key = (lr_mult, wd)
        if group_key not in param_groups:
            param_groups[group_key] = {
                'params': [],
                'lr': cfg.lr * lr_mult,
                'weight_decay': wd,
            }
        param_groups[group_key]['params'].append(param)

    groups = list(param_groups.values())
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def _get_layer_idx(name, n_layers):
    """Извлекает индекс слоя из имени параметра."""
    # Паттерн: core.layers.{idx}.xxx
    parts = name.split('.')
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def get_cosine_schedule(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    """
    Cosine schedule с linear warmup и опциональным min_lr.

    Args:
        step: текущий шаг
        warmup_steps: число шагов warmup
        total_steps: общее число шагов
        max_lr: пиковый learning rate
        min_lr: минимальный LR (по умолчанию 0)

    Returns:
        lr: текущий learning rate
    """
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)  # clamp

    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def get_warmup_stable_decay_schedule(step, warmup_steps, stable_steps, decay_steps,
                                     max_lr, min_lr=0.0):
    """
    Warmup → Stable → Cosine Decay schedule (WSD).

    Используется в Llama 3 и других современных LLM.

    Args:
        step: текущий шаг
        warmup_steps: linear warmup
        stable_steps: шаги на пиковом LR
        decay_steps: шаги cosine decay
        max_lr: пиковый LR
        min_lr: минимальный LR
    """
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    elif step < warmup_steps + stable_steps:
        return max_lr
    else:
        decay_step = step - warmup_steps - stable_steps
        progress = decay_step / max(1, decay_steps)
        progress = min(1.0, progress)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def get_param_stats(model):
    """
    Возвращает статистику по параметрам модели.
    Полезно для мониторинга обучения.
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            info = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'norm': param.data.norm().item(),
            }
            if param.grad is not None:
                info['grad_norm'] = param.grad.norm().item()
                info['grad_mean'] = param.grad.mean().item()
            stats[name] = info
    return stats
