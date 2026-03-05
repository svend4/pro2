"""
Обучение с cosine LR schedule, gradient accumulation и валидацией.
Исправляет основные проблемы тренировочного пайплайна Lila-E8.
"""

import os
import math
import torch
from data_utils.streaming_dataset import get_batch_streaming, create_train_val_iterators
from tokenizer.tokenizer_utils import load_tokenizer
from config.config import YiJingConfig
from models.model import YiJingGPT


def get_lr(step, cfg):
    """Cosine learning rate с warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_val_loss(model, val_iter, cfg, device, sp, num_batches=20):
    """Оценка validation loss."""
    model.eval()
    losses = []
    for _ in range(num_batches):
        xb, yb = get_batch_streaming(val_iter, cfg.batch_size, cfg.block_size, device, sp)
        if xb is None:
            break
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('nan')


def train(checkpoint_dir="checkpoints", resume=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sp = load_tokenizer()
    vocab_size = sp.get_piece_size()

    cfg = YiJingConfig(vocab_size=vocab_size)
    model = YiJingGPT(cfg).to(device)

    total_params, hex_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"YiJing-specific parameters: {hex_params:,} ({100*hex_params/total_params:.2f}%)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    train_iter, val_iter = create_train_val_iterators()
    log_every = 100
    save_every = 2000
    val_every = 500

    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0

    for step in range(1, cfg.total_steps + 1):
        # LR schedule
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        xb, yb = get_batch_streaming(train_iter, cfg.batch_size, cfg.block_size, device, sp)
        if xb is None:
            train_iter, _ = create_train_val_iterators()
            continue

        _, loss = model(xb, yb)
        loss = loss / cfg.grad_accum_steps
        loss.backward()
        accum_loss += loss.item()

        if step % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if step % log_every == 0:
            avg_loss = accum_loss / log_every * cfg.grad_accum_steps
            print(f"Step {step}: train_loss={avg_loss:.4f} lr={lr:.2e}")
            accum_loss = 0.0

        if step % val_every == 0:
            val_loss = estimate_val_loss(model, val_iter, cfg, device, sp)
            print(f"Step {step}: val_loss={val_loss:.4f}")

        if step % save_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
            }, os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt'))

    print("Training complete.")


if __name__ == "__main__":
    train()
