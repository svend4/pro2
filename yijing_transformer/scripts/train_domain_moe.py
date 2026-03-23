"""
Обучение YiJing-Transformer с DomainMoE — экспертами по доменам корпуса.

Каждый из 6 экспертов обучается специализироваться на своём домене:
  0: ai_agents    — ИИ-агенты, скиллы
  1: infosystems  — инфосистемы, метаданные
  2: knowledge    — архетипы, гуманитарные тексты
  3: algorithms   — алгоритмы, оптимизация
  4: data2        — энциклопедия
  5: meta         — метатексты, И-Цзин

Запуск:
    cd /home/user/pro2
    python yijing_transformer/scripts/train_domain_moe.py \\
        --svend4 data/svend4_corpus \\
        --steps 6000 \\
        --resume checkpoints/checkpoint_step_4000.pt

Параметры:
    --steps     : шаги обучения (по умолчанию 6000)
    --resume    : загрузить чекпоинт для дообучения
    --experts   : число экспертов (по умолчанию 6)
    --top-k     : активных экспертов за forward (по умолчанию 2)
    --dsw       : domain supervision weight (по умолчанию 0.1)
    --no-resume : начать с нуля
"""

import sys
import os
import argparse

# Путь к пакету
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from config.config import YiJingConfig
from models.model import YiJingGPT
from data_utils.svend4_dataset import Svend4Corpus


def get_lr(step, warmup, total, base_lr=3e-4, min_lr=3e-5):
    if step < warmup:
        return base_lr * step / warmup
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + __import__('math').cos(__import__('math').pi * t))


def main():
    parser = argparse.ArgumentParser(description='Train YiJing DomainMoE')
    parser.add_argument('--svend4', type=str, default='data/svend4_corpus')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--experts', type=int, default=6,
                        help='Число экспертов (= число доменов)')
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--dsw', type=float, default=0.1,
                        help='Domain supervision weight')
    parser.add_argument('--resume', type=str, default=None,
                        help='Чекпоинт для дообучения')
    parser.add_argument('--no-resume', action='store_true',
                        help='Начать с нуля (игнорировать --resume)')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Загрузка корпуса
    print(f'Loading corpus: {args.svend4}')
    corpus = Svend4Corpus.from_directory(args.svend4, block_size=args.block_size)
    corpus.print_stats()
    vocab_size = corpus.get_vocab_size()
    print(f'Vocab size: {vocab_size}')

    start_step = 0

    # Конфиг с DomainMoE
    cfg = YiJingConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        block_size=args.block_size,
        batch_size=args.batch_size,
        dropout=0.05,
        use_rope=True,
        use_swiglu=False,        # DomainMoE заменяет SwiGLU
        use_bian_gua=True,
        use_hex_moe=False,       # не TrigramMoE
        use_domain_moe=True,     # <-- DomainMoE
        domain_moe_n_experts=args.experts,
        domain_moe_top_k=args.top_k,
        domain_supervision_weight=args.dsw,
        adaptive_temp=True,
        use_nautilus=True,
        nautilus_mode='sequential',
        nautilus_warmup_steps=1000,
        total_steps=args.steps,
        warmup_steps=200,
        weight_decay=0.1,
    )

    model = YiJingGPT(cfg).to(device)

    # Загрузка чекпоинта (если есть)
    if args.resume and not args.no_resume and os.path.exists(args.resume):
        print(f'Loading checkpoint: {args.resume}')
        with torch.serialization.safe_globals([YiJingConfig]):
            ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        ckpt_cfg = ckpt['config']
        old_vocab = ckpt['model_state_dict']['tok_emb.weight'].shape[0]
        if old_vocab != vocab_size:
            print(f'Vocab mismatch: ckpt={old_vocab}, corpus={vocab_size} — reinitializing')
        else:
            try:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
                start_step = ckpt.get('step', 0)
                print(f'Resumed from step {start_step} (strict=False, DomainMoE layers fresh)')
            except Exception as e:
                print(f'Checkpoint load failed ({e}), starting fresh')
    elif not args.resume:
        print('No checkpoint specified, training from scratch')

    total, yijing_params = model.count_parameters()
    print(f'\nModel: {total:,} params ({yijing_params:,} YiJing-specific)')
    print(f'DomainMoE: {args.experts} experts, top-{args.top_k}, dsw={args.dsw}')
    print(f'Training steps: {start_step} → {args.steps}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Цикл обучения
    model.train()
    accum_loss = 0.0
    expert_usage = torch.zeros(args.experts)  # статистика активации экспертов

    for step in range(start_step + 1, args.steps + 1):
        lr = get_lr(step, warmup=200, total=args.steps, base_lr=args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y, domain_ids = corpus.get_batch_with_domain(args.batch_size, device)

        optimizer.zero_grad()
        logits, loss, _ = model(x, y, domain_ids=domain_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        accum_loss += loss.item()

        if step % args.log_every == 0:
            avg_loss = accum_loss / args.log_every
            print(f'step {step:5d} | loss {avg_loss:.4f} | lr {lr:.2e}')
            accum_loss = 0.0

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'domain_moe_step_{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
            }, ckpt_path)
            print(f'Saved: {ckpt_path}')

    # Финальный чекпоинт
    final_path = os.path.join(args.save_dir, 'domain_moe_final.pt')
    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg,
    }, final_path)
    print(f'\nDone! Final checkpoint: {final_path}')


if __name__ == '__main__':
    main()
