"""
Официальный запуск Недели 1: lean_baseline.

Обучает LeanYiJingGPT на реальных данных svend4_corpus.
Минимум 3000 шагов. Результат фиксируется как официальный ceiling PPL.

Использование:
    python experiments/train_lean_baseline.py
    python experiments/train_lean_baseline.py --steps 3000 --lr 3e-4
    python experiments/train_lean_baseline.py --dry-run   # 50 шагов, проверка
"""

import sys
import math
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from yijing_transformer.models.lean_model import LeanYiJingGPT
from corpus_loader import CorpusLoader


# ─── Гиперпараметры ──────────────────────────────────────────────

DEFAULTS = dict(
    vocab_size=256,     # byte-level: покрывает весь Unicode через UTF-8
    d_model=128,
    n_layers=4,
    n_heads=4,
    block_size=256,
    dropout=0.0,
    # Обучение
    steps=3000,
    batch_size=8,
    lr=3e-4,
    warmup_steps=200,
    grad_clip=1.0,
    # Диагностика
    log_every=100,
    xerox_every=500,
    save_every=1000,
    # Протокол
    min_steps_for_verdict=3000,
)


# ─── Данные ──────────────────────────────────────────────────────

def build_dataset(max_per_source: int = 1000) -> list[str]:
    """Загружает реальные данные из svend4_corpus."""
    loader = CorpusLoader()
    corpus = loader.as_training_corpus(max_per_source=max_per_source)
    texts = [d['text'] for d in corpus if len(d['text']) > 50]
    total = sum(len(t) for t in texts)
    print(f"  Корпус: {len(texts)} текстов, {total:,} символов")
    return texts


def text_to_bytes(texts: list[str]) -> list[list[int]]:
    """Конвертирует тексты в byte-level последовательности."""
    return [list(t.encode('utf-8', errors='replace')) for t in texts]


def get_batch(
    encoded: list[list[int]],
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple:
    """Случайный батч из корпуса."""
    x_list, y_list = [], []
    for _ in range(batch_size):
        idx = torch.randint(len(encoded), (1,)).item()
        seq = encoded[idx]
        if len(seq) < block_size + 1:
            # Короткий текст — padding нулями
            seq = seq + [0] * (block_size + 1 - len(seq))
        start = torch.randint(0, max(1, len(seq) - block_size), (1,)).item()
        chunk = seq[start: start + block_size + 1]
        x_list.append(chunk[:block_size])
        y_list.append(chunk[1: block_size + 1])

    x = torch.tensor(x_list, dtype=torch.long, device=device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)
    return x, y


# ─── LR schedule ─────────────────────────────────────────────────

def get_lr(step: int, cfg: dict) -> float:
    """Cosine decay с warmup."""
    if step < cfg['warmup_steps']:
        return cfg['lr'] * step / max(cfg['warmup_steps'], 1)
    progress = (step - cfg['warmup_steps']) / max(
        1, cfg['steps'] - cfg['warmup_steps']
    )
    return cfg['lr'] * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Ксерокс-тест (встроенный) ───────────────────────────────────

XEROX_PROBES = [
    "def neural_network(x):",
    "class GradientDescent:",
    "SELECT * FROM experts",
    "Hexagram as archetype",
]


def run_xerox_probe(model: LeanYiJingGPT, step: int, device: str):
    """Мини ксерокс-тест: PPL на диагностических строках."""
    model.eval()
    results = []
    with torch.no_grad():
        for text in XEROX_PROBES:
            encoded = list(text.encode('utf-8', errors='replace'))
            if len(encoded) < 2:
                continue
            x = torch.tensor([encoded[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([encoded[1:]], dtype=torch.long, device=device)
            _, loss = model(x, y)
            ppl = loss.exp().item()
            results.append((text[:25], ppl))
    model.train()

    avg_ppl = sum(p for _, p in results) / max(len(results), 1)
    print(f"  [Xerox step={step}] avg_PPL={avg_ppl:.2f}: " +
          " | ".join(f"'{t}' {p:.1f}" for t, p in results))
    return avg_ppl


# ─── Обучение ────────────────────────────────────────────────────

def train(cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*64}")
    print(f"  LEAN BASELINE (Неделя 1)")
    print(f"{'='*64}")
    print(f"  Устройство : {device}")
    print(f"  Шаги       : {cfg['steps']} (минимум для валидного вердикта: {cfg['min_steps_for_verdict']})")
    print(f"  d_model    : {cfg['d_model']}, layers={cfg['n_layers']}, heads={cfg['n_heads']}")
    print(f"  block_size : {cfg['block_size']}, batch={cfg['batch_size']}")
    print(f"  lr         : {cfg['lr']}, warmup={cfg['warmup_steps']}")

    # Данные
    print("\n  Загрузка корпуса...")
    texts = build_dataset()
    encoded = text_to_bytes(texts)
    print(f"  Закодировано: {len(encoded)} последовательностей")

    # Модель
    model = LeanYiJingGPT(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        block_size=cfg['block_size'],
        dropout=cfg['dropout'],
    ).to(device)
    print(f"  Параметры  : {model.num_parameters:,}")

    # Оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Лог
    log = {
        'config': cfg,
        'steps': [],
        'losses': [],
        'ppls': [],
        'xerox_ppls': [],
    }

    best_loss = float('inf')
    t0 = time.time()
    model.train()

    print(f"\n{'─'*64}")
    print(f"  ОБУЧЕНИЕ")
    print(f"{'─'*64}")

    for step in range(1, cfg['steps'] + 1):
        # LR update
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Батч
        x, y = get_batch(encoded, cfg['block_size'], cfg['batch_size'], device)

        # Forward
        _, loss = model(x, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()

        loss_val = loss.item()
        ppl_val = math.exp(min(loss_val, 20))

        if loss_val < best_loss:
            best_loss = loss_val

        # Логирование
        if step % cfg['log_every'] == 0 or step == 1:
            elapsed = time.time() - t0
            log['steps'].append(step)
            log['losses'].append(round(loss_val, 4))
            log['ppls'].append(round(ppl_val, 4))
            print(
                f"  step={step:4d}/{cfg['steps']} | "
                f"loss={loss_val:.4f} | ppl={ppl_val:.4f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

        # Ксерокс-тест
        if step % cfg['xerox_every'] == 0:
            xppl = run_xerox_probe(model, step, device)
            log['xerox_ppls'].append({'step': step, 'ppl': round(xppl, 4)})

        # Сохранение чекпоинта
        if step % cfg['save_every'] == 0:
            ckpt_path = Path('experiments') / f'lean_baseline_step{step}.pt'
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': loss_val,
                'config': cfg,
            }, ckpt_path)
            print(f"  Чекпоинт сохранён: {ckpt_path}")

    # Финальная оценка
    total_time = time.time() - t0
    final_loss = log['losses'][-1] if log['losses'] else float('inf')
    final_ppl = log['ppls'][-1] if log['ppls'] else float('inf')

    print(f"\n{'='*64}")
    print(f"  РЕЗУЛЬТАТ lean_baseline")
    print(f"{'='*64}")
    print(f"  Шагов      : {cfg['steps']}")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  Final loss : {final_loss:.4f}")
    print(f"  Final PPL  : {final_ppl:.4f}")
    print(f"  Время      : {total_time:.0f}s")

    if cfg['steps'] < cfg['min_steps_for_verdict']:
        print(f"\n  ⚠  ВНИМАНИЕ: {cfg['steps']} < {cfg['min_steps_for_verdict']} шагов")
        print(f"     Вердикт статистически недействителен!")
        verdict = "inconclusive"
    elif final_ppl < 1.07:
        print(f"\n  ✓  УСПЕХ: PPL={final_ppl:.4f} < 1.07 (цель достигнута)")
        verdict = "proven"
    else:
        print(f"\n  ?  PPL={final_ppl:.4f} (цель: < 1.07, нужно больше данных/шагов)")
        verdict = "inconclusive"

    # Сохранить финальный чекпоинт
    final_ckpt = Path('experiments') / 'lean_baseline_final.pt'
    torch.save({
        'step': cfg['steps'],
        'model_state': model.state_dict(),
        'final_loss': final_loss,
        'final_ppl': final_ppl,
        'best_loss': best_loss,
        'config': cfg,
        'verdict': verdict,
    }, final_ckpt)
    print(f"  Финал сохранён: {final_ckpt}")

    # Сохранить лог
    log_path = Path('experiments') / 'lean_baseline_log.json'
    log['final_ppl'] = final_ppl
    log['final_loss'] = final_loss
    log['best_loss'] = best_loss
    log['verdict'] = verdict
    log['total_time_s'] = round(total_time, 1)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Лог сохранён: {log_path}")

    return final_ppl, verdict


# ─── CLI ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lean Baseline Training')
    parser.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--d-model', type=int, default=DEFAULTS['d_model'])
    parser.add_argument('--n-layers', type=int, default=DEFAULTS['n_layers'])
    parser.add_argument('--block-size', type=int, default=DEFAULTS['block_size'])
    parser.add_argument('--dry-run', action='store_true',
                        help='50 шагов: проверка что всё запускается')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg['steps'] = 50 if args.dry_run else args.steps
    cfg['lr'] = args.lr
    cfg['batch_size'] = args.batch_size
    cfg['d_model'] = args.d_model
    cfg['n_layers'] = args.n_layers
    cfg['block_size'] = args.block_size
    cfg['log_every'] = 10 if args.dry_run else cfg['log_every']
    cfg['xerox_every'] = 25 if args.dry_run else cfg['xerox_every']
    cfg['save_every'] = 9999 if args.dry_run else cfg['save_every']

    train(cfg)
