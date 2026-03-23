"""
Официальный запуск Недели 2: interlingua_fixed.

LeanYiJingGPT + ArchetypalInterlingua (per-source trit_proj).
Сравнивается с lean_baseline PPL из experiments/lean_baseline_log.json.

Использование:
    python experiments/train_interlingua_fixed.py
    python experiments/train_interlingua_fixed.py --dry-run
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yijing_transformer.models.lean_model import LeanYiJingGPT, LeanYiJingBlock
from yijing_transformer.models.geometry.interlingua_fixed import ArchetypalInterlinguaFixed
from corpus_loader import CorpusLoader


DEFAULTS = dict(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=4,
    block_size=256,
    dropout=0.0,
    steps=3000,
    batch_size=8,
    lr=3e-4,
    warmup_steps=200,
    grad_clip=1.0,
    # ArchetypalInterlingua параметры
    n_archetypes=64,
    interlingua_warmup=3000,   # cosine annealing на весь прогон
    diversity_weight=0.01,     # штраф за одинаковые триты
    log_every=100,
    xerox_every=500,
    save_every=1000,
)


class LeanWithInterlingua(torch.nn.Module):
    """LeanYiJingGPT + ArchetypalInterlingua как residual поверх блоков.

    Interlingua собирает выходы всех 4 блоков как n_sources=4 источника,
    вычисляет per-source тернарные коды и добавляет к финальному представлению.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.base = LeanYiJingGPT(
            vocab_size=cfg['vocab_size'],
            d_model=cfg['d_model'],
            n_layers=cfg['n_layers'],
            n_heads=cfg['n_heads'],
            block_size=cfg['block_size'],
            dropout=cfg['dropout'],
        )
        self.interlingua = ArchetypalInterlinguaFixed(
            d_model=cfg['d_model'],
            n_sources=cfg['n_layers'],  # каждый блок = источник
            n_archetypes=cfg['n_archetypes'],
            diversity_weight=cfg['diversity_weight'],
            warmup_steps=cfg['interlingua_warmup'],
        )
        self.diversity_weight = cfg['diversity_weight']
        self.d_model = cfg['d_model']
        self.block_size = cfg['block_size']

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple:
        B, T = idx.shape
        base = self.base

        # Прогоняем через embedding
        pos = torch.arange(T, device=idx.device)
        x = base.drop(base.embed(idx) + base.pos_embed(pos))

        # Собираем выходы всех блоков
        block_outputs = []
        for block in base.blocks:
            x = block(x)
            block_outputs.append(x)

        # ArchetypalInterlinguaFixed: per-source тернарные коды
        # forward возвращает (output, aux_loss) где aux_loss уже включает diversity_weight
        interlingua_repr, aux_loss = self.interlingua(block_outputs, core_hidden=x)

        # Residual: основной поток + interlingua (gate внутри interlingua)
        x = base.norm(interlingua_repr)
        logits = base.head(x)

        loss = None
        diversity_loss_val = 0.0
        if targets is not None:
            lm_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            diversity_loss_val = aux_loss
            loss = lm_loss + aux_loss

        return logits, loss, diversity_loss_val


def build_dataset():
    loader = CorpusLoader()
    corpus = loader.as_training_corpus(max_per_source=1000)
    texts = [d['text'] for d in corpus if len(d['text']) > 50]
    print(f"  Корпус: {len(texts)} текстов, {sum(len(t) for t in texts):,} символов")
    return [list(t.encode('utf-8', errors='replace')) for t in texts]


def get_batch(encoded, block_size, batch_size, device):
    x_list, y_list = [], []
    for _ in range(batch_size):
        idx = torch.randint(len(encoded), (1,)).item()
        seq = encoded[idx]
        if len(seq) < block_size + 1:
            seq = seq + [0] * (block_size + 1 - len(seq))
        start = torch.randint(0, max(1, len(seq) - block_size), (1,)).item()
        chunk = seq[start: start + block_size + 1]
        x_list.append(chunk[:block_size])
        y_list.append(chunk[1: block_size + 1])
    x = torch.tensor(x_list, dtype=torch.long, device=device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)
    return x, y


def get_lr(step, cfg):
    if step < cfg['warmup_steps']:
        return cfg['lr'] * step / max(cfg['warmup_steps'], 1)
    progress = (step - cfg['warmup_steps']) / max(1, cfg['steps'] - cfg['warmup_steps'])
    return cfg['lr'] * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_baseline_ppl() -> float | None:
    """Загружает официальный baseline PPL из lean_baseline_log.json."""
    log_path = Path('experiments') / 'lean_baseline_log.json'
    if log_path.exists():
        with open(log_path) as f:
            data = json.load(f)
        return data.get('final_ppl')
    return None


def train(cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    baseline_ppl = load_baseline_ppl()
    baseline_str = f"{baseline_ppl:.4f}" if baseline_ppl else "не готов"

    print(f"\n{'='*64}")
    print(f"  INTERLINGUA FIXED (Неделя 2)")
    print(f"{'='*64}")
    print(f"  Устройство      : {device}")
    print(f"  Шаги            : {cfg['steps']}")
    print(f"  Baseline PPL    : {baseline_str}")
    print(f"  n_archetypes    : {cfg['n_archetypes']}")
    print(f"  diversity_weight: {cfg['diversity_weight']}")
    print(f"  interlingua_T   : cosine 1.0→0.05 за {cfg['interlingua_warmup']} шагов")

    encoded = build_dataset()

    model = LeanWithInterlingua(cfg).to(device)
    print(f"  Параметры  : {model.num_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], betas=(0.9, 0.95), weight_decay=0.1
    )

    log = {
        'config': cfg,
        'baseline_ppl': baseline_ppl,
        'steps': [], 'losses': [], 'ppls': [],
        'diversity_losses': [], 'temperatures': [],
    }

    best_loss = float('inf')
    t0 = time.time()
    model.train()

    print(f"\n{'─'*64}")
    for step in range(1, cfg['steps'] + 1):
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch(encoded, cfg['block_size'], cfg['batch_size'], device)
        logits, loss, div_loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()

        loss_val = loss.item()
        ppl_val = math.exp(min(loss_val, 20))
        T_current = model.interlingua._get_temperature()

        if loss_val < best_loss:
            best_loss = loss_val

        if step % cfg['log_every'] == 0 or step == 1:
            div_val = div_loss.item() if hasattr(div_loss, 'item') else float(div_loss)
            elapsed = time.time() - t0
            log['steps'].append(step)
            log['losses'].append(round(loss_val, 4))
            log['ppls'].append(round(ppl_val, 4))
            log['diversity_losses'].append(round(div_val, 4))
            log['temperatures'].append(round(T_current, 4))
            print(
                f"  step={step:4d}/{cfg['steps']} | "
                f"loss={loss_val:.4f} | ppl={ppl_val:.4f} | "
                f"T={T_current:.3f} | div={div_val:.4f} | {elapsed:.0f}s"
            )

        if step % cfg['save_every'] == 0:
            ckpt = Path('experiments') / f'interlingua_fixed_step{step}.pt'
            torch.save({'step': step, 'model_state': model.state_dict(),
                        'loss': loss_val, 'config': cfg}, ckpt)
            print(f"  Чекпоинт: {ckpt}")

    final_ppl = log['ppls'][-1] if log['ppls'] else float('inf')
    total_time = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  РЕЗУЛЬТАТ interlingua_fixed")
    print(f"{'='*64}")
    print(f"  Final PPL  : {final_ppl:.4f}")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  Baseline   : {baseline_str}")

    verdict = "inconclusive"
    if cfg['steps'] < 3000:
        print(f"  ⚠  < 3000 шагов — вердикт недействителен")
    elif baseline_ppl and final_ppl < baseline_ppl:
        print(f"  ✓  УСПЕХ: PPL улучшился vs baseline ({final_ppl:.4f} < {baseline_ppl:.4f})")
        verdict = "proven"
    elif baseline_ppl:
        diff = final_ppl - baseline_ppl
        print(f"  ✗  Хуже baseline на {diff:.4f}")
        verdict = "disproven"

    final_ckpt = Path('experiments') / 'interlingua_fixed_final.pt'
    torch.save({'step': cfg['steps'], 'model_state': model.state_dict(),
                'final_ppl': final_ppl, 'verdict': verdict, 'config': cfg}, final_ckpt)

    log.update({'final_ppl': final_ppl, 'best_loss': best_loss,
                'verdict': verdict, 'total_time_s': round(total_time, 1)})
    log_path = Path('experiments') / 'interlingua_fixed_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Лог: {log_path}")

    return final_ppl, verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.dry_run:
        cfg['steps'] = 50
        cfg['log_every'] = 10
        cfg['xerox_every'] = 25
        cfg['save_every'] = 9999

    train(cfg)
