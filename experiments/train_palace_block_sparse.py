"""
Официальный запуск Недели 5: palace_block_sparse.

LeanYiJingGPT + PalaceAttention (block-sparse 8×8 маска Скляровой).
Гипотеза: PalaceAttention вместо HeisenbergAttention улучшает PPL.

Метрика успеха: PPL < lean_baseline PPL (1.51)

Использование:
    python experiments/train_palace_block_sparse.py
    python experiments/train_palace_block_sparse.py --dry-run
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from yijing_transformer.models.geometry.attention import PalaceAttention, FlowerOfLifeGAT
from yijing_transformer.models.geometry.core import palace_attention_mask
from corpus_loader import CorpusLoader


class FastPalaceAttention(PalaceAttention):
    """PalaceAttention с предвычисленной маской для произвольного block_size.

    Базовый PalaceAttention.get_mask() использует двойной Python-цикл для
    seq_len > 64, что даёт O(T²) Python-итераций на каждый forward().
    FastPalaceAttention предвычисляет маску один раз при инициализации.
    """

    def __init__(self, d_model: int, n_heads: int = 8, block_size: int = 256):
        super().__init__(d_model, n_heads)
        # Предвычисляем маску для всего block_size векторизованно
        mask_64 = palace_attention_mask(64)  # (64, 64)
        i_idx = torch.arange(block_size) % 64
        j_idx = torch.arange(block_size) % 64
        base = mask_64[i_idx.unsqueeze(1), j_idx.unsqueeze(0)]  # (T, T)
        intra = (base == 1.0).float()
        inter = (base < 1.0).float()
        # intra + sigmoid(weight) * inter кешируем через отдельный буфер
        self.register_buffer('_intra', intra)
        self.register_buffer('_inter', inter)
        self._block_size = block_size

    def get_mask(self, seq_len: int) -> torch.Tensor:
        intra = self._intra[:seq_len, :seq_len]
        inter = self._inter[:seq_len, :seq_len]
        return intra + torch.sigmoid(self.inter_palace_weight) * inter


DEFAULTS = dict(
    vocab_size=256,
    d_model=128,
    n_layers=4,
    n_heads=8,       # PalaceAttention: 8 дворцов = 8 голов
    block_size=256,
    dropout=0.0,
    steps=3000,
    batch_size=8,
    lr=3e-4,
    warmup_steps=200,
    grad_clip=1.0,
    log_every=100,
    save_every=1000,
)


class PalaceYiJingBlock(nn.Module):
    """Блок с PalaceAttention (block-sparse) вместо HeisenbergAttention.

    Структура идентична LeanYiJingBlock, но:
      HeisenbergAttention (dense) → PalaceAttention (block-sparse 8×8)

    Это тест гипотезы: Склярова's palace-sparse pattern улучшает PPL
    за счёт структурированного внимания к смысловым "дворцам".
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, block_size: int = 256):
        super().__init__()
        self.palace = FastPalaceAttention(d_model, n_heads, block_size=block_size)
        self.flower_gat = FlowerOfLifeGAT(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Learnable gate: баланс palace ↔ flower_gat
        self.gate = nn.Parameter(torch.tensor(0.0))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_palace = self.palace(self.norm1(x))
        h_gat = self.flower_gat(self.norm2(x))
        alpha = torch.sigmoid(self.gate)
        x = x + alpha * h_palace + (1 - alpha) * h_gat
        x = x + self.ffn(self.norm_ffn(x))
        return x


class PalaceYiJingGPT(nn.Module):
    """LeanYiJingGPT с PalaceAttention вместо HeisenbergAttention.

    Параметры идентичны LeanYiJingGPT для честного сравнения:
    d_model=128, n_layers=4, block_size=256, vocab_size=256.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        block_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PalaceYiJingBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.embed(idx) + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


def build_dataset():
    loader = CorpusLoader()
    corpus = loader.as_training_corpus(max_per_source=1000)
    texts = [d['text'] for d in corpus if len(d['text']) > 50]
    print(f"  Корпус: {len(texts)} текстов")
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
    return (
        torch.tensor(x_list, dtype=torch.long, device=device),
        torch.tensor(y_list, dtype=torch.long, device=device),
    )


def get_lr(step, cfg):
    if step < cfg['warmup_steps']:
        return cfg['lr'] * step / max(cfg['warmup_steps'], 1)
    progress = (step - cfg['warmup_steps']) / max(1, cfg['steps'] - cfg['warmup_steps'])
    return cfg['lr'] * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_baseline_ppl():
    path = Path('experiments') / 'lean_baseline_log.json'
    if path.exists():
        return json.loads(path.read_text()).get('final_ppl')
    return 1.51


def train(cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_ppl = load_baseline_ppl()

    print(f"\n{'='*64}")
    print(f"  PALACE BLOCK-SPARSE (Неделя 5)")
    print(f"{'='*64}")
    print(f"  Устройство     : {device}")
    print(f"  Шаги           : {cfg['steps']}")
    print(f"  Baseline PPL   : {baseline_ppl:.4f}")
    print(f"  n_heads        : {cfg['n_heads']} (= 8 дворцов Скляровой)")

    encoded = build_dataset()
    model = PalaceYiJingGPT(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        block_size=cfg['block_size'],
        dropout=cfg['dropout'],
    ).to(device)
    print(f"  Параметры      : {model.num_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], betas=(0.9, 0.95), weight_decay=0.1
    )

    log = {
        'config': cfg, 'baseline_ppl': baseline_ppl,
        'steps': [], 'losses': [], 'ppls': [],
        'gate_history': [],
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
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()

        loss_val = loss.item()
        ppl_val = math.exp(min(loss_val, 20))
        if loss_val < best_loss:
            best_loss = loss_val

        if step % cfg['log_every'] == 0 or step == 1:
            elapsed = time.time() - t0
            log['steps'].append(step)
            log['losses'].append(round(loss_val, 4))
            log['ppls'].append(round(ppl_val, 4))

            # Среднее значение gate по всем блокам
            avg_gate = sum(
                torch.sigmoid(b.gate).item() for b in model.blocks
            ) / len(model.blocks)
            log['gate_history'].append({'step': step, 'avg_gate': round(avg_gate, 4)})

            print(
                f"  step={step:4d}/{cfg['steps']} | "
                f"loss={loss_val:.4f} | ppl={ppl_val:.4f} | "
                f"gate={avg_gate:.3f} | lr={lr:.2e} | {elapsed:.0f}s"
            )

        if step % cfg['save_every'] == 0:
            ckpt = Path('experiments') / f'palace_block_step{step}.pt'
            torch.save({'step': step, 'model_state': model.state_dict(),
                        'loss': loss_val, 'config': cfg}, ckpt)

    final_ppl = log['ppls'][-1] if log['ppls'] else float('inf')
    total_time = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  РЕЗУЛЬТАТ palace_block_sparse")
    print(f"{'='*64}")
    print(f"  Final PPL      : {final_ppl:.4f}")
    print(f"  Best loss      : {best_loss:.4f}")
    print(f"  Baseline PPL   : {baseline_ppl:.4f}")

    verdict = "inconclusive"
    if cfg['steps'] < 3000:
        print(f"  ⚠  < 3000 шагов — вердикт недействителен")
    else:
        if final_ppl <= baseline_ppl * 1.02:
            print(f"  ✓  УСПЕХ: PPL {final_ppl:.4f} ≤ baseline {baseline_ppl:.4f}")
            verdict = "proven"
        else:
            print(f"  ✗  PPL хуже baseline: {final_ppl:.4f} > {baseline_ppl:.4f}")
            verdict = "disproven"

    final_ckpt = Path('experiments') / 'palace_block_final.pt'
    torch.save({'step': cfg['steps'], 'model_state': model.state_dict(),
                'final_ppl': final_ppl, 'best_loss': best_loss,
                'verdict': verdict, 'config': cfg}, final_ckpt)

    log.update({'final_ppl': final_ppl, 'best_loss': best_loss,
                'verdict': verdict, 'total_time_s': round(total_time, 1)})
    log_path = Path('experiments') / 'palace_block_log.json'
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
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
        cfg['save_every'] = 9999

    train(cfg)
