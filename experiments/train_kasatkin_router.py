"""
Официальный запуск Недели 3: kasatkin_as_router.

LeanYiJingGPT + KasatkinQ6Router как слой маршрутизации.
Метрика успеха: routing_confidence > 15% при PPL не хуже lean_baseline.

Использование:
    python experiments/train_kasatkin_router.py
    python experiments/train_kasatkin_router.py --dry-run
"""

import sys
import math
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from yijing_transformer.models.lean_model import LeanYiJingGPT
from yijing_transformer.models.geometry.kasatkin_router import KasatkinQ6Router
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
    n_experts=6,
    routing_temperature=0.5,
    log_every=100,
    conf_every=500,    # как часто замерять routing_confidence
    save_every=1000,
    # Критерий успеха
    target_confidence=0.15,
)


class LeanWithKasatkinRouter(nn.Module):
    """LeanYiJingGPT с KasatkinQ6Router как дополнительным слоем.

    Router применяется после последнего блока: взвешивает
    репликации выхода по 6 экспертным «осям» куба.
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
        self.router = KasatkinQ6Router(
            d_model=cfg['d_model'],
            n_experts=cfg['n_experts'],
            routing_temperature=cfg['routing_temperature'],
        )
        # 6 лёгких экспертных проекторов (каждый: d_model → d_model)
        self.expert_projs = nn.ModuleList([
            nn.Linear(cfg['d_model'], cfg['d_model'], bias=False)
            for _ in range(cfg['n_experts'])
        ])
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
        base = self.base
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = base.drop(base.embed(idx) + base.pos_embed(pos))
        for block in base.blocks:
            x = block(x)

        # Router: routing_weights (B, T, 6)
        routing_weights = self.router(x)

        # Каждый эксперт: линейный проектор
        expert_outs = torch.stack(
            [proj(x) for proj in self.expert_projs], dim=-1
        )  # (B, T, d_model, 6)

        # Взвешенная смесь экспертов
        routed = (expert_outs * routing_weights.unsqueeze(2)).sum(dim=-1)  # (B, T, d_model)

        # Residual: основной поток + routed
        x = base.norm(x + routed)
        logits = base.head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss, routing_weights

    def get_routing_confidence(self, x: torch.Tensor) -> float:
        return self.router.get_routing_confidence(x)


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
    return None


def train(cfg: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_ppl = load_baseline_ppl()

    print(f"\n{'='*64}")
    print(f"  KASATKIN AS ROUTER (Неделя 3)")
    print(f"{'='*64}")
    print(f"  Устройство     : {device}")
    print(f"  Шаги           : {cfg['steps']}")
    baseline_str = f"{baseline_ppl:.4f}" if baseline_ppl else "не готов"
    print(f"  Baseline PPL   : {baseline_str}")
    print(f"  n_experts      : {cfg['n_experts']} (= 6 осей куба)")
    print(f"  routing_T      : {cfg['routing_temperature']}")
    print(f"  Цель           : routing_conf > {cfg['target_confidence']}")

    encoded = build_dataset()
    model = LeanWithKasatkinRouter(cfg).to(device)
    print(f"  Параметры : {model.num_parameters:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], betas=(0.9, 0.95), weight_decay=0.1
    )

    log = {
        'config': cfg, 'baseline_ppl': baseline_ppl,
        'steps': [], 'losses': [], 'ppls': [], 'routing_confidences': [],
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
        logits, loss, routing_weights = model(x, y)

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
            print(
                f"  step={step:4d}/{cfg['steps']} | "
                f"loss={loss_val:.4f} | ppl={ppl_val:.4f} | "
                f"lr={lr:.2e} | {elapsed:.0f}s"
            )

        # Routing confidence
        if step % cfg['conf_every'] == 0:
            with torch.no_grad():
                conf = model.get_routing_confidence(
                    model.base.drop(
                        model.base.embed(x) + model.base.pos_embed(torch.arange(x.shape[1], device=device))
                    )
                )
            log['routing_confidences'].append({'step': step, 'conf': round(conf, 4)})
            # Распределение по экспертам
            avg_weights = routing_weights.mean(dim=(0, 1))  # (6,)
            experts_str = " ".join(
                f"{KasatkinQ6Router.EXPERT_NAMES[i]}:{avg_weights[i]:.2f}"
                for i in range(6)
            )
            print(f"  [Routing step={step}] conf={conf:.4f} | {experts_str}")

        if step % cfg['save_every'] == 0:
            ckpt = Path('experiments') / f'kasatkin_router_step{step}.pt'
            torch.save({'step': step, 'model_state': model.state_dict(),
                        'loss': loss_val, 'config': cfg}, ckpt)

    final_ppl = log['ppls'][-1] if log['ppls'] else float('inf')
    final_conf = (log['routing_confidences'][-1]['conf']
                  if log['routing_confidences'] else 0.0)
    total_time = time.time() - t0

    print(f"\n{'='*64}")
    print(f"  РЕЗУЛЬТАТ kasatkin_as_router")
    print(f"{'='*64}")
    print(f"  Final PPL         : {final_ppl:.4f}")
    print(f"  Routing confidence: {final_conf:.4f} (цель: > {cfg['target_confidence']})")
    print(f"  Baseline PPL      : {f'{baseline_ppl:.4f}' if baseline_ppl else 'N/A'}")

    verdict = "inconclusive"
    if cfg['steps'] < 3000:
        print(f"  ⚠  < 3000 шагов — вердикт недействителен")
    else:
        conf_ok = final_conf > cfg['target_confidence']
        ppl_ok = (baseline_ppl is None) or (final_ppl <= baseline_ppl * 1.02)  # допуск 2%
        if conf_ok and ppl_ok:
            print(f"  ✓  УСПЕХ: conf={final_conf:.4f} > {cfg['target_confidence']} И PPL в норме")
            verdict = "proven"
        elif not conf_ok:
            print(f"  ✗  Routing confidence недостаточна: {final_conf:.4f}")
            verdict = "disproven"
        else:
            print(f"  ✗  PPL хуже baseline: {final_ppl:.4f} > {baseline_ppl:.4f}")
            verdict = "disproven"

    final_ckpt = Path('experiments') / 'kasatkin_router_final.pt'
    torch.save({'step': cfg['steps'], 'model_state': model.state_dict(),
                'final_ppl': final_ppl, 'routing_confidence': final_conf,
                'verdict': verdict, 'config': cfg}, final_ckpt)

    log.update({'final_ppl': final_ppl, 'routing_confidence': final_conf,
                'verdict': verdict, 'total_time_s': round(total_time, 1)})
    log_path = Path('experiments') / 'kasatkin_router_log.json'
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"  Лог: {log_path}")

    return final_ppl, final_conf, verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=DEFAULTS['steps'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.dry_run:
        cfg['steps'] = 50
        cfg['log_every'] = 10
        cfg['conf_every'] = 25
        cfg['save_every'] = 9999

    train(cfg)
