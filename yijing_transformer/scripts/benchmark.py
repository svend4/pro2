"""
Сравнительный бенчмарк: YiJing vs Vanilla Transformer.

Запускает оба варианта на синтетических данных (без внешних зависимостей)
и сравнивает: loss, скорость, число параметров, вклад геометрии.

Использование:
    python scripts/benchmark.py
    python scripts/benchmark.py --steps 2000 --device cuda
"""

import argparse
import sys
import os
import time
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT
from models.baseline import VanillaGPT


def generate_synthetic_batch(batch_size, block_size, vocab_size, device):
    """Синтетические данные: случайные токены (для benchmark, не для качества)."""
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    return x, y


def get_lr(step, warmup, total, lr):
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(model, name, cfg, steps, device):
    """Обучает модель заданное число шагов, возвращает метрики."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    model.train()
    losses = []
    start_time = time.time()

    for step in range(1, steps + 1):
        lr = get_lr(step, cfg.warmup_steps, steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = generate_synthetic_batch(cfg.batch_size, cfg.block_size, cfg.vocab_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            print(f"  [{name}] Step {step}/{steps}: loss={avg:.4f}")

    elapsed = time.time() - start_time
    return {
        'final_loss': sum(losses[-50:]) / 50,
        'time_s': elapsed,
        'steps_per_sec': steps / elapsed,
        'losses': losses,
    }


def measure_hex_contribution(model):
    """Измеряет реальный вклад геометрических компонентов."""
    contributions = {}
    for i, layer in enumerate(model.core.layers):
        scale = layer.hex_scale.item()
        attn_scales = layer.attn.head_scales.data.abs().mean().item()
        contributions[f'layer_{i}'] = {
            'hex_scale': scale,
            'mean_head_scale': attn_scales,
        }
        if layer.bian_gua is not None:
            bg_scale = layer.bian_gua.scale.item()
            change_probs = torch.sigmoid(layer.bian_gua.change_logits).tolist()
            contributions[f'layer_{i}']['bian_gua_scale'] = bg_scale
            contributions[f'layer_{i}']['change_probs'] = [round(p, 3) for p in change_probs]
    return contributions


def main():
    parser = argparse.ArgumentParser(description='YiJing vs Vanilla benchmark')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--block_size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = YiJingConfig(
        vocab_size=512,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=8,
        block_size=args.block_size,
        batch_size=4,
        warmup_steps=100,
        use_bian_gua=True,
    )

    print("=" * 60)
    print("BENCHMARK: YiJing-Transformer vs Vanilla Transformer")
    print("=" * 60)
    print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"block_size={cfg.block_size}, steps={args.steps}")
    print()

    # --- YiJing ---
    print("[1/2] YiJing-Transformer")
    yijing_model = YiJingGPT(cfg).to(device)
    yj_total, yj_hex = yijing_model.count_parameters()
    print(f"  Parameters: {yj_total:,} (YiJing-specific: {yj_hex:,}, {100*yj_hex/yj_total:.2f}%)")
    yj_results = train_model(yijing_model, "YiJing", cfg, args.steps, device)

    print()

    # --- Vanilla ---
    print("[2/2] Vanilla Transformer")
    vanilla_model = VanillaGPT(cfg).to(device)
    vn_total, _ = vanilla_model.count_parameters()
    print(f"  Parameters: {vn_total:,}")
    vn_results = train_model(vanilla_model, "Vanilla", cfg, args.steps, device)

    print()

    # --- Вклад геометрии ---
    print("=" * 60)
    print("YiJing Geometry Contribution (после обучения)")
    print("=" * 60)
    contributions = measure_hex_contribution(yijing_model)
    for layer_name, vals in contributions.items():
        parts = [f"hex_scale={vals['hex_scale']:.4f}",
                 f"head_scale={vals['mean_head_scale']:.4f}"]
        if 'bian_gua_scale' in vals:
            parts.append(f"bian_gua={vals['bian_gua_scale']:.4f}")
            parts.append(f"changes={vals['change_probs']}")
        print(f"  {layer_name}: {', '.join(parts)}")

    print()

    # --- Итоги ---
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'YiJing':>12} {'Vanilla':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'Parameters':.<25} {yj_total:>12,} {vn_total:>12,} {yj_total-vn_total:>+12,}")
    print(f"{'Final loss':.<25} {yj_results['final_loss']:>12.4f} {vn_results['final_loss']:>12.4f} "
          f"{yj_results['final_loss']-vn_results['final_loss']:>+12.4f}")
    print(f"{'Time (sec)':.<25} {yj_results['time_s']:>12.1f} {vn_results['time_s']:>12.1f} "
          f"{yj_results['time_s']-vn_results['time_s']:>+12.1f}")
    print(f"{'Steps/sec':.<25} {yj_results['steps_per_sec']:>12.1f} {vn_results['steps_per_sec']:>12.1f}")
    overhead = (yj_results['time_s'] / vn_results['time_s'] - 1) * 100
    print(f"{'Overhead (%)':.<25} {overhead:>+12.1f}%")

    # YiJing-specific: какие линии гексаграмм модель научилась изменять?
    if cfg.use_bian_gua:
        print()
        print("变卦 Analysis (learned line change probabilities):")
        yao_names = ["初爻(1я линия)", "二爻(2я)", "三爻(3я)",
                     "四爻(4я)", "五爻(5я)", "上爻(6я)"]
        for i, layer in enumerate(yijing_model.core.layers):
            if layer.bian_gua is not None:
                probs = torch.sigmoid(layer.bian_gua.change_logits).tolist()
                active = [(name, p) for name, p in zip(yao_names, probs) if p > 0.55]
                if active:
                    active_str = ", ".join(f"{n}={p:.2f}" for n, p in active)
                    print(f"  Layer {i}: {active_str}")


if __name__ == "__main__":
    main()
