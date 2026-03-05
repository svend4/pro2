"""
Сравнительный бенчмарк: YiJing vs Vanilla Transformer.

Варианты сравнения:
1. YiJing (полный) vs Vanilla — влияние геометрии
2. Ablation: RoPE, SwiGLU, BianGua, MoE — вклад каждого компонента
3. Адаптивная vs фиксированная температура

Использование:
    python scripts/benchmark.py
    python scripts/benchmark.py --steps 2000 --device cuda
    python scripts/benchmark.py --ablation
"""

import argparse
import sys
import os
import time
import math
import json

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT
from models.baseline import VanillaGPT


def generate_synthetic_batch(batch_size, block_size, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, block_size + 1), device=device)
    for i in range(batch_size):
        if i % 3 == 0:
            src_start = torch.randint(0, block_size // 2, (1,)).item()
            src_len = min(block_size // 4, block_size - src_start)
            dst_start = src_start + block_size // 2
            if dst_start + src_len <= block_size + 1:
                x[i, dst_start:dst_start + src_len] = x[i, src_start:src_start + src_len]
    return x[:, :-1], x[:, 1:]


def get_lr(step, warmup, total, lr):
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(model, name, cfg, steps, device):
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
        _, loss, _ = model(x, y)
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
        'final_loss': sum(losses[-50:]) / min(50, len(losses)),
        'time_s': elapsed,
        'steps_per_sec': steps / elapsed,
        'losses': losses,
    }


def measure_hex_contribution(model):
    contributions = {}
    for i, layer in enumerate(model.core.layers):
        info = {
            'hex_scale': layer.hex_scale.item(),
            'mean_head_scale': layer.attn.head_scales.data.abs().mean().item(),
        }
        if hasattr(layer.quantizer, 'log_temp'):
            info['temp'] = layer.quantizer.current_temp.item()
        if layer.bian_gua is not None:
            info['bian_gua_scale'] = layer.bian_gua.scale.item()
            info['change_probs'] = [round(p, 3) for p in
                                     torch.sigmoid(layer.bian_gua.change_logits).tolist()]
        contributions[f'layer_{i}'] = info
    return contributions


def run_single(name, cfg, steps, device):
    """Обучает одну модель и возвращает результаты."""
    print(f"\n--- {name} ---")
    if 'Vanilla' in name:
        model = VanillaGPT(cfg).to(device)
    else:
        model = YiJingGPT(cfg).to(device)

    total, hex_params = model.count_parameters()
    print(f"  Parameters: {total:,}" +
          (f" (YiJing: {hex_params:,}, {100*hex_params/total:.1f}%)" if hex_params else ""))

    results = train_model(model, name, cfg, steps, device)
    results['params'] = total
    results['hex_params'] = hex_params

    if hasattr(model, 'core'):
        results['geometry'] = measure_hex_contribution(model)

    return results


def print_comparison(results_dict):
    """Красивый вывод сравнительной таблицы."""
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    names = list(results_dict.keys())
    header = f"{'Metric':<25}" + "".join(f"{n:>15}" for n in names)
    print(header)
    print("-" * 70)

    for metric in ['params', 'final_loss', 'time_s', 'steps_per_sec']:
        row = f"{metric:<25}"
        for name in names:
            val = results_dict[name].get(metric, 'N/A')
            if isinstance(val, float):
                row += f"{val:>15.4f}"
            elif isinstance(val, int):
                row += f"{val:>15,}"
            else:
                row += f"{val:>15}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='YiJing vs Vanilla benchmark')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--ablation', action='store_true', default=False,
                        help='Run ablation study')
    parser.add_argument('--quantizers', action='store_true', default=False,
                        help='Compare all quantizer types (incl. E8)')
    parser.add_argument('--gqa', action='store_true', default=False,
                        help='Compare GQA configurations')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Save results to JSON')
    args = parser.parse_args()

    device = torch.device(args.device)

    base_cfg = dict(
        vocab_size=512,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=8,
        block_size=args.block_size,
        batch_size=4,
        warmup_steps=100,
    )

    results = {}

    if args.ablation:
        # Ablation study
        configs = {
            'YiJing (full)': dict(use_rope=True, use_swiglu=True, use_bian_gua=True, adaptive_temp=True),
            'No RoPE': dict(use_rope=False, use_swiglu=True, use_bian_gua=True, adaptive_temp=True),
            'No SwiGLU': dict(use_rope=True, use_swiglu=False, use_bian_gua=True, adaptive_temp=True),
            'No BianGua': dict(use_rope=True, use_swiglu=True, use_bian_gua=False, adaptive_temp=True),
            'Fixed temp': dict(use_rope=True, use_swiglu=True, use_bian_gua=True, adaptive_temp=False),
            'Vanilla': dict(use_rope=True, use_swiglu=True, use_bian_gua=False, adaptive_temp=False),
        }

        for name, overrides in configs.items():
            cfg = YiJingConfig(**base_cfg, **overrides)
            results[name] = run_single(name, cfg, args.steps, device)

    elif args.quantizers:
        # Сравнение всех квантизаторов
        configs = {
            'Factored 6D': dict(quantizer_type='factored6', quant_total_dim=6, use_bian_gua=True),
            'Hierarchical 6D': dict(quantizer_type='hierarchical', quant_total_dim=6, quant_group_dim=2, use_bian_gua=True),
            'Octogram 8D': dict(quantizer_type='octogram', quant_total_dim=8, use_bian_gua=True),
            'E8 (240 roots)': dict(quantizer_type='e8', quant_total_dim=8, use_bian_gua=True),
            'Gumbel 6D': dict(quantizer_type='gumbel', quant_total_dim=6, quant_group_dim=3, use_bian_gua=True),
            'Deformable 6D': dict(quantizer_type='deformable', quant_total_dim=6, quant_group_dim=3, use_bian_gua=True),
        }

        for name, overrides in configs.items():
            cfg = YiJingConfig(**base_cfg, use_rope=True, use_swiglu=True,
                               adaptive_temp=True, **overrides)
            results[name] = run_single(name, cfg, args.steps, device)

        # Добавляем Vanilla для сравнения
        cfg_vn = YiJingConfig(**base_cfg, use_rope=True, use_swiglu=True)
        results['Vanilla'] = run_single('Vanilla', cfg_vn, args.steps, device)

    elif args.gqa:
        # Сравнение GQA конфигураций
        for n_kv in [None, 4, 2, 1]:
            name = f"MHA" if n_kv is None else f"GQA-{n_kv}kv"
            cfg = YiJingConfig(**base_cfg, use_rope=True, use_swiglu=True,
                               use_bian_gua=True, adaptive_temp=True,
                               n_kv_heads=n_kv)
            results[name] = run_single(name, cfg, args.steps, device)

    else:
        # Стандартное сравнение
        cfg_yj = YiJingConfig(**base_cfg, use_rope=True, use_swiglu=True,
                               use_bian_gua=True, adaptive_temp=True)
        results['YiJing'] = run_single('YiJing', cfg_yj, args.steps, device)

        cfg_vn = YiJingConfig(**base_cfg, use_rope=True, use_swiglu=True)
        results['Vanilla'] = run_single('Vanilla', cfg_vn, args.steps, device)

    print_comparison(results)

    # YiJing geometry analysis
    for name, res in results.items():
        if 'geometry' in res:
            print(f"\n{name} — Geometry Contribution:")
            for layer_name, vals in res['geometry'].items():
                parts = [f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.4f}"
                         for k, v in vals.items()]
                print(f"  {layer_name}: {', '.join(parts)}")

    # Save
    if args.save_results:
        save_data = {}
        for name, res in results.items():
            save_data[name] = {k: v for k, v in res.items()
                               if k not in ('losses',)}
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
