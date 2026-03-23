"""
Визуализация attention паттернов и геометрических bias'ов YiJing-Transformer.

Показывает:
1. Attention weights по слоям и головам
2. Геометрический bias (триграммный вклад)
3. Head scales — какие головы используют геометрию
4. E8 vs Hypercube сравнение (точки в пространстве)

Использование:
    python scripts/visualize_attention.py
    python scripts/visualize_attention.py --checkpoint model.pt --save-dir plots/
"""

import sys
import os
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT
from models.geometry import (
    generate_trigrams, generate_hexagrams, generate_e8_roots,
    compare_e8_vs_hypercube,
)


def extract_attention_weights(model, x):
    """Извлекает attention weights из каждого слоя (без gradient)."""
    model.eval()
    attention_maps = []
    geo_biases = []

    # Hook для захвата attention
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Перевычисляем attention для анализа
            attn = module
            B, T, C = input[0].shape
            q = attn.q_proj(input[0]).reshape(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(input[0]).reshape(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)

            if attn.use_rope:
                cos, sin = attn.rotary(T)
                from models.geometry import apply_rotary_emb
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)

            k = attn._repeat_kv(k)
            scores = (q @ k.transpose(-2, -1)) * attn.scale

            # Causal mask
            causal = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attention_maps.append(attn_weights.detach().cpu())

            # Geometric bias
            if attn.head_dim >= 3:
                q3 = q[..., :3]
                k3 = k[..., :3]
                q_proj = torch.einsum('bhtd,hd->bht', q3, attn.head_dirs)
                k_proj = torch.einsum('bhtd,hd->bht', k3, attn.head_dirs)
                geo = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
                geo_biases.append({
                    'bias': geo.detach().cpu(),
                    'scales': attn.head_scales.detach().cpu(),
                })
        return hook_fn

    for i, layer in enumerate(model.core.layers):
        h = layer.attn.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return attention_maps, geo_biases


def print_attention_analysis(model, attention_maps, geo_biases):
    """Текстовая визуализация attention паттернов."""
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS")
    print("=" * 60)

    for i, (attn_map, geo) in enumerate(zip(attention_maps, geo_biases)):
        # attn_map: (B, H, T, T)
        B, H, T, _ = attn_map.shape

        print(f"\n--- Layer {i} ---")
        print(f"  Attention shape: {attn_map.shape}")

        # Entropy per head (high = uniform, low = peaked)
        entropy = -(attn_map * (attn_map + 1e-10).log()).sum(-1).mean(dim=(0, 2))
        for h in range(H):
            scale = geo['scales'][h].item()
            print(f"  Head {h}: entropy={entropy[h]:.3f}, "
                  f"geo_scale={scale:.4f}")

    # Summary
    print(f"\n--- Summary ---")
    all_scales = torch.cat([g['scales'] for g in geo_biases])
    active = (all_scales.abs() > 0.01).sum().item()
    print(f"  Active geometric heads: {active}/{len(all_scales)}")
    print(f"  Mean |geo_scale|: {all_scales.abs().mean():.4f}")
    print(f"  Max  |geo_scale|: {all_scales.abs().max():.4f}")


def print_e8_comparison():
    """Сравнение E8 и гиперкубов."""
    print("\n" + "=" * 60)
    print("E8 vs HYPERCUBE COMPARISON")
    print("=" * 60)

    results = compare_e8_vs_hypercube()
    header = f"{'Metric':<20}" + "".join(f"{n:>22}" for n in results.keys())
    print(header)
    print("-" * (20 + 22 * len(results)))

    for metric in ['n_points', 'dim', 'norm', 'min_dist', 'mean_dist', 'max_dist', 'bits']:
        row = f"{metric:<20}"
        for name in results:
            val = results[name][metric]
            if isinstance(val, float):
                row += f"{val:>22.4f}"
            else:
                row += f"{val:>22}"
        print(row)

    # Отношение softmax операций
    print(f"\n{'Softmax complexity':<20}", end="")
    complexities = {
        'E8 (240 roots)': '240 (brute)',
        'Hexagrams {-1,+1}⁶': '2×8=16 (factored)',
        'Octograms {-1,+1}⁸': '2×16=32 (factored)',
    }
    for name in results:
        print(f"{complexities.get(name, '?'):>22}", end="")
    print()


def print_quantizer_analysis(model):
    """Анализ квантизаторов по слоям."""
    if not hasattr(model, 'quantization_analytics'):
        return
    print("\n" + "=" * 60)
    print("QUANTIZER ANALYSIS")
    print("=" * 60)

    analytics = model.quantization_analytics()
    for name, info in analytics.items():
        print(f"\n  {name}:")
        for k, v in info.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description='YiJing Attention Visualization')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.checkpoint:
        with torch.serialization.safe_globals([YiJingConfig]):
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        cfg = ckpt['config']
        model = YiJingGPT(cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from step {ckpt['step']}")
    else:
        cfg = YiJingConfig(
            vocab_size=128, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=args.n_heads,
            block_size=64, use_rope=True, use_swiglu=True,
            use_bian_gua=True, adaptive_temp=True,
        )
        model = YiJingGPT(cfg).to(device)
        print("Using random model (no checkpoint)")

    total, hex_p = model.count_parameters()
    print(f"Parameters: {total:,} (YiJing: {hex_p:,})")

    # Извлекаем attention
    x = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=device)
    attention_maps, geo_biases = extract_attention_weights(model, x)

    # Анализ
    print_attention_analysis(model, attention_maps, geo_biases)
    print_quantizer_analysis(model)
    print_e8_comparison()

    print("\nDone!")


if __name__ == "__main__":
    main()
