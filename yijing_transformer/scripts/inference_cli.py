"""
Интерактивная генерация текста с YiJing-Transformer.

Использование:
    # С сохранённой моделью
    python scripts/inference_cli.py --checkpoint checkpoints/checkpoint_step_2000.pt

    # Быстрый тест (создаёт модель и генерирует random tokens)
    python scripts/inference_cli.py --demo

    # Параметры генерации
    python scripts/inference_cli.py --checkpoint model.pt --temp 0.8 --top-k 40 --top-p 0.95
"""

import argparse
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT


@torch.no_grad()
def generate_interactive(model, tokenizer, cfg, device,
                         temperature=0.8, top_k=50, top_p=0.9,
                         repetition_penalty=1.2, max_tokens=200):
    """Интерактивная генерация: ввод промпта → вывод продолжения."""
    model.eval()

    while True:
        try:
            prompt = input("\n>>> Prompt (или 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if prompt.lower() in ('quit', 'exit', 'q'):
            break
        if not prompt:
            continue

        # Токенизация
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            print("Пустой ввод после токенизации.")
            continue

        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Генерация
        generated = []
        for _ in range(max_tokens):
            idx_cond = context[:, -cfg.block_size:]
            logits, _, _ = model(idx_cond)
            logits = logits[0, -1, :].clone()

            # Repetition penalty
            past = context[0, -50:].tolist()
            for t_idx in set(past):
                if t_idx < logits.size(0):
                    logits[t_idx] /= repetition_penalty

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            # Top-K
            if top_k > 0:
                v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                probs[probs < v[-1]] = 0.0

            # Top-P
            if 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                probs[sorted_idx[mask]] = 0.0

            probs = probs / probs.sum()
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
            token_id = next_token.item()
            generated.append(token_id)

        # Декодирование
        text = tokenizer.decode(generated)
        print(f"\n--- Генерация ({len(generated)} токенов) ---")
        print(prompt + text)
        print("---")


def print_model_info(model, cfg):
    """Вывод информации о модели и её геометрии."""
    total, hex_params = model.count_parameters()
    print(f"\n{'='*50}")
    print(f"YiJing-Transformer Model Info")
    print(f"{'='*50}")
    print(f"Parameters: {total:,}")
    if hex_params:
        print(f"YiJing-specific: {hex_params:,} ({100*hex_params/total:.2f}%)")
    print(f"d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
    print(f"block_size={cfg.block_size}, vocab_size={cfg.vocab_size}")
    print(f"RoPE={cfg.use_rope}, SwiGLU={cfg.use_swiglu}, BianGua={cfg.use_bian_gua}")
    print(f"Quantizer: {cfg.quantizer_type} ({cfg.quant_total_dim}D)")

    # Геометрия
    if hasattr(model, 'core'):
        print(f"\nGeometry per layer:")
        for i, layer in enumerate(model.core.layers):
            info = f"  layer {i}: hex_scale={layer.hex_scale.item():.4f}"
            if hasattr(layer.quantizer, 'log_temp'):
                info += f", temp={layer.quantizer.current_temp.item():.4f}"
            if layer.bian_gua is not None:
                probs = torch.sigmoid(layer.bian_gua.change_logits)
                active = [f"{j+1}:{p:.2f}" for j, p in enumerate(probs.tolist()) if p > 0.55]
                if active:
                    info += f", active_lines=[{','.join(active)}]"
            print(info)
    print()


def main():
    parser = argparse.ArgumentParser(description='YiJing-Transformer Inference')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Quick demo with random model')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--temp', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--rep-penalty', type=float, default=1.2)
    parser.add_argument('--max-tokens', type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model = YiJingGPT(cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded from step {ckpt['step']}")

        # Пробуем загрузить SentencePiece токенизатор
        try:
            from tokenizer.tokenizer_utils import load_tokenizer
            tokenizer = load_tokenizer()
            print("Using SentencePiece tokenizer")
        except Exception:
            from tokenizer.char_tokenizer import CharTokenizer
            tokenizer = CharTokenizer()
            print("Using char-level tokenizer")

    elif args.demo:
        print("Demo mode: creating small random model...")
        cfg = YiJingConfig(
            vocab_size=128, d_model=64, n_layers=2, n_heads=4,
            block_size=64, use_rope=True, use_swiglu=True, use_bian_gua=True,
        )
        model = YiJingGPT(cfg).to(device)

        from tokenizer.char_tokenizer import CharTokenizer
        tokenizer = CharTokenizer()
        cfg.vocab_size = tokenizer.get_piece_size()
        # Пересоздаём модель с правильным vocab_size
        cfg = YiJingConfig(
            vocab_size=tokenizer.get_piece_size(),
            d_model=64, n_layers=2, n_heads=4, block_size=64,
            use_rope=True, use_swiglu=True, use_bian_gua=True,
        )
        model = YiJingGPT(cfg).to(device)
        print("Note: модель не обучена, генерация будет случайной")
    else:
        print("Укажите --checkpoint или --demo")
        return

    print_model_info(model, cfg)

    generate_interactive(
        model, tokenizer, cfg, device,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
