"""
Демо: обучение YiJing-Transformer на текстовом корпусе с char-level токенизатором.

Создаёт синтетический текстовый корпус (или использует переданный файл),
обучает модель, генерирует текст.

Использование:
    # Автоматический синтетический текст
    python scripts/train_text_demo.py --steps 500

    # Свой текстовый файл
    python scripts/train_text_demo.py --text-file my_corpus.txt --steps 2000

    # С визуализацией обучения
    python scripts/train_text_demo.py --steps 1000 --show-progress
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
from tokenizer.char_tokenizer import CharTokenizer


SYNTHETIC_TEXT = """
The ancient Book of Changes, known as the I Ching or Yi Jing, is one of the oldest
Chinese classical texts. It describes a system of cosmology and philosophy based on
the interplay of yin and yang, represented by broken and unbroken lines.

Each hexagram consists of six lines, either yin (broken) or yang (unbroken).
The 64 hexagrams represent all possible combinations of these six lines.
This creates a natural binary encoding: each hexagram is a vertex of the
six-dimensional hypercube {-1, +1}^6.

The eight trigrams (bagua) are the building blocks: Heaven, Earth, Water, Fire,
Thunder, Wind, Mountain, and Lake. Each trigram has three lines and represents
a fundamental force of nature.

A hexagram is composed of two trigrams: lower (inner) and upper (outer).
This factorization is key to our transformer architecture: instead of computing
distances to all 64 hexagrams (softmax over 64), we can factorize into two
independent softmax operations over 8 trigrams each.

The E8 lattice has 240 roots in 8 dimensions. Our octograms {-1,+1}^8 have
256 vertices - comparable in count but with a fundamentally different geometry.
The hypercube vertices are equidistant (Hamming geometry) while E8 roots form
a denser, more uniform packing.

Transformations between hexagrams, called bian gua (变卦), involve changing
individual lines. In our model, this corresponds to learned coordinate flips
in the quantized representation, enabling the network to explore the geometric
structure of the hypercube during training.
""".strip()


def make_batches(text_ids, block_size, batch_size, device):
    """Создаёт батчи из токенизированного текста."""
    data = torch.tensor(text_ids, dtype=torch.long, device=device)
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(f"Text too short ({len(data)} tokens) for block_size={block_size}")

    def get_batch():
        ix = torch.randint(0, n, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    return get_batch


def get_lr(step, warmup, total, lr):
    if step < warmup:
        return lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=40):
    """Генерирует текст с KV-cache."""
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [2]  # fallback
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature,
                                top_k=top_k, use_cache=True)
    return tokenizer.decode(generated[0].tolist()[len(ids):])


def main():
    parser = argparse.ArgumentParser(description='YiJing text training demo')
    parser.add_argument('--text-file', type=str, default=None)
    parser.add_argument('--model', type=str, default='yijing', choices=['yijing', 'vanilla'])
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--quantizer', type=str, default='factored6',
                        choices=['factored6', 'hierarchical', 'octogram', 'e8', 'gumbel', 'deformable'])
    parser.add_argument('--show-progress', action='store_true', default=False)
    parser.add_argument('--generate-samples', type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Текст
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text)} chars from {args.text_file}")
    else:
        text = SYNTHETIC_TEXT * 5  # Повторяем для большего объёма
        print(f"Using synthetic I Ching text ({len(text)} chars)")

    # Токенизатор из текста
    tokenizer = CharTokenizer.from_text(text)
    vocab_size = tokenizer.get_piece_size()
    text_ids = tokenizer.encode(text)
    print(f"Vocabulary: {vocab_size} tokens, text: {len(text_ids)} tokens")

    # Конфигурация
    quant_dim = 8 if args.quantizer in ('octogram', 'e8') else 6
    cfg = YiJingConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        block_size=args.block_size,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=min(100, args.steps // 5),
        total_steps=args.steps,
        use_rope=True,
        use_swiglu=True,
        use_bian_gua=True,
        adaptive_temp=True,
        quantizer_type=args.quantizer,
        quant_total_dim=quant_dim,
        weight_tying=True,
    )

    # Модель
    if args.model == 'yijing':
        model = YiJingGPT(cfg).to(device)
    else:
        model = VanillaGPT(cfg).to(device)

    total, hex_p = model.count_parameters()
    print(f"\nModel: {args.model} ({args.quantizer})")
    print(f"Parameters: {total:,}" + (f" (YiJing: {hex_p:,})" if hex_p else ""))

    # Обучение
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    get_batch = make_batches(text_ids, args.block_size, args.batch_size, device)

    model.train()
    losses = []
    start_time = time.time()

    for step in range(1, args.steps + 1):
        lr = get_lr(step, cfg.warmup_steps, cfg.total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x, y = get_batch()
        _, loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if args.show_progress and step % 50 == 0:
            avg = sum(losses[-50:]) / min(50, len(losses))
            elapsed = time.time() - start_time
            print(f"  Step {step}/{args.steps}: loss={avg:.4f}, lr={lr:.6f}, "
                  f"speed={step/elapsed:.1f} steps/s")

    elapsed = time.time() - start_time
    final_loss = sum(losses[-50:]) / min(50, len(losses))
    print(f"\nTraining done: {args.steps} steps in {elapsed:.1f}s")
    print(f"Final loss: {final_loss:.4f} (started at {losses[0]:.4f})")

    # Quantization analytics
    if args.model == 'yijing' and hasattr(model, 'quantization_analytics'):
        print("\nQuantization Analytics:")
        qa = model.quantization_analytics()
        for layer_name, info in qa.items():
            parts = []
            for k, v in info.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            print(f"  {layer_name}: {', '.join(parts)}")

    # Генерация
    if args.generate_samples > 0:
        print(f"\n{'='*50}")
        print("GENERATED TEXT SAMPLES")
        print(f"{'='*50}")
        prompts = ["The ", "Each hexagram ", "In the ancient ",
                    "The E8 ", "Transformation "][:args.generate_samples]
        for prompt in prompts:
            generated = generate_text(model, tokenizer, prompt,
                                       max_tokens=150, temperature=0.8, top_k=30)
            print(f"\n>>> {prompt}")
            print(f"{prompt}{generated}")
            print("---")


if __name__ == "__main__":
    main()
