"""
Полная демонстрация YiJing-Transformer: train → eval → generate → export.

Показывает весь пайплайн:
1. Создание модели из preset
2. Обучение на синтетическом тексте с LLRD и cosine schedule
3. Quantization analytics
4. Генерация текста (sampling, beam search)
5. Model card
6. Save/load pretrained

Использование:
    python scripts/full_demo.py
    python scripts/full_demo.py --preset small --steps 500
    python scripts/full_demo.py --text-file my_corpus.txt --steps 2000
"""

import argparse
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT
from models.export import create_model_card
from tokenizer.char_tokenizer import CharTokenizer, ByteTokenizer
from training.optim import build_optimizer, get_cosine_schedule


DEMO_TEXT = """\
The I Ching, or Book of Changes, uses a binary system of broken and unbroken
lines to represent the fundamental forces of nature. Each hexagram consists of
six lines arranged in two trigrams. The eight trigrams represent Heaven, Earth,
Water, Fire, Thunder, Wind, Mountain, and Lake.

In our transformer architecture, we map these ancient geometric structures to
modern deep learning. Each token passes through a quantization bottleneck that
projects to the vertices of a hypercube {-1,+1}^6, creating 64 discrete states
that mirror the 64 hexagrams.

The factored quantization uses two independent softmax operations over the 8
trigrams, reducing computational complexity from O(64) to O(16). This geometric
regularization provides an inductive bias that encourages structured latent
representations.

Transformations between hexagrams are called bian gua. In our model, learned
coordinate flips in the quantized space enable the network to explore the
geometric structure of the hypercube during training, moving between related
hexagram states.

The E8 lattice with 240 roots in 8 dimensions offers an alternative geometric
structure. While the hypercube {-1,+1}^8 has 256 vertices, E8 provides denser
packing but lacks the factorization structure of the hexagram system.
""".strip()


def main():
    parser = argparse.ArgumentParser(description='YiJing-Transformer Full Demo')
    parser.add_argument('--preset', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'])
    parser.add_argument('--text-file', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='char',
                        choices=['char', 'byte'])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--llrd', type=float, default=0.9,
                        help='Layer-wise LR decay factor')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save', type=str, default=None,
                        help='Save model to path')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 60)
    print("YiJing-Transformer Full Demo")
    print("=" * 60)

    # === 1. Текст и токенизатор ===
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"\nLoaded {len(text)} chars from {args.text_file}")
    else:
        text = DEMO_TEXT * 5
        print(f"\nUsing synthetic text ({len(text)} chars)")

    if args.tokenizer == 'byte':
        tokenizer = ByteTokenizer()
        print(f"Tokenizer: ByteTokenizer (vocab={tokenizer.get_piece_size()})")
    else:
        tokenizer = CharTokenizer.from_text(text)
        print(f"Tokenizer: CharTokenizer (vocab={tokenizer.get_piece_size()})")

    token_ids = tokenizer.encode(text)
    vocab_size = tokenizer.get_piece_size()
    print(f"Text: {len(token_ids)} tokens")

    # === 2. Модель ===
    preset_fn = getattr(YiJingConfig, args.preset)
    cfg = preset_fn(
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        warmup_steps=min(50, args.steps // 5),
        total_steps=args.steps,
        lr=3e-3,
    )

    # Подгоняем block_size под текст
    cfg.block_size = min(cfg.block_size, len(token_ids) // 2)

    model = YiJingGPT(cfg).to(device)
    total_params, hex_params = model.count_parameters()
    print(f"\nModel: {args.preset} preset")
    print(f"  d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}")
    print(f"  Parameters: {total_params:,} (YiJing: {hex_params:,})")
    print(f"  FLOPS: {model.estimate_flops_str(cfg.block_size)}")

    # === 3. Обучение с LLRD ===
    print(f"\n--- Training ({args.steps} steps, LLRD={args.llrd}) ---")
    optimizer = build_optimizer(model, cfg, llrd_factor=args.llrd)

    data = torch.tensor(token_ids, dtype=torch.long, device=device)
    n = len(data) - cfg.block_size - 1

    model.train()
    losses = []
    start_time = time.time()

    for step in range(1, args.steps + 1):
        lr = get_cosine_schedule(step, cfg.warmup_steps, cfg.total_steps, cfg.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr * (pg['lr'] / cfg.lr) if cfg.lr > 0 else lr

        ix = torch.randint(0, n, (cfg.batch_size,))
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])

        _, loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % max(1, args.steps // 5) == 0:
            avg = sum(losses[-50:]) / min(50, len(losses))
            elapsed = time.time() - start_time
            print(f"  Step {step}/{args.steps}: loss={avg:.4f}, "
                  f"lr={lr:.6f}, {step/elapsed:.1f} steps/s")

    elapsed = time.time() - start_time
    final_loss = sum(losses[-50:]) / min(50, len(losses))
    print(f"\nTraining done: {elapsed:.1f}s, final loss={final_loss:.4f}")

    # === 4. Quantization Analytics ===
    print("\n--- Quantization Analytics ---")
    model.eval()
    qa = model.quantization_analytics()
    for layer_name, info in qa.items():
        parts = []
        for k, v in info.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(f"  {layer_name}: {', '.join(parts)}")

    # === 5. Генерация текста ===
    print("\n--- Text Generation ---")
    prompts = ["The ", "Each hexagram ", "In the "]

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        # Sampling
        out_sample = model.generate(idx.clone(), max_new_tokens=100,
                                     temperature=0.8, top_k=30, top_p=0.9,
                                     repetition_penalty=1.2)
        text_sample = tokenizer.decode(out_sample[0].tolist()[len(ids):])

        # Beam search
        out_beam = model.beam_search(idx.clone(), max_new_tokens=50,
                                      beam_width=3, temperature=0.7)
        text_beam = tokenizer.decode(out_beam[0].tolist()[len(ids):])

        print(f"\n  Prompt: '{prompt}'")
        print(f"  [Sampling] {prompt}{text_sample[:150]}")
        print(f"  [Beam]     {prompt}{text_beam[:150]}")

    # === 6. Model Card ===
    print("\n--- Model Card ---")
    card = create_model_card(model)
    for section, data in card.items():
        if isinstance(data, dict):
            print(f"  {section}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {section}: {data}")

    # === 7. Save ===
    if args.save:
        model.save_pretrained(args.save)
        print(f"\nModel saved to {args.save}")

        # Verify load
        model2 = YiJingGPT.from_pretrained(args.save, device=str(device))
        print(f"Model loaded and verified: {model2.cfg.d_model}d, {model2.cfg.n_layers}L")

    print("\nDone!")


if __name__ == "__main__":
    main()
