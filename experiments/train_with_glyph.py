#!/usr/bin/env python3
"""
train_with_glyph.py — Тренировка с SOLAN GlyphTokenizer.

Переключает bidir_train_v2.py с CharTokenizer на GlyphTokenizer.
Тестирует: улучшает ли геометрическая токенизация качество представлений.

Запуск:
    python experiments/train_with_glyph.py --steps 200 --fast
"""

import sys
import json
import random
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_glyph_tokenizer():
    """Загрузить GlyphTokenizer если доступен."""
    try:
        from yijing_transformer.tokenizer.glyph_tokenizer import GlyphTokenizer
        tok = GlyphTokenizer()
        print(f"GlyphTokenizer загружен ({tok.vocab_size} символов)")
        return tok
    except Exception as e:
        print(f"GlyphTokenizer недоступен ({e}), используется char-fallback")
        return None


class GlyphAwareEmbedding(nn.Module):
    """
    Embedding слой с Q6-геометрией из SOLAN.
    Каждый токен получает стандартный embedding + Q6-позицию.
    """

    def __init__(self, vocab_size: int, d_model: int, tokenizer=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokenizer  = tokenizer
        self.std_embed  = nn.Embedding(vocab_size, d_model)

        # Q6-проектор: 6 бит → d_model (фиксированный, не обучается)
        q6_basis = torch.randn(6, d_model) / (6 ** 0.5)
        self.register_buffer('q6_basis', q6_basis)

        # Hamming bias: близкие токены → ближе в attention
        self.hamming_scale = nn.Parameter(torch.tensor(0.1))

    def encode_to_q6(self, text: str) -> torch.Tensor:
        """Текст → Q6-матрица (T, 6)."""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text).float()
        else:
            # Char-level fallback: псевдо-Q6 через биты ASCII
            chars = [ord(c) % 64 for c in text]
            bits = []
            for c in chars:
                row = [1.0 if (c >> i) & 1 else -1.0 for i in range(6)]
                bits.append(row)
            return torch.tensor(bits)

    def forward(self, token_ids: torch.Tensor, text: str = None):
        """
        Args:
            token_ids: (B, T) — целочисленные индексы
            text:      опциональный текст для Q6-геометрии
        Returns:
            embeddings: (B, T, d_model)
            hamming_bias: (T, T) или None
        """
        std = self.std_embed(token_ids)  # (B, T, d_model)

        hamming_bias = None
        if text is not None:
            T = token_ids.shape[1]
            q6 = self.encode_to_q6(text[:T])  # (T, 6)

            # Гарантировать правильную длину
            if q6.shape[0] < T:
                pad = torch.zeros(T - q6.shape[0], 6)
                q6  = torch.cat([q6, pad], dim=0)
            q6 = q6[:T]

            # Q6-embedding: q6 @ q6_basis → (T, d_model)
            geo_emb = q6.to(std.device) @ self.q6_basis  # (T, d_model)
            std = std + geo_emb.unsqueeze(0)  # broadcast over batch

            # Hamming bias matrix для attention
            diff = (q6.unsqueeze(0) != q6.unsqueeze(1)).float().sum(-1)  # (T, T)
            hamming_bias = -self.hamming_scale * diff  # (T, T)

        return std, hamming_bias


def run_comparison(steps: int = 200, fast: bool = False):
    """
    Сравнить обучение с CharTokenizer vs GlyphTokenizer.
    Метрика: косинусная близость между семантически близкими концептами.
    """
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ: CharTokenizer vs GlyphTokenizer")
    print("=" * 60)

    tok = load_glyph_tokenizer()

    # Тестовые пары (должны быть близкими)
    concept_pairs = [
        ("crystal structure", "lattice symmetry"),
        ("gradient descent",  "backpropagation"),
        ("hexagram pattern",  "binary sequence"),
    ]

    d_model   = 64
    vocab_size = 256
    results = {}

    for mode in ['char', 'glyph']:
        use_glyph = (mode == 'glyph' and tok is not None)
        print(f"\n  Режим: {mode.upper()}")

        embed = GlyphAwareEmbedding(vocab_size, d_model,
                                    tokenizer=(tok if use_glyph else None))
        proj  = nn.Linear(d_model, d_model)
        opt   = torch.optim.Adam(
            list(embed.parameters()) + list(proj.parameters()), lr=1e-3
        )

        n_steps = steps // 2 if fast else steps
        for step in range(n_steps):
            a_text, b_text = random.choice(concept_pairs)
            a_ids = torch.tensor(
                [[min(ord(c), 255) for c in a_text[:16]]], dtype=torch.long
            )
            b_ids = torch.tensor(
                [[min(ord(c), 255) for c in b_text[:16]]], dtype=torch.long
            )

            a_emb, _ = embed(a_ids, a_text if use_glyph else None)
            b_emb, _ = embed(b_ids, b_text if use_glyph else None)

            a_h = proj(a_emb.mean(dim=1))
            b_h = proj(b_emb.mean(dim=1))

            # Contrastive: близкие → высокое cosine
            sim  = F.cosine_similarity(a_h, b_h)
            loss = 1 - sim.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % max(n_steps // 5, 1) == 0:
                print(f"    Step {step:4d}: loss={loss.item():.4f} sim={sim.item():.4f}")

        # Финальная оценка
        embed.eval()
        sims = []
        for a_text, b_text in concept_pairs:
            a_ids = torch.tensor(
                [[min(ord(c), 255) for c in a_text[:16]]], dtype=torch.long
            )
            b_ids = torch.tensor(
                [[min(ord(c), 255) for c in b_text[:16]]], dtype=torch.long
            )
            with torch.no_grad():
                a_emb, _ = embed(a_ids, a_text if use_glyph else None)
                b_emb, _ = embed(b_ids, b_text if use_glyph else None)
                a_h = proj(a_emb.mean(1))
                b_h = proj(b_emb.mean(1))
                sims.append(F.cosine_similarity(a_h, b_h).item())

        avg_sim = sum(sims) / len(sims)
        results[mode] = avg_sim
        print(f"  Средняя cosine similarity: {avg_sim:.4f}")

    # Итог
    print("\n" + "=" * 60)
    char_score  = results.get('char', 0)
    glyph_score = results.get('glyph', 0)
    improvement = glyph_score - char_score
    print(f"  char:  {char_score:.4f}")
    print(f"  glyph: {glyph_score:.4f}")
    print(f"  delta: {improvement:+.4f}")

    if improvement > 0.02:
        print("  [OK] GlyphTokenizer улучшает семантические представления")
    elif improvement > -0.02:
        print("  [~]  Разница незначительна — нужно больше шагов")
    else:
        print("  [!!] GlyphTokenizer ухудшает результат на этих данных")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    results = run_comparison(steps=args.steps, fast=args.fast)

    out = ROOT / 'experiments' / 'glyph_comparison.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nРезультат: {out}")
