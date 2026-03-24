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
    Метрика: triplet accuracy (anchor близко к positive, далеко от negative).

    Triplet loss = max(0, sim(a, neg) - sim(a, pos) + margin) — не насыщается до 1.0.
    """
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ: CharTokenizer vs GlyphTokenizer")
    print("=" * 60)

    tok = load_glyph_tokenizer()

    # Триплеты: (anchor, positive, negative) — positive похож, negative нет
    triplets = [
        ("crystal structure",   "lattice symmetry",    "gradient descent"),
        ("gradient descent",    "backpropagation",     "hexagram pattern"),
        ("hexagram pattern",    "binary sequence",     "crystal structure"),
        ("yin yang balance",    "complementary forces", "matrix algebra"),
        ("neural network",      "deep learning",       "ancient philosophy"),
        ("recursion pattern",   "self-similar loop",   "cooking recipe"),
    ]

    d_model   = 64
    vocab_size = 256
    margin    = 0.3
    results   = {}

    n_steps = steps // 2 if fast else steps

    for mode in ['char', 'glyph']:
        use_glyph = (mode == 'glyph' and tok is not None)
        print(f"\n  Режим: {mode.upper()}")

        embed = GlyphAwareEmbedding(vocab_size, d_model,
                                    tokenizer=(tok if use_glyph else None))
        proj  = nn.Linear(d_model, d_model)
        opt   = torch.optim.Adam(
            list(embed.parameters()) + list(proj.parameters()), lr=1e-3
        )

        def encode(text):
            ids = torch.tensor(
                [[min(ord(c), 255) for c in text[:16]]], dtype=torch.long
            )
            emb, _ = embed(ids, text if use_glyph else None)
            return proj(emb.mean(dim=1))  # (1, d_model)

        for step in range(n_steps):
            anc_text, pos_text, neg_text = random.choice(triplets)

            a = encode(anc_text)
            p = encode(pos_text)
            n = encode(neg_text)

            sim_pos = F.cosine_similarity(a, p)
            sim_neg = F.cosine_similarity(a, n)
            loss = F.relu(sim_neg - sim_pos + margin).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % max(n_steps // 5, 1) == 0:
                acc = (sim_pos > sim_neg).float().mean().item()
                print(f"    Step {step:4d}: loss={loss.item():.4f}  "
                      f"sim_pos={sim_pos.item():.3f}  sim_neg={sim_neg.item():.3f}  "
                      f"acc={acc:.1f}")

        # Финальная оценка: triplet accuracy на всех парах
        embed.eval()
        correct = 0
        pos_sims = []
        neg_sims = []
        for anc_text, pos_text, neg_text in triplets:
            with torch.no_grad():
                a = encode(anc_text)
                p = encode(pos_text)
                n = encode(neg_text)
                sp = F.cosine_similarity(a, p).item()
                sn = F.cosine_similarity(a, n).item()
                pos_sims.append(sp)
                neg_sims.append(sn)
                if sp > sn:
                    correct += 1

        acc_final  = correct / len(triplets)
        margin_avg = sum(p - n for p, n in zip(pos_sims, neg_sims)) / len(triplets)
        results[mode] = {
            'accuracy': acc_final,
            'avg_pos_sim': sum(pos_sims) / len(pos_sims),
            'avg_neg_sim': sum(neg_sims) / len(neg_sims),
            'margin': margin_avg,
        }
        print(f"  Accuracy: {acc_final:.1%}  margin: {margin_avg:+.4f}")

    # Итог
    print("\n" + "=" * 60)
    char_acc   = results.get('char',  {}).get('accuracy', 0)
    glyph_acc  = results.get('glyph', {}).get('accuracy', 0)
    char_marg  = results.get('char',  {}).get('margin', 0)
    glyph_marg = results.get('glyph', {}).get('margin', 0)
    delta_acc  = glyph_acc - char_acc
    delta_marg = glyph_marg - char_marg

    print(f"  char:  accuracy={char_acc:.1%}  margin={char_marg:+.4f}")
    print(f"  glyph: accuracy={glyph_acc:.1%}  margin={glyph_marg:+.4f}")
    print(f"  delta: accuracy={delta_acc:+.1%}  margin={delta_marg:+.4f}")

    # Решение по интеграции: accuracy delta > 5% или margin delta > 0.02
    if delta_acc > 0.05 or delta_marg > 0.02:
        print("  [OK] GlyphTokenizer улучшает семантические представления → интегрировать")
        results['integrate'] = True
    elif delta_acc > -0.05 and delta_marg > -0.02:
        print("  [~]  Разница незначительна — нейтральный результат")
        results['integrate'] = None
    else:
        print("  [!!] GlyphTokenizer ухудшает результат — не интегрировать")
        results['integrate'] = False

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
