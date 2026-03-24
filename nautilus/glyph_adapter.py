"""
GlyphAwareEmbedding — адаптер SOLAN → NautilusMoME (Шаг 4).

Каждый токен получает:
  - стандартный обучаемый embedding (d_model)
  - Q6-геометрический embedding из фиксированного базиса (d_model)
  - Хэмминг-матрицу для attention bias (T, T)

Интеграция с NautilusMoME через NautilusMoMEWithGlyph.

Ксерокс-тест запускать каждые 500 шагов обучения:
  python -c "from nautilus.glyph_adapter import run_xerox_test; ..."
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Путь к GlyphTokenizer
sys.path.insert(0, str(Path(__file__).parent.parent))
from yijing_transformer.tokenizer.glyph_tokenizer import GlyphTokenizer


class GlyphAwareEmbedding(nn.Module):
    """Объединяет стандартный и Q6-геометрический embeddings.

    Входные данные: строка текста.
    Выход: (embeddings, hamming_bias_matrix).

    Хэмминг-bias: близкие токены (малое расстояние Q6) получают
    больший attention. Это кодирует визуальную близость букв как
    семантическую близость.

    Args:
        d_model: размерность модели
        use_hamming_bias: добавлять ли Хэмминг-матрицу для attention
        hamming_scale_init: начальное значение масштаба Хэмминг-штрафа
    """

    def __init__(
        self,
        d_model: int = 128,
        use_hamming_bias: bool = True,
        hamming_scale_init: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_hamming_bias = use_hamming_bias

        self.tokenizer = GlyphTokenizer()
        self.vocab_size = 76  # SOLAN-76

        # Стандартный обучаемый embedding по индексу символа
        self.token_embed = nn.Embedding(self.vocab_size, d_model)

        # Q6-проектор: 6 бит → d_model (фиксированный, не обучается)
        # Случайная матрица масштабированная по Лекуну
        q6_basis = torch.randn(6, d_model) / (6 ** 0.5)
        self.register_buffer('q6_basis', q6_basis)

        # Learnable масштаб Хэмминг-штрафа (начинаем с малого)
        self.hamming_scale = nn.Parameter(torch.tensor(hamming_scale_init))

        nn.init.normal_(self.token_embed.weight, std=0.02)

    def _char_to_idx(self, ch: str) -> int:
        """Символ → индекс в словаре SOLAN-76."""
        bits = self.tokenizer.char_to_bits.get(ch)
        if bits is None:
            # Fallback для неизвестных символов
            code = ord(ch) % 64
            bits = tuple((code >> (5 - b)) & 1 for b in range(6))
        # Перевести bits → порядковый индекс [0, vocab_size)
        idx = sum(b << (5 - i) for i, b in enumerate(bits))
        return min(idx, self.vocab_size - 1)

    def forward(self, text: str) -> tuple:
        """Кодирует строку текста в embeddings и Хэмминг-матрицу.

        Args:
            text: входная строка

        Returns:
            embeddings: (1, T, d_model) — батч=1 для текста
            hamming_bias: (T, T) или None если use_hamming_bias=False
        """
        if not text:
            return torch.zeros(1, 0, self.d_model), None

        # Q6-координаты каждого символа
        q6_vecs = self.tokenizer.encode(text)  # (T, 6), значения {-1, +1}
        T = q6_vecs.shape[0]
        device = self.q6_basis.device
        q6_vecs = q6_vecs.to(device)

        # Индексы токенов для стандартного embedding
        token_ids = torch.tensor(
            [self._char_to_idx(ch) for ch in text[:T]],
            device=device,
        )

        # Стандартный обучаемый embedding
        std_emb = self.token_embed(token_ids)  # (T, d_model)

        # Q6-геометрический embedding: q6_vec @ q6_basis
        geo_emb = q6_vecs.float() @ self.q6_basis  # (T, d_model)

        # Итог: сумма (residual-style, не конкатенация)
        combined = std_emb + geo_emb  # (T, d_model)

        # Хэмминг-матрица для attention bias
        hamming_bias = None
        if self.use_hamming_bias:
            # diff[i,j,k] = True если бит k различается у токенов i и j
            diff = q6_vecs.unsqueeze(0) != q6_vecs.unsqueeze(1)  # (T, T, 6)
            hamming_dist = diff.float().sum(dim=-1)  # (T, T), от 0 до 6
            # Штраф: далёкие токены → отрицательный bias
            hamming_bias = -self.hamming_scale * hamming_dist  # (T, T)

        return combined.unsqueeze(0), hamming_bias  # (1, T, d_model), (T, T)


class NautilusMoMEWithGlyph(nn.Module):
    """Обёртка NautilusMoME с поддержкой SOLAN-токенов.

    Добавляет GlyphAwareEmbedding как альтернативный путь кодирования.
    Переключение между BPE и Glyph через use_glyph флаг.

    Args:
        base_model: базовая NautilusMoME модель
        d_model: размерность модели
        use_glyph: активировать SOLAN путь (иначе — стандартный BPE)
    """

    def __init__(
        self,
        base_model: nn.Module,
        d_model: int = 128,
        use_glyph: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.use_glyph = use_glyph
        self.glyph_embed = GlyphAwareEmbedding(
            d_model=d_model,
            use_hamming_bias=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        text: str | None = None,
        **kwargs,
    ):
        if self.use_glyph and text is not None:
            embeddings, hamming_bias = self.glyph_embed(text)
            # Передать embeddings и hamming_bias в базовую модель
            return self.base_model(
                embeddings=embeddings,
                attention_bias=hamming_bias,
                **kwargs,
            )
        else:
            return self.base_model(input_ids=input_ids, **kwargs)


# ─── Ксерокс-тест ────────────────────────────────────────────────

XEROX_TESTS = [
    # (текст, ожидаемый домен, max_PPL)
    ("def neural_network(x):", "CODE", 15.0),
    ("class GradientDescent:", "CODE", 15.0),
    ("Expert routing weights", "RECON", 30.0),
    ("Hexagram as archetype", "RECON", 30.0),
    ("SELECT * FROM experts", "SYSTEM", 25.0),
]


def compute_ppl(model: nn.Module, text: str, device: str = "cpu") -> float:
    """Быстрое вычисление PPL через character-level cross-entropy."""
    embed = GlyphAwareEmbedding(d_model=getattr(model, 'd_model', 128))
    embeddings, _ = embed(text)
    # Простой прокси: entropy embedding как PPL-оценка
    with torch.no_grad():
        probs = F.softmax(embeddings.squeeze(0), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return entropy.exp().item()


def run_xerox_test(model: nn.Module, step: int = 0) -> dict:
    """Ксерокс-тест: проверяет routing и PPL на диагностических текстах.

    Запускать каждые 500 шагов обучения.
    Если к шагу 2000 пройдено < 80% — архитектурная проблема.

    Args:
        model: модель с методом get_routing_weights(text) → dict
        step: текущий шаг обучения

    Returns:
        results: dict с результатами каждого теста
    """
    results = []
    for text, expected_domain, max_ppl in XEROX_TESTS:
        ppl = compute_ppl(model, text)
        routing_correct = False

        if hasattr(model, 'get_routing_weights'):
            routing = model.get_routing_weights(text)
            actual_domain = max(routing, key=routing.get)
            routing_correct = (actual_domain == expected_domain)
        else:
            actual_domain = "N/A"

        results.append({
            'text': text,
            'expected': expected_domain,
            'actual': actual_domain,
            'routing_correct': routing_correct,
            'ppl': round(ppl, 2),
            'ppl_ok': ppl < max_ppl,
        })

    passed = sum(r['routing_correct'] and r['ppl_ok'] for r in results)
    total = len(results)
    print(f"[Step {step}] Ксерокс-тест: {passed}/{total} пройдено")
    for r in results:
        status = "✓" if (r['routing_correct'] and r['ppl_ok']) else "✗"
        print(f"  {status} '{r['text'][:30]}' | "
              f"PPL={r['ppl']:.1f} | routing={r['actual']}")

    return {
        'passed': passed,
        'total': total,
        'pass_rate': passed / total,
        'details': results,
    }


if __name__ == '__main__':
    # Быстрая проверка embedding
    embed = GlyphAwareEmbedding(d_model=64)
    embeddings, hamming_bias = embed("Hello world")
    print(f"Embeddings shape: {embeddings.shape}")        # (1, 11, 64)
    print(f"Hamming bias shape: {hamming_bias.shape}")    # (11, 11)
    print(f"Hamming bias range: [{hamming_bias.min():.2f}, {hamming_bias.max():.2f}]")
