"""
self_train_common.py — Общая конфигурация и утилиты для self_train v1/v2/v3.

Экспортирует:
  CFG          — Variant3Config (d=128, Q6-архитектура)
  hexagrams    — (64, 6) матрица вершин Q6
  biangua      — (64, 64) матрица смежности (Хэмминг-расстояние = 1)
  evaluator    — HexagramEvaluator
  qfilter      — TextQualityFilter
  text_to_ids  — текст → байтовые токены (тензор)
"""

import torch

from yijing_transformer.models.variant3 import Variant3Config
from yijing_transformer.models.variant3_extensions import (
    HexagramEvaluator, TextQualityFilter,
    get_hexagrams, get_biangua,
)

CFG = Variant3Config(
    vocab_size=256,
    block_size=32,
    d_model=128,
    n_heads=4,
    n_layers=4,
    ffn_mult=4,
    hamming_lambda=0.15,
    uncertainty_budget=0.25,
    dropout=0.05,
    use_domain_routing=True,
)

hexagrams = get_hexagrams()   # (64, 6)
biangua   = get_biangua()     # (64, 64)
evaluator = HexagramEvaluator(threshold=0.01)
qfilter   = TextQualityFilter(CFG.d_model)


def text_to_ids(text: str, block_size: int = 32) -> torch.Tensor:
    """Кодировать текст в байтовые токены длиной block_size."""
    ids = list(text.encode("utf-8"))[:block_size] or [0]
    return torch.tensor(ids, dtype=torch.long)
