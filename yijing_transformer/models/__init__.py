from .model import YiJingGPT, YiJingTransformer
from .baseline import VanillaGPT
from .geometry import (
    generate_trigrams,
    generate_hexagrams,
    generate_octograms,
    generate_tetragrams,
    generate_hypercube,
    generate_e8_roots,
    compare_e8_vs_hypercube,
    YiJingQuantizer,
    FactoredYiJingQuantizer,
    HierarchicalQuantizer,
    DeformableQuantizer,
    GumbelQuantizer,
    E8Quantizer,
    HexagramAttentionPattern,
    RotaryEmbedding,
    SwiGLU,
    TrigramMoE,
)
