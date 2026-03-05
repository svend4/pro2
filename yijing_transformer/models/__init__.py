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
from .lora import (
    LoRALinear,
    apply_lora,
    freeze_non_lora,
    unfreeze_all,
    merge_lora,
    unmerge_lora,
    count_lora_parameters,
    save_lora_weights,
    load_lora_weights,
)
from .speculative import (
    build_draft_model,
    speculative_generate,
    measure_acceptance_rate,
)
