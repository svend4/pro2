"""
И-Цзин геометрия: пакет подмодулей.

Обратная совместимость: все классы и функции доступны через
`from .geometry import ...` как и раньше.

Подмодули:
- core: генерация кодбуков, порядки, структурные функции
- quantizers: все квантизаторы
- attention: attention паттерны и модули
- positional: RoPE, ALiBi, FourLevelPE
- equivariant: D4, DualEmbedding, BianGua
- routing: gates, routers, curriculum, logging
- ffn: SwiGLU, TrigramMoE, GeometricFFN, MultiScale
"""

# Core functions
from .core import (
    generate_trigrams,
    generate_hexagrams,
    generate_hypercube,
    generate_octograms,
    generate_tetragrams,
    generate_e8_roots,
    compare_e8_vs_hypercube,
    verify_yijing_properties,
    get_trigrams,
    get_hexagrams,
    FUXI_TO_BINARY,
    fuxi_order,
    wenwang_order,
    palace_clusters,
    palace_attention_mask,
    antipodal_pairs,
    antipodal_index,
    loshu_kernel,
    triangular_position,
    triangular_distance_matrix,
    andreev_matrix,
    hermann_packing,
    collision_test,
    valid_codebook_sizes,
    find_fixed_points,
    e8_collision_proof,
    generate_four_state_codebook,
    kasatkin_embedding,
    kasatkin_distance_matrix,
    kasatkin_axis_projection,
)

# Quantizers
from .quantizers import (
    YiJingQuantizer,
    E8Quantizer,
    FactoredYiJingQuantizer,
    FourStateQuantizer,
    AntipodalQuantizer,
    HierarchicalQuantizer,
    DeformableQuantizer,
    GumbelQuantizer,
    GroupedQuantizer,
    TernaryQuantizer,
    PairedBitQuantizer,
    MatryoshkaQuantizer,
    WHT_Quantizer,
)

# Attention patterns & modules
from .attention import (
    TriangularAttentionBias,
    PalaceAttention,
    QuadrantAttention,
    RecursiveCubeAttention,
    WeavingLoomArchitecture,
    BidirectionalTriangularAttention,
    CubeDiagonalAttention,
    CubicAttentionBias,
    HeisenbergAttention,
    FlowerOfLifeGAT,
    MobiusAttentionPattern,
    PrivilegedAxisAttention,
    DualModeHead,
    StructuralDefectLayer,
    HexagramAttentionPattern,
    GeometricAttention,
    MagicSquareInitializer,
    TriangularCurriculumScheduler,
)

# Positional encodings
from .positional import (
    RotaryEmbedding,
    rotate_half,
    apply_rotary_emb,
    ALiBi,
    FourLevelPositionalEncoding,
    CubicPositionalEncoding,
)

# Equivariant & structural layers
from .equivariant import (
    BianGuaTransform,
    GraduatedBianGuaTransform,
    D4EquivariantLayer,
    DualEmbedding,
)

# ArchetypalInterlinguaFixed — исправленная версия (раздельные trit_proj на источник)
# Примечание: оригинальный ArchetypalInterlingua в routing.py содержит баг
# (единый trit_proj делит веса между источниками → PPL ≈ 2.93 вместо 2.75)
from .interlingua_fixed import ArchetypalInterlinguaFixed

# Kasatkin Q6 routing — WHT-based routing с hex-метками доменов
from .kasatkin_router import KasatkinQ6Router, Q6ExpertBank

# Routing, gating, curriculum
from .routing import (
    GatedPathSelector,
    AdaptiveGatedPathSelector,
    TaskAwareRouter,
    GateLogger,
    GeometryCurriculumScheduler,
    DynamicCurriculumController,
    GeometricSourceRouter,
    GeometricSourceMixer,
    SequentialSourceCurriculum,
    PairwiseBridge,
    LightweightBridge,
    BridgeOfModules,
    AbrialeBridgeMediator,
    AdaptiveBridgeOfModules,
    SourceSpecializer,
    ArchetypalInterlingua,
    BridgedInterlingua,
)

# FFN modules
from .ffn import (
    SwiGLU,
    TrigramMoE,
    DomainMoE,
    GeometricFFN,
    MultiScaleHypercubeLayer,
)

# Convergence Bridge: гибридная иерархия глифов ↔ токенов
from .convergence import (
    GlyphComposer,
    TokenAbstractor,
    ConvergenceLayer,
    ConvergenceBridge,
    MatrixGrammar,
)

# Nautilus Hierarchy: иерархическое упорядочивание геометрических модулей
from .nautilus import (
    NautilusChamber,
    NautilusHierarchy,
    NautilusScheduler,
    MatryoshkaNautilus,
    PostCascadeMatryoshkaNautilus,
)

# Core: ternary hypercube & spacetime
from .core import (
    generate_ternary_hypercube,
    generate_ternary_trigrams,
    hex_digit_semantics,
    generate_spacetime_pairs,
)

# Q6 Algebra: Z₂^6 арифметика (Теорема 3) + GERMES-нотация + Bent seeds
from .q6_algebra import (
    Q6Arithmetic,
    YiJingOps,
    BentFunctions,
    BentPrototypeQuantizer,
    YiJingV4Layer,
    Q6ArithmeticLayer,
    verify_theorem3,
    verify_v4_group,
    verify_bent_functions,
    verify_all as verify_q6_all,
)
