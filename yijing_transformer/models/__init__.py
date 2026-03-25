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
    ALiBi,
    SwiGLU,
    TrigramMoE,
    GroupedQuantizer,
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
from .export import (
    export_onnx,
    export_torchscript,
    create_model_card,
)
from .polyglot import (
    PolyglotQuartet,
    PolyglotConfig,
    PolyglotMusician,
    PolyglotConductor,
    RosettaBridge,
    RepresentationHead,
    build_polyglot,
)
from .polyglot_translation import (
    TranslationHead,
    CrossTranslator,
    CycleConsistencyLoss,
    TranslationDemo,
)
from .polyglot_curriculum import (
    DifficultyEstimator,
    CurriculumScheduler,
    CurriculumDataLoader,
    CurriculumTrainer,
    build_curriculum_trainer,
)
from .polyglot_supervised import (
    SupervisedConfig,
    SupervisedTrainer,
    DistillationLoss,
    ContrastiveSupervisedLoss,
    AnnotatedPairLoss,
    LayerFreezer,
    WarmupCosineScheduler,
    build_supervised_trainer,
)
