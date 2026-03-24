"""
4-LEVEL KNOWLEDGE SYSTEM FOR YIJING-TRANSFORMER
================================================

Maps every module to the 4-level framework:
  Level 1: FORMULA   (Mathematics)  — pure geometry, group theory
  Level 2: ARCHETYPE (Physics)      — how geometry maps to neural architectures
  Level 3: ALGORITHM (Chemistry)    — training procedures, data pipelines
  Level 4: THEOREM   (Linguistics)  — inference, benchmarks, real language results

The "golden middle" between Formula (1) and Theorem (4) is
Archetype (2) + Algorithm (3). Without them, geometry cannot
connect to language — which explains the interlingua gate ≈ 0.5 problem.

Usage:
    from yijing_transformer.knowledge_system import KnowledgeSystem

    ks = KnowledgeSystem()
    ks.summary()
    ks.diagnose()
    modules = ks.get_level(2)  # all Archetype modules
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum
from pathlib import Path


class Level(IntEnum):
    FORMULA = 1     # Mathematics: what IS the hypercube
    ARCHETYPE = 2   # Physics: how geometry MAPS to attention/routing
    ALGORITHM = 3   # Chemistry: how to COMBINE and TRAIN
    THEOREM = 4     # Linguistics: does it WORK on language


LEVEL_NAMES = {
    Level.FORMULA: "FORMULA (Mathematics)",
    Level.ARCHETYPE: "ARCHETYPE (Physics)",
    Level.ALGORITHM: "ALGORITHM (Chemistry)",
    Level.THEOREM: "THEOREM (Linguistics)",
}

LEVEL_QUESTIONS = {
    Level.FORMULA: "What IS the hypercube geometry?",
    Level.ARCHETYPE: "How does geometry MAP to neural architectures?",
    Level.ALGORITHM: "How to COMBINE and TRAIN geometric components?",
    Level.THEOREM: "Does it WORK on real language data?",
}


@dataclass
class Module:
    """A module classified within the 4-level system."""
    path: str
    level: Level
    role: str
    key_classes: List[str] = field(default_factory=list)


# ============================================================================
# COMPLETE MODULE REGISTRY
# ============================================================================

MODULES: List[Module] = [
    # ── LEVEL 1: FORMULA (Mathematics) ──────────────────────────────────
    Module("models/geometry/core.py", Level.FORMULA,
           "Hypercube vertices Z₂ⁿ, trigrams, hexagrams, octograms, E8 roots",
           ["generate_hexagrams", "generate_trigrams", "hamming_distance"]),
    Module("models/geometry/quantizers.py", Level.FORMULA,
           "Projection onto geometric vertices",
           ["YiJingQuantizer", "E8Quantizer", "FactoredYiJingQuantizer",
            "GumbelQuantizer", "HierarchicalQuantizer"]),
    Module("models/geometry/equivariant.py", Level.FORMULA,
           "BianGua transformation, D4 equivariance, yin-yang duality",
           ["BianGuaTransform", "DualEmbedding"]),
    Module("models/geometry/convergence.py", Level.FORMULA,
           "GlyphComposer: hierarchical composition of edges/faces/sigils",
           ["GlyphComposer"]),
    Module("models/geometry/abriale.py", Level.FORMULA,
           "Isotropic networks, event-driven N-ary relations",
           ["AbrialeLayer", "IsotropicNet"]),
    Module("models/geometry/positional.py", Level.FORMULA,
           "RoPE, ALiBi, CubicPE with NTK/linear scaling",
           ["RotaryEmbedding", "ALiBi", "CubicPE"]),
    Module("config/config.py", Level.FORMULA,
           "YiJingConfig: hyperparameters defining the geometry system",
           ["YiJingConfig"]),

    # ── LEVEL 2: ARCHETYPE (Physics) ───────────────────────────────────
    Module("models/geometry/attention.py", Level.ARCHETYPE,
           "15 attention patterns from hypercube geometry",
           ["TriangularAttention", "PalaceAttention", "HeisenbergAttention",
            "FlowerOfLifeGAT", "MobiusPattern", "CubeDiagonalAttention"]),
    Module("models/geometry/nautilus.py", Level.ARCHETYPE,
           "NautilusHierarchy: 7 chambers from micro to macro",
           ["NautilusHierarchy", "NautilusChamber"]),
    Module("models/geometry/routing.py", Level.ARCHETYPE,
           "Expert routing via geometric distance",
           ["GatedPathSelector", "AdaptiveGatedPathSelector", "TaskAwareRouter"]),
    Module("models/geometry/ffn.py", Level.ARCHETYPE,
           "Geometric FFN: SwiGLU, TrigramMoE, DomainMoE",
           ["SwiGLU", "TrigramMoE", "DomainMoE", "GeometricFFN"]),
    Module("models/model.py", Level.ARCHETYPE,
           "YiJingGPT: main architecture integrating geometry",
           ["YiJingGPT", "YiJingAttention", "YiJingBlock"]),
    Module("models/baseline.py", Level.ARCHETYPE,
           "VanillaGPT: baseline without geometry",
           ["VanillaGPT"]),
    Module("models/diff_attn.py", Level.ARCHETYPE,
           "DifferentialAttention: softmax difference mechanism",
           ["DifferentialAttention"]),
    Module("models/expert_choice.py", Level.ARCHETYPE,
           "Expert-Choice routing with capacity factor",
           ["ExpertChoiceRouter"]),
    Module("models/hierarchical_moe.py", Level.ARCHETYPE,
           "Matryoshka MoE: Q2→Q3→Q6 multi-scale hierarchy",
           ["HierarchicalMoE", "Q6ExpertBank"]),
    Module("models/variant3.py", Level.ARCHETYPE,
           "Q6-hypercube core with BianGuaAttention",
           ["Variant3GPT", "ArchetypeLayer"]),
    Module("models/variant3_extensions.py", Level.ARCHETYPE,
           "10 extensions: SixLineAttention, TernaryKVCache, etc.",
           ["HexagramPositionalEncoding", "SixLineAttention"]),
    Module("models/hierarchical_e2.py", Level.ARCHETYPE,
           "E2: 5 hierarchical levels (Glyph→Core→Method→Theory→Philosophy)",
           ["HierarchicalE2"]),
    Module("models/nautilus_yijing.py", Level.ARCHETYPE,
           "Full YiJing-Nautilus integration",
           ["Q6GeometricRouter", "YiJingCoreBlock", "YiJingMicroExpert"]),
    Module("models/lora.py", Level.ARCHETYPE,
           "LoRA for parameter-efficient fine-tuning",
           ["LoRALayer"]),
    Module("models/prefix_tuning.py", Level.ARCHETYPE,
           "Prefix tuning, Logit Lens, Multi-Token Prediction",
           ["PrefixTuning", "LogitLens"]),
    Module("models/speculative.py", Level.ARCHETYPE,
           "Speculative decoding with draft models",
           ["SpeculativeDecoder"]),
    Module("models/extensions.py", Level.ARCHETYPE,
           "A2-E14: Walsh-Hadamard, Reed-Muller, Z₂¹²",
           ["MultiHeadGeometricAttention"]),

    # ── LEVEL 3: ALGORITHM (Chemistry) ─────────────────────────────────
    Module("training/train.py", Level.ALGORITHM,
           "Main training loop: warmup, gradient accum, checkpointing",
           ["Trainer", "train_epoch"]),
    Module("training/optim.py", Level.ALGORITHM,
           "Optimizer factory with LLRD",
           ["build_optimizer"]),
    Module("training/bridge.py", Level.ALGORITHM,
           "Adapter bridging 41 experimental optimizer modules",
           ["TrainingBridge"]),
    Module("training/regularization.py", Level.ALGORITHM,
           "Token Merging, Cosine Annealing, Gradient Noise",
           ["TokenMerger", "GradientNoise"]),
    Module("training/distillation.py", Level.ALGORITHM,
           "Knowledge distillation: soft targets + feature matching",
           ["DistillationLoss"]),
    Module("training/ema.py", Level.ALGORITHM,
           "Exponential Moving Average",
           ["EMA"]),
    Module("training/bridge_optimizers.py", Level.ALGORITHM,
           "Sophia, LAMB, Lion, SAM from experimental modules",
           ["Sophia", "LAMB", "Lion"]),
    Module("training/bridge_schedulers.py", Level.ALGORITHM,
           "WSD, Curriculum, LLRD scheduling",
           ["WarmupStableDecay"]),
    Module("training/bridge_regularization.py", Level.ALGORITHM,
           "Z-Loss, AGC, Label Smoothing, Mixup",
           ["ZLoss", "AdaptiveGradientClipping"]),
    Module("training/bridge_monitors.py", Level.ALGORITHM,
           "Loss spike detection, Grokking patterns",
           ["LossMonitor", "GradientFlowAnalyzer"]),
    Module("training/bridge_model_surgery.py", Level.ALGORITHM,
           "muP scaling, Pruning, DoRA, Merging",
           ["muP", "StructuredPruning"]),
    Module("data_utils/text_dataset.py", Level.ALGORITHM,
           "TextDataset: char-level from files/directories",
           ["TextDataset"]),
    Module("data_utils/streaming_dataset.py", Level.ALGORITHM,
           "StreamingDataset: online data loading",
           ["StreamingDataset"]),
    Module("data_utils/svend4_dataset.py", Level.ALGORITHM,
           "Svend4: multi-domain corpus with domain routing",
           ["Svend4Dataset"]),
    Module("data_utils/wikitext_dataset.py", Level.ALGORITHM,
           "WikiText dataset loader",
           ["WikiTextDataset"]),
    Module("data_utils/bridge_augmentation.py", Level.ALGORITHM,
           "Data pipeline: BPE dropout, RAG, packing, frequency weighting",
           ["DataPipeline"]),
    Module("tokenizer/char_tokenizer.py", Level.ALGORITHM,
           "CharTokenizer: character-level, no dependencies",
           ["CharTokenizer"]),
    Module("tokenizer/glyph_tokenizer.py", Level.ALGORITHM,
           "GlyphTokenizer: SOLAN alphabet → Q6 priors",
           ["GlyphTokenizer"]),
    Module("tokenizer/tokenizer_utils.py", Level.ALGORITHM,
           "Tokenizer utilities",
           ["load_tokenizer"]),

    # ── LEVEL 4: THEOREM (Linguistics) ─────────────────────────────────
    Module("inference/generate.py", Level.THEOREM,
           "Text generation: temperature, top-k, top-p",
           ["generate"]),
    Module("inference/bridge_inference.py", Level.THEOREM,
           "AdvancedGenerator: beam, nucleus, speculative, KV cache",
           ["AdvancedGenerator", "DynamicTemperature"]),
    Module("models/export.py", Level.THEOREM,
           "ONNX/TorchScript export, model cards",
           ["export_onnx", "export_torchscript"]),
    Module("scripts/ablation_archetypes.py", Level.THEOREM,
           "Archetype ablation: which Level-2 patterns help on language (Level-4)?",
           ["run_ablation", "PATTERNS"]),
    Module("scripts/benchmark_level4.py", Level.THEOREM,
           "Level 4 benchmark: LM, mod-64, XOR, copy tasks",
           ["run_benchmark", "TASK_REGISTRY"]),
]


class KnowledgeSystem:
    """
    4-level knowledge system for the yijing-transformer project.

    Maps the entire codebase to Formula → Archetype → Algorithm → Theorem.
    """

    def __init__(self):
        self.modules = MODULES
        self._by_level: Dict[Level, List[Module]] = {level: [] for level in Level}
        for m in self.modules:
            self._by_level[m.level].append(m)

    def get_level(self, level: int) -> List[Module]:
        """Get all modules at a given level (1-4)."""
        return self._by_level[Level(level)]

    def get_formula_modules(self) -> List[Module]:
        return self.get_level(1)

    def get_archetype_modules(self) -> List[Module]:
        return self.get_level(2)

    def get_algorithm_modules(self) -> List[Module]:
        return self.get_level(3)

    def get_theorem_modules(self) -> List[Module]:
        return self.get_level(4)

    def find_module(self, keyword: str) -> List[Module]:
        """Search modules by keyword in path, role, or classes."""
        kw = keyword.lower()
        results = []
        for m in self.modules:
            searchable = f"{m.path} {m.role} {' '.join(m.key_classes)}".lower()
            if kw in searchable:
                results.append(m)
        return results

    def summary(self) -> str:
        """Print summary of all levels."""
        lines = [
            "=" * 70,
            "  YIJING-TRANSFORMER: 4-LEVEL KNOWLEDGE SYSTEM",
            "=" * 70,
            "",
        ]
        for level in Level:
            modules = self._by_level[level]
            lines.append(f"[{level.value}] {LEVEL_NAMES[level]}")
            lines.append(f"    Question: {LEVEL_QUESTIONS[level]}")
            lines.append(f"    Modules: {len(modules)}")
            lines.append("")
        lines.append(f"Total: {len(self.modules)} modules")
        lines.append("=" * 70)
        return "\n".join(lines)

    def diagnose(self) -> str:
        """
        Diagnose the connection between Formula and Theorem.

        The golden middle (Archetype + Algorithm) must be strong enough
        to bridge pure mathematics to practical language modeling.
        """
        formula = self._by_level[Level.FORMULA]
        archetype = self._by_level[Level.ARCHETYPE]
        algorithm = self._by_level[Level.ALGORITHM]
        theorem = self._by_level[Level.THEOREM]

        lines = [
            "=" * 70,
            "  DIAGNOSTIC: Formula → Theorem Connection",
            "=" * 70,
            "",
            f"[1] FORMULA:   {len(formula)} modules  (mathematical foundation)",
            f"[2] ARCHETYPE: {len(archetype)} modules  (structural patterns)",
            f"[3] ALGORITHM: {len(algorithm)} modules  (training procedures)",
            f"[4] THEOREM:   {len(theorem)} modules  (empirical results)",
            "",
            "Golden Middle (2+3): "
            f"{len(archetype) + len(algorithm)} modules",
            "",
        ]

        # Diagnose balance
        ratio_21 = len(archetype) / max(len(formula), 1)
        ratio_32 = len(algorithm) / max(len(archetype), 1)
        ratio_43 = len(theorem) / max(len(algorithm), 1)

        lines.append("Level transitions:")
        lines.append(f"  1→2 (Formula→Archetype):  {ratio_21:.1f}x expansion")
        lines.append(f"  2→3 (Archetype→Algorithm): {ratio_32:.1f}x expansion")
        lines.append(f"  3→4 (Algorithm→Theorem):   {ratio_43:.1f}x expansion")
        lines.append("")

        # Identify weaknesses
        lines.append("Issues:")

        if ratio_21 < 2.0:
            lines.append("  [!] Level 2 may be thin — not enough architectural "
                         "patterns to explore how geometry maps to attention")
        else:
            lines.append("  [ok] Level 2 has sufficient archetype diversity")

        if len(theorem) < 5:
            lines.append("  [!] Level 4 is thin — need more empirical validation")
        else:
            lines.append("  [ok] Level 4 has benchmark infrastructure")

        lines.append("")
        lines.append("Key bottleneck: Archetype selection")
        lines.append("  15 attention patterns exist, but which 2-3 actually help")
        lines.append("  on real language data? Systematic ablation needed.")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def level_map(self) -> str:
        """Generate a visual map of all modules by level."""
        lines = []
        for level in Level:
            lines.append(f"\n{'─' * 70}")
            lines.append(f"  [{level.value}] {LEVEL_NAMES[level]}")
            lines.append(f"{'─' * 70}")
            for m in self._by_level[level]:
                classes = ", ".join(m.key_classes[:3])
                lines.append(f"  {m.path:<45s} {classes}")
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS (matching info3 API)
# ============================================================================

def get_formula():
    """Get Level 1 modules (pure mathematics)."""
    return KnowledgeSystem().get_formula_modules()

def get_archetype():
    """Get Level 2 modules (structural patterns)."""
    return KnowledgeSystem().get_archetype_modules()

def get_algorithm():
    """Get Level 3 modules (training procedures)."""
    return KnowledgeSystem().get_algorithm_modules()

def get_theorem():
    """Get Level 4 modules (empirical results)."""
    return KnowledgeSystem().get_theorem_modules()

def search(keyword: str):
    """Search across all levels."""
    return KnowledgeSystem().find_module(keyword)


if __name__ == "__main__":
    ks = KnowledgeSystem()
    print(ks.summary())
    print()
    print(ks.diagnose())
    print()
    print(ks.level_map())
