"""
quartet.py — Четыре Бременских Музыканта (Four Bremen Musicians)

Четырёхмодельная архитектура, где каждая модель — отдельная «профессия»,
специализирующаяся на своём типе мышления. Вместе они образуют
конвейер, где каждый уровень дополняет остальные.

Четыре профессии (горизонтальные эксперты):

    ① ФОРМАЛИСТ (Formalist) — математика, формулы, точные структуры
       Инструмент: скрипка (точность, чистота линий)
       Элемент: Огонь (энергия преобразования)
       Сезон: Лето (максимальная активность)
       Модули из репозитория:
         - geometry/core.py: generate_hypercube, generate_hexagrams, E8 lattice
         - geometry/quantizers.py: GumbelQuantizer, E8Quantizer, WHT
         - geometry/positional.py: RotaryEmbedding, CubicPE
         - geometry/equivariant.py: D4EquivariantLayer, BianGuaTransform
         - geometry/q6_algebra.py: Z₂⁶ group algebra, bent functions

    ② АРХЕТИПИСТ (Archetypist) — физика, архетипы, сложные формулы
       Инструмент: виолончель (глубина, обертоны)
       Элемент: Земля (фундаментальность)
       Сезон: Осень (сбор урожая паттернов)
       Модули из репозитория:
         - geometry/attention.py: 15 attention patterns (Heisenberg, Palace,
           CubeDiagonal, Mobius, FlowerOfLife, SOLAN, Triangular, etc.)
         - geometry/routing.py: GatedPathSelector, GeometricSourceRouter
         - geometry/nautilus.py: NautilusHierarchy (7 chambers micro→macro)
         - geometry/interlingua_fixed.py: 64 archetypes as mediator hub
         - geometry/convergence.py: GlyphComposer, TokenAbstractor

    ③ АЛГОРИТМИСТ (Algorithmist) — химия, алгоритмы, комбинаторика
       Инструмент: духовые (трансформация, потоки)
       Элемент: Вода (адаптивность, текучесть)
       Сезон: Зима (кристаллизация структур)
       Модули из репозитория:
         - geometry/ffn.py: TrigramMoE, DomainMoE, GeometricFFN
         - geometry/six_sources.py: SixSourceLayer (6 theories unified)
         - geometry/kasatkin_router.py: KasatkinQ6Router
         - hierarchical_moe.py: HierarchicalMoE, Q6ExpertBank
         - expert_choice.py: ExpertChoiceRouter
         - geometry/abriale.py: AbrialeLayer (event-driven relations)
         - diff_attn.py: DifferentialAttention

    ④ ЛИНГВИСТ (Linguist) — язык, философия, психология, биология
       Инструмент: ударные (ритм, паттерн)
       Элемент: Воздух (связь, коммуникация)
       Сезон: Весна (порождение нового)
       Модули из репозитория:
         - geometry/quantizers.py: TernaryQuantizer ({-1,0,+1} = yes/no/unknown)
         - geometry/convergence.py: MatrixGrammar (2D syntax)
         - pseudo_rag.py: PseudoRAGProjection (Q4→Q6 embedding)
         - tokenizer/glyph_tokenizer.py: GlyphTokenizer (SOLAN visual alphabet)
         - inference/generate.py: generation strategies
         - cross_domain_analogy.py: hexagram-based analogies

Каждый Музыкант — это миди-эксперт, состоящий из микро-экспертов.
Они не иерархия (не вертикаль), а четыре горизонтальных профессии
в одном оркестре, играющие на разных инструментах одну мелодию.
"""

import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# Cluster Registry: mapping existing modules to four professions
# ═══════════════════════════════════════════════════════════════

# Each musician has access to specific geometric modules.
# This registry defines which "instruments" each profession plays.
CLUSTER_REGISTRY = {
    'formalist': {
        'description': 'Математика, формулы, точные структуры',
        'element': 'Огонь', 'season': 'Лето', 'instrument': 'скрипка',
        'geometric_modules': [
            # Exact geometric structures (pure math)
            'D4EquivariantLayer',     # D₄ group symmetry
            'CubeDiagonalAttention',  # corner symmetry patterns
            'PrivilegedAxisAttention', # main axis extraction
        ],
        'quantizer': 'gumbel',  # discrete → exact vertices
        'micro_expert_style': 'precise',  # narrow, focused experts
    },
    'archetypist': {
        'description': 'Физика, архетипы, сложные паттерны',
        'element': 'Земля', 'season': 'Осень', 'instrument': 'виолончель',
        'geometric_modules': [
            # Pattern emergence (physics of attention)
            'HeisenbergAttention',    # uncertainty principle
            'PalaceAttention',        # 8-palace block structure
            'FlowerOfLifeGAT',        # sacred geometry graph attention
        ],
        'quantizer': 'ternary',  # {-1, 0, +1} = yin/void/yang
        'micro_expert_style': 'deep',  # fewer but larger experts
    },
    'algorithmist': {
        'description': 'Химия, алгоритмы, комбинаторика',
        'element': 'Вода', 'season': 'Зима', 'instrument': 'духовые',
        'geometric_modules': [
            # Transformations and flows (algorithmic)
            'MobiusAttentionPattern',    # topological wrapping
            'TriangularAttentionBias',   # distance-based routing
            'DualEmbedding',             # yin-yang dual space
        ],
        'quantizer': 'factored',  # 3+3 factored = combinatorial
        'micro_expert_style': 'combinatorial',  # many small experts
    },
    'linguist': {
        'description': 'Язык, философия, психология, биология',
        'element': 'Воздух', 'season': 'Весна', 'instrument': 'ударные',
        'geometric_modules': [
            # Communication and meaning (linguistic)
            'HexagramAttentionPattern',  # semantic distances
            'SOLANAttention',            # visual language attention
            'WeavingLoomArchitecture',   # bidirectional weave
        ],
        'quantizer': 'ternary',  # {-1, 0, +1} = yes/no/unknown epistemic
        'micro_expert_style': 'broad',  # broader but shallower
    },
}


def _build_geometric_module(name: str, d_model: int) -> Optional[nn.Module]:
    """Safely import and instantiate a geometric module by name.

    Returns None if the module is not available (graceful degradation).
    """
    try:
        import importlib
        # Import from absolute path since quartet.py lives in models/
        attn_mod = importlib.import_module(
            'yijing_transformer.models.geometry.attention')
        equiv_mod = importlib.import_module(
            'yijing_transformer.models.geometry.equivariant')

        registry = {
            'D4EquivariantLayer': lambda: equiv_mod.D4EquivariantLayer(d_model),
            'DualEmbedding': lambda: equiv_mod.DualEmbedding(d_model),
            'CubeDiagonalAttention': lambda: attn_mod.CubeDiagonalAttention(d_model),
            'PrivilegedAxisAttention': lambda: attn_mod.PrivilegedAxisAttention(d_model),
            'HeisenbergAttention': lambda: attn_mod.HeisenbergAttention(d_model),
            'PalaceAttention': lambda: attn_mod.PalaceAttention(d_model),
            'FlowerOfLifeGAT': lambda: attn_mod.FlowerOfLifeGAT(d_model),
            'MobiusAttentionPattern': lambda: attn_mod.MobiusAttentionPattern(d_model),
            'TriangularAttentionBias': lambda: attn_mod.TriangularAttentionBias(),
            'HexagramAttentionPattern': lambda: attn_mod.HexagramAttentionPattern(d_model, block_size=512),
            'SOLANAttention': lambda: attn_mod.SOLANAttention(d_model),
            'WeavingLoomArchitecture': lambda: attn_mod.WeavingLoomArchitecture(d_model),
        }
        if name in registry:
            return registry[name]()
    except (ImportError, Exception):
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class MusicianConfig:
    """Конфигурация одного Музыканта (миди-эксперта)."""
    name: str = 'formalist'
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_micro_experts: int = 8       # микро-экспертов внутри каждого слоя
    micro_expert_dim: int = 64     # размерность каждого микро-эксперта
    top_k_micro: int = 2           # сколько микро-экспертов активировать
    dropout: float = 0.1
    # Специализация: какие геометрические модули включены
    use_quantizer: bool = True
    quantizer_type: str = 'ternary'
    # Geometric modules from CLUSTER_REGISTRY (loaded by name)
    geometric_modules: List[str] = field(default_factory=list)
    use_geometric: bool = True     # whether to use geometric enrichment


@dataclass
class QuartetConfig:
    """Конфигурация ансамбля из четырёх Музыкантов."""
    vocab_size: int = 4096
    d_model: int = 256             # общая размерность (обмен между музыкантами)
    block_size: int = 512
    n_archetypes: int = 64         # размер общего архетипного пространства
    dropout: float = 0.1

    # Каждый Музыкант может иметь свою конфигурацию
    formalist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='formalist', n_micro_experts=8, micro_expert_dim=64,
        quantizer_type='gumbel',
    ))
    archetypist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='archetypist', n_micro_experts=6, micro_expert_dim=96,
        quantizer_type='ternary',
    ))
    algorithmist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='algorithmist', n_micro_experts=8, micro_expert_dim=64,
        quantizer_type='factored',
    ))
    linguist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='linguist', n_micro_experts=6, micro_expert_dim=96,
        quantizer_type='ternary',
    ))

    # Оркестр: как музыканты взаимодействуют
    conductor_heads: int = 4       # голов внимания у Дирижёра
    rehearsal_rounds: int = 2      # сколько раундов обмена между музыкантами
    curriculum_warmup: int = 2000  # шагов до полной активации всех музыкантов


# ═══════════════════════════════════════════════════════════════
# Micro-Expert Layer (внутренний строительный блок)
# ═══════════════════════════════════════════════════════════════

class MicroExpertFFN(nn.Module):
    """Один микро-эксперт: маленький FFN со специализацией.

    Каждый микро-эксперт — это «подмастерье» внутри Музыканта.
    """
    def __init__(self, d_model: int, d_expert: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d_model, d_expert, bias=False)
        self.gate = nn.Linear(d_model, d_expert, bias=False)
        self.down = nn.Linear(d_expert, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # Инициализация: маленький масштаб для стабильности
        nn.init.trunc_normal_(self.up.weight, std=0.02)
        nn.init.trunc_normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class MicroExpertMoE(nn.Module):
    """MoE из микро-экспертов внутри одного Музыканта.

    Soft top-k routing: каждый токен активирует top_k микро-экспертов
    через дифференцируемые веса.
    """
    def __init__(self, d_model: int, n_experts: int, d_expert: int,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)

        self.experts = nn.ModuleList([
            MicroExpertFFN(d_model, d_expert, dropout)
            for _ in range(n_experts)
        ])

        # Learnable router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.zeros_(self.router.weight)
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # temperature

        # Auxiliary load balancing
        self._aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Router logits
        temp = F.softplus(self.log_temp) + 0.1
        logits = self.router(x) / temp  # (B, T, n_experts)

        # Soft top-k selection
        if self.top_k < self.n_experts:
            top_vals, top_idx = logits.topk(self.top_k, dim=-1)
            threshold = top_vals[..., -1:].detach()
            suppression = torch.sigmoid(10.0 * (logits - threshold))
            logits = logits * suppression

        weights = F.softmax(logits, dim=-1)  # (B, T, n_experts)

        # Load balancing loss
        avg_probs = weights.mean(dim=(0, 1))
        self._aux_loss = self.n_experts * (avg_probs * avg_probs).sum()

        # Compute all expert outputs (vectorized for small expert count)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (B, T, D, n_experts)

        # Weighted combination
        result = (expert_outputs * weights.unsqueeze(2)).sum(dim=-1)
        return result

    def get_aux_loss(self) -> torch.Tensor:
        return self._aux_loss


# ═══════════════════════════════════════════════════════════════
# Musician Layer (один слой внутри Музыканта)
# ═══════════════════════════════════════════════════════════════

class GeometricEnrichment(nn.Module):
    """Geometric enrichment from a set of specialized modules.

    Each Musician has its own set of geometric "instruments" — modules from
    geometry/ that define its professional specialization. The enrichment
    is soft-gated: the model learns how much each geometric module contributes.
    """
    def __init__(self, d_model: int, module_names: List[str]):
        super().__init__()
        self.d_model = d_model
        self.module_names = []
        modules = []
        for name in module_names:
            mod = _build_geometric_module(name, d_model)
            if mod is not None:
                modules.append(mod)
                self.module_names.append(name)

        self.geo_modules = nn.ModuleList(modules)
        n = len(self.geo_modules)
        if n > 0:
            # Per-module learnable gate (how much this instrument contributes)
            self.gate_logits = nn.Parameter(torch.zeros(n))
            # Projection from geometric output to d_model (if shapes differ)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply geometric modules and return soft-gated enrichment delta."""
        if len(self.geo_modules) == 0:
            return torch.zeros_like(x)

        gates = torch.sigmoid(self.gate_logits)  # (n_modules,)
        enrichment = torch.zeros_like(x)

        for i, mod in enumerate(self.geo_modules):
            try:
                # Geometric modules have diverse interfaces:
                # - Some return bias (B, T, T) for attention
                # - Some return transformed x (B, T, D)
                # We handle both via duck typing
                if hasattr(mod, 'get_bias'):
                    bias = mod.get_bias(x)
                    T = x.shape[1]
                    causal = torch.tril(torch.ones(T, T, device=x.device))
                    bias = bias.masked_fill(causal.unsqueeze(0) == 0, float('-inf'))
                    weights = F.softmax(bias, dim=-1).nan_to_num(0.0)
                    delta = torch.bmm(weights, x) - x
                else:
                    out = mod(x)
                    if out.shape == x.shape:
                        delta = out - x
                    else:
                        delta = torch.zeros_like(x)
            except Exception:
                delta = torch.zeros_like(x)

            enrichment = enrichment + gates[i] * delta

        return self.proj(enrichment)


class MusicianLayer(nn.Module):
    """Один трансформер-слой внутри Музыканта.

    Attention → [Geometric Enrichment] → MicroExpert MoE → LayerNorm.
    """
    def __init__(self, cfg: MusicianConfig, layer_idx: int = 0):
        super().__init__()
        d = cfg.d_model

        # Self-attention
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        # Geometric enrichment (profession-specific instruments)
        self.has_geo = cfg.use_geometric and len(cfg.geometric_modules) > 0
        if self.has_geo:
            self.ln_geo = nn.LayerNorm(d)
            self.geo = GeometricEnrichment(d, cfg.geometric_modules)
            # Learnable gate: how much geometry vs vanilla
            self.geo_gate = nn.Parameter(torch.tensor(0.0))

        # Micro-Expert MoE
        self.ln2 = nn.LayerNorm(d)
        self.moe = MicroExpertMoE(
            d_model=d,
            n_experts=cfg.n_micro_experts,
            d_expert=cfg.micro_expert_dim,
            top_k=cfg.top_k_micro,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out

        # Geometric enrichment (profession-specific)
        if self.has_geo:
            geo_delta = self.geo(self.ln_geo(x))
            x = x + torch.sigmoid(self.geo_gate) * geo_delta

        # Micro-expert MoE
        x = x + self.moe(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════
# Musician (один из четырёх Бременских Музыкантов)
# ═══════════════════════════════════════════════════════════════

class Musician(nn.Module):
    """Один Музыкант — миди-эксперт со своей «профессией».

    Каждый Музыкант:
    - Имеет свой стек трансформер-слоёв с MicroExpert MoE
    - Принимает общий вход + сигнал от других музыкантов
    - Возвращает свою «партию» (вклад в общий результат)
    """
    def __init__(self, cfg: MusicianConfig, d_shared: int):
        super().__init__()
        self.name = cfg.name
        self.d_model = cfg.d_model
        self.d_shared = d_shared

        # Проекция из общего пространства в пространство музыканта
        self.proj_in = nn.Linear(d_shared, cfg.d_model, bias=False)

        # Стек слоёв (собственный трансформер)
        self.layers = nn.ModuleList([
            MusicianLayer(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ])

        # Проекция обратно в общее пространство
        self.proj_out = nn.Linear(cfg.d_model, d_shared, bias=False)
        nn.init.zeros_(self.proj_out.weight)  # начинаем с нулевого вклада

        # Финальная LayerNorm
        self.ln_out = nn.LayerNorm(d_shared)

        # Learnable «громкость» этого музыканта
        self.volume = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

    def forward(self, x_shared: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_shared: (B, T, d_shared) — общее представление
            attn_mask: маска внимания

        Returns:
            contribution: (B, T, d_shared) — вклад этого музыканта
        """
        # Проекция в своё пространство
        h = self.proj_in(x_shared)

        # Обработка своим трансформером
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        # Проекция обратно + volume control
        contribution = self.proj_out(h)
        volume = torch.sigmoid(self.volume)
        return self.ln_out(contribution * volume)

    def get_aux_loss(self) -> torch.Tensor:
        """Суммарный auxiliary loss от всех MoE слоёв."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total = total + layer.moe.get_aux_loss()
        return total


# ═══════════════════════════════════════════════════════════════
# Conductor (Дирижёр — orchestrates the musicians)
# ═══════════════════════════════════════════════════════════════

class Conductor(nn.Module):
    """Дирижёр — управляет взаимодействием четырёх Музыкантов.

    Дирижёр не «умнее» музыкантов, он просто координирует:
    1. Принимает партии всех четырёх музыкантов
    2. Cross-attention: каждый музыкант «слышит» остальных
    3. Возвращает скоординированный сигнал каждому

    Это не иерархия — Дирижёр просто обеспечивает коммуникацию,
    как круглый стол, за которым сидят четыре учёных.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Cross-attention между музыкантами
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Soft routing: какой музыкант сколько влияет
        self.blend_proj = nn.Linear(d_model, 4, bias=True)
        nn.init.zeros_(self.blend_proj.weight)
        nn.init.zeros_(self.blend_proj.bias)

        # Post-processing
        self.ln_post = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.ffn[-2].weight)

    def forward(self, shared_state: torch.Tensor,
                contributions: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            shared_state: (B, T, D) — текущее общее состояние
            contributions: list of 4 × (B, T, D) — партии музыкантов

        Returns:
            orchestrated: (B, T, D) — скоординированный результат
        """
        B, T, D = shared_state.shape

        # Concatenate all musician contributions as KV sequence
        # (B, 4*T, D) — каждый музыкант добавляет T токенов
        all_music = torch.cat(contributions, dim=1)  # (B, 4*T, D)

        # Cross-attention: shared_state attends to all musicians
        q = self.ln_q(shared_state)
        kv = self.ln_kv(all_music)
        attended, _ = self.cross_attn(q, kv, kv)  # (B, T, D)

        # Soft blend weights per token (who to listen to more)
        blend_logits = self.blend_proj(shared_state)  # (B, T, 4)
        blend_weights = F.softmax(blend_logits, dim=-1)  # (B, T, 4)

        # Weighted sum of individual contributions
        stacked = torch.stack(contributions, dim=-1)  # (B, T, D, 4)
        weighted = (stacked * blend_weights.unsqueeze(2)).sum(dim=-1)  # (B, T, D)

        # Combine cross-attention + weighted blend
        combined = attended + weighted

        # Post-processing
        combined = combined + self.ffn(self.ln_post(combined))

        return combined


# ═══════════════════════════════════════════════════════════════
# Seasonal Cycle (Годовой Круг — curriculum & phase dynamics)
# ═══════════════════════════════════════════════════════════════

class SeasonalCycle(nn.Module):
    """Годовой Круг — циклическая динамика четырёх Музыкантов.

    Как времена года: каждый Музыкант имеет свой «пик активности»,
    но все работают постоянно. Цикл определяет относительную
    «громкость» каждого музыканта в зависимости от фазы обучения.

    Формалист  ⟷  Лето   (phase 0.00 — 0.25) — точность, формулы
    Архетипист ⟷  Осень  (phase 0.25 — 0.50) — глубина, паттерны
    Алгоритмист ⟷ Зима   (phase 0.50 — 0.75) — кристаллизация
    Лингвист   ⟷  Весна  (phase 0.75 — 1.00) — порождение

    Но это не жёсткое расписание — скорее мягкий curriculum bias.
    """
    def __init__(self, warmup_steps: int = 2000):
        super().__init__()
        self.warmup_steps = warmup_steps
        # Learnable phase offsets — модель может сдвигать «пики»
        self.phase_offsets = nn.Parameter(torch.tensor([0.0, 0.25, 0.5, 0.75]))
        # Learnable concentration — насколько острый пик
        self.log_concentration = nn.Parameter(torch.tensor(1.0))

    def forward(self, step: int) -> torch.Tensor:
        """Возвращает 4 весовых коэффициента для Музыкантов.

        Returns:
            weights: (4,) — мягкие веса, сумма ≈ 1
        """
        # Нормализованная фаза [0, 1) — циклическая
        if self.warmup_steps > 0:
            progress = min(step / self.warmup_steps, 1.0)
        else:
            progress = 1.0

        # Базовый вес: все музыканты работают с самого начала
        base = torch.ones(4, device=self.phase_offsets.device) * 0.25

        if progress < 1.0:
            # Во время warmup: постепенно добавляем сезонную модуляцию
            phase = (step % self.warmup_steps) / self.warmup_steps
            concentration = self.log_concentration.exp()

            # Косинусная близость к «пику» каждого музыканта
            phase_diff = torch.cos(2 * math.pi * (phase - self.phase_offsets))
            seasonal_bias = F.softmax(concentration * phase_diff, dim=0)

            # Мягкий переход от равномерного к сезонному
            weights = (1 - progress) * base + progress * seasonal_bias
        else:
            # После warmup: полностью learnable (сезонная модуляция)
            weights = base  # или можно оставить сезонную

        return weights


# ═══════════════════════════════════════════════════════════════
# Quartet (полный ансамбль — Четыре Бременских Музыканта)
# ═══════════════════════════════════════════════════════════════

class BremenQuartet(nn.Module):
    """Четыре Бременских Музыканта — полная языковая модель.

    Архитектура:
        1. Token embedding → shared space
        2. Для каждого rehearsal_round:
           a. Каждый Музыкант обрабатывает shared_state → contribution
           b. Дирижёр координирует contributions → new shared_state
        3. Output head → logits

    Это не ensemble (голосование) и не mixture (выбор одного).
    Это оркестр — каждый играет свою партию, и только вместе
    получается музыка.
    """
    def __init__(self, cfg: QuartetConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # ── Shared embeddings ──
        self.tok_emb = nn.Embedding(cfg.vocab_size, D)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, D))
        self.emb_drop = nn.Dropout(cfg.dropout)

        # ── Четыре Музыканта ──
        self.musicians = nn.ModuleDict({
            'formalist': Musician(cfg.formalist, d_shared=D),
            'archetypist': Musician(cfg.archetypist, d_shared=D),
            'algorithmist': Musician(cfg.algorithmist, d_shared=D),
            'linguist': Musician(cfg.linguist, d_shared=D),
        })

        # Порядок (для стабильной итерации)
        self._musician_order = ['formalist', 'archetypist', 'algorithmist', 'linguist']

        # ── Дирижёр (по одному на каждый rehearsal round) ──
        self.conductors = nn.ModuleList([
            Conductor(D, n_heads=cfg.conductor_heads, dropout=cfg.dropout)
            for _ in range(cfg.rehearsal_rounds)
        ])

        # ── Годовой Круг ──
        self.seasonal_cycle = SeasonalCycle(warmup_steps=cfg.curriculum_warmup)

        # ── Learnable residual gate for rehearsal rounds ──
        self.rehearsal_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))
            for _ in range(cfg.rehearsal_rounds)
        ])

        # ── Output ──
        self.ln_out = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Tracking
        self._current_step = 0
        self._last_info: Dict = {}

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def set_step(self, step: int):
        self._current_step = step

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices (optional, for loss)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
            info: diagnostic dict
        """
        B, T = idx.shape
        device = idx.device

        # ── Embedding ──
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        shared_state = self.emb_drop(tok + pos)

        # ── Causal mask ──
        attn_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )  # True = masked positions (for nn.MultiheadAttention)

        # ── Seasonal weights ──
        seasonal_w = self.seasonal_cycle(self._current_step)

        # ── Rehearsal rounds (оркестровые репетиции) ──
        info = {'seasonal_weights': seasonal_w.detach().tolist(), 'rounds': []}

        for r, conductor in enumerate(self.conductors):
            # Каждый Музыкант играет свою партию
            contributions = []
            round_info = {}

            for i, name in enumerate(self._musician_order):
                musician = self.musicians[name]
                c = musician(shared_state, attn_mask=attn_mask)
                # Модулируем сезонным весом
                c = c * seasonal_w[i]
                contributions.append(c)
                round_info[name] = {
                    'volume': torch.sigmoid(musician.volume).item(),
                    'seasonal_w': seasonal_w[i].item(),
                }

            # Дирижёр координирует
            orchestrated = conductor(shared_state, contributions)

            # Residual с learnable gate
            gate = torch.sigmoid(self.rehearsal_gates[r])
            shared_state = shared_state + gate * orchestrated

            round_info['gate'] = gate.item()
            info['rounds'].append(round_info)

        # ── Output ──
        logits = self.head(self.ln_out(shared_state))

        # ── Loss ──
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            # Auxiliary losses от MoE
            aux_loss = torch.tensor(0.0, device=device)
            for name in self._musician_order:
                aux_loss = aux_loss + self.musicians[name].get_aux_loss()
            loss = loss + 0.01 * aux_loss

        self._last_info = info
        return logits, loss, info

    def get_info(self) -> Dict:
        return self._last_info

    def count_parameters(self) -> Dict[str, int]:
        """Подсчёт параметров по компонентам."""
        result = {}
        for name in self._musician_order:
            result[name] = sum(p.numel() for p in self.musicians[name].parameters())
        result['conductors'] = sum(
            p.numel() for c in self.conductors for p in c.parameters()
        )
        result['embeddings'] = (
            self.tok_emb.weight.numel() + self.pos_emb.numel()
        )
        result['total'] = sum(p.numel() for p in self.parameters())
        return result


# ═══════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════

def build_quartet(
    vocab_size: int = 4096,
    d_model: int = 256,
    musician_layers: int = 4,
    micro_experts: int = 8,
    micro_dim: int = 64,
    rehearsal_rounds: int = 2,
    **kwargs
) -> BremenQuartet:
    """Создаёт Квартет с разумными defaults.

    Каждый Музыкант ~0.5-1M параметров (миди-эксперт),
    весь Квартет ~3-5M параметров.

    Usage:
        model = build_quartet(vocab_size=4096, d_model=256)
        logits, loss, info = model(token_ids, targets=labels)
    """
    # Propagate common settings to all musicians
    musician_defaults = dict(
        d_model=d_model,
        n_layers=musician_layers,
        n_heads=max(d_model // 64, 2),
        dropout=kwargs.get('dropout', 0.1),
    )

    cfg = QuartetConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=kwargs.get('block_size', 512),
        dropout=kwargs.get('dropout', 0.1),
        rehearsal_rounds=rehearsal_rounds,
        conductor_heads=max(d_model // 64, 2),
        curriculum_warmup=kwargs.get('curriculum_warmup', 2000),

        formalist=MusicianConfig(
            name='formalist',
            n_micro_experts=micro_experts,
            micro_expert_dim=micro_dim,
            top_k_micro=2,
            quantizer_type=CLUSTER_REGISTRY['formalist']['quantizer'],
            geometric_modules=CLUSTER_REGISTRY['formalist']['geometric_modules'],
            **musician_defaults,
        ),
        archetypist=MusicianConfig(
            name='archetypist',
            n_micro_experts=max(micro_experts * 3 // 4, 2),
            micro_expert_dim=int(micro_dim * 1.5),
            top_k_micro=2,
            quantizer_type=CLUSTER_REGISTRY['archetypist']['quantizer'],
            geometric_modules=CLUSTER_REGISTRY['archetypist']['geometric_modules'],
            **musician_defaults,
        ),
        algorithmist=MusicianConfig(
            name='algorithmist',
            n_micro_experts=micro_experts,
            micro_expert_dim=micro_dim,
            top_k_micro=2,
            quantizer_type=CLUSTER_REGISTRY['algorithmist']['quantizer'],
            geometric_modules=CLUSTER_REGISTRY['algorithmist']['geometric_modules'],
            **musician_defaults,
        ),
        linguist=MusicianConfig(
            name='linguist',
            n_micro_experts=max(micro_experts * 3 // 4, 2),
            micro_expert_dim=int(micro_dim * 1.5),
            top_k_micro=2,
            quantizer_type=CLUSTER_REGISTRY['linguist']['quantizer'],
            geometric_modules=CLUSTER_REGISTRY['linguist']['geometric_modules'],
            **musician_defaults,
        ),
    )

    return BremenQuartet(cfg)


# ═══════════════════════════════════════════════════════════════
# Cluster Inventory: full mapping of repo files → four professions
# ═══════════════════════════════════════════════════════════════

FULL_CLUSTER_INVENTORY = {
    'formalist': {
        'description': '① ФОРМАЛИСТ — Математика, формулы, точные структуры',
        'element': 'Огонь 🔥', 'season': 'Лето', 'instrument': 'Скрипка',
        'files': {
            # ─── Ядро: чистая математика ───
            'geometry/core.py': [
                'generate_hypercube()',    # Z₂ⁿ вершины
                'generate_hexagrams()',    # 64 гексаграммы
                'generate_e8_roots()',     # E8 решётка (240 корней)
                'generate_ternary_hypercube()',  # Z₃ⁿ
                'generate_ternary_trigrams()',   # 27 тернарных триграмм
            ],
            # ─── Квантизация (проекция на вершины) ───
            'geometry/quantizers.py': [
                'GumbelQuantizer',          # Gumbel-Softmax дискретизация
                'E8Quantizer',              # E8 решётка
                'WHTQuantizer',             # Walsh-Hadamard
                'HierarchicalQuantizer',    # Q3→Q6 иерархия
                'DeformableQuantizer',      # обучаемые вершины
                'GroupedQuantizer',         # группированный
            ],
            # ─── Позиционное кодирование (формулы) ───
            'geometry/positional.py': [
                'RotaryEmbedding',          # RoPE
                'CubicPositionalEncoding',  # 3D Касаткин
                'FourLevelPositionalEncoding',  # 4-уровневый Андреев
            ],
            # ─── Группы симметрий ───
            'geometry/equivariant.py': [
                'D4EquivariantLayer',       # D₄ группа (8 операций)
                'BianGuaTransform',         # 变卦 трансформация
                'GraduatedBianGuaTransform',  # мягкая BianGua
            ],
            # ─── Алгебра ───
            'geometry/q6_algebra.py': [
                'Q6Arithmetic',             # Z₂⁶ групповая алгебра
                # XOR, Hamming, GERMES, bent-функции
            ],
            # ─── Конфигурация ───
            'config/config.py': ['YiJingConfig'],
            'constants.py': ['HEXAGRAM_NAMES', 'DOMAIN_NAMES'],
        },
    },

    'archetypist': {
        'description': '② АРХЕТИПИСТ — Физика, архетипы, сложные паттерны',
        'element': 'Земля 🌍', 'season': 'Осень', 'instrument': 'Виолончель',
        'files': {
            # ─── 15 паттернов внимания (как геометрия → нейросеть) ───
            'geometry/attention.py': [
                'HeisenbergAttention',       # принцип неопределённости
                'PalaceAttention',           # 8 дворцов × 8 гексаграмм
                'FlowerOfLifeGAT',           # граф-внимание «Цветок жизни»
                'CubeDiagonalAttention',     # симметрии углов куба
                'PrivilegedAxisAttention',   # выделенные оси
                'MobiusAttentionPattern',    # топологическая лента
                'TriangularAttentionBias',   # треугольные расстояния
                'SOLANAttention',            # визуальный алфавит
                'HexagramAttentionPattern',  # семантические расстояния
                'WeavingLoomArchitecture',   # ткацкий станок
                'QuadrantAttention',         # 4-квадрантное
                'RecursiveCubeAttention',    # рекурсивный куб
                'StructuralDefectLayer',     # дефекты структуры
                'FourLevelAttention',        # 4-уровневое
                'BidirectionalTriangularAttention',  # двунаправленное
            ],
            # ─── Маршрутизация ───
            'geometry/routing.py': [
                'GatedPathSelector',         # standard ↔ geometric gate
                'AdaptiveGatedPathSelector', # с адаптивным gate
                'GeometricSourceRouter',     # маршрутизатор источников
                'GeometricSourceMixer',      # смеситель источников
                'GateLogger',                # мониторинг gates
            ],
            # ─── Наутилус (7 камер) ───
            'geometry/nautilus.py': [
                'NautilusHierarchy',         # иерархия 7 камер
                'NautilusChamber',           # одна камера
                'NautilusScheduler',         # curriculum активации
            ],
            # ─── Интерлингва (64 архетипа) ───
            'geometry/interlingua_fixed.py': [
                'ArchetypalInterlinguaFixed', # 64 архетипа-медиатора
            ],
            # ─── Конвергенция (глифы ↔ токены) ───
            'geometry/convergence.py': [
                'GlyphComposer',             # bottom-up композиция
                'TokenAbstractor',           # top-down абстракция
                'ConvergenceBridge',         # мост глифы↔токены
            ],
        },
    },

    'algorithmist': {
        'description': '③ АЛГОРИТМИСТ — Химия, алгоритмы, комбинаторика',
        'element': 'Вода 💧', 'season': 'Зима', 'instrument': 'Духовые',
        'files': {
            # ─── FFN с геометрической маршрутизацией ───
            'geometry/ffn.py': [
                'SwiGLU',                    # LLaMA-style FFN
                'TrigramMoE',                # 8 экспертов по триграммам
                'DomainMoE',                 # 6 доменных экспертов
                'GeometricFFN',              # FFN через гиперкуб
                'MultiScaleHypercubeLayer',  # multi-scale проекция
            ],
            # ─── 6 теоретических источников ───
            'geometry/six_sources.py': [
                'PalaceSource',              # Склярова
                'AntipodalSource',           # Фомюк
                'TriangularSource',          # Андреев
                'KasatkinSource',            # Касаткин
                'HermannSource',             # Германн
                'BelyaevSource',             # Беляев
                'SixSourceLayer',            # все 6 вместе
            ],
            # ─── Иерархический MoE ───
            'hierarchical_moe.py': [
                'HierarchicalMoE',           # Q2→Q3→Q6 маршрутизация
                'Q6ExpertBank',              # 64 эксперта
                'BidirBridgeExpert',         # мост между группами
            ],
            # ─── Модели-алгоритмы ───
            'model.py': [
                'YiJingGPT',                 # основная модель
                'YiJingTransformer',         # backbone
                'YiJingTransformerLayer',    # один слой
            ],
            'variant3.py': ['Variant3GPT'],
            'hierarchical_e2.py': ['HierarchicalE2'],
            'nautilus_yijing.py': ['NautilusYiJing'],
            # ─── Маршрутизаторы ───
            'geometry/kasatkin_router.py': ['KasatkinQ6Router'],
            'expert_choice.py': ['ExpertChoiceRouter'],
            'diff_attn.py': ['DifferentialAttention'],
            'geometry/abriale.py': ['AbrialeLayer', 'IsotropicNet'],
            # ─── Оптимизация обучения ───
            'training/optim.py': ['optimizer factory + LLRD'],
            'training/regularization.py': ['TokenMerger', 'GradientNoise'],
            'training/distillation.py': ['DistillationLoss'],
            'lora.py': ['LoRALayer'],
        },
    },

    'linguist': {
        'description': '④ ЛИНГВИСТ — Язык, философия, психология, биология',
        'element': 'Воздух 🌬️', 'season': 'Весна', 'instrument': 'Ударные',
        'files': {
            # ─── Тернарное квантование (да/нет/не знаю) ───
            'geometry/quantizers.py': [
                'TernaryQuantizer',          # {-1, 0, +1} эпистемическая
                'FactoredYiJingQuantizer',   # 3+3 факторизация
                'FourStateQuantizer',        # 4 состояния
                'AntipodalQuantizer',        # антиподальная пара
            ],
            # ─── Грамматика и синтаксис ───
            'geometry/convergence.py': [
                'MatrixGrammar',             # 2D матричная грамматика
            ],
            # ─── Мосты и аналогии ───
            'pseudo_rag.py': [
                'PseudoRAGProjection',       # Q4→Q6 вложение
            ],
            # ─── Токенизация (семантика букв) ───
            'tokenizer/glyph_tokenizer.py': [
                'GlyphTokenizer',            # SOLAN-76 визуальный алфавит
            ],
            # ─── Генерация текста ───
            'inference/generate.py': ['generate()'],
            'inference/bridge_inference.py': ['AdvancedGenerator'],
            'speculative.py': ['SpeculativeDecoder'],
            # ─── Оценка и интерпретация ───
            'evaluation/eval_suite.py': ['evaluation harness'],
            # ─── Кросс-доменные аналогии ───
            'scripts/cross_domain_analogy.py': ['hexagram analogies'],
            'scripts/self_description.py': ['model self-reflection'],
            # ─── Данные (обработка языка) ───
            'data_utils/text_dataset.py': ['TextDataset'],
            'data_utils/svend4_dataset.py': ['Svend4Dataset (6 domains)'],
            # ─── Обучение NautilusMoME ───
            'scripts/train_nautilus_mome.py': [
                'organic_detect_domain()',
                'ExpertRouter',
                'Phase 0-11 training pipeline',
            ],
        },
    },
}


def cluster_inventory() -> str:
    """Выводит полную карту: какие файлы → какая профессия.

    Usage:
        print(cluster_inventory())
    """
    lines = []
    lines.append('=' * 70)
    lines.append('ЧЕТЫРЕ БРЕМЕНСКИХ МУЗЫКАНТА — Карта репозитория')
    lines.append('=' * 70)

    for name, cluster in FULL_CLUSTER_INVENTORY.items():
        lines.append('')
        lines.append(f'  {cluster["description"]}')
        lines.append(f'  {cluster["element"]} | {cluster["season"]} | {cluster["instrument"]}')
        lines.append(f'  {"─" * 60}')
        for filepath, items in cluster['files'].items():
            lines.append(f'    {filepath}:')
            for item in items:
                lines.append(f'      • {item}')
        lines.append('')

    lines.append('=' * 70)
    lines.append('Всего: 4 кластера × ~3 геометрических модуля = 12 инструментов')
    lines.append('Каждый Музыкант играет на СВОИХ инструментах.')
    lines.append('Вместе — оркестр. По одиночке — бессмысленны.')
    lines.append('=' * 70)
    return '\n'.join(lines)
