"""
YiJing-Transformer: трансформер с геометрической регуляризацией
на основе гиперкубов {-1,+1}^n (64 гексаграмм, 256 октограмм).

v8: RoPE scaling (NTK/linear), distillation, FLOPS profiler,
    improved generate (rep penalty, top-p, stop tokens),
    ONNX/TorchScript export, save/load pretrained.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .geometry import (
    get_trigrams,
    FactoredYiJingQuantizer,
    HierarchicalQuantizer,
    DeformableQuantizer,
    GumbelQuantizer,
    E8Quantizer,
    BianGuaTransform,
    HexagramAttentionPattern,
    RotaryEmbedding,
    apply_rotary_emb,
    ALiBi,
    SwiGLU,
    TrigramMoE,
    GatedPathSelector,
    GeometricAttention,
    GeometricFFN,
    GateLogger,
    GeometryCurriculumScheduler,
    # Phase 3: Adaptive specialization
    AdaptiveGatedPathSelector,
    TaskAwareRouter,
    DynamicCurriculumController,
    MultiScaleHypercubeLayer,
    # v51: Six-source integration modules
    FourStateQuantizer,
    AntipodalQuantizer,
    TriangularAttentionBias,
    PalaceAttention,
    DualEmbedding,
    QuadrantAttention,
    GraduatedBianGuaTransform,
    D4EquivariantLayer,
    DualModeHead,
    RecursiveCubeAttention,
    WeavingLoomArchitecture,
    FourLevelPositionalEncoding,
    BidirectionalTriangularAttention,
    CubeDiagonalAttention,
    MobiusAttentionPattern,
    PrivilegedAxisAttention,
    HeisenbergAttention,
    FlowerOfLifeGAT,
    StructuralDefectLayer,
    HexagramAttentionPattern,
    # v54: Anti-interference routing
    GeometricSourceRouter,
    GeometricSourceMixer,
    # v58: Bridge of Modules
    BridgeOfModules,
    # v59: AbrialeBridge + AdaptiveBridge + SourceSpecializer
    AbrialeBridgeMediator,
    AdaptiveBridgeOfModules,
    SourceSpecializer,
    # v60: Archetypal Interlingua
    ArchetypalInterlingua,
    # v61: BridgedInterlingua (двойная прослойка)
    BridgedInterlingua,
    # v54: Kasatkin 3D embedding
    CubicAttentionBias,
    CubicPositionalEncoding,
    # v55: Convergence Bridge
    ConvergenceBridge,
    MatrixGrammar,
    # v56: Ternary Quantizer
    TernaryQuantizer,
)
# v57: Abriale — событийно-управляемые изотропные N-местные связи (Пацкин)
from yijing_transformer.models.geometry.abriale import AbrialeLayer


def build_quantizer(cfg):
    """Фабрика квантизаторов по конфигурации."""
    if cfg.quantizer_type == 'factored6':
        return FactoredYiJingQuantizer(
            temp=cfg.temp, adaptive_temp=cfg.adaptive_temp
        )
    elif cfg.quantizer_type == 'hierarchical':
        return HierarchicalQuantizer(
            total_dim=cfg.quant_total_dim,
            group_dim=cfg.quant_group_dim,
            temp=cfg.temp,
            adaptive_temp=cfg.adaptive_temp,
        )
    elif cfg.quantizer_type == 'octogram':
        return HierarchicalQuantizer(
            total_dim=8,
            group_dim=4,  # 2 тетраграммы: 16×16 = 256
            temp=cfg.temp,
            adaptive_temp=cfg.adaptive_temp,
        )
    elif cfg.quantizer_type == 'deformable':
        return DeformableQuantizer(
            total_dim=cfg.quant_total_dim,
            group_dim=cfg.quant_group_dim,
            temp=cfg.temp,
        )
    elif cfg.quantizer_type == 'e8':
        return E8Quantizer(
            temp=cfg.temp,
            adaptive_temp=cfg.adaptive_temp,
        )
    elif cfg.quantizer_type == 'gumbel':
        return GumbelQuantizer(
            total_dim=cfg.quant_total_dim,
            group_dim=cfg.quant_group_dim,
            temp=cfg.temp,
            hard=cfg.use_gumbel,
            commitment_weight=cfg.commitment_weight,
        )
    elif cfg.quantizer_type == 'ternary':
        return TernaryQuantizer(
            total_dim=cfg.quant_total_dim,
            mode=getattr(cfg, 'ternary_mode', 'factored'),
            temp=cfg.temp,
            adaptive_temp=cfg.adaptive_temp,
            uncertainty_budget=getattr(cfg, 'ternary_uncertainty', 0.3),
            max_zeros=getattr(cfg, 'ternary_max_zeros', 2),
        )
    else:
        raise ValueError(f"Unknown quantizer_type: {cfg.quantizer_type}")


class YiJingAttention(nn.Module):
    """
    Multi-head attention с геометрическим bias из триграмм.

    Поддерживает:
    - GQA (Grouped Query Attention): n_kv_heads < n_heads
    - Sliding window attention: ограничение контекста
    - RoPE, FlashAttention, KV-cache

    Каждая из 8 голов получает направление одной из 8 триграмм.
    Bias — ранг-1 матрица (внешнее произведение проекций q и k на направление).
    Инициализация head_scales=0: модель начинает как стандартный трансформер.
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = cfg.use_rope and not getattr(cfg, 'use_alibi', False)
        self.use_alibi = getattr(cfg, 'use_alibi', False)
        self.use_flash = cfg.use_flash_attn
        self.sliding_window = cfg.sliding_window
        self.attention_sinks = getattr(cfg, 'attention_sinks', 0)

        # GQA: число KV голов (None = MHA, т.е. n_kv_heads = n_heads)
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
        assert cfg.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({cfg.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.n_rep = cfg.n_heads // self.n_kv_heads  # сколько раз повторять KV

        # Раздельные проекции для GQA
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        # RoPE (с поддержкой scaling для расширения контекста)
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                self.head_dim, max_seq_len=cfg.block_size, base=cfg.rope_base,
                scaling=getattr(cfg, 'rope_scaling', None),
                scaling_factor=getattr(cfg, 'rope_scaling_factor', 1.0),
            )

        # ALiBi (альтернатива RoPE)
        if self.use_alibi:
            self.alibi = ALiBi(cfg.n_heads, max_seq_len=cfg.block_size)

        # 8 триграмм как направления для голов
        trigrams = get_trigrams()  # (8, 3)
        trigrams_norm = F.normalize(trigrams, p=2, dim=1)
        self.register_buffer('head_dirs', trigrams_norm[:cfg.n_heads])
        self.head_scales = nn.Parameter(torch.zeros(cfg.n_heads))

    def _repeat_kv(self, x):
        """Повторяем KV головы для GQA: (B, n_kv_heads, T, D) -> (B, n_heads, T, D)."""
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, self.n_rep, T, D).reshape(B, H * self.n_rep, T, D)

    def forward(self, x, kv_cache=None, extra_attn_bias=None):
        B, T, C = x.shape

        # Раздельные проекции Q, K, V (для GQA)
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Применяем RoPE (с учётом позиции при KV-cache)
        if self.use_rope:
            offset = 0
            if kv_cache is not None:
                offset = kv_cache[0].size(2)
            cos, sin = self.rotary(T + offset)
            q = apply_rotary_emb(q, cos[offset:offset+T], sin[offset:offset+T])
            k = apply_rotary_emb(k, cos[offset:offset+T], sin[offset:offset+T])

        # KV-cache: сохраняем/конкатенируем (до repeat для GQA — экономим память)
        new_kv_cache = (k, v)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # GQA: повторяем KV головы до n_heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        S = k.size(2)  # полная длина (с кэшем)

        if self.use_flash and hasattr(F, 'scaled_dot_product_attention') and kv_cache is None and extra_attn_bias is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
            )

            # Геометрический bias
            if self.head_dim >= 3:
                q3, k3 = q[..., :3], k[..., :3]
                q_proj = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)
                k_proj = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
                geo_correction = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
                causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                geo_correction = geo_correction.masked_fill(~causal, 0.0)
                geo_attn = F.softmax(
                    geo_correction.masked_fill(~causal, float('-inf')),
                    dim=-1
                )
                geo_out = geo_attn @ v
                scales = self.head_scales.view(1, -1, 1, 1)
                out = out + scales * (geo_out - out).detach()
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale

            # ALiBi bias
            if self.use_alibi:
                offset = S - T
                alibi_bias = self.alibi(T, offset=offset)
                scores = scores + alibi_bias

            # v53: external attention biases (triangular, mobius, cube diagonal, etc.)
            if extra_attn_bias is not None:
                # Normalize to (B, 1, T, S) — broadcasts across heads
                if extra_attn_bias.dim() == 2:
                    bias = extra_attn_bias.unsqueeze(0).unsqueeze(0)
                elif extra_attn_bias.dim() == 3:
                    bias = extra_attn_bias.unsqueeze(1)
                else:
                    bias = extra_attn_bias
                # Handle KV-cache: bias is (*, T, T) but scores are (*, T, S)
                if bias.size(-1) < S:
                    pad_size = S - bias.size(-1)
                    bias = F.pad(bias, (pad_size, 0))
                scores = scores + bias

            # Causal mask с учётом KV-cache
            causal = torch.tril(torch.ones(S, S, device=x.device))
            causal = causal[S-T:S, :S]

            # Sliding window: маскируем всё за пределами окна
            if self.sliding_window is not None:
                for i in range(T):
                    pos = S - T + i  # абсолютная позиция в последовательности
                    window_start = max(0, pos - self.sliding_window + 1)
                    # Attention sinks: всегда видим первые N токенов
                    if self.attention_sinks > 0:
                        causal[i, :min(self.attention_sinks, window_start)] = 1.0
                    causal[i, min(self.attention_sinks, S):window_start] = 0.0
            elif self.attention_sinks > 0:
                # Attention sinks без sliding window: первые N всегда видны
                # (уже видны благодаря causal mask, но полезно при стриминге)
                pass

            causal = causal.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal == 0, float('-inf'))

            # Геометрический bias из триграмм
            if self.head_dim >= 3:
                q3 = q[..., :3]
                k3 = k[..., :3]
                q_proj = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)
                k_proj = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
                geo_bias = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
                scores = scores + self.head_scales.view(1, -1, 1, 1) * geo_bias

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out), new_kv_cache


class YiJingTransformerLayer(nn.Module):
    """
    Слой YiJing-трансформера:

    1. Attention с триграммным bias и RoPE (+ v51 attention модули)
    2. Факторизованная квантизация к гексаграммам (6D гиперкуб)
    3. 变卦 трансформация (опционально)
    4. D₄-эквивариантный слой (v51, опционально)
    5. FFN (SwiGLU или GELU) или MoE на триграммах
    """
    def __init__(self, cfg, layer_quant_dim=None):
        super().__init__()
        self.cfg = cfg

        # Attention — может быть заменён на v51-модуль
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.use_quadrant_attention = getattr(cfg, 'use_quadrant_attention', False)
        self.use_recursive_cube = getattr(cfg, 'use_recursive_cube', False)
        self.use_weaving_loom = getattr(cfg, 'use_weaving_loom', False)

        if self.use_quadrant_attention:
            self.attn = QuadrantAttention(cfg.d_model, cfg.n_heads)
            self._attn_returns_cache = False
        elif self.use_recursive_cube:
            self.attn = RecursiveCubeAttention(cfg.d_model, cfg.n_heads)
            self._attn_returns_cache = False
        elif self.use_weaving_loom:
            self.attn = WeavingLoomArchitecture(
                cfg.d_model,
                max_level=getattr(cfg, 'weaving_max_level', 3),
            )
            self._attn_returns_cache = False
        else:
            self.attn = YiJingAttention(cfg)
            self._attn_returns_cache = True

        # v51: additional attention biases (composable with standard attention)
        self.use_triangular_bias = getattr(cfg, 'use_triangular_bias', False)
        if self.use_triangular_bias:
            self.triangular_bias = TriangularAttentionBias(
                max_seq_len=cfg.block_size
            )

        self.use_palace_attention = getattr(cfg, 'use_palace_attention', False)
        if self.use_palace_attention:
            self.palace_attn = PalaceAttention(cfg.d_model, cfg.n_heads)

        self.use_mobius_bias = getattr(cfg, 'use_mobius_bias', False)
        if self.use_mobius_bias:
            self.mobius_pattern = MobiusAttentionPattern(
                max_seq_len=cfg.block_size
            )

        self.use_privileged_axis = getattr(cfg, 'use_privileged_axis', False)
        if self.use_privileged_axis:
            self.privileged_axis = PrivilegedAxisAttention(cfg.d_model)

        self.use_cube_diagonal = getattr(cfg, 'use_cube_diagonal', False)
        if self.use_cube_diagonal:
            self.cube_diag = CubeDiagonalAttention(cfg.d_model)

        # v54: Kasatkin cubic attention bias (3D distance-based)
        self.use_cubic_bias = getattr(cfg, 'use_cubic_bias', False)
        if self.use_cubic_bias:
            self.cubic_bias = CubicAttentionBias(max_seq_len=cfg.block_size)

        # v53: Heisenberg attention (Беляев 6.1) — additive enrichment
        self.use_heisenberg = getattr(cfg, 'use_heisenberg_attention', False)
        if self.use_heisenberg and not (self.use_quadrant_attention or self.use_recursive_cube or self.use_weaving_loom):
            self.heisenberg_attn = HeisenbergAttention(cfg.d_model)

        # v53: Hexagram attention pattern (64 fixed patterns)
        self.use_hex_attn_pattern = getattr(cfg, 'use_hex_attn_pattern', False)
        if self.use_hex_attn_pattern:
            self.hex_pattern = HexagramAttentionPattern(cfg.d_model, cfg.block_size)

        # v53: Flower of Life GAT (Беляев 6.6)
        self.use_flower_gat = getattr(cfg, 'use_flower_gat', False)
        if self.use_flower_gat:
            self.flower_gat = FlowerOfLifeGAT(cfg.d_model)

        # v53: Structural Defect bottleneck (Беляев)
        self.use_structural_defect = getattr(cfg, 'use_structural_defect', False)
        if self.use_structural_defect:
            self.structural_defect = StructuralDefectLayer(cfg.d_model)

        # v58: early flag reads for expanded enrichment sources
        self.use_cube_diagonal = getattr(cfg, 'use_cube_diagonal', False)
        self.use_d4_equivariant = getattr(cfg, 'use_d4_equivariant', False)
        self.use_dual_embedding = getattr(cfg, 'use_dual_embedding', False)

        # v54: Geometric Source Mixer — replaces fixed coefficients with learnable gates
        self.use_source_mixer = getattr(cfg, 'use_source_mixer', False)
        self.use_source_router = getattr(cfg, 'use_source_router', False)
        self.use_bridge_of_modules = getattr(cfg, 'use_bridge_of_modules', False)
        self.use_abriale_bridge = getattr(cfg, 'use_abriale_bridge', False)
        self.use_adaptive_bridge = getattr(cfg, 'use_adaptive_bridge', False)
        self.use_source_specialization = getattr(cfg, 'use_source_specialization', False)
        self.use_archetypal_interlingua = getattr(cfg, 'use_archetypal_interlingua', False)
        self.use_bridged_interlingua = getattr(cfg, 'use_bridged_interlingua', False)
        if (self.use_source_mixer or self.use_source_router or self.use_bridge_of_modules
                or self.use_abriale_bridge or self.use_adaptive_bridge
                or self.use_source_specialization or self.use_archetypal_interlingua
                or self.use_bridged_interlingua):
            # Count how many enrichment sources are active
            # v58: expanded to cover all 6 mathematical source groups
            self._enrichment_sources = []
            if self.use_heisenberg:
                self._enrichment_sources.append('heisenberg')
            if self.use_palace_attention:
                self._enrichment_sources.append('palace')
            if self.use_privileged_axis:
                self._enrichment_sources.append('privileged_axis')
            if self.use_flower_gat:
                self._enrichment_sources.append('flower_gat')
            if self.use_cube_diagonal:
                self._enrichment_sources.append('cube_diagonal')
            if self.use_d4_equivariant:
                self._enrichment_sources.append('d4_equivariant')
            if self.use_dual_embedding:
                self._enrichment_sources.append('dual_embedding')
            n_sources = len(self._enrichment_sources)
            if n_sources > 0:
                if self.use_bridged_interlingua:
                    # v61: BridgedInterlingua — двойная прослойка (Bridge → Archetype)
                    self.archetypal_interlingua = BridgedInterlingua(
                        cfg.d_model, n_sources,
                        n_archetypes=getattr(cfg, 'interlingua_n_archetypes', 64),
                        bridge_mode=getattr(cfg, 'bridged_bridge_mode', 'lightweight'),
                        use_ternary=getattr(cfg, 'interlingua_use_ternary', True),
                        uncertainty_budget=getattr(cfg, 'interlingua_uncertainty', 0.3),
                        n_heads=getattr(cfg, 'interlingua_n_heads', 4),
                        bridge_n_heads=getattr(cfg, 'bridged_bridge_n_heads', 2),
                        bridge_dropout=getattr(cfg, 'bridged_bridge_dropout', 0.1),
                        use_paired_bit=getattr(cfg, 'interlingua_use_paired_bit', False),
                    )
                    self.use_archetypal_interlingua = True  # reuse forward path
                elif self.use_archetypal_interlingua:
                    # v60: Archetypal Interlingua — hub-and-spoke посредник
                    self.archetypal_interlingua = ArchetypalInterlingua(
                        cfg.d_model, n_sources,
                        n_archetypes=getattr(cfg, 'interlingua_n_archetypes', 64),
                        d_bottleneck=getattr(cfg, 'interlingua_d_bottleneck', 0),
                        use_ternary=getattr(cfg, 'interlingua_use_ternary', True),
                        uncertainty_budget=getattr(cfg, 'interlingua_uncertainty', 0.3),
                        n_heads=getattr(cfg, 'interlingua_n_heads', 4),
                        use_paired_bit=getattr(cfg, 'interlingua_use_paired_bit', False),
                    )
                elif self.use_abriale_bridge:
                    # v59: AbrialeBridge — гибрид Abriale + Bridge
                    self.bridge_of_modules = AbrialeBridgeMediator(
                        cfg.d_model, n_sources,
                        n_heads=getattr(cfg, 'bridge_n_heads', 2),
                        dropout=getattr(cfg, 'bridge_dropout', 0.1),
                        bridge_mode=getattr(cfg, 'bridge_mode', 'lightweight'),
                        d_event=getattr(cfg, 'abriale_bridge_d_event', 64),
                        n_rules=getattr(cfg, 'abriale_bridge_n_rules', 64),
                        arity=getattr(cfg, 'abriale_bridge_arity', 2),
                    )
                    self.use_bridge_of_modules = True  # reuse forward path
                elif self.use_adaptive_bridge:
                    # v59: Adaptive Bridge — адаптивная глубина
                    self.bridge_of_modules = AdaptiveBridgeOfModules(
                        cfg.d_model, n_sources,
                        n_heads=getattr(cfg, 'bridge_n_heads', 2),
                        dropout=getattr(cfg, 'bridge_dropout', 0.1),
                        bridge_mode=getattr(cfg, 'bridge_mode', 'lightweight'),
                        max_levels=getattr(cfg, 'adaptive_bridge_max_levels', 0),
                    )
                    self.use_bridge_of_modules = True  # reuse forward path
                elif self.use_bridge_of_modules:
                    # v58: Bridge of Modules — иерархическая cross-attention медиация
                    self.bridge_of_modules = BridgeOfModules(
                        cfg.d_model, n_sources,
                        n_heads=getattr(cfg, 'bridge_n_heads', 2),
                        dropout=getattr(cfg, 'bridge_dropout', 0.1),
                        bridge_mode=getattr(cfg, 'bridge_mode', 'full'),
                    )
                elif self.use_source_specialization:
                    # v59: Source Specialization — доменная специализация
                    self.source_specializer = SourceSpecializer(
                        cfg.d_model, n_sources,
                        n_domains=getattr(cfg, 'n_domains', 4),
                    )
                elif self.use_source_router:
                    self.source_router = GeometricSourceRouter(
                        cfg.d_model, n_sources,
                        top_k=getattr(cfg, 'source_router_top_k', min(2, n_sources)),
                    )
                else:
                    self.source_mixer = GeometricSourceMixer(cfg.d_model, n_sources)
            else:
                self.use_source_mixer = False
                self.use_source_router = False
                self.use_bridge_of_modules = False
                self.use_abriale_bridge = False
                self.use_adaptive_bridge = False
                self.use_source_specialization = False

        # Квантизация к вершинам гиперкуба (bottleneck)
        # Multi-scale: layer_quant_dim overrides cfg.quant_total_dim
        quant_dim = layer_quant_dim if layer_quant_dim is not None else cfg.quant_total_dim
        self.quant_dim = quant_dim
        self.ln_hex = nn.LayerNorm(cfg.d_model)
        self.to_qd = nn.Linear(cfg.d_model, quant_dim, bias=False)
        self.from_qd = nn.Linear(quant_dim, cfg.d_model, bias=False)

        # Для multi-scale: создаём временный cfg с правильным quant_total_dim
        if layer_quant_dim is not None and layer_quant_dim != cfg.quant_total_dim:
            from dataclasses import replace
            layer_cfg = replace(cfg, quant_total_dim=quant_dim)
            self.quantizer = build_quantizer(layer_cfg)
        else:
            self.quantizer = build_quantizer(cfg)
        self.hex_scale = nn.Parameter(torch.tensor(cfg.hex_strength))

        # v51: Antipodal regularization (Фомюк/Герман)
        self.use_antipodal_reg = getattr(cfg, 'use_antipodal_reg', False)
        if self.use_antipodal_reg:
            self.antipodal = AntipodalQuantizer(
                temp=cfg.temp, adaptive_temp=cfg.adaptive_temp
            )

        # 变卦 (трансформация гексаграмм) — standard or graduated
        self.use_graduated_biangua = getattr(cfg, 'use_graduated_biangua', False)
        if self.use_graduated_biangua:
            self.bian_gua = GraduatedBianGuaTransform(cfg.d_model)
        elif cfg.use_bian_gua:
            self.bian_gua = BianGuaTransform(cfg.d_model)
        else:
            self.bian_gua = None

        # v51: D₄-эквивариантный слой (Фомюк 2.2)
        self.use_d4_equivariant = getattr(cfg, 'use_d4_equivariant', False)
        if self.use_d4_equivariant:
            self.d4_layer = D4EquivariantLayer(cfg.d_model)

        # v51: Dual Embedding (Касаткин 4.4)
        self.use_dual_embedding = getattr(cfg, 'use_dual_embedding', False)
        if self.use_dual_embedding:
            self.dual_emb = DualEmbedding(cfg.d_model)

        # FFN или MoE
        self.use_moe = cfg.use_hex_moe
        self.ln_ffn = nn.LayerNorm(cfg.d_model)

        if cfg.use_hex_moe:
            self.ffn = TrigramMoE(
                d_model=cfg.d_model,
                n_experts=cfg.n_experts,
                top_k=cfg.moe_top_k,
                ffn_hidden=cfg.ffn_hidden,
                dropout=cfg.dropout,
            )
        elif cfg.use_swiglu:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.ffn_hidden),
                nn.GELU(),
                nn.Linear(cfg.ffn_hidden, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # 1. Attention (standard or v51 module)
        h = self.ln_attn(x)
        new_kv = None

        # v53: compose external attention biases — injected into score computation
        extra_bias = None
        if self.use_triangular_bias:
            tri_bias = self.triangular_bias(T).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            extra_bias = tri_bias if extra_bias is None else extra_bias + tri_bias
        if self.use_mobius_bias:
            mob_bias = self.mobius_pattern(T).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            extra_bias = mob_bias if extra_bias is None else extra_bias + mob_bias
        if self.use_cube_diagonal:
            cd_bias = self.cube_diag.get_bias(h).unsqueeze(1)  # (B,1,T,T)
            extra_bias = cd_bias if extra_bias is None else extra_bias + cd_bias
        if self.use_hex_attn_pattern:
            hex_bias = self.hex_pattern(h, T)  # (B,1,T,T)
            extra_bias = hex_bias if extra_bias is None else extra_bias + hex_bias
        if self.use_cubic_bias:
            cub_bias = self.cubic_bias(T).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            extra_bias = cub_bias if extra_bias is None else extra_bias + cub_bias

        if self._attn_returns_cache:
            attn_out, new_kv = self.attn(h, kv_cache=kv_cache, extra_attn_bias=extra_bias)
        else:
            attn_out = self.attn(h)
            # For alternative attentions: compose bias post-hoc
            if extra_bias is not None:
                bias_w = F.softmax(extra_bias.squeeze(1), dim=-1)  # (B,T,T)
                attn_out = attn_out + 0.05 * torch.bmm(bias_w, h)

        # v54/v58/v59: Geometric source mixing (replaces fixed coefficients)
        _has_routing = (self.use_source_mixer or self.use_source_router
                        or self.use_bridge_of_modules or self.use_source_specialization
                        or self.use_archetypal_interlingua)
        if _has_routing:
            enrichments = []
            if self.use_heisenberg:
                enrichments.append(self.heisenberg_attn(h))
            if self.use_palace_attention:
                enrichments.append(self.palace_attn(h))
            if self.use_privileged_axis:
                axis_bias = self.privileged_axis.get_bias(h)
                axis_w = F.softmax(axis_bias, dim=-1)
                enrichments.append(torch.bmm(axis_w, h))
            if self.use_flower_gat:
                enrichments.append(self.flower_gat.enrich(h) if hasattr(self.flower_gat, 'enrich') else self.flower_gat(h) - h)
            # v58: expanded enrichments for 6-source bridge
            if self.use_cube_diagonal:
                cd_bias = self.cube_diag.get_bias(h)  # (B, T, T)
                cd_w = F.softmax(cd_bias, dim=-1)
                enrichments.append(torch.bmm(cd_w, h))
            if self.use_d4_equivariant:
                enrichments.append(self.d4_layer(h) - h)  # delta only
            if self.use_dual_embedding:
                enrichments.append(self.dual_emb(h) - h)  # delta only

            if enrichments:
                if self.use_archetypal_interlingua:
                    # v60: Archetypal Interlingua — hub-and-spoke посредник
                    attn_out = self.archetypal_interlingua(attn_out, enrichments)
                    self._source_routing_aux = 0.01 * self.archetypal_interlingua.get_interlingua_loss()
                elif self.use_bridge_of_modules:
                    # v58/v59: Bridge of Modules / AbrialeBridge / AdaptiveBridge
                    attn_out = self.bridge_of_modules(attn_out, enrichments)
                    # v59: AbrialeBridge has additional aux loss
                    if hasattr(self.bridge_of_modules, 'get_abriale_aux_loss'):
                        self._source_routing_aux = 0.01 * self.bridge_of_modules.get_abriale_aux_loss()
                    else:
                        self._source_routing_aux = 0.0
                elif self.use_source_specialization:
                    # v59: Source Specialization
                    attn_out = self.source_specializer(attn_out, enrichments)
                    self._source_routing_aux = 0.0
                elif self.use_source_router:
                    mixed = self.source_router(h, enrichments)
                    attn_out = attn_out + mixed
                    self._source_routing_aux = self.source_router._aux_loss
                else:
                    attn_out = self.source_mixer(attn_out, enrichments)
                    self._source_routing_aux = 0.0
            else:
                self._source_routing_aux = 0.0
        else:
            self._source_routing_aux = 0.0
            # v53: Heisenberg attention enrichment (additive)
            if self.use_heisenberg:
                attn_out = attn_out + 0.1 * self.heisenberg_attn(h)

            # v51: compose additional attention biases (additive post-processing)
            if self.use_palace_attention:
                palace_out = self.palace_attn(h)
                attn_out = attn_out + 0.1 * palace_out

            if self.use_privileged_axis:
                axis_bias = self.privileged_axis.get_bias(h)  # (B, T, T)
                axis_w = F.softmax(axis_bias, dim=-1)
                attn_out = attn_out + 0.05 * torch.bmm(axis_w, h)

        x = x + attn_out

        # v51: Dual Embedding enrichment (skip if handled by bridge/router)
        if self.use_dual_embedding and not _has_routing:
            x = self.dual_emb(x)

        # v53: Flower of Life GAT enrichment (only if not handled by source mixer/bridge)
        if self.use_flower_gat and not _has_routing:
            x = self.flower_gat(x)

        # 2. Квантизация к вершинам гиперкуба
        h = self.ln_hex(x)
        zq = self.to_qd(h)
        zq_out = self.quantizer(zq)
        if self.training:
            zq_out = zq_out * (1.0 + 0.001 * torch.randn_like(zq_out))
        x = x + self.hex_scale * self.from_qd(zq_out)

        # v51: Antipodal regularization (adds penalty during training)
        if self.use_antipodal_reg and self.training:
            self._antipodal_loss = self.antipodal.antipodal_loss(zq_out)
        else:
            self._antipodal_loss = 0.0

        # 3. 变卦 трансформация
        if self.bian_gua is not None:
            x = self.bian_gua(x)

        # 4. D₄-эквивариантный слой (skip if handled by bridge/router)
        if self.use_d4_equivariant and not _has_routing:
            x = self.d4_layer(x)

        # v53: Structural Defect bottleneck (geometric compression)
        if self.use_structural_defect:
            compressed = self.structural_defect(x)  # (B, 12, C)
            if compressed.size(1) < T:
                weights = F.softmax(
                    torch.matmul(x, compressed.transpose(-2, -1)) / math.sqrt(C),
                    dim=-1
                )
                x = x + 0.1 * torch.matmul(weights, compressed)

        # 5. FFN или MoE
        h_ffn = self.ln_ffn(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(h_ffn)
            x = x + ffn_out
            self._aux_loss = aux_loss
        else:
            x = x + self.ffn(h_ffn)
            self._aux_loss = 0.0

        # Aggregate v51+ auxiliary losses
        if isinstance(self._antipodal_loss, torch.Tensor):
            antipodal_w = getattr(self.cfg, 'antipodal_weight', 0.01)
            self._aux_loss = self._aux_loss + antipodal_w * self._antipodal_loss
        # v54: source routing balance loss
        if isinstance(getattr(self, '_source_routing_aux', 0.0), torch.Tensor):
            source_w = getattr(self.cfg, 'source_routing_weight', 0.01)
            self._aux_loss = self._aux_loss + source_w * self._source_routing_aux

        return x, new_kv


class YiJingTransformer(nn.Module):
    """Core transformer (без language model head)."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Multi-scale quantization: разная размерность по слоям
        if cfg.multi_scale_quant and cfg.quant_dim_schedule is not None:
            assert len(cfg.quant_dim_schedule) == cfg.n_layers, \
                f"quant_dim_schedule length {len(cfg.quant_dim_schedule)} != n_layers {cfg.n_layers}"
            self.layers = nn.ModuleList([
                YiJingTransformerLayer(cfg, layer_quant_dim=dim)
                for dim in cfg.quant_dim_schedule
            ])
        else:
            self.layers = nn.ModuleList(
                [YiJingTransformerLayer(cfg) for _ in range(cfg.n_layers)]
            )
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # Mixture of Depths: router для пропуска слоёв
        self.mod_capacity = getattr(cfg, 'mod_capacity', 1.0)
        if self.mod_capacity < 1.0:
            # Один router на слой: скаляр per-token решение
            self.mod_routers = nn.ModuleList([
                nn.Linear(cfg.d_model, 1, bias=False) for _ in range(cfg.n_layers)
            ])
        else:
            self.mod_routers = None

    def forward(self, x, kv_cache=None):
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None

            # Mixture of Depths: часть токенов пропускает слой
            if self.mod_routers is not None and self.training:
                B, T, D = x.shape
                router_logits = self.mod_routers[i](x).squeeze(-1)  # (B, T)
                k = max(1, int(T * self.mod_capacity))

                # Top-k токенов проходят через слой
                _, top_idx = router_logits.topk(k, dim=-1)  # (B, k)

                # Собираем выбранные токены
                top_idx_sorted, _ = top_idx.sort(dim=-1)
                batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
                x_selected = x[batch_idx, top_idx_sorted]  # (B, k, D)

                # Прогоняем через слой (без KV-cache при MoD)
                out_selected, new_kv = layer(x_selected, kv_cache=None)

                # Записываем обратно (residual для пропущенных токенов = identity)
                x_out = x.clone()
                x_out[batch_idx, top_idx_sorted] = out_selected
                x = x_out
                new_kv_cache.append(new_kv)

                # Сохраняем router loss (balance)
                layer._mod_loss = self._compute_mod_balance_loss(router_logits, k)
            else:
                if self.cfg.use_gradient_ckpt and self.training and kv_cache is None:
                    x, new_kv = grad_checkpoint(
                        layer, x, layer_cache, use_reentrant=False
                    )
                else:
                    x, new_kv = layer(x, kv_cache=layer_cache)
                new_kv_cache.append(new_kv)
        return self.final_norm(x), new_kv_cache

    @staticmethod
    def _compute_mod_balance_loss(router_logits, k):
        """Balance loss для MoD router — стимулирует равномерный выбор."""
        probs = torch.sigmoid(router_logits)  # (B, T)
        target = k / router_logits.shape[1]
        return ((probs.mean(dim=-1) - target) ** 2).mean() * 0.01

    def get_aux_loss(self):
        """Собирает aux loss со всех MoE слоёв + commitment loss + MoD loss."""
        total = 0.0
        for layer in self.layers:
            total = total + layer._aux_loss
            # Commitment loss от Gumbel квантизатора
            if hasattr(layer.quantizer, 'get_commitment_loss'):
                total = total + layer.quantizer.get_commitment_loss()
            # v56: Ternary uncertainty loss
            if hasattr(layer.quantizer, 'get_uncertainty_loss'):
                total = total + layer.quantizer.get_uncertainty_loss()
            # MoD balance loss
            if hasattr(layer, '_mod_loss'):
                total = total + layer._mod_loss
        return total


class YiJingGPT(nn.Module):
    """Полная языковая модель с И-Цзин архитектурой."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # v51: 4-уровневое позиционное кодирование Андреева
        self.use_four_level_pe = getattr(cfg, 'use_four_level_pe', False)
        # v54: Kasatkin 3D positional encoding
        self.use_cubic_pe = getattr(cfg, 'use_cubic_pe', False)
        if self.use_four_level_pe:
            self.four_level_pe = FourLevelPositionalEncoding(
                cfg.d_model, max_seq_len=cfg.block_size
            )
            self.pos_emb = None  # FourLevelPE заменяет стандартный PE
        elif self.use_cubic_pe:
            self.cubic_pe = CubicPositionalEncoding(
                cfg.d_model, max_seq_len=cfg.block_size
            )
            self.pos_emb = None  # CubicPE заменяет стандартный PE
        elif not cfg.use_rope and not getattr(cfg, 'use_alibi', False):
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        else:
            self.pos_emb = None

        # v51: Bidirectional triangular attention (Андреев 3.3)
        self.use_bidirectional_tri = getattr(cfg, 'use_bidirectional_tri', False)
        if self.use_bidirectional_tri:
            self.bi_tri_attn = BidirectionalTriangularAttention(
                cfg.d_model, max_seq_len=cfg.block_size
            )

        # Token Merging (ToMe) — сокращение числа токенов в FFN
        self.token_merge_ratio = getattr(cfg, 'token_merge_ratio', 0.0)

        # Label smoothing
        self.label_smoothing = getattr(cfg, 'label_smoothing', 0.0)

        # v55: Convergence Bridge — гибридная иерархия глифов ↔ токенов
        self.use_convergence_bridge = getattr(cfg, 'use_convergence_bridge', False)
        if self.use_convergence_bridge:
            self.convergence_bridge = ConvergenceBridge(
                d_model=cfg.d_model,
                n_clusters=getattr(cfg, 'convergence_n_clusters', 64),
                window_size=getattr(cfg, 'convergence_window_size', 4),
                stride=getattr(cfg, 'convergence_stride', 2),
                n_compose_layers=getattr(cfg, 'convergence_compose_layers', 1),
                n_heads=getattr(cfg, 'convergence_n_heads', 4),
            )
            # Проекция token ids → Q6 вершины (learned, не зависит от GlyphTokenizer)
            # Каждый токен vocab → 6D вершина Q6
            self.tok_to_q6 = nn.Linear(cfg.d_model, 6, bias=False)

        # v56: Matrix Grammar — 2D матричная грамматика (Atamiri/Аймара)
        self.use_matrix_grammar = getattr(cfg, 'use_matrix_grammar', False)
        if self.use_matrix_grammar:
            self.matrix_grammar = MatrixGrammar(
                d_model=cfg.d_model,
                n_rows=getattr(cfg, 'matrix_grammar_rows', 8),
                n_cols=getattr(cfg, 'matrix_grammar_cols', 8),
                n_heads=getattr(cfg, 'matrix_grammar_heads', 4),
            )

        # v57: Abriale — событийно-управляемые изотропные N-местные связи (Пацкин)
        self.use_abriale = getattr(cfg, 'use_abriale', False)
        if self.use_abriale:
            self.abriale = AbrialeLayer(
                d_model=cfg.d_model,
                d_event=getattr(cfg, 'abriale_d_event', 64),
                n_heads=getattr(cfg, 'abriale_n_heads', 4),
                arity=getattr(cfg, 'abriale_arity', 2),
                n_rules=getattr(cfg, 'abriale_n_rules', 64),
                n_hits=getattr(cfg, 'abriale_n_hits', 4),
                n_alternatives=getattr(cfg, 'abriale_n_alternatives', 2),
                n_event_types=getattr(cfg, 'abriale_n_event_types', 8),
                dropout=cfg.dropout,
            )
            self._abriale_balance_weight = getattr(cfg, 'abriale_balance_weight', 0.01)

        self.core = YiJingTransformer(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

        # Weight tying: share weights between embedding and output head
        if cfg.weight_tying:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        b, t = idx.size()
        x = self.tok_emb(idx)

        # Позиционное кодирование
        if self.use_four_level_pe:
            x = x + self.four_level_pe(t, device=idx.device)
        elif self.use_cubic_pe:
            x = x + self.cubic_pe(t, device=idx.device)
        elif self.pos_emb is not None:
            offset = 0
            if kv_cache is not None and kv_cache[0] is not None:
                offset = kv_cache[0][0].size(2)
            x = x + self.pos_emb[:, offset:offset+t, :]

        # v53: Bidirectional triangular pre-processing (Андреев 3.3)
        if self.use_bidirectional_tri:
            bi_mask = self.bi_tri_attn.get_mask(t)  # (t, t) soft directional mask
            x = x * bi_mask.diag().unsqueeze(0).unsqueeze(-1)

        # v55: Convergence Bridge — обогащение через гибридную иерархию
        convergence_info = None
        if self.use_convergence_bridge:
            # Генерируем Q6 вершины из token embeddings (learned projection)
            glyph_vertices = torch.tanh(self.tok_to_q6(x))  # (B, T, 6) → soft Q6
            x, convergence_info = self.convergence_bridge(x, glyph_vertices)

        # v56: Matrix Grammar — 2D axial attention обогащение
        if self.use_matrix_grammar:
            x = x + self.matrix_grammar(x)

        # v57: Abriale — событийно-управляемые изотропные N-местные связи
        abriale_info = None
        if self.use_abriale:
            x, abriale_info = self.abriale(x)

        hidden, new_kv_cache = self.core(x, kv_cache=kv_cache)
        logits = self.head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                label_smoothing=self.label_smoothing,
            )
            # Добавляем MoE aux loss
            aux = self.core.get_aux_loss()
            if isinstance(aux, torch.Tensor):
                loss = loss + aux

            # v55: Convergence auxiliary loss
            if convergence_info is not None:
                conv_loss = self.convergence_bridge.get_convergence_loss(
                    convergence_info['assignments']
                )
                loss = loss + conv_loss

            # v57: Abriale auxiliary loss (балансировка правил)
            if abriale_info is not None and 'hit_weights' in abriale_info:
                abriale_loss = self.abriale.get_auxiliary_loss(
                    abriale_info['hit_weights']
                )
                loss = loss + self._abriale_balance_weight * abriale_loss

        return logits, loss, new_kv_cache

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, repetition_window=64, stop_tokens=None,
                 use_cache=True):
        """
        Авторегрессивная генерация с KV-cache.

        Args:
            idx: (B, T) начальная последовательность
            max_new_tokens: максимум новых токенов
            temperature: температура сэмплирования
            top_k: top-k фильтрация
            top_p: nucleus sampling (top-p)
            repetition_penalty: штраф за повторения (1.0 = нет штрафа)
            repetition_window: окно для repetition penalty
            stop_tokens: list[int] — стоп-токены (генерация прекращается)
            use_cache: использовать KV-cache
        """
        kv_cache = None
        for _ in range(max_new_tokens):
            if use_cache and kv_cache is not None:
                idx_input = idx[:, -1:]
            else:
                idx_input = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]

            logits, _, kv_cache = self(idx_input, kv_cache=kv_cache if use_cache else None)
            logits = logits[:, -1, :]  # (B, V)

            # Repetition penalty
            if repetition_penalty != 1.0:
                past = idx[:, -repetition_window:]
                for b in range(idx.size(0)):
                    seen = past[b].unique()
                    for token_id in seen:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            logits = logits / temperature

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus sampling)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Убираем токены с cumulative > top_p
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                for b in range(logits.size(0)):
                    indices_to_remove = sorted_indices[b][sorted_mask[b]]
                    logits[b, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop tokens
            if stop_tokens is not None:
                if idx_next.item() in stop_tokens:
                    break

            # Обрезаем кэш если превышаем block_size
            if use_cache and kv_cache is not None:
                cache_len = kv_cache[0][0].size(2)
                if cache_len >= self.cfg.block_size:
                    kv_cache = None
        return idx

    @torch.no_grad()
    def beam_search(self, idx, max_new_tokens, beam_width=4, temperature=1.0,
                    length_penalty=0.6):
        """
        Beam search decoding.

        Args:
            idx: (1, T) начальная последовательность (batch=1)
            max_new_tokens: максимум новых токенов
            beam_width: ширина пучка
            temperature: температура для logits
            length_penalty: штраф за длину (α в score/len^α)

        Returns:
            best_sequence: (1, T+generated) лучшая последовательность
        """
        assert idx.size(0) == 1, "Beam search supports batch_size=1"
        device = idx.device

        # Инициализация: beam_width копий
        beams = [(idx, 0.0)]  # (sequence, log_prob)

        def _score(seq, score):
            length = seq.size(1) - idx.size(1)
            return score / (length ** length_penalty) if length > 0 else score

        for step in range(max_new_tokens):
            candidates = []
            for seq, score in beams:
                seq_input = seq if seq.size(1) <= self.cfg.block_size else seq[:, -self.cfg.block_size:]
                logits, _, _ = self(seq_input)
                logits = logits[:, -1, :] / temperature
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    token = top_indices[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, token], dim=1)
                    new_score = score + top_log_probs[i].item()
                    candidates.append((new_seq, new_score))

            candidates.sort(key=lambda x: _score(x[0], x[1]), reverse=True)
            beams = candidates[:beam_width]

        best_seq, _ = max(beams, key=lambda x: _score(x[0], x[1]))
        return best_seq

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        hex_params = 0
        hex_keys = ['to_qd', 'from_qd', 'to_6d', 'from_6d', 'hex_scale',
                     'head_scales', 'head_dirs', 'bian_gua', 'quantizer',
                     'trigram_dirs', 'router_proj', 'hex_attn_pattern']
        for name, p in self.named_parameters():
            if any(k in name for k in hex_keys):
                hex_params += p.numel()
        return total, hex_params

    def save_pretrained(self, path):
        """Сохраняет модель + config в один файл."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'config': self.cfg,
            'model_state_dict': self.state_dict(),
        }, path)

    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        """Загружает модель из файла."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model = cls(cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    def estimate_flops(self, seq_len=None):
        """
        Оценка FLOPS для одного forward pass.

        Считает основные операции:
        - Attention: QKV projections + attention scores + output projection
        - FFN: 2-3 линейных слоя
        - Quantizer: projection + softmax
        """
        if seq_len is None:
            seq_len = self.cfg.block_size
        d = self.cfg.d_model
        h = self.cfg.n_heads
        hd = self.cfg.head_dim
        L = self.cfg.n_layers
        V = self.cfg.vocab_size
        T = seq_len
        ffn_h = self.cfg.ffn_hidden
        kv_h = self.cfg.n_kv_heads or h

        flops = 0

        # Embedding lookup (negligible) + per layer:
        for _ in range(L):
            # Q, K, V projections
            flops += 2 * T * d * (h * hd)          # Q
            flops += 2 * T * d * (kv_h * hd)       # K
            flops += 2 * T * d * (kv_h * hd)       # V

            # Attention scores: Q @ K^T
            flops += 2 * T * T * h * hd

            # Attention @ V
            flops += 2 * T * T * h * hd

            # Output projection
            flops += 2 * T * d * d

            # FFN
            if self.cfg.use_swiglu:
                flops += 2 * T * d * ffn_h * 3  # w1, w2, w3
            else:
                flops += 2 * T * d * ffn_h * 2  # up + down

            # Quantizer projection (small)
            qd = self.cfg.quant_total_dim
            flops += 2 * T * d * qd * 2  # to_qd + from_qd

        # LM head
        flops += 2 * T * d * V

        return flops

    def estimate_flops_str(self, seq_len=None):
        """Человекочитаемая строка с FLOPS."""
        flops = self.estimate_flops(seq_len)
        if flops >= 1e12:
            return f"{flops/1e12:.2f} TFLOPS"
        elif flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOPS"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOPS"
        return f"{flops:.0f} FLOPS"

    @torch.no_grad()
    def quantization_analytics(self):
        """Анализ квантизации по всем слоям: расстояние до кодбука, использование кодовых слов."""
        analytics = {}
        x_probe = torch.randn(1, 16, self.cfg.d_model, device=next(self.parameters()).device)

        for i, layer in enumerate(self.core.layers):
            info = {
                'hex_scale': layer.hex_scale.item(),
                'quant_dim': layer.quant_dim,
            }

            # Пропускаем через проекцию и квантизатор
            h = layer.ln_hex(x_probe)
            zq = layer.to_qd(h)
            zq_out = layer.quantizer(zq)

            # Расстояние до кодбука (мера "жёсткости" квантизации)
            info['quant_error'] = (zq - zq_out).pow(2).mean().item()
            info['quant_snr'] = (zq_out.pow(2).mean() / (zq - zq_out).pow(2).mean().clamp(min=1e-8)).item()

            # Температура квантизатора
            if hasattr(layer.quantizer, 'current_temp'):
                temp = layer.quantizer.current_temp
                info['temp'] = temp.item() if isinstance(temp, torch.Tensor) else temp

            # GQA info
            info['n_kv_heads'] = layer.attn.n_kv_heads
            info['n_rep'] = layer.attn.n_rep

            # Commitment loss
            if hasattr(layer.quantizer, 'get_commitment_loss'):
                info['commitment_loss'] = layer.quantizer.get_commitment_loss().item()

            # BianGua stats
            if layer.bian_gua is not None:
                probs = torch.sigmoid(layer.bian_gua.change_logits)
                info['bian_gua_scale'] = layer.bian_gua.scale.item()
                info['active_lines'] = (probs > 0.5).sum().item()

            analytics[f'layer_{i}'] = info

        return analytics


# ==================== ЧИСТЫЙ ГЕОМЕТРИЧЕСКИЙ СЛОЙ ====================

class PureGeometricLayer(nn.Module):
    """
    Слой, использующий ТОЛЬКО геометрические компоненты.

    Заменяет стандартный attention на GeometricAttention,
    FFN на TrigramMoE, и использует BianGua + квантизацию
    с hex_strength=1.0 (полный вклад).

    Цель: изолировать геометрический потенциал для чистого тестирования.
    """
    def __init__(self, cfg):
        super().__init__()
        # Геометрический attention (без стандартного dot-product)
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = GeometricAttention(cfg)

        # Квантизация (полный вклад)
        self.ln_hex = nn.LayerNorm(cfg.d_model)
        self.to_qd = nn.Linear(cfg.d_model, cfg.quant_total_dim, bias=False)
        self.from_qd = nn.Linear(cfg.quant_total_dim, cfg.d_model, bias=False)
        self.quantizer = build_quantizer(cfg)
        # hex_strength = 1.0 для чистого геометрического теста
        self.hex_scale = nn.Parameter(torch.tensor(1.0))

        # BianGua всегда включена
        self.bian_gua = BianGuaTransform(cfg.d_model)

        # FFN: геометрический MoE на триграммах
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = GeometricFFN(cfg)

        self._aux_loss = 0.0

    def forward(self, x, kv_cache=None):
        # 1. Геометрический attention
        x = x + self.attn(self.ln_attn(x))

        # 2. Квантизация к вершинам гиперкуба (полный вклад)
        h = self.ln_hex(x)
        zq = self.to_qd(h)
        zq_out = self.quantizer(zq)
        if self.training:
            zq_out = zq_out * (1.0 + 0.001 * torch.randn_like(zq_out))
        x = x + self.hex_scale * self.from_qd(zq_out)

        # 3. BianGua трансформация
        x = self.bian_gua(x)

        # 4. Геометрический FFN (TrigramMoE)
        ffn_out, aux_loss = self.ffn(self.ln_ffn(x))
        x = x + ffn_out
        self._aux_loss = aux_loss

        # Нет KV-cache в чистом геометрическом режиме
        return x, None


# ==================== ГИБРИДНЫЙ СЛОЙ С ГЕЙТОМ ====================

class HybridGatedLayer(nn.Module):
    """
    Гибридный слой: стандартный и геометрический пути с гейтовым выбором.

    Для КАЖДОГО подблока (attention, quantization+bian_gua, FFN)
    модель самостоятельно выбирает, какой путь использовать,
    через обучаемый гейт.

    v51: поддержка QuadrantAttention, RecursiveCubeAttention, WeavingLoom,
    D₄-эквивариантного слоя, и антиподальной регуляризации.
    """
    def __init__(self, cfg, layer_quant_dim=None):
        super().__init__()
        self.cfg = cfg
        gate_bias = cfg.gate_init_bias

        # === Attention: стандартный и геометрический ===
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.std_attn = YiJingAttention(cfg)  # стандартный (с trigram bias)

        # v51: выбор геометрического attention
        use_quadrant = getattr(cfg, 'use_quadrant_attention', False)
        use_recursive = getattr(cfg, 'use_recursive_cube', False)
        use_weaving = getattr(cfg, 'use_weaving_loom', False)

        if use_quadrant:
            self.geo_attn = QuadrantAttention(cfg.d_model, cfg.n_heads)
        elif use_recursive:
            self.geo_attn = RecursiveCubeAttention(cfg.d_model, cfg.n_heads)
        elif use_weaving:
            self.geo_attn = WeavingLoomArchitecture(
                cfg.d_model, max_level=getattr(cfg, 'weaving_max_level', 3)
            )
        else:
            self.geo_attn = GeometricAttention(cfg)  # default
        self.attn_gate = GatedPathSelector(cfg.d_model, init_bias=gate_bias)

        # === Квантизация (геометрический путь) ===
        quant_dim = layer_quant_dim if layer_quant_dim is not None else cfg.quant_total_dim
        self.ln_hex = nn.LayerNorm(cfg.d_model)
        self.to_qd = nn.Linear(cfg.d_model, quant_dim, bias=False)
        self.from_qd = nn.Linear(quant_dim, cfg.d_model, bias=False)
        self.quantizer = build_quantizer(cfg)
        self.hex_scale = nn.Parameter(torch.tensor(cfg.hex_strength))

        # BianGua (геометрический путь) — standard or graduated
        if getattr(cfg, 'use_graduated_biangua', False):
            self.bian_gua = GraduatedBianGuaTransform(cfg.d_model)
        elif cfg.use_bian_gua:
            self.bian_gua = BianGuaTransform(cfg.d_model)
        else:
            self.bian_gua = None

        # v51: Antipodal regularization
        self.use_antipodal_reg = getattr(cfg, 'use_antipodal_reg', False)
        if self.use_antipodal_reg:
            self.antipodal = AntipodalQuantizer(
                temp=cfg.temp, adaptive_temp=cfg.adaptive_temp
            )

        # Гейт для квантизации+BianGua (применять vs пропускать)
        self.quant_gate = GatedPathSelector(cfg.d_model, init_bias=gate_bias)

        # v51: D₄-эквивариантный слой
        self.use_d4_equivariant = getattr(cfg, 'use_d4_equivariant', False)
        if self.use_d4_equivariant:
            self.d4_layer = D4EquivariantLayer(cfg.d_model)

        # === FFN: стандартный и геометрический ===
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        if cfg.use_swiglu:
            self.std_ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        else:
            self.std_ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.ffn_hidden),
                nn.GELU(),
                nn.Linear(cfg.ffn_hidden, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )
        self.geo_ffn = GeometricFFN(cfg)
        self.ffn_gate = GatedPathSelector(cfg.d_model, init_bias=gate_bias)

        # Ссылка на гейт для логирования
        self.path_gate = self.attn_gate  # главный гейт для мониторинга

        self._aux_loss = 0.0

    def forward(self, x, kv_cache=None):
        # 1. Attention с гейтом
        h = self.ln_attn(x)
        std_out, new_kv = self.std_attn(h, kv_cache=kv_cache)
        geo_out = self.geo_attn(h)
        attn_out = self.attn_gate(std_out, geo_out)
        x = x + attn_out

        # 2. Квантизация + BianGua с гейтом
        h = self.ln_hex(x)
        # Стандартный путь: identity (ничего не делаем)
        identity = torch.zeros_like(h)
        # Геометрический путь: квантизация + BianGua
        zq = self.to_qd(h)
        zq_out = self.quantizer(zq)
        if self.training:
            zq_out = zq_out * (1.0 + 0.001 * torch.randn_like(zq_out))
        geo_contrib = self.hex_scale * self.from_qd(zq_out)
        if self.bian_gua is not None:
            # BianGua применяется к геометрическому вкладу
            geo_contrib = geo_contrib + self.bian_gua.scale * self.bian_gua.proj_from_6d(
                self.bian_gua.proj_to_6d(h) * (1 - 2 * torch.sigmoid(self.bian_gua.change_logits))
            )
        quant_out = self.quant_gate(identity, geo_contrib)
        x = x + quant_out

        # v51: Antipodal regularization
        self._antipodal_loss = 0.0
        if self.use_antipodal_reg and self.training:
            self._antipodal_loss = self.antipodal.antipodal_loss(zq_out)

        # v51: D₄-эквивариантный слой (между quantization и FFN)
        if self.use_d4_equivariant:
            x = self.d4_layer(x)

        # 3. FFN с гейтом
        h = self.ln_ffn(x)
        std_ffn_out = self.std_ffn(h)
        geo_ffn_out, aux_loss = self.geo_ffn(h)
        ffn_out = self.ffn_gate(std_ffn_out, geo_ffn_out)
        x = x + ffn_out
        self._aux_loss = aux_loss

        # Aggregate v51 auxiliary losses
        if isinstance(self._antipodal_loss, torch.Tensor):
            antipodal_w = getattr(self.cfg, 'antipodal_weight', 0.01)
            self._aux_loss = self._aux_loss + antipodal_w * self._antipodal_loss

        return x, new_kv

    def get_all_gate_stats(self):
        """Статистика всех трёх гейтов."""
        return {
            'attention': self.attn_gate.get_gate_stats(),
            'quantization': self.quant_gate.get_gate_stats(),
            'ffn': self.ffn_gate.get_gate_stats(),
        }


# ==================== PURE GEOMETRIC GPT ====================

class PureGeometricGPT(nn.Module):
    """
    Языковая модель, использующая ТОЛЬКО геометрические компоненты.

    Для чистого ablation study: показывает потолок геометрии без помощи
    стандартных механизмов. Позволяет ответить на вопрос:
    «Насколько геометрия способна решать задачу самостоятельно?»
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if not cfg.use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        else:
            self.pos_emb = None

        self.layers = nn.ModuleList(
            [PureGeometricLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

        if cfg.weight_tying:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        b, t = idx.size()
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :t, :]

        for layer in self.layers:
            x, _ = layer(x)

        x = self.final_norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            # Aux loss от MoE
            for layer in self.layers:
                aux = layer._aux_loss
                if isinstance(aux, torch.Tensor):
                    loss = loss + aux

        return logits, loss, None

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        # В чисто геометрической модели все параметры — геометрические
        return total, total

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Простая авторегрессивная генерация."""
        for _ in range(max_new_tokens):
            idx_input = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _, _ = self(idx_input)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==================== HYBRID GATED GPT ====================

class HybridGatedGPT(nn.Module):
    """
    Языковая модель с гейтовым выбором между стандартным и геометрическим путём.

    Принцип: не навязывать решение, а предоставить свободу выбора.
    Модель самостоятельно определяет оптимальный баланс через обучение.

    Поддерживает:
    - GateLogger для прозрачности
    - GeometryCurriculumScheduler для постепенного обучения
    - Детальную статистику по каждому гейту
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if not cfg.use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        else:
            self.pos_emb = None

        self.layers = nn.ModuleList(
            [HybridGatedLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Gate logger
        self.gate_logger = GateLogger()

        # Curriculum scheduler
        if cfg.curriculum_strategy_geo != 'none':
            self.curriculum = GeometryCurriculumScheduler(
                strategy=cfg.curriculum_strategy_geo,
                total_steps=cfg.total_steps,
                warmup_fraction=cfg.curriculum_warmup_fraction,
                target_strength=cfg.curriculum_target_strength,
            )
        else:
            self.curriculum = None

        self.apply(self._init_weights)

        if cfg.weight_tying:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def update_curriculum(self, step):
        """Обновляет hex_strength по curriculum schedule."""
        if self.curriculum is not None:
            strength = self.curriculum.get_strength(step)
            for layer in self.layers:
                layer.hex_scale.data.fill_(strength)

    def log_gates(self, step):
        """Логирует состояние всех гейтов."""
        return self.gate_logger.log_step(step, self.layers)

    def forward(self, idx, targets=None, kv_cache=None):
        b, t = idx.size()
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :t, :]

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, kv_cache=layer_cache)
            new_kv_cache.append(new_kv)

        x = self.final_norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            for layer in self.layers:
                aux = layer._aux_loss
                if isinstance(aux, torch.Tensor):
                    loss = loss + aux

        return logits, loss, new_kv_cache

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        geo_keys = ['to_qd', 'from_qd', 'hex_scale', 'head_scales', 'head_dirs',
                     'bian_gua', 'quantizer', 'trigram_dirs', 'router_proj',
                     'geo_attn', 'geo_ffn', 'gate_proj', 'path_gate',
                     'attn_gate', 'quant_gate', 'ffn_gate']
        geo_params = 0
        for name, p in self.named_parameters():
            if any(k in name for k in geo_keys):
                geo_params += p.numel()
        return total, geo_params

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Авторегрессивная генерация."""
        kv_cache = None
        for _ in range(max_new_tokens):
            if kv_cache is not None:
                idx_input = idx[:, -1:]
            else:
                idx_input = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]

            logits, _, kv_cache = self(idx_input, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if kv_cache is not None and kv_cache[0] is not None:
                cache_len = kv_cache[0][0].size(2)
                if cache_len >= self.cfg.block_size:
                    kv_cache = None
        return idx

    def get_gate_summary(self):
        """Сводка по всем гейтам для отчёта."""
        summary = {}
        for i, layer in enumerate(self.layers):
            summary[f'layer_{i}'] = layer.get_all_gate_stats()
        return summary


# ==================== PHASE 3: ADAPTIVE HYBRID GPT ====================

class AdaptiveHybridLayer(nn.Module):
    """
    Адаптивный гибридный слой с расширенными гейтами:
    - AdaptiveGatedPathSelector (контентно-зависимый, с температурой)
    - TaskAwareRouter (sequence-level routing)
    - MultiScaleHypercubeLayer (разный масштаб гиперкуба по слоям)
    """
    def __init__(self, cfg, layer_idx: int, hypercube_dim: int = 3):
        super().__init__()
        self.layer_idx = layer_idx
        gate_bias = cfg.gate_init_bias

        # === Attention ===
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.std_attn = YiJingAttention(cfg)
        self.geo_attn = GeometricAttention(cfg)
        self.attn_gate = AdaptiveGatedPathSelector(
            cfg.d_model, n_heads=min(cfg.n_heads, 4), init_bias=gate_bias
        )

        # === Multi-scale hypercube quantization ===
        self.ln_hex = nn.LayerNorm(cfg.d_model)
        self.multiscale_quant = MultiScaleHypercubeLayer(
            cfg.d_model, hypercube_dim=hypercube_dim
        )
        self.quant_gate = AdaptiveGatedPathSelector(cfg.d_model, init_bias=gate_bias)

        # BianGua
        if cfg.use_bian_gua:
            self.bian_gua = BianGuaTransform(cfg.d_model)
        else:
            self.bian_gua = None

        # === FFN ===
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        if cfg.use_swiglu:
            self.std_ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        else:
            self.std_ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.ffn_hidden),
                nn.GELU(),
                nn.Linear(cfg.ffn_hidden, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )
        self.geo_ffn = GeometricFFN(cfg)
        self.ffn_gate = AdaptiveGatedPathSelector(cfg.d_model, init_bias=gate_bias)

        # Task-aware router (sequence-level)
        self.task_router = TaskAwareRouter(cfg.d_model, n_strategies=4)

        self._aux_loss = 0.0

    def forward(self, x, kv_cache=None):
        # Task-aware bias (sequence-level)
        task_bias = self.task_router(x)  # (B, 1, 1)

        # 1. Attention
        h = self.ln_attn(x)
        std_out, new_kv = self.std_attn(h, kv_cache=kv_cache)
        geo_out = self.geo_attn(h)
        attn_out = self.attn_gate(std_out, geo_out)
        x = x + attn_out

        # 2. Multi-scale quantization + BianGua
        h = self.ln_hex(x)
        identity = torch.zeros_like(h)
        geo_contrib = self.multiscale_quant(h)
        if self.bian_gua is not None:
            geo_contrib = geo_contrib + self.bian_gua(h) - h
        quant_out = self.quant_gate(identity, geo_contrib - h)
        x = x + quant_out

        # 3. FFN
        h = self.ln_ffn(x)
        std_ffn_out = self.std_ffn(h)
        geo_ffn_out, aux_loss = self.geo_ffn(h)
        ffn_out = self.ffn_gate(std_ffn_out, geo_ffn_out)
        x = x + ffn_out
        self._aux_loss = aux_loss

        return x, new_kv

    def get_all_gate_stats(self):
        stats = {
            'attention': self.attn_gate.get_gate_stats(),
            'quantization': self.quant_gate.get_gate_stats(),
            'ffn': self.ffn_gate.get_gate_stats(),
            'task_router': self.task_router.get_strategy_stats(),
        }
        return stats


class AdaptiveHybridGPT(nn.Module):
    """
    Phase 3: Adaptive Hybrid GPT.

    Расширения по сравнению с HybridGatedGPT:
    - AdaptiveGatedPathSelector (контентно-зависимый + температура)
    - TaskAwareRouter (определяет тип задачи на уровне последовательности)
    - MultiScaleHypercubeLayer (разный масштаб по слоям)
    - DynamicCurriculumController (адаптирует силу геометрии)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if not cfg.use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        else:
            self.pos_emb = None

        # Multi-scale: нижние слои — малые гиперкубы, верхние — большие
        # Layer 0,1 → dim 2 (биграммы), Layer 2,3 → dim 3 (триграммы),
        # Layer 4,5 → dim 4 (тетраграммы), Layer 6+ → dim 6 (гексаграммы)
        def get_hypercube_dim(layer_idx, n_layers):
            progress = layer_idx / max(n_layers - 1, 1)
            if progress < 0.25:
                return 2
            elif progress < 0.5:
                return 3
            elif progress < 0.75:
                return 4
            else:
                return 6

        self.layers = nn.ModuleList([
            AdaptiveHybridLayer(
                cfg, layer_idx=i,
                hypercube_dim=get_hypercube_dim(i, cfg.n_layers)
            )
            for i in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Gate logger
        self.gate_logger = GateLogger()

        # Dynamic curriculum
        self.dynamic_curriculum = DynamicCurriculumController(
            base_strength=cfg.curriculum_target_strength,
            adapt_rate=0.005,
        )

        self.apply(self._init_weights)
        if cfg.weight_tying:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def update_curriculum(self, step):
        """Обновляет curriculum на основе текущих гейтов."""
        # Собираем среднее значение гейтов
        gate_values = []
        gate_stds = []
        for layer in self.layers:
            stats = layer.get_all_gate_stats()
            for gate_name, s in stats.items():
                if 'gate_mean' in s:
                    gate_values.append(s['gate_mean'])
                if 'gate_std' in s:
                    gate_stds.append(s['gate_std'])

        if gate_values:
            avg_gate = sum(gate_values) / len(gate_values)
            avg_std = sum(gate_stds) / len(gate_stds) if gate_stds else 0.1
            new_strength = self.dynamic_curriculum.update(avg_gate, avg_std)

            # Обновляем масштаб геометрии
            for layer in self.layers:
                layer.multiscale_quant.scale.data.fill_(new_strength)

    def log_gates(self, step):
        return self.gate_logger.log_step(step, self.layers)

    def forward(self, idx, targets=None, kv_cache=None):
        b, t = idx.size()
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :t, :]

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_kv = layer(x, kv_cache=layer_cache)
            new_kv_cache.append(new_kv)

        x = self.final_norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            for layer in self.layers:
                aux = layer._aux_loss
                if isinstance(aux, torch.Tensor):
                    loss = loss + aux

        return logits, loss, new_kv_cache

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        geo_keys = ['to_qd', 'from_qd', 'hex_scale', 'head_scales', 'head_dirs',
                     'bian_gua', 'quantizer', 'trigram_dirs', 'router_proj',
                     'geo_attn', 'geo_ffn', 'gate_proj', 'path_gate',
                     'attn_gate', 'quant_gate', 'ffn_gate', 'multiscale',
                     'task_router', 'strategy', 'vertices']
        geo_params = 0
        for name, p in self.named_parameters():
            if any(k in name for k in geo_keys):
                geo_params += p.numel()
        return total, geo_params

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_input = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _, _ = self(idx_input)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_gate_summary(self):
        summary = {}
        for i, layer in enumerate(self.layers):
            summary[f'layer_{i}'] = layer.get_all_gate_stats()
        return summary
