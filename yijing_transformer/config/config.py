from dataclasses import dataclass, field
from typing import Optional


@dataclass
class YiJingConfig:
    vocab_size: int = 2048
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    block_size: int = 512
    dropout: float = 0.05
    bias: bool = False

    # И-Цзин параметры
    hex_strength: float = 0.01   # масштаб вклада гексаграммной ветки
    temp: float = 0.3            # температура квантизации (мягче, чем E8: 0.1)
    use_bian_gua: bool = True    # использовать 变卦 трансформацию
    adaptive_temp: bool = True   # обучаемая температура квантизации

    # Иерархическая квантизация
    quantizer_type: str = 'factored6'  # 'factored6', 'hierarchical', 'octogram', 'deformable'
    quant_total_dim: int = 6     # размерность бутылочного горлышка
    quant_group_dim: int = 3     # размерность группы для иерархической квант.
    use_hex_attn_pattern: bool = False  # гексаграммные паттерны attention

    # v4: новые параметры
    weight_tying: bool = True    # разделять веса tok_emb и head
    use_gumbel: bool = False     # Gumbel-Softmax вместо soft quantization
    commitment_weight: float = 0.25  # вес commitment loss
    use_gradient_ckpt: bool = False  # gradient checkpointing
    multi_scale_quant: bool = False  # разная размерность квантизации по слоям
    quant_dim_schedule: Optional[list] = None  # [6, 6, 8, 8, ...] размерности по слоям

    # Архитектурные расширения
    use_rope: bool = True        # Rotary Position Embeddings
    use_swiglu: bool = True      # SwiGLU вместо GELU FFN
    use_flash_attn: bool = False # FlashAttention (требует flash-attn)
    ffn_multiplier: float = 4.0  # множитель для FFN hidden dim
    rope_base: float = 10000.0   # base для RoPE

    # v5: GQA и sliding window
    n_kv_heads: Optional[int] = None  # число KV голов для GQA (None = MHA)
    sliding_window: Optional[int] = None  # размер окна sliding window attention
    use_amp: bool = False        # mixed precision training (AMP)

    # v7: LoRA адаптеры
    use_lora: bool = False       # включить LoRA адаптеры
    lora_rank: int = 8           # ранг LoRA
    lora_alpha: float = 16.0     # scaling factor
    lora_dropout: float = 0.0    # dropout на LoRA
    lora_targets: Optional[list] = None  # целевые модули: ['q_proj', 'v_proj', ...], None = все

    # v7: Speculative decoding
    draft_n_layers: int = 2      # число слоёв в draft модели
    draft_d_model: Optional[int] = None  # d_model для draft (None = cfg.d_model // 2)
    speculative_k: int = 4       # число токенов для спекулятивной генерации

    # v8: RoPE scaling для расширения контекста
    rope_scaling: Optional[str] = None  # None, 'ntk', 'linear'
    rope_scaling_factor: float = 1.0    # множитель расширения контекста

    # v8: Distillation
    distill_alpha: float = 0.5   # баланс hard/soft loss (1.0 = только soft)
    distill_temp: float = 2.0    # температура для soft targets

    # v10: ALiBi, attention sinks, EMA
    use_alibi: bool = False      # ALiBi вместо RoPE (нельзя с use_rope=True)
    attention_sinks: int = 0     # число «sink» токенов для streaming (0 = выкл)
    use_ema: bool = False        # EMA model averaging
    ema_decay: float = 0.999     # EMA decay rate
    early_stop_patience: int = 0 # early stopping (0 = выкл)

    # v11: Token Merging, label smoothing, gradient noise, cosine restarts
    token_merge_ratio: float = 0.0   # доля токенов для слияния (0 = выкл, 0.25 = 25%)
    label_smoothing: float = 0.0     # label smoothing (0 = выкл, 0.1 = типично)
    gradient_noise_eta: float = 0.0  # gradient noise scale (0 = выкл)
    cosine_restarts: int = 0         # число warm restarts (0 = обычный cosine)

    # v12: Mixture of Depths, µP, dynamic temperature
    mod_capacity: float = 1.0        # MoD capacity ratio (1.0 = все токены, 0.5 = 50%)
    use_mup: bool = False            # µP инициализация для width scaling
    mup_base_width: int = 128        # базовая ширина для µP

    # v13: selective checkpointing, weight decay scheduling
    checkpoint_every: int = 0        # selective checkpoint каждые N слоёв (0 = выкл)
    wd_end: float = 0.01             # конечный weight decay (для scheduling)

    # v14: differential attention, KV-cache quantization, layerwise LR
    use_diff_attn: bool = False      # Differential Attention
    quantize_kv_cache: bool = False  # INT8 KV-cache
    layerwise_lr_decay: float = 1.0  # LR decay per layer (1.0 = выкл)

    # v17: Prefix Tuning, Multi-Token Prediction
    prefix_len: int = 0              # Prefix tuning длина (0 = выкл)
    mtp_n_future: int = 0            # Multi-Token Prediction горизонт (0 = выкл)

    # v18: WSD scheduler, NEFTune, LAMB
    wsd_decay_steps: int = 0         # WSD decay фаза (0 = выкл)
    neftune_alpha: float = 0.0       # NEFTune noise scale (0 = выкл, 5-15 типично)
    bpe_dropout: float = 0.0         # BPE-Dropout rate (0 = выкл, 0.1 типично)

    # v19: Sophia, RMSNorm, Chunked Prefill, EMA decay warmup
    use_rmsnorm: bool = False        # RMSNorm вместо LayerNorm
    sophia_rho: float = 0.04         # Sophia clipping threshold
    chunk_prefill_size: int = 0      # Chunked prefill (0 = выкл)
    ema_decay_warmup: int = 0        # EMA decay warmup шагов (0 = выкл)

    # v20: DoRA, Sparse Attention, Cyclic Batch
    use_dora: bool = False           # DoRA вместо LoRA
    dora_rank: int = 8               # DoRA rank
    sparse_attn_window: int = 0      # Sparse attention window (0 = выкл)
    cyclic_max_batch: int = 0        # Cyclic batch max size (0 = выкл)

    # v21: AGC, GradEMA
    agc_clip_factor: float = 0.0     # AGC clip factor (0 = выкл, 0.01 типично)
    grad_ema_decay: float = 0.0      # GradEMA decay (0 = выкл, 0.95 типично)

    # v22: Matryoshka, SWA, Reptile
    matryoshka_dims: Optional[list] = None  # Matryoshka dimensions (None = выкл)
    swa_start: int = 0               # SWA start step (0 = выкл)
    swa_freq: int = 1                # SWA averaging frequency

    # v23: Lookahead, EMA, Gradient Noise, Curriculum
    lookahead_k: int = 0             # Lookahead steps (0 = выкл, 5 типично)
    lookahead_alpha: float = 0.5     # Lookahead interpolation
    ema_decay: float = 0.0           # EMA decay (0 = выкл, 0.999 типично)
    grad_noise_eta: float = 0.0      # Gradient noise eta (0 = выкл)
    curriculum_strategy: str = 'none'  # Curriculum: none, linear, sqrt, step

    # v24: LLRD, Weight Decay Scheduler, Loss Spike
    llrd_decay: float = 0.0          # LLRD decay factor (0 = выкл, 0.8 типично)
    wd_schedule: str = 'constant'    # Weight decay schedule: constant, linear, cosine
    spike_threshold: float = 3.0     # Loss spike detection threshold (z-score)

    # v25: Gradient Centralization, Token Mixing
    use_grad_centralization: bool = False  # Gradient Centralization
    use_token_mixing: bool = False        # Token Mixing MLP вместо attention
    token_mixing_expansion: int = 2       # Token Mixing expansion factor

    # v26: Sparse Attention, Progressive Resizing, NCE
    sparse_attention_pattern: str = 'none'  # none, local, strided, axial, combined
    sparse_window_size: int = 8             # Sparse attention window
    progressive_min_len: int = 0            # Progressive resizing min (0 = выкл)
    nce_negatives: int = 0                  # NCE negatives (0 = выкл, use CE)

    # v27: SAM, Dynamic Temperature, Sequence Packing
    sam_rho: float = 0.0             # SAM perturbation radius (0 = выкл, 0.05 типично)
    dynamic_temperature: bool = False  # Dynamic temperature scaling
    use_sequence_packing: bool = False  # Sequence packing for efficiency

    # v28: Lookahead, SWA, Grad Accumulation, Label Smoothing
    lookahead_k: int = 0               # Lookahead sync every k steps (0 = выкл)
    lookahead_alpha: float = 0.5       # Lookahead interpolation
    swa_start: int = 0                 # SWA start step (0 = выкл)
    grad_accum_steps: int = 1          # Gradient accumulation steps
    label_smoothing_warmup: int = 0    # Label smoothing warmup steps (0 = выкл)

    # v29: EMA, Curriculum, Gradient Noise, Weight Decay Schedule
    ema_decay: float = 0.0             # EMA decay (0 = выкл, 0.999 типично)
    curriculum_strategy: str = 'none'  # none, linear, sqrt, step, exponential
    gradient_noise_eta: float = 0.0    # Gradient noise amplitude (0 = выкл)
    wd_schedule: str = 'constant'      # constant, linear, cosine, proportional

    # v30: Gradient Centralization, AGC, Freeze Schedule
    use_grad_centralization: bool = False  # Gradient Centralization
    agc_clipping: float = 0.0             # AGC clipping factor (0 = выкл, 0.01 типично)
    freeze_strategy: str = 'none'         # none, top_down, bottom_up, all_at_once
    loss_spike_threshold: float = 3.0     # Loss spike detection threshold (std devs)

    # v31: Spectral Reg, Gradient Surgery, Token Freq Weighting
    spectral_lambda: float = 0.0          # Spectral regularization (0 = выкл)
    use_gradient_surgery: bool = False     # Gradient surgery for multi-task
    token_freq_strategy: str = 'none'     # none, inverse, sqrt_inverse, log_inverse, smoothed

    # v32: Cosine Warm Restarts, Multi-Scale Loss, Data Mixing
    cosine_T0: int = 0                    # SGDR period (0 = выкл)
    cosine_T_mult: int = 2               # SGDR period multiplier
    multi_scale_chunks: str = ''          # Chunk sizes for multi-scale loss (e.g. "4,8")
    data_mixing_strategy: str = 'constant'  # constant, linear_shift, loss_based, temperature

    # v33: Gradient Penalty, Polyak, Hard Example Mining
    gradient_penalty_lambda: float = 0.0  # Gradient penalty (0 = выкл)
    gradient_penalty_mode: str = 'r2'     # r1, r2, gp
    polyak_tau: float = 0.0               # Polyak averaging tau (0 = выкл, 0.005 типично)
    hard_example_fraction: float = 0.0    # Hard example mining fraction (0 = выкл)

    # v34: LLRD, Token Dropout, Convergence Detection, Activation Checkpointing
    llrd_decay_rate: float = 1.0          # LLRD decay rate (1.0 = выкл, 0.9 типично)
    token_drop_prob: float = 0.0          # Token dropout probability (0 = выкл)
    convergence_patience: int = 0         # Convergence detector patience (0 = выкл)
    activation_checkpoint_ratio: float = 0.0  # Activation checkpointing ratio (0 = выкл)

    # v35: Curriculum, Gradient Noise, Dynamic Batch, Stability Monitor
    curriculum_strategy: str = 'none'     # Curriculum learning: 'none', 'linear', 'sqrt', 'step'
    grad_noise_eta: float = 0.0           # Gradient noise eta (0 = выкл)
    dynamic_batch_size: bool = False      # Dynamic batch size scaling
    stability_monitor: bool = False       # Training stability monitor

    # v36: EMA, Gradient Vaccine, AGC, Weight Standardization
    ema_decay: float = 0.0                # EMA decay (0 = выкл, 0.999 типично)
    gradient_vaccine_lambda: float = 0.0  # Gradient vaccine strength (0 = выкл)
    agc_clip_factor: float = 0.0          # Adaptive Gradient Clipping factor (0 = выкл, 0.01 типично)
    weight_standardization: bool = False  # Weight Standardization

    # v37: Gradient Centralization, Lookahead, SWA, Batch Warmup, Gradient Penalty
    gradient_centralization: bool = False  # Gradient Centralization
    lookahead_k: int = 0                  # Lookahead k (0 = выкл, 5 типично)
    swa_start_epoch: int = 0              # SWA start epoch (0 = выкл)
    batch_size_warmup_steps: int = 0      # Batch size warmup steps (0 = выкл)
    gradient_penalty_lambda: float = 0.0  # R1 gradient penalty (0 = выкл)

    # v38: Multi-Scale Loss, Grad Accum Scheduler, Freezing, Checkpoints, SGDR
    multi_scale_loss: bool = False        # Multi-scale loss
    grad_accum_schedule: bool = False     # Dynamic gradient accumulation
    gradual_unfreezing: bool = False      # Gradual layer unfreezing (ULMFiT)
    checkpoint_top_k: int = 0            # Top-k checkpoint saving (0 = выкл)
    cosine_warm_restarts: bool = False    # SGDR cosine warm restarts

    # v39: Gradient Projection, Loss Spike Recovery, Scheduled Dropout, WD Scheduler
    gradient_projection: bool = False     # PCGrad gradient projection
    loss_spike_recovery: bool = False     # Auto rollback on loss spike
    scheduled_dropout: bool = False       # Dynamic dropout rate
    weight_decay_schedule: str = 'none'   # WD schedule: 'none', 'linear', 'cosine'

    # v40: Spectral Norm, Gradient Histogram, Mixed Precision, Progress Estimator
    spectral_norm: bool = False           # Spectral normalization
    gradient_histogram: bool = False      # Gradient histogram tracking
    mixed_precision: bool = False         # Mixed precision manager
    progress_estimator: bool = False      # Training progress estimator

    # v41: Activation Checkpointing, Parameter Freezer, Loss Landscape, Optimizer Inspector
    activation_checkpointing: bool = False # Activation checkpointing
    param_freezer: bool = False           # Parameter freezer
    loss_landscape_probe: bool = False    # Loss landscape probing
    optimizer_inspector: bool = False     # Optimizer state inspector
    batch_size_finder: bool = False       # Batch size finder

    # v42: SGDR, Gradient Accumulation, Model EMA, Curriculum, Stability Monitor
    cosine_warm_restarts: bool = False    # SGDR scheduler
    gradient_accumulation: int = 1        # Gradient accumulation steps
    model_ema: bool = False               # Model EMA
    curriculum_learning: bool = False     # Curriculum learning
    stability_monitor: bool = False       # Training stability monitor

    # v43: Knowledge Distillation, Label Smoothing, Focal Loss, Contrastive, R-Drop
    knowledge_distillation: bool = False  # Knowledge distillation
    label_smoothing: float = 0.0          # Label smoothing epsilon
    focal_loss_gamma: float = 0.0         # Focal loss gamma (0 = disabled)
    contrastive_loss: bool = False        # Contrastive loss
    rdrop_alpha: float = 0.0              # R-Drop alpha (0 = disabled)

    # v44: Gradient Noise, Lookahead, Layer-wise LR, SWA, WSD Schedule
    gradient_noise: bool = False          # Gradient noise injection
    lookahead_k: int = 0                  # Lookahead k (0 = disabled)
    layerwise_lr_decay: float = 1.0       # Layer-wise LR decay (1.0 = disabled)
    swa: bool = False                     # Stochastic weight averaging
    wsd_schedule: bool = False            # Warmup-Stable-Decay schedule

    # v45: Gradient Centralization, AdaFactor LR, Gradient Penalty, SAM, Lion
    gradient_centralization: bool = False  # Gradient centralization
    adafactor_lr_scaling: bool = False    # AdaFactor-like LR scaling
    gradient_penalty: float = 0.0         # Gradient penalty lambda (0 = disabled)
    sam_rho: float = 0.0                  # SAM rho (0 = disabled)
    lion_optimizer: bool = False          # Lion optimizer

    # v46: Token Loss Weighting, Seq Packing, Dynamic Padding, Attn Sink, Spec Decoding
    token_loss_weighting: str = 'uniform' # Token loss weighting mode
    sequence_packing: bool = False        # Sequence packing
    dynamic_padding: bool = False         # Dynamic padding
    attention_sink_cache: bool = False    # Attention sink cache
    speculative_decoding: bool = False    # Speculative decoding

    # v47: GradAccum+LossScaling, ParamNoise, EMA Schedule, MultiObjLoss, CheckpointMgr
    grad_accum_loss_scaling: bool = False  # Gradient accumulation + loss scaling
    parameter_noise: float = 0.0          # Parameter noise std (0 = disabled)
    ema_schedule: bool = False            # EMA with schedule
    multi_objective_loss: bool = False    # Multi-objective loss balancer
    checkpoint_manager: bool = False      # Training checkpoint manager

    # v48: BeamSearch, Nucleus Sampling, RepetitionPenalty, TempScheduler, KVCache
    beam_search: bool = False             # Beam search decoding
    nucleus_sampling: bool = False        # Top-k/Top-p sampling
    repetition_penalty: float = 1.0       # Repetition penalty (1.0 = disabled)
    temperature_scheduler: str = 'constant'  # Temperature schedule mode
    kv_cache_manager: bool = False        # KV cache manager

    # v49: Архитектурный режим и гейтовый выбор пути
    architecture_mode: str = 'standard'  # 'standard', 'pure_geometry', 'hybrid'
    gate_init_bias: float = 0.0          # начальный bias гейта (0 = равные шансы)
    curriculum_strategy_geo: str = 'none'  # 'none', 'linear', 'warmup_hold', 'cosine', 'step', 'geometric_first'
    curriculum_warmup_fraction: float = 0.3  # доля шагов на warmup curriculum
    curriculum_target_strength: float = 0.1  # целевая сила геометрии
    log_gate_every: int = 100            # логировать гейты каждые N шагов
    # v49: Gradient Surgery, AGC, Cosine Loss, Mixup, CutMix
    gradient_surgery: bool = False        # PCGrad gradient surgery
    adaptive_grad_clip: float = 0.0       # AGC clip factor (0 = disabled)
    cosine_loss: bool = False             # Cosine similarity loss
    mixup_alpha: float = 0.0              # Mixup alpha (0 = disabled)
    cutmix_alpha: float = 0.0             # CutMix alpha (0 = disabled)

    # v50: Entropy Reg, Confidence Penalty, Distill Temp Anneal, Prog Freeze, Metrics
    entropy_regularization: float = 0.0   # Entropy reg weight (0 = disabled)
    confidence_penalty: float = 0.0       # Confidence penalty weight (0 = disabled)
    distill_temp_annealing: bool = False  # Distillation temperature annealing
    progressive_freezing: bool = False    # Progressive layer freezing
    metrics_aggregator: bool = False      # Training metrics aggregator

    # v51: Gradient Vaccine, NormFree, Sharpness, LR Finder, Grad Flow
    gradient_vaccine: bool = False        # Gradient vaccine filtering
    norm_free: bool = False               # Scaled weight standardization
    sharpness_estimator: bool = False     # Sharpness estimation
    lr_finder: bool = False               # Learning rate finder
    gradient_flow_monitor: bool = False   # Gradient flow monitoring

    # v52: Sliding Window, ALiBi, RoPE, Flash Attn, MQA/GQA
    sliding_window_size: int = 0          # Sliding window attention (0 = disabled)
    alibi_bias: bool = False              # ALiBi positional bias
    rope_embeddings: bool = False         # Rotary position embeddings
    flash_attention: bool = False         # Flash attention approximation
    # n_kv_heads already defined above as Optional[int] = None

    # MoE на гексаграммах
    use_hex_moe: bool = False    # Mixture of Experts на 8 триграммах
    moe_top_k: int = 2           # сколько экспертов активировать
    n_experts: int = 8           # число экспертов (= число триграмм)

    # DomainMoE: эксперты по доменам корпуса (ai_agents, infosystems, ...)
    use_domain_moe: bool = False          # включить доменную MoE вместо SwiGLU/TrigramMoE
    domain_moe_n_experts: int = 6        # один эксперт на домен
    domain_moe_top_k: int = 2            # активных экспертов за forward
    domain_supervision_weight: float = 0.1  # вес loss доменной специализации

    # v51 геометрия: интеграция шести источников
    codebook_order: str = 'fuxi'         # 'fuxi' (binary), 'wenwang' (traditional), 'learned'
    use_palace_attention: bool = False   # block-sparse attention по 8 дворцам (Склярова)
    use_antipodal_reg: bool = False      # антиподальная регуляризация (Фомюк/Герман)
    antipodal_weight: float = 0.01       # вес антиподального штрафа
    use_triangular_bias: bool = False    # треугольный attention bias (Андреев)
    use_four_state: bool = False         # 4-состояния линий: 4096 кодбук (Склярова 1.2)
    use_dual_embedding: bool = False     # dual 6D+3D embedding (Касаткин)
    use_quadrant_attention: bool = False # 4-квадрантный attention (Беляев)
    use_graduated_biangua: bool = False  # градуированная 变卦 (Склярова 1.3)
    use_d4_equivariant: bool = False    # D₄-эквивариантный слой (Фомюк 2.2)
    use_heisenberg_attention: bool = False  # Гейзенберг-attention (Беляев 6.1)
    use_dual_mode_head: bool = False    # мезонный/барионный head (Беляев 6.4)
    use_recursive_cube: bool = False    # рекурсивный куб-attention (Беляев 6.5)
    use_weaving_loom: bool = False      # 4-уровневый ткацкий станок (Беляев 6.8)
    weaving_max_level: int = 3          # макс. уровень ткацкого станка (1-4)
    use_mobius_bias: bool = False       # Мёбиусов attention-паттерн (Беляев 6.3)
    use_privileged_axis: bool = False   # привилегированная ось (Касаткин 4.1)
    use_cube_diagonal: bool = False     # 4 типа диагоналей куба (Касаткин 4.2)
    use_four_level_pe: bool = False     # 4-уровневое позиционное кодирование (Андреев 3.1)
    use_bidirectional_tri: bool = False # двунаправленный треугольный attention (Андреев 3.3)
    use_flower_gat: bool = False        # Цветок Жизни GAT (Беляев 6.6)
    use_structural_defect: bool = False  # Structural Defect bottleneck 16→12 (Беляев)
    curriculum_strategy: str = 'linear' # 'linear', 'geometric_first', 'triangular' (Андреев 3.4)

    # v54: Anti-interference source routing
    use_source_mixer: bool = False      # learnable per-source gates (lightweight)
    use_source_router: bool = False     # MoE-style top-k source routing
    source_router_top_k: int = 2        # how many sources to select per token
    source_routing_weight: float = 0.01 # aux loss weight for load balancing

    # v58: Bridge of Modules — иерархическая медиация между источниками
    use_bridge_of_modules: bool = False  # мост модулей (замена source_router)
    bridge_n_heads: int = 2              # головы cross-attention в каждом мосте
    bridge_dropout: float = 0.1          # dropout в cross-attention мостов
    bridge_mode: str = 'full'            # 'full' (cross-attn) или 'lightweight' (bilinear)

    # v54: Kasatkin 3D embedding
    use_cubic_bias: bool = False        # 3D distance-based attention bias (Касаткин)
    use_cubic_pe: bool = False          # 3D positional encoding (x,y,z in 4×4×4 cube)

    # v55: Convergence Bridge — гибридная иерархия глифов ↔ токенов
    use_convergence_bridge: bool = False  # конвергентный мост глиф↔токен
    use_glyph_tokenizer: bool = False     # использовать GlyphTokenizer (SOLAN-76) вместо learned tok_to_q6
    convergence_n_clusters: int = 64      # число кластеров (64 = гексаграммы)
    convergence_window_size: int = 4      # окно для GlyphComposer
    convergence_stride: int = 2           # шаг окна
    convergence_compose_layers: int = 1   # слои self-attention в GlyphComposer
    convergence_n_heads: int = 4          # головы cross-attention в ConvergenceLayer

    # v63: Geometric prior — SOLAN Q6 таблица как индуктивный bias
    use_glyph_prior: bool = False         # blend learned Q6 с фиксированным SOLAN lookup

    # v56: Ternary Quantizer — трёхзначная логика {-1,0,+1} (Лукасевич/Аймара/变爻)
    use_ternary_quantizer: bool = False   # использовать тернарный квантизатор
    ternary_mode: str = 'factored'        # 'full' (729), 'factored' (2×27), 'sparse'
    ternary_uncertainty: float = 0.3      # бюджет неопределённости [0,1]
    ternary_max_zeros: int = 2            # макс. число 变爻 (для sparse режима)

    # v56: Matrix Grammar — 2D матричная грамматика сигилов (Atamiri/Аймара)
    use_matrix_grammar: bool = False      # матричная грамматика
    matrix_grammar_rows: int = 8          # строки (синтаксические роли)
    matrix_grammar_cols: int = 8          # столбцы (семантические слоты)
    matrix_grammar_heads: int = 4         # головы axial attention

    # v59: AbrialeBridge — гибрид Abriale + Bridge (событийная медиация)
    use_abriale_bridge: bool = False      # AbrialeBridge вместо BridgeOfModules
    abriale_bridge_d_event: int = 64      # размерность событий в AbrialeBridge
    abriale_bridge_n_rules: int = 64      # число правил Абриале в bridge
    abriale_bridge_arity: int = 2         # арность N-местных связей (2 или 3)

    # v59: Adaptive Bridge — адаптивная глубина bridge
    use_adaptive_bridge: bool = False     # адаптивная глубина bridge
    adaptive_bridge_max_levels: int = 0   # макс. уровней (0 = полное дерево)

    # v59: Source specialization — доменная специализация источников
    use_source_specialization: bool = False  # специализация источников по доменам
    n_domains: int = 4                       # число доменов для специализации

    # v60: Archetypal Interlingua — hub-and-spoke посредник (Atamiri/Aymara siwi)
    use_archetypal_interlingua: bool = False  # интерлингва вместо мостов
    interlingua_n_archetypes: int = 64       # число архетипов (64 = гексаграммы)
    interlingua_d_bottleneck: int = 0        # bottleneck в кодировщиках (0 = d//4)
    interlingua_use_ternary: bool = True     # тернарная квантизация {-1,0,+1}
    interlingua_uncertainty: float = 0.3     # бюджет неопределённости [0,1]
    interlingua_n_heads: int = 4             # головы cross-attention в readout

    # v61: BridgedInterlingua — двойная прослойка (Module→Bridge→Archetype→Core)
    use_bridged_interlingua: bool = False   # гибрид мостов + архетипов
    bridged_bridge_mode: str = 'lightweight'  # 'lightweight' или 'full' для мостов
    bridged_bridge_n_heads: int = 2          # головы cross-attention в мостах (full mode)
    bridged_bridge_dropout: float = 0.1      # dropout в мостах

    # v63: NautilusHierarchy — иерархическое упорядочивание геометрических модулей
    use_nautilus: bool = False            # включить Наутилус-иерархию
    nautilus_mode: str = 'sequential'     # 'sequential' (каскад) или 'parallel'
    nautilus_init_scale: float = 0.01     # начальный масштаб камер
    nautilus_warmup_steps: int = 2000     # шагов для прогрессивной активации
    nautilus_chambers: str = 'all'        # 'all' или список через запятую
    # v62: Строительная логика — трит из пары битов (paired bit quantization)
    # Вместо пороговой квантизации {-1,0,+1} — два бита с STE:
    # (1,1)→+1 (jisa/лето), (0,0)→-1 (jani/зима),
    # (0,1)→0↑ (весна), (1,0)→0↓ (осень)
    # Решает проблему STE zero-gradient trap: нет мёртвой зоны
    interlingua_use_paired_bit: bool = False  # строительная логика для тритов

    # v57: Абриале — событийно-управляемые изотропные N-местные связи (Пацкин)
    use_abriale: bool = False            # Абриале-слой (событийное управление)
    abriale_d_event: int = 64            # размерность пространства событий
    abriale_n_heads: int = 4             # число голов изотропного attention
    abriale_arity: int = 2               # арность связей (2=бинарные, 3=тернарные)
    abriale_n_rules: int = 64            # число правил в банке (64 = гексаграммы)
    abriale_n_hits: int = 4              # макс. число хитов на событие
    abriale_n_alternatives: int = 2      # число альтернатив (действий) на правило
    abriale_n_event_types: int = 8       # число типов событий (8 = триграммы)
    abriale_balance_weight: float = 0.01 # вес aux loss для балансировки правил

    # Обучение
    lr: float = 3e-4
    warmup_steps: int = 2000
    batch_size: int = 8
    grad_accum_steps: int = 4    # эффективный batch = 8 × 4 = 32
    total_steps: int = 50000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Логирование
    log_every: int = 100
    save_every: int = 2000
    val_every: int = 500
    use_wandb: bool = False
    use_tensorboard: bool = False
    project_name: str = "yijing-transformer"
    run_name: Optional[str] = None

    @classmethod
    def tiny(cls, vocab_size=2048, **kwargs):
        """Tiny preset: ~2M params, быстрые эксперименты."""
        return cls(vocab_size=vocab_size, d_model=128, n_layers=4, n_heads=4,
                   block_size=256, **kwargs)

    @classmethod
    def small(cls, vocab_size=2048, **kwargs):
        """Small preset: ~15M params, для обучения на CPU/одном GPU."""
        return cls(vocab_size=vocab_size, d_model=256, n_layers=6, n_heads=8,
                   block_size=512, **kwargs)

    @classmethod
    def medium(cls, vocab_size=32000, **kwargs):
        """Medium preset: ~85M params, для серьёзных экспериментов."""
        return cls(vocab_size=vocab_size, d_model=512, n_layers=12, n_heads=8,
                   block_size=1024, n_kv_heads=4, **kwargs)

    @classmethod
    def large(cls, vocab_size=32000, **kwargs):
        """Large preset: ~300M params, для полноценного обучения."""
        return cls(vocab_size=vocab_size, d_model=1024, n_layers=16, n_heads=16,
                   block_size=2048, n_kv_heads=4, **kwargs)

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self):
        h = int(self.d_model * self.ffn_multiplier)
        # SwiGLU использует 2/3 от стандартного hidden для той же параметрики
        if self.use_swiglu:
            h = int(h * 2 / 3)
            # Округление до кратного 8 для эффективности
            h = ((h + 7) // 8) * 8
        return h
