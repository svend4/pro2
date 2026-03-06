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

    # MoE на гексаграммах
    use_hex_moe: bool = False    # Mixture of Experts на 8 триграммах
    moe_top_k: int = 2           # сколько экспертов активировать
    n_experts: int = 8           # число экспертов (= число триграмм)

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
