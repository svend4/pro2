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
