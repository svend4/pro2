from dataclasses import dataclass


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

    # Обучение
    lr: float = 3e-4
    warmup_steps: int = 2000
    batch_size: int = 8
    grad_accum_steps: int = 4    # эффективный batch = 8 × 4 = 32
    total_steps: int = 50000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
