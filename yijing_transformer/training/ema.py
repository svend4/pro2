"""
EMA (Exponential Moving Average) и Early Stopping для YiJing-Transformer.

EMA поддерживает скользящее среднее весов модели для более стабильного инференса.
Early Stopping прекращает обучение при отсутствии улучшений.

Использование:
    ema = EMA(model, decay=0.999)
    for step in training:
        loss.backward(); optimizer.step()
        ema.update()
    # Инференс с EMA весами:
    with ema.average_parameters():
        model.eval(); model(x)
"""

import copy
from contextlib import contextmanager
import torch


class EMA:
    """
    Exponential Moving Average весов модели.

    shadow_param = decay * shadow_param + (1 - decay) * param

    При инференсе используются shadow параметры (более стабильные).
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow()

    def _init_shadow(self):
        """Инициализирует shadow параметры текущими значениями."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Обновляет shadow параметры."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        """Заменяет параметры модели на shadow (для инференса)."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Восстанавливает оригинальные параметры."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    @contextmanager
    def average_parameters(self):
        """Context manager для использования EMA параметров."""
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state):
        self.shadow = state['shadow']
        self.decay = state['decay']


class EarlyStopping:
    """
    Early stopping с patience.

    Отслеживает метрику (обычно val_loss) и сигнализирует
    о прекращении обучения, если нет улучшений за patience шагов.
    """
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        """
        Args:
            patience: число шагов без улучшения до остановки
            min_delta: минимальное улучшение, считающееся значимым
            mode: 'min' для loss (меньше = лучше), 'max' для accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metric):
        """
        Проверяет метрику. Возвращает True если нужно остановиться.

        Args:
            metric: текущее значение метрики

        Returns:
            True если нужно остановить обучение
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == 'min':
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False


def compute_activation_memory(model, batch_size, seq_len):
    """
    Оценивает память, необходимую для активаций (forward pass).

    Returns:
        dict с деталями по компонентам (в bytes)
    """
    cfg = model.cfg
    d = cfg.d_model
    h = cfg.n_heads
    hd = cfg.head_dim
    L = cfg.n_layers
    T = seq_len
    B = batch_size
    ffn_h = cfg.ffn_hidden
    kv_h = cfg.n_kv_heads or h

    bytes_per_param = 4  # float32

    mem = {}

    # Per layer
    per_layer = 0

    # Attention: Q, K, V, scores, attn weights, output
    per_layer += B * h * T * hd * bytes_per_param  # Q
    per_layer += B * kv_h * T * hd * bytes_per_param  # K
    per_layer += B * kv_h * T * hd * bytes_per_param  # V
    per_layer += B * h * T * T * bytes_per_param  # attention scores
    per_layer += B * h * T * T * bytes_per_param  # attention weights
    per_layer += B * T * d * bytes_per_param  # output

    # FFN
    if cfg.use_swiglu:
        per_layer += B * T * ffn_h * 2 * bytes_per_param  # w1 + gate
    else:
        per_layer += B * T * ffn_h * bytes_per_param
    per_layer += B * T * d * bytes_per_param  # output

    # Residual + LayerNorm
    per_layer += B * T * d * 3 * bytes_per_param  # residuals + norms

    mem['per_layer'] = per_layer
    mem['all_layers'] = per_layer * L

    # Embeddings
    mem['embeddings'] = B * T * d * bytes_per_param

    # Logits
    mem['logits'] = B * T * cfg.vocab_size * bytes_per_param

    mem['total'] = mem['all_layers'] + mem['embeddings'] + mem['logits']

    # Human readable
    total_mb = mem['total'] / (1024 ** 2)
    mem['total_mb'] = round(total_mb, 2)
    mem['total_gb'] = round(total_mb / 1024, 3)

    return mem
