"""
v13 утилиты: Ring Attention, selective checkpointing, WD scheduling, throughput.

Ring Attention: scaffold для распределённого attention на длинных контекстах.
Каждый GPU обрабатывает свой сегмент, KV передаются кольцом.

Selective Checkpointing: checkpoint только каждый N-й слой,
баланс между памятью и скоростью (вместо all-or-nothing).

Weight Decay Scheduling: линейное убывание weight decay к концу обучения.

Throughput Benchmark: измерение tokens/sec и Model FLOPs Utilization (MFU).
"""

import math
import time
import torch
import torch.nn as nn


# ==================== Ring Attention (scaffold) ====================

class RingAttentionConfig:
    """
    Конфигурация Ring Attention для распределённого длинного контекста.

    Ring Attention разбивает длинную последовательность на сегменты
    по числу GPU. Каждый GPU хранит свой сегмент Q и получает KV
    от соседей в кольцевом порядке.

    Ref: Liu et al., "Ring Attention with Blockwise Transformers" (2023)

    Args:
        world_size: число GPU
        segment_len: длина сегмента на одном GPU
        overlap_len: перекрытие между сегментами (для causal)
    """
    def __init__(self, world_size=1, segment_len=512, overlap_len=0):
        self.world_size = world_size
        self.segment_len = segment_len
        self.overlap_len = overlap_len
        self.total_len = world_size * segment_len

    def get_rank_range(self, rank):
        """Возвращает диапазон позиций для данного rank."""
        start = rank * self.segment_len
        end = start + self.segment_len
        return start, end

    def get_kv_source_ranks(self, rank, step):
        """
        Для данного rank и шага ring: откуда получаем KV.
        Step 0: свои KV, step 1: от rank-1, step 2: от rank-2, ...
        """
        source = (rank - step) % self.world_size
        return source


def simulate_ring_attention(seq_len, n_gpus, segment_len=None):
    """
    Симулирует Ring Attention на одном GPU (для тестирования).

    Разбивает последовательность на сегменты и вычисляет attention
    блок за блоком, имитируя ring communication.

    Args:
        seq_len: полная длина последовательности
        n_gpus: число (виртуальных) GPU
        segment_len: длина сегмента (None = seq_len / n_gpus)

    Returns:
        dict: статистика (segments, steps, total_attention_blocks)
    """
    if segment_len is None:
        segment_len = seq_len // n_gpus

    n_segments = (seq_len + segment_len - 1) // segment_len
    total_blocks = 0

    schedule = []
    for rank in range(min(n_gpus, n_segments)):
        for step in range(min(n_gpus, n_segments)):
            source = (rank - step) % n_segments
            # Causal: rank может видеть только source <= rank (в ring порядке)
            q_start = rank * segment_len
            kv_start = source * segment_len
            if kv_start <= q_start + segment_len:  # causal check
                total_blocks += 1
                schedule.append({
                    'rank': rank, 'step': step,
                    'q_range': (q_start, min(q_start + segment_len, seq_len)),
                    'kv_source': source,
                    'kv_range': (kv_start, min(kv_start + segment_len, seq_len)),
                })

    return {
        'n_segments': n_segments,
        'n_gpus': n_gpus,
        'total_blocks': total_blocks,
        'schedule': schedule,
        'total_seq_len': seq_len,
        'segment_len': segment_len,
        'memory_per_gpu_tokens': segment_len,
    }


# ==================== Selective Activation Checkpointing ====================

class SelectiveCheckpointing:
    """
    Selective activation checkpointing: checkpoint только каждый N-й слой.

    Вместо checkpoint всех слоёв (max memory saving, min speed)
    или ни одного (min memory saving, max speed), выбираем баланс.

    Правила:
    - checkpoint_every=1: все слои (= gradient_checkpointing)
    - checkpoint_every=2: каждый второй
    - checkpoint_every=0: ни один (отключено)

    Args:
        n_layers: число слоёв в модели
        checkpoint_every: checkpoint каждый N-й слой (0 = выкл)
    """
    def __init__(self, n_layers, checkpoint_every=2):
        self.n_layers = n_layers
        self.checkpoint_every = checkpoint_every
        self._checkpoint_layers = set()
        if checkpoint_every > 0:
            self._checkpoint_layers = {
                i for i in range(n_layers) if (i + 1) % checkpoint_every == 0
            }

    def should_checkpoint(self, layer_idx):
        """Проверяет, нужен ли checkpoint для данного слоя."""
        return layer_idx in self._checkpoint_layers

    def get_checkpoint_layers(self):
        """Возвращает множество слоёв с checkpoint."""
        return sorted(self._checkpoint_layers)

    def estimate_memory_saving(self):
        """
        Оценивает экономию памяти (доля от полного checkpointing).
        1.0 = все слои, 0.0 = ни одного.
        """
        if self.n_layers == 0:
            return 0.0
        return len(self._checkpoint_layers) / self.n_layers


# ==================== Weight Decay Scheduling ====================

class WeightDecayScheduler:
    """
    Планировщик weight decay: линейное убывание к концу обучения.

    WD(t) = wd_start * (1 - t/T) + wd_end * (t/T)

    Мотивация: в начале обучения сильная регуляризация помогает,
    к концу — ослабляем для точной подгонки.

    Args:
        optimizer: PyTorch optimizer
        wd_start: начальный weight decay
        wd_end: конечный weight decay
        total_steps: общее число шагов
    """
    def __init__(self, optimizer, wd_start=0.1, wd_end=0.01, total_steps=10000):
        self.optimizer = optimizer
        self.wd_start = wd_start
        self.wd_end = wd_end
        self.total_steps = total_steps
        self.step_count = 0

    def step(self):
        """Обновляет weight decay."""
        self.step_count += 1
        progress = min(self.step_count / max(1, self.total_steps), 1.0)
        wd = self.wd_start + (self.wd_end - self.wd_start) * progress

        for pg in self.optimizer.param_groups:
            if pg.get('weight_decay', 0) > 0:
                pg['weight_decay'] = wd

        return wd

    def get_wd(self):
        """Возвращает текущий weight decay."""
        progress = min(self.step_count / max(1, self.total_steps), 1.0)
        return self.wd_start + (self.wd_end - self.wd_start) * progress

    def state_dict(self):
        return {'step_count': self.step_count}

    def load_state_dict(self, state):
        self.step_count = state['step_count']


# ==================== Throughput Benchmark ====================

def estimate_model_flops(cfg, seq_len=None):
    """
    Оценивает FLOPS для одного forward pass модели.

    Основные компоненты:
    - Embedding lookup: O(B*T*D)
    - Attention QKV projection: O(3 * B*T*D²) per layer
    - Attention scores: O(B*H*T²*d_h) per layer
    - FFN: O(2 * B*T*D*FFN_H) per layer (или 3x для SwiGLU)
    - Output head: O(B*T*D*V)

    Returns:
        dict с деталями по компонентам
    """
    T = seq_len or cfg.block_size
    D = cfg.d_model
    L = cfg.n_layers
    H = cfg.n_heads
    d_h = D // H
    FFN_H = cfg.ffn_hidden
    V = cfg.vocab_size
    kv_h = cfg.n_kv_heads or H

    flops = {}

    # Per layer
    # QKV projections
    flops['qkv_proj'] = L * (2 * T * D * (H * d_h + 2 * kv_h * d_h))
    # Attention scores + weighted sum
    flops['attn_scores'] = L * (2 * T * T * H * d_h)
    flops['attn_output'] = L * (2 * T * H * d_h * D)
    # FFN
    ffn_mult = 3 if cfg.use_swiglu else 2
    flops['ffn'] = L * (ffn_mult * 2 * T * D * FFN_H)

    # Embedding + head
    flops['embedding'] = 2 * T * D  # lookup
    flops['head'] = 2 * T * D * V

    flops['total'] = sum(flops.values())
    flops['total_gflops'] = flops['total'] / 1e9

    return flops


def benchmark_throughput(model, cfg, batch_size=8, seq_len=None, n_steps=10,
                         device='cpu', use_amp=False):
    """
    Измеряет throughput модели.

    Args:
        model: YiJingGPT модель
        cfg: конфиг
        batch_size: размер batch
        seq_len: длина последовательности (None = block_size)
        n_steps: число шагов для измерения
        device: устройство
        use_amp: использовать AMP

    Returns:
        dict: tokens_per_sec, ms_per_step, mfu (если GPU)
    """
    T = seq_len or cfg.block_size
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(2):
        x = torch.randint(0, cfg.vocab_size, (batch_size, T), device=device)
        y = torch.randint(0, cfg.vocab_size, (batch_size, T), device=device)
        _, loss, _ = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    total_tokens = 0

    for _ in range(n_steps):
        x = torch.randint(0, cfg.vocab_size, (batch_size, T), device=device)
        y = torch.randint(0, cfg.vocab_size, (batch_size, T), device=device)

        if use_amp and device != 'cpu':
            with torch.cuda.amp.autocast():
                _, loss, _ = model(x, y)
        else:
            _, loss, _ = model(x, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += batch_size * T

    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    tokens_per_sec = total_tokens / elapsed
    ms_per_step = (elapsed / n_steps) * 1000

    # MFU estimate
    flops_info = estimate_model_flops(cfg, seq_len=T)
    flops_per_step = flops_info['total'] * batch_size * 3  # fwd + bwd ≈ 3x fwd

    result = {
        'tokens_per_sec': tokens_per_sec,
        'ms_per_step': round(ms_per_step, 2),
        'total_time_sec': round(elapsed, 2),
        'batch_size': batch_size,
        'seq_len': T,
        'n_steps': n_steps,
        'flops_per_step': flops_per_step,
    }

    # MFU (Model FLOPs Utilization) — только для GPU
    if device != 'cpu' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Грубая оценка peak FLOPS для распространённых GPU
        peak_flops_map = {
            'A100': 312e12, 'H100': 990e12, 'V100': 125e12,
            'A6000': 155e12, 'RTX 3090': 142e12, 'RTX 4090': 330e12,
        }
        peak = None
        for name, flops in peak_flops_map.items():
            if name in gpu_name:
                peak = flops
                break

        if peak:
            achieved_flops = flops_per_step / (elapsed / n_steps)
            mfu = achieved_flops / peak
            result['mfu'] = round(mfu, 4)
            result['gpu'] = gpu_name

    return result
