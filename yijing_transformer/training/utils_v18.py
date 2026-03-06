"""
v18 утилиты: WSD Scheduler, Attention Map Capture, BPE Dropout, LAMB, NEFTune.

WSD (Warmup-Stable-Decay): LR schedule с тремя фазами для стабильного обучения.
Ref: MiniCPM (Hu et al., 2024)

Attention Map Capture: hook-based захват attention паттернов для анализа.

BPE Dropout: стохастическая сегментация для регуляризации токенизации.
Ref: Provilkov et al., "BPE-Dropout" (2020)

LAMB Optimizer: Layer-wise Adaptive Moments для больших батчей.
Ref: You et al., "Large Batch Optimization for Deep Learning" (2020)

NEFTune: Noisy Embeddings Fine-Tuning — шум в embedding для регуляризации.
Ref: Jain et al., "NEFTune" (2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ==================== WSD LR Scheduler ====================

class WSDScheduler:
    """
    Warmup-Stable-Decay LR scheduler.

    Три фазы:
    1. Warmup: линейный рост от 0 до lr
    2. Stable: постоянный lr
    3. Decay: cosine decay до min_lr

    Args:
        optimizer: оптимизатор
        total_steps: общее число шагов
        warmup_steps: шагов для warmup
        decay_steps: шагов для decay (в конце)
        min_lr_ratio: отношение min_lr/lr
    """
    def __init__(self, optimizer, total_steps, warmup_steps, decay_steps,
                 min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.stable_steps = total_steps - warmup_steps - decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        """Выполняет один шаг scheduler."""
        self._step += 1
        multiplier = self._get_multiplier(self._step)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * multiplier

    def _get_multiplier(self, step):
        """Вычисляет множитель LR для данного шага."""
        if step <= self.warmup_steps:
            # Линейный warmup
            return step / max(1, self.warmup_steps)
        elif step <= self.warmup_steps + self.stable_steps:
            # Стабильная фаза
            return 1.0
        else:
            # Cosine decay
            decay_progress = (step - self.warmup_steps - self.stable_steps) / max(1, self.decay_steps)
            decay_progress = min(decay_progress, 1.0)
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * decay_progress)
            )

    def get_lr(self):
        """Текущие LR для всех param groups."""
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {'step': self._step, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state):
        self._step = state['step']
        self.base_lrs = state['base_lrs']


# ==================== Attention Map Capture ====================

class AttentionMapCapture:
    """
    Захватывает attention weights из каждого слоя модели.

    Использует forward hooks для перехвата attention паттернов.
    Полезно для визуализации и анализа attention.

    Использование:
        capture = AttentionMapCapture(model)
        capture.start()
        model(input_ids)
        maps = capture.get_maps()  # dict[layer_name] → (B, H, T, T)
        capture.stop()
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_maps = {}
        self._running = False

    def start(self):
        """Начинает захват attention."""
        self.stop()
        self._running = True
        self.attention_maps = {}

        for name, module in self.model.named_modules():
            # Ищем attention модули
            cls_name = module.__class__.__name__
            if 'Attention' in cls_name or 'attn' in name.split('.')[-1]:
                if hasattr(module, 'register_forward_hook'):
                    handle = module.register_forward_hook(
                        self._make_hook(name)
                    )
                    self.hooks.append(handle)

    def stop(self):
        """Останавливает захват."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._running = False

    def _make_hook(self, name):
        def hook(module, input, output):
            # output может быть tuple (out, attn_weights) или просто tensor
            if isinstance(output, tuple) and len(output) >= 2:
                attn = output[1]
                if isinstance(attn, torch.Tensor) and attn.dim() >= 3:
                    self.attention_maps[name] = attn.detach().cpu()
        return hook

    def get_maps(self):
        """Возвращает захваченные attention maps."""
        return dict(self.attention_maps)

    def get_entropy(self):
        """
        Вычисляет entropy каждого attention map.

        Returns:
            dict[str, float]: entropy для каждого слоя
        """
        entropies = {}
        for name, attn in self.attention_maps.items():
            # attn: (B, H, T, T) или (B, T, T)
            probs = attn.float()
            probs = probs.clamp(min=1e-10)
            ent = -(probs * probs.log()).sum(dim=-1).mean().item()
            entropies[name] = ent
        return entropies

    def get_sparsity(self, threshold=0.01):
        """
        Вычисляет sparsity (доля near-zero weights).

        Args:
            threshold: порог для "нулевого" веса

        Returns:
            dict[str, float]: sparsity для каждого слоя
        """
        sparsities = {}
        for name, attn in self.attention_maps.items():
            sparse = (attn.abs() < threshold).float().mean().item()
            sparsities[name] = sparse
        return sparsities

    def reset(self):
        """Очищает захваченные maps."""
        self.attention_maps = {}


# ==================== BPE Dropout ====================

class BPEDropout:
    """
    BPE-Dropout: стохастический BPE для регуляризации.

    При каждом encode с вероятностью p пропускает merge,
    создавая разные сегментации одного текста.

    Args:
        merges: list[(str, str)] — merge rules из BPE
        vocab: dict[str, int] — словарь token → id
        dropout: float — вероятность пропуска merge (0.0 = стандартный BPE)
    """
    def __init__(self, merges, vocab, dropout=0.1):
        self.merges = merges
        self.vocab = vocab
        self.dropout = dropout
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text, dropout=None):
        """
        Encode с BPE-dropout.

        Args:
            text: входной текст
            dropout: override dropout (None = self.dropout)

        Returns:
            list[int]: token ids
        """
        p = dropout if dropout is not None else self.dropout
        tokens = list(text)

        for left, right in self.merges:
            merged = left + right
            i = 0
            new_tokens = []
            while i < len(tokens):
                if (i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right):
                    # С вероятностью (1 - p) делаем merge
                    import random
                    if random.random() > p:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to ids
        ids = []
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            else:
                # Fallback to individual chars
                for c in t:
                    ids.append(self.vocab.get(c, 0))
        return ids

    def encode_deterministic(self, text):
        """Encode без dropout (обычный BPE)."""
        return self.encode(text, dropout=0.0)

    def decode(self, ids):
        """Decode token ids обратно в текст."""
        tokens = [self.id_to_token.get(i, '') for i in ids]
        return ''.join(tokens)


# ==================== LAMB Optimizer ====================

class LAMB(torch.optim.Optimizer):
    """
    LAMB (Layer-wise Adaptive Moments) optimizer.

    Адаптивный optimizer для больших батчей. Использует
    trust ratio для масштабирования Adam update per-layer.

    Args:
        params: параметры модели
        lr: learning rate
        betas: (beta1, beta2) для momentum/variance
        eps: epsilon для числовой стабильности
        weight_decay: L2 регуляризация
        exclude_from_layer_adaptation: list параметров без layer adaptation
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Один шаг LAMB."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                corrected_avg = exp_avg / bc1
                corrected_avg_sq = exp_avg_sq / bc2

                # Adam direction
                adam_update = corrected_avg / (corrected_avg_sq.sqrt() + group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    adam_update.add_(p.data, alpha=group['weight_decay'])

                # Trust ratio (LAMB specific)
                weight_norm = p.data.norm(2)
                update_norm = adam_update.norm(2)

                if weight_norm > 0 and update_norm > 0:
                    trust_ratio = weight_norm / update_norm
                else:
                    trust_ratio = 1.0

                p.data.add_(adam_update, alpha=-group['lr'] * trust_ratio)

        return loss


# ==================== NEFTune ====================

class NEFTune(nn.Module):
    """
    NEFTune: Noisy Embedding Fine-Tuning.

    Добавляет uniform noise к embedding во время training.
    Noise масштабируется по sqrt(seq_len * d_model) / alpha.

    Простой, но эффективный способ регуляризации при fine-tuning.

    Args:
        embedding: nn.Embedding layer
        noise_alpha: масштаб шума (5-15 типично)
    """
    def __init__(self, embedding, noise_alpha=5.0):
        super().__init__()
        self.embedding = embedding
        self.noise_alpha = noise_alpha

    def forward(self, input_ids):
        """
        Forward с noisy embeddings.

        Args:
            input_ids: (B, T) token ids

        Returns:
            embeddings: (B, T, D) с шумом при training
        """
        embeds = self.embedding(input_ids)

        if self.training and self.noise_alpha > 0:
            dims = torch.tensor(embeds.shape[1] * embeds.shape[2], dtype=embeds.dtype)
            mag = self.noise_alpha / torch.sqrt(dims)
            noise = torch.zeros_like(embeds).uniform_(-1, 1) * mag
            embeds = embeds + noise

        return embeds

    @staticmethod
    def apply_to_model(model, noise_alpha=5.0):
        """
        Оборачивает tok_emb модели в NEFTune.

        Args:
            model: YiJingGPT
            noise_alpha: масштаб шума

        Returns:
            NEFTune wrapper (также заменяет model.tok_emb)
        """
        neft = NEFTune(model.tok_emb, noise_alpha)
        model.tok_emb = neft
        return neft

    @staticmethod
    def remove_from_model(model, original_embedding):
        """Убирает NEFTune, возвращает оригинальный embedding."""
        model.tok_emb = original_embedding
