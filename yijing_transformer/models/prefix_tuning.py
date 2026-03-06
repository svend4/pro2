"""
Prefix Tuning, Logit Lens, Multi-Token Prediction.

Prefix Tuning: обучаемые prefix-токены, prepended к каждому слою.
Основная модель заморожена, обучаются только prefix параметры.
Ref: Li & Liang, "Prefix-Tuning" (2021)

Logit Lens: проецирует hidden states промежуточных слоёв через
output head для интерпретации — что «думает» модель на каждом слое.
Ref: nostalgebraist, "Logit Lens" (2020)

Multi-Token Prediction (MTP): предсказание N следующих токенов
одновременно через отдельные головы. Улучшает представления.
Ref: Gloeckle et al., "Better & Faster LLMs via Multi-token Prediction" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Prefix Tuning ====================

class PrefixTuning(nn.Module):
    """
    Prefix Tuning: обучаемые виртуальные токены перед каждым слоем.

    Вместо fine-tuning всей модели, обучаем только prefix:
    - prefix_key, prefix_value: (n_layers, prefix_len, n_heads, head_dim)
    - Добавляются к началу K, V в каждом attention слое.

    Args:
        cfg: YiJingConfig
        prefix_len: число виртуальных prefix-токенов
    """
    def __init__(self, cfg, prefix_len=16):
        super().__init__()
        self.prefix_len = prefix_len
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        n_kv_heads = cfg.n_kv_heads or cfg.n_heads

        # Reparameterization через MLP для стабильности
        self.prefix_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.Tanh(),
            nn.Linear(cfg.d_model * 2, cfg.n_layers * 2 * n_kv_heads * self.head_dim),
        )

        # Обучаемые prefix embeddings
        self.prefix_emb = nn.Parameter(
            torch.randn(prefix_len, cfg.d_model) * 0.01
        )

    def forward(self):
        """
        Генерирует prefix K, V для всех слоёв.

        Returns:
            list[(prefix_k, prefix_v)]: по одной паре на слой
                prefix_k: (1, n_kv_heads, prefix_len, head_dim)
                prefix_v: (1, n_kv_heads, prefix_len, head_dim)
        """
        # prefix_emb: (prefix_len, D)
        prefix = self.prefix_mlp(self.prefix_emb)  # (prefix_len, L*2*H_kv*hd)
        prefix = prefix.view(
            self.prefix_len, self.n_layers, 2, -1, self.head_dim
        )
        # → (prefix_len, L, 2, n_kv_heads, head_dim)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        # → (L, 2, n_kv_heads, prefix_len, head_dim)

        result = []
        for layer_idx in range(self.n_layers):
            pk = prefix[layer_idx, 0].unsqueeze(0)  # (1, n_kv_heads, prefix_len, hd)
            pv = prefix[layer_idx, 1].unsqueeze(0)
            result.append((pk, pv))

        return result

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def freeze_model_for_prefix(model):
    """Замораживает все параметры модели (для prefix tuning)."""
    for param in model.parameters():
        param.requires_grad = False
    return model


# ==================== Logit Lens ====================

class LogitLens:
    """
    Logit Lens: проецирует hidden states каждого слоя через output head.

    Показывает, какие токены «предсказывает» каждый промежуточный слой,
    позволяя понять, как информация трансформируется по глубине.

    Использование:
        lens = LogitLens(model)
        results = lens.analyze(input_ids)
        # results[layer_idx] = {top_tokens, probs, entropy}
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def get_layer_predictions(self, input_ids, position=-1):
        """
        Получает предсказания каждого слоя для данной позиции.

        Args:
            input_ids: (1, T) — входные токены
            position: позиция для анализа (-1 = последняя)

        Returns:
            list[dict]: по одному dict на слой + embedding
        """
        model = self.model
        x = model.tok_emb(input_ids)
        if model.pos_emb is not None:
            x = x + model.pos_emb[:, :input_ids.shape[1], :]

        results = []

        # Embedding layer
        logits = model.head(model.core.final_norm(x))
        results.append(self._analyze_logits(logits, position, 'embedding'))

        # Each transformer layer
        for i, layer in enumerate(model.core.layers):
            x, _ = layer(x)
            logits = model.head(model.core.final_norm(x))
            results.append(self._analyze_logits(logits, position, f'layer_{i}'))

        return results

    def _analyze_logits(self, logits, position, name):
        """Анализирует logits одного слоя."""
        pos_logits = logits[0, position, :]  # (V,)
        probs = F.softmax(pos_logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        top_vals, top_ids = probs.topk(5)

        return {
            'name': name,
            'top_token_ids': top_ids.tolist(),
            'top_probs': top_vals.tolist(),
            'entropy': entropy,
            'max_prob': top_vals[0].item(),
        }

    @torch.no_grad()
    def layer_similarity(self, input_ids):
        """
        Cosine similarity между представлениями соседних слоёв.

        Returns:
            list[float]: similarity[i] = cos_sim(layer_i, layer_{i+1})
        """
        model = self.model
        x = model.tok_emb(input_ids)
        if model.pos_emb is not None:
            x = x + model.pos_emb[:, :input_ids.shape[1], :]

        hidden_states = [x.clone()]
        for layer in model.core.layers:
            x, _ = layer(x)
            hidden_states.append(x.clone())

        similarities = []
        for i in range(len(hidden_states) - 1):
            h1 = hidden_states[i].flatten()
            h2 = hidden_states[i + 1].flatten()
            sim = F.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
            similarities.append(sim)

        return similarities


# ==================== Multi-Token Prediction ====================

class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction: N отдельных голов для предсказания N следующих токенов.

    Вместо предсказания только следующего токена, модель учится
    предсказывать k токенов вперёд одновременно. Каждый горизонт
    имеет свою голову (shared trunk).

    Loss = sum(CE(head_i(hidden), target_{t+i}) for i in 1..k)

    Args:
        d_model: размерность модели
        vocab_size: размер словаря
        n_future: число будущих токенов для предсказания
    """
    def __init__(self, d_model, vocab_size, n_future=4):
        super().__init__()
        self.n_future = n_future
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_future)
        ])

    def forward(self, hidden):
        """
        Args:
            hidden: (B, T, D) — hidden states из transformer

        Returns:
            list[Tensor]: logits для каждого горизонта, каждый (B, T, V)
        """
        return [head(hidden) for head in self.heads]

    def compute_loss(self, hidden, targets):
        """
        Вычисляет MTP loss.

        Args:
            hidden: (B, T, D)
            targets: (B, T) — полная целевая последовательность

        Returns:
            total_loss: усреднённый loss по всем горизонтам
            per_horizon_loss: list[float]
        """
        B, T, D = hidden.shape
        total_loss = 0.0
        per_horizon = []

        for i, head in enumerate(self.heads):
            logits = head(hidden)  # (B, T, V)

            # Target для горизонта i: targets[:, i+1:]
            # Logits: logits[:, :T-i-1]
            shift = i + 1
            if shift >= T:
                break

            shifted_logits = logits[:, :T - shift, :].reshape(-1, logits.size(-1))
            shifted_targets = targets[:, shift:].reshape(-1)

            loss_i = F.cross_entropy(shifted_logits, shifted_targets)
            total_loss = total_loss + loss_i
            per_horizon.append(loss_i.item())

        n = len(per_horizon)
        if n > 0:
            total_loss = total_loss / n

        return total_loss, per_horizon
