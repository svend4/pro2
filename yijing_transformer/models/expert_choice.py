"""
Интегрирован в YiJingGPT через use_expert_choice=True в YiJingConfig.
Также используется в knowledge_system.py.

Expert Choice MoE routing и Cross-Layer Parameter Sharing.

Expert Choice: вместо token→expert routing (top-k),
каждый эксперт выбирает свои top-C токенов.
Это обеспечивает идеальный load balancing без aux loss.

Ref: Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022)

Cross-Layer Parameter Sharing (ALBERT-style):
Один набор параметров переиспользуется для всех (или группы) слоёв.
Радикально сокращает число параметров при сохранении глубины.

Ref: Lan et al., "ALBERT" (2020)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing для MoE.

    Вместо top-k per token, каждый эксперт выбирает top-C токенов.
    C = capacity = (n_tokens * capacity_factor) / n_experts

    Преимущества:
    - Идеальный load balancing (каждый эксперт = ровно C токенов)
    - Нет aux loss
    - Токен может попасть к 0, 1 или нескольким экспертам

    Args:
        d_model: размерность модели
        n_experts: число экспертов
        capacity_factor: множитель ёмкости (1.0 = каждый токен → 1 эксперт в среднем)
        ffn_hidden: скрытая размерность FFN
        dropout: dropout rate
    """
    def __init__(self, d_model, n_experts=8, capacity_factor=1.0,
                 ffn_hidden=None, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        if ffn_hidden is None:
            ffn_hidden = 4 * d_model

        # Router
        self.gate = nn.Linear(d_model, n_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_hidden, bias=False),
                nn.GELU(),
                nn.Linear(ffn_hidden, d_model, bias=False),
                nn.Dropout(dropout),
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, T, D)

        Returns:
            output: (B, T, D)
            aux_info: dict с routing статистикой
        """
        B, T, D = x.shape

        # Router scores: (B, T, E) → transpose → (B, E, T)
        scores = self.gate(x)  # (B, T, E)
        scores_t = scores.transpose(1, 2)  # (B, E, T)

        # Softmax per expert (каждый эксперт оценивает все токены)
        expert_weights = F.softmax(scores_t, dim=-1)  # (B, E, T)

        # Capacity: сколько токенов каждый эксперт обрабатывает
        C = min(T, max(1, int(T * self.capacity_factor / self.n_experts)))

        # Каждый эксперт выбирает top-C токенов
        output = torch.zeros_like(x)
        tokens_processed = torch.zeros(B, T, device=x.device, dtype=x.dtype)

        for e in range(self.n_experts):
            # Top-C токены для этого эксперта
            weights_e = expert_weights[:, e, :]  # (B, T)
            topk_vals, topk_idx = weights_e.topk(C, dim=-1)  # (B, C)

            # Собираем токены
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, C)
            selected = x[batch_idx, topk_idx]  # (B, C, D)

            # Применяем эксперт
            expert_out = self.experts[e](selected)  # (B, C, D)

            # Взвешиваем выход
            weighted_out = expert_out * topk_vals.unsqueeze(-1)

            # Scatter обратно
            output.scatter_add_(
                1,
                topk_idx.unsqueeze(-1).expand(-1, -1, D),
                weighted_out,
            )
            tokens_processed.scatter_add_(
                1,
                topk_idx,
                topk_vals,
            )

        # Нормализуем по сумме весов экспертов, обработавших токен
        norm = tokens_processed.unsqueeze(-1).clamp(min=1e-8)
        output = output / norm

        aux_info = {
            'tokens_per_expert': C,
            'avg_experts_per_token': tokens_processed.mean().item(),
            'max_experts_per_token': tokens_processed.max().item(),
            'unused_tokens_frac': (tokens_processed == 0).float().mean().item(),
        }

        return output, aux_info


# ==================== Cross-Layer Parameter Sharing ====================

class SharedTransformerLayer(nn.Module):
    """
    Обёртка для cross-layer parameter sharing.

    Один набор параметров (shared_layer) переиспользуется N раз,
    как в ALBERT. Опционально: разные LayerNorm для каждого «виртуального» слоя.

    Args:
        shared_layer: слой для sharing (YiJingTransformerLayer)
        n_virtual_layers: число виртуальных слоёв
        share_ln: делить ли LayerNorm (False = свои нормы на каждый виртуальный слой)
        d_model: размерность (нужна если share_ln=False)
    """
    def __init__(self, shared_layer, n_virtual_layers, share_ln=False, d_model=None):
        super().__init__()
        self.shared_layer = shared_layer
        self.n_virtual_layers = n_virtual_layers
        self.share_ln = share_ln

        if not share_ln and d_model is not None:
            # Свои LayerNorm для каждого виртуального слоя
            self.layer_norms_attn = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_virtual_layers)
            ])
            self.layer_norms_ffn = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_virtual_layers)
            ])
        else:
            self.layer_norms_attn = None
            self.layer_norms_ffn = None

    def forward(self, x, virtual_layer_idx=0, kv_cache=None):
        """
        Forward через shared layer с (опционально) уникальным LayerNorm.

        Args:
            x: (B, T, D)
            virtual_layer_idx: индекс виртуального слоя (для выбора LayerNorm)
            kv_cache: KV-cache (tuple)

        Returns:
            x: (B, T, D)
            new_kv: KV-cache tuple
        """
        if self.layer_norms_attn is not None:
            # Подменяем LayerNorm в shared слое
            orig_ln_attn = self.shared_layer.ln_attn
            orig_ln_ffn = self.shared_layer.ln_ffn
            self.shared_layer.ln_attn = self.layer_norms_attn[virtual_layer_idx]
            self.shared_layer.ln_ffn = self.layer_norms_ffn[virtual_layer_idx]

        out, new_kv = self.shared_layer(x, kv_cache=kv_cache)

        if self.layer_norms_attn is not None:
            # Восстанавливаем
            self.shared_layer.ln_attn = orig_ln_attn
            self.shared_layer.ln_ffn = orig_ln_ffn

        return out, new_kv

    def parameter_savings(self, full_params_per_layer):
        """
        Оценивает экономию параметров от sharing.

        Returns:
            dict: savings info
        """
        shared = full_params_per_layer  # 1 слой
        full = full_params_per_layer * self.n_virtual_layers
        ln_overhead = 0
        if self.layer_norms_attn is not None:
            ln_overhead = sum(p.numel() for p in self.layer_norms_attn.parameters())
            ln_overhead += sum(p.numel() for p in self.layer_norms_ffn.parameters())

        actual = shared + ln_overhead
        return {
            'full_params': full,
            'shared_params': actual,
            'compression_ratio': full / max(actual, 1),
            'savings_pct': (1 - actual / max(full, 1)) * 100,
        }
