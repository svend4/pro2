"""
FFN модули: SwiGLU, TrigramMoE, GeometricFFN, MultiScaleHypercubeLayer.

Векторизованный dispatch для DomainMoE и TrigramMoE:
  Вместо двойного Python-цикла (top_k × n_experts с mask) используется
  батчевый gather по предвычисленным выходам всех экспертов.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import get_trigrams, generate_hypercube


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward: LLaMA-style."""
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def _vectorized_dispatch(x_flat: torch.Tensor,
                         W1: torch.Tensor, W2: torch.Tensor,
                         top_k_probs: torch.Tensor,
                         top_k_indices: torch.Tensor,
                         dropout: nn.Dropout) -> torch.Tensor:
    """Векторизованный MoE dispatch без Python-цикла по экспертам.

    Args:
        x_flat:        (N, D)
        W1:            (n_experts, H, D)  — вход FFN
        W2:            (n_experts, D, H)  — выход FFN
        top_k_probs:   (N, top_k)
        top_k_indices: (N, top_k)
        dropout:       Dropout модуль
    Returns:
        output: (N, D)
    """
    n_experts, H, D = W1.shape
    N = x_flat.shape[0]
    top_k = top_k_probs.shape[1]

    # Вычислить все n_experts выходов: (N, n_experts, D)
    # x_flat: (N, D) → (N, 1, D) → bmm с W1^T (n_experts, D, H) = (N, n_experts, H)
    h = torch.einsum('nd,ehd->neh', x_flat, W1)   # (N, n_experts, H)
    h = F.gelu(h)
    out_all = torch.einsum('neh,edh->ned', h, W2)  # (N, n_experts, D)

    # Выбрать top-k и взвесить: gather (N, top_k, D), затем dot-product
    idx = top_k_indices.unsqueeze(-1).expand(N, top_k, D)  # (N, top_k, D)
    topk_out = out_all.gather(1, idx)                       # (N, top_k, D)
    output = (dropout(topk_out) * top_k_probs.unsqueeze(-1)).sum(1)  # (N, D)
    return output


class DomainMoE(nn.Module):
    """
    Mixture of Experts с доменной специализацией.

    Каждый эксперт соответствует домену корпуса (ai_agents, infosystems, ...).
    Маршрутизация — по представлению (top-k softmax).
    Во время обучения добавляется domain_supervision_loss: эксперт домена D
    поощряется активироваться для токенов из домена D.

    Dispatch векторизован: все эксперты вычисляются параллельно через einsum,
    затем top-k выходы выбираются через gather и взвешиваются.
    """
    DOMAINS = ["ai_agents", "infosystems", "knowledge", "algorithms", "data2", "meta"]

    def __init__(self, d_model: int, n_experts: int = 6, top_k: int = 2,
                 ffn_hidden: int = None, dropout: float = 0.0,
                 domain_supervision_weight: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.domain_supervision_weight = domain_supervision_weight
        self.d_model = d_model

        if ffn_hidden is None:
            ffn_hidden = 4 * d_model
        self.ffn_hidden = ffn_hidden

        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Векторизованные банки: (n_experts, ffn_hidden, d_model)
        scale_in  = d_model ** -0.5
        scale_out = ffn_hidden ** -0.5
        self.W1 = nn.Parameter(torch.randn(n_experts, ffn_hidden, d_model) * scale_in)
        self.W2 = nn.Parameter(torch.randn(n_experts, d_model, ffn_hidden) * scale_out)
        self.drop = nn.Dropout(dropout)
        self.aux_loss_coeff = 0.01

    def forward(self, x, domain_ids=None):
        """
        Args:
            x          : (B, T, D)
            domain_ids : (B,) — индекс домена каждого сэмпла, или None
        Returns:
            output   : (B, T, D)
            aux_loss : скаляр
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)                              # (N, D)

        router_logits = self.router(x_flat)                  # (N, n_experts)
        router_probs  = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        output = _vectorized_dispatch(
            x_flat, self.W1, self.W2,
            top_k_probs, top_k_indices, self.drop,
        )

        # Балансировочный aux loss
        tokens_per_expert = router_probs.mean(dim=0)
        uniform   = torch.ones_like(tokens_per_expert) / self.n_experts
        aux_loss  = self.aux_loss_coeff * F.mse_loss(tokens_per_expert, uniform)

        # Domain supervision loss
        if domain_ids is not None and self.training and self.domain_supervision_weight > 0:
            domain_flat = domain_ids.unsqueeze(1).expand(B, T).reshape(-1)
            domain_flat = domain_flat.clamp(0, self.n_experts - 1)
            correct_expert_prob = router_probs[torch.arange(B * T, device=x.device), domain_flat]
            supervision_loss = -torch.log(correct_expert_prob + 1e-8).mean()
            aux_loss = aux_loss + self.domain_supervision_weight * supervision_loss

        return output.view(B, T, D), aux_loss


class TrigramMoE(nn.Module):
    """Mixture of Experts, где каждый эксперт соответствует триграмме.

    Dispatch векторизован через _vectorized_dispatch.
    """
    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        if ffn_hidden is None:
            ffn_hidden = 4 * d_model
        self.ffn_hidden = ffn_hidden

        trigrams = get_trigrams()[:n_experts]
        self.register_buffer('trigram_dirs', F.normalize(trigrams, p=2, dim=1))
        self.router_proj = nn.Linear(d_model, 3, bias=False)

        # Векторизованные банки: (n_experts, ffn_hidden, d_model)
        scale_in  = d_model ** -0.5
        scale_out = ffn_hidden ** -0.5
        self.W1 = nn.Parameter(torch.randn(n_experts, ffn_hidden, d_model) * scale_in)
        self.W2 = nn.Parameter(torch.randn(n_experts, d_model, ffn_hidden) * scale_out)
        self.drop = nn.Dropout(dropout)
        self.aux_loss_coeff = 0.01

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)

        z3           = self.router_proj(x_flat)
        router_logits = z3 @ self.trigram_dirs.T
        router_probs  = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        output = _vectorized_dispatch(
            x_flat, self.W1, self.W2,
            top_k_probs, top_k_indices, self.drop,
        )

        tokens_per_expert = router_probs.mean(dim=0)
        uniform   = torch.ones_like(tokens_per_expert) / self.n_experts
        aux_loss  = self.aux_loss_coeff * F.mse_loss(tokens_per_expert, uniform)
        return output.view(B, T, D), aux_loss


class GeometricFFN(nn.Module):
    """FFN через геометрическую маршрутизацию (TrigramMoE)."""
    def __init__(self, cfg):
        super().__init__()
        self.moe = TrigramMoE(
            d_model=cfg.d_model,
            n_experts=min(cfg.n_heads, 8),
            top_k=2,
            ffn_hidden=cfg.ffn_hidden,
            dropout=cfg.dropout,
        )

    def forward(self, x):
        return self.moe(x)


class MultiScaleHypercubeLayer(nn.Module):
    """Multi-scale hypercube — разные размерности гиперкуба по слоям."""
    def __init__(self, d_model: int, hypercube_dim: int = 3, temp: float = 0.3):
        super().__init__()
        self.dim = hypercube_dim
        vertices = generate_hypercube(hypercube_dim)
        self.register_buffer('vertices', vertices)
        self.proj_to = nn.Linear(d_model, hypercube_dim, bias=False)
        self.proj_from = nn.Linear(hypercube_dim, d_model, bias=False)
        self.temp = nn.Parameter(torch.tensor(temp).log())
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        z = self.proj_to(x)
        temp = F.softplus(self.temp) + 1e-4  # soft positive constraint, no hard bounds
        z_flat = z.reshape(-1, self.dim)
        dists = torch.cdist(z_flat.unsqueeze(0), self.vertices.unsqueeze(0)).squeeze(0)
        weights = F.softmax(-dists / temp, dim=-1)
        quantized = weights @ self.vertices
        quantized = quantized.reshape_as(z)
        return x + self.scale * self.proj_from(quantized)
