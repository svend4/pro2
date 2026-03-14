"""
FFN модули: SwiGLU, TrigramMoE, GeometricFFN, MultiScaleHypercubeLayer.
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


class DomainMoE(nn.Module):
    """
    Mixture of Experts с доменной специализацией.

    Каждый эксперт соответствует домену корпуса (ai_agents, infosystems, ...).
    Маршрутизация — по представлению (top-k softmax).
    Во время обучения добавляется domain_supervision_loss: эксперт домена D
    поощряется активироваться для токенов из домена D.
    """
    DOMAINS = ["ai_agents", "infosystems", "knowledge", "algorithms", "data2", "meta"]

    def __init__(self, d_model: int, n_experts: int = 6, top_k: int = 2,
                 ffn_hidden: int = None, dropout: float = 0.0,
                 domain_supervision_weight: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.domain_supervision_weight = domain_supervision_weight

        if ffn_hidden is None:
            ffn_hidden = 4 * d_model

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_hidden, bias=False),
                nn.GELU(),
                nn.Linear(ffn_hidden, d_model, bias=False),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])
        self.aux_loss_coeff = 0.01

    def forward(self, x, domain_ids=None):
        """
        Args:
            x          : (B, T, D)
            domain_ids : (B,) — индекс домена каждого сэмпла, или None
        Returns:
            output : (B, T, D)
            aux_loss : скаляр
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)

        router_logits = self.router(x_flat)          # (B*T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]
            for e in range(self.n_experts):
                mask = (expert_indices == e)
                if mask.any():
                    output[mask] += expert_weights[mask].unsqueeze(-1) * self.experts[e](x_flat[mask])

        # Балансировочный aux loss (эксперты загружены равномерно)
        tokens_per_expert = router_probs.mean(dim=0)
        uniform = torch.ones_like(tokens_per_expert) / self.n_experts
        aux_loss = self.aux_loss_coeff * F.mse_loss(tokens_per_expert, uniform)

        # Domain supervision loss: эксперт e должен активироваться для домена e
        if domain_ids is not None and self.training and self.domain_supervision_weight > 0:
            # domain_ids: (B,) -> expand to (B*T,)
            domain_flat = domain_ids.unsqueeze(1).expand(B, T).reshape(-1)  # (B*T,)
            domain_flat = domain_flat.clamp(0, self.n_experts - 1)
            # Вероятность «правильного» эксперта для каждого токена
            correct_expert_prob = router_probs[torch.arange(B * T, device=x.device), domain_flat]
            # Максимизируем её → минимизируем отрицательный лог
            supervision_loss = -torch.log(correct_expert_prob + 1e-8).mean()
            aux_loss = aux_loss + self.domain_supervision_weight * supervision_loss

        return output.view(B, T, D), aux_loss


class TrigramMoE(nn.Module):
    """Mixture of Experts, где каждый эксперт соответствует триграмме."""
    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2,
                 ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_model = d_model
        if ffn_hidden is None:
            ffn_hidden = 4 * d_model
        trigrams = get_trigrams()[:n_experts]
        self.register_buffer('trigram_dirs', F.normalize(trigrams, p=2, dim=1))
        self.router_proj = nn.Linear(d_model, 3, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_hidden, bias=False),
                nn.GELU(),
                nn.Linear(ffn_hidden, d_model, bias=False),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])
        self.aux_loss_coeff = 0.01

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        z3 = self.router_proj(x_flat)
        router_logits = z3 @ self.trigram_dirs.T
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]
            for e in range(self.n_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output
        tokens_per_expert = router_probs.mean(dim=0)
        uniform = torch.ones_like(tokens_per_expert) / self.n_experts
        aux_loss = self.aux_loss_coeff * F.mse_loss(tokens_per_expert, uniform)
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
        temp = self.temp.exp().clamp(min=0.01, max=5.0)
        z_flat = z.reshape(-1, self.dim)
        dists = torch.cdist(z_flat.unsqueeze(0), self.vertices.unsqueeze(0)).squeeze(0)
        weights = F.softmax(-dists / temp, dim=-1)
        quantized = weights @ self.vertices
        quantized = quantized.reshape_as(z)
        return x + self.scale * self.proj_from(quantized)
