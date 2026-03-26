"""
Advanced Routing: GrMoE + SIPS + LatFormer soft masks.

Implements three key improvements from recent research:

1. GrMoE (Grassmannian MoE) — each expert is a subspace, not a vector.
   Routing via Matrix Bingham distribution on Grassmann manifold.
   Result: 0% expert collapse, 15-30% better load balancing.

2. SIPS (Saturated Inner-Product Scoring) — Lipschitz-controlled routing.
   Bounded sensitivity prevents routing instability at scale.

3. LatFormer soft masks — learnable geometric masks for attention.
   Encodes Q6 symmetry transformations as soft attention biases.
"""

import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_hexagrams() -> torch.Tensor:
    verts = list(itertools.product([-1.0, 1.0], repeat=6))
    return torch.tensor(verts, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════
# 1. GrMoE — Grassmannian Mixture-of-Experts Router
# ═══════════════════════════════════════════════════════════════════

class GrassmannianRouter(nn.Module):
    """Routes tokens to experts via subspace proximity on Grassmann manifold.

    Each expert is represented by a k-dimensional subspace in d-dimensional
    space (a point on Gr(k, d)). Token-expert affinity = ||U_e^T z||^2,
    where U_e is the orthonormal basis of expert e's subspace.

    The concentration parameter Lambda controls sparsity continuously:
    - Lambda → 0: uniform routing (all experts equally likely)
    - Lambda → inf: hard routing (single expert dominates)

    Args:
        d_model: input dimension
        n_experts: number of experts
        subspace_dim: dimension of each expert's subspace (k in Gr(k,d))
        top_k: number of experts to activate
        latent_dim: dimension of low-rank projection (L2R)
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        subspace_dim: int = 4,
        top_k: int = 2,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.subspace_dim = subspace_dim
        self.top_k = top_k
        self.latent_dim = latent_dim

        # Low-rank projection (L2R): d_model -> latent_dim
        self.proj = nn.Linear(d_model, latent_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

        # Expert subspaces: each is (latent_dim, subspace_dim)
        # Initialized as random orthonormal frames
        self.expert_bases = nn.Parameter(
            torch.zeros(n_experts, latent_dim, subspace_dim)
        )
        for i in range(n_experts):
            nn.init.orthogonal_(self.expert_bases.data[i])

        # Concentration parameter Lambda (log-domain for stability)
        self.log_lambda = nn.Parameter(torch.zeros(n_experts))

        # SIPS parameters for Lipschitz control
        self.sips_alpha = nn.Parameter(torch.ones(1))
        self.sips_beta = nn.Parameter(torch.ones(1) * 2.0)

        # Load balancing
        self.register_buffer('expert_load_ema', torch.zeros(n_experts))
        self.ema_decay = 0.99

    def _orthogonalize(self):
        """Re-orthogonalize expert bases via QR decomposition."""
        with torch.no_grad():
            for i in range(self.n_experts):
                Q, _ = torch.linalg.qr(self.expert_bases.data[i])
                self.expert_bases.data[i] = Q

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            weights: (B, T, n_experts) — routing weights (sum to 1)
            indices: (B, T, top_k) — selected expert indices
            aux_loss: scalar — load balancing loss
        """
        B, T, _ = x.shape

        # Project to low-rank latent space
        z = self.proj(x)  # (B, T, latent_dim)

        # Compute subspace affinity: ||U_e^T z||^2 for each expert
        # z: (B, T, latent_dim) -> (B, T, 1, latent_dim)
        z_exp = z.unsqueeze(2)  # (B, T, 1, latent_dim)

        # expert_bases: (n_experts, latent_dim, subspace_dim)
        # Projection: z @ U_e -> (B, T, n_experts, subspace_dim)
        projections = torch.einsum('btd,edk->btek', z, self.expert_bases)

        # Affinity = squared norm of projection
        affinity = (projections ** 2).sum(dim=-1)  # (B, T, n_experts)

        # Apply concentration (Lambda) — learned per expert
        lam = F.softplus(self.log_lambda)  # positive
        affinity = affinity * lam.unsqueeze(0).unsqueeze(0)

        # SIPS scoring: bounded sensitivity
        scores = self.sips_alpha * torch.tanh(affinity / self.sips_beta)

        # Top-k selection
        topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)

        # Softmax over selected experts only
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B, T, top_k)

        # Scatter to full weight tensor
        weights = torch.zeros(B, T, self.n_experts, device=x.device)
        weights.scatter_(2, topk_idx, topk_weights)

        # Load balancing loss
        with torch.no_grad():
            expert_frac = weights.sum(dim=(0, 1)) / (B * T)
            if self.training:
                self.expert_load_ema = (
                    self.ema_decay * self.expert_load_ema
                    + (1 - self.ema_decay) * expert_frac
                )

        # Auxiliary loss: encourage uniform expert usage
        uniform = torch.ones_like(expert_frac) / self.n_experts
        aux_loss = F.mse_loss(expert_frac, uniform) * self.n_experts

        return weights, topk_idx, aux_loss

    def get_diagnostics(self):
        """Return routing diagnostics."""
        lam = F.softplus(self.log_lambda).detach()
        return {
            'lambda_mean': lam.mean().item(),
            'lambda_std': lam.std().item(),
            'sips_alpha': self.sips_alpha.item(),
            'sips_beta': self.sips_beta.item(),
            'load_balance': self.expert_load_ema.detach().cpu().tolist(),
        }


# ═══════════════════════════════════════════════════════════════════
# 2. Q6 Grassmannian Router — combines GrMoE with hexagram geometry
# ═══════════════════════════════════════════════════════════════════

class Q6GrassmannianRouter(nn.Module):
    """GrMoE router with Q6 hexagram geometric priors.

    Each expert's subspace is initialized near its Q6 anchor hexagram.
    The geometric prior biases routing toward Q6-compatible assignments
    while still allowing learned deviation.

    Combines:
    - Grassmannian subspace routing (GrMoE)
    - Q6 Hamming distance prior
    - SIPS Lipschitz-controlled scoring (L2R)
    """

    # Domain anchors: Q6 vertex indices for 7 experts
    DOMAIN_ANCHORS = {
        'MATH':   63,  # 111111 — Qian/Heaven (pure abstract)
        'CODE':    6,  # 000110 — Wind/巽
        'HUMAN':   0,  # 000000 — Kun/Earth (pure concrete)
        'SYSTEM': 18,  # 010010 — infrastructure
        'RECON':  45,  # 101101 — pattern recognition
        'INFO':   27,  # 011011 — information flow
        'SYNTH':  36,  # 100100 — cross-domain synthesis
    }

    def __init__(
        self,
        d_model: int,
        n_experts: int = 7,
        subspace_dim: int = 4,
        top_k: int = 2,
        latent_dim: int = 32,
        hamming_weight: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.hamming_weight = hamming_weight

        # Core Grassmannian router
        self.grass_router = GrassmannianRouter(
            d_model=d_model,
            n_experts=n_experts,
            subspace_dim=subspace_dim,
            top_k=top_k,
            latent_dim=latent_dim,
        )

        # Q6 projection: d_model -> 6D soft hexagram
        self.q6_proj = nn.Linear(d_model, 6, bias=False)
        nn.init.normal_(self.q6_proj.weight, std=0.02)

        # Hexagram vertices (64, 6) and domain anchors (n_experts, 6)
        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)

        # Build anchor vectors from domain indices
        anchor_indices = list(self.DOMAIN_ANCHORS.values())[:n_experts]
        anchors = hexagrams[anchor_indices]  # (n_experts, 6)
        self.register_buffer('expert_anchors', anchors)

        # Learnable blend between Grassmannian and Hamming scores
        self.geo_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            weights: (B, T, n_experts)
            indices: (B, T, top_k)
            aux_loss: scalar
        """
        # Grassmannian routing
        grass_weights, indices, aux_loss = self.grass_router(x)

        # Q6 geometric routing
        soft_bits = torch.tanh(self.q6_proj(x))  # (B, T, 6)

        # Hamming affinity to expert anchors
        # affinity = <soft_bits, anchor> / 6 -> [-1, 1]
        hamming_affinity = torch.einsum(
            'btd,ed->bte', soft_bits, self.expert_anchors
        ) / 6.0  # (B, T, n_experts)

        # Blend Grassmannian and geometric scores
        alpha = torch.sigmoid(self.geo_gate)
        combined = (1 - alpha) * grass_weights + alpha * F.softmax(hamming_affinity * 3.0, dim=-1)

        # Re-select top-k from combined
        topk_vals, topk_idx = torch.topk(combined, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals * 5.0, dim=-1)

        weights = torch.zeros_like(combined)
        weights.scatter_(2, topk_idx, topk_weights)

        return weights, topk_idx, aux_loss

    def get_diagnostics(self):
        diag = self.grass_router.get_diagnostics()
        diag['geo_gate'] = torch.sigmoid(self.geo_gate).item()
        return diag


# ═══════════════════════════════════════════════════════════════════
# 3. LatFormer Soft Masks — geometric attention masks for Q6
# ═══════════════════════════════════════════════════════════════════

class Q6SoftAttentionMask(nn.Module):
    """Learnable soft attention mask based on Q6 hexagram geometry.

    Implements LatFormer-style soft masks that encode geometric
    transformations of the Q6 hypercube as attention biases.

    For each pair of tokens (i, j), computes:
        mask(i,j) = sum_t w_t * kernel(hex_i, T_t(hex_j))

    where T_t are Q6 transformations (bit flips, reflections)
    and w_t are learned transformation weights.

    This allows the model to attend based on geometric relationships
    (Hamming distance, antipodal pairs, bian gua transitions).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_transforms: int = 7,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_transforms = n_transforms

        # Project to soft Q6 coordinates
        self.q6_proj = nn.Linear(d_model, 6, bias=False)
        nn.init.normal_(self.q6_proj.weight, std=0.02)

        # Transformation matrices: each is a 6x6 matrix
        # Initialized as bit-flip operators (flip one bit each)
        self.transforms = nn.Parameter(torch.zeros(n_transforms, 6, 6))
        with torch.no_grad():
            # First 6: single bit flips (changing lines / bian yao)
            for i in range(min(6, n_transforms)):
                self.transforms.data[i] = torch.eye(6)
                self.transforms.data[i, i, i] = -1.0  # flip bit i
            # 7th: full antipodal (reverse all bits)
            if n_transforms > 6:
                self.transforms.data[6] = -torch.eye(6)

        # Per-head, per-transform weights
        self.transform_weights = nn.Parameter(
            torch.zeros(n_heads, n_transforms)
        )
        nn.init.normal_(self.transform_weights, std=0.01)

        # Temperature and scale
        self.temperature = temperature
        self.scale_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft geometric attention bias.

        Args:
            x: (B, T, d_model)

        Returns:
            mask: (B, n_heads, T, T) — additive attention bias
        """
        B, T, _ = x.shape

        # Project to Q6 coordinates
        hex_coords = torch.tanh(self.q6_proj(x))  # (B, T, 6)

        # Apply each transformation: hex_coords @ T_n^T for each transform n
        # hex_coords: (B, T, 6), transforms: (n_transforms, 6, 6)
        # Result: (B, T, n_transforms, 6)
        transformed = torch.einsum('btd,nde->btne', hex_coords, self.transforms)
        # transformed: (B, T, n_transforms, 6)

        # Kernel: inner product between original and transformed
        # For each transform t: K_t(i,j) = <hex_i, T_t(hex_j)> / 6
        # hex_coords: (B, T, 6), transformed: (B, T, n_transforms, 6)

        # Pairwise kernel for each transform:
        # (B, T_i, 6) @ (B, T_j, n_t, 6).T
        # = (B, T_i, 1, 6) @ (B, 1, T_j, n_t, 6).permute -> (B, T_i, T_j, n_t)
        kernels = torch.einsum(
            'bid,bjnd->bijn', hex_coords, transformed
        ) / 6.0  # (B, T, T, n_transforms)

        # Weight by per-head transform importance
        # transform_weights: (n_heads, n_transforms)
        w = torch.softmax(self.transform_weights / self.temperature, dim=-1)

        # Weighted sum: (B, T, T, n_transforms) @ (n_heads, n_transforms).T
        # -> (B, n_heads, T, T)
        mask = torch.einsum('bijt,ht->bhij', kernels, w)

        # Scale
        scale = torch.sigmoid(self.scale_logit) * 2.0  # [0, 2]
        mask = mask * scale

        return mask


# ═══════════════════════════════════════════════════════════════════
# 4. Enhanced BianGuaAttention — with GrMoE + LatFormer integration
# ═══════════════════════════════════════════════════════════════════

class EnhancedBianGuaAttention(nn.Module):
    """BianGuaAttention enhanced with LatFormer soft masks and SIPS scoring.

    Combines:
    - Original Hamming-distance geometric bias
    - LatFormer learnable transformation masks
    - SIPS-controlled scaling for stability

    Used as a drop-in replacement for BianGuaAttention in Variant3.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        block_size: int = 256,
        dropout: float = 0.05,
        hamming_lambda: float = 0.1,
        use_latformer: bool = True,
        n_transforms: int = 7,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_latformer = use_latformer

        # Standard attention projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

        # Hexagram projection for Hamming bias
        self.hex_proj = nn.Linear(d_model, 6, bias=False)
        nn.init.normal_(self.hex_proj.weight, std=0.02)
        self.hamming_lambda_logit = nn.Parameter(
            torch.tensor(math.log(hamming_lambda / (1 - hamming_lambda)))
        )

        # LatFormer soft mask
        if use_latformer:
            self.latformer_mask = Q6SoftAttentionMask(
                d_model=d_model,
                n_heads=n_heads,
                n_transforms=n_transforms,
            )

        # SIPS parameters for attention scaling
        self.sips_alpha = nn.Parameter(torch.ones(1))
        self.sips_beta = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # SIPS-controlled attention scores
        raw_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
        scores = self.sips_alpha * torch.tanh(raw_scores / self.sips_beta)

        # Hamming geometric bias
        soft_bits = torch.tanh(self.hex_proj(x))  # (B, T, 6)
        hamming_bias = torch.matmul(soft_bits, soft_bits.transpose(-2, -1)) / 2.0
        lam = torch.sigmoid(self.hamming_lambda_logit)
        scores = scores + lam * hamming_bias.unsqueeze(1)

        # LatFormer soft mask (additive)
        if self.use_latformer:
            lat_mask = self.latformer_mask(x)  # (B, n_heads, T, T)
            scores = scores + lat_mask

        # Causal masking
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )

        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Output
        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════
# 5. Expert FFN with Grassmannian routing
# ═══════════════════════════════════════════════════════════════════

class GrassmannianMoEFFN(nn.Module):
    """MoE FFN layer with Grassmannian routing.

    Each expert is a small FFN, selected via GrMoE subspace proximity.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 6,
        d_expert: int = 128,
        top_k: int = 2,
        subspace_dim: int = 4,
        dropout: float = 0.05,
        use_q6: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Router
        if use_q6:
            self.router = Q6GrassmannianRouter(
                d_model=d_model,
                n_experts=n_experts,
                subspace_dim=subspace_dim,
                top_k=top_k,
                latent_dim=min(32, d_model // 2),
            )
        else:
            self.router = GrassmannianRouter(
                d_model=d_model,
                n_experts=n_experts,
                subspace_dim=subspace_dim,
                top_k=top_k,
                latent_dim=min(32, d_model // 2),
            )

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_expert),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_expert, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

        # Output gate
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
            aux_loss: scalar (routing balance loss)
        """
        B, T, D = x.shape

        # Route
        weights, indices, aux_loss = self.router(x)  # weights: (B, T, n_experts)

        # Compute expert outputs (batch all experts)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )  # (B, T, n_experts, d_model)

        # Weighted combination
        combined = (expert_outputs * weights.unsqueeze(-1)).sum(dim=2)

        # Gated residual
        gate = torch.sigmoid(self.gate)
        output = x + gate * combined

        return output, aux_loss
