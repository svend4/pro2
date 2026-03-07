"""
Эквивариантные и структурные слои: D4, DualEmbedding, BianGua.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BianGuaTransform(nn.Module):
    """变卦 (Трансформация гексаграмм) — покоординатное отражение в {-1,+1}⁶."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.proj_from_6d = nn.Linear(6, d_model, bias=False)
        self.change_logits = nn.Parameter(torch.zeros(6))
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        z = self.proj_to_6d(x)
        change_prob = torch.sigmoid(self.change_logits)
        z_transformed = z * (1 - 2 * change_prob)
        return x + self.scale * self.proj_from_6d(z_transformed)


class GraduatedBianGuaTransform(nn.Module):
    """Градуированная 变卦: мутируют только «старые» линии (Склярова)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.proj_from_6d = nn.Linear(6, d_model, bias=False)
        self.change_logits = nn.Parameter(torch.zeros(6))
        self.stability_threshold = nn.Parameter(torch.tensor(0.7))
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        z = self.proj_to_6d(x)
        change_prob = torch.sigmoid(self.change_logits)
        extremeness = z.abs()
        threshold = torch.sigmoid(self.stability_threshold)
        mutation_mask = (extremeness > threshold).float()
        z_transformed = z * (1 - 2 * change_prob * mutation_mask)
        return x + self.scale * self.proj_from_6d(z_transformed)


class D4EquivariantLayer(nn.Module):
    """D₄-эквивариантный слой для триграмм (Фомюк)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        self.proj_from_3d = nn.Linear(3, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.01))
        ops = torch.zeros(8, 3, 3)
        ops[0] = torch.eye(3)
        ops[1] = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        ops[2] = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        ops[3] = -torch.eye(3)
        ops[4] = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
        ops[5] = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
        ops[6] = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
        ops[7] = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32)
        self.register_buffer('d4_ops', ops)
        self.op_weights = nn.Parameter(torch.zeros(8))

    def forward(self, x):
        z = self.proj_to_3d(x)
        w = F.softmax(self.op_weights, dim=0)
        z_flat = z.reshape(-1, 3)
        transformed = torch.zeros_like(z_flat)
        for i in range(8):
            transformed += w[i] * (z_flat @ self.d4_ops[i].T)
        transformed = transformed.reshape_as(z)
        return x + self.scale * self.proj_from_3d(transformed)


class DualEmbedding(nn.Module):
    """Dual embedding: 6D гиперкубное + 3D кубическое (Касаткин)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_to_6d = nn.Linear(d_model, 6, bias=False)
        self.proj_to_3d = nn.Linear(d_model, 3, bias=False)
        self.proj_6d_to_3d = nn.Linear(6, 3, bias=False)
        self.proj_3d_to_6d = nn.Linear(3, 6, bias=False)
        self.proj_from = nn.Linear(9, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        z6 = self.proj_to_6d(x)
        z3 = self.proj_to_3d(x)
        z3_from_6 = self.proj_6d_to_3d(z6)
        z3_aligned = (z3 + z3_from_6) / 2
        combined = torch.cat([z6, z3_aligned], dim=-1)
        return x + self.scale * self.proj_from(combined)

    def consistency_loss(self, x):
        z6 = self.proj_to_6d(x)
        z3 = self.proj_to_3d(x)
        z3_from_6 = self.proj_6d_to_3d(z6)
        return F.mse_loss(z3_from_6, z3)
