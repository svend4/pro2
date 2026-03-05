"""
Vanilla Transformer baseline — для честного сравнения.
Та же архитектура, что YiJingGPT, но без геометрических компонентов.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class VanillaTransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = VanillaAttention(cfg)
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_attn(x))
        x = x + self.ffn(self.ln_ffn(x))
        return x


class VanillaGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.layers = nn.ModuleList(
            [VanillaTransformerLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()), 0
