"""
Vanilla Transformer baseline — для честного сравнения.
Та же архитектура, что YiJingGPT, но без геометрических компонентов.
Поддерживает RoPE и SwiGLU для честного A/B сравнения.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import RotaryEmbedding, apply_rotary_emb, SwiGLU


class VanillaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = cfg.use_rope

        # GQA support
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
        assert cfg.n_heads % self.n_kv_heads == 0
        self.n_rep = cfg.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        if self.use_rope:
            self.rotary = RotaryEmbedding(
                self.head_dim, max_seq_len=cfg.block_size, base=cfg.rope_base
            )

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size))
        )

    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, self.n_rep, T, D).reshape(B, H * self.n_rep, T, D)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rotary(T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

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

        if cfg.use_swiglu:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.ffn_hidden),
                nn.GELU(),
                nn.Linear(cfg.ffn_hidden, cfg.d_model),
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

        if not cfg.use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        else:
            self.pos_emb = None

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
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :t, :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()), 0
