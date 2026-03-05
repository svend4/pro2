"""
YiJing-Transformer: трансформер с геометрической регуляризацией
на основе 64 гексаграмм (вершины гиперкуба {-1,+1}⁶).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    get_trigrams,
    FactoredYiJingQuantizer,
    BianGuaTransform,
)


class YiJingAttention(nn.Module):
    """
    Multi-head attention с геометрическим bias из триграмм.

    Каждая из 8 голов получает направление одной из 8 триграмм.
    Bias — ранг-1 матрица (внешнее произведение проекций q и k на направление).
    Инициализация head_scales=0: модель начинает как стандартный трансформер.
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        # 8 триграмм как направления для голов
        trigrams = get_trigrams()  # (8, 3)
        trigrams_norm = F.normalize(trigrams, p=2, dim=1)
        self.register_buffer('head_dirs', trigrams_norm[:cfg.n_heads])
        self.head_scales = nn.Parameter(torch.zeros(cfg.n_heads))

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

        # Геометрический bias из триграмм (проекция первых 3 измерений)
        if self.head_dim >= 3:
            q3 = q[..., :3]
            k3 = k[..., :3]
            q_proj = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)
            k_proj = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
            geo_bias = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
            scores = scores + self.head_scales.view(1, -1, 1, 1) * geo_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class YiJingTransformerLayer(nn.Module):
    """
    Слой YiJing-трансформера:

    1. Attention с триграммным bias
    2. Факторизованная квантизация к гексаграммам (6D гиперкуб)
    3. 变卦 трансформация (опционально)
    4. FFN
    """
    def __init__(self, cfg):
        super().__init__()
        # Attention
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = YiJingAttention(cfg)

        # Гексаграммная квантизация (6D бутылочное горлышко)
        self.ln_hex = nn.LayerNorm(cfg.d_model)
        self.to_6d = nn.Linear(cfg.d_model, 6, bias=False)
        self.from_6d = nn.Linear(6, cfg.d_model, bias=False)
        self.quantizer = FactoredYiJingQuantizer(temp=cfg.temp)
        self.hex_scale = nn.Parameter(torch.tensor(cfg.hex_strength))

        # 变卦 (трансформация гексаграмм)
        if cfg.use_bian_gua:
            self.bian_gua = BianGuaTransform(cfg.d_model)
        else:
            self.bian_gua = None

        # FFN
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        # 1. Attention с триграммным bias
        x = x + self.attn(self.ln_attn(x))

        # 2. Квантизация к гексаграммам
        h = self.ln_hex(x)
        z6 = self.to_6d(h)
        z6_q = self.quantizer(z6)
        if self.training:
            z6_q = z6_q * (1.0 + 0.001 * torch.randn_like(z6_q))
        x = x + self.hex_scale * self.from_6d(z6_q)

        # 3. 变卦 трансформация
        if self.bian_gua is not None:
            x = self.bian_gua(x)

        # 4. FFN
        x = x + self.ffn(self.ln_ffn(x))
        return x


class YiJingTransformer(nn.Module):
    """Core transformer (без language model head)."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [YiJingTransformerLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class YiJingGPT(nn.Module):
    """Полная языковая модель с И-Цзин архитектурой."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.core = YiJingTransformer(cfg)
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
        hidden = self.core(x)
        logits = self.head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        hex_params = 0
        for name, p in self.named_parameters():
            if any(k in name for k in ['to_6d', 'from_6d', 'hex_scale',
                                        'head_scales', 'head_dirs',
                                        'bian_gua', 'quantizer']):
                hex_params += p.numel()
        return total, hex_params
