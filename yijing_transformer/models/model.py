"""
YiJing-Transformer: трансформер с геометрической регуляризацией
на основе гиперкубов {-1,+1}^n (64 гексаграмм, 256 октограмм).

v3: RoPE, SwiGLU, адаптивная температура, MoE на триграммах, FlashAttention,
    иерархическая/деформируемая квантизация, октограммы, гексаграммные attention паттерны.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    get_trigrams,
    FactoredYiJingQuantizer,
    HierarchicalQuantizer,
    DeformableQuantizer,
    BianGuaTransform,
    HexagramAttentionPattern,
    RotaryEmbedding,
    apply_rotary_emb,
    SwiGLU,
    TrigramMoE,
)


def build_quantizer(cfg):
    """Фабрика квантизаторов по конфигурации."""
    if cfg.quantizer_type == 'factored6':
        return FactoredYiJingQuantizer(
            temp=cfg.temp, adaptive_temp=cfg.adaptive_temp
        )
    elif cfg.quantizer_type == 'hierarchical':
        return HierarchicalQuantizer(
            total_dim=cfg.quant_total_dim,
            group_dim=cfg.quant_group_dim,
            temp=cfg.temp,
            adaptive_temp=cfg.adaptive_temp,
        )
    elif cfg.quantizer_type == 'octogram':
        return HierarchicalQuantizer(
            total_dim=8,
            group_dim=4,  # 2 тетраграммы: 16×16 = 256
            temp=cfg.temp,
            adaptive_temp=cfg.adaptive_temp,
        )
    elif cfg.quantizer_type == 'deformable':
        return DeformableQuantizer(
            total_dim=cfg.quant_total_dim,
            group_dim=cfg.quant_group_dim,
            temp=cfg.temp,
        )
    else:
        raise ValueError(f"Unknown quantizer_type: {cfg.quantizer_type}")


class YiJingAttention(nn.Module):
    """
    Multi-head attention с геометрическим bias из триграмм.

    Каждая из 8 голов получает направление одной из 8 триграмм.
    Bias — ранг-1 матрица (внешнее произведение проекций q и k на направление).
    Инициализация head_scales=0: модель начинает как стандартный трансформер.

    Поддерживает: RoPE, FlashAttention.
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rope = cfg.use_rope
        self.use_flash = cfg.use_flash_attn

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        # RoPE
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                self.head_dim, max_seq_len=cfg.block_size, base=cfg.rope_base
            )

        # 8 триграмм как направления для голов
        trigrams = get_trigrams()  # (8, 3)
        trigrams_norm = F.normalize(trigrams, p=2, dim=1)
        self.register_buffer('head_dirs', trigrams_norm[:cfg.n_heads])
        self.head_scales = nn.Parameter(torch.zeros(cfg.n_heads))

        if not self.use_flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size))
            )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, H, T, D)

        # Применяем RoPE
        if self.use_rope:
            cos, sin = self.rotary(T)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ SDPA (включает FlashAttention v2 на CUDA)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
            )

            # Геометрический bias (добавляем к результату, а не к scores)
            if self.head_dim >= 3:
                q3, k3 = q[..., :3], k[..., :3]
                q_proj = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)
                k_proj = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
                geo_correction = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
                # Causal mask
                causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                geo_correction = geo_correction.masked_fill(~causal, 0.0)
                geo_attn = F.softmax(
                    geo_correction.masked_fill(~causal, float('-inf')),
                    dim=-1
                )
                geo_out = geo_attn @ v
                scales = self.head_scales.view(1, -1, 1, 1)
                out = out + scales * (geo_out - out).detach()  # мягкая коррекция
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float('-inf')
            )

            # Геометрический bias из триграмм
            if self.head_dim >= 3:
                q3 = q[..., :3]
                k3 = k[..., :3]
                q_proj = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)
                k_proj = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
                geo_bias = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
                scores = scores + self.head_scales.view(1, -1, 1, 1) * geo_bias

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class YiJingTransformerLayer(nn.Module):
    """
    Слой YiJing-трансформера:

    1. Attention с триграммным bias и RoPE
    2. Факторизованная квантизация к гексаграммам (6D гиперкуб)
    3. 变卦 трансформация (опционально)
    4. FFN (SwiGLU или GELU) или MoE на триграммах
    """
    def __init__(self, cfg):
        super().__init__()
        # Attention
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = YiJingAttention(cfg)

        # Квантизация к вершинам гиперкуба (bottleneck)
        self.quant_dim = cfg.quant_total_dim
        self.ln_hex = nn.LayerNorm(cfg.d_model)
        self.to_qd = nn.Linear(cfg.d_model, self.quant_dim, bias=False)
        self.from_qd = nn.Linear(self.quant_dim, cfg.d_model, bias=False)
        self.quantizer = build_quantizer(cfg)
        self.hex_scale = nn.Parameter(torch.tensor(cfg.hex_strength))

        # 变卦 (трансформация гексаграмм)
        if cfg.use_bian_gua:
            self.bian_gua = BianGuaTransform(cfg.d_model)
        else:
            self.bian_gua = None

        # FFN или MoE
        self.use_moe = cfg.use_hex_moe
        self.ln_ffn = nn.LayerNorm(cfg.d_model)

        if cfg.use_hex_moe:
            self.ffn = TrigramMoE(
                d_model=cfg.d_model,
                n_experts=cfg.n_experts,
                top_k=cfg.moe_top_k,
                ffn_hidden=cfg.ffn_hidden,
                dropout=cfg.dropout,
            )
        elif cfg.use_swiglu:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.ffn_hidden),
                nn.GELU(),
                nn.Linear(cfg.ffn_hidden, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )

    def forward(self, x):
        # 1. Attention с триграммным bias
        x = x + self.attn(self.ln_attn(x))

        # 2. Квантизация к вершинам гиперкуба
        h = self.ln_hex(x)
        zq = self.to_qd(h)
        zq_out = self.quantizer(zq)
        if self.training:
            zq_out = zq_out * (1.0 + 0.001 * torch.randn_like(zq_out))
        x = x + self.hex_scale * self.from_qd(zq_out)

        # 3. 变卦 трансформация
        if self.bian_gua is not None:
            x = self.bian_gua(x)

        # 4. FFN или MoE
        h_ffn = self.ln_ffn(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(h_ffn)
            x = x + ffn_out
            # Сохраняем aux_loss для добавления к основному loss
            self._aux_loss = aux_loss
        else:
            x = x + self.ffn(h_ffn)
            self._aux_loss = 0.0

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

    def get_aux_loss(self):
        """Собирает aux loss со всех MoE слоёв."""
        total = 0.0
        for layer in self.layers:
            total = total + layer._aux_loss
        return total


class YiJingGPT(nn.Module):
    """Полная языковая модель с И-Цзин архитектурой."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Позиционные эмбеддинги: если RoPE — не нужны отдельные
        if not cfg.use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        else:
            self.pos_emb = None

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
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :t, :]

        hidden = self.core(x)
        logits = self.head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            # Добавляем MoE aux loss
            aux = self.core.get_aux_loss()
            if isinstance(aux, torch.Tensor):
                loss = loss + aux

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
        hex_keys = ['to_qd', 'from_qd', 'to_6d', 'from_6d', 'hex_scale',
                     'head_scales', 'head_dirs', 'bian_gua', 'quantizer',
                     'trigram_dirs', 'router_proj', 'hex_attn_pattern']
        for name, p in self.named_parameters():
            if any(k in name for k in hex_keys):
                hex_params += p.numel()
        return total, hex_params
