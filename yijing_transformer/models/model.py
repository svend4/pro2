"""
YiJing-Transformer: трансформер с геометрической регуляризацией
на основе гиперкубов {-1,+1}^n (64 гексаграмм, 256 октограмм).

v8: RoPE scaling (NTK/linear), distillation, FLOPS profiler,
    improved generate (rep penalty, top-p, stop tokens),
    ONNX/TorchScript export, save/load pretrained.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .geometry import (
    get_trigrams,
    FactoredYiJingQuantizer,
    HierarchicalQuantizer,
    DeformableQuantizer,
    GumbelQuantizer,
    E8Quantizer,
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
    elif cfg.quantizer_type == 'e8':
        return E8Quantizer(
            temp=cfg.temp,
            adaptive_temp=cfg.adaptive_temp,
        )
    elif cfg.quantizer_type == 'gumbel':
        return GumbelQuantizer(
            total_dim=cfg.quant_total_dim,
            group_dim=cfg.quant_group_dim,
            temp=cfg.temp,
            hard=cfg.use_gumbel,
            commitment_weight=cfg.commitment_weight,
        )
    else:
        raise ValueError(f"Unknown quantizer_type: {cfg.quantizer_type}")


class YiJingAttention(nn.Module):
    """
    Multi-head attention с геометрическим bias из триграмм.

    Поддерживает:
    - GQA (Grouped Query Attention): n_kv_heads < n_heads
    - Sliding window attention: ограничение контекста
    - RoPE, FlashAttention, KV-cache

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
        self.use_rope = cfg.use_rope
        self.use_flash = cfg.use_flash_attn
        self.sliding_window = cfg.sliding_window

        # GQA: число KV голов (None = MHA, т.е. n_kv_heads = n_heads)
        self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads
        assert cfg.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({cfg.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.n_rep = cfg.n_heads // self.n_kv_heads  # сколько раз повторять KV

        # Раздельные проекции для GQA
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        # RoPE (с поддержкой scaling для расширения контекста)
        if self.use_rope:
            self.rotary = RotaryEmbedding(
                self.head_dim, max_seq_len=cfg.block_size, base=cfg.rope_base,
                scaling=getattr(cfg, 'rope_scaling', None),
                scaling_factor=getattr(cfg, 'rope_scaling_factor', 1.0),
            )

        # 8 триграмм как направления для голов
        trigrams = get_trigrams()  # (8, 3)
        trigrams_norm = F.normalize(trigrams, p=2, dim=1)
        self.register_buffer('head_dirs', trigrams_norm[:cfg.n_heads])
        self.head_scales = nn.Parameter(torch.zeros(cfg.n_heads))

    def _repeat_kv(self, x):
        """Повторяем KV головы для GQA: (B, n_kv_heads, T, D) -> (B, n_heads, T, D)."""
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, self.n_rep, T, D).reshape(B, H * self.n_rep, T, D)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # Раздельные проекции Q, K, V (для GQA)
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Применяем RoPE (с учётом позиции при KV-cache)
        if self.use_rope:
            offset = 0
            if kv_cache is not None:
                offset = kv_cache[0].size(2)
            cos, sin = self.rotary(T + offset)
            q = apply_rotary_emb(q, cos[offset:offset+T], sin[offset:offset+T])
            k = apply_rotary_emb(k, cos[offset:offset+T], sin[offset:offset+T])

        # KV-cache: сохраняем/конкатенируем (до repeat для GQA — экономим память)
        new_kv_cache = (k, v)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # GQA: повторяем KV головы до n_heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        S = k.size(2)  # полная длина (с кэшем)

        if self.use_flash and hasattr(F, 'scaled_dot_product_attention') and kv_cache is None:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
            )

            # Геометрический bias
            if self.head_dim >= 3:
                q3, k3 = q[..., :3], k[..., :3]
                q_proj = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)
                k_proj = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
                geo_correction = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
                causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                geo_correction = geo_correction.masked_fill(~causal, 0.0)
                geo_attn = F.softmax(
                    geo_correction.masked_fill(~causal, float('-inf')),
                    dim=-1
                )
                geo_out = geo_attn @ v
                scales = self.head_scales.view(1, -1, 1, 1)
                out = out + scales * (geo_out - out).detach()
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale

            # Causal mask с учётом KV-cache
            causal = torch.tril(torch.ones(S, S, device=x.device))
            causal = causal[S-T:S, :S]

            # Sliding window: маскируем всё за пределами окна
            if self.sliding_window is not None:
                for i in range(T):
                    pos = S - T + i  # абсолютная позиция в последовательности
                    window_start = max(0, pos - self.sliding_window + 1)
                    causal[i, :window_start] = 0.0

            causal = causal.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal == 0, float('-inf'))

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
        return self.out(out), new_kv_cache


class YiJingTransformerLayer(nn.Module):
    """
    Слой YiJing-трансформера:

    1. Attention с триграммным bias и RoPE
    2. Факторизованная квантизация к гексаграммам (6D гиперкуб)
    3. 变卦 трансформация (опционально)
    4. FFN (SwiGLU или GELU) или MoE на триграммах
    """
    def __init__(self, cfg, layer_quant_dim=None):
        super().__init__()
        # Attention
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = YiJingAttention(cfg)

        # Квантизация к вершинам гиперкуба (bottleneck)
        # Multi-scale: layer_quant_dim overrides cfg.quant_total_dim
        quant_dim = layer_quant_dim if layer_quant_dim is not None else cfg.quant_total_dim
        self.quant_dim = quant_dim
        self.ln_hex = nn.LayerNorm(cfg.d_model)
        self.to_qd = nn.Linear(cfg.d_model, quant_dim, bias=False)
        self.from_qd = nn.Linear(quant_dim, cfg.d_model, bias=False)

        # Для multi-scale: создаём временный cfg с правильным quant_total_dim
        if layer_quant_dim is not None and layer_quant_dim != cfg.quant_total_dim:
            from dataclasses import replace
            layer_cfg = replace(cfg, quant_total_dim=quant_dim)
            self.quantizer = build_quantizer(layer_cfg)
        else:
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

    def forward(self, x, kv_cache=None):
        # 1. Attention с триграммным bias (+ KV-cache)
        attn_out, new_kv = self.attn(self.ln_attn(x), kv_cache=kv_cache)
        x = x + attn_out

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

        return x, new_kv


class YiJingTransformer(nn.Module):
    """Core transformer (без language model head)."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Multi-scale quantization: разная размерность по слоям
        if cfg.multi_scale_quant and cfg.quant_dim_schedule is not None:
            assert len(cfg.quant_dim_schedule) == cfg.n_layers, \
                f"quant_dim_schedule length {len(cfg.quant_dim_schedule)} != n_layers {cfg.n_layers}"
            self.layers = nn.ModuleList([
                YiJingTransformerLayer(cfg, layer_quant_dim=dim)
                for dim in cfg.quant_dim_schedule
            ])
        else:
            self.layers = nn.ModuleList(
                [YiJingTransformerLayer(cfg) for _ in range(cfg.n_layers)]
            )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, kv_cache=None):
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            if self.cfg.use_gradient_ckpt and self.training and kv_cache is None:
                # Gradient checkpointing (только при обучении без KV-cache)
                x, new_kv = grad_checkpoint(
                    layer, x, layer_cache, use_reentrant=False
                )
            else:
                x, new_kv = layer(x, kv_cache=layer_cache)
            new_kv_cache.append(new_kv)
        return self.final_norm(x), new_kv_cache

    def get_aux_loss(self):
        """Собирает aux loss со всех MoE слоёв + commitment loss."""
        total = 0.0
        for layer in self.layers:
            total = total + layer._aux_loss
            # Commitment loss от Gumbel квантизатора
            if hasattr(layer.quantizer, 'get_commitment_loss'):
                total = total + layer.quantizer.get_commitment_loss()
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

        # Weight tying: share weights between embedding and output head
        if cfg.weight_tying:
            self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None):
        b, t = idx.size()
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            offset = 0
            if kv_cache is not None and kv_cache[0] is not None:
                offset = kv_cache[0][0].size(2)
            x = x + self.pos_emb[:, offset:offset+t, :]

        hidden, new_kv_cache = self.core(x, kv_cache=kv_cache)
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

        return logits, loss, new_kv_cache

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=1.0, repetition_window=64, stop_tokens=None,
                 use_cache=True):
        """
        Авторегрессивная генерация с KV-cache.

        Args:
            idx: (B, T) начальная последовательность
            max_new_tokens: максимум новых токенов
            temperature: температура сэмплирования
            top_k: top-k фильтрация
            top_p: nucleus sampling (top-p)
            repetition_penalty: штраф за повторения (1.0 = нет штрафа)
            repetition_window: окно для repetition penalty
            stop_tokens: list[int] — стоп-токены (генерация прекращается)
            use_cache: использовать KV-cache
        """
        kv_cache = None
        for _ in range(max_new_tokens):
            if use_cache and kv_cache is not None:
                idx_input = idx[:, -1:]
            else:
                idx_input = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]

            logits, _, kv_cache = self(idx_input, kv_cache=kv_cache if use_cache else None)
            logits = logits[:, -1, :]  # (B, V)

            # Repetition penalty
            if repetition_penalty != 1.0:
                past = idx[:, -repetition_window:]
                for b in range(idx.size(0)):
                    seen = past[b].unique()
                    for token_id in seen:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            logits = logits / temperature

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus sampling)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Убираем токены с cumulative > top_p
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                for b in range(logits.size(0)):
                    indices_to_remove = sorted_indices[b][sorted_mask[b]]
                    logits[b, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop tokens
            if stop_tokens is not None:
                if idx_next.item() in stop_tokens:
                    break

            # Обрезаем кэш если превышаем block_size
            if use_cache and kv_cache is not None:
                cache_len = kv_cache[0][0].size(2)
                if cache_len >= self.cfg.block_size:
                    kv_cache = None
        return idx

    @torch.no_grad()
    def beam_search(self, idx, max_new_tokens, beam_width=4, temperature=1.0,
                    length_penalty=0.6):
        """
        Beam search decoding.

        Args:
            idx: (1, T) начальная последовательность (batch=1)
            max_new_tokens: максимум новых токенов
            beam_width: ширина пучка
            temperature: температура для logits
            length_penalty: штраф за длину (α в score/len^α)

        Returns:
            best_sequence: (1, T+generated) лучшая последовательность
        """
        assert idx.size(0) == 1, "Beam search supports batch_size=1"
        device = idx.device

        # Инициализация: beam_width копий
        beams = [(idx, 0.0)]  # (sequence, log_prob)

        def _score(seq, score):
            length = seq.size(1) - idx.size(1)
            return score / (length ** length_penalty) if length > 0 else score

        for step in range(max_new_tokens):
            candidates = []
            for seq, score in beams:
                seq_input = seq if seq.size(1) <= self.cfg.block_size else seq[:, -self.cfg.block_size:]
                logits, _, _ = self(seq_input)
                logits = logits[:, -1, :] / temperature
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                for i in range(beam_width):
                    token = top_indices[i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, token], dim=1)
                    new_score = score + top_log_probs[i].item()
                    candidates.append((new_seq, new_score))

            candidates.sort(key=lambda x: _score(x[0], x[1]), reverse=True)
            beams = candidates[:beam_width]

        best_seq, _ = max(beams, key=lambda x: _score(x[0], x[1]))
        return best_seq

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

    def save_pretrained(self, path):
        """Сохраняет модель + config в один файл."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'config': self.cfg,
            'model_state_dict': self.state_dict(),
        }, path)

    @classmethod
    def from_pretrained(cls, path, device='cpu'):
        """Загружает модель из файла."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model = cls(cfg).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    def estimate_flops(self, seq_len=None):
        """
        Оценка FLOPS для одного forward pass.

        Считает основные операции:
        - Attention: QKV projections + attention scores + output projection
        - FFN: 2-3 линейных слоя
        - Quantizer: projection + softmax
        """
        if seq_len is None:
            seq_len = self.cfg.block_size
        d = self.cfg.d_model
        h = self.cfg.n_heads
        hd = self.cfg.head_dim
        L = self.cfg.n_layers
        V = self.cfg.vocab_size
        T = seq_len
        ffn_h = self.cfg.ffn_hidden
        kv_h = self.cfg.n_kv_heads or h

        flops = 0

        # Embedding lookup (negligible) + per layer:
        for _ in range(L):
            # Q, K, V projections
            flops += 2 * T * d * (h * hd)          # Q
            flops += 2 * T * d * (kv_h * hd)       # K
            flops += 2 * T * d * (kv_h * hd)       # V

            # Attention scores: Q @ K^T
            flops += 2 * T * T * h * hd

            # Attention @ V
            flops += 2 * T * T * h * hd

            # Output projection
            flops += 2 * T * d * d

            # FFN
            if self.cfg.use_swiglu:
                flops += 2 * T * d * ffn_h * 3  # w1, w2, w3
            else:
                flops += 2 * T * d * ffn_h * 2  # up + down

            # Quantizer projection (small)
            qd = self.cfg.quant_total_dim
            flops += 2 * T * d * qd * 2  # to_qd + from_qd

        # LM head
        flops += 2 * T * d * V

        return flops

    def estimate_flops_str(self, seq_len=None):
        """Человекочитаемая строка с FLOPS."""
        flops = self.estimate_flops(seq_len)
        if flops >= 1e12:
            return f"{flops/1e12:.2f} TFLOPS"
        elif flops >= 1e9:
            return f"{flops/1e9:.2f} GFLOPS"
        elif flops >= 1e6:
            return f"{flops/1e6:.2f} MFLOPS"
        return f"{flops:.0f} FLOPS"

    @torch.no_grad()
    def quantization_analytics(self):
        """Анализ квантизации по всем слоям: расстояние до кодбука, использование кодовых слов."""
        analytics = {}
        x_probe = torch.randn(1, 16, self.cfg.d_model, device=next(self.parameters()).device)

        for i, layer in enumerate(self.core.layers):
            info = {
                'hex_scale': layer.hex_scale.item(),
                'quant_dim': layer.quant_dim,
            }

            # Пропускаем через проекцию и квантизатор
            h = layer.ln_hex(x_probe)
            zq = layer.to_qd(h)
            zq_out = layer.quantizer(zq)

            # Расстояние до кодбука (мера "жёсткости" квантизации)
            info['quant_error'] = (zq - zq_out).pow(2).mean().item()
            info['quant_snr'] = (zq_out.pow(2).mean() / (zq - zq_out).pow(2).mean().clamp(min=1e-8)).item()

            # Температура квантизатора
            if hasattr(layer.quantizer, 'current_temp'):
                temp = layer.quantizer.current_temp
                info['temp'] = temp.item() if isinstance(temp, torch.Tensor) else temp

            # GQA info
            info['n_kv_heads'] = layer.attn.n_kv_heads
            info['n_rep'] = layer.attn.n_rep

            # Commitment loss
            if hasattr(layer.quantizer, 'get_commitment_loss'):
                info['commitment_loss'] = layer.quantizer.get_commitment_loss().item()

            # BianGua stats
            if layer.bian_gua is not None:
                probs = torch.sigmoid(layer.bian_gua.change_logits)
                info['bian_gua_scale'] = layer.bian_gua.scale.item()
                info['active_lines'] = (probs > 0.5).sum().item()

            analytics[f'layer_{i}'] = info

        return analytics
