"""
NautilusYiJing — полная интеграция YiJingGPT и NautilusMoME.

Три варианта интеграции реализованы совместно:

  Вариант 1 — Q6GeometricRouter:
    Заменяет ExpertRouter. Вместо обученного линейного классификатора
    использует FactoredYiJingQuantizer для получения Q6-координаты токена,
    затем маршрутизирует по расстоянию Хэмминга до якорей экспертов.
    Роутинг детерминированный, интерпретируемый, без вспомогательного loss.

  Вариант 2 — YiJingCoreBlock:
    Заменяет TransformerBlock в core_first/core_second.
    Использует YiJingAttention (с геометрическим trigram-bias и RoPE) +
    FactoredYiJingQuantizer (Q6 на скрытых состояниях) +
    BianGuaTransform (опциональные координатные флипы) + SwiGLU FFN.

  Вариант 3 — YiJingMicroExpert:
    Заменяет MicroExpert (простой FFN-адаптер).
    Каждый эксперт — мини-YiJing-слой с собственным Q6-якорем (гексаграммой).
    Якорь задаёт геометрическое смещение: эксперт MATH "думает" иначе
    чем HUMAN на уровне гиперкуба Q6.

Архитектура:
  Input tokens
      ↓ tok_emb + pos_emb + SOLAN Q6 gate
  [YiJingCoreBlock × n_first]       ← Вариант 2: геометрическое ядро
      ↓
  [Q6GeometricRouter]               ← Вариант 1: геометрический роутинг
      ↓
  [YiJingMicroExpert × n_experts]   ← Вариант 3: геометрические эксперты
      ↓
  [NautilusBridge]
      ↓
  [CrossDomainAnalogy]
      ↓
  [ArchetypeLayer]
      ↓
  [SYNTH expert (entropy-triggered)]
      ↓
  [YiJingCoreBlock × n_second]      ← Вариант 2: геометрическая финализация
      ↓
  LN → head → logits

Использование:
    from models.nautilus_yijing import NautilusYiJing, NautilusYiJingConfig
    cfg = NautilusYiJingConfig()
    model = NautilusYiJing(cfg)
"""

import sys
import os
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Пути для импортов
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_ROOT))

from models.geometry.core import get_trigrams, get_hexagrams
from models.geometry.quantizers import FactoredYiJingQuantizer
from models.geometry.equivariant import BianGuaTransform


# ─── Якоря гексаграмм для экспертов (Q6 индексы) ──────────────────────────────
# Совпадают с HEXAGRAM_MAP из train_nautilus_mome.py
EXPERT_HEXAGRAM_ANCHORS = {
    'MATH':   63,   # 乾 Цянь — Небо, Творчество
    'CODE':    6,   # 巽 Сюнь — Ветер, Проникновение
    'HUMAN':   0,   # 坤 Кунь — Земля, Восприятие
    'SYSTEM': 18,   # 坎 Кань — Вода, Поток
    'RECON':  45,   # 離 Ли   — Огонь, Ясность
    'INFO':   27,   # 兌 Дуй  — Озеро, Обмен
    'SYNTH':  36,   # 革 Гэ   — Смена, Синтез
}

EXPERT_NAMES = ['MATH', 'CODE', 'HUMAN', 'SYSTEM', 'RECON', 'INFO', 'SYNTH']


def _q6_idx_to_vertex(idx: int) -> torch.Tensor:
    """Преобразует индекс гексаграммы (0..63) в вершину {-1,+1}^6."""
    bits = [(idx >> (5 - b)) & 1 for b in range(6)]
    return torch.tensor([2.0 * b - 1.0 for b in bits])


def hamming_distance_q6(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Расстояние Хэмминга в {-1,+1}^6: d_H = (6 - <a,b>) / 2.

    Args:
        a: (..., 6)
        b: (n, 6)
    Returns:
        (..., n)
    """
    dot = torch.einsum('...d,nd->...n', a, b)
    return (6.0 - dot) / 2.0


# ─── Конфигурация ──────────────────────────────────────────────────────────────

class NautilusYiJingConfig:
    """Конфигурация интегрированной модели NautilusYiJing."""

    def __init__(
        self,
        vocab_size: int = 128,       # char-level по умолчанию
        d_model: int = 192,
        n_layers: int = 4,           # суммарное число слоёв ядра
        n_heads: int = 6,
        block_size: int = 256,
        d_expert: int = 96,          # внутренняя размерность YiJingMicroExpert
        n_experts: int = 6,          # число активных экспертов (без SYNTH)
        top_k: int = 2,              # top-k роутинг
        dropout: float = 0.05,
        # YiJing-специфичные опции
        use_rope: bool = True,
        quant_temp: float = 0.3,
        use_bian_gua: bool = True,
        # SYNTH expert
        enable_synth: bool = True,
        # SOLAN Q6 таблица для токенов (опционально)
        solan_table: Optional[torch.Tensor] = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.d_expert = d_expert
        self.n_experts = n_experts
        self.top_k = top_k
        self.dropout = dropout
        self.use_rope = use_rope
        self.quant_temp = quant_temp
        self.use_bian_gua = use_bian_gua
        self.enable_synth = enable_synth
        self.solan_table = solan_table

        # Для совместимости с YiJingAttention / YiJingTransformerLayer
        self.bias = False
        self.use_flash_attn = True
        self.sliding_window = None
        self.attention_sinks = 0
        self.n_kv_heads = None
        self.use_alibi = False
        self.rope_base = 10000
        self.rope_scaling = None
        self.rope_scaling_factor = 1.0
        self.quantizer_type = 'factored6'
        self.quant_total_dim = 6
        self.use_swiglu = True
        self.ffn_mult = 4
        self.use_rope = use_rope
        # Отключаем тяжёлые v51+ модули для чистоты
        self.use_quadrant_attention = False
        self.use_recursive_cube = False
        self.use_weaving_loom = False
        self.use_triangular_bias = False
        self.use_palace_attention = False
        self.use_mobius_bias = False
        self.use_privileged_axis = False
        self.use_cube_diagonal = False
        self.use_cubic_bias = False
        self.use_heisenberg_attention = False
        self.use_hex_attn_pattern = False
        self.use_flower_gat = False
        self.use_structural_defect = False
        self.use_four_level_pe = False
        self.use_cubic_pe = False
        self.use_convergence_bridge = False
        self.use_glyph_tokenizer = False
        self.use_matrix_grammar = False
        self.use_abriale = False
        self.use_nautilus = False
        self.use_bidirectional_tri = False
        self.token_merge_ratio = 0.0
        self.label_smoothing = 0.0


# ─── Вариант 1: Q6GeometricRouter ─────────────────────────────────────────────

class Q6GeometricRouter(nn.Module):
    """Вариант 1 — геометрический роутер через Q6-квантизатор.

    Алгоритм:
      1. Проецируем скрытый вектор x в 6D: proj(x) → z ∈ R^6
      2. FactoredYiJingQuantizer: z → q ∈ {-1,+1}^6 (soft)
      3. Считаем расстояние Хэмминга от q до якоря каждого эксперта
      4. Маршрутизируем в top-k ближайших экспертов (наименьшее d_H)

    В отличие от ExpertRouter нет вспомогательного loss —
    геометрия сама обеспечивает разнообразие маршрутизации.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 quant_temp: float = 0.3, expert_names: Optional[List[str]] = None):
        super().__init__()
        if expert_names is None and n_experts > len(EXPERT_NAMES):
            raise ValueError(
                f"n_experts={n_experts} exceeds available EXPERT_NAMES ({len(EXPERT_NAMES)})")
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.expert_names = expert_names or EXPERT_NAMES[:n_experts]

        # Проекция d_model → 6 (Q6-пространство)
        self.proj = nn.Linear(d_model, 6, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        # Q6-квантизатор (soft, факторизованный по триграммам)
        self.quantizer = FactoredYiJingQuantizer(temp=quant_temp, adaptive_temp=True)

        # Якоря экспертов в Q6 — буфер, не обучаемый
        anchors = torch.stack([
            _q6_idx_to_vertex(EXPERT_HEXAGRAM_ANCHORS.get(name, i * 9 % 64))
            for i, name in enumerate(self.expert_names)
        ])  # (n_experts, 6)
        self.register_buffer('expert_anchors', anchors)

        # Температура softmax для весов роутинга (обучаемая)
        self.routing_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            expert_weights: (B, T, n_experts) — sparse top-k
            q6_coords: (B, T, 6) — Q6-координаты для мониторинга
        """
        B, T, D = x.shape

        # Проецируем в Q6-пространство
        z = self.proj(x)  # (B, T, 6)
        q = self.quantizer(z)  # (B, T, 6) — soft Q6

        # Расстояние Хэмминга до якорей экспертов
        # d_H(q, anchor) = (6 - <q, anchor>) / 2 ∈ [0, 6]
        dot = torch.einsum('btd,ed->bte', q, self.expert_anchors)  # (B, T, n_experts)
        hamming = (6.0 - dot) / 2.0  # (B, T, n_experts)

        # Превращаем расстояние в веса (ближе = больший вес)
        # Используем negative distance как logits
        logits = -hamming / self.routing_temp.clamp(min=0.1)

        # Top-k отбор
        top_k_logits, top_k_idx = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        expert_weights = torch.zeros(B, T, self.n_experts, device=x.device, dtype=x.dtype)
        expert_weights.scatter_(-1, top_k_idx, top_k_weights)

        return expert_weights, q


# ─── Вариант 2: YiJingCoreBlock ───────────────────────────────────────────────

class YiJingCoreBlock(nn.Module):
    """Вариант 2 — блок ядра с полной YiJing-геометрией.

    Заменяет простой TransformerBlock. Включает:
    - YiJingAttention: multi-head с trigram-bias и RoPE
    - FactoredYiJingQuantizer: Q6-квантизация скрытых состояний
    - BianGuaTransform: опциональные обученные флипы линий
    - SwiGLU FFN
    """

    def __init__(self, cfg: NautilusYiJingConfig):
        super().__init__()
        d = cfg.d_model

        self.ln_attn = nn.LayerNorm(d)
        self.ln_ffn = nn.LayerNorm(d)
        self.ln_quant = nn.LayerNorm(d)

        # YiJing Attention — с геометрическим trigram-bias
        self.attn = _YiJingCoreAttention(cfg)

        # Q6 квантизатор (проецируем d_model → 6, квантизуем, проецируем обратно)
        self.quant_proj_in = nn.Linear(d, 6, bias=False)
        self.quantizer = FactoredYiJingQuantizer(temp=cfg.quant_temp, adaptive_temp=True)
        self.quant_proj_out = nn.Linear(6, d, bias=False)
        self.hex_scale = nn.Parameter(torch.zeros(1))  # начинаем с нуля

        # BianGua (опционально)
        self.use_bian_gua = cfg.use_bian_gua
        if self.use_bian_gua:
            self.bian_gua = BianGuaTransform(d)

        # SwiGLU FFN
        d_ff = int(d * cfg.ffn_mult * 2 / 3)
        d_ff = (d_ff // 64) * 64 or 64
        self.ffn_gate = nn.Linear(d, d_ff, bias=False)
        self.ffn_up = nn.Linear(d, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention
        h = self.ln_attn(x)
        h = self.attn(h)
        x = x + h

        # Q6-квантизация (геометрический сигнал, масштабируемый)
        z = self.ln_quant(x)
        z6 = self.quant_proj_in(z)        # (B, T, 6)
        z6q = self.quantizer(z6)          # soft-квантизованный
        geo = self.quant_proj_out(z6q)    # обратно в d_model
        x = x + torch.tanh(self.hex_scale) * geo

        # BianGua
        if self.use_bian_gua:
            x = self.bian_gua(x)

        # SwiGLU FFN
        h = self.ln_ffn(x)
        gate = F.silu(self.ffn_gate(h))
        up = self.ffn_up(h)
        x = x + self.drop(self.ffn_down(gate * up))

        return x


class _YiJingCoreAttention(nn.Module):
    """Упрощённая YiJing-геометрическая attention для YiJingCoreBlock.

    Multi-head attention с:
    - RoPE позиционными эмбеддингами
    - Trigram-направленными геометрическими bias по головам
    """

    def __init__(self, cfg: NautilusYiJingConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        d = cfg.d_model

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        # RoPE
        self.use_rope = cfg.use_rope
        if self.use_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer('inv_freq', inv_freq)

        # Trigram-направления для голов (8 триграмм → первые n_heads)
        trigrams = get_trigrams()          # (8, 3)
        trigrams_norm = F.normalize(trigrams, p=2, dim=1)
        self.register_buffer('head_dirs', trigrams_norm[:cfg.n_heads])  # (n_heads, 3)
        self.head_scales = nn.Parameter(torch.zeros(cfg.n_heads))

    def _rope(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """Применяет RoPE к тензору (B, H, T, D)."""
        pos = torch.arange(T, device=x.device).float()
        freqs = torch.outer(pos, self.inv_freq)      # (T, D/2)
        cos = freqs.cos()[None, None]                 # (1,1,T,D/2)
        sin = freqs.sin()[None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        if self.use_rope:
            q = self._rope(q, T)
            k = self._rope(k, T)

        # Стандартное scaled dot-product attention (с causal mask)
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 dropout_p=self.drop.p if self.training else 0.0)
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            out = F.softmax(scores, dim=-1) @ v

        # Trigram геометрический bias (rank-1, поверх стандартного attention)
        if self.head_dim >= 3 and self.head_dirs.shape[0] > 0:
            q3 = q[..., :3]
            k3 = k[..., :3]
            q_proj = torch.einsum('bhtd,hd->bht', q3, self.head_dirs)  # (B,H,T)
            k_proj = torch.einsum('bhtd,hd->bht', k3, self.head_dirs)
            geo_bias = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)     # (B,H,T,T)
            causal = torch.tril(torch.ones(T, T, device=x.device)).bool()
            geo_bias = geo_bias.masked_fill(~causal, 0.0)
            geo_attn = F.softmax(geo_bias.masked_fill(~causal, float('-inf')), dim=-1)
            geo_out = geo_attn @ v
            scales = self.head_scales.view(1, -1, 1, 1)
            out = out + scales * (geo_out - out.detach())

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ─── Вариант 3: YiJingMicroExpert ─────────────────────────────────────────────

class YiJingMicroExpert(nn.Module):
    """Вариант 3 — мини-YiJing-слой как domain-специфичный эксперт.

    В отличие от простого FFN (MicroExpert), здесь:
    - Attention с trigram-bias (геометрическое восприятие контекста)
    - Q6-квантизатор с якорем на конкретной гексаграмме эксперта
    - Q6-якорь смещает квантизатор: эксперт "видит" пространство
      через призму своей гексаграммы

    MATH (乾, q6=63): все 6 бит yang (+1,+1,+1,+1,+1,+1) — максимальная структура
    HUMAN (坤, q6=0): все 6 бит yin (-1,-1,-1,-1,-1,-1) — максимальная рецептивность
    CODE (巽, q6=6): смешанный, "мягкое проникновение"
    и т.д.
    """

    def __init__(self, d_model: int, d_expert: int, dropout: float,
                 q6_anchor: torch.Tensor, expert_name: str = ''):
        super().__init__()
        self.expert_name = expert_name
        self.d_model = d_model
        self.d_expert = d_expert

        # Регистрируем Q6-якорь как буфер (геометрическая константа)
        self.register_buffer('q6_anchor', q6_anchor.float())  # (6,)

        # Проекция в Q6-пространство и обратно
        self.proj_in = nn.Linear(d_model, 6, bias=False)
        self.quantizer = FactoredYiJingQuantizer(temp=0.3, adaptive_temp=True)
        self.proj_out = nn.Linear(6, d_model, bias=False)

        # Адаптер: d_model → d_expert → d_model (domain-specific)
        self.ln = nn.LayerNorm(d_model)
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_expert),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_expert, d_expert),
            nn.GELU(),
            nn.Linear(d_expert, d_model),
            nn.Dropout(dropout),
        )

        # Q6-anchor bias: смещаем квантизатор к якорю эксперта
        # Инициализируем малым значением — якорь мягко направляет
        self.anchor_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

        # Общий масштаб вклада эксперта
        self.gate_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            delta: (B, T, d_model) — вклад эксперта
        """
        h = self.ln(x)

        # Q6-геометрический сигнал с якорём
        z6 = self.proj_in(h)              # (B, T, 6)

        # Смещаем к якорю: q6_anchor {-1,+1}^6 притягивает
        anchor_bias = torch.sigmoid(self.anchor_gate) * self.q6_anchor  # (6,)
        z6_biased = z6 + anchor_bias      # (B, T, 6)

        q6 = self.quantizer(z6_biased)    # soft Q6 с якорным смещением
        geo = self.proj_out(q6)           # (B, T, d_model)

        # FFN адаптер
        adapter_out = self.adapter(h + geo)

        return adapter_out * self.gate_scale


# ─── Вспомогательные модули (упрощённые из train_nautilus_mome.py) ─────────────

class SimpleBridge(nn.Module):
    """Упрощённый NautilusBridge: взвешенное слияние expert outputs."""

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        n_pairs = (n_experts + 1) // 2
        self.pair_merge = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU())
            for _ in range(n_pairs)
        ])
        self.global_merge = nn.Sequential(
            nn.Linear(d_model * n_pairs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.1))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, core_h: torch.Tensor,
                expert_outputs: List[torch.Tensor],
                expert_weights: torch.Tensor) -> torch.Tensor:
        B, T, D = core_h.shape
        weighted = [expert_outputs[i] * expert_weights[:, :, i:i+1]
                    for i in range(len(expert_outputs))]

        pair_outputs = []
        for i in range(0, len(weighted), 2):
            a = weighted[i]
            b = weighted[i + 1] if i + 1 < len(weighted) else torch.zeros_like(a)
            pair_outputs.append(self.pair_merge[i // 2](torch.cat([a, b], dim=-1)))

        global_in = torch.cat(pair_outputs, dim=-1)
        expected = self.global_merge[0].in_features
        if global_in.size(-1) < expected:
            pad = torch.zeros(B, T, expected - global_in.size(-1), device=global_in.device)
            global_in = torch.cat([global_in, pad], dim=-1)

        merged = self.global_merge(global_in)
        return core_h + self.residual_gate * self.ln(merged)


class SimpleAnalogy(nn.Module):
    """Упрощённый CrossDomainAnalogy: pairwise Q6-аналогии между экспертами."""

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.n_experts = n_experts
        d_a = d_model // 4

        # Один общий слой аналогии (вместо C(n,2) пар)
        self.proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.01))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                expert_outputs: List[torch.Tensor],
                expert_weights: torch.Tensor) -> torch.Tensor:
        if len(expert_outputs) < 2:
            return x

        # Берём два наиболее активных эксперта
        top2_idx = expert_weights.mean(dim=(0, 1)).topk(2).indices
        a = expert_outputs[top2_idx[0].item()]
        b = expert_outputs[top2_idx[1].item()]

        analogy = self.proj(torch.cat([a, b], dim=-1))
        return x + self.gate * self.ln(analogy)


# ─── Основная модель NautilusYiJing ───────────────────────────────────────────

class NautilusYiJing(nn.Module):
    """
    Интегрированная модель: NautilusMoME + YiJingGPT.

    Все три варианта интеграции активны одновременно:
      1. Q6GeometricRouter     — геометрический роутер
      2. YiJingCoreBlock       — геометрические слои ядра
      3. YiJingMicroExpert     — геометрические эксперты с якорями
    """

    def __init__(self, cfg: NautilusYiJingConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ── Эмбеддинги ────────────────────────────────────────────────────
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.block_size, d)
        self.drop = nn.Dropout(cfg.dropout)

        # SOLAN Q6 auxiliary signal (опционально, для BPE-токенов)
        solan = getattr(cfg, 'solan_table', None)
        self.use_solan = solan is not None
        if self.use_solan:
            self.register_buffer('solan_table', solan.float())
            self.glyph_proj = nn.Linear(6, d, bias=False)
            nn.init.normal_(self.glyph_proj.weight, std=0.01)
            self.solan_gate = nn.Parameter(torch.tensor(-3.0))

        # ── Вариант 2: YiJingCoreBlock — геометрическое ядро ─────────────
        n_first = cfg.n_layers // 2
        n_second = cfg.n_layers - n_first
        self.core_first = nn.ModuleList([
            YiJingCoreBlock(cfg) for _ in range(n_first)
        ])
        self.core_second = nn.ModuleList([
            YiJingCoreBlock(cfg) for _ in range(n_second)
        ])

        # ── Вариант 1: Q6GeometricRouter ─────────────────────────────────
        active_names = EXPERT_NAMES[:cfg.n_experts]
        self.router = Q6GeometricRouter(
            d_model=d,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            quant_temp=cfg.quant_temp,
            expert_names=active_names,
        )

        # ── Вариант 3: YiJingMicroExpert — геометрические эксперты ───────
        self.experts = nn.ModuleDict()
        for name in active_names:
            anchor_idx = EXPERT_HEXAGRAM_ANCHORS.get(name, 0)
            anchor = _q6_idx_to_vertex(anchor_idx)
            self.experts[name] = YiJingMicroExpert(
                d_model=d,
                d_expert=cfg.d_expert,
                dropout=cfg.dropout,
                q6_anchor=anchor,
                expert_name=name,
            )

        # ── NautilusBridge (иерархическое слияние) ────────────────────────
        self.bridge = SimpleBridge(d, cfg.n_experts)

        # ── CrossDomainAnalogy (упрощённый) ──────────────────────────────
        self.analogy = SimpleAnalogy(d, cfg.n_experts)

        # ── SYNTH expert (энтропийный) ────────────────────────────────────
        self.enable_synth = cfg.enable_synth
        if cfg.enable_synth:
            synth_anchor = _q6_idx_to_vertex(EXPERT_HEXAGRAM_ANCHORS['SYNTH'])
            self.synth_expert = YiJingMicroExpert(
                d_model=d, d_expert=cfg.d_expert,
                dropout=cfg.dropout,
                q6_anchor=synth_anchor,
                expert_name='SYNTH',
            )
            self.synth_entropy_threshold = nn.Parameter(
                torch.tensor(0.55), requires_grad=False
            )
            self.synth_gate = nn.Parameter(torch.tensor(0.05))

        # ── Регистрируем Q6-якоря экспертов (для мониторинга) ─────────────
        expert_anchors = torch.stack([
            _q6_idx_to_vertex(EXPERT_HEXAGRAM_ANCHORS.get(n, i * 9))
            for i, n in enumerate(active_names)
        ])
        self.register_buffer('expert_q6_anchors', expert_anchors)

        # ── Выходной слой ─────────────────────────────────────────────────
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        yijing = sum(
            p.numel() for m in [*self.core_first, *self.core_second,
                                 self.router, *self.experts.values()]
            for p in m.parameters()
        )
        return total, yijing

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None):
        """
        Args:
            idx: (B, T) токены
            targets: (B, T) следующие токены для loss

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar или None
            info: dict с диагностикой
        """
        B, T = idx.shape
        device = idx.device

        # ── Эмбеддинги ────────────────────────────────────────────────────
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))

        if self.use_solan:
            safe_idx = idx.clamp(0, self.solan_table.shape[0] - 1)
            glyph = self.glyph_proj(self.solan_table[safe_idx])
            gate = torch.sigmoid(self.solan_gate)
            x = tok + pos + gate * glyph
        else:
            x = tok + pos

        x = self.drop(x)

        # ── Вариант 2: ядро first half (YiJingCoreBlock) ──────────────────
        for block in self.core_first:
            x = block(x)

        # ── Вариант 1: Q6GeometricRouter ──────────────────────────────────
        expert_weights, q6_coords = self.router(x)  # (B,T,n_experts), (B,T,6)

        # ── Вариант 3: YiJingMicroExpert ──────────────────────────────────
        expert_names = list(self.experts.keys())
        expert_outputs = []
        for i, name in enumerate(expert_names):
            # Sparse: считаем только если эксперт получил ненулевой вес
            if expert_weights[:, :, i].sum() > 0:
                out = self.experts[name](x)
            else:
                out = torch.zeros_like(x)
            expert_outputs.append(out)

        # ── SimpleBridge: иерархическое слияние ───────────────────────────
        x = self.bridge(x, expert_outputs, expert_weights)

        # ── CrossDomainAnalogy ────────────────────────────────────────────
        x = self.analogy(x, expert_outputs, expert_weights)

        # ── SYNTH expert (при высокой энтропии роутинга) ──────────────────
        synth_info = {}
        if self.enable_synth:
            ew = expert_weights.clamp(min=1e-8)
            entropy = -(ew * ew.log()).sum(dim=-1)  # (B, T)
            synth_act = (entropy - self.synth_entropy_threshold).clamp(min=0)
            if synth_act.sum() > 0:
                synth_act = synth_act / synth_act.max().clamp(min=1e-8)
                synth_out = self.synth_expert(x)
                x = x + synth_out * synth_act.unsqueeze(-1) * self.synth_gate
                synth_info = {
                    'avg_entropy': entropy.mean().item(),
                    'synth_frac': (synth_act > 0).float().mean().item(),
                }

        # ── Вариант 2: ядро second half (YiJingCoreBlock) ─────────────────
        for block in self.core_second:
            x = block(x)

        # ── Выход ─────────────────────────────────────────────────────────
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )

        info = {
            'q6_coords': q6_coords.detach(),
            'expert_weights': expert_weights.detach(),
            'synth': synth_info,
        }
        return logits, loss, info
