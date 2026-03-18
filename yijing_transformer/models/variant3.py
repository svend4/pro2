"""
Variant 3 — архитектура с архетипами как сквозной осью.

Реализует концептуальную схему из CONCEPTUAL_STAGE.md:

    Input (tokens)
        ↓  [Embedding + Positional]
        ↓  [HexagramProjection]    ← проекция h на ближайшие архетипы Q6
        ↓  [BianGuaAttention]      ← внимание с метрикой Хэмминга Q6
        ↓  [TernaryGate {-1,0,+1}] ← тернарная активация (три уровня знания)
        ↓  [ArchetypalInterlingua] ← 64 архетипа как посредник-хаб
        ↓  [CrossHexagramAnalogy]  ← 変爻-аналогии через Хэмминг-1 переходы
        ↓  [NautilusYiJinRouter]   ← роутинг по 6 доменам через Q6
        ↓  [Output projection]
        ↓  Выход (логиты)

Домены (шесть линий гексаграммы):
    Линия 1: GEO   / CODE   — 地 Chi,   Земля       — Мастер/Инженер
    Линия 2: HYDRO / RECON  — 水 Sui,   Вода        — Разведчик/Аналитик
    Линия 3: PYRO  / SYSTEM — 火 Ka,    Огонь       — Архитектор
    Линия 4: AERO  / MATH   — 風 Fu,    Ветер       — Математик/Логик
    Линия 5: COSMO / HUMAN  — 空 Kū,    Пустота     — Лидер/Дипломат
    Линия 6: NOOS  / INFO   — 識 Shiki, Сознание    — Философ/Мудрец
"""

import math
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты Q6
# ──────────────────────────────────────────────────────────────────────────────

def _make_hexagrams() -> torch.Tensor:
    """64 гексаграммы — все вершины гиперкуба {-1, +1}^6."""
    verts = list(itertools.product([-1.0, 1.0], repeat=6))
    return torch.tensor(verts, dtype=torch.float32)  # (64, 6)


def _make_biangua_matrix(hexagrams: torch.Tensor) -> torch.Tensor:
    """64×64 матрица смежности: M[i,j]=1 iff hamming(i,j)==1.

    Каждая строка имеет ровно 6 единиц — по одной на каждую изменяющуюся
    линию (変爻). Векторизованное вычисление через dot-product:
      hamming(i,j) == 1  ↔  dot(hex_i, hex_j) == 4
    """
    dot = hexagrams @ hexagrams.T  # (64, 64)
    return (dot == 4.0).float()


def hamming_distance_soft(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Мягкое расстояние Хэмминга между soft-гексаграммами.

    hamming_soft(a, b) = (6 - <a, b>) / 2  при a,b ∈ {-1,+1}^6
    Работает с непрерывными значениями (soft approximation).

    Args:
        a: (*, 6)
        b: (*, 6)
    Returns:
        (*)  ∈ [0, 6]
    """
    return (6.0 - (a * b).sum(dim=-1)) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. HexagramProjection — «мышление начинается здесь»
# ──────────────────────────────────────────────────────────────────────────────

class HexagramProjection(nn.Module):
    """Проецирует скрытое состояние h ∈ R^d на 64 гексаграммы Q6.

    Это не квантизация — это «голосование»: каждый из 64 архетипов
    получает вес, пропорциональный cos-близости h к этому архетипу.

    Схема:
        h  →  proj_to_6d  →  tanh  →  soft_bits ∈ (-1,+1)^6
              soft_bits  ·  hexagrams.T  →  similarity (64,)
              softmax(similarity / T)    →  hex_weights (64,)   [сумма=1]
              hex_weights  ·  hexagrams  →  hex_embed ∈ R^6
              proj_from_6d(hex_embed)    →  signal ∈ R^d
        output = h + gate * signal

    Возвращает:
        h_enriched: (B, T, d)   — обогащённое представление
        hex_weights: (B, T, 64) — мягкое распределение по архетипам
    """

    def __init__(self, d_model: int, temperature: float = 0.5):
        super().__init__()
        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)  # (64, 6)

        self.proj_to_6d   = nn.Linear(d_model, 6,       bias=False)
        self.proj_from_6d = nn.Linear(6,       d_model, bias=False)
        self.gate         = nn.Parameter(torch.tensor(0.1))
        self.log_temp     = nn.Parameter(torch.tensor(temperature).log())

        nn.init.normal_(self.proj_to_6d.weight,   std=0.02)
        nn.init.normal_(self.proj_from_6d.weight, std=0.02)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.1, 5.0)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, d)
        soft_bits  = torch.tanh(self.proj_to_6d(h))               # (B, T, 6)
        similarity = soft_bits @ self.hexagrams.T                  # (B, T, 64)
        hex_weights = F.softmax(similarity / self.temperature, dim=-1)  # (B, T, 64)

        # Reconstruct: soft hexagram position → project back to d_model
        hex_embed = hex_weights @ self.hexagrams                   # (B, T, 6)
        signal    = self.proj_from_6d(hex_embed)                   # (B, T, d)

        h_enriched = h + torch.sigmoid(self.gate) * signal
        return h_enriched, hex_weights


# ──────────────────────────────────────────────────────────────────────────────
# 2. BianGuaAttention — внимание, знающее топологию
# ──────────────────────────────────────────────────────────────────────────────

class BianGuaAttention(nn.Module):
    """Multi-head attention с топологическим bias из Q6.

    Стандартное внимание:
        score(q, k) = q·k / √d

    BianGuaAttention (変卦-внимание):
        score(q, k) = q·k / √d + λ · <soft_hex_q, soft_hex_k> / 2

    Смысл: токены с близкими архетипами (малое расстояние Хэмминга)
    притягиваются сильнее. Токены с далёкими архетипами требуют
    высокой косинусной близости для получения высокого внимания.
    """

    @property
    def hamming_lambda(self):
        """Bounded hamming_lambda ∈ (0, 1) via sigmoid."""
        return torch.sigmoid(self.hamming_lambda_logit)

    def __init__(self, d_model: int, n_heads: int,
                 hamming_lambda: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.hamming_lambda_logit = nn.Parameter(torch.tensor(hamming_lambda))

        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)  # (64, 6)

    def forward(self, x: torch.Tensor,
                hex_weights: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D    = self.n_heads, self.head_dim
        scale   = D ** -0.5

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) * scale                  # (B, H, T, T)

        # Topological bias via soft hexagram positions
        # soft_hex[b, t] = weighted average of codebook vertices ∈ (-1, +1)^6
        soft_hex = hex_weights @ self.hexagrams                    # (B, T, 6)
        # hamming_affinity[b, i, j] = <soft_hex_i, soft_hex_j> / 2 ∈ [-3, 3]
        # Low Hamming ↔ High affinity ↔ Positive bias
        ham_affinity = torch.bmm(soft_hex, soft_hex.transpose(-2, -1)) / 2.0
        ham_bias = torch.sigmoid(self.hamming_lambda_logit) * ham_affinity.unsqueeze(1) # (B, 1, T, T)

        scores = scores + ham_bias

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0) == 0,
                                    float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = attn.nan_to_num(0.0)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# 3. TernaryGate {-1, 0, +1} — три уровня знания
# ──────────────────────────────────────────────────────────────────────────────

class TernaryGate(nn.Module):
    """Тернарный гейт: три состояния знания.

        +1 = ян — знаю и подтверждаю
         0 = 変爻 — неопределённость, суперпозиция
        -1 = инь — знаю, что нет

    Механизм:
        scores = tanh(gate_proj(x) / T)   ∈ (-1, +1)
        hard:  +1 if scores > threshold
                0 if |scores| ≤ threshold
               -1 if scores < -threshold
        STE: gate = scores + (hard - scores).detach()
        output = x + scale * gate * x
    """

    def __init__(self, d_model: int, uncertainty_budget: float = 0.3):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        # Learnable uncertainty budget: how often gate is 0
        self.log_uncertainty = nn.Parameter(
            torch.tensor(uncertainty_budget).clamp(0.01, 0.99).logit()
        )
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.scale    = nn.Parameter(torch.tensor(0.1))

    @property
    def uncertainty_budget(self) -> torch.Tensor:
        return torch.sigmoid(self.log_uncertainty)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.05, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        scores = torch.tanh(self.gate_proj(x) / self.temperature)  # (B, T, d)

        # Dynamic threshold from uncertainty budget (detached for STE stability)
        # budget=0.3 → ~30% of values are zero → threshold ≈ 0.35
        threshold = ((1.0 - self.uncertainty_budget) * 0.5 + 0.1).detach()

        gate_hard = torch.zeros_like(scores)
        gate_hard[scores >  threshold] =  1.0
        gate_hard[scores < -threshold] = -1.0

        # STE: backward through soft, forward through hard
        gate = scores + (gate_hard - scores).detach()

        # uncertainty_budget participates in computation:
        # higher budget → gate effect is scaled down (more uncertainty = less impact)
        active_scale = 1.0 - self.uncertainty_budget * 0.5  # ∈ (0.5, 1.0)
        return x + self.scale * active_scale * gate * x


# ──────────────────────────────────────────────────────────────────────────────
# 4. CrossHexagramAnalogy — 変爻 аналогия через Хэмминг-1
# ──────────────────────────────────────────────────────────────────────────────

class CrossHexagramAnalogy(nn.Module):
    """変爻 (biàn yáo) — кросс-доменная аналогия через изменяющиеся линии.

    В И-Цзин «変爻» — одна линия гексаграммы меняется с ян на инь
    или обратно. Это эквивалентно Хэмминг-расстоянию d=1 в Q6.

    Механизм:
        Для каждого токена с распределением hex_weights (B, T, 64):
        1. analogy_weights = hex_weights · biangua_matrix  (Хэмминг-1 соседи)
        2. analogy_embed = analogy_weights · hexagram_embed
        3. gate = sigmoid(gate_proj(x))
        4. output = x + scale · out_proj(gate * analogy_embed)

    Результат: каждый токен «слышит» что произошло бы, если бы
    одна из его линий изменилась — это формализует аналогическое мышление.
    """

    def __init__(self, d_model: int, n_archetypes: int = 64):
        super().__init__()
        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)                      # (64, 6)

        biangua = _make_biangua_matrix(hexagrams)
        self.register_buffer('biangua_matrix', biangua)                  # (64, 64)

        # Learnable embedding for each of the 64 archetypes
        self.hexagram_embed = nn.Parameter(
            torch.randn(n_archetypes, d_model) * 0.02
        )
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.scale     = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor,
                hex_weights: torch.Tensor) -> torch.Tensor:
        # hex_weights: (B, T, 64)

        # Which hexagrams are Hamming-1 away from current position?
        # analogy_w[b, t, j] = sum_i hex_weights[b,t,i] * biangua[i,j]
        analogy_w = hex_weights @ self.biangua_matrix                    # (B, T, 64)
        # Normalize to distribution
        analogy_w = analogy_w / (analogy_w.sum(dim=-1, keepdim=True) + 1e-8)

        # Analogy context: what do the neighboring archetypes "say"?
        analogy_embed = analogy_w @ self.hexagram_embed                  # (B, T, d)

        # Gated residual
        gate   = torch.sigmoid(self.gate_proj(x))
        signal = self.out_proj(gate * analogy_embed)

        return x + torch.sigmoid(self.scale) * signal


# ──────────────────────────────────────────────────────────────────────────────
# 5. NautilusYiJinRouter — роутинг по 6 доменам через Q6
# ──────────────────────────────────────────────────────────────────────────────

# Шесть доменов = шесть линий гексаграммы
DOMAINS: List[str] = ["GEO", "HYDRO", "PYRO", "AERO", "COSMO", "NOOS"]

# Якоря доменов в Q6 (индексы гексаграмм 0..63)
DOMAIN_ANCHORS: Dict[str, int] = {
    "GEO":   0,   # ䷁ 坤 Кунь   — Земля,    Линия 1
    "HYDRO": 18,  # ䷒ 坎 Кань   — Вода,     Линия 2
    "PYRO":  45,  # ䷬ 離 Ли     — Огонь,    Линия 3
    "AERO":  6,   # ䷅ 巽 Сюнь   — Ветер,    Линия 4
    "COSMO": 27,  # ䷛ 兌 Дуй    — Пустота,  Линия 5
    "NOOS":  63,  # ䷿ 乾 Цянь   — Сознание, Линия 6
}


class NautilusYiJinRouter(nn.Module):
    """Q6-роутер: гексаграмма = паттерн активации 6 доменов.

    Ключевая идея NautilusYiJin:
        Каждый из 6 экспертов = одна линия гексаграммы.
        Гексаграмма = паттерн активации всех шести доменов одновременно.

        ䷀ ☰ (111111) = все 6 доменов активны   = полный синтез
        ䷁ ☷ (000000) = ни один не доминирует   = нейтраль
        ䷡ (100000) = только GEO/CODE активен   = чистый код

    Роутинг:
        Вместо softmax: query → Q6-encoder → soft_hex
        domain_score[j] = dot(soft_hex, domain_anchor[j])
        domain_weight[j] = sigmoid(domain_score[j])   ← {0,1} per domain
    """

    def __init__(self, d_model: int):
        super().__init__()
        hexagrams = _make_hexagrams()
        self.register_buffer('hexagrams', hexagrams)

        # Q6 encoder
        self.q6_proj  = nn.Linear(d_model, 6, bias=False)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.5)))

        # 6 domain experts (lightweight FFN per domain)
        self.domain_experts = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.GELU(),
                nn.Linear(d_model, d_model, bias=False),
            )
            for name in DOMAINS
        })

        # Register domain Q6 anchors as buffer
        anchors = torch.stack([hexagrams[DOMAIN_ANCHORS[n]] for n in DOMAINS])
        self.register_buffer('domain_q6_anchors', anchors)  # (6, 6)

        self.ln       = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate     = nn.Parameter(torch.tensor(-1.0))  # starts near 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.1, 5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, T, C = x.shape

        # Q6 encoding
        soft_bits = torch.tanh(self.q6_proj(x))                          # (B, T, 6)

        # Soft hexagram assignment
        similarity  = soft_bits @ self.hexagrams.T                       # (B, T, 64)
        hex_weights = F.softmax(similarity / self.temperature, dim=-1)   # (B, T, 64)
        soft_hex    = hex_weights @ self.hexagrams                        # (B, T, 6)

        # Domain routing: line j active ↔ soft_hex[j] aligned with anchor[j]
        # domain_scores ∈ (-6, +6)
        domain_scores  = soft_hex @ self.domain_q6_anchors.T             # (B, T, 6)
        domain_weights = torch.sigmoid(domain_scores)                    # (B, T, 6)

        # Apply each domain expert, weighted by its line activation
        expert_outputs = []
        for j, name in enumerate(DOMAINS):
            w = domain_weights[..., j : j + 1]                          # (B, T, 1)
            expert_out = self.domain_experts[name](x)                    # (B, T, d)
            expert_outputs.append(w * expert_out)

        combined = torch.stack(expert_outputs, dim=0).sum(dim=0)         # (B, T, d)
        out = self.out_proj(self.ln(combined))

        routing_info = {
            'hex_weights':    hex_weights.detach(),    # (B, T, 64)
            'domain_weights': domain_weights.detach(), # (B, T, 6)
            'soft_hex':       soft_hex.detach(),       # (B, T, 6)
        }
        return x + torch.sigmoid(self.gate) * out, routing_info


# ──────────────────────────────────────────────────────────────────────────────
# 6. Variant3Block — один слой (все пять операций)
# ──────────────────────────────────────────────────────────────────────────────

class Variant3Block(nn.Module):
    """Один блок Варианта 3: HexProj → BianGuaAttn → TernGate → Interlingua → Analogy → FFN.

    Поток:
        x → norm → HexagramProjection → hex_weights, h_enriched
        x → norm → BianGuaAttention(x, hex_weights) → attn_out
        attn_out → TernaryGate → ternary_out
        [attn_out, ternary_out] → ArchetypalInterlingua(x) → inter_out
        inter_out → CrossHexagramAnalogy(hex_weights) → analogy_out
        analogy_out → FFN (или HierarchicalMoEFFN) → block output
    """

    def __init__(self, d_model: int, n_heads: int,
                 ffn_mult:          int   = 4,
                 hamming_lambda:    float = 0.1,
                 uncertainty_budget: float = 0.3,
                 use_hierarchical_moe: bool = False):
        super().__init__()
        self.norm_hex  = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn  = nn.LayerNorm(d_model)

        self.hex_proj   = HexagramProjection(d_model)
        self.biangua_attn = BianGuaAttention(d_model, n_heads, hamming_lambda)
        self.ternary_gate = TernaryGate(d_model, uncertainty_budget)

        # ArchetypalInterlingua: 2 sources (attn_out, ternary_out)
        # Imported here to avoid circular import at module level
        from yijing_transformer.models.geometry.routing import ArchetypalInterlingua
        self.interlingua = ArchetypalInterlingua(
            d_model=d_model,
            n_sources=2,
            n_archetypes=64,
            use_ternary=True,
            uncertainty_budget=uncertainty_budget,
        )

        self.analogy = CrossHexagramAnalogy(d_model)

        self.use_hierarchical_moe = use_hierarchical_moe
        if use_hierarchical_moe:
            from yijing_transformer.models.hierarchical_moe import (
                HierarchicalMoEFFN, HMoEConfig,
            )
            self.hmoe = HierarchicalMoEFFN(HMoEConfig(d_model=d_model))
        else:
            # SwiGLU-style FFN (оригинальный)
            d_ff = d_model * ffn_mult
            self.ffn_gate_proj  = nn.Linear(d_model, d_ff, bias=False)
            self.ffn_value_proj = nn.Linear(d_model, d_ff, bias=False)
            self.ffn_out_proj   = nn.Linear(d_ff, d_model, bias=False)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        gate  = F.silu(self.ffn_gate_proj(x))
        value = self.ffn_value_proj(x)
        return self.ffn_out_proj(gate * value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Локализация в Q6
        norm_x = self.norm_hex(x)
        h_enriched, hex_weights = self.hex_proj(norm_x)
        x = x + (h_enriched - norm_x)  # residual: add projection signal

        # 2. Топологическое внимание
        attn_out = x + self.biangua_attn(self.norm_attn(x), hex_weights)

        # 3. Тернарный гейт — три уровня знания
        ternary_out = self.ternary_gate(attn_out)

        # 4. Архетипальная интерлингва — хаб для двух источников
        # Сигнатура: forward(x, source_outputs)
        inter_out = self.interlingua(attn_out, [attn_out, ternary_out])
        self._interlingua_loss = self.interlingua.get_interlingua_loss()

        # 5. Кросс-архетипная аналогия — 変爻 механизм
        analogy_out = self.analogy(inter_out, hex_weights)

        # 6. FFN (стандартный или HierarchicalMoE)
        if self.use_hierarchical_moe:
            moe_out, moe_info = self.hmoe(self.norm_ffn(analogy_out))
            out = analogy_out + moe_out
            # lb_loss передаётся наружу через атрибут для доступа в тренере
            self._last_moe_info = moe_info
        else:
            out = analogy_out + self._ffn(self.norm_ffn(analogy_out))
            self._last_moe_info = None
        return out


# ──────────────────────────────────────────────────────────────────────────────
# 7. Variant3GPT — полная модель
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Variant3Config:
    """Конфигурация Варианта 3."""
    vocab_size:          int   = 256
    block_size:          int   = 128
    d_model:             int   = 128
    n_heads:             int   = 4
    n_layers:            int   = 4
    ffn_mult:            int   = 4
    hamming_lambda:      float = 0.1
    uncertainty_budget:  float = 0.3
    dropout:             float = 0.0
    use_domain_routing:  bool  = True
    use_hierarchical_moe: bool = False   # заменить _ffn на HierarchicalMoEFFN


class Variant3GPT(nn.Module):
    """Полная языковая модель Варианта 3 — архетипы как сквозная ось.

    Архитектура:
        Embedding (token + position)
            ↓
        [Variant3Block × n_layers]
            ↓
        [NautilusYiJinRouter]  (если use_domain_routing=True)
            ↓
        LayerNorm → Linear Head → logits
    """

    def __init__(self, cfg: Variant3Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.block_size, d)
        self.drop    = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            Variant3Block(
                d_model=d,
                n_heads=cfg.n_heads,
                ffn_mult=cfg.ffn_mult,
                hamming_lambda=cfg.hamming_lambda,
                uncertainty_budget=cfg.uncertainty_budget,
                use_hierarchical_moe=cfg.use_hierarchical_moe,
            )
            for _ in range(cfg.n_layers)
        ])

        if cfg.use_domain_routing:
            self.domain_router = NautilusYiJinRouter(d)

        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        tokens:  torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        B, T = tokens.shape
        assert T <= self.cfg.block_size, f"Sequence length {T} > block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=tokens.device)
        x   = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))   # (B, T, d)

        for block in self.blocks:
            x = block(x)

        routing_info = None
        if hasattr(self, 'domain_router'):
            x, routing_info = self.domain_router(x)

        x      = self.ln_f(x)
        logits = self.head(x)                                         # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss, routing_info

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        lines = [
            f"Variant3GPT — архетипы как сквозная ось",
            f"  vocab_size:  {self.cfg.vocab_size}",
            f"  block_size:  {self.cfg.block_size}",
            f"  d_model:     {self.cfg.d_model}",
            f"  n_heads:     {self.cfg.n_heads}",
            f"  n_layers:    {self.cfg.n_layers}",
            f"  parameters:  {self.count_parameters():,}",
            f"  Q6 domains:  {' / '.join(DOMAINS)}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты для анализа и интерпретации
# ──────────────────────────────────────────────────────────────────────────────

HEXAGRAM_NAMES = [
    "☷坤", "☶艮", "☵坎", "☴巽", "☳震", "☲離", "☱兌", "☰乾",
    # ... полный список для отображения (первые 8 как пример)
]


def get_dominant_hexagram(hex_weights: torch.Tensor) -> torch.Tensor:
    """Возвращает индекс доминирующей гексаграммы для каждого токена.

    Args:
        hex_weights: (B, T, 64)
    Returns:
        indices: (B, T) LongTensor
    """
    return hex_weights.argmax(dim=-1)


def get_active_domains(domain_weights: torch.Tensor,
                       threshold: float = 0.5) -> List[List[List[str]]]:
    """Возвращает список активных доменов для каждого токена.

    Args:
        domain_weights: (B, T, 6) ∈ (0, 1)
        threshold: порог активации
    Returns:
        list[batch][seq] = list of active domain names
    """
    B, T, _ = domain_weights.shape
    result = []
    for b in range(B):
        batch_result = []
        for t in range(T):
            active = [DOMAINS[j] for j in range(6)
                      if domain_weights[b, t, j] > threshold]
            batch_result.append(active)
        result.append(batch_result)
    return result


def biangua_path(start_hex: int, end_hex: int) -> Optional[List[int]]:
    """Находит путь по 変爻-переходам от start_hex до end_hex.

    Greedy BFS по Хэмминг-1 переходам. Если расстояние Хэмминга = k,
    путь состоит из k шагов (k изменяющихся линий).

    Args:
        start_hex: индекс начальной гексаграммы (0..63)
        end_hex:   индекс конечной гексаграммы (0..63)
    Returns:
        list of hexagram indices from start to end, or None if unreachable
    """
    hexagrams = _make_hexagrams()
    biangua   = _make_biangua_matrix(hexagrams)
    adj       = biangua.nonzero(as_tuple=False)  # (6*64, 2)

    # BFS
    from collections import deque
    queue   = deque([(start_hex, [start_hex])])
    visited = {start_hex}

    while queue:
        current, path = queue.popleft()
        if current == end_hex:
            return path
        neighbors = biangua[current].nonzero(as_tuple=False).squeeze(-1).tolist()
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None
