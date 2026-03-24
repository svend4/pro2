"""
quartet.py — Четыре Бременских Музыканта (Four Bremen Musicians)

Четырёхмодельная архитектура, где каждая модель — отдельная «профессия»,
специализирующаяся на своём типе мышления. Вместе они образуют
конвейер, где каждый уровень дополняет остальные.

Четыре профессии (горизонтальные эксперты):

    ① ФОРМАЛИСТ (Formalist) — математика, формулы, точные структуры
       Инструмент: скрипка (точность, чистота линий)
       Элемент: Огонь (энергия преобразования)
       Сезон: Лето (максимальная активность)

    ② АРХЕТИПИСТ (Archetypist) — физика, архетипы, сложные формулы
       Инструмент: виолончель (глубина, обертоны)
       Элемент: Земля (фундаментальность)
       Сезон: Осень (сбор урожая паттернов)

    ③ АЛГОРИТМИСТ (Algorithmist) — химия, алгоритмы, комбинаторика
       Инструмент: духовые (трансформация, потоки)
       Элемент: Вода (адаптивность, текучесть)
       Сезон: Зима (кристаллизация структур)

    ④ ЛИНГВИСТ (Linguist) — язык, философия, психология, биология
       Инструмент: ударные (ритм, паттерн)
       Элемент: Воздух (связь, коммуникация)
       Сезон: Весна (порождение нового)

Каждый Музыкант — это миди-эксперт, состоящий из микро-экспертов.
Они не иерархия (не вертикаль), а четыре горизонтальных профессии
в одном оркестре, играющие на разных инструментах одну мелодию.
"""

import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class MusicianConfig:
    """Конфигурация одного Музыканта (миди-эксперта)."""
    name: str = 'formalist'
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_micro_experts: int = 8       # микро-экспертов внутри каждого слоя
    micro_expert_dim: int = 64     # размерность каждого микро-эксперта
    top_k_micro: int = 2           # сколько микро-экспертов активировать
    dropout: float = 0.1
    # Специализация: какие геометрические модули включены
    use_quantizer: bool = True
    quantizer_type: str = 'ternary'


@dataclass
class QuartetConfig:
    """Конфигурация ансамбля из четырёх Музыкантов."""
    vocab_size: int = 4096
    d_model: int = 256             # общая размерность (обмен между музыкантами)
    block_size: int = 512
    n_archetypes: int = 64         # размер общего архетипного пространства
    dropout: float = 0.1

    # Каждый Музыкант может иметь свою конфигурацию
    formalist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='formalist', n_micro_experts=8, micro_expert_dim=64,
        quantizer_type='gumbel',
    ))
    archetypist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='archetypist', n_micro_experts=6, micro_expert_dim=96,
        quantizer_type='ternary',
    ))
    algorithmist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='algorithmist', n_micro_experts=8, micro_expert_dim=64,
        quantizer_type='factored',
    ))
    linguist: MusicianConfig = field(default_factory=lambda: MusicianConfig(
        name='linguist', n_micro_experts=6, micro_expert_dim=96,
        quantizer_type='ternary',
    ))

    # Оркестр: как музыканты взаимодействуют
    conductor_heads: int = 4       # голов внимания у Дирижёра
    rehearsal_rounds: int = 2      # сколько раундов обмена между музыкантами
    curriculum_warmup: int = 2000  # шагов до полной активации всех музыкантов


# ═══════════════════════════════════════════════════════════════
# Micro-Expert Layer (внутренний строительный блок)
# ═══════════════════════════════════════════════════════════════

class MicroExpertFFN(nn.Module):
    """Один микро-эксперт: маленький FFN со специализацией.

    Каждый микро-эксперт — это «подмастерье» внутри Музыканта.
    """
    def __init__(self, d_model: int, d_expert: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Linear(d_model, d_expert, bias=False)
        self.gate = nn.Linear(d_model, d_expert, bias=False)
        self.down = nn.Linear(d_expert, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # Инициализация: маленький масштаб для стабильности
        nn.init.trunc_normal_(self.up.weight, std=0.02)
        nn.init.trunc_normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class MicroExpertMoE(nn.Module):
    """MoE из микро-экспертов внутри одного Музыканта.

    Soft top-k routing: каждый токен активирует top_k микро-экспертов
    через дифференцируемые веса.
    """
    def __init__(self, d_model: int, n_experts: int, d_expert: int,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)

        self.experts = nn.ModuleList([
            MicroExpertFFN(d_model, d_expert, dropout)
            for _ in range(n_experts)
        ])

        # Learnable router
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.zeros_(self.router.weight)
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # temperature

        # Auxiliary load balancing
        self._aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Router logits
        temp = F.softplus(self.log_temp) + 0.1
        logits = self.router(x) / temp  # (B, T, n_experts)

        # Soft top-k selection
        if self.top_k < self.n_experts:
            top_vals, top_idx = logits.topk(self.top_k, dim=-1)
            threshold = top_vals[..., -1:].detach()
            suppression = torch.sigmoid(10.0 * (logits - threshold))
            logits = logits * suppression

        weights = F.softmax(logits, dim=-1)  # (B, T, n_experts)

        # Load balancing loss
        avg_probs = weights.mean(dim=(0, 1))
        self._aux_loss = self.n_experts * (avg_probs * avg_probs).sum()

        # Compute all expert outputs (vectorized for small expert count)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (B, T, D, n_experts)

        # Weighted combination
        result = (expert_outputs * weights.unsqueeze(2)).sum(dim=-1)
        return result

    def get_aux_loss(self) -> torch.Tensor:
        return self._aux_loss


# ═══════════════════════════════════════════════════════════════
# Musician Layer (один слой внутри Музыканта)
# ═══════════════════════════════════════════════════════════════

class MusicianLayer(nn.Module):
    """Один трансформер-слой внутри Музыканта.

    Attention → MicroExpert MoE → LayerNorm.
    """
    def __init__(self, cfg: MusicianConfig):
        super().__init__()
        d = cfg.d_model

        # Self-attention
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )

        # Micro-Expert MoE
        self.ln2 = nn.LayerNorm(d)
        self.moe = MicroExpertMoE(
            d_model=d,
            n_experts=cfg.n_micro_experts,
            d_expert=cfg.micro_expert_dim,
            top_k=cfg.top_k_micro,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out

        # Micro-expert MoE
        x = x + self.moe(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════
# Musician (один из четырёх Бременских Музыкантов)
# ═══════════════════════════════════════════════════════════════

class Musician(nn.Module):
    """Один Музыкант — миди-эксперт со своей «профессией».

    Каждый Музыкант:
    - Имеет свой стек трансформер-слоёв с MicroExpert MoE
    - Принимает общий вход + сигнал от других музыкантов
    - Возвращает свою «партию» (вклад в общий результат)
    """
    def __init__(self, cfg: MusicianConfig, d_shared: int):
        super().__init__()
        self.name = cfg.name
        self.d_model = cfg.d_model
        self.d_shared = d_shared

        # Проекция из общего пространства в пространство музыканта
        self.proj_in = nn.Linear(d_shared, cfg.d_model, bias=False)

        # Стек слоёв (собственный трансформер)
        self.layers = nn.ModuleList([
            MusicianLayer(cfg) for _ in range(cfg.n_layers)
        ])

        # Проекция обратно в общее пространство
        self.proj_out = nn.Linear(cfg.d_model, d_shared, bias=False)
        nn.init.zeros_(self.proj_out.weight)  # начинаем с нулевого вклада

        # Финальная LayerNorm
        self.ln_out = nn.LayerNorm(d_shared)

        # Learnable «громкость» этого музыканта
        self.volume = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

    def forward(self, x_shared: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_shared: (B, T, d_shared) — общее представление
            attn_mask: маска внимания

        Returns:
            contribution: (B, T, d_shared) — вклад этого музыканта
        """
        # Проекция в своё пространство
        h = self.proj_in(x_shared)

        # Обработка своим трансформером
        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        # Проекция обратно + volume control
        contribution = self.proj_out(h)
        volume = torch.sigmoid(self.volume)
        return self.ln_out(contribution * volume)

    def get_aux_loss(self) -> torch.Tensor:
        """Суммарный auxiliary loss от всех MoE слоёв."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total = total + layer.moe.get_aux_loss()
        return total


# ═══════════════════════════════════════════════════════════════
# Conductor (Дирижёр — orchestrates the musicians)
# ═══════════════════════════════════════════════════════════════

class Conductor(nn.Module):
    """Дирижёр — управляет взаимодействием четырёх Музыкантов.

    Дирижёр не «умнее» музыкантов, он просто координирует:
    1. Принимает партии всех четырёх музыкантов
    2. Cross-attention: каждый музыкант «слышит» остальных
    3. Возвращает скоординированный сигнал каждому

    Это не иерархия — Дирижёр просто обеспечивает коммуникацию,
    как круглый стол, за которым сидят четыре учёных.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Cross-attention между музыкантами
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Soft routing: какой музыкант сколько влияет
        self.blend_proj = nn.Linear(d_model, 4, bias=True)
        nn.init.zeros_(self.blend_proj.weight)
        nn.init.zeros_(self.blend_proj.bias)

        # Post-processing
        self.ln_post = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.ffn[-2].weight)

    def forward(self, shared_state: torch.Tensor,
                contributions: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            shared_state: (B, T, D) — текущее общее состояние
            contributions: list of 4 × (B, T, D) — партии музыкантов

        Returns:
            orchestrated: (B, T, D) — скоординированный результат
        """
        B, T, D = shared_state.shape

        # Concatenate all musician contributions as KV sequence
        # (B, 4*T, D) — каждый музыкант добавляет T токенов
        all_music = torch.cat(contributions, dim=1)  # (B, 4*T, D)

        # Cross-attention: shared_state attends to all musicians
        q = self.ln_q(shared_state)
        kv = self.ln_kv(all_music)
        attended, _ = self.cross_attn(q, kv, kv)  # (B, T, D)

        # Soft blend weights per token (who to listen to more)
        blend_logits = self.blend_proj(shared_state)  # (B, T, 4)
        blend_weights = F.softmax(blend_logits, dim=-1)  # (B, T, 4)

        # Weighted sum of individual contributions
        stacked = torch.stack(contributions, dim=-1)  # (B, T, D, 4)
        weighted = (stacked * blend_weights.unsqueeze(2)).sum(dim=-1)  # (B, T, D)

        # Combine cross-attention + weighted blend
        combined = attended + weighted

        # Post-processing
        combined = combined + self.ffn(self.ln_post(combined))

        return combined


# ═══════════════════════════════════════════════════════════════
# Seasonal Cycle (Годовой Круг — curriculum & phase dynamics)
# ═══════════════════════════════════════════════════════════════

class SeasonalCycle(nn.Module):
    """Годовой Круг — циклическая динамика четырёх Музыкантов.

    Как времена года: каждый Музыкант имеет свой «пик активности»,
    но все работают постоянно. Цикл определяет относительную
    «громкость» каждого музыканта в зависимости от фазы обучения.

    Формалист  ⟷  Лето   (phase 0.00 — 0.25) — точность, формулы
    Архетипист ⟷  Осень  (phase 0.25 — 0.50) — глубина, паттерны
    Алгоритмист ⟷ Зима   (phase 0.50 — 0.75) — кристаллизация
    Лингвист   ⟷  Весна  (phase 0.75 — 1.00) — порождение

    Но это не жёсткое расписание — скорее мягкий curriculum bias.
    """
    def __init__(self, warmup_steps: int = 2000):
        super().__init__()
        self.warmup_steps = warmup_steps
        # Learnable phase offsets — модель может сдвигать «пики»
        self.phase_offsets = nn.Parameter(torch.tensor([0.0, 0.25, 0.5, 0.75]))
        # Learnable concentration — насколько острый пик
        self.log_concentration = nn.Parameter(torch.tensor(1.0))

    def forward(self, step: int) -> torch.Tensor:
        """Возвращает 4 весовых коэффициента для Музыкантов.

        Returns:
            weights: (4,) — мягкие веса, сумма ≈ 1
        """
        # Нормализованная фаза [0, 1) — циклическая
        if self.warmup_steps > 0:
            progress = min(step / self.warmup_steps, 1.0)
        else:
            progress = 1.0

        # Базовый вес: все музыканты работают с самого начала
        base = torch.ones(4, device=self.phase_offsets.device) * 0.25

        if progress < 1.0:
            # Во время warmup: постепенно добавляем сезонную модуляцию
            phase = (step % self.warmup_steps) / self.warmup_steps
            concentration = self.log_concentration.exp()

            # Косинусная близость к «пику» каждого музыканта
            phase_diff = torch.cos(2 * math.pi * (phase - self.phase_offsets))
            seasonal_bias = F.softmax(concentration * phase_diff, dim=0)

            # Мягкий переход от равномерного к сезонному
            weights = (1 - progress) * base + progress * seasonal_bias
        else:
            # После warmup: полностью learnable (сезонная модуляция)
            weights = base  # или можно оставить сезонную

        return weights


# ═══════════════════════════════════════════════════════════════
# Quartet (полный ансамбль — Четыре Бременских Музыканта)
# ═══════════════════════════════════════════════════════════════

class BremenQuartet(nn.Module):
    """Четыре Бременских Музыканта — полная языковая модель.

    Архитектура:
        1. Token embedding → shared space
        2. Для каждого rehearsal_round:
           a. Каждый Музыкант обрабатывает shared_state → contribution
           b. Дирижёр координирует contributions → new shared_state
        3. Output head → logits

    Это не ensemble (голосование) и не mixture (выбор одного).
    Это оркестр — каждый играет свою партию, и только вместе
    получается музыка.
    """
    def __init__(self, cfg: QuartetConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # ── Shared embeddings ──
        self.tok_emb = nn.Embedding(cfg.vocab_size, D)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, D))
        self.emb_drop = nn.Dropout(cfg.dropout)

        # ── Четыре Музыканта ──
        self.musicians = nn.ModuleDict({
            'formalist': Musician(cfg.formalist, d_shared=D),
            'archetypist': Musician(cfg.archetypist, d_shared=D),
            'algorithmist': Musician(cfg.algorithmist, d_shared=D),
            'linguist': Musician(cfg.linguist, d_shared=D),
        })

        # Порядок (для стабильной итерации)
        self._musician_order = ['formalist', 'archetypist', 'algorithmist', 'linguist']

        # ── Дирижёр (по одному на каждый rehearsal round) ──
        self.conductors = nn.ModuleList([
            Conductor(D, n_heads=cfg.conductor_heads, dropout=cfg.dropout)
            for _ in range(cfg.rehearsal_rounds)
        ])

        # ── Годовой Круг ──
        self.seasonal_cycle = SeasonalCycle(warmup_steps=cfg.curriculum_warmup)

        # ── Learnable residual gate for rehearsal rounds ──
        self.rehearsal_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))
            for _ in range(cfg.rehearsal_rounds)
        ])

        # ── Output ──
        self.ln_out = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Tracking
        self._current_step = 0
        self._last_info: Dict = {}

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def set_step(self, step: int):
        self._current_step = step

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices (optional, for loss)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
            info: diagnostic dict
        """
        B, T = idx.shape
        device = idx.device

        # ── Embedding ──
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        shared_state = self.emb_drop(tok + pos)

        # ── Causal mask ──
        attn_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )  # True = masked positions (for nn.MultiheadAttention)

        # ── Seasonal weights ──
        seasonal_w = self.seasonal_cycle(self._current_step)

        # ── Rehearsal rounds (оркестровые репетиции) ──
        info = {'seasonal_weights': seasonal_w.detach().tolist(), 'rounds': []}

        for r, conductor in enumerate(self.conductors):
            # Каждый Музыкант играет свою партию
            contributions = []
            round_info = {}

            for i, name in enumerate(self._musician_order):
                musician = self.musicians[name]
                c = musician(shared_state, attn_mask=attn_mask)
                # Модулируем сезонным весом
                c = c * seasonal_w[i]
                contributions.append(c)
                round_info[name] = {
                    'volume': torch.sigmoid(musician.volume).item(),
                    'seasonal_w': seasonal_w[i].item(),
                }

            # Дирижёр координирует
            orchestrated = conductor(shared_state, contributions)

            # Residual с learnable gate
            gate = torch.sigmoid(self.rehearsal_gates[r])
            shared_state = shared_state + gate * orchestrated

            round_info['gate'] = gate.item()
            info['rounds'].append(round_info)

        # ── Output ──
        logits = self.head(self.ln_out(shared_state))

        # ── Loss ──
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            # Auxiliary losses от MoE
            aux_loss = torch.tensor(0.0, device=device)
            for name in self._musician_order:
                aux_loss = aux_loss + self.musicians[name].get_aux_loss()
            loss = loss + 0.01 * aux_loss

        self._last_info = info
        return logits, loss, info

    def get_info(self) -> Dict:
        return self._last_info

    def count_parameters(self) -> Dict[str, int]:
        """Подсчёт параметров по компонентам."""
        result = {}
        for name in self._musician_order:
            result[name] = sum(p.numel() for p in self.musicians[name].parameters())
        result['conductors'] = sum(
            p.numel() for c in self.conductors for p in c.parameters()
        )
        result['embeddings'] = (
            self.tok_emb.weight.numel() + self.pos_emb.numel()
        )
        result['total'] = sum(p.numel() for p in self.parameters())
        return result


# ═══════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════

def build_quartet(
    vocab_size: int = 4096,
    d_model: int = 256,
    musician_layers: int = 4,
    micro_experts: int = 8,
    micro_dim: int = 64,
    rehearsal_rounds: int = 2,
    **kwargs
) -> BremenQuartet:
    """Создаёт Квартет с разумными defaults.

    Каждый Музыкант ~0.5-1M параметров (миди-эксперт),
    весь Квартет ~3-5M параметров.

    Usage:
        model = build_quartet(vocab_size=4096, d_model=256)
        logits, loss, info = model(token_ids, targets=labels)
    """
    # Propagate common settings to all musicians
    musician_defaults = dict(
        d_model=d_model,
        n_layers=musician_layers,
        n_heads=max(d_model // 64, 2),
        dropout=kwargs.get('dropout', 0.1),
    )

    cfg = QuartetConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=kwargs.get('block_size', 512),
        dropout=kwargs.get('dropout', 0.1),
        rehearsal_rounds=rehearsal_rounds,
        conductor_heads=max(d_model // 64, 2),
        curriculum_warmup=kwargs.get('curriculum_warmup', 2000),

        formalist=MusicianConfig(
            name='formalist',
            n_micro_experts=micro_experts,
            micro_expert_dim=micro_dim,
            top_k_micro=2,
            quantizer_type='gumbel',
            **musician_defaults,
        ),
        archetypist=MusicianConfig(
            name='archetypist',
            n_micro_experts=max(micro_experts * 3 // 4, 2),
            micro_expert_dim=int(micro_dim * 1.5),
            top_k_micro=2,
            quantizer_type='ternary',
            **musician_defaults,
        ),
        algorithmist=MusicianConfig(
            name='algorithmist',
            n_micro_experts=micro_experts,
            micro_expert_dim=micro_dim,
            top_k_micro=2,
            quantizer_type='factored',
            **musician_defaults,
        ),
        linguist=MusicianConfig(
            name='linguist',
            n_micro_experts=max(micro_experts * 3 // 4, 2),
            micro_expert_dim=int(micro_dim * 1.5),
            top_k_micro=2,
            quantizer_type='ternary',
            **musician_defaults,
        ),
    )

    return BremenQuartet(cfg)
