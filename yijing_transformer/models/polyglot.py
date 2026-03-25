"""
polyglot.py — Четыре Языка Одной Истины (Polyglot Quartet)

Каждый Музыкант говорит на СВОЁМ языке:

  ① ФОРМАЛИСТ  → формулы:  E = m × c²
  ② АРХЕТИПИСТ → соответствия: (масса → земля), (энергия → вода)
  ③ АЛГОРИТМИСТ → графы:  [m] ──→ [×] ──→ [E]
  ④ ЛИНГВИСТ   → слова:  «Крупица вещества, помноженная на...»

Все четверо описывают ОДНО явление, но каждый — по-своему.
Как Розеттский камень: один текст на четырёх языках.

Научная основа:
  - Мультимодальное обучение (multimodal representation learning)
  - Теория категорий (изоморфизмы между описаниями)
  - Contrastive loss (CLIP-style) для выравнивания представлений
"""

import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Четыре Специализированных Словаря
# ═══════════════════════════════════════════════════════════════

# ① ФОРМАЛИСТ: математические символы
FORMULA_VOCAB = [
    '<pad>', '<unk>', '<sep>',
    # Цифры
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # Переменные
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u',
    'v', 'w', 'x', 'y', 'z',
    # Греческие
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'λ', 'μ',
    'ν', 'π', 'ρ', 'σ', 'τ', 'φ', 'ψ', 'ω',
    # Операторы
    '=', '+', '-', '×', '÷', '^', '√', '∫', '∂', '∑',
    'Π', '∞', '≈', '≠', '≤', '≥', '<', '>',
    # Скобки и структура
    '(', ')', '[', ']', '{', '}', ',', '.', ':', ';',
    # Логика
    '∀', '∃', '¬', '∧', '∨', '→', '↔', '∈', '∉', '⊂',
    '∅', '∪', '∩',
    # Спецсимволы
    'ℝ', 'ℤ', 'ℕ', 'ℂ', 'ℚ',
    # Пробел
    ' ',
]

# ② АРХЕТИПИСТ: элементы и соответствия
ARCHETYPE_VOCAB = [
    '<pad>', '<unk>', '<sep>',
    # Четыре стихии
    'ОГОНЬ', 'ВОДА', 'ЗЕМЛЯ', 'ВОЗДУХ',
    # Расширенные элементы
    'СВЕТ', 'ТЬМА', 'ВРЕМЯ', 'ПРОСТРАНСТВО',
    'ДВИЖЕНИЕ', 'ПОКОЙ', 'ФОРМА', 'ПУСТОТА',
    'ЭНЕРГИЯ', 'МАССА', 'СИЛА', 'ВОЛНА',
    'ЧАСТИЦА', 'ПОЛЕ', 'ПОРЯДОК', 'ХАОС',
    # Инь-Ян
    'ИНЬ', 'ЯН', 'ДАО',
    # Архетипы действий
    'РОЖДЕНИЕ', 'РОСТ', 'ЗРЕЛОСТЬ', 'УВЯДАНИЕ', 'СМЕРТЬ', 'ВОЗРОЖДЕНИЕ',
    # Отношения
    '→', '↔', '⊃', '≡', '∘', '⊕',
    'СТАНОВИТСЯ', 'СОДЕРЖИТ', 'ПРОТИВОПОЛОЖНО',
    'ПОРОЖДАЕТ', 'РАЗРУШАЕТ', 'ПИТАЕТ', 'ОГРАНИЧИВАЕТ',
    # Триграммы
    '☰', '☱', '☲', '☳', '☴', '☵', '☶', '☷',
    # Качества
    'ТВЁРДОЕ', 'МЯГКОЕ', 'ГОРЯЧЕЕ', 'ХОЛОДНОЕ',
    'БЫСТРОЕ', 'МЕДЛЕННОЕ', 'ТЯЖЁЛОЕ', 'ЛЁГКОЕ',
    'ВИДИМОЕ', 'НЕВИДИМОЕ', 'КОНЕЧНОЕ', 'БЕСКОНЕЧНОЕ',
    # Паттерны
    'СПИРАЛЬ', 'КРУГ', 'ЛИНИЯ', 'ТОЧКА', 'ВОЛНА_ПАТТЕРН',
    'ДЕРЕВО', 'СЕТЬ', 'КРИСТАЛЛ',
    # Пространство
    'ВЕРХ', 'НИЗ', 'ЦЕНТР', 'КРАЙ',
    'ВНУТРИ', 'СНАРУЖИ', 'МЕЖДУ',
]

# ③ АЛГОРИТМИСТ: узлы графа и операции
GRAPH_VOCAB = [
    '<pad>', '<unk>', '<sep>',
    # Узлы (абстрактные)
    'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7',
    'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15',
    # Типы узлов
    'INPUT', 'OUTPUT', 'PROCESS', 'DECISION', 'STORE',
    'TRANSFORM', 'FILTER', 'MERGE', 'SPLIT', 'BUFFER',
    # Рёбра
    '──→', '←──', '←→', '──', '╌╌→',
    'EDGE', 'FLOW', 'FEEDBACK', 'SKIP',
    # Операции
    'ADD', 'MUL', 'DIV', 'SUB', 'POW', 'SQRT', 'LOG', 'EXP',
    'SUM', 'PROD', 'MAX', 'MIN', 'AVG',
    'MAP', 'REDUCE', 'SCAN', 'ZIP', 'UNZIP',
    # Контроль потока
    'IF', 'THEN', 'ELSE', 'LOOP', 'BREAK', 'RETURN',
    'PARALLEL', 'SEQUENTIAL', 'PIPELINE',
    # Структуры данных
    'STACK', 'QUEUE', 'TREE', 'GRAPH', 'LIST', 'SET', 'MAP_DS',
    # Метки
    'LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3',
    'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7',
    # Свойства рёбер
    'WEIGHT', 'CAPACITY', 'COST', 'DELAY',
    # Паттерны
    'STAR', 'CHAIN', 'RING', 'MESH', 'BIPARTITE',
    # Скобки
    '[', ']', '(', ')', '{', '}', ',', ':', ';',
]

# ④ ЛИНГВИСТ: использует общий CharTokenizer (стандартный текст)
# — не нуждается в специальном словаре


# ═══════════════════════════════════════════════════════════════
# Специализированный Токенизатор для каждого языка
# ═══════════════════════════════════════════════════════════════

class SpecializedVocab:
    """Словарь одного специализированного языка."""

    def __init__(self, tokens: List[str], name: str = 'unnamed'):
        self.name = name
        self.tokens = list(tokens)
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2tok = {i: t for i, t in enumerate(self.tokens)}

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, token_list: List[str]) -> List[int]:
        """Кодирует список токенов в ID."""
        return [self.tok2id.get(t, 1) for t in token_list]  # 1 = <unk>

    def decode(self, ids: List[int]) -> List[str]:
        """Декодирует ID в список токенов."""
        return [self.id2tok.get(i, '<unk>') for i in ids]

    def decode_str(self, ids: List[int]) -> str:
        """Декодирует в строку с разделителями."""
        tokens = self.decode(ids)
        # Убираем pad и unk
        tokens = [t for t in tokens if t not in ('<pad>', '<unk>')]
        return ' '.join(tokens)


# Создаём четыре словаря
VOCABS = {
    'formalist': SpecializedVocab(FORMULA_VOCAB, 'формулы'),
    'archetypist': SpecializedVocab(ARCHETYPE_VOCAB, 'архетипы'),
    'algorithmist': SpecializedVocab(GRAPH_VOCAB, 'графы'),
    # linguist использует общий словарь — создаётся динамически
}


# ═══════════════════════════════════════════════════════════════
# Representation Head — проекция в свой язык
# ═══════════════════════════════════════════════════════════════

class RepresentationHead(nn.Module):
    """Голова, проецирующая скрытое состояние в специализированный словарь.

    Каждый Музыкант имеет свою голову, которая «переводит» общее
    представление на свой язык.
    """

    def __init__(self, d_model: int, vocab_size: int, name: str = ''):
        super().__init__()
        self.name = name
        self.vocab_size = vocab_size
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, D) — скрытое состояние музыканта
        Returns:
            logits: (B, T, vocab_size) — вероятности токенов этого языка
        """
        return self.proj(self.ln(h))

    @torch.no_grad()
    def decode_greedy(self, h: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
        """Жадное декодирование: скрытое состояние → последовательность токенов."""
        logits = self.forward(h) / temperature
        return logits.argmax(dim=-1)  # (B, T)


# ═══════════════════════════════════════════════════════════════
# Rosetta Bridge — выравнивание четырёх представлений
# ═══════════════════════════════════════════════════════════════

class RosettaBridge(nn.Module):
    """Розеттский мост — контрастивное выравнивание представлений.

    Как Розеттский камень содержал один текст на трёх языках,
    так и этот модуль обеспечивает, что четыре музыканта
    описывают одно и то же явление, хоть и на разных языках.

    Метод: CLIP-style contrastive loss между парами представлений.
    """

    def __init__(self, d_model: int, n_musicians: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_musicians = n_musicians

        # Проекция каждого музыканта в общее «семантическое» пространство
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model, bias=False),
            )
            for _ in range(n_musicians)
        ])

        # Learnable temperature (как в CLIP)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, representations: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            representations: list of 4 × (B, T, D) скрытых состояний

        Returns:
            alignment_loss: скаляр — насколько представления не совпадают
            info: диагностика
        """
        assert len(representations) == self.n_musicians

        # Проецируем в общее пространство и берём среднее по T
        projected = []
        for i, (rep, proj) in enumerate(zip(representations, self.projections)):
            z = proj(rep)                    # (B, T, D)
            z = z.mean(dim=1)               # (B, D) — среднее по позициям
            z = F.normalize(z, dim=-1)      # L2-нормализация
            projected.append(z)

        # Contrastive loss: все пары музыкантов должны быть близки
        temp = self.log_temp.exp()
        total_loss = torch.tensor(0.0, device=representations[0].device)
        n_pairs = 0
        cosine_sims = {}

        for i in range(self.n_musicians):
            for j in range(i + 1, self.n_musicians):
                # Косинусное сходство между i и j
                sim = (projected[i] * projected[j]).sum(dim=-1)  # (B,)
                cosine_sims[f'{i}-{j}'] = sim.mean().item()

                # Loss: хотим sim → 1.0 (максимальное согласие)
                pair_loss = (1.0 - sim).mean()
                total_loss = total_loss + pair_loss
                n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        info = {
            'cosine_sims': cosine_sims,
            'temperature': temp.item(),
        }

        return total_loss, info


# ═══════════════════════════════════════════════════════════════
# Polyglot Musician — музыкант со своим языком
# ═══════════════════════════════════════════════════════════════

class PolyglotMusician(nn.Module):
    """Музыкант, говорящий на своём специализированном языке.

    В отличие от базового Musician, имеет:
    - Собственную голову (RepresentationHead) для своего словаря
    - Учебный сигнал на своём языке (не только общий CE loss)
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 spec_vocab_size: int, name: str = '',
                 n_micro_experts: int = 4, micro_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.spec_vocab_size = spec_vocab_size

        # Стек трансформер-слоёв
        self.layers = nn.ModuleList([
            PolyglotLayer(d_model, n_heads, n_micro_experts, micro_dim, dropout)
            for _ in range(n_layers)
        ])

        # Голова для своего специализированного языка
        self.spec_head = RepresentationHead(d_model, spec_vocab_size, name)

        # Проекция из общего пространства
        self.proj_in = nn.Linear(d_model, d_model, bias=False)

        # Проекция обратно в общее пространство
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj_out.weight)

        self.ln_out = nn.LayerNorm(d_model)

        # Громкость
        self.volume = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_shared: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            contribution: (B, T, d_model) — вклад в общее состояние
            hidden: (B, T, d_model) — скрытое для специализированной головы
        """
        h = self.proj_in(x_shared)

        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask)

        contribution = self.proj_out(h)
        volume = torch.sigmoid(self.volume)
        contribution = self.ln_out(contribution * volume)

        return contribution, h  # h — для spec_head

    def get_spec_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Логиты в специализированном словаре."""
        return self.spec_head(hidden)


class PolyglotLayer(nn.Module):
    """Один слой полиглот-музыканта."""

    def __init__(self, d_model: int, n_heads: int,
                 n_micro_experts: int = 4, micro_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)

        # SwiGLU FFN
        self.w_gate = nn.Linear(d_model, micro_dim * n_micro_experts, bias=False)
        self.w_up = nn.Linear(d_model, micro_dim * n_micro_experts, bias=False)
        self.w_down = nn.Linear(micro_dim * n_micro_experts, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.w_down.weight)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out

        h = self.ln2(x)
        x = x + self.drop(self.w_down(F.silu(self.w_gate(h)) * self.w_up(h)))
        return x


# ═══════════════════════════════════════════════════════════════
# Polyglot Conductor — дирижёр со знанием языков
# ═══════════════════════════════════════════════════════════════

class PolyglotConductor(nn.Module):
    """Дирижёр, понимающий все четыре языка."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.blend = nn.Linear(d_model, 4, bias=True)
        nn.init.zeros_(self.blend.weight)
        nn.init.zeros_(self.blend.bias)

        self.ln_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.ffn[-2].weight)

    def forward(self, shared: torch.Tensor,
                contributions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            result: (B, T, D) — скоординированный результат
            blend_weights: (B, T, 4) — веса музыкантов
        """
        all_music = torch.cat(contributions, dim=1)
        q = self.ln_q(shared)
        kv = self.ln_kv(all_music)
        attended, _ = self.cross_attn(q, kv, kv)

        blend_weights = F.softmax(self.blend(shared), dim=-1)
        stacked = torch.stack(contributions, dim=-1)
        weighted = (stacked * blend_weights.unsqueeze(2)).sum(dim=-1)

        combined = attended + weighted
        combined = combined + self.ffn(self.ln_ff(combined))

        return combined, blend_weights


# ═══════════════════════════════════════════════════════════════
# PolyglotQuartet — полная модель
# ═══════════════════════════════════════════════════════════════

@dataclass
class PolyglotConfig:
    """Конфигурация полиглот-квартета."""
    vocab_size: int = 4096        # общий словарь (для лингвиста и эмбеддинга)
    d_model: int = 128
    n_layers: int = 2             # слоёв у каждого музыканта
    n_heads: int = 2
    n_micro_experts: int = 4
    micro_dim: int = 32
    block_size: int = 256
    dropout: float = 0.1
    rehearsal_rounds: int = 2
    # Веса лоссов
    rosetta_weight: float = 0.1   # вес contrastive loss
    spec_weight: float = 0.05     # вес специализированного loss


class PolyglotQuartet(nn.Module):
    """Четыре Языка Одной Истины.

    Каждый Музыкант:
    1. Обрабатывает общий вход через свой трансформер
    2. Генерирует вывод на СВОЁМ языке (формулы / архетипы / графы / слова)
    3. Вносит вклад в общее состояние

    Розеттский мост обеспечивает, что все четверо описывают одно и то же.
    """

    def __init__(self, cfg: PolyglotConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # ── Shared embedding ──
        self.tok_emb = nn.Embedding(cfg.vocab_size, D)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, D))
        self.emb_drop = nn.Dropout(cfg.dropout)

        # ── Четыре Музыканта с разными словарями ──
        self._musician_order = ['formalist', 'archetypist', 'algorithmist', 'linguist']

        spec_vocab_sizes = {
            'formalist': len(FORMULA_VOCAB),
            'archetypist': len(ARCHETYPE_VOCAB),
            'algorithmist': len(GRAPH_VOCAB),
            'linguist': cfg.vocab_size,  # общий словарь
        }

        self.musicians = nn.ModuleDict({
            name: PolyglotMusician(
                d_model=D,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                spec_vocab_size=spec_vocab_sizes[name],
                name=name,
                n_micro_experts=cfg.n_micro_experts,
                micro_dim=cfg.micro_dim,
                dropout=cfg.dropout,
            )
            for name in self._musician_order
        })

        # ── Дирижёры ──
        self.conductors = nn.ModuleList([
            PolyglotConductor(D, n_heads=cfg.n_heads, dropout=cfg.dropout)
            for _ in range(cfg.rehearsal_rounds)
        ])

        # ── Розеттский мост ──
        self.rosetta = RosettaBridge(D, n_musicians=4)

        # ── Rehearsal gates ──
        self.rehearsal_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0))
            for _ in range(cfg.rehearsal_rounds)
        ])

        # ── Output head (общий, для языковой модели) ──
        self.ln_out = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self._current_step = 0

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
                targets: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        B, T = idx.shape
        device = idx.device

        # ── Embedding ──
        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        shared_state = self.emb_drop(tok + pos)

        # ── Causal mask ──
        attn_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # ── Rehearsal rounds ──
        info = {'rounds': [], 'spec_logits': {}, 'rosetta': {}}
        all_hiddens = []  # для Rosetta

        for r, conductor in enumerate(self.conductors):
            contributions = []
            hiddens = []
            round_info = {}

            for i, name in enumerate(self._musician_order):
                musician = self.musicians[name]
                contrib, hidden = musician(shared_state, attn_mask=attn_mask)
                contributions.append(contrib)
                hiddens.append(hidden)
                round_info[name] = {
                    'volume': torch.sigmoid(musician.volume).item(),
                }

            # Дирижёр координирует
            orchestrated, blend_weights = conductor(shared_state, contributions)

            # Rehearsal gate
            gate = torch.sigmoid(self.rehearsal_gates[r])
            shared_state = shared_state + gate * orchestrated

            round_info['gate'] = gate.item()
            round_info['blend'] = blend_weights.mean(dim=(0, 1)).detach().tolist()
            info['rounds'].append(round_info)

            # Запоминаем hiddens последнего раунда для Rosetta
            if r == len(self.conductors) - 1:
                all_hiddens = hiddens

        # ── Специализированные логиты (каждый на своём языке) ──
        for i, name in enumerate(self._musician_order):
            spec_logits = self.musicians[name].get_spec_logits(all_hiddens[i])
            info['spec_logits'][name] = spec_logits

        # ── Output ──
        logits = self.head(self.ln_out(shared_state))

        # ── Loss ──
        loss = None
        if targets is not None:
            # Основной loss: языковая модель
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            # Rosetta loss: контрастивное выравнивание
            rosetta_loss, rosetta_info = self.rosetta(all_hiddens)
            info['rosetta'] = rosetta_info

            # Специализированный loss: лингвист учится предсказывать
            # те же targets на своём языке (его vocab = общий vocab)
            linguist_logits = info['spec_logits']['linguist']
            spec_loss = F.cross_entropy(
                linguist_logits.view(-1, linguist_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

            loss = ce_loss + self.cfg.rosetta_weight * rosetta_loss + \
                   self.cfg.spec_weight * spec_loss

            info['ce_loss'] = ce_loss.item()
            info['rosetta_loss'] = rosetta_loss.item()
            info['spec_loss'] = spec_loss.item()

        return logits, loss, info

    def count_parameters(self) -> Dict[str, int]:
        result = {}
        for name in self._musician_order:
            result[name] = sum(p.numel() for p in self.musicians[name].parameters())
        result['conductors'] = sum(
            p.numel() for c in self.conductors for p in c.parameters()
        )
        result['rosetta'] = sum(p.numel() for p in self.rosetta.parameters())
        result['embeddings'] = self.tok_emb.weight.numel() + self.pos_emb.numel()
        result['total'] = sum(p.numel() for p in self.parameters())
        return result

    @torch.no_grad()
    def speak(self, idx: torch.Tensor, musician_name: str) -> torch.Tensor:
        """Заставить одного музыканта «высказаться» на своём языке.

        Returns:
            token_ids: (B, T) — ID в специализированном словаре
        """
        self.eval()
        B, T = idx.shape
        device = idx.device

        tok = self.tok_emb(idx)
        pos = self.pos_emb[:, :T, :]
        shared = self.emb_drop(tok + pos)

        attn_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        musician = self.musicians[musician_name]
        _, hidden = musician(shared, attn_mask=attn_mask)
        spec_logits = musician.get_spec_logits(hidden)

        return spec_logits.argmax(dim=-1)


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

def build_polyglot(vocab_size: int = 4096, d_model: int = 128,
                   n_layers: int = 2, **kwargs) -> PolyglotQuartet:
    """Создаёт Полиглот-Квартет."""
    cfg = PolyglotConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=kwargs.get('n_heads', max(d_model // 64, 2)),
        n_micro_experts=kwargs.get('n_micro_experts', 4),
        micro_dim=kwargs.get('micro_dim', max(d_model // 4, 16)),
        block_size=kwargs.get('block_size', 256),
        dropout=kwargs.get('dropout', 0.1),
        rehearsal_rounds=kwargs.get('rehearsal_rounds', 2),
        rosetta_weight=kwargs.get('rosetta_weight', 0.1),
        spec_weight=kwargs.get('spec_weight', 0.05),
    )
    return PolyglotQuartet(cfg)
