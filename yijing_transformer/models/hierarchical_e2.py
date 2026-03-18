"""
hierarchical_e2.py — Вариант Е2: Полная иерархическая интеграция.

Пять уровней абстракции (от конкретного к философскому),
соответствующих α-уровням методологии info1:

  α = −4  ║  GLYPH LEVEL   ─ GlyphComposer + TokenAbstractor
          ║                  байты → Q6-глифы → ранняя кластеризация
          ║
  α = −2  ║  CORE LEVEL    ─ Variant3Block × n_core
          ║                  Q6-внимание + тернарные ворота + аналогии
          ║                  (может загружать checkpoint_bidir_v2.pt)
          ║
  α =  0  ║  METHOD LEVEL  ─ ArchetypalInterlingua (64 архетипа)
          ║                  синтез двух потоков через тернарное голосование
          ║
  α = +2  ║  THEORY LEVEL  ─ NautilusHierarchy (7 камер)
          ║                  геометрическое обогащение: куб → дворец → цветок
          ║
  α = +4  ║  PHILO LEVEL   ─ ConvergenceBridge + MatrixGrammar
          ║                  2D синтаксис + встреча глифов и кластеров

Обучение снизу вверх — каждый уровень сначала заморожен:
  Фаза 1: α=−4  (glyph_level)
  Фаза 2: α=−2  (core_level, опционально от checkpoint)
  Фаза 3: α= 0  (method_level)
  Фаза 4: α=+2  (theory_level)
  Фаза 5: α=+4  (philo_level)

Usage:
  from yijing_transformer.models.hierarchical_e2 import HierarchicalE2, E2Config
  cfg = E2Config(vocab_size=256, d_model=128, n_core=4, block_size=32)
  model = HierarchicalE2(cfg)
  # Загрузить Variant3 ядро из checkpoint:
  model.load_core_from_v3(path="checkpoint_bidir_v2.pt")
  # Обучать по уровням:
  model.set_training_phase(1)  # только glyph
  ...
  model.set_training_phase(5)  # всё разморожено
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

# ── импорты компонентов ────────────────────────────────────────────────────────
from yijing_transformer.models.variant3 import (
    Variant3Block, Variant3Config, DOMAINS,
)
from yijing_transformer.models.geometry.convergence import (
    ConvergenceBridge, MatrixGrammar, TokenAbstractor,
)
from yijing_transformer.models.geometry.routing import (
    ArchetypalInterlingua,
)
from yijing_transformer.models.geometry.nautilus import (
    NautilusHierarchy,
)


# ══════════════════════════════════════════════════════════════════════════════
# Конфигурация
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class E2Config:
    """Конфигурация HierarchicalE2."""

    # Базовые параметры
    vocab_size:  int   = 256
    d_model:     int   = 128
    block_size:  int   = 32
    dropout:     float = 0.05

    # α = −2: ядро (Variant3Block)
    n_core:            int   = 4     # число Variant3Block
    n_heads:           int   = 4
    hamming_lambda:    float = 0.15
    uncertainty_budget: float = 0.25
    ffn_mult:          int   = 4

    # α = 0: интерлингва
    n_archetypes:      int   = 64
    il_use_ternary:    bool  = True

    # α = +2: наутилус
    nautilus_warmup:   int   = 500
    nautilus_mode:     str   = "sequential"    # или "parallel"
    nautilus_chambers: Optional[List[str]] = None  # None = все 7

    # α = +4: конвергентный мост
    conv_window:       int   = 4
    conv_stride:       int   = 2
    grammar_rows:      int   = 8
    grammar_cols:      int   = 8

    # Имена уровней для freeze/unfreeze
    LEVEL_NAMES = ["glyph_level", "core_level", "method_level",
                   "theory_level", "philo_level"]
    ALPHA_MAP   = {-4: "glyph_level", -2: "core_level",
                    0: "method_level",  2: "theory_level",
                    4: "philo_level"}


# ══════════════════════════════════════════════════════════════════════════════
# Уровень α = −4: Glyph Level
# ══════════════════════════════════════════════════════════════════════════════

class GlyphLevel(nn.Module):
    """α = −4: байты → Q6-координаты → ранняя кластеризация.

    Компоненты:
      q6_proj      — проецирует d_model → 6D (мягкие Q6-координаты)
      early_abstr  — TokenAbstractor: токены → 64 кластера (гексаграммы)

    Выход:
      x_enriched   — x + scaled abstraction  (B, T, D)
      q6_coords    — (B, T, 6) — передаётся на α=+4 для ConvergenceBridge
      assignments  — (B, T, 64) — мягкие кластерные присваивания
    """

    def __init__(self, d_model: int):
        super().__init__()
        # d_model → 6D мягкие Q6 координаты (tanh ∈ (−1, +1))
        self.q6_proj = nn.Linear(d_model, 6, bias=False)
        nn.init.normal_(self.q6_proj.weight, std=0.02)

        # Ранняя кластеризация (64 = число гексаграмм)
        self.early_abstr = TokenAbstractor(d_model, n_clusters=64)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,
                                                 torch.Tensor,
                                                 torch.Tensor]:
        """
        Args:
            x: (B, T, D) — сырые эмбеддинги токенов

        Returns:
            x_enriched: (B, T, D)
            q6_coords:  (B, T, 6)
            assignments: (B, T, 64)
        """
        q6_coords = torch.tanh(self.q6_proj(x))          # (B, T, 6)
        abstract, assignments = self.early_abstr(x)      # (B,T,D), (B,T,64)
        x_enriched = x + abstract                        # residual
        return x_enriched, q6_coords, assignments


# ══════════════════════════════════════════════════════════════════════════════
# Уровень α = −2: Core Level (Variant3 blocks)
# ══════════════════════════════════════════════════════════════════════════════

class CoreLevel(nn.Module):
    """α = −2: Variant3Block × n_core.

    Каждый блок: HexProj → BianGuaAttn → TernaryGate → Interlingua → Analogy → FFN.
    Веса можно загрузить из checkpoint_bidir_v2.pt.
    """

    def __init__(self, d_model: int, n_heads: int, n_core: int,
                 hamming_lambda: float, uncertainty_budget: float,
                 ffn_mult: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            Variant3Block(
                d_model=d_model,
                n_heads=n_heads,
                ffn_mult=ffn_mult,
                hamming_lambda=hamming_lambda,
                uncertainty_budget=uncertainty_budget,
            )
            for _ in range(n_core)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def load_from_v3_checkpoint(self, path: str) -> bool:
        """Загружает веса из Variant3GPT checkpoint (только блоки).

        checkpoint_bidir_v2.pt хранит {'model_state': {...}, ...}
        model_state содержит ключи "blocks.0.xxx", "blocks.1.xxx"...
        """
        if not os.path.exists(path):
            return False
        raw = torch.load(path, map_location="cpu", weights_only=True)
        # Поддержка обоих форматов: dict с model_state или плоский state_dict
        if isinstance(raw, dict) and "model_state" in raw:
            state = raw["model_state"]
        else:
            state = raw
        block_state = {
            k.replace("blocks.", "", 1): v
            for k, v in state.items()
            if k.startswith("blocks.")
        }
        if not block_state:
            return False
        missing, _ = self.blocks.load_state_dict(block_state, strict=False)
        loaded = len(block_state) - len(missing)
        print(f"  CoreLevel: загружено {loaded} тензоров из {path}")
        return loaded > 0


# ══════════════════════════════════════════════════════════════════════════════
# Уровень α = 0: Method Level (ArchetypalInterlingua)
# ══════════════════════════════════════════════════════════════════════════════

class MethodLevel(nn.Module):
    """α = 0: синтез двух потоков (glyph + core) через 64 архетипа.

    ArchetypalInterlingua принимает:
      x             — базовый сигнал (core_output используется как identity)
      source_outputs — [glyph_output, core_output]

    Тернарное голосование: каждый источник голосует +1/0/−1
    по каждому из 64 архетипов.
    """

    def __init__(self, d_model: int, n_archetypes: int, use_ternary: bool):
        super().__init__()
        self.interlingua = ArchetypalInterlingua(
            d_model=d_model,
            n_sources=2,                  # glyph + core
            n_archetypes=n_archetypes,
            use_ternary=use_ternary,
            uncertainty_budget=0.25,
            n_heads=4,
            ternary_warmup_steps=1000,
        )

    def forward(self, x_glyph: torch.Tensor,
                x_core: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_glyph: (B, T, D) — выход GlyphLevel
            x_core:  (B, T, D) — выход CoreLevel

        Returns:
            x_method: (B, T, D) — синтез через архетипы
        """
        # identity = x_core (более информативный)
        return self.interlingua(x_core, [x_glyph, x_core])


# ══════════════════════════════════════════════════════════════════════════════
# Уровень α = +2: Theory Level (NautilusHierarchy)
# ══════════════════════════════════════════════════════════════════════════════

class TheoryLevel(nn.Module):
    """α = +2: геометрическое обогащение через 7 камер Наутилуса.

    Камеры (от малого масштаба к большому):
      CubeDiagonal   → PrivilegedAxis → DualEmbedding → D4Equivariant
      → Palace → Heisenberg → FlowerOfLife

    Каждая камера добавляет свой геометрический «язык».
    Curriculum warmup: камеры активируются постепенно.
    """

    def __init__(self, d_model: int, warmup_steps: int,
                 mode: str, chambers: Optional[List[str]]):
        super().__init__()
        self.nautilus = NautilusHierarchy(
            d_model=d_model,
            init_scale=0.01,
            warmup_steps=warmup_steps,
            mode=mode,
            enabled_chambers=chambers,
        )

    def set_step(self, step: int) -> None:
        self.nautilus.set_step(step)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return self.nautilus(x)


# ══════════════════════════════════════════════════════════════════════════════
# Уровень α = +4: Philo Level (ConvergenceBridge + MatrixGrammar)
# ══════════════════════════════════════════════════════════════════════════════

class PhiloLevel(nn.Module):
    """α = +4: философский уровень — встреча снизу (глифы) и сверху (кластеры).

    ConvergenceBridge:
      - GlyphComposer: Q6-координаты → сигилы (нижний поток)
      - TokenAbstractor: токены → 64 кластера (верхний поток)
      - ConvergenceLayer: cross-attention сигилов и кластеров
      → Встреча в середине (аналог «Атамири» Гусмана де Рохаса)

    MatrixGrammar:
      - 2D синтаксис: 8 строк (роли) × 8 столбцов (слоты) = 64 ячейки
      - Axial attention: по строкам (синтаксис) + по столбцам (семантика)
      → Обобщение 1D цепочки до 2D матрицы смыслов
    """

    def __init__(self, d_model: int, window_size: int, stride: int,
                 n_rows: int, n_cols: int):
        super().__init__()
        self.conv_bridge = ConvergenceBridge(
            d_model=d_model,
            n_clusters=64,
            window_size=window_size,
            stride=stride,
            n_compose_layers=1,
            n_heads=4,
        )
        self.matrix_gram = MatrixGrammar(
            d_model=d_model,
            n_rows=n_rows,
            n_cols=n_cols,
            n_heads=4,
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                q6_coords: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x:          (B, T, D) — выход TheoryLevel
            q6_coords:  (B, T, 6) — из GlyphLevel

        Returns:
            x_out:  (B, T, D)
            info:   dict с диагностикой
        """
        x_conv, conv_info = self.conv_bridge(x, q6_coords)
        x_gram = self.matrix_gram(x_conv)
        return self.ln_out(x_gram), conv_info


# ══════════════════════════════════════════════════════════════════════════════
# Главная модель
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalE2(nn.Module):
    """Вариант Е2: пятиуровневая иерархическая интеграция.

    Архитектура соответствует α-уровням info1:
      α=−4 → α=−2 → α=0 → α=+2 → α=+4

    Параметры:
      ~2M (d=128) до ~8M (d=256) в зависимости от конфигурации.

    Обучение снизу вверх через set_training_phase(1..5).
    """

    def __init__(self, cfg: E2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ── Общие эмбеддинги ─────────────────────────────────────────────────
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.block_size, d)
        self.drop    = nn.Dropout(cfg.dropout)
        self.ln_f    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

        # ── Уровни ───────────────────────────────────────────────────────────
        self.glyph_level = GlyphLevel(d)

        self.core_level = CoreLevel(
            d_model=d,
            n_heads=cfg.n_heads,
            n_core=cfg.n_core,
            hamming_lambda=cfg.hamming_lambda,
            uncertainty_budget=cfg.uncertainty_budget,
            ffn_mult=cfg.ffn_mult,
        )

        self.method_level = MethodLevel(
            d_model=d,
            n_archetypes=cfg.n_archetypes,
            use_ternary=cfg.il_use_ternary,
        )

        self.theory_level = TheoryLevel(
            d_model=d,
            warmup_steps=cfg.nautilus_warmup,
            mode=cfg.nautilus_mode,
            chambers=cfg.nautilus_chambers,
        )

        self.philo_level = PhiloLevel(
            d_model=d,
            window_size=cfg.conv_window,
            stride=cfg.conv_stride,
            n_rows=cfg.grammar_rows,
            n_cols=cfg.grammar_cols,
        )

        # ── Текущий обучающий шаг (для NautilusHierarchy curriculum) ────────
        self._train_step = 0
        # Текущая тренировочная фаза (1..5)
        self._phase = 5  # по умолчанию — всё разморожено

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Управление фазами обучения ────────────────────────────────────────────

    def set_training_phase(self, phase: int) -> None:
        """Устанавливает фазу обучения (1..5), размораживает нужные уровни.

        Фаза 1: только glyph_level + embeddings
        Фаза 2: + core_level
        Фаза 3: + method_level
        Фаза 4: + theory_level
        Фаза 5: + philo_level (всё разморожено)
        """
        assert 1 <= phase <= 5, f"Phase must be 1..5, got {phase}"
        self._phase = phase

        # Заморозить всё
        for p in self.parameters():
            p.requires_grad_(False)

        # Embeddings + head — всегда обучаемы
        for p in list(self.tok_emb.parameters()) + \
                 list(self.pos_emb.parameters()) + \
                 list(self.head.parameters()):
            p.requires_grad_(True)

        # Размораживаем уровни по фазе
        level_map = {
            1: [self.glyph_level],
            2: [self.glyph_level, self.core_level],
            3: [self.glyph_level, self.core_level, self.method_level],
            4: [self.glyph_level, self.core_level, self.method_level,
                self.theory_level],
            5: [self.glyph_level, self.core_level, self.method_level,
                self.theory_level, self.philo_level],
        }
        for level in level_map[phase]:
            for p in level.parameters():
                p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Фаза {phase}: {trainable:,} / {total:,} параметров обучаемы "
              f"({trainable/total*100:.1f}%)")

    def get_phase_description(self) -> str:
        descs = {
            1: "α=−4  GlyphLevel   (Q6-проекция + кластеризация)",
            2: "α=−2  CoreLevel    (Variant3Block × {})".format(self.cfg.n_core),
            3: "α= 0  MethodLevel  (ArchetypalInterlingua 64 архетипа)",
            4: "α=+2  TheoryLevel  (NautilusHierarchy 7 камер)",
            5: "α=+4  PhiloLevel   (ConvergenceBridge + MatrixGrammar)",
        }
        return descs.get(self._phase, "?")

    def load_core_from_v3(self, path: str) -> bool:
        """Загружает Variant3 ядро из checkpoint."""
        return self.core_level.load_from_v3_checkpoint(path)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        tokens:  torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            tokens:  (B, T) LongTensor
            targets: (B, T) LongTensor для обучения (или None)

        Returns:
            logits:  (B, T, vocab_size)
            loss:    скалярный loss (или None)
            info:    словарь с диагностикой по всем уровням
        """
        B, T = tokens.shape
        assert T <= self.cfg.block_size

        # ── Embeddings ────────────────────────────────────────────────────────
        pos = torch.arange(T, device=tokens.device)
        x   = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))  # (B, T, D)

        # ── α = −4: Glyph Level ───────────────────────────────────────────────
        x_glyph, q6_coords, assignments = self.glyph_level(x)

        # ── α = −2: Core Level ────────────────────────────────────────────────
        x_core = self.core_level(x_glyph)

        # ── α = 0: Method Level ───────────────────────────────────────────────
        x_method = self.method_level(x_glyph, x_core)

        # ── α = +2: Theory Level ──────────────────────────────────────────────
        self.theory_level.set_step(self._train_step)
        x_theory, naut_info = self.theory_level(x_method)

        # ── α = +4: Philo Level ───────────────────────────────────────────────
        x_philo, conv_info = self.philo_level(x_theory, q6_coords)

        # ── LM Head ───────────────────────────────────────────────────────────
        logits = self.head(self.ln_f(x_philo))          # (B, T, vocab)

        # ── Loss ──────────────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            # Auxiliary: конвергентный баланс кластеров
            conv_loss = self.philo_level.conv_bridge.get_convergence_loss(
                conv_info["assignments"]
            )

            # Auxiliary: энтропия ранних кластеров (α=−4)
            # Поощряем чёткое распределение по гексаграммам
            assign_entropy = -(assignments * (assignments + 1e-9).log()).sum(-1).mean()
            target_entropy = math.log(8.0)
            glyph_loss = (assign_entropy - target_entropy).pow(2)

            loss = lm_loss + 0.05 * conv_loss + 0.02 * glyph_loss

            if self.training:
                self._train_step += 1

        info = {
            "phase":         self._phase,
            "train_step":    self._train_step,
            "q6_coords":     q6_coords,       # (B, T, 6)
            "assignments":   assignments,      # (B, T, 64)
            "naut_info":     naut_info,
            "conv_info":     conv_info,
        }

        return logits, loss, info

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        result = {}
        for name in E2Config.LEVEL_NAMES + ["tok_emb", "pos_emb", "head"]:
            if hasattr(self, name):
                result[name] = sum(p.numel() for p in
                                   getattr(self, name).parameters())
        result["total"] = sum(p.numel() for p in self.parameters())
        return result

    def describe(self) -> str:
        counts = self.count_parameters()
        lines = [
            "HierarchicalE2 — Вариант Е2 (5 уровней α)",
            f"  vocab={self.cfg.vocab_size}  d={self.cfg.d_model}  "
            f"T={self.cfg.block_size}  core×{self.cfg.n_core}",
            "",
            f"  α=−4  glyph_level  : {counts.get('glyph_level',0):>8,} п.",
            f"  α=−2  core_level   : {counts.get('core_level',0):>8,} п.",
            f"  α= 0  method_level : {counts.get('method_level',0):>8,} п.",
            f"  α=+2  theory_level : {counts.get('theory_level',0):>8,} п.",
            f"  α=+4  philo_level  : {counts.get('philo_level',0):>8,} п.",
            f"  emb+head           : {counts.get('tok_emb',0)+counts.get('pos_emb',0)+counts.get('head',0):>8,} п.",
            f"  {'─'*30}",
            f"  ИТОГО              : {counts.get('total',0):>8,} п.",
            "",
            f"  Текущая фаза: {self._phase} — {self.get_phase_description()}",
        ]
        return "\n".join(lines)

    def embed_text(self, text: str) -> Dict:
        """Q6-эмбеддинг текста → координата + гексаграмма."""
        ids = [min(b, self.cfg.vocab_size - 1)
               for b in text.encode("utf-8")][:self.cfg.block_size]
        if not ids:
            ids = [32]
        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            _, _, info = self(tokens)
        q6 = info["q6_coords"][0].mean(0)          # (6,)
        q6_bin = (q6 > 0).int().tolist()
        hex_i  = sum(b << i for i, b in enumerate(q6_bin))
        return {
            "text":    text[:60],
            "q6":      q6_bin,
            "hex_idx": hex_i,
            "q6_soft": q6.tolist(),
        }
