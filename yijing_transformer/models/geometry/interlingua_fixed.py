"""
interlingua_fixed.py — Исправленная ArchetypalInterlingua.

Проблема оригинала (из REPORT-v60-v61):
    Один общий trit_proj → все модули дают одинаковые триты →
    64 архетипа неразличимы → PPL = vanilla.

Исправление:
    1. Per-source trit_proj: каждый источник имеет свой проектор
    2. Diversity loss: штраф за одинаковое голосование
    3. Gumbel-Softmax вместо STE (см. quantizer_fixed.py)
    4. Cosine annealing температуры: start_temp → end_temp за warmup_steps
"""

import math
import itertools
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArchetypalInterlinguaFixed(nn.Module):
    """
    Исправленная версия Archetypal Interlingua.

    Ключевое отличие от оригинала: N отдельных проекторов
    вместо одного общего. Каждый источник голосует
    по-своему за каждый из 64 архетипов.
    """

    def __init__(
        self,
        d_model: int,
        n_sources: int,
        n_archetypes: int = 64,
        diversity_weight: float = 0.01,
        warmup_steps: int = 3000,
        start_temp: float = 1.0,
        end_temp: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources
        self.n_archetypes = n_archetypes
        self.diversity_weight = diversity_weight
        self.warmup_steps = warmup_steps
        self.start_temp = start_temp
        self.end_temp = end_temp

        # ── ИСПРАВЛЕНИЕ 1: Per-source проекторы ──────────────────
        # БЫЛО: self.trit_proj = nn.Linear(d_model, n_archetypes)
        # СТАЛО: по одному для каждого источника
        self.trit_projs = nn.ModuleList([
            nn.Linear(d_model, n_archetypes, bias=True)
            for _ in range(n_sources)
        ])

        # Инициализация: небольшой разброс чтобы источники стартовали по-разному
        for i, proj in enumerate(self.trit_projs):
            nn.init.normal_(proj.weight, mean=0, std=0.02 * (i + 1) / n_sources)
            nn.init.zeros_(proj.bias)

        # ── Архетипические якоря (фиксированные Q6-вершины) ───────
        all_anchors = self._generate_q6_vertices()  # (64, 6)
        anchors = all_anchors[:n_archetypes]
        self.register_buffer('archetype_anchors', anchors)

        # Проектор: n_archetypes-мерный консенсус → d_model через Q6-якоря
        self.anchor_to_hidden = nn.Linear(6, d_model, bias=False)

        # Readout attention
        self.readout_attn = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True, dropout=0.0
        )

        # Gate: сколько брать из интерлингвы vs прямого пути
        self.gate = nn.Parameter(torch.zeros(1))

        # ── ИСПРАВЛЕНИЕ 2: Temperature annealing (Gumbel) ─────────
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    def _generate_q6_vertices(self) -> torch.Tensor:
        """Все 64 вершины гиперкуба {-1,+1}^6."""
        vertices = list(itertools.product([-1., 1.], repeat=6))
        return torch.tensor(vertices, dtype=torch.float32)  # (64, 6)

    def _get_temperature(self) -> float:
        """Косинусное расписание температуры: start_temp → end_temp за warmup_steps."""
        step = self._step.item()
        progress = min(step / max(self.warmup_steps, 1), 1.0)
        factor = (1 - math.cos(math.pi * progress)) / 2
        return self.start_temp + (self.end_temp - self.start_temp) * factor

    def _compute_trits(
        self, source_outputs: List[torch.Tensor]
    ) -> tuple:
        """
        Вычисляет тернарные веса для каждого источника.

        Returns:
            trit_stack: (n_sources, B, n_archetypes)
            diversity_loss: scalar
        """
        T = self._get_temperature()
        trits = []

        for i, h in enumerate(source_outputs):
            # h: (B, T_seq, d_model) → усреднить по seq
            h_mean = h.mean(dim=1)  # (B, d_model)

            logits = self.trit_projs[i](h_mean)  # (B, n_archetypes)

            if self.training:
                # Gumbel-Softmax: 3 категории {-1, 0, +1}
                logits_3 = torch.stack([
                    -logits,
                    torch.zeros_like(logits),
                    logits,
                ], dim=-1)  # (B, n_archetypes, 3)

                soft = F.gumbel_softmax(logits_3, tau=T, hard=False, dim=-1)
                trit = soft[..., 2] - soft[..., 0]  # (B, n_archetypes)
            else:
                trit = torch.sign(logits)

            trits.append(trit)

        trit_stack = torch.stack(trits, dim=0)  # (n_sources, B, n_archetypes)

        # ── ИСПРАВЛЕНИЕ 3: Diversity loss ─────────────────────────
        n = self.n_sources
        if n > 1:
            flat = trit_stack.mean(dim=1)  # (n_sources, n_archetypes)
            norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            normed = flat / norms
            sim_matrix = normed @ normed.T  # (n_sources, n_sources)
            eye = torch.eye(n, device=sim_matrix.device)
            off_diag_mean = (sim_matrix * (1 - eye)).sum() / (n * (n - 1))
            diversity_loss = off_diag_mean
        else:
            diversity_loss = torch.tensor(0., device=trit_stack.device)

        if self.training:
            self._step += 1

        return trit_stack, diversity_loss

    def forward(
        self,
        source_outputs: List[torch.Tensor],
        core_hidden: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            source_outputs: list of (B, T_seq, d_model) — выходы каждого источника
            core_hidden: (B, T_seq, d_model) — прямой путь (для gate)

        Returns:
            output: (B, T_seq, d_model)
            aux_loss: diversity_loss * diversity_weight
        """
        B = source_outputs[0].shape[0]
        T_seq = source_outputs[0].shape[1]

        trit_stack, diversity_loss = self._compute_trits(source_outputs)
        # trit_stack: (n_sources, B, n_archetypes)

        # Консенсус по источникам
        consensus = trit_stack.mean(dim=0)  # (B, n_archetypes)

        # Преобразовать консенсус через архетипные якоря
        soft_q6 = F.softmax(consensus, dim=-1)          # (B, n_archetypes)
        archetype_pos = soft_q6 @ self.archetype_anchors  # (B, 6)
        interlingua_signal = self.anchor_to_hidden(archetype_pos)  # (B, d_model)
        interlingua_signal = interlingua_signal.unsqueeze(1).expand(B, T_seq, -1)

        # Gate: смешать с прямым путём
        if core_hidden is not None:
            alpha = torch.sigmoid(self.gate)
            output = alpha * core_hidden + (1 - alpha) * interlingua_signal
        else:
            output = interlingua_signal

        aux_loss = self.diversity_weight * diversity_loss
        return output, aux_loss

    def diagnostics(self, source_outputs: List[torch.Tensor]) -> dict:
        """
        Диагностика: проверить дифференциацию тритов.
        Вызывать каждые 200 шагов для мониторинга.
        Здоровые значения: diversity_loss > 0.1, gate между 0.3 и 0.7.
        """
        with torch.no_grad():
            trit_stack, div_loss = self._compute_trits(source_outputs)

        d = {
            'step':           self._step.item(),
            'temperature':    self._get_temperature(),
            'gate':           torch.sigmoid(self.gate).item(),
            'diversity_loss': div_loss.item() if hasattr(div_loss, 'item') else float(div_loss),
        }
        for i in range(self.n_sources):
            trits = trit_stack[i]  # (B, n_archetypes)
            d[f'src{i}_pos']  = (trits > 0.5).float().mean().item()
            d[f'src{i}_neg']  = (trits < -0.5).float().mean().item()
            d[f'src{i}_zero'] = (trits.abs() < 0.5).float().mean().item()
        return d

    # Alias for backward compat
    get_diagnostics = diagnostics


# ─── Быстрый тест модуля ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Тест ArchetypalInterlinguaFixed...")

    model = ArchetypalInterlinguaFixed(
        d_model=64, n_sources=3, n_archetypes=16,
        diversity_weight=0.1, warmup_steps=50,
        start_temp=1.0, end_temp=0.05,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(30):
        sources = [torch.randn(2, 8, 64) * (i + 1) * 0.3 for i in range(3)]
        core = torch.randn(2, 8, 64)
        out, aux = model(sources, core)
        loss = out.pow(2).mean() + aux
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    sources = [torch.randn(2, 8, 64) * (i + 1) for i in range(3)]
    d = model.diagnostics(sources)

    print(f"  Температура:      {d['temperature']:.3f}")
    print(f"  Gate:             {d['gate']:.3f}")
    print(f"  Diversity loss:   {d['diversity_loss']:.4f}")

    pos_vals = [d[f'src{i}_pos'] for i in range(3)]
    spread = max(pos_vals) - min(pos_vals)
    print(f"  Разброс trit_pos: {spread:.3f}")

    ok = spread > 0.03
    print(f"\n{'✅ ТЕСТ ПРОЙДЕН' if ok else '⚠️  Малый разброс — увеличьте diversity_weight'}")
