"""
polyglot_ensemble.py — Ансамбль: NautilusMoME + PolyglotQuartet

Две модели говорят на РАЗНЫХ языках:
  - NautilusMoME → Сумеречный язык (неологизмы: «Началость», «Единологится»)
  - PolyglotQuartet → Четыре языка (формулы, архетипы, графы, текст)

Этот модуль объединяет их в единый ансамбль четырьмя способами:

  ① ДИСТИЛЛЯЦИЯ   — NautilusMoME как учитель, PolyglotQuartet учится
  ② ПЯТЫЙ МУЗЫКАНТ — NautilusMoME становится 5-м музыкантом квартета → квинтет
  ③ ОБЩАЯ ПАМЯТЬ   — обе модели делят скрытое пространство через мост
  ④ ПЕРЕКЛЮЧЕНИЕ   — роутер выбирает модель в зависимости от входа

Научная основа:
  - Ensemble Distillation (Hinton et al., 2015)
  - Mixture of Experts routing (Shazeer et al., 2017)
  - Shared representation learning (Ruder, 2017)
"""

import sys
import os
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# TwilightMusician — NautilusMoME как музыкант в Полиглот-квартете
# ═══════════════════════════════════════════════════════════════

class TwilightMusician(nn.Module):
    """Сумеречный Музыкант — обёртка NautilusMoME для встраивания в PolyglotQuartet.

    NautilusMoME говорит на сумеречном языке (неологизмы).
    Этот модуль извлекает скрытые представления NautilusMoME
    и проецирует их в пространство PolyglotQuartet.

    Квартет становится Квинтетом:
      формалист, архетипист, алгоритмист, лингвист, *сумеречник*
    """

    def __init__(self, nautilus_model: nn.Module, d_quartet: int,
                 freeze_nautilus: bool = True):
        """
        Args:
            nautilus_model: обученная NautilusMoME
            d_quartet: размерность d_model в PolyglotQuartet
            freeze_nautilus: заморозить веса NautilusMoME (по умолчанию да)
        """
        super().__init__()
        self.nautilus = nautilus_model
        d_nautilus = nautilus_model.d_model

        # Заморозка NautilusMoME — она уже обучена, знания зафиксированы
        if freeze_nautilus:
            for p in self.nautilus.parameters():
                p.requires_grad = False

        # Проекция из пространства NautilusMoME в пространство PolyglotQuartet
        self.proj_in = nn.Linear(d_quartet, d_nautilus, bias=False)
        self.proj_out = nn.Linear(d_nautilus, d_quartet, bias=False)
        nn.init.eye_(self.proj_in.weight[:min(d_quartet, d_nautilus),
                                         :min(d_quartet, d_nautilus)])
        nn.init.eye_(self.proj_out.weight[:min(d_quartet, d_nautilus),
                                          :min(d_quartet, d_nautilus)])

        # Нормализация для стабильного смешивания
        self.ln = nn.LayerNorm(d_quartet)

        # Громкость сумеречного музыканта (как volume у остальных)
        self.volume = nn.Parameter(torch.tensor(0.0))

    def forward(self, shared_state: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            shared_state: (B, T, d_quartet) — общее состояние квартета
            input_ids: (B, T) — входные токены (нужны для forward NautilusMoME)
            attn_mask: не используется (NautilusMoME имеет свою маску)

        Returns:
            contribution: (B, T, d_quartet) — вклад сумеречного музыканта
            hidden: (B, T, d_quartet) — скрытое состояние для кросс-перевода
            info: диагностика (twilight_strength, synth_activation и т.д.)
        """
        vol = torch.sigmoid(self.volume)

        if input_ids is not None:
            # Прогоняем через NautilusMoME, извлекаем скрытые состояния
            with torch.no_grad() if not any(p.requires_grad for p in self.nautilus.parameters()) else torch.enable_grad():
                nautilus_hidden = self._extract_hidden(input_ids)
            # Проецируем в пространство квартета
            hidden = self.proj_out(nautilus_hidden)
        else:
            # Если нет input_ids — проецируем shared_state в Nautilus и обратно
            nautilus_input = self.proj_in(shared_state)
            hidden = self.proj_out(nautilus_input)

        hidden = self.ln(hidden)
        contribution = vol * hidden

        info = {
            'volume': vol.item(),
            'hidden_norm': hidden.norm(dim=-1).mean().item(),
        }

        return contribution, hidden, info

    def _extract_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Извлекает скрытые состояния из NautilusMoME перед LM head.

        Проходим через:
          embedding → core_first → router → experts → bridge →
          analogy → archetype → synth → twilight → core_second
        Возвращаем состояние после core_second (перед head).
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Обрезаем до block_size NautilusMoME
        max_T = min(T, self.nautilus.block_size)
        idx = input_ids[:, :max_T]

        tok = self.nautilus.tok_emb(idx)
        pos = self.nautilus.pos_emb(torch.arange(max_T, device=device))

        if self.nautilus.use_solan and self.nautilus.solan_table is not None:
            safe_idx = idx.clamp(0, self.nautilus.solan_table.shape[0] - 1)
            glyph_vecs = self.nautilus.solan_table[safe_idx]
            glyph_proj = self.nautilus.glyph_proj(glyph_vecs)
            gate = torch.sigmoid(self.nautilus.solan_gate)
            x = tok + pos + gate * glyph_proj
        else:
            x = tok + pos

        x = self.nautilus.drop(x)

        # Core first half
        for layer in self.nautilus.core_first:
            x = layer(x)

        # Router + experts
        expert_weights, _ = self.nautilus.router(x)
        expert_outputs = []
        expert_names = self.nautilus.EXPERT_NAMES[:self.nautilus.n_experts]
        for i, name in enumerate(expert_names):
            mask = expert_weights[:, :, i].sum() > 0
            if mask:
                exp_out = self.nautilus.experts[name](x)
            else:
                exp_out = torch.zeros_like(x)
            expert_outputs.append(exp_out)

        x = self.nautilus.bridge(x, expert_outputs, expert_weights)
        x, _ = self.nautilus.analogy(x, expert_outputs, expert_weights, expert_names)
        x, _ = self.nautilus.archetype_layer(x, expert_weights)

        # SYNTH + Twilight
        if self.nautilus.enable_synth:
            ew_safe = expert_weights.clamp(min=1e-8)
            token_entropy = -(ew_safe * ew_safe.log()).sum(dim=-1)
            synth_activation = torch.sigmoid(
                self.nautilus.synth_sharpness * (token_entropy - self.nautilus.synth_center)
            )
            synth_out = self.nautilus.synth_expert(x)
            x = x + synth_out * synth_activation.unsqueeze(-1) * self.nautilus.synth_gate

        x, _ = self.nautilus.twilight(x, {})

        # Core second half
        for layer in self.nautilus.core_second:
            x = layer(x)

        # Дополняем нулями, если T > max_T
        if T > max_T:
            pad = torch.zeros(B, T - max_T, x.size(-1), device=device)
            x = torch.cat([x, pad], dim=1)

        return x  # (B, T, d_nautilus)


# ═══════════════════════════════════════════════════════════════
# SharedMemoryBridge — общая память между двумя моделями
# ═══════════════════════════════════════════════════════════════

class SharedMemoryBridge(nn.Module):
    """Мост общей памяти между NautilusMoME и PolyglotQuartet.

    Обе модели пишут в общую память и читают из неё.
    Это позволяет сумеречным неологизмам обогащать формальные
    представления, а формулам — направлять сумеречные интуиции.

    Архитектура:
      hidden_A → write_gate_A → ОБЩАЯ ПАМЯТЬ → read_gate_B → hidden_B
      hidden_B → write_gate_B → ОБЩАЯ ПАМЯТЬ → read_gate_A → hidden_A
    """

    def __init__(self, d_nautilus: int, d_quartet: int, memory_size: int = 64):
        """
        Args:
            d_nautilus: размерность NautilusMoME
            d_quartet: размерность PolyglotQuartet
            memory_size: число ячеек памяти
        """
        super().__init__()
        self.memory_size = memory_size
        d_mem = max(d_nautilus, d_quartet)

        # Общая память — обучаемый буфер
        self.memory = nn.Parameter(torch.randn(memory_size, d_mem) * 0.02)

        # Проекции записи/чтения для NautilusMoME
        self.write_nautilus = nn.Linear(d_nautilus, d_mem, bias=False)
        self.read_nautilus = nn.Linear(d_mem, d_nautilus, bias=False)
        self.gate_nautilus = nn.Parameter(torch.tensor(-2.0))  # начинаем слабо

        # Проекции записи/чтения для PolyglotQuartet
        self.write_quartet = nn.Linear(d_quartet, d_mem, bias=False)
        self.read_quartet = nn.Linear(d_mem, d_quartet, bias=False)
        self.gate_quartet = nn.Parameter(torch.tensor(-2.0))

        self.ln_mem = nn.LayerNorm(d_mem)

    def forward(
        self,
        h_nautilus: Optional[torch.Tensor] = None,
        h_quartet: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """Обмен информацией через общую память.

        Args:
            h_nautilus: (B, T, d_nautilus) — скрытое состояние NautilusMoME
            h_quartet: (B, T, d_quartet) — скрытое состояние PolyglotQuartet

        Returns:
            delta_nautilus: добавка к h_nautilus (или None)
            delta_quartet: добавка к h_quartet (или None)
            info: диагностика
        """
        info = {}
        mem = self.ln_mem(self.memory)  # (M, d_mem)

        # NautilusMoME → память → PolyglotQuartet
        delta_quartet = None
        if h_nautilus is not None and h_quartet is not None:
            # Запись NautilusMoME в память
            query_n = self.write_nautilus(h_nautilus.mean(dim=1))  # (B, d_mem)
            attn_n = torch.softmax(query_n @ mem.T / math.sqrt(mem.size(-1)), dim=-1)  # (B, M)
            read_n = attn_n @ mem  # (B, d_mem)

            gate_q = torch.sigmoid(self.gate_quartet)
            delta_quartet = gate_q * self.read_quartet(read_n).unsqueeze(1)  # (B, 1, d_quartet)
            info['gate_quartet'] = gate_q.item()

        # PolyglotQuartet → память → NautilusMoME
        delta_nautilus = None
        if h_quartet is not None and h_nautilus is not None:
            query_q = self.write_quartet(h_quartet.mean(dim=1))
            attn_q = torch.softmax(query_q @ mem.T / math.sqrt(mem.size(-1)), dim=-1)
            read_q = attn_q @ mem

            gate_n = torch.sigmoid(self.gate_nautilus)
            delta_nautilus = gate_n * self.read_nautilus(read_q).unsqueeze(1)
            info['gate_nautilus'] = gate_n.item()

        return delta_nautilus, delta_quartet, info


# ═══════════════════════════════════════════════════════════════
# EnsembleRouter — выбор модели в зависимости от входа
# ═══════════════════════════════════════════════════════════════

class EnsembleRouter(nn.Module):
    """Роутер ансамбля: решает, кому отдать вход.

    Для формул и кода → NautilusMoME (эксперты MATH/CODE).
    Для мифов и философии → PolyglotQuartet (архетипист/лингвист).
    Для сложных случаев → обе модели + смешивание.

    Входной текст анализируется малой сетью, которая выдаёт
    веса [w_nautilus, w_quartet] ∈ [0, 1].
    """

    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # [nautilus_weight, quartet_weight]
        )
        # Температура softmax (обучаемая)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: (B, T)

        Returns:
            weights: (B, 2) — [w_nautilus, w_quartet], sum=1
            info: диагностика
        """
        x = self.embed(input_ids)  # (B, T, d)
        x = x.mean(dim=1)  # (B, d) — средний пулинг
        logits = self.router(x) / self.temperature.abs().clamp(min=0.1)
        weights = F.softmax(logits, dim=-1)

        info = {
            'nautilus_weight': weights[:, 0].mean().item(),
            'quartet_weight': weights[:, 1].mean().item(),
            'temperature': self.temperature.item(),
        }

        return weights, info


# ═══════════════════════════════════════════════════════════════
# PolyglotEnsemble — главный ансамбль
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    """Конфигурация ансамбля."""
    mode: str = 'quintet'           # 'distill' | 'quintet' | 'memory' | 'router'
    freeze_nautilus: bool = True    # заморозить NautilusMoME
    memory_size: int = 64           # размер общей памяти
    blend_weight: float = 0.3      # вес NautilusMoME в смеси логитов


class PolyglotEnsemble(nn.Module):
    """Ансамбль NautilusMoME + PolyglotQuartet.

    Четыре режима работы:

    ① 'distill' — NautilusMoME учитель, PolyglotQuartet ученик
       Логиты учителя → soft targets → обучение ученика.
       После обучения NautilusMoME не нужна для инференса.

    ② 'quintet' — Квинтет (5 музыкантов)
       NautilusMoME как 5-й музыкант «Сумеречник».
       Его вклад координируется через общий Conductor.

    ③ 'memory' — Общая память
       Обе модели работают параллельно, обмениваясь
       через SharedMemoryBridge. Каждая обогащает другую.

    ④ 'router' — Переключение
       Роутер выбирает модель для каждого входа.
       Для формул → NautilusMoME, для мифов → Quartet.
    """

    def __init__(
        self,
        quartet: nn.Module,
        nautilus: nn.Module,
        config: EnsembleConfig,
    ):
        super().__init__()
        self.quartet = quartet
        self.nautilus = nautilus
        self.config = config

        d_quartet = quartet.cfg.d_model
        d_nautilus = nautilus.d_model
        vocab_size = quartet.cfg.vocab_size

        # Модули для разных режимов
        # Quintet: сумеречный музыкант
        self.twilight_musician = TwilightMusician(
            nautilus, d_quartet, freeze_nautilus=config.freeze_nautilus,
        )

        # Memory: мост общей памяти
        self.memory_bridge = SharedMemoryBridge(
            d_nautilus, d_quartet, memory_size=config.memory_size,
        )

        # Router: переключатель
        self.router = EnsembleRouter(vocab_size)

        # Проекция логитов NautilusMoME → vocab_size квартета (если словари разные)
        self.logit_proj = nn.Linear(
            nautilus.vocab_size, vocab_size, bias=False
        ) if nautilus.vocab_size != vocab_size else nn.Identity()

        # Вес смешивания
        self.blend_weight = nn.Parameter(torch.tensor(config.blend_weight))

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Args:
            input_ids: (B, T)
            targets: (B, T) — метки
            mode: режим (если None, берётся из config)

        Returns:
            logits, loss, info
        """
        mode = mode or self.config.mode

        if mode == 'distill':
            return self._forward_distill(input_ids, targets)
        elif mode == 'quintet':
            return self._forward_quintet(input_ids, targets)
        elif mode == 'memory':
            return self._forward_memory(input_ids, targets)
        elif mode == 'router':
            return self._forward_router(input_ids, targets)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

    def _forward_distill(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Дистилляция: учитель (Nautilus) → ученик (Quartet)."""
        # Ученик
        student_logits, student_loss, student_info = self.quartet(input_ids, targets)

        # Учитель (без градиентов)
        with torch.no_grad():
            teacher_logits, _, teacher_info = self.nautilus(input_ids)
            teacher_logits = self.logit_proj(teacher_logits)

        info = {'student': student_info, 'teacher_twilight': teacher_info.get('twilight', {})}

        if targets is not None:
            T_temp = 3.0
            soft_student = F.log_softmax(student_logits / T_temp, dim=-1)
            soft_teacher = F.softmax(teacher_logits / T_temp, dim=-1)
            kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T_temp ** 2)

            loss = 0.5 * student_loss + 0.5 * kl_loss
            info['kl_loss'] = kl_loss.item()
            info['student_loss'] = student_loss.item()
            return student_logits, loss, info

        return student_logits, student_loss, info

    def _forward_quintet(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Квинтет: NautilusMoME как 5-й музыкант."""
        B, T = input_ids.shape
        device = input_ids.device

        # Получаем shared_state от квартета (до оркестровки)
        tok = self.quartet.tok_emb(input_ids)
        pos = self.quartet.pos_emb[:, :T, :]
        shared_state = self.quartet.emb_drop(tok + pos)

        attn_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )

        # Получаем вклад сумеречного музыканта
        twilight_contrib, twilight_hidden, twilight_info = self.twilight_musician(
            shared_state, input_ids=input_ids,
        )

        # Раунды репетиций (как в квартете, но с 5-м участником)
        info = {'rounds': [], 'twilight': twilight_info}
        all_hiddens = []

        for r, conductor in enumerate(self.quartet.conductors):
            contributions = []
            hiddens = []

            for name in self.quartet._musician_order:
                musician = self.quartet.musicians[name]
                contrib, hidden = musician(shared_state, attn_mask=attn_mask)
                contributions.append(contrib)
                hiddens.append(hidden)

            # Добавляем 5-го музыканта
            contributions.append(twilight_contrib)

            # Дирижёр берёт только первых 4 (его blend рассчитан на 4)
            orchestrated, blend_weights = conductor(shared_state, contributions[:4])

            # Сумеречный вклад добавляется отдельно с обучаемым весом
            twilight_w = torch.sigmoid(self.blend_weight)
            orchestrated = orchestrated + twilight_w * twilight_contrib

            gate = torch.sigmoid(self.quartet.rehearsal_gates[r])
            shared_state = shared_state + gate * orchestrated

            if r == len(self.quartet.conductors) - 1:
                all_hiddens = hiddens

        # Output
        logits = self.quartet.head(self.quartet.ln_out(shared_state))

        # Loss
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )
            rosetta_loss, rosetta_info = self.quartet.rosetta(all_hiddens)
            info['rosetta'] = rosetta_info

            linguist_logits = self.quartet.musicians['linguist'].get_spec_logits(all_hiddens[3])
            spec_loss = F.cross_entropy(
                linguist_logits.view(-1, linguist_logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )

            loss = ce_loss + self.quartet.cfg.rosetta_weight * rosetta_loss + \
                   self.quartet.cfg.spec_weight * spec_loss
            info['ce_loss'] = ce_loss.item()
            info['rosetta_loss'] = rosetta_loss.item()
            info['spec_loss'] = spec_loss.item()

        return logits, loss, info

    def _forward_memory(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Общая память: обе модели обмениваются через мост."""
        # NautilusMoME forward
        with torch.no_grad() if self.config.freeze_nautilus else torch.enable_grad():
            nautilus_logits, _, nautilus_info = self.nautilus(input_ids)

        # Извлекаем скрытые состояния NautilusMoME
        h_nautilus = self.twilight_musician._extract_hidden(input_ids)

        # PolyglotQuartet forward (до head)
        B, T = input_ids.shape
        tok = self.quartet.tok_emb(input_ids)
        pos = self.quartet.pos_emb[:, :T, :]
        shared_state = self.quartet.emb_drop(tok + pos)
        attn_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1
        )

        for r, conductor in enumerate(self.quartet.conductors):
            contribs = []
            for name in self.quartet._musician_order:
                c, _ = self.quartet.musicians[name](shared_state, attn_mask=attn_mask)
                contribs.append(c)
            orch, _ = conductor(shared_state, contribs)
            gate = torch.sigmoid(self.quartet.rehearsal_gates[r])
            shared_state = shared_state + gate * orch

        # Обмен через мост памяти
        delta_n, delta_q, mem_info = self.memory_bridge(h_nautilus, shared_state)

        if delta_q is not None:
            shared_state = shared_state + delta_q

        # Output
        logits = self.quartet.head(self.quartet.ln_out(shared_state))

        info = {'memory': mem_info, 'nautilus_twilight': nautilus_info.get('twilight', {})}

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )
            info['ce_loss'] = loss.item()

        return logits, loss, info

    def _forward_router(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Переключение: роутер выбирает модель."""
        # Веса роутера
        weights, router_info = self.router(input_ids)  # (B, 2)

        # Обе модели
        q_logits, q_loss, q_info = self.quartet(input_ids, targets)

        with torch.no_grad() if self.config.freeze_nautilus else torch.enable_grad():
            n_logits, n_loss, n_info = self.nautilus(input_ids)
        n_logits = self.logit_proj(n_logits)

        # Взвешенная сумма логитов
        w_n = weights[:, 0].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        w_q = weights[:, 1].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        logits = w_n * n_logits + w_q * q_logits

        info = {'router': router_info, 'quartet': q_info}

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )
            info['ce_loss'] = loss.item()

        return logits, loss, info


# ═══════════════════════════════════════════════════════════════
# Фабричные функции
# ═══════════════════════════════════════════════════════════════

def build_ensemble(
    quartet: nn.Module,
    nautilus: nn.Module,
    mode: str = 'quintet',
    freeze_nautilus: bool = True,
) -> PolyglotEnsemble:
    """Создать ансамбль из двух моделей.

    Args:
        quartet: обученный PolyglotQuartet
        nautilus: обученная NautilusMoME
        mode: 'distill' | 'quintet' | 'memory' | 'router'
        freeze_nautilus: заморозить NautilusMoME

    Returns:
        PolyglotEnsemble
    """
    config = EnsembleConfig(
        mode=mode,
        freeze_nautilus=freeze_nautilus,
    )
    ensemble = PolyglotEnsemble(quartet, nautilus, config)

    params_total = sum(p.numel() for p in ensemble.parameters())
    params_train = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
    print(f"  Ансамбль ({mode}):")
    print(f"    Параметры: {params_total:,} всего, {params_train:,} обучаемых")
    print(f"    NautilusMoME: {'заморожена' if freeze_nautilus else 'обучается'}")
    print(f"    PolyglotQuartet: обучается")

    return ensemble


def load_nautilus_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cpu',
) -> nn.Module:
    """Загрузить NautilusMoME из чекпоинта.

    Args:
        checkpoint_path: путь к .pt файлу
        device: устройство

    Returns:
        NautilusMoME (eval mode)
    """
    # Добавляем путь к скриптам для импорта NautilusMoME
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'yijing_transformer', 'scripts',
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Извлекаем конфигурацию из чекпоинта
    if 'model_config' in ckpt:
        cfg = ckpt['model_config']
    else:
        cfg = {
            'vocab_size': 4096, 'd_model': 192, 'n_layers': 4,
            'n_heads': 6, 'block_size': 512, 'd_expert': 128,
        }

    from train_nautilus_mome import NautilusMoME

    model = NautilusMoME(
        vocab_size=cfg.get('vocab_size', 4096),
        d_model=cfg.get('d_model', 192),
        n_layers=cfg.get('n_layers', 4),
        n_heads=cfg.get('n_heads', 6),
        block_size=cfg.get('block_size', 512),
        d_expert=cfg.get('d_expert', 128),
    )

    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    print(f"  NautilusMoME загружена из {checkpoint_path}")
    print(f"    d_model={model.d_model}, vocab={model.vocab_size}")

    return model
