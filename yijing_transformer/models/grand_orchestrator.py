"""
grand_orchestrator.py — Гранд-Оркестратор: все модели репозитория в едином ансамбле

Объединяет ВСЕ обученные и экспериментальные модели:

  ① NautilusMoME      — сумеречный язык, 6 экспертов, неологизмы
  ② PolyglotQuartet   — 4 языка (формулы, архетипы, графы, текст)
  ③ YiJingGPT         — геометрический трансформер, Q6 гиперкуб
  ④ Variant3GPT       — тернарные гейты, гексаграммные проекции
  ⑤ HierarchicalE2    — 5 уровней абстракции (глиф→философия)
  ⑥ NautilusYiJing    — MoME + геометрия И-Цзин
  ⑦ HierarchicalMoE   — 4-уровневая маршрутизация Q2→Q3→Q6

Три режима оркестровки:
  - 'blend'    — взвешенная сумма логитов всех моделей
  - 'cascade'  — каскад: каждая модель обогащает следующую
  - 'expert'   — роутер выбирает лучшую модель для каждого входа

Научная основа:
  - Mixture of Experts at model level (Jacobs et al., 1991)
  - Ensemble methods (Dietterich, 2000)
  - Cascaded inference (Viola & Jones, 2001)
"""

import math
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# ModelWrapper — унифицированная обёртка для любой модели
# ═══════════════════════════════════════════════════════════════

class ModelWrapper(nn.Module):
    """Обёртка, нормализующая интерфейс любой модели.

    Все модели в репозитории имеют разные сигнатуры forward().
    Эта обёртка приводит их к единому интерфейсу:
      forward(idx, targets) → (logits, loss, info)

    Также обеспечивает:
      - проекцию логитов к единому vocab_size
      - проекцию скрытых состояний к единому d_model
      - извлечение скрытых состояний (если модель их отдаёт)
    """

    def __init__(
        self,
        model: nn.Module,
        name: str,
        model_type: str,
        target_vocab_size: int,
        target_d_model: int,
        freeze: bool = True,
    ):
        """
        Args:
            model: исходная модель
            name: имя в оркестре ('nautilus', 'quartet', 'yijing', ...)
            model_type: тип forward ('standard', 'hmoe_component')
            target_vocab_size: общий размер словаря оркестра
            target_d_model: общая размерность d_model оркестра
            freeze: заморозить веса модели
        """
        super().__init__()
        self.model = model
        self.name = name
        self.model_type = model_type
        self.target_vocab_size = target_vocab_size
        self.target_d_model = target_d_model

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        # Определяем параметры модели
        src_vocab = self._detect_vocab_size()
        src_d = self._detect_d_model()

        # Проекция логитов если vocab_size не совпадает
        if src_vocab != target_vocab_size and src_vocab > 0:
            self.logit_proj = nn.Linear(src_vocab, target_vocab_size, bias=False)
        else:
            self.logit_proj = nn.Identity()

        # Проекция скрытых состояний если d_model не совпадает
        if src_d != target_d_model and src_d > 0:
            self.hidden_proj = nn.Linear(src_d, target_d_model, bias=False)
        else:
            self.hidden_proj = nn.Identity()

        self.src_vocab = src_vocab
        self.src_d = src_d

    def _detect_vocab_size(self) -> int:
        """Автодетекция vocab_size модели."""
        for attr in ['vocab_size', 'cfg']:
            obj = getattr(self.model, attr, None)
            if isinstance(obj, int):
                return obj
            if obj and hasattr(obj, 'vocab_size'):
                return obj.vocab_size
        # Ищем tok_emb
        if hasattr(self.model, 'tok_emb'):
            return self.model.tok_emb.weight.shape[0]
        return 0

    def _detect_d_model(self) -> int:
        """Автодетекция d_model модели."""
        for attr in ['d_model', 'cfg']:
            obj = getattr(self.model, attr, None)
            if isinstance(obj, int):
                return obj
            if obj and hasattr(obj, 'd_model'):
                return obj.d_model
        if hasattr(self.model, 'tok_emb'):
            return self.model.tok_emb.weight.shape[1]
        return 0

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Унифицированный forward.

        Returns:
            logits: (B, T, target_vocab_size)
            loss: скаляр или None
            info: диагностика
        """
        if self.model_type == 'hmoe_component':
            # HierarchicalMoEFFN — принимает hidden states, не token IDs
            # Не может использоваться напрямую для генерации логитов
            return torch.zeros(idx.shape[0], idx.shape[1], self.target_vocab_size,
                               device=idx.device), None, {'type': 'hmoe_component'}

        # Клампим input_ids если vocab модели меньше общего
        if self.src_vocab > 0 and self.src_vocab < self.target_vocab_size:
            idx = idx.clamp(0, self.src_vocab - 1)
            if targets is not None:
                targets = targets.clamp(0, self.src_vocab - 1)

        # Вызываем forward модели
        try:
            result = self.model(idx, targets)
        except TypeError:
            # Некоторые модели принимают targets как positional arg
            try:
                result = self.model(idx)
            except Exception:
                return torch.zeros(idx.shape[0], idx.shape[1], self.target_vocab_size,
                                   device=idx.device), None, {'error': 'forward_failed'}

        # Разбираем результат
        if isinstance(result, tuple):
            if len(result) == 3:
                logits, loss, info = result
            elif len(result) == 2:
                logits, loss = result
                info = {}
            else:
                logits = result[0]
                loss = None
                info = {}
        else:
            logits = result
            loss = None
            info = {}

        if info is None or not isinstance(info, dict):
            info = {}

        # Проецируем логиты к общему vocab_size
        if logits.shape[-1] != self.target_vocab_size:
            logits = self.logit_proj(logits)

        info['wrapper_name'] = self.name
        return logits, loss, info


# ═══════════════════════════════════════════════════════════════
# OrchestraRouter — маршрутизация между моделями
# ═══════════════════════════════════════════════════════════════

class OrchestraRouter(nn.Module):
    """Маршрутизатор оркестра: решает, какой модели отдать вход.

    Малая сеть анализирует входные токены и выдаёт веса
    для каждой модели. Поддерживает:
      - мягкий выбор (все модели с разными весами)
      - жёсткий выбор (top-k моделей)
    """

    def __init__(self, vocab_size: int, n_models: int, d_router: int = 64):
        super().__init__()
        self.n_models = n_models
        self.embed = nn.Embedding(vocab_size, d_router)
        self.router = nn.Sequential(
            nn.Linear(d_router, d_router * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_router * 2, n_models),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor,
                top_k: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: (B, T)
            top_k: если > 0, обнуляет веса всех кроме top-k моделей

        Returns:
            weights: (B, n_models), sum=1
            info: диагностика
        """
        x = self.embed(input_ids).mean(dim=1)  # (B, d_router)
        logits = self.router(x) / self.temperature.abs().clamp(min=0.1)

        if top_k > 0 and top_k < self.n_models:
            # Обнуляем все кроме top-k
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter(-1, topk_idx, 1.0)
            logits = logits * mask + (1 - mask) * (-1e9)

        weights = F.softmax(logits, dim=-1)

        info = {f'w_{i}': weights[:, i].mean().item() for i in range(self.n_models)}
        info['temperature'] = self.temperature.item()
        return weights, info


# ═══════════════════════════════════════════════════════════════
# SharedOrchestraMemory — общая память всех моделей
# ═══════════════════════════════════════════════════════════════

class SharedOrchestraMemory(nn.Module):
    """Общая память оркестра: все модели пишут и читают.

    Каждая модель вносит свои скрытые представления через attention
    к общей памяти. Модели обогащают друг друга:
      - NautilusMoME записывает сумеречные интуиции
      - PolyglotQuartet записывает мультиязыковые паттерны
      - YiJingGPT записывает геометрические координаты
      - и т.д.
    """

    def __init__(self, d_model: int, n_models: int, memory_size: int = 128):
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model

        # Общая память
        self.memory = nn.Parameter(torch.randn(memory_size, d_model) * 0.02)
        self.ln_mem = nn.LayerNorm(d_model)

        # Проекции записи/чтения для каждой модели
        self.write_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_models)
        ])
        self.read_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_models)
        ])
        # Гейты чтения (начинаем слабо)
        self.read_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.0)) for _ in range(n_models)
        ])

    def exchange(
        self, hiddens: List[Optional[torch.Tensor]],
    ) -> Tuple[List[Optional[torch.Tensor]], Dict]:
        """Обмен через общую память.

        Args:
            hiddens: список скрытых состояний каждой модели,
                     (B, T, d_model) или None если модель не активна

        Returns:
            deltas: добавки к скрытым состояниям каждой модели
            info: диагностика
        """
        mem = self.ln_mem(self.memory)  # (M, d)
        info = {}
        deltas = []

        # Фаза записи: все модели пишут в память
        # (в текущей реализации память статична, обновляется через градиенты)

        # Фаза чтения: каждая модель читает из памяти
        for i, h in enumerate(hiddens):
            if h is None:
                deltas.append(None)
                continue

            # Средний пулинг по времени
            query = self.write_projs[i](h.mean(dim=1))  # (B, d)
            attn = torch.softmax(
                query @ mem.T / math.sqrt(self.d_model), dim=-1
            )  # (B, M)
            read_val = attn @ mem  # (B, d)

            gate = torch.sigmoid(self.read_gates[i])
            delta = gate * self.read_projs[i](read_val).unsqueeze(1)  # (B, 1, d)
            deltas.append(delta)

            info[f'gate_{i}'] = gate.item()

        return deltas, info


# ═══════════════════════════════════════════════════════════════
# CascadeConnector — каскадное соединение моделей
# ═══════════════════════════════════════════════════════════════

class CascadeConnector(nn.Module):
    """Каскадное соединение: выход модели A обогащает вход модели B.

    Цепочка:
      Variant3 → YiJingGPT → NautilusMoME → PolyglotQuartet → E2
    Каждое звено передаёт «знание» следующему через проекцию логитов.
    """

    def __init__(self, vocab_size: int, n_models: int):
        super().__init__()
        # Гейт для каждого перехода
        self.cascade_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)) for _ in range(n_models - 1)
        ])
        # Проекция логитов в soft embedding для следующей модели
        self.soft_proj = nn.Linear(vocab_size, vocab_size, bias=False)
        nn.init.eye_(self.soft_proj.weight)

    def blend_logits(
        self, prev_logits: torch.Tensor, curr_logits: torch.Tensor, step: int,
    ) -> torch.Tensor:
        """Смешать логиты предыдущей и текущей модели."""
        if step >= len(self.cascade_gates):
            return curr_logits

        gate = torch.sigmoid(self.cascade_gates[step])
        # Soft-проекция логитов предыдущей модели
        prev_soft = self.soft_proj(F.softmax(prev_logits.detach(), dim=-1))
        return curr_logits + gate * prev_soft


# ═══════════════════════════════════════════════════════════════
# GrandOrchestrator — главный оркестратор
# ═══════════════════════════════════════════════════════════════

@dataclass
class OrchestraConfig:
    """Конфигурация оркестра."""
    mode: str = 'blend'                # 'blend' | 'cascade' | 'expert'
    vocab_size: int = 4096             # общий vocab_size
    d_model: int = 128                 # общий d_model
    memory_size: int = 128             # размер общей памяти
    expert_top_k: int = 3              # top-k моделей в режиме expert
    freeze_models: bool = True         # заморозить исходные модели
    cascade_order: List[str] = field(  # порядок в каскаде
        default_factory=lambda: [
            'variant3', 'yijing', 'nautilus_mome',
            'quartet', 'hierarchical_e2',
        ]
    )


class GrandOrchestrator(nn.Module):
    """Гранд-Оркестратор: все модели играют вместе.

    Архитектура:
      ┌─────────────────────────────────────────────────────┐
      │              ГРАНД-ОРКЕСТРАТОР                       │
      │                                                       │
      │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │
      │  │NautilusMoME│ │PolyglotQ │ │ YiJingGPT│  ...       │
      │  │(сумеречный)│ │(4 языка) │ │(геометр.)│             │
      │  └─────┬──────┘ └────┬─────┘ └────┬─────┘             │
      │        │              │            │                   │
      │        ▼              ▼            ▼                   │
      │  ┌──────────────────────────────────────────┐         │
      │  │        ОБЩАЯ ПАМЯТЬ ОРКЕСТРА              │         │
      │  │    (128 ячеек, attention read/write)      │         │
      │  └──────────────────────────────────────────┘         │
      │        │              │            │                   │
      │        ▼              ▼            ▼                   │
      │  ┌──────────────────────────────────────────┐         │
      │  │          МАРШРУТИЗАТОР / КАСКАД            │         │
      │  │     (blend / cascade / expert routing)    │         │
      │  └──────────────────────────────────────────┘         │
      │                      │                                 │
      │                      ▼                                 │
      │              ФИНАЛЬНЫЕ ЛОГИТЫ                          │
      └─────────────────────────────────────────────────────┘

    Три режима:
      'blend'   — все модели работают параллельно, логиты смешиваются
      'cascade' — каждая модель обогащает следующую в цепочке
      'expert'  — роутер выбирает top-k моделей для каждого входа
    """

    def __init__(self, config: OrchestraConfig):
        super().__init__()
        self.config = config
        self.model_names: List[str] = []
        self.wrappers = nn.ModuleDict()

        # Будут инициализированы при добавлении моделей
        self._router = None
        self._memory = None
        self._cascade = None
        self._finalized = False

    def add_model(
        self,
        name: str,
        model: nn.Module,
        model_type: str = 'standard',
        freeze: bool = True,
    ) -> 'GrandOrchestrator':
        """Добавить модель в оркестр.

        Args:
            name: уникальное имя ('nautilus_mome', 'quartet', 'yijing', ...)
            model: экземпляр модели
            model_type: 'standard' (forward(idx, targets)) или 'hmoe_component'
            freeze: заморозить ли веса

        Returns:
            self (для цепочки вызовов)
        """
        assert not self._finalized, "Оркестр уже финализирован, нельзя добавлять модели"

        wrapper = ModelWrapper(
            model=model,
            name=name,
            model_type=model_type,
            target_vocab_size=self.config.vocab_size,
            target_d_model=self.config.d_model,
            freeze=freeze if self.config.freeze_models else False,
        )
        self.wrappers[name] = wrapper
        self.model_names.append(name)
        return self

    def finalize(self) -> 'GrandOrchestrator':
        """Финализировать оркестр: создать роутер, память, каскад."""
        n = len(self.model_names)
        assert n > 0, "Нет моделей в оркестре"

        self._router = OrchestraRouter(
            self.config.vocab_size, n, d_router=64,
        )
        self._memory = SharedOrchestraMemory(
            self.config.d_model, n, memory_size=self.config.memory_size,
        )
        self._cascade = CascadeConnector(self.config.vocab_size, n)

        # Финальная нормализация
        self.ln_final = nn.LayerNorm(self.config.vocab_size)

        self._finalized = True

        # Статистика
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n  ╔══════════════════════════════════════════╗")
        print(f"  ║       ГРАНД-ОРКЕСТРАТОР ФИНАЛИЗИРОВАН     ║")
        print(f"  ╠══════════════════════════════════════════╣")
        print(f"  ║  Моделей:    {n:3d}                          ║")
        print(f"  ║  Режим:      {self.config.mode:10s}                ║")
        print(f"  ║  Параметры:  {total:>12,}              ║")
        print(f"  ║  Обучаемые:  {trainable:>12,}              ║")
        print(f"  ║  Замороженные:{frozen:>11,}              ║")
        print(f"  ╠══════════════════════════════════════════╣")
        for i, name in enumerate(self.model_names):
            w = self.wrappers[name]
            n_p = sum(p.numel() for p in w.model.parameters())
            status = "❄" if not any(p.requires_grad for p in w.model.parameters()) else "🔥"
            print(f"  ║  {status} {name:20s} {n_p:>10,} п. ║")
        print(f"  ╚══════════════════════════════════════════╝")

        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Args:
            input_ids: (B, T)
            targets: (B, T) или None
            mode: режим (если None, берётся из config)

        Returns:
            logits: (B, T, vocab_size)
            loss: скаляр или None
            info: диагностика
        """
        assert self._finalized, "Вызовите finalize() перед forward()"
        mode = mode or self.config.mode

        if mode == 'blend':
            return self._forward_blend(input_ids, targets)
        elif mode == 'cascade':
            return self._forward_cascade(input_ids, targets)
        elif mode == 'expert':
            return self._forward_expert(input_ids, targets)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

    def _forward_blend(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Blend: все модели параллельно, логиты смешиваются."""
        # Получаем веса от роутера
        weights, router_info = self._router(input_ids)  # (B, N)

        all_logits = []
        all_infos = {}

        for i, name in enumerate(self.model_names):
            logits_i, loss_i, info_i = self.wrappers[name](input_ids, targets)
            all_logits.append(logits_i)
            all_infos[name] = info_i

        # Взвешенная сумма логитов
        # weights: (B, N) → (B, N, 1, 1) для broadcasting
        w = weights.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        stacked = torch.stack(all_logits, dim=1)  # (B, N, T, V)
        logits = (w * stacked).sum(dim=1)  # (B, T, V)

        info = {'router': router_info, 'models': all_infos, 'mode': 'blend'}

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )
            info['ce_loss'] = loss.item()

        return logits, loss, info

    def _forward_cascade(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Cascade: каждая модель обогащает следующую."""
        # Определяем порядок
        order = [n for n in self.config.cascade_order if n in self.model_names]
        # Добавляем модели, которых нет в cascade_order
        for n in self.model_names:
            if n not in order:
                order.append(n)

        prev_logits = None
        info = {'cascade_order': order, 'mode': 'cascade', 'steps': {}}

        for step, name in enumerate(order):
            logits_i, loss_i, info_i = self.wrappers[name](input_ids, targets)

            if prev_logits is not None:
                logits_i = self._cascade.blend_logits(prev_logits, logits_i, step - 1)

            prev_logits = logits_i
            info['steps'][name] = {
                'logits_norm': logits_i.norm().item(),
            }

        logits = prev_logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )
            info['ce_loss'] = loss.item()

        return logits, loss, info

    def _forward_expert(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Expert: роутер выбирает top-k моделей."""
        weights, router_info = self._router(
            input_ids, top_k=self.config.expert_top_k,
        )

        all_logits = []
        active_models = []

        for i, name in enumerate(self.model_names):
            # Пропускаем модели с нулевым весом
            if weights[:, i].max() < 1e-6:
                all_logits.append(None)
                continue

            logits_i, _, info_i = self.wrappers[name](input_ids, targets)
            all_logits.append(logits_i)
            active_models.append(name)

        # Собираем только активные
        w = weights.unsqueeze(-1).unsqueeze(-1)
        logits = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.config.vocab_size,
            device=input_ids.device,
        )
        for i, lg in enumerate(all_logits):
            if lg is not None:
                logits = logits + w[:, i] * lg

        info = {
            'router': router_info,
            'active_models': active_models,
            'mode': 'expert',
            'top_k': self.config.expert_top_k,
        }

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1,
            )
            info['ce_loss'] = loss.item()

        return logits, loss, info

    # ── Утилиты ──

    def list_models(self) -> List[Dict]:
        """Список моделей с их параметрами."""
        result = []
        for name in self.model_names:
            w = self.wrappers[name]
            n_p = sum(p.numel() for p in w.model.parameters())
            n_t = sum(p.numel() for p in w.model.parameters() if p.requires_grad)
            result.append({
                'name': name,
                'type': w.model_type,
                'src_vocab': w.src_vocab,
                'src_d_model': w.src_d,
                'params': n_p,
                'trainable': n_t,
            })
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_len: int = 80,
        temperature: float = 0.8,
        top_k: int = 40,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Авторегрессивная генерация через оркестр."""
        self.eval()
        idx = input_ids.clone()

        for _ in range(max_len):
            idx_cond = idx[:, -256:]
            logits, _, _ = self(idx_cond, mode=mode)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

    def get_model(self, name: str) -> nn.Module:
        """Получить исходную модель по имени."""
        return self.wrappers[name].model


# ═══════════════════════════════════════════════════════════════
# Фабричные функции
# ═══════════════════════════════════════════════════════════════

def build_grand_orchestrator(
    models: Dict[str, nn.Module],
    mode: str = 'blend',
    vocab_size: int = 4096,
    d_model: int = 128,
    freeze: bool = True,
    expert_top_k: int = 3,
) -> GrandOrchestrator:
    """Создать оркестр из словаря моделей.

    Args:
        models: {'name': model_instance, ...}
        mode: 'blend' | 'cascade' | 'expert'
        vocab_size: общий размер словаря
        d_model: общая размерность
        freeze: заморозить исходные модели
        expert_top_k: top-k моделей для режима expert

    Returns:
        финализированный GrandOrchestrator

    Пример:
        orchestrator = build_grand_orchestrator({
            'nautilus': nautilus_model,
            'quartet': quartet_model,
            'yijing': yijing_model,
        }, mode='blend')
    """
    config = OrchestraConfig(
        mode=mode,
        vocab_size=vocab_size,
        d_model=d_model,
        freeze_models=freeze,
        expert_top_k=expert_top_k,
    )

    orch = GrandOrchestrator(config)
    for name, model in models.items():
        orch.add_model(name, model, freeze=freeze)
    orch.finalize()

    return orch
