"""
Абриале-слой: событийно-управляемые изотропные N-местные связи.

По мотивам А.И. Пацкина (РосНИИ ИИ):
  «Применение техники управления событиями для анализа текста в системе Абриаль»
  Диалог 2003, https://dialogue-conf.org/media/2673/packin.pdf

Три ключевых идеи Пацкина, реализованных в нейросетевой форме:

1. ИЗОТРОПНАЯ СЕТЬ (Isotropic Network):
   Все связи двусторонние, нет привилегированного направления.
   → Симметричный attention: A_ij = A_ji, все токены видят друг друга
   одинаково в обоих направлениях.

2. СОБЫТИЯ = ВРЕМЕННЫЕ СВЯЗИ (Events = Temporary Links):
   События существуют только в пределах транзакции (commit/rollback).
   → Soft-computed N-местные связи с гейтовым commit:
   удачные связи фиксируются (residual), неудачные — откатываются.

3. ИНВЕРСИЯ УПРАВЛЕНИЯ (Event-Driven Activation):
   Не правила ищут данные, а события активизируют правила.
   → Каждый токен порождает «событие» — вектор активации,
   который динамически находит подходящие «правила» (паттерны)
   через attention к банку паттернов. Порядок проверки условий
   определяется динамически, как в Абриале.

4. N-МЕСТНЫЕ СВЯЗИ (N-ary Relations):
   Ri(O1, O2, ..., On) — не бинарные, а N-арные.
   → Гиперрёберный attention: вместо пар (qi, kj) вычисляем
   совместную функцию от N объектов через тензорное произведение
   в проектированном пространстве.

Архитектура AbrialeLayer:
    Input: x (B, T, d_model)
      ↓
    [EventGenerator] — каждый токен → событие (активация)
      ↓
    [IsotropicMatcher] — симметричный attention для N-ary связей
      ↓ (дерево хитов)
    [RuleBank] — банк паттернов, активизируемых событиями
      ↓
    [TransactionGate] — commit/rollback решение для каждой связи
      ↓
    Output: x + scale * committed_events
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IsotropicAttention(nn.Module):
    """Изотропный attention: симметричный, двусторонний.

    В отличие от стандартного attention (Q≠K → асимметрия),
    здесь используется единая проекция: Q=K=proj(x).
    Результат: A_ij = A_ji — связи проходимы в обе стороны
    одинаково быстро, как в модели памяти Абриаля.

    Дополнительно поддерживает N-арные связи через
    гиперрёберное расширение: вместо пар (i,j) рассматриваются
    тройки (i,j,k) с разделённым тензорным произведением.

    Args:
        d_model: размерность модели
        n_heads: число голов
        arity: арность связей (2 = бинарные, 3 = тернарные)
        dropout: dropout rate
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 arity: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.arity = arity
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        # Единая проекция (изотропность: Q=K)
        self.node_proj = nn.Linear(d_model, d_model, bias=False)
        # Отдельная проекция для значений
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        # Выходная проекция
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Для N-арных связей (arity > 2): разделённое тензорное произведение
        # Вместо полного тензора R^{d^N} используем rank-1 факторизацию
        if arity > 2:
            self.arity_weights = nn.ParameterList([
                nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
                for _ in range(arity - 2)  # доп. оси сверх бинарного
            ])

    def _symmetric_scores(self, nodes: Tensor) -> Tensor:
        """Вычисляет симметричную матрицу сходства.

        Ключевое свойство: A_ij = A_ji (изотропность).

        Args:
            nodes: (B, H, T, head_dim) — проекции узлов

        Returns:
            scores: (B, H, T, T) — симметричная матрица
        """
        # Стандартное QK^T уже симметрично при Q=K
        scores = torch.matmul(nodes, nodes.transpose(-2, -1))
        # Явная симметризация для численной стабильности
        scores = (scores + scores.transpose(-2, -1)) / 2.0
        return scores / math.sqrt(self.head_dim)

    def _nary_scores(self, nodes: Tensor) -> Tensor:
        """Вычисляет N-арные баллы через тензорную факторизацию.

        Для тернарных связей (arity=3): s_ijk = (n_i · w) * (n_j · n_k)
        — третий узел k «наблюдает» за парой (i,j), модулируя связь.

        Это аналог условий в образце правила Абриаля:
        условие1 + условие2 + условие3 → хит.

        Args:
            nodes: (B, H, T, head_dim) — проекции узлов

        Returns:
            scores: (B, H, T, T) — эффективные 2D баллы
                    (N-арность свёрнута по доп. осям)
        """
        # Бинарная часть: (B, H, T, T)
        binary_scores = self._symmetric_scores(nodes)

        # Дополнительные оси арности
        for w in self.arity_weights:
            # w: (H, head_dim) — весовой вектор для дополнительной оси
            # Проекция каждого узла на этот вектор: (B, H, T)
            axis_scores = torch.einsum('bhtd,hd->bht', nodes, w)
            # Модуляция: каждая ось добавляет «голос» третьего участника
            # Среднее по всем возможным третьим → (B, H, 1, T) * (B, H, T, 1)
            modulation = axis_scores.unsqueeze(-1) + axis_scores.unsqueeze(-2)
            # Sigma → мягкое gate [0, 1]
            modulation = torch.sigmoid(modulation / math.sqrt(self.head_dim))
            binary_scores = binary_scores * modulation

        return binary_scores

    def forward(self, x: Tensor) -> tuple:
        """Изотропный N-арный attention.

        Args:
            x: (B, T, d_model) — входные токены

        Returns:
            output: (B, T, d_model) — обработанные токены
            attention_weights: (B, H, T, T) — веса (для диагностики)
        """
        B, T, D = x.shape
        H = self.n_heads

        # Единая проекция для Q=K (изотропность)
        nodes = self.node_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)
        values = self.value_proj(x).reshape(B, T, H, self.head_dim).transpose(1, 2)

        # Вычисляем баллы
        if self.arity > 2:
            scores = self._nary_scores(nodes)
        else:
            scores = self._symmetric_scores(nodes)

        # Softmax → веса
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Применяем к значениям
        out = torch.matmul(attn_weights, values)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        return out, attn_weights


class EventGenerator(nn.Module):
    """Генератор событий: каждый токен порождает «событие».

    В системе Абриаля: событие = временная связь, существующая
    в пределах транзакции. Событие активизирует правила.

    Здесь: событие = проекция токена в «пространство событий»,
    которое затем ищет подходящие паттерны в банке правил.

    Типы событий (как у Пацкина):
    - Стартовое событие E0(T) — активизация всего текста
    - Вторичные события E_i(C_i) — активизация отдельных знаков
    - Следственные события — порождённые правилами

    Args:
        d_model: размерность модели
        d_event: размерность пространства событий
        n_event_types: число типов событий (аналог типов связей)
    """

    def __init__(self, d_model: int, d_event: int = 64,
                 n_event_types: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_event = d_event
        self.n_event_types = n_event_types

        # Проекция токена → событие
        self.event_proj = nn.Linear(d_model, d_event, bias=False)

        # Классификатор типа события (мягкий)
        self.type_classifier = nn.Linear(d_model, n_event_types, bias=False)

        # Порождение «следственных» событий через self-attention
        # (событие E_i порождает E_j через правило)
        self.causal_proj = nn.Linear(d_event, d_event, bias=False)

    def forward(self, x: Tensor) -> tuple:
        """Генерирует события из токенов.

        Args:
            x: (B, T, d_model)

        Returns:
            events: (B, T, d_event) — векторы событий
            event_types: (B, T, n_event_types) — мягкая классификация
        """
        events = self.event_proj(x)  # (B, T, d_event)
        event_types = F.softmax(
            self.type_classifier(x), dim=-1
        )  # (B, T, n_event_types)

        return events, event_types


class RuleBank(nn.Module):
    """Банк правил (паттернов), активизируемых событиями.

    В Абриале: правило = образец (условия) + альтернативы (действия).
    Ключевая инверсия: не правила применяются к данным,
    а события АКТИВИЗИРУЮТ правила.

    Здесь: банк из N_rules learnable паттернов.
    Каждый паттерн = вектор в пространстве событий.
    Событие активизирует ближайшие паттерны через attention.

    Хит = совпадение события с паттерном → запуск «действия»
    (трансформации токена).

    Дерево хитов: одно событие может активизировать
    несколько правил (soft top-k).

    Args:
        d_event: размерность пространства событий
        d_model: размерность модели
        n_rules: число правил в банке
        n_hits: макс. число хитов (top-k активизаций)
        n_alternatives: число альтернатив (действий) на правило
    """

    def __init__(self, d_event: int = 64, d_model: int = 512,
                 n_rules: int = 64, n_hits: int = 4,
                 n_alternatives: int = 2):
        super().__init__()
        self.d_event = d_event
        self.d_model = d_model
        self.n_rules = n_rules
        self.n_hits = min(n_hits, n_rules)
        self.n_alternatives = n_alternatives

        # Паттерны (образцы): каждый паттерн = вектор в d_event
        # 64 правила = 64 гексаграммы — не совпадение!
        self.patterns = nn.Parameter(
            torch.randn(n_rules, d_event) * 0.02
        )

        # Действия (alternatives): каждое правило → n_alternatives действий
        # Каждое действие = линейная трансформация d_model → d_model
        self.actions = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_alternatives)
        ])

        # Селектор альтернатив: какое действие выбрать для данного хита
        self.alt_selector = nn.Linear(d_event, n_alternatives, bias=False)

        # Температура для soft matching
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=10.0)

    def forward(self, events: Tensor, x: Tensor) -> tuple:
        """Активизирует правила событиями.

        Args:
            events: (B, T, d_event) — векторы событий
            x: (B, T, d_model) — токены для трансформации

        Returns:
            actions_out: (B, T, d_model) — результат действий
            hit_weights: (B, T, n_rules) — веса хитов (для диагностики)
        """
        B, T, _ = events.shape

        # 1. Сходство событий с паттернами: (B, T, n_rules)
        events_norm = F.normalize(events, dim=-1)
        patterns_norm = F.normalize(self.patterns, dim=-1)
        similarity = torch.matmul(events_norm, patterns_norm.T)

        # 2. Soft top-k: только n_hits лучших паттернов активизируются
        # (аналог «дерева хитов» Абриаля)
        topk_vals, topk_idx = similarity.topk(self.n_hits, dim=-1)

        # Мягкие веса хитов
        hit_weights_sparse = F.softmax(topk_vals / self.temperature, dim=-1)

        # 3. Выбор альтернативы для каждого хита
        # Берём паттерны хитов: (B, T, n_hits, d_event)
        hit_patterns = self.patterns[topk_idx]  # (B, T, n_hits, d_event)

        # Средневзвешенный паттерн для селекции альтернативы
        weighted_pattern = (
            hit_patterns * hit_weights_sparse.unsqueeze(-1)
        ).sum(dim=2)  # (B, T, d_event)

        # Мягкий выбор альтернативы: (B, T, n_alternatives)
        alt_logits = self.alt_selector(weighted_pattern)
        alt_weights = F.softmax(alt_logits / self.temperature, dim=-1)

        # 4. Применяем действия (взвешенная сумма альтернатив)
        actions_out = torch.zeros_like(x)
        for i, action in enumerate(self.actions):
            actions_out = actions_out + alt_weights[:, :, i:i+1] * action(x)

        # 5. Масштабируем по силе хитов (сильный хит → сильное действие)
        hit_strength = topk_vals.max(dim=-1).values.unsqueeze(-1)  # (B, T, 1)
        actions_out = actions_out * torch.sigmoid(hit_strength)

        # Полные веса хитов для диагностики (sparse → dense)
        hit_weights = torch.zeros(B, T, self.n_rules, device=events.device, dtype=events.dtype)
        hit_weights.scatter_(2, topk_idx, hit_weights_sparse)

        return actions_out, hit_weights


class TransactionGate(nn.Module):
    """Транзакционный гейт: commit/rollback для событий.

    В Абриале: транзакция — неделимая операция.
    Удача → фиксация всех изменений (commit).
    Неудача → полная отмена (rollback/бэктрекинг).

    Здесь: мягкий commit/rollback.
    Гейт g ∈ [0, 1] определяет долю «зафиксированных» событий.
    g ≈ 1 → commit (событие полезно, фиксируем).
    g ≈ 0 → rollback (событие бесполезно, откатываем).

    Решение принимается на основе:
    - Согласованности события с контекстом (attention)
    - Силы хитов в банке правил
    - Типа события

    Args:
        d_model: размерность модели
        d_event: размерность событий
    """

    def __init__(self, d_model: int, d_event: int = 64):
        super().__init__()

        # Решение commit/rollback на основе:
        # [original_token, action_result, event_type_entropy]
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, actions: Tensor,
                event_types: Tensor) -> Tensor:
        """Commit/rollback решение.

        Args:
            x: (B, T, d_model) — исходные токены
            actions: (B, T, d_model) — результат действий правил
            event_types: (B, T, n_types) — мягкая классификация событий

        Returns:
            committed: (B, T, d_model) — зафиксированные изменения
        """
        # Энтропия типов событий: высокая энтропия → неуверенность → rollback
        entropy = -(event_types * (event_types + 1e-10).log()).sum(dim=-1, keepdim=True)
        # Нормализуем энтропию в [0, 1]
        max_entropy = math.log(event_types.shape[-1])
        entropy_norm = entropy / max_entropy  # (B, T, 1)

        # Конкатенируем входы для gate
        gate_input = torch.cat([x, actions, entropy_norm], dim=-1)

        # Commit gate
        g = self.gate(gate_input)  # (B, T, 1)

        # Мягкий commit/rollback
        committed = g * actions  # g≈0 → rollback, g≈1 → commit

        return committed


class AbrialeLayer(nn.Module):
    """Полный Абриале-слой: событийно-управляемые изотропные N-местные связи.

    Архитектура (по мотивам Пацкина, Диалог 2003):

        Input: x (B, T, d_model)
          ↓
        [EventGenerator] — токены → события (временные связи)
          ↓
        [IsotropicAttention] — симметричный N-арный attention
          ↓ (обогащённые токены)
        [RuleBank] — события активизируют правила → действия
          ↓
        [TransactionGate] — commit/rollback для каждого действия
          ↓
        Output: x + scale * committed

    Ключевые свойства:
    1. Изотропность: A_ij = A_ji (все связи двусторонние)
    2. N-арность: связи могут быть тернарными (arity=3)
    3. Событийность: токены → события → хиты → действия
    4. Транзакционность: мягкий commit/rollback

    Args:
        d_model: размерность модели
        d_event: размерность пространства событий
        n_heads: число голов изотропного attention
        arity: арность связей (2 или 3)
        n_rules: число правил в банке (64 = гексаграммы)
        n_hits: макс. число хитов на событие
        n_alternatives: число альтернатив (действий) на правило
        n_event_types: число типов событий
        dropout: dropout rate
    """

    def __init__(self, d_model: int, d_event: int = 64,
                 n_heads: int = 4, arity: int = 2,
                 n_rules: int = 64, n_hits: int = 4,
                 n_alternatives: int = 2,
                 n_event_types: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_hits = n_hits

        # 1. Изотропный attention (симметричные N-арные связи)
        self.isotropic_attn = IsotropicAttention(
            d_model=d_model,
            n_heads=n_heads,
            arity=arity,
            dropout=dropout,
        )

        # 2. Генератор событий
        self.event_gen = EventGenerator(
            d_model=d_model,
            d_event=d_event,
            n_event_types=n_event_types,
        )

        # 3. Банк правил
        self.rule_bank = RuleBank(
            d_event=d_event,
            d_model=d_model,
            n_rules=n_rules,
            n_hits=n_hits,
            n_alternatives=n_alternatives,
        )

        # 4. Транзакционный гейт
        self.transaction = TransactionGate(
            d_model=d_model,
            d_event=d_event,
        )

        # Layer norms
        self.norm_pre = nn.LayerNorm(d_model)
        self.norm_post = nn.LayerNorm(d_model)

        # Learnable scale (начинаем с малого вклада)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: Tensor) -> tuple:
        """Forward pass Абриале-слоя.

        Args:
            x: (B, T, d_model)

        Returns:
            enriched: (B, T, d_model) — обогащённые токены
            info: dict с диагностикой
        """
        residual = x
        x_norm = self.norm_pre(x)

        # 1. Изотропный attention: симметричные N-арные связи
        iso_out, attn_weights = self.isotropic_attn(x_norm)

        # 2. Генерируем события из обогащённых токенов
        enriched = x_norm + iso_out
        events, event_types = self.event_gen(enriched)

        # 3. События активизируют правила → действия
        actions, hit_weights = self.rule_bank(events, enriched)

        # 4. Транзакционный commit/rollback
        committed = self.transaction(enriched, actions, event_types)

        # 5. Residual с learnable scale
        output = residual + self.scale * self.norm_post(committed)

        # Диагностика
        info = {
            'attn_symmetry': self._measure_symmetry(attn_weights),
            'commit_rate': (committed.abs().mean() / (actions.abs().mean() + 1e-10)).item(),
            'hit_entropy': self._hit_entropy(hit_weights).item(),
            'event_type_entropy': self._type_entropy(event_types).item(),
            'scale': self.scale.item(),
            'hit_weights': hit_weights,  # для aux loss
        }

        return output, info

    def _measure_symmetry(self, attn: Tensor) -> float:
        """Измеряет степень симметрии attention матрицы.

        Идеал: 1.0 (полная симметрия = изотропность).
        """
        diff = (attn - attn.transpose(-2, -1)).abs().mean()
        return 1.0 - diff.item()

    def _hit_entropy(self, hit_weights: Tensor) -> Tensor:
        """Энтропия распределения хитов по правилам.

        Высокая энтропия = правила используются равномерно.
        Низкая = только несколько правил активны (специализация).
        """
        avg = hit_weights.mean(dim=(0, 1))  # (n_rules,)
        avg = avg / (avg.sum() + 1e-10)
        return -(avg * (avg + 1e-10).log()).sum()

    def _type_entropy(self, event_types: Tensor) -> Tensor:
        """Средняя энтропия классификации событий."""
        entropy = -(event_types * (event_types + 1e-10).log()).sum(dim=-1)
        return entropy.mean()

    def get_auxiliary_loss(self, hit_weights: Tensor) -> Tensor:
        """Вспомогательный loss для балансировки использования правил.

        Аналог: в Абриале все правила должны иметь шанс сработать.
        Если часть правил никогда не активизируется — потеря ресурсов.

        Args:
            hit_weights: (B, T, n_rules)

        Returns:
            loss: скаляр — штраф за неравномерное использование
        """
        avg_usage = hit_weights.mean(dim=(0, 1))  # (n_rules,)
        # target = n_hits/n_rules (т.к. scatter помещает mass только на n_hits правил)
        target = self.n_hits / hit_weights.shape[-1]
        return ((avg_usage - target) ** 2).sum() * hit_weights.shape[-1]
