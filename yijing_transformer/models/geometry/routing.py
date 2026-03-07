"""
Routing, gating, curriculum, logging.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedPathSelector(nn.Module):
    """Гейтовый механизм выбора между геометрическим и стандартным путём."""
    def __init__(self, d_model: int, init_bias: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)
        self._last_gate_mean = 0.5
        self._last_gate_std = 0.0

    def forward(self, x_standard, x_geometric):
        combined = (x_standard + x_geometric) * 0.5
        gate = torch.sigmoid(self.gate_proj(combined))
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_gate_std = gate.std().item()
        return gate * x_geometric + (1 - gate) * x_standard

    def get_gate_stats(self):
        return {
            'gate_mean': self._last_gate_mean,
            'gate_std': self._last_gate_std,
            'prefers_geometry': self._last_gate_mean > 0.5,
        }


class AdaptiveGatedPathSelector(nn.Module):
    """Расширенный гейт с контентно-зависимым поведением и температурой."""
    def __init__(self, d_model: int, n_heads: int = 1, init_bias: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        self._last_gate_mean = 0.5
        self._last_gate_std = 0.0
        self._last_gate_entropy = 0.0

    def forward(self, x_standard, x_geometric):
        combined = (x_standard + x_geometric) * 0.5
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        raw_gate = self.gate_proj(combined) / temp
        gate = torch.sigmoid(raw_gate)
        if self.n_heads > 1:
            gate = gate.mean(dim=-1, keepdim=True)
        with torch.no_grad():
            self._last_gate_mean = gate.mean().item()
            self._last_gate_std = gate.std().item()
            g = gate.mean().clamp(1e-6, 1 - 1e-6)
            self._last_gate_entropy = -(g * g.log() + (1-g) * (1-g).log()).item()
        return gate * x_geometric + (1 - gate) * x_standard

    def get_gate_stats(self):
        return {
            'gate_mean': self._last_gate_mean,
            'gate_std': self._last_gate_std,
            'gate_entropy': self._last_gate_entropy,
            'temperature': self.log_temperature.exp().item(),
            'prefers_geometry': self._last_gate_mean > 0.5,
        }


class TaskAwareRouter(nn.Module):
    """Task-aware routing: определяет тип входного паттерна."""
    def __init__(self, d_model: int, n_strategies: int = 4):
        super().__init__()
        self.n_strategies = n_strategies
        self.strategy_proj = nn.Linear(d_model, n_strategies, bias=True)
        self.strategy_biases = nn.Parameter(torch.linspace(-1.0, 1.0, n_strategies))
        self._last_strategy_probs = None

    def forward(self, x):
        x_mean = x.mean(dim=1)
        logits = self.strategy_proj(x_mean)
        probs = F.softmax(logits, dim=-1)
        gate_bias = (probs * self.strategy_biases.unsqueeze(0)).sum(dim=-1)
        with torch.no_grad():
            self._last_strategy_probs = probs.mean(dim=0).tolist()
        return gate_bias.unsqueeze(1).unsqueeze(2)

    def get_strategy_stats(self):
        if self._last_strategy_probs is not None:
            return {f'strategy_{i}': p for i, p in enumerate(self._last_strategy_probs)}
        return {}


class GateLogger:
    """Собирает и агрегирует статистику гейтов."""
    def __init__(self):
        self.history = []

    def log_step(self, step: int, layers):
        entry = {'step': step, 'gates': {}}
        for i, layer in enumerate(layers):
            if hasattr(layer, 'path_gate') and layer.path_gate is not None:
                stats = layer.path_gate.get_gate_stats()
                entry['gates'][f'layer_{i}'] = stats
        self.history.append(entry)
        return entry

    def summary(self):
        if not self.history:
            return {}
        last = self.history[-1]
        geo_layers = sum(1 for g in last['gates'].values() if g['prefers_geometry'])
        std_layers = sum(1 for g in last['gates'].values() if not g['prefers_geometry'])
        return {
            'step': last['step'],
            'layers_prefer_geometry': geo_layers,
            'layers_prefer_standard': std_layers,
            'gates': last['gates'],
        }

    def get_trajectory(self):
        trajectories = {}
        for entry in self.history:
            for layer_name, stats in entry['gates'].items():
                if layer_name not in trajectories:
                    trajectories[layer_name] = {'steps': [], 'means': []}
                trajectories[layer_name]['steps'].append(entry['step'])
                trajectories[layer_name]['means'].append(stats['gate_mean'])
        return trajectories


class GeometryCurriculumScheduler:
    """Планировщик curriculum learning для геометрических компонентов."""
    def __init__(self, strategy: str = 'linear', total_steps: int = 10000,
                 warmup_fraction: float = 0.3, target_strength: float = 0.1,
                 n_step_stages: int = 4):
        self.strategy = strategy
        self.total_steps = total_steps
        self.warmup_fraction = warmup_fraction
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.target_strength = target_strength
        self.n_step_stages = n_step_stages

    def get_strength(self, step: int) -> float:
        progress = min(step / max(self.total_steps, 1), 1.0)
        if self.strategy == 'linear':
            return self.target_strength * progress
        elif self.strategy == 'warmup_hold':
            if step < self.warmup_steps:
                return self.target_strength * step / self.warmup_steps
            return self.target_strength
        elif self.strategy == 'cosine':
            return self.target_strength * 0.5 * (1 - math.cos(math.pi * progress))
        elif self.strategy == 'step':
            stage = min(int(progress * self.n_step_stages), self.n_step_stages)
            return self.target_strength * stage / self.n_step_stages
        elif self.strategy == 'geometric_first':
            if step < self.warmup_steps:
                return 1.0
            decay_progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            return self.target_strength + (1.0 - self.target_strength) * (1 - decay_progress)
        else:
            return self.target_strength

    def get_gate_bias(self, step: int) -> float:
        if self.strategy == 'geometric_first':
            progress = min(step / max(self.total_steps, 1), 1.0)
            return 2.0 * (1 - progress)
        return 0.0


class GeometricSourceRouter(nn.Module):
    """
    MoE-style router для геометрических источников.

    Вместо фиксированных коэффициентов (0.05, 0.1) при каждом attention bias,
    обучаемый маршрутизатор выбирает top-k из N источников per-token.

    Решает проблему интерференции: модель сама решает, какие источники
    использовать для каждого входа, вместо усреднения всех.

    Sources: triangular, mobius, cube_diagonal, heisenberg, palace,
             privileged_axis, flower_gat, hex_pattern, etc.
    """
    def __init__(self, d_model: int, n_sources: int, top_k: int = 2,
                 temperature: float = 1.0):
        super().__init__()
        self.n_sources = n_sources
        self.top_k = min(top_k, n_sources)
        self.router_proj = nn.Linear(d_model, n_sources, bias=True)
        nn.init.zeros_(self.router_proj.weight)
        nn.init.zeros_(self.router_proj.bias)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        # Per-source learnable scale (initialized to small value)
        self.source_scales = nn.Parameter(torch.full((n_sources,), 0.05))
        # Statistics
        self._last_routing_probs = None
        self._aux_loss = 0.0

    def forward(self, x, source_outputs):
        """
        Args:
            x: input tensor (B, T, C) — used for routing decision
            source_outputs: list of N tensors, each (B, T, C) — outputs from sources

        Returns:
            mixed: (B, T, C) — weighted combination of selected sources
        """
        B, T, C = x.shape
        n_actual = len(source_outputs)
        assert n_actual == self.n_sources, \
            f"Expected {self.n_sources} sources, got {n_actual}"

        # Routing logits: (B, T, N)
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        logits = self.router_proj(x) / temp  # (B, T, N)

        # Top-k selection
        if self.top_k < self.n_sources:
            top_vals, top_idx = logits.topk(self.top_k, dim=-1)  # (B, T, k)
            mask = torch.zeros_like(logits).scatter(-1, top_idx, 1.0)
            logits = logits.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(logits, dim=-1)  # (B, T, N)

        # Load balancing auxiliary loss (importance vs frequency)
        with torch.no_grad():
            self._last_routing_probs = weights.mean(dim=(0, 1)).tolist()
        # Encourage uniform routing across sources
        avg_probs = weights.mean(dim=(0, 1))  # (N,)
        self._aux_loss = self.n_sources * (avg_probs * avg_probs).sum()

        # Apply per-source scale and mix
        scales = self.source_scales.abs()  # ensure positive
        stacked = torch.stack(source_outputs, dim=-1)  # (B, T, C, N)
        scaled_weights = weights * scales.unsqueeze(0).unsqueeze(0)  # (B, T, N)
        mixed = (stacked * scaled_weights.unsqueeze(2)).sum(dim=-1)  # (B, T, C)

        return mixed

    def get_routing_stats(self):
        if self._last_routing_probs is not None:
            stats = {f'source_{i}': p for i, p in enumerate(self._last_routing_probs)}
            stats['aux_loss'] = self._aux_loss.item() if isinstance(self._aux_loss, torch.Tensor) else self._aux_loss
            stats['scales'] = self.source_scales.abs().detach().tolist()
            return stats
        return {}


class GeometricSourceMixer(nn.Module):
    """
    Lightweight mixer: каждый источник получает обучаемый residual gate.

    Проще чем GeometricSourceRouter — нет top-k, просто sigmoid gate per source.
    Gate=0 при инициализации → модель начинает как vanilla transformer.
    Каждый источник включается/выключается независимо.
    """
    def __init__(self, d_model: int, n_sources: int):
        super().__init__()
        self.n_sources = n_sources
        # Per-source gate: initialized to -2 → sigmoid(-2) ≈ 0.12 (nearly off)
        self.gate_logits = nn.Parameter(torch.full((n_sources,), -2.0))
        # Per-source learnable scale
        self.source_scales = nn.Parameter(torch.full((n_sources,), 0.05))
        self._last_gates = None

    def forward(self, x, source_outputs):
        """
        Args:
            x: base output (B, T, C) — identity path
            source_outputs: list of N tensors (B, T, C) — additive enrichments

        Returns:
            x + sum(gate_i * scale_i * source_i for i in sources)
        """
        gates = torch.sigmoid(self.gate_logits)  # (N,)
        scales = self.source_scales  # (N,)

        with torch.no_grad():
            self._last_gates = gates.tolist()

        result = x
        for i, src in enumerate(source_outputs):
            result = result + gates[i] * scales[i] * src
        return result

    def get_gate_stats(self):
        if self._last_gates is not None:
            return {
                'gates': self._last_gates,
                'scales': self.source_scales.detach().tolist(),
                'active_sources': sum(1 for g in self._last_gates if g > 0.5),
            }
        return {}


class SequentialSourceCurriculum:
    """
    Curriculum: включает геометрические источники один за другим.

    Этап 0: только vanilla (все gates заблокированы)
    Этап 1: разблокируется источник 1 (Belyaev)
    Этап 2: разблокируется источник 2 (Fomyuk)
    ...
    Этап N: все источники разблокированы

    Работает путём модификации gate_logits у GeometricSourceMixer.
    """
    def __init__(self, n_sources: int, steps_per_source: int = 500,
                 warmup_steps: int = 200, source_order=None):
        self.n_sources = n_sources
        self.steps_per_source = steps_per_source
        self.warmup_steps = warmup_steps
        # Default order: strongest first (Belyaev=0, then others)
        self.source_order = source_order or list(range(n_sources))

    def get_active_mask(self, step: int):
        """Returns bool list of which sources are active at given step."""
        if step < self.warmup_steps:
            return [False] * self.n_sources
        effective_step = step - self.warmup_steps
        n_active = min(
            effective_step // self.steps_per_source + 1,
            self.n_sources
        )
        mask = [False] * self.n_sources
        for i in range(n_active):
            mask[self.source_order[i]] = True
        return mask

    def apply_to_mixer(self, mixer, step: int):
        """Clamp inactive source gates to near-zero."""
        mask = self.get_active_mask(step)
        with torch.no_grad():
            for i, active in enumerate(mask):
                if not active:
                    mixer.gate_logits.data[i] = -10.0  # sigmoid(-10) ≈ 0


class PairwiseBridge(nn.Module):
    """Мост между двумя источниками: cross-attention медиатор.

    Вместо линейного смешивания (w₁·src₁ + w₂·src₂) мост находит
    «общий язык» через cross-attention:
    - src₁ спрашивает у src₂: "что из тебя совместимо со мной?"
    - src₂ спрашивает у src₁: "что из тебя совместимо со мной?"
    - Gated merge → согласованное представление

    Семантическая аналогия с RAG: мост «извлекает» из каждого источника
    только совместимую информацию, подавляя конфликтующие сигналы.
    """
    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        # Cross-attention: src_a ← src_b
        self.cross_ab = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: src_b ← src_a
        self.cross_ba = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        # Gated merge
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)
        # Learnable scale (starts small → non-coercion)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, src_a: torch.Tensor, src_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_a: (B, T, C) — выход источника A
            src_b: (B, T, C) — выход источника B
        Returns:
            mediated: (B, T, C) — согласованное представление
        """
        a = self.norm_a(src_a)
        b = self.norm_b(src_b)
        # A спрашивает B: "что ты можешь предложить мне?"
        a_from_b, _ = self.cross_ab(query=a, key=b, value=b)
        # B спрашивает A: "что ты можешь предложить мне?"
        b_from_a, _ = self.cross_ba(query=b, key=a, value=a)
        # Residuals
        a_enriched = src_a + a_from_b
        b_enriched = src_b + b_from_a
        # Gated merge: выбираем пропорцию каждого
        g = self.gate(torch.cat([a_enriched, b_enriched], dim=-1))
        merged = g * a_enriched + (1 - g) * b_enriched
        return self.out_norm(merged) * self.scale


class BridgeOfModules(nn.Module):
    """Мост Модулей: иерархическая медиация между геометрическими источниками.

    Вместо прямого смешивания (линейная комбинация как в MoE),
    источники соединяются попарно через PairwiseBridge, который
    выполняет cross-attention медиацию. Это:

    1. Устраняет деструктивную интерференцию — несовместимые сигналы
       подавляются cross-attention (низкие веса для конфликтующих).
    2. Строит иерархию согласования:
       - Уровень 1: попарные мосты (Bridge₁₂, Bridge₃₄, Bridge₅₆)
       - Уровень 2: мета-мосты между результатами уровня 1
       - Уровень 3: финальный мост → единый выход
    3. Семантический RAG: каждый мост «извлекает» из пары источников
       только совместимую информацию, как RAG извлекает релевантные
       документы, но здесь — в пространстве представлений.

    Принцип «золотой середины»: мост не выбирает один из источников,
    а находит согласованное представление, в котором оба вносят вклад
    пропорционально их совместимости.

    При нечётном числе источников: последний источник проецируется
    напрямую на следующий уровень.

    Args:
        d_model: размерность модели
        n_sources: число геометрических источников
        n_heads: число голов cross-attention в каждом мосте
        dropout: dropout в cross-attention
    """
    def __init__(self, d_model: int, n_sources: int, n_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources

        # Строим дерево мостов снизу вверх
        self.bridge_tree = nn.ModuleList()
        self.tree_structure = []  # [(level, pairs)] для forward

        remaining = n_sources
        level = 0
        while remaining > 1:
            n_pairs = remaining // 2
            has_odd = (remaining % 2 == 1)
            bridges = nn.ModuleList([
                PairwiseBridge(d_model, n_heads, dropout)
                for _ in range(n_pairs)
            ])
            self.bridge_tree.append(bridges)
            self.tree_structure.append((n_pairs, has_odd))
            remaining = n_pairs + (1 if has_odd else 0)
            level += 1

        # Глобальный gate: возможность полностью отключить мост
        self.global_gate = nn.Parameter(torch.tensor(0.0))
        # Per-source input projections (alignment перед медиацией)
        self.source_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_sources)
        ])

        # Статистика
        self._last_global_gate = 0.5

    def forward(self, x: torch.Tensor,
                source_outputs: list) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — базовый выход (identity path)
            source_outputs: list of N tensors (B, T, C)
        Returns:
            enriched: (B, T, C) — x + bridge_contribution
        """
        assert len(source_outputs) == self.n_sources

        # Нормализуем каждый источник
        current_level = [
            self.source_norms[i](src)
            for i, src in enumerate(source_outputs)
        ]

        # Иерархическая медиация
        for bridges, (n_pairs, has_odd) in zip(
            self.bridge_tree, self.tree_structure
        ):
            next_level = []
            for p in range(n_pairs):
                a = current_level[2 * p]
                b = current_level[2 * p + 1]
                mediated = bridges[p](a, b)
                next_level.append(mediated)
            # Нечётный элемент проходит напрямую
            if has_odd:
                next_level.append(current_level[-1])
            current_level = next_level

        # Финальный результат — единственный оставшийся элемент
        bridge_output = current_level[0]

        # Глобальный gate: residual connection
        gate = torch.sigmoid(self.global_gate)
        with torch.no_grad():
            self._last_global_gate = gate.item()

        return x + gate * bridge_output

    def get_bridge_stats(self) -> dict:
        stats = {
            'global_gate': self._last_global_gate,
            'n_levels': len(self.bridge_tree),
            'bridge_scales': [],
        }
        for level_idx, bridges in enumerate(self.bridge_tree):
            for bridge_idx, bridge in enumerate(bridges):
                stats['bridge_scales'].append({
                    'level': level_idx,
                    'pair': bridge_idx,
                    'scale': bridge.scale.item(),
                })
        return stats


class DynamicCurriculumController:
    """Dynamic curriculum: адаптирует стратегию на основе гейтов."""
    def __init__(self, base_strength: float = 0.1, adapt_rate: float = 0.01):
        self.base_strength = base_strength
        self.adapt_rate = adapt_rate
        self.current_strength = base_strength
        self.history = []

    def update(self, avg_gate_value: float, gate_std: float):
        if gate_std > 0.2:
            pass
        elif avg_gate_value > 0.6:
            self.current_strength = min(
                self.current_strength + self.adapt_rate,
                self.base_strength * 3.0
            )
        elif avg_gate_value < 0.35:
            self.current_strength = max(
                self.current_strength - self.adapt_rate,
                self.base_strength * 0.1
            )
        self.history.append({
            'strength': self.current_strength,
            'avg_gate': avg_gate_value,
            'gate_std': gate_std,
        })
        return self.current_strength
