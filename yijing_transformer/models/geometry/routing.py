"""
Routing, gating, curriculum, logging.
"""

import math
import threading
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
        self._local = threading.local()

    def forward(self, x_standard, x_geometric):
        combined = (x_standard + x_geometric) * 0.5
        gate = torch.sigmoid(self.gate_proj(combined))
        with torch.no_grad():
            self._local.gate_mean = gate.mean().item()
            self._local.gate_std = gate.std().item()
        return gate * x_geometric + (1 - gate) * x_standard

    def get_gate_stats(self):
        mean = getattr(self._local, 'gate_mean', 0.5)
        std  = getattr(self._local, 'gate_std',  0.0)
        return {
            'gate_mean': mean,
            'gate_std': std,
            'prefers_geometry': mean > 0.5,
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
        self._local = threading.local()

    def forward(self, x_standard, x_geometric):
        combined = (x_standard + x_geometric) * 0.5
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        raw_gate = self.gate_proj(combined) / temp
        gate = torch.sigmoid(raw_gate)
        if self.n_heads > 1:
            gate = gate.mean(dim=-1, keepdim=True)
        with torch.no_grad():
            self._local.gate_mean = gate.mean().item()
            self._local.gate_std = gate.std().item()
            g = gate.mean().clamp(1e-6, 1 - 1e-6)
            self._local.gate_entropy = -(g * g.log() + (1-g) * (1-g).log()).item()
        return gate * x_geometric + (1 - gate) * x_standard

    def get_gate_stats(self):
        mean    = getattr(self._local, 'gate_mean',    0.5)
        std     = getattr(self._local, 'gate_std',     0.0)
        entropy = getattr(self._local, 'gate_entropy', 0.0)
        return {
            'gate_mean': mean,
            'gate_std': std,
            'gate_entropy': entropy,
            'temperature': self.log_temperature.exp().item(),
            'prefers_geometry': mean > 0.5,
        }


class TaskAwareRouter(nn.Module):
    """Task-aware routing: определяет тип входного паттерна."""
    def __init__(self, d_model: int, n_strategies: int = 4):
        super().__init__()
        self.n_strategies = n_strategies
        self.strategy_proj = nn.Linear(d_model, n_strategies, bias=True)
        self.strategy_biases = nn.Parameter(torch.linspace(-1.0, 1.0, n_strategies))
        self._local = threading.local()

    def forward(self, x):
        x_mean = x.mean(dim=1)
        logits = self.strategy_proj(x_mean)
        probs = F.softmax(logits, dim=-1)
        gate_bias = (probs * self.strategy_biases.unsqueeze(0)).sum(dim=-1)
        with torch.no_grad():
            self._local.strategy_probs = probs.mean(dim=0).tolist()
        return gate_bias.unsqueeze(1).unsqueeze(2)

    def get_strategy_stats(self):
        probs = getattr(self._local, 'strategy_probs', None)
        if probs is not None:
            return {f'strategy_{i}': p for i, p in enumerate(probs)}
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
            decay_progress = min((step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1), 1.0)
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


class LightweightBridge(nn.Module):
    """Облегчённый мост: bilinear compatibility вместо full cross-attention.

    Вместо O(d²) cross-attention использует:
    1. Bilinear compatibility score: score = src_a^T · W · src_b (bottleneck)
    2. Gated blend на основе score
    3. ~10x меньше параметров чем PairwiseBridge

    Идея: если cross-attention — это «подробный разговор» между источниками,
    то bilinear bridge — это «быстрый взгляд» на совместимость.
    Для многих задач достаточно знать «совместимы ли эти два сигнала»,
    не нужно детально разбирать каждую позицию.

    Args:
        d_model: размерность модели
        bottleneck: размер bottleneck для bilinear (d_model // 4)
    """
    def __init__(self, d_model: int, bottleneck: int = 0):
        super().__init__()
        bn = bottleneck or max(d_model // 4, 8)
        # Bottleneck projections для bilinear scoring
        self.proj_a = nn.Linear(d_model, bn, bias=False)
        self.proj_b = nn.Linear(d_model, bn, bias=False)
        # Compatibility → gate
        self.compat_proj = nn.Linear(bn, 1, bias=True)
        nn.init.zeros_(self.compat_proj.bias)
        # Norm
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, src_a: torch.Tensor, src_b: torch.Tensor) -> torch.Tensor:
        a = self.norm_a(src_a)
        b = self.norm_b(src_b)
        # Bilinear compatibility: element-wise product in bottleneck space
        za = self.proj_a(a)  # (B, T, bn)
        zb = self.proj_b(b)  # (B, T, bn)
        compat = torch.sigmoid(self.compat_proj(za * zb))  # (B, T, 1)
        # Gated blend: высокая совместимость → больше от обоих, низкая → suppress
        merged = compat * (src_a + src_b) * 0.5 + (1 - compat) * src_a
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
        bridge_mode: 'full' (cross-attention) или 'lightweight' (bilinear)
    """
    def __init__(self, d_model: int, n_sources: int, n_heads: int = 2,
                 dropout: float = 0.1, bridge_mode: str = 'full'):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources
        self.bridge_mode = bridge_mode

        # Строим дерево мостов снизу вверх
        self.bridge_tree = nn.ModuleList()
        self.tree_structure = []  # [(level, pairs)] для forward

        def make_bridge():
            if bridge_mode == 'lightweight':
                return LightweightBridge(d_model)
            return PairwiseBridge(d_model, n_heads, dropout)

        remaining = n_sources
        level = 0
        while remaining > 1:
            n_pairs = remaining // 2
            has_odd = (remaining % 2 == 1)
            bridges = nn.ModuleList([
                make_bridge() for _ in range(n_pairs)
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


class AbrialeBridgeMediator(nn.Module):
    """Абриале-мост: событийно-управляемая медиация между уровнями bridge.

    Комбинирует BridgeOfModules (иерархическая cross-attention медиация)
    с AbrialeLayer (событийно-управляемые N-местные связи):

    1. Иерархический bridge строит дерево согласования источников
    2. AbrialeLayer на каждом уровне генерирует «события» из медиированных
       представлений и активизирует правила для дополнительной модуляции
    3. N-местные связи позволяют учитывать тройки/четвёрки источников,
       а не только пары

    Это v59 гибрид — объединяет лучшее из v57 (Abriale) и v58 (Bridge).

    Args:
        d_model: размерность модели
        n_sources: число геометрических источников
        n_heads: число голов cross-attention
        dropout: dropout
        bridge_mode: 'full' или 'lightweight'
        d_event: размерность пространства событий Абриале
        n_rules: число правил в банке
        arity: арность N-местных связей (2 или 3)
    """
    def __init__(self, d_model: int, n_sources: int, n_heads: int = 2,
                 dropout: float = 0.1, bridge_mode: str = 'lightweight',
                 d_event: int = 64, n_rules: int = 64, arity: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources

        # --- Bridge tree (как в BridgeOfModules) ---
        self.bridge_mode = bridge_mode

        def make_bridge():
            if bridge_mode == 'lightweight':
                return LightweightBridge(d_model)
            return PairwiseBridge(d_model, n_heads, dropout)

        self.bridge_tree = nn.ModuleList()
        self.tree_structure = []

        remaining = n_sources
        level = 0
        while remaining > 1:
            n_pairs = remaining // 2
            has_odd = (remaining % 2 == 1)
            bridges = nn.ModuleList([make_bridge() for _ in range(n_pairs)])
            self.bridge_tree.append(bridges)
            self.tree_structure.append((n_pairs, has_odd))
            remaining = n_pairs + (1 if has_odd else 0)
            level += 1

        # --- Abriale компонент на финальном уровне ---
        # EventGenerator: медиированный сигнал → событие
        from .abriale import EventGenerator, RuleBank, TransactionGate, IsotropicAttention
        self.iso_attn = IsotropicAttention(d_model, n_heads=n_heads, arity=arity, dropout=dropout)
        self.event_gen = EventGenerator(d_model, d_event=d_event)
        self.rule_bank = RuleBank(d_event=d_event, d_model=d_model, n_rules=n_rules)
        self.transaction = TransactionGate(d_model, d_event=d_event)
        self.abriale_norm = nn.LayerNorm(d_model)
        self.abriale_scale = nn.Parameter(torch.tensor(0.05))

        # --- Global gate и norms ---
        self.global_gate = nn.Parameter(torch.tensor(0.0))
        self.source_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_sources)])
        self._last_global_gate = 0.5
        self._last_abriale_commit = 0.0

    def forward(self, x: torch.Tensor, source_outputs: list) -> torch.Tensor:
        assert len(source_outputs) == self.n_sources

        # Нормализуем источники
        current_level = [self.source_norms[i](src) for i, src in enumerate(source_outputs)]

        # Иерархическая медиация (bridge tree)
        for bridges, (n_pairs, has_odd) in zip(self.bridge_tree, self.tree_structure):
            next_level = []
            for p in range(n_pairs):
                mediated = bridges[p](current_level[2*p], current_level[2*p+1])
                next_level.append(mediated)
            if has_odd:
                next_level.append(current_level[-1])
            current_level = next_level

        bridge_output = current_level[0]

        # Abriale модуляция: событийно-управляемая коррекция bridge output
        h = self.abriale_norm(bridge_output)
        iso_out, _ = self.iso_attn(h)
        enriched = h + iso_out
        events, event_types = self.event_gen(enriched)
        actions, hit_weights = self.rule_bank(events, enriched)
        committed = self.transaction(enriched, actions, event_types)
        bridge_output = bridge_output + self.abriale_scale * committed

        # Сохраняем для диагностики и aux loss
        self._last_hit_weights = hit_weights
        with torch.no_grad():
            self._last_abriale_commit = (committed.abs().mean() / (actions.abs().mean() + 1e-10)).item()

        # Global gate
        gate = torch.sigmoid(self.global_gate)
        with torch.no_grad():
            self._last_global_gate = gate.item()

        return x + gate * bridge_output

    def get_bridge_stats(self) -> dict:
        stats = {
            'global_gate': self._last_global_gate,
            'abriale_commit_rate': self._last_abriale_commit,
            'abriale_scale': self.abriale_scale.item(),
            'n_levels': len(self.bridge_tree),
            'bridge_scales': [],
        }
        for level_idx, bridges in enumerate(self.bridge_tree):
            for bridge_idx, bridge in enumerate(bridges):
                stats['bridge_scales'].append({
                    'level': level_idx, 'pair': bridge_idx,
                    'scale': bridge.scale.item(),
                })
        return stats

    def get_abriale_aux_loss(self) -> torch.Tensor:
        """Aux loss для балансировки правил Абриале."""
        if hasattr(self, '_last_hit_weights'):
            hw = self._last_hit_weights
            avg_usage = hw.mean(dim=(0, 1))
            target = 1.0 / hw.shape[-1]
            return ((avg_usage - target) ** 2).sum() * hw.shape[-1]
        return torch.tensor(0.0)


class AdaptiveBridgeOfModules(nn.Module):
    """Bridge of Modules с адаптивной глубиной.

    Вместо фиксированного числа уровней bridge, глубина определяется
    динамически на основе сложности входа:

    1. Complexity estimator оценивает «сложность» набора источников
       (дисперсия, попарное расхождение, энтропия)
    2. На основе сложности выбирается число уровней mediation:
       - Простой вход (источники согласованы) → 1 уровень (быстро)
       - Сложный вход (источники конфликтуют) → полное дерево (тщательно)
    3. Экономия вычислений на простых примерах

    Args:
        d_model: размерность модели
        n_sources: число источников
        n_heads: число голов cross-attention
        dropout: dropout
        bridge_mode: 'full' или 'lightweight'
        max_levels: максимальное число уровней (None = полное дерево)
    """
    def __init__(self, d_model: int, n_sources: int, n_heads: int = 2,
                 dropout: float = 0.1, bridge_mode: str = 'lightweight',
                 max_levels: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources
        self.bridge_mode = bridge_mode

        def make_bridge():
            if bridge_mode == 'lightweight':
                return LightweightBridge(d_model)
            return PairwiseBridge(d_model, n_heads, dropout)

        # Строим полное дерево
        self.bridge_tree = nn.ModuleList()
        self.tree_structure = []
        remaining = n_sources
        while remaining > 1:
            n_pairs = remaining // 2
            has_odd = (remaining % 2 == 1)
            bridges = nn.ModuleList([make_bridge() for _ in range(n_pairs)])
            self.bridge_tree.append(bridges)
            self.tree_structure.append((n_pairs, has_odd))
            remaining = n_pairs + (1 if has_odd else 0)

        self.total_levels = len(self.bridge_tree)
        self.max_levels = max_levels if max_levels > 0 else self.total_levels

        # Complexity estimator: оценивает сколько уровней нужно
        # Вход: попарные расхождения источников → скаляр [0, 1]
        self.complexity_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Per-level reduction: промежуточные average pools для ранней остановки
        self.level_reducers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(self.total_levels)
        ])

        # Global gate
        self.global_gate = nn.Parameter(torch.tensor(0.0))
        self.source_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_sources)])

        # Stats
        self._last_global_gate = 0.5
        self._last_complexity = 0.5
        self._last_active_levels = self.total_levels

    def _estimate_complexity(self, source_outputs: list) -> torch.Tensor:
        """Оценка сложности: насколько источники расходятся.

        Высокая дисперсия между источниками → высокая сложность → больше уровней.
        """
        # Средний вектор всех источников
        stacked = torch.stack(source_outputs, dim=0)  # (N, B, T, C)
        mean_src = stacked.mean(dim=0)  # (B, T, C)

        # Дисперсия между источниками
        variance = ((stacked - mean_src.unsqueeze(0)) ** 2).mean(dim=(0, 2))  # (B, C)
        # → скаляр сложности [0, 1]
        complexity = self.complexity_proj(variance)  # (B, 1)
        return complexity.squeeze(-1)  # (B,)

    def forward(self, x: torch.Tensor, source_outputs: list) -> torch.Tensor:
        assert len(source_outputs) == self.n_sources

        # Нормализуем
        current_level = [self.source_norms[i](src) for i, src in enumerate(source_outputs)]

        # Оценка сложности
        complexity = self._estimate_complexity(current_level)  # (B,)
        with torch.no_grad():
            self._last_complexity = complexity.mean().item()

        # Определяем число активных уровней
        # complexity ∈ [0, 1] → n_levels ∈ [1, max_levels]
        n_active = max(1, int(round(self._last_complexity * self.max_levels)))
        n_active = min(n_active, self.total_levels)
        with torch.no_grad():
            self._last_active_levels = n_active

        # Иерархическая медиация до n_active уровней
        for lvl in range(n_active):
            bridges = self.bridge_tree[lvl]
            n_pairs, has_odd = self.tree_structure[lvl]
            next_level = []
            for p in range(n_pairs):
                mediated = bridges[p](current_level[2*p], current_level[2*p+1])
                next_level.append(mediated)
            if has_odd:
                next_level.append(current_level[-1])
            current_level = next_level

        # Если остановились рано, суммируем оставшиеся элементы
        if len(current_level) > 1:
            bridge_output = sum(current_level) / len(current_level)
        else:
            bridge_output = current_level[0]

        # Global gate
        gate = torch.sigmoid(self.global_gate)
        with torch.no_grad():
            self._last_global_gate = gate.item()

        return x + gate * bridge_output

    def get_bridge_stats(self) -> dict:
        stats = {
            'global_gate': self._last_global_gate,
            'complexity': self._last_complexity,
            'active_levels': self._last_active_levels,
            'total_levels': self.total_levels,
            'bridge_scales': [],
        }
        for level_idx, bridges in enumerate(self.bridge_tree):
            for bridge_idx, bridge in enumerate(bridges):
                stats['bridge_scales'].append({
                    'level': level_idx, 'pair': bridge_idx,
                    'scale': bridge.scale.item(),
                })
        return stats


class SourceSpecializer(nn.Module):
    """Доменная специализация источников.

    Каждый из N источников получает soft domain assignment:
    - Оценщик домена определяет, к какому домену принадлежит текущий вход
    - Каждый источник имеет learnable domain affinity (какие домены ему «близки»)
    - Источники автоматически специализируются на разных доменах

    Это позволяет разным источникам отвечать за разные типы данных:
    например, HeisenbergAttention → числовые паттерны,
    PalaceAttention → синтаксические структуры, и т.д.

    Args:
        d_model: размерность модели
        n_sources: число источников
        n_domains: число доменов
    """
    def __init__(self, d_model: int, n_sources: int, n_domains: int = 4):
        super().__init__()
        self.n_sources = n_sources
        self.n_domains = n_domains

        # Domain detector: определяет домен входа
        self.domain_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, n_domains),
        )

        # Per-source domain affinity: (n_sources, n_domains)
        # Инициализируем как identity-like → каждый источник предпочитает свой домен
        init = torch.zeros(n_sources, n_domains)
        for i in range(min(n_sources, n_domains)):
            init[i, i] = 1.0
        self.domain_affinity = nn.Parameter(init)

        # Per-source scale
        self.source_scales = nn.Parameter(torch.full((n_sources,), 0.1))

        # Stats
        self._last_domain_probs = None
        self._last_source_weights = None

    def forward(self, x: torch.Tensor, source_outputs: list) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — базовый выход
            source_outputs: list of N tensors (B, T, C)
        Returns:
            enriched: x + специализированная комбинация источников
        """
        B, T, C = x.shape
        n = len(source_outputs)

        # Определяем домен: mean-pool → domain logits
        x_mean = x.mean(dim=1)  # (B, C)
        domain_logits = self.domain_proj(x_mean)  # (B, n_domains)
        domain_probs = F.softmax(domain_logits, dim=-1)  # (B, n_domains)

        # Source weights: domain_probs × domain_affinity^T → (B, n_sources)
        affinity = F.softmax(self.domain_affinity, dim=-1)  # (n_sources, n_domains)
        source_weights = torch.matmul(domain_probs, affinity.T)  # (B, n_sources)
        source_weights = source_weights * self.source_scales.abs().unsqueeze(0)

        with torch.no_grad():
            self._last_domain_probs = domain_probs.mean(dim=0).tolist()
            self._last_source_weights = source_weights.mean(dim=0).tolist()

        # Взвешенная комбинация
        result = x
        for i, src in enumerate(source_outputs):
            w = source_weights[:, i].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            result = result + w * src

        return result

    def get_specialization_stats(self) -> dict:
        stats = {}
        if self._last_domain_probs is not None:
            stats['domain_probs'] = self._last_domain_probs
            stats['source_weights'] = self._last_source_weights
            # Domain affinity matrix
            affinity = F.softmax(self.domain_affinity, dim=-1)
            stats['affinity'] = affinity.detach().tolist()
        return stats


class ArchetypalInterlingua(nn.Module):
    """Архетипальная Интерлингва: hub-and-spoke посредник для N модулей.

    .. warning::
        ИЗВЕСТНЫЙ БАГ: В данной реализации все N источников разделяют
        *один* ``trit_proj`` (строка ~990). Это создаёт bottleneck:
        градиенты от разных источников конкурируют за один проектор,
        что снижает качество представления (PPL ≈ 2.93 вместо 2.75).

        Используйте ``ArchetypalInterlinguaFixed`` из модуля
        ``geometry.interlingua_fixed`` — там каждый источник имеет
        собственный ``trit_proj[i]``.

        Ссылка: docs/THEORY_VS_PRACTICE.md, CHANGELOG.md 2026-03-24.

    Вместо дерева попарных мостов (BridgeOfModules) — единое промежуточное
    представление из 64 архетипов, через которое проходят все источники.

    Архитектура (по аналогии с Atamiri Гусмана де Рохаса):
    1. Фаза 1 — КОДИРОВАНИЕ: каждый модуль проецирует свой выход
       в архетипальное пространство (64 × d_model) через bottleneck
    2. Фаза 2 — АГРЕГАЦИЯ: вклады всех модулей агрегируются
       в единое архетипальное представление с тернарной квантизацией
       {-1, 0, +1} (кольцо Aymara siwi)
    3. Фаза 3 — ДЕКОДИРОВАНИЕ: архетипальное представление
       проецируется обратно в пространство токенов через readout

    Преимущества:
    - O(N) вместо O(N log N) — полный параллелизм кодировщиков
    - Масштабируемость: добавление модуля = +1 кодировщик
    - Интерпретируемость: 64 архетипа × 3 состояния = «язык» системы
    - Тернарная логика: модуль может голосовать «за» (+1), «против» (-1),
      или «воздержаться» (0 = 变爻 = ina аймара)

    Теоретическое обоснование: archetypal-interlingua-theory.md,
    Теоремы 16–19.

    Args:
        d_model: размерность модели
        n_sources: число геометрических источников (модулей)
        n_archetypes: число архетипов (64 = гексаграммы Q6)
        d_bottleneck: размерность bottleneck в кодировщиках
        use_ternary: использовать тернарную квантизацию {-1,0,+1}
        uncertainty_budget: бюджет неопределённости [0, 1] для тернарного режима
        n_heads: число голов readout cross-attention
        ternary_warmup_steps: шагов temperature annealing (1.0 → min_temp)
        ternary_min_temp: минимальная температура после annealing
    """
    def __init__(self, d_model: int, n_sources: int,
                 n_archetypes: int = 64, d_bottleneck: int = 0,
                 use_ternary: bool = True, uncertainty_budget: float = 0.3,
                 n_heads: int = 4,
                 ternary_warmup_steps: int = 3000,
                 ternary_min_temp: float = 0.1,
                 use_paired_bit: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources
        self.n_archetypes = n_archetypes
        self.use_ternary = use_ternary
        self.use_paired_bit = use_paired_bit
        d_bn = d_bottleneck or max(d_model // 4, 16)

        # --- Фаза 1: Кодировщики (по одному на источник) ---
        # Каждый кодировщик проецирует (B, T, d_model) → (B, n_archetypes, d_model)
        self.encoders = nn.ModuleList()
        self.source_norms = nn.ModuleList()
        for _ in range(n_sources):
            self.source_norms.append(nn.LayerNorm(d_model))
            self.encoders.append(nn.Sequential(
                nn.Linear(d_model, d_bn, bias=False),
                nn.SiLU(),
                nn.Linear(d_bn, d_model, bias=False),
            ))

        # Архетипальные якоря (learnable queries для cross-attention)
        self.archetype_queries = nn.Parameter(
            torch.randn(n_archetypes, d_model) * 0.02
        )

        # Cross-attention: архетипы attend к каждому источнику
        self.encode_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.encode_norm = nn.LayerNorm(d_model)

        # --- Фаза 2: Тернарная агрегация ---
        if use_ternary:
            if use_paired_bit:
                # Строительная логика: проекция в 2 бита (пару) на архетип
                # trit = bit_a + bit_b - 1; direction = bit_a - bit_b
                self.paired_bit_proj = nn.Linear(d_model, 2, bias=True)
                nn.init.zeros_(self.paired_bit_proj.bias)
                self.paired_bit_temp = 1.0  # начальная температура сигмоиды
            else:
                # ИСПРАВЛЕНИЕ БАГА v60: каждый источник получает НЕЗАВИСИМУЮ
                # проекцию trit_proj. Ранее использовался общий trit_proj для
                # всех источников → все модули голосовали одинаково → 64 архетипа
                # все идентичные → readout видел const → PPL = vanilla.
                # Теперь: n_sources отдельных матриц → дифференцированное голосование.
                self.trit_projs = nn.ModuleList([
                    nn.Linear(d_model, 1, bias=True)
                    for _ in range(n_sources)
                ])
                for proj in self.trit_projs:
                    nn.init.zeros_(proj.bias)
            # Learnable uncertainty budget
            self.log_uncertainty = nn.Parameter(
                torch.tensor(uncertainty_budget).clamp(0.01, 0.99).logit()
            )

        # Агрегационная проекция: объединяет N вкладов
        self.aggregate_proj = nn.Linear(d_model, d_model, bias=False)
        self.aggregate_norm = nn.LayerNorm(d_model)

        # --- Фаза 3: Декодирование (readout) ---
        # Cross-attention: токены attend к архетипам
        self.readout_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.readout_norm = nn.LayerNorm(d_model)
        self.readout_proj = nn.Linear(d_model, d_model, bias=False)

        # Global gate: возможность отключить интерлингву
        # Инициализация с bias=+0.5 → sigmoid ≈ 0.62 — стимулирует использование
        # interlingua pathway. Если бесполезна — gate быстро уйдёт к 0.
        self.global_gate = nn.Parameter(torch.tensor(0.5))
        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(0.1))

        # --- Temperature annealing для тернарной квантизации ---
        # Начинаем с мягких тритов (temp=1.0), постепенно хардим (temp→0.1)
        # Это обеспечивает gradient flow на ранних этапах обучения
        if use_ternary:
            self.ternary_warmup_steps = ternary_warmup_steps
            self.ternary_min_temp = ternary_min_temp
            self.register_buffer('_ternary_step', torch.tensor(0, dtype=torch.long))

        # Статистика
        self._last_global_gate = 0.5
        self._last_trit_distribution = {'pos': 0.33, 'zero': 0.33, 'neg': 0.33}
        self._last_direction_stats = {'spring': 0.0, 'autumn': 0.0}
        self._last_raw_scores = None  # для activation loss
        self._last_all_trit_scores = None  # для diversity loss (Variant C)
        self._last_archetype_usage = None

        # Q6 якоря для корреляционного анализа
        self._init_q6_anchors()

    def _init_q6_anchors(self):
        """Инициализирует Q6 координаты 64 архетипов = 64 гексаграммы."""
        anchors = []
        for i in range(min(self.n_archetypes, 64)):
            vertex = tuple(2 * ((i >> (5 - b)) & 1) - 1 for b in range(6))
            anchors.append(vertex)
        # Дополняем если n_archetypes > 64
        while len(anchors) < self.n_archetypes:
            anchors.append(tuple(0.0 for _ in range(6)))
        self.register_buffer(
            'q6_anchors',
            torch.tensor(anchors, dtype=torch.float32)
        )

    @property
    def uncertainty_budget(self) -> torch.Tensor:
        if self.use_ternary:
            return torch.sigmoid(self.log_uncertainty)
        return torch.tensor(0.0)

    def _encode_source(self, source_output: torch.Tensor,
                       encoder: nn.Module, norm: nn.Module) -> torch.Tensor:
        """Кодирует один источник в архетипальное пространство.

        Args:
            source_output: (B, T, d_model) — выход модуля-источника
            encoder: bottleneck encoder для этого источника
            norm: LayerNorm для этого источника

        Returns:
            contribution: (B, n_archetypes, d_model) — вклад в архетипы
        """
        B = source_output.shape[0]
        h = norm(source_output)
        h = encoder(h)  # (B, T, d_model) — bottleneck transformation

        # Cross-attention: архетипальные queries attend к трансформированному источнику
        queries = self.archetype_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 64, d_model)
        contribution, _ = self.encode_attn(
            query=queries, key=h, value=h
        )  # (B, n_archetypes, d_model)

        return contribution

    @property
    def ternary_temperature(self) -> float:
        """Текущая температура тернарной квантизации.

        Анилируется от 1.0 (мягкие триты, полный gradient flow)
        до ternary_min_temp (жёсткие триты) за ternary_warmup_steps шагов.
        """
        if not self.use_ternary:
            return 1.0
        step = self._ternary_step.item()
        progress = min(step / max(self.ternary_warmup_steps, 1), 1.0)
        # cosine annealing: плавный переход
        temp = self.ternary_min_temp + (1.0 - self.ternary_min_temp) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return temp

    def _paired_bit_quantize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Строительная квантизация: трит из пары битов.

        Каждый бит — независимая сигмоида с STE. Трит = bit_a + bit_b - 1.
        Нет мёртвой зоны: градиент течёт через оба бита всегда.

        (1,1) → +1 (jisa, лето)    (0,0) → -1 (jani, зима)
        (0,1) →  0 (весна, ↑)      (1,0) →  0 (осень, ↓)

        Args:
            contribution: (B, n_archetypes, d_model)

        Returns:
            trit_scores: (B, n_archetypes) — триты {-1, 0, +1}
        """
        logits = self.paired_bit_proj(contribution)  # (B, n_archetypes, 2)
        self._last_raw_scores = logits.sum(dim=-1)  # совместимость с activation loss

        temp = self.ternary_temperature
        # Температура влияет на жёсткость сигмоиды
        effective_temp = max(temp, 0.1)

        probs = torch.sigmoid(logits / effective_temp)  # (B, n_archetypes, 2)

        # STE: soft forward, hard backward
        bits_hard = (probs > 0.5).float()
        bits = probs + (bits_hard - probs).detach()

        bit_a = bits[..., 0]  # (B, n_archetypes)
        bit_b = bits[..., 1]  # (B, n_archetypes)

        # Трит = bit_a + bit_b - 1
        trit_scores = bit_a + bit_b - 1.0

        # Статистика направления переходов
        with torch.no_grad():
            hard_a = bits_hard[..., 0]
            hard_b = bits_hard[..., 1]
            hard_trits = hard_a + hard_b - 1.0
            zero_mask = (hard_trits == 0)
            if zero_mask.any():
                direction = hard_a - hard_b  # +1=осень, -1=весна
                spring = ((direction == -1) & zero_mask).float().sum().item()
                autumn = ((direction == 1) & zero_mask).float().sum().item()
                n_zeros = zero_mask.float().sum().item()
                self._last_direction_stats = {
                    'spring': spring / max(n_zeros, 1),
                    'autumn': autumn / max(n_zeros, 1),
                }

        return trit_scores

    def _ternary_quantize(self, contribution: torch.Tensor,
                          source_idx: int = 0) -> torch.Tensor:
        """Тернарная квантизация вклада: {-1, 0, +1} per archetype.

        С temperature annealing: на ранних шагах обучения используются мягкие
        триты (tanh(scores/T) с большим T), обеспечивая gradient flow.
        По мере обучения T → 0.1, триты становятся жёсткими.

        Args:
            contribution: (B, n_archetypes, d_model)
            source_idx: индекс источника — выбирает ИНДИВИДУАЛЬНЫЙ trit_proj.
                Каждый источник имеет свою проекцию, что обеспечивает
                дифференцированное голосование за архетипы.

        Returns:
            trit_scores: (B, n_archetypes) — мягкие или жёсткие триты
        """
        # Строительная логика: трит из пары битов
        if self.use_paired_bit:
            return self._paired_bit_quantize(contribution)

        # Используем trit_proj ЭТОГО конкретного источника (не общий!)
        proj = self.trit_projs[source_idx]
        scores = proj(contribution).squeeze(-1)  # (B, n_archetypes)

        # Сохраняем raw scores для activation loss (последний источник)
        self._last_raw_scores = scores

        temp = self.ternary_temperature

        if temp > 0.15:
            # Тёплая фаза: мягкие триты, градиент течёт напрямую
            trit_scores = torch.tanh(scores / temp)
        else:
            # Холодная фаза: жёсткие триты с STE
            raw = torch.tanh(scores)
            threshold = (1.0 - self.uncertainty_budget) * 0.5 + 0.1

            hard = torch.zeros_like(raw)
            hard[raw > threshold] = 1.0
            hard[raw < -threshold] = -1.0
            trit_scores = raw + (hard - raw).detach()

        return trit_scores

    def forward(self, x: torch.Tensor,
                source_outputs: list) -> torch.Tensor:
        """Forward pass Архетипальной Интерлингвы.

        Args:
            x: (B, T, C) — базовый выход (identity path)
            source_outputs: list of N tensors (B, T, C)

        Returns:
            enriched: (B, T, C) — x + interlingua_contribution
        """
        assert len(source_outputs) == self.n_sources
        B, T, C = x.shape

        # === Фаза 1: Кодирование (параллельно для каждого модуля) ===
        contributions = []
        trit_scores_list = []
        for i, src in enumerate(source_outputs):
            contrib = self._encode_source(src, self.encoders[i], self.source_norms[i])
            contributions.append(contrib)  # (B, n_archetypes, d_model)

            if self.use_ternary:
                # Передаём source_idx — каждый источник использует свой trit_proj
                trits = self._ternary_quantize(contrib, source_idx=i)
                trit_scores_list.append(trits)

        # === Фаза 2: Агрегация ===
        if self.use_ternary and trit_scores_list:
            # Инкремент шага для temperature annealing (только в training)
            if self.training:
                self._ternary_step += 1

            # Сохраняем все trit scores для diversity loss (Variant C)
            self._last_all_trit_scores = torch.stack(trit_scores_list, dim=0)  # (N, B, archetypes)

            # Тернарное голосование: суммируем триты всех модулей
            trit_sum = self._last_all_trit_scores.sum(dim=0)  # (B, n_archetypes)
            consensus = torch.tanh(trit_sum / max(self.n_sources, 1))  # (B, n_archetypes)

            # Epsilon leak: гарантирует gradient flow к encoders даже при consensus≈0
            weights = (consensus.abs() + 0.01).unsqueeze(-1)  # (B, n_archetypes, 1)

            # Взвешенное агрегирование: каждый источник взвешен по согласованности
            # с консенсусом. Если source_i проголосовал +1 и consensus=+0.8,
            # его вес = softmax(trit_i * consensus) — выше для согласных источников.
            # Это решает проблему mean_contrib: при идентичных голосах работает
            # как среднее, но при расхождении усиливает мнение большинства.
            stacked = torch.stack(contributions, dim=0)  # (N, B, archetypes, d_model)
            all_trits = self._last_all_trit_scores      # (N, B, archetypes)
            # alignment: насколько каждый source согласен с консенсусом
            alignment = all_trits * consensus.unsqueeze(0)  # (N, B, archetypes)
            # softmax по sources: (N, B, archetypes) → мягкие веса
            source_weights = F.softmax(alignment, dim=0).unsqueeze(-1)  # (N, B, arch, 1)
            weighted_contrib = (stacked * source_weights).sum(dim=0)  # (B, arch, d_model)
            aggregated = weighted_contrib * weights  # (B, archetypes, d_model)

            # Статистика тритов
            with torch.no_grad():
                all_trits = torch.stack(trit_scores_list, dim=0)  # (N, B, archetypes)
                hard_trits = all_trits.sign()
                total = hard_trits.numel()
                self._last_trit_distribution = {
                    'pos': (hard_trits > 0).float().sum().item() / total,
                    'zero': (hard_trits == 0).float().sum().item() / total,
                    'neg': (hard_trits < 0).float().sum().item() / total,
                }
                self._last_archetype_usage = consensus.abs().mean(dim=0).detach()
        else:
            # Без тернарной квантизации: простое среднее
            stacked = torch.stack(contributions, dim=0)
            aggregated = stacked.mean(dim=0)  # (B, n_archetypes, d_model)

        # Проекция агрегированного представления
        aggregated = self.aggregate_norm(self.aggregate_proj(aggregated))

        # === Фаза 3: Декодирование (readout) ===
        # Токены attend к архетипам: "какие архетипы релевантны мне?"
        readout, _ = self.readout_attn(
            query=x, key=aggregated, value=aggregated
        )  # (B, T, d_model)
        readout = self.readout_proj(self.readout_norm(readout))

        # Global gate + scale
        gate = torch.sigmoid(self.global_gate)
        with torch.no_grad():
            self._last_global_gate = gate.item()

        return x + gate * self.scale * readout

    def get_interlingua_loss(self) -> torch.Tensor:
        """Вспомогательный loss для обучения интерлингвы.

        Три компонента:
        1. Archetype balance: все архетипы используются примерно одинаково
        2. Uncertainty penalty: штраф за несоответствие тритового баланса бюджету
        3. Activation encouragement: толкает trit_projs scores от нуля
           (обеспечивает gradient flow через все per-source trit_projs)
        """
        loss = self.archetype_queries.new_tensor(0.0)

        if self._last_archetype_usage is not None:
            usage = self._last_archetype_usage
            target = usage.mean()
            balance_loss = ((usage - target) ** 2).mean()
            loss = loss + 0.1 * balance_loss

        if self.use_ternary and not self.use_paired_bit:
            zero_frac = self._last_trit_distribution.get('zero', 0.33)
            target_frac = self.uncertainty_budget.item()
            uncertainty_loss = (zero_frac - target_frac) ** 2
            loss = loss + 0.05 * uncertainty_loss

            # Activation encouragement: применяем ко всем trit_projs
            # Это гарантирует gradient flow к каждому источнику независимо
            if self._last_raw_scores is not None:
                temp = self.ternary_temperature
                encouragement_weight = max(temp - self.ternary_min_temp, 0.0)
                if encouragement_weight > 0:
                    activation_loss = -self._last_raw_scores.abs().clamp(max=2.0).mean()
                    loss = loss + 0.02 * encouragement_weight * activation_loss

            # ── Variant C: Diversity loss ───────────────────────────────
            # Штраф за идентичные тритовые паттерны между источниками.
            # Без этого per-source trit_projs могут сколлапсировать к одной и той же
            # проекции (информационный bottleneck v60-v61).
            # Метрика: средняя попарная cos-similarity между trit-паттернами
            # источников. При n=2: одна пара. При n>2: C(n,2) пар.
            if (self._last_all_trit_scores is not None
                    and self._last_all_trit_scores.shape[0] >= 2):
                trits = self._last_all_trit_scores  # (N, B, archetypes)
                N = trits.shape[0]
                # Flatten batch: (N, B*archetypes)
                flat = trits.reshape(N, -1)
                # Normalize per source
                flat_norm = F.normalize(flat, dim=-1, eps=1e-8)
                # Pairwise cosine similarity matrix: (N, N)
                cos_sim = flat_norm @ flat_norm.T
                # Mean off-diagonal absolute cosine similarity
                mask = ~torch.eye(N, device=cos_sim.device, dtype=torch.bool)
                mean_cos = cos_sim[mask].abs().mean()
                # Цель: cos_sim → 0 (ортогональные голоса).
                # Вес 0.1 — достаточно сильный чтобы разделить проекции,
                # но не доминирует над основным CE loss.
                loss = loss + 0.1 * mean_cos

        return loss

    def get_interlingua_stats(self) -> dict:
        """Статистика для мониторинга."""
        stats = {
            'global_gate': self._last_global_gate,
            'scale': self.scale.item(),
            'trit_distribution': self._last_trit_distribution,
        }
        if self.use_ternary:
            stats['uncertainty_budget'] = self.uncertainty_budget.item()
            stats['ternary_temperature'] = self.ternary_temperature
            stats['ternary_step'] = self._ternary_step.item()
            if self.use_paired_bit:
                stats['paired_bit'] = True
                stats['direction_stats'] = self._last_direction_stats
        if self._last_archetype_usage is not None:
            usage = self._last_archetype_usage
            stats['archetype_usage_mean'] = usage.mean().item()
            stats['archetype_usage_std'] = usage.std().item()
            stats['active_archetypes'] = (usage > 0.1).sum().item()
        return stats

    def archetype_q6_correlation(self) -> torch.Tensor:
        """Измеряет корреляцию между выученными архетипами и гексаграммами Q6.

        Аналог TokenAbstractor.cluster_hexagram_correlation().
        """
        if self.d_model < 6:
            return self.archetype_queries.new_tensor(0.0)

        # Берём первые 6 компонент archetype_queries
        q6_proj = self.archetype_queries[:, :6]  # (64, 6)
        q6_binary = q6_proj.sign()

        # Сходство с Q6 якорями
        dots = torch.matmul(q6_binary, self.q6_anchors[:64].T)  # (64, 64)
        max_sim = dots.max(dim=1).values  # (64,)
        correlation = max_sim.mean() / 6.0  # [0, 1]

        return correlation


class BridgedInterlingua(nn.Module):
    """Двойная прослойка: Module → Bridge → 64 Archetype → Core (v61).

    Гибрид BridgeOfModules (v58) и ArchetypalInterlingua (v60):
    1. Фаза 1 — МОСТОВАЯ МЕДИАЦИЯ: попарные мосты снимают деструктивную
       интерференцию между несовместимыми модулями. Результат: K медиированных
       сигналов (K ≤ N, где N — число модулей).
    2. Фаза 2 — АРХЕТИПАЛЬНОЕ КОДИРОВАНИЕ: медиированные сигналы проецируются
       в пространство 64 архетипов через per-bridge-output encoders.
    3. Фаза 3 — ТЕРНАРНАЯ АГРЕГАЦИЯ: консенсусное голосование {-1, 0, +1}
       по архетипам (как в v60).
    4. Фаза 4 — ДЕКОДИРОВАНИЕ: readout cross-attention из токенов к архетипам.

    Преимущества над v58 (только мосты):
    - Единое архетипальное представление вместо сжатия дерева в 1 вектор
    - Тернарная семантика: каждый мост-выход голосует за/против/воздержался

    Преимущества над v60 (только интерлингва):
    - Мосты предварительно снимают интерференцию между модулями
    - Архетипы получают уже «очищенные» сигналы
    - Лучшая обработка конфликтующих источников

    Сложность: O(N) для мостов (1 уровень) + O(K) для архетипов = O(N).

    Методология сочетания мостов и архетипов:
    - Мосты используют lightweight режим (bilinear) для скорости
    - Один уровень мостов (без дерева) — пары, не иерархия
    - Нечётный источник проходит напрямую в архетипальный слой
    - Архетипальный слой видит K = ceil(N/2) медиированных входов

    Args:
        d_model: размерность модели
        n_sources: число геометрических источников (модулей)
        n_archetypes: число архетипов (64 = гексаграммы)
        bridge_mode: 'full' (cross-attention) или 'lightweight' (bilinear)
        use_ternary: использовать тернарную квантизацию
        uncertainty_budget: бюджет неопределённости для тернарного режима
        n_heads: число голов cross-attention
        bridge_n_heads: число голов в мостах (для full mode)
    """
    def __init__(self, d_model: int, n_sources: int,
                 n_archetypes: int = 64, bridge_mode: str = 'lightweight',
                 use_ternary: bool = True, uncertainty_budget: float = 0.3,
                 n_heads: int = 4, bridge_n_heads: int = 2,
                 bridge_dropout: float = 0.1,
                 ternary_warmup_steps: int = 3000,
                 ternary_min_temp: float = 0.1,
                 use_paired_bit: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources
        self.n_archetypes = n_archetypes
        self.use_ternary = use_ternary
        self.use_paired_bit = use_paired_bit
        self.bridge_mode = bridge_mode

        # --- Фаза 1: Мостовая медиация (один уровень) ---
        # Пары модулей соединяются через мосты
        self.n_pairs = n_sources // 2
        self.has_odd = (n_sources % 2 == 1)
        self.n_bridge_outputs = self.n_pairs + (1 if self.has_odd else 0)

        self.source_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_sources)
        ])

        self.bridges = nn.ModuleList()
        for _ in range(self.n_pairs):
            if bridge_mode == 'lightweight':
                self.bridges.append(LightweightBridge(d_model))
            else:
                self.bridges.append(PairwiseBridge(d_model, bridge_n_heads, bridge_dropout))

        # Нечётный источник: отдельная проекция для выравнивания
        if self.has_odd:
            self.odd_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model, bias=False),
                nn.SiLU(),
            )

        # --- Фаза 2: Архетипальное кодирование (по одному на bridge-выход) ---
        d_bn = max(d_model // 4, 16)
        self.bridge_encoders = nn.ModuleList()
        self.bridge_enc_norms = nn.ModuleList()
        for _ in range(self.n_bridge_outputs):
            self.bridge_enc_norms.append(nn.LayerNorm(d_model))
            self.bridge_encoders.append(nn.Sequential(
                nn.Linear(d_model, d_bn, bias=False),
                nn.SiLU(),
                nn.Linear(d_bn, d_model, bias=False),
            ))

        # Архетипальные якоря
        self.archetype_queries = nn.Parameter(
            torch.randn(n_archetypes, d_model) * 0.02
        )

        # Cross-attention: архетипы attend к bridge-выходам
        self.encode_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.encode_norm = nn.LayerNorm(d_model)

        # --- Фаза 3: Тернарная агрегация ---
        if use_ternary:
            if use_paired_bit:
                # Строительная логика: 2 бита на архетип
                self.paired_bit_proj = nn.Linear(d_model, 2, bias=True)
                nn.init.zeros_(self.paired_bit_proj.bias)
                self.paired_bit_temp = 1.0
            else:
                # ИСПРАВЛЕНИЕ БАГА v61: per-bridge-output trit_projs.
                # Каждый bridge-выход голосует своей проекцией → дифференциация.
                self.trit_projs = nn.ModuleList([
                    nn.Linear(d_model, 1, bias=True)
                    for _ in range(self.n_bridge_outputs)
                ])
                for proj in self.trit_projs:
                    nn.init.zeros_(proj.bias)
            self.log_uncertainty = nn.Parameter(
                torch.tensor(uncertainty_budget).clamp(0.01, 0.99).logit()
            )

        self.aggregate_proj = nn.Linear(d_model, d_model, bias=False)
        self.aggregate_norm = nn.LayerNorm(d_model)

        # --- Фаза 4: Декодирование ---
        self.readout_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.readout_norm = nn.LayerNorm(d_model)
        self.readout_proj = nn.Linear(d_model, d_model, bias=False)

        # Гейты и масштаб
        self.global_gate = nn.Parameter(torch.tensor(0.5))  # bias=+0.5 → sigmoid≈0.62
        self.scale = nn.Parameter(torch.tensor(0.1))

        # --- Temperature annealing для тернарной квантизации ---
        if use_ternary:
            self.ternary_warmup_steps = ternary_warmup_steps
            self.ternary_min_temp = ternary_min_temp
            self.register_buffer('_ternary_step', torch.tensor(0, dtype=torch.long))

        # Статистика
        self._last_global_gate = 0.5
        self._last_trit_distribution = {'pos': 0.33, 'zero': 0.33, 'neg': 0.33}
        self._last_direction_stats = {'spring': 0.0, 'autumn': 0.0}
        self._last_all_trit_scores = None  # для diversity loss
        self._last_archetype_usage = None
        self._last_bridge_compatibility = []
        self._last_raw_scores = None

        # Q6 якоря
        self._init_q6_anchors()

    def _init_q6_anchors(self):
        """Инициализирует Q6 координаты 64 архетипов."""
        anchors = []
        for i in range(min(self.n_archetypes, 64)):
            vertex = tuple(2 * ((i >> (5 - b)) & 1) - 1 for b in range(6))
            anchors.append(vertex)
        while len(anchors) < self.n_archetypes:
            anchors.append(tuple(0.0 for _ in range(6)))
        self.register_buffer(
            'q6_anchors',
            torch.tensor(anchors, dtype=torch.float32)
        )

    @property
    def uncertainty_budget(self) -> torch.Tensor:
        if self.use_ternary:
            return torch.sigmoid(self.log_uncertainty)
        return torch.tensor(0.0)

    def _bridge_phase(self, source_outputs: list) -> list:
        """Фаза 1: мостовая медиация — попарное соединение модулей.

        Args:
            source_outputs: list of N tensors (B, T, d_model)

        Returns:
            bridge_outputs: list of K tensors (B, T, d_model), K = ceil(N/2)
        """
        # Нормализуем источники
        normed = [self.source_norms[i](src) for i, src in enumerate(source_outputs)]

        bridge_outputs = []
        for p in range(self.n_pairs):
            a = normed[2 * p]
            b = normed[2 * p + 1]
            mediated = self.bridges[p](a, b)
            bridge_outputs.append(mediated)

        # Нечётный источник
        if self.has_odd:
            bridge_outputs.append(self.odd_proj(normed[-1]))

        return bridge_outputs

    def _encode_bridge_output(self, bridge_out: torch.Tensor,
                               encoder: nn.Module, norm: nn.Module) -> torch.Tensor:
        """Кодирует один bridge-выход в архетипальное пространство.

        Args:
            bridge_out: (B, T, d_model) — выход моста
            encoder: bottleneck encoder
            norm: LayerNorm

        Returns:
            contribution: (B, n_archetypes, d_model)
        """
        B = bridge_out.shape[0]
        h = norm(bridge_out)
        h = encoder(h)

        queries = self.archetype_queries.unsqueeze(0).expand(B, -1, -1)
        contribution, _ = self.encode_attn(query=queries, key=h, value=h)
        return contribution

    @property
    def ternary_temperature(self) -> float:
        """Текущая температура тернарной квантизации (cosine annealing)."""
        if not self.use_ternary:
            return 1.0
        step = self._ternary_step.item()
        progress = min(step / max(self.ternary_warmup_steps, 1), 1.0)
        temp = self.ternary_min_temp + (1.0 - self.ternary_min_temp) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return temp

    def _paired_bit_quantize(self, contribution: torch.Tensor) -> torch.Tensor:
        """Строительная квантизация: трит из пары битов (как в ArchetypalInterlingua)."""
        logits = self.paired_bit_proj(contribution)  # (B, n_archetypes, 2)
        self._last_raw_scores = logits.sum(dim=-1)

        temp = self.ternary_temperature
        effective_temp = max(temp, 0.1)

        probs = torch.sigmoid(logits / effective_temp)
        bits_hard = (probs > 0.5).float()
        bits = probs + (bits_hard - probs).detach()

        trit_scores = bits[..., 0] + bits[..., 1] - 1.0

        with torch.no_grad():
            hard_a = bits_hard[..., 0]
            hard_b = bits_hard[..., 1]
            hard_trits = hard_a + hard_b - 1.0
            zero_mask = (hard_trits == 0)
            if zero_mask.any():
                direction = hard_a - hard_b
                spring = ((direction == -1) & zero_mask).float().sum().item()
                autumn = ((direction == 1) & zero_mask).float().sum().item()
                n_zeros = zero_mask.float().sum().item()
                self._last_direction_stats = {
                    'spring': spring / max(n_zeros, 1),
                    'autumn': autumn / max(n_zeros, 1),
                }

        return trit_scores

    def _ternary_quantize(self, contribution: torch.Tensor,
                          source_idx: int = 0) -> torch.Tensor:
        """Тернарная квантизация с temperature annealing.

        Args:
            contribution: (B, n_archetypes, d_model)
            source_idx: индекс bridge-выхода — выбирает ИНДИВИДУАЛЬНЫЙ trit_proj.
        """
        if self.use_paired_bit:
            return self._paired_bit_quantize(contribution)

        proj = self.trit_projs[source_idx]
        scores = proj(contribution).squeeze(-1)
        self._last_raw_scores = scores

        temp = self.ternary_temperature

        if temp > 0.15:
            trit_scores = torch.tanh(scores / temp)
        else:
            raw = torch.tanh(scores)
            threshold = (1.0 - self.uncertainty_budget) * 0.5 + 0.1
            hard = torch.zeros_like(raw)
            hard[raw > threshold] = 1.0
            hard[raw < -threshold] = -1.0
            trit_scores = raw + (hard - raw).detach()

        return trit_scores

    def forward(self, x: torch.Tensor,
                source_outputs: list) -> torch.Tensor:
        """Forward pass двойной прослойки.

        Поток: source_outputs → bridges → encoders → archetypes → readout → x

        Args:
            x: (B, T, C) — базовый выход (identity path)
            source_outputs: list of N tensors (B, T, C)

        Returns:
            enriched: (B, T, C) — x + двойная прослойка contribution
        """
        assert len(source_outputs) == self.n_sources
        B, T, C = x.shape

        # === Фаза 1: Мостовая медиация ===
        bridge_outputs = self._bridge_phase(source_outputs)
        # bridge_outputs: K элементов, каждый (B, T, d_model)

        # === Фаза 2: Архетипальное кодирование ===
        contributions = []
        trit_scores_list = []
        for i, b_out in enumerate(bridge_outputs):
            contrib = self._encode_bridge_output(
                b_out, self.bridge_encoders[i], self.bridge_enc_norms[i]
            )
            contributions.append(contrib)

            if self.use_ternary:
                # Передаём source_idx — каждый bridge-выход использует свой trit_proj
                trits = self._ternary_quantize(contrib, source_idx=i)
                trit_scores_list.append(trits)

        # === Фаза 3: Тернарная агрегация ===
        if self.use_ternary and trit_scores_list:
            if self.training:
                self._ternary_step += 1

            # Сохраняем все trit scores для diversity loss
            self._last_all_trit_scores = torch.stack(trit_scores_list, dim=0)  # (K, B, arch)

            trit_sum = self._last_all_trit_scores.sum(dim=0)
            consensus = torch.tanh(trit_sum / max(self.n_bridge_outputs, 1))
            weights = (consensus.abs() + 0.01).unsqueeze(-1)

            # Взвешенное агрегирование: source weight пропорционален согласию
            stacked = torch.stack(contributions, dim=0)
            alignment = self._last_all_trit_scores * consensus.unsqueeze(0)
            source_weights = F.softmax(alignment, dim=0).unsqueeze(-1)
            weighted_contrib = (stacked * source_weights).sum(dim=0)
            aggregated = weighted_contrib * weights

            with torch.no_grad():
                all_trits = torch.stack(trit_scores_list, dim=0)
                hard_trits = all_trits.sign()
                total = hard_trits.numel()
                self._last_trit_distribution = {
                    'pos': (hard_trits > 0).float().sum().item() / total,
                    'zero': (hard_trits == 0).float().sum().item() / total,
                    'neg': (hard_trits < 0).float().sum().item() / total,
                }
                self._last_archetype_usage = consensus.abs().mean(dim=0).detach()
        else:
            stacked = torch.stack(contributions, dim=0)
            aggregated = stacked.mean(dim=0)

        aggregated = self.aggregate_norm(self.aggregate_proj(aggregated))

        # === Фаза 4: Декодирование (readout) ===
        readout, _ = self.readout_attn(
            query=x, key=aggregated, value=aggregated
        )
        readout = self.readout_proj(self.readout_norm(readout))

        gate = torch.sigmoid(self.global_gate)
        with torch.no_grad():
            self._last_global_gate = gate.item()

        return x + gate * self.scale * readout

    def get_interlingua_loss(self) -> torch.Tensor:
        """Вспомогательный loss (совместим с ArchetypalInterlingua API)."""
        loss = self.archetype_queries.new_tensor(0.0)

        if self._last_archetype_usage is not None:
            usage = self._last_archetype_usage
            target = usage.mean()
            balance_loss = ((usage - target) ** 2).mean()
            loss = loss + 0.1 * balance_loss

        if self.use_ternary:
            zero_frac = self._last_trit_distribution.get('zero', 0.33)
            target_frac = self.uncertainty_budget.item()
            uncertainty_loss = (zero_frac - target_frac) ** 2
            loss = loss + 0.05 * uncertainty_loss

            # Activation encouragement: штрафует scores ≈ 0
            if self._last_raw_scores is not None:
                temp = self.ternary_temperature
                encouragement_weight = max(temp - self.ternary_min_temp, 0.0)
                if encouragement_weight > 0:
                    activation_loss = -self._last_raw_scores.abs().clamp(max=2.0).mean()
                    loss = loss + 0.02 * encouragement_weight * activation_loss

            # Diversity loss (Variant C) — аналогично ArchetypalInterlingua
            if (self._last_all_trit_scores is not None
                    and self._last_all_trit_scores.shape[0] >= 2):
                trits = self._last_all_trit_scores
                N = trits.shape[0]
                flat = trits.reshape(N, -1)
                flat_norm = F.normalize(flat, dim=-1, eps=1e-8)
                cos_sim = flat_norm @ flat_norm.T
                mask = ~torch.eye(N, device=cos_sim.device, dtype=torch.bool)
                mean_cos = cos_sim[mask].abs().mean()
                loss = loss + 0.1 * mean_cos

        return loss

    def get_interlingua_stats(self) -> dict:
        """Статистика для мониторинга."""
        stats = {
            'global_gate': self._last_global_gate,
            'scale': self.scale.item(),
            'trit_distribution': self._last_trit_distribution,
            'n_bridges': self.n_pairs,
            'n_bridge_outputs': self.n_bridge_outputs,
            'bridge_mode': self.bridge_mode,
        }
        if self.use_ternary:
            stats['uncertainty_budget'] = self.uncertainty_budget.item()
            stats['ternary_temperature'] = self.ternary_temperature
            stats['ternary_step'] = self._ternary_step.item()
            if self.use_paired_bit:
                stats['paired_bit'] = True
                stats['direction_stats'] = self._last_direction_stats
        if self._last_archetype_usage is not None:
            usage = self._last_archetype_usage
            stats['archetype_usage_mean'] = usage.mean().item()
            stats['archetype_usage_std'] = usage.std().item()
            stats['active_archetypes'] = (usage > 0.1).sum().item()
        # Статистика мостов
        bridge_scales = []
        for i, bridge in enumerate(self.bridges):
            bridge_scales.append({
                'pair': i,
                'scale': bridge.scale.item(),
            })
        stats['bridge_scales'] = bridge_scales
        return stats

    def archetype_q6_correlation(self) -> torch.Tensor:
        """Корреляция архетипов с гексаграммами Q6."""
        if self.d_model < 6:
            return self.archetype_queries.new_tensor(0.0)
        q6_proj = self.archetype_queries[:, :6]
        q6_binary = q6_proj.sign()
        dots = torch.matmul(q6_binary, self.q6_anchors[:64].T)
        max_sim = dots.max(dim=1).values
        correlation = max_sim.mean() / 6.0
        return correlation


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
