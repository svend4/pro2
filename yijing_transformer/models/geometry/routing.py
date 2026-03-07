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
