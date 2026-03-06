"""
v33 утилиты: Gradient Penalty, Polyak Averaging, Loss Landscape Probe,
Adaptive Batch Sampler, Optimizer State Monitor.

Gradient Penalty: R1/R2 регуляризация через штраф на норму градиента.
Ref: Mescheder et al., "Which Training Methods for GANs do actually Converge?" (2018)

Polyak Averaging: усреднение параметров с target network.

Loss Landscape Probe: зондирование ландшафта loss вокруг текущей точки.
Ref: Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018)

Adaptive Batch Sampler: приоритизация сложных примеров.
Ref: Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining" (2016)

Optimizer State Monitor: мониторинг внутреннего состояния Adam.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque


# ==================== Gradient Penalty ====================

class GradientPenalty:
    """
    Gradient Penalty регуляризация.

    Штрафует за большую норму градиента loss по входам или параметрам.

    Режимы:
    - r1: штраф на ||∇_x D(x)||² (по входам)
    - r2: штраф на ||∇_θ L(θ)||² (по параметрам)
    - gp: штраф на (||∇_x D(x)|| - 1)² (WGAN-GP стиль)

    Args:
        lambda_gp: коэффициент штрафа
        mode: 'r1', 'r2', 'gp'
    """
    def __init__(self, lambda_gp=10.0, mode='r2'):
        self.lambda_gp = lambda_gp
        self.mode = mode

    def compute_r2(self, loss, model):
        """
        R2 penalty: ||∇_θ L||².

        Args:
            loss: scalar loss tensor (must have grad_fn)
            model: nn.Module

        Returns:
            Tensor: scalar penalty
        """
        grads = torch.autograd.grad(
            loss, [p for p in model.parameters() if p.requires_grad],
            create_graph=True, retain_graph=True, allow_unused=True
        )
        penalty = torch.tensor(0.0, device=loss.device)
        for g in grads:
            if g is not None:
                penalty = penalty + g.norm(2) ** 2
        return self.lambda_gp * penalty

    def compute_r1(self, loss, inputs):
        """
        R1 penalty: ||∇_x L||².

        Args:
            loss: scalar loss tensor
            inputs: input tensor (must require grad)

        Returns:
            Tensor: scalar penalty
        """
        grads = torch.autograd.grad(
            loss, inputs,
            create_graph=True, retain_graph=True
        )
        penalty = torch.tensor(0.0, device=loss.device)
        for g in grads:
            if g is not None:
                penalty = penalty + g.norm(2) ** 2
        return self.lambda_gp * penalty

    def compute_gp(self, loss, inputs, target_norm=1.0):
        """
        WGAN-GP style: (||∇_x L|| - target)².

        Args:
            loss: scalar loss
            inputs: input tensor
            target_norm: target gradient norm

        Returns:
            Tensor: scalar penalty
        """
        grads = torch.autograd.grad(
            loss, inputs,
            create_graph=True, retain_graph=True
        )
        grad_norm = torch.tensor(0.0, device=loss.device)
        for g in grads:
            if g is not None:
                grad_norm = grad_norm + g.norm(2) ** 2
        grad_norm = grad_norm.sqrt()
        penalty = (grad_norm - target_norm) ** 2
        return self.lambda_gp * penalty


# ==================== Polyak Averaging ====================

class PolyakAveraging:
    """
    Polyak Averaging с target network.

    Поддерживает target network, который обновляется
    через экспоненциальное усреднение с основной моделью.

    В отличие от EMA, создаёт отдельную копию модели.

    Args:
        model: nn.Module (source)
        tau: коэффициент мягкого обновления (0.005 типично)
    """
    def __init__(self, model, tau=0.005):
        self.tau = tau
        self.target = copy.deepcopy(model)
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def update(self):
        """
        Мягкое обновление target ← τ*source + (1-τ)*target.

        Вызывать source model через self._get_source() нельзя,
        поэтому передаём source в update.
        """
        # This version requires explicit source
        pass

    def update_from(self, source_model):
        """
        Обновляет target из source.

        Args:
            source_model: nn.Module
        """
        with torch.no_grad():
            for tp, sp in zip(self.target.parameters(), source_model.parameters()):
                tp.data.mul_(1 - self.tau).add_(sp.data, alpha=self.tau)

    def hard_update(self, source_model):
        """Полное копирование source → target."""
        with torch.no_grad():
            for tp, sp in zip(self.target.parameters(), source_model.parameters()):
                tp.data.copy_(sp.data)

    def get_target(self):
        """Возвращает target model."""
        return self.target

    def get_distance(self, source_model):
        """
        L2 расстояние между source и target.

        Returns:
            float
        """
        dist = 0.0
        with torch.no_grad():
            for tp, sp in zip(self.target.parameters(), source_model.parameters()):
                dist += (tp.data - sp.data).norm(2).item() ** 2
        return math.sqrt(dist)


# ==================== Loss Landscape Probe ====================

class LossLandscapeProbe:
    """
    Зондирование ландшафта loss.

    Сэмплирует loss в окрестности текущих параметров
    для анализа гладкости / остроты минимума.

    Args:
        n_directions: число случайных направлений
        max_distance: максимальное расстояние зондирования
        n_points: число точек в каждом направлении
    """
    def __init__(self, n_directions=5, max_distance=0.1, n_points=5):
        self.n_directions = n_directions
        self.max_distance = max_distance
        self.n_points = n_points

    def probe(self, model, loss_fn):
        """
        Зондирует ландшафт.

        Args:
            model: nn.Module
            loss_fn: callable() → scalar loss (no args, uses closure)

        Returns:
            dict: {center_loss, profiles, sharpness, smoothness}
        """
        # Save original params
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        # Center loss
        with torch.no_grad():
            center_loss = loss_fn().item()

        profiles = []
        all_losses = []

        for d in range(self.n_directions):
            # Random direction (normalized)
            direction = {}
            norm = 0.0
            for n, p in model.named_parameters():
                dir_vec = torch.randn_like(p.data)
                direction[n] = dir_vec
                norm += dir_vec.norm(2).item() ** 2
            norm = math.sqrt(norm)
            for n in direction:
                direction[n] /= max(norm, 1e-8)

            # Sample along direction
            profile = []
            for i in range(self.n_points):
                alpha = self.max_distance * (i + 1) / self.n_points

                # Move in positive direction
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        p.data.copy_(original_params[n] + alpha * direction[n])
                    loss_pos = loss_fn().item()

                # Move in negative direction
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        p.data.copy_(original_params[n] - alpha * direction[n])
                    loss_neg = loss_fn().item()

                profile.append({
                    'alpha': alpha,
                    'loss_pos': loss_pos,
                    'loss_neg': loss_neg,
                })
                all_losses.extend([loss_pos, loss_neg])

            profiles.append(profile)

        # Restore original params
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(original_params[n])

        # Compute sharpness = max(loss around) - center_loss
        sharpness = max(all_losses) - center_loss if all_losses else 0.0
        avg_loss = sum(all_losses) / len(all_losses) if all_losses else center_loss
        smoothness = avg_loss - center_loss

        return {
            'center_loss': center_loss,
            'profiles': profiles,
            'sharpness': sharpness,
            'smoothness': smoothness,
            'avg_surrounding_loss': avg_loss,
        }

    def quick_sharpness(self, model, loss_fn, epsilon=0.01):
        """
        Быстрая оценка остроты минимума.

        Args:
            model: nn.Module
            loss_fn: callable() → loss
            epsilon: радиус perturbation

        Returns:
            dict: {center_loss, perturbed_loss, sharpness}
        """
        original = {n: p.data.clone() for n, p in model.named_parameters()}

        with torch.no_grad():
            center = loss_fn().item()

            # Random perturbation
            for n, p in model.named_parameters():
                p.data.add_(torch.randn_like(p.data) * epsilon)
            perturbed = loss_fn().item()

            # Restore
            for n, p in model.named_parameters():
                p.data.copy_(original[n])

        return {
            'center_loss': center,
            'perturbed_loss': perturbed,
            'sharpness': perturbed - center,
        }


# ==================== Adaptive Batch Sampler ====================

class AdaptiveBatchSampler:
    """
    Приоритизация сложных примеров (hard example mining).

    Отслеживает per-sample loss и сэмплирует батчи
    с приоритетом для высоко-loss примеров.

    Args:
        dataset_size: размер датасета
        hard_fraction: доля сложных примеров в батче
        history_decay: затухание истории loss
    """
    def __init__(self, dataset_size, hard_fraction=0.5, history_decay=0.99):
        self.dataset_size = dataset_size
        self.hard_fraction = hard_fraction
        self.history_decay = history_decay
        self._losses = torch.zeros(dataset_size)
        self._seen = torch.zeros(dataset_size, dtype=torch.bool)

    def update_losses(self, indices, losses):
        """
        Обновляет loss для примеров.

        Args:
            indices: list/Tensor индексов
            losses: list/Tensor loss'ов
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().tolist()

        for idx, loss in zip(indices, losses):
            if 0 <= idx < self.dataset_size:
                self._losses[idx] = self.history_decay * self._losses[idx] + \
                                    (1 - self.history_decay) * loss
                self._seen[idx] = True

    def sample_batch(self, batch_size):
        """
        Сэмплирует батч с приоритетом для сложных примеров.

        Args:
            batch_size: размер батча

        Returns:
            Tensor: индексы сэмплированных примеров
        """
        n_hard = int(batch_size * self.hard_fraction)
        n_random = batch_size - n_hard

        indices = []

        # Hard examples: top-k by loss
        n_hard_actual = 0
        if n_hard > 0 and self._seen.any():
            seen_mask = self._seen.clone()
            losses = self._losses.clone()
            losses[~seen_mask] = -float('inf')
            k = min(n_hard, seen_mask.sum().item())
            _, topk = losses.topk(k)
            indices.extend(topk.tolist())
            n_hard_actual = k

        # Random examples (fill remaining)
        n_fill = batch_size - n_hard_actual
        if n_fill > 0:
            random_indices = torch.randint(0, self.dataset_size, (n_fill,))
            indices.extend(random_indices.tolist())

        return torch.tensor(indices[:batch_size], dtype=torch.long)

    def get_difficulty_distribution(self, n_bins=10):
        """
        Распределение сложности примеров.

        Returns:
            dict: {bin_edges, counts, mean_loss, median_loss}
        """
        seen_losses = self._losses[self._seen]
        if len(seen_losses) == 0:
            return {'bin_edges': [], 'counts': [], 'mean_loss': 0, 'median_loss': 0}

        hist = torch.histc(seen_losses, bins=n_bins)
        return {
            'counts': hist.tolist(),
            'mean_loss': seen_losses.mean().item(),
            'median_loss': seen_losses.median().item(),
            'n_seen': self._seen.sum().item(),
        }

    def reset(self):
        self._losses.zero_()
        self._seen.zero_()


# ==================== Optimizer State Monitor ====================

class OptimizerStateMonitor:
    """
    Мониторинг внутреннего состояния Adam/AdamW.

    Отслеживает:
    - Первый момент (mean of gradients)
    - Второй момент (variance of gradients)
    - Effective step size
    - Bias correction

    Args:
        optimizer: torch optimizer (Adam/AdamW)
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._history = defaultdict(lambda: deque(maxlen=50))
        self._step = 0

    def record(self, model=None):
        """
        Записывает статистики состояния оптимизатора.

        Returns:
            dict: {param_name: {m1_norm, m2_norm, step_size_est}}
        """
        self._step += 1
        snapshot = {}

        param_names = {}
        if model is not None:
            for name, p in model.named_parameters():
                param_names[id(p)] = name

        for group_idx, group in enumerate(self.optimizer.param_groups):
            lr = group.get('lr', 0)
            beta1, beta2 = group.get('betas', (0.9, 0.999))
            eps = group.get('eps', 1e-8)

            for p_idx, p in enumerate(group['params']):
                state = self.optimizer.state.get(p, {})
                if not state:
                    continue

                name = param_names.get(id(p), f'group{group_idx}_param{p_idx}')

                stats = {}
                if 'exp_avg' in state:
                    m1 = state['exp_avg']
                    stats['m1_norm'] = m1.norm(2).item()
                    stats['m1_mean'] = m1.mean().item()

                if 'exp_avg_sq' in state:
                    m2 = state['exp_avg_sq']
                    stats['m2_norm'] = m2.norm(2).item()
                    stats['m2_mean'] = m2.mean().item()
                    # Effective step size ≈ lr / sqrt(m2) (simplified)
                    avg_m2 = m2.mean().item()
                    stats['effective_step'] = lr / (math.sqrt(avg_m2) + eps) if avg_m2 > 0 else lr

                step_count = state.get('step', 0)
                if isinstance(step_count, torch.Tensor):
                    step_count = step_count.item()
                stats['optimizer_step'] = step_count
                stats['lr'] = lr

                self._history[name].append(stats)
                snapshot[name] = stats

        return snapshot

    def get_summary(self):
        """Сводка по всем параметрам."""
        summary = {}
        for name, history in self._history.items():
            if not history:
                continue
            latest = history[-1]
            summary[name] = {
                'latest': latest,
                'n_records': len(history),
            }
            if 'm1_norm' in latest:
                m1_norms = [h['m1_norm'] for h in history if 'm1_norm' in h]
                summary[name]['m1_norm_trend'] = 'growing' if len(m1_norms) >= 2 and m1_norms[-1] > m1_norms[0] * 1.1 else 'stable'
        return summary

    def get_effective_lr_distribution(self):
        """Распределение effective step sizes."""
        step_sizes = []
        for name, history in self._history.items():
            if history and 'effective_step' in history[-1]:
                step_sizes.append(history[-1]['effective_step'])
        if not step_sizes:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        t = torch.tensor(step_sizes)
        return {
            'mean': t.mean().item(),
            'min': t.min().item(),
            'max': t.max().item(),
            'std': t.std().item() if len(step_sizes) > 1 else 0.0,
        }

    def reset(self):
        self._history.clear()
        self._step = 0
