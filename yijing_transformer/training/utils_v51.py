"""
v51 утилиты: Gradient Vaccine, Norm-Free Normalization,
Sharpness Estimator, Learning Rate Finder, Gradient Flow Monitor.

Gradient Vaccine: cosine-similarity фильтрация градиентов.
Ref: Panda et al., "SparseFed: Mitigating Model Poisoning Attacks" (2022)

Norm-Free Normalization: Scaled Weight Standardization.
Ref: Brock et al., "Characterizing signal propagation" (2021)

Sharpness Estimator: оценка кривизны loss landscape.
Ref: Keskar et al., "On Large-Batch Training for Deep Learning" (2017)

Learning Rate Finder: автоматический подбор LR.
Ref: Smith, "Cyclical Learning Rates" (2017)

Gradient Flow Monitor: мониторинг gradient flow по слоям.
Ref: Standard debugging/analysis technique.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Gradient Vaccine ====================

class GradientVaccine:
    """
    Фильтрация аномальных градиентов через cosine similarity.

    Сравнивает текущий gradient с EMA gradient.
    Отклоняет обновления если similarity слишком низкая.

    Args:
        threshold: минимальная cosine similarity
        ema_decay: decay для reference gradient
    """
    def __init__(self, threshold=0.1, ema_decay=0.99):
        self.threshold = threshold
        self.ema_decay = ema_decay
        self._reference = None
        self._stats = {'accepted': 0, 'rejected': 0}

    def check_and_filter(self, model):
        """
        Проверяет и фильтрует градиенты.

        Args:
            model: nn.Module (после backward)

        Returns:
            dict: {accepted, similarity, action}
        """
        current = self._flatten_grads(model)
        if current is None:
            return {'accepted': True, 'similarity': 1.0, 'action': 'no_grad'}

        if self._reference is None:
            self._reference = current.clone()
            self._stats['accepted'] += 1
            return {'accepted': True, 'similarity': 1.0, 'action': 'init'}

        # Cosine similarity
        sim = F.cosine_similarity(
            current.unsqueeze(0), self._reference.unsqueeze(0)
        ).item()

        if sim >= self.threshold:
            # Accept and update reference
            self._reference.mul_(self.ema_decay).add_(current, alpha=1 - self.ema_decay)
            self._stats['accepted'] += 1
            return {'accepted': True, 'similarity': sim, 'action': 'accept'}
        else:
            # Reject: zero out gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()
            self._stats['rejected'] += 1
            return {'accepted': False, 'similarity': sim, 'action': 'reject'}

    def _flatten_grads(self, model):
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.data.flatten())
        return torch.cat(grads) if grads else None

    def get_stats(self):
        total = self._stats['accepted'] + self._stats['rejected']
        return {
            'accepted': self._stats['accepted'],
            'rejected': self._stats['rejected'],
            'accept_rate': self._stats['accepted'] / max(total, 1),
        }


# ==================== Norm-Free Normalization ====================

class ScaledWeightStandardization(nn.Module):
    """
    Scaled Weight Standardization (Norm-Free).

    Стандартизирует веса вместо активаций.
    Не требует running statistics, работает с любым batch size.

    W_std = γ * (W - mean(W)) / (std(W) + ε)

    Args:
        module: nn.Linear или nn.Conv слой
        gain: начальный gain
        eps: для стабильности
    """
    def __init__(self, module, gain=1.0, eps=1e-4):
        super().__init__()
        self.module = module
        self.gain = nn.Parameter(torch.tensor(gain))
        self.eps = eps

    def forward(self, x):
        weight = self.module.weight
        mean = weight.mean(dim=list(range(1, weight.dim())), keepdim=True)
        std = weight.std(dim=list(range(1, weight.dim())), keepdim=True)
        standardized = (weight - mean) / (std + self.eps)

        # Fan-in scaling
        fan_in = weight[0].numel()
        scaled = self.gain * standardized / math.sqrt(fan_in)

        return F.linear(x, scaled, self.module.bias)

    @staticmethod
    def wrap_model(model, gain=1.0):
        """
        Оборачивает все Linear слои в SWS.

        Args:
            model: nn.Module
            gain: начальный gain

        Returns:
            int: число обёрнутых слоёв
        """
        count = 0
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, ScaledWeightStandardization(module, gain))
                count += 1
            else:
                count += ScaledWeightStandardization.wrap_model(module, gain)
        return count


# ==================== Sharpness Estimator ====================

class SharpnessEstimator:
    """
    Оценка остроты минимума loss landscape.

    Вычисляет max loss в ε-окрестности текущей точки.
    Sharpness = max_loss - current_loss.

    Args:
        n_perturbations: число случайных возмущений
        epsilon: радиус возмущения
    """
    def __init__(self, n_perturbations=5, epsilon=0.01):
        self.n_perturbations = n_perturbations
        self.epsilon = epsilon
        self._history = []

    @torch.no_grad()
    def estimate(self, model, loss_fn, data):
        """
        Оценивает sharpness.

        Args:
            model: nn.Module
            loss_fn: callable(model, data) -> loss
            data: данные для вычисления loss

        Returns:
            dict: {sharpness, base_loss, max_loss, avg_perturbed_loss}
        """
        model.eval()
        base_loss = loss_fn(model, data).item()

        # Save original params
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}

        max_loss = base_loss
        total_loss = 0.0

        for _ in range(self.n_perturbations):
            # Random perturbation
            for p in model.parameters():
                noise = torch.randn_like(p.data) * self.epsilon
                p.data.add_(noise)

            perturbed_loss = loss_fn(model, data).item()
            max_loss = max(max_loss, perturbed_loss)
            total_loss += perturbed_loss

            # Restore
            for n, p in model.named_parameters():
                p.data.copy_(original_params[n])

        sharpness = max_loss - base_loss
        self._history.append(sharpness)

        return {
            'sharpness': sharpness,
            'base_loss': base_loss,
            'max_loss': max_loss,
            'avg_perturbed_loss': total_loss / self.n_perturbations,
        }

    def get_trend(self):
        """Тренд sharpness."""
        if len(self._history) < 2:
            return {'trend': 'insufficient_data'}
        recent = self._history[-5:]
        older = self._history[-10:-5] if len(self._history) >= 10 else self._history[:len(self._history)//2]
        if not older:
            return {'trend': 'insufficient_data'}
        return {
            'current': sum(recent) / len(recent),
            'previous': sum(older) / len(older),
            'trend': 'flattening' if sum(recent)/len(recent) < sum(older)/len(older) else 'sharpening',
        }


# ==================== Learning Rate Finder ====================

class LearningRateFinder:
    """
    Автоматический поиск оптимального LR.

    Экспоненциально увеличивает LR, записывает loss.
    Оптимальный LR — где loss падает быстрее всего.

    Args:
        min_lr: минимальный LR
        max_lr: максимальный LR
        n_steps: число шагов
        smooth_factor: сглаживание loss
    """
    def __init__(self, min_lr=1e-7, max_lr=10.0, n_steps=100,
                 smooth_factor=0.05):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.n_steps = n_steps
        self.smooth_factor = smooth_factor

    def find(self, model, optimizer, loss_fn, data_iter):
        """
        Запускает поиск LR.

        Args:
            model: nn.Module
            optimizer: оптимизатор
            loss_fn: callable(model, batch) -> loss
            data_iter: итератор батчей

        Returns:
            dict: {best_lr, lrs, losses, suggestion}
        """
        # Save state
        original_state = {n: p.data.clone() for n, p in model.named_parameters()}
        original_opt = optimizer.state_dict()

        lrs = []
        losses = []
        smoothed_loss = None
        best_loss = float('inf')
        best_lr = self.min_lr

        mult = (self.max_lr / self.min_lr) ** (1.0 / self.n_steps)

        lr = self.min_lr
        for step in range(self.n_steps):
            # Set LR
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            try:
                batch = next(data_iter)
            except StopIteration:
                break

            optimizer.zero_grad()
            loss = loss_fn(model, batch)

            if torch.isnan(loss) or torch.isinf(loss):
                break

            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            # Smooth
            if smoothed_loss is None:
                smoothed_loss = loss_val
            else:
                smoothed_loss = (1 - self.smooth_factor) * smoothed_loss + \
                               self.smooth_factor * loss_val

            lrs.append(lr)
            losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                best_lr = lr

            # Divergence check
            if smoothed_loss > 4 * best_loss:
                break

            lr *= mult

        # Restore
        for n, p in model.named_parameters():
            p.data.copy_(original_state[n])
        optimizer.load_state_dict(original_opt)

        # Suggestion: LR where steepest descent
        suggestion = best_lr / 10  # Heuristic: 10x before minimum

        return {
            'best_lr': best_lr,
            'suggestion': suggestion,
            'lrs': lrs,
            'losses': losses,
        }


# ==================== Gradient Flow Monitor ====================

class GradientFlowMonitor:
    """
    Мониторинг потока градиентов по слоям.

    Отслеживает gradient magnitude по слоям для
    диагностики vanishing/exploding gradients.
    """
    def __init__(self):
        self._history = []

    def record(self, model):
        """
        Записывает gradient norms для каждого слоя.

        Args:
            model: nn.Module (после backward)

        Returns:
            dict: {layers: [{name, mean_grad, max_grad, has_grad}]}
        """
        layers = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = p.grad.data
                layers.append({
                    'name': name,
                    'mean_grad': grad.abs().mean().item(),
                    'max_grad': grad.abs().max().item(),
                    'std_grad': grad.std().item() if grad.numel() > 1 else 0.0,
                    'has_grad': True,
                    'zero_pct': (grad == 0).float().mean().item(),
                })
            else:
                layers.append({
                    'name': name,
                    'mean_grad': 0.0,
                    'max_grad': 0.0,
                    'std_grad': 0.0,
                    'has_grad': False,
                    'zero_pct': 1.0,
                })

        self._history.append(layers)
        return {'layers': layers}

    def diagnose(self):
        """
        Диагностика gradient flow.

        Returns:
            dict: {vanishing, exploding, dead_layers, healthy}
        """
        if not self._history:
            return {'healthy': True, 'issues': []}

        latest = self._history[-1]
        issues = []

        vanishing = []
        exploding = []
        dead = []

        for layer in latest:
            if not layer['has_grad']:
                dead.append(layer['name'])
            elif layer['mean_grad'] < 1e-7:
                vanishing.append(layer['name'])
            elif layer['max_grad'] > 1000:
                exploding.append(layer['name'])

        if vanishing:
            issues.append(f'vanishing gradients in {len(vanishing)} layers')
        if exploding:
            issues.append(f'exploding gradients in {len(exploding)} layers')
        if dead:
            issues.append(f'{len(dead)} layers without gradients')

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'vanishing': vanishing,
            'exploding': exploding,
            'dead': dead,
        }

    def get_summary(self):
        """Сводка по всем записям."""
        if not self._history:
            return {}
        latest = self._history[-1]
        grads = [l['mean_grad'] for l in latest if l['has_grad']]
        if not grads:
            return {'no_gradients': True}
        return {
            'n_layers': len(latest),
            'avg_grad': sum(grads) / len(grads),
            'min_grad': min(grads),
            'max_grad': max(grads),
            'n_recordings': len(self._history),
        }
