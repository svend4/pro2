"""
v31 утилиты: Mutual Information Estimator, Gradient Surgery,
Spectral Regularization, Training Phase Manager, Token Frequency Weighting.

Mutual Information: оценка MI между слоями.
Ref: Belghazi et al., "MINE: Mutual Information Neural Estimation" (2018)

Gradient Surgery: разрешение конфликтов градиентов.
Ref: Yu et al., "Gradient Surgery for Multi-Task Learning" (2020)

Spectral Regularization: ограничение спектральной нормы.

Training Phase Manager: структурированные фазы обучения.

Token Frequency Weighting: взвешивание loss по частоте токенов.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ==================== Mutual Information Estimator ====================

class MutualInformationEstimator:
    """
    Оценка взаимной информации между представлениями.

    Использует MINE (Mutual Information Neural Estimation)
    с упрощённой верхней/нижней границей.

    Args:
        d_model: размерность представлений
        hidden_dim: скрытый размер статистической сети
    """
    def __init__(self, d_model, hidden_dim=64):
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self._stats_net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def estimate(self, x, y):
        """
        Оценивает MI между x и y (MINE lower bound).

        Args:
            x: (B, D) — представления из слоя 1
            y: (B, D) — представления из слоя 2

        Returns:
            dict: {mi_estimate, joint_score, marginal_score}
        """
        B = x.size(0)
        if B < 2:
            return {'mi_estimate': 0.0, 'joint_score': 0.0, 'marginal_score': 0.0}

        # Joint samples: (x_i, y_i)
        joint = torch.cat([x, y], dim=-1)  # (B, 2D)
        joint_score = self._stats_net(joint).mean()

        # Marginal samples: (x_i, y_j) with shuffled y
        perm = torch.randperm(B, device=y.device)
        y_shuffled = y[perm]
        marginal = torch.cat([x, y_shuffled], dim=-1)
        marginal_score = self._stats_net(marginal)

        # MINE: E[T(x,y)] - log(E[exp(T(x,y'))])
        log_mean_exp = torch.logsumexp(marginal_score, dim=0) - math.log(B)

        mi = joint_score - log_mean_exp.squeeze()

        return {
            'mi_estimate': mi.item(),
            'joint_score': joint_score.item(),
            'marginal_score': log_mean_exp.item(),
        }

    def estimate_simple(self, x, y):
        """
        Простая оценка MI через корреляцию (быстрая, но грубая).

        Args:
            x: (B, D)
            y: (B, D)

        Returns:
            float: приблизительная MI
        """
        # Correlation-based: MI ≈ -0.5 * log(1 - r²)
        x_flat = x.detach().flatten()
        y_flat = y.detach().flatten()

        x_centered = x_flat - x_flat.mean()
        y_centered = y_flat - y_flat.mean()

        cov = (x_centered * y_centered).mean()
        std_x = x_centered.std() + 1e-8
        std_y = y_centered.std() + 1e-8

        r = cov / (std_x * std_y)
        r = r.clamp(-0.999, 0.999)

        mi = -0.5 * math.log(1 - r.item() ** 2)
        return mi

    @property
    def stats_net(self):
        return self._stats_net


# ==================== Gradient Surgery ====================

class GradientSurgery:
    """
    Gradient Surgery для multi-task learning.

    Когда градиенты разных задач конфликтуют (cos < 0),
    проецирует один на нормальную плоскость другого.

    Args:
        reduction: 'mean' или 'sum' для объединения
    """
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def resolve(self, grads):
        """
        Разрешает конфликты между градиентами задач.

        Args:
            grads: list[Tensor] — градиенты от разных задач (одного размера)

        Returns:
            Tensor: объединённый gradient
        """
        if len(grads) == 0:
            raise ValueError("Empty grads list")
        if len(grads) == 1:
            return grads[0].clone()

        # Flatten all grads
        flat_grads = [g.flatten().float() for g in grads]
        n_tasks = len(flat_grads)

        # Project conflicting gradients
        resolved = []
        for i in range(n_tasks):
            gi = flat_grads[i].clone()
            for j in range(n_tasks):
                if i == j:
                    continue
                gj = flat_grads[j]
                dot = (gi * gj).sum()
                if dot < 0:
                    # Project gi onto normal plane of gj
                    # gi' = gi - (gi·gj / gj·gj) * gj
                    gi = gi - (dot / (gj.norm() ** 2 + 1e-8)) * gj
            resolved.append(gi)

        # Combine
        stacked = torch.stack(resolved)
        if self.reduction == 'mean':
            combined = stacked.mean(dim=0)
        else:
            combined = stacked.sum(dim=0)

        return combined.view_as(grads[0])

    def check_conflicts(self, grads):
        """
        Проверяет наличие конфликтов между градиентами.

        Args:
            grads: list[Tensor]

        Returns:
            dict: {n_pairs, n_conflicts, conflict_ratio, cosine_matrix}
        """
        flat_grads = [g.flatten().float() for g in grads]
        n = len(flat_grads)
        n_pairs = 0
        n_conflicts = 0
        cosines = []

        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                    continue
                cos = F.cosine_similarity(
                    flat_grads[i].unsqueeze(0),
                    flat_grads[j].unsqueeze(0)
                ).item()
                row.append(cos)
                if i < j:
                    n_pairs += 1
                    if cos < 0:
                        n_conflicts += 1
            cosines.append(row)

        return {
            'n_pairs': n_pairs,
            'n_conflicts': n_conflicts,
            'conflict_ratio': n_conflicts / max(n_pairs, 1),
            'cosine_matrix': cosines,
        }


# ==================== Spectral Regularization ====================

class SpectralRegularizer:
    """
    Спектральная регуляризация весов.

    Добавляет штраф за большую спектральную норму (σ_max).
    Стабилизирует обучение и улучшает обобщение.

    Args:
        lambda_spectral: коэффициент регуляризации
        n_power_iterations: число итераций power method
    """
    def __init__(self, lambda_spectral=0.01, n_power_iterations=1):
        self.lambda_spectral = lambda_spectral
        self.n_power_iterations = n_power_iterations
        self._cache = {}

    def compute_penalty(self, model):
        """
        Вычисляет спектральный штраф.

        Args:
            model: nn.Module

        Returns:
            Tensor: scalar penalty loss
        """
        penalty = torch.tensor(0.0)
        n_matrices = 0

        for name, p in model.named_parameters():
            if p.ndim < 2:
                continue
            # Reshape to 2D
            w = p.view(p.size(0), -1)
            sigma = self._spectral_norm(w, name)
            penalty = penalty + sigma
            n_matrices += 1

        if n_matrices > 0:
            penalty = penalty / n_matrices

        return self.lambda_spectral * penalty

    def _spectral_norm(self, w, name):
        """Power iteration для оценки σ_max."""
        h, w_dim = w.shape

        # Initialize or retrieve u, v vectors
        if name not in self._cache:
            u = F.normalize(torch.randn(h), dim=0)
            v = F.normalize(torch.randn(w_dim), dim=0)
            self._cache[name] = (u.to(w.device), v.to(w.device))

        u, v = self._cache[name]
        u = u.to(w.device)
        v = v.to(w.device)

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v_new = F.normalize(w.t() @ u, dim=0)
                u_new = F.normalize(w @ v_new, dim=0)
                v = v_new
                u = u_new
            self._cache[name] = (u, v)

        # σ_max = u^T W v
        sigma = u @ w @ v
        return sigma

    def get_spectral_norms(self, model):
        """Спектральные нормы всех матриц весов."""
        norms = {}
        for name, p in model.named_parameters():
            if p.ndim < 2:
                continue
            w = p.view(p.size(0), -1)
            with torch.no_grad():
                sigma = self._spectral_norm(w, name)
            norms[name] = sigma.item()
        return norms


# ==================== Training Phase Manager ====================

class TrainingPhaseManager:
    """
    Менеджер фаз обучения.

    Управляет переходами между структурированными фазами:
    warmup → main → fine-tune → cooldown

    Каждая фаза может иметь свои гиперпараметры.

    Args:
        phases: list[dict] с описанием фаз
    """
    def __init__(self, phases=None):
        if phases is None:
            phases = [
                {'name': 'warmup', 'steps': 100, 'lr_scale': 0.1},
                {'name': 'main', 'steps': 800, 'lr_scale': 1.0},
                {'name': 'cooldown', 'steps': 100, 'lr_scale': 0.01},
            ]
        self.phases = phases
        self._total_steps = sum(p['steps'] for p in phases)
        self._global_step = 0

    def step(self):
        """Продвинуть на один шаг."""
        self._global_step += 1

    def get_current_phase(self):
        """
        Текущая фаза.

        Returns:
            dict: {name, lr_scale, phase_progress, global_progress, phase_idx, ...}
        """
        cumulative = 0
        for idx, phase in enumerate(self.phases):
            if self._global_step < cumulative + phase['steps']:
                phase_step = self._global_step - cumulative
                return {
                    **phase,
                    'phase_idx': idx,
                    'phase_step': phase_step,
                    'phase_progress': phase_step / max(phase['steps'], 1),
                    'global_progress': self._global_step / max(self._total_steps, 1),
                }
            cumulative += phase['steps']

        # Past all phases → return last
        last = self.phases[-1]
        return {
            **last,
            'phase_idx': len(self.phases) - 1,
            'phase_step': last['steps'],
            'phase_progress': 1.0,
            'global_progress': 1.0,
        }

    def get_lr_scale(self):
        """Текущий масштаб LR."""
        return self.get_current_phase().get('lr_scale', 1.0)

    def apply_lr(self, optimizer, base_lr):
        """
        Применяет масштаб LR к optimizer.

        Args:
            optimizer: torch optimizer
            base_lr: базовый LR
        """
        scale = self.get_lr_scale()
        for group in optimizer.param_groups:
            group['lr'] = base_lr * scale

    def is_phase(self, name):
        """Проверяет, текущая ли это фаза."""
        return self.get_current_phase()['name'] == name

    @property
    def global_step(self):
        return self._global_step

    @property
    def total_steps(self):
        return self._total_steps

    def get_phase_boundaries(self):
        """Границы фаз (для логирования)."""
        boundaries = []
        cumulative = 0
        for phase in self.phases:
            boundaries.append({
                'name': phase['name'],
                'start': cumulative,
                'end': cumulative + phase['steps'],
            })
            cumulative += phase['steps']
        return boundaries


# ==================== Token Frequency Weighting ====================

class TokenFrequencyWeighting:
    """
    Взвешивание loss по частоте токенов.

    Редкие токены получают больший вес,
    частые — меньший. Улучшает генерацию редких токенов.

    Стратегии:
    - inverse: weight = 1 / freq
    - sqrt_inverse: weight = 1 / sqrt(freq)
    - log_inverse: weight = 1 / log(1 + freq)
    - smoothed: weight = (max_freq / freq) ^ alpha

    Args:
        vocab_size: размер словаря
        strategy: стратегия взвешивания
        alpha: сглаживание для smoothed стратегии
        min_weight: минимальный вес
        max_weight: максимальный вес
    """
    def __init__(self, vocab_size, strategy='sqrt_inverse',
                 alpha=0.5, min_weight=0.1, max_weight=10.0):
        self.vocab_size = vocab_size
        self.strategy = strategy
        self.alpha = alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self._counts = torch.zeros(vocab_size)
        self._weights = torch.ones(vocab_size)
        self._total = 0

    def update_counts(self, tokens):
        """
        Обновляет счётчики частот.

        Args:
            tokens: Tensor с token ids
        """
        flat = tokens.flatten().long()
        for t in flat:
            if 0 <= t < self.vocab_size:
                self._counts[t] += 1
        self._total += flat.numel()

    def update_counts_batch(self, tokens):
        """
        Обновляет счётчики (батчевая версия).

        Args:
            tokens: Tensor с token ids
        """
        flat = tokens.flatten().long()
        valid = flat[(flat >= 0) & (flat < self.vocab_size)]
        if valid.numel() > 0:
            self._counts.scatter_add_(0, valid.cpu(), torch.ones_like(valid, dtype=torch.float))
        self._total += flat.numel()

    def compute_weights(self):
        """Пересчитывает веса на основе накопленных частот."""
        if self._total == 0:
            return

        freq = self._counts / max(self._total, 1)
        freq = freq.clamp(min=1e-8)

        if self.strategy == 'inverse':
            w = 1.0 / freq
        elif self.strategy == 'sqrt_inverse':
            w = 1.0 / freq.sqrt()
        elif self.strategy == 'log_inverse':
            w = 1.0 / (1 + freq).log()
        elif self.strategy == 'smoothed':
            max_freq = freq.max()
            w = (max_freq / freq) ** self.alpha
        else:
            w = torch.ones_like(freq)

        # Normalize so mean weight = 1
        w = w / w.mean()

        # Clip
        w = w.clamp(self.min_weight, self.max_weight)
        self._weights = w

    def get_weights(self, device=None):
        """Возвращает вектор весов."""
        w = self._weights
        if device is not None:
            w = w.to(device)
        return w

    def weighted_cross_entropy(self, logits, targets, ignore_index=-100):
        """
        Cross-entropy с частотным взвешиванием.

        Args:
            logits: (B, T, V) или (B, V)
            targets: (B, T) или (B,)
            ignore_index: игнорируемый индекс

        Returns:
            Tensor: scalar loss
        """
        weights = self._weights.to(logits.device)
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)

        return F.cross_entropy(logits, targets, weight=weights,
                               ignore_index=ignore_index)

    def get_frequency_stats(self):
        """Статистика частот."""
        if self._total == 0:
            return {'total': 0, 'n_seen': 0, 'n_unseen': 0}
        n_seen = (self._counts > 0).sum().item()
        return {
            'total': self._total,
            'n_seen': n_seen,
            'n_unseen': self.vocab_size - n_seen,
            'coverage': n_seen / self.vocab_size,
            'max_count': self._counts.max().item(),
            'min_nonzero_count': self._counts[self._counts > 0].min().item()
            if n_seen > 0 else 0,
        }

    def reset(self):
        self._counts.zero_()
        self._weights.fill_(1.0)
        self._total = 0
