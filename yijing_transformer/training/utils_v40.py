"""
v40 утилиты: Spectral Normalization, Gradient Histogram,
LR Probing, Mixed Precision Manager, Training Progress Estimator.

Spectral Normalization: нормализация по наибольшему сингулярному значению.
Ref: Miyato et al., "Spectral Normalization for GANs" (2018)

Gradient Histogram: распределение градиентов для диагностики.

LR Probing: проверка чувствительности к LR.

Mixed Precision Manager: управление AMP.
Ref: Micikevicius et al., "Mixed Precision Training" (2018)

Training Progress Estimator: ETA и прогресс обучения.
"""

import math
import time
import torch
import torch.nn as nn
from collections import deque


# ==================== Spectral Normalization Wrapper ====================

class SpectralNormWrapper:
    """
    Обёртка для спектральной нормализации.

    Нормализует вес W по наибольшему сингулярному значению:
    W_norm = W / σ(W), что ограничивает Lipschitz константу.

    Args:
        n_power_iterations: число итераций power method
    """
    def __init__(self, n_power_iterations=1):
        self.n_power_iterations = n_power_iterations

    def compute_spectral_norm(self, weight):
        """
        Вычисляет наибольшее сингулярное значение.

        Args:
            weight: Tensor (>= 2D)

        Returns:
            dict: {sigma, normalized_weight}
        """
        if weight.dim() < 2:
            return {'sigma': weight.norm().item(), 'normalized_weight': weight}

        h, w = weight.shape[0], weight.reshape(weight.shape[0], -1).shape[1]
        weight_mat = weight.reshape(h, w)

        # Power iteration
        u = torch.randn(h, device=weight.device)
        u = u / u.norm()

        for _ in range(self.n_power_iterations):
            v = weight_mat.t() @ u
            v = v / (v.norm() + 1e-12)
            u = weight_mat @ v
            u = u / (u.norm() + 1e-12)

        sigma = (u @ weight_mat @ v).item()
        normalized = weight / max(sigma, 1e-12)

        return {
            'sigma': sigma,
            'normalized_weight': normalized,
        }

    def apply_to_model(self, model):
        """
        Применяет спектральную нормализацию in-place.

        Args:
            model: nn.Module

        Returns:
            dict: {n_normalized, sigmas}
        """
        sigmas = {}
        n = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.dim() >= 2:
                result = self.compute_spectral_norm(module.weight.data)
                module.weight.data.copy_(result['normalized_weight'])
                sigmas[name] = result['sigma']
                n += 1

        return {'n_normalized': n, 'sigmas': sigmas}

    def get_spectral_norms(self, model):
        """
        Возвращает сингулярные значения всех слоёв.

        Args:
            model: nn.Module

        Returns:
            dict: {name: sigma}
        """
        norms = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.dim() >= 2:
                result = self.compute_spectral_norm(module.weight.data)
                norms[name] = result['sigma']
        return norms


# ==================== Gradient Histogram Tracker ====================

class GradientHistogramTracker:
    """
    Отслеживание распределения градиентов.

    Собирает статистики gradient distribution для
    диагностики обучения (vanishing/exploding grads).

    Args:
        n_bins: число бинов гистограммы
        track_every: как часто собирать (каждые N шагов)
    """
    def __init__(self, n_bins=50, track_every=10):
        self.n_bins = n_bins
        self.track_every = track_every
        self._step = 0
        self._history = []

    def track(self, model):
        """
        Собирает гистограмму градиентов.

        Args:
            model: nn.Module (после backward)

        Returns:
            dict or None: {histogram, stats, per_layer} или None если skip
        """
        self._step += 1
        if self._step % self.track_every != 0:
            return None

        all_grads = []
        per_layer = {}

        for name, p in model.named_parameters():
            if p.grad is not None:
                flat = p.grad.data.flatten()
                all_grads.append(flat)
                per_layer[name] = {
                    'mean': flat.mean().item(),
                    'std': flat.std().item() if flat.numel() > 1 else 0.0,
                    'min': flat.min().item(),
                    'max': flat.max().item(),
                    'abs_mean': flat.abs().mean().item(),
                    'zero_fraction': (flat == 0).float().mean().item(),
                }

        if not all_grads:
            return None

        all_flat = torch.cat(all_grads)
        stats = {
            'mean': all_flat.mean().item(),
            'std': all_flat.std().item(),
            'min': all_flat.min().item(),
            'max': all_flat.max().item(),
            'abs_mean': all_flat.abs().mean().item(),
            'n_params': all_flat.numel(),
        }

        # Histogram
        hist = torch.histc(all_flat, bins=self.n_bins)
        histogram = hist.tolist()

        result = {
            'histogram': histogram,
            'stats': stats,
            'per_layer': per_layer,
            'step': self._step,
        }

        self._history.append({
            'step': self._step,
            'mean': stats['mean'],
            'std': stats['std'],
            'abs_mean': stats['abs_mean'],
        })

        return result

    def get_trend(self, last_n=10):
        """
        Тренд gradient statistics.

        Returns:
            dict: {means, stds, abs_means}
        """
        recent = self._history[-last_n:]
        return {
            'means': [h['mean'] for h in recent],
            'stds': [h['std'] for h in recent],
            'abs_means': [h['abs_mean'] for h in recent],
        }

    def detect_issues(self):
        """
        Определяет проблемы с градиентами.

        Returns:
            list[str]: обнаруженные проблемы
        """
        issues = []
        if not self._history:
            return issues

        recent = self._history[-5:]
        avg_abs = sum(h['abs_mean'] for h in recent) / len(recent)

        if avg_abs < 1e-7:
            issues.append('vanishing_gradients')
        if avg_abs > 100:
            issues.append('exploding_gradients')

        # Check trend
        if len(self._history) >= 10:
            early = self._history[:5]
            late = self._history[-5:]
            early_abs = sum(h['abs_mean'] for h in early) / len(early)
            late_abs = sum(h['abs_mean'] for h in late) / len(late)
            if late_abs < early_abs * 0.01:
                issues.append('gradient_decay')

        return issues


# ==================== Learning Rate Probing ====================

class LRProbe:
    """
    Проверка чувствительности к текущему LR.

    Делает пробный шаг с разными LR и измеряет
    изменение loss для диагностики.

    Args:
        probe_factors: множители текущего LR для проб
    """
    def __init__(self, probe_factors=None):
        self.probe_factors = probe_factors or [0.1, 0.5, 1.0, 2.0, 5.0]

    def probe(self, model, optimizer, loss_fn):
        """
        Пробные шаги с разными LR.

        Args:
            model: nn.Module
            optimizer: torch optimizer
            loss_fn: callable() -> loss

        Returns:
            dict: {results, best_factor, current_lr}
        """
        import copy

        # Save state
        model_state = copy.deepcopy(model.state_dict())
        opt_state = copy.deepcopy(optimizer.state_dict())
        current_lr = optimizer.param_groups[0]['lr']

        results = []
        for factor in self.probe_factors:
            # Restore
            model.load_state_dict(copy.deepcopy(model_state))
            optimizer.load_state_dict(copy.deepcopy(opt_state))

            # Set probe LR
            probe_lr = current_lr * factor
            for pg in optimizer.param_groups:
                pg['lr'] = probe_lr

            # Compute loss before
            with torch.no_grad():
                loss_before = loss_fn().item()

            # One step
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()

            # Loss after
            with torch.no_grad():
                loss_after = loss_fn().item()

            results.append({
                'factor': factor,
                'lr': probe_lr,
                'loss_before': loss_before,
                'loss_after': loss_after,
                'improvement': loss_before - loss_after,
            })

        # Restore original
        model.load_state_dict(model_state)
        optimizer.load_state_dict(opt_state)

        best = max(results, key=lambda r: r['improvement'])

        return {
            'results': results,
            'best_factor': best['factor'],
            'current_lr': current_lr,
            'recommended_lr': current_lr * best['factor'],
        }


# ==================== Mixed Precision Manager ====================

class MixedPrecisionManager:
    """
    Менеджер mixed precision training.

    Управляет GradScaler, отслеживает overflow,
    и корректирует scale factor.

    Args:
        enabled: включён ли AMP
        initial_scale: начальный scale factor
        growth_interval: шагов между увеличением scale
    """
    def __init__(self, enabled=True, initial_scale=2**16,
                 growth_interval=2000):
        self.enabled = enabled
        self.initial_scale = initial_scale
        self.growth_interval = growth_interval
        self._overflow_count = 0
        self._total_steps = 0
        self._current_scale = initial_scale
        self._scale_history = []

    def get_scale(self):
        """Текущий scale factor."""
        return self._current_scale

    def update(self, overflow_detected):
        """
        Обновляет scale factor.

        Args:
            overflow_detected: bool

        Returns:
            dict: {scale, overflow, action}
        """
        self._total_steps += 1
        action = 'hold'

        if overflow_detected:
            self._overflow_count += 1
            self._current_scale = max(1.0, self._current_scale / 2)
            action = 'decrease'
        elif self._total_steps % self.growth_interval == 0:
            self._current_scale *= 2
            action = 'increase'

        self._scale_history.append({
            'step': self._total_steps,
            'scale': self._current_scale,
            'action': action,
        })

        return {
            'scale': self._current_scale,
            'overflow': overflow_detected,
            'action': action,
        }

    def scale_loss(self, loss):
        """
        Масштабирует loss для mixed precision.

        Args:
            loss: Tensor

        Returns:
            Tensor: scaled loss
        """
        if not self.enabled:
            return loss
        return loss * self._current_scale

    def unscale_grads(self, model):
        """
        Обратное масштабирование градиентов.

        Args:
            model: nn.Module
        """
        if not self.enabled:
            return
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.div_(self._current_scale)

    @property
    def overflow_rate(self):
        if self._total_steps == 0:
            return 0.0
        return self._overflow_count / self._total_steps

    def get_info(self):
        return {
            'enabled': self.enabled,
            'current_scale': self._current_scale,
            'overflow_count': self._overflow_count,
            'overflow_rate': self.overflow_rate,
            'total_steps': self._total_steps,
        }


# ==================== Training Progress Estimator ====================

class TrainingProgressEstimator:
    """
    Оценка прогресса обучения и ETA.

    Отслеживает скорость обучения, предсказывает
    оставшееся время и целевой loss.

    Args:
        total_steps: общее число шагов
        target_loss: целевой loss (опционально)
    """
    def __init__(self, total_steps=10000, target_loss=None):
        self.total_steps = total_steps
        self.target_loss = target_loss
        self._start_time = None
        self._step_times = deque(maxlen=100)
        self._loss_history = deque(maxlen=200)
        self._current_step = 0

    def update(self, loss_value=None):
        """
        Обновляет прогресс.

        Args:
            loss_value: текущий loss

        Returns:
            dict: {progress, eta_seconds, steps_per_sec,
                   estimated_final_loss}
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        self._current_step += 1

        if self._step_times:
            step_time = now - self._step_times[-1]
            self._step_times.append(now)
        else:
            self._step_times.append(now)
            step_time = 0

        if loss_value is not None:
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            self._loss_history.append(loss_value)

        # Progress
        progress = min(self._current_step / max(self.total_steps, 1), 1.0)

        # Steps per second
        elapsed = now - self._start_time
        steps_per_sec = self._current_step / max(elapsed, 1e-6)

        # ETA
        remaining_steps = max(0, self.total_steps - self._current_step)
        eta_seconds = remaining_steps / max(steps_per_sec, 1e-6)

        # Estimated final loss (linear extrapolation)
        estimated_final = None
        if len(self._loss_history) >= 10:
            recent = list(self._loss_history)
            half = len(recent) // 2
            first_half = sum(recent[:half]) / half
            second_half = sum(recent[half:]) / (len(recent) - half)
            trend = second_half - first_half
            remaining_frac = 1.0 - progress
            estimated_final = second_half + trend * remaining_frac * 2

        return {
            'progress': progress,
            'progress_pct': progress * 100,
            'eta_seconds': eta_seconds,
            'eta_formatted': self._format_time(eta_seconds),
            'steps_per_sec': steps_per_sec,
            'elapsed_seconds': elapsed,
            'elapsed_formatted': self._format_time(elapsed),
            'current_step': self._current_step,
            'estimated_final_loss': estimated_final,
        }

    def _format_time(self, seconds):
        """Форматирует время."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"

    def will_reach_target(self):
        """
        Предсказывает, достигнет ли обучение target loss.

        Returns:
            dict: {reachable, estimated_steps, confidence}
        """
        if self.target_loss is None or len(self._loss_history) < 20:
            return {'reachable': None, 'estimated_steps': None, 'confidence': 0}

        recent = list(self._loss_history)
        current = sum(recent[-10:]) / 10
        earlier = sum(recent[:10]) / 10

        if current >= earlier:
            return {'reachable': False, 'estimated_steps': None, 'confidence': 0.8}

        # Extrapolate
        improvement_per_step = (earlier - current) / len(recent)
        if improvement_per_step <= 0:
            return {'reachable': False, 'estimated_steps': None, 'confidence': 0.5}

        remaining_improvement = current - self.target_loss
        if remaining_improvement <= 0:
            return {'reachable': True, 'estimated_steps': 0, 'confidence': 1.0}

        estimated_steps = remaining_improvement / improvement_per_step

        return {
            'reachable': estimated_steps < self.total_steps * 2,
            'estimated_steps': int(estimated_steps),
            'confidence': min(0.9, len(recent) / 100),
        }

    def get_summary(self):
        return {
            'current_step': self._current_step,
            'total_steps': self.total_steps,
            'progress_pct': self._current_step / max(self.total_steps, 1) * 100,
            'current_loss': self._loss_history[-1] if self._loss_history else None,
        }
