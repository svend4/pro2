"""
v41 утилиты: Activation Checkpointing, Parameter Freezer,
Loss Landscape Probe, Optimizer State Inspector, Batch Size Finder.

Activation Checkpointing: пересчёт активаций при backward вместо хранения.
Ref: Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)

Parameter Freezer: выборочная заморозка параметров для transfer learning.

Loss Landscape Probe: визуализация loss surface в окрестности текущих весов.
Ref: Li et al., "Visualizing the Loss Landscape of Neural Nets" (2018)

Optimizer State Inspector: инспекция momentum, variance и др.

Batch Size Finder: подбор максимального batch size для GPU.
Ref: McCandlish et al., "An Empirical Model of Large-Batch Training" (2018)
"""

import math
import copy
import torch
import torch.nn as nn
from collections import OrderedDict


# ==================== Activation Checkpointing Manager ====================

class ActivationCheckpointManager:
    """
    Управление gradient checkpointing.

    Выборочно включает checkpointing для модулей,
    снижая потребление памяти за счёт пересчёта.

    Args:
        strategy: 'all', 'every_n', 'selective'
        every_n: checkpoint каждые N слоёв (для strategy='every_n')
    """
    def __init__(self, strategy='every_n', every_n=2):
        self.strategy = strategy
        self.every_n = every_n
        self._checkpointed = []
        self._memory_saved_estimate = 0

    def apply(self, model, layer_names=None):
        """
        Применяет checkpointing к модели.

        Args:
            model: nn.Module
            layer_names: список имён слоёв (для 'selective')

        Returns:
            dict: {n_checkpointed, layers}
        """
        self._checkpointed = []
        modules = list(model.named_modules())

        if self.strategy == 'all':
            for name, module in modules:
                if self._is_checkpointable(module):
                    self._enable_checkpoint(name, module)

        elif self.strategy == 'every_n':
            idx = 0
            for name, module in modules:
                if self._is_checkpointable(module):
                    if idx % self.every_n == 0:
                        self._enable_checkpoint(name, module)
                    idx += 1

        elif self.strategy == 'selective' and layer_names:
            name_set = set(layer_names)
            for name, module in modules:
                if name in name_set:
                    self._enable_checkpoint(name, module)

        return {
            'n_checkpointed': len(self._checkpointed),
            'layers': self._checkpointed,
        }

    def _is_checkpointable(self, module):
        """Модуль подходит для checkpointing."""
        return (
            isinstance(module, (nn.TransformerEncoderLayer,
                                nn.TransformerDecoderLayer,
                                nn.Sequential))
            and len(list(module.parameters())) > 0
        )

    def _enable_checkpoint(self, name, module):
        """Включает checkpoint flag."""
        module._checkpoint_enabled = True
        n_params = sum(p.numel() for p in module.parameters())
        self._checkpointed.append(name)
        self._memory_saved_estimate += n_params * 4  # ~4 bytes per activation

    def estimate_memory_saving(self):
        """
        Оценка экономии памяти.

        Returns:
            dict: {saved_bytes, saved_mb}
        """
        return {
            'saved_bytes': self._memory_saved_estimate,
            'saved_mb': self._memory_saved_estimate / (1024 * 1024),
        }

    def get_info(self):
        return {
            'strategy': self.strategy,
            'n_checkpointed': len(self._checkpointed),
            'layers': self._checkpointed,
        }


# ==================== Parameter Freezer ====================

class ParameterFreezer:
    """
    Выборочная заморозка/разморозка параметров.

    Для transfer learning, fine-tuning определённых слоёв,
    и поэтапного обучения.

    Args:
        model: nn.Module
    """
    def __init__(self, model):
        self.model = model
        self._frozen_params = set()
        self._freeze_history = []

    def freeze(self, patterns=None, except_patterns=None):
        """
        Замораживает параметры по паттернам.

        Args:
            patterns: список подстрок в имени (freeze matching)
            except_patterns: исключения (не замораживать)

        Returns:
            dict: {n_frozen, n_total, frozen_params}
        """
        frozen_names = []
        for name, param in self.model.named_parameters():
            should_freeze = False

            if patterns is None:
                should_freeze = True
            else:
                for p in patterns:
                    if p in name:
                        should_freeze = True
                        break

            if except_patterns and should_freeze:
                for ep in except_patterns:
                    if ep in name:
                        should_freeze = False
                        break

            if should_freeze:
                param.requires_grad = False
                self._frozen_params.add(name)
                frozen_names.append(name)

        self._freeze_history.append(('freeze', frozen_names))

        total = sum(1 for _ in self.model.parameters())
        return {
            'n_frozen': len(self._frozen_params),
            'n_total': total,
            'frozen_params': frozen_names,
        }

    def unfreeze(self, patterns=None):
        """
        Размораживает параметры.

        Args:
            patterns: список подстрок (unfreeze matching), None = all

        Returns:
            dict: {n_unfrozen, names}
        """
        unfrozen = []
        for name, param in self.model.named_parameters():
            if name not in self._frozen_params:
                continue

            should_unfreeze = False
            if patterns is None:
                should_unfreeze = True
            else:
                for p in patterns:
                    if p in name:
                        should_unfreeze = True
                        break

            if should_unfreeze:
                param.requires_grad = True
                self._frozen_params.discard(name)
                unfrozen.append(name)

        self._freeze_history.append(('unfreeze', unfrozen))
        return {'n_unfrozen': len(unfrozen), 'names': unfrozen}

    def get_trainable_count(self):
        """Число обучаемых параметров."""
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            'trainable': trainable,
            'frozen': total - trainable,
            'total': total,
            'trainable_pct': trainable / max(total, 1) * 100,
        }

    def get_status(self):
        return {
            'n_frozen': len(self._frozen_params),
            'frozen_params': list(self._frozen_params),
            **self.get_trainable_count(),
        }


# ==================== Loss Landscape Probe ====================

class LossLandscapeProbe:
    """
    Исследование loss surface вокруг текущей точки.

    Делает пробы loss в разных направлениях от текущих весов
    для понимания геометрии loss landscape.

    Args:
        n_directions: число направлений для probing
        step_sizes: размеры шагов вдоль каждого направления
    """
    def __init__(self, n_directions=2, step_sizes=None):
        self.n_directions = n_directions
        self.step_sizes = step_sizes or [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]

    def probe_1d(self, model, loss_fn, direction=None):
        """
        1D срез loss landscape.

        Args:
            model: nn.Module
            loss_fn: callable() -> scalar loss
            direction: dict {name: tensor} или None (random)

        Returns:
            dict: {steps, losses, curvature}
        """
        original_state = copy.deepcopy(model.state_dict())

        # Random direction if not specified
        if direction is None:
            direction = {}
            for name, param in model.named_parameters():
                d = torch.randn_like(param)
                d = d / (d.norm() + 1e-12) * param.data.norm()
                direction[name] = d

        losses = []
        for alpha in self.step_sizes:
            # Move in direction
            for name, param in model.named_parameters():
                if name in direction:
                    param.data.copy_(original_state[name] + alpha * direction[name])

            with torch.no_grad():
                loss = loss_fn()
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
            losses.append(loss)

        # Restore
        model.load_state_dict(original_state)

        # Estimate curvature from central differences
        curvature = None
        if 0.0 in self.step_sizes:
            center_idx = self.step_sizes.index(0.0)
            if center_idx > 0 and center_idx < len(self.step_sizes) - 1:
                h = self.step_sizes[center_idx + 1] - self.step_sizes[center_idx]
                if h > 0:
                    curvature = (losses[center_idx + 1] - 2 * losses[center_idx]
                                 + losses[center_idx - 1]) / (h * h)

        return {
            'steps': self.step_sizes,
            'losses': losses,
            'curvature': curvature,
        }

    def probe_2d(self, model, loss_fn, n_points=5):
        """
        2D срез loss landscape.

        Args:
            model: nn.Module
            loss_fn: callable() -> scalar loss
            n_points: число точек на каждую ось

        Returns:
            dict: {grid_x, grid_y, loss_grid}
        """
        original_state = copy.deepcopy(model.state_dict())

        # Two random directions
        dir1, dir2 = {}, {}
        for name, param in model.named_parameters():
            d1 = torch.randn_like(param)
            d1 = d1 / (d1.norm() + 1e-12) * param.data.norm()
            dir1[name] = d1
            d2 = torch.randn_like(param)
            d2 = d2 / (d2.norm() + 1e-12) * param.data.norm()
            dir2[name] = d2

        alphas = [i / (n_points - 1) * 2 - 1 for i in range(n_points)]
        loss_grid = []

        for a1 in alphas:
            row = []
            for a2 in alphas:
                for name, param in model.named_parameters():
                    if name in dir1:
                        param.data.copy_(
                            original_state[name]
                            + a1 * dir1[name]
                            + a2 * dir2[name]
                        )
                with torch.no_grad():
                    loss = loss_fn()
                    if isinstance(loss, torch.Tensor):
                        loss = loss.item()
                row.append(loss)
            loss_grid.append(row)

        model.load_state_dict(original_state)

        return {
            'grid_x': alphas,
            'grid_y': alphas,
            'loss_grid': loss_grid,
        }

    def estimate_sharpness(self, model, loss_fn, epsilon=0.01):
        """
        Оценка sharpness (Keskar et al., 2017).

        Args:
            model: nn.Module
            loss_fn: callable() -> loss
            epsilon: размер возмущения

        Returns:
            dict: {base_loss, max_loss, sharpness}
        """
        original_state = copy.deepcopy(model.state_dict())

        with torch.no_grad():
            base_loss = loss_fn()
            if isinstance(base_loss, torch.Tensor):
                base_loss = base_loss.item()

        max_loss = base_loss
        n_probes = 10
        for _ in range(n_probes):
            for name, param in model.named_parameters():
                noise = torch.randn_like(param) * epsilon
                param.data.copy_(original_state[name] + noise)
            with torch.no_grad():
                loss = loss_fn()
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
            max_loss = max(max_loss, loss)

        model.load_state_dict(original_state)

        return {
            'base_loss': base_loss,
            'max_loss': max_loss,
            'sharpness': (max_loss - base_loss) / (1 + base_loss),
        }


# ==================== Optimizer State Inspector ====================

class OptimizerStateInspector:
    """
    Инспекция состояния оптимизатора.

    Извлекает и анализирует momentum, variance, step counts
    из внутреннего состояния Adam/AdamW/SGD.

    Args:
        optimizer: torch optimizer
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def inspect(self):
        """
        Полная инспекция состояния.

        Returns:
            dict: {param_groups, state_summary, n_params_with_state}
        """
        groups_info = []
        for i, pg in enumerate(self.optimizer.param_groups):
            info = {k: v for k, v in pg.items() if k != 'params'}
            info['n_params'] = len(pg['params'])
            groups_info.append(info)

        n_with_state = 0
        momentum_stats = []
        variance_stats = []

        for param_id, state in self.optimizer.state.items():
            if not state:
                continue
            n_with_state += 1

            # Adam-style states
            if 'exp_avg' in state:
                m = state['exp_avg']
                momentum_stats.append({
                    'mean': m.mean().item(),
                    'std': m.std().item() if m.numel() > 1 else 0.0,
                    'abs_mean': m.abs().mean().item(),
                })

            if 'exp_avg_sq' in state:
                v = state['exp_avg_sq']
                variance_stats.append({
                    'mean': v.mean().item(),
                    'max': v.max().item(),
                    'min': v.min().item(),
                })

        summary = {
            'n_params_with_state': n_with_state,
        }
        if momentum_stats:
            summary['momentum'] = {
                'avg_abs_mean': sum(s['abs_mean'] for s in momentum_stats) / len(momentum_stats),
                'avg_std': sum(s['std'] for s in momentum_stats) / len(momentum_stats),
            }
        if variance_stats:
            summary['variance'] = {
                'avg_mean': sum(s['mean'] for s in variance_stats) / len(variance_stats),
                'max_max': max(s['max'] for s in variance_stats),
            }

        return {
            'param_groups': groups_info,
            'state_summary': summary,
            'n_params_with_state': n_with_state,
        }

    def get_effective_lr(self):
        """
        Эффективный LR с учётом Adam bias correction.

        Returns:
            list[dict]: per-group effective LR info
        """
        results = []
        for i, pg in enumerate(self.optimizer.param_groups):
            lr = pg['lr']
            effective = lr

            # Check if Adam
            for p in pg['params']:
                state = self.optimizer.state.get(p, {})
                if 'step' in state:
                    step = state['step']
                    if isinstance(step, torch.Tensor):
                        step = step.item()
                    step = int(step)
                    beta1 = pg.get('betas', (0.9, 0.999))[0]
                    beta2 = pg.get('betas', (0.9, 0.999))[1]
                    if step > 0:
                        bc1 = 1 - beta1 ** step
                        bc2 = 1 - beta2 ** step
                        effective = lr * math.sqrt(bc2) / bc1
                    break

            results.append({
                'group': i,
                'base_lr': lr,
                'effective_lr': effective,
            })

        return results

    def detect_anomalies(self):
        """
        Обнаружение аномалий в состоянии.

        Returns:
            list[str]: аномалии
        """
        anomalies = []

        for param_id, state in self.optimizer.state.items():
            if 'exp_avg_sq' in state:
                v = state['exp_avg_sq']
                if torch.isnan(v).any():
                    anomalies.append('nan_in_variance')
                    break
                if v.max().item() > 1e6:
                    anomalies.append('large_variance')
                    break

            if 'exp_avg' in state:
                m = state['exp_avg']
                if torch.isnan(m).any():
                    anomalies.append('nan_in_momentum')
                    break

        return anomalies


# ==================== Batch Size Finder ====================

class BatchSizeFinder:
    """
    Автоматический подбор batch size.

    Бинарным поиском находит максимальный batch size,
    при котором не происходит OOM.

    Args:
        max_batch_size: верхняя граница поиска
        min_batch_size: нижняя граница
    """
    def __init__(self, max_batch_size=256, min_batch_size=1):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

    def find(self, create_batch_fn, model, loss_fn):
        """
        Находит оптимальный batch size.

        Args:
            create_batch_fn: callable(batch_size) -> (input, target)
            model: nn.Module
            loss_fn: callable(model_output, target) -> loss

        Returns:
            dict: {optimal_batch_size, tested}
        """
        tested = []
        lo, hi = self.min_batch_size, self.max_batch_size
        best = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            success = self._try_batch(mid, create_batch_fn, model, loss_fn)
            tested.append({'batch_size': mid, 'success': success})

            if success:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return {
            'optimal_batch_size': best,
            'tested': tested,
        }

    def _try_batch(self, batch_size, create_batch_fn, model, loss_fn):
        """Пробует batch size, возвращает True если OK."""
        try:
            batch_input, batch_target = create_batch_fn(batch_size)
            output = model(batch_input)
            if isinstance(output, tuple):
                output = output[0]
            loss = loss_fn(output, batch_target)
            loss.backward()
            model.zero_grad()
            return True
        except RuntimeError:
            model.zero_grad()
            return False

    def find_efficient(self, create_batch_fn, model, loss_fn, time_budget=5.0):
        """
        Находит batch size, оптимальный по throughput.

        Args:
            create_batch_fn: callable(batch_size) -> (input, target)
            model: nn.Module
            loss_fn: callable(output, target) -> loss
            time_budget: секунд на каждый замер

        Returns:
            dict: {best_batch_size, results}
        """
        import time

        candidates = [2 ** i for i in range(
            int(math.log2(self.min_batch_size)),
            int(math.log2(self.max_batch_size)) + 1
        )]

        results = []
        for bs in candidates:
            success = self._try_batch(bs, create_batch_fn, model, loss_fn)
            if not success:
                break

            # Measure throughput
            start = time.time()
            n_iters = 0
            while time.time() - start < 0.5:  # Quick measurement
                try:
                    inp, tgt = create_batch_fn(bs)
                    out = model(inp)
                    if isinstance(out, tuple):
                        out = out[0]
                    l = loss_fn(out, tgt)
                    l.backward()
                    model.zero_grad()
                    n_iters += 1
                except RuntimeError:
                    break

            elapsed = time.time() - start
            throughput = (bs * n_iters) / max(elapsed, 1e-6)
            results.append({
                'batch_size': bs,
                'throughput': throughput,
                'time_per_batch': elapsed / max(n_iters, 1),
            })

        best = max(results, key=lambda r: r['throughput']) if results else None

        return {
            'best_batch_size': best['batch_size'] if best else self.min_batch_size,
            'results': results,
        }
