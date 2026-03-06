"""
v20 утилиты: DoRA, Sparse Attention, Gradient Vaccine, Cyclic Batch Size, Loss Landscape.

DoRA: Weight-Decomposed Low-Rank Adaptation — разделяет magnitude и direction.
Ref: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024)

Sparse Attention: BigBird-style сочетание random, window и global attention.
Ref: Zaheer et al., "Big Bird: Transformers for Longer Sequences" (2020)

Gradient Vaccine: cosine-similarity alignment градиентов между workers.
Ref: Lin et al., "Gradient Vaccine" (2021)

Cyclic Batch Size: увеличение batch size вместо уменьшения LR.
Ref: Smith et al., "Don't Decay the Learning Rate, Increase the Batch Size" (2018)

Loss Landscape Probe: измерение sharpness/flatness минимума.
Ref: Foret et al., "SAM" (2021) / Li et al., "Visualizing Loss Landscape" (2018)
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== DoRA ====================

class DoRALinear(nn.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation.

    Разделяет вес на magnitude (m) и direction (V/||V||):
        W' = m * (W + BA) / ||W + BA||

    Где BA — стандартный LoRA, а m — обучаемый magnitude вектор.
    Преимущество: лучше LoRA при том же числе параметров.

    Args:
        base_linear: исходный nn.Linear
        rank: ранг LoRA
        alpha: scaling factor
    """
    def __init__(self, base_linear, rank=8, alpha=16.0):
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Замороженные оригинальные веса
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        self.bias = None
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)

        # LoRA компоненты
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        # Magnitude вектор (DoRA specific)
        with torch.no_grad():
            col_norms = self.weight.norm(dim=1)
        self.magnitude = nn.Parameter(col_norms)

    def forward(self, x):
        # W + scaling * B @ A
        lora_weight = self.scaling * (self.lora_B @ self.lora_A)
        adapted_weight = self.weight + lora_weight

        # Normalize direction, apply magnitude
        col_norms = adapted_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        direction = adapted_weight / col_norms
        final_weight = self.magnitude.unsqueeze(1) * direction

        out = F.linear(x, final_weight, self.bias)
        return out

    def num_trainable_params(self):
        return (self.lora_A.numel() + self.lora_B.numel() +
                self.magnitude.numel())


def apply_dora(model, rank=8, alpha=16.0, targets=None):
    """
    Применяет DoRA ко всем Linear слоям модели.

    Args:
        model: nn.Module
        rank: LoRA rank
        alpha: scaling
        targets: list[str] — имена целевых модулей (None = все Linear)

    Returns:
        list[DoRALinear]: созданные DoRA модули
    """
    dora_modules = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if targets and not any(t in name for t in targets):
            continue

        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model
        if parent_name:
            for part in parent_name.split('.'):
                parent = getattr(parent, part)

        dora = DoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, child_name, dora)
        dora_modules.append(dora)

    return dora_modules


# ==================== Sparse Attention ====================

class SparseAttentionMask:
    """
    BigBird-style sparse attention mask generator.

    Комбинирует три типа attention:
    1. Window (local): каждый токен видит w соседей
    2. Random: r случайных связей
    3. Global: g токенов видят всё и видны всем

    Args:
        seq_len: длина последовательности
        window_size: размер локального окна
        n_random: число random связей per token
        n_global: число global токенов (первые n)
    """
    def __init__(self, seq_len, window_size=3, n_random=2, n_global=1):
        self.seq_len = seq_len
        self.window_size = window_size
        self.n_random = n_random
        self.n_global = n_global

    def get_mask(self):
        """
        Генерирует sparse attention mask.

        Returns:
            mask: (seq_len, seq_len) bool tensor — True = attend
        """
        T = self.seq_len
        mask = torch.zeros(T, T, dtype=torch.bool)

        # 1. Window attention
        for i in range(T):
            start = max(0, i - self.window_size)
            end = min(T, i + self.window_size + 1)
            mask[i, start:end] = True

        # 2. Global tokens (first n_global tokens see & are seen by all)
        mask[:self.n_global, :] = True
        mask[:, :self.n_global] = True

        # 3. Random connections
        for i in range(T):
            available = (~mask[i]).nonzero(as_tuple=True)[0]
            if len(available) > 0:
                n = min(self.n_random, len(available))
                perm = torch.randperm(len(available))[:n]
                mask[i, available[perm]] = True

        return mask

    def get_additive_mask(self, bool_mask=None):
        """
        Возвращает additive mask для attention scores.

        Args:
            bool_mask: готовая bool маска (None = генерируем новую)

        Returns:
            mask: (seq_len, seq_len) float — 0.0 где attend, -inf где block
        """
        if bool_mask is None:
            bool_mask = self.get_mask()
        additive = torch.zeros(self.seq_len, self.seq_len, dtype=torch.float)
        additive[~bool_mask] = float('-inf')
        return additive

    def sparsity_ratio(self):
        """Доля нулевых элементов в маске."""
        mask = self.get_mask()
        return 1.0 - mask.float().mean().item()


# ==================== Gradient Vaccine ====================

class GradientVaccine:
    """
    Gradient Vaccine: alignment градиентов по cosine similarity.

    Для distributed/multi-task: проверяет согласованность
    градиентов и корректирует конфликтующие.

    Args:
        threshold: минимальный cosine similarity для согласия
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.history = []

    def align_gradients(self, grad_list):
        """
        Выравнивает список градиентов.

        Args:
            grad_list: list[Tensor] — flat gradients от разных источников

        Returns:
            aligned: Tensor — скорректированный градиент
        """
        grads = torch.stack(grad_list)  # (N, D)
        avg = grads.mean(dim=0)

        # Cosine similarity каждого с средним
        cosines = F.cosine_similarity(grads, avg.unsqueeze(0), dim=1)
        self.history.append(cosines.tolist())

        # Маскируем конфликтующие
        aligned_grads = []
        for i, (g, cos) in enumerate(zip(grads, cosines)):
            if cos.item() >= self.threshold:
                aligned_grads.append(g)
            else:
                # Проецируем на среднее направление
                proj = (g @ avg) / (avg.norm() ** 2 + 1e-8) * avg
                # Смесь проекции и оригинала
                aligned_grads.append(proj)

        return torch.stack(aligned_grads).mean(dim=0)

    def get_agreement_stats(self):
        """Статистика согласованности."""
        if not self.history:
            return {'mean_cosine': 0.0, 'min_cosine': 0.0, 'n_steps': 0}
        all_cos = [c for step in self.history for c in step]
        return {
            'mean_cosine': sum(all_cos) / len(all_cos),
            'min_cosine': min(all_cos),
            'n_steps': len(self.history),
        }

    def reset(self):
        self.history = []


# ==================== Cyclic Batch Size Scheduler ====================

class CyclicBatchScheduler:
    """
    Cyclic Batch Size Scheduler.

    Увеличивает batch size вместо уменьшения LR.
    Эквивалентно cosine decay LR, но через batch size.

    Args:
        initial_batch_size: начальный batch size
        max_batch_size: максимальный batch size
        total_steps: общее число шагов
        warmup_steps: шагов warmup (batch size = initial)
    """
    def __init__(self, initial_batch_size, max_batch_size, total_steps,
                 warmup_steps=0):
        self.initial = initial_batch_size
        self.max_bs = max_batch_size
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        """Один шаг scheduler."""
        self._step += 1
        return self.get_batch_size()

    def get_batch_size(self):
        """Текущий batch size."""
        if self._step <= self.warmup_steps:
            return self.initial

        progress = (self._step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)

        # Cosine schedule: от initial до max
        bs = self.initial + (self.max_bs - self.initial) * (
            1 - math.cos(math.pi * progress)
        ) / 2

        # Округляем до кратного initial (для dataloader)
        bs = max(self.initial, int(bs / self.initial) * self.initial)
        return min(bs, self.max_bs)

    def get_grad_accum_steps(self, base_batch_size=None):
        """
        Количество шагов gradient accumulation для текущего batch size.

        Args:
            base_batch_size: базовый batch size (None = initial)
        """
        base = base_batch_size or self.initial
        return max(1, self.get_batch_size() // base)

    def state_dict(self):
        return {'step': self._step}

    def load_state_dict(self, state):
        self._step = state['step']


# ==================== Loss Landscape Probe ====================

class LossLandscapeProbe:
    """
    Loss Landscape Probe: измеряет sharpness/flatness минимума.

    Добавляет случайные perturbations к весам модели
    и измеряет, как сильно меняется loss. Плоские минимумы
    (маленький delta loss) → лучшая генерализация.

    Args:
        model: nn.Module
        loss_fn: callable(model) → loss
    """
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    @torch.no_grad()
    def measure_sharpness(self, epsilon=0.01, n_samples=5):
        """
        Измеряет sharpness как avg(|L(w+e) - L(w)|).

        Args:
            epsilon: масштаб perturbation
            n_samples: число случайных направлений

        Returns:
            dict: {sharpness, base_loss, perturbed_losses}
        """
        base_loss = self.loss_fn().item()

        # Сохраняем оригинальные веса
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        perturbed_losses = []
        for _ in range(n_samples):
            # Random perturbation
            for p in self.model.parameters():
                noise = torch.randn_like(p) * epsilon
                p.data.add_(noise)

            loss = self.loss_fn().item()
            perturbed_losses.append(loss)

            # Restore
            self.model.load_state_dict(original_state)

        deltas = [abs(pl - base_loss) for pl in perturbed_losses]
        sharpness = sum(deltas) / len(deltas)

        return {
            'sharpness': sharpness,
            'base_loss': base_loss,
            'perturbed_losses': perturbed_losses,
            'max_delta': max(deltas),
            'min_delta': min(deltas),
        }

    @torch.no_grad()
    def directional_sharpness(self, direction=None, n_points=11,
                               epsilon_range=0.05):
        """
        1D slice через loss landscape.

        Args:
            direction: направление (None = random)
            n_points: число точек
            epsilon_range: диапазон [-eps, +eps]

        Returns:
            dict: {epsilons, losses}
        """
        if direction is None:
            direction = {}
            for name, p in self.model.named_parameters():
                direction[name] = torch.randn_like(p)
                direction[name] /= direction[name].norm()

        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        epsilons = torch.linspace(-epsilon_range, epsilon_range, n_points).tolist()
        losses = []

        for eps in epsilons:
            for name, p in self.model.named_parameters():
                if name in direction:
                    p.data.copy_(original_state[name] + eps * direction[name])

            loss = self.loss_fn().item()
            losses.append(loss)
            self.model.load_state_dict(original_state)

        return {
            'epsilons': epsilons,
            'losses': losses,
            'curvature': self._estimate_curvature(epsilons, losses),
        }

    def _estimate_curvature(self, epsilons, losses):
        """Оценка кривизны через finite differences."""
        mid = len(epsilons) // 2
        if mid <= 0 or mid >= len(epsilons) - 1:
            return 0.0
        h = epsilons[mid + 1] - epsilons[mid]
        if h == 0:
            return 0.0
        return (losses[mid + 1] - 2 * losses[mid] + losses[mid - 1]) / (h * h)
