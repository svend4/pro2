"""
v27 утилиты: SAM, Dynamic Temperature, Gradient Projection,
Training Metrics Dashboard, Sequence Packing.

SAM: Sharpness-Aware Minimization для плоских минимумов.
Ref: Foret et al., "Sharpness-Aware Minimization" (2021)

Dynamic Temperature: адаптивная температура softmax.

Gradient Projection: проекция градиентов для continual learning.
Ref: Zeng et al., "Continual Learning via Gradient Projection" (2021)

Training Metrics Dashboard: агрегация всех метрик обучения.

Sequence Packing: упаковка коротких последовательностей для эффективности.
Ref: Krell et al., "Efficient Sequence Packing" (2021)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque


# ==================== Sharpness-Aware Minimization ====================

class SAM:
    """
    Sharpness-Aware Minimization.

    Ищет параметры в плоских минимумах через двухшаговую оптимизацию:
    1. ε = ρ * grad / |grad| (perturbation)
    2. θ' = θ + ε → compute grad at θ'
    3. Update θ using grad at θ'

    Args:
        optimizer: базовый optimizer
        rho: радиус perturbation (0.05 типично)
    """
    def __init__(self, optimizer, rho=0.05):
        self.optimizer = optimizer
        self.rho = rho
        self._param_backup = {}

    def _grad_norm(self):
        """Норма текущих градиентов."""
        norm = 0.0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += p.grad.data.norm(2).item() ** 2
        return math.sqrt(norm)

    def first_step(self):
        """
        Первый шаг: perturbation в направлении градиента.
        Вызывать после первого backward().
        """
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)

        self._param_backup = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self._param_backup[p] = p.data.clone()
                e_w = p.grad.data * scale
                p.data.add_(e_w)

    def second_step(self):
        """
        Второй шаг: восстановить веса и сделать обычный шаг.
        Вызывать после второго backward().
        """
        for p, backup in self._param_backup.items():
            p.data.copy_(backup)

        self.optimizer.step()
        self._param_backup = {}

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


# ==================== Dynamic Temperature Scaling ====================

class DynamicTemperature:
    """
    Адаптивная температура softmax.

    Автоматически настраивает температуру на основе
    энтропии выходного распределения.

    Args:
        initial_temp: начальная температура
        min_temp: минимальная
        max_temp: максимальная
        target_entropy: целевая энтропия (None = auto)
        adapt_rate: скорость адаптации
    """
    def __init__(self, initial_temp=1.0, min_temp=0.1, max_temp=5.0,
                 target_entropy=None, adapt_rate=0.01):
        self.temperature = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.target_entropy = target_entropy
        self.adapt_rate = adapt_rate
        self.history = []

    def apply(self, logits):
        """
        Применяет температуру к logits.

        Args:
            logits: (B, T, V) или (B, V)

        Returns:
            scaled logits
        """
        return logits / self.temperature

    def compute_entropy(self, logits):
        """Вычисляет энтропию softmax распределения."""
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1).mean()
        return entropy.item()

    def adapt(self, logits):
        """
        Адаптирует температуру на основе текущих logits.

        Args:
            logits: (B, T, V)

        Returns:
            dict: {temperature, entropy, target_entropy}
        """
        entropy = self.compute_entropy(logits)

        # Auto target: log(V) * 0.7
        if self.target_entropy is None:
            V = logits.size(-1)
            target = math.log(V) * 0.7
        else:
            target = self.target_entropy

        # If entropy too high → decrease temp (sharper)
        # If entropy too low → increase temp (smoother)
        if entropy > target:
            self.temperature -= self.adapt_rate
        else:
            self.temperature += self.adapt_rate

        self.temperature = max(self.min_temp, min(self.max_temp, self.temperature))

        result = {
            'temperature': self.temperature,
            'entropy': entropy,
            'target_entropy': target,
        }
        self.history.append(result)
        return result

    def reset(self):
        self.temperature = 1.0
        self.history.clear()


# ==================== Gradient Projection ====================

class GradientProjection:
    """
    Gradient Projection для continual learning.

    Запоминает "важные" направления предыдущих задач
    и проецирует новые градиенты на ортогональное подпространство.
    Предотвращает catastrophic forgetting.

    Args:
        model: nn.Module
        n_components: число компонент для запоминания на задачу
    """
    def __init__(self, model, n_components=10):
        self.model = model
        self.n_components = n_components
        self.projection_bases = {}  # param_name → (n_tasks * n_comp, param_size)

    def memorize_task(self, task_id, data_loader_fn, n_batches=10):
        """
        Запоминает важные направления для задачи.

        Args:
            task_id: идентификатор задачи
            data_loader_fn: callable() → (input, target)
            n_batches: число батчей для сбора градиентов
        """
        grad_accum = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                grad_accum[name] = []

        # Collect gradients
        for _ in range(n_batches):
            self.model.zero_grad()
            x, y = data_loader_fn()
            output = self.model(x, y) if y is not None else self.model(x)
            if isinstance(output, tuple) and len(output) >= 2:
                loss = output[1]
            else:
                loss = output.sum() if isinstance(output, torch.Tensor) else output[0].sum()
            loss.backward()

            for name, p in self.model.named_parameters():
                if p.grad is not None and name in grad_accum:
                    grad_accum[name].append(p.grad.data.flatten().clone())

        # SVD to find important directions
        for name, grads in grad_accum.items():
            if not grads:
                continue
            G = torch.stack(grads)  # (n_batches, param_size)
            n_comp = min(self.n_components, G.size(0), G.size(1))
            U, S, Vt = torch.linalg.svd(G, full_matrices=False)
            basis = Vt[:n_comp]  # (n_comp, param_size)

            if name not in self.projection_bases:
                self.projection_bases[name] = basis
            else:
                self.projection_bases[name] = torch.cat(
                    [self.projection_bases[name], basis], dim=0
                )

    def project_gradients(self):
        """
        Проецирует текущие градиенты на ортогональное подпространство.

        Returns:
            dict: {n_projected, n_skipped}
        """
        n_projected = 0
        n_skipped = 0

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None or name not in self.projection_bases:
                    n_skipped += 1
                    continue

                g = p.grad.data.flatten()
                basis = self.projection_bases[name].to(g.device)

                # Project out components along remembered directions
                # g' = g - Σ (g·b_i) * b_i
                projections = basis @ g  # (n_comp,)
                correction = (projections.unsqueeze(1) * basis).sum(dim=0)  # (param_size,)
                g_new = g - correction
                p.grad.data.copy_(g_new.view_as(p.grad.data))
                n_projected += 1

        return {'n_projected': n_projected, 'n_skipped': n_skipped}

    def get_memory_usage(self):
        """Память для хранения баз."""
        total = 0
        for name, basis in self.projection_bases.items():
            total += basis.numel() * basis.element_size()
        return {
            'total_bytes': total,
            'total_mb': total / (1024 * 1024),
            'n_params_tracked': len(self.projection_bases),
        }


# ==================== Training Metrics Dashboard ====================

class MetricsDashboard:
    """
    Агрегатор метрик обучения.

    Собирает все метрики в одном месте: loss, lr, grad norm,
    throughput, memory, custom metrics.

    Args:
        window_size: размер окна для скользящих средних
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.step_times = deque(maxlen=window_size)
        self._step = 0
        self._last_time = None
        self._total_tokens = 0

    def log(self, **kwargs):
        """
        Логирует метрики.

        Args:
            **kwargs: name=value пары
        """
        self._step += 1
        now = time.time()
        if self._last_time is not None:
            self.step_times.append(now - self._last_time)
        self._last_time = now

        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[name].append(value)

    def log_tokens(self, n_tokens):
        """Логирует количество обработанных токенов."""
        self._total_tokens += n_tokens

    def get_metric(self, name):
        """Последнее значение метрики."""
        values = self.metrics.get(name)
        if not values:
            return None
        return values[-1]

    def get_mean(self, name):
        """Среднее метрики в окне."""
        values = self.metrics.get(name)
        if not values:
            return None
        return sum(values) / len(values)

    def get_throughput(self):
        """Tokens per second."""
        if not self.step_times:
            return 0.0
        avg_time = sum(self.step_times) / len(self.step_times)
        return 1.0 / max(avg_time, 1e-8)

    def get_summary(self):
        """Полная сводка."""
        summary = {
            'step': self._step,
            'total_tokens': self._total_tokens,
            'steps_per_sec': self.get_throughput(),
        }
        for name in self.metrics:
            values = list(self.metrics[name])
            if values:
                summary[f'{name}_current'] = values[-1]
                summary[f'{name}_mean'] = sum(values) / len(values)
                summary[f'{name}_min'] = min(values)
                summary[f'{name}_max'] = max(values)
        return summary

    def get_trend(self, name, n_points=10):
        """Тренд метрики за последние n точек."""
        values = list(self.metrics.get(name, []))
        if len(values) < 2:
            return 'insufficient_data'
        recent = values[-n_points:]
        if len(recent) < 2:
            return 'insufficient_data'
        first_half = sum(recent[:len(recent) // 2]) / max(len(recent) // 2, 1)
        second_half = sum(recent[len(recent) // 2:]) / max(len(recent) - len(recent) // 2, 1)
        if second_half < first_half * 0.95:
            return 'decreasing'
        elif second_half > first_half * 1.05:
            return 'increasing'
        return 'stable'

    def reset(self):
        self.metrics.clear()
        self.step_times.clear()
        self._step = 0
        self._last_time = None
        self._total_tokens = 0


# ==================== Sequence Packing ====================

class SequencePacker:
    """
    Упаковка коротких последовательностей в длинные.

    Вместо padding объединяет несколько примеров
    в одну последовательность с разделителями.
    Экономит compute на padding tokens.

    Args:
        max_seq_len: максимальная длина упакованной последовательности
        pad_token_id: ID pad-токена
        sep_token_id: ID разделителя
    """
    def __init__(self, max_seq_len=512, pad_token_id=0, sep_token_id=1):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

    def pack(self, sequences):
        """
        Упаковывает список последовательностей.

        Args:
            sequences: list[Tensor] — 1D tensors разной длины

        Returns:
            dict: {
                packed_ids: (N, max_seq_len),
                attention_mask: (N, max_seq_len),
                position_ids: (N, max_seq_len),
                sequence_ids: (N, max_seq_len),  # какой пример в каждой позиции
                n_packed: int,
                n_sequences: int,
                efficiency: float
            }
        """
        packed_ids = []
        attention_masks = []
        position_ids_list = []
        sequence_ids_list = []

        current_ids = []
        current_positions = []
        current_seq_ids = []
        current_seq_idx = 0
        global_seq_idx = 0

        for seq in sequences:
            seq_len = len(seq)
            if seq_len > self.max_seq_len - 1:
                # Too long, truncate
                seq = seq[:self.max_seq_len - 1]
                seq_len = len(seq)

            # Check if fits in current pack
            needed = seq_len + (1 if current_ids else 0)  # +1 for separator
            if len(current_ids) + needed > self.max_seq_len:
                # Flush current
                if current_ids:
                    self._flush(current_ids, current_positions, current_seq_ids,
                                packed_ids, attention_masks, position_ids_list,
                                sequence_ids_list)
                current_ids = []
                current_positions = []
                current_seq_ids = []
                current_seq_idx = 0

            # Add separator if not first
            if current_ids:
                current_ids.append(self.sep_token_id)
                current_positions.append(0)
                current_seq_ids.append(-1)

            # Add sequence
            for i, token in enumerate(seq.tolist()):
                current_ids.append(token)
                current_positions.append(i)
                current_seq_ids.append(current_seq_idx)

            current_seq_idx += 1
            global_seq_idx += 1

        # Flush remaining
        if current_ids:
            self._flush(current_ids, current_positions, current_seq_ids,
                        packed_ids, attention_masks, position_ids_list,
                        sequence_ids_list)

        if not packed_ids:
            return {
                'packed_ids': torch.zeros(1, self.max_seq_len, dtype=torch.long),
                'attention_mask': torch.zeros(1, self.max_seq_len, dtype=torch.bool),
                'position_ids': torch.zeros(1, self.max_seq_len, dtype=torch.long),
                'sequence_ids': torch.full((1, self.max_seq_len), -1, dtype=torch.long),
                'n_packed': 0,
                'n_sequences': 0,
                'efficiency': 0.0,
            }

        packed = torch.stack(packed_ids)
        masks = torch.stack(attention_masks)
        positions = torch.stack(position_ids_list)
        seq_ids = torch.stack(sequence_ids_list)

        total_tokens = masks.sum().item()
        total_slots = masks.numel()

        return {
            'packed_ids': packed,
            'attention_mask': masks,
            'position_ids': positions,
            'sequence_ids': seq_ids,
            'n_packed': len(packed_ids),
            'n_sequences': global_seq_idx,
            'efficiency': total_tokens / max(total_slots, 1),
        }

    def _flush(self, ids, positions, seq_ids, packed_ids, attention_masks,
               position_ids_list, sequence_ids_list):
        """Финализирует одну packed-последовательность."""
        L = len(ids)
        pad_len = self.max_seq_len - L

        packed = torch.tensor(ids + [self.pad_token_id] * pad_len, dtype=torch.long)
        mask = torch.tensor([True] * L + [False] * pad_len, dtype=torch.bool)
        pos = torch.tensor(positions + [0] * pad_len, dtype=torch.long)
        sids = torch.tensor(seq_ids + [-1] * pad_len, dtype=torch.long)

        packed_ids.append(packed)
        attention_masks.append(mask)
        position_ids_list.append(pos)
        sequence_ids_list.append(sids)

    def estimate_savings(self, sequence_lengths):
        """
        Оценивает экономию от пacking.

        Args:
            sequence_lengths: list[int]

        Returns:
            dict: {naive_tokens, packed_tokens, savings_pct}
        """
        # Naive: pad each to max_seq_len
        naive = len(sequence_lengths) * self.max_seq_len
        actual = sum(sequence_lengths)

        # Estimate packed
        packed_seqs = 0
        current_len = 0
        for sl in sorted(sequence_lengths):
            needed = sl + (1 if current_len > 0 else 0)
            if current_len + needed > self.max_seq_len:
                packed_seqs += 1
                current_len = sl
            else:
                current_len += needed
        if current_len > 0:
            packed_seqs += 1
        packed_tokens = packed_seqs * self.max_seq_len

        return {
            'naive_tokens': naive,
            'packed_tokens': packed_tokens,
            'actual_tokens': actual,
            'savings_pct': (1 - packed_tokens / max(naive, 1)) * 100,
            'packing_efficiency': actual / max(packed_tokens, 1),
        }
