"""
v26 утилиты: Sparse Attention Mask, Gradient Vaccine, Progressive Resizing,
Checkpoint Manager, NCE Loss.

Sparse Attention Mask: структурированные паттерны разреженного attention.
Ref: Child et al., "Generating Long Sequences with Sparse Transformers" (2019)

Gradient Vaccine: снижение конфликтов градиентов в multi-task learning.
Ref: Wang et al., "Gradient Vaccine" (2020)

Progressive Resizing: постепенное увеличение длины последовательности.
Ref: Howard & Ruder, "ULMFiT" (2018)

Checkpoint Manager: умное сохранение моделей с отслеживанием метрик.

NCE Loss: noise contrastive estimation как альтернатива softmax.
Ref: Gutmann & Hyvärinen, "NCE" (2010)
"""

import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ==================== Sparse Attention Mask ====================

class SparseAttentionMask:
    """
    Генератор масок для разреженного attention.

    Поддерживает паттерны: local window, strided, axial, combined.
    Снижает O(n²) → O(n√n) или O(n·w).

    Args:
        seq_len: длина последовательности
        pattern: 'local', 'strided', 'axial', 'combined'
        window_size: размер локального окна
        stride: шаг для strided pattern
    """
    def __init__(self, seq_len, pattern='local', window_size=8, stride=4):
        self.seq_len = seq_len
        self.pattern = pattern
        self.window_size = window_size
        self.stride = stride

    def _local_mask(self, seq_len=None):
        """Локальное окно: каждый токен видит w соседей."""
        T = seq_len or self.seq_len
        mask = torch.zeros(T, T, dtype=torch.bool)
        for i in range(T):
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2 + 1)
            mask[i, start:end] = True
        return mask

    def _strided_mask(self, seq_len=None):
        """Strided: каждый токен видит каждый stride-й токен."""
        T = seq_len or self.seq_len
        mask = torch.zeros(T, T, dtype=torch.bool)
        for i in range(T):
            # Strided positions
            for j in range(0, T, self.stride):
                mask[i, j] = True
            # Also see self
            mask[i, i] = True
        return mask

    def _axial_mask(self, seq_len=None):
        """Axial: строки и столбцы в 2D-представлении."""
        T = seq_len or self.seq_len
        side = max(1, int(math.sqrt(T)))
        mask = torch.zeros(T, T, dtype=torch.bool)
        for i in range(T):
            row_i = i // side
            col_i = i % side
            for j in range(T):
                row_j = j // side
                col_j = j % side
                if row_i == row_j or col_i == col_j:
                    mask[i, j] = True
        return mask

    def get_mask(self, seq_len=None):
        """
        Генерирует маску.

        Args:
            seq_len: длина (None = self.seq_len)

        Returns:
            (T, T) bool tensor: True = attend, False = mask out
        """
        if self.pattern == 'local':
            return self._local_mask(seq_len)
        elif self.pattern == 'strided':
            return self._strided_mask(seq_len)
        elif self.pattern == 'axial':
            return self._axial_mask(seq_len)
        elif self.pattern == 'combined':
            local = self._local_mask(seq_len)
            strided = self._strided_mask(seq_len)
            return local | strided
        else:
            T = seq_len or self.seq_len
            return torch.ones(T, T, dtype=torch.bool)

    def get_float_mask(self, seq_len=None):
        """Возвращает float mask: 0 = attend, -inf = mask."""
        bool_mask = self.get_mask(seq_len)
        float_mask = torch.zeros_like(bool_mask, dtype=torch.float32)
        float_mask[~bool_mask] = float('-inf')
        return float_mask

    def sparsity_ratio(self, seq_len=None):
        """Доля замаскированных позиций."""
        mask = self.get_mask(seq_len)
        total = mask.numel()
        attended = mask.sum().item()
        return 1.0 - attended / total


# ==================== Gradient Vaccine ====================

class GradientVaccine:
    """
    Gradient Vaccine для multi-task learning.

    Снижает конфликты градиентов между задачами,
    проецируя конфликтующие направления.

    Args:
        n_tasks: число задач
    """
    def __init__(self, n_tasks=2):
        self.n_tasks = n_tasks

    def _flatten_grads(self, grads):
        """Сплющивает список градиентов в один вектор."""
        return torch.cat([g.flatten() for g in grads if g is not None])

    def compute_conflict(self, grads_a, grads_b):
        """
        Вычисляет конфликт между двумя наборами градиентов.

        Args:
            grads_a, grads_b: list[Tensor] — градиенты задач

        Returns:
            dict: {cosine_sim, conflict, magnitude_a, magnitude_b}
        """
        flat_a = self._flatten_grads(grads_a)
        flat_b = self._flatten_grads(grads_b)

        cos_sim = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()

        return {
            'cosine_sim': cos_sim,
            'conflict': cos_sim < 0,
            'magnitude_a': flat_a.norm().item(),
            'magnitude_b': flat_b.norm().item(),
        }

    def vaccinate(self, grads_a, grads_b):
        """
        Применяет vaccine: проецирует конфликтующие градиенты.

        Если cos(g_a, g_b) < 0: g_a' = g_a - (g_a·g_b / |g_b|²) * g_b

        Args:
            grads_a, grads_b: list[Tensor]

        Returns:
            list[Tensor]: скорректированные градиенты задачи A
        """
        flat_a = self._flatten_grads(grads_a)
        flat_b = self._flatten_grads(grads_b)

        cos_sim = F.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()

        if cos_sim >= 0:
            return grads_a  # No conflict

        # Project out conflicting component
        proj = (flat_a @ flat_b) / (flat_b @ flat_b + 1e-8)
        flat_corrected = flat_a - proj * flat_b

        # Reshape back
        result = []
        offset = 0
        for g in grads_a:
            if g is None:
                result.append(None)
            else:
                numel = g.numel()
                result.append(flat_corrected[offset:offset + numel].view_as(g))
                offset += numel

        return result

    def merge_gradients(self, task_grads, model):
        """
        Объединяет градиенты нескольких задач с vaccine.

        Args:
            task_grads: list[list[Tensor]] — градиенты каждой задачи
            model: nn.Module для применения

        Returns:
            dict: {n_conflicts, total_pairs}
        """
        n_tasks = len(task_grads)
        n_conflicts = 0
        total_pairs = 0

        # Pairwise vaccine
        corrected = list(task_grads)
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                total_pairs += 1
                conflict = self.compute_conflict(corrected[i], corrected[j])
                if conflict['conflict']:
                    n_conflicts += 1
                    corrected[i] = self.vaccinate(corrected[i], corrected[j])

        # Average corrected gradients and apply
        with torch.no_grad():
            for p_idx, p in enumerate(model.parameters()):
                if p.grad is None:
                    continue
                avg_grad = torch.zeros_like(p.grad)
                for t in range(n_tasks):
                    if corrected[t][p_idx] is not None:
                        avg_grad += corrected[t][p_idx]
                p.grad.copy_(avg_grad / n_tasks)

        return {'n_conflicts': n_conflicts, 'total_pairs': total_pairs}


# ==================== Progressive Resizing ====================

class ProgressiveResizing:
    """
    Постепенное увеличение длины последовательности.

    Начинает с коротких последовательностей для быстрого обучения,
    постепенно увеличивает для лучшего quality.

    Args:
        min_len: начальная длина
        max_len: максимальная длина
        total_steps: общее число шагов
        strategy: 'linear', 'step', 'exponential'
    """
    def __init__(self, min_len=32, max_len=512, total_steps=10000, strategy='linear'):
        self.min_len = min_len
        self.max_len = max_len
        self.total_steps = total_steps
        self.strategy = strategy

    def get_seq_len(self, step):
        """
        Текущая длина последовательности.

        Returns:
            int: длина (кратна 8 для эффективности)
        """
        t = min(step / max(self.total_steps, 1), 1.0)

        if self.strategy == 'linear':
            raw = self.min_len + t * (self.max_len - self.min_len)
        elif self.strategy == 'exponential':
            log_min = math.log(max(self.min_len, 1))
            log_max = math.log(max(self.max_len, 1))
            raw = math.exp(log_min + t * (log_max - log_min))
        elif self.strategy == 'step':
            if t < 0.33:
                raw = self.min_len
            elif t < 0.66:
                raw = (self.min_len + self.max_len) / 2
            else:
                raw = self.max_len
        else:
            raw = self.max_len

        # Round to multiple of 8
        result = max(self.min_len, min(int(raw), self.max_len))
        result = ((result + 7) // 8) * 8
        return min(result, self.max_len)

    def truncate_batch(self, input_ids, step):
        """
        Обрезает батч до текущей длины.

        Args:
            input_ids: (B, T)
            step: текущий шаг

        Returns:
            (B, T') — обрезанный tensor
        """
        seq_len = self.get_seq_len(step)
        return input_ids[:, :seq_len]

    def get_schedule(self, n_points=10):
        """Расписание длин для визуализации."""
        points = []
        for i in range(n_points + 1):
            step = int(i * self.total_steps / n_points)
            points.append((step, self.get_seq_len(step)))
        return points

    def get_speedup_estimate(self, step):
        """Оценка ускорения по сравнению с full length."""
        current = self.get_seq_len(step)
        # Attention is O(n²)
        speedup = (self.max_len / current) ** 2
        return {
            'current_len': current,
            'max_len': self.max_len,
            'attention_speedup': speedup,
            'memory_ratio': current / self.max_len,
        }


# ==================== Checkpoint Manager ====================

class CheckpointManager:
    """
    Умный менеджер чекпоинтов.

    Хранит top-k лучших чекпоинтов по метрике,
    периодические чекпоинты, и last checkpoint.

    Args:
        save_dir: директория для сохранения
        max_keep: максимум лучших чекпоинтов
        metric_name: имя метрики ('loss', 'accuracy', etc.)
        mode: 'min' или 'max'
    """
    def __init__(self, save_dir='checkpoints', max_keep=3,
                 metric_name='loss', mode='min'):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.metric_name = metric_name
        self.mode = mode
        self.best_checkpoints = []  # [(metric, path, step)]
        self.history = []

    def _is_better(self, new_metric, old_metric):
        if self.mode == 'min':
            return new_metric < old_metric
        return new_metric > old_metric

    def should_save(self, metric_value):
        """Нужно ли сохранять чекпоинт?"""
        if len(self.best_checkpoints) < self.max_keep:
            return True
        worst = self.best_checkpoints[-1][0]
        return self._is_better(metric_value, worst)

    def save(self, model, optimizer, step, metric_value, extra=None):
        """
        Сохраняет чекпоинт если он в top-k.

        Args:
            model: nn.Module
            optimizer: optimizer
            step: текущий шаг
            metric_value: значение метрики
            extra: dict с дополнительными данными

        Returns:
            dict: {saved, path, rank}
        """
        self.history.append({
            'step': step,
            'metric': metric_value,
        })

        if not self.should_save(metric_value):
            return {'saved': False, 'path': None, 'rank': None}

        # Build checkpoint
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            self.metric_name: metric_value,
        }
        if extra:
            checkpoint.update(extra)

        # Save path
        path = os.path.join(self.save_dir, f'ckpt_step{step}_{self.metric_name}{metric_value:.4f}.pt')

        # Add to best list
        self.best_checkpoints.append((metric_value, path, step))
        reverse = self.mode == 'max'
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Prune if too many
        rank = next(i for i, (_, p, _) in enumerate(self.best_checkpoints) if p == path) + 1

        while len(self.best_checkpoints) > self.max_keep:
            _, removed_path, _ = self.best_checkpoints.pop()
            if os.path.exists(removed_path):
                os.remove(removed_path)

        # Actually save (create dir if needed)
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(checkpoint, path)

        return {'saved': True, 'path': path, 'rank': rank}

    def get_best(self):
        """Возвращает лучший чекпоинт."""
        if not self.best_checkpoints:
            return None
        return {
            'metric': self.best_checkpoints[0][0],
            'path': self.best_checkpoints[0][1],
            'step': self.best_checkpoints[0][2],
        }

    def get_all_best(self):
        """Все лучшие чекпоинты."""
        return [
            {'metric': m, 'path': p, 'step': s}
            for m, p, s in self.best_checkpoints
        ]

    def load_best(self, model, optimizer=None):
        """Загружает лучший чекпоинт."""
        best = self.get_best()
        if best is None or not os.path.exists(best['path']):
            return None
        ckpt = torch.load(best['path'], weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt


# ==================== NCE Loss ====================

class NCELoss(nn.Module):
    """
    Noise Contrastive Estimation Loss.

    Эффективная альтернатива full softmax для больших словарей.
    Вместо нормализации по всему словарю сравнивает с K
    случайными "шумовыми" примерами.

    Args:
        d_model: размерность hidden states
        vocab_size: размер словаря
        n_negatives: число negative samples
        noise_ratio: отношение noise к data (K)
    """
    def __init__(self, d_model, vocab_size, n_negatives=64, noise_ratio=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_negatives = n_negatives
        self.noise_ratio = noise_ratio

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Uniform noise distribution (can be replaced with unigram)
        self.register_buffer(
            'noise_dist',
            torch.ones(vocab_size) / vocab_size
        )

    def set_noise_distribution(self, counts):
        """
        Устанавливает шумовое распределение из частот.

        Args:
            counts: tensor (vocab_size,) — частоты
        """
        freq = counts.float()
        freq = freq / freq.sum()
        # Smooth with 3/4 power (Word2Vec style)
        freq = freq.pow(0.75)
        freq = freq / freq.sum()
        self.noise_dist.copy_(freq)

    def forward(self, hidden, targets):
        """
        NCE forward pass.

        Args:
            hidden: (B, T, D) — hidden states
            targets: (B, T) — target token IDs

        Returns:
            loss: scalar
        """
        B, T, D = hidden.shape

        # Positive scores
        target_emb = self.embedding(targets)  # (B, T, D)
        pos_score = (hidden * target_emb).sum(dim=-1)  # (B, T)
        pos_score = pos_score + self.bias[targets]

        # Negative samples
        neg_ids = torch.multinomial(
            self.noise_dist.expand(B * T, -1),
            self.n_negatives,
            replacement=True,
        ).view(B, T, self.n_negatives)  # (B, T, K)

        neg_emb = self.embedding(neg_ids)  # (B, T, K, D)
        neg_score = torch.einsum('btd,btkd->btk', hidden, neg_emb)  # (B, T, K)
        neg_score = neg_score + self.bias[neg_ids]

        # NCE loss
        # log σ(s_pos - log K·p_noise) + Σ log σ(-s_neg + log K·p_noise)
        log_noise_pos = torch.log(self.noise_dist[targets] * self.noise_ratio + 1e-10)
        log_noise_neg = torch.log(
            self.noise_dist.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            .gather(2, neg_ids) * self.noise_ratio + 1e-10
        )

        pos_loss = -F.logsigmoid(pos_score - log_noise_pos)
        neg_loss = -F.logsigmoid(-neg_score + log_noise_neg)

        loss = pos_loss.mean() + neg_loss.mean()
        return loss
