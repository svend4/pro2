"""
v49 утилиты: Gradient Surgery (PCGrad), Adaptive Gradient Clipping,
Cosine Similarity Loss, Mixup Augmentation, CutMix for Sequences.

Gradient Surgery: проекция конфликтующих градиентов.
Ref: Yu et al., "Gradient Surgery for Multi-Task Learning" (2020)

AGC: clipping по ratio gradient/parameter norms.
Ref: Brock et al., "High-Performance Large-Scale Image Recognition" (2021)

Cosine Similarity Loss: обучение через cosine distance.
Ref: Standard metric learning technique.

Mixup: интерполяция примеров.
Ref: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)

CutMix for Sequences: замена подпоследовательностей.
Ref: Yun et al., "CutMix" (2019), adapted for NLP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Gradient Surgery (PCGrad) ====================

class GradientSurgery:
    """
    PCGrad: проекция конфликтующих градиентов.

    Если градиенты двух задач конфликтуют (cos < 0),
    проецирует один на нормальную плоскость другого.

    Устраняет negative transfer в multi-task learning.
    """
    def __init__(self):
        self._conflict_count = 0
        self._total_count = 0

    def project(self, grads):
        """
        Проецирует конфликтующие градиенты.

        Args:
            grads: list[Tensor] — градиенты от разных задач (flattened)

        Returns:
            dict: {projected_grads, n_conflicts, conflict_rate}
        """
        n = len(grads)
        projected = [g.clone() for g in grads]
        conflicts = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dot = (projected[i] * grads[j]).sum()
                if dot < 0:
                    # Project: remove conflicting component
                    proj = dot / max((grads[j] * grads[j]).sum(), 1e-12)
                    projected[i] = projected[i] - proj * grads[j]
                    conflicts += 1

        self._conflict_count += conflicts
        self._total_count += n * (n - 1)

        return {
            'projected_grads': projected,
            'n_conflicts': conflicts,
            'conflict_rate': conflicts / max(n * (n - 1), 1),
        }

    def apply_to_model(self, model, task_losses):
        """
        Применяет PCGrad к модели.

        Args:
            model: nn.Module
            task_losses: list[Tensor] — losses от разных задач

        Returns:
            dict: {combined_grad, n_conflicts}
        """
        all_grads = []

        for loss in task_losses:
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad = []
            for p in model.parameters():
                if p.grad is not None:
                    grad.append(p.grad.data.clone().flatten())
                else:
                    grad.append(torch.zeros(p.numel(), device=p.device))
            all_grads.append(torch.cat(grad))

        result = self.project(all_grads)

        # Average projected gradients
        combined = torch.stack(result['projected_grads']).mean(dim=0)

        # Set gradients
        model.zero_grad()
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            if p.requires_grad:
                p.grad = combined[offset:offset + numel].reshape(p.shape).clone()
            offset += numel

        return {
            'n_conflicts': result['n_conflicts'],
            'conflict_rate': result['conflict_rate'],
        }

    def get_stats(self):
        return {
            'total_conflicts': self._conflict_count,
            'overall_conflict_rate': self._conflict_count / max(self._total_count, 1),
        }


# ==================== Adaptive Gradient Clipping ====================

class AdaptiveGradientClipping:
    """
    AGC: clipping по ratio ||grad|| / ||param||.

    Если ||∇W|| / ||W|| > λ, масштабирует градиент.
    Более стабилен чем фиксированный max_norm.

    Args:
        clip_factor: максимальный ratio (0.01-0.1 типично)
        eps: для числовой стабильности
    """
    def __init__(self, clip_factor=0.01, eps=1e-3):
        self.clip_factor = clip_factor
        self.eps = eps
        self._clipped_count = 0
        self._total_count = 0

    def clip(self, model):
        """
        Применяет AGC к градиентам модели.

        Args:
            model: nn.Module (после backward)

        Returns:
            dict: {n_clipped, n_total, max_ratio}
        """
        n_clipped = 0
        n_total = 0
        max_ratio = 0.0

        for p in model.parameters():
            if p.grad is None:
                continue
            n_total += 1

            param_norm = p.data.norm(2)
            grad_norm = p.grad.data.norm(2)

            if param_norm < self.eps:
                continue

            ratio = grad_norm / (param_norm + self.eps)
            max_ratio = max(max_ratio, ratio.item())

            if ratio > self.clip_factor:
                scale = self.clip_factor * param_norm / (grad_norm + self.eps)
                p.grad.data.mul_(scale)
                n_clipped += 1

        self._clipped_count += n_clipped
        self._total_count += n_total

        return {
            'n_clipped': n_clipped,
            'n_total': n_total,
            'max_ratio': max_ratio,
        }

    def get_stats(self):
        return {
            'total_clipped': self._clipped_count,
            'clip_rate': self._clipped_count / max(self._total_count, 1),
        }


# ==================== Cosine Similarity Loss ====================

class CosineSimilarityLoss(nn.Module):
    """
    Loss на основе косинусного сходства.

    Для пар (x, y, label): max(0, margin - label * cos(x, y)).
    label = +1 для похожих, -1 для разных.

    Args:
        margin: маржа (default 0.0)
        reduction: 'mean' или 'sum'
    """
    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, x, y, labels=None):
        """
        Args:
            x: (B, D)
            y: (B, D)
            labels: (B,) +1/-1 (None = все позитивные)

        Returns:
            Tensor: loss
        """
        cos_sim = F.cosine_similarity(x, y, dim=-1)

        if labels is None:
            # Maximize similarity
            loss = 1.0 - cos_sim
        else:
            labels = labels.float()
            loss = torch.clamp(self.margin - labels * cos_sim, min=0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def compute_similarity_matrix(self, embeddings):
        """
        Матрица попарных сходств.

        Args:
            embeddings: (B, D)

        Returns:
            Tensor: (B, B) similarity matrix
        """
        normed = F.normalize(embeddings, dim=-1)
        return torch.mm(normed, normed.t())


# ==================== Mixup Augmentation ====================

class MixupAugmentation:
    """
    Mixup: линейная интерполяция пар примеров.

    x̃ = λ*x_i + (1-λ)*x_j
    ỹ = λ*y_i + (1-λ)*y_j
    λ ~ Beta(α, α)

    Args:
        alpha: параметр Beta-распределения (0.2-0.4 типично)
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def mix(self, x, y=None):
        """
        Применяет mixup к батчу.

        Args:
            x: (B, ...) входные данные
            y: (B, C) one-hot labels или (B,) class indices

        Returns:
            dict: {mixed_x, mixed_y, lam, indices}
        """
        B = x.size(0)
        device = x.device

        # Sample lambda
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
            lam = max(lam.item(), 1 - lam.item())  # Ensure lam >= 0.5
        else:
            lam = 1.0

        # Random permutation
        indices = torch.randperm(B, device=device)

        mixed_x = lam * x + (1 - lam) * x[indices]

        result = {
            'mixed_x': mixed_x,
            'lam': lam,
            'indices': indices,
        }

        if y is not None:
            result['mixed_y'] = (y, y[indices], lam)

        return result

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Mixup loss.

        Args:
            criterion: loss function
            pred: predictions
            y_a, y_b: original and shuffled targets
            lam: interpolation coefficient

        Returns:
            Tensor: mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== CutMix for Sequences ====================

class SequenceCutMix:
    """
    CutMix адаптированный для последовательностей.

    Заменяет случайный отрезок одной последовательности
    отрезком из другой.

    Args:
        alpha: параметр Beta для длины отрезка
        min_cut_ratio: минимальная доля замены
        max_cut_ratio: максимальная доля замены
    """
    def __init__(self, alpha=1.0, min_cut_ratio=0.1, max_cut_ratio=0.5):
        self.alpha = alpha
        self.min_cut_ratio = min_cut_ratio
        self.max_cut_ratio = max_cut_ratio

    def cut_mix(self, input_ids, labels=None):
        """
        Применяет CutMix к батчу последовательностей.

        Args:
            input_ids: (B, T) токены
            labels: (B, T) метки (optional)

        Returns:
            dict: {mixed_ids, mixed_labels, cut_start, cut_end, lam, indices}
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Sample cut ratio
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 0.5

        cut_ratio = self.min_cut_ratio + lam * (self.max_cut_ratio - self.min_cut_ratio)
        cut_len = max(int(T * cut_ratio), 1)

        # Random start position
        cut_start = torch.randint(0, max(T - cut_len, 1), (1,)).item()
        cut_end = min(cut_start + cut_len, T)

        # Random permutation
        indices = torch.randperm(B, device=device)

        # Mix
        mixed_ids = input_ids.clone()
        mixed_ids[:, cut_start:cut_end] = input_ids[indices, cut_start:cut_end]

        # Effective lambda (ratio of original tokens)
        effective_lam = 1.0 - (cut_end - cut_start) / T

        result = {
            'mixed_ids': mixed_ids,
            'cut_start': cut_start,
            'cut_end': cut_end,
            'lam': effective_lam,
            'indices': indices,
        }

        if labels is not None:
            mixed_labels = labels.clone()
            mixed_labels[:, cut_start:cut_end] = labels[indices, cut_start:cut_end]
            result['mixed_labels'] = mixed_labels

        return result

    def cutmix_criterion(self, criterion, pred, y_original, y_mixed, lam):
        """
        CutMix loss.

        Returns:
            Tensor: mixed loss
        """
        return lam * criterion(pred, y_original) + (1 - lam) * criterion(pred, y_mixed)
