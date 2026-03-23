"""
v43 утилиты: Knowledge Distillation, Label Smoothing,
Focal Loss, Contrastive Loss, R-Drop Regularization.

Knowledge Distillation: передача знаний teacher → student.
Ref: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)

Label Smoothing: сглаживание one-hot меток.
Ref: Szegedy et al., "Rethinking the Inception Architecture" (2016)

Focal Loss: фокусировка на сложных примерах.
Ref: Lin et al., "Focal Loss for Dense Object Detection" (2017)

Contrastive Loss: обучение представлений через контраст.
Ref: Chen et al., "A Simple Framework for Contrastive Learning" (2020)

R-Drop: регуляризация через KL-divergence двух проходов.
Ref: Wu et al., "R-Drop: Regularized Dropout for Neural Networks" (2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Knowledge Distillation ====================

class KnowledgeDistillation:
    """
    Knowledge Distillation loss.

    Комбинирует hard loss (CE с ground truth) и soft loss
    (KL-div с teacher logits при температуре T).

    L = α * L_hard + (1 - α) * T² * L_soft

    Args:
        temperature: температура softmax (выше = мягче)
        alpha: вес hard loss (0 = только soft, 1 = только hard)
    """
    def __init__(self, temperature=4.0, alpha=0.5):
        self.temperature = temperature
        self.alpha = alpha
        self._stats = {'hard_loss': 0, 'soft_loss': 0, 'total_loss': 0, 'n': 0}

    def compute(self, student_logits, teacher_logits, targets):
        """
        Вычисляет distillation loss.

        Args:
            student_logits: (B, C) логиты студента
            teacher_logits: (B, C) логиты учителя
            targets: (B,) ground truth labels

        Returns:
            dict: {loss, hard_loss, soft_loss}
        """
        T = self.temperature

        # Hard loss: standard CE
        hard_loss = F.cross_entropy(student_logits, targets)

        # Soft loss: KL divergence with temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        # Combined loss
        total = self.alpha * hard_loss + (1 - self.alpha) * T * T * soft_loss

        self._stats['hard_loss'] += hard_loss.item()
        self._stats['soft_loss'] += soft_loss.item()
        self._stats['total_loss'] += total.item()
        self._stats['n'] += 1

        return {
            'loss': total,
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
        }

    def compute_from_logits(self, student_logits, teacher_logits):
        """
        Только soft distillation (без ground truth).

        Args:
            student_logits: (B, C)
            teacher_logits: (B, C)

        Returns:
            Tensor: soft distillation loss
        """
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * T * T

    def get_stats(self):
        n = max(self._stats['n'], 1)
        return {
            'avg_hard_loss': self._stats['hard_loss'] / n,
            'avg_soft_loss': self._stats['soft_loss'] / n,
            'avg_total_loss': self._stats['total_loss'] / n,
        }


# ==================== Label Smoothing ====================

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy с label smoothing.

    Вместо one-hot: y_smooth = (1 - ε) * y_onehot + ε / C.
    Предотвращает overconfidence модели.

    Args:
        smoothing: коэффициент сглаживания ε (0 = без, 0.1 типично)
        ignore_index: индекс для ignore (padding)
    """
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) или (B, T, C)
            targets: (B,) или (B, T)

        Returns:
            Tensor: smoothed loss
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
        else:
            C = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (C - 1))
            mask = targets != self.ignore_index
            valid_targets = targets.clone()
            valid_targets[~mask] = 0
            smooth_targets.scatter_(1, valid_targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Apply mask
        if self.ignore_index >= 0:
            loss = loss * mask.float()
            n_valid = mask.sum().item()
            if n_valid == 0:
                return loss.new_tensor(0.0)
            return loss.sum() / n_valid

        return loss.mean()


# ==================== Focal Loss ====================

class FocalLoss(nn.Module):
    """
    Focal Loss для работы с class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Уменьшает вклад легко классифицируемых примеров,
    фокусируясь на сложных.

    Args:
        gamma: focusing parameter (0 = CE, 2 типично)
        alpha: class balancing weight (None = без балансировки)
        ignore_index: индекс padding
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) или (B, T, C)
            targets: (B,) или (B, T)

        Returns:
            Tensor: focal loss
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)

        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                  ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                focal_weight = focal_weight * self.alpha
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[targets]
                focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss

        # Mask padding
        mask = targets != self.ignore_index
        return loss[mask].mean() if mask.any() else loss.mean()


# ==================== Contrastive Loss ====================

class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Для self-supervised / contrastive learning. Максимизирует
    сходство позитивных пар и минимизирует для негативных.

    Args:
        temperature: температура масштабирования
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1: (B, D) представления view 1
            z2: (B, D) представления view 2

        Returns:
            Tensor: contrastive loss
        """
        B = z1.size(0)

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Similarity matrix
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss

    def compute_similarity_stats(self, z1, z2):
        """
        Статистики сходства.

        Returns:
            dict: {pos_sim, neg_sim, alignment, uniformity}
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Positive similarity
        pos_sim = (z1 * z2).sum(dim=-1).mean().item()

        # Negative similarity (all pairs)
        all_sim = torch.mm(z1, z2.t())
        mask = ~torch.eye(z1.size(0), device=z1.device).bool()
        neg_sim = all_sim[mask].mean().item() if mask.any() else 0.0

        # Alignment (Wang & Isola, 2020)
        alignment = -(z1 - z2).norm(dim=-1).pow(2).mean().item()

        return {
            'pos_sim': pos_sim,
            'neg_sim': neg_sim,
            'alignment': alignment,
        }


# ==================== R-Drop Regularization ====================

class RDropRegularization:
    """
    R-Drop: регуляризация через KL-divergence.

    Делает два forward pass с dropout, минимизирует
    KL-divergence между выходами для consistency.

    Args:
        alpha: вес R-Drop loss
        reduction: 'mean' или 'sum'
    """
    def __init__(self, alpha=1.0, reduction='mean'):
        self.alpha = alpha
        self.reduction = reduction
        self._stats = {'rdrop_loss': 0.0, 'ce_loss': 0.0, 'n': 0}

    def compute(self, model, inputs, targets, loss_fn=None):
        """
        Два forward pass + R-Drop loss.

        Args:
            model: nn.Module (должен быть в train mode)
            inputs: входные данные (dict или tensor)
            targets: метки
            loss_fn: callable(logits, targets) -> loss (default: CE)

        Returns:
            dict: {loss, ce_loss, rdrop_loss, logits1, logits2}
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        # Two forward passes with different dropout masks
        model.train()
        if isinstance(inputs, dict):
            logits1 = model(**inputs)
            logits2 = model(**inputs)
        else:
            logits1 = model(inputs)
            logits2 = model(inputs)

        # Handle tuple outputs
        if isinstance(logits1, tuple):
            logits1 = logits1[0]
        if isinstance(logits2, tuple):
            logits2 = logits2[0]

        # Reshape for loss if needed
        if logits1.dim() == 3 and targets.dim() == 2:
            B, T, C = logits1.shape
            l1_flat = logits1.reshape(-1, C)
            l2_flat = logits2.reshape(-1, C)
            t_flat = targets.reshape(-1)
        else:
            l1_flat = logits1
            l2_flat = logits2
            t_flat = targets

        # CE loss (average of both passes)
        ce1 = loss_fn(l1_flat, t_flat)
        ce2 = loss_fn(l2_flat, t_flat)
        ce_loss = (ce1 + ce2) / 2

        # KL divergence between two passes
        p1 = F.log_softmax(l1_flat, dim=-1)
        p2 = F.log_softmax(l2_flat, dim=-1)
        q1 = F.softmax(l1_flat, dim=-1)
        q2 = F.softmax(l2_flat, dim=-1)

        kl_1 = F.kl_div(p1, q2, reduction='batchmean')
        kl_2 = F.kl_div(p2, q1, reduction='batchmean')
        rdrop_loss = (kl_1 + kl_2) / 2

        total = ce_loss + self.alpha * rdrop_loss

        self._stats['rdrop_loss'] += rdrop_loss.item()
        self._stats['ce_loss'] += ce_loss.item()
        self._stats['n'] += 1

        return {
            'loss': total,
            'ce_loss': ce_loss.item(),
            'rdrop_loss': rdrop_loss.item(),
            'logits1': logits1,
            'logits2': logits2,
        }

    def kl_divergence(self, logits1, logits2):
        """
        Симметричная KL-divergence между двумя распределениями.

        Args:
            logits1, logits2: (B, C)

        Returns:
            Tensor: symmetric KL
        """
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        q1 = F.softmax(logits1, dim=-1)
        q2 = F.softmax(logits2, dim=-1)

        kl_1 = F.kl_div(p1, q2, reduction='batchmean')
        kl_2 = F.kl_div(p2, q1, reduction='batchmean')
        return (kl_1 + kl_2) / 2

    def get_stats(self):
        n = max(self._stats['n'], 1)
        return {
            'avg_rdrop_loss': self._stats['rdrop_loss'] / n,
            'avg_ce_loss': self._stats['ce_loss'] / n,
        }
