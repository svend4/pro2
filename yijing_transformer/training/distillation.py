"""
EXPERIMENTAL — не используется в training pipeline.
Импортируется только в test_model_pytest.py. Оставлен для будущих экспериментов.

Knowledge Distillation для YiJing-Transformer.

Обучает student модель на soft targets от teacher модели.
Поддерживает:
- Стандартную KD (Hinton et al.): KL(student || teacher) при повышенной температуре
- Feature distillation: MSE между hidden states
- Комбинированный loss: α·soft_loss + (1-α)·hard_loss

Использование:
    teacher = YiJingGPT.from_pretrained("teacher.pt")
    student = YiJingGPT(small_cfg)
    distill_loss = distillation_loss(student_logits, teacher_logits, targets, cfg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, targets,
                      alpha=0.5, temperature=2.0):
    """
    Комбинированный distillation loss.

    Args:
        student_logits: (B, T, V) — логиты от student
        teacher_logits: (B, T, V) — логиты от teacher
        targets: (B, T) — истинные метки
        alpha: баланс soft/hard (1.0 = только soft targets)
        temperature: температура для soft targets

    Returns:
        loss: scalar
    """
    # Hard loss: стандартный cross-entropy
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    # Soft loss: KL divergence при повышенной температуре
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = F.kl_div(
        student_soft.reshape(-1, student_logits.size(-1)),
        teacher_soft.reshape(-1, teacher_logits.size(-1)),
        reduction='batchmean',
    ) * (temperature ** 2)  # масштабирование для корректных градиентов

    return alpha * soft_loss + (1 - alpha) * hard_loss


def feature_distillation_loss(student_hidden, teacher_hidden):
    """
    Feature distillation: MSE между hidden states.

    Если размерности не совпадают, используется линейная проекция.

    Args:
        student_hidden: (B, T, D_student)
        teacher_hidden: (B, T, D_teacher)

    Returns:
        loss: MSE loss (scalar)
    """
    if student_hidden.size(-1) != teacher_hidden.size(-1):
        # Проецируем student → teacher dim через adaptive pooling
        student_hidden = F.adaptive_avg_pool1d(
            student_hidden.transpose(1, 2),
            teacher_hidden.size(-1)
        ).transpose(1, 2)

    return F.mse_loss(student_hidden, teacher_hidden)


class DistillationTrainer:
    """
    Тренер для knowledge distillation.

    Автоматизирует:
    1. Forward pass через teacher (no grad)
    2. Forward pass через student
    3. Комбинированный loss
    4. Backward + optimizer step
    """
    def __init__(self, teacher, student, optimizer, cfg,
                 alpha=0.5, temperature=2.0, feature_weight=0.0):
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.cfg = cfg
        self.alpha = alpha
        self.temperature = temperature
        self.feature_weight = feature_weight

        # Замораживаем teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def step(self, x, y):
        """
        Один шаг distillation.

        Args:
            x: (B, T) input tokens
            y: (B, T) target tokens

        Returns:
            dict с метриками: total_loss, hard_loss, soft_loss
        """
        self.student.train()

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits, _, _ = self.teacher(x)

        # Student forward
        student_logits, hard_loss, _ = self.student(x, y)

        # Distillation loss
        loss = distillation_loss(
            student_logits, teacher_logits, y,
            alpha=self.alpha,
            temperature=self.temperature,
        )

        # Aux loss от student (MoE, commitment)
        aux = self.student.core.get_aux_loss()
        if isinstance(aux, torch.Tensor):
            loss = loss + aux

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), self.cfg.max_grad_norm
        )
        self.optimizer.step()

        return {
            'total_loss': loss.item(),
            'hard_loss': hard_loss.item() if hard_loss is not None else 0.0,
        }
