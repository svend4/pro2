"""
polyglot_supervised.py — Дообучение с учителем (Supervised Fine-Tuning) для PolyglotQuartet

Три режима дообучения:

  ① ДИСТИЛЛЯЦИЯ — старший учитель передаёт знания младшему ученику
  ② АННОТАЦИИ  — обучение на размеченных парах (вход → эталон)
  ③ КОНТРАСТИВНОЕ — правильные пары ближе, неправильные дальше

Научная основа:
  - Knowledge Distillation (Hinton et al., 2015)
  - Supervised Contrastive Learning (Khosla et al., 2020)
  - Task-specific fine-tuning с замораживанием слоёв (adapter-style)

Корпус svend4/info — русскоязычный, охватывает:
  физику, математику, философию, мифологию, музыку,
  биологию, астрономию, числовые системы и многое другое.
"""

import math
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Конфигурация дообучения
# ═══════════════════════════════════════════════════════════════

@dataclass
class SupervisedConfig:
    """Параметры дообучения с учителем.

    Режимы (mode):
      'distill'    — дистилляция из учителя в ученика
      'annotated'  — обучение на размеченных парах
      'contrastive'— контрастивное обучение на парах (позитив/негатив)
    """
    mode: str = 'annotated'           # 'distill' | 'annotated' | 'contrastive'
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 8
    # Дистилляция
    temperature: float = 3.0          # температура софтенинга логитов
    alpha: float = 0.5                # баланс soft/hard loss
    # Контрастивное
    contrastive_margin: float = 0.5   # отступ для негативных пар
    # Замораживание
    freeze_embeddings: bool = True    # заморозить эмбеддинги при дообучении
    freeze_musicians: List[str] = field(default_factory=list)  # имена замораживаемых музыкантов
    # Логирование
    log_every: int = 50


# ═══════════════════════════════════════════════════════════════
# LayerFreezer — заморозка/разморозка частей модели
# ═══════════════════════════════════════════════════════════════

class LayerFreezer:
    """Управление замораживанием слоёв модели.

    Как дирижёр, который говорит одним музыкантам «играйте»,
    а другим — «молчите и слушайте».
    """

    def __init__(self, model: nn.Module, config: SupervisedConfig):
        self.model = model
        self.config = config
        self._frozen_params: List[str] = []

    def freeze(self) -> List[str]:
        """Заморозить слои согласно конфигурации. Возвращает список замороженных параметров."""
        self._frozen_params = []

        # Заморозка эмбеддингов
        if self.config.freeze_embeddings:
            for name, param in self.model.named_parameters():
                if 'tok_emb' in name or 'pos_emb' in name or 'emb_drop' in name:
                    param.requires_grad = False
                    self._frozen_params.append(name)

        # Заморозка указанных музыкантов
        for musician_name in self.config.freeze_musicians:
            prefix = f'musicians.{musician_name}'
            for name, param in self.model.named_parameters():
                if name.startswith(prefix):
                    param.requires_grad = False
                    self._frozen_params.append(name)

        return self._frozen_params

    def unfreeze_all(self) -> None:
        """Разморозить все параметры."""
        for param in self.model.parameters():
            param.requires_grad = True
        self._frozen_params = []

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Только обучаемые параметры (для оптимизатора)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def summary(self) -> Dict[str, int]:
        """Статистика: сколько параметров заморожено/обучаемо."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'frozen_pct': round(100 * (total - trainable) / max(total, 1), 1),
        }


# ═══════════════════════════════════════════════════════════════
# DistillationLoss — потери для дистилляции знаний
# ═══════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    """Потери дистилляции: мягкие + жёсткие метки.

    Мягкие метки (soft targets) — логиты учителя, сглаженные температурой.
    Жёсткие метки (hard targets) — истинные токены.

    Loss = α * KL(soft_student || soft_teacher) * T² + (1-α) * CE(student, labels)

    Множитель T² компенсирует уменьшение градиентов при высокой температуре.
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            student_logits: (B, T, V) — логиты ученика
            teacher_logits: (B, T, V) — логиты учителя (detached)
            labels: (B, T) — истинные метки (для hard loss)

        Returns:
            loss: скалярный тензор
            info: диагностика
        """
        T = self.temperature

        # Мягкие вероятности
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits.detach() / T, dim=-1)

        # KL дивергенция с поправкой T²
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)

        info = {'kl_loss': kl_loss.item()}

        if labels is not None and self.alpha < 1.0:
            # Жёсткий CE loss
            B, S, V = student_logits.shape
            hard_loss = F.cross_entropy(
                student_logits.reshape(B * S, V),
                labels.reshape(B * S),
                ignore_index=0,  # <pad>
            )
            loss = self.alpha * kl_loss + (1.0 - self.alpha) * hard_loss
            info['hard_loss'] = hard_loss.item()
        else:
            loss = kl_loss

        info['total_loss'] = loss.item()
        return loss, info


# ═══════════════════════════════════════════════════════════════
# ContrastiveSupervisedLoss — контрастивное обучение с учителем
# ═══════════════════════════════════════════════════════════════

class ContrastiveSupervisedLoss(nn.Module):
    """Контрастивное обучение на размеченных парах.

    Позитивные пары (один и тот же смысл, разные формулировки) → ближе.
    Негативные пары (разный смысл) → дальше.

    Loss = max(0, margin - (sim_pos - sim_neg))
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            anchor:   (B, D) — якорное представление
            positive: (B, D) — позитивный пример (тот же смысл)
            negative: (B, D) — негативный пример (другой смысл)

        Returns:
            loss, info
        """
        # L2-нормализация
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        sim_pos = (anchor * positive).sum(dim=-1)   # (B,)
        sim_neg = (anchor * negative).sum(dim=-1)   # (B,)

        # Triplet margin loss
        loss = F.relu(self.margin - (sim_pos - sim_neg)).mean()

        info = {
            'sim_pos': sim_pos.mean().item(),
            'sim_neg': sim_neg.mean().item(),
            'loss': loss.item(),
        }
        return loss, info


# ═══════════════════════════════════════════════════════════════
# AnnotatedPairLoss — обучение на размеченных парах вход→выход
# ═══════════════════════════════════════════════════════════════

class AnnotatedPairLoss(nn.Module):
    """Стандартный CE loss для размеченных пар (вход → эталонный выход).

    Используется когда у нас есть корпус с правильными ответами,
    например: «E=mc² → Энергия равна массе...» (перевод между языками).
    """

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits:  (B, T, V)
            targets: (B, T)
            mask:    (B, T) — 1.0 для валидных позиций, 0.0 для паддинга

        Returns:
            loss, info
        """
        B, T, V = logits.shape

        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=0,
            label_smoothing=self.label_smoothing,
        )

        # Точность (accuracy) для мониторинга
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # (B, T)
            if mask is not None:
                correct = ((preds == targets) & (mask > 0)).sum()
                total = mask.sum()
            else:
                valid = targets != 0
                correct = ((preds == targets) & valid).sum()
                total = valid.sum()
            accuracy = (correct / total.clamp(min=1)).item()

        info = {
            'loss': loss.item(),
            'accuracy': accuracy,
        }
        return loss, info


# ═══════════════════════════════════════════════════════════════
# WarmupCosineScheduler — расписание learning rate
# ═══════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """Линейный warmup → косинусное затухание.

    Как восход солнца: сначала медленно светлеет (warmup),
    потом яркий день (пиковый lr), потом постепенный закат (cosine decay).
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_steps: int, max_steps: int,
                 min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._step = 0

    def step(self) -> float:
        """Один шаг расписания. Возвращает текущий lr."""
        self._step += 1
        lr_scale = self._compute_scale(self._step)

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * lr_scale

        return self.base_lrs[0] * lr_scale

    def _compute_scale(self, step: int) -> float:
        if step <= self.warmup_steps:
            # Линейный warmup
            return step / max(self.warmup_steps, 1)

        # Косинусное затухание
        progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


# ═══════════════════════════════════════════════════════════════
# SupervisedTrainer — основной тренер дообучения
# ═══════════════════════════════════════════════════════════════

class SupervisedTrainer:
    """Тренер дообучения с учителем для PolyglotQuartet.

    Поддерживает три режима:
      - 'distill':     дистилляция из модели-учителя
      - 'annotated':   обучение на размеченных парах
      - 'contrastive': контрастивное обучение
    """

    def __init__(
        self,
        student: nn.Module,
        config: SupervisedConfig,
        teacher: Optional[nn.Module] = None,
    ):
        self.student = student
        self.config = config
        self.teacher = teacher
        self.device = next(student.parameters()).device

        # Замораживание слоёв
        self.freezer = LayerFreezer(student, config)
        self.freezer.freeze()

        # Оптимизатор (только обучаемые параметры)
        trainable = self.freezer.get_trainable_params()
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98),
        )

        # Расписание lr
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )

        # Функции потерь
        self.distill_loss = DistillationLoss(config.temperature, config.alpha)
        self.annotated_loss = AnnotatedPairLoss()
        self.contrastive_loss = ContrastiveSupervisedLoss(config.contrastive_margin)

        # Учитель в eval-режиме
        if teacher is not None:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

        # История обучения
        self.history: List[Dict] = []
        self._step = 0

    def train_step_distill(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Один шаг дистилляции.

        Args:
            input_ids: (B, T) — входные токены
            labels:    (B, T) — метки (если None, используются сдвинутые input_ids)
        """
        assert self.teacher is not None, "Для дистилляции нужен учитель (teacher)"

        self.student.train()
        input_ids = input_ids.to(self.device)
        if labels is None:
            labels = input_ids

        # Прямой проход ученика
        student_out = self.student(input_ids)
        student_logits = student_out if isinstance(student_out, torch.Tensor) else student_out[0]

        # Прямой проход учителя (без градиентов)
        with torch.no_grad():
            teacher_out = self.teacher(input_ids)
            teacher_logits = teacher_out if isinstance(teacher_out, torch.Tensor) else teacher_out[0]

        loss, info = self.distill_loss(student_logits, teacher_logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        lr = self.scheduler.step()
        self._step += 1

        info['lr'] = lr
        info['step'] = self._step
        self.history.append(info)
        return info

    def train_step_annotated(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Один шаг обучения на размеченных парах.

        Args:
            input_ids: (B, T) — входные токены
            targets:   (B, T) — эталонные выходные токены
            mask:      (B, T) — маска валидных позиций
        """
        self.student.train()
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        out = self.student(input_ids)
        logits = out if isinstance(out, torch.Tensor) else out[0]

        loss, info = self.annotated_loss(logits, targets, mask)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        lr = self.scheduler.step()
        self._step += 1

        info['lr'] = lr
        info['step'] = self._step
        self.history.append(info)
        return info

    def train_step_contrastive(
        self,
        anchor_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Один шаг контрастивного обучения.

        Args:
            anchor_ids:   (B, T) — якорные примеры
            positive_ids: (B, T) — позитивные (тот же смысл)
            negative_ids: (B, T) — негативные (другой смысл)
        """
        self.student.train()
        anchor_ids = anchor_ids.to(self.device)
        positive_ids = positive_ids.to(self.device)
        negative_ids = negative_ids.to(self.device)

        # Получаем средние представления для каждого примера
        def encode(ids: torch.Tensor) -> torch.Tensor:
            out = self.student(ids)
            hidden = out if isinstance(out, torch.Tensor) else out[0]
            return hidden.mean(dim=1)  # (B, D)

        anchor_emb = encode(anchor_ids)
        positive_emb = encode(positive_ids)
        negative_emb = encode(negative_ids)

        loss, info = self.contrastive_loss(anchor_emb, positive_emb, negative_emb)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        lr = self.scheduler.step()
        self._step += 1

        info['lr'] = lr
        info['step'] = self._step
        self.history.append(info)
        return info

    def train_epoch(
        self,
        dataloader,
        mode: Optional[str] = None,
    ) -> Dict[str, float]:
        """Одна эпоха обучения.

        Args:
            dataloader: итератор батчей (словари с ключами зависят от режима)
            mode: режим ('distill', 'annotated', 'contrastive'), по умолчанию из config
        """
        mode = mode or self.config.mode
        epoch_losses = []

        for batch in dataloader:
            if mode == 'distill':
                info = self.train_step_distill(
                    input_ids=batch['input_ids'],
                    labels=batch.get('labels'),
                )
            elif mode == 'annotated':
                info = self.train_step_annotated(
                    input_ids=batch['input_ids'],
                    targets=batch['targets'],
                    mask=batch.get('mask'),
                )
            elif mode == 'contrastive':
                info = self.train_step_contrastive(
                    anchor_ids=batch['anchor'],
                    positive_ids=batch['positive'],
                    negative_ids=batch['negative'],
                )
            else:
                raise ValueError(f"Неизвестный режим: {mode}")

            epoch_losses.append(info.get('loss', info.get('total_loss', 0.0)))

            if self._step % self.config.log_every == 0:
                avg = sum(epoch_losses[-self.config.log_every:]) / min(
                    len(epoch_losses), self.config.log_every
                )
                print(f"  шаг {self._step}: loss={avg:.4f}, lr={info.get('lr', 0):.6f}")

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        return {'avg_loss': avg_loss, 'steps': len(epoch_losses)}

    def get_summary(self) -> Dict:
        """Сводка обучения."""
        freezer_info = self.freezer.summary()
        return {
            'mode': self.config.mode,
            'total_steps': self._step,
            'params': freezer_info,
            'final_loss': self.history[-1] if self.history else {},
        }


# ═══════════════════════════════════════════════════════════════
# Фабричная функция
# ═══════════════════════════════════════════════════════════════

def build_supervised_trainer(
    student: nn.Module,
    teacher: Optional[nn.Module] = None,
    mode: str = 'annotated',
    learning_rate: float = 1e-4,
    max_steps: int = 2000,
    freeze_musicians: Optional[List[str]] = None,
) -> SupervisedTrainer:
    """Создать тренер дообучения с настройками по умолчанию.

    Пример для корпуса svend4/info:
        trainer = build_supervised_trainer(
            student=model,
            mode='annotated',
            freeze_musicians=['formalist', 'archetypist'],
        )
        # Обучаем только algorithmist и linguist
        trainer.train_epoch(dataloader)
    """
    config = SupervisedConfig(
        mode=mode,
        learning_rate=learning_rate,
        max_steps=max_steps,
        freeze_musicians=freeze_musicians or [],
    )
    return SupervisedTrainer(student=student, config=config, teacher=teacher)
