"""
v12 утилиты: µP инициализация, Dynamic Temperature, Checkpoint Manager, Perplexity.

µP (Mu-Parameterization): масштабирование init и lr при увеличении ширины,
позволяет переносить гиперпараметры с маленькой модели на большую.
Ref: Yang et al., "Tensor Programs V" (2022)

Dynamic Temperature: адаптивная температура сэмплирования на основе энтропии logits.

Checkpoint Manager: автоматическое сохранение лучших k чекпоинтов по метрике.

Perplexity: оценка качества языковой модели на корпусе.
"""

import math
import os
import json
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== µP Initialization ====================

def apply_mup_init(model, base_width=128):
    """
    Применяет µP инициализацию к модели.

    Правила µP для width m (base_width → d_model):
    - Embedding: init ~ O(1)
    - Hidden→Hidden: init ~ O(1/m)
    - Output head: init ~ O(1/m), lr ~ O(1/m)
    - Attention logits: scale by 1/d вместо 1/√d

    Args:
        model: YiJingGPT модель
        base_width: базовая ширина (для которой подбирались гиперпараметры)
    """
    d_model = model.cfg.d_model
    width_mult = d_model / base_width

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue

            if 'tok_emb' in name:
                # Embedding: O(1)
                nn.init.normal_(param, std=0.02)
            elif 'head.weight' in name and not model.cfg.weight_tying:
                # Output: O(1/m)
                nn.init.normal_(param, std=0.02 / width_mult)
            elif 'q_proj' in name or 'k_proj' in name:
                # Attention QK: O(1/√m) для стабильных attention scores
                nn.init.normal_(param, std=0.02 / math.sqrt(width_mult))
            elif 'v_proj' in name or 'out' in name:
                # Attention VO: O(1/m)
                nn.init.normal_(param, std=0.02 / width_mult)
            elif any(n in name for n in ['w1', 'w2', 'w3', 'gate', 'ffn']):
                # FFN: O(1/m)
                nn.init.normal_(param, std=0.02 / width_mult)
            else:
                # Default: O(1/√m)
                nn.init.normal_(param, std=0.02 / math.sqrt(width_mult))

    return model


def get_mup_param_groups(model, base_lr=3e-4, base_width=128):
    """
    Создаёт parameter groups с µP learning rates.

    µP предписывает разные lr для разных компонентов:
    - Embedding: lr (без масштабирования)
    - Hidden: lr (без масштабирования)
    - Output head: lr / width_mult

    Args:
        model: YiJingGPT
        base_lr: базовый learning rate
        base_width: базовая ширина

    Returns:
        list of param groups для optimizer
    """
    width_mult = model.cfg.d_model / base_width

    output_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name and 'weight' in name and not model.cfg.weight_tying:
            output_params.append(param)
        else:
            other_params.append(param)

    groups = [
        {'params': other_params, 'lr': base_lr},
    ]
    if output_params:
        groups.append({'params': output_params, 'lr': base_lr / width_mult})

    return groups


# ==================== Dynamic Temperature ====================

def dynamic_temperature(logits, target_entropy_ratio=0.6, min_temp=0.1, max_temp=2.0):
    """
    Адаптивная температура на основе энтропии распределения.

    Идея: если модель уверена (низкая энтропия) → низкая температура.
    Если не уверена (высокая энтропия) → более высокая температура
    для разнообразия, но не слишком.

    Args:
        logits: (B, V) — raw logits
        target_entropy_ratio: целевая доля от max энтропии (0-1)
        min_temp: минимальная температура
        max_temp: максимальная температура

    Returns:
        temperature: (B, 1) — per-batch температура
    """
    V = logits.shape[-1]
    max_entropy = math.log(V)
    target_entropy = target_entropy_ratio * max_entropy

    # Текущая энтропия
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B,)

    # Температура: если entropy < target → снижаем, если > target → повышаем
    # temp = entropy / target_entropy (нормализация)
    temp = entropy / (target_entropy + 1e-8)
    temp = temp.clamp(min=min_temp, max=max_temp)

    return temp.unsqueeze(-1)  # (B, 1)


# ==================== Checkpoint Manager ====================

class CheckpointManager:
    """
    Менеджер чекпоинтов с отслеживанием лучших k моделей.

    Автоматически:
    - Сохраняет чекпоинты при улучшении метрики
    - Удаляет старые, оставляя только top-k
    - Хранит метаданные (step, metric, config)

    Args:
        save_dir: директория для чекпоинтов
        max_keep: максимум сохранённых чекпоинтов
        mode: 'min' (loss) или 'max' (accuracy)
    """
    def __init__(self, save_dir, max_keep=3, mode='min'):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.mode = mode
        # heap: (score, path) — min-heap
        # Для mode='max' инвертируем знак
        self._heap = []
        os.makedirs(save_dir, exist_ok=True)

    def _score(self, metric):
        return metric if self.mode == 'min' else -metric

    def should_save(self, metric):
        """Проверяет, стоит ли сохранять этот чекпоинт."""
        if len(self._heap) < self.max_keep:
            return True
        # Для min: удаляем worst (largest), сохраняем если новый лучше worst
        worst_score = max(s for s, _ in self._heap)
        return self._score(metric) < worst_score

    def save(self, model, optimizer, step, metric, extra=None):
        """
        Сохраняет чекпоинт если он в top-k.

        Args:
            model: модель
            optimizer: optimizer
            step: текущий шаг
            metric: метрика для ранжирования
            extra: дополнительные данные

        Returns:
            path если сохранён, None если пропущен
        """
        if not self.should_save(metric):
            return None

        filename = f"checkpoint_step{step}_metric{metric:.4f}.pt"
        path = os.path.join(self.save_dir, filename)

        checkpoint = {
            'step': step,
            'metric': metric,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if hasattr(model, 'cfg'):
            checkpoint['config'] = vars(model.cfg)
        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)

        score = self._score(metric)
        heapq.heappush(self._heap, (score, path))

        # Удаляем лишние
        while len(self._heap) > self.max_keep:
            _, old_path = heapq.heappop(self._heap)
            # Для min mode: heappop удаляет минимальный score,
            # но нам нужно удалять WORST (max score)
            # Переделываем: храним все, сортируем, удаляем worst
            pass

        # Перестраиваем: оставляем top-k
        all_items = sorted(self._heap, key=lambda x: x[0])
        to_keep = all_items[:self.max_keep]
        to_remove = all_items[self.max_keep:]

        for _, old_path in to_remove:
            if os.path.exists(old_path):
                os.remove(old_path)

        self._heap = list(to_keep)
        heapq.heapify(self._heap)

        # Сохраняем метаданные
        meta = {
            'checkpoints': [
                {'path': p, 'score': s} for s, p in self._heap
            ],
            'mode': self.mode,
            'max_keep': self.max_keep,
        }
        meta_path = os.path.join(self.save_dir, 'checkpoint_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return path

    def get_best(self):
        """Возвращает путь к лучшему чекпоинту."""
        if not self._heap:
            return None
        best = min(self._heap, key=lambda x: x[0])
        return best[1]

    def list_checkpoints(self):
        """Список всех сохранённых чекпоинтов, отсортированных по метрике."""
        return sorted(self._heap, key=lambda x: x[0])


# ==================== Perplexity Evaluator ====================

@torch.no_grad()
def evaluate_perplexity(model, dataloader, device='cpu', max_batches=None):
    """
    Вычисляет perplexity модели на датасете.

    PPL = exp(average cross-entropy loss)

    Args:
        model: языковая модель с forward(idx, targets) → (logits, loss, _)
        dataloader: итератор batch'ей (x, y) или (x,) где y = x shifted
        device: устройство
        max_batches: ограничение на число batch'ей (None = все)

    Returns:
        dict: {'perplexity': float, 'avg_loss': float, 'n_tokens': int}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x = batch[0].to(device)
                y = x[:, 1:]
                x = x[:, :-1]
        else:
            x = batch.to(device)
            y = x[:, 1:]
            x = x[:, :-1]

        logits, loss, _ = model(x, targets=y)

        if loss is not None:
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    if total_tokens == 0:
        return {'perplexity': float('inf'), 'avg_loss': float('inf'), 'n_tokens': 0}

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # clamp для численной стабильности

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'n_tokens': total_tokens,
    }
