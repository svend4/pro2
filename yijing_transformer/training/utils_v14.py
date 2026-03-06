"""
v14 утилиты: Layerwise LR, Data Mixing Scheduler, Model Surgery.

Layerwise LR: разные learning rates для разных слоёв.
Глубокие слои учатся медленнее (стабильнее), мелкие — быстрее.

Data Mixing Scheduler: управление пропорциями данных во время обучения
(curriculum learning). Например, сначала больше простых данных,
потом сложных.

Model Surgery: утилиты для прунинга голов, добавления/удаления слоёв,
изменения размерности без полного переобучения.
"""

import math
import copy
import torch
import torch.nn as nn


# ==================== Layerwise Learning Rate ====================

def get_layerwise_lr_groups(model, base_lr=3e-4, lr_decay=0.8):
    """
    Создаёт parameter groups с экспоненциально убывающим LR по слоям.

    Самый глубокий слой (output) получает base_lr.
    Каждый предыдущий слой: lr *= lr_decay.

    Пример (3 слоя, decay=0.8):
        layer 2: 3e-4
        layer 1: 2.4e-4
        layer 0: 1.92e-4
        embeddings: 1.536e-4

    Args:
        model: YiJingGPT модель
        base_lr: learning rate для последнего слоя
        lr_decay: множитель убывания на слой

    Returns:
        list of param groups для optimizer
    """
    param_groups = []
    n_layers = model.cfg.n_layers

    # Группируем параметры по глубине
    layer_params = {i: [] for i in range(n_layers)}
    embed_params = []
    head_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'tok_emb' in name or 'pos_emb' in name:
            embed_params.append(param)
        elif 'head' in name:
            head_params.append(param)
        elif 'core.layers.' in name:
            # Извлекаем номер слоя
            layer_idx = int(name.split('core.layers.')[1].split('.')[0])
            layer_params[layer_idx].append(param)
        else:
            other_params.append(param)

    # Head (output) — base_lr
    if head_params:
        param_groups.append({'params': head_params, 'lr': base_lr, 'name': 'head'})

    # Слои — от глубоких к мелким
    for i in range(n_layers - 1, -1, -1):
        depth_from_top = n_layers - 1 - i
        lr = base_lr * (lr_decay ** depth_from_top)
        if layer_params[i]:
            param_groups.append({
                'params': layer_params[i],
                'lr': lr,
                'name': f'layer_{i}',
            })

    # Embeddings — самый низкий lr
    if embed_params:
        lr = base_lr * (lr_decay ** n_layers)
        param_groups.append({'params': embed_params, 'lr': lr, 'name': 'embeddings'})

    # Прочие
    if other_params:
        param_groups.append({'params': other_params, 'lr': base_lr, 'name': 'other'})

    return param_groups


# ==================== Data Mixing Scheduler ====================

class DataMixingScheduler:
    """
    Управление пропорциями данных при обучении (curriculum learning).

    Определяет веса/вероятности для разных источников данных
    в зависимости от шага обучения.

    Стратегии:
    - 'linear': линейная интерполяция от start_weights к end_weights
    - 'step': дискретные переключения в заданные шаги
    - 'exponential': экспоненциальное изменение

    Args:
        sources: список имён источников данных
        strategy: стратегия смешивания
        total_steps: общее число шагов обучения
    """
    def __init__(self, sources, strategy='linear', total_steps=10000):
        self.sources = sources
        self.n_sources = len(sources)
        self.strategy = strategy
        self.total_steps = total_steps

        # Начальные и конечные веса
        uniform = [1.0 / self.n_sources] * self.n_sources
        self.start_weights = uniform[:]
        self.end_weights = uniform[:]

        # Для step стратегии
        self.step_schedule = []  # [(step, weights), ...]

    def set_linear(self, start_weights, end_weights):
        """Устанавливает линейную интерполяцию."""
        assert len(start_weights) == self.n_sources
        assert len(end_weights) == self.n_sources
        self.strategy = 'linear'
        self.start_weights = list(start_weights)
        self.end_weights = list(end_weights)

    def set_step_schedule(self, schedule):
        """
        Устанавливает пошаговое расписание.

        Args:
            schedule: [(step, weights), ...] — при достижении step
                     переключаемся на weights
        """
        self.strategy = 'step'
        self.step_schedule = sorted(schedule, key=lambda x: x[0])

    def get_weights(self, step):
        """
        Возвращает веса для данного шага.

        Args:
            step: текущий шаг обучения

        Returns:
            list of float: веса для каждого источника (суммируются в 1)
        """
        if self.strategy == 'linear':
            progress = min(step / max(1, self.total_steps), 1.0)
            weights = [
                s + (e - s) * progress
                for s, e in zip(self.start_weights, self.end_weights)
            ]
        elif self.strategy == 'step':
            weights = self.start_weights[:]
            for s, w in self.step_schedule:
                if step >= s:
                    weights = list(w)
            return self._normalize(weights)
        elif self.strategy == 'exponential':
            progress = min(step / max(1, self.total_steps), 1.0)
            # Exponential interpolation
            weights = [
                s * ((e / max(s, 1e-8)) ** progress)
                for s, e in zip(self.start_weights, self.end_weights)
            ]
        else:
            weights = self.start_weights[:]

        return self._normalize(weights)

    @staticmethod
    def _normalize(weights):
        """Нормализует веса к сумме 1."""
        total = sum(weights)
        if total == 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]

    def sample_source(self, step):
        """Выбирает источник данных по весам."""
        import random
        weights = self.get_weights(step)
        return random.choices(range(self.n_sources), weights=weights, k=1)[0]


# ==================== Model Surgery ====================

def prune_attention_heads(model, layer_idx, heads_to_prune):
    """
    Удаляет attention головы из указанного слоя.

    Зануляет веса указанных голов (soft pruning).
    Для hard pruning нужно пересоздавать проекции.

    Args:
        model: YiJingGPT модель
        layer_idx: индекс слоя
        heads_to_prune: list[int] — индексы голов для удаления

    Returns:
        dict: информация о прунинге
    """
    layer = model.core.layers[layer_idx]
    attn = layer.attn

    head_dim = attn.head_dim
    n_heads = attn.n_heads

    pruned = 0
    with torch.no_grad():
        for head_idx in heads_to_prune:
            if head_idx >= n_heads:
                continue
            start = head_idx * head_dim
            end = start + head_dim

            # Зануляем Q проекцию для этой головы
            attn.q_proj.weight[start:end, :] = 0
            # Зануляем output проекцию
            attn.out.weight[:, start:end] = 0

            pruned += 1

    return {
        'layer': layer_idx,
        'pruned_heads': heads_to_prune[:pruned],
        'remaining_heads': n_heads - pruned,
        'total_heads': n_heads,
    }


def count_active_heads(model, threshold=1e-6):
    """
    Считает активные (не-нулевые) attention головы в каждом слое.

    Args:
        model: YiJingGPT модель
        threshold: порог для считания головы «нулевой»

    Returns:
        list[dict]: информация по каждому слою
    """
    results = []
    for i, layer in enumerate(model.core.layers):
        attn = layer.attn
        head_dim = attn.head_dim
        n_heads = attn.n_heads

        active = 0
        for h in range(n_heads):
            start = h * head_dim
            end = start + head_dim
            q_norm = attn.q_proj.weight[start:end, :].norm().item()
            if q_norm > threshold:
                active += 1

        results.append({
            'layer': i,
            'active_heads': active,
            'total_heads': n_heads,
            'utilization': active / n_heads if n_heads > 0 else 0,
        })

    return results


def grow_model_depth(model, n_new_layers=1, position='end'):
    """
    Добавляет слои к модели (depth growing).

    Новые слои инициализируются как identity (нулевые residual ветки).

    Args:
        model: YiJingGPT модель
        n_new_layers: число новых слоёв
        position: 'end' или 'middle'

    Returns:
        Новая модель с дополнительными слоями
    """
    cfg = copy.deepcopy(model.cfg)
    old_n_layers = cfg.n_layers

    # Копируем последний слой как шаблон
    template_layer = model.core.layers[-1]

    new_layers = []
    for _ in range(n_new_layers):
        new_layer = copy.deepcopy(template_layer)
        # Инициализируем output проекции нулями → identity при start
        with torch.no_grad():
            new_layer.attn.out.weight.zero_()
            # FFN output → zero
            if hasattr(new_layer.ffn, 'w2'):
                new_layer.ffn.w2.weight.zero_()
            elif isinstance(new_layer.ffn, nn.Sequential):
                # Последний Linear в Sequential
                for m in reversed(list(new_layer.ffn.modules())):
                    if isinstance(m, nn.Linear):
                        m.weight.zero_()
                        break
        new_layers.append(new_layer)

    if position == 'end':
        for nl in new_layers:
            model.core.layers.append(nl)
    elif position == 'middle':
        mid = old_n_layers // 2
        for j, nl in enumerate(new_layers):
            model.core.layers.insert(mid + j, nl)

    # Обновляем конфиг
    model.cfg.n_layers = old_n_layers + n_new_layers

    return model


def shrink_model_depth(model, layers_to_remove):
    """
    Удаляет слои из модели.

    Args:
        model: YiJingGPT модель
        layers_to_remove: list[int] — индексы слоёв для удаления

    Returns:
        Модель с удалёнными слоями
    """
    # Удаляем в обратном порядке чтобы индексы не сбивались
    for idx in sorted(layers_to_remove, reverse=True):
        if 0 <= idx < len(model.core.layers):
            del model.core.layers[idx]

    model.cfg.n_layers = len(model.core.layers)
    return model
