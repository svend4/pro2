"""
Bridge: Хирургия моделей из utils_v12..v52.

Собирает все инструменты для модификации архитектуры модели
после создания или во время обучения.

Источники:
  v12: µP Initialization — масштабируемая инициализация (Yang et al., 2022)
  v14: Model Surgery — pruning heads, grow/shrink depth
  v16: Structured Pruning — удаление целых голов/слоёв
  v16: Model Merging — SLERP, TIES, Average
  v20: DoRA — Weight-Decomposed Low-Rank Adaptation
  v22: Matryoshka Embeddings — вложенные представления
  v34: Activation Checkpointing Manager — экономия памяти
  v50: Progressive Layer Freezing — постепенная заморозка

Использование:
    from training.bridge_model_surgery import ModelSurgeon
    surgeon = ModelSurgeon(model, cfg)
    surgeon.apply_mup_init()  # µP инициализация
    surgeon.prune_heads(threshold=0.01)  # удаление неважных голов
    surgeon.freeze_layers(up_to=4)  # заморозка первых 4 слоёв
"""

from training.utils_v12 import apply_mup_init, get_mup_param_groups
from training.utils_v14 import prune_attention_heads, count_active_heads
from training.utils_v14 import grow_model_depth, shrink_model_depth
from training.utils_v16 import StructuredPruner
from training.utils_v16 import merge_models_average, merge_models_slerp, merge_models_ties
from training.utils_v20 import DoRALinear, apply_dora
from training.utils_v22 import MatryoshkaHead
from training.utils_v34 import ActivationCheckpointManager
from training.utils_v50 import ProgressiveLayerFreezing

import torch


class ModelSurgeon:
    """Инструменты для модификации архитектуры модели.

    Собирает все операции «хирургии» над моделью:
    инициализация, pruning, merging, адаптация.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    # === Инициализация ===

    def apply_mup_init(self, base_width=128):
        """Применяет µP инициализацию для масштабируемого переноса гиперпараметров.

        При µP можно подобрать гиперпараметры на маленькой модели
        и перенести их на большую без потери качества.

        Args:
            base_width: ширина базовой модели (для которой подбирались гиперпараметры)
        """
        apply_mup_init(self.model, base_width=base_width)
        return self

    def get_mup_param_groups(self, base_width=128):
        """Возвращает param groups для µP-корректного обучения."""
        return get_mup_param_groups(self.model, base_width=base_width)

    # === Pruning ===

    def prune_heads(self, threshold=0.01):
        """Удаляет attention головы с малым вкладом.

        Args:
            threshold: головы с scale < threshold будут удалены

        Returns:
            int: число удалённых голов
        """
        return prune_attention_heads(self.model, threshold=threshold)

    def count_active_heads(self):
        """Считает число активных (не обрезанных) голов."""
        return count_active_heads(self.model)

    def structured_prune(self, target_sparsity=0.3):
        """Структурное pruning: удаление целых нейронов/каналов.

        Args:
            target_sparsity: доля параметров для удаления (0.3 = 30%)
        """
        pruner = StructuredPruner(target_sparsity=target_sparsity)
        pruner.prune(self.model)
        return self

    # === Grow / Shrink ===

    def grow_depth(self, n_new_layers=2):
        """Добавляет новые слои в модель (для progressive training).

        Новые слои инициализируются как identity (не меняют поведение).

        Args:
            n_new_layers: число новых слоёв
        """
        grow_model_depth(self.model, n_new_layers)
        return self

    def shrink_depth(self, n_remove=2):
        """Удаляет последние слои из модели.

        Args:
            n_remove: число слоёв для удаления
        """
        shrink_model_depth(self.model, n_remove)
        return self

    # === Adaptation ===

    def apply_dora(self, rank=16, target_modules=None):
        """Применяет DoRA (Weight-Decomposed LoRA) к модели.

        DoRA раскладывает веса на magnitude + direction,
        обучая только direction через LoRA.

        Args:
            rank: ранг LoRA-адаптации
            target_modules: список модулей для адаптации (None = все Linear)
        """
        apply_dora(self.model, rank=rank, target_modules=target_modules)
        return self

    def add_matryoshka_head(self, dimensions=None):
        """Добавляет Matryoshka head для вложенных представлений.

        Позволяет использовать embedding'и разной размерности
        (8, 16, 32, ..., d_model) без дополнительного обучения.

        Args:
            dimensions: list[int] — размерности для Matryoshka (по умолчанию 2^k)
        """
        if dimensions is None:
            d = self.cfg.d_model
            max_k = int(torch.tensor(float(d)).log2().item()) + 1
            dimensions = [2**k for k in range(3, max_k) if 2**k <= d]
            if not dimensions or dimensions[-1] != d:
                dimensions.append(d)
        head = MatryoshkaHead(self.cfg.d_model, dimensions)
        self.model.matryoshka_head = head
        return self

    # === Freezing ===

    def freeze_layers(self, up_to=None, pattern=None):
        """Замораживает параметры модели.

        Args:
            up_to: заморозить первые N слоёв (None = не морозить)
            pattern: regex-паттерн для заморозки (например 'tok_emb|pos_emb')
        """
        import re
        for name, param in self.model.named_parameters():
            should_freeze = False
            if up_to is not None and 'layers.' in name:
                layer_idx = int(name.split('layers.')[1].split('.')[0])
                if layer_idx < up_to:
                    should_freeze = True
            if pattern is not None and re.search(pattern, name):
                should_freeze = True
            if should_freeze:
                param.requires_grad = False
        return self

    def progressive_freeze(self, step, total_steps):
        """Постепенная заморозка слоёв от нижних к верхним.

        Args:
            step: текущий шаг
            total_steps: общее число шагов
        """
        freezer = ProgressiveLayerFreezing(
            model=self.model,
            total_steps=total_steps,
        )
        freezer.step(step)
        return self

    # === Activation Checkpointing ===

    def enable_activation_checkpointing(self, every_n=2):
        """Включает gradient checkpointing для экономии памяти.

        Args:
            every_n: checkpoint каждый N-й слой
        """
        manager = ActivationCheckpointManager(every_n=every_n)
        manager.apply(self.model)
        return self

    # === Model Merging ===

    @staticmethod
    def merge_models(models, method='average', **kwargs):
        """Объединяет несколько моделей в одну.

        Args:
            models: list[nn.Module] — модели для объединения
            method: 'average' | 'slerp' | 'ties'

        Returns:
            merged model
        """
        if method == 'average':
            return merge_models_average(models)
        elif method == 'slerp':
            return merge_models_slerp(models, t=kwargs.get('t', 0.5))
        elif method == 'ties':
            return merge_models_ties(models, threshold=kwargs.get('threshold', 0.1))
        else:
            raise ValueError(f"Unknown merge method: {method}")

    # === Диагностика ===

    def surgery_report(self):
        """Отчёт о состоянии модели после операций."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            'total_params': total,
            'trainable_params': trainable,
            'frozen_params': frozen,
            'frozen_pct': round(100.0 * frozen / total, 2) if total > 0 else 0,
            'active_heads': self.count_active_heads() if hasattr(self.model, 'core') else 'N/A',
        }
