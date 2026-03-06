"""
LoRA (Low-Rank Adaptation) для YiJing-Transformer.

Добавляет обучаемые low-rank матрицы к линейным слоям:
    W' = W + (α/r) · B @ A

где A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, r << min(d_in, d_out).

Использование:
    model = YiJingGPT(cfg)
    apply_lora(model, cfg)           # добавляет LoRA адаптеры
    freeze_non_lora(model)           # замораживает все кроме LoRA
    # ... обучение ...
    merge_lora(model)                # вливает LoRA в основные веса
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Линейный слой с LoRA адаптером.

    Заменяет nn.Linear, сохраняя исходные веса замороженными
    и добавляя low-rank обновление.
    """
    def __init__(self, base_linear: nn.Linear, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = base_linear.in_features
        d_out = base_linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Флаг merged — после merge_lora обновление вливается в base
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
        base_out = self.base(x)
        lora_out = F.linear(
            self.lora_dropout(x),
            self.lora_B @ self.lora_A
        ) * self.scaling
        return base_out + lora_out

    def merge(self):
        """Вливает LoRA обновление в основные веса."""
        if not self.merged:
            with torch.no_grad():
                self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True

    def unmerge(self):
        """Отменяет merge для продолжения обучения."""
        if self.merged:
            with torch.no_grad():
                self.base.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False

    @property
    def weight(self):
        """Совместимость с weight tying."""
        if self.merged:
            return self.base.weight
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def apply_lora(model, cfg):
    """
    Применяет LoRA адаптеры к модели.

    По умолчанию адаптирует q_proj и v_proj (как в оригинальной LoRA).
    Можно расширить на k_proj, out, w1, w2, w3 через cfg.lora_targets.
    """
    targets = cfg.lora_targets or ['q_proj', 'v_proj']
    count = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Проверяем, что имя модуля заканчивается на один из targets
            short_name = name.split('.')[-1]
            if short_name in targets:
                # Находим родительский модуль
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)

                lora_linear = LoRALinear(
                    module,
                    rank=cfg.lora_rank,
                    alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                )
                setattr(parent, parts[-1], lora_linear)
                count += 1

    return count


def freeze_non_lora(model):
    """Замораживает все параметры кроме LoRA."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_all(model):
    """Размораживает все параметры."""
    for param in model.parameters():
        param.requires_grad = True


def merge_lora(model):
    """Вливает все LoRA адаптеры в основные веса."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora(model):
    """Отменяет merge для всех LoRA адаптеров."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def count_lora_parameters(model):
    """Подсчитывает количество LoRA параметров."""
    lora_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_' in name:
            lora_params += param.numel()
    return lora_params, total_params


def save_lora_weights(model, path):
    """Сохраняет только LoRA веса (компактно)."""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state[name] = param.data
    torch.save(lora_state, path)


def load_lora_weights(model, path, device='cpu'):
    """Загружает LoRA веса."""
    lora_state = torch.load(path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
