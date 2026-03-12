"""
Bridge: Продвинутые оптимизаторы из utils_v12..v52.

Собирает и унифицирует все оптимизаторы, разработанные в экспериментальных
версиях, предоставляя единый интерфейс build_advanced_optimizer().

Источники:
  v18: LAMB — Layer-wise Adaptive Moments (large batch training)
  v19: Sophia — second-order optimizer (Hessian-based)
  v23: Lookahead — slow/fast weights (Zhang et al., 2019)
  v27: SAM — Sharpness-Aware Minimization (Foret et al., 2021)
  v45: Lion — Evolved Sign Momentum (Chen et al., 2024)

Использование:
    from training.bridge_optimizers import build_advanced_optimizer
    optimizer = build_advanced_optimizer(model, cfg, optimizer_type='sophia')
"""

from training.utils_v18 import LAMB
from training.utils_v19 import Sophia
from training.utils_v23 import Lookahead
from training.utils_v27 import SAM
from training.utils_v45 import Lion

from training.optim import build_optimizer as _build_base_optimizer


# ==================== Реестр оптимизаторов ====================

_OPTIMIZER_REGISTRY = {
    'adamw': None,  # built-in, обрабатывается через build_optimizer
    'lamb': LAMB,
    'sophia': Sophia,
    'lion': Lion,
}

_WRAPPER_REGISTRY = {
    'lookahead': Lookahead,
    'sam': SAM,
}


def build_advanced_optimizer(model, cfg, optimizer_type='adamw',
                             wrapper=None, **kwargs):
    """Фабрика оптимизаторов с поддержкой всех экспериментальных вариантов.

    Args:
        model: YiJingGPT модель
        cfg: YiJingConfig
        optimizer_type: 'adamw' | 'lamb' | 'sophia' | 'lion'
        wrapper: None | 'lookahead' | 'sam' — обёртка поверх базового
        **kwargs: дополнительные параметры для оптимизатора

    Returns:
        optimizer (может быть обёрнут в Lookahead / SAM)
    """
    # Базовый оптимизатор
    if optimizer_type == 'adamw':
        optimizer = _build_base_optimizer(
            model, cfg,
            llrd_factor=kwargs.get('llrd_factor', 1.0),
            embedding_lr_scale=kwargs.get('embedding_lr_scale', 1.0),
        )
    elif optimizer_type == 'lamb':
        optimizer = LAMB(
            model.parameters(),
            lr=kwargs.get('lr', cfg.lr),
            weight_decay=kwargs.get('weight_decay', cfg.weight_decay),
        )
    elif optimizer_type == 'sophia':
        optimizer = Sophia(
            model.parameters(),
            lr=kwargs.get('lr', cfg.lr),
            weight_decay=kwargs.get('weight_decay', cfg.weight_decay),
            rho=kwargs.get('rho', 0.04),
        )
    elif optimizer_type == 'lion':
        optimizer = Lion(
            model.parameters(),
            lr=kwargs.get('lr', cfg.lr * 0.3),  # Lion рекомендует 3-10x меньший lr
            weight_decay=kwargs.get('weight_decay', cfg.weight_decay * 3),
        )
    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type}. "
            f"Available: {list(_OPTIMIZER_REGISTRY.keys())}"
        )

    # Обёртка
    if wrapper == 'lookahead':
        optimizer = Lookahead(
            optimizer,
            k=kwargs.get('lookahead_k', 5),
            alpha=kwargs.get('lookahead_alpha', 0.5),
        )
    elif wrapper == 'sam':
        optimizer = SAM(
            optimizer,
            rho=kwargs.get('sam_rho', 0.05),
        )
    elif wrapper is not None:
        raise ValueError(
            f"Unknown wrapper: {wrapper}. "
            f"Available: {list(_WRAPPER_REGISTRY.keys())}"
        )

    return optimizer


def list_optimizers():
    """Возвращает список доступных оптимизаторов и обёрток."""
    return {
        'optimizers': list(_OPTIMIZER_REGISTRY.keys()),
        'wrappers': list(_WRAPPER_REGISTRY.keys()),
    }
