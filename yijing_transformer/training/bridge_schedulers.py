"""
Bridge: Продвинутые LR scheduler'ы из utils_v12..v52.

Собирает и унифицирует все планировщики learning rate,
предоставляя единый интерфейс build_scheduler().

Источники:
  v14: Layerwise LR Decay (LLRD) — разные lr для разных глубин
  v18: WSD Scheduler — Warmup → Stable → Decay (Llama 3 style)
  v24: LLRD — продвинутая версия с группами параметров
  v29: Curriculum Scheduler — увеличение сложности данных
  v32: Cosine Annealing Warm Restarts — периодические перезапуски
  v35: Dynamic Batch Size Scaler — увеличение batch size вместо снижения lr
  v44: Warmup-Stable-Decay — вариант WSD с детальной конфигурацией

Использование:
    from training.bridge_schedulers import build_scheduler
    scheduler = build_scheduler(optimizer, cfg, scheduler_type='wsd')
    for step in range(total_steps):
        scheduler.step()
"""

from training.utils_v14 import get_layerwise_lr_groups
from training.utils_v18 import WSDScheduler
from training.utils_v24 import LLRD as LLRDv24, GradAccumulationScheduler
from training.utils_v29 import CurriculumScheduler
from training.utils_v32 import CosineAnnealingWarmRestarts as CosineRestarts
from training.utils_v35 import CurriculumScheduler as CurriculumV35, DynamicBatchSizeScaler
from training.utils_v44 import WarmupStableDecaySchedule

from training.optim import get_cosine_schedule, get_warmup_stable_decay_schedule


# ==================== Реестр планировщиков ====================

_SCHEDULER_REGISTRY = {
    'cosine': 'builtin',
    'wsd': WSDScheduler,
    'wsd_v44': WarmupStableDecaySchedule,
    'cosine_restarts': CosineRestarts,
    'curriculum': CurriculumV35,
}

_AUX_REGISTRY = {
    'llrd': LLRDv24,
    'grad_accum': GradAccumulationScheduler,
    'dynamic_batch': DynamicBatchSizeScaler,
}


def build_scheduler(optimizer, cfg, scheduler_type='cosine', **kwargs):
    """Фабрика LR scheduler'ов.

    Args:
        optimizer: PyTorch optimizer
        cfg: YiJingConfig
        scheduler_type: 'cosine' | 'wsd' | 'wsd_v44' | 'cosine_restarts' | 'curriculum'
        **kwargs: дополнительные параметры

    Returns:
        scheduler object с методом step()
    """
    if scheduler_type == 'cosine':
        return _CosineSchedulerWrapper(optimizer, cfg)

    elif scheduler_type == 'wsd':
        return WSDScheduler(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', cfg.warmup_steps),
            stable_steps=kwargs.get('stable_steps', cfg.total_steps // 2),
            decay_steps=kwargs.get('decay_steps', cfg.total_steps // 2),
            max_lr=kwargs.get('max_lr', cfg.lr),
            min_lr=kwargs.get('min_lr', cfg.lr * 0.1),
        )

    elif scheduler_type == 'wsd_v44':
        return WarmupStableDecaySchedule(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', cfg.warmup_steps),
            stable_steps=kwargs.get('stable_steps', cfg.total_steps // 3),
            decay_steps=kwargs.get('decay_steps', cfg.total_steps // 3),
        )

    elif scheduler_type == 'cosine_restarts':
        return CosineRestarts(
            optimizer,
            T_0=kwargs.get('T_0', cfg.total_steps // 4),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', cfg.lr * 0.01),
            warmup_steps=kwargs.get('warmup_steps', cfg.warmup_steps),
        )

    elif scheduler_type == 'curriculum':
        return CurriculumV35(
            total_steps=cfg.total_steps,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Available: {list(_SCHEDULER_REGISTRY.keys())}"
        )


def build_auxiliary(aux_type, **kwargs):
    """Вспомогательные scheduler'ы (batch size, grad accum).

    Args:
        aux_type: 'llrd' | 'grad_accum' | 'dynamic_batch'

    Returns:
        auxiliary scheduler
    """
    if aux_type == 'dynamic_batch':
        return DynamicBatchSizeScaler(**kwargs)
    elif aux_type == 'grad_accum':
        return GradAccumulationScheduler(**kwargs)
    elif aux_type == 'llrd':
        return LLRDv24(**kwargs)
    else:
        raise ValueError(f"Unknown aux_type: {aux_type}. Available: {list(_AUX_REGISTRY.keys())}")


class _CosineSchedulerWrapper:
    """Обёртка для встроенного cosine schedule, совместимая с интерфейсом step()."""

    def __init__(self, optimizer, cfg):
        self.optimizer = optimizer
        self.cfg = cfg
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr = get_cosine_schedule(
            self.step_count, self.cfg.warmup_steps,
            self.cfg.total_steps, self.cfg.lr,
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {'step_count': self.step_count}

    def load_state_dict(self, state):
        self.step_count = state['step_count']


def list_schedulers():
    """Возвращает список доступных scheduler'ов."""
    return {
        'schedulers': list(_SCHEDULER_REGISTRY.keys()),
        'auxiliary': list(_AUX_REGISTRY.keys()),
    }
