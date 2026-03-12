"""
Bridge: Главная точка входа для всех пробуждённых утилит из utils_v12..v52.

Этот модуль — «мостовая плата», соединяющая 41 экспериментальный файл
(18,484 строки кода) с рабочей системой YiJing-Transformer.

Архитектура мостов:
    bridge.py (ВЫ ЗДЕСЬ)
    ├── bridge_optimizers.py    — Sophia, LAMB, Lion, SAM, Lookahead
    ├── bridge_schedulers.py    — WSD, Cosine Restarts, Curriculum, LLRD
    ├── bridge_regularization.py — Z-Loss, AGC, Label Smoothing, Mixup, etc.
    ├── bridge_monitors.py      — Loss Spike, Grokking, Gradient Flow
    ├── bridge_model_surgery.py — µP, Pruning, DoRA, Merging, Freezing
    └── (inference/)
        └── bridge_inference.py — Beam Search, Speculative, Dynamic Temp
    └── (data_utils/)
        └── bridge_augmentation.py — Packing, BPE Dropout, RAG, Freq Weighting

Каждый мост-файл — это «адаптер» (adapter pattern), который:
1. Импортирует классы из соответствующих utils_v*.py
2. Предоставляет единый API (фабрика / registry / suite)
3. Интегрируется с YiJingConfig через getattr с defaults

Использование:
    from training.bridge import TrainingBridge
    bridge = TrainingBridge(model, cfg)
    optimizer = bridge.build_optimizer(optimizer_type='sophia')
    scheduler = bridge.build_scheduler(scheduler_type='wsd')
    # В training loop:
    bridge.before_backward(logits, targets, loss)
    loss.backward()
    bridge.after_backward(model, step)
    bridge.after_step(step, loss, lr)

Источники (41 файл, v12..v52):
    v12: µP, Dynamic Temperature, Checkpoint Manager, Perplexity
    v13: Ring Attention, Selective Checkpointing, Weight Decay Scheduling
    v14: Layerwise LR, Data Mixing, Model Surgery (prune/grow/shrink)
    v15: Z-Loss, Gradient Accumulation Profiler, Vocab Expansion
    v16: Structured Pruning, Contrastive Loss, Sequence Packing, PCGrad, Merging
    v17: RAG Scaffold, Activation Statistics
    v18: WSD Scheduler, BPE Dropout, LAMB, NEFTune
    v19: Sophia, RMSNorm, Chunked Prefill, CAGrad, EMA Decay
    v20: DoRA, Sparse Attention, Gradient Vaccine, Loss Landscape
    v21: FSDP Simulator, Grokking Detector, GradEMA, AGC
    v22: Matryoshka Embeddings, Reptile, Token Frequency, SWA
    v23: Lookahead, Activation Histogram, Curriculum, EMA, Gradient Noise
    v24: LLRD, Grad Accum Scheduler, Attention Entropy, Loss Spike
    v25: Gradient Centralization, LR Finder, Param Efficiency, Batch Finder
    v26: Sparse Attention, Gradient Vaccine, Progressive Resizing, NCE Loss
    v27: SAM, Dynamic Temperature, Gradient Projection, Metrics Dashboard
    v28: Lookahead, SWA, Label Smoothing Warmup, Batch Finder
    v29: EMA, Curriculum, Gradient Noise, LR Probing, Weight Decay
    v30: Grad Centralization, AGC, Loss Spike, Param Freezing, Snapshotter
    v31: MI Estimator, Gradient Surgery, Spectral Reg, Token Freq Weighting
    v32: Gradient Histogram, Cosine Restarts, Multi-Scale Loss, Data Mixing
    v33: Gradient Penalty, Polyak Averaging, Loss Landscape, Adaptive Batch
    v34: Safe Grad Accum, LLRD, Token Dropout, Convergence Detector
    v35: Curriculum, Grad Noise, Dynamic Batch, Param Efficiency, Stability
    v36: EMA, Gradient Vaccine, AGC, LR Finder, Weight Standardization
    v37: Grad Centralization, Lookahead, SWA, Batch Warmup, Grad Penalty
    v38: Multi-Scale Loss, Grad Accum, Param Freezing, Cosine Restarts
    v39: Grad Projection, Loss Spike Recovery, Optimizer Pruning, Dropout
    v40: Spectral Norm, Grad Histogram, LR Probing, Mixed Precision
    v41: Activation Checkpoint, Param Freezer, Loss Landscape, State Inspector
    v42: Cosine Restarts, Grad Accum, Model EMA, Curriculum, Stability
    v43: Knowledge Distillation, Label Smoothing, Focal Loss, R-Drop
    v44: Grad Noise, Lookahead, LLRD, SWA, WSD Schedule
    v45: Grad Centralization, AdaFactor LR, Grad Penalty, SAM, Lion
    v46: Token Loss Weighting, Seq Packing, Dynamic Padding, Speculative
    v47: Grad Accum + Scaling, Param Noise, EMA Schedule, Loss Balancer
    v48: Beam Search, Nucleus Sampling, Rep Penalty, Temp Scheduler, KV Cache
    v49: PCGrad, AGC, Cosine Sim Loss, Mixup, CutMix
    v50: Entropy Reg, Confidence Penalty, Distillation Temp, Layer Freezing
    v51: Gradient Vaccine, Weight Standardization, Sharpness, LR Finder
    v52: Sliding Window, ALiBi, RoPE, Flash Attention, Multi-Query
"""


class TrainingBridge:
    """Единая точка входа для всех утилит из v12..v52.

    Собирает все мосты в один объект для удобного использования
    в training loop.
    """

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

        # Ленивая инициализация — создаём компоненты только при запросе
        self._optimizer = None
        self._scheduler = None
        self._regularization = None
        self._monitor = None
        self._surgeon = None
        self._data_pipeline = None

    # === Фабрики ===

    def build_optimizer(self, optimizer_type='adamw', wrapper=None, **kwargs):
        """Создаёт оптимизатор через bridge_optimizers."""
        from training.bridge_optimizers import build_advanced_optimizer
        self._optimizer = build_advanced_optimizer(
            self.model, self.cfg,
            optimizer_type=optimizer_type,
            wrapper=wrapper,
            **kwargs,
        )
        return self._optimizer

    def build_scheduler(self, optimizer=None, scheduler_type='cosine', **kwargs):
        """Создаёт LR scheduler через bridge_schedulers."""
        from training.bridge_schedulers import build_scheduler
        opt = optimizer or self._optimizer
        if opt is None:
            raise ValueError("Optimizer not set. Call build_optimizer() first.")
        self._scheduler = build_scheduler(opt, self.cfg, scheduler_type, **kwargs)
        return self._scheduler

    def build_regularization(self):
        """Создаёт RegularizationSuite через bridge_regularization."""
        from training.bridge_regularization import build_regularization_suite
        self._regularization = build_regularization_suite(self.cfg)
        return self._regularization

    def build_monitor(self, verbose=True):
        """Создаёт TrainingMonitor через bridge_monitors."""
        from training.bridge_monitors import TrainingMonitor
        self._monitor = TrainingMonitor(self.model, self.cfg, verbose=verbose)
        return self._monitor

    def build_surgeon(self):
        """Создаёт ModelSurgeon через bridge_model_surgery."""
        from training.bridge_model_surgery import ModelSurgeon
        self._surgeon = ModelSurgeon(self.model, self.cfg)
        return self._surgeon

    def build_data_pipeline(self):
        """Создаёт DataPipeline через bridge_augmentation."""
        from data_utils.bridge_augmentation import DataPipeline
        self._data_pipeline = DataPipeline(self.cfg)
        return self._data_pipeline

    # === Shortcut-методы для training loop ===

    def before_backward(self, logits, targets, base_loss):
        """Вызывается перед backward. Модифицирует loss.

        Returns:
            modified_loss
        """
        if self._regularization is not None:
            return self._regularization.apply_loss_modifiers(
                logits, targets, base_loss,
            )
        return base_loss

    def after_backward(self, step):
        """Вызывается после backward, перед optimizer.step().

        Применяет модификаторы градиентов.
        """
        if self._regularization is not None:
            self._regularization.apply_gradient_modifiers(self.model, step)

    def after_step(self, step, loss, lr=None, grad_norm=None,
                   val_loss=None, val_acc=None, train_acc=None):
        """Вызывается после optimizer.step().

        Обновляет scheduler и мониторинг.

        Returns:
            list[str]: алерты (пустой список = всё ок)
        """
        if self._scheduler is not None:
            self._scheduler.step()

        alerts = []
        if self._monitor is not None:
            alerts = self._monitor.step(
                loss, step, lr=lr, grad_norm=grad_norm,
                val_loss=val_loss, val_acc=val_acc, train_acc=train_acc,
            )

        return alerts

    # === Диагностика ===

    def diagnostic_report(self, step):
        """Полный диагностический отчёт."""
        report = {'step': step}

        if self._monitor is not None:
            report['monitor'] = self._monitor.diagnostic_report(step)

        if self._regularization is not None:
            report['active_regularizers'] = self._regularization.get_active_components()

        if self._surgeon is not None:
            report['model_surgery'] = self._surgeon.surgery_report()

        if self._data_pipeline is not None:
            report['data_pipeline'] = self._data_pipeline.get_active_components()

        return report

    def summary(self):
        """Краткая сводка по подключённым компонентам."""
        components = {
            'optimizer': type(self._optimizer).__name__ if self._optimizer else 'not set',
            'scheduler': type(self._scheduler).__name__ if self._scheduler else 'not set',
            'regularization': (self._regularization.get_active_components()
                               if self._regularization else 'not set'),
            'monitor': 'active' if self._monitor else 'not set',
            'surgeon': 'active' if self._surgeon else 'not set',
            'data_pipeline': (self._data_pipeline.get_active_components()
                              if self._data_pipeline else 'not set'),
        }
        return components
