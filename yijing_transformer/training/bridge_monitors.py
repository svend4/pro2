"""
Bridge: Мониторинг и диагностика обучения из utils_v12..v52.

Собирает все инструменты наблюдения за процессом обучения,
предоставляя единый TrainingMonitor с автоматическим обнаружением аномалий.

Источники:
  v12: Checkpoint Manager, Perplexity — базовые метрики
  v13: Throughput Benchmark — скорость обучения
  v20: Loss Landscape Probe — визуализация ландшафта потерь
  v21: Grokking Detector — обнаружение delayed generalization
  v24: Attention Entropy Monitor — здоровье attention
  v24: Loss Spike Detector — обнаружение скачков loss
  v25: Parameter Efficiency Tracker — эффективность параметров
  v30: State Snapshotter — снимки состояния для дебага
  v31: Mutual Information Estimator — MI между слоями
  v35: Training Stability Monitor — комплексный мониторинг
  v40: Spectral Norm Wrapper, Gradient Histogram — глубокая диагностика
  v47: Multi-Objective Loss Balancer — балансировка нескольких loss'ов
  v51: Gradient Flow Monitor, Sharpness Estimator

Использование:
    from training.bridge_monitors import TrainingMonitor
    monitor = TrainingMonitor(model, cfg)
    # В training loop:
    alerts = monitor.step(loss, step)
    if alerts:
        print(f"Alerts: {alerts}")
    # Периодическая диагностика:
    if step % 100 == 0:
        report = monitor.diagnostic_report(step)
"""

from training.utils_v12 import CheckpointManager, evaluate_perplexity
from training.utils_v13 import estimate_model_flops, benchmark_throughput
from training.utils_v20 import LossLandscapeProbe
from training.utils_v21 import GrokkingDetector
from training.utils_v24 import AttentionEntropyMonitor, LossSpikeDetector
from training.utils_v25 import ParamEfficiencyTracker
from training.utils_v30 import StateSnapshotter
from training.utils_v31 import MutualInformationEstimator
from training.utils_v35 import TrainingStabilityMonitor
from training.utils_v40 import GradientHistogramTracker
from training.utils_v47 import MultiObjectiveLossBalancer
from training.utils_v51 import SharpnessEstimator, GradientFlowMonitor


def _safe_init(cls, *args, **kwargs):
    """Безопасная инициализация: при ошибке возвращает None вместо краша."""
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        print(f"  [bridge_monitors] Warning: {cls.__name__} init failed: {e}")
        return None


class TrainingMonitor:
    """Комплексный мониторинг обучения с автоматическими алертами.

    Объединяет все диагностические инструменты из v12..v52
    в единый мониторинг с порогами и отчётами.
    """

    def __init__(self, model, cfg, verbose=True):
        self.model = model
        self.cfg = cfg
        self.verbose = verbose
        self._alerts = []
        self._history = {'loss': [], 'lr': [], 'grad_norm': []}

        # Safe init: каждый компонент инициализируется с try/except,
        # чтобы несовместимость сигнатуры одного не ломала остальные.

        # === Обнаружение аномалий ===
        self.loss_spike = _safe_init(LossSpikeDetector,
            window_size=getattr(cfg, 'spike_window', 100),
            spike_threshold=getattr(cfg, 'spike_threshold', 5.0),
        )
        self.grokking = _safe_init(GrokkingDetector,
            patience=getattr(cfg, 'grokking_patience', 1000),
        )
        self.stability = _safe_init(TrainingStabilityMonitor)

        # === Диагностика градиентов ===
        self.grad_flow = _safe_init(GradientFlowMonitor)
        self.grad_histogram = _safe_init(GradientHistogramTracker)

        # === Метрики модели ===
        self.attention_entropy = _safe_init(AttentionEntropyMonitor)
        self.param_efficiency = _safe_init(ParamEfficiencyTracker)
        self.sharpness = _safe_init(SharpnessEstimator)

        # === Управление чекпоинтами ===
        self.checkpoint_mgr = _safe_init(CheckpointManager,
            max_checkpoints=getattr(cfg, 'max_checkpoints', 5),
        )

        # === Multi-objective ===
        self.loss_balancer = None
        if getattr(cfg, 'use_loss_balancer', False):
            self.loss_balancer = _safe_init(MultiObjectiveLossBalancer,
                n_losses=getattr(cfg, 'n_loss_objectives', 2),
            )

        # === Snapshots ===
        self.snapshotter = _safe_init(StateSnapshotter)

    def step(self, loss, step, lr=None, grad_norm=None,
             train_acc=None, val_loss=None, val_acc=None):
        """Один шаг мониторинга.

        Args:
            loss: текущий loss
            step: номер шага
            lr: текущий learning rate
            grad_norm: норма градиентов
            train_acc: accuracy на train (опционально)
            val_loss: loss на validation (опционально)
            val_acc: accuracy на validation (опционально)

        Returns:
            list[str]: список алертов (пустой = всё ок)
        """
        alerts = []

        # Записываем историю
        self._history['loss'].append(loss)
        if lr is not None:
            self._history['lr'].append(lr)
        if grad_norm is not None:
            self._history['grad_norm'].append(grad_norm)

        # Проверка spike
        if self.loss_spike is not None:
            try:
                is_spike = self.loss_spike(loss)
                if is_spike:
                    alerts.append(f"[step {step}] Loss spike detected: {loss:.4f}")
            except Exception:
                pass

        # Проверка grokking
        if self.grokking is not None and val_acc is not None and train_acc is not None:
            try:
                grokking_detected = self.grokking(train_acc, val_acc, step)
                if grokking_detected:
                    alerts.append(f"[step {step}] Grokking detected! Train acc >> Val acc")
            except Exception:
                pass

        # Проверка стабильности
        if self.stability is not None:
            try:
                stability_alert = self.stability.check(loss, grad_norm, step)
                if stability_alert:
                    alerts.append(stability_alert)
            except Exception:
                pass

        # Проверка gradient flow
        if self.grad_flow is not None and step % 50 == 0:
            try:
                self.grad_flow.record(self.model)
                diag = self.grad_flow.diagnose()
                if isinstance(diag, dict) and diag.get('dead_layers', 0) > 0:
                    alerts.append(
                        f"[step {step}] Dead layers: {diag['dead_layers']}"
                    )
            except Exception:
                pass

        if self.verbose and alerts:
            for a in alerts:
                print(f"  ALERT: {a}")

        self._alerts.extend(alerts)
        return alerts

    def diagnostic_report(self, step):
        """Полный диагностический отчёт.

        Returns:
            dict с метриками по всем компонентам
        """
        report = {
            'step': step,
            'alerts_total': len(self._alerts),
        }

        # Loss statistics
        if self._history['loss']:
            recent = self._history['loss'][-100:]
            report['loss_mean'] = sum(recent) / len(recent)
            report['loss_std'] = _std(recent)
            report['loss_trend'] = _trend(recent)

        # Gradient flow (safe: utils use .record() + .diagnose())
        if self.grad_flow is not None:
            try:
                self.grad_flow.record(self.model)
                flow = self.grad_flow.diagnose()
                report['grad_flow'] = flow
            except Exception:
                report['grad_flow'] = 'unavailable'

        # Parameter efficiency (safe)
        if self.param_efficiency is not None:
            try:
                report['param_efficiency'] = self.param_efficiency.analyze(self.model)
            except Exception:
                report['param_efficiency'] = 'unavailable'

        return report

    def save_snapshot(self, step, model, optimizer):
        """Сохраняет снимок состояния для дебага."""
        if self.snapshotter is None:
            return
        self.snapshotter.save(step, model, optimizer)

    def should_save_checkpoint(self, val_loss):
        """Проверяет, нужно ли сохранять чекпоинт (top-k по val_loss)."""
        if self.checkpoint_mgr is None:
            return False
        return self.checkpoint_mgr.should_save(val_loss)

    def balance_losses(self, losses_dict):
        """Балансирует несколько loss'ов (если loss_balancer активен).

        Args:
            losses_dict: {'main': loss1, 'aux': loss2, ...}

        Returns:
            balanced_loss: scalar
        """
        if self.loss_balancer is not None:
            return self.loss_balancer(losses_dict)
        return sum(losses_dict.values())

    def estimate_sharpness(self, loss_fn, data_batch):
        """Оценивает sharpness минимума (SAM-style).

        Returns:
            float: sharpness estimate, or None if estimator unavailable
        """
        if self.sharpness is None:
            return None
        return self.sharpness.estimate(self.model, loss_fn, data_batch)

    def get_alerts(self):
        """Возвращает все накопленные алерты."""
        return self._alerts.copy()

    def clear_alerts(self):
        """Очищает историю алертов."""
        self._alerts.clear()


def _std(values):
    """Стандартное отклонение для списка."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _trend(values, window=20):
    """Тренд: разница средних последних window vs предыдущих window."""
    if len(values) < window * 2:
        return 0.0
    recent = sum(values[-window:]) / window
    previous = sum(values[-2*window:-window]) / window
    return recent - previous
