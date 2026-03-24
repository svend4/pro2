"""
Bridge: Методы регуляризации из utils_v12..v52.

Собирает все техники регуляризации, предоставляя единый интерфейс
build_regularization_suite() для подключения к training loop.

Источники:
  v15: Z-Loss — стабилизация logits (PaLM style)
  v19: RMSNorm — быстрая нормализация (Llama style)
  v21: AGC — Adaptive Gradient Clipping (NFNet style)
  v30: Gradient Centralization — центрирование градиентов
  v31: Spectral Regularization — ограничение спектральной нормы
  v33: Gradient Penalty — R1/R2 penalty
  v34: Token Dropout — случайное выбрасывание токенов
  v36: Weight Standardization — нормализация весов
  v43: Label Smoothing, Focal Loss, R-Drop — loss-функции
  v49: Mixup / CutMix — аугментация для последовательностей
  v50: Entropy Regularization, Confidence Penalty

Использование:
    from training.bridge_regularization import build_regularization_suite
    reg = build_regularization_suite(cfg)
    # В training loop:
    loss = reg.apply_loss_modifiers(logits, targets, base_loss)
    reg.apply_gradient_modifiers(model, step)
"""

from training.utils_v15 import z_loss, compute_loss_with_z
from training.utils_v19 import RMSNorm
from training.utils_v21 import AGC
from training.utils_v30 import GradientCentralization, AdaptiveGradientClipping
from training.utils_v31 import SpectralRegularizer
from training.utils_v33 import GradientPenalty
from training.utils_v34 import TokenDropout
from training.utils_v36 import WeightStandardization
from training.utils_v43 import LabelSmoothingLoss, FocalLoss, RDropRegularization
from training.utils_v49 import MixupAugmentation, SequenceCutMix
from training.utils_v50 import EntropyRegularization, ConfidencePenalty

from training.regularization import GradientNoise, AntipodalRegularization


class RegularizationSuite:
    """Комбинированный набор регуляризаторов с единым интерфейсом.

    Активация каждого компонента контролируется через config.
    Неактивные компоненты не добавляют вычислительных затрат.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # === Loss modifiers ===
        self.use_z_loss = getattr(cfg, 'use_z_loss', False)
        self.z_loss_weight = getattr(cfg, 'z_loss_weight', 1e-4)

        self.use_label_smoothing = getattr(cfg, 'label_smoothing', 0.0) > 0
        if self.use_label_smoothing:
            self.label_smoother = LabelSmoothingLoss(
                smoothing=cfg.label_smoothing,
            )

        self.use_focal_loss = getattr(cfg, 'use_focal_loss', False)
        if self.use_focal_loss:
            self.focal_loss = FocalLoss(
                gamma=getattr(cfg, 'focal_gamma', 2.0),
            )

        self.use_entropy_reg = getattr(cfg, 'use_entropy_reg', False)
        if self.use_entropy_reg:
            self.entropy_reg = EntropyRegularization(
                weight=getattr(cfg, 'entropy_reg_weight', 0.01),
            )

        self.use_confidence_penalty = getattr(cfg, 'use_confidence_penalty', False)
        if self.use_confidence_penalty:
            self.confidence_penalty = ConfidencePenalty(
                weight=getattr(cfg, 'confidence_penalty_weight', 0.1),
            )

        # === Gradient modifiers ===
        self.use_gradient_noise = getattr(cfg, 'use_gradient_noise', False)
        if self.use_gradient_noise:
            self.grad_noise = GradientNoise(
                eta=getattr(cfg, 'gradient_noise_eta', 0.01),
            )

        self.use_agc = getattr(cfg, 'use_agc', False)
        if self.use_agc:
            self.agc = AdaptiveGradientClipping(
                clip_factor=getattr(cfg, 'agc_clip_factor', 0.01),
            )

        self.use_grad_centralization = getattr(cfg, 'use_grad_centralization', False)
        if self.use_grad_centralization:
            self.grad_central = GradientCentralization()

        self.use_spectral_reg = getattr(cfg, 'use_spectral_reg', False)
        if self.use_spectral_reg:
            self.spectral_reg = SpectralRegularizer(
                weight=getattr(cfg, 'spectral_reg_weight', 0.01),
            )

        # === Data augmentation ===
        self.use_mixup = getattr(cfg, 'use_mixup', False)
        if self.use_mixup:
            self.mixup = MixupAugmentation(
                alpha=getattr(cfg, 'mixup_alpha', 0.2),
            )

        self.use_cutmix = getattr(cfg, 'use_cutmix', False)
        if self.use_cutmix:
            self.cutmix = SequenceCutMix(
                alpha=getattr(cfg, 'cutmix_alpha', 1.0),
            )

        self.use_token_dropout = getattr(cfg, 'use_token_dropout', False)
        if self.use_token_dropout:
            self.token_dropout = TokenDropout(
                p=getattr(cfg, 'token_dropout_p', 0.1),
            )

        # === Антиподальная регуляризация (родная для YiJing) ===
        self.use_antipodal = getattr(cfg, 'use_antipodal_reg', False)
        if self.use_antipodal:
            self.antipodal_reg = AntipodalRegularization(
                weight=getattr(cfg, 'antipodal_weight', 0.01),
            )

    def apply_loss_modifiers(self, logits, targets, base_loss):
        """Применяет модификаторы к loss.

        Args:
            logits: (B, T, V) — логиты модели
            targets: (B, T) — целевые токены
            base_loss: scalar — базовый cross-entropy loss

        Returns:
            modified_loss: scalar
        """
        loss = base_loss

        if self.use_z_loss:
            loss = loss + z_loss(logits) * self.z_loss_weight

        if self.use_entropy_reg:
            loss = loss + self.entropy_reg(logits)

        if self.use_confidence_penalty:
            loss = loss + self.confidence_penalty(logits)

        return loss

    def apply_gradient_modifiers(self, model, step):
        """Применяет модификаторы к градиентам (вызывать после backward, до step).

        Args:
            model: модель с .parameters()
            step: текущий шаг обучения
        """
        if self.use_grad_centralization:
            self.grad_central.apply(model)

        if self.use_agc:
            self.agc.apply(model)

        if self.use_gradient_noise:
            self.grad_noise.add_noise(model)

    def augment_data(self, embeddings, targets):
        """Применяет data augmentation к embedding'ам.

        Args:
            embeddings: (B, T, D) — embedded токены
            targets: (B, T) — целевые токены

        Returns:
            (augmented_embeddings, augmented_targets)
        """
        if self.training_mode_active and self.use_token_dropout:
            embeddings = self.token_dropout(embeddings)

        if self.use_mixup:
            embeddings, targets = self.mixup(embeddings, targets)

        return embeddings, targets

    def set_model(self, model):
        """Привязка к модели для проверки training mode."""
        self._model = model

    @property
    def training_mode_active(self):
        """Проверка: активированы ли модификаторы (только в train mode)."""
        model = getattr(self, '_model', None)
        if model is not None:
            return model.training
        return True  # fallback если модель не привязана

    def compute_spectral_loss(self, model):
        """Спектральный регуляризатор (отдельно, т.к. нужна модель).

        Returns:
            scalar loss или 0.0
        """
        if self.use_spectral_reg:
            return self.spectral_reg(model)
        return 0.0

    def compute_antipodal_loss(self, quantizer, x):
        """Антиподальная регуляризация для квантизатора.

        Returns:
            scalar loss или 0.0
        """
        if self.use_antipodal:
            return self.antipodal_reg(quantizer, x)
        return 0.0

    def get_active_components(self):
        """Возвращает список активных регуляризаторов."""
        active = []
        for attr in ['use_z_loss', 'use_label_smoothing', 'use_focal_loss',
                      'use_entropy_reg', 'use_confidence_penalty',
                      'use_gradient_noise', 'use_agc', 'use_grad_centralization',
                      'use_spectral_reg', 'use_mixup', 'use_cutmix',
                      'use_token_dropout', 'use_antipodal']:
            if getattr(self, attr, False):
                active.append(attr.replace('use_', ''))
        return active


def build_regularization_suite(cfg):
    """Фабрика для создания RegularizationSuite."""
    return RegularizationSuite(cfg)


# === Standalone утилиты для прямого использования ===

def apply_weight_standardization(model):
    """Применяет Weight Standardization ко всем Conv/Linear слоям."""
    return WeightStandardization.apply_to_model(model)


def apply_rms_norm(model, d_model):
    """Заменяет LayerNorm на RMSNorm (быстрее, используется в Llama)."""
    return RMSNorm(d_model)
