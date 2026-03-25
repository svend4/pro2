"""
tests/test_polyglot_modules.py — Smoke-тесты для трёх полиглот-модулей.

Проверяет ключевые компоненты без реального обучения:
  - polyglot_translation: TranslationHead, CrossTranslator, CycleConsistencyLoss
  - polyglot_curriculum: DifficultyEstimator, CurriculumScheduler, CurriculumDataLoader
  - polyglot_supervised: DistillationLoss, ContrastiveSupervisedLoss, AnnotatedPairLoss,
                         WarmupCosineScheduler, LayerFreezer

pytest tests/test_polyglot_modules.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from yijing_transformer.models.polyglot_translation import (
    TranslationHead, CrossTranslator, CycleConsistencyLoss,
)
from yijing_transformer.models.polyglot_curriculum import (
    DifficultyEstimator, CurriculumScheduler, CurriculumDataLoader,
    split_corpus_to_fragments,
)
from yijing_transformer.models.polyglot_supervised import (
    SupervisedConfig, DistillationLoss, ContrastiveSupervisedLoss,
    AnnotatedPairLoss, WarmupCosineScheduler, LayerFreezer,
)


# ══════════════════════════════════════════════════════════════════
# 1. polyglot_translation — кросс-перевод между музыкантами
# ══════════════════════════════════════════════════════════════════

def test_translation_head_shape():
    """TranslationHead: выходной тензор имеет правильную форму (B, T, tgt_vocab_size)."""
    head = TranslationHead(d_model=64, tgt_vocab_size=100)
    x = torch.randn(2, 8, 64)
    out = head(x)
    assert out.shape == (2, 8, 100), \
        f"Ожидали (2, 8, 100), получили {out.shape}"


def test_translation_head_greedy():
    """translate_greedy: возвращает целочисленный тензор формы (B, T)."""
    head = TranslationHead(d_model=64, tgt_vocab_size=100)
    x = torch.randn(3, 5, 64)
    tokens = head.translate_greedy(x)
    assert tokens.shape == (3, 5), \
        f"Ожидали (3, 5), получили {tokens.shape}"
    assert tokens.dtype in (torch.int64, torch.int32), \
        f"Ожидали целочисленный тип, получили {tokens.dtype}"


def test_cross_translator_creation():
    """CrossTranslator: создаёт ровно 12 голов перевода (4 музыканта × 3 направления)."""
    ct = CrossTranslator(d_model=64)
    assert len(ct.heads) == 12, \
        f"Ожидали 12 голов перевода, получили {len(ct.heads)}"


def test_cross_translator_translate():
    """CrossTranslator.translate: возвращает логиты правильной формы."""
    ct = CrossTranslator(d_model=64)
    hidden = torch.randn(2, 8, 64)
    # Переводим из формалиста в архетиписта
    logits = ct.translate('formalist', 'archetypist', hidden)
    expected_vocab = ct.vocab_sizes['archetypist']
    assert logits.shape == (2, 8, expected_vocab), \
        f"Ожидали (2, 8, {expected_vocab}), получили {logits.shape}"


def test_cycle_loss_runs():
    """cycle_loss: возвращает скалярный тензор с градиентом."""
    ct = CrossTranslator(d_model=64)
    hiddens = {
        name: torch.randn(2, 4, 64, requires_grad=True)
        for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']
    }
    loss = ct.cycle_loss(hiddens)
    assert loss.dim() == 0, f"Ожидали скаляр, получили dim={loss.dim()}"
    # Проверяем, что градиент можно вычислить
    loss.backward()
    # Хотя бы один параметр модели должен получить градиент
    has_grad = any(p.grad is not None for p in ct.parameters())
    assert has_grad, "Ни один параметр не получил градиент после backward()"


def test_pairwise_loss_runs():
    """pairwise_loss: возвращает скалярный loss при мок-целях."""
    ct = CrossTranslator(d_model=64)
    hiddens = {
        'formalist': torch.randn(2, 4, 64),
        'archetypist': torch.randn(2, 4, 64),
    }
    # Создаём мок-целевые токены для архетиписта
    targets = {
        'archetypist': torch.randint(1, ct.vocab_sizes['archetypist'], (2, 4)),
    }
    loss = ct.pairwise_loss(hiddens, targets)
    assert loss.dim() == 0, f"Ожидали скаляр, получили dim={loss.dim()}"


def test_cycle_consistency_loss():
    """CycleConsistencyLoss: обёртка с весом работает корректно."""
    ct = CrossTranslator(d_model=64)
    ccl = CycleConsistencyLoss(ct, weight=0.5)
    hiddens = {
        name: torch.randn(1, 4, 64)
        for name in ['formalist', 'archetypist', 'algorithmist', 'linguist']
    }
    weighted_loss, info = ccl(hiddens)
    assert weighted_loss.dim() == 0, "Ожидали скалярный loss"
    assert 'cycle_loss_raw' in info, "info должен содержать 'cycle_loss_raw'"
    assert 'cycle_loss_weighted' in info, "info должен содержать 'cycle_loss_weighted'"
    # Взвешенный loss = raw * 0.5
    assert abs(info['cycle_loss_weighted'] - info['cycle_loss_raw'] * 0.5) < 1e-5, \
        "Взвешенный loss не совпадает с raw * weight"


# ══════════════════════════════════════════════════════════════════
# 2. polyglot_curriculum — учебный план (curriculum training)
# ══════════════════════════════════════════════════════════════════

def test_difficulty_estimator_simple():
    """DifficultyEstimator: простой текст получает низкую сложность (< 0.3)."""
    est = DifficultyEstimator()
    # Короткий простой текст на кириллице
    score = est.estimate("Это мир и он есть.")
    assert score < 0.3, f"Простой текст получил сложность {score:.3f}, ожидали < 0.3"


def test_difficulty_estimator_complex():
    """DifficultyEstimator: текст с формулами и греческими буквами получает высокую сложность (> 0.3)."""
    est = DifficultyEstimator()
    # Сложный текст с формулами, греческими буквами и спецсимволами
    score = est.estimate(
        "Интеграл ∫₀^∞ e^{-αx²} dx = √(π/α), "
        "где α > 0. Формула Эйлера: e^{iπ} + 1 = 0. "
        "Тензор Римана R^μ_{νρσ} описывает кривизну пространства-времени."
    )
    assert score > 0.3, f"Сложный текст получил сложность {score:.3f}, ожидали > 0.3"


def test_difficulty_estimator_batch():
    """estimate_batch: возвращает список float-ов правильной длины."""
    est = DifficultyEstimator()
    texts = ["Один два три.", "E=mc² + ∫αβγ", "Это просто текст."]
    scores = est.estimate_batch(texts)
    assert len(scores) == 3, f"Ожидали 3 оценки, получили {len(scores)}"
    assert all(isinstance(s, float) for s in scores), "Все оценки должны быть float"


def test_curriculum_scheduler_easy():
    """CurriculumScheduler: на ранней стадии (progress=0.1) максимум сложности <= 0.3."""
    sched = CurriculumScheduler()
    d_min, d_max = sched.get_difficulty_range(0.1)
    assert d_max <= 0.3, \
        f"На progress=0.1 максимум сложности {d_max:.3f}, ожидали <= 0.3"


def test_curriculum_scheduler_expert():
    """CurriculumScheduler: на поздней стадии (progress=0.9) минимум сложности >= 0.5."""
    sched = CurriculumScheduler()
    d_min, d_max = sched.get_difficulty_range(0.9)
    assert d_min >= 0.5, \
        f"На progress=0.9 минимум сложности {d_min:.3f}, ожидали >= 0.5"


def test_curriculum_data_loader():
    """CurriculumDataLoader.get_batch: возвращает тексты и оценки сложности."""
    texts = [
        "Простой текст один.",
        "Простой текст два.",
        "Формула: ∫αβγ dx = Σ.",
        "E=mc² + квантовая теория поля.",
    ]
    loader = CurriculumDataLoader(texts)
    batch_texts, batch_diffs = loader.get_batch(
        difficulty_range=(0.0, 1.0), batch_size=2,
    )
    assert len(batch_texts) == 2, f"Ожидали 2 текста, получили {len(batch_texts)}"
    assert len(batch_diffs) == 2, f"Ожидали 2 сложности, получили {len(batch_diffs)}"
    assert all(isinstance(d, float) for d in batch_diffs), \
        "Сложности должны быть float"


def test_curriculum_data_loader_expand():
    """CurriculumDataLoader: при очень узком диапазоне расширяет и всё равно возвращает результаты."""
    texts = [
        "Один.", "Два.", "Три.", "Четыре.", "Пять.",
    ]
    loader = CurriculumDataLoader(texts)
    # Очень узкий диапазон — вряд ли кто-то попадёт точно
    batch_texts, batch_diffs = loader.get_batch(
        difficulty_range=(0.999, 1.0), batch_size=2,
    )
    assert len(batch_texts) >= 1, "Должен вернуть хотя бы один текст даже при узком диапазоне"


def test_split_corpus_to_fragments():
    """split_corpus_to_fragments: разбивает корпус на непустой список фрагментов."""
    corpus = "А" * 1000  # 1000 символов
    fragments = split_corpus_to_fragments(corpus, block_size=100, stride=50)
    assert len(fragments) > 0, "Должен вернуть хотя бы один фрагмент"
    assert all(len(f) == 100 for f in fragments), \
        "Все фрагменты должны быть длиной block_size"


# ══════════════════════════════════════════════════════════════════
# 3. polyglot_supervised — дообучение с учителем
# ══════════════════════════════════════════════════════════════════

def test_supervised_config_defaults():
    """SupervisedConfig: значения по умолчанию корректны."""
    cfg = SupervisedConfig()
    assert cfg.mode == 'annotated', f"Ожидали mode='annotated', получили '{cfg.mode}'"
    assert cfg.learning_rate == 1e-4, f"Ожидали lr=1e-4, получили {cfg.learning_rate}"
    assert cfg.temperature == 3.0, f"Ожидали temperature=3.0, получили {cfg.temperature}"
    assert cfg.alpha == 0.5, f"Ожидали alpha=0.5, получили {cfg.alpha}"
    assert cfg.warmup_steps == 100, f"Ожидали warmup_steps=100, получили {cfg.warmup_steps}"
    assert cfg.max_steps == 2000, f"Ожидали max_steps=2000, получили {cfg.max_steps}"
    assert cfg.freeze_embeddings is True, "Ожидали freeze_embeddings=True"
    assert cfg.freeze_musicians == [], "Ожидали пустой список freeze_musicians"


def test_distillation_loss_shapes():
    """DistillationLoss: возвращает скаляр и dict при подаче логитов без меток."""
    dl = DistillationLoss(temperature=3.0, alpha=0.5)
    student_logits = torch.randn(2, 8, 100)
    teacher_logits = torch.randn(2, 8, 100)
    loss, info = dl(student_logits, teacher_logits)
    assert loss.dim() == 0, f"Ожидали скаляр, получили dim={loss.dim()}"
    assert isinstance(info, dict), "Второй возврат должен быть dict"
    assert 'kl_loss' in info, "info должен содержать 'kl_loss'"


def test_distillation_loss_with_labels():
    """DistillationLoss: при подаче меток info содержит и kl_loss, и hard_loss."""
    dl = DistillationLoss(temperature=3.0, alpha=0.5)
    student_logits = torch.randn(2, 8, 100)
    teacher_logits = torch.randn(2, 8, 100)
    labels = torch.randint(1, 100, (2, 8))
    loss, info = dl(student_logits, teacher_logits, labels=labels)
    assert loss.dim() == 0, "Ожидали скаляр"
    assert 'kl_loss' in info, "info должен содержать 'kl_loss'"
    assert 'hard_loss' in info, "info должен содержать 'hard_loss'"


def test_contrastive_loss():
    """ContrastiveSupervisedLoss: возвращает скалярный loss."""
    cl = ContrastiveSupervisedLoss(margin=0.5)
    anchor = torch.randn(4, 64)
    positive = torch.randn(4, 64)
    negative = torch.randn(4, 64)
    loss, info = cl(anchor, positive, negative)
    assert loss.dim() == 0, f"Ожидали скаляр, получили dim={loss.dim()}"
    assert 'sim_pos' in info, "info должен содержать 'sim_pos'"
    assert 'sim_neg' in info, "info должен содержать 'sim_neg'"


def test_contrastive_loss_correct_ordering():
    """ContrastiveSupervisedLoss: sim_pos > sim_neg когда positive = anchor."""
    cl = ContrastiveSupervisedLoss(margin=0.5)
    anchor = torch.randn(4, 64)
    # Позитивный пример = якорь (максимальное сходство)
    positive = anchor.clone()
    # Негативный пример — случайный (низкое сходство)
    negative = torch.randn(4, 64)
    _, info = cl(anchor, positive, negative)
    assert info['sim_pos'] > info['sim_neg'], \
        f"sim_pos ({info['sim_pos']:.4f}) должен быть > sim_neg ({info['sim_neg']:.4f})"


def test_annotated_pair_loss():
    """AnnotatedPairLoss: возвращает loss и accuracy."""
    apl = AnnotatedPairLoss()
    logits = torch.randn(2, 8, 100)
    targets = torch.randint(1, 100, (2, 8))
    loss, info = apl(logits, targets)
    assert loss.dim() == 0, f"Ожидали скаляр, получили dim={loss.dim()}"
    assert 'accuracy' in info, "info должен содержать 'accuracy'"
    assert 0.0 <= info['accuracy'] <= 1.0, \
        f"accuracy должен быть в [0, 1], получили {info['accuracy']}"


def test_warmup_cosine_scheduler():
    """WarmupCosineScheduler: lr растёт при warmup и убывает после."""
    # Создаём простую модель и оптимизатор
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=10, max_steps=100, min_lr_ratio=0.1,
    )

    # Собираем lr на каждом шаге
    lrs = []
    for _ in range(50):
        lr = scheduler.step()
        lrs.append(lr)

    # Во время warmup (шаги 1–10) lr должен расти
    assert lrs[4] > lrs[0], \
        f"lr должен расти при warmup: шаг 1={lrs[0]:.6f}, шаг 5={lrs[4]:.6f}"

    # После warmup (шаг 15+) lr должен убывать
    assert lrs[29] < lrs[10], \
        f"lr должен убывать после warmup: шаг 11={lrs[10]:.6f}, шаг 30={lrs[29]:.6f}"


def test_layer_freezer():
    """LayerFreezer: freeze замораживает параметры, unfreeze_all размораживает."""

    # Создаём модель с подмодулями, имитирующими структуру PolyglotQuartet
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(100, 32)
            self.pos_emb = nn.Parameter(torch.randn(1, 16, 32))
            self.musicians = nn.ModuleDict({
                'formalist': nn.Linear(32, 32),
                'archetypist': nn.Linear(32, 32),
            })
            self.output = nn.Linear(32, 100)

    model = DummyModel()
    config = SupervisedConfig(
        freeze_embeddings=True,
        freeze_musicians=['formalist'],
    )
    freezer = LayerFreezer(model, config)

    # Замораживаем
    frozen_names = freezer.freeze()
    assert len(frozen_names) > 0, "Должен заморозить хотя бы один параметр"

    # tok_emb должен быть заморожен
    assert not model.tok_emb.weight.requires_grad, "tok_emb должен быть заморожен"

    # Формалист должен быть заморожен
    assert not model.musicians['formalist'].weight.requires_grad, \
        "musicians.formalist должен быть заморожен"

    # Архетипист НЕ должен быть заморожен
    assert model.musicians['archetypist'].weight.requires_grad, \
        "musicians.archetypist не должен быть заморожен"

    # Размораживаем всё
    freezer.unfreeze_all()
    assert model.tok_emb.weight.requires_grad, \
        "tok_emb должен быть разморожен после unfreeze_all"
    assert model.musicians['formalist'].weight.requires_grad, \
        "musicians.formalist должен быть разморожен после unfreeze_all"
