"""
tests/test_grand_orchestrator.py — Юнит-тесты для Гранд-Оркестратора.

Проверяет ключевые компоненты без реального обучения:
  - ModelWrapper: обёртка, проекция логитов, clamping, обработка info
  - OrchestraRouter: форма выхода, top-k фильтрация
  - SharedOrchestraMemory: обмен скрытыми состояниями, обработка None
  - CascadeConnector: смешивание логитов
  - GrandOrchestrator: blend / cascade / expert, генерация, фабрика

pytest tests/test_grand_orchestrator.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from yijing_transformer.models.grand_orchestrator import (
    ModelWrapper,
    OrchestraRouter,
    SharedOrchestraMemory,
    CascadeConnector,
    GrandOrchestrator,
    OrchestraConfig,
    build_grand_orchestrator,
)


# ══════════════════════════════════════════════════════════════════
# Вспомогательная мини-модель для тестов
# ══════════════════════════════════════════════════════════════════

class MiniModel(nn.Module):
    """Минимальная модель: embedding → linear head, forward(idx, targets)."""

    def __init__(self, vocab_size=64, d_model=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1),
            )
        return logits, loss, {'test': True}


# ══════════════════════════════════════════════════════════════════
# 1. ModelWrapper — унифицированная обёртка
# ══════════════════════════════════════════════════════════════════

def test_model_wrapper_standard():
    """ModelWrapper: стандартная обёртка сохраняет форму (B, T, vocab_size)."""
    model = MiniModel(vocab_size=64, d_model=32)
    wrapper = ModelWrapper(
        model=model, name='mini', model_type='standard',
        target_vocab_size=64, target_d_model=32, freeze=False,
    )
    idx = torch.randint(0, 64, (2, 8))
    logits, loss, info = wrapper(idx)
    assert logits.shape == (2, 8, 64), \
        f"Ожидали (2, 8, 64), получили {logits.shape}"
    assert loss is None, "Без targets loss должен быть None"
    assert isinstance(info, dict), "info должен быть dict"
    assert info.get('wrapper_name') == 'mini', \
        "info должен содержать wrapper_name='mini'"


def test_model_wrapper_vocab_projection():
    """ModelWrapper: при src_vocab=32, target_vocab=64 логиты проецируются через logit_proj."""
    model = MiniModel(vocab_size=32, d_model=32)
    wrapper = ModelWrapper(
        model=model, name='small', model_type='standard',
        target_vocab_size=64, target_d_model=32, freeze=False,
    )
    # Убеждаемся, что logit_proj — это Linear, а не Identity
    assert isinstance(wrapper.logit_proj, nn.Linear), \
        "logit_proj должен быть Linear при разных vocab_size"

    idx = torch.randint(0, 32, (2, 8))
    logits, loss, info = wrapper(idx)
    assert logits.shape == (2, 8, 64), \
        f"Ожидали (2, 8, 64) после проекции, получили {logits.shape}"


def test_model_wrapper_clamp_ids():
    """ModelWrapper: входные id > src_vocab клампятся, модель не падает."""
    model = MiniModel(vocab_size=32, d_model=32)
    wrapper = ModelWrapper(
        model=model, name='clamp_test', model_type='standard',
        target_vocab_size=64, target_d_model=32, freeze=False,
    )
    # Подаём id, выходящие за пределы vocab модели (32..63)
    idx = torch.randint(32, 64, (2, 8))
    logits, loss, info = wrapper(idx)
    # Главное — нет ошибки, форма корректна
    assert logits.shape == (2, 8, 64), \
        f"Ожидали (2, 8, 64), получили {logits.shape}"


def test_model_wrapper_non_dict_info():
    """ModelWrapper: если модель возвращает не-dict info, обёртка заменяет его на {}."""

    class ListInfoModel(nn.Module):
        """Модель, возвращающая список вместо dict в info."""
        def __init__(self):
            super().__init__()
            self.vocab_size = 64
            self.d_model = 32
            self.tok_emb = nn.Embedding(64, 32)
            self.head = nn.Linear(32, 64)

        def forward(self, idx, targets=None):
            x = self.tok_emb(idx)
            logits = self.head(x)
            # Возвращаем список вместо dict
            return logits, None, ['not', 'a', 'dict']

    model = ListInfoModel()
    wrapper = ModelWrapper(
        model=model, name='list_info', model_type='standard',
        target_vocab_size=64, target_d_model=32, freeze=False,
    )
    idx = torch.randint(0, 64, (2, 8))
    logits, loss, info = wrapper(idx)
    # info должен быть dict (список заменён на {})
    assert isinstance(info, dict), \
        f"Ожидали dict, получили {type(info)}"
    assert info.get('wrapper_name') == 'list_info', \
        "info должен содержать wrapper_name"


# ══════════════════════════════════════════════════════════════════
# 2. OrchestraRouter — маршрутизация между моделями
# ══════════════════════════════════════════════════════════════════

def test_router_output_shape():
    """OrchestraRouter: выход (B, n_models) с суммой ≈ 1 по каждому сэмплу."""
    router = OrchestraRouter(vocab_size=64, n_models=4, d_router=32)
    idx = torch.randint(0, 64, (2, 8))
    weights, info = router(idx)
    assert weights.shape == (2, 4), \
        f"Ожидали (2, 4), получили {weights.shape}"
    # Сумма весов по каждому сэмплу должна быть ≈ 1
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5), \
        f"Сумма весов должна быть ≈ 1, получили {sums}"


def test_router_top_k():
    """OrchestraRouter: при top_k=2 не более 2 ненулевых весов на сэмпл."""
    router = OrchestraRouter(vocab_size=64, n_models=4, d_router=32)
    idx = torch.randint(0, 64, (2, 8))
    weights, info = router(idx, top_k=2)
    # Считаем ненулевые веса (порог 1e-6 для числовой стабильности)
    nonzero_counts = (weights > 1e-6).sum(dim=-1)
    assert (nonzero_counts <= 2).all(), \
        f"Ожидали <= 2 ненулевых весов, получили {nonzero_counts}"


# ══════════════════════════════════════════════════════════════════
# 3. SharedOrchestraMemory — общая память оркестра
# ══════════════════════════════════════════════════════════════════

def test_memory_exchange():
    """SharedOrchestraMemory: exchange возвращает дельты правильной формы."""
    mem = SharedOrchestraMemory(d_model=32, n_models=3, memory_size=16)
    hiddens = [torch.randn(2, 8, 32) for _ in range(3)]
    deltas, info = mem.exchange(hiddens)
    assert len(deltas) == 3, f"Ожидали 3 дельты, получили {len(deltas)}"
    for i, delta in enumerate(deltas):
        assert delta is not None, f"Дельта {i} не должна быть None"
        # delta: (B, 1, d_model) — broadcast по T
        assert delta.shape[0] == 2 and delta.shape[2] == 32, \
            f"Дельта {i} имеет неожиданную форму {delta.shape}"


def test_memory_none_handling():
    """SharedOrchestraMemory: None на входе → None на выходе."""
    mem = SharedOrchestraMemory(d_model=32, n_models=3, memory_size=16)
    hiddens = [torch.randn(2, 8, 32), None, torch.randn(2, 8, 32)]
    deltas, info = mem.exchange(hiddens)
    assert len(deltas) == 3, f"Ожидали 3 элемента, получили {len(deltas)}"
    assert deltas[0] is not None, "Первая дельта должна быть тензором"
    assert deltas[1] is None, "Вторая дельта должна быть None (вход был None)"
    assert deltas[2] is not None, "Третья дельта должна быть тензором"


# ══════════════════════════════════════════════════════════════════
# 4. CascadeConnector — каскадное смешивание логитов
# ══════════════════════════════════════════════════════════════════

def test_cascade_blend():
    """CascadeConnector: blend_logits возвращает тензор той же формы."""
    cascade = CascadeConnector(vocab_size=64, n_models=3)
    prev_logits = torch.randn(2, 8, 64)
    curr_logits = torch.randn(2, 8, 64)
    blended = cascade.blend_logits(prev_logits, curr_logits, step=0)
    assert blended.shape == (2, 8, 64), \
        f"Ожидали (2, 8, 64), получили {blended.shape}"


# ══════════════════════════════════════════════════════════════════
# 5. GrandOrchestrator — главный оркестратор
# ══════════════════════════════════════════════════════════════════

def _make_orchestrator(mode='blend', n_models=3, vocab_size=64, d_model=32):
    """Вспомогательная функция: создаёт и финализирует оркестр с n мини-моделями."""
    config = OrchestraConfig(
        mode=mode, vocab_size=vocab_size, d_model=d_model,
        memory_size=16, expert_top_k=2,
    )
    orch = GrandOrchestrator(config)
    for i in range(n_models):
        orch.add_model(f'model_{i}', MiniModel(vocab_size=vocab_size, d_model=d_model))
    orch.finalize()
    return orch


def test_orchestrator_add_finalize():
    """GrandOrchestrator: после добавления 3 моделей и финализации model_names содержит 3 имени."""
    orch = _make_orchestrator(n_models=3)
    assert len(orch.model_names) == 3, \
        f"Ожидали 3 имени, получили {len(orch.model_names)}"
    assert orch.model_names == ['model_0', 'model_1', 'model_2'], \
        f"Неожиданные имена: {orch.model_names}"
    assert orch._finalized is True, "Оркестр должен быть финализирован"


def test_orchestrator_blend_mode():
    """GrandOrchestrator: forward в режиме blend возвращает (logits, loss, info)."""
    orch = _make_orchestrator(mode='blend')
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits, loss, info = orch(idx, targets)
    assert logits.shape == (2, 8, 64), \
        f"Ожидали (2, 8, 64), получили {logits.shape}"
    assert loss is not None, "При targets != None loss должен быть скаляром"
    assert loss.dim() == 0, f"loss должен быть скаляром, получили dim={loss.dim()}"
    assert info.get('mode') == 'blend', \
        f"Ожидали mode='blend' в info, получили {info.get('mode')}"


def test_orchestrator_cascade_mode():
    """GrandOrchestrator: forward в режиме cascade возвращает (logits, loss, info)."""
    orch = _make_orchestrator(mode='cascade')
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits, loss, info = orch(idx, targets)
    assert logits.shape == (2, 8, 64), \
        f"Ожидали (2, 8, 64), получили {logits.shape}"
    assert loss is not None, "При targets != None loss должен быть скаляром"
    assert info.get('mode') == 'cascade', \
        f"Ожидали mode='cascade' в info, получили {info.get('mode')}"


def test_orchestrator_expert_mode():
    """GrandOrchestrator: forward в режиме expert с top_k=2 работает корректно."""
    orch = _make_orchestrator(mode='expert')
    idx = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits, loss, info = orch(idx, targets)
    assert logits.shape == (2, 8, 64), \
        f"Ожидали (2, 8, 64), получили {logits.shape}"
    assert loss is not None, "При targets != None loss должен быть скаляром"
    assert info.get('mode') == 'expert', \
        f"Ожидали mode='expert' в info, получили {info.get('mode')}"
    assert info.get('top_k') == 2, \
        f"Ожидали top_k=2, получили {info.get('top_k')}"


def test_orchestrator_generate():
    """GrandOrchestrator: generate() возвращает расширенную последовательность."""
    orch = _make_orchestrator(mode='blend')
    idx = torch.randint(0, 64, (1, 4))
    gen_len = 10
    output = orch.generate(idx, max_len=gen_len, temperature=1.0, top_k=0)
    expected_len = 4 + gen_len
    assert output.shape == (1, expected_len), \
        f"Ожидали (1, {expected_len}), получили {output.shape}"


def test_orchestrator_list_models():
    """GrandOrchestrator: list_models возвращает корректную информацию о каждой модели."""
    orch = _make_orchestrator(n_models=3)
    models_info = orch.list_models()
    assert len(models_info) == 3, \
        f"Ожидали 3 записи, получили {len(models_info)}"
    for entry in models_info:
        assert 'name' in entry, "Каждая запись должна содержать 'name'"
        assert 'type' in entry, "Каждая запись должна содержать 'type'"
        assert 'params' in entry, "Каждая запись должна содержать 'params'"
        assert entry['params'] > 0, "Количество параметров должно быть > 0"


def test_build_grand_orchestrator():
    """build_grand_orchestrator: фабричная функция создаёт финализированный оркестр."""
    models = {
        f'mini_{i}': MiniModel(vocab_size=64, d_model=32)
        for i in range(3)
    }
    orch = build_grand_orchestrator(
        models=models, mode='blend',
        vocab_size=64, d_model=32, freeze=True, expert_top_k=2,
    )
    assert orch._finalized is True, "Оркестр должен быть финализирован"
    assert len(orch.model_names) == 3, \
        f"Ожидали 3 модели, получили {len(orch.model_names)}"
    # Проверяем, что forward работает
    idx = torch.randint(0, 64, (1, 4))
    logits, loss, info = orch(idx)
    assert logits.shape == (1, 4, 64), \
        f"Ожидали (1, 4, 64), получили {logits.shape}"
