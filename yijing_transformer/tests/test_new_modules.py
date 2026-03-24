"""Pytest-тесты для новых модулей сессии: PseudoRAG, DDP, SixSourceLayer, SOLANAttention."""

import os
import sys
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT
from models.pseudo_rag import (
    PseudoRAGProjection,
    PseudoRAGDistillationLoss,
    q4_to_q6_embedding,
    q6_cluster_membership,
)
from training.ddp import DDPConfig, DDPWrapper, setup_ddp_from_env
from models.geometry.six_sources import SixSourceLayer, SixSourceConfig
from models.geometry.attention import SOLANAttention


def make_cfg(**overrides):
    defaults = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=8,
        block_size=32, batch_size=2, use_rope=True, use_swiglu=True,
        use_bian_gua=True, use_hex_moe=False, use_flash_attn=False,
        adaptive_temp=True,
    )
    defaults.update(overrides)
    return YiJingConfig(**defaults)


B, T, D = 2, 16, 64


# ========================= PseudoRAG =========================

class TestPseudoRAG:
    """Тесты для PseudoRAG-моста Q4↔Q6."""

    def test_projection_shape(self):
        """Проекция hidden → Q6 весов: (B,T,D) → (B,T,64)."""
        proj = PseudoRAGProjection(d_model=D)
        x = torch.randn(B, T, D)
        out = proj(x)
        assert out.shape == (B, T, 64)

    def test_q4_logits_shape(self):
        """Q4-логиты: (B,T,D) → (B,T,16)."""
        proj = PseudoRAGProjection(d_model=D)
        x = torch.randn(B, T, D)
        q4 = proj.get_q4_logits(x)
        assert q4.shape == (B, T, 16)

    def test_cluster_membership(self):
        """Кластерная матрица (64,16): бинарная, 4 гексаграммы на кластер."""
        mem = q6_cluster_membership()
        assert mem.shape == (64, 16)
        # Каждая гексаграмма принадлежит ровно одному кластеру
        assert (mem.sum(dim=1) == 1.0).all()
        # Каждый кластер содержит ровно 4 гексаграммы
        assert (mem.sum(dim=0) == 4.0).all()

    def test_q4_to_q6_embedding(self):
        """Вложение Q4→Q6: форма (16,6), значения ∈ {-1, 0, +1}."""
        embed = q4_to_q6_embedding()
        assert embed.shape == (16, 6)
        unique_vals = set(embed.unique().tolist())
        assert unique_vals <= {-1.0, 0.0, 1.0}
        # Последние 2 координаты всегда 0
        assert (embed[:, 4:] == 0).all()

    def test_distillation_loss_finite(self):
        """Distillation loss — конечный скаляр."""
        loss_fn = PseudoRAGDistillationLoss(temperature=2.0)
        q6_logits = torch.randn(B, T, 64)
        q4_targets = torch.randn(B, T, 16)
        loss = loss_fn(q6_logits, q4_targets)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_distillation_backward(self):
        """Градиенты протекают через distillation loss."""
        loss_fn = PseudoRAGDistillationLoss(temperature=2.0)
        q6_logits = torch.randn(B, T, 64, requires_grad=True)
        q4_targets = torch.randn(B, T, 16)
        loss = loss_fn(q6_logits, q4_targets)
        loss.backward()
        assert q6_logits.grad is not None
        assert q6_logits.grad.shape == (B, T, 64)


# ========================= DDP =========================

class TestDDP:
    """Тесты для DDP-конфигурации и обёртки (без реального distributed)."""

    def test_ddp_config_defaults(self):
        """DDPConfig: значения по умолчанию корректны."""
        cfg = DDPConfig()
        assert cfg.backend == 'nccl'
        assert cfg.world_size == 1
        assert cfg.local_rank == 0
        assert cfg.rank == 0
        assert cfg.master_addr == 'localhost'
        assert cfg.master_port == '12355'

    def test_setup_from_env_no_env(self):
        """setup_ddp_from_env возвращает None без переменных окружения."""
        # Убираем переменные, если вдруг есть
        for var in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
            os.environ.pop(var, None)
        result = setup_ddp_from_env()
        assert result is None

    def test_ddp_wrapper_init(self):
        """DDPWrapper инициализируется без ошибок (без distributed)."""
        model = nn.Linear(D, D)
        cfg = DDPConfig(backend='gloo', world_size=1, local_rank=0, rank=0)
        wrapper = DDPWrapper(model, cfg)
        assert wrapper.model is model
        assert wrapper.config is cfg
        assert wrapper.is_main_process is True


# ========================= SixSourceLayer =========================

class TestSixSourceLayer:
    """Тесты для единого слоя 6 теоретических источников."""

    def test_forward_shape(self):
        """Прямой проход: (B,T,D) → (B,T,D)."""
        layer = SixSourceLayer(d_model=D)
        x = torch.randn(B, T, D)
        out, aux = layer(x)
        assert out.shape == (B, T, D)

    def test_aux_info_keys(self):
        """aux_info содержит 6 ключей gate_* для каждого источника."""
        layer = SixSourceLayer(d_model=D)
        x = torch.randn(B, T, D)
        _, aux = layer(x)
        expected_keys = {
            'gate_palace', 'gate_antipodal', 'gate_triangular',
            'gate_kasatkin', 'gate_hermann', 'gate_belyaev',
        }
        assert set(aux.keys()) == expected_keys

    def test_backward(self):
        """Градиенты протекают через все 6 источников."""
        layer = SixSourceLayer(d_model=D)
        x = torch.randn(B, T, D, requires_grad=True)
        out, _ = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        # Проверяем, что у каждого источника есть grad
        for name, src in layer.sources.items():
            for p in src.parameters():
                assert p.grad is not None, f"Нет градиента у источника {name}"

    def test_individual_sources(self):
        """Каждый из 6 источников выдаёт (B,T,D)."""
        from models.geometry.six_sources import (
            PalaceSource, AntipodalSource, TriangularSource,
            KasatkinSource, HermannSource, BelyaevSource,
        )
        x = torch.randn(B, T, D)
        sources = [
            PalaceSource(D), AntipodalSource(D), TriangularSource(D),
            KasatkinSource(D), HermannSource(D), BelyaevSource(D),
        ]
        for src in sources:
            out = src(x)
            assert out.shape == (B, T, D), f"{src.__class__.__name__} форма != (B,T,D)"


# ========================= SOLANAttention =========================

class TestSOLANAttention:
    """Тесты для SOLAN-76 attention (Q6 геометрический приор)."""

    def test_forward_shape(self):
        """Прямой проход: (B,T,D) → (B,T,D)."""
        attn = SOLANAttention(d_model=D, n_heads=4, block_size=32)
        x = torch.randn(B, T, D)
        out = attn(x)
        assert out.shape == (B, T, D)

    def test_backward(self):
        """Градиенты протекают через SOLANAttention."""
        attn = SOLANAttention(d_model=D, n_heads=4, block_size=32)
        x = torch.randn(B, T, D, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (B, T, D)

    def test_gate_learnable(self):
        """Параметр gate существует и обучаем."""
        attn = SOLANAttention(d_model=D, n_heads=4, block_size=32)
        assert hasattr(attn, 'gate')
        assert isinstance(attn.gate, nn.Parameter)
        assert attn.gate.requires_grad is True


# ========================= Интеграция =========================

class TestExperimentalIntegration:
    """Интеграционные тесты: новые фичи в составе YiJingGPT."""

    def test_diff_attn_in_model(self):
        """YiJingGPT(use_diff_attn=True): forward без ошибок."""
        cfg = make_cfg(use_diff_attn=True)
        model = YiJingGPT(cfg)
        idx = torch.randint(0, 128, (B, T))
        logits, loss, _ = model(idx)
        assert logits.shape == (B, T, 128)

    def test_mtp_in_model(self):
        """YiJingGPT(mtp_n_future=3): forward + loss вычисляется."""
        cfg = make_cfg(mtp_n_future=3)
        model = YiJingGPT(cfg)
        idx = torch.randint(0, 128, (B, T))
        targets = torch.randint(0, 128, (B, T))
        logits, loss, _ = model(idx, targets=targets)
        assert loss is not None
        assert torch.isfinite(loss)

    def test_prefix_tuning_in_model(self):
        """YiJingGPT(prefix_len=8): создаёт модуль prefix_tuning."""
        cfg = make_cfg(prefix_len=8)
        model = YiJingGPT(cfg)
        assert model.prefix_tuning is not None
        assert model.prefix_len == 8

    def test_combined_features(self):
        """Несколько новых фич одновременно: diff_attn + prefix + mtp."""
        cfg = make_cfg(use_diff_attn=True, prefix_len=4, mtp_n_future=2)
        model = YiJingGPT(cfg)
        idx = torch.randint(0, 128, (B, T))
        targets = torch.randint(0, 128, (B, T))
        logits, loss, _ = model(idx, targets=targets)
        assert logits.shape[0] == B
        assert logits.shape[-1] == 128
        assert loss is not None
        assert torch.isfinite(loss)
