"""Pytest-тесты для NautilusHierarchy (v63)."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT
from models.geometry.nautilus import (
    NautilusChamber,
    NautilusHierarchy,
    NautilusScheduler,
)


def make_cfg(**overrides):
    defaults = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=8,
        block_size=32, batch_size=2, use_rope=True, use_swiglu=True,
        use_bian_gua=True, use_hex_moe=False, use_flash_attn=False,
        adaptive_temp=True,
    )
    defaults.update(overrides)
    return YiJingConfig(**defaults)


class TestNautilusScheduler:
    """Тесты прогрессивной активации камер."""

    def test_all_zero_at_start(self):
        sched = NautilusScheduler(n_chambers=7, warmup_steps=700)
        masks = sched.get_masks(0)
        assert len(masks) == 7
        assert masks[0] == 0.0  # first chamber starts at 0
        for m in masks[1:]:
            assert m == 0.0

    def test_all_one_after_warmup(self):
        sched = NautilusScheduler(n_chambers=7, warmup_steps=700)
        masks = sched.get_masks(700)
        for m in masks:
            assert m == 1.0

    def test_monotonic_over_time(self):
        sched = NautilusScheduler(n_chambers=7, warmup_steps=700)
        prev_masks = sched.get_masks(0)
        for step in range(50, 750, 50):
            masks = sched.get_masks(step)
            for i in range(7):
                assert masks[i] >= prev_masks[i] - 1e-6
            prev_masks = masks

    def test_early_chambers_activate_first(self):
        sched = NautilusScheduler(n_chambers=7, warmup_steps=700)
        masks = sched.get_masks(150)  # ~1.5 chamber periods
        # Chamber 0 should be fully active, chamber 6 should be inactive
        assert masks[0] == 1.0
        assert masks[6] == 0.0


class TestNautilusHierarchy:
    """Тесты модуля NautilusHierarchy."""

    def test_sequential_forward_shape(self):
        nh = NautilusHierarchy(d_model=64, mode='sequential', warmup_steps=100)
        nh.set_step(200)  # all chambers active
        x = torch.randn(2, 16, 64)
        out, info = nh(x)
        assert out.shape == x.shape
        assert 'chambers' in info
        assert len(info['chambers']) == 7

    def test_parallel_forward_shape(self):
        nh = NautilusHierarchy(d_model=64, mode='parallel', warmup_steps=100)
        nh.set_step(200)
        x = torch.randn(2, 16, 64)
        out, info = nh(x)
        assert out.shape == x.shape

    def test_enabled_chambers_subset(self):
        nh = NautilusHierarchy(
            d_model=64,
            enabled_chambers=['heisenberg', 'flower_gat'],
            warmup_steps=100,
        )
        assert len(nh.chambers) == 2
        assert nh.chamber_names == ['heisenberg', 'flower_gat']

    def test_zero_step_minimal_change(self):
        """При step=0 все камеры curriculum=0, выход ≈ вход."""
        nh = NautilusHierarchy(d_model=64, warmup_steps=1000)
        nh.set_step(0)
        x = torch.randn(2, 16, 64)
        out, _ = nh(x)
        # С curriculum_mask=0 enrichment должен быть нулевым
        diff = (out - x).abs().max().item()
        assert diff < 1e-5, f"Expected near-zero diff at step 0, got {diff}"

    def test_gradient_flows(self):
        """Проверяем что градиенты протекают через все камеры."""
        nh = NautilusHierarchy(d_model=64, warmup_steps=10)
        nh.set_step(100)  # all active
        x = torch.randn(2, 16, 64, requires_grad=True)
        out, _ = nh(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_stats_reporting(self):
        nh = NautilusHierarchy(d_model=64, warmup_steps=100)
        nh.set_step(200)
        x = torch.randn(2, 16, 64)
        nh(x)
        stats = nh.get_nautilus_stats()
        assert 'nautilus/heisenberg/gate' in stats
        assert 'nautilus/flower_gat/scale' in stats
        assert 'nautilus/residual_gate' in stats


class TestNautilusInYiJingGPT:
    """Тесты интеграции NautilusHierarchy с YiJingGPT."""

    def test_forward_with_nautilus(self):
        cfg = make_cfg(use_nautilus=True, nautilus_warmup_steps=100)
        model = YiJingGPT(cfg)
        assert hasattr(model, 'nautilus')
        model.nautilus.set_step(200)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        targets = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss, _ = model(x, targets=targets)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert loss is not None

    def test_nautilus_subset_chambers(self):
        cfg = make_cfg(
            use_nautilus=True,
            nautilus_chambers='heisenberg,flower_gat,d4_equivariant',
            nautilus_warmup_steps=10,
        )
        model = YiJingGPT(cfg)
        assert len(model.nautilus.chambers) == 3
        model.nautilus.set_step(100)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_nautilus_parallel_mode(self):
        cfg = make_cfg(
            use_nautilus=True,
            nautilus_mode='parallel',
            nautilus_warmup_steps=10,
        )
        model = YiJingGPT(cfg)
        model.nautilus.set_step(100)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)
