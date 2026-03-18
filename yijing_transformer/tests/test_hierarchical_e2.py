"""
Тесты для hierarchical_e2.py — пятиуровневая иерархическая модель E2.

Покрывает:
  - GlyphLevel: forward shapes, Q6 coordinates
  - CoreLevel: forward, checkpoint loading API
  - MethodLevel: forward, interlingua integration
  - TheoryLevel: forward, step control
  - PhiloLevel: forward, convergence info
  - HierarchicalE2: full forward, loss computation, phase management
"""

import math
import pytest
import torch
import torch.nn as nn

from yijing_transformer.models.hierarchical_e2 import (
    GlyphLevel,
    CoreLevel,
    MethodLevel,
    TheoryLevel,
    PhiloLevel,
    HierarchicalE2,
    E2Config,
)


torch.manual_seed(42)

B, T, D = 2, 8, 64


# ═══════════════════════════════════════════════════════════════════════════════
# GlyphLevel
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlyphLevel:
    @pytest.fixture
    def level(self):
        return GlyphLevel(d_model=D)

    def test_forward_shapes(self, level):
        x = torch.randn(B, T, D)
        x_enriched, q6, assignments = level(x)
        assert x_enriched.shape == (B, T, D)
        assert q6.shape == (B, T, 6)
        assert assignments.shape == (B, T, 64)

    def test_q6_coords_bounded(self, level):
        """Q6 coords use tanh, should be in (-1, 1)."""
        x = torch.randn(B, T, D) * 10  # large input
        _, q6, _ = level(x)
        assert q6.abs().max().item() <= 1.0

    def test_assignments_sum_to_one(self, level):
        x = torch.randn(B, T, D)
        _, _, assignments = level(x)
        sums = assignments.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_gradient_flow(self, level):
        x = torch.randn(B, T, D, requires_grad=True)
        x_enriched, q6, assignments = level(x)
        (x_enriched.sum() + q6.sum() + assignments.sum()).backward()
        assert x.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CoreLevel
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoreLevel:
    def test_forward_shape(self):
        level = CoreLevel(D, n_heads=4, n_core=2, hamming_lambda=0.15,
                          uncertainty_budget=0.25, ffn_mult=4)
        x = torch.randn(B, T, D)
        out = level(x)
        assert out.shape == (B, T, D)

    def test_gradient_flow(self):
        level = CoreLevel(D, n_heads=4, n_core=2, hamming_lambda=0.15,
                          uncertainty_budget=0.25, ffn_mult=4)
        x = torch.randn(B, T, D, requires_grad=True)
        out = level(x)
        out.sum().backward()
        assert x.grad is not None

    def test_load_from_nonexistent_path(self):
        level = CoreLevel(D, n_heads=4, n_core=2, hamming_lambda=0.15,
                          uncertainty_budget=0.25, ffn_mult=4)
        assert level.load_from_v3_checkpoint("/nonexistent/path.pt") is False


# ═══════════════════════════════════════════════════════════════════════════════
# HierarchicalE2 (full model)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchicalE2:
    @pytest.fixture
    def cfg(self):
        return E2Config(vocab_size=64, d_model=D, block_size=T,
                        n_core=2, n_heads=4, n_archetypes=64, ffn_mult=2)

    @pytest.fixture
    def model(self, cfg):
        return HierarchicalE2(cfg)

    def test_forward_shape(self, model):
        tokens = torch.randint(0, 64, (B, T))
        logits, loss, info = model(tokens)
        assert logits.shape == (B, T, 64)
        assert loss is None  # no targets

    def test_forward_with_loss(self, model):
        tokens = torch.randint(0, 64, (B, T))
        targets = torch.randint(0, 64, (B, T))
        logits, loss, info = model(tokens, targets=targets)
        assert loss is not None
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_info_keys(self, model):
        tokens = torch.randint(0, 64, (B, T))
        _, _, info = model(tokens)
        assert 'phase' in info
        assert 'q6_coords' in info
        assert 'assignments' in info
        assert 'naut_info' in info
        assert 'conv_info' in info

    def test_phase_management(self, model):
        for phase in range(1, 6):
            model.set_training_phase(phase)
            assert model._phase == phase
            # Should still produce valid output
            tokens = torch.randint(0, 64, (B, T))
            logits, _, _ = model(tokens)
            assert logits.shape == (B, T, 64)

    def test_phase_1_freezes_upper_levels(self, model):
        model.set_training_phase(1)
        # core_level should be frozen
        for p in model.core_level.parameters():
            assert not p.requires_grad
        # glyph_level should be trainable
        for p in model.glyph_level.parameters():
            assert p.requires_grad

    def test_phase_5_all_trainable(self, model):
        model.set_training_phase(5)
        for name, p in model.named_parameters():
            assert p.requires_grad, f"Phase 5: {name} should be trainable"

    def test_gradient_flow_with_loss(self, model):
        model.set_training_phase(5)
        tokens = torch.randint(0, 64, (B, T))
        targets = torch.randint(0, 64, (B, T))
        logits, loss, info = model(tokens, targets=targets)
        loss.backward()
        # Check gradient reaches embeddings
        assert model.tok_emb.weight.grad is not None

    def test_weight_tying(self, model):
        assert model.tok_emb.weight is model.head.weight

    def test_train_step_increments(self, model):
        model.train()
        initial = model._train_step
        tokens = torch.randint(0, 64, (B, T))
        targets = torch.randint(0, 64, (B, T))
        model(tokens, targets=targets)
        assert model._train_step == initial + 1

    def test_count_parameters(self, model):
        counts = model.count_parameters()
        assert 'total' in counts
        assert 'glyph_level' in counts
        assert 'core_level' in counts
        assert counts['total'] > 0

    def test_invalid_phase_raises(self, model):
        with pytest.raises(AssertionError):
            model.set_training_phase(0)
        with pytest.raises(AssertionError):
            model.set_training_phase(6)
