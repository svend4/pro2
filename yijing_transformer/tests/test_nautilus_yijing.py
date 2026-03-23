"""Tests for nautilus_yijing.py — NautilusYiJing model and components.

Covers: Q6GeometricRouter, YiJingMicroExpert, SimpleBridge, SimpleAnalogy,
        YiJingCoreBlock, NautilusYiJing (full model), helper functions.
"""
import pytest
import torch
import torch.nn as nn

from yijing_transformer.models.nautilus_yijing import (
    NautilusYiJingConfig,
    Q6GeometricRouter,
    YiJingMicroExpert,
    SimpleBridge,
    SimpleAnalogy,
    YiJingCoreBlock,
    NautilusYiJing,
    hamming_distance_q6,
    EXPERT_NAMES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_cfg():
    return NautilusYiJingConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4,
        block_size=16, d_expert=16, n_experts=6, top_k=2,
        dropout=0.0, use_rope=True, use_bian_gua=False,
        enable_synth=False,
    )


@pytest.fixture
def tiny_cfg():
    return NautilusYiJingConfig(
        vocab_size=32, d_model=16, n_layers=1, n_heads=2,
        block_size=8, d_expert=8, n_experts=3, top_k=2,
        dropout=0.0, use_rope=False, use_bian_gua=False,
        enable_synth=False,
    )


# ── Helper functions ───────────────────────────────────────────────────────────

class TestHelpers:
    def test_hamming_distance_identical(self):
        a = torch.tensor([1., -1., 1., -1., 1., -1.])
        b = a.unsqueeze(0)  # (1, 6)
        d = hamming_distance_q6(a, b)
        assert d.squeeze().item() == 0

    def test_hamming_distance_opposite(self):
        a = torch.tensor([1., 1., 1., 1., 1., 1.])
        b = torch.tensor([[-1., -1., -1., -1., -1., -1.]])  # (1, 6)
        d = hamming_distance_q6(a, b)
        assert d.squeeze().item() == 6

    def test_hamming_distance_one_flip(self):
        a = torch.tensor([1., 1., 1., 1., 1., 1.])
        b = torch.tensor([[1., 1., 1., 1., 1., -1.]])  # (1, 6)
        d = hamming_distance_q6(a, b)
        assert d.squeeze().item() == 1

    def test_expert_names_count(self):
        assert len(EXPERT_NAMES) >= 6


# ── Q6GeometricRouter ─────────────────────────────────────────────────────────

class TestQ6GeometricRouter:
    def test_output_shapes(self):
        router = Q6GeometricRouter(d_model=32, n_experts=6, top_k=2)
        x = torch.randn(2, 8, 32)
        weights, q6 = router(x)
        assert weights.shape == (2, 8, 6)
        assert q6.shape == (2, 8, 6)

    def test_weights_sum_to_one(self):
        router = Q6GeometricRouter(d_model=32, n_experts=6, top_k=2)
        x = torch.randn(2, 8, 32)
        weights, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_q6_range(self):
        """Q6 coordinates should be in [-1, 1] (tanh output)."""
        router = Q6GeometricRouter(d_model=32, n_experts=6, top_k=2)
        x = torch.randn(4, 16, 32)
        _, q6 = router(x)
        assert q6.min() >= -1.0
        assert q6.max() <= 1.0

    def test_gradient_flow(self):
        router = Q6GeometricRouter(d_model=32, n_experts=6, top_k=2)
        x = torch.randn(2, 4, 32, requires_grad=True)
        weights, q6 = router(x)
        loss = weights.sum() + q6.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_single_token(self):
        router = Q6GeometricRouter(d_model=16, n_experts=3, top_k=2)
        x = torch.randn(1, 1, 16)
        weights, q6 = router(x)
        assert weights.shape == (1, 1, 3)
        assert q6.shape == (1, 1, 6)


# ── YiJingMicroExpert ─────────────────────────────────────────────────────────

class TestYiJingMicroExpert:
    def test_output_shape(self):
        anchor = torch.tensor([1., -1., 1., -1., 1., -1.])
        expert = YiJingMicroExpert(d_model=32, d_expert=16, dropout=0.0,
                                   q6_anchor=anchor)
        x = torch.randn(2, 8, 32)
        out = expert(x)
        assert out.shape == (2, 8, 32)

    def test_gradient_flow(self):
        anchor = torch.tensor([1., 1., 1., -1., -1., -1.])
        expert = YiJingMicroExpert(d_model=32, d_expert=16, dropout=0.0,
                                   q6_anchor=anchor)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = expert(x)
        out.sum().backward()
        assert x.grad is not None


# ── SimpleBridge ───────────────────────────────────────────────────────────────

class TestSimpleBridge:
    def test_output_shape(self):
        bridge = SimpleBridge(d_model=32, n_experts=3)
        core_h = torch.randn(2, 8, 32)
        expert_outputs = [torch.randn(2, 8, 32) for _ in range(3)]
        weights = torch.softmax(torch.randn(2, 8, 3), dim=-1)
        out = bridge(core_h, expert_outputs, weights)
        assert out.shape == (2, 8, 32)


# ── SimpleAnalogy ──────────────────────────────────────────────────────────────

class TestSimpleAnalogy:
    def test_output_shape(self):
        analogy = SimpleAnalogy(d_model=32, n_experts=3)
        x = torch.randn(2, 8, 32)
        expert_outputs = [torch.randn(2, 8, 32) for _ in range(3)]
        weights = torch.softmax(torch.randn(2, 8, 3), dim=-1)
        out = analogy(x, expert_outputs, weights)
        assert out.shape == (2, 8, 32)


# ── YiJingCoreBlock ────────────────────────────────────────────────────────────

class TestYiJingCoreBlock:
    def test_output_shape(self, small_cfg):
        block = YiJingCoreBlock(small_cfg)
        x = torch.randn(2, 8, small_cfg.d_model)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self, small_cfg):
        block = YiJingCoreBlock(small_cfg)
        x = torch.randn(2, 4, small_cfg.d_model, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ── NautilusYiJing (Full Model) ───────────────────────────────────────────────

class TestNautilusYiJing:
    def test_forward_no_targets(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg)
        tokens = torch.randint(0, tiny_cfg.vocab_size, (2, 6))
        logits, loss, info = model(tokens)
        assert logits.shape == (2, 6, tiny_cfg.vocab_size)
        assert loss is None

    def test_forward_with_targets(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg)
        tokens = torch.randint(0, tiny_cfg.vocab_size, (2, 6))
        targets = torch.randint(0, tiny_cfg.vocab_size, (2, 6))
        logits, loss, info = model(tokens, targets=targets)
        assert logits.shape == (2, 6, tiny_cfg.vocab_size)
        assert loss is not None
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_info_dict(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg)
        tokens = torch.randint(0, tiny_cfg.vocab_size, (2, 4))
        _, _, info = model(tokens)
        assert isinstance(info, dict)

    def test_backward(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg)
        tokens = torch.randint(0, tiny_cfg.vocab_size, (2, 6))
        targets = torch.randint(0, tiny_cfg.vocab_size, (2, 6))
        _, loss, _ = model(tokens, targets=targets)
        loss.backward()
        # Check at least some parameters have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_count_parameters(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg)
        total, yijing = model.count_parameters()
        assert total > 0
        assert yijing >= 0
        assert yijing <= total

    def test_single_token_input(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg)
        tokens = torch.randint(0, tiny_cfg.vocab_size, (1, 1))
        logits, _, _ = model(tokens)
        assert logits.shape == (1, 1, tiny_cfg.vocab_size)

    def test_no_nan_in_output(self, small_cfg):
        model = NautilusYiJing(small_cfg)
        tokens = torch.randint(0, small_cfg.vocab_size, (2, 8))
        logits, _, _ = model(tokens)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_eval_mode(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg)
        model.eval()
        with torch.no_grad():
            tokens = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
            logits, _, _ = model(tokens)
        assert logits.shape[0] == 1


# ── Mixed Precision Tests ─────────────────────────────────────────────────────

class TestNautilusYiJingMixedPrecision:
    """Test model behavior under fp16/bf16 (CPU emulation)."""

    def test_forward_float16(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg).half()
        tokens = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
        with torch.no_grad():
            logits, _, _ = model(tokens)
        assert logits.dtype == torch.float16
        assert not torch.isnan(logits).any()

    @pytest.mark.skipif(
        not hasattr(torch, 'bfloat16'),
        reason="bfloat16 not available"
    )
    def test_forward_bfloat16(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg).to(torch.bfloat16)
        tokens = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
        with torch.no_grad():
            logits, _, _ = model(tokens)
        assert logits.dtype == torch.bfloat16
        assert not torch.isnan(logits).any()


# ── Device Transfer Tests ─────────────────────────────────────────────────────

class TestDeviceTransfer:
    """Test model can be moved between devices without breaking."""

    def test_cpu_to_cpu(self, tiny_cfg):
        model = NautilusYiJing(tiny_cfg).to('cpu')
        tokens = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
        logits, _, _ = model(tokens)
        assert logits.device.type == 'cpu'

    def test_state_dict_roundtrip(self, tiny_cfg):
        """Save and reload state dict preserves outputs."""
        model = NautilusYiJing(tiny_cfg)
        model.eval()
        tokens = torch.randint(0, tiny_cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out1, _, _ = model(tokens)

        # Roundtrip
        sd = model.state_dict()
        model2 = NautilusYiJing(tiny_cfg)
        model2.load_state_dict(sd)
        model2.eval()
        with torch.no_grad():
            out2, _, _ = model2(tokens)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_buffers_follow_device(self, tiny_cfg):
        """All registered buffers should be on the same device as model."""
        model = NautilusYiJing(tiny_cfg).to('cpu')
        for name, buf in model.named_buffers():
            assert buf.device.type == 'cpu', f"Buffer {name} on {buf.device}"
