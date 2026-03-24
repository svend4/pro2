"""Tests for geometry/abriale.py — Event-driven N-ary relationships (Abriale layer).

Covers: IsotropicAttention, EventGenerator, RuleBank, TransactionGate, AbrialeLayer.
"""
import pytest
import torch
import torch.nn as nn

from yijing_transformer.models.geometry.abriale import (
    IsotropicAttention,
    EventGenerator,
    RuleBank,
    TransactionGate,
    AbrialeLayer,
)

B, T, D = 2, 8, 32
D_EVENT = 16


# ── IsotropicAttention ────────────────────────────────────────────────────────

class TestIsotropicAttention:
    def test_output_shapes(self):
        attn = IsotropicAttention(d_model=D, n_heads=4, arity=2, dropout=0.0)
        x = torch.randn(B, T, D)
        out, weights = attn(x)
        assert out.shape == (B, T, D)
        assert weights.shape[0] == B

    def test_gradient_flow(self):
        attn = IsotropicAttention(d_model=D, n_heads=4, dropout=0.0)
        x = torch.randn(B, T, D, requires_grad=True)
        out, _ = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_single_token(self):
        attn = IsotropicAttention(d_model=D, n_heads=2, dropout=0.0)
        x = torch.randn(1, 1, D)
        out, _ = attn(x)
        assert out.shape == (1, 1, D)

    def test_no_nan(self):
        attn = IsotropicAttention(d_model=D, n_heads=4, dropout=0.0)
        x = torch.randn(B, T, D)
        out, _ = attn(x)
        assert not torch.isnan(out).any()


# ── EventGenerator ─────────────────────────────────────────────────────────────

class TestEventGenerator:
    def test_output_shapes(self):
        gen = EventGenerator(d_model=D, d_event=D_EVENT, n_event_types=8)
        x = torch.randn(B, T, D)
        events, types = gen(x)
        assert events.shape == (B, T, D_EVENT)
        assert types.shape == (B, T, 8)

    def test_event_types_softmax(self):
        gen = EventGenerator(d_model=D, d_event=D_EVENT, n_event_types=4)
        x = torch.randn(B, T, D)
        _, types = gen(x)
        sums = types.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flow(self):
        gen = EventGenerator(d_model=D, d_event=D_EVENT)
        x = torch.randn(B, T, D, requires_grad=True)
        events, types = gen(x)
        (events.sum() + types.sum()).backward()
        assert x.grad is not None


# ── RuleBank ───────────────────────────────────────────────────────────────────

class TestRuleBank:
    def test_output_shapes(self):
        bank = RuleBank(d_event=D_EVENT, d_model=D, n_rules=16, n_hits=4)
        events = torch.randn(B, T, D_EVENT)
        x = torch.randn(B, T, D)
        actions, hits = bank(events, x)
        assert actions.shape == (B, T, D)
        assert hits.shape == (B, T, 16)

    def test_temperature_property(self):
        bank = RuleBank(d_event=D_EVENT, d_model=D)
        temp = bank.temperature
        assert temp.item() > 0
        assert temp.item() < 100

    def test_gradient_flow(self):
        bank = RuleBank(d_event=D_EVENT, d_model=D, n_rules=8, n_hits=2)
        events = torch.randn(B, T, D_EVENT, requires_grad=True)
        x = torch.randn(B, T, D)
        actions, hits = bank(events, x)
        actions.sum().backward()
        assert events.grad is not None


# ── TransactionGate ────────────────────────────────────────────────────────────

class TestTransactionGate:
    def test_output_shape(self):
        gate = TransactionGate(d_model=D, d_event=D_EVENT)
        x = torch.randn(B, T, D)
        actions = torch.randn(B, T, D)
        types = torch.softmax(torch.randn(B, T, 8), dim=-1)
        out = gate(x, actions, types)
        assert out.shape == (B, T, D)


# ── AbrialeLayer (Full) ───────────────────────────────────────────────────────

class TestAbrialeLayer:
    def test_output_shapes(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, n_heads=4,
                             n_rules=16, n_hits=4, dropout=0.0)
        x = torch.randn(B, T, D)
        out, info = layer(x)
        assert out.shape == (B, T, D)
        assert isinstance(info, dict)

    def test_info_contents(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, n_heads=2,
                             n_rules=8, n_hits=2, dropout=0.0)
        x = torch.randn(B, T, D)
        _, info = layer(x)
        assert 'commit_rate' in info
        assert 'hit_entropy' in info
        assert 'event_type_entropy' in info
        assert 'scale' in info

    def test_residual_connection(self):
        """Output should be x + scaled_contribution."""
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, dropout=0.0)
        x = torch.randn(B, T, D)
        out, _ = layer(x)
        # If scale is small, output should be close to input
        diff = (out - x).abs().mean()
        assert diff < 10.0  # Reasonable bound

    def test_gradient_flow(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, n_heads=2,
                             n_rules=8, dropout=0.0)
        x = torch.randn(B, T, D, requires_grad=True)
        out, info = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_auxiliary_loss(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, n_heads=2,
                             n_rules=8, n_hits=2, dropout=0.0)
        x = torch.randn(B, T, D)
        _, info = layer(x)
        if 'hit_weights' in info:
            aux_loss = layer.get_auxiliary_loss(info['hit_weights'])
            assert aux_loss.shape == ()
            assert not torch.isnan(aux_loss)

    def test_no_nan(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, n_heads=2,
                             n_rules=16, dropout=0.0)
        x = torch.randn(B, T, D)
        out, _ = layer(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_eval_mode(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, dropout=0.1)
        layer.eval()
        x = torch.randn(B, T, D)
        with torch.no_grad():
            out, _ = layer(x)
        assert out.shape == (B, T, D)


# ── Mixed Precision Tests ─────────────────────────────────────────────────────

class TestAbrialeMixedPrecision:
    def test_float16(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, n_heads=2,
                             n_rules=8, dropout=0.0).half()
        x = torch.randn(B, T, D, dtype=torch.float16)
        with torch.no_grad():
            out, _ = layer(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

    @pytest.mark.skipif(
        not hasattr(torch, 'bfloat16'),
        reason="bfloat16 not available"
    )
    def test_bfloat16(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, n_heads=2,
                             n_rules=8, dropout=0.0).to(torch.bfloat16)
        x = torch.randn(B, T, D, dtype=torch.bfloat16)
        with torch.no_grad():
            out, _ = layer(x)
        assert out.dtype == torch.bfloat16


# ── Device Transfer Tests ─────────────────────────────────────────────────────

class TestAbrialeDeviceTransfer:
    def test_state_dict_roundtrip(self):
        layer1 = AbrialeLayer(d_model=D, d_event=D_EVENT, n_rules=8, dropout=0.0)
        layer1.eval()
        x = torch.randn(1, 4, D)
        with torch.no_grad():
            out1, _ = layer1(x)

        sd = layer1.state_dict()
        layer2 = AbrialeLayer(d_model=D, d_event=D_EVENT, n_rules=8, dropout=0.0)
        layer2.load_state_dict(sd)
        layer2.eval()
        with torch.no_grad():
            out2, _ = layer2(x)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_buffers_on_device(self):
        layer = AbrialeLayer(d_model=D, d_event=D_EVENT, dropout=0.0).to('cpu')
        for name, buf in layer.named_buffers():
            assert buf.device.type == 'cpu', f"Buffer {name} on {buf.device}"
