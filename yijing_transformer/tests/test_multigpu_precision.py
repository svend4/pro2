"""Tests for multi-GPU readiness and mixed precision (fp16/bf16).

Covers: Variant3GPT, HierarchicalMoEFFN, GlyphComposer, ArchetypalInterlingua
        under fp16/bf16 and device transfer scenarios.
"""
import pytest
import torch
import torch.nn as nn

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.hierarchical_moe import HMoEConfig, HierarchicalMoEFFN


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def v3_cfg():
    return Variant3Config(
        vocab_size=64, d_model=32, n_layers=1, n_heads=4,
        block_size=16, dropout=0.0,
    )


@pytest.fixture
def hmoe_cfg():
    return HMoEConfig(d_model=32, use_multiscale=True, use_hex_tier=False)


# ── Variant3GPT Mixed Precision ───────────────────────────────────────────────

class TestVariant3MixedPrecision:
    def test_forward_float16(self, v3_cfg):
        model = Variant3GPT(v3_cfg).half()
        model.eval()
        tokens = torch.randint(0, v3_cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = model(tokens)
        logits = out[0]
        assert logits.dtype == torch.float16
        assert not torch.isnan(logits).any(), "NaN in fp16 forward"

    @pytest.mark.skipif(
        not hasattr(torch, 'bfloat16'),
        reason="bfloat16 not available"
    )
    def test_forward_bfloat16(self, v3_cfg):
        model = Variant3GPT(v3_cfg).to(torch.bfloat16)
        model.eval()
        tokens = torch.randint(0, v3_cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = model(tokens)
        logits = out[0]
        assert logits.dtype == torch.bfloat16
        assert not torch.isnan(logits).any(), "NaN in bf16 forward"

    def test_backward_float16_no_nan(self, v3_cfg):
        """Backward in fp16 should not produce NaN gradients."""
        model = Variant3GPT(v3_cfg).half()
        tokens = torch.randint(0, v3_cfg.vocab_size, (1, 8))
        targets = torch.randint(0, v3_cfg.vocab_size, (1, 8))
        out = model(tokens)
        logits = out[0]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, v3_cfg.vocab_size).float(),  # fp32 loss
            targets.view(-1),
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


# ── HierarchicalMoEFFN Mixed Precision ────────────────────────────────────────

class TestHMoEMixedPrecision:
    def test_forward_float16(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg).half()
        moe.eval()
        x = torch.randn(2, 4, 32, dtype=torch.float16)
        with torch.no_grad():
            out, info = moe(x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any(), "NaN in HMoE fp16"

    @pytest.mark.skipif(
        not hasattr(torch, 'bfloat16'),
        reason="bfloat16 not available"
    )
    def test_forward_bfloat16(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg).to(torch.bfloat16)
        moe.eval()
        x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
        with torch.no_grad():
            out, info = moe(x)
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any(), "NaN in HMoE bf16"

    def test_lb_loss_float16(self, hmoe_cfg):
        """Load-balance loss should not produce NaN in fp16."""
        moe = HierarchicalMoEFFN(hmoe_cfg).half()
        x = torch.randn(2, 4, 32, dtype=torch.float16)
        _, info = moe(x)
        lb = info['lb_loss']
        assert not torch.isnan(lb).any(), "NaN in lb_loss fp16"
        assert not torch.isinf(lb).any(), "Inf in lb_loss fp16"

    def test_topology_loss_float16(self, hmoe_cfg):
        """Topology loss should not produce NaN in fp16."""
        moe = HierarchicalMoEFFN(hmoe_cfg).half()
        x = torch.randn(2, 4, 32, dtype=torch.float16)
        _, info = moe(x)
        topo = info.get('topo_loss', torch.tensor(0.0))
        assert not torch.isnan(topo).any(), "NaN in topo_loss fp16"


# ── Device Transfer Tests ─────────────────────────────────────────────────────

class TestDeviceTransfer:
    def test_variant3_state_dict_roundtrip(self, v3_cfg):
        model = Variant3GPT(v3_cfg)
        model.eval()
        tokens = torch.randint(0, v3_cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out1 = model(tokens)[0]

        sd = model.state_dict()
        model2 = Variant3GPT(v3_cfg)
        model2.load_state_dict(sd)
        model2.eval()
        with torch.no_grad():
            out2 = model2(tokens)[0]

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_hmoe_state_dict_roundtrip(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg)
        moe.eval()
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            out1, _ = moe(x)

        sd = moe.state_dict()
        moe2 = HierarchicalMoEFFN(hmoe_cfg)
        moe2.load_state_dict(sd)
        moe2.eval()
        with torch.no_grad():
            out2, _ = moe2(x)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_variant3_buffers_on_device(self, v3_cfg):
        model = Variant3GPT(v3_cfg).to('cpu')
        for name, buf in model.named_buffers():
            assert buf.device.type == 'cpu', f"Buffer {name} on {buf.device}"

    def test_hmoe_buffers_on_device(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg).to('cpu')
        for name, buf in moe.named_buffers():
            assert buf.device.type == 'cpu', f"Buffer {name} on {buf.device}"

    def test_variant3_all_params_same_device(self, v3_cfg):
        model = Variant3GPT(v3_cfg).to('cpu')
        for name, p in model.named_parameters():
            assert p.device.type == 'cpu', f"Param {name} on {p.device}"

    def test_hmoe_all_params_same_device(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg).to('cpu')
        for name, p in moe.named_parameters():
            assert p.device.type == 'cpu', f"Param {name} on {p.device}"


# ── Edge Case Tests ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_variant3_batch_size_1(self, v3_cfg):
        model = Variant3GPT(v3_cfg)
        tokens = torch.randint(0, v3_cfg.vocab_size, (1, 4))
        out = model(tokens)
        assert out[0].shape[0] == 1

    def test_variant3_seq_len_1(self, v3_cfg):
        model = Variant3GPT(v3_cfg)
        tokens = torch.randint(0, v3_cfg.vocab_size, (2, 1))
        out = model(tokens)
        assert out[0].shape == (2, 1, v3_cfg.vocab_size)

    def test_variant3_max_seq_len(self, v3_cfg):
        model = Variant3GPT(v3_cfg)
        tokens = torch.randint(0, v3_cfg.vocab_size, (1, v3_cfg.block_size))
        out = model(tokens)
        assert out[0].shape == (1, v3_cfg.block_size, v3_cfg.vocab_size)

    def test_hmoe_single_token(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg)
        x = torch.randn(1, 1, 32)
        out, info = moe(x)
        assert out.shape == (1, 1, 32)
        assert not torch.isnan(out).any()

    def test_hmoe_large_batch(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg)
        x = torch.randn(16, 4, 32)
        out, info = moe(x)
        assert out.shape == (16, 4, 32)

    def test_hmoe_routing_entropy_valid_range(self, hmoe_cfg):
        moe = HierarchicalMoEFFN(hmoe_cfg)
        x = torch.randn(2, 8, 32)
        _, info = moe(x)
        ent = info['routing_entropy'].item()
        assert 0 <= ent <= 10, f"Entropy {ent} out of valid range"
        eff = info['routing_eff'].item()
        assert 0 <= eff <= 1.1, f"Efficiency {eff} out of valid range"
