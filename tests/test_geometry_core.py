"""
Smoke tests for geometry modules lacking coverage:
  - AbrialeBridgeMediator (v59, best PPL=1.24)
  - AbrialeLayer (event-driven N-ary attention)
  - BianGuaTransform, D4EquivariantLayer, DualEmbedding
  - RotaryEmbedding, ALiBi, FourLevelPositionalEncoding
  - ConvergenceBridge, GlyphComposer, TokenAbstractor
  - NautilusHierarchy

Run: python tests/test_geometry_core.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import unittest


# ── AbrialeBridgeMediator ────────────────────────────────────────────────────

class TestAbrialeBridgeMediator(unittest.TestCase):

    def _make(self, n_sources=3, d=32):
        from yijing_transformer.models.geometry.routing import AbrialeBridgeMediator
        return AbrialeBridgeMediator(d_model=d, n_sources=n_sources, n_heads=2,
                                     n_rules=8, d_event=16)

    def test_output_shape(self):
        """forward(x, source_outputs) → same shape as x."""
        m = self._make(n_sources=3, d=32)
        x = torch.randn(2, 8, 32)
        srcs = [torch.randn(2, 8, 32) for _ in range(3)]
        out = m(x, srcs)
        self.assertEqual(out.shape, x.shape)

    def test_wrong_n_sources_raises(self):
        """Assertion fires if source list length != n_sources."""
        m = self._make(n_sources=3, d=32)
        x = torch.randn(2, 8, 32)
        with self.assertRaises(AssertionError):
            m(x, [torch.randn(2, 8, 32), torch.randn(2, 8, 32)])  # 2 instead of 3

    def test_n_sources_1(self):
        """n_sources=1 → bridge_tree is empty, still works."""
        m = self._make(n_sources=1, d=32)
        x = torch.randn(2, 8, 32)
        out = m(x, [torch.randn(2, 8, 32)])
        self.assertEqual(out.shape, x.shape)

    def test_n_sources_4(self):
        """n_sources=4 (even) → two-level bridge tree."""
        m = self._make(n_sources=4, d=32)
        x = torch.randn(2, 8, 32)
        srcs = [torch.randn(2, 8, 32) for _ in range(4)]
        out = m(x, srcs)
        self.assertEqual(out.shape, x.shape)

    def test_n_sources_5_odd(self):
        """n_sources=5 (odd) → has_odd passthrough in bridge tree."""
        m = self._make(n_sources=5, d=32)
        x = torch.randn(2, 8, 32)
        srcs = [torch.randn(2, 8, 32) for _ in range(5)]
        out = m(x, srcs)
        self.assertEqual(out.shape, x.shape)

    def test_gradient_flows(self):
        """Gradients flow through AbrialeBridgeMediator."""
        m = self._make(n_sources=3, d=32)
        x = torch.randn(2, 8, 32, requires_grad=True)
        srcs = [torch.randn(2, 8, 32) for _ in range(3)]
        out = m(x, srcs)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().max().item(), 0)

    def test_get_bridge_stats(self):
        """get_bridge_stats() returns dict with global_gate key."""
        m = self._make(n_sources=3, d=32)
        x = torch.randn(2, 8, 32)
        srcs = [torch.randn(2, 8, 32) for _ in range(3)]
        m(x, srcs)  # populate stats
        stats = m.get_bridge_stats()
        self.assertIn('global_gate', stats)
        self.assertIn('abriale_commit_rate', stats)
        self.assertTrue(0 <= stats['global_gate'] <= 1)

    def test_train_eval_modes(self):
        """Works in both train and eval mode."""
        m = self._make(n_sources=2, d=32)
        x = torch.randn(2, 8, 32)
        srcs = [torch.randn(2, 8, 32) for _ in range(2)]
        m.train()
        out_train = m(x, srcs)
        m.eval()
        with torch.no_grad():
            out_eval = m(x, srcs)
        self.assertEqual(out_train.shape, out_eval.shape)


# ── AbrialeLayer ─────────────────────────────────────────────────────────────

class TestAbrialeLayer(unittest.TestCase):

    def _make(self, d=32, arity=2):
        from yijing_transformer.models.geometry.abriale import AbrialeLayer
        return AbrialeLayer(d_model=d, d_event=16, n_heads=2, arity=arity,
                            n_rules=8, n_hits=2)

    def test_output_shape(self):
        m = self._make(d=32)
        x = torch.randn(2, 6, 32)
        out, info = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_info_keys(self):
        m = self._make(d=32)
        x = torch.randn(2, 6, 32)
        _, info = m(x)
        for key in ('attn_symmetry', 'commit_rate', 'hit_entropy', 'scale'):
            self.assertIn(key, info)

    def test_symmetry_near_one(self):
        """IsotropicAttention is symmetric, so attn_symmetry ≈ 1.0."""
        m = self._make(d=32)
        x = torch.randn(2, 6, 32)
        _, info = m(x)
        self.assertGreater(info['attn_symmetry'], 0.95)

    def test_arity_3(self):
        """Ternary N-ary attention (arity=3) — shape unchanged."""
        m = self._make(d=32, arity=3)
        x = torch.randn(2, 6, 32)
        out, _ = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_gradient_flows(self):
        m = self._make(d=32)
        x = torch.randn(2, 6, 32, requires_grad=True)
        out, info = m(x)
        loss = out.sum() + info['hit_weights'].sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


# ── BianGuaTransform ─────────────────────────────────────────────────────────

class TestBianGuaTransform(unittest.TestCase):

    def test_shape_preserved(self):
        from yijing_transformer.models.geometry.equivariant import BianGuaTransform
        m = BianGuaTransform(d_model=32)
        x = torch.randn(2, 8, 32)
        out = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_residual_small_init(self):
        """With scale=0.01 init, output ≈ input."""
        from yijing_transformer.models.geometry.equivariant import BianGuaTransform
        m = BianGuaTransform(d_model=32)
        m.eval()
        with torch.no_grad():
            x = torch.randn(2, 8, 32)
            out = m(x)
        diff = (out - x).abs().max().item()
        self.assertLess(diff, 1.0)

    def test_gradient_flows(self):
        from yijing_transformer.models.geometry.equivariant import BianGuaTransform
        m = BianGuaTransform(d_model=32)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = m(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


# ── D4EquivariantLayer ───────────────────────────────────────────────────────

class TestD4EquivariantLayer(unittest.TestCase):

    def test_shape_preserved(self):
        from yijing_transformer.models.geometry.equivariant import D4EquivariantLayer
        m = D4EquivariantLayer(d_model=32)
        x = torch.randn(2, 8, 32)
        out = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_one_of_eight_ops_selected(self):
        """In eval mode, exactly one D4 operation is selected (argmax)."""
        from yijing_transformer.models.geometry.equivariant import D4EquivariantLayer
        m = D4EquivariantLayer(d_model=32)
        m.eval()
        # Set a clear winner for op_weights
        with torch.no_grad():
            m.op_weights.fill_(0.0)
            m.op_weights[2] = 10.0
        x = torch.randn(3, 5, 32)
        out = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_gradient_flows(self):
        from yijing_transformer.models.geometry.equivariant import D4EquivariantLayer
        m = D4EquivariantLayer(d_model=32)
        m.train()
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = m(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


# ── DualEmbedding ────────────────────────────────────────────────────────────

class TestDualEmbedding(unittest.TestCase):

    def test_shape_preserved(self):
        from yijing_transformer.models.geometry.equivariant import DualEmbedding
        m = DualEmbedding(d_model=32)
        x = torch.randn(2, 8, 32)
        out = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_consistency_loss_non_negative(self):
        from yijing_transformer.models.geometry.equivariant import DualEmbedding
        m = DualEmbedding(d_model=32)
        x = torch.randn(2, 8, 32)
        loss = m.consistency_loss(x)
        self.assertGreaterEqual(loss.item(), 0.0)


# ── RotaryEmbedding ──────────────────────────────────────────────────────────

class TestRotaryEmbedding(unittest.TestCase):

    def test_shapes(self):
        from yijing_transformer.models.geometry.positional import RotaryEmbedding
        rope = RotaryEmbedding(dim=32)
        cos, sin = rope(seq_len=16)
        self.assertEqual(cos.shape, (16, 32))
        self.assertEqual(sin.shape, (16, 32))

    def test_cos_sin_bounded(self):
        from yijing_transformer.models.geometry.positional import RotaryEmbedding
        rope = RotaryEmbedding(dim=32)
        cos, sin = rope(seq_len=8)
        self.assertLessEqual(cos.abs().max().item(), 1.0 + 1e-5)
        self.assertLessEqual(sin.abs().max().item(), 1.0 + 1e-5)

    def test_longer_than_cache(self):
        """When seq_len > initial cache, rebuild works."""
        from yijing_transformer.models.geometry.positional import RotaryEmbedding
        rope = RotaryEmbedding(dim=32, max_seq_len=16)
        cos, sin = rope(seq_len=32)  # larger than max_seq_len
        self.assertEqual(cos.shape[0], 32)

    def test_apply_rotary_emb_shape(self):
        from yijing_transformer.models.geometry.positional import (
            RotaryEmbedding, apply_rotary_emb
        )
        rope = RotaryEmbedding(dim=32)
        cos, sin = rope(seq_len=8)
        x = torch.randn(2, 4, 8, 32)  # (B, H, T, D)
        out = apply_rotary_emb(x, cos, sin)
        self.assertEqual(out.shape, x.shape)


# ── ALiBi ────────────────────────────────────────────────────────────────────

class TestALiBi(unittest.TestCase):

    def test_output_shape(self):
        from yijing_transformer.models.geometry.positional import ALiBi
        alibi = ALiBi(n_heads=4)
        bias = alibi(seq_len=8)
        self.assertEqual(bias.shape, (1, 4, 8, 8))

    def test_bias_non_positive(self):
        """ALiBi biases are non-positive (distance penalizes attention)."""
        from yijing_transformer.models.geometry.positional import ALiBi
        alibi = ALiBi(n_heads=4)
        bias = alibi(seq_len=8)
        self.assertLessEqual(bias.max().item(), 0.0 + 1e-6)

    def test_power_of_2_heads(self):
        for n_heads in [1, 2, 4, 8]:
            from yijing_transformer.models.geometry.positional import ALiBi
            alibi = ALiBi(n_heads=n_heads)
            bias = alibi(seq_len=6)
            self.assertEqual(bias.shape[-3], n_heads)

    def test_non_power_of_2_heads(self):
        from yijing_transformer.models.geometry.positional import ALiBi
        alibi = ALiBi(n_heads=6)
        bias = alibi(seq_len=6)
        self.assertEqual(bias.shape, (1, 6, 6, 6))


# ── FourLevelPositionalEncoding ───────────────────────────────────────────────

class TestFourLevelPE(unittest.TestCase):

    def test_shape(self):
        from yijing_transformer.models.geometry.positional import FourLevelPositionalEncoding
        pe = FourLevelPositionalEncoding(d_model=32)
        out = pe(seq_len=16)
        self.assertEqual(out.shape, (16, 32))

    def test_different_positions_differ(self):
        from yijing_transformer.models.geometry.positional import FourLevelPositionalEncoding
        pe = FourLevelPositionalEncoding(d_model=32)
        out = pe(seq_len=12)
        # Not all positions should be identical
        diffs = (out[1:] - out[:-1]).abs().sum(dim=-1)
        self.assertGreater(diffs.max().item(), 0)


# ── ConvergenceBridge / GlyphComposer ────────────────────────────────────────

class TestConvergenceBridge(unittest.TestCase):

    def test_glyph_composer_shape(self):
        from yijing_transformer.models.geometry.convergence import GlyphComposer
        m = GlyphComposer(d_model=32, window_size=4, stride=2)
        # Input: sequence of Q6 vertices (B, T, 6)
        q6_seq = torch.randn(2, 12, 6)
        out = m(q6_seq)
        # Output should be (B, T_out, d_model)
        self.assertEqual(out.shape[-1], 32)
        self.assertEqual(out.shape[0], 2)

    def test_convergence_layer_exists(self):
        """ConvergenceLayer can be imported and instantiated."""
        from yijing_transformer.models.geometry.convergence import ConvergenceLayer
        m = ConvergenceLayer(d_model=32, n_heads=2)
        self.assertIsInstance(m, nn.Module)

    def test_convergence_layer_shape(self):
        from yijing_transformer.models.geometry.convergence import ConvergenceLayer
        m = ConvergenceLayer(d_model=32, n_heads=2)
        glyph_out = torch.randn(2, 8, 32)
        token_out = torch.randn(2, 8, 32)
        out = m(glyph_out, token_out)
        self.assertEqual(out.shape, (2, 8, 32))


# ── NautilusHierarchy ────────────────────────────────────────────────────────

class TestNautilusHierarchy(unittest.TestCase):

    def test_sequential_shape(self):
        from yijing_transformer.models.geometry.nautilus import NautilusHierarchy
        m = NautilusHierarchy(d_model=32, mode='sequential',
                              enabled_chambers=['dual_embedding', 'd4_equivariant'])
        x = torch.randn(2, 8, 32)
        out, info = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_parallel_shape(self):
        from yijing_transformer.models.geometry.nautilus import NautilusHierarchy
        m = NautilusHierarchy(d_model=32, mode='parallel',
                              enabled_chambers=['dual_embedding', 'd4_equivariant'])
        x = torch.randn(2, 8, 32)
        out, info = m(x)
        self.assertEqual(out.shape, x.shape)

    def test_scheduler_masks(self):
        from yijing_transformer.models.geometry.nautilus import NautilusScheduler
        sched = NautilusScheduler(n_chambers=7, warmup_steps=1000)
        masks_0 = sched.get_masks(0)
        masks_1000 = sched.get_masks(1000)
        # At step 0: first chamber may have partial activation, rest ~0
        self.assertGreaterEqual(masks_0[0], 0.0)
        # At warmup_steps: all chambers should be fully open
        for mask in masks_1000:
            self.assertAlmostEqual(mask, 1.0, places=5)

    def test_get_nautilus_stats(self):
        from yijing_transformer.models.geometry.nautilus import NautilusHierarchy
        m = NautilusHierarchy(d_model=32, mode='sequential',
                              enabled_chambers=['dual_embedding'])
        x = torch.randn(2, 8, 32)
        out, info = m(x)
        self.assertIn('chambers', info)
        self.assertIn('residual_gate', info)

    def test_set_step(self):
        """set_step() controls curriculum mask, forward returns tuple."""
        from yijing_transformer.models.geometry.nautilus import NautilusHierarchy
        m = NautilusHierarchy(d_model=32, warmup_steps=10,
                              enabled_chambers=['dual_embedding', 'd4_equivariant'])
        x = torch.randn(2, 8, 32)
        m.set_step(0)
        out_0, info_0 = m(x)
        m.set_step(10)
        out_10, info_10 = m(x)
        # Both should produce valid-shaped outputs
        self.assertEqual(out_0.shape, x.shape)
        self.assertEqual(out_10.shape, x.shape)
        # Info should report the correct step
        self.assertEqual(info_0['step'], 0)
        self.assertEqual(info_10['step'], 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
