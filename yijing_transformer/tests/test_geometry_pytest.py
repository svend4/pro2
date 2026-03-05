"""Pytest-тесты математических свойств И-Цзин геометрии."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry import (
    generate_trigrams,
    generate_hexagrams,
    verify_yijing_properties,
    YiJingQuantizer,
    FactoredYiJingQuantizer,
    BianGuaTransform,
    RotaryEmbedding,
    apply_rotary_emb,
    SwiGLU,
    TrigramMoE,
)


class TestTrigrams:
    def test_shape(self):
        tri = generate_trigrams()
        assert tri.shape == (8, 3)

    def test_binary_values(self):
        tri = generate_trigrams()
        assert torch.all((tri == 1.0) | (tri == -1.0))

    def test_uniqueness(self):
        tri = generate_trigrams()
        assert torch.unique(tri, dim=0).shape[0] == 8

    def test_centered(self):
        tri = generate_trigrams()
        assert torch.norm(tri.sum(dim=0)).item() < 1e-6

    def test_norms(self):
        tri = generate_trigrams()
        norms = torch.norm(tri, dim=1)
        assert torch.allclose(norms, torch.tensor(3.0).sqrt())

    def test_min_pairwise_distance(self):
        tri = generate_trigrams()
        dists = torch.cdist(tri, tri)
        mask = ~torch.eye(8, dtype=torch.bool)
        min_dist = dists[mask].min().item()
        assert abs(min_dist - 2.0) < 0.01


class TestHexagrams:
    def test_shape(self):
        hex = generate_hexagrams()
        assert hex.shape == (64, 6)

    def test_binary_values(self):
        hex = generate_hexagrams()
        assert torch.all((hex == 1.0) | (hex == -1.0))

    def test_uniqueness(self):
        hex = generate_hexagrams()
        assert torch.unique(hex, dim=0).shape[0] == 64

    def test_centered(self):
        hex = generate_hexagrams()
        assert torch.norm(hex.sum(dim=0)).item() < 1e-6

    def test_factorization(self):
        """Каждая гексаграмма = верхняя триграмма ⊗ нижняя триграмма."""
        tri = generate_trigrams()
        hex = generate_hexagrams()
        for i in range(64):
            upper, lower = hex[i, :3], hex[i, 3:]
            assert any(torch.allclose(upper, tri[j]) for j in range(8))
            assert any(torch.allclose(lower, tri[j]) for j in range(8))

    def test_verify_properties(self):
        verify_yijing_properties(generate_trigrams(), generate_hexagrams())


class TestQuantizers:
    def test_naive_output_shape(self):
        q = YiJingQuantizer(temp=0.3)
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == x.shape

    def test_factored_output_shape(self):
        q = FactoredYiJingQuantizer(temp=0.3)
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == x.shape

    def test_hard_quantize_is_sign(self):
        q = FactoredYiJingQuantizer()
        x = torch.randn(100, 6)
        hard = q.hard_quantize(x)
        expected = torch.sign(x)
        assert torch.allclose(hard, expected)

    def test_equivalence_at_low_temp(self):
        naive = YiJingQuantizer(temp=0.01)
        factored = FactoredYiJingQuantizer(temp=0.01)
        x = torch.randn(4, 16, 6)
        naive_hard = torch.sign(naive(x))
        factored_hard = torch.sign(factored(x))
        match_rate = (naive_hard == factored_hard).float().mean().item()
        assert match_rate > 0.95

    def test_adaptive_temp(self):
        q = FactoredYiJingQuantizer(temp=0.3, adaptive_temp=True)
        assert hasattr(q, 'log_temp')
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == x.shape
        # Убедимся что temp можно оптимизировать
        loss = out.sum()
        loss.backward()
        assert q.log_temp.grad is not None

    def test_gradient_flow(self):
        """STE должен пропускать градиенты."""
        q = FactoredYiJingQuantizer(temp=0.3)
        x = torch.randn(2, 4, 6, requires_grad=True)
        out = q(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestBianGua:
    def test_output_shape(self):
        bg = BianGuaTransform(d_model=64)
        x = torch.randn(2, 8, 64)
        out = bg(x)
        assert out.shape == x.shape

    def test_full_inversion(self):
        bg = BianGuaTransform(d_model=64)
        with torch.no_grad():
            bg.change_logits.fill_(10.0)
        z_in = torch.randn(2, 8, 6)
        change_prob = torch.sigmoid(bg.change_logits)
        z_transformed = z_in * (1 - 2 * change_prob)
        assert torch.allclose(z_transformed, -z_in, atol=1e-3)

    def test_no_change_at_init(self):
        """При инициализации (logits=0, scale=0.01) вклад минимален."""
        bg = BianGuaTransform(d_model=64)
        x = torch.randn(2, 8, 64)
        out = bg(x)
        diff = (out - x).abs().max().item()
        assert diff < 1.0  # Малое изменение


class TestRoPE:
    def test_output_shape(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=128)
        cos, sin = rope(64)
        assert cos.shape == (64, 32)
        assert sin.shape == (64, 32)

    def test_cache_extension(self):
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        cos, sin = rope(128)  # больше чем max_seq_len
        assert cos.shape == (128, 32)

    def test_apply_rotary(self):
        rope = RotaryEmbedding(dim=16, max_seq_len=32)
        cos, sin = rope(8)
        x = torch.randn(2, 4, 8, 16)  # B, H, T, D
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_relative_position(self):
        """RoPE должен давать одинаковый скор для одинаковых относительных позиций."""
        rope = RotaryEmbedding(dim=16, max_seq_len=64)
        cos, sin = rope(32)
        q = torch.randn(1, 1, 1, 16).expand(1, 1, 32, 16)
        k = q.clone()
        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)
        # Скалярное произведение q[t] и k[t] должно быть одинаковым для всех t
        dots = (q_rot * k_rot).sum(dim=-1).squeeze()  # (32,)
        assert torch.allclose(dots, dots[0].expand_as(dots), atol=1e-5)


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(d_model=64, hidden=128)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        ffn = SwiGLU(d_model=64, hidden=128)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestTrigramMoE:
    def test_output_shape(self):
        moe = TrigramMoE(d_model=64, n_experts=8, top_k=2, ffn_hidden=128)
        x = torch.randn(2, 8, 64)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss.item() >= 0

    def test_aux_loss_positive(self):
        moe = TrigramMoE(d_model=64, n_experts=8, top_k=2, ffn_hidden=128)
        x = torch.randn(2, 8, 64)
        _, aux_loss = moe(x)
        assert aux_loss.item() > 0

    def test_gradient_flow(self):
        moe = TrigramMoE(d_model=64, n_experts=8, top_k=2, ffn_hidden=128)
        x = torch.randn(2, 4, 64, requires_grad=True)
        out, aux = moe(x)
        (out.sum() + aux).backward()
        assert x.grad is not None
