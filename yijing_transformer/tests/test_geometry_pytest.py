"""Pytest-тесты математических свойств И-Цзин геометрии."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry import (
    generate_trigrams,
    generate_hexagrams,
    generate_octograms,
    generate_tetragrams,
    generate_hypercube,
    generate_e8_roots,
    compare_e8_vs_hypercube,
    verify_yijing_properties,
    YiJingQuantizer,
    FactoredYiJingQuantizer,
    HierarchicalQuantizer,
    DeformableQuantizer,
    GumbelQuantizer,
    E8Quantizer,
    HexagramAttentionPattern,
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


# ==================== НОВЫЕ КОМПОНЕНТЫ v3 ====================

class TestHypercubeHierarchy:
    def test_tetragrams(self):
        t = generate_tetragrams()
        assert t.shape == (16, 4)
        assert torch.all((t == 1.0) | (t == -1.0))
        assert torch.unique(t, dim=0).shape[0] == 16

    def test_octograms(self):
        o = generate_octograms()
        assert o.shape == (256, 8)
        assert torch.all((o == 1.0) | (o == -1.0))
        assert torch.unique(o, dim=0).shape[0] == 256

    def test_hypercube_general(self):
        for n in [1, 2, 3, 4, 5, 6, 7, 8]:
            h = generate_hypercube(n)
            assert h.shape == (2**n, n)
            assert torch.all((h == 1.0) | (h == -1.0))
            assert torch.unique(h, dim=0).shape[0] == 2**n

    def test_centered(self):
        for n in [3, 4, 6, 8]:
            h = generate_hypercube(n)
            assert torch.norm(h.sum(dim=0)).item() < 1e-6

    def test_norms(self):
        for n in [3, 4, 6, 8]:
            h = generate_hypercube(n)
            norms = torch.norm(h, dim=1)
            assert torch.allclose(norms, torch.tensor(float(n)).sqrt())


class TestHierarchicalQuantizer:
    def test_output_shape_6d(self):
        q = HierarchicalQuantizer(total_dim=6, group_dim=3)
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == x.shape

    def test_output_shape_8d(self):
        q = HierarchicalQuantizer(total_dim=8, group_dim=4)
        x = torch.randn(2, 8, 8)
        out = q(x)
        assert out.shape == x.shape

    def test_various_factorizations(self):
        """Разные факторизации должны работать."""
        configs = [
            (4, 2),   # 2 биграммы
            (6, 2),   # 3 биграммы
            (6, 3),   # 2 триграммы
            (8, 2),   # 4 биграммы
            (8, 4),   # 2 тетраграммы
        ]
        for total_dim, group_dim in configs:
            q = HierarchicalQuantizer(total_dim=total_dim, group_dim=group_dim)
            x = torch.randn(2, 4, total_dim)
            out = q(x)
            assert out.shape == x.shape, f"Failed for ({total_dim}, {group_dim})"

    def test_gradient_flow(self):
        q = HierarchicalQuantizer(total_dim=8, group_dim=4, adaptive_temp=True)
        x = torch.randn(2, 4, 8, requires_grad=True)
        out = q(x)
        out.sum().backward()
        assert x.grad is not None

    def test_codebook_info(self):
        q = HierarchicalQuantizer(total_dim=8, group_dim=4)
        info = q.codebook_info()
        assert info['total_codewords'] == 256  # 16^2
        assert info['softmax_ops'] == 32  # 2*16

    def test_hard_quantize(self):
        q = HierarchicalQuantizer(total_dim=8, group_dim=2)
        x = torch.randn(10, 8)
        hard = q.hard_quantize(x)
        assert torch.all((hard == 1.0) | (hard == -1.0))

    def test_equivalence_with_factored(self):
        """При group_dim=3, total_dim=6 должен совпадать с FactoredYiJingQuantizer."""
        hq = HierarchicalQuantizer(total_dim=6, group_dim=3, temp=0.01)
        fq = FactoredYiJingQuantizer(temp=0.01)
        x = torch.randn(4, 16, 6)
        hq_hard = torch.sign(hq(x))
        fq_hard = torch.sign(fq(x))
        match_rate = (hq_hard == fq_hard).float().mean().item()
        assert match_rate > 0.95


class TestDeformableQuantizer:
    def test_output_shape(self):
        q = DeformableQuantizer(total_dim=6, group_dim=3)
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == x.shape

    def test_starts_as_hypercube(self):
        """При deform_scale=0 должен совпадать с обычным квантизатором."""
        q = DeformableQuantizer(total_dim=6, group_dim=3, deform_scale=0.0)
        cb = q.codebook
        base = q.base_codebook
        assert torch.allclose(cb, base)

    def test_deformation_changes_output(self):
        """Deformation should change quantization output vs base."""
        q = DeformableQuantizer(total_dim=6, group_dim=3, deform_scale=0.0)
        x = torch.randn(2, 4, 6)
        out_base = q(x)
        # Apply deformation
        with torch.no_grad():
            q.deform_scale.fill_(1.0)
            q.delta.normal_(0, 0.5)
        out_deformed = q(x)
        # Outputs should differ when codebook is deformed
        assert not torch.allclose(out_base, out_deformed, atol=1e-4)

    def test_deformation_stats(self):
        q = DeformableQuantizer(total_dim=6, group_dim=3)
        stats = q.deformation_stats()
        assert 'delta_norm' in stats
        assert stats['effective_shift'] == 0.0  # deform_scale starts at 0


class TestHexagramAttentionPattern:
    def test_output_shape(self):
        hap = HexagramAttentionPattern(d_model=64, block_size=32)
        x = torch.randn(2, 16, 64)
        bias = hap(x, T=16)
        assert bias.shape == (2, 1, 16, 16)

    def test_starts_at_zero(self):
        """Scale инициализирован нулём — bias должен быть нулевым."""
        hap = HexagramAttentionPattern(d_model=64, block_size=32)
        x = torch.randn(2, 16, 64)
        bias = hap(x, T=16)
        assert bias.abs().max().item() < 1e-6

    def test_gradient_flow(self):
        hap = HexagramAttentionPattern(d_model=64, block_size=32)
        x = torch.randn(2, 16, 64, requires_grad=True)
        bias = hap(x, T=16)
        bias.sum().backward()
        assert x.grad is not None


class TestCharTokenizer:
    def test_encode_decode(self):
        from tokenizer.char_tokenizer import CharTokenizer
        tok = CharTokenizer()
        text = "Hello, World!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        from tokenizer.char_tokenizer import CharTokenizer
        tok = CharTokenizer()
        size = tok.get_piece_size()
        assert size > 90  # ASCII printable + extras

    def test_from_text(self):
        from tokenizer.char_tokenizer import CharTokenizer
        tok = CharTokenizer.from_text("abc123")
        assert tok.get_piece_size() == 8  # 6 chars + PAD + UNK
        ids = tok.encode("abc")
        decoded = tok.decode(ids)
        assert decoded == "abc"


class TestGumbelQuantizer:
    """Тесты Gumbel-Softmax квантизатора."""

    def test_output_shape(self):
        gq = GumbelQuantizer(total_dim=6, group_dim=3)
        x = torch.randn(4, 16, 6)
        out = gq(x)
        assert out.shape == x.shape

    def test_training_mode(self):
        gq = GumbelQuantizer(total_dim=6, group_dim=3)
        gq.train()
        x = torch.randn(4, 8, 6, requires_grad=True)
        out = gq(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_eval_mode_hard(self):
        """В eval режиме должен выдавать точки кодбука."""
        gq = GumbelQuantizer(total_dim=6, group_dim=3)
        gq.eval()
        x = torch.randn(2, 4, 6)
        out = gq(x)
        # Каждая тройка координат должна быть ±1
        for i in range(2):
            upper = out[..., :3]
            lower = out[..., 3:]
            assert torch.all(upper.abs() == 1.0)
            assert torch.all(lower.abs() == 1.0)

    def test_commitment_loss_nonzero(self):
        gq = GumbelQuantizer(total_dim=6, group_dim=3, commitment_weight=0.5)
        gq.train()
        x = torch.randn(2, 4, 6)
        gq(x)
        cl = gq.get_commitment_loss()
        assert cl.item() > 0

    def test_commitment_loss_zero_in_eval(self):
        gq = GumbelQuantizer(total_dim=6, group_dim=3, commitment_weight=0.5)
        gq.eval()
        x = torch.randn(2, 4, 6)
        gq(x)
        cl = gq.get_commitment_loss()
        assert cl.item() == 0.0

    def test_8d_octogram(self):
        gq = GumbelQuantizer(total_dim=8, group_dim=4)
        x = torch.randn(2, 4, 8)
        out = gq(x)
        assert out.shape == (2, 4, 8)

    def test_learnable_temperature(self):
        gq = GumbelQuantizer(total_dim=6, group_dim=3, temp=1.0)
        assert hasattr(gq, 'log_temp')
        assert gq.current_temp.item() > 0


class TestE8Lattice:
    """Тесты решётки E8 и E8Quantizer."""

    def test_e8_roots_count(self):
        e8 = generate_e8_roots()
        assert e8.shape == (240, 8)

    def test_e8_norms(self):
        """Все корни E8 имеют норму √2."""
        e8 = generate_e8_roots()
        norms = torch.norm(e8, dim=1)
        assert torch.allclose(norms, torch.tensor(2.0).sqrt(), atol=1e-5)

    def test_e8_unique(self):
        e8 = generate_e8_roots()
        unique = torch.unique(e8, dim=0)
        assert unique.shape[0] == 240

    def test_e8_centered(self):
        """E8 корни центрированы (сумма = 0)."""
        e8 = generate_e8_roots()
        center = e8.mean(dim=0)
        assert torch.norm(center).item() < 1e-5

    def test_e8_min_distance(self):
        """Минимальное расстояние между корнями E8 = √2."""
        e8 = generate_e8_roots()
        dists = torch.cdist(e8, e8)
        mask = ~torch.eye(240, dtype=torch.bool)
        min_dist = dists[mask].min()
        assert abs(min_dist.item() - 2.0**0.5) < 0.01

    def test_e8_quantizer_shape(self):
        eq = E8Quantizer(temp=0.3)
        x = torch.randn(2, 4, 8)
        out = eq(x)
        assert out.shape == (2, 4, 8)

    def test_e8_quantizer_gradient(self):
        eq = E8Quantizer(temp=0.3, adaptive_temp=True)
        x = torch.randn(2, 4, 8, requires_grad=True)
        out = eq(x)
        out.sum().backward()
        assert x.grad is not None

    def test_e8_hard_quantize(self):
        eq = E8Quantizer(temp=0.3)
        x = torch.randn(2, 4, 8)
        hard = eq.hard_quantize(x)
        assert hard.shape == x.shape
        # Результат должен быть корнем E8
        norms = torch.norm(hard.reshape(-1, 8), dim=1)
        assert torch.allclose(norms, torch.tensor(2.0).sqrt(), atol=1e-5)

    def test_compare_e8_vs_hypercube(self):
        results = compare_e8_vs_hypercube()
        assert 'E8 (240 roots)' in results
        assert 'Hexagrams {-1,+1}⁶' in results
        assert 'Octograms {-1,+1}⁸' in results
        # E8 имеет меньший минимальный зазор чем октограммы
        assert results['E8 (240 roots)']['min_dist'] < results['Octograms {-1,+1}⁸']['min_dist']
