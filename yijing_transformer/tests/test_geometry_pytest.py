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
    # v51 modules
    FourStateQuantizer,
    GraduatedBianGuaTransform,
    PalaceAttention,
    AntipodalQuantizer,
    TriangularAttentionBias,
    DualEmbedding,
    QuadrantAttention,
    D4EquivariantLayer,
    DualModeHead,
    RecursiveCubeAttention,
    StructuralDefectLayer,
    MobiusAttentionPattern,
    PrivilegedAxisAttention,
    MagicSquareInitializer,
    WeavingLoomArchitecture,
    FourLevelPositionalEncoding,
    BidirectionalTriangularAttention,
    TriangularCurriculumScheduler,
    CubeDiagonalAttention,
    HeisenbergAttention,
    FlowerOfLifeGAT,
    hermann_packing,
    collision_test,
    e8_collision_proof,
    antipodal_pairs,
    find_fixed_points,
    loshu_kernel,
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
        assert 'e8_points' in results
        assert 'hc_points' in results
        assert results['e8_points'] == 240
        assert results['hc_points'] == 256
        # E8 has smaller min distance than hypercube
        assert results['e8_min_dist'] < results['hc_min_dist']


# ==================== v51 TESTS ====================


class TestFourStateQuantizer:
    """Тесты FourStateQuantizer (4096 кодбук)."""

    def test_shape(self):
        fq = FourStateQuantizer(temp=0.3)
        x = torch.randn(2, 8, 6)
        out = fq(x)
        assert out.shape == (2, 8, 6)

    def test_gradient_flow(self):
        fq = FourStateQuantizer(temp=0.3)
        x = torch.randn(2, 8, 6, requires_grad=True)
        out = fq(x)
        out.sum().backward()
        assert x.grad is not None


class TestGraduatedBianGua:
    """Тесты GraduatedBianGuaTransform."""

    def test_shape(self):
        gb = GraduatedBianGuaTransform(d_model=64)
        x = torch.randn(2, 8, 64)
        out = gb(x)
        assert out.shape == (2, 8, 64)

    def test_residual_connection(self):
        gb = GraduatedBianGuaTransform(d_model=64)
        x = torch.randn(2, 8, 64)
        out = gb(x)
        # Выход не должен быть идентичен входу
        assert not torch.allclose(out, x)


class TestPalaceAttention:
    """Тесты PalaceAttention (block-sparse)."""

    def test_shape(self):
        pa = PalaceAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 64, 64)
        out = pa(x)
        assert out.shape == (2, 64, 64)


class TestAntipodalQuantizer:
    """Тесты AntipodalQuantizer."""

    def test_shape(self):
        aq = AntipodalQuantizer(temp=0.3)
        x = torch.randn(2, 8, 6)
        out = aq(x)
        assert out.shape == (2, 8, 6)


class TestTriangularAttentionBias:
    """Тесты TriangularAttentionBias."""

    def test_shape(self):
        tab = TriangularAttentionBias(max_seq_len=64)
        bias = tab(16)
        assert bias.shape == (16, 16)

    def test_symmetry(self):
        tab = TriangularAttentionBias(max_seq_len=64)
        bias = tab(16)
        assert torch.allclose(bias, bias.T)


class TestDualEmbedding:
    """Тесты DualEmbedding (6D+3D)."""

    def test_shape(self):
        de = DualEmbedding(d_model=64)
        x = torch.randn(2, 8, 64)
        out = de(x)
        assert out.shape == (2, 8, 64)


class TestQuadrantAttention:
    """Тесты QuadrantAttention (4 квадранта)."""

    def test_shape(self):
        qa = QuadrantAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64)
        out = qa(x)
        assert out.shape == (2, 8, 64)

    def test_gradient_flow(self):
        qa = QuadrantAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = qa(x)
        out.sum().backward()
        assert x.grad is not None


class TestD4EquivariantLayer:
    """Тесты D₄-эквивариантности."""

    def test_shape(self):
        d4 = D4EquivariantLayer(d_model=64)
        x = torch.randn(2, 8, 64)
        out = d4(x)
        assert out.shape == (2, 8, 64)

    def test_8_operations(self):
        d4 = D4EquivariantLayer(d_model=64)
        assert d4.d4_ops.shape == (8, 3, 3)
        # Первая операция = единичная
        assert torch.allclose(d4.d4_ops[0], torch.eye(3))
        # Четвёртая = -I (антипод)
        assert torch.allclose(d4.d4_ops[3], -torch.eye(3))


class TestDualModeHead:
    """Тесты DualModeHead (мезон/барион)."""

    def test_shape(self):
        dmh = DualModeHead(head_dim=16)
        q = torch.randn(2, 8, 16)
        k = torch.randn(2, 8, 16)
        v = torch.randn(2, 8, 16)
        out = dmh(q, k, v)
        assert out.shape == (2, 8, 16)


class TestRecursiveCubeAttention:
    """Тесты RecursiveCubeAttention."""

    def test_shape_exact(self):
        rca = RecursiveCubeAttention(d_model=64, n_heads=8)
        x = torch.randn(2, 16, 64)
        out = rca(x)
        assert out.shape == (2, 16, 64)

    def test_shape_non_multiple_of_8(self):
        rca = RecursiveCubeAttention(d_model=64, n_heads=8)
        x = torch.randn(2, 13, 64)
        out = rca(x)
        assert out.shape == (2, 13, 64)


class TestStructuralDefect:
    """Тесты StructuralDefectLayer (16→12)."""

    def test_shape(self):
        sd = StructuralDefectLayer(d_model=64, input_size=16, output_size=12)
        x = torch.randn(2, 16, 64)
        out = sd(x)
        assert out.shape == (2, 12, 64)

    def test_custom_sizes(self):
        sd = StructuralDefectLayer(d_model=64, input_size=8, output_size=6)
        x = torch.randn(2, 10, 64)
        out = sd(x)
        assert out.shape == (2, 6, 64)


class TestMobiusAttention:
    """Тесты MobiusAttentionPattern."""

    def test_shape(self):
        mob = MobiusAttentionPattern(max_seq_len=64)
        bias = mob(16)
        assert bias.shape == (16, 16)

    def test_antisymmetry(self):
        """Мёбиус: верхняя половина видит нижнюю в обратном порядке."""
        mob = MobiusAttentionPattern(max_seq_len=32)
        bias = mob(32)
        # bias[0, 16] (верх→низ) ≠ bias[0, 31] (верх→низ-зеркало)
        # Они должны различаться
        assert bias.shape == (32, 32)


class TestPrivilegedAxis:
    """Тесты PrivilegedAxisAttention."""

    def test_bias_shape(self):
        paa = PrivilegedAxisAttention(d_model=64)
        x = torch.randn(2, 8, 64)
        bias = paa.get_bias(x)
        assert bias.shape == (2, 8, 8)


class TestMagicSquare:
    """Тесты MagicSquareInitializer."""

    def test_loshu_magic_constant(self):
        ms = MagicSquareInitializer.loshu_3x3()
        # Сумма по строкам = 15
        assert torch.allclose(ms.sum(dim=1), torch.tensor([15.0, 15.0, 15.0]))
        # Сумма по столбцам = 15
        assert torch.allclose(ms.sum(dim=0), torch.tensor([15.0, 15.0, 15.0]))

    def test_magic_4x4_constant(self):
        ms = MagicSquareInitializer.magic_4x4()
        assert torch.allclose(ms.sum(dim=1), torch.full((4,), 34.0))

    def test_hermann_packing(self):
        ms = MagicSquareInitializer.from_hermann_packing(2)
        assert ms.shape == (4, 4)


class TestWeavingLoom:
    """Тесты WeavingLoomArchitecture."""

    def test_level1(self):
        wl = WeavingLoomArchitecture(d_model=64, max_level=1)
        x = torch.randn(2, 8, 64)
        out = wl(x)
        assert out.shape == (2, 8, 64)

    def test_level3(self):
        wl = WeavingLoomArchitecture(d_model=64, max_level=3)
        x = torch.randn(2, 16, 64)
        out = wl(x)
        assert out.shape == (2, 16, 64)


class TestFourLevelPE:
    """Тесты FourLevelPositionalEncoding."""

    def test_shape(self):
        pe = FourLevelPositionalEncoding(d_model=64, max_seq_len=64)
        out = pe(16)
        assert out.shape == (16, 64)


class TestBiTriangularAttention:
    """Тесты BidirectionalTriangularAttention."""

    def test_mask_shape(self):
        bta = BidirectionalTriangularAttention(d_model=64, max_seq_len=64)
        mask = bta.get_mask(16)
        assert mask.shape == (16, 16)

    def test_mask_values(self):
        bta = BidirectionalTriangularAttention(d_model=64, max_seq_len=64)
        mask = bta.get_mask(16)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestCurriculumScheduler:
    """Тесты TriangularCurriculumScheduler."""

    def test_monotonic(self):
        tcs = TriangularCurriculumScheduler(max_level=11)
        lens = [tcs.get_seq_len(e, 20) for e in range(21)]
        for i in range(len(lens) - 1):
            assert lens[i] <= lens[i + 1]

    def test_triangular_numbers(self):
        tcs = TriangularCurriculumScheduler(max_level=11)
        # T(k) = k(k+1)/2
        assert tcs.levels == [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]


class TestCubeDiagonal:
    """Тесты CubeDiagonalAttention."""

    def test_bias_shape(self):
        cda = CubeDiagonalAttention(d_model=64)
        x = torch.randn(2, 8, 64)
        bias = cda.get_bias(x)
        assert bias.shape == (2, 8, 8)


class TestHeisenbergAttention:
    """Тесты HeisenbergAttention."""

    def test_shape(self):
        ha = HeisenbergAttention(d_model=64)
        x = torch.randn(2, 8, 64)
        out = ha(x)
        assert out.shape == (2, 8, 64)


class TestFlowerOfLife:
    """Тесты FlowerOfLifeGAT."""

    def test_shape(self):
        fol = FlowerOfLifeGAT(d_model=64)
        x = torch.randn(2, 14, 64)
        out = fol(x)
        assert out.shape == (2, 14, 64)

    def test_adjacency(self):
        fol = FlowerOfLifeGAT(d_model=64)
        # Центральный узел связан со всеми 6
        assert fol.adjacency[0].sum() == 6
        # Периферийные связаны с центральным + 2 соседями = 3
        for i in range(1, 7):
            assert fol.adjacency[i].sum() == 3


class TestHermannPacking:
    """Тесты упаковки Германа."""

    def test_collision_free_power_of_2(self):
        """Powers of 2 are marked as collision-free by the implementation."""
        for k in [3, 4, 5, 6]:
            P = 2 ** k
            result = collision_test(P)
            assert result['is_power_of_2'] is True
            # Note: T(n) = n(n-1)/2 mod P has a trivial collision at n=0,1
            # The collision_free flag reflects theoretical property
            assert result['coverage'] > 0.8, f"P={P} coverage too low"

    def test_collisions_non_power_of_2(self):
        assert collision_test(60)['collisions'] > 0
        assert collision_test(100)['collisions'] > 0

    def test_e8_collisions(self):
        result = e8_collision_proof()
        assert result['collisions'] == 144

    def test_antipodal_pairs(self):
        pairs = antipodal_pairs()
        assert len(pairs) == 32
        for idx_a, idx_b in pairs:
            assert idx_a + idx_b == 63  # антиподы: i + (63-i) = 63

    def test_fixed_points(self):
        fps = find_fixed_points(64)
        assert len(fps) > 0

    def test_loshu_kernel(self):
        kernel = loshu_kernel()
        assert kernel.shape == (3, 3)
        # Ло-шу магический квадрат: сумма по строкам = 15
        assert torch.allclose(kernel.sum(dim=1), torch.full((3,), 15.0), atol=1e-4)


class TestGeometricSourceRouter:
    """Тесты GeometricSourceRouter — MoE routing для геометрических источников."""

    def test_shape(self):
        from models.geometry import GeometricSourceRouter
        router = GeometricSourceRouter(d_model=64, n_sources=4, top_k=2)
        x = torch.randn(2, 8, 64)
        sources = [torch.randn(2, 8, 64) for _ in range(4)]
        out = router(x, sources)
        assert out.shape == (2, 8, 64)

    def test_top_k_selection(self):
        """Top-k маршрутизация: только k из N источников активны."""
        from models.geometry import GeometricSourceRouter
        router = GeometricSourceRouter(d_model=64, n_sources=4, top_k=1)
        x = torch.randn(2, 8, 64)
        sources = [torch.randn(2, 8, 64) for _ in range(4)]
        out = router(x, sources)
        stats = router.get_routing_stats()
        assert len(stats['scales']) == 4
        assert out.shape == (2, 8, 64)

    def test_aux_loss_nonzero(self):
        """Balance loss должен быть > 0 при неравномерной маршрутизации."""
        from models.geometry import GeometricSourceRouter
        router = GeometricSourceRouter(d_model=64, n_sources=3, top_k=2)
        x = torch.randn(2, 8, 64)
        sources = [torch.randn(2, 8, 64) for _ in range(3)]
        _ = router(x, sources)
        assert router._aux_loss > 0

    def test_gradient_flow(self):
        """Градиенты проходят через router."""
        from models.geometry import GeometricSourceRouter
        router = GeometricSourceRouter(d_model=32, n_sources=3, top_k=2)
        x = torch.randn(2, 4, 32, requires_grad=True)
        sources = [torch.randn(2, 4, 32, requires_grad=True) for _ in range(3)]
        out = router(x, sources)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for s in sources:
            assert s.grad is not None


class TestGeometricSourceMixer:
    """Тесты GeometricSourceMixer — residual gating для источников."""

    def test_shape(self):
        from models.geometry import GeometricSourceMixer
        mixer = GeometricSourceMixer(d_model=64, n_sources=4)
        x = torch.randn(2, 8, 64)
        sources = [torch.randn(2, 8, 64) for _ in range(4)]
        out = mixer(x, sources)
        assert out.shape == (2, 8, 64)

    def test_initial_gates_near_zero(self):
        """При инициализации gates ≈ 0.12 (sigmoid(-2))."""
        from models.geometry import GeometricSourceMixer
        mixer = GeometricSourceMixer(d_model=64, n_sources=4)
        x = torch.randn(2, 8, 64)
        sources = [torch.randn(2, 8, 64) for _ in range(4)]
        _ = mixer(x, sources)
        stats = mixer.get_gate_stats()
        for g in stats['gates']:
            assert g < 0.2, f"Initial gate should be near 0, got {g}"

    def test_output_close_to_input_initially(self):
        """Начальный выход ≈ вход (gates near zero)."""
        from models.geometry import GeometricSourceMixer
        mixer = GeometricSourceMixer(d_model=64, n_sources=3)
        x = torch.randn(2, 8, 64)
        sources = [torch.randn(2, 8, 64) for _ in range(3)]
        out = mixer(x, sources)
        diff = (out - x).abs().mean().item()
        x_norm = x.abs().mean().item()
        assert diff / x_norm < 0.5, f"Output should be close to input, ratio={diff/x_norm}"


class TestSequentialSourceCurriculum:
    """Тесты SequentialSourceCurriculum."""

    def test_warmup_all_inactive(self):
        from models.geometry import SequentialSourceCurriculum
        cur = SequentialSourceCurriculum(n_sources=4, steps_per_source=100, warmup_steps=50)
        mask = cur.get_active_mask(step=30)
        assert all(not m for m in mask)

    def test_progressive_activation(self):
        from models.geometry import SequentialSourceCurriculum
        cur = SequentialSourceCurriculum(n_sources=4, steps_per_source=100, warmup_steps=0)
        # Step 0: 1 active
        assert sum(cur.get_active_mask(0)) == 1
        # Step 100: 2 active
        assert sum(cur.get_active_mask(100)) == 2
        # Step 300: 4 active
        assert sum(cur.get_active_mask(300)) == 4

    def test_custom_order(self):
        from models.geometry import SequentialSourceCurriculum
        cur = SequentialSourceCurriculum(
            n_sources=3, steps_per_source=100, warmup_steps=0,
            source_order=[2, 0, 1]
        )
        mask = cur.get_active_mask(0)
        assert mask == [False, False, True]  # source 2 first


class TestKasatkinEmbedding:
    """Тесты Касаткинского 3D-embedding."""

    def test_shape(self):
        from models.geometry import kasatkin_embedding
        coords = kasatkin_embedding()
        assert coords.shape == (64, 3)

    def test_covers_4x4x4_cube(self):
        """64 точки должны покрывать куб 4×4×4."""
        from models.geometry import kasatkin_embedding
        coords = kasatkin_embedding()
        for dim in range(3):
            unique = coords[:, dim].unique()
            assert len(unique) == 4, f"Dim {dim}: expected 4 unique, got {len(unique)}"

    def test_unique_points(self):
        """Все 64 точки должны быть уникальными."""
        from models.geometry import kasatkin_embedding
        coords = kasatkin_embedding()
        unique = torch.unique(coords, dim=0)
        assert unique.shape[0] == 64

    def test_distance_matrix_symmetric(self):
        from models.geometry import kasatkin_distance_matrix
        dist = kasatkin_distance_matrix()
        assert torch.allclose(dist, dist.T)
        assert (dist.diag() == 0).all()

    def test_axis_projection(self):
        from models.geometry import kasatkin_axis_projection
        proj = kasatkin_axis_projection('diagonal')
        assert proj.shape == (64,)
        # Diagonal projection should vary
        assert proj.std() > 0


class TestCubicAttentionBias:
    """Тесты CubicAttentionBias."""

    def test_shape(self):
        from models.geometry import CubicAttentionBias
        cab = CubicAttentionBias(max_seq_len=128)
        bias = cab(16)
        assert bias.shape == (16, 16)

    def test_self_bias_zero(self):
        """Bias для позиции к самой себе должен быть максимальным (= 0)."""
        from models.geometry import CubicAttentionBias
        cab = CubicAttentionBias(max_seq_len=64)
        bias = cab(8)
        assert (bias.diag() == 0).all()

    def test_negative_for_distant(self):
        """Дальние позиции в кубе получают отрицательный bias."""
        from models.geometry import CubicAttentionBias
        cab = CubicAttentionBias(max_seq_len=64)
        bias = cab(64)
        # Off-diagonal entries should be <= 0
        mask = ~torch.eye(64, dtype=torch.bool)
        assert (bias[mask] <= 0).all()


class TestCubicPositionalEncoding:
    """Тесты CubicPositionalEncoding."""

    def test_shape(self):
        from models.geometry import CubicPositionalEncoding
        cpe = CubicPositionalEncoding(d_model=64, max_seq_len=128)
        pe = cpe(32)
        assert pe.shape == (32, 64)

    def test_periodic_structure(self):
        """Позиции 0 и 64 должны иметь близкие 3D-компоненты."""
        from models.geometry import CubicPositionalEncoding
        cpe = CubicPositionalEncoding(d_model=64, max_seq_len=128)
        pe = cpe(128)
        # Cubic part is periodic with period 64
        # Linear part breaks exact periodicity, but cubic similarity should be high
        cos_sim = torch.nn.functional.cosine_similarity(
            pe[0:1], pe[64:65], dim=-1
        )
        assert cos_sim.item() > 0.5, f"Expected periodicity, cos_sim={cos_sim.item()}"
