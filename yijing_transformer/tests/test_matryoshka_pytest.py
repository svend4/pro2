"""Pytest-тесты для MatryoshkaQuantizer (v65): иерархическое кодирование бит→трит→гекс."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry.quantizers import MatryoshkaQuantizer
from models.geometry.core import (
    generate_hypercube,
    hex_digit_semantics,
    generate_spacetime_pairs,
)


# ===================== Core helpers =====================

class TestHexDigitSemantics:
    """Тесты для hex_digit_semantics()."""

    def test_returns_16_states(self):
        sem = hex_digit_semantics()
        assert len(sem) == 16

    def test_all_hex_digits_present(self):
        sem = hex_digit_semantics()
        hex_vals = {s['hex'] for s in sem}
        assert hex_vals == {format(i, 'x') for i in range(16)}

    def test_stable_states_exist(self):
        """Должны быть стабильные состояния (now == ref)."""
        sem = hex_digit_semantics()
        stable = [s for s in sem if s['stable']]
        assert len(stable) == 4  # 4 пары одинаковых: ++/++, --/--, +-/+-, -+/-+

    def test_spatial_labels_valid(self):
        sem = hex_digit_semantics()
        valid_labels = {'yin', 'yang', 'spring', 'autumn'}
        for s in sem:
            assert s['spatial_now'] in valid_labels
            assert s['spatial_ref'] in valid_labels


class TestSpacetimePairs:
    """Тесты для generate_spacetime_pairs()."""

    def test_default_q12(self):
        cb = generate_spacetime_pairs(3)
        assert cb.shape == (4096, 12)

    def test_q8_for_2_pairs(self):
        cb = generate_spacetime_pairs(2)
        assert cb.shape == (256, 8)

    def test_all_binary(self):
        cb = generate_spacetime_pairs(3)
        assert ((cb == -1) | (cb == 1)).all()


# ===================== MatryoshkaQuantizer =====================

class TestMatryoshkaInit:
    """Тесты инициализации."""

    def test_default_init(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        assert mq.total_dim == 6
        assert mq.n_pairs == 3
        assert mq.d_model == 64

    def test_codebook_shapes(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        assert mq.binary_codebook.shape == (64, 6)
        assert mq.pair_codebook.shape == (4, 2)
        assert mq.hex_codebook.shape == (16, 4)

    def test_trit_table_values(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        # (-1,-1)→-1, (-1,+1)→0, (+1,-1)→0, (+1,+1)→+1
        assert mq.trit_table.tolist() == [-1.0, 0.0, 0.0, 1.0]

    def test_direction_table_values(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        # (-1,-1)→0, (-1,+1)→+1(spring), (+1,-1)→-1(autumn), (+1,+1)→0
        assert mq.direction_table.tolist() == [0.0, 1.0, -1.0, 0.0]

    def test_even_dim_required(self):
        with pytest.raises(AssertionError):
            MatryoshkaQuantizer(total_dim=5, d_model=64)

    def test_adaptive_temp(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64, adaptive_temp=True)
        assert hasattr(mq, 'log_temp')
        assert isinstance(mq.log_temp, torch.nn.Parameter)

    def test_level_gates_init(self):
        """Nautilus principle: higher levels get larger initial gate."""
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        gates = torch.sigmoid(mq.level_gates)
        assert gates[0] < gates[1] < gates[2]


class TestMatryoshkaForward:
    """Тесты forward pass."""

    @pytest.fixture
    def mq(self):
        return MatryoshkaQuantizer(total_dim=6, d_model=64)

    def test_output_shape_no_ref(self, mq):
        x = torch.randn(2, 10, 6)
        out, info = mq(x)
        assert out.shape == (2, 10, 64)

    def test_output_shape_with_ref(self, mq):
        x = torch.randn(2, 10, 6)
        x_ref = torch.randn(2, 10, 6)
        out, info = mq(x, x_ref)
        assert out.shape == (2, 10, 64)

    def test_info_has_trits(self, mq):
        x = torch.randn(2, 10, 6)
        _, info = mq(x)
        assert 'trits' in info
        assert info['trits'].shape == (2, 10, 3)

    def test_info_has_directions(self, mq):
        x = torch.randn(2, 10, 6)
        _, info = mq(x)
        assert 'directions' in info
        assert info['directions'].shape == (2, 10, 3)

    def test_info_has_hex_with_ref(self, mq):
        x = torch.randn(2, 10, 6)
        x_ref = torch.randn(2, 10, 6)
        _, info = mq(x, x_ref)
        assert 'hex_features' in info
        assert info['hex_features'].shape == (2, 10, 12)  # 3 pairs × 4

    def test_no_hex_without_ref(self, mq):
        x = torch.randn(2, 10, 6)
        _, info = mq(x)
        assert 'hex_features' not in info

    def test_gradient_flows(self, mq):
        x = torch.randn(2, 10, 6, requires_grad=True)
        out, _ = mq(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flows_with_ref(self, mq):
        x = torch.randn(2, 10, 6, requires_grad=True)
        x_ref = torch.randn(2, 10, 6, requires_grad=True)
        out, _ = mq(x, x_ref)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x_ref.grad is not None

    def test_level_gates_in_info(self, mq):
        x = torch.randn(2, 10, 6)
        _, info = mq(x)
        assert 'level_gates' in info
        assert info['level_gates'].shape == (3,)

    def test_deterministic_eval(self, mq):
        mq.eval()
        x = torch.randn(2, 10, 6)
        out1, _ = mq(x)
        out2, _ = mq(x)
        assert torch.allclose(out1, out2)


class TestMatryoshkaHardQuantize:
    """Тесты жёсткой квантизации."""

    @pytest.fixture
    def mq(self):
        return MatryoshkaQuantizer(total_dim=6, d_model=64)

    def test_bits_are_signs(self, mq):
        x = torch.randn(4, 6)
        result = mq.hard_quantize(x)
        assert ((result['bits'] == -1) | (result['bits'] == 1)).all()

    def test_trits_values(self, mq):
        """Триты должны быть из {-1, 0, +1}."""
        x = torch.randn(4, 6)
        result = mq.hard_quantize(x)
        trits = result['trits']
        valid = (trits == -1) | (trits == 0) | (trits == 1)
        assert valid.all()

    def test_directions_values(self, mq):
        """Направления из {-1, 0, +1}."""
        x = torch.randn(4, 6)
        result = mq.hard_quantize(x)
        dirs = result['directions']
        valid = (dirs == -1) | (dirs == 0) | (dirs == 1)
        assert valid.all()

    def test_trit_direction_consistency(self, mq):
        """Направление ≠ 0 только для переходных тритов (=0)."""
        x = torch.randn(100, 6)
        result = mq.hard_quantize(x)
        trits = result['trits']
        dirs = result['directions']
        # Where trit is ±1 (stable), direction must be 0
        stable_mask = trits.abs() > 0.5
        assert (dirs[stable_mask] == 0).all()
        # Where trit is 0 (transition), direction must be ±1
        transition_mask = trits.abs() < 0.5
        if transition_mask.any():
            assert (dirs[transition_mask].abs() > 0.5).all()

    def test_known_input_yang(self, mq):
        """Все +1 → все триты = +1 (ян), все направления = 0."""
        x = torch.ones(1, 6) * 2  # clearly positive
        result = mq.hard_quantize(x)
        assert (result['bits'] == 1).all()
        assert (result['trits'] == 1).all()
        assert (result['directions'] == 0).all()

    def test_known_input_yin(self, mq):
        """Все -1 → все триты = -1 (инь), все направления = 0."""
        x = torch.ones(1, 6) * -2  # clearly negative
        result = mq.hard_quantize(x)
        assert (result['bits'] == -1).all()
        assert (result['trits'] == -1).all()
        assert (result['directions'] == 0).all()

    def test_known_input_spring(self, mq):
        """(-1,+1,-1,+1,-1,+1) → все триты = 0, все направления = +1 (весна)."""
        x = torch.tensor([[-2, 2, -2, 2, -2, 2]], dtype=torch.float32)
        result = mq.hard_quantize(x)
        assert (result['trits'] == 0).all()
        assert (result['directions'] == 1).all()  # spring

    def test_known_input_autumn(self, mq):
        """(+1,-1,+1,-1,+1,-1) → все триты = 0, все направления = -1 (осень)."""
        x = torch.tensor([[2, -2, 2, -2, 2, -2]], dtype=torch.float32)
        result = mq.hard_quantize(x)
        assert (result['trits'] == 0).all()
        assert (result['directions'] == -1).all()  # autumn

    def test_hex_digits_range(self, mq):
        """Гекс-цифры в диапазоне [0, 15]."""
        x = torch.randn(8, 6)
        x_ref = torch.randn(8, 6)
        result = mq.hard_quantize(x, x_ref)
        assert 'hex_digits' in result
        assert (result['hex_digits'] >= 0).all()
        assert (result['hex_digits'] <= 15).all()

    def test_hex_digits_shape(self, mq):
        x = torch.randn(4, 6)
        x_ref = torch.randn(4, 6)
        result = mq.hard_quantize(x, x_ref)
        assert result['hex_digits'].shape == (4, 3)  # 3 hex digits

    def test_hex_stable_identity(self, mq):
        """x == x_ref → стабильные hex digits."""
        x = torch.randn(4, 6)
        result = mq.hard_quantize(x, x.clone())
        # When ref == current: hex bits = (d1,d2,d1,d2)
        # These should be the "stable" states (0,5,10,15 in hex)
        stable_hex = {0, 5, 10, 15}
        for row in result['hex_digits']:
            for digit in row:
                assert digit.item() in stable_hex

    def test_hex_vectors_shape(self, mq):
        x = torch.randn(4, 6)
        x_ref = torch.randn(4, 6)
        result = mq.hard_quantize(x, x_ref)
        assert result['hex_vectors'].shape == (4, 12)  # Q12


class TestMatryoshkaAnalysis:
    """Тесты matryoshka_analysis()."""

    def test_two_levels_without_ref(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(4, 6)
        analysis = mq.matryoshka_analysis(x)
        assert analysis['n_levels'] == 2
        assert 'level0' in analysis
        assert 'level1' in analysis
        assert 'level2' not in analysis

    def test_three_levels_with_ref(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(4, 6)
        x_ref = torch.randn(4, 6)
        analysis = mq.matryoshka_analysis(x, x_ref)
        assert analysis['n_levels'] == 3
        assert 'level2' in analysis
        assert analysis['level2']['total_states'] == 4096
        assert analysis['level2']['new_dimension'] == 12

    def test_distribution_sums_to_one(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(100, 6)
        analysis = mq.matryoshka_analysis(x)
        dist = analysis['level1']['distribution']
        total = dist['yang'] + dist['yin'] + dist['transition']
        assert abs(total - 1.0) < 0.01


class TestMatryoshkaStats:
    """Тесты get_stats() мониторинга."""

    def test_stats_populated_after_forward(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(4, 10, 6)
        mq(x)
        stats = mq.get_stats()
        assert 'trit_yang' in stats
        assert 'trit_yin' in stats
        assert 'trit_transition' in stats
        assert 'gate_L0' in stats
        assert 'gate_L1' in stats
        assert 'gate_L2' in stats

    def test_stats_with_ref(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(4, 10, 6)
        x_ref = torch.randn(4, 10, 6)
        mq(x, x_ref)
        stats = mq.get_stats()
        assert stats['has_spacetime'] is True

    def test_stats_without_ref(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(4, 10, 6)
        mq(x)
        stats = mq.get_stats()
        assert stats['has_spacetime'] is False


class TestMatryoshkaQ8:
    """Тесты для total_dim=8 (Q8, 4 пары)."""

    def test_q8_init(self):
        mq = MatryoshkaQuantizer(total_dim=8, d_model=64)
        assert mq.n_pairs == 4
        assert mq.binary_codebook.shape == (256, 8)

    def test_q8_forward(self):
        mq = MatryoshkaQuantizer(total_dim=8, d_model=64)
        x = torch.randn(2, 10, 8)
        out, info = mq(x)
        assert out.shape == (2, 10, 64)
        assert info['trits'].shape == (2, 10, 4)

    def test_q8_hex_digits(self):
        mq = MatryoshkaQuantizer(total_dim=8, d_model=64)
        x = torch.randn(4, 8)
        x_ref = torch.randn(4, 8)
        result = mq.hard_quantize(x, x_ref)
        assert result['hex_digits'].shape == (4, 4)  # 4 hex digits for Q8
        assert result['hex_vectors'].shape == (4, 16)  # Q16


class TestMatryoshkaGradientIntegrity:
    """Тесты целостности градиентов через все уровни."""

    def test_all_projections_receive_grad(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(2, 10, 6, requires_grad=True)
        x_ref = torch.randn(2, 10, 6, requires_grad=True)
        out, _ = mq(x, x_ref)
        loss = out.sum()
        loss.backward()
        # All level projections should have gradients
        assert mq.proj_level0.weight.grad is not None
        assert mq.proj_level1.weight.grad is not None
        assert mq.proj_level2.weight.grad is not None

    def test_level_gates_receive_grad(self):
        mq = MatryoshkaQuantizer(total_dim=6, d_model=64)
        x = torch.randn(2, 10, 6)
        x_ref = torch.randn(2, 10, 6)
        out, _ = mq(x, x_ref)
        loss = out.sum()
        loss.backward()
        assert mq.level_gates.grad is not None
