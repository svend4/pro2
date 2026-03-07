"""Pytest-тесты: TernaryQuantizer и MatrixGrammar (v56)."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry.quantizers import TernaryQuantizer
from models.geometry.convergence import MatrixGrammar
from models.geometry.core import generate_ternary_hypercube, generate_ternary_trigrams


# ── Ternary Hypercube Generation ─────────────────────────────


class TestTernaryHypercube:
    def test_ternary_trigrams_shape(self):
        t = generate_ternary_trigrams()
        assert t.shape == (27, 3)  # 3³ = 27

    def test_ternary_hexagrams_shape(self):
        t = generate_ternary_hypercube(6)
        assert t.shape == (729, 6)  # 3⁶ = 729

    def test_values_in_range(self):
        t = generate_ternary_hypercube(4)
        unique = t.unique().sort().values
        assert torch.allclose(unique, torch.tensor([-1.0, 0.0, 1.0]))

    def test_includes_binary(self):
        """Тернарный гиперкуб содержит все бинарные вершины."""
        t = generate_ternary_hypercube(3)  # 27 vertices
        # Бинарные вершины = те, где нет 0
        binary_mask = (t != 0).all(dim=1)
        n_binary = binary_mask.sum().item()
        assert n_binary == 8  # 2³ = 8 бинарных в 3³ = 27


# ── TernaryQuantizer ─────────────────────────────────────────


class TestTernaryQuantizer:
    def test_full_mode_output_shape(self):
        q = TernaryQuantizer(total_dim=6, mode='full', temp=0.3)
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == (2, 8, 6)

    def test_factored_mode_output_shape(self):
        q = TernaryQuantizer(total_dim=6, mode='factored', temp=0.3)
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == (2, 8, 6)

    def test_sparse_mode_output_shape(self):
        q = TernaryQuantizer(total_dim=6, mode='sparse', max_zeros=2)
        x = torch.randn(2, 8, 6)
        out = q(x)
        assert out.shape == (2, 8, 6)

    def test_sparse_codebook_size(self):
        q = TernaryQuantizer(total_dim=6, mode='sparse', max_zeros=0)
        # max_zeros=0 → only binary vertices → 2⁶ = 64
        assert q.codebook.shape[0] == 64

    def test_sparse_codebook_with_zeros(self):
        q = TernaryQuantizer(total_dim=6, mode='sparse', max_zeros=1)
        # Vertices with 0 zeros: 2⁶ = 64
        # Vertices with 1 zero: C(6,1) * 2⁵ = 6 * 32 = 192
        # Total: 64 + 192 = 256
        assert q.codebook.shape[0] == 256

    def test_hard_quantize_values(self):
        q = TernaryQuantizer(total_dim=6, mode='factored')
        x = torch.tensor([[0.8, -0.9, 0.05, -0.02, 0.7, -0.6]])
        hard = q.hard_quantize(x)
        # Values should be in {-1, 0, +1}
        unique = hard.unique().sort().values
        for v in unique:
            assert v.item() in [-1.0, 0.0, 1.0]

    def test_uncertainty_budget_learnable(self):
        q = TernaryQuantizer(total_dim=6, mode='factored', uncertainty_budget=0.3)
        budget = q.uncertainty_budget
        assert 0.0 < budget.item() < 1.0
        # The underlying parameter should be learnable
        assert q.log_uncertainty.requires_grad

    def test_uncertainty_loss(self):
        q = TernaryQuantizer(total_dim=6, mode='factored')
        x = torch.randn(2, 8, 6)
        _ = q(x)
        loss = q.get_uncertainty_loss()
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_analyze_bian_yao(self):
        q = TernaryQuantizer(total_dim=6, mode='factored')
        x = torch.randn(4, 16, 6)
        info = q.analyze_bian_yao(x)
        assert 'n_bian_yao' in info
        assert 'bian_yao_mask' in info
        assert 'uncertainty_per_dim' in info
        assert info['uncertainty_per_dim'].shape == (6,)
        assert 0 <= info['n_bian_yao'] <= 6

    def test_gradient_flow(self):
        q = TernaryQuantizer(total_dim=6, mode='factored', adaptive_temp=True)
        x = torch.randn(2, 8, 6, requires_grad=True)
        out = q(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_codebook_size_property(self):
        q_full = TernaryQuantizer(total_dim=6, mode='full')
        assert q_full.codebook_size == 729

        q_fact = TernaryQuantizer(total_dim=6, mode='factored')
        assert q_fact.codebook_size == 27


# ── MatrixGrammar ────────────────────────────────────────────


class TestMatrixGrammar:
    def test_output_shape(self):
        mg = MatrixGrammar(d_model=64, n_rows=8, n_cols=8, n_heads=4)
        x = torch.randn(2, 16, 64)
        out = mg(x)
        assert out.shape == (2, 16, 64)

    def test_different_matrix_sizes(self):
        mg = MatrixGrammar(d_model=64, n_rows=4, n_cols=4, n_heads=2)
        x = torch.randn(2, 32, 64)
        out = mg(x)
        assert out.shape == (2, 32, 64)

    def test_slot_assignment(self):
        mg = MatrixGrammar(d_model=64, n_rows=8, n_cols=8)
        x = torch.randn(2, 16, 64)
        slots = mg._assign_to_slots(x)
        assert slots.shape == (2, 64, 64)  # n_slots = 8*8 = 64

    def test_axial_attention(self):
        mg = MatrixGrammar(d_model=64, n_rows=4, n_cols=4)
        matrix = torch.randn(2, 4, 4, 64)
        out = mg._axial_attention(matrix)
        assert out.shape == (2, 4, 4, 64)

    def test_gradient_flow(self):
        mg = MatrixGrammar(d_model=64, n_rows=4, n_cols=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = mg(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_scale_initialization(self):
        mg = MatrixGrammar(d_model=64)
        # Scale should start small for safe residual
        assert mg.scale.item() < 1.0

    def test_64_slots_equals_hexagrams(self):
        """8×8 = 64 слотов = число гексаграмм."""
        mg = MatrixGrammar(d_model=64, n_rows=8, n_cols=8)
        assert mg.n_slots == 64


# ── Integration ──────────────────────────────────────────────


class TestIntegrationV56:
    def test_model_with_ternary_quantizer(self):
        from config.config import YiJingConfig
        from models.model import YiJingGPT

        cfg = YiJingConfig.tiny(
            vocab_size=256,
            quantizer_type='ternary',
            ternary_mode='factored',
            ternary_uncertainty=0.3,
        )
        model = YiJingGPT(cfg)

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))
        logits, loss, _ = model(idx, targets=targets)

        assert logits.shape == (2, 32, 256)
        assert loss is not None
        assert loss.requires_grad

    def test_model_with_matrix_grammar(self):
        from config.config import YiJingConfig
        from models.model import YiJingGPT

        cfg = YiJingConfig.tiny(
            vocab_size=256,
            use_matrix_grammar=True,
            matrix_grammar_rows=4,
            matrix_grammar_cols=4,
        )
        model = YiJingGPT(cfg)

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))
        logits, loss, _ = model(idx, targets=targets)

        assert logits.shape == (2, 32, 256)
        assert loss is not None

    def test_model_with_all_v56_features(self):
        """Все три новые фичи одновременно."""
        from config.config import YiJingConfig
        from models.model import YiJingGPT

        cfg = YiJingConfig.tiny(
            vocab_size=256,
            quantizer_type='ternary',
            ternary_mode='factored',
            use_convergence_bridge=True,
            use_matrix_grammar=True,
            matrix_grammar_rows=4,
            matrix_grammar_cols=4,
        )
        model = YiJingGPT(cfg)

        idx = torch.randint(0, 256, (2, 16))
        targets = torch.randint(0, 256, (2, 16))
        logits, loss, _ = model(idx, targets=targets)

        assert logits.shape == (2, 16, 256)
        assert loss is not None
        assert loss.requires_grad
