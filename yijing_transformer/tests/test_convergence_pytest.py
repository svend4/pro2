"""Pytest-тесты конвергентного моста: GlyphComposer, TokenAbstractor, ConvergenceBridge."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry.convergence import (
    GlyphComposer,
    TokenAbstractor,
    ConvergenceLayer,
    ConvergenceBridge,
)


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 16


# ── GlyphComposer ──────────────────────────────────────────────


class TestGlyphComposer:
    def test_output_shape(self, d_model, batch_size, seq_len):
        composer = GlyphComposer(d_model=d_model, window_size=4, stride=2)
        vertices = torch.randn(batch_size, seq_len, 6).sign()  # Q6 вершины
        out = composer(vertices)
        assert out.dim() == 3
        assert out.shape[0] == batch_size
        assert out.shape[2] == d_model
        # n_sigils = (seq_len - window_size) // stride + 1 = (16-4)//2+1 = 7
        assert out.shape[1] == 7

    def test_short_sequence(self, d_model, batch_size):
        """Последовательность короче окна — должна работать."""
        composer = GlyphComposer(d_model=d_model, window_size=4, stride=2)
        vertices = torch.randn(batch_size, 2, 6).sign()
        out = composer(vertices)
        assert out.dim() == 3
        assert out.shape[0] == batch_size

    def test_edge_computation(self, d_model, batch_size, seq_len):
        composer = GlyphComposer(d_model=d_model)
        vertices = torch.randn(batch_size, seq_len, 6).sign()
        edges = composer._compute_edges(vertices)
        assert edges.shape == (batch_size, seq_len - 1, 12)

    def test_sigil_features(self, d_model, batch_size, seq_len):
        composer = GlyphComposer(d_model=d_model, window_size=4, stride=2)
        vertices = torch.randn(batch_size, seq_len, 6).sign()
        features = composer._compute_sigil_features(vertices)
        # spectral_dim = 6 + 1 + 4 = 11
        assert features.shape[2] == 11
        assert features.shape[0] == batch_size

    def test_gradient_flow(self, d_model, batch_size, seq_len):
        composer = GlyphComposer(d_model=d_model, window_size=4, stride=2)
        vertices = torch.randn(batch_size, seq_len, 6, requires_grad=True)
        out = composer(vertices)
        loss = out.sum()
        loss.backward()
        assert vertices.grad is not None


# ── TokenAbstractor ─────────────────────────────────────────────


class TestTokenAbstractor:
    def test_output_shape(self, d_model, batch_size, seq_len):
        abstractor = TokenAbstractor(d_model=d_model, n_clusters=64)
        x = torch.randn(batch_size, seq_len, d_model)
        abstract, assignments = abstractor(x)
        assert abstract.shape == (batch_size, seq_len, d_model)
        assert assignments.shape == (batch_size, seq_len, 64)

    def test_assignments_are_probabilities(self, d_model, batch_size, seq_len):
        abstractor = TokenAbstractor(d_model=d_model, n_clusters=64)
        x = torch.randn(batch_size, seq_len, d_model)
        _, assignments = abstractor(x)
        # Сумма по кластерам = 1
        sums = assignments.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # Все >= 0
        assert (assignments >= 0).all()

    def test_hexagram_anchors(self, d_model):
        abstractor = TokenAbstractor(d_model=d_model, n_clusters=64)
        assert abstractor.hexagram_anchors.shape == (64, 6)
        # Все значения в {-1, +1}
        assert ((abstractor.hexagram_anchors == 1) | (abstractor.hexagram_anchors == -1)).all()

    def test_correlation_range(self, d_model):
        abstractor = TokenAbstractor(d_model=d_model, n_clusters=64)
        corr = abstractor.cluster_hexagram_correlation()
        assert 0.0 <= corr.item() <= 1.0

    def test_temperature_positive(self, d_model):
        abstractor = TokenAbstractor(d_model=d_model, init_temperature=0.5)
        assert abstractor.temperature.item() > 0

    def test_gradient_flow(self, d_model, batch_size, seq_len):
        abstractor = TokenAbstractor(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        abstract, assignments = abstractor(x)
        loss = abstract.sum()
        loss.backward()
        assert x.grad is not None


# ── ConvergenceLayer ────────────────────────────────────────────


class TestConvergenceLayer:
    def test_output_shape(self, d_model, batch_size, seq_len):
        layer = ConvergenceLayer(d_model=d_model, n_heads=4)
        sigils = torch.randn(batch_size, 7, d_model)  # n_sigils
        abstracts = torch.randn(batch_size, seq_len, d_model)
        merged = layer(sigils, abstracts)
        assert merged.shape == (batch_size, seq_len, d_model)

    def test_same_length(self, d_model, batch_size, seq_len):
        """Когда sigils и abstracts одинаковой длины."""
        layer = ConvergenceLayer(d_model=d_model, n_heads=4)
        sigils = torch.randn(batch_size, seq_len, d_model)
        abstracts = torch.randn(batch_size, seq_len, d_model)
        merged = layer(sigils, abstracts)
        assert merged.shape == (batch_size, seq_len, d_model)


# ── ConvergenceBridge (full pipeline) ──────────────────────────


class TestConvergenceBridge:
    def test_full_forward(self, d_model, batch_size, seq_len):
        bridge = ConvergenceBridge(
            d_model=d_model, n_clusters=64,
            window_size=4, stride=2
        )
        token_emb = torch.randn(batch_size, seq_len, d_model)
        glyph_verts = torch.randn(batch_size, seq_len, 6).sign()

        enriched, info = bridge(token_emb, glyph_verts)

        assert enriched.shape == (batch_size, seq_len, d_model)
        assert 'assignments' in info
        assert 'correlation' in info
        assert 'n_sigils' in info
        assert info['assignments'].shape == (batch_size, seq_len, 64)

    def test_convergence_loss(self, d_model, batch_size, seq_len):
        bridge = ConvergenceBridge(d_model=d_model)
        token_emb = torch.randn(batch_size, seq_len, d_model)
        glyph_verts = torch.randn(batch_size, seq_len, 6).sign()

        _, info = bridge(token_emb, glyph_verts)
        loss = bridge.get_convergence_loss(info['assignments'])

        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_residual_connection(self, d_model, batch_size, seq_len):
        """Enriched ≈ token_emb при bridge_scale ≈ 0."""
        bridge = ConvergenceBridge(d_model=d_model)
        with torch.no_grad():
            bridge.bridge_scale.fill_(0.0)

        token_emb = torch.randn(batch_size, seq_len, d_model)
        glyph_verts = torch.randn(batch_size, seq_len, 6).sign()
        enriched, _ = bridge(token_emb, glyph_verts)

        assert torch.allclose(enriched, token_emb, atol=1e-6)

    def test_parameter_count(self, d_model):
        bridge = ConvergenceBridge(d_model=d_model)
        n_params = sum(p.numel() for p in bridge.parameters())
        # Должен быть разумным (не взрывать бюджет модели)
        assert n_params < 500_000, f"Too many params: {n_params}"
        assert n_params > 0

    def test_gradient_flow_full(self, d_model, batch_size, seq_len):
        bridge = ConvergenceBridge(d_model=d_model)
        token_emb = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        glyph_verts = torch.randn(batch_size, seq_len, 6, requires_grad=True)

        enriched, info = bridge(token_emb, glyph_verts)
        loss = enriched.sum() + bridge.get_convergence_loss(info['assignments'])
        loss.backward()

        assert token_emb.grad is not None
        assert glyph_verts.grad is not None


# ── Integration with YiJingGPT ──────────────────────────────────


class TestIntegration:
    def test_model_with_convergence_bridge(self):
        from config.config import YiJingConfig
        from models.model import YiJingGPT

        cfg = YiJingConfig.tiny(
            vocab_size=256,
            use_convergence_bridge=True,
            convergence_n_clusters=64,
        )
        model = YiJingGPT(cfg)

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        logits, loss, _ = model(idx, targets=targets)

        assert logits.shape == (2, 32, 256)
        assert loss is not None
        assert loss.requires_grad

    def test_model_without_bridge(self):
        """Без моста — стандартное поведение."""
        from config.config import YiJingConfig
        from models.model import YiJingGPT

        cfg = YiJingConfig.tiny(vocab_size=256, use_convergence_bridge=False)
        model = YiJingGPT(cfg)

        idx = torch.randint(0, 256, (2, 32))
        targets = torch.randint(0, 256, (2, 32))

        logits, loss, _ = model(idx, targets=targets)
        assert logits.shape == (2, 32, 256)
        assert loss is not None
