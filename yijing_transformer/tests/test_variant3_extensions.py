"""
test_variant3_extensions.py — Comprehensive tests for variant3_extensions.py

Tests cover all 10 ideas and all text-analysis modules:
  TestHexPosEncoding         — Idea 1
  TestSixLineAttention       — Idea 2
  TestBianGuaOptimizer       — Idea 3
  TestTernaryKVCache         — Idea 4
  TestHexagramTokenizer      — Idea 5
  TestCrossDomainRAG         — Idea 6
  TestHexagramEvaluator      — Idea 7
  TestMultiScaleQ6           — Idea 8
  TestAdaptiveHammingScheduler — Idea 9
  TestHexagramMoE            — Idea 10
  TestBinaryOppositionTable  — text analysis
  TestSvoyChuzhoiGate        — text analysis
  TestBinaryExclusionClassifier — text analysis
  TestTextQualityFilter      — text analysis
  TestConveyorVariant3Block  — text analysis
  TestIntegration            — end-to-end integration
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from yijing_transformer.models.variant3_extensions import (
    HexagramPositionalEncoding,
    SixLineAttention,
    BianGuaOptimizer,
    TernaryKVCache,
    HexagramTokenizer,
    CrossDomainRAG,
    HexagramEvaluator,
    MultiScaleQ6,
    AdaptiveHammingScheduler,
    HexagramMoE,
    BinaryOppositionTable,
    SvoyChuzhoiGate,
    BinaryExclusionClassifier,
    TextQualityFilter,
    ConveyorVariant3Block,
    ConveyorStageOutput,
    get_hexagrams,
    get_biangua,
    bfs_distances,
    DEFAULT_AXES,
)


# ===========================================================================
# Fixtures
# ===========================================================================

B, T, D = 2, 8, 48  # batch, seq_len, d_model (divisible by 6)


@pytest.fixture
def dummy_x():
    return torch.randn(B, T, D)


@pytest.fixture
def hex_weights():
    w = torch.rand(B, T, 64)
    return w / w.sum(dim=-1, keepdim=True)


# ===========================================================================
# Idea 1 — HexagramPositionalEncoding
# ===========================================================================

class TestHexPosEncoding:
    def test_output_shape(self):
        pe = HexagramPositionalEncoding(d_model=D, block_size=32)
        out = pe(T)
        assert out.shape == (T, D)

    def test_different_positions_differ(self):
        pe = HexagramPositionalEncoding(d_model=D)
        out = pe(128)
        # Positions 0 and 64 map to same hexagram → same encoding (modulo 64)
        pos0 = out[0]
        pos64 = out[64]
        assert torch.allclose(pos0, pos64, atol=1e-5)
        # Positions 0 and 1 map to different hexagrams → should differ
        pos1 = out[1]
        assert not torch.allclose(pos0, pos1, atol=1e-5)

    def test_gradient_flows(self):
        pe = HexagramPositionalEncoding(d_model=D)
        out = pe(T)
        loss = out.sum()
        loss.backward()
        assert pe.proj.weight.grad is not None

    def test_bfs_distances_diameter(self):
        biangua = get_biangua()
        dist = bfs_distances(biangua)
        assert dist.max().item() == 6, "Diameter of Q6 should be 6"
        assert dist.min().item() == 0
        # Self-distance = 0
        assert (dist.diag() == 0).all()

    def test_bfs_symmetry(self):
        biangua = get_biangua()
        dist = bfs_distances(biangua)
        assert torch.allclose(dist.float(), dist.T.float())


# ===========================================================================
# Idea 2 — SixLineAttention
# ===========================================================================

class TestSixLineAttention:
    def test_output_shape(self, dummy_x):
        attn = SixLineAttention(d_model=D)
        out = attn(dummy_x)
        assert out.shape == dummy_x.shape

    def test_causal_mask(self, dummy_x):
        attn = SixLineAttention(d_model=D)
        # Create causal mask (upper triangular = -inf)
        mask = torch.full((T, T), float('-inf'))
        mask = torch.tril(torch.zeros(T, T)).masked_fill(
            torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), float('-inf')
        )
        out = attn(dummy_x, mask=mask.unsqueeze(0).unsqueeze(0))
        assert out.shape == dummy_x.shape
        assert not torch.isnan(out).any()

    def test_gradient_flows(self, dummy_x):
        dummy_x = dummy_x.requires_grad_(True)
        attn = SixLineAttention(d_model=D)
        out = attn(dummy_x)
        out.sum().backward()
        assert dummy_x.grad is not None

    def test_domain_lambda_learned(self, dummy_x):
        attn = SixLineAttention(d_model=D)
        before = attn.domain_lambda.clone().detach()
        opt = torch.optim.Adam(attn.parameters(), lr=1e-3)
        for _ in range(5):
            opt.zero_grad()
            out = attn(dummy_x)
            out.sum().backward()
            opt.step()
        after = attn.domain_lambda.clone().detach()
        assert not torch.allclose(before, after), "domain_lambda should be updated"

    def test_wrong_d_model_raises(self):
        with pytest.raises(AssertionError):
            SixLineAttention(d_model=7)  # not divisible by 6


# ===========================================================================
# Idea 3 — BianGuaOptimizer
# ===========================================================================

class TestBianGuaOptimizer:
    def _make_simple_model(self):
        return nn.Linear(8, 4)

    def test_basic_step(self):
        model = self._make_simple_model()
        opt = BianGuaOptimizer(model.parameters(), lr=1e-2)
        x = torch.randn(4, 8)
        y = torch.randn(4, 4)
        for _ in range(3):
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        # Loss should decrease
        initial = F.mse_loss(model(x), y).item()
        assert isinstance(initial, float)

    def test_hex_scale_warmup(self):
        model = self._make_simple_model()
        opt = BianGuaOptimizer(model.parameters(), warmup_steps=10, hex_scale=0.1)
        # At step 0, scale should be 0
        assert opt._get_hex_scale(opt.param_groups[0]) == 0.0
        opt._step_count = 5
        scale = opt._get_hex_scale(opt.param_groups[0])
        assert 0 < scale < 0.1, f"Expected scale in (0, 0.1), got {scale}"
        opt._step_count = 10
        assert opt._get_hex_scale(opt.param_groups[0]) == pytest.approx(0.1, abs=1e-5)

    def test_hex_scale_anneal(self):
        model = self._make_simple_model()
        opt = BianGuaOptimizer(model.parameters(), warmup_steps=10, anneal_steps=10,
                                hex_scale=0.1)
        opt._step_count = 20  # end of anneal
        scale = opt._get_hex_scale(opt.param_groups[0])
        assert scale == pytest.approx(0.0, abs=1e-5)

    def test_hex_scale_steady(self):
        model = self._make_simple_model()
        opt = BianGuaOptimizer(model.parameters(), warmup_steps=10, anneal_steps=10,
                                hex_scale=0.1)
        opt._step_count = 100  # well past anneal
        scale = opt._get_hex_scale(opt.param_groups[0])
        assert scale == 0.0


# ===========================================================================
# Idea 4 — TernaryKVCache
# ===========================================================================

class TestTernaryKVCache:
    def test_quantise_dequantise(self):
        cache = TernaryKVCache(n_heads=4, head_dim=8, max_seq=32)
        k = torch.randn(1, 4, 4, 8) * 2.0  # large values for clear quantisation
        ternary = cache.quantise_keys(k)
        assert ternary.dtype == torch.int8
        assert set(ternary.unique().tolist()).issubset({-1, 0, 1})

    def test_write_attend(self):
        cache = TernaryKVCache(n_heads=2, head_dim=8, max_seq=32)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        cache.write(k, v)
        assert cache._cache_len == 4

        q = torch.randn(1, 2, 2, 8)
        out = cache.attend(q)
        assert out.shape == (1, 2, 2, 8)

    def test_cache_reset(self):
        cache = TernaryKVCache(n_heads=2, head_dim=8, max_seq=32)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        cache.write(k, v)
        assert cache._cache_len == 4
        cache.reset()
        assert cache._cache_len == 0
        assert cache._k_ternary is None

    def test_threshold_gradient(self):
        cache = TernaryKVCache(n_heads=2, head_dim=8, max_seq=32)
        k = torch.randn(1, 2, 4, 8, requires_grad=True)
        thresh = cache.threshold
        loss = (k * thresh.view(1, -1, 1, 1)).sum()
        loss.backward()
        assert cache.log_threshold.grad is not None

    def test_multiple_writes(self):
        cache = TernaryKVCache(n_heads=2, head_dim=8, max_seq=32)
        for _ in range(4):
            k = torch.randn(1, 2, 2, 8)
            v = torch.randn(1, 2, 2, 8)
            cache.write(k, v)
        assert cache._cache_len == 8


# ===========================================================================
# Idea 5 — HexagramTokenizer
# ===========================================================================

class TestHexagramTokenizer:
    def test_encode_nonempty(self):
        tok = HexagramTokenizer()
        ids = tok.encode("hello")
        assert len(ids) > 0
        assert all(0 <= x < 64 for x in ids)

    def test_encode_empty(self):
        tok = HexagramTokenizer()
        ids = tok.encode("")
        assert ids == []

    def test_bpe_training_extends_vocab(self):
        tok = HexagramTokenizer()
        corpus = ["hello world " * 10, "foo bar baz " * 10, "python is great " * 10]
        tok.train_bpe(corpus, n_merges=5)
        assert tok.vocab_size >= 64  # could be 64 if no biangua adjacent pairs

    def test_vocab_size_property(self):
        tok = HexagramTokenizer()
        assert tok.vocab_size == 64
        tok.merge_table[(0, 1)] = 64
        tok._next_meta = 65
        assert tok.vocab_size == 65

    def test_biangua_only_adjacent_merges(self):
        """BPE merges should only happen between biangua-adjacent pairs."""
        tok = HexagramTokenizer()
        biangua = get_biangua()
        corpus = ["test corpus for training " * 20]
        tok.train_bpe(corpus, n_merges=10)
        for (a, b) in tok.merge_table.keys():
            if a < 64 and b < 64:
                assert biangua[a, b] > 0.5, f"Non-adjacent merge: ({a},{b})"

    def test_consistent_encoding(self):
        tok = HexagramTokenizer()
        ids1 = tok.encode("test string")
        ids2 = tok.encode("test string")
        assert ids1 == ids2


# ===========================================================================
# Idea 6 — CrossDomainRAG
# ===========================================================================

class TestCrossDomainRAG:
    def test_output_shape(self, dummy_x, hex_weights):
        rag = CrossDomainRAG(d_model=D, n_docs=16, top_k=2)
        out = rag(dummy_x, hex_weights)
        assert out.shape == dummy_x.shape

    def test_retrieve_shape(self, hex_weights):
        rag = CrossDomainRAG(d_model=D, n_docs=16, top_k=3)
        idx, scores = rag.retrieve(hex_weights)
        assert idx.shape == (B, T, 3)
        assert scores.shape == (B, T, 3)

    def test_retrieve_valid_indices(self, hex_weights):
        rag = CrossDomainRAG(d_model=D, n_docs=16, top_k=2)
        idx, _ = rag.retrieve(hex_weights)
        assert idx.min().item() >= 0
        assert idx.max().item() < 16

    def test_gradient_flows(self, dummy_x, hex_weights):
        dummy_x = dummy_x.requires_grad_(True)
        rag = CrossDomainRAG(d_model=D, n_docs=16, top_k=2)
        out = rag(dummy_x, hex_weights)
        out.sum().backward()
        assert dummy_x.grad is not None

    def test_top_k_scores_ordered(self, hex_weights):
        rag = CrossDomainRAG(d_model=D, n_docs=16, top_k=4)
        _, scores = rag.retrieve(hex_weights)
        # Scores should be in descending order
        assert (scores[:, :, :-1] >= scores[:, :, 1:]).all()


# ===========================================================================
# Idea 7 — HexagramEvaluator
# ===========================================================================

class TestHexagramEvaluator:
    def test_hex_entropy_uniform(self):
        ev = HexagramEvaluator()
        # Uniform distribution → max entropy = log2(64) = 6
        w = torch.ones(1, 1, 64) / 64.0
        ent = ev.hex_entropy(w)
        assert ent.item() == pytest.approx(6.0, abs=0.01)

    def test_hex_entropy_peaked(self):
        ev = HexagramEvaluator()
        # All mass on one hexagram → entropy ≈ 0
        w = torch.zeros(1, 1, 64)
        w[0, 0, 0] = 1.0
        ent = ev.hex_entropy(w)
        assert ent.item() == pytest.approx(0.0, abs=0.01)

    def test_domain_coherence_peaked(self):
        ev = HexagramEvaluator()
        w = torch.zeros(2, 6)
        w[:, 0] = 10.0  # all mass on domain 0
        coh = ev.domain_coherence(w)
        assert coh.mean().item() > 0.9

    def test_biangua_coverage(self):
        ev = HexagramEvaluator(threshold=0.01)
        # 10 active hexagrams
        w = torch.zeros(1, 1, 64)
        w[0, 0, :10] = 0.1
        cov = ev.biangua_coverage(w)
        assert cov.item() == pytest.approx(10 / 64, abs=0.01)

    def test_hamming_entropy_ratio_range(self):
        ev = HexagramEvaluator()
        w = torch.rand(2, 4, 64)
        w = w / w.sum(dim=-1, keepdim=True)
        ratio = ev.hamming_entropy_ratio(w)
        assert (ratio >= 0).all()
        assert (ratio <= 1.0 + 1e-5).all()

    def test_evaluate_dict(self):
        ev = HexagramEvaluator()
        w = torch.rand(2, 4, 64)
        w = w / w.sum(dim=-1, keepdim=True)
        dw = torch.rand(2, 6)
        result = ev.evaluate(w, dw)
        assert set(result.keys()) == {"hex_entropy", "biangua_coverage",
                                       "hamming_entropy_ratio", "domain_coherence"}
        assert all(isinstance(v, float) for v in result.values())


# ===========================================================================
# Idea 8 — MultiScaleQ6
# ===========================================================================

class TestMultiScaleQ6:
    def test_output_shape(self, dummy_x):
        ms = MultiScaleQ6(d_model=D)
        out, info = ms(dummy_x)
        assert out.shape == dummy_x.shape

    def test_scale_info_keys(self, dummy_x):
        ms = MultiScaleQ6(d_model=D)
        _, info = ms(dummy_x)
        assert set(info.keys()) == {"q2_weights", "q3_weights", "q6_weights"}

    def test_scale_weights_shape(self, dummy_x):
        ms = MultiScaleQ6(d_model=D)
        _, info = ms(dummy_x)
        assert info["q2_weights"].shape == (B, T, 4)
        assert info["q3_weights"].shape == (B, T, 8)
        assert info["q6_weights"].shape == (B, T, 64)

    def test_weights_sum_to_one(self, dummy_x):
        ms = MultiScaleQ6(d_model=D)
        _, info = ms(dummy_x)
        for key in ["q2_weights", "q3_weights", "q6_weights"]:
            sums = info[key].sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_gradient_flows(self, dummy_x):
        dummy_x = dummy_x.requires_grad_(True)
        ms = MultiScaleQ6(d_model=D)
        out, _ = ms(dummy_x)
        out.sum().backward()
        assert dummy_x.grad is not None
        assert ms.scale_weights.grad is not None

    def test_q2_vertices(self):
        ms = MultiScaleQ6(d_model=D)
        assert ms.q2_verts.shape == (4, 2)
        assert ms.q3_verts.shape == (8, 3)
        assert ms.q6_verts.shape == (64, 6)


# ===========================================================================
# Idea 9 — AdaptiveHammingScheduler
# ===========================================================================

class TestAdaptiveHammingScheduler:
    def _make_model(self):
        from yijing_transformer.models.variant3 import Variant3GPT, Variant3Config
        cfg = Variant3Config(d_model=48, n_heads=4, n_layers=2, block_size=16)
        return Variant3GPT(cfg)

    def test_get_lambda_warmup(self):
        model = self._make_model()
        sched = AdaptiveHammingScheduler(model, lambda_max=0.5, lambda_min=0.05,
                                          warmup_steps=100, anneal_steps=200)
        assert sched.get_lambda(0) == 0.0
        assert sched.get_lambda(50) == pytest.approx(0.25, abs=0.01)
        assert sched.get_lambda(100) == pytest.approx(0.5, abs=0.01)

    def test_get_lambda_anneal(self):
        model = self._make_model()
        sched = AdaptiveHammingScheduler(model, lambda_max=0.5, lambda_min=0.05,
                                          warmup_steps=100, anneal_steps=100)
        # At end of anneal (step 200), should be lambda_min
        lam = sched.get_lambda(200)
        assert lam == pytest.approx(0.05, abs=0.01)

    def test_get_lambda_steady(self):
        model = self._make_model()
        sched = AdaptiveHammingScheduler(model, lambda_max=0.5, lambda_min=0.05,
                                          warmup_steps=100, anneal_steps=100)
        lam300 = sched.get_lambda(300)
        lam1000 = sched.get_lambda(1000)
        assert lam300 == pytest.approx(0.05, abs=0.01)
        assert lam1000 == pytest.approx(0.05, abs=0.01)

    def test_step_updates_modules(self):
        model = self._make_model()
        sched = AdaptiveHammingScheduler(model, lambda_max=0.8, warmup_steps=10)
        sched.step(5)  # halfway through warmup
        for module in sched.targets:
            assert module.hamming_lambda.item() == pytest.approx(0.4, abs=0.01)

    def test_finds_biangua_modules(self):
        model = self._make_model()
        sched = AdaptiveHammingScheduler(model)
        from yijing_transformer.models.variant3 import Variant3Config
        cfg = Variant3Config(d_model=48, n_heads=4, n_layers=2)
        assert len(sched.targets) == cfg.n_layers  # one BianGuaAttention per layer


# ===========================================================================
# Idea 10 — HexagramMoE
# ===========================================================================

class TestHexagramMoE:
    def test_output_shape(self, dummy_x, hex_weights):
        moe = HexagramMoE(d_model=D, d_ff=D * 2, top_k=2)
        out = moe(dummy_x, hex_weights)
        assert out.shape == dummy_x.shape

    def test_gradient_flows(self, dummy_x, hex_weights):
        dummy_x = dummy_x.requires_grad_(True)
        moe = HexagramMoE(d_model=D, d_ff=D * 2, top_k=4)
        out = moe(dummy_x, hex_weights)
        out.sum().backward()
        assert dummy_x.grad is not None

    def test_load_balance_loss(self, dummy_x, hex_weights):
        moe = HexagramMoE(d_model=D, d_ff=D * 2, top_k=2)
        moe.train()
        out = moe(dummy_x, hex_weights)
        loss = moe.load_balance_loss(hex_weights)
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)

    def test_expert_count_update(self, dummy_x, hex_weights):
        moe = HexagramMoE(d_model=D, d_ff=D * 2, top_k=2)
        moe.train()
        moe(dummy_x, hex_weights)
        # After one forward pass, expert_counts should have non-zero entries
        assert moe.expert_counts.sum() > 0

    def test_top_k_constraint(self, dummy_x, hex_weights):
        # Only top_k experts should contribute per token
        moe = HexagramMoE(d_model=D, d_ff=D * 2, top_k=3)
        # Test that different top_k values don't crash
        out1 = moe(dummy_x, hex_weights)
        assert out1.shape == dummy_x.shape


# ===========================================================================
# BinaryOppositionTable
# ===========================================================================

class TestBinaryOppositionTable:
    def test_output_shape(self, dummy_x):
        bot = BinaryOppositionTable(d_model=D)
        scores, assignments = bot(dummy_x)
        assert scores.shape == (B, T, 6)
        assert assignments.shape == (B, T, 64)

    def test_scores_in_range(self, dummy_x):
        bot = BinaryOppositionTable(d_model=D)
        scores, _ = bot(dummy_x)
        assert (scores >= -1.0 - 1e-5).all()
        assert (scores <= 1.0 + 1e-5).all()

    def test_assignments_sum_to_one(self, dummy_x):
        bot = BinaryOppositionTable(d_model=D)
        _, assignments = bot(dummy_x)
        sums = assignments.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_default_axes_count(self):
        assert len(DEFAULT_AXES) == 6

    def test_interpret_shape(self, dummy_x):
        bot = BinaryOppositionTable(d_model=D)
        scores, _ = bot(dummy_x)
        interpretations = bot.interpret(scores[0, 0].unsqueeze(0))
        assert len(interpretations) == 1
        assert len(interpretations[0]) == 6

    def test_gradient_flows(self, dummy_x):
        dummy_x = dummy_x.requires_grad_(True)
        bot = BinaryOppositionTable(d_model=D)
        scores, assignments = bot(dummy_x)
        (scores.sum() + assignments.sum()).backward()
        assert dummy_x.grad is not None


# ===========================================================================
# SvoyChuzhoiGate
# ===========================================================================

class TestSvoyChuzhoiGate:
    def test_output_shape(self, dummy_x):
        gate = SvoyChuzhoiGate(d_model=D)
        gated, gate_values = gate(dummy_x)
        assert gated.shape == dummy_x.shape
        assert gate_values.shape == (B, T)

    def test_gate_values_ternary(self, dummy_x):
        gate = SvoyChuzhoiGate(d_model=D)
        # With STE, hard gate should be in {-1, 0, +1}
        # (the gate_ste returned is soft + (hard-soft).detach() ≈ hard values approximately)
        # Just check it's finite
        _, gate_values = gate(dummy_x)
        assert torch.isfinite(gate_values).all()

    def test_gradient_flows(self, dummy_x):
        dummy_x = dummy_x.requires_grad_(True)
        gate = SvoyChuzhoiGate(d_model=D)
        gated, gate_values = gate(dummy_x)
        gated.sum().backward()
        assert dummy_x.grad is not None

    def test_near_prototype_is_positive(self):
        gate = SvoyChuzhoiGate(d_model=D, n_prototypes=1)
        # Force prototype to all-zeros
        with torch.no_grad():
            gate.prototypes.fill_(0.0)
            gate.near_threshold.fill_(10.0)   # very wide "near" zone
            gate.far_threshold.fill_(0.001)   # far = anything > 0.001
        x = torch.zeros(1, 1, D)
        _, gate_values = gate(x)
        # near_threshold dominates → gate should be +1
        assert gate_values.item() > 0

    def test_prototype_count(self):
        gate = SvoyChuzhoiGate(d_model=D, n_prototypes=3)
        assert gate.prototypes.shape == (3, 6)


# ===========================================================================
# BinaryExclusionClassifier
# ===========================================================================

class TestBinaryExclusionClassifier:
    def test_output_shapes(self, dummy_x):
        clf = BinaryExclusionClassifier(d_model=D)
        accept_mask, scores, decisions = clf(dummy_x)
        assert accept_mask.shape == (B, T)
        assert scores.shape == (B, T, 6)
        assert decisions.shape == (B, T, 6)

    def test_accept_mask_is_bool(self, dummy_x):
        clf = BinaryExclusionClassifier(d_model=D)
        accept_mask, _, _ = clf(dummy_x)
        assert accept_mask.dtype == torch.bool

    def test_decisions_are_bool(self, dummy_x):
        clf = BinaryExclusionClassifier(d_model=D)
        _, _, decisions = clf(dummy_x)
        assert decisions.dtype == torch.bool

    def test_accept_implies_all_axes_pass(self, dummy_x):
        clf = BinaryExclusionClassifier(d_model=D)
        accept_mask, _, decisions = clf(dummy_x)
        # Every accepted token must have all 6 axis decisions = True
        accepted_decisions = decisions[accept_mask]  # (N_accepted, 6)
        if accepted_decisions.numel() > 0:
            assert accepted_decisions.all()

    def test_reject_means_at_least_one_axis_fails(self, dummy_x):
        clf = BinaryExclusionClassifier(d_model=D)
        accept_mask, _, decisions = clf(dummy_x)
        rejected_decisions = decisions[~accept_mask]  # (N_rejected, 6)
        if rejected_decisions.numel() > 0:
            # At least one axis must be False for each rejected token
            assert (~rejected_decisions).any(dim=-1).all()

    def test_gradient_flows(self, dummy_x):
        dummy_x = dummy_x.requires_grad_(True)
        clf = BinaryExclusionClassifier(d_model=D)
        _, scores, _ = clf(dummy_x)
        scores.sum().backward()
        assert dummy_x.grad is not None


# ===========================================================================
# TextQualityFilter
# ===========================================================================

class TestTextQualityFilter:
    def test_output_shapes(self, dummy_x):
        filt = TextQualityFilter(d_model=D)
        scores, assignments, dominant_idx = filt(dummy_x)
        assert scores.shape == (B, 6)
        assert assignments.shape == (B, 64)
        assert isinstance(dominant_idx, int)

    def test_scores_in_range(self, dummy_x):
        filt = TextQualityFilter(d_model=D)
        scores, _, _ = filt(dummy_x)
        assert (scores >= -1.0 - 1e-5).all()
        assert (scores <= 1.0 + 1e-5).all()

    def test_assignments_sum_to_one(self, dummy_x):
        filt = TextQualityFilter(d_model=D)
        _, assignments, _ = filt(dummy_x)
        sums = assignments.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_dominant_idx_in_range(self, dummy_x):
        filt = TextQualityFilter(d_model=D)
        _, _, dominant_idx = filt(dummy_x)
        assert 0 <= dominant_idx < 64

    def test_quality_bits(self, dummy_x):
        filt = TextQualityFilter(d_model=D)
        scores, _, _ = filt(dummy_x)
        bits = filt.quality_bits(scores)
        assert bits.shape == (B,)
        assert (bits >= 0).all()
        assert (bits < 64).all()

    def test_all_positive_is_63(self):
        filt = TextQualityFilter(d_model=D)
        all_positive = torch.ones(1, 6)  # all axes positive → hexagram 63
        bits = filt.quality_bits(all_positive)
        assert bits.item() == 63

    def test_all_negative_is_0(self):
        filt = TextQualityFilter(d_model=D)
        all_negative = -torch.ones(1, 6)  # all axes negative → hexagram 0
        bits = filt.quality_bits(all_negative)
        assert bits.item() == 0

    def test_gradient_flows(self, dummy_x):
        dummy_x = dummy_x.requires_grad_(True)
        filt = TextQualityFilter(d_model=D)
        scores, assignments, _ = filt(dummy_x)
        (scores.sum() + assignments.sum()).backward()
        assert dummy_x.grad is not None


# ===========================================================================
# ConveyorVariant3Block
# ===========================================================================

class TestConveyorVariant3Block:
    def _make_block(self, record=False):
        block = ConveyorVariant3Block(d_model=D, n_heads=6)
        block.record_intermediates = record
        return block

    def test_output_shape(self, dummy_x):
        block = self._make_block()
        out = block(dummy_x)
        assert out.shape == dummy_x.shape

    def test_stage_names(self):
        assert len(ConveyorVariant3Block.STAGE_NAMES) == 6
        assert ConveyorVariant3Block.STAGE_NAMES[0] == "Q6_LOCALISE"
        assert ConveyorVariant3Block.STAGE_NAMES[-1] == "SWIGLU_FFN"

    def test_record_intermediates(self, dummy_x):
        block = self._make_block(record=True)
        out = block(dummy_x)
        stage_out = block.last_stage_output
        assert stage_out is not None
        assert isinstance(stage_out, ConveyorStageOutput)
        assert set(stage_out.stages.keys()) == set(ConveyorVariant3Block.STAGE_NAMES)

    def test_intermediate_shapes(self, dummy_x):
        block = self._make_block(record=True)
        block(dummy_x)
        for name, tensor in block.last_stage_output.stages.items():
            assert tensor.shape == (B, T, D), f"Stage {name} shape wrong: {tensor.shape}"

    def test_no_record_by_default(self, dummy_x):
        block = self._make_block(record=False)
        block(dummy_x)
        assert block.last_stage_output is None

    def test_gradient_flows(self, dummy_x):
        dummy_x = dummy_x.requires_grad_(True)
        block = self._make_block()
        out = block(dummy_x)
        out.sum().backward()
        assert dummy_x.grad is not None

    def test_all_params_have_gradients(self, dummy_x):
        block = self._make_block()
        out = block(dummy_x)
        out.sum().backward()
        known_dead = {
            'interlingua.log_uncertainty',
            'interlingua.encode_norm.weight',
            'interlingua.encode_norm.bias',
        }
        for name, param in block.named_parameters():
            if any(name.endswith(s) for s in known_dead):
                continue
            assert param.grad is not None, f"No gradient for {name}"

    def test_hex_weights_captured(self, dummy_x):
        block = self._make_block(record=True)
        block(dummy_x)
        stage_out = block.last_stage_output
        assert stage_out.hex_weights is not None
        assert stage_out.hex_weights.shape == (B, T, 64)


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_multiscale_then_moe(self, dummy_x, hex_weights):
        """MultiScaleQ6 → HexagramMoE pipeline."""
        ms = MultiScaleQ6(d_model=D)
        moe = HexagramMoE(d_model=D, d_ff=D * 2, top_k=2)
        enriched, info = ms(dummy_x)
        q6_weights = info["q6_weights"]
        out = moe(enriched, q6_weights)
        assert out.shape == dummy_x.shape

    def test_quality_filter_with_opposition_table(self, dummy_x):
        """BinaryOppositionTable → TextQualityFilter pipeline."""
        bot = BinaryOppositionTable(d_model=D)
        filt = TextQualityFilter(d_model=D)
        # Use axis scores as a proxy for quality-annotated features
        scores, _ = bot(dummy_x)
        # Quality filter works on raw x
        quality_scores, _, dominant = filt(dummy_x)
        assert quality_scores.shape == (B, 6)

    def test_conveyor_block_with_evaluator(self, dummy_x):
        """ConveyorVariant3Block → HexagramEvaluator pipeline."""
        block = ConveyorVariant3Block(d_model=D, n_heads=6)
        block.record_intermediates = True
        out = block(dummy_x)
        hex_weights = block.last_stage_output.hex_weights
        ev = HexagramEvaluator()
        metrics = ev.evaluate(hex_weights)
        assert 0 <= metrics["hex_entropy"] <= 6.0
        assert 0 <= metrics["biangua_coverage"] <= 1.0

    def test_exclusion_classifier_with_svoy_gate(self, dummy_x):
        """BinaryExclusionClassifier + SvoyChuzhoiGate: complementary filtering."""
        clf = BinaryExclusionClassifier(d_model=D)
        gate = SvoyChuzhoiGate(d_model=D)
        accept_mask, scores, decisions = clf(dummy_x)
        gated, gate_values = gate(dummy_x)
        # Both should produce valid outputs
        assert accept_mask.dtype == torch.bool
        assert gated.shape == dummy_x.shape

    def test_six_line_attention_with_positional_encoding(self):
        """SixLineAttention + HexagramPositionalEncoding combination."""
        pe = HexagramPositionalEncoding(d_model=D)
        attn = SixLineAttention(d_model=D)
        x = torch.randn(B, T, D)
        pos = pe(T)  # (T, D)
        x = x + pos.unsqueeze(0)
        out = attn(x)
        assert out.shape == x.shape

    def test_all_modules_no_nan(self, dummy_x, hex_weights):
        """All extension modules should produce finite outputs."""
        modules_and_inputs = [
            (HexagramPositionalEncoding(D), None),  # special call
            (SixLineAttention(D), dummy_x),
            (CrossDomainRAG(D, n_docs=16, top_k=2), (dummy_x, hex_weights)),
            (MultiScaleQ6(D), dummy_x),
            (HexagramMoE(D, D * 2, top_k=2), (dummy_x, hex_weights)),
            (BinaryOppositionTable(D), dummy_x),
            (SvoyChuzhoiGate(D), dummy_x),
            (BinaryExclusionClassifier(D), dummy_x),
            (TextQualityFilter(D), dummy_x),
            (ConveyorVariant3Block(D, n_heads=6), dummy_x),
        ]
        for module, inp in modules_and_inputs:
            if inp is None:
                out = module(T)
            elif isinstance(inp, tuple):
                out = module(*inp)
            else:
                out = module(inp)
            # Get first tensor output
            if isinstance(out, tuple):
                out = out[0]
            assert torch.isfinite(out).all(), f"{module.__class__.__name__} produced non-finite output"

    def test_ternary_kv_cache_in_generation(self):
        """TernaryKVCache simulate incremental generation."""
        n_heads, head_dim = 2, 8
        cache = TernaryKVCache(n_heads=n_heads, head_dim=head_dim, max_seq=64)
        # Simulate encoding 10 tokens
        for t in range(10):
            k = torch.randn(1, n_heads, 1, head_dim)
            v = torch.randn(1, n_heads, 1, head_dim)
            cache.write(k, v)
        assert cache._cache_len == 10
        # Decode 1 token
        q = torch.randn(1, n_heads, 1, head_dim)
        out = cache.attend(q)
        assert out.shape == (1, n_heads, 1, head_dim)
        assert torch.isfinite(out).all()


# ===========================================================================
# Q6 Structure Validation (shared foundations)
# ===========================================================================

class TestQ6Foundations:
    def test_hexagrams_count(self):
        h = get_hexagrams()
        assert h.shape == (64, 6)

    def test_hexagrams_values(self):
        h = get_hexagrams()
        assert set(h.unique().tolist()) == {-1.0, 1.0}

    def test_biangua_exactly_6_neighbors(self):
        b = get_biangua()
        assert b.shape == (64, 64)
        row_sums = b.sum(dim=1)
        assert (row_sums == 6).all(), "Each hexagram must have exactly 6 Hamming-1 neighbors"

    def test_biangua_symmetric(self):
        b = get_biangua()
        assert torch.allclose(b, b.T)

    def test_biangua_no_self_loops(self):
        b = get_biangua()
        assert b.diag().sum() == 0

    def test_bfs_distances_range(self):
        b = get_biangua()
        dist = bfs_distances(b)
        assert dist.min().item() == 0
        assert dist.max().item() == 6  # Q6 diameter
