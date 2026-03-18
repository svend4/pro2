"""
Тесты для hierarchical_moe.py — 4-уровневая Mixture-of-Experts архитектура.

Покрывает:
  - Q6ExpertBank: forward, load_balance_loss, gradient flow
  - MicroExpert: forward shape, gradient flow
  - BidirBridgeExpert: forward, alpha balance, causal mask
  - GroupRouter: forward, lb_loss sign, anticircle penalty
  - MultiScaleGlobalRouter: forward, matryoshka routing, lb_loss
  - GlobalRouter (legacy): forward, lb_loss sign
  - HierarchicalMoEFFN: full forward, stage progression, freeze/unfreeze
"""

import math
import pytest
import torch
import torch.nn as nn

from yijing_transformer.models.hierarchical_moe import (
    Q6ExpertBank,
    MicroExpert,
    BidirBridgeExpert,
    GroupRouter,
    GlobalRouter,
    MultiScaleGlobalRouter,
    HierarchicalMoEFFN,
    HMoEConfig,
    DOMAIN_GROUPS,
    CLUSTER_TO_DOMAIN,
    TRAINING_STAGES,
    set_moe_stage,
    get_stage_info,
)


torch.manual_seed(42)

B, T, D = 2, 8, 64  # batch, seq_len, d_model


# ═══════════════════════════════════════════════════════════════════════════════
# Q6ExpertBank
# ═══════════════════════════════════════════════════════════════════════════════

class TestQ6ExpertBank:
    @pytest.fixture
    def bank(self):
        return Q6ExpertBank(d_model=D, d_ff=D * 2, top_k=4)

    def test_forward_shape(self, bank):
        x = torch.randn(B, T, D)
        hw = torch.softmax(torch.randn(B, T, 64), dim=-1)
        out = bank(x, hw)
        assert out.shape == (B, T, D)

    def test_no_nan(self, bank):
        x = torch.randn(B, T, D)
        hw = torch.softmax(torch.randn(B, T, 64), dim=-1)
        out = bank(x, hw)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, bank):
        x = torch.randn(B, T, D, requires_grad=True)
        hw = torch.softmax(torch.randn(B, T, 64), dim=-1)
        out = bank(x, hw)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_load_balance_loss_sign(self, bank):
        """lb_loss = Σ p·log(p) should be negative (entropy)."""
        hw = torch.softmax(torch.randn(B, T, 64), dim=-1)
        loss = bank.load_balance_loss(hw)
        assert loss.item() <= 0, "lb_loss should be ≤ 0 (negative entropy)"

    def test_load_balance_loss_uniform_is_minimal(self, bank):
        """Uniform distribution should give maximum negative entropy."""
        hw_uniform = torch.ones(B, T, 64) / 64
        hw_peaked = torch.zeros(B, T, 64)
        hw_peaked[:, :, 0] = 1.0
        loss_uniform = bank.load_balance_loss(hw_uniform)
        loss_peaked = bank.load_balance_loss(hw_peaked)
        # Uniform = max entropy → most negative lb_loss → best balance
        assert loss_uniform.item() < loss_peaked.item()


# ═══════════════════════════════════════════════════════════════════════════════
# MicroExpert
# ═══════════════════════════════════════════════════════════════════════════════

class TestMicroExpert:
    def test_forward_shape(self):
        expert = MicroExpert(D, expansion=2)
        x = torch.randn(B, T, D)
        out = expert(x)
        assert out.shape == (B, T, D)

    def test_gradient_flow(self):
        expert = MicroExpert(D, expansion=2)
        x = torch.randn(B, T, D, requires_grad=True)
        out = expert(x)
        out.sum().backward()
        assert x.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# BidirBridgeExpert
# ═══════════════════════════════════════════════════════════════════════════════

class TestBidirBridgeExpert:
    @pytest.fixture
    def bridge(self):
        return BidirBridgeExpert(d_model=D)

    def test_forward_shape(self, bridge):
        xa = torch.randn(B, T, D)
        xb = torch.randn(B, T, D)
        out = bridge(xa, xb)
        assert out.shape == (B, T, D)

    def test_initial_gate_near_zero(self, bridge):
        """Gate initialized to sigmoid(-2.0) ≈ 0.12, so output starts small."""
        xa = torch.randn(B, T, D)
        xb = torch.randn(B, T, D)
        out = bridge(xa, xb)
        # With gate ≈ 0.12, output should be dampened
        assert out.abs().mean().item() < xa.abs().mean().item()

    def test_alpha_starts_balanced(self, bridge):
        """log_alpha=0 → sigmoid(0)=0.5 → equal fwd/bwd weight."""
        alpha = torch.sigmoid(bridge.log_alpha)
        assert abs(alpha.item() - 0.5) < 1e-6

    def test_gradient_flow_both_inputs(self, bridge):
        xa = torch.randn(B, T, D, requires_grad=True)
        xb = torch.randn(B, T, D, requires_grad=True)
        out = bridge(xa, xb)
        out.sum().backward()
        assert xa.grad is not None
        assert xb.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# GroupRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroupRouter:
    @pytest.fixture
    def router(self):
        return GroupRouter(D, expert_names=["Theory", "Models"], streak_limit=4)

    def test_forward_shape(self, router):
        x = torch.randn(B, T, D)
        weights, indices, lb = router(x, top_k=2)
        assert weights.shape == (B, T, 2)
        assert indices.shape == (B, T, 2)
        assert lb.shape == ()

    def test_weights_sum_to_one(self, router):
        x = torch.randn(B, T, D)
        weights, _, _ = router(x, top_k=2)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_lb_loss_sign(self, router):
        """lb_loss core term Σ p·log(p) ≤ 0."""
        x = torch.randn(B, T, D)
        _, _, lb = router(x, top_k=2)
        # lb = Σp·log(p) + anticircle_penalty
        # At init, anticircle_penalty should be 0 (uniform EMA)
        # Core entropy term is always ≤ 0
        assert lb.item() <= 0.1, "lb_loss should not be large positive"

    def test_anticircle_penalty_triggers(self):
        """When one expert dominates, anticircle penalty > 0."""
        router = GroupRouter(D, expert_names=["A", "B", "C"], streak_limit=4,
                             anticircle_weight=1.0)
        # Force EMA to be very skewed
        router._ema_load.fill_(0.0)
        router._ema_load[0] = 1.0
        x = torch.randn(B, T, D)
        # Compare with a balanced version
        router_balanced = GroupRouter(D, expert_names=["A", "B", "C"],
                                      streak_limit=4, anticircle_weight=1.0)
        # Copy weights so only EMA differs
        router_balanced.load_state_dict(router.state_dict(), strict=False)
        router_balanced._ema_load.fill_(1.0 / 3)
        _, _, lb_skewed = router(x, top_k=1)
        _, _, lb_balanced = router_balanced(x, top_k=1)
        # Skewed EMA should produce higher lb_loss (anticircle adds positive penalty)
        assert lb_skewed.item() > lb_balanced.item(), \
            "anticircle penalty should increase lb_loss for skewed EMA"

    def test_gradient_flow(self, router):
        x = torch.randn(B, T, D, requires_grad=True)
        weights, _, lb = router(x, top_k=2)
        (weights.sum() + lb).backward()
        assert x.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# GlobalRouter (legacy)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalRouter:
    def test_forward_shape(self):
        router = GlobalRouter(D, list(DOMAIN_GROUPS.keys()))
        x = torch.randn(B, T, D)
        gw, lb = router(x)
        assert gw.shape == (B, T, 3)
        assert lb.shape == ()

    def test_lb_loss_not_positive(self):
        """After .neg() removal, lb_loss = Σ p·log(p) ≤ 0."""
        router = GlobalRouter(D, list(DOMAIN_GROUPS.keys()))
        x = torch.randn(B, T, D)
        _, lb = router(x)
        assert lb.item() <= 0.01, f"lb_loss should be ≤ 0, got {lb.item()}"


# ═══════════════════════════════════════════════════════════════════════════════
# MultiScaleGlobalRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiScaleGlobalRouter:
    @pytest.fixture
    def router(self):
        return MultiScaleGlobalRouter(D, list(DOMAIN_GROUPS.keys()))

    def test_forward_shape(self, router):
        x = torch.randn(B, T, D)
        gw, hw, lb = router(x)
        assert gw.shape == (B, T, 3), f"group_weights wrong shape: {gw.shape}"
        assert hw.shape == (B, T, 64), f"hex_weights wrong shape: {hw.shape}"
        assert lb.shape == ()

    def test_group_weights_sum_to_one(self, router):
        x = torch.randn(B, T, D)
        gw, _, _ = router(x)
        sums = gw.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_hex_weights_sum_to_one(self, router):
        x = torch.randn(B, T, D)
        _, hw, _ = router(x)
        sums = hw.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_scale_mix_initial(self, router):
        """Initial log_scale_mix = [0,0,0] → equal mix of Q2, Q3, Q6."""
        mix = torch.softmax(router.log_scale_mix, dim=0)
        assert torch.allclose(mix, torch.ones(3) / 3, atol=1e-5)

    def test_gradient_flow(self, router):
        x = torch.randn(B, T, D, requires_grad=True)
        gw, hw, lb = router(x)
        (gw.sum() + hw.sum() + lb).backward()
        assert x.grad is not None

    def test_temperature_bounded(self, router):
        router.log_temp.data.fill_(10.0)  # exp(10) ≈ 22000
        assert router.temperature.item() <= 5.0
        router.log_temp.data.fill_(-10.0)
        assert router.temperature.item() >= 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# HierarchicalMoEFFN
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchicalMoEFFN:
    @pytest.fixture
    def hmoe(self):
        cfg = HMoEConfig(d_model=D, expert_expansion=2, top_k_experts=2,
                         use_multiscale=True, use_hex_tier=False)
        return HierarchicalMoEFFN(cfg)

    @pytest.fixture
    def hmoe_hex(self):
        cfg = HMoEConfig(d_model=D, expert_expansion=2, top_k_experts=2,
                         use_multiscale=True, use_hex_tier=True,
                         hex_tier_d_ff_mult=2)
        return HierarchicalMoEFFN(cfg)

    def test_forward_shape(self, hmoe):
        x = torch.randn(B, T, D)
        out, info = hmoe(x)
        assert out.shape == (B, T, D)
        assert 'lb_loss' in info
        assert 'group_weights' in info

    def test_forward_with_hex_tier(self, hmoe_hex):
        x = torch.randn(B, T, D)
        out, info = hmoe_hex(x)
        assert out.shape == (B, T, D)
        assert 'hex_tier_active' in info

    def test_no_nan(self, hmoe):
        x = torch.randn(B, T, D)
        out, info = hmoe(x)
        assert not torch.isnan(out).any()
        assert not torch.isnan(info['lb_loss'])

    def test_gradient_flow_full(self, hmoe):
        x = torch.randn(B, T, D, requires_grad=True)
        out, info = hmoe(x)
        loss = out.sum() + info['lb_loss']
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_lb_loss_contributes_to_grad(self, hmoe):
        """lb_loss should have gradient (not detached)."""
        x = torch.randn(B, T, D)
        out, info = hmoe(x)
        lb = info['lb_loss']
        assert lb.requires_grad, "lb_loss should have gradient"

    def test_crossing_info(self, hmoe):
        x = torch.randn(B, T, D)
        _, info = hmoe(x)
        assert 'crossing_alpha' in info
        alpha = info['crossing_alpha'].item()
        assert 0.0 < alpha < 1.0, f"crossing_alpha should be in (0,1), got {alpha}"

    def test_all_groups_active(self, hmoe):
        x = torch.randn(B, T, D)
        _, info = hmoe(x)
        for group in ["ABSTRACT", "CONCRETE", "DYNAMIC"]:
            assert f'lb_{group}' in info, f"Missing lb_{group} in info"

    def test_info_group_weights_detached(self, hmoe):
        x = torch.randn(B, T, D)
        _, info = hmoe(x)
        assert not info['group_weights'].requires_grad


# ═══════════════════════════════════════════════════════════════════════════════
# Stage progression
# ═══════════════════════════════════════════════════════════════════════════════

class TestStageProgression:
    def test_stages_defined(self):
        for stage in range(1, 7):
            assert stage in TRAINING_STAGES

    def test_set_moe_stage_freezes_correctly(self):
        cfg = HMoEConfig(d_model=D, use_multiscale=True, use_hex_tier=True,
                         hex_tier_d_ff_mult=1)
        hmoe = HierarchicalMoEFFN(cfg)

        # Stage 1: only micro_experts unfrozen
        set_moe_stage(hmoe, 1)
        for name, p in hmoe.named_parameters():
            if 'micro_experts' in name:
                assert p.requires_grad, f"Stage 1: {name} should be unfrozen"
            elif 'global_router' in name or 'group_routers' in name:
                assert not p.requires_grad, f"Stage 1: {name} should be frozen"

    def test_set_moe_stage_5_all_unfrozen(self):
        cfg = HMoEConfig(d_model=D, use_multiscale=True)
        hmoe = HierarchicalMoEFFN(cfg)
        set_moe_stage(hmoe, 5)
        for name, p in hmoe.named_parameters():
            assert p.requires_grad, f"Stage 5: {name} should be unfrozen"

    def test_get_stage_info(self):
        info = get_stage_info(3)
        assert 'GlobalRouter' in info
        assert 'Stage 3' in info

    def test_forward_works_after_stage_change(self):
        cfg = HMoEConfig(d_model=D)
        hmoe = HierarchicalMoEFFN(cfg)
        x = torch.randn(B, T, D)
        for stage in [1, 2, 3, 4, 5]:
            set_moe_stage(hmoe, stage)
            out, info = hmoe(x)
            assert out.shape == (B, T, D), f"Stage {stage} forward failed"
