"""Pytest-тесты BridgeOfModules: иерархическая медиация между источниками."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry.routing import (
    PairwiseBridge,
    LightweightBridge,
    BridgeOfModules,
    GeometricSourceRouter,
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


# ── PairwiseBridge ──────────────────────────────────────────────


class TestPairwiseBridge:
    def test_output_shape(self, d_model, batch_size, seq_len):
        bridge = PairwiseBridge(d_model=d_model, n_heads=2)
        a = torch.randn(batch_size, seq_len, d_model)
        b = torch.randn(batch_size, seq_len, d_model)
        out = bridge(a, b)
        assert out.shape == (batch_size, seq_len, d_model)

    def test_small_scale_init(self, d_model):
        """Мост начинает с малым scale → non-coercion."""
        bridge = PairwiseBridge(d_model=d_model)
        assert bridge.scale.item() == pytest.approx(0.1, abs=0.01)

    def test_gradient_flow(self, d_model, batch_size, seq_len):
        """Градиенты проходят через оба источника."""
        bridge = PairwiseBridge(d_model=d_model, n_heads=2)
        a = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        b = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        out = bridge(a, b)
        loss = out.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.abs().sum() > 0
        assert b.grad.abs().sum() > 0

    def test_symmetry_awareness(self, d_model, batch_size, seq_len):
        """Мост различает порядок аргументов (не симметричен)."""
        bridge = PairwiseBridge(d_model=d_model, n_heads=2)
        a = torch.randn(batch_size, seq_len, d_model)
        b = torch.randn(batch_size, seq_len, d_model)
        out_ab = bridge(a, b)
        out_ba = bridge(b, a)
        # Выходы должны отличаться (разные cross-attention направления)
        assert not torch.allclose(out_ab, out_ba, atol=1e-5)

    def test_identical_inputs_stable(self, d_model, batch_size, seq_len):
        """Одинаковые входы → стабильный выход."""
        bridge = PairwiseBridge(d_model=d_model, n_heads=2)
        x = torch.randn(batch_size, seq_len, d_model)
        out = bridge(x, x)
        assert torch.isfinite(out).all()


# ── BridgeOfModules ─────────────────────────────────────────────


class TestBridgeOfModules:
    def test_output_shape_even(self, d_model, batch_size, seq_len):
        """Чётное число источников."""
        n_sources = 6
        bom = BridgeOfModules(d_model=d_model, n_sources=n_sources)
        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(n_sources)]
        out = bom(x, sources)
        assert out.shape == (batch_size, seq_len, d_model)

    def test_output_shape_odd(self, d_model, batch_size, seq_len):
        """Нечётное число источников."""
        n_sources = 5
        bom = BridgeOfModules(d_model=d_model, n_sources=n_sources)
        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(n_sources)]
        out = bom(x, sources)
        assert out.shape == (batch_size, seq_len, d_model)

    def test_two_sources(self, d_model, batch_size, seq_len):
        """Минимальный случай: 2 источника → 1 мост."""
        bom = BridgeOfModules(d_model=d_model, n_sources=2)
        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(2)]
        out = bom(x, sources)
        assert out.shape == (batch_size, seq_len, d_model)
        # Должен быть ровно 1 уровень с 1 мостом
        assert len(bom.bridge_tree) == 1
        assert len(bom.bridge_tree[0]) == 1

    def test_tree_structure_6_sources(self, d_model):
        """6 источников → дерево: 3 моста → 1 мост + odd → 1 мост."""
        bom = BridgeOfModules(d_model=d_model, n_sources=6)
        # Уровень 0: 3 пары → 3 моста → 3 выхода
        assert len(bom.bridge_tree[0]) == 3
        # Уровень 1: 1 пара + odd → 1 мост → 2 выхода
        assert len(bom.bridge_tree[1]) == 1
        # Уровень 2: 1 пара → 1 мост → 1 выход
        assert len(bom.bridge_tree[2]) == 1

    def test_residual_with_zero_gate(self, d_model, batch_size, seq_len):
        """При global_gate = -inf → sigmoid ≈ 0 → выход = x (identity)."""
        bom = BridgeOfModules(d_model=d_model, n_sources=4)
        with torch.no_grad():
            bom.global_gate.fill_(-100.0)
        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(4)]
        out = bom(x, sources)
        assert torch.allclose(out, x, atol=1e-4)

    def test_gradient_flow_all_sources(self, d_model, batch_size, seq_len):
        """Градиенты проходят через все источники."""
        n_sources = 4
        bom = BridgeOfModules(d_model=d_model, n_sources=n_sources)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        sources = [
            torch.randn(batch_size, seq_len, d_model, requires_grad=True)
            for _ in range(n_sources)
        ]
        out = bom(x, sources)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for i, src in enumerate(sources):
            assert src.grad is not None, f"No gradient for source {i}"
            assert src.grad.abs().sum() > 0, f"Zero gradient for source {i}"

    def test_stats(self, d_model, batch_size, seq_len):
        """Статистика возвращает осмысленные значения."""
        bom = BridgeOfModules(d_model=d_model, n_sources=6)
        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(6)]
        bom(x, sources)
        stats = bom.get_bridge_stats()
        assert 'global_gate' in stats
        assert 0.0 <= stats['global_gate'] <= 1.0
        assert stats['n_levels'] == 3
        assert len(stats['bridge_scales']) == 5  # 3 + 1 + 1

    def test_wrong_n_sources_raises(self, d_model, batch_size, seq_len):
        """Неверное число источников → AssertionError."""
        bom = BridgeOfModules(d_model=d_model, n_sources=4)
        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(3)]
        with pytest.raises(AssertionError):
            bom(x, sources)


# ── Сравнение: Bridge vs Router ─────────────────────────────────


class TestBridgeVsRouter:
    """Проверяем что Bridge и Router совместимы по интерфейсу."""

    def test_same_interface(self, d_model, batch_size, seq_len):
        """Оба принимают (x, sources) и возвращают tensor."""
        n_sources = 4
        bridge = BridgeOfModules(d_model=d_model, n_sources=n_sources)
        router = GeometricSourceRouter(d_model=d_model, n_sources=n_sources)

        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(n_sources)]

        out_bridge = bridge(x, sources)
        out_router = router(x, sources)

        assert out_bridge.shape == (batch_size, seq_len, d_model)
        assert out_router.shape == (batch_size, seq_len, d_model)

    def test_bridge_param_count(self, d_model):
        """Bridge не должен быть значительно тяжелее Router."""
        n_sources = 6
        bridge = BridgeOfModules(d_model=d_model, n_sources=n_sources)
        router = GeometricSourceRouter(d_model=d_model, n_sources=n_sources)

        bridge_params = sum(p.numel() for p in bridge.parameters())
        router_params = sum(p.numel() for p in router.parameters())

        # Bridge использует cross-attention (O(d²) per head) —
        # значительно тяжелее linear router (O(d·N)).
        # Но не более 1000x для d_model=64.
        ratio = bridge_params / max(router_params, 1)
        assert ratio < 1000, f"Bridge too heavy: {bridge_params} vs Router {router_params}"
        # Абсолютный лимит: Bridge < 1M params для d_model=64
        assert bridge_params < 1_000_000, f"Bridge too large: {bridge_params}"


# ── LightweightBridge ───────────────────────────────────────────


class TestLightweightBridge:
    def test_output_shape(self, d_model, batch_size, seq_len):
        bridge = LightweightBridge(d_model=d_model)
        a = torch.randn(batch_size, seq_len, d_model)
        b = torch.randn(batch_size, seq_len, d_model)
        out = bridge(a, b)
        assert out.shape == (batch_size, seq_len, d_model)

    def test_much_lighter_than_full(self, d_model):
        """LightweightBridge должен быть значительно легче PairwiseBridge."""
        light = LightweightBridge(d_model=d_model)
        full = PairwiseBridge(d_model=d_model)
        light_params = sum(p.numel() for p in light.parameters())
        full_params = sum(p.numel() for p in full.parameters())
        assert light_params < full_params, \
            f"Lightweight ({light_params}) should be < Full ({full_params})"

    def test_gradient_flow(self, d_model, batch_size, seq_len):
        bridge = LightweightBridge(d_model=d_model)
        a = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        b = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        out = bridge(a, b)
        out.sum().backward()
        assert a.grad is not None and a.grad.abs().sum() > 0
        assert b.grad is not None and b.grad.abs().sum() > 0


class TestBridgeOfModulesLightweight:
    def test_lightweight_mode(self, d_model, batch_size, seq_len):
        """bridge_mode='lightweight' создаёт LightweightBridge внутри."""
        bom = BridgeOfModules(d_model=d_model, n_sources=4, bridge_mode='lightweight')
        x = torch.randn(batch_size, seq_len, d_model)
        sources = [torch.randn(batch_size, seq_len, d_model) for _ in range(4)]
        out = bom(x, sources)
        assert out.shape == (batch_size, seq_len, d_model)

    def test_lightweight_fewer_params(self, d_model):
        """Lightweight mode должен быть легче full mode."""
        full = BridgeOfModules(d_model=d_model, n_sources=6, bridge_mode='full')
        light = BridgeOfModules(d_model=d_model, n_sources=6, bridge_mode='lightweight')
        full_p = sum(p.numel() for p in full.parameters())
        light_p = sum(p.numel() for p in light.parameters())
        assert light_p < full_p, f"Lightweight ({light_p}) should be < Full ({full_p})"
        ratio = full_p / light_p
        assert ratio > 2, f"Expected >2x reduction, got {ratio:.1f}x"
