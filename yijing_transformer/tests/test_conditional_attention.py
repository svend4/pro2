"""
Тесты для 8 условных attention-паттернов геометрии.

Эти паттерны активируются через конфигурационные флаги, но не включены по умолчанию.
Каждый тест:
1. Создаёт YiJingGPT с соответствующим конфиг-флагом
2. Проверяет forward pass
3. Проверяет backward (gradient flow)
4. Проверяет что loss конечен

Покрываемые паттерны:
- TriangularAttentionBias (use_triangular_bias)
- PalaceAttention (use_palace_attention)
- QuadrantAttention (use_quadrant_attention)
- RecursiveCubeAttention (use_recursive_cube)
- WeavingLoomArchitecture (use_weaving_loom)
- BidirectionalTriangularAttention (use_bidirectional_tri)
- CubeDiagonalAttention (use_cube_diagonal)
- MobiusAttentionPattern (use_mobius_bias)
- HeisenbergAttention (use_heisenberg_attention)
- FlowerOfLifeGAT (use_flower_gat)
- PrivilegedAxisAttention (use_privileged_axis)
- CubicAttentionBias (use_cubic_bias)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT


def make_cfg(**overrides):
    defaults = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=8,
        block_size=32, batch_size=2, use_rope=True, use_swiglu=True,
        use_bian_gua=True, use_hex_moe=False, use_flash_attn=False,
        adaptive_temp=True, dropout=0.0,
    )
    defaults.update(overrides)
    return YiJingConfig(**defaults)


def _run_forward_backward(cfg, seq_len=16):
    """Общий helper: forward + backward + gradient check."""
    model = YiJingGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, seq_len))
    y = torch.randint(0, cfg.vocab_size, (2, seq_len))

    model.train()
    logits, loss, _ = model(x, y)

    assert logits.shape == (2, seq_len, cfg.vocab_size), \
        f"Wrong shape: {logits.shape}"
    assert loss is not None
    assert torch.isfinite(torch.tensor(loss.item())), f"Loss is not finite: {loss.item()}"

    loss.backward()

    grads_found = sum(1 for _, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None)
    total_params = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    ratio = grads_found / total_params
    assert ratio > 0.85, \
        f"Only {grads_found}/{total_params} ({ratio:.0%}) params have gradients"

    return model, loss


# ==================== Bias-based паттерны (добавляют bias к attention scores) ====================

class TestTriangularBias:
    """TriangularAttentionBias (Андреев): треугольное расстояние → attention bias."""

    def test_forward_backward(self):
        cfg = make_cfg(use_triangular_bias=True)
        _run_forward_backward(cfg)

    def test_loss_decreases(self):
        cfg = make_cfg(use_triangular_bias=True)
        model = YiJingGPT(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        losses = []
        for _ in range(15):
            model.train()
            _, loss, _ = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


class TestMobiusBias:
    """MobiusAttentionPattern (Беляев 6.3): Мёбиусов attention bias."""

    def test_forward_backward(self):
        cfg = make_cfg(use_mobius_bias=True)
        _run_forward_backward(cfg)


class TestCubicBias:
    """CubicAttentionBias (Касаткин): 3D distance bias в 4x4x4 кубе."""

    def test_forward_backward(self):
        cfg = make_cfg(use_cubic_bias=True)
        _run_forward_backward(cfg)


class TestPrivilegedAxis:
    """PrivilegedAxisAttention (Касаткин 4.1): привилегированная ось bias."""

    def test_forward_backward(self):
        cfg = make_cfg(use_privileged_axis=True)
        _run_forward_backward(cfg)


# ==================== Full attention replacement паттерны ====================

class TestPalaceAttention:
    """PalaceAttention (Склярова): block-sparse по 8 дворцам."""

    def test_forward_backward(self):
        cfg = make_cfg(use_palace_attention=True)
        _run_forward_backward(cfg)


class TestQuadrantAttention:
    """QuadrantAttention (Беляев): 4-квадрантный attention splitting."""

    def test_forward_backward(self):
        cfg = make_cfg(use_quadrant_attention=True)
        _run_forward_backward(cfg)


class TestRecursiveCube:
    """RecursiveCubeAttention (Беляев 6.5): куб-в-кубе рекурсивный."""

    def test_forward_backward(self):
        cfg = make_cfg(use_recursive_cube=True)
        _run_forward_backward(cfg)


class TestWeavingLoom:
    """WeavingLoomArchitecture (Беляев 6.8): 4-уровневый ткацкий станок."""

    def test_forward_backward(self):
        cfg = make_cfg(use_weaving_loom=True, weaving_max_level=3)
        _run_forward_backward(cfg)


class TestBidirectionalTriangular:
    """BidirectionalTriangularAttention (Андреев 3.3): upper + lower triangular."""

    def test_forward_backward(self):
        cfg = make_cfg(use_bidirectional_tri=True)
        _run_forward_backward(cfg)


class TestCubeDiagonal:
    """CubeDiagonalAttention (Касаткин 4.2): 4 типа диагоналей куба."""

    def test_forward_backward(self):
        cfg = make_cfg(use_cube_diagonal=True)
        _run_forward_backward(cfg)


class TestHeisenbergAttention:
    """HeisenbergAttention (Беляев 6.1): uncertainty-based temperature."""

    def test_forward_backward(self):
        cfg = make_cfg(use_heisenberg_attention=True)
        _run_forward_backward(cfg)


class TestFlowerOfLifeGAT:
    """FlowerOfLifeGAT (Беляев 6.6): 7-node Flower of Life graph attention."""

    def test_forward_backward(self):
        cfg = make_cfg(use_flower_gat=True)
        _run_forward_backward(cfg)


# ==================== Комбинации паттернов ====================

class TestCombinedPatterns:
    """Проверяем что несколько паттернов можно включить одновременно."""

    def test_heisenberg_plus_triangular(self):
        cfg = make_cfg(
            use_heisenberg_attention=True,
            use_triangular_bias=True,
        )
        _run_forward_backward(cfg)

    def test_flower_plus_mobius_plus_weaving(self):
        cfg = make_cfg(
            use_flower_gat=True,
            use_mobius_bias=True,
            use_weaving_loom=True,
        )
        _run_forward_backward(cfg)

    def test_all_biases_together(self):
        cfg = make_cfg(
            use_triangular_bias=True,
            use_mobius_bias=True,
            use_cubic_bias=True,
        )
        _run_forward_backward(cfg)

    def test_palace_plus_heisenberg(self):
        cfg = make_cfg(
            use_palace_attention=True,
            use_heisenberg_attention=True,
        )
        _run_forward_backward(cfg)


# ==================== Config examples (документация через тесты) ====================

class TestConfigExamples:
    """Примеры конфигураций, демонстрирующие использование attention паттернов.

    Каждый тест = рабочий пример конфигурации для пользователя.
    """

    def test_geometric_analysis_config(self):
        """Конфиг для геометрического анализа текста.

        Включает HeisenbergAttention (uncertainty) + FlowerOfLife (graph) +
        TriangularBias (distance).

        Использование:
            cfg = YiJingConfig.tiny(
                use_heisenberg_attention=True,
                use_flower_gat=True,
                use_triangular_bias=True,
            )
        """
        cfg = YiJingConfig.tiny(
            vocab_size=128,
            use_heisenberg_attention=True,
            use_flower_gat=True,
            use_triangular_bias=True,
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, 128, (1, 16))
        logits, _, _ = model(x)
        assert logits.shape == (1, 16, 128)

    def test_topological_moe_config(self):
        """Конфиг для топологической MoE с Expert Choice.

        Expert Choice routing (perfect load balance) + Weaving Loom hierarchy +
        Möbius topology.

        Использование:
            cfg = YiJingConfig.tiny(
                use_expert_choice=True,
                n_experts=8,
                use_weaving_loom=True,
                use_mobius_bias=True,
            )
        """
        cfg = YiJingConfig.tiny(
            vocab_size=128,
            use_expert_choice=True,
            n_experts=8,
            use_weaving_loom=True,
            use_mobius_bias=True,
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, 128, (1, 16))
        logits, _, _ = model(x)
        assert logits.shape == (1, 16, 128)

    def test_kasatkin_3d_config(self):
        """Конфиг с 3D-геометрией Касаткина.

        CubicBias (4x4x4) + PrivilegedAxis + CubeDiagonal.

        Использование:
            cfg = YiJingConfig.tiny(
                use_cubic_bias=True,
                use_privileged_axis=True,
                use_cube_diagonal=True,
            )
        """
        cfg = YiJingConfig.tiny(
            vocab_size=128,
            use_cubic_bias=True,
            use_privileged_axis=True,
            use_cube_diagonal=True,
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, 128, (1, 16))
        logits, _, _ = model(x)
        assert logits.shape == (1, 16, 128)

    def test_sklyarova_palace_config(self):
        """Конфиг с дворцовой архитектурой Скляровой.

        PalaceAttention (block-sparse по 8 дворцам) + GraduatedBianGua.

        Использование:
            cfg = YiJingConfig.tiny(
                use_palace_attention=True,
                use_graduated_biangua=True,
            )
        """
        cfg = YiJingConfig.tiny(
            vocab_size=128,
            use_palace_attention=True,
            use_graduated_biangua=True,
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, 128, (1, 16))
        logits, _, _ = model(x)
        assert logits.shape == (1, 16, 128)
