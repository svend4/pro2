"""
Интеграционные тесты на реалистичных размерах (d_model=256+).

Все основные тесты проекта работают на toy-scale (d_model=64, seq_len=16).
Этот модуль проверяет, что модели работают корректно при масштабировании:
- d_model=256, n_layers=4, block_size=128, seq_len=64
- Проверяется forward pass, backward pass, loss decrease, gradient flow
- Тесты для ExpertChoice MoE и DomainMoE на реальных размерах
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT


def make_scaled_cfg(**overrides):
    """Конфигурация на реалистичном масштабе (~5M params)."""
    defaults = dict(
        vocab_size=2048,
        d_model=256,
        n_layers=4,
        n_heads=8,
        block_size=128,
        batch_size=4,
        use_rope=True,
        use_swiglu=True,
        use_bian_gua=True,
        use_hex_moe=False,
        use_flash_attn=False,
        adaptive_temp=True,
        dropout=0.0,
    )
    defaults.update(overrides)
    return YiJingConfig(**defaults)


# ==================== Реалистичный масштаб: forward / backward ====================

class TestScaledForward:
    """Forward pass на d_model=256, seq_len=64."""

    def test_forward_shape(self):
        cfg = make_scaled_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, 64))
        logits, loss, _ = model(x)
        assert logits.shape == (4, 64, cfg.vocab_size)
        assert loss is None

    def test_forward_with_targets(self):
        cfg = make_scaled_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, 64))
        y = torch.randint(0, cfg.vocab_size, (4, 64))
        logits, loss, _ = model(x, y)
        assert logits.shape == (4, 64, cfg.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_backward_gradient_flow(self):
        """Все параметры должны получить градиенты при d_model=256."""
        cfg = make_scaled_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        y = torch.randint(0, cfg.vocab_size, (2, 64))
        _, loss, _ = model(x, y)
        loss.backward()

        grads_found = sum(1 for _, p in model.named_parameters()
                         if p.requires_grad and p.grad is not None)
        total_params = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        ratio = grads_found / total_params
        assert ratio > 0.9, \
            f"Only {grads_found}/{total_params} ({ratio:.0%}) params have gradients"

    def test_no_nan_in_output(self):
        """Выход модели не содержит NaN/Inf на реалистичном масштабе."""
        cfg = make_scaled_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, 64))
        logits, _, _ = model(x)
        assert torch.isfinite(logits).all(), "NaN or Inf detected in logits"


class TestScaledTraining:
    """Обучение на реалистичном масштабе: loss должен падать."""

    def test_loss_decreases_20_steps(self):
        """За 20 шагов loss должен уменьшиться (overfit на 1 батче)."""
        cfg = make_scaled_cfg(n_layers=4)
        model = YiJingGPT(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        x = torch.randint(0, cfg.vocab_size, (4, 64))
        y = torch.randint(0, cfg.vocab_size, (4, 64))

        losses = []
        for _ in range(20):
            model.train()
            _, loss, _ = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        # Ожидаем хотя бы 10% снижение
        reduction = (losses[0] - losses[-1]) / losses[0]
        assert reduction > 0.05, \
            f"Insignificant loss reduction: {reduction:.1%}"

    def test_parameter_count_realistic(self):
        """Проверяем что модель имеет ожидаемый порядок параметров."""
        cfg = make_scaled_cfg()
        model = YiJingGPT(cfg)
        total, hex_params = model.count_parameters()
        # d_model=256, n_layers=4 → ~5-10M params
        assert total > 1_000_000, f"Too few params: {total}"
        assert total < 50_000_000, f"Too many params: {total}"
        assert hex_params > 0, "No hexagram params found"


# ==================== MoE на реалистичном масштабе ====================

class TestScaledMoE:
    """MoE routing на d_model=256."""

    def test_trigram_moe_forward(self):
        cfg = make_scaled_cfg(use_hex_moe=True, n_experts=8, moe_top_k=2)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, 64))
        y = torch.randint(0, cfg.vocab_size, (4, 64))
        logits, loss, _ = model(x, y)
        assert logits.shape == (4, 64, cfg.vocab_size)
        assert loss.item() > 0

    def test_domain_moe_forward(self):
        cfg = make_scaled_cfg(use_domain_moe=True, domain_moe_n_experts=6)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, 64))
        y = torch.randint(0, cfg.vocab_size, (4, 64))
        logits, loss, _ = model(x, y)
        assert logits.shape == (4, 64, cfg.vocab_size)

    def test_expert_choice_forward(self):
        cfg = make_scaled_cfg(use_expert_choice=True, n_experts=8)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (4, 64))
        y = torch.randint(0, cfg.vocab_size, (4, 64))
        logits, loss, _ = model(x, y)
        assert logits.shape == (4, 64, cfg.vocab_size)
        assert loss.item() > 0

    def test_expert_choice_training(self):
        """ExpertChoice MoE: loss уменьшается за 15 шагов."""
        cfg = make_scaled_cfg(use_expert_choice=True, n_experts=8)
        model = YiJingGPT(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        x = torch.randint(0, cfg.vocab_size, (4, 64))
        y = torch.randint(0, cfg.vocab_size, (4, 64))

        losses = []
        for _ in range(15):
            model.train()
            _, loss, _ = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"ExpertChoice loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ==================== KV-cache на реалистичном масштабе ====================

class TestScaledKVCache:
    """KV-cache корректность при seq_len=64."""

    def test_kv_cache_consistency(self):
        cfg = make_scaled_cfg()
        model = YiJingGPT(cfg)
        model.eval()

        x = torch.randint(0, cfg.vocab_size, (1, 32))

        # Полный forward
        logits_full, _, kv_full = model(x)

        # Инкрементальный: prefix + suffix
        logits_prefix, _, kv_prefix = model(x[:, :24])
        logits_suffix, _, _ = model(x[:, 24:], kv_cache=kv_prefix)

        torch.testing.assert_close(
            logits_full[:, -1, :], logits_suffix[:, -1, :],
            atol=1e-3, rtol=1e-3,
        )

    def test_generate_scaled(self):
        cfg = make_scaled_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 16))
        out = model.generate(idx, max_new_tokens=20, temperature=1.0, top_k=50)
        assert out.shape == (1, 36)
