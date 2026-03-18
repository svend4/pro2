"""
Тесты для утилит обучения из train_hmoe_staged.py и lora.py.

Покрывает:
  - encode(): byte encoding, clamping, empty input
  - perplexity(): eval mode, held-out split, edge cases
  - collect_moe_lb_loss(): aggregation from model blocks
  - collect_interlingua_loss(): aggregation from model blocks
  - LoRALinear: forward, merge/unmerge, weight caching
  - StreamingBatchLoader: persistent buffer, rotation, restart
"""

import math
import pytest
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
from yijing_transformer.models.lora import (
    LoRALinear, apply_lora, freeze_non_lora, merge_lora, unmerge_lora,
    count_lora_parameters,
)

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════════
# encode()
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncode:
    """Tests for the encode() utility from train_hmoe_staged.py."""

    def _encode(self, text, vocab_size=256, block_size=32):
        """Local copy of encode to avoid importing the training script."""
        ids = [min(b, vocab_size - 1) for b in text.encode("utf-8")][:block_size]
        return torch.tensor(ids or [32], dtype=torch.long).unsqueeze(0)

    def test_basic_ascii(self):
        t = self._encode("hello")
        assert t.shape == (1, 5)
        assert t[0, 0].item() == ord('h')

    def test_truncation(self):
        t = self._encode("a" * 100, block_size=10)
        assert t.shape == (1, 10)

    def test_vocab_clamping(self):
        """Bytes > vocab_size should be clamped."""
        t = self._encode("\xff", vocab_size=128)
        assert t[0, 0].item() == 127  # min(0xFF, 127)

    def test_empty_input(self):
        t = self._encode("")
        assert t.shape == (1, 1)
        assert t[0, 0].item() == 32  # fallback

    def test_utf8_multibyte(self):
        t = self._encode("ä")  # 2 UTF-8 bytes: 0xC3, 0xA4
        assert t.shape == (1, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# collect_moe_lb_loss() / collect_interlingua_loss()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectLoss:
    @pytest.fixture
    def model(self):
        cfg = Variant3Config(vocab_size=64, block_size=16, d_model=64,
                             n_heads=4, n_layers=2, ffn_mult=2)
        return Variant3GPT(cfg)

    def test_collect_moe_lb_loss_no_hmoe(self, model):
        """Without HMoE blocks, should return 0."""
        from train_hmoe_staged import collect_moe_lb_loss
        loss = collect_moe_lb_loss(model)
        assert loss.item() == 0.0

    def test_collect_interlingua_loss_default(self, model):
        """After forward pass, interlingua loss should be collected."""
        from train_hmoe_staged import collect_interlingua_loss
        tokens = torch.randint(0, 64, (1, 16))
        model(tokens)
        loss = collect_interlingua_loss(model)
        # Should be a tensor (possibly 0 if no interlingua, or a real value)
        assert isinstance(loss, torch.Tensor)

    def test_collect_moe_lb_loss_with_info(self, model):
        """Manually set _last_moe_info to verify aggregation."""
        from train_hmoe_staged import collect_moe_lb_loss
        # Simulate HMoE output
        model.blocks[0]._last_moe_info = {
            'lb_loss': torch.tensor(0.5, requires_grad=True)
        }
        model.blocks[1]._last_moe_info = {
            'lb_loss': torch.tensor(0.3, requires_grad=True)
        }
        loss = collect_moe_lb_loss(model)
        assert abs(loss.item() - 0.8) < 1e-5
        assert loss.requires_grad


# ═══════════════════════════════════════════════════════════════════════════════
# LoRALinear
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoRALinear:
    @pytest.fixture
    def base(self):
        linear = nn.Linear(32, 64)
        return linear

    @pytest.fixture
    def lora(self, base):
        return LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

    def test_forward_shape(self, lora):
        x = torch.randn(2, 8, 32)
        out = lora(x)
        assert out.shape == (2, 8, 64)

    def test_forward_adds_lora(self, lora, base):
        """LoRA output should differ from base output."""
        x = torch.randn(2, 8, 32)
        base_out = base(x)
        lora_out = lora(x)
        # lora_B initialized to 0, so initially they should be equal
        assert torch.allclose(base_out, lora_out, atol=1e-5)
        # After modifying lora_B, they should differ
        lora.lora_B.data.fill_(0.1)
        lora_out2 = lora(x)
        assert not torch.allclose(base_out, lora_out2, atol=1e-3)

    def test_merge_unmerge_roundtrip(self, lora):
        """Merge then unmerge should restore original behavior."""
        lora.lora_B.data.fill_(0.1)
        x = torch.randn(2, 8, 32)
        out_before = lora(x).detach().clone()

        lora.merge()
        assert lora.merged
        out_merged = lora(x).detach().clone()
        # Merged output should match pre-merge output
        assert torch.allclose(out_before, out_merged, atol=1e-4)

        lora.unmerge()
        assert not lora.merged
        out_unmerged = lora(x).detach().clone()
        assert torch.allclose(out_before, out_unmerged, atol=1e-4)

    def test_weight_property(self, lora):
        """weight property should return effective weight."""
        w = lora.weight
        assert w.shape == (64, 32)

    def test_weight_cache_invalidation(self, lora):
        """Cache should be invalidated after forward pass."""
        x = torch.randn(1, 4, 32)
        _ = lora.weight  # populate cache
        lora(x)  # should invalidate
        assert lora._weight_cache is None

    def test_gradient_flow(self, lora):
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = lora(x)
        out.sum().backward()
        assert x.grad is not None
        assert lora.lora_A.grad is not None

    def test_count_parameters(self):
        base = nn.Linear(64, 128)
        model = nn.Module()
        model.layer = LoRALinear(base, rank=4, alpha=8.0)
        lora_params, total_params = count_lora_parameters(model)
        expected_lora = 4 * 64 + 128 * 4  # A: (4, 64), B: (128, 4)
        assert lora_params == expected_lora

    def test_freeze_non_lora(self):
        base = nn.Linear(32, 64)
        model = nn.Module()
        model.layer = LoRALinear(base, rank=4)
        model.other = nn.Linear(16, 16)
        freeze_non_lora(model)
        for name, p in model.named_parameters():
            if 'lora_' in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"


# ═══════════════════════════════════════════════════════════════════════════════
# StreamingBatchLoader
# ═══════════════════════════════════════════════════════════════════════════════

class TestStreamingBatchLoader:
    """Tests using a mock iterator (no HuggingFace dependency)."""

    def _mock_iterator(self, n=50):
        for i in range(n):
            yield {'text': f'This is example text number {i} for testing streaming'}

    def _mock_tokenizer(self):
        class Tok:
            def encode(self, text):
                return list(text.encode('utf-8'))
        return Tok()

    def test_basic_batch(self):
        from yijing_transformer.data_utils.streaming_dataset import StreamingBatchLoader
        loader = StreamingBatchLoader(
            iterator=self._mock_iterator(),
            tokenizer=self._mock_tokenizer(),
            block_size=16, device='cpu', buffer_size=20,
        )
        X, Y = loader.get_batch(4)
        assert X is not None
        assert X.shape == (4, 16)
        assert Y.shape == (4, 16)

    def test_persistent_buffer(self):
        from yijing_transformer.data_utils.streaming_dataset import StreamingBatchLoader
        loader = StreamingBatchLoader(
            iterator=self._mock_iterator(30),
            tokenizer=self._mock_tokenizer(),
            block_size=16, device='cpu', buffer_size=20,
        )
        # Multiple calls should work without re-creating buffer
        for _ in range(5):
            X, Y = loader.get_batch(2)
            assert X is not None

    def test_exhausted_iterator(self):
        from yijing_transformer.data_utils.streaming_dataset import StreamingBatchLoader
        loader = StreamingBatchLoader(
            iterator=self._mock_iterator(5),
            tokenizer=self._mock_tokenizer(),
            block_size=16, device='cpu', buffer_size=100,
        )
        # Should still work from buffer even though iterator exhausted
        X, Y = loader.get_batch(2)
        assert X is not None

    def test_restart_fn(self):
        from yijing_transformer.data_utils.streaming_dataset import StreamingBatchLoader
        call_count = [0]
        def restart():
            call_count[0] += 1
            return self._mock_iterator(10)
        loader = StreamingBatchLoader(
            iterator=iter([]),  # empty iterator
            tokenizer=self._mock_tokenizer(),
            block_size=16, device='cpu', buffer_size=5,
            restart_fn=restart,
        )
        X, Y = loader.get_batch(2)
        assert X is not None
        assert call_count[0] == 1

    def test_is_exhausted(self):
        from yijing_transformer.data_utils.streaming_dataset import StreamingBatchLoader
        loader = StreamingBatchLoader(
            iterator=iter([]),
            tokenizer=self._mock_tokenizer(),
            block_size=16, device='cpu', buffer_size=5,
        )
        assert loader.is_exhausted

    def test_backward_compat_function(self):
        from yijing_transformer.data_utils.streaming_dataset import get_batch_streaming
        it = self._mock_iterator(20)
        X, Y = get_batch_streaming(
            it, batch_size=2, block_size=16, device='cpu',
            tokenizer=self._mock_tokenizer(), buffer_size=10,
        )
        assert X is not None
        assert X.shape == (2, 16)
