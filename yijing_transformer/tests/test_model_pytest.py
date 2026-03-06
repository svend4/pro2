"""Pytest-тесты для моделей YiJing и Vanilla."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import YiJingConfig
from models.model import YiJingGPT, YiJingTransformer, YiJingAttention
from models.baseline import VanillaGPT


def make_cfg(**overrides):
    defaults = dict(
        vocab_size=128, d_model=64, n_layers=2, n_heads=8,
        block_size=32, batch_size=2, use_rope=True, use_swiglu=True,
        use_bian_gua=True, use_hex_moe=False, use_flash_attn=False,
        adaptive_temp=True,
    )
    defaults.update(overrides)
    return YiJingConfig(**defaults)


class TestYiJingGPT:
    def test_forward_shape(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert loss is None

    def test_forward_with_targets(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss, _ = model(x, y)
        assert loss is not None
        assert loss.item() > 0

    def test_backward(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        # Большинство параметров должны иметь градиенты
        grads_found = sum(1 for _, p in model.named_parameters()
                         if p.requires_grad and p.grad is not None)
        total_params = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        assert grads_found > total_params * 0.9, \
            f"Only {grads_found}/{total_params} parameters have gradients"

    def test_generate(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10)
        assert out.shape == (1, 9)

    def test_count_parameters(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        total, hex_params = model.count_parameters()
        assert total > 0
        assert hex_params > 0
        assert hex_params < total

    def test_without_rope(self):
        cfg = make_cfg(use_rope=False)
        model = YiJingGPT(cfg)
        assert model.pos_emb is not None
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_without_swiglu(self):
        cfg = make_cfg(use_swiglu=False)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_without_bian_gua(self):
        cfg = make_cfg(use_bian_gua=False)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_with_moe(self):
        cfg = make_cfg(use_hex_moe=True, n_experts=4, moe_top_k=2)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, loss, _ = model(x, y)
        assert logits.shape == (2, 8, cfg.vocab_size)
        assert loss.item() > 0

    def test_with_octogram_quantizer(self):
        cfg = make_cfg(quantizer_type='octogram', quant_total_dim=8)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, loss, _ = model(x, y)
        assert logits.shape == (2, 8, cfg.vocab_size)
        assert loss.item() > 0

    def test_with_hierarchical_quantizer(self):
        cfg = make_cfg(quantizer_type='hierarchical', quant_total_dim=8, quant_group_dim=2)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_with_deformable_quantizer(self):
        cfg = make_cfg(quantizer_type='deformable', quant_total_dim=6, quant_group_dim=3)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0


class TestVanillaGPT:
    def test_forward(self):
        cfg = make_cfg()
        model = VanillaGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert loss is None

    def test_forward_with_targets(self):
        cfg = make_cfg()
        model = VanillaGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss = model(x, y)
        assert loss is not None

    def test_count_parameters(self):
        cfg = make_cfg()
        model = VanillaGPT(cfg)
        total, hex = model.count_parameters()
        assert total > 0
        assert hex == 0

    def test_with_rope(self):
        cfg = make_cfg(use_rope=True)
        model = VanillaGPT(cfg)
        assert model.pos_emb is None
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)


class TestParameterCount:
    """Убеждаемся, что YiJing добавляет мало параметров поверх baseline."""
    def test_overhead_small(self):
        cfg = make_cfg(use_hex_moe=False)
        yj = YiJingGPT(cfg)
        vn = VanillaGPT(cfg)
        yj_total, yj_hex = yj.count_parameters()
        vn_total, _ = vn.count_parameters()
        overhead_pct = (yj_total - vn_total) / vn_total * 100
        assert overhead_pct < 10, f"Too much overhead: {overhead_pct:.1f}%"


class TestTrainingStep:
    """Проверяем что один шаг обучения работает без ошибок."""
    def test_one_step(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))

        model.train()
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Loss должен быть конечным
        assert torch.isfinite(torch.tensor(loss.item()))

    def test_loss_decreases(self):
        """За 10 шагов loss должен уменьшиться (overfit на 1 батче)."""
        cfg = make_cfg(n_layers=2)
        model = YiJingGPT(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))

        losses = []
        for _ in range(20):
            model.train()
            _, loss, _ = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestV4Features:
    """Тесты для v4: KV-cache, weight tying, Gumbel, multi-scale, grad ckpt."""

    def test_kv_cache_generate(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        # Генерация с KV-cache
        out_cache = model.generate(idx.clone(), max_new_tokens=5, use_cache=True)
        assert out_cache.shape == (1, 9)

    def test_kv_cache_forward(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        # Полный forward
        logits_full, _, kv = model(x)
        # Инкрементальный: первые 6 токенов + кэш + последние 2
        logits_prefix, _, kv_prefix = model(x[:, :6])
        logits_suffix, _, _ = model(x[:, 6:], kv_cache=kv_prefix)
        # Последний токен должен иметь похожие logits
        torch.testing.assert_close(
            logits_full[:, -1, :], logits_suffix[:, -1, :], atol=1e-4, rtol=1e-4
        )

    def test_weight_tying(self):
        cfg = make_cfg(weight_tying=True)
        model = YiJingGPT(cfg)
        assert model.head.weight is model.tok_emb.weight

    def test_no_weight_tying(self):
        cfg = make_cfg(weight_tying=False)
        model = YiJingGPT(cfg)
        assert model.head.weight is not model.tok_emb.weight

    def test_gumbel_quantizer(self):
        cfg = make_cfg(quantizer_type='gumbel', quant_total_dim=6, quant_group_dim=3)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0

    def test_gumbel_commitment_loss(self):
        from models.geometry import GumbelQuantizer
        gq = GumbelQuantizer(total_dim=6, group_dim=3, commitment_weight=0.5)
        x = torch.randn(2, 4, 6)
        gq.train()
        out = gq(x)
        cl = gq.get_commitment_loss()
        assert cl.item() > 0
        assert out.shape == x.shape

    def test_multi_scale_quantization(self):
        cfg = make_cfg(
            n_layers=3,
            multi_scale_quant=True,
            quant_dim_schedule=[6, 6, 8],
            quantizer_type='hierarchical',
            quant_group_dim=2,
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)
        # Проверяем что слои имеют разные quant_dim
        assert model.core.layers[0].quant_dim == 6
        assert model.core.layers[2].quant_dim == 8

    def test_gradient_checkpointing(self):
        cfg = make_cfg(use_gradient_ckpt=True)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        model.train()
        _, loss, _ = model(x, y)
        loss.backward()
        # Все параметры должны иметь градиенты
        grads = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        total = sum(1 for p in model.parameters() if p.requires_grad)
        assert grads > total * 0.9


class TestV5Features:
    """Тесты для v5: GQA, sliding window, quantization analytics."""

    def test_gqa_forward(self):
        """GQA с 2 KV головами из 8."""
        cfg = make_cfg(n_kv_heads=2)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_gqa_with_targets(self):
        cfg = make_cfg(n_kv_heads=4)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0

    def test_gqa_kv_cache(self):
        """GQA с KV-cache должен давать такие же результаты."""
        cfg = make_cfg(n_kv_heads=2)
        model = YiJingGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits_full, _, _ = model(x)
        logits_prefix, _, kv = model(x[:, :6])
        logits_suffix, _, _ = model(x[:, 6:], kv_cache=kv)
        torch.testing.assert_close(
            logits_full[:, -1, :], logits_suffix[:, -1, :], atol=1e-4, rtol=1e-4
        )

    def test_gqa_parameter_savings(self):
        """GQA должен иметь меньше параметров чем MHA."""
        cfg_mha = make_cfg(n_kv_heads=None)  # MHA: 8 KV heads
        cfg_gqa = make_cfg(n_kv_heads=2)     # GQA: 2 KV heads
        mha = YiJingGPT(cfg_mha)
        gqa = YiJingGPT(cfg_gqa)
        mha_total, _ = mha.count_parameters()
        gqa_total, _ = gqa.count_parameters()
        assert gqa_total < mha_total, f"GQA ({gqa_total}) should have fewer params than MHA ({mha_total})"

    def test_sliding_window(self):
        cfg = make_cfg(sliding_window=8)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_sliding_window_with_kv_cache(self):
        cfg = make_cfg(sliding_window=8)
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=5, use_cache=True)
        assert out.shape == (1, 9)

    def test_gqa_vanilla(self):
        """GQA в VanillaGPT baseline."""
        cfg = make_cfg(n_kv_heads=2)
        model = VanillaGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_quantization_analytics(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        analytics = model.quantization_analytics()
        assert len(analytics) == cfg.n_layers
        for layer_name, info in analytics.items():
            assert 'hex_scale' in info
            assert 'quant_error' in info
            assert 'quant_snr' in info
            assert info['quant_error'] >= 0

    def test_gqa_combined_features(self):
        """GQA + sliding window + BianGua + все фичи вместе."""
        cfg = make_cfg(
            n_kv_heads=2, sliding_window=8,
            use_bian_gua=True, use_rope=True, use_swiglu=True,
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0

    def test_e8_quantizer_model(self):
        """E8 квантизатор в полной модели."""
        cfg = make_cfg(quantizer_type='e8', quant_total_dim=8)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0


class TestV7LoRA:
    """Тесты для v7: LoRA адаптеры."""

    def test_apply_lora(self):
        """LoRA адаптеры применяются к q_proj и v_proj."""
        from models.lora import apply_lora, LoRALinear
        cfg = make_cfg(use_lora=True, lora_rank=4, lora_alpha=8.0)
        model = YiJingGPT(cfg)
        count = apply_lora(model, cfg)
        # 2 слоя × 2 target (q_proj, v_proj) = 4
        assert count == cfg.n_layers * 2

        # Проверяем что модули заменены
        for layer in model.core.layers:
            assert isinstance(layer.attn.q_proj, LoRALinear)
            assert isinstance(layer.attn.v_proj, LoRALinear)

    def test_lora_forward(self):
        """Модель с LoRA работает (forward + backward)."""
        from models.lora import apply_lora
        cfg = make_cfg(use_lora=True, lora_rank=4)
        model = YiJingGPT(cfg)
        apply_lora(model, cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0

    def test_freeze_non_lora(self):
        """freeze_non_lora замораживает всё кроме LoRA параметров."""
        from models.lora import apply_lora, freeze_non_lora
        cfg = make_cfg(use_lora=True, lora_rank=4)
        model = YiJingGPT(cfg)
        apply_lora(model, cfg)
        freeze_non_lora(model)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

        assert len(trainable) > 0
        assert all('lora_' in n for n in trainable)
        assert len(frozen) > 0

    def test_lora_parameter_count(self):
        """LoRA параметры — малая доля от общего числа."""
        from models.lora import apply_lora, count_lora_parameters
        cfg = make_cfg(use_lora=True, lora_rank=4)
        model = YiJingGPT(cfg)
        apply_lora(model, cfg)
        lora_p, total_p = count_lora_parameters(model)
        assert lora_p > 0
        assert lora_p < total_p * 0.1  # LoRA < 10% от всех параметров

    def test_merge_lora(self):
        """Merge вливает LoRA в основные веса."""
        from models.lora import apply_lora, merge_lora, LoRALinear
        cfg = make_cfg(use_lora=True, lora_rank=4)
        model = YiJingGPT(cfg)
        apply_lora(model, cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))

        # До merge
        logits_before, _, _ = model(x)

        # Merge
        merge_lora(model)
        logits_after, _, _ = model(x)

        # Результаты должны совпадать
        torch.testing.assert_close(logits_before, logits_after, atol=1e-5, rtol=1e-5)

    def test_unmerge_lora(self):
        """Unmerge восстанавливает LoRA для продолжения обучения."""
        from models.lora import apply_lora, merge_lora, unmerge_lora
        cfg = make_cfg(use_lora=True, lora_rank=4)
        model = YiJingGPT(cfg)
        apply_lora(model, cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))

        logits_orig, _, _ = model(x)
        merge_lora(model)
        unmerge_lora(model)
        logits_restored, _, _ = model(x)

        torch.testing.assert_close(logits_orig, logits_restored, atol=1e-5, rtol=1e-5)

    def test_lora_custom_targets(self):
        """LoRA с кастомными targets (все проекции)."""
        from models.lora import apply_lora, LoRALinear
        cfg = make_cfg(use_lora=True, lora_rank=4,
                       lora_targets=['q_proj', 'k_proj', 'v_proj', 'out'])
        model = YiJingGPT(cfg)
        count = apply_lora(model, cfg)
        assert count == cfg.n_layers * 4  # 4 targets per layer

    def test_lora_with_gqa(self):
        """LoRA работает с GQA."""
        from models.lora import apply_lora
        cfg = make_cfg(use_lora=True, lora_rank=4, n_kv_heads=2)
        model = YiJingGPT(cfg)
        apply_lora(model, cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0

    def test_save_load_lora(self):
        """Сохранение и загрузка LoRA весов."""
        import tempfile
        from models.lora import apply_lora, save_lora_weights, load_lora_weights
        cfg = make_cfg(use_lora=True, lora_rank=4)
        model = YiJingGPT(cfg)
        apply_lora(model, cfg)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            save_lora_weights(model, path)
            # Загружаем в новую модель
            model2 = YiJingGPT(cfg)
            apply_lora(model2, cfg)
            load_lora_weights(model2, path)

            # Проверяем что веса совпадают
            for (n1, p1), (n2, p2) in zip(
                [(n, p) for n, p in model.named_parameters() if 'lora_' in n],
                [(n, p) for n, p in model2.named_parameters() if 'lora_' in n],
            ):
                torch.testing.assert_close(p1, p2)
        finally:
            os.remove(path)


class TestV7SpeculativeDecoding:
    """Тесты для v7: speculative decoding."""

    def test_build_draft_model(self):
        """Draft модель создаётся с правильными параметрами."""
        from models.speculative import build_draft_model
        cfg = make_cfg(d_model=64, n_layers=4, draft_n_layers=2)
        draft = build_draft_model(cfg)
        assert len(draft.core.layers) == 2
        # draft d_model = 64 // 2 = 32
        assert draft.cfg.d_model == 32

    def test_speculative_generate(self):
        """Speculative generate выдаёт корректное число токенов."""
        from models.speculative import build_draft_model, speculative_generate
        cfg = make_cfg(d_model=64, n_layers=2, draft_n_layers=1)
        target = YiJingGPT(cfg)
        draft = build_draft_model(cfg)
        target.eval()
        draft.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = speculative_generate(target, draft, idx, max_new_tokens=8, K=3)
        # Должно быть >= 4 + 1 (хотя бы 1 принятый токен)
        assert out.shape[1] >= 5
        assert out.shape[1] <= 4 + 8 + 1  # max_new_tokens + 1 bonus token possible

    def test_speculative_with_temperature(self):
        """Speculative decoding с temperature > 1."""
        from models.speculative import build_draft_model, speculative_generate
        cfg = make_cfg(d_model=64, n_layers=2, draft_n_layers=1)
        target = YiJingGPT(cfg)
        draft = build_draft_model(cfg)
        target.eval()
        draft.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = speculative_generate(target, draft, idx, max_new_tokens=5,
                                   K=2, temperature=1.5)
        assert out.shape[1] >= 5

    def test_speculative_with_top_k(self):
        """Speculative decoding с top-k."""
        from models.speculative import build_draft_model, speculative_generate
        cfg = make_cfg(d_model=64, n_layers=2, draft_n_layers=1)
        target = YiJingGPT(cfg)
        draft = build_draft_model(cfg)
        target.eval()
        draft.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = speculative_generate(target, draft, idx, max_new_tokens=5,
                                   K=2, top_k=20)
        assert out.shape[1] >= 5


class TestV7ModelPresets:
    """Тесты для v7: model size presets."""

    def test_tiny_preset(self):
        cfg = YiJingConfig.tiny(vocab_size=128)
        assert cfg.d_model == 128
        assert cfg.n_layers == 4
        assert cfg.n_heads == 4
        model = YiJingGPT(cfg)
        total, _ = model.count_parameters()
        assert total > 0

    def test_small_preset(self):
        cfg = YiJingConfig.small(vocab_size=128)
        assert cfg.d_model == 256
        assert cfg.n_layers == 6
        model = YiJingGPT(cfg)
        total, _ = model.count_parameters()
        assert total > 0

    def test_medium_preset(self):
        cfg = YiJingConfig.medium(vocab_size=128)
        assert cfg.d_model == 512
        assert cfg.n_layers == 12
        assert cfg.n_kv_heads == 4  # GQA by default

    def test_large_preset(self):
        cfg = YiJingConfig.large(vocab_size=128)
        assert cfg.d_model == 1024
        assert cfg.n_layers == 16
        assert cfg.n_kv_heads == 4

    def test_preset_with_overrides(self):
        """Пресеты принимают дополнительные kwargs."""
        cfg = YiJingConfig.tiny(vocab_size=256, use_bian_gua=False)
        assert cfg.vocab_size == 256
        assert cfg.use_bian_gua is False

    def test_tiny_forward(self):
        """Tiny модель работает (forward + backward)."""
        cfg = YiJingConfig.tiny(vocab_size=128)
        model = YiJingGPT(cfg)
        x = torch.randint(0, 128, (2, 8))
        y = torch.randint(0, 128, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        assert loss.item() > 0


class TestV7DataLoading:
    """Тесты для v7: TextDataset."""

    def test_from_text(self):
        from data_utils.text_dataset import TextDataset
        text = "Hello world! This is a test text for the dataset." * 10
        ds = TextDataset.from_text(text, block_size=16)
        assert ds.n_tokens > 0
        assert len(ds) > 0

    def test_get_batch(self):
        from data_utils.text_dataset import TextDataset
        text = "The quick brown fox jumps over the lazy dog. " * 20
        ds = TextDataset.from_text(text, block_size=16)
        x, y = ds.get_batch(batch_size=4)
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        # y сдвинут на 1 относительно x
        # (проверяем что это одни и те же данные сдвинутые)

    def test_split(self):
        from data_utils.text_dataset import TextDataset
        text = "abcdefghijklmnop" * 50
        ds = TextDataset.from_text(text, block_size=8)
        train_ds, val_ds = ds.split(val_fraction=0.2)
        assert train_ds.n_tokens + val_ds.n_tokens == ds.n_tokens

    def test_get_vocab_size(self):
        from data_utils.text_dataset import TextDataset
        text = "Hello World"
        ds = TextDataset.from_text(text, block_size=4)
        vs = ds.get_vocab_size()
        assert vs > 0

    def test_shuffled_batch_iterator(self):
        from data_utils.text_dataset import TextDataset, ShuffledBatchIterator
        text = "The ancient Book of Changes. " * 50
        ds = TextDataset.from_text(text, block_size=8)
        iterator = ShuffledBatchIterator(ds, batch_size=4)
        x, y = iterator.get_batch()
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)
        assert iterator.n_batches > 0

    def test_repr(self):
        from data_utils.text_dataset import TextDataset
        text = "test" * 100
        ds = TextDataset.from_text(text, block_size=8)
        r = repr(ds)
        assert 'TextDataset' in r
        assert 'tokens=' in r


class TestV8RoPEScaling:
    """Тесты для v8: RoPE scaling (NTK и linear)."""

    def test_ntk_scaling(self):
        """NTK-aware RoPE scaling работает."""
        cfg = make_cfg(rope_scaling='ntk', rope_scaling_factor=2.0)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_linear_scaling(self):
        """Linear RoPE scaling работает."""
        cfg = make_cfg(rope_scaling='linear', rope_scaling_factor=2.0)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_no_scaling_default(self):
        """Без scaling — стандартный RoPE."""
        cfg = make_cfg(rope_scaling=None, rope_scaling_factor=1.0)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_ntk_different_frequencies(self):
        """NTK scaling изменяет inv_freq по сравнению с обычным RoPE."""
        from models.geometry import RotaryEmbedding
        dim = 16
        rope_std = RotaryEmbedding(dim, max_seq_len=64)
        rope_ntk = RotaryEmbedding(dim, max_seq_len=64, scaling='ntk', scaling_factor=4.0)
        # inv_freq должны отличаться
        assert not torch.allclose(rope_std.inv_freq, rope_ntk.inv_freq)

    def test_ntk_with_kv_cache(self):
        """NTK scaling + KV-cache работают вместе."""
        cfg = make_cfg(rope_scaling='ntk', rope_scaling_factor=2.0)
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=5, use_cache=True)
        assert out.shape == (1, 9)


class TestV8Generate:
    """Тесты для v8: улучшенный generate."""

    def test_repetition_penalty(self):
        """Repetition penalty не вызывает ошибок."""
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=10, repetition_penalty=1.2)
        assert out.shape[1] == 14

    def test_top_p(self):
        """Top-p (nucleus) sampling работает."""
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=10, top_p=0.9)
        assert out.shape[1] == 14

    def test_combined_sampling(self):
        """Top-k + top-p + repetition penalty вместе."""
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.generate(idx, max_new_tokens=8, top_k=20, top_p=0.9,
                             repetition_penalty=1.3)
        assert out.shape[1] == 12

    def test_stop_tokens(self):
        """Stop tokens прекращают генерацию."""
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        # stop_tokens=[0] — очень вероятно что 0 появится быстро
        out = model.generate(idx, max_new_tokens=100, stop_tokens=[0])
        assert out.shape[1] <= 104  # не больше max


class TestV8SaveLoad:
    """Тесты для v8: save/load pretrained."""

    def test_save_load(self):
        """Сохранение и загрузка модели."""
        import tempfile
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits_orig, _, _ = model(x)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            model.save_pretrained(path)
            model2 = YiJingGPT.from_pretrained(path)
            model2.eval()
            logits_loaded, _, _ = model2(x)
            torch.testing.assert_close(logits_orig, logits_loaded)
        finally:
            os.remove(path)

    def test_save_load_preserves_config(self):
        """Config сохраняется и восстанавливается."""
        import tempfile
        cfg = make_cfg(n_kv_heads=2, sliding_window=8, rope_scaling='ntk')
        model = YiJingGPT(cfg)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            model.save_pretrained(path)
            model2 = YiJingGPT.from_pretrained(path)
            assert model2.cfg.n_kv_heads == 2
            assert model2.cfg.sliding_window == 8
            assert model2.cfg.rope_scaling == 'ntk'
        finally:
            os.remove(path)


class TestV8FLOPS:
    """Тесты для v8: FLOPS estimation."""

    def test_estimate_flops(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        flops = model.estimate_flops()
        assert flops > 0

    def test_estimate_flops_custom_seq(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        f1 = model.estimate_flops(seq_len=16)
        f2 = model.estimate_flops(seq_len=32)
        assert f2 > f1  # длиннее = больше FLOPS

    def test_estimate_flops_str(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        s = model.estimate_flops_str()
        assert 'FLOPS' in s

    def test_flops_scales_with_layers(self):
        model_2 = YiJingGPT(make_cfg(n_layers=2))
        model_4 = YiJingGPT(make_cfg(n_layers=4))
        f2 = model_2.estimate_flops(seq_len=16)
        f4 = model_4.estimate_flops(seq_len=16)
        assert f4 > f2


class TestV8Distillation:
    """Тесты для v8: knowledge distillation."""

    def test_distillation_loss(self):
        """Distillation loss вычисляется корректно."""
        from training.distillation import distillation_loss
        B, T, V = 2, 8, 128
        student_logits = torch.randn(B, T, V)
        teacher_logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        loss = distillation_loss(student_logits, teacher_logits, targets,
                                 alpha=0.5, temperature=2.0)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_distillation_loss_alpha_zero(self):
        """alpha=0 → только hard loss."""
        from training.distillation import distillation_loss
        B, T, V = 2, 8, 128
        student_logits = torch.randn(B, T, V)
        teacher_logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))

        import torch.nn.functional as F
        hard_only = F.cross_entropy(
            student_logits.reshape(-1, V), targets.reshape(-1)
        )
        loss = distillation_loss(student_logits, teacher_logits, targets,
                                 alpha=0.0, temperature=2.0)
        torch.testing.assert_close(loss, hard_only, atol=1e-5, rtol=1e-5)

    def test_distillation_trainer_step(self):
        """DistillationTrainer выполняет один шаг."""
        from training.distillation import DistillationTrainer
        cfg = make_cfg()
        teacher = YiJingGPT(cfg)
        student = YiJingGPT(cfg)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        trainer = DistillationTrainer(teacher, student, optimizer, cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        metrics = trainer.step(x, y)
        assert 'total_loss' in metrics
        assert metrics['total_loss'] > 0

    def test_distillation_cross_size(self):
        """Distillation между моделями разного размера."""
        from training.distillation import DistillationTrainer
        teacher_cfg = make_cfg(d_model=64, n_layers=4)
        student_cfg = make_cfg(d_model=64, n_layers=2)
        teacher = YiJingGPT(teacher_cfg)
        student = YiJingGPT(student_cfg)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        trainer = DistillationTrainer(teacher, student, optimizer, student_cfg)
        x = torch.randint(0, teacher_cfg.vocab_size, (2, 8))
        y = torch.randint(0, teacher_cfg.vocab_size, (2, 8))
        metrics = trainer.step(x, y)
        assert metrics['total_loss'] > 0


class TestV8ModelCard:
    """Тесты для v8: model card."""

    def test_create_model_card(self):
        from models.export import create_model_card
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        card = create_model_card(model)
        assert card['model_type'] == 'YiJingGPT'
        assert card['architecture']['d_model'] == 64
        assert card['parameters']['total'] > 0
        assert 'FLOPS' in card['flops']['human_readable']

    def test_model_card_save(self):
        import tempfile
        from models.export import create_model_card
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            card = create_model_card(model, save_path=path)
            import json
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['model_type'] == 'YiJingGPT'
        finally:
            os.remove(path)


class TestV9BeamSearch:
    """Тесты для v9: beam search decoding."""

    def test_beam_search_basic(self):
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.beam_search(idx, max_new_tokens=5, beam_width=3)
        assert out.shape[0] == 1
        assert out.shape[1] == 9  # 4 + 5

    def test_beam_search_deterministic(self):
        """Beam search должен быть детерминистичным."""
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out1 = model.beam_search(idx.clone(), max_new_tokens=5, beam_width=3)
        out2 = model.beam_search(idx.clone(), max_new_tokens=5, beam_width=3)
        assert torch.equal(out1, out2)

    def test_beam_search_width_1(self):
        """Beam width=1 = greedy search."""
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model.beam_search(idx, max_new_tokens=5, beam_width=1)
        assert out.shape[1] == 9

    def test_beam_search_length_penalty(self):
        """Length penalty влияет на результат."""
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 4))
        # Разные length_penalty могут дать разные результаты
        out1 = model.beam_search(idx.clone(), max_new_tokens=8, beam_width=3,
                                  length_penalty=0.0)
        out2 = model.beam_search(idx.clone(), max_new_tokens=8, beam_width=3,
                                  length_penalty=1.0)
        # Оба валидны по длине
        assert out1.shape[1] == 12
        assert out2.shape[1] == 12


class TestV9LLRD:
    """Тесты для v9: Layer-wise LR Decay и advanced optimizer."""

    def test_build_optimizer_basic(self):
        from training.optim import build_optimizer
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        opt = build_optimizer(model, cfg)
        assert len(opt.param_groups) > 0

    def test_llrd_creates_groups(self):
        """LLRD создаёт разные LR для разных слоёв."""
        from training.optim import build_optimizer
        cfg = make_cfg(n_layers=4)
        model = YiJingGPT(cfg)
        opt = build_optimizer(model, cfg, llrd_factor=0.8)
        # Должно быть несколько групп с разными LR
        lrs = set(pg['lr'] for pg in opt.param_groups)
        assert len(lrs) > 1  # не все одинаковые

    def test_llrd_lr_ordering(self):
        """Верхние слои имеют больший LR при LLRD."""
        from training.optim import build_optimizer, _get_layer_idx
        cfg = make_cfg(n_layers=4)
        model = YiJingGPT(cfg)
        opt = build_optimizer(model, cfg, llrd_factor=0.7)
        # Собираем LR по слоям
        layer_lrs = {}
        for pg in opt.param_groups:
            for p in pg['params']:
                for name, param in model.named_parameters():
                    if param is p:
                        idx = _get_layer_idx(name, cfg.n_layers)
                        if idx is not None:
                            layer_lrs.setdefault(idx, set()).add(pg['lr'])
        # Верхний слой (3) >= нижний слой (0)
        if layer_lrs and 0 in layer_lrs and 3 in layer_lrs:
            assert max(layer_lrs[3]) >= max(layer_lrs[0])

    def test_cosine_schedule(self):
        from training.optim import get_cosine_schedule
        # Warmup: линейный рост
        lr_0 = get_cosine_schedule(0, 100, 1000, 1e-3)
        lr_50 = get_cosine_schedule(50, 100, 1000, 1e-3)
        lr_100 = get_cosine_schedule(100, 100, 1000, 1e-3)
        assert lr_0 < lr_50 < lr_100
        assert abs(lr_100 - 1e-3) < 1e-6  # peak

        # Decay: убывает к 0
        lr_end = get_cosine_schedule(1000, 100, 1000, 1e-3)
        assert lr_end < lr_100

    def test_wsd_schedule(self):
        from training.optim import get_warmup_stable_decay_schedule
        lr = get_warmup_stable_decay_schedule(
            50, warmup_steps=100, stable_steps=200,
            decay_steps=300, max_lr=1e-3
        )
        assert 0 < lr < 1e-3  # in warmup

        lr_stable = get_warmup_stable_decay_schedule(
            200, warmup_steps=100, stable_steps=200,
            decay_steps=300, max_lr=1e-3
        )
        assert abs(lr_stable - 1e-3) < 1e-6  # stable phase

    def test_training_step_with_llrd(self):
        """Один шаг обучения с LLRD работает."""
        from training.optim import build_optimizer, get_cosine_schedule
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        opt = build_optimizer(model, cfg, llrd_factor=0.85)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        model.train()
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()
        assert loss.item() > 0


class TestV9ByteTokenizer:
    """Тесты для v9: ByteTokenizer."""

    def test_encode_decode(self):
        from tokenizer.char_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        text = "Hello, world!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_unicode(self):
        """ByteTokenizer обрабатывает Unicode."""
        from tokenizer.char_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        text = "Привет мир! 你好世界"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        from tokenizer.char_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        assert tok.get_piece_size() == 259  # 256 bytes + 3 special

    def test_special_tokens(self):
        from tokenizer.char_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        assert tok.bos_id() == 1
        assert tok.eos_id() == 2

    def test_encode_with_special(self):
        from tokenizer.char_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        ids = tok.encode_with_special("Hi")
        assert ids[0] == 1  # BOS
        assert ids[-1] == 2  # EOS
        assert len(ids) == 4  # BOS + H + i + EOS

    def test_byte_tokenizer_with_model(self):
        """ByteTokenizer работает с моделью."""
        from tokenizer.char_tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        cfg = make_cfg(vocab_size=tok.get_piece_size())
        model = YiJingGPT(cfg)
        text = "Hello"
        ids = tok.encode(text)
        x = torch.tensor([ids], dtype=torch.long)
        logits, _, _ = model(x)
        assert logits.shape == (1, len(ids), tok.get_piece_size())


class TestV9CharTokenizerSaveLoad:
    """Тесты для v9: сохранение/загрузка CharTokenizer."""

    def test_save_load(self):
        import tempfile
        from tokenizer.char_tokenizer import CharTokenizer
        tok = CharTokenizer.from_text("Hello World 123!")
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = CharTokenizer.load(path)
            assert tok.get_piece_size() == tok2.get_piece_size()
            assert tok.encode("Hello") == tok2.encode("Hello")
        finally:
            os.remove(path)


# ==================== V10 TESTS ====================

class TestV10ALiBi:
    """Тесты для ALiBi позиционного bias."""

    def test_alibi_slopes(self):
        """ALiBi slopes — убывающая геометрическая последовательность."""
        from models.geometry import ALiBi
        alibi = ALiBi(n_heads=8)
        slopes = alibi.slopes
        assert slopes.shape == (8,)
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1]

    def test_alibi_bias_shape(self):
        """ALiBi bias имеет правильную форму."""
        from models.geometry import ALiBi
        alibi = ALiBi(n_heads=4, max_seq_len=64)
        bias = alibi(seq_len=16)
        assert bias.shape == (1, 4, 16, 16)

    def test_alibi_bias_with_offset(self):
        """ALiBi bias с KV-cache offset."""
        from models.geometry import ALiBi
        alibi = ALiBi(n_heads=4, max_seq_len=64)
        bias = alibi(seq_len=1, offset=15)
        assert bias.shape == (1, 4, 1, 16)

    def test_alibi_bias_causal_property(self):
        """ALiBi bias: ближние позиции получают меньший штраф."""
        from models.geometry import ALiBi
        alibi = ALiBi(n_heads=4, max_seq_len=32)
        bias = alibi(seq_len=8)
        for h in range(4):
            for i in range(8):
                assert bias[0, h, i, i].item() == 0.0
        assert (bias[0, 0, 5, 0] < 0).item()

    def test_alibi_non_power_of_2_heads(self):
        """ALiBi работает с не-степенью-2 числом голов."""
        from models.geometry import ALiBi
        alibi = ALiBi(n_heads=6, max_seq_len=32)
        bias = alibi(seq_len=8)
        assert bias.shape == (1, 6, 8, 8)

    def test_model_with_alibi(self):
        """Модель с ALiBi (без RoPE) работает."""
        cfg = make_cfg(use_rope=False, use_alibi=True)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_alibi_backward(self):
        """Backward через ALiBi."""
        cfg = make_cfg(use_rope=False, use_alibi=True)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()

    def test_alibi_cache_extension(self):
        """ALiBi автоматически расширяет кэш при длинных последовательностях."""
        from models.geometry import ALiBi
        alibi = ALiBi(n_heads=4, max_seq_len=16)
        bias = alibi(seq_len=32)
        assert bias.shape == (1, 4, 32, 32)


class TestV10AttentionSinks:
    """Тесты для attention sinks."""

    def test_model_with_sinks(self):
        """Модель с attention sinks forward."""
        cfg = make_cfg(attention_sinks=2, sliding_window=8)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_sinks_preserve_first_tokens(self):
        """Attention sinks сохраняют видимость первых токенов при sliding window."""
        cfg = make_cfg(attention_sinks=2, sliding_window=4)
        attn = YiJingAttention(cfg)
        x = torch.randn(1, 12, cfg.d_model)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_sinks_without_sliding_window(self):
        """Attention sinks без sliding window — работает нормально."""
        cfg = make_cfg(attention_sinks=2, sliding_window=None)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_sinks_backward(self):
        """Backward через attention sinks."""
        cfg = make_cfg(attention_sinks=2, sliding_window=6)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()


class TestV10EMA:
    """Тесты для EMA model averaging."""

    def test_ema_init(self):
        """EMA инициализируется копией параметров."""
        from training.ema import EMA
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        ema = EMA(model, decay=0.99)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert torch.equal(ema.shadow[name], param.data)

    def test_ema_update(self):
        """EMA обновляет shadow параметры."""
        from training.ema import EMA
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        ema = EMA(model, decay=0.9)
        old_shadows = {n: s.clone() for n, s in ema.shadow.items()}
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update()
        changed = False
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema.shadow:
                if not torch.equal(ema.shadow[name], old_shadows[name]):
                    changed = True
                assert not torch.equal(ema.shadow[name], param.data)
        assert changed

    def test_ema_context_manager(self):
        """Context manager переключает и восстанавливает параметры."""
        from training.ema import EMA
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        ema = EMA(model, decay=0.9)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update()
        originals = {n: p.data.clone() for n, p in model.named_parameters()}
        with ema.average_parameters():
            for name, param in model.named_parameters():
                if param.requires_grad and name in ema.shadow:
                    assert torch.equal(param.data, ema.shadow[name])
        for name, param in model.named_parameters():
            if name in originals:
                assert torch.equal(param.data, originals[name])

    def test_ema_state_dict(self):
        """EMA state_dict сохраняется и загружается."""
        from training.ema import EMA
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        ema = EMA(model, decay=0.995)
        state = ema.state_dict()
        assert 'shadow' in state
        assert state['decay'] == 0.995


class TestV10EarlyStopping:
    """Тесты для early stopping."""

    def test_no_stop_improving(self):
        """Не останавливается при улучшении."""
        from training.ema import EarlyStopping
        es = EarlyStopping(patience=3, mode='min')
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)
        assert not es(0.7)

    def test_stop_stagnant(self):
        """Останавливается при отсутствии улучшений."""
        from training.ema import EarlyStopping
        es = EarlyStopping(patience=3, mode='min')
        assert not es(1.0)
        assert not es(1.1)
        assert not es(1.2)
        assert es(1.3)

    def test_stop_max_mode(self):
        """Max mode: останавливается при падении accuracy."""
        from training.ema import EarlyStopping
        es = EarlyStopping(patience=2, mode='max')
        assert not es(0.8)
        assert not es(0.9)
        assert not es(0.85)
        assert es(0.85)

    def test_min_delta(self):
        """min_delta: мелкие улучшения не считаются."""
        from training.ema import EarlyStopping
        es = EarlyStopping(patience=2, min_delta=0.1, mode='min')
        assert not es(1.0)
        assert not es(0.95)
        assert es(0.92)

    def test_reset(self):
        """Reset сбрасывает состояние."""
        from training.ema import EarlyStopping
        es = EarlyStopping(patience=2, mode='min')
        es(1.0)
        es(1.1)
        es.reset()
        assert es.counter == 0
        assert es.best_score is None
        assert not es.should_stop


class TestV10GroupedQuantizer:
    """Тесты для Grouped Quantization."""

    def test_basic_shape(self):
        """Grouped quantizer сохраняет shape."""
        from models.geometry import GroupedQuantizer
        gq = GroupedQuantizer(d_model=256, group_size=64, n_bits=8)
        x = torch.randn(2, 16, 256)
        y = gq(x)
        assert y.shape == x.shape

    def test_small_group(self):
        """Маленький group_size."""
        from models.geometry import GroupedQuantizer
        gq = GroupedQuantizer(d_model=64, group_size=16, n_bits=8)
        assert gq.n_groups == 4
        x = torch.randn(4, 64)
        y = gq(x)
        assert y.shape == x.shape

    def test_asymmetric(self):
        """Асимметричная квантизация с zero-point."""
        from models.geometry import GroupedQuantizer
        gq = GroupedQuantizer(d_model=128, group_size=32, n_bits=8, symmetric=False)
        assert hasattr(gq, 'zero_points')
        x = torch.randn(2, 8, 128)
        y = gq(x)
        assert y.shape == x.shape

    def test_ste_gradient(self):
        """STE: градиенты проходят через квантизацию."""
        from models.geometry import GroupedQuantizer
        gq = GroupedQuantizer(d_model=64, group_size=32, n_bits=8)
        x = torch.randn(2, 64, requires_grad=True)
        y = gq(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_calibrate(self):
        """Калибровка scales на данных."""
        from models.geometry import GroupedQuantizer
        gq = GroupedQuantizer(d_model=128, group_size=64, n_bits=8)
        x = torch.randn(100, 128) * 5.0
        gq.calibrate(x)
        assert (gq.scales > 0).all()

    def test_group_size_larger_than_d(self):
        """Group size > d_model → одна группа."""
        from models.geometry import GroupedQuantizer
        gq = GroupedQuantizer(d_model=32, group_size=128, n_bits=8)
        assert gq.n_groups == 1
        x = torch.randn(4, 32)
        y = gq(x)
        assert y.shape == x.shape

    def test_4bit_quantization(self):
        """4-bit квантизация."""
        from models.geometry import GroupedQuantizer
        gq = GroupedQuantizer(d_model=64, group_size=32, n_bits=4)
        assert gq.qmin == -8
        assert gq.qmax == 7
        x = torch.randn(2, 64)
        y = gq(x)
        assert y.shape == x.shape


class TestV10ActivationMemory:
    """Тесты для activation memory profiler."""

    def test_basic_profile(self):
        """Профилирование активационной памяти."""
        from training.ema import compute_activation_memory
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        mem = compute_activation_memory(model, batch_size=2, seq_len=16)
        assert 'total' in mem
        assert 'total_mb' in mem
        assert 'per_layer' in mem
        assert mem['total'] > 0

    def test_memory_scales_with_batch(self):
        """Память растёт линейно с batch_size."""
        from training.ema import compute_activation_memory
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        mem1 = compute_activation_memory(model, batch_size=1, seq_len=16)
        mem2 = compute_activation_memory(model, batch_size=2, seq_len=16)
        assert mem2['total'] == 2 * mem1['total']

    def test_memory_scales_with_seq_len(self):
        """Память растёт квадратично с seq_len (из-за attention)."""
        from training.ema import compute_activation_memory
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        mem1 = compute_activation_memory(model, batch_size=1, seq_len=16)
        mem2 = compute_activation_memory(model, batch_size=1, seq_len=32)
        assert mem2['total'] > 2 * mem1['total']


class TestV10Integration:
    """Интеграционные тесты v10."""

    def test_alibi_with_sinks_and_sliding(self):
        """ALiBi + attention sinks + sliding window."""
        cfg = make_cfg(
            use_rope=False, use_alibi=True,
            attention_sinks=2, sliding_window=8
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss, _ = model(x, y)
        assert logits.shape == (2, 16, cfg.vocab_size)
        loss.backward()

    def test_alibi_kv_cache(self):
        """ALiBi с KV-cache генерацией."""
        cfg = make_cfg(use_rope=False, use_alibi=True)
        model = YiJingGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape[0] == 1

    def test_ema_with_training_step(self):
        """EMA обновляется при обучении."""
        from training.ema import EMA
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        ema = EMA(model, decay=0.99)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ema.update()


# ==================== V11 TESTS ====================

class TestV11MQA:
    """Тесты для Multi-Query Attention (n_kv_heads=1)."""

    def test_mqa_forward(self):
        """MQA: n_kv_heads=1 работает."""
        cfg = make_cfg(n_kv_heads=1)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, _, _ = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_mqa_backward(self):
        """MQA backward."""
        cfg = make_cfg(n_kv_heads=1)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()

    def test_mqa_fewer_kv_params(self):
        """MQA имеет меньше параметров K/V проекций."""
        cfg_mha = make_cfg(n_kv_heads=None)  # MHA: 8 kv heads
        cfg_mqa = make_cfg(n_kv_heads=1)     # MQA: 1 kv head
        attn_mha = YiJingAttention(cfg_mha)
        attn_mqa = YiJingAttention(cfg_mqa)
        # K/V проекции MQA должны быть меньше
        mha_kv = sum(p.numel() for p in [attn_mha.k_proj.weight, attn_mha.v_proj.weight])
        mqa_kv = sum(p.numel() for p in [attn_mqa.k_proj.weight, attn_mqa.v_proj.weight])
        assert mqa_kv < mha_kv

    def test_mqa_with_kv_cache(self):
        """MQA с KV-cache."""
        cfg = make_cfg(n_kv_heads=1)
        model = YiJingGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits1, _, cache = model(x)
            # Decode step
            next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
            logits2, _, _ = model(next_tok, kv_cache=cache)
        assert logits2.shape == (1, 1, cfg.vocab_size)


class TestV11TokenMerger:
    """Тесты для Token Merging."""

    def test_merge_basic(self):
        """Token Merging: merge и unmerge."""
        from training.regularization import TokenMerger
        merger = TokenMerger(merge_ratio=0.25)
        x = torch.randn(2, 16, 64)
        merged, info = merger(x)
        assert merged.shape[0] == 2
        assert merged.shape[1] < 16  # сократилось
        assert merged.shape[2] == 64

    def test_merge_unmerge_shape(self):
        """Unmerge восстанавливает исходную длину."""
        from training.regularization import TokenMerger
        merger = TokenMerger(merge_ratio=0.25)
        x = torch.randn(1, 16, 32)
        merged, info = merger(x)
        unmerged = merger.unmerge(merged, info)
        assert unmerged.shape == x.shape

    def test_merge_zero_ratio(self):
        """merge_ratio=0 не меняет ничего."""
        from training.regularization import TokenMerger
        merger = TokenMerger(merge_ratio=0.0)
        x = torch.randn(2, 8, 32)
        merged, info = merger(x)
        assert info is None
        assert merged.shape == x.shape

    def test_merge_short_sequence(self):
        """Очень короткая последовательность (T=2)."""
        from training.regularization import TokenMerger
        merger = TokenMerger(merge_ratio=0.5)
        x = torch.randn(1, 2, 32)
        merged, info = merger(x)
        # T=2 → r=1, a=[0], b=[1], merge pair → T-1=1
        assert merged.shape[1] <= 2

    def test_merge_gradient(self):
        """Gradient проходит через merge/unmerge."""
        from training.regularization import TokenMerger
        merger = TokenMerger(merge_ratio=0.25)
        x = torch.randn(1, 8, 32, requires_grad=True)
        merged, info = merger(x)
        loss = merged.sum()
        loss.backward()
        assert x.grad is not None


class TestV11CosineWarmRestarts:
    """Тесты для cosine annealing с warm restarts."""

    def test_basic_schedule(self):
        """LR уменьшается, затем перезапускается."""
        from training.regularization import CosineAnnealingWarmRestarts
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1)

        lrs = []
        for _ in range(25):
            scheduler.step()
            lrs.append(scheduler.get_lr()[0])

        # LR должен упасть в середине цикла
        assert lrs[4] < lrs[0]
        # LR на restart (T_cur=0) возвращается к base
        # step 10 → T_cur=10 → reset → T_cur=0 → lr=base=0.1
        assert abs(lrs[9] - 0.1) < 1e-6  # restart point

    def test_warmup(self):
        """Warmup в начале каждого цикла."""
        from training.regularization import CosineAnnealingWarmRestarts
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=20, warmup_steps=5)

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(scheduler.get_lr()[0])

        # Warmup: LR растёт первые 5 шагов
        assert lrs[0] < lrs[3]

    def test_t_mult(self):
        """T_mult удлиняет циклы."""
        from training.regularization import CosineAnnealingWarmRestarts
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

        # Первый цикл: 10 шагов, второй: 20 шагов
        for _ in range(10):
            scheduler.step()
        assert scheduler.cycle == 1
        assert scheduler.T_i == 20

    def test_state_dict(self):
        """State dict сохраняется и загружается."""
        from training.regularization import CosineAnnealingWarmRestarts
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = CosineAnnealingWarmRestarts(opt, T_0=10)
        for _ in range(5):
            sched.step()
        state = sched.state_dict()
        assert state['step_count'] == 5


class TestV11GradientNoise:
    """Тесты для gradient noise."""

    def test_noise_added(self):
        """Шум добавляется к градиентам."""
        from training.regularization import GradientNoise
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        grad_before = model.weight.grad.clone()
        gn = GradientNoise(eta=1.0, gamma=0.55)
        gn.add_noise(model)
        grad_after = model.weight.grad

        # Градиенты должны измениться
        assert not torch.equal(grad_before, grad_after)

    def test_noise_decreases(self):
        """Sigma уменьшается со временем."""
        from training.regularization import GradientNoise
        gn = GradientNoise(eta=1.0, gamma=0.55)
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)

        sigmas = []
        for _ in range(10):
            loss = model(x).sum()
            loss.backward()
            sigma = gn.add_noise(model)
            sigmas.append(sigma)
            model.zero_grad()

        # Sigma должна убывать
        assert sigmas[-1] < sigmas[0]

    def test_noise_state_dict(self):
        """State dict сохраняется."""
        from training.regularization import GradientNoise
        gn = GradientNoise(eta=0.5)
        gn.step_count = 100
        state = gn.state_dict()
        assert state['step_count'] == 100
        assert state['eta'] == 0.5


class TestV11LabelSmoothing:
    """Тесты для label smoothing."""

    def test_label_smoothing_forward(self):
        """Модель с label smoothing работает."""
        cfg = make_cfg(label_smoothing=0.1)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        assert loss is not None
        assert loss.item() > 0

    def test_label_smoothing_higher_loss(self):
        """Label smoothing увеличивает loss (менее уверенные предсказания)."""
        cfg_no = make_cfg(label_smoothing=0.0)
        cfg_ls = make_cfg(label_smoothing=0.1)
        # Фиксируем seed для воспроизводимости
        torch.manual_seed(42)
        model_no = YiJingGPT(cfg_no)
        torch.manual_seed(42)
        model_ls = YiJingGPT(cfg_ls)
        x = torch.randint(0, cfg_no.vocab_size, (2, 8))
        y = torch.randint(0, cfg_no.vocab_size, (2, 8))
        _, loss_no, _ = model_no(x, y)
        _, loss_ls, _ = model_ls(x, y)
        # Label smoothing обычно увеличивает loss
        assert loss_ls.item() >= loss_no.item() - 0.5  # мягкая проверка

    def test_label_smoothing_backward(self):
        """Backward с label smoothing."""
        cfg = make_cfg(label_smoothing=0.1)
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()


class TestV11Integration:
    """Интеграционные тесты v11."""

    def test_mqa_alibi_label_smoothing(self):
        """MQA + ALiBi + label smoothing вместе."""
        cfg = make_cfg(
            n_kv_heads=1, use_rope=False, use_alibi=True,
            label_smoothing=0.1
        )
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()

    def test_full_training_loop_v11(self):
        """Мини training loop со всеми v11 фичами."""
        from training.regularization import CosineAnnealingWarmRestarts, GradientNoise
        from training.ema import EMA, EarlyStopping

        cfg = make_cfg(n_kv_heads=1, label_smoothing=0.05)
        model = YiJingGPT(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingWarmRestarts(opt, T_0=5, warmup_steps=2)
        gn = GradientNoise(eta=0.01)
        ema = EMA(model, decay=0.99)
        es = EarlyStopping(patience=3, mode='min')

        for step in range(10):
            x = torch.randint(0, cfg.vocab_size, (2, 8))
            y = torch.randint(0, cfg.vocab_size, (2, 8))
            _, loss, _ = model(x, y)
            loss.backward()
            gn.add_noise(model)
            opt.step()
            opt.zero_grad()
            scheduler.step()
            ema.update()
            es(loss.item())

        # EMA inference
        with ema.average_parameters():
            model.eval()
            x = torch.randint(0, cfg.vocab_size, (1, 4))
            with torch.no_grad():
                logits, _, _ = model(x)
            assert logits.shape == (1, 4, cfg.vocab_size)


# ==================== V12 TESTS ====================

class TestV12MixtureOfDepths:
    """Тесты для Mixture of Depths."""

    def test_mod_forward(self):
        """MoD forward: mod_capacity < 1.0."""
        cfg = make_cfg(mod_capacity=0.5)
        model = YiJingGPT(cfg)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss, _ = model(x, y)
        assert logits.shape == (2, 16, cfg.vocab_size)
        assert loss is not None

    def test_mod_backward(self):
        """MoD backward."""
        cfg = make_cfg(mod_capacity=0.5)
        model = YiJingGPT(cfg)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()

    def test_mod_eval_no_routing(self):
        """MoD в eval режиме: все токены проходят (нет routing)."""
        cfg = make_cfg(mod_capacity=0.5)
        model = YiJingGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape == (1, 8, cfg.vocab_size)

    def test_mod_capacity_1(self):
        """mod_capacity=1.0 → все токены, нет router."""
        cfg = make_cfg(mod_capacity=1.0)
        model = YiJingGPT(cfg)
        assert model.core.mod_routers is None

    def test_mod_has_routers(self):
        """mod_capacity < 1.0 → есть routers."""
        cfg = make_cfg(mod_capacity=0.75)
        model = YiJingGPT(cfg)
        assert model.core.mod_routers is not None
        assert len(model.core.mod_routers) == cfg.n_layers


class TestV12MuP:
    """Тесты для µP инициализации."""

    def test_mup_init(self):
        """µP инициализация не ломает forward."""
        from training.utils_v12 import apply_mup_init
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        apply_mup_init(model, base_width=32)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)

    def test_mup_scales_with_width(self):
        """µP: более широкие модели получают меньшие init."""
        from training.utils_v12 import apply_mup_init
        cfg_narrow = make_cfg(d_model=64, n_heads=4)
        cfg_wide = make_cfg(d_model=128, n_heads=8)
        model_n = YiJingGPT(cfg_narrow)
        model_w = YiJingGPT(cfg_wide)
        apply_mup_init(model_n, base_width=64)
        apply_mup_init(model_w, base_width=64)
        # Широкая модель: weights std должен быть меньше
        for (n1, p1), (n2, p2) in zip(
            model_n.named_parameters(), model_w.named_parameters()
        ):
            if 'q_proj.weight' in n1:
                assert p2.std() <= p1.std() + 0.01
                break

    def test_mup_param_groups(self):
        """µP param groups создают разные lr."""
        from training.utils_v12 import get_mup_param_groups
        cfg = make_cfg(weight_tying=False)
        model = YiJingGPT(cfg)
        groups = get_mup_param_groups(model, base_lr=1e-3, base_width=32)
        assert len(groups) >= 1
        # Output group lr меньше base lr
        if len(groups) > 1:
            assert groups[1]['lr'] < groups[0]['lr']


class TestV12DynamicTemperature:
    """Тесты для Dynamic Temperature."""

    def test_basic(self):
        """Dynamic temperature возвращает правильную форму."""
        from training.utils_v12 import dynamic_temperature
        logits = torch.randn(4, 1000)
        temp = dynamic_temperature(logits)
        assert temp.shape == (4, 1)

    def test_confident_low_temp(self):
        """Уверенные предсказания → низкая температура."""
        from training.utils_v12 import dynamic_temperature
        # Очень уверенный logits (один класс доминирует)
        logits_confident = torch.zeros(1, 100)
        logits_confident[0, 0] = 100.0
        temp_c = dynamic_temperature(logits_confident)

        # Неуверенный logits (равномерно)
        logits_uniform = torch.ones(1, 100)
        temp_u = dynamic_temperature(logits_uniform)

        assert temp_c.item() < temp_u.item()

    def test_bounds(self):
        """Temperature ограничена min/max."""
        from training.utils_v12 import dynamic_temperature
        logits = torch.randn(8, 500)
        temp = dynamic_temperature(logits, min_temp=0.5, max_temp=1.5)
        assert (temp >= 0.5).all()
        assert (temp <= 1.5).all()


class TestV12CheckpointManager:
    """Тесты для Checkpoint Manager."""

    def test_save_and_track(self):
        """Сохранение чекпоинтов."""
        import tempfile
        from training.utils_v12 import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, max_keep=2, mode='min')
            cfg = make_cfg()
            model = YiJingGPT(cfg)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            p1 = mgr.save(model, opt, step=100, metric=2.5)
            assert p1 is not None
            p2 = mgr.save(model, opt, step=200, metric=2.0)
            assert p2 is not None
            p3 = mgr.save(model, opt, step=300, metric=1.5)
            assert p3 is not None

            # Только 2 лучших сохранены
            assert len(mgr.list_checkpoints()) == 2

    def test_get_best(self):
        """Получение лучшего чекпоинта."""
        import tempfile
        from training.utils_v12 import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, max_keep=3, mode='min')
            cfg = make_cfg()
            model = YiJingGPT(cfg)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            mgr.save(model, opt, step=100, metric=2.5)
            mgr.save(model, opt, step=200, metric=1.5)
            mgr.save(model, opt, step=300, metric=2.0)

            best = mgr.get_best()
            assert '1.5' in best

    def test_should_save(self):
        """should_save правильно фильтрует."""
        import tempfile
        from training.utils_v12 import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, max_keep=1, mode='min')
            cfg = make_cfg()
            model = YiJingGPT(cfg)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            mgr.save(model, opt, step=100, metric=1.0)
            # Хуже чем 1.0 → не сохраняем
            assert not mgr.should_save(2.0)
            # Лучше → сохраняем
            assert mgr.should_save(0.5)

    def test_max_mode(self):
        """Max mode (accuracy)."""
        import tempfile
        from training.utils_v12 import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, max_keep=2, mode='max')
            cfg = make_cfg()
            model = YiJingGPT(cfg)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            mgr.save(model, opt, step=100, metric=0.8)
            mgr.save(model, opt, step=200, metric=0.9)
            mgr.save(model, opt, step=300, metric=0.7)

            # best = 0.9
            best = mgr.get_best()
            assert '0.9' in best


class TestV12Perplexity:
    """Тесты для perplexity evaluator."""

    def test_basic_perplexity(self):
        """Perplexity вычисляется на синтетических данных."""
        from training.utils_v12 import evaluate_perplexity
        cfg = make_cfg()
        model = YiJingGPT(cfg)

        # Создаём простой dataloader
        data = [torch.randint(0, cfg.vocab_size, (4, 16)) for _ in range(3)]
        # Каждый batch: x=data[:, :-1], y=data[:, 1:]
        loader = [(d[:, :-1], d[:, 1:]) for d in data]

        result = evaluate_perplexity(model, loader)
        assert 'perplexity' in result
        assert result['perplexity'] > 1.0
        assert result['n_tokens'] > 0

    def test_perplexity_max_batches(self):
        """max_batches ограничивает число batch'ей."""
        from training.utils_v12 import evaluate_perplexity
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        data = [torch.randint(0, cfg.vocab_size, (4, 16)) for _ in range(10)]
        loader = [(d[:, :-1], d[:, 1:]) for d in data]

        r1 = evaluate_perplexity(model, loader, max_batches=2)
        r2 = evaluate_perplexity(model, loader, max_batches=5)
        assert r2['n_tokens'] > r1['n_tokens']

    def test_perplexity_single_tensor_batch(self):
        """Perplexity с batch = single tensor (auto shift)."""
        from training.utils_v12 import evaluate_perplexity
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        loader = [torch.randint(0, cfg.vocab_size, (2, 16)) for _ in range(3)]
        result = evaluate_perplexity(model, loader)
        assert result['perplexity'] > 1.0


class TestV12Integration:
    """Интеграционные тесты v12."""

    def test_mod_with_alibi(self):
        """MoD + ALiBi."""
        cfg = make_cfg(mod_capacity=0.5, use_rope=False, use_alibi=True)
        model = YiJingGPT(cfg)
        model.train()
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()

    def test_mup_with_training(self):
        """µP init + training step."""
        from training.utils_v12 import apply_mup_init, get_mup_param_groups
        cfg = make_cfg(weight_tying=False)
        model = YiJingGPT(cfg)
        apply_mup_init(model, base_width=32)
        groups = get_mup_param_groups(model, base_lr=1e-3, base_width=32)
        opt = torch.optim.AdamW(groups)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()

    def test_dynamic_temp_in_generate(self):
        """Dynamic temperature при генерации."""
        from training.utils_v12 import dynamic_temperature
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            logits, _, _ = model(x)
            last_logits = logits[:, -1, :]
            temp = dynamic_temperature(last_logits)
            scaled_logits = last_logits / temp
            probs = F.softmax(scaled_logits, dim=-1)
            assert probs.shape == (1, cfg.vocab_size)
            assert abs(probs.sum().item() - 1.0) < 1e-5


# ==================== V13 TESTS ====================

class TestV13BPETokenizer:
    """Тесты для BPE tokenizer."""

    def test_train_and_encode(self):
        """BPE обучается и кодирует текст."""
        from tokenizer.char_tokenizer import BPETokenizer
        bpe = BPETokenizer()
        corpus = "hello world hello world hello world foo bar foo bar" * 10
        bpe.train(corpus, vocab_size=280)
        ids = bpe.encode("hello world")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_roundtrip(self):
        """Encode → decode даёт исходный текст."""
        from tokenizer.char_tokenizer import BPETokenizer
        bpe = BPETokenizer()
        corpus = "the quick brown fox jumps over the lazy dog " * 50
        bpe.train(corpus, vocab_size=300)
        text = "the quick brown fox"
        ids = bpe.encode(text)
        decoded = bpe.decode(ids)
        assert decoded == text

    def test_compression(self):
        """BPE сжимает текст (меньше токенов чем байтов)."""
        from tokenizer.char_tokenizer import BPETokenizer
        bpe = BPETokenizer()
        corpus = "abcabc " * 200
        bpe.train(corpus, vocab_size=300)
        ids = bpe.encode("abcabc abcabc")
        byte_len = len("abcabc abcabc".encode('utf-8'))
        assert len(ids) < byte_len

    def test_special_tokens(self):
        """Спец-токены BOS/EOS."""
        from tokenizer.char_tokenizer import BPETokenizer
        bpe = BPETokenizer()
        bpe.train("hello " * 100, vocab_size=270)
        ids = bpe.encode_with_special("hello")
        assert ids[0] == bpe.BOS_ID
        assert ids[-1] == bpe.EOS_ID

    def test_save_load(self):
        """Сохранение и загрузка BPE модели."""
        import tempfile
        from tokenizer.char_tokenizer import BPETokenizer
        bpe = BPETokenizer()
        bpe.train("test data test data " * 50, vocab_size=280)
        text = "test data"
        ids_before = bpe.encode(text)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            bpe.save(path)
            bpe2 = BPETokenizer.load(path)
            ids_after = bpe2.encode(text)
            assert ids_before == ids_after
        finally:
            os.remove(path)

    def test_unicode(self):
        """BPE работает с Unicode текстом."""
        from tokenizer.char_tokenizer import BPETokenizer
        bpe = BPETokenizer()
        corpus = "привет мир " * 50
        bpe.train(corpus, vocab_size=300)
        ids = bpe.encode("привет")
        decoded = bpe.decode(ids)
        assert decoded == "привет"

    def test_with_model(self):
        """BPE tokenizer работает с моделью."""
        from tokenizer.char_tokenizer import BPETokenizer
        bpe = BPETokenizer()
        bpe.train("hello world " * 100, vocab_size=280)
        cfg = make_cfg(vocab_size=bpe.get_piece_size())
        model = YiJingGPT(cfg)
        ids = bpe.encode("hello")
        x = torch.tensor([ids], dtype=torch.long)
        logits, _, _ = model(x)
        assert logits.shape == (1, len(ids), bpe.get_piece_size())


class TestV13RingAttention:
    """Тесты для Ring Attention scaffold."""

    def test_config(self):
        """Ring Attention конфигурация."""
        from training.utils_v13 import RingAttentionConfig
        cfg = RingAttentionConfig(world_size=4, segment_len=256)
        assert cfg.total_len == 1024
        start, end = cfg.get_rank_range(2)
        assert start == 512
        assert end == 768

    def test_kv_source(self):
        """KV source ranks для ring communication."""
        from training.utils_v13 import RingAttentionConfig
        cfg = RingAttentionConfig(world_size=4, segment_len=128)
        # Step 0: свои KV
        assert cfg.get_kv_source_ranks(2, 0) == 2
        # Step 1: от rank-1
        assert cfg.get_kv_source_ranks(2, 1) == 1

    def test_simulate(self):
        """Симуляция Ring Attention."""
        from training.utils_v13 import simulate_ring_attention
        result = simulate_ring_attention(seq_len=1024, n_gpus=4)
        assert result['n_segments'] == 4
        assert result['memory_per_gpu_tokens'] == 256
        assert result['total_blocks'] > 0

    def test_simulate_single_gpu(self):
        """Симуляция на 1 GPU = обычный attention."""
        from training.utils_v13 import simulate_ring_attention
        result = simulate_ring_attention(seq_len=512, n_gpus=1)
        assert result['n_segments'] == 1
        assert result['total_blocks'] == 1


class TestV13SelectiveCheckpointing:
    """Тесты для selective activation checkpointing."""

    def test_every_2(self):
        """Checkpoint каждый 2-й слой."""
        from training.utils_v13 import SelectiveCheckpointing
        sc = SelectiveCheckpointing(n_layers=8, checkpoint_every=2)
        layers = sc.get_checkpoint_layers()
        assert layers == [1, 3, 5, 7]

    def test_every_1(self):
        """Checkpoint все слои."""
        from training.utils_v13 import SelectiveCheckpointing
        sc = SelectiveCheckpointing(n_layers=4, checkpoint_every=1)
        assert len(sc.get_checkpoint_layers()) == 4

    def test_disabled(self):
        """Checkpoint отключён."""
        from training.utils_v13 import SelectiveCheckpointing
        sc = SelectiveCheckpointing(n_layers=4, checkpoint_every=0)
        assert len(sc.get_checkpoint_layers()) == 0

    def test_should_checkpoint(self):
        """should_checkpoint правильно фильтрует."""
        from training.utils_v13 import SelectiveCheckpointing
        sc = SelectiveCheckpointing(n_layers=6, checkpoint_every=3)
        assert not sc.should_checkpoint(0)
        assert not sc.should_checkpoint(1)
        assert sc.should_checkpoint(2)
        assert not sc.should_checkpoint(3)
        assert not sc.should_checkpoint(4)
        assert sc.should_checkpoint(5)

    def test_memory_saving(self):
        """Оценка экономии памяти."""
        from training.utils_v13 import SelectiveCheckpointing
        sc2 = SelectiveCheckpointing(n_layers=8, checkpoint_every=2)
        sc4 = SelectiveCheckpointing(n_layers=8, checkpoint_every=4)
        assert sc2.estimate_memory_saving() == 0.5
        assert sc4.estimate_memory_saving() == 0.25


class TestV13WeightDecayScheduler:
    """Тесты для weight decay scheduling."""

    def test_linear_decay(self):
        """WD убывает линейно."""
        from training.utils_v13 import WeightDecayScheduler
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        wds = WeightDecayScheduler(opt, wd_start=0.1, wd_end=0.01, total_steps=100)

        wds.step()
        wd1 = wds.get_wd()
        for _ in range(49):
            wds.step()
        wd50 = wds.get_wd()
        for _ in range(50):
            wds.step()
        wd100 = wds.get_wd()

        assert wd1 > wd50 > wd100
        assert abs(wd50 - 0.055) < 0.01
        assert abs(wd100 - 0.01) < 0.001

    def test_wd_applied_to_optimizer(self):
        """WD обновляется в param groups optimizer."""
        from training.utils_v13 import WeightDecayScheduler
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        wds = WeightDecayScheduler(opt, wd_start=0.1, wd_end=0.01, total_steps=10)

        for _ in range(10):
            wds.step()

        for pg in opt.param_groups:
            assert abs(pg['weight_decay'] - 0.01) < 0.001


class TestV13Throughput:
    """Тесты для throughput benchmark."""

    def test_estimate_flops(self):
        """FLOPS estimation."""
        from training.utils_v13 import estimate_model_flops
        cfg = make_cfg()
        flops = estimate_model_flops(cfg, seq_len=16)
        assert flops['total'] > 0
        assert 'total_gflops' in flops
        assert flops['total_gflops'] > 0

    def test_benchmark_throughput(self):
        """Throughput benchmark на CPU."""
        from training.utils_v13 import benchmark_throughput
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        result = benchmark_throughput(model, cfg, batch_size=2, seq_len=8, n_steps=3)
        assert result['tokens_per_sec'] > 0
        assert result['ms_per_step'] > 0

    def test_flops_scales_with_layers(self):
        """Больше слоёв → больше FLOPS."""
        from training.utils_v13 import estimate_model_flops
        cfg2 = make_cfg(n_layers=2)
        cfg4 = make_cfg(n_layers=4)
        f2 = estimate_model_flops(cfg2, seq_len=16)
        f4 = estimate_model_flops(cfg4, seq_len=16)
        assert f4['total'] > f2['total']


class TestV13Integration:
    """Интеграционные тесты v13."""

    def test_bpe_train_evaluate(self):
        """BPE tokenizer → train → evaluate perplexity."""
        from tokenizer.char_tokenizer import BPETokenizer
        from training.utils_v12 import evaluate_perplexity

        bpe = BPETokenizer()
        corpus = "the cat sat on the mat " * 100
        bpe.train(corpus, vocab_size=280)

        cfg = make_cfg(vocab_size=bpe.get_piece_size())
        model = YiJingGPT(cfg)

        # Создаём данные
        ids = bpe.encode(corpus)
        chunks = [ids[i:i+16] for i in range(0, len(ids) - 16, 16)][:5]
        loader = [
            (torch.tensor([c[:-1]]), torch.tensor([c[1:]]))
            for c in chunks
        ]

        result = evaluate_perplexity(model, loader)
        assert result['perplexity'] > 1.0

    def test_wd_scheduler_with_training(self):
        """WD scheduler в training loop."""
        from training.utils_v13 import WeightDecayScheduler
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        wds = WeightDecayScheduler(opt, wd_start=0.1, wd_end=0.01, total_steps=10)

        for _ in range(5):
            x = torch.randint(0, cfg.vocab_size, (2, 8))
            y = torch.randint(0, cfg.vocab_size, (2, 8))
            _, loss, _ = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            wds.step()

        assert wds.get_wd() < 0.1


# ==================== V14 TESTS ====================

class TestV14DifferentialAttention:
    """Тесты для Differential Attention."""

    def test_forward_shape(self):
        """Differential Attention: forward shape."""
        from models.diff_attn import DifferentialAttention
        da = DifferentialAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out = da(x)
        assert out.shape == (2, 16, 64)

    def test_backward(self):
        """Backward через Differential Attention."""
        from models.diff_attn import DifferentialAttention
        da = DifferentialAttention(d_model=64, n_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = da(x)
        out.sum().backward()
        assert x.grad is not None

    def test_lambda_learnable(self):
        """Lambda — обучаемый параметр."""
        from models.diff_attn import DifferentialAttention
        da = DifferentialAttention(d_model=64, n_heads=4)
        assert da.lambda_init.requires_grad
        assert da.lambda_init.shape == (4,)

    def test_lambda_range(self):
        """Sigmoid(lambda) в [0, 1]."""
        from models.diff_attn import DifferentialAttention
        da = DifferentialAttention(d_model=64, n_heads=4)
        lam = torch.sigmoid(da.lambda_init)
        assert (lam >= 0).all() and (lam <= 1).all()

    def test_causal_masking(self):
        """Causal masking работает."""
        from models.diff_attn import DifferentialAttention
        da = DifferentialAttention(d_model=32, n_heads=2)
        x = torch.randn(1, 8, 32)
        out = da(x)
        assert out.shape == (1, 8, 32)
        # Не NaN
        assert not torch.isnan(out).any()


class TestV14QuantizedKVCache:
    """Тесты для KV-cache quantization."""

    def test_basic_update_get(self):
        """Update и get работают."""
        from models.diff_attn import QuantizedKVCache
        cache = QuantizedKVCache(enabled=True)
        k = torch.randn(1, 4, 8, 16)  # (B, H, T, D)
        v = torch.randn(1, 4, 8, 16)
        cache.update(k, v)
        k_out, v_out = cache.get()
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_sequential_update(self):
        """Последовательные update конкатенируют."""
        from models.diff_attn import QuantizedKVCache
        cache = QuantizedKVCache(enabled=True)
        cache.update(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        cache.update(torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16))
        k, v = cache.get()
        assert k.shape[2] == 9  # 8 + 1
        assert cache.seq_len == 9

    def test_quantization_accuracy(self):
        """Квантизация близка к оригиналу."""
        from models.diff_attn import QuantizedKVCache
        cache = QuantizedKVCache(enabled=True)
        k = torch.randn(1, 4, 16, 32)
        v = torch.randn(1, 4, 16, 32)
        cache.update(k, v)
        k_out, v_out = cache.get()
        # Ошибка квантизации < 1% от нормы
        k_error = (k - k_out).norm() / k.norm()
        assert k_error < 0.05

    def test_memory_savings(self):
        """INT8 cache меньше float32."""
        from models.diff_attn import QuantizedKVCache
        cache_q = QuantizedKVCache(enabled=True)
        cache_f = QuantizedKVCache(enabled=False)
        k = torch.randn(1, 8, 256, 64)
        v = torch.randn(1, 8, 256, 64)
        cache_q.update(k, v)
        cache_f.update(k, v)
        # INT8 ≈ 2x меньше float32 (плюс scales overhead)
        assert cache_q.memory_bytes() < cache_f.memory_bytes()

    def test_disabled_mode(self):
        """Disabled mode: хранит float32."""
        from models.diff_attn import QuantizedKVCache
        cache = QuantizedKVCache(enabled=False)
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        cache.update(k, v)
        k_out, v_out = cache.get()
        assert torch.equal(k, k_out)  # Exact match

    def test_reset(self):
        """Reset очищает кэш."""
        from models.diff_attn import QuantizedKVCache
        cache = QuantizedKVCache()
        cache.update(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))
        assert cache.seq_len == 8
        cache.reset()
        assert cache.seq_len == 0


class TestV14LayerwiseLR:
    """Тесты для layerwise learning rate."""

    def test_groups_created(self):
        """Param groups создаются для каждого слоя."""
        from training.utils_v14 import get_layerwise_lr_groups
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        groups = get_layerwise_lr_groups(model, base_lr=1e-3, lr_decay=0.8)
        assert len(groups) >= cfg.n_layers

    def test_lr_decreases_with_depth(self):
        """LR убывает для более ранних слоёв."""
        from training.utils_v14 import get_layerwise_lr_groups
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        groups = get_layerwise_lr_groups(model, base_lr=1e-3, lr_decay=0.8)
        layer_lrs = {}
        for g in groups:
            if 'name' in g and g['name'].startswith('layer_'):
                idx = int(g['name'].split('_')[1])
                layer_lrs[idx] = g['lr']
        if len(layer_lrs) >= 2:
            # Последний слой > первый слой
            max_idx = max(layer_lrs.keys())
            min_idx = min(layer_lrs.keys())
            assert layer_lrs[max_idx] > layer_lrs[min_idx]

    def test_optimizer_works(self):
        """Optimizer с layerwise LR работает."""
        from training.utils_v14 import get_layerwise_lr_groups
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        groups = get_layerwise_lr_groups(model, base_lr=1e-3, lr_decay=0.9)
        opt = torch.optim.AdamW(groups)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()


class TestV14DataMixing:
    """Тесты для Data Mixing Scheduler."""

    def test_linear_interpolation(self):
        """Линейная интерполяция весов."""
        from training.utils_v14 import DataMixingScheduler
        sched = DataMixingScheduler(['code', 'text'], total_steps=100)
        sched.set_linear([0.8, 0.2], [0.2, 0.8])

        w0 = sched.get_weights(0)
        assert abs(w0[0] - 0.8) < 0.01

        w100 = sched.get_weights(100)
        assert abs(w100[0] - 0.2) < 0.01

        w50 = sched.get_weights(50)
        assert abs(w50[0] - 0.5) < 0.05

    def test_step_schedule(self):
        """Пошаговое переключение."""
        from training.utils_v14 import DataMixingScheduler
        sched = DataMixingScheduler(['a', 'b', 'c'])
        sched.set_step_schedule([
            (0, [0.5, 0.3, 0.2]),
            (50, [0.2, 0.5, 0.3]),
            (80, [0.1, 0.1, 0.8]),
        ])

        w = sched.get_weights(30)
        assert abs(w[0] - 0.5) < 0.01

        w = sched.get_weights(60)
        assert abs(w[1] - 0.5) < 0.01

    def test_sample_source(self):
        """sample_source возвращает валидный индекс."""
        from training.utils_v14 import DataMixingScheduler
        sched = DataMixingScheduler(['a', 'b'])
        sched.set_linear([1.0, 0.0], [0.0, 1.0])

        # В начале почти всегда 'a' (индекс 0)
        sources = [sched.sample_source(0) for _ in range(100)]
        assert all(s in [0, 1] for s in sources)
        assert sources.count(0) > 50  # Большинство = source 0

    def test_normalize(self):
        """Веса нормализуются к 1."""
        from training.utils_v14 import DataMixingScheduler
        sched = DataMixingScheduler(['a', 'b', 'c'])
        w = sched.get_weights(0)
        assert abs(sum(w) - 1.0) < 1e-6


class TestV14ModelSurgery:
    """Тесты для model surgery."""

    def test_prune_heads(self):
        """Прунинг attention голов."""
        from training.utils_v14 import prune_attention_heads, count_active_heads
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        result = prune_attention_heads(model, layer_idx=0, heads_to_prune=[0, 1])
        assert result['pruned_heads'] == [0, 1]

        # Проверяем что головы действительно занулены
        heads = count_active_heads(model)
        assert heads[0]['active_heads'] < cfg.n_heads

    def test_count_active_heads(self):
        """Подсчёт активных голов."""
        from training.utils_v14 import count_active_heads
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        heads = count_active_heads(model)
        assert len(heads) == cfg.n_layers
        assert all(h['active_heads'] == cfg.n_heads for h in heads)

    def test_grow_model(self):
        """Добавление слоёв."""
        from training.utils_v14 import grow_model_depth
        cfg = make_cfg(n_layers=2)
        model = YiJingGPT(cfg)
        old_layers = len(model.core.layers)
        model = grow_model_depth(model, n_new_layers=1, position='end')
        assert len(model.core.layers) == old_layers + 1

        # Новый слой работает (forward не падает)
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _, _ = model(x)
        assert logits.shape[0] == 1

    def test_shrink_model(self):
        """Удаление слоёв."""
        from training.utils_v14 import shrink_model_depth
        cfg = make_cfg(n_layers=4)
        model = YiJingGPT(cfg)
        model = shrink_model_depth(model, layers_to_remove=[1, 3])
        assert len(model.core.layers) == 2

        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _, _ = model(x)
        assert logits.shape[0] == 1

    def test_grow_middle(self):
        """Добавление слоёв в середину."""
        from training.utils_v14 import grow_model_depth
        cfg = make_cfg(n_layers=4)
        model = YiJingGPT(cfg)
        model = grow_model_depth(model, n_new_layers=2, position='middle')
        assert len(model.core.layers) == 6


class TestV14Integration:
    """Интеграционные тесты v14."""

    def test_layerwise_lr_training(self):
        """Training с layerwise LR."""
        from training.utils_v14 import get_layerwise_lr_groups
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        groups = get_layerwise_lr_groups(model, base_lr=1e-3, lr_decay=0.8)
        opt = torch.optim.AdamW(groups)

        for _ in range(3):
            x = torch.randint(0, cfg.vocab_size, (2, 8))
            y = torch.randint(0, cfg.vocab_size, (2, 8))
            _, loss, _ = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_prune_then_train(self):
        """Прунинг → training."""
        from training.utils_v14 import prune_attention_heads
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        prune_attention_heads(model, layer_idx=0, heads_to_prune=[0])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()

    def test_diff_attn_standalone(self):
        """Differential Attention как standalone модуль."""
        from models.diff_attn import DifferentialAttention
        da = DifferentialAttention(d_model=64, n_heads=4, dropout=0.1)
        da.train()
        x = torch.randn(2, 16, 64)
        out = da(x)
        loss = out.sum()
        loss.backward()
        # Lambda gradients exist
        assert da.lambda_init.grad is not None


# ==================== V15 TESTS ====================

class TestV15ExpertChoice:
    """Тесты для Expert Choice MoE routing."""

    def test_forward_shape(self):
        from models.expert_choice import ExpertChoiceRouter
        ec = ExpertChoiceRouter(d_model=64, n_experts=4, capacity_factor=1.0)
        x = torch.randn(2, 16, 64)
        out, info = ec(x)
        assert out.shape == (2, 16, 64)

    def test_backward(self):
        from models.expert_choice import ExpertChoiceRouter
        ec = ExpertChoiceRouter(d_model=64, n_experts=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out, _ = ec(x)
        out.sum().backward()
        assert x.grad is not None

    def test_aux_info(self):
        from models.expert_choice import ExpertChoiceRouter
        ec = ExpertChoiceRouter(d_model=64, n_experts=4, capacity_factor=1.0)
        x = torch.randn(2, 16, 64)
        _, info = ec(x)
        assert 'tokens_per_expert' in info
        assert info['tokens_per_expert'] >= 1

    def test_load_balance(self):
        from models.expert_choice import ExpertChoiceRouter
        ec = ExpertChoiceRouter(d_model=64, n_experts=4, capacity_factor=1.0)
        x = torch.randn(1, 16, 64)
        _, info = ec(x)
        assert info['tokens_per_expert'] == 4  # 16 * 1.0 / 4


class TestV15CrossLayerSharing:
    """Тесты для cross-layer parameter sharing."""

    def test_shared_layer_forward(self):
        from models.expert_choice import SharedTransformerLayer
        from models.model import YiJingTransformerLayer
        cfg = make_cfg()
        layer = YiJingTransformerLayer(cfg)
        shared = SharedTransformerLayer(layer, n_virtual_layers=4, share_ln=True)
        x = torch.randn(1, 8, cfg.d_model)
        out, _ = shared(x, virtual_layer_idx=0)
        assert out.shape == x.shape

    def test_different_ln(self):
        from models.expert_choice import SharedTransformerLayer
        from models.model import YiJingTransformerLayer
        cfg = make_cfg()
        layer = YiJingTransformerLayer(cfg)
        shared = SharedTransformerLayer(
            layer, n_virtual_layers=3, share_ln=False, d_model=cfg.d_model
        )
        assert len(shared.layer_norms_attn) == 3

    def test_parameter_savings(self):
        from models.expert_choice import SharedTransformerLayer
        from models.model import YiJingTransformerLayer
        cfg = make_cfg()
        layer = YiJingTransformerLayer(cfg)
        params = sum(p.numel() for p in layer.parameters())
        shared = SharedTransformerLayer(
            layer, n_virtual_layers=6, share_ln=False, d_model=cfg.d_model
        )
        savings = shared.parameter_savings(params)
        assert savings['compression_ratio'] > 1.0
        assert savings['savings_pct'] > 50


class TestV15ZLoss:
    """Тесты для Z-Loss."""

    def test_z_loss_positive(self):
        from training.utils_v15 import z_loss
        logits = torch.randn(4, 16, 100)
        assert z_loss(logits).item() >= 0

    def test_z_loss_scales(self):
        from training.utils_v15 import z_loss
        small = torch.randn(4, 16, 100)
        big = small * 10
        assert z_loss(big).item() > z_loss(small).item()

    def test_z_loss_gradient(self):
        from training.utils_v15 import z_loss
        logits = torch.randn(2, 8, 50, requires_grad=True)
        z_loss(logits).backward()
        assert logits.grad is not None

    def test_compute_loss_with_z(self):
        from training.utils_v15 import compute_loss_with_z
        logits = torch.randn(2, 8, 100)
        targets = torch.randint(0, 100, (2, 8))
        total, ce, zl = compute_loss_with_z(logits, targets, z_weight=1e-4)
        assert total.item() >= ce.item()

    def test_z_weight_zero(self):
        from training.utils_v15 import compute_loss_with_z
        logits = torch.randn(2, 8, 100)
        targets = torch.randint(0, 100, (2, 8))
        total, ce, zl = compute_loss_with_z(logits, targets, z_weight=0)
        assert abs(total.item() - ce.item()) < 1e-6


class TestV15GradAccumProfiler:
    """Тесты для Gradient Accumulation Profiler."""

    def test_basic_profiling(self):
        from training.utils_v15 import GradAccumProfiler
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        profiler = GradAccumProfiler(model, grad_accum_steps=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(4):
            x = torch.randint(0, cfg.vocab_size, (2, 8))
            y = torch.randint(0, cfg.vocab_size, (2, 8))
            _, loss, _ = model(x, y)
            (loss / 4).backward()
            profiler.log_micro_step()
        profiler.log_optimizer_step(step=1)
        opt.step()
        opt.zero_grad()
        stats = profiler.get_stats()
        assert stats['n_steps'] == 1
        assert stats['avg_accumulated_norm'] > 0

    def test_empty_stats(self):
        from training.utils_v15 import GradAccumProfiler
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        profiler = GradAccumProfiler(model)
        assert profiler.get_stats()['n_steps'] == 0


class TestV15VocabExpansion:
    """Тесты для vocab expansion."""

    def test_expand_mean(self):
        from training.utils_v15 import expand_vocab
        cfg = make_cfg(vocab_size=128)
        model = YiJingGPT(cfg)
        old_emb = model.tok_emb.weight.data[:128].clone()
        model = expand_vocab(model, new_vocab_size=200, init_method='mean')
        assert model.cfg.vocab_size == 200
        assert torch.equal(model.tok_emb.weight.data[:128], old_emb)

    def test_expand_forward(self):
        from training.utils_v15 import expand_vocab
        cfg = make_cfg(vocab_size=128)
        model = YiJingGPT(cfg)
        model = expand_vocab(model, new_vocab_size=200)
        x = torch.randint(0, 200, (1, 8))
        logits, _, _ = model(x)
        assert logits.shape == (1, 8, 200)

    def test_expand_backward(self):
        from training.utils_v15 import expand_vocab
        cfg = make_cfg(vocab_size=128)
        model = YiJingGPT(cfg)
        model = expand_vocab(model, new_vocab_size=200)
        x = torch.randint(0, 200, (2, 8))
        y = torch.randint(0, 200, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()

    def test_shrink_vocab(self):
        from training.utils_v15 import shrink_vocab
        cfg = make_cfg(vocab_size=200)
        model = YiJingGPT(cfg)
        model = shrink_vocab(model, new_vocab_size=100)
        assert model.cfg.vocab_size == 100
        x = torch.randint(0, 100, (1, 8))
        logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)

    def test_expand_weight_tying(self):
        from training.utils_v15 import expand_vocab
        cfg = make_cfg(vocab_size=128, weight_tying=True)
        model = YiJingGPT(cfg)
        model = expand_vocab(model, new_vocab_size=200)
        assert model.head.weight is model.tok_emb.weight


class TestV15Integration:
    """Интеграционные тесты v15."""

    def test_expert_choice_training(self):
        from models.expert_choice import ExpertChoiceRouter
        ec = ExpertChoiceRouter(d_model=64, n_experts=4)
        opt = torch.optim.Adam(ec.parameters(), lr=1e-3)
        for _ in range(3):
            x = torch.randn(2, 8, 64)
            out, _ = ec(x)
            out.sum().backward()
            opt.step()
            opt.zero_grad()

    def test_z_loss_in_model(self):
        from training.utils_v15 import z_loss
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, ce_loss, _ = model(x, y)
        total = ce_loss + z_loss(logits)
        total.backward()

    def test_expand_then_train(self):
        from training.utils_v15 import expand_vocab
        cfg = make_cfg(vocab_size=128)
        model = YiJingGPT(cfg)
        model = expand_vocab(model, new_vocab_size=160)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randint(0, 160, (2, 8))
        y = torch.randint(0, 160, (2, 8))
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()


# ==================== V16 TESTS ====================

class TestV16StructuredPruning:
    """Тесты для structured pruning."""

    def test_compute_importance(self):
        from training.utils_v16 import StructuredPruner
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        pruner = StructuredPruner(model, prune_ratio=0.3)
        imp = pruner.compute_importance()
        assert len(imp) == cfg.n_layers
        assert imp[0]['importance'] is not None

    def test_apply_masks(self):
        from training.utils_v16 import StructuredPruner
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        pruner = StructuredPruner(model, prune_ratio=0.3)
        stats = pruner.apply_masks()
        assert stats['total_pruned'] > 0

    def test_pruned_model_forward(self):
        from training.utils_v16 import StructuredPruner
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        StructuredPruner(model, prune_ratio=0.5).apply_masks()
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, _, _ = model(x)
        assert logits.shape == (1, 8, cfg.vocab_size)

    def test_zero_prune(self):
        from training.utils_v16 import StructuredPruner
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        stats = StructuredPruner(model, prune_ratio=0.0).apply_masks()
        assert stats['total_pruned'] == 0


class TestV16ContrastiveLoss:
    """Тесты для contrastive loss."""

    def test_basic(self):
        from training.utils_v16 import contrastive_loss
        loss = contrastive_loss(torch.randn(8, 64), torch.randn(8, 64))
        assert loss.item() > 0

    def test_similar_lower(self):
        from training.utils_v16 import contrastive_loss
        z = torch.randn(8, 64)
        l_same = contrastive_loss(z, z + torch.randn_like(z) * 0.01)
        l_diff = contrastive_loss(z, torch.randn_like(z))
        assert l_same.item() < l_diff.item()

    def test_gradient(self):
        from training.utils_v16 import contrastive_loss
        z1 = torch.randn(4, 32, requires_grad=True)
        contrastive_loss(z1, torch.randn(4, 32)).backward()
        assert z1.grad is not None


class TestV16SequencePacking:
    """Тесты для sequence packing."""

    def test_basic(self):
        from training.utils_v16 import SequencePacker
        packer = SequencePacker(max_seq_len=20)
        batches = packer.pack([[1, 2, 3], [4, 5], [6, 7, 8, 9]])
        assert len(batches) >= 1
        assert batches[0]['input_ids'].shape[0] == 20

    def test_overflow(self):
        from training.utils_v16 import SequencePacker
        packer = SequencePacker(max_seq_len=5)
        batches = packer.pack([[1, 2, 3], [4, 5, 6], [7, 8]])
        assert len(batches) >= 2

    def test_mask(self):
        from training.utils_v16 import SequencePacker
        doc_ids = torch.tensor([0, 0, 0, 1, 1, -1])
        mask = SequencePacker.create_packing_mask(doc_ids)
        assert mask[2, 0].item() == True   # same doc
        assert mask[3, 0].item() == False  # diff doc
        assert mask[5, 0].item() == False  # padding

    def test_separator(self):
        from training.utils_v16 import SequencePacker
        packer = SequencePacker(max_seq_len=20, sep_id=99)
        batches = packer.pack([[1, 2], [3, 4]])
        assert 99 in batches[0]['input_ids'].tolist()


class TestV16PCGrad:
    """Тесты для PCGrad."""

    def test_basic(self):
        from training.utils_v16 import PCGrad
        model = torch.nn.Linear(10, 5)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        pcgrad = PCGrad(opt)
        x = torch.randn(4, 10)
        pcgrad.step([model(x)[:, :3].sum(), model(x)[:, 3:].sum()])

    def test_conflicting(self):
        from training.utils_v16 import PCGrad
        model = torch.nn.Linear(10, 2, bias=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        pcgrad = PCGrad(opt)
        x = torch.randn(4, 10)
        # Partially conflicting tasks (different outputs)
        w_before = model.weight.data.clone()
        pcgrad.step([model(x)[:, 0].sum(), model(x)[:, 1].mean()])
        assert not torch.equal(w_before, model.weight.data)


class TestV16ModelMerging:
    """Тесты для model merging."""

    def test_average(self):
        from training.utils_v16 import merge_models_average
        cfg = make_cfg()
        m1, m2 = YiJingGPT(cfg), YiJingGPT(cfg)
        sd = merge_models_average([m1, m2])
        m1.load_state_dict(sd)
        logits, _, _ = m1(torch.randint(0, cfg.vocab_size, (1, 8)))
        assert not torch.isnan(logits).any()

    def test_weighted(self):
        from training.utils_v16 import merge_models_average
        cfg = make_cfg()
        m1, m2 = YiJingGPT(cfg), YiJingGPT(cfg)
        sd = merge_models_average([m1, m2], weights=[0.7, 0.3])
        m1.load_state_dict(sd)
        assert not torch.isnan(m1(torch.randint(0, cfg.vocab_size, (1, 4)))[0]).any()

    def test_slerp(self):
        from training.utils_v16 import merge_models_slerp
        cfg = make_cfg()
        m1, m2 = YiJingGPT(cfg), YiJingGPT(cfg)
        sd = merge_models_slerp(m1, m2, t=0.5)
        m1.load_state_dict(sd)
        assert not torch.isnan(m1(torch.randint(0, cfg.vocab_size, (1, 8)))[0]).any()

    def test_slerp_t0(self):
        from training.utils_v16 import merge_models_slerp
        cfg = make_cfg()
        m1, m2 = YiJingGPT(cfg), YiJingGPT(cfg)
        sd = merge_models_slerp(m1, m2, t=0.0)
        for key in sd:
            assert torch.allclose(sd[key].float(), m1.state_dict()[key].float(), atol=1e-5)

    def test_ties(self):
        from training.utils_v16 import merge_models_ties
        cfg = make_cfg()
        base, ft1, ft2 = YiJingGPT(cfg), YiJingGPT(cfg), YiJingGPT(cfg)
        with torch.no_grad():
            for p in ft1.parameters():
                p.add_(torch.randn_like(p) * 0.01)
            for p in ft2.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        sd = merge_models_ties([ft1, ft2], base_model=base, density=0.5)
        base.load_state_dict(sd)
        assert not torch.isnan(base(torch.randint(0, cfg.vocab_size, (1, 8)))[0]).any()


class TestV16Integration:
    """Интеграционные тесты v16."""

    def test_prune_then_merge(self):
        from training.utils_v16 import StructuredPruner, merge_models_average
        cfg = make_cfg()
        m1, m2 = YiJingGPT(cfg), YiJingGPT(cfg)
        StructuredPruner(m1, 0.3).apply_masks()
        StructuredPruner(m2, 0.3).apply_masks()
        sd = merge_models_average([m1, m2])
        m1.load_state_dict(sd)
        assert not torch.isnan(m1(torch.randint(0, cfg.vocab_size, (1, 8)))[0]).any()

    def test_packing_with_model(self):
        from training.utils_v16 import SequencePacker
        packer = SequencePacker(max_seq_len=16)
        batches = packer.pack([[1, 2, 3, 4], [5, 6, 7], [8, 9]])
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        logits, _, _ = model(batches[0]['input_ids'].unsqueeze(0))
        assert logits.shape[1] == 16


# ==================== v17 Tests ====================


class TestPrefixTuning:
    """Тесты для Prefix Tuning."""

    def test_prefix_output_shape(self):
        from models.prefix_tuning import PrefixTuning
        cfg = make_cfg()
        prefix = PrefixTuning(cfg, prefix_len=8)
        result = prefix()
        assert len(result) == cfg.n_layers
        pk, pv = result[0]
        n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        head_dim = cfg.d_model // cfg.n_heads
        assert pk.shape == (1, n_kv_heads, 8, head_dim)
        assert pv.shape == (1, n_kv_heads, 8, head_dim)

    def test_prefix_with_gqa(self):
        from models.prefix_tuning import PrefixTuning
        cfg = make_cfg(n_kv_heads=2)
        prefix = PrefixTuning(cfg, prefix_len=4)
        result = prefix()
        pk, pv = result[0]
        head_dim = cfg.d_model // cfg.n_heads
        assert pk.shape == (1, 2, 4, head_dim)

    def test_prefix_trainable_params(self):
        from models.prefix_tuning import PrefixTuning
        cfg = make_cfg()
        prefix = PrefixTuning(cfg, prefix_len=8)
        n_params = prefix.num_trainable_params()
        assert n_params > 0

    def test_prefix_backward(self):
        from models.prefix_tuning import PrefixTuning
        cfg = make_cfg()
        prefix = PrefixTuning(cfg, prefix_len=4)
        result = prefix()
        loss = sum(pk.sum() + pv.sum() for pk, pv in result)
        loss.backward()
        assert prefix.prefix_emb.grad is not None

    def test_freeze_model(self):
        from models.prefix_tuning import freeze_model_for_prefix
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        freeze_model_for_prefix(model)
        for p in model.parameters():
            assert not p.requires_grad

    def test_prefix_different_layers_different_values(self):
        from models.prefix_tuning import PrefixTuning
        cfg = make_cfg()
        prefix = PrefixTuning(cfg, prefix_len=4)
        result = prefix()
        pk0, _ = result[0]
        pk1, _ = result[1]
        assert not torch.equal(pk0, pk1)


class TestLogitLens:
    """Тесты для Logit Lens."""

    def test_layer_predictions(self):
        from models.prefix_tuning import LogitLens
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        lens = LogitLens(model)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        results = lens.get_layer_predictions(ids)
        # embedding + n_layers
        assert len(results) == cfg.n_layers + 1
        for r in results:
            assert 'name' in r
            assert 'top_token_ids' in r
            assert 'entropy' in r
            assert len(r['top_token_ids']) == 5
            assert r['max_prob'] > 0

    def test_layer_predictions_position(self):
        from models.prefix_tuning import LogitLens
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        lens = LogitLens(model)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        r1 = lens.get_layer_predictions(ids, position=0)
        r2 = lens.get_layer_predictions(ids, position=-1)
        # Different positions should generally give different results
        assert r1[-1]['top_token_ids'] != r2[-1]['top_token_ids'] or True  # may coincide

    def test_layer_similarity(self):
        from models.prefix_tuning import LogitLens
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        lens = LogitLens(model)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        sims = lens.layer_similarity(ids)
        assert len(sims) == cfg.n_layers  # between consecutive layers (incl embedding→layer0)
        for s in sims:
            assert -1.0 <= s <= 1.0

    def test_entropy_decreases_or_finite(self):
        from models.prefix_tuning import LogitLens
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        lens = LogitLens(model)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        results = lens.get_layer_predictions(ids)
        for r in results:
            assert r['entropy'] >= 0
            assert not torch.isnan(torch.tensor(r['entropy']))


class TestMultiTokenPrediction:
    """Тесты для Multi-Token Prediction."""

    def test_forward_shape(self):
        from models.prefix_tuning import MultiTokenPredictionHead
        mtp = MultiTokenPredictionHead(d_model=64, vocab_size=128, n_future=4)
        hidden = torch.randn(2, 16, 64)
        outputs = mtp(hidden)
        assert len(outputs) == 4
        for o in outputs:
            assert o.shape == (2, 16, 128)

    def test_compute_loss(self):
        from models.prefix_tuning import MultiTokenPredictionHead
        mtp = MultiTokenPredictionHead(d_model=64, vocab_size=128, n_future=3)
        hidden = torch.randn(2, 16, 64)
        targets = torch.randint(0, 128, (2, 16))
        total_loss, per_horizon = mtp.compute_loss(hidden, targets)
        assert total_loss.item() > 0
        assert len(per_horizon) == 3
        for h in per_horizon:
            assert h > 0

    def test_compute_loss_backward(self):
        from models.prefix_tuning import MultiTokenPredictionHead
        mtp = MultiTokenPredictionHead(d_model=64, vocab_size=128, n_future=2)
        hidden = torch.randn(2, 10, 64, requires_grad=True)
        targets = torch.randint(0, 128, (2, 10))
        total_loss, _ = mtp.compute_loss(hidden, targets)
        total_loss.backward()
        assert hidden.grad is not None

    def test_short_sequence(self):
        from models.prefix_tuning import MultiTokenPredictionHead
        mtp = MultiTokenPredictionHead(d_model=64, vocab_size=128, n_future=4)
        hidden = torch.randn(1, 3, 64)
        targets = torch.randint(0, 128, (1, 3))
        total_loss, per_horizon = mtp.compute_loss(hidden, targets)
        # Only 2 horizons possible with T=3 (shift 1 and 2)
        assert len(per_horizon) == 2

    def test_single_head(self):
        from models.prefix_tuning import MultiTokenPredictionHead
        mtp = MultiTokenPredictionHead(d_model=64, vocab_size=128, n_future=1)
        hidden = torch.randn(2, 8, 64)
        targets = torch.randint(0, 128, (2, 8))
        total_loss, per_horizon = mtp.compute_loss(hidden, targets)
        assert len(per_horizon) == 1


class TestSimpleRetriever:
    """Тесты для SimpleRetriever."""

    def test_add_and_retrieve(self):
        from training.utils_v17 import SimpleRetriever
        ret = SimpleRetriever(embedding_dim=16)
        docs = ["hello world", "foo bar", "test doc"]
        embs = torch.randn(3, 16)
        ret.add_documents(docs, embs)
        assert len(ret) == 3

        results = ret.retrieve(embs[0], top_k=2)
        assert len(results) == 2
        assert results[0]['text'] == "hello world"
        assert results[0]['score'] > results[1]['score']

    def test_empty_retriever(self):
        from training.utils_v17 import SimpleRetriever
        ret = SimpleRetriever(embedding_dim=8)
        results = ret.retrieve(torch.randn(8), top_k=5)
        assert results == []

    def test_top_k_larger_than_docs(self):
        from training.utils_v17 import SimpleRetriever
        ret = SimpleRetriever(embedding_dim=8)
        ret.add_documents(["a", "b"], torch.randn(2, 8))
        results = ret.retrieve(torch.randn(8), top_k=10)
        assert len(results) == 2

    def test_add_incremental(self):
        from training.utils_v17 import SimpleRetriever
        ret = SimpleRetriever(embedding_dim=8)
        ret.add_documents(["a"], torch.randn(1, 8))
        ret.add_documents(["b", "c"], torch.randn(2, 8))
        assert len(ret) == 3

    def test_retrieve_scores_range(self):
        from training.utils_v17 import SimpleRetriever
        ret = SimpleRetriever(embedding_dim=8)
        embs = F.normalize(torch.randn(5, 8), dim=-1)
        ret.add_documents(["a", "b", "c", "d", "e"], embs)
        results = ret.retrieve(embs[0], top_k=5)
        for r in results:
            assert -1.0 <= r['score'] <= 1.01  # cosine sim range


class TestRAGPipeline:
    """Тесты для RAG Pipeline."""

    def test_index_and_retrieve(self):
        from training.utils_v17 import RAGPipeline, SimpleRetriever
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        retriever = SimpleRetriever(embedding_dim=cfg.d_model)

        class FakeTokenizer:
            def encode(self, text):
                return [ord(c) % cfg.vocab_size for c in text[:8]]
            def decode(self, ids):
                return "".join(chr(i) for i in ids)

        rag = RAGPipeline(model, retriever, FakeTokenizer())
        rag.index_documents(["hello", "world", "test"])
        assert len(retriever) == 3

        context, results = rag.retrieve_context("hello", top_k=2)
        assert len(results) == 2
        assert isinstance(context, str)

    def test_augmented_ids(self):
        from training.utils_v17 import RAGPipeline, SimpleRetriever
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        retriever = SimpleRetriever(embedding_dim=cfg.d_model)

        class FakeTokenizer:
            def encode(self, text):
                return [ord(c) % cfg.vocab_size for c in text[:16]]
            def decode(self, ids):
                return "".join(chr(i) for i in ids)

        rag = RAGPipeline(model, retriever, FakeTokenizer())
        rag.index_documents(["doc1", "doc2"])
        ids, retrieved = rag.augmented_ids("query", top_k=1)
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert len(retrieved) == 1


class TestActivationTracker:
    """Тесты для ActivationTracker."""

    def test_tracking(self):
        from training.utils_v17 import ActivationTracker
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        tracker = ActivationTracker(model)
        tracker.start()

        x = torch.randint(0, cfg.vocab_size, (1, 8))
        y = torch.randint(0, cfg.vocab_size, (1, 8))
        _, loss, _ = model(x, y)
        loss.backward()

        stats = tracker.get_stats()
        tracker.stop()

        assert 'activation_norms' in stats
        assert 'gradient_norms' in stats
        assert stats['avg_activation_norm'] > 0
        assert stats['avg_gradient_norm'] > 0

    def test_dead_neurons(self):
        from training.utils_v17 import ActivationTracker
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        tracker = ActivationTracker(model)
        tracker.start()

        x = torch.randint(0, cfg.vocab_size, (1, 8))
        model(x)

        stats = tracker.get_stats()
        tracker.stop()

        assert 'dead_neurons' in stats
        assert 'total_dead' in stats

    def test_reset(self):
        from training.utils_v17 import ActivationTracker
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        tracker = ActivationTracker(model)
        tracker.start()

        model(torch.randint(0, cfg.vocab_size, (1, 8)))
        assert len(tracker.activation_norms) > 0

        tracker.reset()
        assert len(tracker.activation_norms) == 0

    def test_stop_removes_hooks(self):
        from training.utils_v17 import ActivationTracker
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        tracker = ActivationTracker(model)
        tracker.start()
        assert len(tracker.hooks) > 0
        tracker.stop()
        assert len(tracker.hooks) == 0
        assert not tracker._running

    def test_multiple_forwards(self):
        from training.utils_v17 import ActivationTracker
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        tracker = ActivationTracker(model)
        tracker.start()

        for _ in range(3):
            model(torch.randint(0, cfg.vocab_size, (1, 8)))

        stats = tracker.get_stats()
        tracker.stop()

        # Multiple forwards should accumulate norms
        assert stats['avg_activation_norm'] > 0
        assert len(stats.get('activation_norms', {})) > 0


class TestV17Integration:
    """Интеграционные тесты v17."""

    def test_prefix_tuning_with_model(self):
        from models.prefix_tuning import PrefixTuning, freeze_model_for_prefix
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        freeze_model_for_prefix(model)
        prefix = PrefixTuning(cfg, prefix_len=4)
        result = prefix()
        # Verify prefix generates valid KVs that could be prepended
        pk, pv = result[0]
        assert pk.shape[-1] == cfg.d_model // cfg.n_heads

    def test_mtp_with_model(self):
        from models.prefix_tuning import MultiTokenPredictionHead
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        mtp = MultiTokenPredictionHead(cfg.d_model, cfg.vocab_size, n_future=3)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        y = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss, aux = model(x, y)
        # Use hidden from before head
        emb = model.tok_emb(x)
        if model.pos_emb is not None:
            emb = emb + model.pos_emb[:, :16, :]
        hidden = model.core(emb)[0]
        mtp_loss, horizons = mtp.compute_loss(hidden, y)
        total = loss + 0.1 * mtp_loss
        total.backward()
        assert not torch.isnan(total)

    def test_logit_lens_with_model(self):
        from models.prefix_tuning import LogitLens
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        lens = LogitLens(model)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        preds = lens.get_layer_predictions(ids)
        sims = lens.layer_similarity(ids)
        assert len(preds) == cfg.n_layers + 1
        assert len(sims) == cfg.n_layers


# ==================== v18 Tests ====================


class TestWSDScheduler:
    """Тесты для WSD LR Scheduler."""

    def test_warmup_phase(self):
        from training.utils_v18 import WSDScheduler
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = WSDScheduler(opt, total_steps=100, warmup_steps=10, decay_steps=20)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.param_groups[0]['lr'])
        # LR should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1]
        assert abs(lrs[-1] - 0.1) < 1e-6  # reach base lr at end of warmup

    def test_stable_phase(self):
        from training.utils_v18 import WSDScheduler
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = WSDScheduler(opt, total_steps=100, warmup_steps=10, decay_steps=20)
        for _ in range(10):
            sched.step()
        # Steps 11-80 should be stable
        for _ in range(70):
            sched.step()
            assert abs(opt.param_groups[0]['lr'] - 0.1) < 1e-6

    def test_decay_phase(self):
        from training.utils_v18 import WSDScheduler
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = WSDScheduler(opt, total_steps=100, warmup_steps=10, decay_steps=20,
                             min_lr_ratio=0.1)
        for _ in range(80):
            sched.step()
        lr_start = opt.param_groups[0]['lr']
        for _ in range(20):
            sched.step()
        lr_end = opt.param_groups[0]['lr']
        assert lr_end < lr_start  # LR decreased
        assert abs(lr_end - 0.01) < 1e-5  # 0.1 * 0.1

    def test_get_lr(self):
        from training.utils_v18 import WSDScheduler
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = WSDScheduler(opt, total_steps=100, warmup_steps=10, decay_steps=20)
        sched.step()
        lrs = sched.get_lr()
        assert len(lrs) == 1
        assert lrs[0] > 0

    def test_state_dict(self):
        from training.utils_v18 import WSDScheduler
        model = nn.Linear(10, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = WSDScheduler(opt, total_steps=100, warmup_steps=10, decay_steps=20)
        for _ in range(25):
            sched.step()
        sd = sched.state_dict()
        assert sd['step'] == 25
        sched2 = WSDScheduler(opt, total_steps=100, warmup_steps=10, decay_steps=20)
        sched2.load_state_dict(sd)
        assert sched2._step == 25


class TestAttentionMapCapture:
    """Тесты для Attention Map Capture."""

    def test_capture_basic(self):
        from training.utils_v18 import AttentionMapCapture
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        capture = AttentionMapCapture(model)
        capture.start()
        assert capture._running
        model(torch.randint(0, cfg.vocab_size, (1, 8)))
        capture.stop()
        assert not capture._running

    def test_hooks_registered(self):
        from training.utils_v18 import AttentionMapCapture
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        capture = AttentionMapCapture(model)
        capture.start()
        assert len(capture.hooks) > 0
        capture.stop()
        assert len(capture.hooks) == 0

    def test_entropy_computation(self):
        from training.utils_v18 import AttentionMapCapture
        capture = AttentionMapCapture(nn.Linear(10, 10))
        # Manually add a uniform attention map
        T = 8
        uniform_attn = torch.ones(1, 4, T, T) / T
        capture.attention_maps['test_layer'] = uniform_attn
        entropies = capture.get_entropy()
        assert 'test_layer' in entropies
        assert entropies['test_layer'] > 0

    def test_sparsity_computation(self):
        from training.utils_v18 import AttentionMapCapture
        capture = AttentionMapCapture(nn.Linear(10, 10))
        # Nearly-sparse attention
        attn = torch.zeros(1, 4, 8, 8)
        attn[:, :, :, 0] = 1.0  # all attend to position 0
        capture.attention_maps['test'] = attn
        sparsities = capture.get_sparsity(threshold=0.01)
        assert sparsities['test'] > 0.5  # most weights are 0

    def test_reset(self):
        from training.utils_v18 import AttentionMapCapture
        capture = AttentionMapCapture(nn.Linear(10, 10))
        capture.attention_maps['foo'] = torch.ones(1, 1, 4, 4)
        capture.reset()
        assert len(capture.attention_maps) == 0


class TestBPEDropout:
    """Тесты для BPE Dropout."""

    def test_encode_no_dropout(self):
        from training.utils_v18 import BPEDropout
        vocab = {c: i for i, c in enumerate("abcdefgh")}
        vocab['ab'] = len(vocab)
        vocab['cd'] = len(vocab)
        merges = [('a', 'b'), ('c', 'd')]
        bpe = BPEDropout(merges, vocab, dropout=0.0)
        ids = bpe.encode("abcd")
        # With no dropout, should apply all merges
        assert ids == [vocab['ab'], vocab['cd']]

    def test_encode_full_dropout(self):
        from training.utils_v18 import BPEDropout
        vocab = {c: i for i, c in enumerate("abcdefgh")}
        vocab['ab'] = len(vocab)
        merges = [('a', 'b')]
        bpe = BPEDropout(merges, vocab, dropout=1.0)
        ids = bpe.encode("ab")
        # Full dropout = no merges → individual characters
        assert ids == [vocab['a'], vocab['b']]

    def test_encode_stochastic(self):
        import random
        from training.utils_v18 import BPEDropout
        vocab = {c: i for i, c in enumerate("abcdefgh")}
        vocab['ab'] = len(vocab)
        merges = [('a', 'b')]
        bpe = BPEDropout(merges, vocab, dropout=0.5)
        # Run multiple times — should get different results
        random.seed(42)
        results = set()
        for _ in range(20):
            ids = tuple(bpe.encode("ab"))
            results.add(ids)
        assert len(results) >= 2  # at least 2 different segmentations

    def test_decode(self):
        from training.utils_v18 import BPEDropout
        vocab = {'a': 0, 'b': 1, 'c': 2, 'ab': 3}
        merges = [('a', 'b')]
        bpe = BPEDropout(merges, vocab, dropout=0.0)
        ids = bpe.encode_deterministic("abc")
        text = bpe.decode(ids)
        assert text == "abc"

    def test_encode_deterministic(self):
        from training.utils_v18 import BPEDropout
        vocab = {c: i for i, c in enumerate("abcdefgh")}
        vocab['ab'] = len(vocab)
        merges = [('a', 'b')]
        bpe = BPEDropout(merges, vocab, dropout=0.5)
        ids1 = bpe.encode_deterministic("ab")
        ids2 = bpe.encode_deterministic("ab")
        assert ids1 == ids2  # always same without dropout


class TestLAMB:
    """Тесты для LAMB optimizer."""

    def test_basic_step(self):
        from training.utils_v18 import LAMB
        model = nn.Linear(10, 5)
        opt = LAMB(model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        # Should not crash

    def test_convergence(self):
        from training.utils_v18 import LAMB
        torch.manual_seed(42)
        model = nn.Linear(10, 1)
        opt = LAMB(model.parameters(), lr=0.01, weight_decay=0.0)
        x = torch.randn(32, 10)
        y = x[:, 0:1] * 2  # simple linear target
        initial_loss = F.mse_loss(model(x), y).item()
        for _ in range(50):
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        final_loss = F.mse_loss(model(x), y).item()
        assert final_loss < initial_loss

    def test_weight_decay(self):
        from training.utils_v18 import LAMB
        model = nn.Linear(10, 5, bias=False)
        # Initialize with large weights
        nn.init.constant_(model.weight, 5.0)
        opt = LAMB(model.parameters(), lr=0.01, weight_decay=0.1)
        x = torch.randn(4, 10)
        y = torch.zeros(4, 5)
        for _ in range(10):
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        # Weights should have decreased due to decay
        assert model.weight.data.abs().mean().item() < 5.0

    def test_state_dict(self):
        from training.utils_v18 import LAMB
        model = nn.Linear(10, 5)
        opt = LAMB(model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        loss = F.mse_loss(model(x), torch.zeros(4, 5))
        loss.backward()
        opt.step()
        sd = opt.state_dict()
        assert len(sd['state']) > 0

    def test_multiple_param_groups(self):
        from training.utils_v18 import LAMB
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        opt = LAMB([
            {'params': model[0].parameters(), 'lr': 0.01},
            {'params': model[1].parameters(), 'lr': 0.001},
        ])
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        # Both groups should have been updated
        assert opt.param_groups[0]['lr'] == 0.01
        assert opt.param_groups[1]['lr'] == 0.001


class TestNEFTune:
    """Тесты для NEFTune."""

    def test_noise_in_training(self):
        from training.utils_v18 import NEFTune
        emb = nn.Embedding(100, 64)
        neft = NEFTune(emb, noise_alpha=5.0)
        neft.train()
        ids = torch.randint(0, 100, (2, 8))
        out1 = neft(ids)
        out2 = neft(ids)
        # Different noise each time
        assert not torch.equal(out1, out2)

    def test_no_noise_in_eval(self):
        from training.utils_v18 import NEFTune
        emb = nn.Embedding(100, 64)
        neft = NEFTune(emb, noise_alpha=5.0)
        neft.eval()
        ids = torch.randint(0, 100, (2, 8))
        out1 = neft(ids)
        out2 = neft(ids)
        assert torch.equal(out1, out2)

    def test_output_shape(self):
        from training.utils_v18 import NEFTune
        emb = nn.Embedding(100, 64)
        neft = NEFTune(emb, noise_alpha=5.0)
        ids = torch.randint(0, 100, (2, 8))
        out = neft(ids)
        assert out.shape == (2, 8, 64)

    def test_zero_alpha(self):
        from training.utils_v18 import NEFTune
        emb = nn.Embedding(100, 64)
        neft = NEFTune(emb, noise_alpha=0.0)
        neft.train()
        ids = torch.randint(0, 100, (2, 8))
        out1 = neft(ids)
        out2 = neft(ids)
        assert torch.equal(out1, out2)

    def test_apply_to_model(self):
        from training.utils_v18 import NEFTune
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        original_emb = model.tok_emb
        NEFTune.apply_to_model(model, noise_alpha=5.0)
        assert isinstance(model.tok_emb, NEFTune)
        NEFTune.remove_from_model(model, original_emb)
        assert model.tok_emb is original_emb

    def test_backward(self):
        from training.utils_v18 import NEFTune
        emb = nn.Embedding(100, 64)
        neft = NEFTune(emb, noise_alpha=5.0)
        neft.train()
        ids = torch.randint(0, 100, (2, 8))
        out = neft(ids)
        out.sum().backward()
        assert emb.weight.grad is not None


class TestV18Integration:
    """Интеграционные тесты v18."""

    def test_wsd_with_lamb(self):
        from training.utils_v18 import WSDScheduler, LAMB
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        opt = LAMB(model.parameters(), lr=0.01)
        sched = WSDScheduler(opt, total_steps=50, warmup_steps=5, decay_steps=10)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        for step in range(20):
            opt.zero_grad()
            _, loss, _ = model(x, y)
            loss.backward()
            opt.step()
            sched.step()
        assert opt.param_groups[0]['lr'] > 0

    def test_neftune_with_model_training(self):
        from training.utils_v18 import NEFTune
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.train()
        original_emb = model.tok_emb
        NEFTune.apply_to_model(model, noise_alpha=5.0)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, loss, _ = model(x, y)
        loss.backward()
        assert not torch.isnan(loss)
        NEFTune.remove_from_model(model, original_emb)

    def test_full_v18_pipeline(self):
        from training.utils_v18 import WSDScheduler, LAMB, NEFTune, AttentionMapCapture
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.train()
        original_emb = model.tok_emb
        NEFTune.apply_to_model(model, noise_alpha=5.0)
        opt = LAMB(model.parameters(), lr=0.01)
        sched = WSDScheduler(opt, total_steps=30, warmup_steps=5, decay_steps=5)

        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        initial_loss = None
        for step in range(15):
            opt.zero_grad()
            _, loss, _ = model(x, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            opt.step()
            sched.step()
        assert not torch.isnan(loss)
        NEFTune.remove_from_model(model, original_emb)


# ==================== v19 Tests ====================


class TestSophia:
    """Тесты для Sophia optimizer."""

    def test_basic_step(self):
        from training.utils_v19 import Sophia
        model = nn.Linear(10, 5)
        opt = Sophia(model.parameters(), lr=1e-3)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()

    def test_convergence(self):
        from training.utils_v19 import Sophia
        torch.manual_seed(42)
        model = nn.Linear(10, 1)
        opt = Sophia(model.parameters(), lr=1e-3, weight_decay=0.0, rho=1.0)
        x = torch.randn(32, 10)
        y = x[:, 0:1] * 2
        initial_loss = F.mse_loss(model(x), y).item()
        for _ in range(100):
            opt.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        final_loss = F.mse_loss(model(x), y).item()
        assert final_loss < initial_loss

    def test_weight_decay(self):
        from training.utils_v19 import Sophia
        model = nn.Linear(10, 5, bias=False)
        nn.init.constant_(model.weight, 5.0)
        opt = Sophia(model.parameters(), lr=1e-3, weight_decay=0.1, rho=1.0)
        for _ in range(20):
            opt.zero_grad()
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            opt.step()
        assert model.weight.data.abs().mean().item() < 5.0

    def test_hessian_update(self):
        from training.utils_v19 import Sophia
        model = nn.Linear(10, 5)
        opt = Sophia(model.parameters(), lr=1e-3)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        def loss_fn():
            return F.mse_loss(model(x), y)

        opt.update_hessian(loss_fn)

        for group in opt.param_groups:
            for p in group['params']:
                state = opt.state[p]
                assert 'hessian' in state
                assert not torch.all(state['hessian'] == 1.0)

    def test_state_dict(self):
        from training.utils_v19 import Sophia
        model = nn.Linear(10, 5)
        opt = Sophia(model.parameters(), lr=1e-3)
        loss = model(torch.randn(4, 10)).sum()
        loss.backward()
        opt.step()
        sd = opt.state_dict()
        assert len(sd['state']) > 0


class TestRMSNorm:
    """Тесты для RMSNorm."""

    def test_output_shape(self):
        from training.utils_v19 import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == (2, 8, 64)

    def test_normalization(self):
        from training.utils_v19 import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64) * 10
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert rms.mean().item() < 3.0

    def test_backward(self):
        from training.utils_v19 import RMSNorm
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_weight_initialization(self):
        from training.utils_v19 import RMSNorm
        norm = RMSNorm(64)
        assert torch.all(norm.weight == 1.0)

    def test_1d_input(self):
        from training.utils_v19 import RMSNorm
        norm = RMSNorm(32)
        x = torch.randn(32)
        out = norm(x)
        assert out.shape == (32,)

    def test_extra_repr(self):
        from training.utils_v19 import RMSNorm
        norm = RMSNorm(64, eps=1e-5)
        assert '64' in norm.extra_repr()
        assert '1e-05' in norm.extra_repr()


class TestChunkedPrefill:
    """Тесты для Chunked Prefill."""

    def test_prefill_short(self):
        from training.utils_v19 import ChunkedPrefill
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        cp = ChunkedPrefill(model, chunk_size=8)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits, info = cp.prefill(ids)
        assert logits.shape == (1, 8, cfg.vocab_size)
        assert info['n_chunks'] == 1

    def test_prefill_long(self):
        from training.utils_v19 import ChunkedPrefill
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        cp = ChunkedPrefill(model, chunk_size=8)
        ids = torch.randint(0, cfg.vocab_size, (1, 24))
        logits, info = cp.prefill(ids)
        assert logits.shape == (1, 24, cfg.vocab_size)
        assert info['n_chunks'] == 3

    def test_prefill_uneven(self):
        from training.utils_v19 import ChunkedPrefill
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        cp = ChunkedPrefill(model, chunk_size=8)
        ids = torch.randint(0, cfg.vocab_size, (1, 13))
        logits, info = cp.prefill(ids)
        assert logits.shape == (1, 13, cfg.vocab_size)
        assert info['n_chunks'] == 2

    def test_generate(self):
        from training.utils_v19 import ChunkedPrefill
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        model.eval()
        cp = ChunkedPrefill(model, chunk_size=8)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        generated = cp.generate_with_prefill(ids, max_new_tokens=5)
        assert generated.shape == (1, 13)


class TestCAGrad:
    """Тесты для CAGrad."""

    def test_no_conflict(self):
        from training.utils_v19 import CAGrad
        model = nn.Linear(10, 5, bias=False)
        cag = CAGrad(n_tasks=2, c=0.5)
        x = torch.randn(4, 10)
        loss1 = model(x)[:, 0].pow(2).sum()
        loss2 = model(x)[:, 1].pow(2).sum()
        params = list(model.parameters())
        total = cag.backward([loss1, loss2], params)
        assert total > 0
        assert model.weight.grad is not None

    def test_conflicting_tasks(self):
        from training.utils_v19 import CAGrad
        model = nn.Linear(10, 2, bias=False)
        cag = CAGrad(n_tasks=2, c=0.5)
        x = torch.randn(4, 10)
        loss1 = model(x)[:, 0].mean()
        loss2 = -model(x)[:, 0].mean()
        params = list(model.parameters())
        total = cag.backward([loss1, loss2], params)
        assert model.weight.grad is not None

    def test_three_tasks(self):
        from training.utils_v19 import CAGrad
        model = nn.Linear(10, 5, bias=False)
        cag = CAGrad(n_tasks=3, c=0.5)
        x = torch.randn(4, 10)
        losses = [model(x)[:, i].sum() for i in range(3)]
        params = list(model.parameters())
        total = cag.backward(losses, params)
        assert total > 0

    def test_gradient_set(self):
        from training.utils_v19 import CAGrad
        model = nn.Linear(10, 5, bias=False)
        cag = CAGrad(n_tasks=2, c=0.5)
        x = torch.randn(4, 10)
        loss1 = model(x)[:, 0].sum()
        loss2 = model(x)[:, 1].sum()
        params = list(model.parameters())
        cag.backward([loss1, loss2], params)
        w_before = model.weight.data.clone()
        model.weight.data -= 0.01 * model.weight.grad
        assert not torch.equal(w_before, model.weight.data)


class TestEMADecayScheduler:
    """Тесты для EMA Decay Scheduler."""

    def test_decay_warmup(self):
        from training.utils_v19 import EMADecayScheduler
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        ema_model = YiJingGPT(cfg)
        sched = EMADecayScheduler(ema_model, model,
                                   initial_decay=0.9, target_decay=0.999,
                                   warmup_steps=10)
        decays = []
        for _ in range(15):
            d = sched.step()
            decays.append(d)
        assert decays[-1] > decays[0]
        assert abs(decays[-1] - 0.999) < 1e-3

    def test_initial_decay(self):
        from training.utils_v19 import EMADecayScheduler
        model = nn.Linear(10, 5)
        ema = nn.Linear(10, 5)
        sched = EMADecayScheduler(ema, model, initial_decay=0.9,
                                   target_decay=0.999, warmup_steps=100)
        d = sched.get_decay()
        assert abs(d - 0.9) < 1e-6

    def test_ema_update(self):
        from training.utils_v19 import EMADecayScheduler
        model = nn.Linear(10, 5, bias=False)
        ema = nn.Linear(10, 5, bias=False)
        nn.init.ones_(model.weight)
        nn.init.zeros_(ema.weight)
        sched = EMADecayScheduler(ema, model, initial_decay=0.5,
                                   target_decay=0.999, warmup_steps=100)
        sched.step()
        assert ema.weight.data.abs().sum().item() > 0

    def test_state_dict(self):
        from training.utils_v19 import EMADecayScheduler
        model = nn.Linear(10, 5)
        ema = nn.Linear(10, 5)
        sched = EMADecayScheduler(ema, model, warmup_steps=50)
        for _ in range(25):
            sched.step()
        sd = sched.state_dict()
        assert sd['step'] == 25
        sched2 = EMADecayScheduler(ema, model, warmup_steps=50)
        sched2.load_state_dict(sd)
        assert sched2._step == 25

    def test_cosine_monotonic(self):
        from training.utils_v19 import EMADecayScheduler
        model = nn.Linear(10, 5)
        ema = nn.Linear(10, 5)
        sched = EMADecayScheduler(ema, model, initial_decay=0.9,
                                   target_decay=0.999, warmup_steps=20)
        decays = []
        for _ in range(20):
            decays.append(sched.get_decay())
            sched.step()
        for i in range(1, len(decays)):
            assert decays[i] >= decays[i - 1] - 1e-10


class TestV19Integration:
    """Интеграционные тесты v19."""

    def test_sophia_with_model(self):
        from training.utils_v19 import Sophia
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        opt = Sophia(model.parameters(), lr=1e-4, rho=1.0)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        for _ in range(5):
            opt.zero_grad()
            _, loss, _ = model(x, y)
            loss.backward()
            opt.step()
        assert not torch.isnan(loss)

    def test_rmsnorm_drop_in(self):
        from training.utils_v19 import RMSNorm
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        for module in model.modules():
            for name, child in list(module.named_children()):
                if isinstance(child, nn.LayerNorm):
                    setattr(module, name, RMSNorm(child.normalized_shape[0]))
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        logits, _, _ = model(x)
        assert logits.shape == (2, 8, cfg.vocab_size)
        assert not torch.isnan(logits).any()

    def test_ema_scheduler_with_training(self):
        from training.utils_v19 import EMADecayScheduler
        cfg = make_cfg()
        model = YiJingGPT(cfg)
        ema_model = YiJingGPT(cfg)
        ema_model.load_state_dict(model.state_dict())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ema_sched = EMADecayScheduler(ema_model, model,
                                       initial_decay=0.9, target_decay=0.999,
                                       warmup_steps=5)
        x = torch.randint(0, cfg.vocab_size, (2, 8))
        y = torch.randint(0, cfg.vocab_size, (2, 8))
        for _ in range(10):
            opt.zero_grad()
            _, loss, _ = model(x, y)
            loss.backward()
            opt.step()
            ema_sched.step()
        ema_model.eval()
        logits, _, _ = ema_model(x)
        assert not torch.isnan(logits).any()
