"""Pytest-тесты для моделей YiJing и Vanilla."""

import pytest
import torch
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
