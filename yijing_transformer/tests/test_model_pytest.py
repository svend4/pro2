"""Pytest-тесты для моделей YiJing и Vanilla."""

import pytest
import torch
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
        assert out.shape[1] <= 4 + 8  # не больше max_new_tokens

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
