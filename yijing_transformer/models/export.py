"""
Утилиты экспорта для YiJing-Transformer.

Поддерживает:
- ONNX export (для inference на ONNX Runtime)
- TorchScript export (для C++/мобильный inference)
- Модельная карточка с метриками

Использование:
    model = YiJingGPT.from_pretrained("model.pt")
    export_onnx(model, "model.onnx", seq_len=128)
    export_torchscript(model, "model.pt", seq_len=128)
"""

import torch
import json
from dataclasses import asdict


def export_onnx(model, path, seq_len=128, opset_version=14):
    """
    Экспортирует модель в ONNX формат.

    Args:
        model: YiJingGPT
        path: путь для сохранения .onnx
        seq_len: длина последовательности для трейсинга
        opset_version: версия ONNX opset
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randint(
        0, model.cfg.vocab_size, (1, seq_len), device=device
    )

    # Обёртка для single-output (ONNX не любит tuple)
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, idx):
            logits, _, _ = self.model(idx)
            return logits

    wrapper = OnnxWrapper(model)

    torch.onnx.export(
        wrapper,
        dummy_input,
        path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    return path


def export_torchscript(model, path, seq_len=128):
    """
    Экспортирует модель в TorchScript формат.

    Args:
        model: YiJingGPT
        path: путь для сохранения .pt
        seq_len: длина последовательности для трейсинга
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randint(
        0, model.cfg.vocab_size, (1, seq_len), device=device
    )

    # Используем torch.jit.trace
    class TraceWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, idx):
            logits, _, _ = self.model(idx)
            return logits

    wrapper = TraceWrapper(model)
    traced = torch.jit.trace(wrapper, dummy_input)
    traced.save(path)
    return path


def create_model_card(model, save_path=None):
    """
    Создаёт модельную карточку с метриками.

    Returns:
        dict с информацией о модели
    """
    cfg = model.cfg
    total_params, hex_params = model.count_parameters()

    card = {
        'model_type': 'YiJingGPT',
        'architecture': {
            'd_model': cfg.d_model,
            'n_layers': cfg.n_layers,
            'n_heads': cfg.n_heads,
            'n_kv_heads': cfg.n_kv_heads,
            'block_size': cfg.block_size,
            'vocab_size': cfg.vocab_size,
            'ffn_hidden': cfg.ffn_hidden,
        },
        'features': {
            'rope': cfg.use_rope,
            'swiglu': cfg.use_swiglu,
            'bian_gua': cfg.use_bian_gua,
            'gqa': cfg.n_kv_heads is not None and cfg.n_kv_heads != cfg.n_heads,
            'sliding_window': cfg.sliding_window,
            'quantizer_type': cfg.quantizer_type,
            'adaptive_temp': cfg.adaptive_temp,
            'rope_scaling': cfg.rope_scaling if hasattr(cfg, 'rope_scaling') else None,
        },
        'parameters': {
            'total': total_params,
            'hex_specific': hex_params,
            'hex_overhead_pct': round(100 * hex_params / max(1, total_params), 2),
        },
        'flops': {
            'per_token': model.estimate_flops(1),
            'block_size': model.estimate_flops(),
            'human_readable': model.estimate_flops_str(),
        },
    }

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(card, f, indent=2, default=str)

    return card
