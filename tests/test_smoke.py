"""
tests/test_smoke.py — Быстрый smoke-test pipeline за < 30 секунд.

Проверяет, что ключевые компоненты не падают при запуске:
  - HierarchicalMoEFFN с новым Hamming prior
  - meta_q6: bent seeds, ecube_route, hamming_prior_matrix
  - nautilus_15agent: импорт и _build_agents()
  - pipeline.py: read_avg_lci + run_pipeline signature

pytest tests/test_smoke.py -v
"""

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── 1. Hamming prior в HierarchicalMoEFFN ────────────────────────────────────

def test_hamming_prior_multiscale():
    """MultiScaleGlobalRouter: gate не коллапсирует к 0.5."""
    from yijing_transformer.models.hierarchical_moe import MultiScaleGlobalRouter
    router = MultiScaleGlobalRouter(128, ["ABSTRACT", "DYNAMIC", "CONCRETE"])

    # Проверяем, что hamming_prior_matrix существует и правильного размера
    assert hasattr(router, "hamming_prior_matrix"), "hamming_prior_matrix не найден"
    assert router.hamming_prior_matrix.shape == (64, 3), \
        f"Ожидали (64,3), получили {router.hamming_prior_matrix.shape}"

    # Строки должны быть нормализованы: sum ≈ 1
    row_sums = router.hamming_prior_matrix.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(64), atol=1e-5), \
        "hamming_prior_matrix строки не нормализованы"

    # Прогон: gate не должен быть константой
    x = torch.randn(2, 8, 128)
    gw, hw, lb = router(x)
    assert gw.shape == (2, 8, 3)
    gate_mean = gw.mean(dim=(0, 1))
    # При 3 группах равномерный gate = 0.333. Не должно быть > 0.48 для любой группы
    max_gate = gate_mean.max().item()
    assert max_gate < 0.48, f"Gate коллапсирует: max={max_gate:.4f} (должно быть < 0.48)"


def test_hamming_prior_global_router():
    """GlobalRouter (legacy): Hamming prior работает."""
    from yijing_transformer.models.hierarchical_moe import GlobalRouter
    router = GlobalRouter(128, ["ABSTRACT", "DYNAMIC", "CONCRETE"])
    assert hasattr(router, "hamming_prior_matrix")
    x = torch.randn(1, 4, 128)
    gw, lb = router(x)
    assert gw.shape == (1, 4, 3)
    # Суммы по группам ≈ 1
    assert torch.allclose(gw.sum(-1), torch.ones(1, 4), atol=1e-5)


# ── 2. meta_q6 ────────────────────────────────────────────────────────────────

def test_bent_seeds():
    """bent_seed_texts: 20 текстов, все разные."""
    from meta_q6 import bent_seed_texts
    texts = bent_seed_texts(n=20)
    assert len(texts) == 20
    assert len(set(texts)) == 20, "Не все bent seed тексты уникальны"


def test_ecube_route():
    """ecube_route(0, 63): длина пути = 6 (hamming distance)."""
    from meta_q6 import ecube_route
    path = ecube_route(0, 63)
    assert path[0] == 0 and path[-1] == 63
    assert len(path) - 1 == 6, f"Ожидали длину 6, получили {len(path)-1}"


def test_q4_tesseracts():
    """q4_tesseracts: C(6,4)×4=60 тессерактов, каждый из 16 вершин."""
    from meta_q6 import q4_tesseracts
    tess = q4_tesseracts()
    # C(6,4)=15 наборов осей × 4 сдвига base = 60 (или fallback даёт 15 уникальных по осям)
    assert len(tess) >= 15, f"Ожидали ≥15, получили {len(tess)}"
    assert all(len(t) == 16 for t in tess), "Не все тессеракты имеют 16 вершин"
    # 15 первых тессерактов (для nautilus_15agent) должны покрыть >60 вершин
    covered = {v for t in tess[:15] for v in t}
    assert len(covered) > 60, f"Топ-15 покрывают только {len(covered)}/64 вершин"


# ── 3. nautilus_15agent ───────────────────────────────────────────────────────

def test_nautilus_15agent_build():
    """nautilus_15agent._build_agents: 15 агентов, каждый с 16 вершинами."""
    from nautilus_15agent import _build_agents
    agents = _build_agents()
    assert len(agents) == 15
    for ag in agents:
        assert len(ag["verts"]) == 16, f"Агент {ag['id']} имеет {len(ag['verts'])} вершин"
    total_covered = len({v for ag in agents for v in ag["verts"]})
    assert total_covered > 60, f"Покрыто только {total_covered}/64 вершин"


# ── 4. pipeline.py импорт ─────────────────────────────────────────────────────

def test_pipeline_import():
    """pipeline.py: импорт и сигнатура run_pipeline."""
    import inspect
    import pipeline
    sig = inspect.signature(pipeline.run_pipeline)
    params = list(sig.parameters.keys())
    assert "adaptive_lr"    in params, "adaptive_lr не найден в run_pipeline"
    assert "reset_rag_pass" in params, "reset_rag_pass не найден в run_pipeline"
    assert "start_checkpoint" in params


# ── 5. HierarchicalMoEFFN end-to-end ─────────────────────────────────────────

def test_hmoe_ffn_forward():
    """HierarchicalMoEFFN: прямой проход без ошибок."""
    from yijing_transformer.models.hierarchical_moe import HierarchicalMoEFFN, HMoEConfig
    cfg = HMoEConfig(d_model=64, use_multiscale=True, use_hex_tier=False)
    ffn = HierarchicalMoEFFN(cfg)
    x = torch.randn(1, 4, 64)
    out, info = ffn(x)
    assert out.shape == x.shape, f"Ожидали {x.shape}, получили {out.shape}"
    assert isinstance(info, dict), "Второй возврат должен быть dict"
    assert "lb_loss" in info, "info должен содержать 'lb_loss'"


# ── 6. bench_stability.py импорт ─────────────────────────────────────────────

def test_bench_stability_import():
    """bench_stability.py: можно импортировать."""
    import bench_stability
    assert hasattr(bench_stability, "run_once")
    assert hasattr(bench_stability, "main")
