"""
tests/test_training_smoke.py — Smoke-тесты обучающих скриптов.

Проверяет, что ключевые обучающие скрипты:
  - Импортируются без ошибок
  - Выполняют 1-2 шага без падений

Покрытие:
  - self_train_common: CFG, hexagrams, biangua, evaluator, qfilter, text_to_ids
  - self_train_hmoe:   lci_from_routing, lci_from_embeddings, micro_train, RagBuffer
  - nautilus_4agent:   RINGS, _AGENTS, конфигурация орбит
  - figure8_turbine:   kirchhoff_balance, константы
  - pipeline:          run_phase сигнатура

pytest tests/test_training_smoke.py -v
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── self_train_common ─────────────────────────────────────────────────────────

class TestSelfTrainCommon:
    def test_cfg_instantiated(self):
        from self_train_common import CFG
        assert CFG.vocab_size == 256
        assert CFG.d_model == 128
        assert CFG.n_layers == 4

    def test_hexagrams_shape(self):
        from self_train_common import hexagrams
        assert hexagrams.shape == (64, 6), f"Ожидали (64, 6), получили {hexagrams.shape}"
        # Все вершины должны быть ±1
        assert hexagrams.abs().max().item() == 1.0

    def test_biangua_shape_and_symmetry(self):
        from self_train_common import biangua
        assert biangua.shape == (64, 64), f"Ожидали (64, 64), получили {biangua.shape}"
        # biangua — матрица смежности, должна быть симметричной
        assert (biangua == biangua.T).all(), "BianGua-матрица не симметрична"

    def test_text_to_ids(self):
        from self_train_common import text_to_ids
        ids = text_to_ids("Hello, world!", block_size=16)
        assert ids.dtype == torch.long
        assert ids.shape[0] <= 16
        assert ids.numel() > 0

    def test_text_to_ids_empty_string(self):
        """Пустая строка → тензор с нулевым токеном."""
        from self_train_common import text_to_ids
        ids = text_to_ids("", block_size=8)
        assert ids.numel() > 0  # fallback: [0]

    def test_evaluator_and_qfilter_instantiated(self):
        from self_train_common import evaluator, qfilter
        assert evaluator is not None
        assert qfilter is not None


# ── self_train_hmoe ───────────────────────────────────────────────────────────

class TestSelfTrainHMoE:
    @pytest.fixture
    def tiny_model(self):
        """Tiny HMoE-модель для быстрых тестов."""
        from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
        cfg = Variant3Config(
            vocab_size=256, block_size=16, d_model=32,
            n_heads=2, n_layers=2, ffn_mult=2,
            use_domain_routing=False, use_hierarchical_moe=True,
        )
        return Variant3GPT(cfg)

    def test_model_cfg_keys(self):
        from self_train_hmoe import MODEL_CFG
        assert "vocab_size" in MODEL_CFG
        assert "d_model" in MODEL_CFG
        assert MODEL_CFG["use_hierarchical_moe"] is True

    def test_rag_buffer_instantiated(self):
        from self_train_hmoe import RagBuffer
        buf = RagBuffer(max_size=10)
        assert len(buf) == 0
        emb = torch.zeros(32)
        buf.add("test text", emb=emb, q6_vert=0)
        assert len(buf) == 1

    def test_quality_filter(self):
        from self_train_hmoe import quality_filter
        assert quality_filter("Hello world") is True
        assert quality_filter("") is False
        assert quality_filter("   ") is False
        assert quality_filter("Hi") is False   # len < 10
        assert quality_filter("Hello world!") is True

    def test_lci_from_routing_tiny(self, tiny_model):
        from self_train_hmoe import lci_from_routing
        ids = torch.randint(0, 256, (1, 8))
        lci, gw_dict = lci_from_routing(tiny_model, ids)
        assert isinstance(lci, float), "lci_from_routing должен возвращать float"
        assert 0.0 <= lci <= math.pi + 0.01, f"LCI={lci:.3f} вне [0, π]"
        assert isinstance(gw_dict, dict)

    def test_lci_from_embeddings_tiny(self, tiny_model):
        from self_train_hmoe import lci_from_embeddings
        start = torch.randint(0, 256, (1, 8))
        end   = torch.randint(0, 256, (1, 8))
        lci = lci_from_embeddings(tiny_model, start, end)
        assert isinstance(lci, float)
        # Formula B: acos(cos)*4, диапазон [0, 4π]
        assert 0.0 <= lci <= 4 * math.pi + 0.01

    def test_micro_train_1_step(self, tiny_model):
        from self_train_hmoe import micro_train
        ids = torch.randint(0, 256, (1, 10))
        loss = micro_train(tiny_model, ids, lr=1e-4, n_steps=1)
        assert isinstance(loss, float)
        assert not math.isnan(loss), "micro_train вернул nan"

    def test_generate_runs(self, tiny_model):
        from self_train_hmoe import _generate
        prompt = torch.randint(0, 256, (1, 4))
        out = _generate(tiny_model, prompt, block_size=16, temperature=1.0, n_tokens=3)
        assert out.shape[1] >= 4   # prompt + хотя бы несколько токенов


# ── nautilus_4agent ───────────────────────────────────────────────────────────

class TestNautilus4Agent:
    def test_rings_structure(self):
        from nautilus_4agent import RINGS, _RING_BY_NAME, _TOTAL_STEPS
        assert len(RINGS) == 4
        ring_names = {r["name"] for r in RINGS}
        assert ring_names == {"META", "ABSTRACT", "DYNAMIC", "CONCRETE"}
        assert _TOTAL_STEPS == sum(r["steps"] for r in RINGS)

    def test_agents_structure(self):
        from nautilus_4agent import _AGENTS
        assert len(_AGENTS) == 4
        homes = {a["home"] for a in _AGENTS}
        assert homes == {"META", "ABSTRACT", "DYNAMIC", "CONCRETE"}

    def test_orbit_verts_cover_q6(self):
        """Все 64 вершины должны быть покрыты орбитами."""
        from nautilus_4agent import _AUT_Q6_ORBITS
        all_verts = set()
        for k, verts in _AUT_Q6_ORBITS.items():
            all_verts |= set(verts)
        assert all_verts == set(range(64)), "Орбиты не покрывают все 64 вершины Q6"

    def test_orbit_sizes(self):
        """Размеры орбит = C(6,k)."""
        from nautilus_4agent import _AUT_Q6_ORBITS
        expected = {0: 1, 1: 6, 2: 15, 3: 20, 4: 15, 5: 6, 6: 1}
        for k, size in expected.items():
            assert len(_AUT_Q6_ORBITS[k]) == size, \
                f"Орбита k={k}: ожидали {size}, получили {len(_AUT_Q6_ORBITS[k])}"


# ── figure8_turbine ───────────────────────────────────────────────────────────

class TestFigure8Turbine:
    def test_constants(self):
        from figure8_turbine import _TURBINE_EXPERTS, _TURBINE_ROLES, _KIRCHHOFF_EPSILON
        assert set(_TURBINE_EXPERTS) == {"ABSTRACT", "DYNAMIC", "CONCRETE", "META"}
        assert set(_TURBINE_ROLES.keys()) == set(_TURBINE_EXPERTS)
        assert 0 < _KIRCHHOFF_EPSILON < math.pi

    def test_kirchhoff_balance_tiny(self):
        from figure8_turbine import kirchhoff_balance
        from yijing_transformer.models.variant3 import Variant3Config, Variant3GPT
        cfg = Variant3Config(
            vocab_size=256, block_size=16, d_model=32,
            n_heads=2, n_layers=2, ffn_mult=2,
            use_domain_routing=False, use_hierarchical_moe=True,
        )
        model = Variant3GPT(cfg)
        ids = torch.randint(0, 256, (1, 8))
        # kirchhoff_balance(model, ids, expert_lci_map)
        expert_lci_map = {"ABSTRACT": math.pi, "DYNAMIC": math.pi, "CONCRETE": math.pi}
        result = kirchhoff_balance(model, ids, expert_lci_map)
        # Returns Tuple[float, float]: (kirchhoff_sum, energy)
        assert isinstance(result, tuple) and len(result) == 2

    def test_expert_freeze_map_completeness(self):
        from figure8_turbine import _EXPERT_FREEZE_MAP, _TURBINE_EXPERTS
        for expert in _TURBINE_EXPERTS:
            assert expert in _EXPERT_FREEZE_MAP, f"Нет freeze-map для {expert}"
            assert isinstance(_EXPERT_FREEZE_MAP[expert], list)


# ── pipeline ──────────────────────────────────────────────────────────────────

class TestPipeline:
    def test_run_phase_importable(self):
        from pipeline import run_phase
        import inspect
        sig = inspect.signature(run_phase)
        params = list(sig.parameters.keys())
        assert "cmd" in params
        assert "phase_name" in params

    def test_pipeline_constants(self):
        """pipeline.py имеет ключевые конфигурационные константы."""
        import pipeline
        assert hasattr(pipeline, "_PI")
        assert abs(pipeline._PI - math.pi) < 1e-10
        assert hasattr(pipeline, "_ROOT")
        assert os.path.isdir(pipeline._ROOT)
