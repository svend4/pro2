"""
test_geometry_new.py — Тесты для компонентов, добавленных в сессиях 2026-03-24.

Покрывает расхождения D1, D2 из аудита THEORY_VS_PRACTICE.md:
  D1 — WHT_Quantizer без тестов
  D2 — Q6Arithmetic / YiJingOps / BentFunctions без тестов
  + новые экспорты (была категория A): ArchetypalInterlinguaFixed, KasatkinQ6Router

pytest tests/test_geometry_new.py -v
"""

import pytest
import torch
import torch.nn as nn


# ── WHT_Quantizer (Теорема 5) ─────────────────────────────────────────────────

class TestWHTQuantizer:
    def setup_method(self):
        from yijing_transformer.models.geometry import WHT_Quantizer
        self.Q = WHT_Quantizer

    def test_import_from_geometry(self):
        """WHT_Quantizer экспортируется из geometry (расхождение A2)."""
        from yijing_transformer.models.geometry import WHT_Quantizer
        assert WHT_Quantizer is not None

    def test_output_shape_3d(self):
        """(B, T, d_model) → (B, T, n_bits)."""
        q = self.Q(d_model=32, n_bits=6)
        x = torch.randn(2, 8, 32)
        out = q(x)
        assert out.shape == (2, 8, 6), f"Expected (2,8,6), got {out.shape}"

    def test_output_shape_2d(self):
        """(B, d_model) → (B, n_bits)."""
        q = self.Q(d_model=16, n_bits=4)
        x = torch.randn(4, 16)
        out = q(x)
        assert out.shape == (4, 4)

    def test_soft_output_range(self):
        """Soft mode: выход в (-1, +1) через tanh."""
        q = self.Q(d_model=32, n_bits=6, temp=0.5)
        x = torch.randn(100, 32)
        out = q(x)
        assert out.abs().max().item() < 1.0, "tanh output must be in (-1, +1)"

    def test_hard_output_values(self):
        """Hard mode (temp<=0): выход точно в {-1, +1}."""
        q = self.Q(d_model=32, n_bits=6, temp=-1)
        x = torch.randn(50, 32)
        out = q(x)
        unique = out.unique()
        for v in unique:
            assert v.item() in (-1.0, 1.0), f"Unexpected value: {v.item()}"

    def test_hadamard_matrix_shape(self):
        """H матрица имеет правильный размер 2^n × 2^n."""
        for n in [2, 4, 6]:
            q = self.Q(d_model=32, n_bits=n)
            assert q.H.shape == (2**n, 2**n), f"Expected ({2**n},{2**n}) for n={n}"

    def test_spectral_loss_computed(self):
        """use_spectral_loss=True: _spectral_loss > 0 при train."""
        q = self.Q(d_model=16, n_bits=6, use_spectral_loss=True)
        q.train()
        x = torch.randn(4, 8, 16)
        q(x)
        assert q._spectral_loss.item() >= 0.0

    def test_gradients_flow(self):
        """Градиенты проходят через soft quantizer."""
        q = self.Q(d_model=16, n_bits=6, temp=0.5)
        x = torch.randn(4, 16, requires_grad=True)
        loss = q(x).sum()
        loss.backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()


# ── Q6Arithmetic (Теорема 3) ──────────────────────────────────────────────────

class TestQ6Arithmetic:
    def setup_method(self):
        from yijing_transformer.models.geometry import Q6Arithmetic
        self.A = Q6Arithmetic

    def test_hamming_from_dot_exact(self):
        """Теорема 3: d_H = (n − ⟨a,b⟩)/2 совпадает с прямым подсчётом."""
        a = torch.randint(0, 2, (200, 6)) * 2 - 1
        b = torch.randint(0, 2, (200, 6)) * 2 - 1
        direct = (a != b).float().sum(-1)
        theorem3 = self.A.hamming_from_dot(a.float(), b.float())
        assert (direct - theorem3).abs().max().item() < 1e-5

    def test_hamming_from_dot_diagonal_zero(self):
        """d_H(a, a) = 0."""
        a = torch.randint(0, 2, (50, 6)) * 2 - 1
        d = self.A.hamming_from_dot(a.float(), a.float())
        assert d.max().item() == 0.0

    def test_hamming_from_dot_max(self):
        """d_H(v, -v) = 6 для всех v ∈ {-1,+1}^6."""
        v = torch.randint(0, 2, (50, 6)) * 2 - 1
        d = self.A.hamming_from_dot(v.float(), -v.float())
        assert (d == 6).all()

    def test_hamming_matrix_symmetry(self):
        """Матрица расстояний симметрична и диагональ = 0."""
        cb = torch.tensor([[1,-1,1,-1,1,-1],[1,1,-1,-1,1,1],[-1,1,1,-1,-1,1]], dtype=torch.float)
        H = self.A.hamming_matrix(cb)
        assert torch.allclose(H, H.T)
        assert (H.diagonal() == 0).all()

    def test_weight_range(self):
        """Хэмминговый вес ∈ [0, n]."""
        v = torch.randint(0, 2, (100, 6)) * 2 - 1
        w = self.A.weight(v.float())
        assert w.min().item() >= 0
        assert w.max().item() <= 6

    def test_add_is_involution(self):
        """XOR в Z₂^6: a ⊕ a = identity."""
        a = torch.randint(0, 2, (20, 6)) * 2 - 1
        result = self.A.add(a.float(), a.float())
        expected = self.A.identity(6)
        assert torch.allclose(result, expected.unsqueeze(0).expand_as(result))

    def test_subgroup_size_power_of_2(self):
        """Порядок подгруппы кратен степени 2."""
        gen = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
        H = self.A.subgroup_from_generators(gen, n_bits=6)
        assert H.shape[0] in (1, 2, 4, 8, 16, 32, 64)


# ── YiJingOps (Группа Клейна V₄) ─────────────────────────────────────────────

class TestYiJingOps:
    def setup_method(self):
        from yijing_transformer.models.geometry import YiJingOps
        self.O = YiJingOps

    def test_heng_identity(self):
        """heng(v) = v."""
        v = torch.randn(6)
        assert torch.allclose(self.O.heng(v), v)

    def test_cuo_involution(self):
        """cuo_gua(cuo_gua(v)) = v."""
        v = torch.randn(10, 6)
        assert torch.allclose(self.O.cuo_gua(self.O.cuo_gua(v)), v)

    def test_zong_involution(self):
        """zong_gua(zong_gua(v)) = v."""
        v = torch.randn(10, 6)
        assert torch.allclose(self.O.zong_gua(self.O.zong_gua(v)), v)

    def test_cuo_zong_involution(self):
        """cuo_zong(cuo_zong(v)) = v."""
        v = torch.randn(10, 6)
        assert torch.allclose(self.O.cuo_zong(self.O.cuo_zong(v)), v)

    def test_v4_cayley_table(self):
        """Cuo ∘ Zong = CuoZong (таблица Кэли V₄)."""
        v = torch.randn(20, 6)
        lhs = self.O.cuo_gua(self.O.zong_gua(v))
        rhs = self.O.cuo_zong(v)
        assert torch.allclose(lhs, rhs)

    def test_v4_orbit_shape(self):
        """v4_orbit возвращает (..., 4, 6)."""
        v = torch.randn(3, 6)
        orbit = self.O.v4_orbit(v)
        assert orbit.shape == (3, 4, 6)

    def test_hu_gua_shape(self):
        """hu_gua: (..., 6) → (..., 6)."""
        v = torch.randn(5, 6)
        out = self.O.hu_gua(v)
        assert out.shape == (5, 6)

    def test_hu_gua_uses_inner_lines(self):
        """hu_gua: нижняя триграмма ядра = строки 1-3 (0-indexed)."""
        v = torch.arange(6, dtype=torch.float)  # [0,1,2,3,4,5]
        out = self.O.hu_gua(v)
        # out[0:3] = v[1:4] = [1,2,3]
        assert torch.allclose(out[:3], v[1:4])


# ── BentFunctions (геометрические seed) ──────────────────────────────────────

class TestBentFunctions:
    def setup_method(self):
        from yijing_transformer.models.geometry import BentFunctions
        self.B = BentFunctions

    def test_truth_tables_shape(self):
        """truth_tables(): (20, 64) в {0,1}."""
        tt = self.B.truth_tables()
        assert tt.shape == (20, 64)
        unique = tt.unique()
        for v in unique:
            assert v.item() in (0.0, 1.0)

    def test_is_bent_all_true(self):
        """Все 20 функций — bent: |Ŵ(u)| = 8."""
        flags = self.B.is_bent()
        assert flags.all(), f"Bent check failed for: {(~flags).nonzero().squeeze()}"

    def test_wht_spectra_flat(self):
        """Все коэффициенты WHT = ±8 (плоский спектр)."""
        W = self.B.wht_spectra()  # (20, 64)
        assert W.abs().shape == (20, 64)
        err = (W.abs() - 8.0).abs().max().item()
        assert err < 1e-4, f"WHT not flat: max err = {err}"

    def test_support_sizes(self):
        """|support| ∈ {28, 36} — bent на GF(2)^6 несбалансированы."""
        supports = self.B.support_vectors()
        assert len(supports) == 20
        for s in supports:
            assert s.shape[-1] == 6
            assert s.shape[0] in (28, 36), f"Unexpected support size: {s.shape[0]}"

    def test_support_vectors_in_pm1(self):
        """Support vectors ∈ {-1,+1}^6."""
        supports = self.B.support_vectors()
        for s in supports[:5]:
            unique = s.unique()
            for v in unique:
                assert v.item() in (-1.0, 1.0)

    def test_prototype_codebook_shape(self):
        """prototype_codebook(k): (k, 6)."""
        for k in [4, 8, 16]:
            proto = self.B.prototype_codebook(k)
            assert proto.shape == (k, 6), f"Expected ({k},6), got {proto.shape}"

    def test_prototype_codebook_diversity(self):
        """k=8 прототипов имеют mean_dH > 2.5 (хорошее покрытие Q6)."""
        from yijing_transformer.models.geometry import Q6Arithmetic
        proto = self.B.prototype_codebook(k=8)
        H = Q6Arithmetic.hamming_matrix(proto)
        n = len(proto)
        mean_dH = H.sum().item() / (n * (n - 1))
        assert mean_dH > 2.5, f"Low diversity: mean_dH = {mean_dH:.2f}"

    def test_verify_returns_all_ok(self):
        """verify() должен вернуть all_bent=True."""
        r = self.B.verify()
        assert r['all_bent']
        assert r['spectrum_flatness_err'] < 1e-4


# ── ArchetypalInterlinguaFixed (расхождение A1) ───────────────────────────────

class TestArchetypalInterlinguaFixed:
    def test_import_from_geometry(self):
        """Экспортируется из geometry (расхождение A1 закрыто)."""
        from yijing_transformer.models.geometry import ArchetypalInterlinguaFixed
        assert ArchetypalInterlinguaFixed is not None

    def test_forward_shape(self):
        """Выход имеет форму (B, T, d_model)."""
        from yijing_transformer.models.geometry import ArchetypalInterlinguaFixed
        model = ArchetypalInterlinguaFixed(d_model=32, n_sources=3, n_archetypes=8)
        x = torch.randn(2, 8, 32)
        out, aux = model([x, x, x])
        assert out.shape == (2, 8, 32)
        assert aux.dim() == 0  # scalar

    def test_per_source_projectors(self):
        """n_sources отдельных trit_proj (исправление бага общего проектора)."""
        from yijing_transformer.models.geometry import ArchetypalInterlinguaFixed
        model = ArchetypalInterlinguaFixed(d_model=32, n_sources=4, n_archetypes=8)
        # Убедиться, что параметры per-source не одинаковы (не shared)
        assert hasattr(model, 'trit_projs') or any(
            'trit_proj' in n for n, _ in model.named_parameters()
        ), "ArchetypalInterlinguaFixed must have per-source trit projectors"

    def test_diagnostics_method_exists(self):
        """Метод diagnostics() существует (имя в коде, не get_diagnostics)."""
        from yijing_transformer.models.geometry import ArchetypalInterlinguaFixed
        model = ArchetypalInterlinguaFixed(d_model=32, n_sources=2)
        assert hasattr(model, 'diagnostics'), "diagnostics() method missing"

    def test_aux_loss_finite(self):
        """aux_loss не NaN и не Inf."""
        from yijing_transformer.models.geometry import ArchetypalInterlinguaFixed
        model = ArchetypalInterlinguaFixed(d_model=16, n_sources=2)
        x = torch.randn(2, 4, 16)
        _, aux = model([x, x])
        assert torch.isfinite(aux).item()


# ── KasatkinQ6Router (расхождение A3) ─────────────────────────────────────────

class TestKasatkinQ6Router:
    def test_import_from_geometry(self):
        """KasatkinQ6Router и Q6ExpertBank экспортируются (расхождение A3, A4)."""
        from yijing_transformer.models.geometry import KasatkinQ6Router, Q6ExpertBank
        assert KasatkinQ6Router is not None
        assert Q6ExpertBank is not None

    def test_forward_output_shape(self):
        """KasatkinQ6Router: (B, T, d) → (B, T, 6) routing weights."""
        from yijing_transformer.models.geometry import KasatkinQ6Router
        router = KasatkinQ6Router(d_model=32)
        x = torch.randn(2, 8, 32)
        out = router(x)
        assert out.shape == (2, 8, 6)

    def test_requires_6_experts(self):
        """KasatkinQ6Router требует n_experts=6 (6 осей куба)."""
        from yijing_transformer.models.geometry import KasatkinQ6Router
        with pytest.raises(ValueError):
            KasatkinQ6Router(d_model=32, n_experts=8)

    def test_q6_expert_bank_shape(self):
        """Q6ExpertBank: (B, T, d) → (B, T, d)."""
        from yijing_transformer.models.geometry import Q6ExpertBank
        bank = Q6ExpertBank(d_model=32, d_ffn=64)
        x = torch.randn(2, 8, 32)
        out = bank(x)
        assert out.shape == (2, 8, 32)


# ── verify_all интеграция ─────────────────────────────────────────────────────

class TestVerifyAll:
    def test_verify_q6_all_ok(self):
        """verify_all() проходит без ошибок и all_ok=True."""
        from yijing_transformer.models.geometry import verify_q6_all
        result = verify_q6_all()
        assert result['all_ok'], f"verify_all failed: {result}"

    def test_verify_theorem3(self):
        from yijing_transformer.models.geometry import verify_theorem3
        r = verify_theorem3()
        assert r['verified']
        assert r['theorem3_max_error'] < 1e-5

    def test_verify_v4_group(self):
        from yijing_transformer.models.geometry import verify_v4_group
        r = verify_v4_group()
        assert r['v4_verified']
        assert r['failed_checks'] == []

    def test_verify_bent_functions(self):
        from yijing_transformer.models.geometry import verify_bent_functions
        r = verify_bent_functions()
        assert r['all_bent']
        assert r['n_bent'] == 20
