"""Cola física O(e²) del canal Atlas (E3_Atlas): formas cerradas de los
residuos del polo 1/(λ_K − 1) fijadas en cosmology/mu_eta_atlas.py.
Sin sympy, sin datos: identidades, límites y la relación con la cola QS
truncada (no física) que el módulo conserva como referencia.
"""

import numpy as np
import pytest

from cosmology.mu_eta_atlas import (
    eta_tail_coefficient,
    eta_tail_physical,
    growth_index_matter_era,
    khronon_cs2,
    mu_local_tail_physical,
    tail_pole_residues,
)


class TestPoleResidues:
    @pytest.mark.parametrize("al", [0.1, 0.3, 0.6, 0.012])
    def test_P_eta_closed_form_and_ratio_identity(self, al):
        r = tail_pole_residues(al)
        assert r["P_eta"] == pytest.approx(3 * al / (2 - al), rel=1e-14)
        p = growth_index_matter_era(1.0, 1.0, al)
        # p(p+½) = 3/(2−α) en λ_K = 1
        assert p * (p + 0.5) == pytest.approx(3 / (2 - al), rel=1e-12)
        assert r["P_mu_local"] == pytest.approx(-r["P_eta"] * p * (2 * p - 1) / 3,
                                                rel=1e-14)
        assert r["ratio_P_mu_over_P_eta"] == pytest.approx(-p * (2 * p - 1) / 3,
                                                           rel=1e-14)

    def test_signs_and_gr_limit(self):
        # η − 1 > 0 (residuo positivo), µ_loc − 1 < 0; ambos → 0 con α_a → 0
        for al in (0.05, 0.3, 1.0):
            r = tail_pole_residues(al)
            assert r["P_eta"] > 0 and r["P_mu_local"] < 0
        assert tail_pole_residues(1e-6)["P_eta"] == pytest.approx(1.5e-6, rel=1e-3)
        # en GR (α → 0, p → 1): P_µ/P_η → −1/3
        assert tail_pole_residues(1e-9)["ratio_P_mu_over_P_eta"] == \
            pytest.approx(-1 / 3, rel=1e-6)

    def test_harness_point_numbers(self):
        # α_a = 0.3: P_η = 0.9/1.7; el cociente QS/físico es 2(2−α)/(3α) = 3.78
        r = tail_pole_residues(0.3)
        assert r["P_eta"] == pytest.approx(0.9 / 1.7, rel=1e-14)
        assert r["qs_over_physical_eta"] == pytest.approx(2 * 1.7 / 0.9, rel=1e-14)
        assert tail_pole_residues(0.1)["qs_over_physical_eta"] == \
            pytest.approx(12.667, rel=1e-3)

    def test_qs_truncated_residue_is_2_and_alpha_independent(self):
        """La cola QS truncada tiene residuo 2 (λ→1, ξ=1) y no depende de
        α_a: por eso no es física."""
        for lam in (1.001, 1.0001):
            assert (lam - 1) * eta_tail_coefficient(lam, 1.0) == \
                pytest.approx(2.0, rel=5e-3)
        r = tail_pole_residues(0.3)
        assert (1.0001 - 1) * eta_tail_coefficient(1.0001) / r["P_eta"] == \
            pytest.approx(r["qs_over_physical_eta"], rel=5e-3)

    def test_domain_guards(self):
        with pytest.raises(ValueError):
            tail_pole_residues(0.0)
        with pytest.raises(ValueError):
            tail_pole_residues(2.0)
        with pytest.raises(ValueError):
            tail_pole_residues(0.3, xi=1.02)          # solo ξ = 1
        with pytest.raises(ValueError):
            eta_tail_physical(0.01, 1.05, 0.3, xi=1.02)


class TestPhysicalTails:
    def test_eta_tail_is_three_halves_over_cs2(self):
        e = np.array([0.001, 0.01, 0.02])
        lam, al = 1.05, 0.3
        cs2 = khronon_cs2(lam, 1.0, al)
        np.testing.assert_allclose(eta_tail_physical(e, lam, al), 1.5 * e ** 2 / cs2,
                                   rtol=1e-14)
        # el parámetro pequeño es aH/(c_s k): cola ∝ e² y ∝ 1/(λ_K − 1)
        assert eta_tail_physical(0.02, lam, al) / eta_tail_physical(0.01, lam, al) \
            == pytest.approx(4.0, rel=1e-12)
        assert eta_tail_physical(0.01, 1.01, al) / eta_tail_physical(0.01, 1.05, al) \
            == pytest.approx(khronon_cs2(1.05, 1, al) / khronon_cs2(1.01, 1, al),
                             rel=1e-12)

    def test_mu_local_tail_and_ratio(self):
        lam, al, e = 1.05, 0.3, 0.01
        p = growth_index_matter_era(1.0, 1.0, al)
        ratio = mu_local_tail_physical(e, lam, al) / eta_tail_physical(e, lam, al)
        assert ratio == pytest.approx(-p * (2 * p - 1) / 3, rel=1e-12)

    def test_residue_matches_leading_pole_of_tail(self):
        """(λ−1)·(cola física/e²) → P cuando λ → 1 (residuo exacto en la
        variable e, porque c_s² ∝ (λ−1)/(3λ−1) y 3λ−1 → 2)."""
        al = 0.3
        r = tail_pole_residues(al)
        lam = 1.0 + 1e-6
        assert (lam - 1) * eta_tail_physical(1.0, lam, al) == \
            pytest.approx(r["P_eta"], rel=1e-5)
        assert (lam - 1) * mu_local_tail_physical(1.0, lam, al) == \
            pytest.approx(r["P_mu_local"], rel=1e-5)

    def test_pole_part_only_is_declared(self):
        """La función devuelve SOLO el residuo del polo: en el punto del
        arnés (1.05, 0.3) vale P_η/(λ−1) = 10.59·e² frente al coeficiente
        completo de la escalera 17.07·e² (Q_η ≈ 6.1); en α_a = ε_K = 0.012
        (λ_K = 1.012) el polo aporta solo 1.51·e² y la parte regular domina.
        El módulo lo declara; los coeficientes completos viven en el
        artefacto E3."""
        # en la variable física (3/2)e²/c_s² el residuo lleva el factor
        # (3λ−1)/2 = 1 + O(λ−1) respecto de P_η/(λ−1): ambigüedad declarada
        e = 1.0
        assert eta_tail_physical(e, 1.05, 0.3) == pytest.approx(
            0.9 / 1.7 / 0.05 * (3 * 1.05 - 1) / 2, rel=1e-9)
        assert eta_tail_physical(e, 1.05, 0.3) < 17.07          # < coef completo
        assert eta_tail_physical(e, 1.012, 0.012) == pytest.approx(
            (3 * 0.012 / 1.988) / 0.012 * (3 * 1.012 - 1) / 2, rel=1e-9)
        assert "SOLO el residuo del polo" in eta_tail_physical.__doc__
