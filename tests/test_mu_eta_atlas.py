"""E1_Atlas — identidades de las formas cerradas del canal Atlas
(cosmology/mu_eta_atlas.py): límites GR con e → 0 PRIMERO, cancelación
G_growth = G_local, salud del khronon, c_T², cola 1/(λ_K−1) y guardia de
«sin datos». El candado de preinscripción/artefacto vive en
tests/test_mu_eta_atlas_lock.py.
"""

from pathlib import Path

import numpy as np
import pytest

from cosmology.mu_eta_atlas import (
    G_cosmo_over_G_local,
    G_cosmo_over_GB,
    G_growth_over_GB,
    G_local_over_GB,
    c_T2,
    eta_atlas_qs,
    eta_tail_coefficient,
    growth_index_matter_era,
    is_healthy,
    khronon_cs2,
    khronon_kinetic_sign,
    mu_atlas_qs,
    mu_atlas_relative_to_local,
    mu_atlas_subhorizon,
    residues_ratio_first_order,
)
from mcmc_ontology import constants as C

REPO = Path(__file__).resolve().parent.parent
HARNESS = (1.05, 1.02, 0.30)          # punto del arnés E2
TREATISE = (1.0 + C.EPSILON_K, 1.0)   # λ_K = 1 + ε_K, ξ = 1 (GW170817)
E = np.array([0.0, 1e-3, 1e-2, 3e-2])


class TestGRLimit:
    def test_mu_eta_one_at_gr_with_e_to_zero_first(self):
        assert mu_atlas_qs(0.0, 1.0, 1.0, 0.0) == 1.0
        assert eta_atlas_qs(0.0, 1.0, 1.0, 0.0) == 1.0
        assert mu_atlas_subhorizon(1.0, 1.0, 0.0) == 1.0
        assert G_cosmo_over_GB(1.0) == 1.0 == G_local_over_GB(1.0, 1.0, 0.0)

    def test_eta_closed_form_is_degenerate_at_lamK_one(self):
        """Precisión (i): en λ_K = 1 exacto la forma cerrada es 0/0 y a e
        finito daría 1/3; el módulo devuelve el orden dominante (1)."""
        assert np.array_equal(eta_atlas_qs(E, 1.0, 1.0, 0.0), np.ones_like(E))
        # y a λ_K = 1 + δ con δ → 0 la forma cerrada se acerca a 1/3 a e
        # finito: la degeneración es real, no un artefacto del código
        val = eta_atlas_qs(1e-2, 1.0 + 1e-9, 1.0, 0.0)
        assert val == pytest.approx(1.0 / 3.0, rel=1e-3)


class TestCancellation:
    def test_subhorizon_offset_and_cancellation(self):
        lam, xi_, al = HARNESS
        mu_sub = mu_atlas_subhorizon(lam, xi_, al)
        assert mu_sub * (1.0 - al / (2 * xi_)) == pytest.approx(1.0, rel=1e-15)
        assert mu_sub == pytest.approx(mu_atlas_qs(0.0, lam, xi_, al), rel=1e-15)
        # respecto de G_B/ξ: 1/(1 − 0.30/2.04) = 1.17241… (el traspaso: 1.172);
        # respecto de G_B sería 1/0.87 = 1.1494 = G_growth/G_B — distinta
        # normalización, mismo objeto
        assert mu_sub == pytest.approx(1.0 / (1.0 - 0.30 / 2.04), rel=1e-12)
        assert mu_sub / xi_ == pytest.approx(1.0 / 0.87, rel=1e-12)

    def test_G_growth_equals_G_local(self):
        for lam, xi_, al in (HARNESS, (1.012, 1.0, 0.1), (1.3, 0.9, 1.0)):
            assert G_growth_over_GB(lam, xi_, al) == G_local_over_GB(lam, xi_, al)
            assert G_growth_over_GB(lam, xi_, al) == pytest.approx(
                1.0 / (xi_ - al / 2), rel=1e-15)

    def test_mu_relative_to_local_has_no_pole_and_tends_to_one(self):
        lam, xi_, al = HARNESS
        m_loc = mu_atlas_relative_to_local(E, lam, xi_, al)
        assert m_loc[0] == pytest.approx(1.0, rel=1e-15)
        exact = 1.0 / (1.0 + 3 * (3 * lam - 1) * E ** 2 / (2 * xi_ - al))
        assert np.max(np.abs(m_loc - exact)) < 1e-15
        first_order = 1.0 - 3 * (3 * lam - 1) * E ** 2 / (2 * xi_ - al)
        # residual O(e⁴): (3(3λ−1)/(2ξ−α))²·e⁴ ≈ 1.1e-5 en e = 0.03
        assert np.max(np.abs(m_loc - first_order)) < 2e-5
        # sin polo en λ_K → 1: coeficiente finito
        m1 = mu_atlas_relative_to_local(1e-2, 1.0 + 1e-9, 1.0, 0.1)
        assert abs(m1 - 1.0) < 1e-3

    def test_lamK_independent_at_leading_order(self):
        assert mu_atlas_subhorizon(1.05, 1.0, 0.3) == \
            mu_atlas_subhorizon(1.30, 1.0, 0.3)


class TestResiduesRefined:
    def test_G_cosmo_over_G_local_closed_form(self):
        lam, xi_, al = HARNESS
        assert G_cosmo_over_G_local(lam, xi_, al) == pytest.approx(
            (2 * xi_ - al) / (3 * lam - 1), rel=1e-15)

    def test_first_order_matches_eq_9_5_when_alpha_zero(self):
        from cosmology.residues_test import predicted_ratio
        assert residues_ratio_first_order(C.EPSILON_K, 0.0) == pytest.approx(
            predicted_ratio(C.EPSILON_K), rel=1e-15)

    def test_first_order_expansion_is_correct(self):
        eK, al = 0.012, 0.02
        exact = G_cosmo_over_G_local(1.0 + eK, 1.0, al)
        approx = residues_ratio_first_order(eK, al)
        # residual de segundo orden: 2.25ε² + 0.75αε ≈ 5.0e-4 ≪ 0.028
        assert abs(exact - approx) < 1e-3
        assert abs(exact - approx) < 0.05 * abs(1.0 - approx)


class TestKhrononHealth:
    def test_cs2_closed_form_and_window(self):
        lam, xi_, al = HARNESS
        assert khronon_cs2(lam, xi_, al) == pytest.approx(
            xi_ * (2 * xi_ - al) * (lam - 1) / (al * (3 * lam - 1)), rel=1e-15)
        assert khronon_cs2(lam, xi_, al) > 0
        assert is_healthy(lam, xi_, al)
        assert not is_healthy(0.95, 1.0, 0.3)        # fantasma
        assert not is_healthy(1.05, 1.0, 2.5)        # c_s² < 0
        assert not is_healthy(1.05, 1.0, 0.0)        # α_a = 0: sin c_s

    def test_kinetic_sign(self):
        assert khronon_kinetic_sign(1.05) > 0
        assert khronon_kinetic_sign(0.9) < 0
        assert khronon_kinetic_sign(0.2) > 0
        assert khronon_kinetic_sign(1.0) == 0.0

    def test_cs2_diverges_as_alpha_to_zero_at_fixed_lamK(self):
        """Erratum candidata H.2.2: c_s² NO → 0 cuando α_a → 0 a λ_K fijo;
        diverge (acoplamiento fuerte). Sí → 0 cuando λ_K → 1."""
        assert khronon_cs2(1.05, 1.0, 1e-6) > 1e4
        assert khronon_cs2(1.0 + 1e-9, 1.0, 0.3) < 1e-7
        assert khronon_cs2(1.05, 1.0, 0.0) == float("inf")


class TestTensorAndGrowthIndex:
    def test_cT2_is_xi(self):
        assert c_T2(1.0) == 1.0 and c_T2(1.02) == 1.02

    def test_growth_index(self):
        assert growth_index_matter_era(1.0, 1.0, 0.0) == pytest.approx(1.0)
        p = growth_index_matter_era(*HARNESS)
        assert p == pytest.approx(1.1342, abs=2e-4)
        g = (3 * HARNESS[0] - 1) / (2 * HARNESS[1] - HARNESS[2])
        assert p * (p + 0.5) == pytest.approx(1.5 * g, rel=1e-12)


class TestEtaTail:
    def test_tail_coefficient_matches_expansion(self):
        lam, xi_, al = HARNESS
        c = eta_tail_coefficient(lam, xi_)
        e = 1e-3
        num = (eta_atlas_qs(e, lam, xi_, al) - 1.0) / e ** 2
        assert num == pytest.approx(c, rel=1e-4)

    def test_tail_has_one_over_lamK_minus_one_pole(self):
        c1 = eta_tail_coefficient(1.0 + 0.012, 1.0)
        c2 = eta_tail_coefficient(1.0 + 0.024, 1.0)
        assert c1 == pytest.approx(2 * c2, rel=0.05)
        assert 150 < c1 < 200                        # ~170 con ε_K = 0.012
        assert eta_tail_coefficient(1.0, 1.0) == float("inf")

    def test_tail_size_at_treatise_point(self):
        """Con ε_K = 0.012 la cola QS de η a k = 0.05 h/Mpc, z = 0 es del
        orden del 1 %: la ventana sub-horizonte se estrecha ×~9 respecto
        de e ≪ 1 (el coeficiente exacto es frontera declarada)."""
        lam, xi_ = TREATISE
        e = 67.87 / (299792.458 * 0.05 * 0.6787)
        tail = eta_atlas_qs(e, lam, xi_, 0.1) - 1.0
        assert 3e-3 < tail < 2e-2


class TestNoDataGuard:
    def test_modules_load_no_data_and_fit_nothing(self):
        for rel in ("cosmology/mu_eta_atlas.py",
                    "validation/atlas_derivation.py",
                    "scripts/run_mu_eta_atlas.py",
                    "scripts/run_mu_eta_atlas_prereg.py"):
            src = (REPO / rel).read_text(encoding="utf-8")
            for pat in ("load_fsigma8_data", "np.loadtxt", "loadtxt(",
                        "require_available", "scipy.optimize", "minimize(",
                        "curve_fit", "emcee", "EnsembleSampler"):
                assert pat not in src, f"{rel}: {pat}"
