"""µ(k,a), η, Σ desde la ontología (canal Cronos): identidades exactas,
límite GR, cierres, crecimiento D(k,a) y guardias de «sin datos».
El candado de la preinscripción vive en tests/test_mu_eta_lock.py.
"""

from pathlib import Path

import numpy as np
import pytest

from cosmology.extended_likelihoods import growth_D_f
from cosmology.mu_eta_cronos import (
    ALPHA0_INV_MAX,
    K_MAX_LINEAR_HMPC,
    RHO_C_OVER_MEAN,
    Sigma_cronos,
    atlas_offset,
    background_above_rho_c_z,
    epsilon_c_background,
    epsilon_c_nonperturbative_z,
    eta_cronos,
    fsigma8_ratio_cronos,
    growth_Dk,
    in_validity_window,
    mu_cronos,
    mu_minus_one_cronos,
    mu_table,
    z_where_mu_minus_one_reaches,
)
from cronos.cronos_v3 import epsilon_c

REPO = Path(__file__).resolve().parent.parent
THETA = (67.87, 0.3263, 0.0172, 9.09)     # medianas del posterior v1
K = np.array([0.01, 0.05, 0.1, 0.2])
A = np.array([1.0, 0.5, 0.25])


class TestGRLimit:
    def test_alpha_zero_gives_exactly_one(self):
        for mode in ("comoving", "physical"):
            m = mu_cronos(K, 0.5, THETA, 0.0, mode)
            e = eta_cronos(K, 0.5, THETA, 0.0, mode)
            s = Sigma_cronos(K, 0.5, THETA, 0.0, mode)
            assert np.array_equal(m, np.ones_like(m))
            assert np.array_equal(e, np.ones_like(e))
            assert np.array_equal(s, np.ones_like(s))

    def test_bound_11_5_enforced_by_canonical_check(self):
        with pytest.raises(ValueError, match="cota dura"):
            mu_cronos(K, 1.0, THETA, 2.0 * ALPHA0_INV_MAX)


class TestIdentities:
    @pytest.mark.parametrize("mode", ["comoving", "physical"])
    def test_sigma_is_half_mu_exact(self, mode):
        for a in A:
            m1 = mu_minus_one_cronos(K, a, THETA, ALPHA0_INV_MAX, mode)
            S = Sigma_cronos(K, a, THETA, ALPHA0_INV_MAX, mode)
            assert np.max(np.abs((S - 1.0) - 0.5 * m1)) <= 1e-15

    @pytest.mark.parametrize("mode", ["comoving", "physical"])
    def test_eta_is_minus_mu_to_second_order(self, mode):
        for a in A:
            m1 = mu_minus_one_cronos(K, a, THETA, ALPHA0_INV_MAX, mode)
            E = eta_cronos(K, a, THETA, ALPHA0_INV_MAX, mode)
            # η = 1/µ exacto ⟹ (η−1) + (µ−1) = (µ−1)²/µ ≤ 1.01·(µ−1)²
            assert np.all(np.abs((E - 1.0) + m1) <= 1.01 * m1 ** 2)
            assert np.all(E < 1.0)                    # slip Φ/Ψ < 1

    def test_k_squared_tail(self):
        m1 = mu_minus_one_cronos(K, 0.5, THETA, ALPHA0_INV_MAX)
        ratio = m1[1:] / m1[:-1]
        expected = (K[1:] / K[:-1]) ** 2
        assert np.max(np.abs(ratio / expected - 1.0)) <= 1e-12

    def test_linear_in_alpha(self):
        full = mu_minus_one_cronos(K, 0.5, THETA, ALPHA0_INV_MAX)
        half = mu_minus_one_cronos(K, 0.5, THETA, 0.5 * ALPHA0_INV_MAX)
        assert np.max(np.abs(half / (0.5 * full) - 1.0)) <= 1e-12

    def test_physical_over_comoving_is_1pz_to_9_halves(self):
        z = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        com = mu_table(THETA, ALPHA0_INV_MAX, K, z, "comoving")
        phy = mu_table(THETA, ALPHA0_INV_MAX, K, z, "physical")
        expected = (1.0 + z)[:, None] ** 4.5
        assert np.max(np.abs(phy / com / expected - 1.0)) <= 1e-10

    def test_linearization_matches_canonical_epsilon_c(self):
        """Cierre 1: d ln ε_c / d ln ρ = 3/2 para la ε_c canónica ⟹
        δε_c/ε_c = (3/2)δ — la linealización no es una hipótesis
        aparte, es la derivada de la Def. 11.1."""
        rho, rho_c = 0.7, 200.0
        e0 = epsilon_c(np.array([rho]), ALPHA0_INV_MAX, rho_c)[0]
        d = 1e-6
        e1 = epsilon_c(np.array([rho * (1 + d)]), ALPHA0_INV_MAX, rho_c)[0]
        assert (e1 - e0) / e0 / d == pytest.approx(1.5, rel=1e-5)

    def test_epsilon_c_background_modes(self):
        a = np.array([1.0, 0.5])
        com = epsilon_c_background(a, ALPHA0_INV_MAX, "comoving")
        phy = epsilon_c_background(a, ALPHA0_INV_MAX, "physical")
        assert com[0] == com[1] == pytest.approx(
            ALPHA0_INV_MAX * RHO_C_OVER_MEAN ** -1.5, rel=1e-14)
        assert phy[0] == pytest.approx(com[0], rel=1e-14)
        assert phy[1] == pytest.approx(com[1] * 2.0 ** 4.5, rel=1e-12)
        with pytest.raises(ValueError):
            epsilon_c_background(a, ALPHA0_INV_MAX, "volumetric")


class TestMagnitudes:
    def test_bound_saturated_is_small_in_window(self):
        """Con la cota saturada y cierre comoving, µ−1 ≲ 1e-3 en la
        ventana lineal a z = 0 (orden de magnitud de la nota, que la
        implementación recomputa)."""
        m1 = mu_minus_one_cronos(K, 1.0, THETA, ALPHA0_INV_MAX)
        assert np.all(m1 > 0)
        assert m1[K == 0.1][0] == pytest.approx(1e-4, rel=0.5)
        assert np.all(m1 < 1e-3)

    def test_validity_window_flag(self):
        flags = in_validity_window(np.array([0.05, K_MAX_LINEAR_HMPC, 0.3]))
        assert flags.tolist() == [True, True, False]

    def test_input_validation(self):
        with pytest.raises(ValueError):
            mu_minus_one_cronos(K, 1.5, THETA, ALPHA0_INV_MAX)
        with pytest.raises(ValueError):
            mu_minus_one_cronos(np.array([-0.1]), 1.0, THETA, ALPHA0_INV_MAX)


class TestAtlasHook:
    def test_default_contributes_nothing(self):
        assert atlas_offset() == 0.0

    def test_explicit_coefficient_is_callers_hypothesis(self):
        assert atlas_offset(epsilon_K=0.012, c_mu=1.0) == pytest.approx(0.012)


class TestGrowthWithMu:
    def test_mu_one_reproduces_growth_D_f_bitwise(self):
        """growth_D_f con mu_of_a ≡ 1 es idéntico bit a bit al camino
        sin µ: la generalización no altera el integrador validado."""
        z = np.array([0.0, 0.5, 1.0, 2.0])
        D0, f0 = growth_D_f(z, THETA)
        D1, f1 = growth_D_f(z, THETA, mu_of_a=lambda a: np.ones_like(a))
        assert np.array_equal(D0, D1) and np.array_equal(f0, f1)

    def test_alpha_zero_ratio_is_exactly_one(self):
        z = np.array([0.0, 1.0, 2.0])
        R = fsigma8_ratio_cronos(z, THETA, 0.1, 0.0)
        assert np.array_equal(R, np.ones_like(R))

    def test_ratio_small_positive_and_monotone_in_alpha(self):
        z = np.array([0.0, 0.5, 1.0])
        R_full = fsigma8_ratio_cronos(z, THETA, 0.2, ALPHA0_INV_MAX)
        R_half = fsigma8_ratio_cronos(z, THETA, 0.2, 0.5 * ALPHA0_INV_MAX)
        assert np.all(R_full > 1.0)          # µ > 1 refuerza el crecimiento
        assert np.all(R_full - 1.0 > R_half - 1.0)
        assert np.max(np.abs(R_full - 1.0)) < 1e-3

    def test_growth_Dk_normalized_today(self):
        D, f = growth_Dk(np.array([0.0, 1.0]), THETA, 0.1, ALPHA0_INV_MAX)
        assert D[0] == pytest.approx(1.0, abs=1e-12)
        assert 0.0 < D[1] < 1.0 and 0.0 < f[1] < 1.2


class TestNonperturbativeDiagnostics:
    def test_physical_closure_diverges_and_fails_closed(self):
        """Bajo 2b con la cota saturada, µ−1 ∝ (1+z)^{7/2} no es
        perturbativo a z alto: la razón fσ8 falla cerrado (nunca NaN)."""
        with pytest.raises(FloatingPointError, match="divergente"):
            fsigma8_ratio_cronos(np.array([0.0, 1.0]), THETA, 0.2,
                                 ALPHA0_INV_MAX, "physical")

    def test_epsilon_c_nonperturbative_z_analytic(self):
        # physical: 1 + z = (f^{3/2}/α)^{2/9}
        z1 = epsilon_c_nonperturbative_z(ALPHA0_INV_MAX, "physical", 1.0)
        expected = (RHO_C_OVER_MEAN ** 1.5 / ALPHA0_INV_MAX) ** (2 / 9) - 1
        assert z1 == pytest.approx(expected, rel=1e-12)
        assert 100 < z1 < 200
        assert epsilon_c_nonperturbative_z(ALPHA0_INV_MAX, "physical",
                                           0.1) < z1
        # comoving: ε̄_c constante ≪ 1 ⟹ nunca
        assert epsilon_c_nonperturbative_z(ALPHA0_INV_MAX, "comoving") \
            == float("inf")
        assert epsilon_c_nonperturbative_z(0.0, "physical") == float("inf")

    def test_z_where_mu_reaches_one(self):
        z_phys = z_where_mu_minus_one_reaches(0.2, THETA, ALPHA0_INV_MAX,
                                              "physical")
        assert 3.0 < z_phys < 30.0          # z de un dígito o dos
        # a k menor el cruce ocurre más tarde (µ−1 ∝ k²)
        assert z_where_mu_minus_one_reaches(0.05, THETA, ALPHA0_INV_MAX,
                                            "physical") > z_phys
        # comoving: µ−1 ≲ 1e-3 en todo z ⟹ nunca alcanza 1
        assert z_where_mu_minus_one_reaches(0.2, THETA, ALPHA0_INV_MAX,
                                            "comoving") == float("inf")
        # la malla fina y la interpolación reproducen la fórmula
        z_c = z_where_mu_minus_one_reaches(0.1, THETA, ALPHA0_INV_MAX,
                                           "physical")
        m1 = mu_minus_one_cronos(0.1, 1.0 / (1.0 + z_c), THETA,
                                 ALPHA0_INV_MAX, "physical")
        assert m1 == pytest.approx(1.0, rel=2e-3)

    def test_background_above_rho_c(self):
        assert background_above_rho_c_z() == pytest.approx(
            200.0 ** (1 / 3) - 1.0)


class TestNoDataGuard:
    def test_prediction_modules_load_no_data_and_fit_nothing(self):
        """Predicción pura: sin cargadores de datos (RSD, lensing,
        registro) ni optimizadores/samplers en el módulo y sus scripts —
        la prohibición «µ, η nunca desde lensing» es ejecutable."""
        for rel in ("cosmology/mu_eta_cronos.py",
                    "scripts/run_mu_eta_cronos.py",
                    "scripts/run_mu_eta_prereg.py"):
            src = (REPO / rel).read_text(encoding="utf-8")
            for pat in ("load_fsigma8_data", "np.loadtxt", "loadtxt(",
                        "require_available", "lensing_data", "C_L_data",
                        "scipy.optimize", "minimize(", "curve_fit",
                        "emcee", "EnsembleSampler"):
                assert pat not in src, f"{rel}: {pat}"
