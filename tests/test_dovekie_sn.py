"""Unidades del pipeline SN Dovekie (PR #15): fórmula, parsers,
covarianzas, modelo de distancias y maquinaria de mocks.
"""

import numpy as np
import pytest

from cosmology.dovekie_sn import (
    chi2_sn_marginalized,
    load_dovekie_design,
    load_dovekie_inv_cov,
    mu_model,
)
from validation.dovekie_independent import (
    chi2_marginalized_independent,
    mu_model_independent,
)
from validation.dovekie_mocks import official_cov_log_likelihood


@pytest.fixture(scope="module")
def small_problem():
    """Problema sintético pequeño con W SPD y vectores reproducibles."""
    rng = np.random.default_rng(7)
    n = 40
    A = rng.normal(size=(n, n))
    W = A @ A.T + n * np.eye(n)
    mu_mod = 38.0 + rng.normal(scale=0.5, size=n)
    mu_dat = mu_mod + rng.normal(scale=0.1, size=n)
    return mu_mod, mu_dat, W


class TestChi2Formula:
    def test_M_shift_invariance(self, small_problem):
        """La marginalización analítica de M hace χ̃² EXACTAMENTE
        invariante bajo un offset constante del dato."""
        mu_mod, mu_dat, W = small_problem
        base = chi2_sn_marginalized(mu_mod, mu_dat, W)
        for c in (-3.0, 0.7, 19.0):
            shifted = chi2_sn_marginalized(mu_mod, mu_dat + c, W)
            assert shifted == pytest.approx(base, abs=1e-8)

    def test_A_equals_B(self, small_problem):
        mu_mod, mu_dat, W = small_problem
        a = chi2_sn_marginalized(mu_mod, mu_dat, W)
        b = chi2_marginalized_independent(mu_mod, mu_dat, W)
        assert a == pytest.approx(b, abs=1e-9)

    def test_A_equals_official_bytes(self, small_problem):
        """La fórmula de producción coincide con cov_log_likelihood
        extraída por AST de los bytes del release (χ² = −2·logL)."""
        mu_mod, mu_dat, W = small_problem
        official = official_cov_log_likelihood()
        chi_off = -2.0 * float(official(mu_mod, mu_dat, W))
        assert chi2_sn_marginalized(mu_mod, mu_dat, W) == \
            pytest.approx(chi_off, abs=1e-9)


class TestRelease:
    def test_design_count_and_columns(self):
        d = load_dovekie_design()
        assert d["n_sn"] == 1820
        assert len(d["zHD"]) == len(d["zHEL"]) == 1820
        assert np.all(d["zHD"] > 0)

    def test_inv_cov_symmetric_and_sized(self):
        for kind in ("STATONLY", "STAT+SYS"):
            W = load_dovekie_inv_cov(kind)
            assert W.shape == (1820, 1820)
            assert np.array_equal(W, W.T)
            assert np.all(np.diag(W) > 0)

    def test_inv_cov_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            load_dovekie_inv_cov("SYSONLY")


class TestDistanceModel:
    def test_lcdm_limit_z_trans_irrelevant(self):
        """Con ε = 0 el fondo es ΛCDM exacto: z_trans no cambia μ."""
        z = np.array([0.05, 0.2, 0.6, 1.1])
        a = mu_model(z, z, 0.315, eps=0.0, z_trans=1.0)
        b = mu_model(z, z, 0.315, eps=0.0, z_trans=20.0)
        assert np.array_equal(a, b)

    def test_monotonic_in_z(self):
        z = np.linspace(0.01, 1.1, 50)
        mu = mu_model(z, z, 0.315)
        assert np.all(np.diff(mu) > 0)

    def test_production_vs_independent_quadrature(self):
        """Puerta 1b en miniatura: integrador O(h⁴) de producción vs
        cuadratura adaptativa independiente."""
        z = np.array([0.03, 0.15, 0.42, 0.78, 1.10])
        zhel = z * 1.001
        for om, eps, zt in ((0.315, 0.0, 8.9), (0.20, 0.05, 3.0)):
            a = mu_model(z, zhel, om, eps=eps, z_trans=zt)
            b = mu_model_independent(z, zhel, om, eps=eps, z_trans=zt)
            assert np.max(np.abs(a - b)) < 1e-5

    def test_out_of_grid_fails_closed(self):
        with pytest.raises(ValueError, match="fuera de la malla"):
            mu_model(np.array([2.0]), np.array([2.0]), 0.3)


class TestMockGrid:
    def test_grid_posterior_recovers_synthetic_truth(self, monkeypatch):
        """Humo del brazo ΛCDM: sobre un problema sintético pequeño el
        posterior en malla recupera la verdad dentro de su CI."""
        import validation.dovekie_mocks as dm
        monkeypatch.setattr(dm, "OMEGA_GRID",
                            np.linspace(0.05, 0.7, 131))
        rng = np.random.default_rng(11)
        z = np.linspace(0.02, 1.05, 60)
        zhel = z.copy()
        om_true = 0.315
        sigma = 0.08
        W = np.eye(60) / sigma ** 2
        mu_true = mu_model(z, zhel, om_true)
        data = mu_true + 0.3 + rng.normal(scale=sigma, size=60)
        grid = dm.lcdm_grid_mu(z, zhel)
        pct, chi2_min, om_hat = dm.lcdm_posterior_percentiles(
            grid, W, data)
        assert pct[0.025] <= om_true <= pct[0.975]
        assert chi2_min < chi2_sn_marginalized(
            mu_model(z, zhel, 0.10), data, W)
        assert 0.05 <= om_hat <= 0.7
