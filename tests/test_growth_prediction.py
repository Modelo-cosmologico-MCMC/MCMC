"""Sector perturbativo (rama theory/perturbations-linear-growth):
identidades del crecimiento exacto y maquinaria de la banda. El
candado de la preinscripción vive en tests/test_perturbations_lock.py.
"""

from pathlib import Path

import numpy as np
import pytest

from cosmology.growth_prediction import (
    Z_BAND,
    fD_of_z,
    load_background_posterior,
    posterior_band,
    prior_envelope,
    ratio_fsigma8,
)

REPO = Path(__file__).resolve().parent.parent

THETA_REF = (67.87, 0.326, 0.017, 8.9)   # medianas publicadas del v1


class TestGrowthIdentities:
    def test_ratio_is_exactly_one_at_eps_zero(self):
        """Con ε = 0 el fondo es ΛCDM exacto y R(z) ≡ 1 (mismo
        integrador, misma malla): identidad, no aproximación."""
        z = np.linspace(0.0, 2.0, 21)
        R = ratio_fsigma8(z, (67.87, 0.326, 0.0, 8.9))
        assert np.array_equal(R, np.ones_like(R))

    def test_ratio_small_for_prior_eps(self):
        """Para ε de la escala del prior, la desviación del sector
        lineal es pequeña (el fondo casi no cambia): |R − 1| < 0.05."""
        z = np.linspace(0.0, 2.0, 21)
        for eps in (-0.05, 0.05, 0.10):
            R = ratio_fsigma8(z, (67.87, 0.326, eps, 8.9))
            assert np.max(np.abs(R - 1.0)) < 0.05

    def test_fD_positive_and_finite(self):
        z = np.linspace(0.0, 2.0, 21)
        fD = fD_of_z(z, THETA_REF)
        assert np.all(np.isfinite(fD)) and np.all(fD > 0)

    def test_envelope_positive(self):
        env = prior_envelope(Z_BAND, THETA_REF)
        assert env.shape == Z_BAND.shape
        assert np.all(env >= 0) and np.any(env > 0)


class TestPosteriorMachinery:
    def test_loader_validates_param_order(self, tmp_path):
        bad = tmp_path / "chains.npz"
        np.savez(bad, chain=np.zeros((10, 4)), log_prob=np.zeros(10),
                 params=np.array(["a", "b", "c", "d"]))
        with pytest.raises(RuntimeError, match="orden de parámetros"):
            load_background_posterior(bad)

    def test_committed_chains_load(self):
        chain = load_background_posterior()
        assert chain.shape == (16000, 4)
        assert 60 < np.median(chain[:, 0]) < 75

    def test_band_percentiles_ordered(self):
        rng = np.random.default_rng(3)
        tiny = np.column_stack([
            rng.normal(67.9, 0.5, 6), rng.normal(0.32, 0.01, 6),
            rng.normal(0.02, 0.02, 6), rng.uniform(5, 12, 6)])
        band = posterior_band(tiny, z=np.linspace(0, 2, 5),
                              n_draws=6, seed=1)
        assert np.all(band["ratio_p16"] <= band["ratio_p50"] + 1e-15)
        assert np.all(band["ratio_p50"] <= band["ratio_p84"] + 1e-15)
        assert band["n_draws"] == 6


class TestNoFittingGuard:
    def test_prediction_modules_fit_nothing(self):
        """Predicción, no ajuste: sin optimizadores ni samplers en el
        módulo de predicción ni en sus scripts (σ8 jamás se ajusta)."""
        for rel in ("cosmology/growth_prediction.py",
                    "scripts/run_perturbations_prediction.py",
                    "scripts/run_perturbations_prereg.py"):
            src = (REPO / rel).read_text(encoding="utf-8")
            for pat in ("scipy.optimize", "minimize(", "curve_fit",
                        "least_squares", "emcee", "EnsembleSampler"):
                assert pat not in src, f"{rel}: {pat}"
