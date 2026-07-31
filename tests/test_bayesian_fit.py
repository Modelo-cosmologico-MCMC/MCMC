"""Tests de la maquinaria de inferencia (sin emcee ni datos reales)."""

import numpy as np

from mcmc_ontology import constants as C
from cosmology.bayesian_fit import (
    synthetic_selftest_dataset, log_prior, log_like_Hz, log_prob,
    comoving_distance, distance_modulus, BAOData, log_like_bao,
)


THETA0 = (C.H0_MCMC, 0.300, C.EPSILON_0, C.Z_TRANS)


def test_selftest_dataset_shapes():
    data = synthetic_selftest_dataset(n=16, seed=1)
    assert len(data.z) == len(data.H) == len(data.sig) == 16
    assert np.all(data.sig > 0)


def test_log_prior_finite_at_reference():
    assert np.isfinite(log_prior(THETA0))
    assert log_prior((90.0, 0.3, 0.012, 8.9)) == -np.inf


def test_log_prob_selftest():
    """El log-posterior en el punto generador debe ser finito y razonable."""
    data = synthetic_selftest_dataset(n=16, seed=1)
    lp = log_prob(THETA0, data)
    assert np.isfinite(lp)
    # χ² esperado ~ n para datos generados por el propio modelo
    assert log_like_Hz(THETA0, data) > -3 * len(data.z)


def test_comoving_distance_monotone():
    z = np.array([0.1, 0.5, 1.0, 2.0])
    d = comoving_distance(z, THETA0)
    assert np.all(np.diff(d) > 0)
    assert np.all(d > 0)


def test_distance_modulus_range():
    """μ(z~0.5) ≈ 42 para una cosmología razonable."""
    mu = distance_modulus(np.array([0.5]), THETA0)
    assert 41.0 < mu[0] < 43.5


def test_log_like_bao_evaluates():
    bao = BAOData(
        z=np.array([0.38, 0.38]),
        kind=["DM_over_rd", "DH_over_rd"],
        value=np.array([10.27, 24.89]),
        sig=np.array([0.15, 0.58]),
    )
    ll = log_like_bao(THETA0, bao)
    assert np.isfinite(ll)
