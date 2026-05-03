"""Tests del módulo cosmológico."""

import numpy as np

from mcmc_ontology import constants as C
from cosmology.background import H_of_z, Lambda_rel
from cosmology.perturbations import f_sigma8


def test_H0_in_range():
    """H(z=0) ≈ H0_MCMC."""
    H = H_of_z(0.0)
    assert 68.5 < H < 71.0


def test_Lambda_at_today():
    """Ω_Λ(z=0) ≈ Ω_Λ0 (1 + ε)."""
    Om = Lambda_rel(0.0)
    assert 0.65 < Om < 0.75


def test_Hz_monotone():
    """H(z) creciente en z."""
    z = np.linspace(0.0, 3.0, 50)
    H = H_of_z(z)
    assert np.all(np.diff(H) > 0)


def test_fsigma8_finite():
    z = np.linspace(0.05, 1.5, 10)
    fs = f_sigma8(z)
    assert np.all(np.isfinite(fs))
    assert np.all(fs > 0)


def test_epsilon_value():
    assert abs(C.EPSILON_0 - 0.012) < 1e-9


def test_delta_BIC_negative():
    assert C.DELTA_BIC < -5.0


def test_w_id_near_minus_one():
    """w_id(z) ≈ -1 (sector de energía oscura ~ Λ con dinámica leve)."""
    from cosmology.background import w_id
    z = np.linspace(0.0, 5.0, 20)
    w = w_id(z)
    assert np.all(np.isfinite(w))
    # No debe alejarse mucho de -1 con la parametrización fiducial.
    assert np.all(np.abs(w + 1.0) < 0.5)


def test_cs2_id():
    """c²_s,id = 1 (Tratado §6.7)."""
    from cosmology.background import cs2_id
    assert cs2_id() == 1.0
