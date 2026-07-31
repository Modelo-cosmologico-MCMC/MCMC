"""Tests del acoplo de Wilson entrópico y el mapa Sₙ→jₙ (v35, Ap. D)."""

import numpy as np

from mcmc_ontology import constants as C
from lattice.wilson_entropic import (
    beta_S, lqg_area, seal_spin, seal_area_table, SEAL_TO_SPIN, S3_QCD,
)


def test_beta_S_at_seal():
    """En S = S3 el acoplo vale β0 + β1 (el exponente se anula)."""
    assert abs(beta_S(S3_QCD, beta0=6.0, beta1=1.0) - 7.0) < 1e-12


def test_beta_S_decays_after_seal():
    """β(S) decrece hacia β0 para S > S3 (D.1)."""
    S = np.linspace(1.0, 2.0, 50)
    b = beta_S(S)
    assert np.all(np.diff(b) < 0)
    assert b[-1] > 6.0  # asintótico a beta0 por arriba


def test_seal_to_spin_map():
    """Prop. D.1: los cuatro sellos de la Década ↔ j = 1/2, 3/2, 5/2, 7/2."""
    assert seal_spin(0.009) == 0.5
    assert seal_spin(0.099) == 1.5
    assert seal_spin(0.999) == 2.5
    assert seal_spin(1.001) == 3.5
    assert set(SEAL_TO_SPIN) == {0.009, 0.099, 0.999, 1.001}


def test_lqg_area_monotone_in_j():
    """A(j) = 8πγℓ_P²√(j(j+1)) crece con j; usa γ* = 0.274 sellado."""
    areas = [lqg_area(j) for j in (0.5, 1.5, 2.5, 3.5)]
    assert all(a > 0 for a in areas)
    assert areas == sorted(areas)
    assert abs(lqg_area(0.5) - 8.0 * np.pi * C.GAMMA_LQG * np.sqrt(0.75)) < 1e-12


def test_seal_area_table_complete():
    table = seal_area_table()
    assert set(table) == set(SEAL_TO_SPIN)
