"""Tests del acoplo de Wilson entrópico y el mapa Sₙ→jₙ (v35, Ap. D)."""

import numpy as np

from lattice.wilson_entropic import (
    S3_QCD,
    SEAL_TO_SPIN,
    beta_S,
    lqg_area,
    seal_area_table,
    seal_spin,
)
from mcmc_ontology import constants as C


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


def test_vertex_amplitude_ratio():
    """D.4: A_v crece con j; cociente (2j_n+1)/(2j_prev+1) elevado a ΔN."""
    from lattice.wilson_entropic import partition_ratio, vertex_amplitude_ratio
    r = vertex_amplitude_ratio(1.5, 0.5, delta_N=1.0)
    assert abs(r - 2.0) < 1e-12  # (2·1.5+1)/(2·0.5+1) = 4/2
    assert vertex_amplitude_ratio(2.5, 1.5) > 1.0
    assert abs(partition_ratio(1e-3) - np.exp(1e-3)) < 1e-15


def test_E_min_lattice_step_at_higgs():
    """D.6: E_min(S) tiene escalón en ~1.0 y post-escalón = sqrt(2β3)·v3
    con la elección φ*²=v3²(1+2/3), α=0 (identidad de la Prop. 12.1)."""
    from lattice.mass_gap import E_min_lattice
    below = E_min_lattice(0.5)
    above = E_min_lattice(1.001)
    assert below < 1e-3 * above          # escalón: casi nulo antes
    expected = np.sqrt(2.0 * 0.13) * C.V3_GEV
    assert abs(above - expected) / expected < 0.02
    # 125.436 GeV: la identidad del Higgs (con su auditoría, Obs. 12.2)
    assert abs(above - 125.436) < 0.5
