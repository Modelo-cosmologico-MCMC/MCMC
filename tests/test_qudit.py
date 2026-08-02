"""Tests del qudit ontológico (d=5) y de las compuertas de colapso."""

import numpy as np

from quantum.gates import GATE_THRESHOLDS_S, X_collapse, apply_gate_at_S
from quantum.hamiltonian import H_MCMC
from quantum.qudit import D, basis, fidelity, state


def test_gate_thresholds_are_decade():
    """Los umbrales de disparo son los de la Ley de la Década (v35 C.2)."""
    assert list(GATE_THRESHOLDS_S.values()) == [0.009, 0.099, 0.999, 1.001]


def test_apply_gate_at_S_fires_only_at_threshold():
    """X̂0→1 no dispara antes de S=0.009 y sí a partir de él."""
    psi = basis(0)
    before = apply_gate_at_S(psi, 0, S=0.005)
    after = apply_gate_at_S(psi, 0, S=0.009)
    assert np.allclose(before, basis(0))
    assert np.allclose(after, basis(1))


def test_basis_dim():
    assert basis(0).shape == (D,)
    assert np.allclose(basis(2)[2], 1.0)


def test_state_normalization():
    s = state(np.ones(D))
    assert abs(np.linalg.norm(s) - 1.0) < 1e-12


def test_fidelity_self():
    s = basis(3)
    assert abs(fidelity(s, s) - 1.0) < 1e-12


def test_H_hermitian():
    H = H_MCMC()
    assert np.allclose(H, H.conj().T)


def test_X_collapse_swap():
    U = X_collapse(0)
    s0 = basis(0)
    s1 = U @ s0
    assert np.allclose(s1, basis(1))
