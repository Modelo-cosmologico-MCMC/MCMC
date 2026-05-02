"""Tests del qudit ontológico (d=5)."""

import numpy as np

from quantum.qudit import basis, state, fidelity, D
from quantum.hamiltonian import H_MCMC
from quantum.gates import X_collapse


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
