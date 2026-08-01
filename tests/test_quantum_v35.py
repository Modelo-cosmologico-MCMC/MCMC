"""Tests del apéndice C de la v35: H_ten, D̂, Û(θ), Lindblad y Γn/Γ0."""

import numpy as np
import pytest

from quantum.qudit import basis, D
from quantum.hamiltonian import H_ten, dimension_operator
from quantum.gates import U_collapse, gate_condition, X_collapse
from quantum.decoherence import (
    evolve, fidelity_with, gamma_ratio,
)
from quantum.qutip_simulation import EXPECTED_FIDELITIES


def test_H_ten_diagonal_increasing():
    """H_ten = Σ E_n|Sn⟩⟨Sn| con E0 < ... < E4 (C.2)."""
    H = H_ten()
    assert np.allclose(H, np.diag(np.diag(H)))
    assert np.all(np.diff(np.real(np.diag(H))) > 0)
    with pytest.raises(ValueError):
        H_ten(np.array([1.0, 1.0, 2.0, 3.0, 4.0]))  # no estrictamente creciente


def test_dimension_operator_expectation():
    """⟨D̂⟩ sigue el conteo de dimensiones emergidas (C.1)."""
    Dop = dimension_operator()
    for n in range(D):
        psi = basis(n)
        assert abs(np.real(psi.conj() @ Dop @ psi) - n) < 1e-12


def test_U_collapse_unitary_and_full_transfer():
    """Û(θ) es unitaria; con θ=π/2 transfiere toda la población (C.3)."""
    for n in range(D - 1):
        U = U_collapse(n, 0.3)
        assert np.allclose(U @ U.conj().T, np.eye(D), atol=1e-12)
    U = U_collapse(0, np.pi / 2.0)
    psi = U @ basis(0)
    assert abs(abs(psi[1]) ** 2 - 1.0) < 1e-12
    # Coincide con X_collapse salvo fase global −i en el subespacio:
    assert np.allclose(np.abs(psi), np.abs(X_collapse(0) @ basis(0)))


def test_gate_condition_double():
    """Disparo con condición doble τ ≥ τ_crit Y S ≥ S_min (C.2)."""
    assert gate_condition(1.1, 1.0, 0.010, 0.009)
    assert not gate_condition(0.9, 1.0, 0.010, 0.009)   # τ insuficiente
    assert not gate_condition(1.1, 1.0, 0.008, 0.009)   # S insuficiente


def test_lindblad_preserves_trace_and_decoheres():
    """(C.4): la traza se conserva y la coherencia fuera de diagonal decae."""
    psi = (basis(0) + basis(1)) / np.sqrt(2.0)
    rho0 = np.outer(psi, psi.conj())
    H = H_ten()
    Phi = dimension_operator()  # acopla a la estructura de niveles
    rho_t = evolve(rho0, H, Phi, lam_ten=1.0, t_final=2.0, n_steps=400)
    assert abs(np.trace(rho_t).real - 1.0) < 1e-6
    assert abs(rho_t[0, 1]) < abs(rho0[0, 1])  # decoherencia
    # Sin canal tensional (λ=0) la evolución es unitaria: coherencia intacta
    rho_u = evolve(rho0, H, Phi, lam_ten=0.0, t_final=2.0, n_steps=400)
    assert abs(abs(rho_u[0, 1]) - abs(rho0[0, 1])) < 1e-6


def test_fidelity_decays():
    """F(t) = ⟨Ψ0|ρ(t)|Ψ0⟩ decae bajo el canal tensional (C.3)."""
    psi = (basis(0) + basis(2)) / np.sqrt(2.0)
    rho0 = np.outer(psi, psi.conj())
    rho_t = evolve(rho0, H_ten(), dimension_operator(), 0.5, 1.0)
    assert fidelity_with(rho_t, psi) < fidelity_with(rho0, psi)


def test_gamma_ratio_signature():
    """(C.5): Γn/Γ0 crece con n si ξ_ten > 0; ξ_ten = 0 → sin exceso."""
    E = np.arange(1.0, 6.0)
    ratios = [gamma_ratio(n, xi_ten=0.5, eta_void_n=1.0,
                          E_n=E[n], E_4=E[4]) for n in range(5)]
    assert all(np.diff(ratios) > 0)
    assert gamma_ratio(3, 0.0, 1.0, E[3], E[4]) == 1.0


def test_expected_fidelities_v35_pattern():
    """La tabla v35 (C.4) cumple el patrón decreciente F0 > F1 > F2 > F3."""
    vals = [EXPECTED_FIDELITIES[k] for k in
            ("S0->S1", "S1->S2", "S2->S3", "S3->S4")]
    assert vals == [0.981, 0.976, 0.968, 0.961]
    assert all(a > b for a, b in zip(vals, vals[1:]))
