"""Tests del WKB analítico (III.A, Ecs. 478-482)."""

import numpy as np
import pytest

from mass_program.B3_wkb_analytic import (
    transmission_analytic, m_effective_at_seal,
    transmission_amplitude, family_transmission_analytic,
    reconstruction_consistency,
)


def test_T_in_unit_interval():
    """|T|² ∈ [0, 1] para argumentos físicos."""
    T2 = transmission_analytic(0.9, 0.5, 0.7)
    assert 0.0 <= T2 <= 1.0


def test_T_kinematically_forbidden_with_wide_barrier():
    """E < min(m_-, m_+) y barrera ancha → T fuertemente suprimida.

    Con una barrera de grosor λ=10 (S-units), la transmisión está
    exponencialmente suprimida por |Δm| y por κ.
    """
    T2 = transmission_analytic(E=0.1, m_minus=0.99, m_plus=0.95, lam=10.0)
    assert T2 < 1e-3


def test_T_kinematically_forbidden_narrow_barrier_partial():
    """E < min(m) con barrera estrecha (λ=0.01) → T < 1 pero no exponencial.

    Refleja que las barreras estrechas no suprimen totalmente: el modo
    evanescente atraviesa el sello con T ≈ exp(-2λκ) ≈ 0.98.
    """
    T2 = transmission_analytic(E=0.1, m_minus=0.99, m_plus=0.95, lam=0.01)
    assert 0.0 < T2 < 1.0


def test_T_unit_when_no_step():
    """Sin salto de masa (m_- = m_+) → kinemático = 1, atenuación = 1."""
    T2 = transmission_analytic(0.95, 0.5, 0.5)
    assert abs(T2 - 1.0) < 1e-10


def test_m_effective_consistent_with_profile():
    """m^(±) en C2 difiere por el grosor 2λ del perfil m_P(S)."""
    m_minus, m_plus = m_effective_at_seal("C2")
    assert m_minus > m_plus  # m_P decrece con S
    assert m_minus < 0.99 + 1e-9  # acotado por valor inicial
    assert m_plus  > 0.50 - 1e-9


def test_transmission_amplitude_F2_finite():
    """|T_n^(F2)| analítico finito en los 4 sellos."""
    out = family_transmission_analytic("F2")
    for seal in ("C1", "C2", "C3", "C4"):
        assert 0.0 <= out[seal] <= 1.0


def test_consistency_report_has_all_families():
    """`reconstruction_consistency()` devuelve los 3×4 entradas."""
    rep = reconstruction_consistency()
    assert set(rep.keys()) == {"F1", "F2", "F3"}
    for fam in rep:
        assert set(rep[fam].keys()) == {"C1", "C2", "C3", "C4"}
        for seal in rep[fam]:
            entry = rep[fam][seal]
            assert "analytic" in entry and "tabulated" in entry
