"""Tests del WKB espinorial.

Los test_regression_* comparan las funciones de B3_wkb contra los valores
calibrados de constants.py: verifican que la calibración declarada sigue
siendo autoconsistente, NO que los pesos estén derivados de primeros
principios (frente abierto nº 7 del Tratado de Fundamentos).
"""

from mass_program.B3_wkb import (
    E_F2,
    E_F3,
    kappa_gap_12,
    kappa_gap_23,
    sequential_tunnel,
    transmission,
    verify_calibration,
)
from mcmc_ontology import constants as C


def test_regression_kappa_gap_12():
    assert abs(kappa_gap_12() - C.KAPPA_GAP_12) < 1e-3


def test_regression_kappa_gap_23():
    assert abs(kappa_gap_23() - C.KAPPA_GAP_23) < 1e-3


def test_regression_E_F2():
    assert abs(E_F2() - C.E_F2) < 1e-3


def test_regression_E_F3():
    assert abs(E_F3() - C.E_F3) < 1e-3


def test_regression_T_F2_C1():
    """|T_1^(F2)| ≈ 4.30e-4 dentro del 0.1% (entrada calibrada, Tabla 3 v32)."""
    assert abs(transmission("F2", "C1") - C.T_UNIVERSAL["F2"][0]) / C.T_UNIVERSAL["F2"][0] < 1e-3


def test_regression_T_F3_C2():
    """|T_2^(F3)| ≈ 2.10e-3 (entrada calibrada, Tabla 3 v32)."""
    assert abs(transmission("F3", "C2") - C.T_UNIVERSAL["F3"][1]) / C.T_UNIVERSAL["F3"][1] < 1e-3


def test_sequential_tunnel_consistency():
    """Túnel secuencial pre-emergencia: π de exp(-κΔS/λ)."""
    # F3 desde C1 (n=0) hasta su emergencia C3 (n=2):
    seq = sequential_tunnel("F3", "C1")
    # Debe ser muy pequeño (electrón pre-pre-emergente)
    assert 0.0 < seq < 1e-3


def test_regression_calibration_verifier():
    rep = verify_calibration()
    for k, v in rep.items():
        assert v[2], f"{k} fuera de tolerancia: {v}"
