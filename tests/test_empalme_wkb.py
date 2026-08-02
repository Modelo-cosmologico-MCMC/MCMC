"""Tests del frente nº 7: empalme C¹ (B7) y WKB ab initio (B8) — H.2.4."""

import numpy as np
import pytest

from mcmc_ontology import constants as C
import mass_program.B7_empalme as B7
from mass_program.B8_wkb_ab_initio import (
    MASS_LADDER_GEV, barrier, W_integral, survival_weight, kappa_minimum,
    ladder,
)


# ------------------------------ B7 ---------------------------------------

def test_conventions_consistent():
    """Los dos convenios de β3 difieren exactamente en el factor 4 y
    λ_H medido = m_H²/2v² ≈ 0.130 (aviso de normalización de H.8)."""
    assert abs(B7.BETA3_CONVENIO_12_1 / B7.BETA3_CONVENIO_H8 - 4.0) < 1e-12
    assert abs(B7.lambda_H_measured() - 0.1296) < 1e-3


def test_sealed_curvature_analytic_equals_numeric():
    """λ_Ad = √D: la analítica y las diferencias finitas sobre el Basal
    coinciden — la curvatura sellada es la del potencial real."""
    a = B7.sealed_curvature_lambda(0.012)
    n = B7.sealed_curvature_lambda(0.012, numeric=True)
    assert abs(a - n) / a < 1e-3


def test_no_higgs_input():
    """La derivación NO usa la masa medida: perturbar M_HIGGS_PDG no
    cambia λ_Ad (no circularidad de H.8, verificada de facto)."""
    before = B7.sealed_curvature_lambda(0.02)
    saved = C.M_HIGGS_PDG
    try:
        C.M_HIGGS_PDG = 999.0
        after = B7.sealed_curvature_lambda(0.02)
    finally:
        C.M_HIGGS_PDG = saved
    assert before == after


def test_closure_requires_delta_star():
    """El desenlace cuantificado: λ_Ad(δ0*) = 0.130 con δ0* ≈ 0.0581;
    con δ0 = 0.012 (el valor de ε_Λ que el v32 identificaba con δ0)
    el empalme da m_H ≈ 57 GeV, no 125.3."""
    d0_star = B7.delta0_required()
    assert abs(d0_star - 0.0581) < 5e-4
    assert abs(B7.sealed_curvature_lambda(d0_star) - 0.130) < 1e-12
    assert abs(B7.m_H_predicted(d0_star) - 125.4) < 0.5
    assert B7.m_H_predicted(C.EPSILON_0) < 60.0


def test_audit_report_honest():
    """El informe declara el estatuto condicional, no lo resuelve."""
    r = B7.audit_report()
    assert "CONDICIONAL" in r and "Obs. 12.2" in r


# ------------------------------ B8 ---------------------------------------

DELTA0 = 0.05


def test_W_decreasing_in_E():
    """W(E) decrece con E: más profundo (E menor) → más barrera → W mayor."""
    bar = barrier(DELTA0)
    Es = np.linspace(0.05, 0.9, 6) * bar["V_max"]
    Ws = [W_integral(E, DELTA0, kappa=1.0) for E in Es]
    assert all(a > b for a, b in zip(Ws, Ws[1:]))


def test_exponential_sensitivity():
    """H.2.4: una variación O(1) en E produce órdenes de magnitud en c."""
    bar = barrier(DELTA0)
    kappa = 1.2 * kappa_minimum(DELTA0)
    c_hi = survival_weight(0.5 * bar["V_max"], DELTA0, kappa)
    c_lo = survival_weight(0.25 * bar["V_max"], DELTA0, kappa)
    assert c_hi / c_lo > 1e3


def test_single_kappa_covers_13_orders():
    """Una sola escala de acción cubre m_t/m_ν ≈ 3×10¹³ (κ_min existe y
    la escalera con κ = 1.2κ_min contiene TODOS los modos)."""
    ratio = MASS_LADDER_GEV["t"] / MASS_LADDER_GEV["nu_tau"]
    assert ratio > 1e13
    lad = ladder(DELTA0)
    assert set(lad["E"]) == set(MASS_LADDER_GEV)
    bar = barrier(DELTA0)
    for E in lad["E"].values():
        assert 0.0 < E < bar["V_max"]


def test_ladder_monotone_with_mass():
    """La escalera: más pesado → más somero; el neutrino el más profundo
    (enlace con el seesaw, Prop. 12.5)."""
    lad = ladder(DELTA0)["E"]
    by_mass = sorted(MASS_LADDER_GEV, key=MASS_LADDER_GEV.get, reverse=True)
    depths = [lad[k] for k in by_mass]
    assert all(a > b for a, b in zip(depths, depths[1:]))
    assert min(lad, key=lad.get) == "nu_tau"
    assert max(lad, key=lad.get) == "t"


def test_kappa_too_small_fails_honestly():
    """Con κ insuficiente la escalera no cubre el rango y lo dice."""
    with pytest.raises(ValueError):
        ladder(DELTA0, kappa=0.5 * kappa_minimum(DELTA0))
