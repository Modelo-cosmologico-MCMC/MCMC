"""Ronda de registro 17-sep: la amplitud de Cronos es una (A_Sculptor).
Candado del cálculo de consistencia: los dos cierres difieren en ~1e7-1e8,
el cosmológico viola la subdominancia dentro de halos NFW y el galáctico
la cumple; con A_Sculptor la cola k² de µ es ~1e-12.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cosmology.mu_eta_cronos import RHO_C_OVER_MEAN, mu_minus_one_cronos
from cronos.cronos_v3 import ALPHA0_INV_MAX
from dynamics.cronos_amplitude_validity import (
    A_SCULPTOR,
    cosmological_closure_amplitude,
    dominance_ratio,
    galactic_closure_f_over_mean,
    galactic_closure_rho_c,
    max_subdominant_amplitude,
    nfw_halo,
    radius_where_dominance_is_one,
    rho_mean_matter_today,
    validity_report,
)

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "results" / "2026-09-17_cronos_amplitude_validity" / "cronos_amplitude_validity.json"
H0, OM = 67.867, 0.3263


def test_amplitude_is_the_frozen_5e_value():
    assert A_SCULPTOR == pytest.approx(6.201589e-7, rel=1e-5)


def test_two_closures_differ_by_seven_orders():
    rho0 = rho_mean_matter_today(H0, OM)
    A_c = cosmological_closure_amplitude(ALPHA0_INV_MAX, rho0, RHO_C_OVER_MEAN)
    assert 3e7 < A_c / A_SCULPTOR < 1.5e8
    # ρ_c galáctica equivalente ~ 1.4 M☉/pc³ ≫ 200 ρ̄_m
    assert galactic_closure_rho_c(A_SCULPTOR) == pytest.approx(1.375, rel=0.02)
    assert galactic_closure_f_over_mean(A_SCULPTOR, rho0) > 1e7


def test_nfw_halo_reference_numbers():
    h = nfw_halo(1e11, 10.0, H0, 0.10)
    assert h["r200_kpc"] == pytest.approx(95.8, rel=0.03)
    assert h["r_s_kpc"] == pytest.approx(9.58, rel=0.03)
    assert h["v_circ_rs_kms"] == pytest.approx(76.0, rel=0.05)
    assert np.all(np.diff(h["M_enc"]) > 0)


def test_cosmological_closure_excluded_and_galactic_subdominant():
    rho0 = rho_mean_matter_today(H0, OM)
    A_c = cosmological_closure_amplitude(ALPHA0_INV_MAX, rho0, RHO_C_OVER_MEAN)
    h = nfw_halo(1e11, 10.0, H0, 0.10)
    D_c = dominance_ratio(h, A_c)
    D_g = dominance_ratio(h, A_SCULPTOR)
    assert D_c[-1] > 1.0 and D_c.max() > 1e3           # excluido en todo el halo
    assert D_g.max() < 1.0                              # subdominante en r ≥ 0.1 kpc
    assert D_g[-1] < 1e-3
    # A_Sculptor está en la amplitud máxima subdominante (mismo orden)
    assert 0.5 < max_subdominant_amplitude(h) / A_SCULPTOR < 2.5
    # en r200 el cierre cosmológico sigue en régimen débil (la exclusión
    # allí es por subdominancia), pero en el interior rompe también el
    # régimen débil: ε_c > 1 por debajo de ~0.4 kpc
    eps_c = A_c * h["rho"] ** 1.5
    assert eps_c[-1] < 1e-3 and eps_c.max() > 1.0
    assert (A_SCULPTOR * h["rho"] ** 1.5).max() < 1e-3
    # halo enano de 1e10 M☉ (c = 13, suavizado 50 pc): A_Sculptor deja de
    # ser subdominante por debajo de ~0.15 kpc
    h10 = nfw_halo(1e10, 13.0, H0, 0.05)
    r1 = radius_where_dominance_is_one(h10, A_SCULPTOR)
    assert r1 is not None and 0.08 < r1 < 0.3
    assert radius_where_dominance_is_one(h, A_SCULPTOR) is None


def test_k2_tail_dies_with_galactic_amplitude():
    rho0 = rho_mean_matter_today(H0, OM)
    theta = (H0, OM, 0.0172, 9.09)
    f_gal = galactic_closure_f_over_mean(A_SCULPTOR, rho0)
    mu_gal = float(mu_minus_one_cronos(0.2, 1.0, theta, ALPHA0_INV_MAX, "physical", f_gal))
    mu_cos = float(mu_minus_one_cronos(0.2, 1.0, theta, ALPHA0_INV_MAX, "comoving", RHO_C_OVER_MEAN))
    assert 0 < mu_gal < 1e-10
    assert mu_cos / mu_gal > 1e6


def test_force_ratio_dominates_inside_kpc_at_galactic_amplitude():
    """D_F ≫ D_Φ en la cúspide: con A_Sculptor la fuerza de Cronos iguala
    a la gravedad en ~0.9 kpc del halo de 1e11 M☉ y la supera ×100 en
    0.1 kpc, aunque D_Φ < 1 en todo r ≥ 0.1 kpc."""
    from dynamics.cronos_amplitude_validity import (
        force_dominance_ratio,
        radius_where_force_ratio_is_one,
    )
    h = nfw_halo(1e11, 10.0, H0, 0.10)
    DF = force_dominance_ratio(h, A_SCULPTOR)
    DP = dominance_ratio(h, A_SCULPTOR)
    assert DF[0] > 100.0 and DP[0] < 1.0
    inner = h["r_kpc"] <= 10.0
    assert np.all(DF[inner] / DP[inner] > 10.0)
    r1 = radius_where_force_ratio_is_one(h, A_SCULPTOR)
    assert 0.6 < r1 < 1.2
    i23 = np.argmin(np.abs(h["r_kpc"] - 2.3))
    assert DF[i23] < 0.2


def test_report_is_serializable_and_consistent():
    rep = validity_report(H0, OM)
    json.dumps(rep)
    h11 = rep["halos"][0]["closures"]
    assert h11["cosmological"]["subdominant_everywhere"] is False
    assert h11["galactic"]["subdominant_everywhere"] is True
    assert rep["epsilon_c_background_today"]["galactic"] < 1e-16


def test_artifact_if_present_matches_module():
    if not ARTIFACT.exists():
        pytest.skip("artefacto de la ronda de registro ausente")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rep = validity_report(doc["inputs"]["H0"], doc["inputs"]["Omega_m"])
    assert doc["A_cosmo_over_A_galactic"] == pytest.approx(rep["A_cosmo_over_A_galactic"], rel=1e-9)
    assert doc["halos"][0]["closures"]["cosmological"]["D_r200"] == pytest.approx(
        rep["halos"][0]["closures"]["cosmological"]["D_r200"], rel=1e-9)
    assert doc["mu_minus_one_at_k0p2"]["0.0"]["galactic_A_sculptor"] < 1e-10
