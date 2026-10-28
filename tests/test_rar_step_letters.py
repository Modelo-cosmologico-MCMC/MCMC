"""Letras del escalón de la RAR (dynamics/rar_step_letters.py), sin datos: la predicción congelada intacta, la RAR de un
parámetro, R*, el estadístico de ventanas, las letras por punto con sus puertas y la lectura literal, sobre galaxias
SINTÉTICAS declaradas (no observacionales)."""

import math

import numpy as np
import pytest

from dynamics import mcv_baryons as mb
from dynamics import mcv_sparc as ms
from dynamics import rar_step_letters as rs
from dynamics.epsilon_c_saturating import exponential_disc_g_bar
from dynamics.sparc_bank import KMS2_PER_KPC_SI

G_DAGGER = 1.2e-10


def _galaxies(n=40, step_point=None, noise=0.0, seed=0):
    """Discos exponenciales SINTÉTICOS con g_obs = g_ref(g_bar) (sin escalón) o g_ref + g_C (con el escalón de la ley
    congelada en step_point), con ruido gaussiano en dex."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        Sigma0 = 10 ** rng.uniform(1.8, 3.0)               # M☉/pc² con Υ_d = 0.5
        R_d = rng.uniform(1.0, 4.0)
        R = np.geomspace(0.2 * R_d, 8.0 * R_d, 60)
        v2b = exponential_disc_g_bar(R, Sigma0, R_d) * R
        g_bar = v2b / R
        g_obs = rs.g_ref(g_bar, G_DAGGER / KMS2_PER_KPC_SI)
        gal = ms.Galaxy(f"S{k:02d}", R, np.zeros_like(R), np.ones_like(R), np.zeros_like(R), np.sqrt(v2b / 0.5),
                        np.zeros_like(R), 10.0, 1.0, 60.0, 4.0, R_d=R_d, SB_disk0=Sigma0 / 0.5, synthetic=True)
        if step_point is not None:
            acc = rs.galaxy_accelerations(gal)
            rho = 0.5 * gal.SB_disk0 * np.exp(-R / R_d) / (2 * 0.15 * R_d * 1e3)
            g_obs = g_obs + mb.C_KMS ** 2 * rs.deps_c_drho(rho, step_point["eps_max"], step_point["rho_star"]) * rho / R_d
            assert acc["R"].size == R.size
        g_obs = g_obs * 10 ** (noise * rng.normal(size=R.size))
        gal.v_obs = np.sqrt(g_obs * R)
        out.append(gal)
    return out


def test_frozen_prediction_is_checked(tmp_path):
    pts = rs.frozen_points()
    assert [p["rho_star"] for p in pts][:2] == [0.001, 0.01] and len(pts) == 4
    bad = tmp_path / "p.json"
    bad.write_text('{"points": []}', encoding="utf-8")
    with pytest.raises(RuntimeError):
        rs.frozen_points(bad)


def test_one_parameter_rar_and_its_fit():
    g = np.geomspace(1e-2, 1e2, 50) * G_DAGGER / KMS2_PER_KPC_SI
    gr = rs.g_ref(g, G_DAGGER / KMS2_PER_KPC_SI)
    assert gr[-1] / g[-1] == pytest.approx(1.0, abs=1e-4) and gr[0] ** 2 == pytest.approx(g[0] * G_DAGGER / KMS2_PER_KPC_SI, rel=0.11)
    assert rs.fit_g_dagger(g, gr) == pytest.approx(G_DAGGER, rel=1e-6)


def test_r_star_and_the_window_statistic():
    gal = _galaxies(1)[0]
    Rs = rs.r_star(gal, 0.001, 0.15)
    rho = 0.5 * gal.SB_disk0 * math.exp(-Rs / gal.R_d) / (2 * 0.15 * gal.R_d * 1e3)
    assert rho == pytest.approx(0.001, rel=1e-12)
    assert rs.r_star(gal, 1e6, 0.15) is None
    x = np.array([0.55, 0.6, 0.9, 1.0, 1.2, 1.7, 2.0])
    d = np.array([0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0])
    assert rs.window_statistic(x, d) == pytest.approx(0.3)
    assert rs.window_statistic(x[:3], d[:3]) is None


def test_letters_without_and_with_the_step():
    pts = rs.frozen_points()
    p1 = pts[0]                                   # ρ* = 0.001: el punto comprobable
    flat = _galaxies(40, None, noise=0.01, seed=1)
    gd = rs.fit_g_dagger(np.concatenate([rs.galaxy_accelerations(g)["g_bar"] for g in flat]),
                         np.concatenate([rs.galaxy_accelerations(g)["g_obs"] for g in flat]))
    assert gd == pytest.approx(G_DAGGER, rel=0.02)
    no_step = rs.step_letters(flat, gd, True, [p1])[0]
    assert no_step["letter"] == "C"
    stepped = _galaxies(40, p1, noise=0.01, seed=1)
    gd2 = rs.fit_g_dagger(np.concatenate([rs.galaxy_accelerations(g)["g_bar"] for g in stepped]),
                          np.concatenate([rs.galaxy_accelerations(g)["g_obs"] for g in stepped]))
    # Hallazgo del 27-oct: con la ley aditiva exacta y ρ* = 0.001, la forma del documento deja S por encima de S_pred
    # (B) y la autoconsistente da A.
    doc = rs.step_letters(stepped, gd2, True, [p1])[0]["per_zeta"]["0.15"]
    auto = rs.step_letters(stepped, gd2, True, [p1], form="autoconsistente")[0]["per_zeta"]["0.15"]
    assert doc["letter"] == "B" and doc["S"] - doc["S_pred"] > 3.0 * doc["sigma_S"]
    assert auto["letter"] == "A"
    assert rs.step_letters(stepped, gd2, False, [p1])[0]["letter"] == "INDETERMINADO"
    few = rs.step_letters(stepped[:10], gd2, True, [p1])[0]
    assert few["letter"] == "INDETERMINADO" and "20 galaxias" in few["per_zeta"]["0.15"]["reason"]


def test_saturated_points_are_not_testable_with_realistic_noise():
    """Con ρ* = 0.01 el escalón previsto es de milésimas de dex: con 0.05 dex de ruido por punto, no comprobable."""
    gals = _galaxies(40, None, noise=0.05, seed=2)
    out = rs.step_letters(gals, G_DAGGER, True, [rs.frozen_points()[1]])[0]
    assert out["per_zeta"]["0.15"]["letter"] == "INDETERMINADO" and "no comprobable" in out["per_zeta"]["0.15"]["reason"]


def test_literal_reading_runs_without_letter():
    lit = rs.literal_reading(_galaxies(10), rs.frozen_points()[0])
    assert lit["n"] > 0 and lit["predicted_mean_dex"] >= 0.0
