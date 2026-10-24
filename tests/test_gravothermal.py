"""Código gravotérmico 1D de la MCV (c) con borde de ECV (bloque P1 del autor del 23-oct; borde del 24-oct): el estado
inicial en equilibrio, la conducción implícita y su balance de energía (también el intercambio con la ECV), las unidades
frente a la fórmula de Essig et al. (2019), t_c dentro de la banda declarada y universal en el régimen LMFP, el borde fuera
de la influencia térmica, gravitatoria y mecánica, la pendiente autosemejante y las guardas. Configuración barata (100
capas, eps = 4·10⁻³: unos 2 s por colapso); las corridas de 150 a 400 capas están en results/2026-10-24_gravothermal/."""

from dataclasses import replace

import numpy as np
import pytest

from dynamics.gravothermal import (
    BORDER_TOL,
    THERMAL_TOL,
    GravothermalConfig,
    GravothermalHalo,
    border_shift,
    collapse_time_gyr,
    nfw_mass,
    run_collapse,
    sigma_hat,
    time_unit_gyr,
)
from dynamics.mcv_equilibrium import gravothermal_collapse_time, nfw_halo

FAST = GravothermalConfig(sigma_hat=0.01, n_shells=100, eps=4e-3)


@pytest.fixture(scope="module")
def reference():
    return run_collapse(FAST)


def test_initial_state_is_the_truncated_nfw_in_equilibrium():
    h = GravothermalHalo(replace(FAST, n_shells=40))
    assert h.hydrostatic_residual() < 1e-10
    assert h.M[-1] == pytest.approx(float(nfw_mass(100.0)), rel=1e-12) and h.r[0] == 0.0 and h.r[-1] == 100.0
    assert np.all(np.diff(h.r) > 0.0) and np.all(h.u > 0.0) and h.u_ecv == h.u[-1]


@pytest.mark.parametrize("border", ["cerrado", "intercambio"])
def test_implicit_conduction_energy_budget(border):
    h = GravothermalHalo(replace(FAST, n_shells=40, border=border))
    h.u = h.u * (1.0 + 0.05 * np.sin(np.arange(len(h.u))))       # gradientes, también entre el borde y la ECV
    e0 = float(np.sum(h.u * h.dM))
    _, g_ecv = h._conductances()
    dt = 0.3
    u1 = h._conduct(dt)
    l_border = 0.0 if border == "cerrado" else -g_ecv * (h.u_ecv - u1[-1])       # L_N del paso implícito
    assert float(np.sum(u1 * h.dM)) - e0 == pytest.approx(-dt * l_border, abs=1e-13 * e0)
    assert (l_border != 0.0) == (border == "intercambio") and np.all(u1 > 0.0)


def test_units_reproduce_the_essig_formula():
    halo = nfw_halo(1e12, 10.0)
    sh = sigma_hat(10.0, halo["rho_s"], halo["r_s"])
    essig = gravothermal_collapse_time(10.0, halo["rho_s"], halo["r_s"], beta=0.75)
    assert (150.0 / 0.75) / sh * time_unit_gyr(halo["rho_s"]) == pytest.approx(essig, rel=1e-12)
    assert essig == pytest.approx(44.9, abs=0.05)                                  # cifra de referencia del autor


def test_lmfp_collapse_time_in_the_declared_band_and_universal(reference):
    r = reference
    assert r["collapsed"] and 120.0 <= r["C_sigma_t_c"] <= 180.0                 # 150/β ± 20 %, declarada antes de correr
    assert r["rho_c_min"] < 1e-2 * r["rho_c_initial"] and 0.05 < r["t_core_max"] / r["t_c"] < 0.3   # núcleo y colapso
    assert abs(r["energy_rel_change"]) < 1e-2
    for other in (replace(FAST, sigma_hat=0.001), replace(FAST, C=0.45)):
        assert run_collapse(other)["C_sigma_t_c"] == pytest.approx(r["C_sigma_t_c"], rel=1e-3)


def test_border_is_outside_the_influence(reference):
    exchange = run_collapse(replace(FAST, border="intercambio"))
    assert reference["outside_thermal_influence"] and exchange["outside_thermal_influence"]
    assert exchange["exchange_flux_rel"] < THERMAL_TOL
    assert exchange["t_c"] == pytest.approx(reference["t_c"], rel=1e-6)
    shift = border_shift(FAST)
    assert shift["outside_influence"] and abs(shift["rel"]) < BORDER_TOL


def test_self_similar_slope_far_from_the_core():
    r = run_collapse(GravothermalConfig(sigma_hat=0.001, n_shells=200, eps=4e-3, rho_stop=1e6), snapshot_rho=(1e6,))
    s = r["snapshots"][0]
    assert s["n_shells"] >= 20 and -2.3 < s["slope"] < -2.1                      # Balberg et al. (2002): −2.19


def test_guards():
    with pytest.raises(ValueError, match="borde"):
        GravothermalHalo(replace(FAST, border="abierto"))
    with pytest.raises(ValueError, match="dominio"):
        GravothermalHalo(replace(FAST, r_min=200.0))
    with pytest.raises(ValueError, match="dentro de r200"):
        collapse_time_gyr(1e9, 20.0, 1.0, replace(FAST, r_border=15.0))
