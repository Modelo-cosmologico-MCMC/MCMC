"""Intercambio de masa entre la burbuja de MCV y la ECV en el borde (documento vivo del 24-oct, PR-64): las tasas por halo
del documento con el NFW prolongado (evaporación 0.051·H0; neto +0.21·H0 con ξ = 1 y +0.08·H0 con ξ = 0.5 para c = 10),
el flujo nulo en el equilibrio, la segunda ley en las dos formas del contacto y, en el código gravotérmico, el balance
exacto de masa y energía del paso de intercambio, el reajuste de las capas del borde y el cierre de la energía."""

import math
from dataclasses import replace

import numpy as np
import pytest

from dynamics.ecv_exchange import (
    border_rates,
    contact_flux,
    entropy_production,
    environment_outskirts,
    nfw_outskirts,
)
from dynamics.gravothermal import (
    GravothermalConfig,
    GravothermalHalo,
    code_outskirts,
    ecv_config,
    run_collapse,
)

FAST = GravothermalConfig(sigma_hat=0.249427, n_shells=100, eps=4e-3)      # el σ̂ del halo del autor


@pytest.mark.parametrize("c, evap, xi_eq, net1, net05", [(5, 0.062, 0.237, 0.198, 0.068), (10, 0.051, 0.199, 0.206, 0.078),
                                                       (15, 0.046, 0.181, 0.210, 0.082), (20, 0.043, 0.169, 0.212, 0.084)])
def test_per_halo_rates_of_the_document(c, evap, xi_eq, net1, net05):
    o = nfw_outskirts(c)
    assert border_rates(o)["evaporation"] == pytest.approx(evap, abs=5e-4)
    r1, r05 = border_rates(o, xi=1.0), border_rates(o, xi=0.5)
    assert r1["xi_eq"] == pytest.approx(xi_eq, abs=5e-4)
    assert r1["net"] == pytest.approx(net1, abs=5e-4) and r05["net"] == pytest.approx(net05, abs=5e-4)
    assert border_rates(o, xi=r1["xi_eq"])["net"] == pytest.approx(0.0, abs=1e-15)          # flujo nulo en el equilibrio
    a = border_rates(o, xi=0.3, xi_s=0.3, evaporation=False, accretion=0.67)                # brazo A con acreción
    assert a["sweep"] == pytest.approx(0.15 / 0.85 * (0.67 + a["contact"]), rel=1e-12)


def test_environment_variant_and_second_law():
    env = border_rates(environment_outskirts(10.0))
    assert env["evaporation"] == pytest.approx(0.159, abs=5e-4) and env["xi_eq"] == pytest.approx(0.460, abs=5e-4)
    for form in ("hertz-knudsen", "onsager-lineal"):
        for rho_b in (0.05, 0.2, 1.0, 3.0):
            for xi in (0.1, 0.5, 1.0, 2.0):
                assert entropy_production(rho_b, 0.3, xi, 1.0, form) >= 0.0
    hk, lin = contact_flux(0.2, 0.3, 0.2 * 1.001, 1.0), contact_flux(0.2, 0.3, 0.2 * 1.001, 1.0, "onsager-lineal")
    assert lin == pytest.approx(hk, rel=1e-3)                                              # su límite lineal


def test_null_flux_at_the_code_equilibrium_and_guards():
    base = ecv_config(FAST.sigma_hat, 10.0, base=FAST)
    xi_eq = GravothermalHalo(base).rho[-1] / base.q_E
    r = GravothermalHalo(replace(base, xi=xi_eq, evaporation=True)).exchange_rates()
    assert abs(r["total"]) < 1e-14 * r["evaporation"] and r["energy_in"] == pytest.approx(r["energy_out"], rel=1e-12)
    with pytest.raises(ValueError, match="pared «R0»"):
        GravothermalHalo(replace(FAST, xi=1.0))
    with pytest.raises(ValueError, match="Onsager"):
        GravothermalHalo(replace(base, xi=1.0, contact_form="onsager-lineal"))


def test_exchange_step_and_border_regrid_conserve_mass_and_energy():
    h = GravothermalHalo(replace(ecv_config(FAST.sigma_hat, 10.0, base=FAST), xi=1.0, xi_s=0.5, evaporation=True))
    rates, dt = h.exchange_rates(), 3.0
    m0, U0, u_new = h.M[-1], float(np.sum(h.u * h.dM)), h.u.copy()
    h._apply_exchange(rates, dt, u_new)
    assert h.M[-1] - m0 == pytest.approx(rates["total"] * dt, rel=1e-12)
    assert float(np.sum(u_new * h.dM)) - U0 == pytest.approx((rates["energy_in"] - rates["energy_out"]) * dt, rel=1e-9)
    assert rates["sweep"] == pytest.approx(0.25 / 0.75 * rates["contact"], rel=1e-12)       # barrido autoalimentado
    h.u = u_new
    for factor in (3.0, 0.1):                                                              # partir y fundir
        h.dM[-1] *= factor
        h.M = np.cumsum(h.dM)
        h.K = (h.u / 1.5) * h.rho ** (-2.0 / 3.0)
        mass, U, n = float(h.M[-1]), float(np.sum(h.u * h.dM)), len(h.dM)
        h._regrid_border()
        assert len(h.dM) == n + (1 if factor > 1 else -1) and float(h.M[-1]) == pytest.approx(mass, rel=1e-14)
        assert float(np.sum(h.u * h.dM)) == pytest.approx(U, rel=1e-12)


def test_arms_and_energy_budget_in_a_full_run():
    base = ecv_config(FAST.sigma_hat, 10.0, base=FAST)
    drift = run_collapse(base)["energy_rel_change"]                                         # la del código sin intercambio
    gain = run_collapse(replace(base, xi=0.5, xi_s=0.5))                                    # brazo A: solo atrapa
    loss = run_collapse(replace(base, evaporation=True))                                    # E2: solo escapa
    for r in (gain, loss):
        ex = r["exchange"]
        assert r["collapsed"] and abs(ex["energy_budget_residual"] - drift) < 2e-3 and abs(r["wall_net_force_end"]) < 1e-9
        assert ex["wall_ratio_end"] == pytest.approx(ex["mass_ratio_end"] ** (1.0 / 3.0), rel=1e-9)   # la pared sigue a R₀
    assert gain["exchange"]["mass_ratio_end"] > 1.2 and gain["exchange"]["evaporated"] == 0.0
    assert loss["exchange"]["mass_ratio_end"] < 0.9 and loss["exchange"]["condensed"] == 0.0


def test_code_outskirts_with_the_ecv_force():
    o, code = nfw_outskirts(10.0), code_outskirts(10.0, base=GravothermalConfig(sigma_hat=0.01, n_shells=150))
    assert code["x_R0"] == o["x_R0"] and 1.5 < code["rho_b"] / o["rho_b"] < 1.9 and 1.1 < code["nu_b"] / o["nu_b"] < 1.25
    assert 0.3 < border_rates(code)["xi_eq"] < 0.36 and math.isfinite(border_rates(code, xi=1.0)["net"])
