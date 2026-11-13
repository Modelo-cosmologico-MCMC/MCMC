"""Cascada de habilitación (R6): el instrumento frente al juguete de existencia de la revisión del 11-nov.

El juguete radial (sin inclinación, pesos 0.9 / 9 / 90 / 0.1 %) NO cuenta como letra: sus cifras ya eran públicas. Estos
tests fijan que el módulo lo reproduce, que la Monotonía queda activa, que sin semilla no hay cascada y que el modo no
toca el reloj por defecto. Ninguna celda del cribado preinscrito (reloj completo, con inclinación y semillas) se corre aquí.

Dos cifras del oráculo se leen con su resolución: sus eventos están tomados en la primera muestra posterior al cruce, y en
el colapso rápido de V2D el reloj avanza mucho por muestra. Los valores convergidos (paso limitado en S) son 0.99006 para
el tercer gatillo (oráculo: 0.9916) y 0.7737 para la media habilitación de V2D (oráculo: 0.7924).
"""

import dataclasses

import numpy as np
import pytest

from core.enabling_cascade import (
    CascadeClock,
    CascadeConfig,
    cascade_letter,
    cumulative_thresholds,
    gate_mass,
    ratios,
    run_cascade,
    scale_law_shares,
)
from core.s_clock import ClockConfig, cascade_clock, run_clock
from mcmc_ontology import constants as C

RADIAL = {"e_bar": 0.0}          # el juguete del oráculo: Basal radial, sin inclinación
RULES = {"A_interval": [7.0, 14.0], "n_collapses_required": 3}


def _events(r):
    return {e["kind"]: e for e in r["events"]}


@pytest.fixture(scope="module")
def toy():
    return run_cascade(0.01, clock_kwargs=RADIAL)


def test_scale_law_is_the_decade_with_one_number():
    sh = scale_law_shares(C.LAMBDA_DECADE)
    assert sh == pytest.approx((0.009, 0.09, 0.9, 0.001), abs=1e-15) and sum(sh) == pytest.approx(1.0, abs=1e-15)
    assert cumulative_thresholds(sh) == pytest.approx(C.decade_thresholds()[:3], abs=1e-12)
    for lam in (2.0, 5.0, 30.0):
        s = scale_law_shares(lam)
        assert sum(s) == pytest.approx(1.0, abs=1e-14)
        assert s[1] / s[0] == pytest.approx(lam) and s[2] / s[1] == pytest.approx(lam)      # progresión geométrica
        assert cumulative_thresholds(s) == pytest.approx([lam ** (k - 3) - lam ** -3 for k in (1, 2, 3)], abs=1e-14)
    with pytest.raises(ValueError):
        scale_law_shares(1.0)


def test_gate_closes_the_direction_until_the_previous_stage_has_advanced():
    m, d, dd = gate_mass(0.9, 0.9, 0.05)
    assert m == 0.0 and d == pytest.approx(-20.0) and dd == 0.0
    assert gate_mass(0.0, 0.9, 0.05)[0] == pytest.approx(1.0, abs=1e-12)       # cerrada: masa positiva
    assert gate_mass(1.5, 0.9, 0.05)[0] == pytest.approx(-1.0, abs=1e-9)       # abierta
    h = 1e-6
    for Y in (0.7, 0.88, 0.93, 1.05):
        m0, d0, dd0 = gate_mass(Y, 0.9, 0.05)
        assert d0 == pytest.approx((gate_mass(Y + h, 0.9, 0.05)[0] - gate_mass(Y - h, 0.9, 0.05)[0]) / (2 * h), rel=1e-6, abs=1e-8)
        assert dd0 == pytest.approx((gate_mass(Y + h, 0.9, 0.05)[1] - gate_mass(Y - h, 0.9, 0.05)[1]) / (2 * h), rel=1e-5, abs=1e-6)


def test_gradient_and_hessian_match_finite_differences_with_the_tilt():
    cc = CascadeClock(CascadeConfig(clock=ClockConfig(delta0=0.0554)))
    rng = np.random.default_rng(20261113)
    h = 1e-6
    for _ in range(40):
        y = [rng.uniform(0.2, 1.8), rng.uniform(0.0, 0.8)] + list(rng.uniform(0.0, 1.2, 3))
        g, H = np.array(cc.grad_hat(y)), cc.hessian_hat(y)
        for i in range(5):
            yp, ym = list(y), list(y)
            yp[i] += h
            ym[i] -= h
            assert g[i] == pytest.approx((cc.V_hat(yp) - cc.V_hat(ym)) / (2 * h), rel=1e-6, abs=1e-6)
            assert H[:, i] == pytest.approx((np.array(cc.grad_hat(yp)) - np.array(cc.grad_hat(ym))) / (2 * h), rel=1e-6, abs=1e-5)
        assert np.allclose(H, H.T)


def test_toy_reproduces_the_oracle_thresholds_and_tension(toy):
    assert toy["T0_basal_hat"] == pytest.approx(0.840847, abs=1e-6)             # c̄: el Basal radial
    assert toy["T_total_nominal_hat"] == pytest.approx(93.43, abs=5e-3)          # c̄/0.009
    assert toy["T_total_hat"] == pytest.approx(93.20, abs=5e-3)                  # 0.25 % menos, por la retroacción
    S = toy["S_triggers"]
    assert toy["n_collapses"] == 3 and S[0] == pytest.approx(0.0087, abs=1e-4) and S[1] == pytest.approx(0.0979, abs=1e-4)
    assert S[2] == pytest.approx(0.9916, abs=2e-3)                               # el oráculo lo lee una muestra tarde
    assert S == pytest.approx([0.008643, 0.097822, 0.990063], abs=2e-6)          # valores convergidos
    ev = _events(toy)
    assert ev["gatillo_1"]["sigma_hat"] == pytest.approx(0.525, abs=3e-3) and ev["gatillo_2"]["sigma_hat"] == pytest.approx(1.538, abs=3e-3)
    assert ev["gatillo_1"]["Y_prev_at_trigger"] == pytest.approx(0.9, abs=2e-3)  # M_1² = 0 ⟺ Y_0 = θ_g
    assert toy["ratios"] == pytest.approx([11.318, 10.121], abs=2e-3)
    half = toy["S_half_enabling"]
    assert half[0] == pytest.approx(0.0766, abs=5e-4) and half[2] == pytest.approx(0.9998, abs=1e-4)
    assert half[1] == pytest.approx(0.77365, abs=2e-4) and 0.76 < half[1] < 0.80  # oráculo: 0.7924, una muestra tarde
    assert cascade_letter(S, RULES)["letter"] == "A"                             # ya era público: no cuenta como letra


def test_toy_keeps_the_monotonia_and_ends_in_a_stable_state(toy):
    ck = toy["checks"]
    assert ck["monotonia_4_5"]["pass"] and ck["monotonia_4_5"]["max_dV_over_T_total"] <= 1e-12
    assert ck["produccion_entropica_4_5"]["pass"] and ck["S_equals_f_identity"]["pass"]
    assert ck["S_equals_f_identity"]["W_floor_over_T_total"] == 0.0              # sin fuente: un solo potencial
    assert ck["secuencia_jerarquica"]["pass"] and ck["secuencia_jerarquica"]["order"] == ["colapso_1D", "colapso_2D", "colapso_3D"]
    assert ck["gatillo_es_autovalor_minimo"]["pass"]
    fin = ck["estado_final"]
    assert toy["finished"] and fin["stable"] and fin["reduced_hessian_min_eigenvalue"] > 0.0
    assert abs(fin["V_final_minus_V_min_over_T_total"]) < 1e-12 and toy["f_final"] == pytest.approx(1.0, abs=1e-12)
    assert [c["algebra"] for c in toy["collapses"]] == ["C(2,0)", "C(3,0)", "C(4,0)"] and all(c["anticommute"] for c in toy["collapses"])
    assert toy["underflow_modes"] == []


def test_seals_appear_as_settlings_and_are_not_exact(toy):
    S, Y1 = np.array(toy["trajectory"]["S"]), np.array(toy["trajectory"]["Y1"])
    sel = toy["checks"]["sellos_como_asentamientos"]
    assert float(np.interp(0.05, S, Y1)) == pytest.approx(0.27, abs=0.01)
    assert sel["Y1_at_S_0.099"] == pytest.approx(0.999, abs=1e-3) and sel["Y2_at_S_0.999"] == pytest.approx(0.992, abs=2e-3)
    assert sel["Y_final"][0] == pytest.approx(1.12, abs=5e-3)                    # q_1² deriva un 12 % por la retroacción
    assert sel["Y0_final"] > 1.05                                                # y el Basal queda sobredescargado


def test_seed_delays_the_enabling_logarithmically_but_not_the_thresholds(toy):
    runs = {1e-6: toy}
    for s in (1e-3, 1e-9):
        runs[s] = run_cascade(0.01, clock_kwargs=RADIAL, seeds=(s, s, s), max_sigma_hat=2.0)
    t = {s: _events(r)["habilitacion_1"]["sigma_hat"] for s, r in runs.items()}
    assert t[1e-3] == pytest.approx(1.29, abs=5e-3) and t[1e-6] == pytest.approx(1.51, abs=6e-3) and t[1e-9] == pytest.approx(1.72, abs=5e-3)
    assert (t[1e-9] - t[1e-6]) == pytest.approx(t[1e-6] - t[1e-3], rel=0.02)     # retraso ∝ ln(semilla)
    for s in (1e-3, 1e-9):
        ev = _events(runs[s])
        assert ev["gatillo_1"]["S"] == pytest.approx(toy["S_triggers"][0], abs=2e-5)
        assert ev["gatillo_2"]["S"] == pytest.approx(toy["S_triggers"][1], abs=2e-5)


def test_without_seed_there_is_a_trigger_but_no_cascade():
    r = run_cascade(0.01, clock_kwargs=RADIAL, seeds=(0.0, 0.0, 0.0))
    assert [e["kind"] for e in r["events"]] == ["gatillo_1"]                     # la inestabilidad existe…
    assert r["n_collapses"] == 0 and r["S_triggers"] == []                       # …pero la perfección es estéril
    assert r["finished"] and r["f_final"] == pytest.approx(r["T0_basal_hat"] / r["T_total_hat"], rel=1e-9)
    assert r["f_final"] == pytest.approx(0.0090, abs=5e-5)                       # la parte del Basal
    assert cascade_letter(r["S_triggers"], RULES)["letter"] == "C"


def test_unseeded_cascade_is_the_default_clock_step_by_step():
    d0 = 0.0554
    r = run_cascade(d0, seeds=(0.0, 0.0, 0.0))                                  # con la inclinación del reloj
    rc = run_clock(d0)
    n = len(rc["trajectory"]["sigma"]) - 1                                       # la última muestra del reloj es la de parada
    sig_c, f_c = np.array(rc["trajectory"]["sigma"][:n]) * d0 ** 2, np.array(rc["trajectory"]["f"][:n])
    sig_k = np.array(r["trajectory"]["sigma_hat"][:n])
    f_k = np.array(r["trajectory"]["f"][:n]) * r["T_total_hat"] / r["T0_basal_hat"]
    assert n > 300 and np.max(np.abs(sig_c - sig_k)) < 1e-9 and np.max(np.abs(f_c - f_k)) < 1e-9
    assert r["T0_basal_hat"] * d0 ** 3 == pytest.approx(rc["T0"], rel=1e-12)     # la Tensión Primordial medida del reloj


def test_noise_floor_reproduces_the_oracle_third_trigger_without_moving_the_thresholds(toy):
    r = run_cascade(0.01, clock_kwargs=RADIAL, noise_floor=1e-18, max_sigma_hat=2.0)
    ev, ev0 = _events(r), _events(toy)
    assert ev["gatillo_3"]["sigma_hat"] == pytest.approx(1.675, abs=3e-3)        # el σ̂ del oráculo: su integrador tenía suelo
    assert ev0["gatillo_3"]["sigma_hat"] == pytest.approx(3.184, abs=5e-3)       # dinámica exacta: q_2 decae con la compuerta cerrada
    for k in ("gatillo_1", "gatillo_2", "gatillo_3"):
        assert ev[k]["S"] == pytest.approx(ev0[k]["S"], abs=2e-5)                # en S, el mismo umbral
    assert 0.0 <= r["checks"]["S_equals_f_identity"]["W_floor_over_T_total"] < 1e-12 and r["checks"]["monotonia_4_5"]["pass"]


def test_negative_controls_no_gate_and_broken_hierarchy():
    no_gate = run_cascade(0.01, clock_kwargs=RADIAL, gated=False)
    assert no_gate["events"] == [] and no_gate["n_collapses"] == 0 and no_gate["finished"]
    assert no_gate["f_final"] == pytest.approx(no_gate["T0_basal_hat"] / no_gate["T_total_hat"], rel=1e-6)
    # sin sembrar q_1, las direcciones superiores no se abren aunque estén sembradas
    broken = run_cascade(0.01, clock_kwargs=RADIAL, seeds=(0.0, 1e-6, 1e-6))
    assert [e["kind"] for e in broken["events"]] == ["gatillo_1"] and broken["n_collapses"] == 0
    assert broken["f_final"] == pytest.approx(0.0090, abs=5e-5)


def test_equal_weights_give_three_collapses_and_fail_the_letter_rule():
    r = run_cascade(0.01, clock_kwargs=RADIAL, shares=(1.0, 1.0, 1.0, 1.0))
    assert r["finished"] and r["n_collapses"] == 3 and r["nominal_thresholds"] == pytest.approx([0.25, 0.5, 0.75])
    assert r["S_triggers"] == pytest.approx([0.242, 0.498, 0.747], abs=2e-3)       # cerca de los acumulados del reparto
    L = cascade_letter(r["S_triggers"], RULES)
    assert L["letter"] == "C" and L["ratios"] == pytest.approx([2.06, 1.50], abs=0.01)   # la regla no da A por construcción


def test_letter_rule():
    assert cascade_letter([0.009, 0.099, 0.999], RULES)["letter"] == "A"
    assert cascade_letter([0.01, 0.1, 0.5], RULES)["letter"] == "B"
    assert cascade_letter([0.25, 0.5, 0.75], RULES)["letter"] == "C"
    assert cascade_letter([0.009, 0.099], RULES)["letter"] == "C" and cascade_letter([], RULES)["letter"] == "C"
    assert ratios([0.009, 0.099, 0.999]) == pytest.approx([11.0, 0.999 / 0.099])


def test_cascade_mode_demands_fixed_couplings_and_pure_gradient_flow():
    with pytest.raises(ValueError):
        CascadeClock(CascadeConfig(clock=ClockConfig(thresholds="emergent", couplings_flow=True)))
    with pytest.raises(ValueError):
        CascadeClock(CascadeConfig(clock=ClockConfig(J_circ=0.5)))
    with pytest.raises(ValueError):
        CascadeClock(CascadeConfig(clock=ClockConfig(kappa_conv=1.0)))
    with pytest.raises(ValueError):
        CascadeClock(CascadeConfig(seeds=(1e-6, 1e-6)))
    with pytest.raises(ValueError):
        CascadeClock(CascadeConfig(lam_scale=1.0))


def test_optional_mode_leaves_the_default_clock_untouched():
    r = cascade_clock(0.01, clock_kwargs=RADIAL, max_sigma_hat=0.1)
    assert r["mode"] == "cascade" and "comprobación interna (E8)" in r["status"]
    fields = {f.name for f in dataclasses.fields(ClockConfig)}
    assert not any("cascade" in f or "latent" in f for f in fields)              # el modo no añade entradas al reloj
    assert ClockConfig().thresholds == "imposed" and ClockConfig().theta_nuc == 0.0 and ClockConfig().angular_A is None
