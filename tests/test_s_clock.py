"""El reloj S como simulador de consistencia (core/s_clock.py).

Lo que se comprueba: identidades del flujo (S ≡ f, monotonía,
exclusión), la secuencia de eventos impuestos con su álgebra, la
Rotación de Florencia con su control negativo, los sellados por
congelación, el hallazgo de metastabilidad con la inclinación, el
control negativo del modo emergente (acoplos congelados ⟹ D constante,
ningún cruce) y que el estado entregado se declara ilegible para la
cosmología. Nada aquí es demostración física (E8).
"""

import numpy as np
import pytest

from core.basal import T0_analytic
from core.s_clock import (
    DECLARED_FORMS,
    ClockConfig,
    S_flip_first_order,
    SClock,
    T0_tilt_first_order,
    delta0_metastability_max,
    delta0_metastability_max_analytic,
    emergent_diagnostic,
    kappa1_tilt,
    naturalness_sweep,
    radial_landscape,
    run_clock,
    sextic_coupling_dimension,
    tau_decade_table,
)
from mcmc_ontology import constants as C


@pytest.fixture(scope="module")
def run():
    return run_clock(0.01)


def test_metastability_with_tilt_has_a_maximum_delta0():
    d0max = delta0_metastability_max()
    assert 0.05 < d0max < 0.2
    assert radial_landscape(0.5 * d0max)["metastable"]
    assert not radial_landscape(1.5 * d0max)["metastable"]
    with pytest.raises(ValueError, match="borra la barrera"):
        SClock(ClockConfig(delta0=1.5 * d0max))


def test_escape_point_between_barrier_and_vacuum(run):
    ld = run["landscape"]
    assert ld["rho_barrier"] < ld["rho_esc"] < ld["rho_tv"]
    assert abs(run["checks"]["S0"]["nucleation"]["V_esc_minus_V_fv_over_T0"]) < 1e-9


def test_S_equals_f_identity_and_flow_theorems(run):
    d = run["checks"]["descent"]
    assert d["S_equals_f_identity"]["pass"], d["S_equals_f_identity"]
    assert d["monotonia_4_5"]["pass"] and d["produccion_entropica_4_5"]["pass"]
    assert d["exclusion_4_7"]["pass"] and d["exit_to_mass_pole_3_5"]["pass"]
    assert d["descent_finished"]["pass"]
    assert run["checks"]["S0"]["T0_scaling_3_4"]["pass"]
    assert 0.0 < run["checks"]["S0"]["T0_scaling_3_4"]["tilt_correction"] < 0.5


def test_imposed_events_in_order_with_algebra_chain(run):
    kinds = [e["kind"] for e in run["events"]]
    assert kinds == ["colapso_1D", "colapso_2D", "colapso_3D", "V3D", "Florencia"]
    col = run["events"][:3]
    assert [e["S"] for e in col] == pytest.approx(C.decade_thresholds()[:3])
    assert [e["algebra"] for e in col] == ["C(2,0)", "C(3,0)", "C(4,0)"]
    assert all(e["anticommute"] and all(s == 1 for s in e["signature"]) for e in col)
    assert all(e["elliptic_symbol_positive"] for e in col)
    sig = [e["sigma"] for e in col]
    assert sig[0] > 0.0 and sig == sorted(sig)
    # S integrado coincide con el reloj f en cada evento (Teo. 4.5)
    assert all(abs(e["S_integrated"] - e["S"]) < 2e-3 for e in col)
    assert run["events"][3]["marginal_in_d3"] and sextic_coupling_dimension(3) == 0.0


def test_florencia_rotation_signature_and_controls(run):
    fl = run["events"][-1]
    assert fl["signature_after"] == [-1, 1, 1, 1] and fl["anticommute_after"]
    assert fl["negative_control_pass"] and fl["negative_control_two_rotations"].count(-1) == 2
    assert fl["regime_after"] == "post"
    assert fl["rp_slice_pass"] and fl["rp_negative_control_pass"]
    assert fl["m_H_identity_pass"]


def test_seals_freeze_on_the_clock(run):
    s = run["checks"]["seals"]
    assert s["c_eff"]["sealed"] and s["c_eff"]["rate_after_seal"] == 0.0
    assert s["c_eff"]["S_of_max"] < C.S_SEALS["C2"]
    assert s["m_eff"]["sealed"] and s["m_eff"]["rate_after_seal"] == 0.0
    assert "FORMA DECLARADA" in DECLARED_FORMS["c_eff"]


def test_diagonal_not_crossed_by_flow_but_imposed(run):
    dg = run["checks"]["diagonal"]
    assert dg["flow"]["crossed"] is False and dg["flow"]["status"] == "publicado"
    assert dg["imposed"]["S_at_crossing"] == C.S_SEALS["V3D"]


def test_delivered_state_declares_unreadable(run):
    st = run["delivered_state"]
    assert st["S"] == C.S_SEALS["C4"] and st["algebra"] == "C(3,1)" and st["regime"] == "post"
    assert st["readable_by_cosmology"] is False and st["channels_initial"] is None
    assert st["delta0_next_cycle"] is None
    assert abs(st["T_residual_over_T0"] - C.DELTA_S) < 5e-4
    hs = run["checks"]["handshake_cosmology"]
    assert hs["newton_seal_9_3"]["pass"] and hs["lcdm_recovery_A_1"]["pass"]
    assert hs["dictionary"]["exists"] is False


def test_emergent_negative_control_frozen_couplings():
    """Con τ = 0 los acoplos no se mueven: D constante, ningún cruce."""
    r = emergent_diagnostic(0.01, tau=0.0)
    assert r["S_crossings"] == [] and r["D_initial"] == pytest.approx(r["D_final"])
    assert r["stop_reason"].startswith(("S ≥ S_end", "descenso completado"))


def test_emergent_diagnostic_reports_without_verdict():
    r = emergent_diagnostic(0.01, tau=1e-2)
    assert "sin estatuto" in r["status"]
    assert isinstance(r["S_crossings"], list) and isinstance(r["D_sinks"], bool)


def test_scale_invariance_in_sigma_hat():
    """σ̂ = σ·δ0²: dos δ0 dan la misma σ̂ de descenso salvo la corrección
    de la inclinación (O(δ0^{1/2}))."""
    a = run_clock(0.005)["checks"]["descent"]["descent_finished"]["sigma_hat_final"]
    b = run_clock(0.02)["checks"]["descent"]["descent_finished"]["sigma_hat_final"]
    assert abs(a - b) / a < 0.35


def test_trajectory_is_jsonable(run):
    import json
    json.dumps(run)
    assert np.isfinite(run["T0"])


# ------------------------------------------------------------------ v1 (21-sep)

def test_origin_root_without_tilt_gives_infinite_delta0_max():
    """Sin inclinación el origen es el falso vacío exacto (raíz que la
    malla v0 no veía): δ₀_max = ∞ y el reloj corre con corrección 0."""
    ld = radial_landscape(0.01, e_bar=0.0)
    assert ld["roots"][0] == 0.0 and ld["metastable"] and ld["rho_esc"] is not None
    assert ld["V_fv"] == 0.0
    assert delta0_metastability_max(e_bar=0.0) == float("inf")
    assert delta0_metastability_max_analytic(e_bar=0.0) == float("inf")
    assert radial_landscape(0.01, theta=np.pi / 4)["roots"][0] == 0.0     # sobre la diagonal, tilt = 0
    r = run_clock(0.01, e_bar=0.0)
    assert r["checks"]["S0"]["T0_scaling_3_4"]["tilt_correction"] == 0.0
    assert [e["kind"] for e in r["events"]][:3] == ["colapso_1D", "colapso_2D", "colapso_3D"]


def test_kappa1_first_order_tilt_correction(run):
    """κ₁ = ē√κ₊/(√2c̄) = 1.361 con las formas por defecto; la corrección
    medida sigue κ₁√δ₀ con residuo O(δ₀)."""
    assert kappa1_tilt() == pytest.approx(1.3607, abs=1e-3)
    t = run["checks"]["S0"]["T0_tilt_first_order"]
    assert t["pass"] and 0.95 < t["ratio_measured_over_first_order"] < 1.0
    for d0 in (0.003, 0.03):
        c = radial_landscape(d0)["T0_full"] / T0_analytic(d0) - 1.0
        assert abs(c - kappa1_tilt() * np.sqrt(d0)) < 0.5 * kappa1_tilt() * d0
    assert T0_tilt_first_order(0.01) == pytest.approx(T0_analytic(0.01) * (1 + kappa1_tilt() * 0.1))


def test_delta0_max_closed_form_scales_as_e_bar_minus_two(run):
    assert run["checks"]["S0"]["delta0_max_closed_form"]["pass"]
    for e in (0.5, 1.0, 2.0):
        an = delta0_metastability_max_analytic(e_bar=e)
        assert an == pytest.approx(0.1028 / e ** 2, rel=2e-3)
        assert an == pytest.approx(delta0_metastability_max(e_bar=e), rel=1e-6)


def test_naturalness_sweep_publishes_fractions():
    ns = naturalness_sweep(n=300, n_check=6)
    fr = ns["fraction_metastable_at"]
    assert set(fr) == {"0.012", "0.0581"} and 0.0 <= fr["0.0581"] <= fr["0.012"] <= 1.0
    assert ns["closed_form_vs_bisection_max_rel_err"] < 1e-4
    assert ns["status"].startswith("publicado")


def test_mass_instability_event_published_and_integration_continues():
    """En modo emergente M0² = 0 ya no detiene el bucle: se publica
    `inestabilidad_masa` con S_flip frente a su forma cerrada y el
    recorrido sigue (se detiene solo con C0 ≤ 0 o al agotar el presupuesto)."""
    r = emergent_diagnostic(0.01, tau=0.1, max_steps=20000)
    mi = r["mass_instability"]
    assert mi is not None and 0.9 < mi["ratio"] < 1.05
    assert r["couplings_out_of_domain"] is None and "M0²" not in (r["stop_reason"] or "")
    assert r["S_final"] > mi["S_flip"]
    assert S_flip_first_order(0.01, 1.0) == pytest.approx(0.10566 * 0.01, rel=1e-3)
    assert S_flip_first_order(0.01, 0.0) is None and S_flip_first_order(0.01, 1.0, b_bar=2.0) is None
    tk = tau_decade_table(0.01)
    assert tk["tau_k"][0] == pytest.approx(11.74 * 0.01, rel=1e-2)
    assert tk["tau_k"][1] / tk["tau_k"][0] == pytest.approx(0.009 / 0.099)
    assert "E13" in tk["status"] and "E13" in mi["status"]


def test_tau_per_dim_hypothesis_is_emergent_only_and_resets_mass():
    with pytest.raises(ValueError, match="emergent"):
        SClock(ClockConfig(delta0=0.01, tau_per_dim=(1.0, 0.1, 0.01))).run()
    r = SClock(ClockConfig(delta0=0.01, thresholds="emergent", couplings_flow=True,
                           tau_per_dim=(0.1174, 0.01174, 0.001174), max_steps=30000)).run()
    flips = r["mass_instability_events"]
    assert flips and "reset" in flips[0] and flips[0]["reset"]["tau_next"] == 0.01174
    col = [e for e in r["events"] if e["kind"].startswith("colapso")]
    assert col and all("hipótesis τ_d" in e["trigger"] for e in col)
    assert all("E13" in e["trigger"] for e in col)


def test_circulation_C_V_is_orthogonal_and_keeps_identity_in_interior():
    """J∇V ⊥ ∇V: la Monotonía y S = f − f₀ + W_J/T₀ se conservan; en el
    interior W_J = 0. Con |J| justo sobre el umbral el flujo cruza la
    diagonal (θ_max > π/4) y vuelve hacia el polo de masa."""
    from core.s_clock import diagonal_crossing_S
    r = diagonal_crossing_S(0.01, 1.5)
    assert r["crossed"] and 0.5 < r["S_cross"] < 0.95
    assert r["S_equals_f_max_diff"] < 1e-6 and r["monotonia_pass"]
    assert abs(r["W_J_over_T0"]) < 1e-9
    assert r["theta_final"] < 0.1 < r["theta_max"]           # vuelve hacia θ = 0
    r0 = diagonal_crossing_S(0.01, 1.0)
    assert not r0["crossed"] and r0["theta_max"] < np.pi / 4
    assert "J_circulation" in DECLARED_FORMS and "Monotonía" in DECLARED_FORMS["J_circulation"]


def test_circulation_rigid_costs_monotonicity():
    from core.s_clock import diagonal_crossing_S
    r = diagonal_crossing_S(0.01, 1.0, "rho2", max_steps=60000)
    assert r["W_J_over_T0"] > 1e-3 and not r["monotonia_pass"]
    with pytest.raises(ValueError, match="circulation_C"):
        SClock(ClockConfig(delta0=0.01, J_circ=1.0, circulation_C="x")).run()


def test_path_flow_circulation_term_is_antisymmetric():
    from core.path_flow import EPS, circulation, flow, grad_V
    phi = np.array([0.1, 0.02])
    g = grad_V(phi, 0.01)
    assert abs(g @ circulation(phi, g, 1.7, "V")) < 1e-18          # J∇V ⊥ ∇V
    assert abs(phi @ circulation(phi, g, 1.7, "rho2")) < 1e-18     # rotación rígida ⊥ Φ
    assert np.allclose(EPS @ EPS, -np.eye(2))
    out = flow(np.array([0.12, 0.0]), 0.01, n_steps=200, J=1.0)
    assert out["J"] == 1.0 and abs(out["W_J"]) < 1e-12 and out["trajectory"][-1][1] > 0.0


def test_declared_forms_include_post_florencia_unit_gap():
    assert "S_post_unit" in DECLARED_FORMS and "hueco" in DECLARED_FORMS["S_post_unit"]
    assert "diccionario-unidad-S-post-florencia" in DECLARED_FORMS["S_post_unit"]
    assert "E13" in DECLARED_FORMS["mass_instability"]


# ---------------------------------------------------------------- v2 (22-sep): decisiones A, B, C
def test_decision_B_kramers_mode_publishes_rate_and_requires_D_ent():
    """Decisión B: n = 1 con Kramers (Axioma 4). El modo exige D_ent
    declarada, arranca en el punto de escape (misma convención que Gamow),
    publica Γ_K y la espera 1/Γ_K, y el bounce queda etiquetado como
    control negativo."""
    import pytest

    from core.s_clock import ClockConfig, SClock, radial_landscape
    with pytest.raises(ValueError):
        SClock(ClockConfig(delta0=0.01, nucleation="kramers"))
    ld = radial_landscape(0.01)
    r = SClock(ClockConfig(delta0=0.01, nucleation="kramers", D_ent=ld["barrier_height"])).run()
    nuc = r["checks"]["S0"]["nucleation"]
    assert nuc["mode"] == "kramers" and nuc["kramers"]["Gamma_K"] > 0.0
    assert nuc["Gamma0"] == nuc["kramers"]["Gamma_K"]
    assert abs(nuc["sigma_wait_before_nucleation"] * nuc["Gamma0"] - 1.0) < 1e-12
    assert abs(nuc["x_esc"] - ld["rho_esc"] ** 2) < 1e-12          # mismo punto de escape que la convención v0
    assert "Kramers" in nuc["status"] and "n_dim = 1" in nuc["status"]
    assert r["checks"]["descent"]["S_equals_f_identity"]["max_abs_diff"] < 1e-9
    ctrl = SClock(ClockConfig(delta0=0.01, nucleation="bounce")).run()["checks"]["S0"]["nucleation"]
    assert "CONTROL NEGATIVO" in ctrl["status"]


def test_decision_C_collapse_trigger_declared_as_D_zero():
    """Decisión C: «el colapso» es D = 0 (Def. 8.4 / Obs. 8.6); M0² = 0 es
    diagnóstico. Declarado en DECLARED_FORMS, no derivado."""
    from core.s_clock import DECLARED_FORMS
    f = DECLARED_FORMS["collapse_trigger"]
    assert "D(S) = B² − 4C0M0² = 0" in f and "Obs. 8.6" in f
    assert "diagnóstico" in f and "no-evento" in f
    assert "frente 2" in f            # el cierre canónico no lo produce: las β son del frente 2
    assert "DECISIÓN B" in DECLARED_FORMS["nucleation"] and "CONTROL NEGATIVO" in DECLARED_FORMS["nucleation"]
