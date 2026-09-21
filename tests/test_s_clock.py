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

from core.s_clock import (
    DECLARED_FORMS,
    ClockConfig,
    SClock,
    delta0_metastability_max,
    emergent_diagnostic,
    radial_landscape,
    run_clock,
    sextic_coupling_dimension,
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
    assert r["stop_reason"].startswith(("S ≥ S_max", "descenso completado"))


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
