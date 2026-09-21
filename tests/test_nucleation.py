"""Nucleación (core/nucleation.py) y su consumo por el reloj S.

Se comprueba: el bounce converge y su centro está entre la barrera y el
vacío verdadero; B crece al bajar δ₀ para d = 4 (Γ₀(0) = 0 sin
prefactor) y el exponente medido cae cerca del argumento de escala; el
escape de Kramers tiene prefactor ∝ δ₀² y barrera ∝ δ₀³; el reloj S con
nucleación por bounce mantiene S ≡ f − f₀ y publica f₀. Nada es
demostración física (E8)."""

import numpy as np
import pytest

from core.nucleation import (
    bounce,
    delta0_for_B,
    kramers_escape,
    kramers_scaling,
    scaling_law,
)
from core.s_clock import ClockConfig, SClock, delta0_metastability_max


def test_bounce_converges_and_center_is_beyond_escape_point():
    for d in (3, 4):
        b = bounce(0.01, d=d)
        assert b.converged and b.n_shots < 80
        assert b.phi_barrier < b.phi_esc_energy < b.phi0 <= b.phi_tv * (1 + 1e-9)
        assert b.B > 0.0 and 0.0 < b.f0 <= 1.0
        assert b.thin_wall_applicable is False and b.barrier_over_T0 < 0.1


def test_bounce_action_grows_as_delta0_shrinks_for_d4():
    s = scaling_law([0.004, 0.008, 0.016, 0.032], d=4)
    assert s["Gamma0_vanishes_at_delta0_zero_without_prefactor"]
    assert -1.7 < s["exponent_measured"] < -0.8          # argumento: −1, corrección de la inclinación
    Bs = [r["B"] for r in s["rows"]]
    assert Bs == sorted(Bs, reverse=True)


def test_bounce_d3_action_is_order_one_and_weakly_dependent():
    s = scaling_law([0.004, 0.008, 0.016, 0.032], d=3)
    assert -0.6 < s["exponent_measured"] < 0.1           # argumento: 0
    assert all(1.0 < r["B"] < 30.0 for r in s["rows"])


def test_kramers_scaling_exponents():
    k = kramers_scaling([0.004, 0.008, 0.016, 0.032], D_ent=1e-8)
    assert abs(k["prefactor_exponent_measured"] - 2.0) < 0.15
    # la barrera escala como δ₀³ SIN inclinación; la inclinación (∝ δ₀^{7/2}) la
    # erosiona al crecer δ₀ y baja el exponente medido (corrección O(δ₀^{1/2}))
    assert 2.3 < k["barrier_exponent_measured"] < 3.0
    assert k["Gamma_K_vanishes_at_delta0_zero"]
    r = kramers_escape(0.01, D_ent=1e-8)
    assert r["Gamma_K"] > 0.0 and r["exponent"] == pytest.approx(r["barrier_height"] / 1e-8)


def test_delta0_for_B_inverts():
    d0 = delta0_for_B(10.0, d=3, lo=0.003, hi=0.09)
    assert bounce(d0, d=3).B == pytest.approx(10.0, rel=1e-3)


def test_clock_with_bounce_nucleation_keeps_identity_and_publishes_f0():
    out = SClock(ClockConfig(delta0=0.01, nucleation="bounce", bounce_d=3)).run()
    chk = out["checks"]["descent"]["S_equals_f_identity"]
    assert chk["pass"] and 0.5 < chk["f0_nucleation"] < 1.0
    nuc = out["checks"]["S0"]["nucleation"]
    assert nuc["mode"] == "bounce" and nuc["bounce"]["d"] == 3 and nuc["Gamma0"] > 0.0
    kinds = [e["kind"] for e in out["events"]]
    assert kinds[:3] == ["colapso_1D", "colapso_2D", "colapso_3D"]
    # los umbrales ya superados al emerger caen en σ = 0 con la nota
    early = [e for e in out["events"][:3] if e["S"] < chk["f0_nucleation"]]
    assert all(e["sigma"] == 0.0 and "dentro de la nucleación" in e["trigger"] for e in early)
    assert out["ledger"]["derived_checks_failed"] == []


def test_clock_bounce_d4_discharges_inside_nucleation():
    out = SClock(ClockConfig(delta0=0.01, nucleation="bounce", bounce_d=4)).run()
    f0 = out["checks"]["descent"]["S_equals_f_identity"]["f0_nucleation"]
    assert f0 > 0.99
    assert sum("dentro de la nucleación" in e["trigger"] for e in out["events"][:3]) >= 2


# ------------------------------------------------------------ ronda 2 (n_dim)

def test_gamow_tunnel_b1_scales_as_delta0_squared_and_prefactor_is_linear():
    """n = 1: B₁ = b₁δ₀²(1 + O(√δ₀)) con b₁ → b₁(sin inclinación); ω_fv = m̄δ₀."""
    from core.nucleation import b1_no_tilt, gamow_tunnel
    b1 = b1_no_tilt()
    assert 0.4 < b1 < 0.6
    g_lo, g_hi = gamow_tunnel(1e-4), gamow_tunnel(1e-3)
    assert g_lo.converged and g_hi.converged
    assert abs(g_lo.b1_over_delta0_sq - b1) < 0.10 * b1
    # el exponente local de B₁ entre 1e-4 y 1e-3 está cerca de 2
    p = np.log(g_hi.B1 / g_lo.B1) / np.log(10.0)
    assert 1.8 < p < 2.1
    assert abs(g_lo.omega_fv - 1e-4) < 1e-6 and abs(g_lo.Gamma0 - 1e-4 / (2 * np.pi)) < 1e-6
    # sin inclinación la ley es exacta
    g0 = gamow_tunnel(0.01, e_bar=0.0)
    assert abs(g0.b1_over_delta0_sq - b1) < 1e-4 * b1


def test_gamma0_dispatch_is_conditional_on_declared_n_dim():
    from core.nucleation import N_DIM_DECLARED, gamma0
    assert N_DIM_DECLARED == (1, 3, 4)
    r1, r4 = gamma0(0.01, 1), gamma0(0.01, 4)
    assert r1["Gamma0"] is not None and r1["prefactor"] is not None
    assert r4["Gamma0"] is None and r4["Gamma0_over_A"] is not None and r4["prefactor"] is None
    with pytest.raises(ValueError, match="declarado"):
        gamma0(0.01, 2)


def test_path_deformation_returns_ratio_and_ray_baseline():
    from core.nucleation import gamow_tunnel, path_deformation
    pd = path_deformation(0.01, n_s=401, maxiter=60)
    assert 0.0 < pd["B_min"] <= pd["B_ray"] and 0.0 < pd["ratio_min_over_ray"] <= 1.0
    # el rayo recto reproduce la acción del túnel 1D
    assert abs(pd["B_ray"] - gamow_tunnel(0.01).B1) < 2e-2 * gamow_tunnel(0.01).B1
    assert 0.0 <= pd["theta_end_opt"] <= np.pi / 2


def test_clock_gamow_mode_starts_at_escape_point_and_publishes_rate():
    r = SClock(ClockConfig(delta0=0.01, nucleation="gamow")).run()
    nuc = r["checks"]["S0"]["nucleation"]
    assert nuc["mode"] == "gamow" and nuc["gamow"] is not None
    assert abs(nuc["x_esc"] - r["landscape"]["rho_esc"] ** 2) < 1e-12
    assert nuc["Gamma0"] > 0 and abs(nuc["sigma_wait_before_nucleation"] * nuc["Gamma0"] - 1.0) < 1e-12
    assert "n_dim = 1" in nuc["status"]
    assert r["checks"]["descent"]["S_equals_f_identity"]["pass"]
    assert r["checks"]["descent"]["S_equals_f_identity"]["f0_nucleation"] == pytest.approx(0.0, abs=1e-9)


def test_bounce_refuses_without_false_vacuum():
    with pytest.raises(ValueError):
        bounce(1.5 * delta0_metastability_max(), d=4)
    assert np.isfinite(bounce(0.02, d=4).B)
