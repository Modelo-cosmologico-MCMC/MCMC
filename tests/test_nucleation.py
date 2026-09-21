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


def test_bounce_refuses_without_false_vacuum():
    with pytest.raises(ValueError):
        bounce(1.5 * delta0_metastability_max(), d=4)
    assert np.isfinite(bounce(0.02, d=4).B)
