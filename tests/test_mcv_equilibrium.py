"""MCV (c) — Jeans isotermo con autointercambio empalmado con una envoltura NFW (decisión del autor del 23-oct): las
cifras de referencia del autor (t_edad = 10 Gyr, H0 = 67.4, β = 0.75), todas las ramas del empalme frente a una solución
autosemejante independiente, la continuidad en r₁, la regla de r₁, los bariones y la ECV en el potencial, t_c y la
cúspide junto al agujero negro."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from dynamics.mcv_equilibrium import (
    G_KPC,
    bh_cusp_density,
    bh_cusp_slope,
    bh_influence_radius,
    dm_profile,
    gravothermal_collapse_time,
    interaction_rate,
    isothermal_profile,
    jeans_match,
    nfw_density,
    nfw_halo,
    nfw_mass,
    r1_from_rate,
)

_ISO = solve_ivp(lambda x, y: [y[1], math.exp(-y[0]) - 2.0 * y[1] / x], (1e-6, 1e9), [1e-12 / 6.0, 1e-6 / 3.0],
                 method="LSODA", rtol=1e-11, atol=1e-14, dense_output=True)


def _self_similar_branches(M200, c, som, t_age=10.0):
    """Referencia independiente sin bariones: la esfera isoterma adimensional ψ'' + 2ψ'/ξ = e^{−ψ} y la condición de
    densidad parametrizada por ξ₁ = r₁/r₀ (A = ξ₁·e^{−ψ/2}); las ramas son las raíces de la condición de masa en ξ₁."""
    h = nfw_halo(M200, c)

    def a_of(s0):
        r1 = r1_from_rate(s0, som, t_age, h["rho_s"], h["r_s"])
        rho1 = float(nfw_density(r1, h["rho_s"], h["r_s"]))
        return r1 * math.sqrt(4.0 * math.pi * G_KPC * rho1) / s0, r1, rho1

    def g(xi):
        p, dp = _ISO.sol(xi)
        a = xi * math.exp(-p / 2.0)
        s0 = math.exp(brentq(lambda ls: math.log(a_of(math.exp(ls))[0] / a), math.log(1e-3), math.log(1e5)))
        _, r1, rho1 = a_of(s0)
        return math.log(s0 ** 2 * r1 * xi * dp / G_KPC / float(nfw_mass(r1, h["rho_s"], h["r_s"]))), s0, rho1 * math.exp(p)
    xs = np.geomspace(0.05, 1e6, 700)
    gs = [g(x)[0] for x in xs]
    out = [g(brentq(lambda x: g(x)[0], xs[i], xs[i + 1]))[1:] for i in range(len(xs) - 1) if gs[i] * gs[i + 1] < 0]
    return sorted(out, key=lambda t: t[1])


def test_reference_numbers_of_the_author():
    m = jeans_match(1e9, 15.0, 2.0)
    assert [b["branch"] for b in m["branches"]] == ["con núcleo"]
    b = m["branches"][0]
    assert b["sigma0"] == pytest.approx(11.0, abs=0.05)
    assert b["rho0"] * 1e-9 == pytest.approx(0.069, abs=0.001)              # M☉/pc³
    assert b["r1"] == pytest.approx(0.88, abs=0.005) and b["r_c"] == pytest.approx(0.41, abs=0.005)
    h = nfw_halo(1e12, 10.0)
    assert gravothermal_collapse_time(10.0, h["rho_s"], h["r_s"], beta=0.75) == pytest.approx(45.0, abs=0.5)
    assert gravothermal_collapse_time(10.0, h["rho_s"], h["r_s"], beta=0.45) == pytest.approx(45.0 * 0.75 / 0.45, rel=0.02)


def test_all_branches_agree_with_the_self_similar_reference():
    for M200, c, som in ((1e9, 15.0, 2.0), (1e9, 15.0, 50.0)):
        got = [(b["sigma0"], b["rho0"]) for b in jeans_match(M200, c, som)["branches"]]
        ref = _self_similar_branches(M200, c, som)
        assert len(got) == len(ref) and len(got) >= 1
        for (s, r), (s_ref, r_ref) in zip(got, ref):
            assert s == pytest.approx(s_ref, rel=1e-5) and r == pytest.approx(r_ref, rel=1e-5)
    two = jeans_match(1e9, 15.0, 50.0)["branches"]
    assert [b["branch"] for b in two] == ["con núcleo", "densa"] and two[0]["rho0"] < two[1]["rho0"]


def test_matching_conditions_and_rate_rule():
    m = jeans_match(1e9, 15.0, 2.0)
    b, h = m["branches"][0], m["halo"]
    eps = 1e-6 * b["r1"]
    p = dm_profile(m, b, [b["r1"] - eps, b["r1"] + eps])
    assert p["rho_M"][0] == pytest.approx(p["rho_M"][1], rel=1e-5) and p["M_M"][0] == pytest.approx(p["M_M"][1], rel=1e-5)
    assert float(interaction_rate(nfw_density(b["r1"], h["rho_s"], h["r_s"]), 2.0, b["sigma0"])) * 10.0 == pytest.approx(1.0, rel=1e-9)
    inner = isothermal_profile(b["rho0"], b["sigma0"], [b["r_c"]])
    assert inner["rho_M"][0] == pytest.approx(b["rho0"] / 2.0, rel=1e-8)


def test_velocity_dependent_cross_section_reduces_to_constant():
    const = jeans_match(1e9, 15.0, 2.0)["branches"][0]
    flat = jeans_match(1e9, 15.0, lambda v: 2.0 * (v / 30.0) ** 0.0)["branches"][0]
    assert flat["sigma0"] == pytest.approx(const["sigma0"], rel=1e-10)
    steep = jeans_match(1e9, 15.0, lambda v: 2.0 * (v / 30.0) ** -1.0)["branches"][0]
    assert steep["r1"] > const["r1"]                                          # ⟨v_rel⟩ ≈ 25 km/s < 30: σ_x/m mayor


def test_baryons_and_ecv_enter_the_potential():
    no_b = jeans_match(1e9, 15.0, 2.0)["branches"][0]
    with_b = jeans_match(1e9, 15.0, 2.0, plummer={"M": 2e6, "a": 0.2})
    b = with_b["branches"][0]
    assert max(abs(v) for v in b["residuals"]) < 1e-8 and b["sigma0"] > no_b["sigma0"]
    # ∇²Φ = 4πG(ρ_M − 2ρ_E): con ρ_E = ρ₀/2 el núcleo es plano junto al centro (curvatura central nula)
    r = np.array([1e-3, 1e-2])
    flat = isothermal_profile(1e7, 10.0, r, rho_E=0.5e7)
    assert flat["rho_M"][0] / 1e7 == pytest.approx(1.0, abs=1e-10)
    plain = isothermal_profile(1e7, 10.0, r)
    assert plain["rho_M"][1] < flat["rho_M"][1]


def test_black_hole_cusp():
    assert bh_cusp_slope(0.0) == 0.75 and bh_cusp_slope(1.0) == 1.0
    r_h = bh_influence_radius(1e6, 10.0)
    assert r_h == pytest.approx(G_KPC * 1e6 / 100.0)
    rho = bh_cusp_density([r_h / 16.0, r_h * (1 - 1e-12), 2 * r_h], 1e6, 10.0, 1e8, a=0.0)
    assert rho[0] == pytest.approx(1e8 * 16.0 ** 0.75) and rho[1] == pytest.approx(1e8) and rho[2] == 1e8
