"""Cuarta ecuación, brazo «término angular»: el collado de salida de S0 y el umbral A_c (core/angular_term.py; revisión
del 10-nov-2026, hallazgo 2; PR R2 del plan del 11-nov).

Con V = V0 + ½μ²χ² los puntos críticos interiores están en R(ρ²) = 0 con χ = η/μ²: el vacío en el anillo κ₊ (el θ* del
registro) y el collado en el anillo κ₋, que existe si A ≥ A_c = (t/4)·√(2κ₊/κ₋). Se comprueban las cifras de la
revisión, la signatura del hessiano, la cota con salida por el polo, que el rayo declarado θ = 0 sigue dando los θ* del
registro y la corrida de consistencia con θ_nuc en el collado. Comprobación interna (E8), sin letra: no elige qué mitad
de la Prop. 3.5 se conserva (decisión 2 del autor)."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from core.angular_term import (
    angular_mu2,
    barrier_ratio,
    critical_A,
    exit_route,
    gradient,
    hessian,
    interior_saddle,
    interior_vacuum,
    kappa_minus,
    min_diagonal_distance_with_pole_exit,
    pole_ray,
    potential,
    saddle_distance_for_vacuum_distance,
    saddle_landscape,
    saddle_ray_run,
    tilt_weight,
)
from core.basal import E_BAR, kappa_plus
from core.s_clock import ClockConfig, SClock

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-11-12_angular_saddle"
DELTA0 = (0.003, 0.01, 0.0554)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_critical_A_is_where_the_saddle_leaves_the_boundary():
    """A_c = 0,098 / 0,178 / 0,419 para δ0 = 0,003 / 0,01 / 0,0554; por debajo no hay collado interior."""
    assert [critical_A(d) for d in DELTA0] == pytest.approx([0.098, 0.178, 0.419], abs=5e-4)
    assert [tilt_weight(d) for d in DELTA0] == pytest.approx([0.105, 0.192, 0.453], abs=5e-4)
    old = _script("run_angular_term")
    for d in DELTA0:
        assert tilt_weight(d) == pytest.approx(old.tilt_weight(d), rel=1e-14)     # la t del guion del 23-oct
        A_c = critical_A(d)
        assert interior_saddle(d, 0.999 * A_c) is None and exit_route(d, 0.999 * A_c) == "polo"
        just_above = interior_saddle(d, 1.001 * A_c)
        assert just_above is not None and 0.0 < just_above["theta"] < 2e-3 and exit_route(d, 1.001 * A_c) == "collado interior"
        # en A_c el collado interior coincide con la barrera del rayo del polo y allí ∂V/∂φ_E = 0
        ray = pole_ray(d, A_c)
        assert ray["rho_barrier"] ** 2 == pytest.approx(kappa_minus() * d, rel=1e-6)
        assert abs(ray["dV_dphiE_at_barrier"]) < 1e-6 * d ** 3
    assert angular_mu2(0.01, 0.7) == pytest.approx(SClock(ClockConfig(delta0=0.01, angular_A=0.7)).mu2, rel=1e-14)


@pytest.mark.parametrize("delta0, angle, ratio", [(0.003, 0.716, 2.58), (0.01, 0.659, 2.34), (0.0554, 0.484, 1.78)])
def test_saddle_angle_and_barrier_ratio_at_A_equal_one(delta0, angle, ratio):
    """Con A = 1: ángulo del collado 0,716 / 0,659 / 0,484 rad y cociente de barreras 2,58 / 2,34 / 1,78."""
    sad = interior_saddle(delta0, 1.0)
    assert sad["theta"] == pytest.approx(angle, abs=5e-4)
    assert sad["rho"] ** 2 == pytest.approx(kappa_minus() * delta0, rel=1e-12)                    # ρ̂² = κ₋
    assert sad["chi"] == pytest.approx(E_BAR * delta0 ** 3 / angular_mu2(delta0, 1.0), rel=1e-12)  # χ̂ = ē√δ0/μ̂²
    assert sad["distance_to_diagonal"] == pytest.approx(
        math.asin(0.25 * tilt_weight(delta0) * math.sqrt(kappa_plus() / kappa_minus())), rel=1e-12)
    br = barrier_ratio(delta0, 1.0)
    assert br["ratio"] == pytest.approx(ratio, abs=5e-3) and br["ratio"] > 1.0
    assert br["V_saddle"] == pytest.approx(potential(np.array(sad["phi"]), delta0, 1.0), rel=1e-12)   # forma cerrada de V_c
    # con A = 1 la barrera del rayo declarado no es un collado: el campo baja hacia el interior del cuadrante
    assert pole_ray(delta0, 1.0)["dV_dphiE_at_barrier"] < 0.0 and exit_route(delta0, 1.0) == "collado interior"
    # el paisaje de este módulo es el que integra el reloj
    clk = SClock(ClockConfig(delta0=delta0, angular_A=1.0))
    for q in (1.1 * np.array(sad["phi"]), np.array([0.7, 0.2]) * sad["rho"], np.array([0.3, 1.4]) * sad["rho"]):
        assert potential(q, delta0, 1.0) == pytest.approx(clk._V(q, clk.lam0), rel=1e-12, abs=1e-30)
        assert gradient(q, delta0, 1.0) == pytest.approx(clk._grad(q, clk.lam0), rel=1e-12, abs=1e-30)


@pytest.mark.parametrize("delta0", DELTA0)
def test_hessian_signature_one_negative_eigenvalue_at_the_saddle(delta0):
    land = saddle_landscape(delta0, 1.0)
    es, ev = land["hessian_eigenvalues_saddle"], land["hessian_eigenvalues_vacuum"]
    assert es[0] < 0.0 < es[1] and 0.0 < ev[0] < ev[1]
    assert land["gradient_norm_saddle"] < 1e-10 * delta0 ** 3 and land["gradient_norm_vacuum"] < 1e-10 * delta0 ** 3
    # el hessiano analítico frente a diferencias finitas del gradiente
    q, h = np.array(land["saddle"]["phi"]), 1e-6 * land["saddle"]["rho"]
    fd = np.array([(gradient(q + h * e, delta0, 1.0) - gradient(q - h * e, delta0, 1.0)) / (2.0 * h) for e in np.eye(2)]).T
    assert hessian(q, delta0, 1.0) == pytest.approx(fd, rel=1e-6, abs=1e-8 * delta0 ** 2)


def test_minimum_distance_to_the_diagonal_with_pole_exit():
    """Con salida por el polo el vacío queda a arcsin√(κ₋/2κ₊) = 0,2735 rad (15,7°) o más, sea cual sea δ0."""
    d_min = min_diagonal_distance_with_pole_exit()
    assert d_min == pytest.approx(0.2735, abs=5e-5) and math.degrees(d_min) == pytest.approx(15.7, abs=0.05)
    for d in DELTA0:
        A_c = critical_A(d)
        assert interior_vacuum(d, A_c)["distance_to_diagonal"] == pytest.approx(d_min, rel=1e-12)
        assert interior_vacuum(d, 0.9 * A_c)["distance_to_diagonal"] > d_min > interior_vacuum(d, 1.1 * A_c)["distance_to_diagonal"]
    # según las formas: 24,1° en el borde de la casi-cancelación (b̄² = (16/3)C0m̄²) y 4,1° con b̄² = 100·C0m̄²
    assert math.degrees(min_diagonal_distance_with_pole_exit(b_bar=math.sqrt(16.0 / 3.0))) == pytest.approx(24.1, abs=0.05)
    assert math.degrees(min_diagonal_distance_with_pole_exit(b_bar=10.0)) == pytest.approx(4.1, abs=0.05)
    # vacío y collado comparten χ = η/μ²: acercar el vacío a la diagonal aleja el collado del polo
    assert saddle_distance_for_vacuum_distance(0.05) == pytest.approx(0.1312, abs=5e-5)
    assert saddle_distance_for_vacuum_distance(d_min * (1.0 - 1e-9)) == pytest.approx(math.pi / 4.0, abs=1e-4)
    assert saddle_distance_for_vacuum_distance(1.01 * d_min) is None
    for d in DELTA0:
        land = saddle_landscape(d, 1.0)
        assert saddle_distance_for_vacuum_distance(land["vacuum"]["distance_to_diagonal"]) == pytest.approx(
            land["saddle"]["distance_to_diagonal"], rel=1e-12)


@pytest.mark.parametrize("delta0", DELTA0)
def test_below_A_c_the_exit_is_through_the_pole_and_the_vacuum_stays_at_0_3444(delta0):
    """Con A = 0,8·A_c la salida es por el polo y el vacío queda a 0,3444 rad de la diagonal, en los tres δ0."""
    A = 0.8 * critical_A(delta0)
    assert interior_saddle(delta0, A) is None and exit_route(delta0, A) == "polo"
    assert pole_ray(delta0, A)["dV_dphiE_at_barrier"] > 0.0             # la frontera sostiene el campo: collado de frontera
    vac = interior_vacuum(delta0, A)
    assert vac["distance_to_diagonal"] == pytest.approx(0.3444, abs=5e-5)
    assert vac["distance_to_diagonal"] == pytest.approx(math.asin(tilt_weight(delta0) / (4.0 * A)), rel=1e-12)
    clk = SClock(ClockConfig(delta0=delta0, angular_A=A))                # el mínimo global que el reloj encuentra
    assert clk.true_vacuum_2d["theta"] == pytest.approx(vac["theta"], abs=1e-4)      # tolerancia del minimizador
    assert clk.true_vacuum_2d["rho"] ** 2 == pytest.approx(kappa_plus() * delta0, rel=1e-4)


def test_flow_from_the_pole_escape_point_ends_at_that_vacuum():
    d0 = 0.0554
    A = 0.8 * critical_A(d0)
    out = SClock(ClockConfig(delta0=d0, angular_A=A, eps_res=1e-9, max_steps=2_000_000)).run()
    theta = np.asarray(out["trajectory"]["theta"])
    assert out["finished"] and theta[0] == 0.0 and np.max(np.maximum.accumulate(theta) - theta) <= 1e-12
    assert math.pi / 4.0 - theta[-1] == pytest.approx(0.3444, abs=2e-4)
    assert out["checks"]["descent"]["S_equals_f_identity"]["pass"] and out["checks"]["descent"]["monotonia_4_5"]["pass"]


@pytest.mark.parametrize("delta0, theta_star", [(0.003, 0.759), (0.01, 0.737), (0.0554, 0.672)])
def test_registry_theta_star_still_comes_out_with_the_declared_ray(delta0, theta_star):
    """Los θ* del registro (0,759 / 0,737 / 0,672) siguen saliendo: son el vacío del anillo κ₊, y el rayo declarado
    del reloj sigue siendo θ = 0."""
    cfg = ClockConfig()
    assert cfg.theta_nuc == 0.0 and cfg.nucleation == "escape_point" and cfg.angular_A is None
    vac = interior_vacuum(delta0, 1.0)
    assert vac["theta"] == pytest.approx(theta_star, abs=5e-4)
    clk = SClock(ClockConfig(delta0=delta0, angular_A=1.0))
    assert clk.cfg.theta_nuc == 0.0 and clk.true_vacuum_2d["theta"] == pytest.approx(vac["theta"], abs=1e-4)
    assert interior_vacuum(delta0, 0.2 * tilt_weight(delta0)) is None    # t/4A > 1/√2: el vacío sigue en el polo


@pytest.mark.parametrize("delta0", DELTA0)
def test_consistency_run_from_the_saddle_ray(delta0):
    """El reloj con θ_nuc en el collado: arranca más allá de él, θ crece sin cruzar la diagonal y de la rotación del
    rayo del polo solo queda el último tramo. Sin letra."""
    r = saddle_ray_run(delta0, 1.0)
    assert r["theta_start"] == pytest.approx(r["theta_saddle"], abs=1e-12)
    assert r["rho_barrier_on_ray_over_rho_saddle"] == pytest.approx(1.0, rel=1e-6)   # la barrera del rayo ES el collado
    assert r["finished"] and r["S_equals_f_pass"] and r["monotonia_pass"]
    assert r["theta_max_drop"] <= 1e-12 and not r["crosses_diagonal"]
    assert r["theta_start"] < r["theta_final"] < r["theta_star"] < math.pi / 4.0
    assert r["rotation_left_from_saddle"] == pytest.approx(r["theta_star"] - r["theta_saddle"], rel=1e-12)
    assert 0.05 < r["fraction_of_pole_rotation_left"] < 0.30
    with pytest.raises(ValueError, match="Sin collado interior"):
        saddle_ray_run(delta0, 0.8 * critical_A(delta0))


@pytest.mark.skipif(not (OUT / "results.json").exists(), reason="corrida del collado aún no publicada")
def test_published_artifact_matches_the_closed_forms():
    doc = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    assert [r["delta0"] for r in doc["rows"]] == list(DELTA0)
    for r in doc["rows"]:
        d, land = r["delta0"], r["A_ref"]["landscape"]
        assert r["A_c"] == pytest.approx(critical_A(d), rel=1e-12) and r["t"] == pytest.approx(tilt_weight(d), rel=1e-12)
        assert land["saddle"]["theta"] == pytest.approx(interior_saddle(d, 1.0)["theta"], rel=1e-12)
        assert land["vacuum"]["theta"] == pytest.approx(interior_vacuum(d, 1.0)["theta"], rel=1e-12)
        assert land["barriers"]["ratio"] == pytest.approx(barrier_ratio(d, 1.0)["ratio"], rel=1e-9)
        assert r["below_A_c"]["distance_closed_form"] == pytest.approx(0.3444, abs=5e-5)
        assert r["below_A_c"]["run_from_pole_ray_converged"]["distance_to_diagonal_final"] == pytest.approx(0.3444, abs=2e-4)
        assert 0.05 < r["A_ref"]["run_from_saddle_ray"]["fraction_of_pole_rotation_left"] < 0.30
    s = doc["summary"]
    assert s["interior_saddle_at_A_ref"] and not s["pole_ray_barrier_is_a_saddle_at_A_ref"] and s["pole_exit_below_A_c"]
    assert not s["any_crossing"] and s["identities_pass_all"] and s["unexplained_failed_checks"] == []
    assert doc["min_distance_with_pole_exit_rad"] == pytest.approx(0.2735, abs=5e-5)
    assert "decisión 2 del autor" in doc["reading"] and "sin letra" in doc["reading"]
