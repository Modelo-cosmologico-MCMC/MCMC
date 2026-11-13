"""Residuo del falso vacío dentro del dominio físico C4 (revisión del 10-nov-2026, hallazgo 4; PR R1 del plan del 11-nov).

La Prop. 3.4 y `core.basal.chi_residue` dan ⟨χ⟩ ≃ (ē/m̄²)·δ0: es el mínimo SIN RESTRINGIR, que cae en φ_E < 0. Dentro
del dominio físico C4 (primer cuadrante del Plano Dual) el falso vacío está sobre la frontera φ_E = 0 y el residuo es
la mitad, (ē/2m̄²)·δ0 (`chi_residue_c4`). Se comprueba: el cociente medido/C4 con las cifras de la revisión, que el
falso vacío del cuadrante está en φ_E = 0, que `chi_residue` no cambia de valor y que el reloj S informa de la
comparación y la pasa. Comprobación interna (E8): nada aquí es una demostración física."""

import numpy as np
import pytest

from core.basal import chi_residue, chi_residue_c4
from core.path_flow import flow, grad_V
from core.s_clock import (
    ClockConfig,
    SClock,
    delta0_metastability_max_analytic,
    radial_landscape,
    run_clock,
)


@pytest.mark.parametrize("delta0, ratio", [(0.003, 1.005), (0.01, 1.016), (0.0554, 1.113)])
def test_measured_residue_is_the_c4_value(delta0, ratio):
    """Cociente medido/C4 = 1,005 / 1,016 / 1,113 (tolerancia 3·δ0); frente a `chi_residue`, 0,502 / 0,508 / 0,557."""
    chi_fv = radial_landscape(delta0)["rho_fv"] / np.sqrt(2.0)          # χ = φ_M/√2 sobre la frontera φ_E = 0
    assert chi_fv / chi_residue_c4(delta0) == pytest.approx(ratio, abs=5e-4)
    assert abs(chi_fv / chi_residue_c4(delta0) - 1.0) < 3.0 * delta0
    assert chi_fv / chi_residue(delta0) == pytest.approx(0.5 * ratio, abs=3e-4)
    # la desviación es la corrección de primer orden (b̄ē²/2m̄⁶)·δ0 = 1.5·δ0 con las formas por defecto, más O(δ0²)
    assert 1.5 * delta0 < chi_fv / chi_residue_c4(delta0) - 1.0 < 2.1 * delta0


@pytest.mark.parametrize("delta0", [0.003, 0.01, 0.0554])
def test_false_vacuum_of_the_quadrant_sits_on_phi_E_zero(delta0):
    """Dentro de C4 el falso vacío está en φ_E = 0 (ligadura activa); el mínimo sin restringir cae en φ_E < 0."""
    ld = radial_landscape(delta0)
    scale = delta0 ** 3
    # el Flujo del Camino proyectado sobre el cuadrante, desde un punto interior a la barrera, acaba en (ρ_fv, 0)
    res = flow(0.3 * ld["rho_barrier"] * np.array([1.0, 1.0]), delta0, d_sigma=0.05 / delta0 ** 2, n_steps=1500)
    assert res["final"][1] == 0.0 and res["final"][0] == pytest.approx(ld["rho_fv"], rel=1e-9)
    assert np.diff(res["V"]).max() <= 1e-12 * scale                     # Monotonía (Teo. 4.5) en el recorrido
    g = grad_V(np.array([ld["rho_fv"], 0.0]), delta0)
    assert abs(g[0]) < 1e-9 * scale and g[1] > 0.0                       # ∂V/∂φ_E = +η/√2: la frontera sostiene el campo
    # sin la ligadura, el mínimo está sobre la antidiagonal (φ_M = −φ_E) y es el de la Prop. 3.4; solo existe por debajo
    # de la mitad del δ0 máximo de C4: en δ_H = 0.0554 no hay falso vacío sin restringir
    free = radial_landscape(delta0, theta=-np.pi / 4.0)
    assert delta0_metastability_max_analytic(theta=-np.pi / 4.0) == pytest.approx(0.5 * delta0_metastability_max_analytic())
    if delta0 < delta0_metastability_max_analytic(theta=-np.pi / 4.0):
        q = free["rho_fv"] * np.array([1.0, -1.0]) / np.sqrt(2.0)
        assert q[1] < 0.0 and np.linalg.norm(grad_V(q, delta0)) < 1e-9 * scale
        assert abs(free["rho_fv"] / chi_residue(delta0) - 1.0) < 6.0 * delta0   # χ = ρ sobre la antidiagonal
    else:
        assert not free["metastable"]


def test_chi_residue_keeps_its_value_and_c4_is_half_of_it():
    assert chi_residue(0.003) == 0.003 and chi_residue(0.01) == 0.01 and chi_residue(0.0554) == 0.0554
    assert chi_residue(0.02, m_bar=2.0, e_bar=3.0) == pytest.approx(0.75 * 0.02)
    for d in (0.003, 0.01, 0.0554):
        assert chi_residue_c4(d) == 0.5 * chi_residue(d)
    assert chi_residue_c4(0.02, m_bar=2.0, e_bar=3.0) == pytest.approx(0.375 * 0.02)


def test_clock_reports_the_comparison_and_passes_it():
    out = run_clock(delta0=0.01)
    chk = out["checks"]["S0"]["chi_residue_3_4"]
    # las dos cifras que el reloj ya publicaba no se mueven (results/2026-09-21_s_clock)
    assert chk["analytic"] == 0.01 and chk["false_vacuum_chi"] == pytest.approx(0.005078450849811868, rel=1e-12)
    assert chk["analytic_c4"] == 0.005 and chk["reference_on_ray"] == pytest.approx(0.005, rel=1e-12)
    assert chk["ratio_false_vacuum_over_c4"] == pytest.approx(1.016, abs=5e-4) and chk["tolerance"] == pytest.approx(0.03)
    assert chk["ratio_false_vacuum_over_analytic"] == pytest.approx(0.508, abs=5e-4)
    assert chk["pass"] is True and chk["status"] == "derivado" and "C4" in chk["note"]
    assert out["ledger"]["derived_checks_failed"] == []
    row = [r for r in out["ledger"]["rows"] if r["item"] == "chi_residue_3_4"]
    assert len(row) == 1 and row[0]["pass"] is True
    dl = out["delivered_state"]
    assert dl["chi_residue_analytic_3_4"] == 0.01 and dl["chi_residue_c4"] == 0.005
    # en un rayo declarado distinto y con el término angular, la referencia es la de C4 sobre ese rayo
    for cfg in (ClockConfig(delta0=0.01, theta_nuc=0.3), ClockConfig(delta0=0.01, angular_A=1.0),
                ClockConfig(delta0=0.0554, angular_A=1.0, theta_nuc=0.484)):
        c = SClock(cfg)._s0_checks()["chi_residue_3_4"]
        assert c["pass"] is True and c["reference_on_ray"] < c["analytic_c4"]
    # sin inclinación (ē = 0) no hay residuo que comparar: sin veredicto, no un fallo
    flat = SClock(ClockConfig(delta0=0.01, e_bar=0.0))._s0_checks()["chi_residue_3_4"]
    assert flat["pass"] is None and flat["false_vacuum_chi"] == 0.0 and flat["ratio_false_vacuum_over_c4"] is None
