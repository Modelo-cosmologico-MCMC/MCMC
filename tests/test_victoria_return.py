"""El Cruce de Victoria del estado ocupado como retorno (revisión del 10-nov-2026, hallazgo 1; PR R3 del plan del 11-nov).

Con r = B²/(4·C0·M0²) y E0 = M0³/√C0, el vacío exterior del Basal radial vale V(x+) = −E0·t(r) con t(r) = r^{3/2}/3 −
√r/2 + (r − 1)^{3/2}/3 (`core.decade.outer_vacuum_energy`). En la espinodal D = 0 el vacío ocupado queda en +E0/6, por
encima del origen: llegar a ella cuesta W_ret = T0 + E0′/6 > T0 (`retension_cost`) y lo que el cruce dispara es la vuelta
del campo a ρ = 0 (`adiabatic_B_ramp`). Comprobación interna (E8): no elige dónde vive el Cruce de Victoria (decisión 1
del autor) y no cambia el comportamiento del reloj S."""

from decimal import Decimal, getcontext

import numpy as np
import pytest

import core.s_clock as s_clock
from core.basal import T0_analytic, T0_numeric, c_bar, scaled_params
from core.decade import (
    RETENSION_PATHS,
    adiabatic_B_ramp,
    basal_energy_scale,
    fixed_points,
    outer_vacuum_energy,
    retension_cost,
    tension_function,
    tension_ratio,
    vacuum_regime,
)
from core.s_clock import DECLARED_FORMS


def test_t_of_r_limits():
    """t(9/4) = c̄ = 0,840847; t(4/3) = 0; t(1) = −1/6."""
    assert tension_function(9.0 / 4.0) == pytest.approx(0.840847, abs=5e-7)
    assert tension_function(9.0 / 4.0) == pytest.approx(c_bar(), rel=1e-12)
    assert abs(tension_function(4.0 / 3.0)) < 1e-15
    assert tension_function(1.0) == pytest.approx(-1.0 / 6.0, rel=1e-15)
    assert outer_vacuum_energy(1.0) == pytest.approx(+1.0 / 6.0, rel=1e-15)      # en D = 0, por encima del origen
    # con otras formas, la Tensión Primordial de la ley 3.4 es E0·t(r): c̄·δ0³ = (m̄δ0)³/√C0 · t(b̄²/4C0m̄²)
    for m_bar, b_bar, C0 in ((1.0, 3.0, 1.0), (0.8, 2.9, 1.3), (1.2, 4.0, 0.7)):
        p = scaled_params(0.02, m_bar, b_bar)
        r = tension_ratio(p["B"], p["M0_sq"], C0)
        assert r == pytest.approx(b_bar ** 2 / (4.0 * C0 * m_bar ** 2), rel=1e-12)
        assert basal_energy_scale(p["M0_sq"], C0) * tension_function(r) == pytest.approx(T0_analytic(0.02, m_bar, b_bar, C0), rel=1e-12)
        assert T0_numeric(0.02, m_bar, b_bar, C0) == pytest.approx(T0_analytic(0.02, m_bar, b_bar, C0), rel=1e-12)


@pytest.mark.parametrize("r", ["1.1", "2", "5", "37"])
def test_closed_form_against_direct_substitution_to_30_digits(r):
    """V(x+) = −E0·t(r) sustituyendo x+ en el Basal, con 50 dígitos de trabajo (sin sympy: la suite no lo importa)."""
    getcontext().prec = 50
    r = Decimal(r)
    a, b = r.sqrt(), (r - 1).sqrt()
    y = a + b                                              # x+·√C0/M0
    v = y / 2 - a / 2 * y ** 2 + y ** 3 / 6                # V/E0 en y
    t = a ** 3 / 3 - a / 2 + b ** 3 / 3
    assert abs(v + t) < Decimal("1e-30")
    assert abs(1 - 2 * a * y + y ** 2) < Decimal("1e-30")  # dV/dy = ½·(1 − 2√r·y + y²) = 0 en y+
    assert float(-t) == pytest.approx(outer_vacuum_energy(float(r)), rel=1e-12)


def test_three_regimes_and_role_inversion():
    """r > 4/3: x+ es el vacío verdadero; 1 < r < 4/3: papeles invertidos; r = 1: +E0/6 sobre el origen."""
    assert vacuum_regime(9.0 / 4.0) == "vacio_exterior_verdadero" and outer_vacuum_energy(9.0 / 4.0) < 0.0
    for r in (1.05, 1.2, 1.3):
        assert vacuum_regime(r) == "vacio_exterior_falso" and 0.0 < outer_vacuum_energy(r) < 1.0 / 6.0
        # x+ sigue siendo un mínimo local (V'' > 0) aunque quede por encima del origen
        y = np.sqrt(r) + np.sqrt(r - 1.0)
        assert -2.0 * np.sqrt(r) + 2.0 * y > 0.0
    assert vacuum_regime(1.0) == "vacio_exterior_falso" and vacuum_regime(0.9) == "solo_origen"
    rs = np.linspace(1.0, 4.0, 301)
    assert np.all(np.diff(outer_vacuum_energy(rs)) < 0.0)                         # la energía del vacío exterior baja con r
    # la metastabilidad de la Obs. 8.6 (D > (4/3)·C0·M0²) es r > 4/3: el régimen con Tensión Primordial
    p = scaled_params(0.01)
    assert tension_ratio(p["B"], p["M0_sq"]) == pytest.approx(9.0 / 4.0)


@pytest.mark.parametrize("r", [0.5, 0.9, 0.999])
def test_only_minimum_is_the_origin_for_negative_D(r):
    """Para D < 0 (r < 1) el único mínimo es el origen: dV/dρ > 0 para todo ρ > 0."""
    M0_sq, C0 = 1.0, 1.0
    B = 2.0 * np.sqrt(r * C0 * M0_sq)
    assert B ** 2 - 4.0 * C0 * M0_sq < 0.0 and fixed_points(B, M0_sq, C0)[0] == "complex"
    rho = np.linspace(1e-6, 5.0, 20001)
    assert np.all(M0_sq * rho - B * rho ** 3 + C0 * rho ** 5 > 0.0)
    with pytest.raises(ValueError, match="Sin vacío exterior"):
        outer_vacuum_energy(r)
    assert vacuum_regime(r) == "solo_origen"


def test_retension_cost_by_the_three_single_coupling_paths():
    """Coste de re-tensar 1,198 / 1,669 / 1,132·T0 si solo baja B, solo sube M0² o solo sube C0; siempre mayor que T0."""
    r0 = 9.0 / 4.0
    cost = {c: retension_cost(r0, c) for c in RETENSION_PATHS}
    assert cost["B"] == pytest.approx(1.198, abs=5e-4) and cost["M0_sq"] == pytest.approx(1.669, abs=5e-4)
    assert cost["C0"] == pytest.approx(1.132, abs=5e-4)
    # lo que el retorno libera después: 0,198 / 0,669 / 0,132·T0
    assert [cost[c] - 1.0 for c in RETENSION_PATHS] == pytest.approx([0.198, 0.669, 0.132], abs=5e-4)
    # frente a la cuenta directa con los acoplos del cruce (D = 0) por cada camino
    M0_sq, B, C0 = 1.0, 3.0, 1.0
    T0 = basal_energy_scale(M0_sq, C0) * tension_function(tension_ratio(B, M0_sq, C0))
    at_cross = {"B": (M0_sq, 2.0 * np.sqrt(C0 * M0_sq), C0), "M0_sq": (B ** 2 / (4.0 * C0), B, C0),
                "C0": (M0_sq, B, B ** 2 / (4.0 * M0_sq))}
    for c, (m2, b, c0) in at_cross.items():
        assert tension_ratio(b, m2, c0) == pytest.approx(1.0, rel=1e-12)
        V_cross = basal_energy_scale(m2, c0) * outer_vacuum_energy(1.0)           # +E0′/6
        assert (V_cross + T0) / T0 == pytest.approx(cost[c], rel=1e-12)
    for r in (1.5, 2.25, 5.0, 25.0):
        assert all(retension_cost(r, c) > 1.0 for c in RETENSION_PATHS)
    with pytest.raises(ValueError, match="r0 > 4/3"):
        retension_cost(1.2, "B")
    with pytest.raises(ValueError, match="camino"):
        retension_cost(r0, "eta")


def test_adiabatic_ramp_rises_above_the_origin_and_returns_to_it():
    """Bajando B despacio con el campo en x+: máximo de V/E0 entre +0,166 y +0,169 y campo final en el origen."""
    out = adiabatic_B_ramp()
    assert out["V_start_over_E0"] == pytest.approx(-c_bar(), rel=1e-12)
    assert 0.166 < out["V_max_over_E0"] < 0.169 and 0.98 < out["r_at_max"] < 1.0
    assert out["V_max_over_E0"] > out["V_at_spinodal_closed_form"] == 1.0 / 6.0   # la rampa finita rebasa +1/6
    assert out["returns_to_origin"] and out["y_final"] < 1e-12
    # la subida total supera una Tensión Primordial entera: con acoplos en movimiento V no es función de Lyapunov
    assert out["rise_over_T0"] == pytest.approx(1.1995, abs=1e-3) and out["rise_over_T0"] > 1.0
    assert out["rise_over_T0"] == pytest.approx(out["retension_cost_closed_form"], abs=2e-3)
    # mientras el vacío ocupado existe (r > 1), el campo lo sigue: y = √r + √(r − 1) salvo el retraso adiabático
    follow = out["r"] > 1.3
    y_plus = np.sqrt(out["r"][follow]) + np.sqrt(out["r"][follow] - 1.0)
    assert np.max(np.abs(out["y"][follow] / y_plus - 1.0)) < 1e-3
    # una rampa más rápida rebasa más: el máximo tiende a +1/6 desde arriba al alargarla
    fast = adiabatic_B_ramp(ramp_time=400.0)
    assert fast["V_max_over_E0"] > out["V_max_over_E0"] > 1.0 / 6.0 and fast["returns_to_origin"]
    with pytest.raises(ValueError, match="rampa"):
        adiabatic_B_ramp(r_end=1.1)


def test_emergent_mode_says_the_crossing_is_a_return_and_the_declared_trigger_is_untouched():
    """El docstring del modo emergente dice que el evento D = 0 del estado ocupado es un retorno; la decisión C del
    22-sep sigue declarada tal cual (el comportamiento del reloj no cambia)."""
    assert "el evento D = 0 del estado ocupado es un RETORNO" in s_clock.__doc__
    assert "RETORNO" in s_clock.emergent_diagnostic.__doc__ and "decisión 1 del autor" in s_clock.emergent_diagnostic.__doc__
    f = DECLARED_FORMS["collapse_trigger"]
    assert f.startswith("DECISIÓN C (autor, 22-sep-2026)") and "retorno a la frontera" in f and "RETORNO" not in f
