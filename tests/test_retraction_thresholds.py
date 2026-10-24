"""Umbrales de pérdida de dimensión del Maestro: el test reproduce las tres incoherencias del registro del 23-oct (hueco
declarado) con las cifras del guion p1_comprobaciones.py del autor — orden de (708) frente al Cuadro 19, valores de (709)
con los VEV del Cuadro 32 y época de S_mx según (699)/(700) con Ω_m = 0.315 — y la lectura del lapso, sin letra."""

import pytest

from core.retraction_thresholds import (
    S_RETRACTION_710,
    SPIN_ASSIGNMENTS,
    TABLE_19,
    incoherences,
    is_strictly_decreasing,
    lapse_crossing_radius,
    lookback_time_gyr,
    t_crit_709_assignment,
    t_eff_699,
    z_at_t_eff,
)


def test_order_708_against_table_19():
    assert not is_strictly_decreasing(TABLE_19) and is_strictly_decreasing(tuple(reversed(TABLE_19)))
    assert is_strictly_decreasing(S_RETRACTION_710)          # (710): la retracción recorre la emergencia al revés


def test_values_709_with_the_vevs_of_table_32():
    expected_v2 = [[0.209, 0.382, 0.000, 1.000], [0.146, 0.209, 0.382, 0.000], [0.146, 0.209, 0.382, 1.000]]
    for (spins, v0), exp in zip(SPIN_ASSIGNMENTS.values(), expected_v2):
        got = t_crit_709_assignment(spins, v0)
        assert got == pytest.approx(exp, abs=5e-4)
        for power in (1.0, 0.5):                              # con v_n² (guion del autor) o con v_n
            first = t_crit_709_assignment(spins, v0, power)[:3]
            assert all(0.0 <= t <= 0.3825 for t in first)
            assert max(abs(a - b) for a, b in zip(first, TABLE_19[:3])) > 0.1   # ninguna reproduce 10⁻³, 0.002, 0.9


def test_epoch_699_700_with_planck_densities():
    assert t_eff_699(0.0) == pytest.approx(-0.370, abs=5e-4)
    assert z_at_t_eff(0.0) == pytest.approx(0.2956, abs=5e-5)
    z1 = z_at_t_eff(1e-3)
    assert z1 == pytest.approx(0.2964, abs=5e-5) and lookback_time_gyr(z1) == pytest.approx(3.5, abs=0.05)


def test_lapse_reading_without_letter():
    assert lapse_crossing_radius(0.9) == pytest.approx(10.0) and lapse_crossing_radius(1e-3) == pytest.approx(1.001, abs=2e-6)
    inc = incoherences()
    assert inc["order"]["table_19_decreasing"] is False and inc["lapse_reading"]["giro"] == pytest.approx(10.0)
