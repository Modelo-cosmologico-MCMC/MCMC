"""Vecindad solar como contraste de la Ley de Cronos débil (dynamics/local_kz.py).

Se comprueba: el dataset transcrito pasa el guard (AVAILABLE, esquema,
sha256) y declara que los bytes oficiales no están verificados; el
incremento efectivo es lineal en A y escala como h⁻² con las alturas de
la losa; el margen observacional es positivo; la cota A_2σ invierte la
linealidad. Nada es resultado observacional verificado (E8)."""

import numpy as np
import pytest

from dynamics.local_kz import (
    A_SCULPTOR,
    SLAB_DECLARED,
    cronos_increments,
    load_bounds,
    room,
    significance,
)


def test_dataset_guard_and_caveat():
    b = load_bounds()
    assert b["official_bytes_verified"] is False and "verificar" in b["values"]["rho_dyn_0_HF2000"]["transcribed_from"]
    assert set(b["values"]) == {"rho_dyn_0_HF2000", "rho_dyn_0_MPH2015", "rho_bar_0_MPH2015", "Sigma_1p1_BT2012", "Sigma_bar_1p1_MPH2015"}


def test_increments_linear_in_A_and_h_minus_two():
    a, b = cronos_increments(A_SCULPTOR), cronos_increments(0.1 * A_SCULPTOR)
    assert a["d_rho_eff_0_msun_pc3"] == pytest.approx(10.0 * b["d_rho_eff_0_msun_pc3"])
    assert a["d_sigma_eff_msun_pc2"] == pytest.approx(10.0 * b["d_sigma_eff_msun_pc2"])
    half = cronos_increments(A_SCULPTOR, h_factor=0.5)
    assert half["d_rho_eff_0_msun_pc3"] == pytest.approx(4.0 * a["d_rho_eff_0_msun_pc3"])
    # el límite de Oort efectivo de Cronos con A_Sculptor (ronda del criterio: ≈ 0.75 M☉/pc³)
    assert 0.7 < a["d_rho_eff_0_msun_pc3"] < 0.8
    assert SLAB_DECLARED["rho_dm"] == 0.013


def test_room_positive_and_A_2sigma_inverts_linearity():
    b = load_bounds()
    rm = room(b)
    assert rm["rho_room_0"] > 0.0 and rm["Sigma_room_1p1"] > 0.0
    s = significance(A_SCULPTOR, b)
    a2 = s["A_2sigma_rho_0"]
    s2 = significance(a2, b)
    assert s2["z_rho_0"] == pytest.approx(2.0, abs=1e-9)
    assert s["z_rho_0"] > 5.0                       # con la amplitud única el plano queda excedido
    assert np.isfinite(s["A_2sigma_over_A_sculptor"]) and 0.0 < s["A_2sigma_over_A_sculptor"] < 1.0
