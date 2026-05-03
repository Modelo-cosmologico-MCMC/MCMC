"""Tests del factor de color ξ_c en C4 (Ec. 487-488)."""

from mass_program.P4_gut_quarks import xi_color, color_factor_at_seal
from mass_program.B4_masses import predict_fermion_masses


def test_xi_color_value():
    """ξ_c ≈ 3.25e-4 (Ec. 487)."""
    xc = xi_color()
    assert 2e-4 < xc < 5e-4


def test_color_factor_only_quarks_C4():
    """El factor (1 + ξ_c) sólo aplica a quarks en C4."""
    assert color_factor_at_seal("C4", "up") > 1.0
    assert color_factor_at_seal("C4", "down") > 1.0
    assert color_factor_at_seal("C4", "lepton") == 1.0
    assert color_factor_at_seal("C3", "up") == 1.0
    assert color_factor_at_seal("C2", "down") == 1.0
    assert color_factor_at_seal("C1", "up") == 1.0


def test_color_factor_small_correction():
    """La corrección (~3e-4) no degrada significativamente el espectro."""
    res = predict_fermion_masses()
    # Quarks: desviaciones siguen siendo razonables
    for q in ("u", "d", "s", "c", "b", "t"):
        assert res[q]["dev_pct"] < 12.0
