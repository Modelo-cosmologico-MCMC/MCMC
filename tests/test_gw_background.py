"""Tests del fondo de GWs (III.G, Ecs. 469-470)."""

import numpy as np

from cosmology.gw_background import (
    Omega_GW_MCMC, Omega_GW_SMBH, characteristic_strain,
    snr_pta, predict_for_pta, discriminate_sech2_vs_powerlaw,
    PTA_DATASETS,
)


def test_Omega_GW_in_PTA_band():
    """Ω_GW(10⁻⁸ Hz) detectable (rango 10⁻¹² .. 10⁻⁸)."""
    Omega = float(Omega_GW_MCMC(1e-8))
    assert 1e-12 < Omega < 1e-7


def test_Omega_GW_smbh_power_law():
    """Ω_GW^SMBH(f) decrece según power-law en banda PTA."""
    f = np.array([1e-9, 1e-8, 1e-7])
    Omega = Omega_GW_SMBH(f)
    assert np.all(np.isfinite(Omega))
    assert np.all(Omega > 0.0)


def test_characteristic_strain_finite():
    """h_c(f) finito y positivo en banda PTA."""
    h = characteristic_strain(np.logspace(-9, -7, 20))
    assert np.all(np.isfinite(h))
    assert np.all(h > 0.0)


def test_snr_ska_pta_order():
    """SNR SKA-PTA (15 yr, 150 pulsares, 50 ns) > 0."""
    snr = snr_pta()
    assert snr > 0.0


def test_predict_for_each_pta():
    """`predict_for_pta` devuelve resultado válido para cada dataset."""
    for name in PTA_DATASETS:
        out = predict_for_pta(name)
        assert "name" in out and "ref" in out


def test_discriminate_sech2_powerlaw():
    """El cociente Ω_MCMC/Ω_SMBH varía en órdenes de magnitud sobre el pico."""
    f = np.logspace(-10, -5, 200)
    out = discriminate_sech2_vs_powerlaw(f)
    ratios = out["ratio"]
    assert np.max(ratios) / np.min(ratios[ratios > 0]) > 10.0
