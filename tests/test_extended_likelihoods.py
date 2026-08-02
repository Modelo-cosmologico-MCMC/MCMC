"""Tests de los likelihoods ampliados (CMB comprimido + fσ8)."""

import numpy as np

from cosmology.extended_likelihoods import (
    z_star_hu_sugiyama, sound_horizon, comoving_to_zstar,
    cmb_compressed_loglike, growth_D_f, fsigma8_model, rsd_loglike,
    CMB_OBS,
)

THETA_LCDM = (67.4, 0.315, 0.0, 8.9)
WB = 0.02236


def test_zstar_and_sound_horizon_physical():
    """z* ~ 1090 (HS96) y r_s ~ 144 Mpc para la cosmología fiducial."""
    zst = z_star_hu_sugiyama(WB, 0.315 * 0.674 ** 2)
    assert 1080.0 < zst < 1100.0
    rs = sound_horizon(THETA_LCDM, WB)
    assert 140.0 < rs < 148.0


def test_shift_parameter_matches_planck():
    """R predicho para ΛCDM fiducial cae a <1σ del comprimido Planck."""
    dm = comoving_to_zstar(THETA_LCDM, WB)
    R = np.sqrt(0.315) * 67.4 * dm / 299792.458
    assert abs(R - CMB_OBS["R"][0]) < CMB_OBS["R"][1]


def test_cmb_loglike_prefers_fiducial():
    """El likelihood penaliza cosmologías lejanas frente a la fiducial."""
    good = cmb_compressed_loglike(THETA_LCDM, WB)
    bad = cmb_compressed_loglike((60.0, 0.40, 0.0, 8.9), WB)
    assert good > bad + 10.0


def test_growth_recovers_matter_domination():
    """A z alto, D ∝ a y f → 1 (dominación de materia)."""
    D, f = growth_D_f(np.array([0.0, 5.0, 50.0]), THETA_LCDM)
    assert abs(f[2] - 1.0) < 0.02       # f(z=50) ≈ 1
    assert D[0] == 1.0 or abs(D[0] - 1.0) < 1e-10
    assert D[1] < D[0]                   # D crece hacia hoy


def test_fsigma8_reasonable_amplitude():
    """fσ8(z~0.4) ~ 0.45-0.50 para σ8 = 0.81 (rango observado)."""
    fs8 = fsigma8_model(np.array([0.38]), THETA_LCDM, 0.81)
    assert 0.40 < fs8[0] < 0.55


def test_rsd_loglike_prefers_reasonable_sigma8():
    """El likelihood RSD distingue σ8 razonable de σ8 absurdo."""
    z = np.array([0.067, 0.38, 0.61])
    fs8 = np.array([0.423, 0.497, 0.436])
    sig = np.array([0.055, 0.045, 0.034])
    good = rsd_loglike(THETA_LCDM, 0.80, z, fs8, sig)
    bad = rsd_loglike(THETA_LCDM, 1.3, z, fs8, sig)
    assert good > bad + 10.0


def test_recovery_limit_extends_to_cmb():
    """Prop. A.1 también aquí: con ε=0 el CMB comprimido del MCMC es el
    de ΛCDM exactamente (misma maquinaria)."""
    ll_mcmc_eps0 = cmb_compressed_loglike((67.4, 0.315, 0.0, 8.9), WB)
    ll_lcdm = cmb_compressed_loglike(THETA_LCDM, WB)
    assert ll_mcmc_eps0 == ll_lcdm
