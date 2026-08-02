"""Tests de los canales del sector oscuro en forma v35 (Apéndice A)."""

import numpy as np

from cosmology.dark_channels import (
    H2_normalized,
    Omega_channel,
    S_of_z,
    f_steps,
    w_DE,
    w_id,
    w_lat,
)


def test_f_steps_identity_without_alphas():
    """Sin escalones, f ≡ 1 (A.5 con α = ())."""
    S = np.linspace(1.0, 100.0, 50)
    assert np.allclose(f_steps(S), 1.0)


def test_f_steps_single_step():
    """Un escalón α=0.1 en S=10: f pasa de 1 a 1.1 (A.5)."""
    f_lo = f_steps(2.0, alphas=[0.1], S_post=[10.0], dS_n=0.5)
    f_hi = f_steps(50.0, alphas=[0.1], S_post=[10.0], dS_n=0.5)
    assert abs(f_lo - 1.0) < 1e-6
    assert abs(f_hi - 1.1) < 1e-6


def test_H2_recovers_lcdm_without_steps():
    """Prop. A.1: sin escalones, (A.4) == ΛCDM con Ω_Λ = Ω_id0 + Ω_lat0."""
    z = np.linspace(0.0, 1100.0, 200)
    kw = {"Omega_b0": 0.0489, "Omega_cdm0": 0.2511, "Omega_r0": 9.2e-5,
          "Omega_id0": 0.65, "Omega_lat0": 0.05, "Omega_k0": 0.0}
    h2 = H2_normalized(z, **kw)
    zp1 = 1.0 + z
    h2_lcdm = 0.30 * zp1 ** 3 + 9.2e-5 * zp1 ** 4 + 0.70
    assert np.allclose(h2, h2_lcdm, rtol=1e-12)


def test_omega_channel_normalized_at_z0():
    """Ω_canal(0) = Ω_0 exactamente (normalización f(z=0)=1, A.4)."""
    om = Omega_channel(0.0, 0.65, alphas=[0.2], S_post=[50.0], dS_n=2.0)
    assert abs(om - 0.65) < 1e-12


def test_w_id_limits():
    """A.6: w_id → 0 a z alto y → −1 a z bajo."""
    assert abs(w_id(100.0)) < 1e-6
    assert abs(w_id(0.0) + 1.0) < 0.01


def test_w_lat_power_law():
    """Para ρ_lat ∝ (1+z)^n, w_lat = −1 + n/3 exactamente (A.6)."""
    for n in (0.0, 1.5, 3.0):
        w = w_lat(1.0, lambda z, n=n: (1.0 + np.asarray(z)) ** n)
        assert abs(w - (-1.0 + n / 3.0)) < 1e-6


def test_w_DE_weighted_mean():
    """w_DE es la media ponderada por densidades (A.6)."""
    w = w_DE(0.0, rho_id_z=3.0, rho_lat_z=1.0, w_id_z=-1.0, w_lat_z=0.0)
    assert abs(w - (-0.75)) < 1e-12


def test_S_of_z_monotone_decreasing():
    """S(z) decrece con z (más pasado, menos estructuración)."""
    z = np.linspace(0.0, 10.0, 50)
    S = S_of_z(z)
    assert np.all(np.diff(S) < 0)
    assert abs(S_of_z(0.0) - 95.0) < 1e-12
