"""Test del límite de recuperación ΛCDM (Tratado de Fundamentos, Prop. A.1).

Si ε → 0 (y las tasas de conversión → 0), el fondo del MCMC debe devolver
exactamente Friedmann con Λ constante: Λ_rel(z) → Λ0 y H(z) → H_ΛCDM(z).
Es el test que demuestra que la extensión es controlada: el MCMC contiene
a ΛCDM como límite exacto, no como aproximación.
"""

import numpy as np

from cosmology.background import H_of_z, Lambda_rel, OMEGA_M0, OMEGA_R0, OMEGA_L0


def H_lcdm(z, H0, Omega_m=OMEGA_M0, Omega_r=OMEGA_R0):
    """Friedmann ΛCDM plano de referencia, independiente del MCMC."""
    z = np.asarray(z, dtype=float)
    Omega_L = 1.0 - Omega_m - Omega_r
    return H0 * np.sqrt(
        Omega_m * (1.0 + z) ** 3 + Omega_r * (1.0 + z) ** 4 + Omega_L
    )


def test_lambda_rel_recovers_constant():
    """Con ε = 0, Λ_rel(z) = Ω_Λ0 exactamente, a todo z."""
    z = np.linspace(0.0, 1100.0, 500)
    Om = Lambda_rel(z, eps=0.0)
    assert np.allclose(Om, OMEGA_L0, rtol=0.0, atol=1e-15)


def test_H_of_z_recovers_lcdm():
    """Con ε = 0, H(z) del MCMC == H_ΛCDM(z) hasta precisión de máquina."""
    z = np.linspace(0.0, 1100.0, 500)
    H0 = 67.4
    H_mcmc = H_of_z(z, H0=H0, eps=0.0)
    H_ref = H_lcdm(z, H0=H0)
    assert np.allclose(H_mcmc, H_ref, rtol=1e-12, atol=0.0)


def test_epsilon_perturbation_is_controlled():
    """Con ε pequeño, la desviación relativa respecto de ΛCDM es O(ε)."""
    z = np.linspace(0.0, 20.0, 200)
    H0 = 67.4
    eps = 0.012
    dev = np.abs(H_of_z(z, H0=H0, eps=eps) / H_lcdm(z, H0=H0) - 1.0)
    assert np.all(dev < eps)
