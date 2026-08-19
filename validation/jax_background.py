"""Implementación INDEPENDIENTE en JAX del fondo cosmológico (6A-P3).

REGLA DE LA RONDA (preinscrita en la misión del frente): NO copiar
funciones internas de la implementación NumPy de referencia
(cosmology/background.py, cosmology/desi_background_fit.py,
cosmology/bayesian_fit.py) — solo se comparten:

1. Las FÓRMULAS declaradas del corpus (v35, Apéndice A.2/A.3):
       F(z)   = 1 + ε·tanh((z_trans − z)/Δz)
       Ω_Λ(z) = (1 − Ω_m − Ω_r)·F(z)/F(0)      [clausura plana,
                                                normalizada hoy]
       E²(z)  = Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_Λ(z)
   y el mapa operativo S ↔ a (v32, LEGACY declarado en S_map.py):
       a(S) = exp(−(S_hoy − S)/(S_hoy − S₄))   para S ≥ S₄
2. Las CONSTANTES FÍSICAS COMUNES EXPLÍCITAS (lista SHARED_CONSTANTS).

La integración de distancias es de Gauss-Legendre compuesto con nodos
fijos (determinista), NO el trapecio de la referencia: la comparación
cruzada mide así, además de la equivalencia algebraica, el error de
truncamiento de los integradores NumPy de producción — y lo publica.

Estatuto: validación interna de implementación (E8) — equivalencia
numérica entre dos implementaciones, no validación física.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# constantes físicas comunes explícitas (las ÚNICAS piezas compartidas
# con la implementación de referencia, aparte de las fórmulas):
SHARED_CONSTANTS = {
    "C_KMS": 299792.458,      # velocidad de la luz [km/s]
    "OMEGA_R0": 9.2e-5,       # radiación + neutrinos relativistas
    "S4": 1.001,              # sello C4 (nacimiento del tiempo)
    "S_TODAY": 95.0,          # parametrización operativa v32 (LEGACY)
}

# nodos de Gauss-Legendre (constantes numéricas, precomputadas):
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(96)
_GL_NODES = jnp.asarray(_GL_NODES)
_GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS)


def E_jax(z, Omega_m, eps=0.0, z_trans=8.9, dz=1.5,
          Omega_r=SHARED_CONSTANTS["OMEGA_R0"]):
    """E(z) = H(z)/H0 desde las fórmulas A.2/A.3 (clausura plana por
    llamada, transición normalizada hoy)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    F = 1.0 + eps * jnp.tanh((z_trans - z) / dz)
    F0 = 1.0 + eps * jnp.tanh(z_trans / dz)
    Omega_DE0 = 1.0 - Omega_m - Omega_r
    arg = (Omega_m * (1.0 + z) ** 3 + Omega_r * (1.0 + z) ** 4
           + Omega_DE0 * F / F0)
    return jnp.sqrt(arg)


def H_jax(z, H0, Omega_m, eps=0.0, z_trans=8.9, dz=1.5):
    """H(z) = H0·E(z) [km/s/Mpc]."""
    return H0 * E_jax(z, Omega_m, eps, z_trans, dz)


def DH_jax(z, H0, Omega_m, eps=0.0, z_trans=8.9, dz=1.5):
    """D_H(z) = c/H(z) [Mpc]."""
    return SHARED_CONSTANTS["C_KMS"] / H_jax(z, H0, Omega_m, eps,
                                             z_trans, dz)


def _inv_E_integral(z_max, Omega_m, eps, z_trans, dz):
    """∫₀^z dz'/E(z') por Gauss-Legendre de 96 nodos (integrando
    analítico y suave: convergencia espectral, error ≪ 1e-13)."""
    half = 0.5 * z_max
    zz = half * (_GL_NODES + 1.0)
    return half * jnp.sum(_GL_WEIGHTS
                          / E_jax(zz, Omega_m, eps, z_trans, dz))


_inv_E_integral_v = jax.vmap(_inv_E_integral, in_axes=(0, None, None,
                                                       None, None))


def DM_jax(z, H0, Omega_m, eps=0.0, z_trans=8.9, dz=1.5):
    """D_M(z) = c/H0·∫₀^z dz'/E(z') [Mpc] (comóvil transversal,
    universo plano)."""
    z = jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64))
    integral = _inv_E_integral_v(z, Omega_m, eps, z_trans, dz)
    return SHARED_CONSTANTS["C_KMS"] / H0 * integral


def DV_jax(z, H0, Omega_m, eps=0.0, z_trans=8.9, dz=1.5):
    """D_V(z) = [z·D_M²·D_H]^(1/3) [Mpc]."""
    dm = DM_jax(z, H0, Omega_m, eps, z_trans, dz)
    dh = DH_jax(z, H0, Omega_m, eps, z_trans, dz)
    return (jnp.asarray(z) * dm ** 2 * dh) ** (1.0 / 3.0)


def DL_jax(z, H0, Omega_m, eps=0.0, z_trans=8.9, dz=1.5):
    """D_L(z) = (1+z)·D_M(z) [Mpc] (universo plano)."""
    return (1.0 + jnp.asarray(z)) * DM_jax(z, H0, Omega_m, eps,
                                           z_trans, dz)


def mu_jax(z, H0, Omega_m, eps=0.0, z_trans=8.9, dz=1.5):
    """Módulo de distancia μ = 5·log10(d_L/10 pc)."""
    return 5.0 * jnp.log10(DL_jax(z, H0, Omega_m, eps, z_trans, dz)
                           * 1e5)


# --- mapa S ↔ z (forma operativa v32, declarada LEGACY) --------------

def s_to_a_jax(S):
    """a(S) = exp(−(S_hoy − S)/(S_hoy − S₄)) para S ≥ S₄; 0 antes."""
    S = jnp.asarray(S, dtype=jnp.float64)
    s4 = SHARED_CONSTANTS["S4"]
    s_today = SHARED_CONSTANTS["S_TODAY"]
    return jnp.where(S >= s4,
                     jnp.exp(-(s_today - S) / (s_today - s4)), 0.0)


def a_to_s_jax(a):
    """Inversa analítica: S = S_hoy + (S_hoy − S₄)·ln a (a ∈ (0, 1])."""
    a = jnp.asarray(a, dtype=jnp.float64)
    s4 = SHARED_CONSTANTS["S4"]
    s_today = SHARED_CONSTANTS["S_TODAY"]
    return s_today + (s_today - s4) * jnp.log(a)


def a_to_z_jax(a):
    return 1.0 / jnp.asarray(a, dtype=jnp.float64) - 1.0


def z_to_a_jax(z):
    return 1.0 / (1.0 + jnp.asarray(z, dtype=jnp.float64))


# --- vector DESI DR2 y χ² -------------------------------------------

def predict_desi_vector_jax(Omega_m, H0rd_kms, eps, z_trans, dz,
                            z_eff, quant):
    """Las 13 componentes adimensionales en el orden del release:
    DH/rd = c/(E·H0rd); DM/rd = (c/H0rd)∫dz'/E; DV/rd =
    [z·(DM/rd)²·(DH/rd)]^(1/3). H0rd_kms es la calibración común
    H0·r_d [km/s] (contrato r_d de la ronda 6A)."""
    c = SHARED_CONSTANTS["C_KMS"]
    z_eff = jnp.asarray(z_eff, dtype=jnp.float64)
    E = E_jax(z_eff, Omega_m, eps, z_trans, dz)
    dh = c / (E * H0rd_kms)
    dm = c / H0rd_kms * _inv_E_integral_v(z_eff, Omega_m, eps,
                                          z_trans, dz)
    dv = (z_eff * dm ** 2 * dh) ** (1.0 / 3.0)
    sel = jnp.asarray([{"DH_over_rs": 0, "DM_over_rs": 1,
                        "DV_over_rs": 2}[q] for q in quant])
    stacked = jnp.stack([dh, dm, dv], axis=0)
    return stacked[sel, jnp.arange(len(z_eff))]


def chi2_jax(model, data_vec, cov):
    """χ² = rᵀ·C⁻¹·r vía solve (sin inversa explícita)."""
    r = jnp.asarray(model) - jnp.asarray(data_vec)
    return float(r @ jnp.linalg.solve(jnp.asarray(cov), r))
