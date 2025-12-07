#!/usr/bin/env python3
"""
================================================================================
MÓDULO CLASS-MCMC: Integración de Λ_rel(z) para C_ℓ y P(k)
================================================================================

Implementa la conexión entre el modelo MCMC y el código CLASS
(Cosmic Linear Anisotropy Solving System) para calcular:
- C_ℓ: Espectro de potencias angular del CMB
- P(k): Espectro de potencias de materia

La modificación principal es reemplazar Λ → Λ_rel(z) = Λ * (1 + ε*tanh((z_trans-z)/Δz))

Esto afecta:
1. La tasa de expansión H(z) → modifica distancias
2. El crecimiento de estructura D(z) → modifica σ₈ y S₈
3. El potencial de Weyl Ψ+Φ → modifica ISW y lensing

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from scipy.integrate import quad, odeint, solve_ivp
from scipy.interpolate import interp1d, CubicSpline
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar funciones ECV
from .ontologia_ecv_mcv import (
    EPSILON_ECV, Z_TRANS, DELTA_Z,
    OMEGA_M_MCMC, OMEGA_LAMBDA_MCMC, H0_MCMC, C_LIGHT,
    Lambda_rel, E_MCMC_ECV, H_MCMC_ECV,
    distancia_comovil_ECV, distancia_luminosidad_ECV,
)


# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================

# Temperatura CMB hoy
T_CMB_0 = 2.7255  # K

# Densidad de radiación
OMEGA_R = 9.24e-5  # Planck 2018

# Densidad de bariones
OMEGA_B = 0.0493

# Constante de Boltzmann
K_B = 8.617e-5  # eV/K

# Masa del electrón
M_E = 0.511e6  # eV

# Redshift de recombinación
Z_STAR = 1089.92  # Planck 2018

# Redshift de drag (decoupling)
Z_DRAG = 1059.94  # Planck 2018


# =============================================================================
# PARÁMETROS CLASS-MCMC
# =============================================================================

@dataclass
class ParametrosCLASS:
    """Parámetros para CLASS-MCMC."""
    # Cosmología
    H0: float = H0_MCMC
    Omega_b: float = OMEGA_B
    Omega_cdm: float = OMEGA_M_MCMC - OMEGA_B
    Omega_Lambda: float = OMEGA_LAMBDA_MCMC

    # ECV
    epsilon_ecv: float = EPSILON_ECV
    z_trans: float = Z_TRANS
    delta_z: float = DELTA_Z

    # Primordial
    A_s: float = 2.1e-9  # Amplitud primordial
    n_s: float = 0.9649  # Índice espectral

    # Reionización
    tau_reio: float = 0.0544
    z_reio: float = 7.67

    # Precisión
    l_max: int = 2500
    k_max: float = 10.0  # h/Mpc


PARAMS_CLASS = ParametrosCLASS()


# =============================================================================
# FUNCIÓN E(z) EXTENDIDA CON RADIACIÓN
# =============================================================================

def E_MCMC_full(z: float, params: ParametrosCLASS = None) -> float:
    """
    Función E(z) completa incluyendo radiación para el MCMC.

    E²(z) = Ω_r(1+z)⁴ + Ω_m(1+z)³ + Ω_Λ*Λ_rel(z)
    """
    p = params or PARAMS_CLASS

    # OMEGA_R ya incluye neutrinos (~9.24e-5)
    # No multiplicar de nuevo por el factor de neutrinos
    Omega_r = OMEGA_R

    # ECV: Λ_rel(z)
    Lambda_z = Lambda_rel(z)

    # Término de radiación (importante a z alto)
    rad_term = Omega_r * (1 + z)**4

    # Término de materia
    matter_term = p.Omega_cdm + p.Omega_b
    matter_term *= (1 + z)**3

    # Término de ECV
    ecv_term = p.Omega_Lambda * Lambda_z

    return np.sqrt(rad_term + matter_term + ecv_term)


def H_MCMC_full(z: float, params: ParametrosCLASS = None) -> float:
    """H(z) en km/s/Mpc con radiación."""
    p = params or PARAMS_CLASS
    return p.H0 * E_MCMC_full(z, params)


# =============================================================================
# CRECIMIENTO DE ESTRUCTURA D(z)
# =============================================================================

def growth_factor_ode(a, y, params):
    """
    EDO para el factor de crecimiento D(a).

    D'' + (3/a + H'/H)D' - (3/2)Ω_m(a)/a²E²(a) D = 0

    donde a = 1/(1+z)

    Nota: solve_ivp usa (t, y, *args), por eso a va primero.
    """
    D, dD = y
    z = 1/a - 1

    p = params
    E_z = E_MCMC_full(z, p)

    # Derivada de E respecto a a (aproximada numéricamente)
    h = 1e-5
    a_plus = min(a + h, 1.0)
    a_minus = max(a - h, 1e-6)
    z_plus = 1/a_plus - 1
    z_minus = 1/a_minus - 1
    E_plus = E_MCMC_full(z_plus, p)
    E_minus = E_MCMC_full(z_minus, p)
    dE_da = (E_plus - E_minus) / (a_plus - a_minus)

    # Coeficientes de la EDO
    coef1 = 3/a + dE_da/E_z

    Omega_m_a = (p.Omega_cdm + p.Omega_b) * (1+z)**3 / E_z**2
    coef2 = -1.5 * Omega_m_a / a**2

    dD_da = dD
    d2D_da2 = -coef1 * dD + coef2 * D

    return [dD_da, d2D_da2]


def calcular_D_MCMC(z_array: np.ndarray, params: ParametrosCLASS = None,
                    normalizar: bool = True) -> np.ndarray:
    """
    Calcula el factor de crecimiento D(z) para MCMC.

    Usa aproximación de Carroll et al. (1992):
    D(z) ≈ g(z) / (1+z)

    donde g(z) = 5/2 * Ω_m(z) / [Ω_m(z)^(4/7) - Ω_Λ(z) + (1+Ω_m(z)/2)(1+Ω_Λ(z)/70)]

    Args:
        z_array: Array de redshifts
        params: Parámetros cosmológicos
        normalizar: Si True, normaliza D(0) = 1

    Returns:
        Array de D(z)
    """
    p = params or PARAMS_CLASS

    z_array = np.atleast_1d(z_array)
    D_z = np.zeros_like(z_array, dtype=float)

    for i, z in enumerate(z_array):
        E_z = E_MCMC_full(z, p)

        # Ω_m(z) y Ω_Λ(z)
        Omega_m_z = (p.Omega_cdm + p.Omega_b) * (1+z)**3 / E_z**2
        Omega_L_z = p.Omega_Lambda * Lambda_rel(z) / E_z**2

        # Fórmula de Carroll et al.
        g_z = 2.5 * Omega_m_z / (
            Omega_m_z**(4/7) - Omega_L_z + (1 + Omega_m_z/2) * (1 + Omega_L_z/70)
        )

        D_z[i] = g_z / (1 + z)

    # Normalizar a D(0) = 1
    if normalizar:
        D_0 = D_z[np.argmin(np.abs(z_array))] if 0 in z_array else calcular_D_MCMC(np.array([0.0]), p, False)[0]
        if D_0 > 0:
            D_z = D_z / D_0

    return D_z


def calcular_f_MCMC(z: float, params: ParametrosCLASS = None) -> float:
    """
    Calcula f(z) = d ln D / d ln a ≈ Ω_m(z)^γ

    donde γ ≈ 0.55 para ΛCDM, modificado para MCMC
    """
    p = params or PARAMS_CLASS

    # Ω_m(z)
    E_z = E_MCMC_full(z, p)
    Omega_m_z = (p.Omega_cdm + p.Omega_b) * (1+z)**3 / E_z**2

    # γ efectivo para MCMC (ligeramente diferente de ΛCDM)
    gamma_eff = 0.55 + 0.02 * EPSILON_ECV

    return Omega_m_z**gamma_eff


def calcular_sigma8_MCMC(params: ParametrosCLASS = None) -> float:
    """
    Calcula σ₈ para el modelo MCMC.

    σ₈ = σ₈_fid * D_MCMC(0) / D_LCDM(0)

    donde σ₈_fid = 0.8111 (Planck 2018)
    """
    p = params or PARAMS_CLASS

    # σ₈ fiducial Planck
    sigma8_planck = 0.8111

    # Factor de corrección por ECV
    # D_MCMC crece más lento debido a Λ_rel > 1 a z bajo
    z_array = np.linspace(0, 10, 100)
    D_mcmc = calcular_D_MCMC(z_array, p)

    # Calcular D para ΛCDM (ε=0)
    params_lcdm = ParametrosCLASS(epsilon_ecv=0)
    D_lcdm = calcular_D_MCMC(z_array, params_lcdm)

    # Ratio de D(0)
    ratio_D = D_mcmc[0] / D_lcdm[0]

    return sigma8_planck * ratio_D


def calcular_S8_MCMC(params: ParametrosCLASS = None) -> float:
    """
    Calcula S₈ = σ₈ * √(Ω_m/0.3) para MCMC.
    """
    p = params or PARAMS_CLASS
    sigma8 = calcular_sigma8_MCMC(p)
    Omega_m = p.Omega_cdm + p.Omega_b
    return sigma8 * np.sqrt(Omega_m / 0.3)


# =============================================================================
# ESPECTRO DE POTENCIAS DE MATERIA P(k)
# =============================================================================

def transfer_function_BBKS(k, params: ParametrosCLASS = None):
    """
    Función de transferencia BBKS (Bardeen et al. 1986).

    T(k) = ln(1+2.34q)/(2.34q) * [1 + 3.89q + (16.1q)² + (5.46q)³ + (6.71q)⁴]^(-1/4)

    donde q = k/(Ω_m h² Mpc⁻¹)

    Vectorizado para arrays de k.
    """
    p = params or PARAMS_CLASS

    h = p.H0 / 100
    Omega_m = p.Omega_cdm + p.Omega_b

    # Shape parameter
    Gamma = Omega_m * h * np.exp(-p.Omega_b * (1 + np.sqrt(2*h)/Omega_m))

    k_arr = np.atleast_1d(k)
    q = k_arr / Gamma

    # Vectorized computation
    result = np.ones_like(q, dtype=float)
    mask = q > 1e-10

    q_valid = q[mask]
    term1 = np.log(1 + 2.34*q_valid) / (2.34*q_valid)
    term2 = 1 + 3.89*q_valid + (16.1*q_valid)**2 + (5.46*q_valid)**3 + (6.71*q_valid)**4
    result[mask] = term1 * term2**(-0.25)

    return result.item() if np.ndim(k) == 0 else result


def P_primordial(k, params: ParametrosCLASS = None):
    """
    Espectro de potencias primordial (Harrison-Zel'dovich modificado).

    P_prim(k) = A_s * (k/k_pivot)^(n_s - 1)

    Vectorizado para arrays de k.
    """
    p = params or PARAMS_CLASS
    k_pivot = 0.05  # Mpc⁻¹
    return p.A_s * (k / k_pivot)**(p.n_s - 1)


def P_k_MCMC(k, z: float = 0, params: ParametrosCLASS = None):
    """
    Espectro de potencias de materia P(k,z) para MCMC.

    P(k,z) = P_prim(k) * T²(k) * D²(z) * k

    Vectorizado para arrays de k.
    """
    p = params or PARAMS_CLASS
    k_arr = np.atleast_1d(k)

    P_prim = P_primordial(k_arr, p)
    T_k = transfer_function_BBKS(k_arr, p)

    # Factor de crecimiento
    D_z = calcular_D_MCMC(np.array([z]), p)[0]

    result = P_prim * T_k**2 * D_z**2 * k_arr
    return result.item() if np.ndim(k) == 0 else result


def calcular_Pk_array(k_array: np.ndarray, z: float = 0,
                      params: ParametrosCLASS = None) -> np.ndarray:
    """Calcula P(k) para un array de k."""
    return np.array([P_k_MCMC(k, z, params) for k in k_array])


# =============================================================================
# ESPECTRO DE POTENCIAS CMB C_ℓ (APROXIMACIÓN)
# =============================================================================

def distancia_angular_MCMC(z: float, params: ParametrosCLASS = None,
                           comoving: bool = True) -> float:
    """
    Distancia angular D_A(z) en Mpc.

    Para CMB (z ~ 1100), usar comoving=True (por defecto).
    Para otros usos, comoving=False da la distancia física.
    """
    p = params or PARAMS_CLASS
    chi = distancia_comovil_ECV(z)  # Mpc
    if comoving:
        return chi  # Para CMB usamos distancia comóvil
    return chi / (1 + z)  # Distancia angular física


def horizonte_sonido_MCMC(z: float, params: ParametrosCLASS = None) -> float:
    """
    Horizonte de sonido r_s(z) en Mpc.

    r_s(z) = ∫_z^∞ c_s(z')/H(z') dz'

    donde c_s = c/√(3(1+R)) y R = 3ρ_b/(4ρ_γ)
    """
    p = params or PARAMS_CLASS

    # Omega_gamma (solo fotones, sin neutrinos)
    # OMEGA_R incluye neutrinos, así que dividimos
    Omega_gamma = OMEGA_R / (1 + 0.2271 * 3.046)  # ~5.4e-5

    def integrand(zp):
        # Baryon loading R = 3Ω_b/(4Ω_γ) * 1/(1+z)
        # A z alto, R → 0 y c_s → c/√3
        R = (3 * p.Omega_b) / (4 * Omega_gamma) / (1 + zp)
        c_s = C_LIGHT / np.sqrt(3 * (1 + R))
        return c_s / H_MCMC_full(zp, p)

    # Integramos hasta z alto para capturar toda la contribución
    result, _ = quad(integrand, z, 1e5, limit=500)
    return result


def theta_star_MCMC(params: ParametrosCLASS = None) -> float:
    """
    Escala angular del horizonte de sonido θ_* = r_s(z_*)/D_A(z_*).

    θ_* ≈ 0.0104 rad para ΛCDM
    """
    p = params or PARAMS_CLASS

    r_s = horizonte_sonido_MCMC(Z_STAR, p)
    D_A = distancia_angular_MCMC(Z_STAR, p)

    return r_s / D_A


def l_acoustic_MCMC(params: ParametrosCLASS = None) -> float:
    """
    Multipolo acústico ℓ_A = π/θ_*.

    ℓ_A ≈ 302 para ΛCDM
    """
    theta_s = theta_star_MCMC(params)
    return np.pi / theta_s


def C_l_TT_approx(l: int, params: ParametrosCLASS = None) -> float:
    """
    Aproximación al espectro TT del CMB.

    C_ℓ^TT ≈ A * (ℓ/ℓ_A)^(n_s-1) * exp(-ℓ(ℓ+1)/ℓ_D²) * [1 + cos(πℓ/ℓ_A)]

    donde ℓ_D es el damping scale y ℓ_A es el acoustic scale.
    """
    p = params or PARAMS_CLASS

    l_A = l_acoustic_MCMC(p)
    l_D = 1500  # Damping scale aproximado

    # Amplitud normalizada
    A = 5e-10 * (p.A_s / 2.1e-9)

    # Término de potencia primordial
    power_term = (l / l_A)**(p.n_s - 1)

    # Damping exponencial (Silk damping)
    damping = np.exp(-l*(l+1) / l_D**2)

    # Picos acústicos - CMB peaks at l_n ≈ l_A * (n - 0.267) for n = 1, 2, 3...
    # First peak at l ≈ 220 when l_A ≈ 302
    # Use phase-shifted formula: cos(π(l/l_1 - 1)) where l_1 ≈ 220
    l_1 = l_A * 0.733  # First peak location (~221 for l_A=302)
    acoustic = 1 + 0.7 * np.cos(np.pi * (l / l_1 - 1))

    # ISW a ℓ bajo (modificado por ECV) - vectorizado
    l_arr = np.atleast_1d(l)
    isw_factor = np.ones_like(l_arr, dtype=float)
    low_l_mask = l_arr < 30
    isw_factor[low_l_mask] = 1 + 0.1 * EPSILON_ECV * (30 - l_arr[low_l_mask]) / 30

    # Supresión por reionización - vectorizado
    reio_factor = np.where(l_arr > 10, np.exp(-2 * p.tau_reio), 1.0)

    result = A * power_term * damping * acoustic * isw_factor * reio_factor

    # Retornar escalar si la entrada era escalar
    return result.item() if np.ndim(l) == 0 else result


def calcular_Cl_array(l_array: np.ndarray,
                      params: ParametrosCLASS = None) -> np.ndarray:
    """Calcula C_ℓ para un array de multipolos."""
    return np.array([C_l_TT_approx(l, params) for l in l_array])


# =============================================================================
# COMPARACIÓN MCMC vs ΛCDM
# =============================================================================

def comparar_con_LCDM(verbose: bool = True) -> Dict:
    """
    Compara las predicciones MCMC con ΛCDM estándar.
    """
    if verbose:
        print("\n" + "="*65)
        print("  CLASS-MCMC: Comparación con ΛCDM")
        print("="*65)

    params_mcmc = PARAMS_CLASS
    params_lcdm = ParametrosCLASS(epsilon_ecv=0)

    resultados = {}

    # 1. θ_* (escala angular del horizonte de sonido)
    theta_mcmc = theta_star_MCMC(params_mcmc)
    theta_lcdm = theta_star_MCMC(params_lcdm)
    diff_theta = (theta_mcmc - theta_lcdm) / theta_lcdm * 100

    resultados['theta_star'] = {
        'MCMC': theta_mcmc * 180/np.pi * 60,  # arcmin
        'LCDM': theta_lcdm * 180/np.pi * 60,
        'diff_percent': diff_theta
    }

    if verbose:
        print(f"\n  θ_* (escala acústica):")
        print(f"    MCMC: {theta_mcmc*180/np.pi*60:.4f} arcmin")
        print(f"    ΛCDM: {theta_lcdm*180/np.pi*60:.4f} arcmin")
        print(f"    Diferencia: {diff_theta:.3f}%")

    # 2. ℓ_A (multipolo acústico)
    l_A_mcmc = l_acoustic_MCMC(params_mcmc)
    l_A_lcdm = l_acoustic_MCMC(params_lcdm)
    diff_l_A = (l_A_mcmc - l_A_lcdm) / l_A_lcdm * 100

    resultados['l_acoustic'] = {
        'MCMC': l_A_mcmc,
        'LCDM': l_A_lcdm,
        'diff_percent': diff_l_A
    }

    if verbose:
        print(f"\n  ℓ_A (multipolo acústico):")
        print(f"    MCMC: {l_A_mcmc:.1f}")
        print(f"    ΛCDM: {l_A_lcdm:.1f}")
        print(f"    Diferencia: {diff_l_A:.3f}%")

    # 3. σ₈
    sigma8_mcmc = calcular_sigma8_MCMC(params_mcmc)
    sigma8_lcdm = 0.8111
    diff_sigma8 = (sigma8_mcmc - sigma8_lcdm) / sigma8_lcdm * 100

    resultados['sigma8'] = {
        'MCMC': sigma8_mcmc,
        'LCDM': sigma8_lcdm,
        'diff_percent': diff_sigma8
    }

    if verbose:
        print(f"\n  σ₈ (amplitud de fluctuaciones):")
        print(f"    MCMC: {sigma8_mcmc:.4f}")
        print(f"    ΛCDM: {sigma8_lcdm:.4f}")
        print(f"    Diferencia: {diff_sigma8:.2f}%")

    # 4. S₈
    S8_mcmc = calcular_S8_MCMC(params_mcmc)
    S8_lcdm = 0.832
    diff_S8 = (S8_mcmc - S8_lcdm) / S8_lcdm * 100

    resultados['S8'] = {
        'MCMC': S8_mcmc,
        'LCDM': S8_lcdm,
        'diff_percent': diff_S8
    }

    if verbose:
        print(f"\n  S₈ = σ₈√(Ω_m/0.3):")
        print(f"    MCMC: {S8_mcmc:.4f}")
        print(f"    ΛCDM (Planck): {S8_lcdm:.4f}")
        print(f"    DES Y3: 0.776 ± 0.017")
        print(f"    Diferencia MCMC-ΛCDM: {diff_S8:.2f}%")

    # 5. D(z) factor de crecimiento
    z_test = np.array([0, 0.5, 1.0, 2.0])
    D_mcmc = calcular_D_MCMC(z_test, params_mcmc)
    D_lcdm = calcular_D_MCMC(z_test, params_lcdm)

    resultados['D_z'] = {
        'z': z_test.tolist(),
        'D_MCMC': D_mcmc.tolist(),
        'D_LCDM': D_lcdm.tolist()
    }

    if verbose:
        print(f"\n  D(z) (factor de crecimiento normalizado):")
        print(f"    {'z':>5} {'D_MCMC':>10} {'D_ΛCDM':>10} {'Diff%':>8}")
        for i, z in enumerate(z_test):
            diff = (D_mcmc[i] - D_lcdm[i]) / D_lcdm[i] * 100
            print(f"    {z:5.1f} {D_mcmc[i]:10.4f} {D_lcdm[i]:10.4f} {diff:8.2f}%")

    if verbose:
        print("\n" + "="*65)

    return resultados


def test_CLASS_MCMC(verbose: bool = True) -> Dict:
    """
    Test del módulo CLASS-MCMC.
    """
    if verbose:
        print("\n" + "="*65)
        print("  TEST CLASS-MCMC: C_ℓ y P(k) con Λ_rel(z)")
        print("="*65)

    # Comparar con ΛCDM
    comparacion = comparar_con_LCDM(verbose=False)

    # Calcular P(k)
    k_array = np.logspace(-3, 1, 50)  # h/Mpc
    Pk_z0 = calcular_Pk_array(k_array, z=0)
    Pk_z1 = calcular_Pk_array(k_array, z=1)

    # Calcular C_ℓ
    l_array = np.arange(2, 2000, 10)
    Cl_mcmc = calcular_Cl_array(l_array)

    if verbose:
        print(f"\n  Parámetros ECV:")
        print(f"    ε = {EPSILON_ECV}")
        print(f"    z_trans = {Z_TRANS}")
        print(f"    Δz = {DELTA_Z}")

        print(f"\n  Resultados clave:")
        print(f"    θ_*: {comparacion['theta_star']['MCMC']:.4f} arcmin")
        print(f"    ℓ_A: {comparacion['l_acoustic']['MCMC']:.1f}")
        print(f"    σ₈: {comparacion['sigma8']['MCMC']:.4f}")
        print(f"    S₈: {comparacion['S8']['MCMC']:.4f}")

        print(f"\n  P(k) a z=0:")
        print(f"    k = 0.01 h/Mpc: P(k) = {P_k_MCMC(0.01):.2e} (Mpc/h)³")
        print(f"    k = 0.1 h/Mpc:  P(k) = {P_k_MCMC(0.1):.2e} (Mpc/h)³")
        print(f"    k = 1.0 h/Mpc:  P(k) = {P_k_MCMC(1.0):.2e} (Mpc/h)³")

        print(f"\n  C_ℓ (TT):")
        print(f"    ℓ = 100:  ℓ(ℓ+1)C_ℓ = {100*101*C_l_TT_approx(100):.2e}")
        print(f"    ℓ = 220:  ℓ(ℓ+1)C_ℓ = {220*221*C_l_TT_approx(220):.2e}")
        print(f"    ℓ = 1000: ℓ(ℓ+1)C_ℓ = {1000*1001*C_l_TT_approx(1000):.2e}")

    # Criterios de éxito
    passed = True

    # θ_* debe estar cerca de 1.04° (62.4 arcmin)
    theta_ok = 60 < comparacion['theta_star']['MCMC'] < 65
    passed &= theta_ok

    # σ₈ debe reducirse respecto a Planck
    sigma8_ok = comparacion['sigma8']['MCMC'] < comparacion['sigma8']['LCDM']
    passed &= sigma8_ok

    # S₈ debe acercarse a DES/KiDS
    S8_ok = 0.78 < comparacion['S8']['MCMC'] < 0.84
    passed &= S8_ok

    if verbose:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  Estado: {status}")
        print("="*65)

    return {
        'comparacion': comparacion,
        'k_array': k_array.tolist(),
        'Pk_z0': Pk_z0.tolist(),
        'Pk_z1': Pk_z1.tolist(),
        'l_array': l_array.tolist(),
        'Cl_mcmc': Cl_mcmc.tolist(),
        'passed': passed
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_CLASS_MCMC(verbose=True)
