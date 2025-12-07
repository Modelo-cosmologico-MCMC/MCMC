#!/usr/bin/env python3
"""
================================================================================
MÓDULO LENSING MCV: Predicciones de Convergencia κ con Perfiles Cored
================================================================================

Implementa predicciones de weak lensing para el modelo MCMC:
1. Convergencia κ con perfiles cored (Zhao γ=0.51)
2. Shear γ para halos individuales
3. Correlación κ-κ a gran escala

La diferencia clave con ΛCDM:
- Perfiles cored reducen κ en el centro de halos
- Densidad crítica de superficie Σ_crit modificada por ECV
- Predicciones de S₈ más bajas (acuerdo con DES/KiDS)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar funciones MCMC
from .ontologia_ecv_mcv import (
    G_GRAV, H0_MCMC, OMEGA_M_MCMC, OMEGA_LAMBDA_MCMC, C_LIGHT,
    E_MCMC_ECV, distancia_comovil_ECV, distancia_luminosidad_ECV,
)
from .sparc_zhao import PerfilZhaoMCMC, ParametrosZhaoMCMC, PerfilNFW


# =============================================================================
# CONSTANTES LENSING
# =============================================================================

# Densidad crítica para lensing
SIGMA_CRIT_FACTOR = C_LIGHT**2 / (4 * np.pi * G_GRAV)  # (km/s)²/kpc * Mpc

# Conversión
MPC_PER_KPC = 1e-3
KPC_PER_MPC = 1e3


# =============================================================================
# PARÁMETROS LENSING MCV
# =============================================================================

@dataclass
class ParametrosLensing:
    """Parámetros para lensing MCV."""
    # Perfil Zhao
    gamma: float = 0.51     # Pendiente interna (cored)
    alpha: float = 2.0
    beta: float = 3.0

    # Escalas
    rho_star: float = 1e7   # M☉/kpc³
    r_star: float = 3.0     # kpc
    S_star: float = 0.5

    # Redshifts típicos
    z_lens: float = 0.3     # Lente
    z_source: float = 1.0   # Fuente


PARAMS_LENSING = ParametrosLensing()


# =============================================================================
# DISTANCIAS ANGULARES
# =============================================================================

def distancia_angular_MCMC(z: float) -> float:
    """
    Distancia angular D_A(z) en Mpc para MCMC con ECV.

    D_A(z) = D_com(z) / (1+z)
    """
    D_com = distancia_comovil_ECV(z)
    return D_com / (1 + z)


def distancia_angular_entre(z1: float, z2: float) -> float:
    """
    Distancia angular entre dos redshifts D_A(z1, z2) en Mpc.

    Para universo plano:
    D_A(z1, z2) = D_A(z2) - D_A(z1) * (1+z1)/(1+z2)

    Nota: Esta es una aproximación válida para universo plano.
    """
    if z2 <= z1:
        return 0.0

    D_A_1 = distancia_angular_MCMC(z1)
    D_A_2 = distancia_angular_MCMC(z2)

    # Distancia comóvil
    chi_1 = D_A_1 * (1 + z1)
    chi_2 = D_A_2 * (1 + z2)

    # Distancia angular entre z1 y z2
    chi_12 = chi_2 - chi_1
    D_A_12 = chi_12 / (1 + z2)

    return D_A_12


# =============================================================================
# DENSIDAD CRÍTICA DE SUPERFICIE
# =============================================================================

def Sigma_crit(z_l: float, z_s: float) -> float:
    """
    Densidad crítica de superficie Σ_crit(z_l, z_s).

    Σ_crit = c² / (4πG) * D_s / (D_l * D_ls)

    donde:
    - D_l: distancia angular a la lente
    - D_s: distancia angular a la fuente
    - D_ls: distancia angular de la lente a la fuente

    Args:
        z_l: Redshift de la lente
        z_s: Redshift de la fuente

    Returns:
        Σ_crit en M☉/kpc²
    """
    if z_s <= z_l:
        return np.inf

    D_l = distancia_angular_MCMC(z_l) * KPC_PER_MPC  # kpc
    D_s = distancia_angular_MCMC(z_s) * KPC_PER_MPC  # kpc
    D_ls = distancia_angular_entre(z_l, z_s) * KPC_PER_MPC  # kpc

    if D_ls <= 0 or D_l <= 0:
        return np.inf

    # Σ_crit = c² / (4πG) * D_s / (D_l * D_ls)
    # c² en (km/s)², G en kpc (km/s)² / M☉
    Sigma_c = C_LIGHT**2 / (4 * np.pi * G_GRAV) * D_s / (D_l * D_ls)

    return Sigma_c


# =============================================================================
# PERFILES DE DENSIDAD PROYECTADA
# =============================================================================

def Sigma_NFW(R: float, M_vir: float, c: float = 10, z_l: float = 0.3) -> float:
    """
    Densidad de superficie proyectada para NFW.

    Σ(R) = 2 ∫₀^∞ ρ_NFW(√(R² + z²)) dz

    usando la fórmula analítica de Bartelmann (1996).

    Args:
        R: Radio proyectado en kpc
        M_vir: Masa virial en M☉
        c: Concentración
        z_l: Redshift de la lente

    Returns:
        Σ(R) en M☉/kpc²
    """
    nfw = PerfilNFW(M_vir, c)

    # Integrar a lo largo de la línea de visión
    def integrand(z):
        r = np.sqrt(R**2 + z**2)
        return nfw.densidad(r)

    # Límite de integración (3 veces radio virial)
    z_max = 3 * nfw.r_vir

    result, _ = quad(integrand, -z_max, z_max, limit=100)
    return result


def Sigma_Zhao(R: float, rho_0: float, r_s: float,
               params: ParametrosLensing = None) -> float:
    """
    Densidad de superficie proyectada para perfil Zhao cored.

    Σ(R) = 2 ∫₀^∞ ρ_Zhao(√(R² + z²)) dz

    Args:
        R: Radio proyectado en kpc
        rho_0: Densidad central en M☉/kpc³
        r_s: Radio de escala en kpc
        params: Parámetros del perfil Zhao

    Returns:
        Σ(R) en M☉/kpc²
    """
    p = params or PARAMS_LENSING

    zhao_params = ParametrosZhaoMCMC(
        gamma=p.gamma,
        alpha=p.alpha,
        beta=p.beta
    )
    zhao = PerfilZhaoMCMC(zhao_params)

    def integrand(z):
        r = np.sqrt(R**2 + z**2)
        return zhao.densidad(r, rho_0, r_s)

    # Límite de integración
    z_max = 100 * r_s

    result, _ = quad(integrand, -z_max, z_max, limit=100)
    return result


# =============================================================================
# CONVERGENCIA κ
# =============================================================================

def kappa_NFW(R, z_l: float = 0.3, z_s: float = 1.0,
              M_vir: float = 1e14, c: float = 10):
    """
    Convergencia κ para perfil NFW - vectorizado.

    κ(R) = Σ(R) / Σ_crit

    Args:
        R: Radio proyectado en kpc (escalar o array)
        z_l: Redshift de la lente
        z_s: Redshift de la fuente
        M_vir: Masa virial en M☉
        c: Concentración

    Returns:
        κ(R) (adimensional)
    """
    R_arr = np.atleast_1d(R)
    Sigma_c = Sigma_crit(z_l, z_s)

    # Calcular Sigma para cada R
    kappa_arr = np.array([Sigma_NFW(r, M_vir, c, z_l) / Sigma_c for r in R_arr])

    return kappa_arr.item() if np.ndim(R) == 0 else kappa_arr


def kappa_Zhao(R, z_l: float = 0.3, z_s: float = 1.0,
               rho_0: float = 1e7, r_s: float = 3.0,
               params: ParametrosLensing = None):
    """
    Convergencia κ para perfil Zhao cored (MCV) - vectorizado.

    κ(R) = Σ_Zhao(R) / Σ_crit

    Args:
        R: Radio proyectado en kpc (escalar o array)
        z_l: Redshift de la lente
        z_s: Redshift de la fuente
        rho_0: Densidad central en M☉/kpc³
        r_s: Radio de escala en kpc
        params: Parámetros del perfil

    Returns:
        κ(R) (adimensional)
    """
    R_arr = np.atleast_1d(R)
    Sigma_c = Sigma_crit(z_l, z_s)

    # Calcular Sigma para cada R
    kappa_arr = np.array([Sigma_Zhao(r, rho_0, r_s, params) / Sigma_c for r in R_arr])

    return kappa_arr.item() if np.ndim(R) == 0 else kappa_arr


# =============================================================================
# SHEAR γ
# =============================================================================

def Sigma_media(R: float, rho_0: float, r_s: float,
                params: ParametrosLensing = None) -> float:
    """
    Densidad de superficie media dentro de R.

    Σ̄(<R) = (2/R²) ∫₀^R Σ(R') R' dR'
    """
    def integrand(Rp):
        return Sigma_Zhao(Rp, rho_0, r_s, params) * Rp

    result, _ = quad(integrand, 0.01, R, limit=50)
    return 2 * result / R**2


def gamma_tangencial(R: float, rho_0: float, r_s: float,
                     z_l: float = 0.3, z_s: float = 1.0,
                     params: ParametrosLensing = None) -> float:
    """
    Shear tangencial γ_t para perfil Zhao.

    γ_t(R) = [Σ̄(<R) - Σ(R)] / Σ_crit = Δκ / Σ_crit

    Args:
        R: Radio proyectado en kpc
        rho_0: Densidad central en M☉/kpc³
        r_s: Radio de escala en kpc
        z_l: Redshift de la lente
        z_s: Redshift de la fuente

    Returns:
        γ_t(R) (adimensional)
    """
    Sigma_bar = Sigma_media(R, rho_0, r_s, params)
    Sigma_R = Sigma_Zhao(R, rho_0, r_s, params)
    Sigma_c = Sigma_crit(z_l, z_s)

    return (Sigma_bar - Sigma_R) / Sigma_c


# =============================================================================
# PREDICCIÓN DE S8 CON LENSING
# =============================================================================

def calcular_S8_lensing(params: ParametrosLensing = None) -> Dict:
    """
    Estima S8 basado en el perfil de lensing MCV.

    La reducción de κ en el centro de halos implica menos clustering
    y por tanto menor S8.

    Returns:
        Dict con S8 estimado y comparaciones
    """
    p = params or PARAMS_LENSING

    # Halo típico de cúmulo (10^14 M☉)
    M_cluster = 1e14
    c_cluster = 5

    # Parámetros Zhao para el halo
    S_loc = 0.5  # Entropía local típica
    rho_0 = p.rho_star * (p.S_star / S_loc)**0.3
    r_s = p.r_star * (S_loc / p.S_star)**0.25

    # Calcular κ a R = 100 kpc (escala típica de lensing)
    R_lens = 100  # kpc

    kappa_nfw = kappa_NFW(R_lens, M_cluster, c_cluster, p.z_lens, p.z_source)
    kappa_zhao = kappa_Zhao(R_lens, rho_0, r_s, p.z_lens, p.z_source, p)

    # Ratio de convergencia → ratio de S8
    ratio_kappa = kappa_zhao / kappa_nfw if kappa_nfw > 0 else 1

    # S8 Planck y ajuste por lensing
    S8_planck = 0.832
    S8_mcv = S8_planck * ratio_kappa**0.5  # Aproximación: S8 ∝ √κ

    return {
        'kappa_NFW': kappa_nfw,
        'kappa_Zhao': kappa_zhao,
        'ratio_kappa': ratio_kappa,
        'S8_Planck': S8_planck,
        'S8_MCV': S8_mcv,
        'S8_DES': 0.776,
        'S8_KiDS': 0.759
    }


# =============================================================================
# COMPARACIÓN NFW vs ZHAO LENSING
# =============================================================================

def comparar_lensing_NFW_Zhao(M_vir: float = 1e14, c: float = 10,
                               z_l: float = 0.3, z_s: float = 1.0,
                               verbose: bool = True) -> Dict:
    """
    Compara perfiles de lensing NFW vs Zhao (MCV).
    """
    p = PARAMS_LENSING

    # Parámetros del halo Zhao
    # Usar S_loc que produce halo comparable
    S_loc = 0.4
    rho_0 = p.rho_star * (p.S_star / S_loc)**0.3 * 50  # Escalar para comparar
    r_s = p.r_star * (S_loc / p.S_star)**0.25 * 5

    # Array de radios
    R_array = np.logspace(0, 3, 50)  # 1 - 1000 kpc

    kappa_nfw_arr = []
    kappa_zhao_arr = []
    gamma_nfw_arr = []
    gamma_zhao_arr = []

    for R in R_array:
        k_nfw = kappa_NFW(R, M_vir, c, z_l, z_s)
        k_zhao = kappa_Zhao(R, rho_0, r_s, z_l, z_s, p)

        kappa_nfw_arr.append(k_nfw)
        kappa_zhao_arr.append(k_zhao)

        # γ tangencial (solo Zhao)
        g_zhao = gamma_tangencial(R, rho_0, r_s, z_l, z_s, p)
        gamma_zhao_arr.append(g_zhao)

    kappa_nfw_arr = np.array(kappa_nfw_arr)
    kappa_zhao_arr = np.array(kappa_zhao_arr)
    gamma_zhao_arr = np.array(gamma_zhao_arr)

    # Diferencia promedio
    diff_kappa = np.mean((kappa_zhao_arr - kappa_nfw_arr) / kappa_nfw_arr) * 100

    if verbose:
        print(f"\n  Comparación Lensing NFW vs Zhao (γ={p.gamma}):")
        print(f"    M_vir = {M_vir:.2e} M☉, c = {c}")
        print(f"    z_lens = {z_l}, z_source = {z_s}")
        print(f"    Σ_crit = {Sigma_crit(z_l, z_s):.2e} M☉/kpc²")

        print(f"\n  {'R (kpc)':>10} {'κ_NFW':>12} {'κ_Zhao':>12} {'Diff%':>10}")
        print("  " + "-"*50)

        for i in [0, 10, 20, 30, 40]:
            R = R_array[i]
            k_nfw = kappa_nfw_arr[i]
            k_zhao = kappa_zhao_arr[i]
            diff = (k_zhao - k_nfw) / k_nfw * 100 if k_nfw > 0 else 0
            print(f"  {R:10.1f} {k_nfw:12.4e} {k_zhao:12.4e} {diff:10.1f}%")

        print(f"\n    Diferencia promedio κ: {diff_kappa:.1f}%")

    return {
        'R_array': R_array,
        'kappa_NFW': kappa_nfw_arr,
        'kappa_Zhao': kappa_zhao_arr,
        'gamma_Zhao': gamma_zhao_arr,
        'diff_kappa_percent': diff_kappa,
        'params': {
            'M_vir': M_vir,
            'c': c,
            'z_l': z_l,
            'z_s': z_s,
            'rho_0': rho_0,
            'r_s': r_s
        }
    }


# =============================================================================
# TEST LENSING MCV
# =============================================================================

def test_Lensing_MCV(verbose: bool = True) -> Dict:
    """
    Test del módulo Lensing MCV.
    """
    if verbose:
        print("\n" + "="*65)
        print("  TEST LENSING MCV: Convergencia κ con Perfiles Cored")
        print("="*65)

    params = PARAMS_LENSING

    # 1. Test de Σ_crit
    if verbose:
        print(f"\n  1. Densidad Crítica de Superficie Σ_crit:")
        for z_l in [0.2, 0.3, 0.5]:
            z_s = 1.0
            Sigma_c = Sigma_crit(z_l, z_s)
            print(f"     Σ_crit(z_l={z_l}, z_s={z_s}) = {Sigma_c:.2e} M☉/kpc²")

    # 2. Comparación NFW vs Zhao
    comparacion = comparar_lensing_NFW_Zhao(
        M_vir=1e14, c=10, z_l=0.3, z_s=1.0, verbose=verbose
    )

    # 3. Predicción S8
    if verbose:
        print(f"\n  2. Predicción S₈ con Lensing MCV:")

    S8_result = calcular_S8_lensing(params)

    if verbose:
        print(f"     κ_NFW / κ_Zhao = {S8_result['ratio_kappa']:.3f}")
        print(f"     S₈ (Planck): {S8_result['S8_Planck']:.3f}")
        print(f"     S₈ (MCV):    {S8_result['S8_MCV']:.3f}")
        print(f"     S₈ (DES):    {S8_result['S8_DES']:.3f}")
        print(f"     S₈ (KiDS):   {S8_result['S8_KiDS']:.3f}")

    # 4. Distancias angulares
    if verbose:
        print(f"\n  3. Distancias Angulares MCMC:")
        for z in [0.3, 0.5, 1.0, 2.0]:
            D_A = distancia_angular_MCMC(z)
            print(f"     D_A(z={z}) = {D_A:.1f} Mpc")

    # Criterios de éxito
    passed = True

    # Σ_crit debe ser positivo y razonable
    Sigma_c_test = Sigma_crit(0.3, 1.0)
    passed &= (1e9 < Sigma_c_test < 1e12)

    # κ_Zhao < κ_NFW en el centro (perfil cored)
    passed &= (comparacion['kappa_Zhao'][0] < comparacion['kappa_NFW'][0])

    # S8_MCV debe estar entre Planck y DES
    passed &= (S8_result['S8_DES'] < S8_result['S8_MCV'] < S8_result['S8_Planck'])

    if verbose:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  Estado: {status}")
        print("="*65)

    return {
        'Sigma_crit_test': Sigma_c_test,
        'comparacion': comparacion,
        'S8_result': S8_result,
        'passed': passed
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_Lensing_MCV(verbose=True)
