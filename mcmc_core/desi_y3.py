#!/usr/bin/env python3
"""
================================================================================
MÓDULO DESI Y3: Validación con Datos DESI Year 3
================================================================================

Compara las predicciones del modelo MCMC con los datos de DESI Year 3 (2024):
1. BAO: D_V/r_d, D_M/r_d, D_H/r_d a múltiples redshifts
2. Ajuste de parámetros: ε y z_trans
3. Comparación con ΛCDM

Datos DESI Y3 (arXiv:2404.03002):
- Trazadores: BGS, LRG, ELG, QSO, Lyman-α
- Rango de redshift: 0.1 < z < 4.2
- Precisión: ~1% en distancias a z < 2

El modelo MCMC predice desviaciones sutiles de ΛCDM debido a:
- Λ_rel(z) ≈ 1 + ε*tanh((z_trans - z)/Δz)
- Efectos mayores a z > z_trans ≈ 1.0

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar funciones MCMC
from .ontologia_ecv_mcv import (
    EPSILON_ECV, Z_TRANS, DELTA_Z,
    OMEGA_M_MCMC, OMEGA_LAMBDA_MCMC, H0_MCMC, C_LIGHT,
    Lambda_rel, E_MCMC_ECV, H_MCMC_ECV,
    distancia_comovil_ECV, distancia_luminosidad_ECV,
    distancia_volumen_ECV,
    E_LCDM_standard, H_LCDM_standard,
    distancia_comovil_LCDM, distancia_volumen_LCDM,
)


# =============================================================================
# DATOS DESI YEAR 3 (2024)
# =============================================================================

@dataclass
class PuntoDESI:
    """Un punto de medición BAO de DESI."""
    z_eff: float           # Redshift efectivo
    trazador: str          # BGS, LRG, ELG, QSO, Lya
    observable: str        # D_V/r_d, D_M/r_d, D_H/r_d
    valor: float           # Valor medido
    error: float           # Error (1σ)


# DESI Y3 BAO measurements (arXiv:2404.03002)
# Combined with Planck prior r_d = 147.09 Mpc

DESI_Y3_DATA = [
    # BGS (Bright Galaxy Survey) - z bajo
    PuntoDESI(0.295, "BGS", "D_V/r_d", 7.93, 0.15),

    # LRG (Luminous Red Galaxies) - z intermedio
    PuntoDESI(0.510, "LRG1", "D_M/r_d", 13.62, 0.25),
    PuntoDESI(0.510, "LRG1", "D_H/r_d", 20.98, 0.61),
    PuntoDESI(0.706, "LRG2", "D_M/r_d", 17.85, 0.31),
    PuntoDESI(0.706, "LRG2", "D_H/r_d", 20.08, 0.54),
    PuntoDESI(0.930, "LRG3", "D_M/r_d", 21.71, 0.28),
    PuntoDESI(0.930, "LRG3", "D_H/r_d", 17.88, 0.35),

    # ELG (Emission Line Galaxies) - z ~ 1
    PuntoDESI(1.317, "ELG", "D_M/r_d", 27.79, 0.69),
    PuntoDESI(1.317, "ELG", "D_H/r_d", 13.82, 0.42),

    # QSO (Quasars) - z alto
    PuntoDESI(1.491, "QSO", "D_M/r_d", 30.69, 0.80),
    PuntoDESI(1.491, "QSO", "D_H/r_d", 13.26, 0.55),

    # Lyman-α Forest - z muy alto
    PuntoDESI(2.330, "Lya", "D_M/r_d", 39.71, 0.94),
    PuntoDESI(2.330, "Lya", "D_H/r_d", 8.52, 0.17),
]

# Radio del horizonte de sonido en drag epoch
R_DRAG_PLANCK = 147.09  # Mpc (Planck 2018)


# =============================================================================
# FUNCIONES DE PREDICCIÓN MCMC
# =============================================================================

def E_MCMC_parametrizado(z: float, epsilon: float, z_trans: float,
                          delta_z: float = 1.5) -> float:
    """
    E(z) para MCMC con parámetros ajustables.

    E²(z) = Ω_m(1+z)³ + Ω_Λ*Λ_rel(z)

    donde Λ_rel(z) = 1 + ε*tanh((z_trans - z)/Δz)
    """
    Lambda_z = 1 + epsilon * np.tanh((z_trans - z) / delta_z)

    matter_term = OMEGA_M_MCMC * (1 + z)**3
    lambda_term = OMEGA_LAMBDA_MCMC * Lambda_z

    return np.sqrt(matter_term + lambda_term)


def H_MCMC_parametrizado(z: float, epsilon: float, z_trans: float,
                          delta_z: float = 1.5) -> float:
    """H(z) en km/s/Mpc con parámetros ajustables."""
    return H0_MCMC * E_MCMC_parametrizado(z, epsilon, z_trans, delta_z)


def distancia_comovil_parametrizada(z: float, epsilon: float, z_trans: float,
                                     delta_z: float = 1.5) -> float:
    """
    Distancia comóvil D_c(z) en Mpc con parámetros ajustables.
    """
    def integrand(zp):
        return C_LIGHT / H_MCMC_parametrizado(zp, epsilon, z_trans, delta_z)

    result, _ = quad(integrand, 0, z, limit=100)
    return result


def D_H_parametrizado(z: float, epsilon: float, z_trans: float,
                       delta_z: float = 1.5) -> float:
    """Distancia de Hubble D_H(z) = c/H(z) en Mpc."""
    return C_LIGHT / H_MCMC_parametrizado(z, epsilon, z_trans, delta_z)


def D_V_parametrizado(z: float, epsilon: float, z_trans: float,
                       delta_z: float = 1.5) -> float:
    """
    Distancia de volumen D_V(z) = [z D_M² D_H]^(1/3) en Mpc.
    """
    D_M = distancia_comovil_parametrizada(z, epsilon, z_trans, delta_z)
    D_H = D_H_parametrizado(z, epsilon, z_trans, delta_z)

    return (z * D_M**2 * D_H)**(1/3)


# =============================================================================
# CHI-CUADRADO Y AJUSTE
# =============================================================================

def calcular_chi2_DESI(epsilon: float, z_trans: float, delta_z: float = 1.5,
                        r_d: float = R_DRAG_PLANCK) -> float:
    """
    Calcula χ² para los datos DESI Y3.

    Args:
        epsilon: Parámetro ECV
        z_trans: Redshift de transición
        delta_z: Anchura de transición
        r_d: Radio del horizonte de sonido

    Returns:
        χ² total
    """
    chi2 = 0.0

    for punto in DESI_Y3_DATA:
        z = punto.z_eff
        obs = punto.valor
        err = punto.error

        if punto.observable == "D_V/r_d":
            pred = D_V_parametrizado(z, epsilon, z_trans, delta_z) / r_d
        elif punto.observable == "D_M/r_d":
            pred = distancia_comovil_parametrizada(z, epsilon, z_trans, delta_z) / r_d
        elif punto.observable == "D_H/r_d":
            pred = D_H_parametrizado(z, epsilon, z_trans, delta_z) / r_d
        else:
            continue

        chi2 += ((pred - obs) / err)**2

    return chi2


def calcular_chi2_LCDM_DESI(r_d: float = R_DRAG_PLANCK) -> float:
    """
    Calcula χ² para ΛCDM estándar con datos DESI Y3.
    """
    chi2 = 0.0

    for punto in DESI_Y3_DATA:
        z = punto.z_eff
        obs = punto.valor
        err = punto.error

        if punto.observable == "D_V/r_d":
            pred = distancia_volumen_LCDM(z) / r_d
        elif punto.observable == "D_M/r_d":
            pred = distancia_comovil_LCDM(z) / r_d
        elif punto.observable == "D_H/r_d":
            pred = C_LIGHT / (H_LCDM_standard(z) * r_d)
        else:
            continue

        chi2 += ((pred - obs) / err)**2

    return chi2


def ajustar_epsilon_z_trans(verbose: bool = True) -> Dict:
    """
    Ajusta ε y z_trans minimizando χ² con DESI Y3.
    """
    if verbose:
        print("\n  Ajustando parámetros (ε, z_trans)...")

    def chi2_func(params):
        epsilon, z_trans = params
        if epsilon < 0 or epsilon > 0.1:
            return 1e10
        if z_trans < 1 or z_trans > 20:
            return 1e10
        return calcular_chi2_DESI(epsilon, z_trans)

    # Optimización global
    bounds = [(0.001, 0.05), (2.0, 15.0)]

    result = differential_evolution(
        chi2_func,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6
    )

    epsilon_opt = result.x[0]
    z_trans_opt = result.x[1]
    chi2_opt = result.fun

    # χ² para ΛCDM
    chi2_lcdm = calcular_chi2_LCDM_DESI()

    # χ² para parámetros por defecto
    chi2_default = calcular_chi2_DESI(EPSILON_ECV, Z_TRANS)

    if verbose:
        print(f"\n  Parámetros óptimos:")
        print(f"    ε = {epsilon_opt:.4f}")
        print(f"    z_trans = {z_trans_opt:.2f}")
        print(f"\n  χ² comparación:")
        print(f"    χ²(ΛCDM) = {chi2_lcdm:.2f}")
        print(f"    χ²(default ε={EPSILON_ECV}, z_trans={Z_TRANS}) = {chi2_default:.2f}")
        print(f"    χ²(óptimo) = {chi2_opt:.2f}")

    return {
        'epsilon_opt': epsilon_opt,
        'z_trans_opt': z_trans_opt,
        'chi2_opt': chi2_opt,
        'chi2_lcdm': chi2_lcdm,
        'chi2_default': chi2_default,
        'mejora_vs_lcdm': (chi2_lcdm - chi2_opt) / chi2_lcdm * 100
    }


# =============================================================================
# COMPARACIÓN DETALLADA
# =============================================================================

def comparar_DESI_detallado(epsilon: float = EPSILON_ECV,
                            z_trans: float = Z_TRANS,
                            verbose: bool = True) -> Dict:
    """
    Comparación detallada punto por punto con DESI Y3.
    """
    if verbose:
        print(f"\n  Comparación DESI Y3 (ε={epsilon}, z_trans={z_trans}):")
        print(f"\n  {'z':>6} {'Traz':>6} {'Obs':>10} {'Pred':>10} "
              f"{'ΛCDM':>10} {'Δσ_MCMC':>8} {'Δσ_ΛCDM':>8}")
        print("  " + "-"*70)

    resultados = []
    chi2_mcmc = 0.0
    chi2_lcdm = 0.0

    for punto in DESI_Y3_DATA:
        z = punto.z_eff
        obs = punto.valor
        err = punto.error

        if punto.observable == "D_V/r_d":
            pred_mcmc = D_V_parametrizado(z, epsilon, z_trans) / R_DRAG_PLANCK
            pred_lcdm = distancia_volumen_LCDM(z) / R_DRAG_PLANCK
        elif punto.observable == "D_M/r_d":
            pred_mcmc = distancia_comovil_parametrizada(z, epsilon, z_trans) / R_DRAG_PLANCK
            pred_lcdm = distancia_comovil_LCDM(z) / R_DRAG_PLANCK
        elif punto.observable == "D_H/r_d":
            pred_mcmc = D_H_parametrizado(z, epsilon, z_trans) / R_DRAG_PLANCK
            pred_lcdm = C_LIGHT / (H_LCDM_standard(z) * R_DRAG_PLANCK)
        else:
            continue

        delta_mcmc = (pred_mcmc - obs) / err
        delta_lcdm = (pred_lcdm - obs) / err

        chi2_mcmc += delta_mcmc**2
        chi2_lcdm += delta_lcdm**2

        resultados.append({
            'z': z,
            'trazador': punto.trazador,
            'observable': punto.observable,
            'obs': obs,
            'pred_mcmc': pred_mcmc,
            'pred_lcdm': pred_lcdm,
            'delta_mcmc': delta_mcmc,
            'delta_lcdm': delta_lcdm
        })

        if verbose:
            print(f"  {z:6.3f} {punto.trazador:>6} {obs:10.2f} {pred_mcmc:10.2f} "
                  f"{pred_lcdm:10.2f} {delta_mcmc:+8.2f}σ {delta_lcdm:+8.2f}σ")

    n_puntos = len(DESI_Y3_DATA)
    chi2_red_mcmc = chi2_mcmc / (n_puntos - 2)
    chi2_red_lcdm = chi2_lcdm / (n_puntos - 2)

    mejora = (chi2_lcdm - chi2_mcmc) / chi2_lcdm * 100

    if verbose:
        print(f"\n  Resumen:")
        print(f"    χ²_MCMC = {chi2_mcmc:.2f} (red: {chi2_red_mcmc:.3f})")
        print(f"    χ²_ΛCDM = {chi2_lcdm:.2f} (red: {chi2_red_lcdm:.3f})")
        print(f"    Mejora MCMC: {mejora:.1f}%")

    return {
        'resultados': resultados,
        'chi2_mcmc': chi2_mcmc,
        'chi2_lcdm': chi2_lcdm,
        'chi2_red_mcmc': chi2_red_mcmc,
        'chi2_red_lcdm': chi2_red_lcdm,
        'mejora_percent': mejora,
        'n_puntos': n_puntos
    }


# =============================================================================
# ANÁLISIS DE TENSIONES
# =============================================================================

def analizar_tensiones_DESI(verbose: bool = True) -> Dict:
    """
    Analiza tensiones entre DESI Y3 y Planck.
    """
    if verbose:
        print("\n" + "="*65)
        print("  ANÁLISIS DE TENSIONES DESI Y3")
        print("="*65)

    # Valores Planck 2018
    Omega_m_planck = 0.3153
    H0_planck = 67.36

    # DESI Y3 implica (aproximación)
    # De D_V/r_d a z ~ 0.3: implica H0 ligeramente más alto si Omega_m fijo
    # Tensión ~2σ con Planck

    # Calcular H0 efectivo si ajustamos a DESI
    # H0_eff ≈ H0_planck * (D_V_LCDM / D_V_obs)^...

    # Para MCMC con ECV:
    # Λ_rel > 1 a z bajo → H_eff > H_Planck → reduce tensión H0

    Lambda_0 = Lambda_rel(0)
    E_0_mcmc = E_MCMC_ECV(0)
    H0_eff_mcmc = H0_planck * E_0_mcmc

    # Tensión original
    H0_SH0ES = 73.04
    H0_SH0ES_err = 1.04
    diff_std = H0_SH0ES - H0_planck
    err_std = np.sqrt(H0_SH0ES_err**2 + 0.54**2)
    sigma_std = diff_std / err_std

    # Tensión con MCMC
    diff_mcmc = H0_SH0ES - H0_eff_mcmc
    sigma_mcmc = diff_mcmc / err_std

    if verbose:
        print(f"\n  Parámetros:")
        print(f"    Λ_rel(z=0) = {Lambda_0:.4f}")
        print(f"    E_MCMC(z=0) = {E_0_mcmc:.4f}")
        print(f"    H0_eff (MCMC) = {H0_eff_mcmc:.2f} km/s/Mpc")

        print(f"\n  Tensión H0 (SH0ES vs Planck):")
        print(f"    ΛCDM: {sigma_std:.1f}σ")
        print(f"    MCMC: {sigma_mcmc:.1f}σ")
        print(f"    Reducción: {(sigma_std - sigma_mcmc)/sigma_std*100:.0f}%")

    # Comparar χ² DESI
    chi2_mcmc = calcular_chi2_DESI(EPSILON_ECV, Z_TRANS)
    chi2_lcdm = calcular_chi2_LCDM_DESI()

    if verbose:
        print(f"\n  χ² DESI Y3:")
        print(f"    ΛCDM: {chi2_lcdm:.2f}")
        print(f"    MCMC: {chi2_mcmc:.2f}")

    return {
        'Lambda_rel_0': Lambda_0,
        'E_0_mcmc': E_0_mcmc,
        'H0_eff_mcmc': H0_eff_mcmc,
        'sigma_H0_lcdm': sigma_std,
        'sigma_H0_mcmc': sigma_mcmc,
        'reduccion_H0_percent': (sigma_std - sigma_mcmc) / sigma_std * 100,
        'chi2_desi_mcmc': chi2_mcmc,
        'chi2_desi_lcdm': chi2_lcdm
    }


# =============================================================================
# TEST DESI Y3
# =============================================================================

def test_DESI_Y3(verbose: bool = True) -> Dict:
    """
    Test del módulo DESI Y3.
    """
    if verbose:
        print("\n" + "="*65)
        print("  TEST DESI Y3: Validación ε y z_trans con Datos Reales")
        print("="*65)

    # 1. Comparación detallada con parámetros default
    if verbose:
        print(f"\n  1. Comparación con parámetros por defecto:")
        print(f"     ε = {EPSILON_ECV}, z_trans = {Z_TRANS}")

    comparacion = comparar_DESI_detallado(EPSILON_ECV, Z_TRANS, verbose=verbose)

    # 2. Ajuste de parámetros
    if verbose:
        print(f"\n  2. Ajuste óptimo de parámetros:")

    ajuste = ajustar_epsilon_z_trans(verbose=verbose)

    # 3. Comparación con parámetros óptimos
    if verbose:
        print(f"\n  3. Comparación con parámetros óptimos:")

    comparacion_opt = comparar_DESI_detallado(
        ajuste['epsilon_opt'],
        ajuste['z_trans_opt'],
        verbose=verbose
    )

    # 4. Análisis de tensiones
    tensiones = analizar_tensiones_DESI(verbose=verbose)

    # Criterios de éxito
    passed = True

    # MCMC debe ser competitivo con ΛCDM
    passed &= (comparacion['chi2_mcmc'] <= comparacion['chi2_lcdm'] * 1.2)

    # Ajuste óptimo debe mejorar
    passed &= (ajuste['chi2_opt'] <= ajuste['chi2_lcdm'])

    # ε óptimo debe ser pequeño y positivo
    passed &= (0.001 < ajuste['epsilon_opt'] < 0.05)

    if verbose:
        print(f"\n  Resumen DESI Y3:")
        print(f"    Puntos DESI: {len(DESI_Y3_DATA)}")
        print(f"    ε (default): {EPSILON_ECV} → χ² = {comparacion['chi2_mcmc']:.1f}")
        print(f"    ε (óptimo): {ajuste['epsilon_opt']:.4f} → χ² = {ajuste['chi2_opt']:.1f}")
        print(f"    χ² ΛCDM: {comparacion['chi2_lcdm']:.1f}")

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  Estado: {status}")
        print("="*65)

    return {
        'comparacion_default': comparacion,
        'ajuste': ajuste,
        'comparacion_optima': comparacion_opt,
        'tensiones': tensiones,
        'passed': passed
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_DESI_Y3(verbose=True)
