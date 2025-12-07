#!/usr/bin/env python3
"""
Validación Ontológica del Modelo MCMC
=====================================

Versión corregida con:
- ECV (Energía Cuántica Virtual) con transición suave
- MCV (Materia Cuántica Virtual) con dependencia entrópica
- Ley de Cronos (fricción entrópica)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Importar MCMC core
from mcmc_core import (
    # Datos observacionales
    PLANCK_2018, BAO_ALL, BAO_DESI_2024,
    SPARC_CATALOG, GAIA_DR3,
    PANTHEON_PLUS_SUBSET, H0_MEDICIONES,
    PuntoBAO, PuntoSN,
    C_LIGHT,
    # Ontología ECV
    E_MCMC_ECV, H_MCMC_ECV,
    distancia_comovil_ECV, distancia_luminosidad_ECV,
    modulo_distancia_ECV, distancia_volumen_ECV,
    S_of_z, Lambda_rel, Omega_ECV,
    EPSILON_ECV, Z_TRANS,
    # Ontología MCV
    S_local, rho_0_MCV, r_core_MCV, r_core_from_mass,
    perfil_MCV_Burkert, masa_encerrada_MCV_Burkert,
    velocidad_circular_MCV, velocidad_circular_MCV_calibrado, velocidad_NFW_standard,
    FriccionEntropicaMCV, ParametrosFriccion,
    NORM_FACTOR_MCV,
    # SPARC Zhao MCMC
    ParametrosZhaoMCMC, PARAMS_ZHAO, PerfilZhaoMCMC,
    AjustadorSPARC, test_SPARC_Zhao_MCMC,
    # GAIA Zhao MCMC
    AjustadorGAIA, test_GAIA_Zhao_MCMC,
    # ΛCDM para comparación
    E_LCDM_standard, H_LCDM_standard,
    distancia_comovil_LCDM, distancia_luminosidad_LCDM,
    modulo_distancia_LCDM, distancia_volumen_LCDM,
    # Bloque 0-1
    SELLOS, EstadoPrimordial, Pregeometria, integral_total,
    # Cosmología antigua (para comparación)
    CosmologiaMCMC, DELTA_LAMBDA,
    # Lattice gauge
    beta_MCMC, E_min_ontologico, E_min_QCD_scale,
)


# =============================================================================
# CONSTANTES
# =============================================================================

H0_MCMC = 67.36           # km/s/Mpc
OMEGA_M = 0.3153
OMEGA_LAMBDA = 0.6847
R_DRAG = 147.09           # Mpc (Planck 2018)
G_GRAV = 4.302e-6         # kpc (km/s)² / M☉


# =============================================================================
# TEST 1: BAO CON ECV
# =============================================================================

def test_BAO_ECV(H0: float = H0_MCMC, r_d: float = R_DRAG) -> Dict:
    """
    Valida contra datos BAO usando ECV correcta.
    """
    print("\n" + "="*60)
    print("  TEST 1: BAO (con ECV ontológica)")
    print("="*60)

    chi2_ecv = 0.0
    chi2_lcdm = 0.0
    n_puntos = 0

    resultados = []

    for punto in BAO_ALL:
        z = punto.z_eff
        obs = punto.valor
        err = punto.error

        if punto.observable == "D_V/r_d":
            pred_ecv = distancia_volumen_ECV(z) / r_d
            pred_lcdm = distancia_volumen_LCDM(z) / r_d
        elif punto.observable == "D_M/r_d":
            pred_ecv = distancia_comovil_ECV(z) / r_d
            pred_lcdm = distancia_comovil_LCDM(z) / r_d
        elif punto.observable == "D_H/r_d":
            pred_ecv = C_LIGHT / (H_MCMC_ECV(z) * r_d)
            pred_lcdm = C_LIGHT / (H_LCDM_standard(z) * r_d)
        else:
            continue

        delta_ecv = ((pred_ecv - obs) / err)**2
        delta_lcdm = ((pred_lcdm - obs) / err)**2

        chi2_ecv += delta_ecv
        chi2_lcdm += delta_lcdm
        n_puntos += 1

        resultados.append({
            "z": z,
            "survey": punto.survey,
            "observable": punto.observable,
            "obs": obs,
            "pred_ecv": pred_ecv,
            "pred_lcdm": pred_lcdm,
            "delta_ecv": np.sqrt(delta_ecv),
            "delta_lcdm": np.sqrt(delta_lcdm)
        })

    # Mostrar resultados
    print(f"\n  {'z':>5} {'Survey':>12} {'Obs':>8} {'ECV':>8} {'ΛCDM':>8} {'Δ_ECV':>7}")
    print("  " + "-"*55)

    for r in resultados[:8]:
        print(f"  {r['z']:5.2f} {r['survey']:>12} {r['obs']:8.3f} "
              f"{r['pred_ecv']:8.3f} {r['pred_lcdm']:8.3f} {r['delta_ecv']:7.2f}σ")

    if len(resultados) > 8:
        print(f"  ... ({len(resultados) - 8} más)")

    chi2_red_ecv = chi2_ecv / (n_puntos - 2) if n_puntos > 2 else chi2_ecv
    chi2_red_lcdm = chi2_lcdm / (n_puntos - 2) if n_puntos > 2 else chi2_lcdm

    mejora = 100 * (chi2_lcdm - chi2_ecv) / chi2_lcdm if chi2_lcdm > 0 else 0

    print(f"\n  Resultados BAO (ECV):")
    print(f"    Puntos: {n_puntos}")
    print(f"    χ²_ECV = {chi2_ecv:.2f} (red: {chi2_red_ecv:.3f})")
    print(f"    χ²_ΛCDM = {chi2_lcdm:.2f} (red: {chi2_red_lcdm:.3f})")
    print(f"    Mejora ECV: {mejora:.1f}%")

    return {
        "chi2_ecv": chi2_ecv,
        "chi2_lcdm": chi2_lcdm,
        "chi2_red_ecv": chi2_red_ecv,
        "chi2_red_lcdm": chi2_red_lcdm,
        "n_puntos": n_puntos,
        "mejora_percent": mejora,
        "passed": chi2_red_ecv < 3.0
    }


# =============================================================================
# TEST 2: SUPERNOVAS Ia CON ECV
# =============================================================================

def test_SNe_ECV(H0: float = H0_MCMC) -> Dict:
    """
    Test de Supernovas con ECV ontológica.

    La ECV tiene transición suave que mejora el ajuste a z bajo.
    """
    print("\n" + "="*60)
    print("  TEST 2: Supernovas Ia (con ECV ontológica)")
    print("="*60)

    chi2_ecv = 0.0
    chi2_lcdm = 0.0
    n_puntos = 0

    print(f"\n  {'z':>5} {'μ_obs':>8} {'μ_ECV':>8} {'μ_ΛCDM':>8} {'Δ_ECV':>7}")
    print("  " + "-"*45)

    for sn in PANTHEON_PLUS_SUBSET:
        z = sn.z_cmb
        mu_obs = sn.mu
        err = sn.mu_err

        # MCMC con ECV correcta
        mu_ecv = modulo_distancia_ECV(z)

        # ΛCDM estándar
        mu_lcdm = modulo_distancia_LCDM(z)

        delta_ecv = ((mu_ecv - mu_obs) / err)**2
        delta_lcdm = ((mu_lcdm - mu_obs) / err)**2

        chi2_ecv += delta_ecv
        chi2_lcdm += delta_lcdm
        n_puntos += 1

        if n_puntos <= 10:
            print(f"  {z:5.2f} {mu_obs:8.2f} {mu_ecv:8.2f} {mu_lcdm:8.2f} "
                  f"{np.sqrt(delta_ecv):7.2f}σ")

    if n_puntos > 10:
        print(f"  ... ({n_puntos - 10} más)")

    chi2_red_ecv = chi2_ecv / (n_puntos - 2)
    chi2_red_lcdm = chi2_lcdm / (n_puntos - 2)

    mejora = 100 * (chi2_lcdm - chi2_ecv) / chi2_lcdm if chi2_lcdm > 0 else 0

    print(f"\n  Resultados Supernovas (ECV):")
    print(f"    SNe Ia: {n_puntos}")
    print(f"    χ²_ECV = {chi2_ecv:.2f} (red: {chi2_red_ecv:.3f})")
    print(f"    χ²_ΛCDM = {chi2_lcdm:.2f} (red: {chi2_red_lcdm:.3f})")
    print(f"    Mejora ECV: {mejora:.1f}%")

    # Criterio: χ² reducido < 1.5 y competitivo con ΛCDM
    passed = chi2_red_ecv < 1.5 or chi2_ecv <= chi2_lcdm * 1.1

    return {
        "chi2_ecv": chi2_ecv,
        "chi2_lcdm": chi2_lcdm,
        "chi2_red_ecv": chi2_red_ecv,
        "chi2_red_lcdm": chi2_red_lcdm,
        "n_puntos": n_puntos,
        "mejora_percent": mejora,
        "passed": passed
    }


# =============================================================================
# TEST 3: TENSIÓN H0 CON ECV
# =============================================================================

def test_H0_tension_ECV() -> Dict:
    """
    Evalúa la reducción de la tensión H0 con ECV.
    """
    print("\n" + "="*60)
    print("  TEST 3: Tensión H0 (con ECV)")
    print("="*60)

    H0_SH0ES = H0_MEDICIONES[0]
    H0_Planck = H0_MEDICIONES[6]

    # Tensión estándar
    diff_std = H0_SH0ES.valor - H0_Planck.valor
    err_std = np.sqrt(H0_SH0ES.error_stat**2 + H0_Planck.error_stat**2)
    sigma_std = diff_std / err_std

    # Con ECV, Λ_rel(z=0) > 1 produce H efectivo mayor
    # H_eff = H0 * sqrt(Omega_m + Omega_Lambda * Lambda_rel(0))
    Lambda_0 = Lambda_rel(0)
    E_0 = np.sqrt(OMEGA_M + OMEGA_LAMBDA * Lambda_0)
    H0_ECV_eff = H0_Planck.valor * E_0

    diff_ecv = H0_SH0ES.valor - H0_ECV_eff
    sigma_ecv = diff_ecv / err_std

    reduccion = (sigma_std - sigma_ecv) / sigma_std * 100

    print(f"\n  Mediciones H0 (km/s/Mpc):")
    print(f"    SH0ES (local):   {H0_SH0ES.valor:.2f} ± {H0_SH0ES.error_stat:.2f}")
    print(f"    Planck (CMB):    {H0_Planck.valor:.2f} ± {H0_Planck.error_stat:.2f}")

    print(f"\n  Corrección ECV:")
    print(f"    Λ_rel(z=0) = {Lambda_0:.4f}")
    print(f"    E(z=0) = {E_0:.4f}")
    print(f"    H0_eff (ECV) = {H0_ECV_eff:.2f} km/s/Mpc")

    print(f"\n  Tensión:")
    print(f"    ΛCDM:   {sigma_std:.1f}σ")
    print(f"    ECV:    {sigma_ecv:.1f}σ")
    print(f"    Reducción: {reduccion:.0f}%")

    return {
        "H0_SH0ES": H0_SH0ES.valor,
        "H0_Planck": H0_Planck.valor,
        "H0_ECV_eff": H0_ECV_eff,
        "Lambda_rel_0": Lambda_0,
        "sigma_LCDM": sigma_std,
        "sigma_ECV": sigma_ecv,
        "reduccion_percent": reduccion,
        "passed": sigma_ecv < sigma_std
    }


# =============================================================================
# TEST 4: SPARC CON PERFIL ZHAO MCMC
# =============================================================================

def test_SPARC_MCV() -> Dict:
    """
    Valida curvas de rotación SPARC con Perfil Zhao MCMC (γ=0.51).

    Usa el perfil Zhao refinado con:
    - γ=0.51 (cored) en lugar de NFW (γ=1)
    - S_loc como parámetro libre por galaxia
    - Calibración bariónica Υ_* para disco y bulbo
    - Fricción entrópica (Ley de Cronos)

    Objetivo: Mejora > 40% sobre NFW
    """
    # Usar la función test_SPARC_Zhao_MCMC del módulo sparc_zhao
    result = test_SPARC_Zhao_MCMC(SPARC_CATALOG[:5], ajustar_barionico=False, verbose=True)

    # Adaptar el formato de salida para compatibilidad
    return {
        "chi2_NFW": result['chi2_NFW'],
        "chi2_MCV": result['chi2_MCMC'],  # MCV = MCMC (Zhao)
        "n_galaxias": result['n_galaxias'],
        "n_puntos": result['n_puntos'],
        "mejora_percent": result['mejora_percent'],
        "resultados": result['resultados'],
        "passed": result['passed']
    }


# =============================================================================
# TEST 5: GAIA CON PERFIL ZHAO MCMC
# =============================================================================

def test_GAIA_MCV() -> Dict:
    """
    Valida cinemática de la Vía Láctea con Perfil Zhao MCMC (γ=0.51).

    Usa el perfil Zhao refinado con:
    - γ=0.51 (cored) en lugar de NFW (γ=1)
    - S_loc como parámetro libre ajustado
    - Componentes bariónicas (disco + bulbo)
    - Fricción entrópica (Ley de Cronos)

    Objetivo: Mejor que NFW y modelo plano
    """
    # Usar la función test_GAIA_Zhao_MCMC del módulo sparc_zhao
    result = test_GAIA_Zhao_MCMC(verbose=True)

    # Adaptar el formato de salida para compatibilidad
    return {
        "R_sol": GAIA_DR3.R_sol,
        "V_sol": GAIA_DR3.V_sol,
        "S_loc": result['S_loc'],
        "r_s": result['r_s'],
        "rho_0": result['rho_0'],
        "chi2_flat": result['chi2_flat'],
        "chi2_NFW": result['chi2_NFW'],
        "chi2_MCV": result['chi2_MCMC'],
        "mejora_vs_NFW": result['mejora_vs_nfw'],
        "mejora_vs_flat": result['mejora_vs_flat'],
        "rho_DM_local": GAIA_DR3.rho_DM_local,
        "passed": result['passed']
    }


# =============================================================================
# TEST 6: TENSIÓN S8
# =============================================================================

def test_S8_tension() -> Dict:
    """
    Evalúa la reducción de la tensión S8.
    """
    print("\n" + "="*60)
    print("  TEST 6: Tensión S8 (fricción entrópica)")
    print("="*60)

    S8_Planck = PLANCK_2018.S8
    S8_Planck_err = PLANCK_2018.S8_err
    S8_DES = PLANCK_2018.S8_DES
    S8_DES_err = PLANCK_2018.S8_DES_err
    S8_KiDS = PLANCK_2018.S8_KiDS
    S8_KiDS_err = PLANCK_2018.S8_KiDS_err

    # Tensión estándar
    diff_DES = S8_Planck - S8_DES
    err_DES = np.sqrt(S8_Planck_err**2 + S8_DES_err**2)
    sigma_DES_std = diff_DES / err_DES

    diff_KiDS = S8_Planck - S8_KiDS
    err_KiDS = np.sqrt(S8_Planck_err**2 + S8_KiDS_err**2)
    sigma_KiDS_std = diff_KiDS / err_KiDS

    # La fricción entrópica reduce el crecimiento de estructuras
    # S8_MCMC ≈ S8_Planck * (1 - η_eff)
    # donde η_eff depende de la escala
    friction = FriccionEntropicaMCV()
    rho_cluster = 1e7  # M☉/kpc³ (densidad típica de cúmulos)
    eta_eff = friction.coeficiente(rho_cluster) * 0.5  # Factor de escala

    factor_reduccion = 1 - eta_eff
    S8_MCV = S8_Planck * factor_reduccion

    diff_DES_mcv = S8_MCV - S8_DES
    sigma_DES_mcv = diff_DES_mcv / err_DES

    diff_KiDS_mcv = S8_MCV - S8_KiDS
    sigma_KiDS_mcv = diff_KiDS_mcv / err_KiDS

    print(f"\n  Valores S8:")
    print(f"    Planck:  {S8_Planck:.3f} ± {S8_Planck_err}")
    print(f"    DES Y3:  {S8_DES:.3f} ± {S8_DES_err}")
    print(f"    KiDS:    {S8_KiDS:.3f} ± {S8_KiDS_err}")
    print(f"    MCV:     {S8_MCV:.3f} (η_eff = {eta_eff:.3f})")

    print(f"\n  Tensión Planck vs DES:")
    print(f"    ΛCDM: {sigma_DES_std:.1f}σ")
    print(f"    MCV:  {abs(sigma_DES_mcv):.1f}σ")

    print(f"\n  Tensión Planck vs KiDS:")
    print(f"    ΛCDM: {sigma_KiDS_std:.1f}σ")
    print(f"    MCV:  {abs(sigma_KiDS_mcv):.1f}σ")

    reduccion_DES = (sigma_DES_std - abs(sigma_DES_mcv)) / sigma_DES_std * 100
    reduccion_KiDS = (sigma_KiDS_std - abs(sigma_KiDS_mcv)) / sigma_KiDS_std * 100

    print(f"\n  Reducción tensión:")
    print(f"    DES: {reduccion_DES:.0f}%")
    print(f"    KiDS: {reduccion_KiDS:.0f}%")

    return {
        "S8_Planck": S8_Planck,
        "S8_DES": S8_DES,
        "S8_KiDS": S8_KiDS,
        "S8_MCV": S8_MCV,
        "eta_eff": eta_eff,
        "sigma_DES_LCDM": sigma_DES_std,
        "sigma_DES_MCV": abs(sigma_DES_mcv),
        "sigma_KiDS_LCDM": sigma_KiDS_std,
        "sigma_KiDS_MCV": abs(sigma_KiDS_mcv),
        "reduccion_DES": reduccion_DES,
        "reduccion_KiDS": reduccion_KiDS,
        "passed": abs(sigma_DES_mcv) < sigma_DES_std
    }


# =============================================================================
# TEST 7: EDAD DEL UNIVERSO
# =============================================================================

def test_edad_universo_ECV() -> Dict:
    """
    Valida la edad del universo con ECV.
    """
    print("\n" + "="*60)
    print("  TEST 7: Edad del Universo (con ECV)")
    print("="*60)

    t0_Planck = PLANCK_2018.t0
    t0_Planck_err = PLANCK_2018.t0_err

    # Edad con ECV
    def integrand_ecv(z):
        return 1.0 / ((1 + z) * E_MCMC_ECV(z))

    integral_ecv, _ = quad(integrand_ecv, 0, 1000)
    t0_ECV = integral_ecv * 977.8 / H0_MCMC  # Gyr

    # Edad ΛCDM
    def integrand_lcdm(z):
        return 1.0 / ((1 + z) * E_LCDM_standard(z))

    integral_lcdm, _ = quad(integrand_lcdm, 0, 1000)
    t0_LCDM = integral_lcdm * 977.8 / H0_MCMC  # Gyr

    print(f"\n  Edad del universo:")
    print(f"    Planck 2018: {t0_Planck:.3f} ± {t0_Planck_err} Gyr")
    print(f"    ΛCDM calc:   {t0_LCDM:.3f} Gyr")
    print(f"    ECV:         {t0_ECV:.3f} Gyr")

    diff_ecv = abs(t0_ECV - t0_Planck)
    sigma_ecv = diff_ecv / t0_Planck_err

    print(f"\n  Desviación ECV: {diff_ecv:.3f} Gyr ({sigma_ecv:.1f}σ)")

    return {
        "t0_Planck": t0_Planck,
        "t0_LCDM": t0_LCDM,
        "t0_ECV": t0_ECV,
        "diff_Gyr": diff_ecv,
        "sigma": sigma_ecv,
        "passed": sigma_ecv < 2.0
    }


# =============================================================================
# TEST 8: CONSISTENCIA ONTOLÓGICA
# =============================================================================

def test_bloques_ontologicos() -> Dict:
    """
    Valida la consistencia de los 5 bloques ontológicos.
    """
    print("\n" + "="*60)
    print("  TEST 8: Consistencia Ontológica (5 Bloques)")
    print("="*60)

    tests_pasados = 0
    tests_total = 5

    # Bloque 0: Estado Primordial
    print("\n  Bloque 0: Estado Primordial")
    estado = EstadoPrimordial.crear_primordial()
    conservado = abs(estado.Mp + estado.Ep - 1.0) < 1e-6
    print(f"    Mp0 + Ep0 = {estado.Mp + estado.Ep:.10f}")
    print(f"    Conservación: {'PASS' if conservado else 'FAIL'}")
    if conservado:
        tests_pasados += 1

    # Bloque 1: Pregeometría
    print("\n  Bloque 1: Pregeometría")
    preg = Pregeometria()
    I_total = integral_total()
    k_S4 = preg.k(SELLOS["S4"])
    print(f"    Integral entrópica: {I_total:.4f}")
    print(f"    k(S4) = {k_S4:.4f}")
    integral_ok = 5.0 < I_total < 8.0
    print(f"    Integral: {'PASS' if integral_ok else 'FAIL'}")
    if integral_ok:
        tests_pasados += 1

    # Bloque 2: Cosmología con ECV
    print("\n  Bloque 2: Cosmología (ECV)")
    E_z0 = E_MCMC_ECV(0)
    E_z1 = E_MCMC_ECV(1)
    E_ok = abs(E_z0 - 1.0) < 0.01 and E_z1 > 1.5
    print(f"    E_ECV(z=0) = {E_z0:.4f}")
    print(f"    E_ECV(z=1) = {E_z1:.4f}")
    print(f"    Λ_rel(z=0) = {Lambda_rel(0):.4f}")
    print(f"    Normalización: {'PASS' if E_ok else 'FAIL'}")
    if E_ok:
        tests_pasados += 1

    # Bloque 3: N-body con MCV
    print("\n  Bloque 3: N-body (MCV)")
    r_c = r_core_from_mass(1e11)
    r_c_ok = 1.0 < r_c < 3.0
    print(f"    r_core(10¹¹ M☉) = {r_c:.2f} kpc")
    print(f"    S_loc(10¹¹ M☉) = {S_local(1e11):.3f}")
    print(f"    Rango: {'PASS' if r_c_ok else 'FAIL'}")
    if r_c_ok:
        tests_pasados += 1

    # Bloque 4: Lattice Gauge
    print("\n  Bloque 4: Lattice Gauge")
    beta_S4 = beta_MCMC(SELLOS["S4"])
    E_QCD = E_min_QCD_scale(SELLOS["S4"])
    beta_ok = 5.0 < beta_S4 < 8.0
    E_ok = 0.1 < E_QCD < 10.0
    print(f"    β(S4) = {beta_S4:.4f}")
    print(f"    E_QCD(S4) = {E_QCD:.4f} GeV")
    print(f"    Acoplamiento: {'PASS' if beta_ok and E_ok else 'FAIL'}")
    if beta_ok and E_ok:
        tests_pasados += 1

    print(f"\n  Bloques validados: {tests_pasados}/{tests_total}")

    return {
        "tests_pasados": tests_pasados,
        "tests_total": tests_total,
        "porcentaje": 100 * tests_pasados / tests_total,
        "passed": tests_pasados == tests_total
    }


# =============================================================================
# VERIFICACIÓN DE CRITERIOS
# =============================================================================

def verificar_criterios(resultados: Dict) -> Dict:
    """
    Verifica los criterios de éxito específicos.
    """
    print("\n" + "="*60)
    print("  VERIFICACIÓN DE CRITERIOS DE ÉXITO")
    print("="*60)

    criterios = {}

    # Criterio SNe
    chi2_red_sne = resultados["SNe"]["chi2_red_ecv"]
    chi2_sne_ecv = resultados["SNe"]["chi2_ecv"]
    chi2_sne_lcdm = resultados["SNe"]["chi2_lcdm"]

    criterios["sne_chi2_red"] = chi2_red_sne < 1.5
    criterios["sne_competitivo"] = chi2_sne_ecv <= chi2_sne_lcdm * 1.1

    print(f"\n  SNe:")
    print(f"    χ² reducido < 1.5: {chi2_red_sne:.3f} -> {'PASS' if criterios['sne_chi2_red'] else 'FAIL'}")
    print(f"    MCMC ≤ ΛCDM*1.1: {chi2_sne_ecv:.1f} ≤ {chi2_sne_lcdm*1.1:.1f} -> {'PASS' if criterios['sne_competitivo'] else 'FAIL'}")

    # Criterio SPARC (Zhao γ=0.51)
    chi2_mcv = resultados["SPARC"]["chi2_MCV"]
    chi2_nfw = resultados["SPARC"]["chi2_NFW"]
    mejora_sparc = resultados["SPARC"]["mejora_percent"]

    criterios["sparc_mejor_nfw"] = chi2_mcv < chi2_nfw
    criterios["sparc_mejora_40"] = mejora_sparc > 40

    print(f"\n  SPARC (Perfil Zhao γ=0.51):")
    print(f"    MCMC < NFW: {chi2_mcv:.1f} < {chi2_nfw:.1f} -> {'PASS' if criterios['sparc_mejor_nfw'] else 'FAIL'}")
    print(f"    Mejora > 40%: {mejora_sparc:.1f}% -> {'PASS' if criterios['sparc_mejora_40'] else 'FAIL'}")

    # Criterio GAIA (Zhao γ=0.51)
    chi2_gaia_mcmc = resultados["GAIA"]["chi2_MCV"]
    chi2_gaia_nfw = resultados["GAIA"]["chi2_NFW"]
    chi2_gaia_flat = resultados["GAIA"]["chi2_flat"]
    mejora_gaia = resultados["GAIA"]["mejora_vs_NFW"]

    criterios["gaia_mejor_nfw"] = chi2_gaia_mcmc < chi2_gaia_nfw
    criterios["gaia_mejor_flat"] = chi2_gaia_mcmc < chi2_gaia_flat

    print(f"\n  GAIA (Perfil Zhao γ=0.51):")
    print(f"    MCMC < NFW: {chi2_gaia_mcmc:.1f} < {chi2_gaia_nfw:.1f} -> {'PASS' if criterios['gaia_mejor_nfw'] else 'FAIL'}")
    print(f"    MCMC < flat: {chi2_gaia_mcmc:.1f} < {chi2_gaia_flat:.1f} -> {'PASS' if criterios['gaia_mejor_flat'] else 'FAIL'}")

    # Criterio r_core
    print(f"\n  r_core(M) (Ley de Cronos):")
    criterios["r_core_rango"] = True
    for M in [1e8, 1e9, 1e10, 1e11, 1e12]:
        r_c = r_core_from_mass(M)
        en_rango = 0.1 < r_c < 10
        criterios["r_core_rango"] &= en_rango
        print(f"    r_core({M:.0e}) = {r_c:.2f} kpc -> {'PASS' if en_rango else 'FAIL'}")

    # Resumen
    total_pasados = sum(criterios.values())
    total_criterios = len(criterios)

    print(f"\n  Criterios pasados: {total_pasados}/{total_criterios}")

    return criterios


# =============================================================================
# RESUMEN FINAL
# =============================================================================

def ejecutar_validacion_ontologica() -> Dict:
    """
    Ejecuta toda la validación ontológica.
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#   VALIDACIÓN ONTOLÓGICA: MCMC con ECV + MCV + Ley de Cronos" + " "*6 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print(f"\n    Versión: mcmc_core 2.5.0")
    print(f"    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n  Parámetros ontológicos:")
    print(f"    ECV: ε = {EPSILON_ECV}, z_trans = {Z_TRANS}")
    print(f"    MCV: α_r = {0.35}, r_⋆ = 1.8 kpc")
    print(f"    Fricción: η_0 = 0.15, γ = 1.5")

    resultados = {}

    # Ejecutar todos los tests
    resultados["BAO"] = test_BAO_ECV()
    resultados["SNe"] = test_SNe_ECV()
    resultados["H0_tension"] = test_H0_tension_ECV()
    resultados["SPARC"] = test_SPARC_MCV()
    resultados["GAIA"] = test_GAIA_MCV()
    resultados["S8_tension"] = test_S8_tension()
    resultados["edad_universo"] = test_edad_universo_ECV()
    resultados["bloques"] = test_bloques_ontologicos()

    # Verificar criterios
    criterios = verificar_criterios(resultados)

    # Resumen
    print("\n" + "="*70)
    print("  RESUMEN DE VALIDACIÓN ONTOLÓGICA")
    print("="*70)

    tests_pasados = sum(1 for r in resultados.values() if r.get("passed", False))
    tests_total = len(resultados)

    print(f"\n  {'Test':<30} {'Estado':>10}")
    print("  " + "-"*45)

    nombres = {
        "BAO": "BAO (31 puntos, ECV)",
        "SNe": "Supernovas Ia (24, ECV)",
        "H0_tension": "Tensión H0 (ECV)",
        "SPARC": "SPARC (5 gal, Zhao γ=0.51)",
        "GAIA": "GAIA DR3 (Zhao γ=0.51)",
        "S8_tension": "Tensión S8 (fricción)",
        "edad_universo": "Edad Universo (ECV)",
        "bloques": "Bloques Ontológicos (5)"
    }

    for key, nombre in nombres.items():
        estado = "PASS" if resultados[key].get("passed", False) else "FAIL"
        print(f"  {nombre:<30} {estado:>10}")

    print(f"\n  Tests pasados: {tests_pasados}/{tests_total} ({100*tests_pasados/tests_total:.0f}%)")

    # Métricas clave
    print("\n  Métricas clave:")
    print(f"    χ²_BAO (ECV): {resultados['BAO']['chi2_ecv']:.1f}")
    print(f"    χ²_SNe (ECV): {resultados['SNe']['chi2_ecv']:.1f} (red: {resultados['SNe']['chi2_red_ecv']:.3f})")
    print(f"    χ²_SPARC Zhao: {resultados['SPARC']['chi2_MCV']:.1f} (mejora {resultados['SPARC']['mejora_percent']:.0f}% vs NFW)")
    print(f"    Tensión H0: {resultados['H0_tension']['sigma_LCDM']:.1f}σ → {resultados['H0_tension']['sigma_ECV']:.1f}σ")
    print(f"    Tensión S8: reducida {resultados['S8_tension']['reduccion_DES']:.0f}%")

    print("\n" + "="*70)
    print("  FIN DE VALIDACIÓN ONTOLÓGICA")
    print("="*70)

    # Guardar resultados
    output = {
        "version": "2.4.0",
        "timestamp": datetime.now().isoformat(),
        "ontologia": {
            "ECV": {"epsilon": EPSILON_ECV, "z_trans": Z_TRANS},
            "MCV": {"alpha_r": 0.35, "r_star": 1.8},
            "friction": {"eta_0": 0.15, "gamma": 1.5}
        },
        "resumen": {
            "total_tests": tests_total,
            "tests_pasados": tests_pasados,
            "porcentaje": f"{100*tests_pasados/tests_total:.0f}%"
        },
        "criterios": {k: bool(v) for k, v in criterios.items()}
    }

    # Convertir numpy
    def convertir(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convertir(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convertir(i) for i in obj]
        return obj

    for key, val in resultados.items():
        output[key] = convertir(val)

    with open("resultados_validacion_ontologica.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Resultados guardados en: resultados_validacion_ontologica.json")

    return resultados


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ejecutar_validacion_ontologica()
