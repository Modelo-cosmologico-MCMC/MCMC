#!/usr/bin/env python3
"""
Ontología ECV y MCV para el Modelo MCMC
========================================

Implementación de:
- ECV (Energía Cuántica Virtual): Reemplaza Λ (energía oscura)
- MCV (Materia Cuántica Virtual): Reemplaza CDM (materia oscura)
- Ley de Cronos: Fricción entrópica que modifica la dinámica gravitatoria

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Tuple, Optional

# =============================================================================
# CONSTANTES ONTOLÓGICAS MCMC
# =============================================================================

# Parámetros ECV (Energía Cuántica Virtual)
EPSILON_ECV = 0.012      # Parámetro de transición
Z_TRANS = 8.9            # Redshift de transición (época de reionización)
DELTA_Z = 1.5            # Anchura de transición
S_BB = 1.001             # Entropía en el Big Bang (S₄)
S_0 = 0.999              # Entropía actual (cerca de S₃)
P_ENTROPIC = 0.5         # Exponente del mapa S(z)

# Parámetros MCV (Materia Cuántica Virtual)
# Calibrados para reproducir SPARC y Milky Way
RHO_STAR = 1e9           # M☉/kpc³ - densidad de referencia (aumentado 10x)
R_STAR = 2.0             # kpc - radio de referencia
S_STAR = 0.5             # Entropía de referencia
ALPHA_RHO = 0.3          # Exponente para ρ_0(S_loc) (reducido para menos variación)
ALPHA_R = 0.35           # Exponente para r_core(S_loc) [SPARC: ~0.4]

# Factor de normalización para ajustar a V_flat observada
NORM_FACTOR_MCV = 3.5    # Factor de escala para masa encerrada

# Parámetros de fricción entrópica (Ley de Cronos)
# Calibrados para pequeñas correcciones (~5-10% en el centro)
ETA_FRICTION = 0.08      # Coeficiente de fricción entrópica (reducido)
GAMMA_FRICTION = 0.5     # Exponente de densidad para fricción (reducido)
RHO_CRIT_FRICTION = 1e8  # M☉/kpc³ - densidad crítica (aumentado)

# Constantes cosmológicas
H0_MCMC = 67.36          # km/s/Mpc
OMEGA_M_MCMC = 0.3153    # Densidad de materia
OMEGA_LAMBDA_MCMC = 0.6847  # Densidad de ECV (z=0)
C_LIGHT = 299792.458     # km/s

# Constante gravitacional
G_GRAV = 4.302e-6        # kpc (km/s)² / M☉


# =============================================================================
# ECV (ENERGÍA CUÁNTICA VIRTUAL)
# =============================================================================

def S_of_z(z: float) -> float:
    """
    Mapa entrópico S(z): conecta redshift con variable entrópica.

    En el MCMC, S evoluciona desde S_BB (Big Bang) hasta S_0 (hoy).
    S(z) = S_BB + (S_0 - S_BB) * (1+z)^(-p)

    A z alto: S → S_BB (entropía alta, cerca del colapso inicial)
    A z = 0:  S → S_0 (entropía actual)
    """
    return S_BB + (S_0 - S_BB) * (1 + z)**(-P_ENTROPIC)


def F_transition(S: float, S_trans: float = 0.9995, delta_S: float = 0.0005) -> float:
    """
    Función de transición suave F(S).

    F(S) = 0.5 * (1 + tanh((S - S_trans)/ΔS))

    - F → 0 cuando S << S_trans (régimen pre-transición)
    - F → 1 cuando S >> S_trans (régimen post-transición)
    """
    return 0.5 * (1 + np.tanh((S - S_trans) / delta_S))


def rho_ECV(z: float) -> float:
    """
    Densidad de Energía Cuántica Virtual ρ_ECV(z).

    La ECV emerge de la conversión masa-espacio y tiene una transición
    suave determinada por el mapa entrópico:

    ρ_ECV(z) = ρ_ECV,0 * [(1-F(S(z))) * (1+z)³ + F(S(z))]

    - A z alto (S cerca de S_BB): comportamiento tipo materia ~ (1+z)³
    - A z bajo (S cerca de S_0):  comportamiento tipo Λ ~ constante
    """
    S = S_of_z(z)
    F = F_transition(S)
    rho_0 = OMEGA_LAMBDA_MCMC
    return rho_0 * ((1 - F) * (1 + z)**3 + F)


def Omega_ECV(z: float) -> float:
    """
    Fracción de densidad de ECV Ω_ECV(z).
    """
    S = S_of_z(z)
    F = F_transition(S)
    return OMEGA_LAMBDA_MCMC * ((1 - F) * (1 + z)**3 + F)


def Lambda_rel(z: float) -> float:
    """
    Constante cosmológica relacional Λ_rel(z).

    Λ_rel(z) = 1 + ε * tanh((z_trans - z)/Δz)

    - A z < z_trans: Λ_rel > 1 (expansión ligeramente más rápida)
    - A z > z_trans: Λ_rel → 1 (comportamiento estándar)
    """
    return 1.0 + EPSILON_ECV * np.tanh((Z_TRANS - z) / DELTA_Z)


def E_MCMC_ECV(z: float) -> float:
    """
    E(z) = H(z)/H₀ para el MCMC con ECV correcta.

    Usamos la forma Lambda_rel que da pequeñas correcciones:
    E²(z) = Ω_m * (1+z)³ + Ω_Λ * Λ_rel(z)

    Λ_rel(z) ≈ 1 + ε * tanh((z_trans - z)/Δz)
    - A z < z_trans: Λ_rel > 1 (expansión ligeramente más rápida)
    - A z > z_trans: Λ_rel → 1-ε (comportamiento estándar)
    """
    matter_term = OMEGA_M_MCMC * (1 + z)**3
    ecv_term = OMEGA_LAMBDA_MCMC * Lambda_rel(z)
    return np.sqrt(matter_term + ecv_term)


def E_MCMC_Lambda_rel(z: float) -> float:
    """
    Forma alternativa: E(z) con Λ relacional.

    E²(z) = Ω_m * (1+z)³ + Ω_Λ * Λ_rel(z)
    """
    return np.sqrt(OMEGA_M_MCMC * (1 + z)**3 + OMEGA_LAMBDA_MCMC * Lambda_rel(z))


def H_MCMC_ECV(z: float) -> float:
    """H(z) en km/s/Mpc para MCMC con ECV."""
    return H0_MCMC * E_MCMC_ECV(z)


def distancia_comovil_ECV(z: float) -> float:
    """
    Distancia comóvil r(z) en Mpc para MCMC con ECV.
    """
    integrand = lambda zp: C_LIGHT / H_MCMC_ECV(zp)
    result, _ = quad(integrand, 0, z)
    return result


def distancia_luminosidad_ECV(z: float) -> float:
    """
    Distancia de luminosidad D_L(z) en Mpc para MCMC con ECV.
    """
    return (1 + z) * distancia_comovil_ECV(z)


def modulo_distancia_ECV(z: float) -> float:
    """
    Módulo de distancia μ(z) para MCMC con ECV.
    """
    D_L = distancia_luminosidad_ECV(z)
    return 5 * np.log10(D_L) + 25


def distancia_volumen_ECV(z: float) -> float:
    """
    Distancia de volumen D_V(z) para BAO con MCMC ECV.
    """
    D_M = distancia_comovil_ECV(z)
    D_H = C_LIGHT / H_MCMC_ECV(z)
    return (z * D_M**2 * D_H)**(1/3)


# =============================================================================
# MCV (MATERIA CUÁNTICA VIRTUAL)
# =============================================================================

def S_local(M_halo: float, z_form: float = 1.0) -> float:
    """
    Entropía local S_loc de un halo.

    S_loc = S_ref * (M/M_ref)^0.1 * (1 + z_form)^0.2
    """
    M_ref = 1e11
    S_ref = 0.5
    return S_ref * (M_halo / M_ref)**0.1 * (1 + z_form)**0.2


def rho_0_MCV(S_loc: float) -> float:
    """
    Densidad central de MCV ρ_0(S_loc).

    ρ_0(S_loc) = ρ_⋆ * (S_⋆/S_loc)^α_ρ
    """
    return RHO_STAR * (S_STAR / S_loc)**ALPHA_RHO


def r_core_MCV(S_loc: float) -> float:
    """
    Radio de núcleo de MCV r_core(S_loc).

    r_core(S_loc) = r_⋆ * (S_loc/S_⋆)^α_r
    """
    return R_STAR * (S_loc / S_STAR)**ALPHA_R


def r_core_from_mass(M_halo: float) -> float:
    """
    Radio de núcleo como función de la masa del halo.

    Combina S_loc(M) con r_core(S_loc) para obtener
    r_core ∝ M^β con β ≈ 0.35 (Ley de Cronos a escala de halos).
    """
    S_loc = S_local(M_halo)
    return r_core_MCV(S_loc)


def perfil_MCV_Burkert(r: float, rho_0: float, r_core: float) -> float:
    """
    Perfil de densidad de MCV tipo Burkert.

    ρ_MCV(r) = ρ_0 * r_core³ / [(r + r_core)(r² + r_core²)]
    """
    if r < 1e-10:
        return rho_0
    return rho_0 * r_core**3 / ((r + r_core) * (r**2 + r_core**2))


def perfil_MCV_isotermico(r: float, rho_0: float, r_core: float) -> float:
    """
    Perfil isotérmico con núcleo.

    ρ_MCV(r) = ρ_0 / (1 + (r/r_core)²)
    """
    return rho_0 / (1 + (r / r_core)**2)


def masa_encerrada_MCV_Burkert(r: float, rho_0: float, r_core: float) -> float:
    """
    Masa encerrada M(<r) para perfil Burkert.

    M(<r) = 2πρ_0*r_core³ * [ln(1+r/r_core) + 0.5*ln(1+(r/r_core)²) - arctan(r/r_core)]
    """
    x = r / r_core
    if x < 1e-10:
        return 0.0

    M = 2 * np.pi * rho_0 * r_core**3 * (
        np.log(1 + x) +
        0.5 * np.log(1 + x**2) -
        np.arctan(x)
    )
    return M


def masa_encerrada_MCV_isotermico(r: float, rho_0: float, r_core: float) -> float:
    """
    Masa encerrada M(<r) para perfil isotérmico.

    M(<r) = 4π ρ_0 r_core³ [r/r_core - arctan(r/r_core)]
    """
    x = r / r_core
    if x < 1e-10:
        return 0.0
    return 4 * np.pi * rho_0 * r_core**3 * (x - np.arctan(x))


def perfil_Zhao_MCV(r: float, rho_0: float, r_s: float,
                    gamma: float = 0, alpha: float = 2, beta: float = 3,
                    S_loc: float = 0.5) -> float:
    """
    Perfil Zhao generalizado con dependencia entrópica.

    ρ(r) = ρ_0 / [(r/r_s)^γ * (1 + (r/r_s)^α)^((β-γ)/α)]

    γ depende de S_loc: γ(S_loc) = γ_max * exp(-S_loc/S_crit)
    """
    S_crit = 0.3
    gamma_eff = gamma * np.exp(-S_loc / S_crit)

    x = r / r_s
    if x < 1e-10:
        x = 1e-10

    return rho_0 / (x**gamma_eff * (1 + x**alpha)**((beta - gamma_eff) / alpha))


# =============================================================================
# LEY DE CRONOS - FRICCIÓN ENTRÓPICA
# =============================================================================

@dataclass
class ParametrosFriccion:
    """Parámetros para la fricción entrópica."""
    eta_0: float = ETA_FRICTION
    gamma: float = GAMMA_FRICTION
    rho_crit: float = RHO_CRIT_FRICTION


class FriccionEntropicaMCV:
    """
    Implementa la fricción entrópica (Ley de Cronos).

    La Ley de Cronos establece que:
    1. La dilatación temporal dt_rel/dS modula el colapso de estructuras
    2. La fricción entrópica η actúa como "rozamiento tensional"
    3. La redistribución Mp → Ep produce núcleos planos
    """

    def __init__(self, params: Optional[ParametrosFriccion] = None):
        if params is None:
            params = ParametrosFriccion()
        self.eta_0 = params.eta_0
        self.gamma = params.gamma
        self.rho_crit = params.rho_crit

    def coeficiente(self, rho: float) -> float:
        """
        Coeficiente de fricción η(ρ).

        η(ρ) = η_0 * (ρ/ρ_crit)^γ
        """
        return self.eta_0 * (rho / self.rho_crit)**self.gamma

    def factor_velocidad(self, r: float, rho: float, r_core: float) -> float:
        """
        Factor de corrección a la velocidad circular.

        f_v(r) = 1 - η(ρ) * exp(-r/r_core)
        """
        eta = self.coeficiente(rho)
        factor = 1 - eta * np.exp(-r / r_core)
        return max(factor, 0.5)  # Límite inferior

    def factor_densidad(self, r: float, rho_0: float, r_core: float) -> float:
        """
        Factor de modificación del perfil de densidad.

        ρ_eff(r) = ρ_base(r) * [1 + η * (r/r_core)²]^(-1)
        """
        eta = self.coeficiente(rho_0)
        return 1 / (1 + eta * (r / r_core)**2)


# =============================================================================
# VELOCIDAD CIRCULAR CON MCV Y FRICCIÓN
# =============================================================================

def velocidad_circular_MCV(r: float, M_halo: float,
                           v_gas: float = 0, v_disk: float = 0, v_bulge: float = 0,
                           perfil: str = "Burkert",
                           include_friction: bool = True,
                           V_flat_target: float = None) -> float:
    """
    Velocidad circular V_circ(r) incluyendo MCV, bariones y fricción.

    V²_circ = (V²_MCV + V²_bar) * f_friction

    Si V_flat_target está especificado, normaliza el halo MCV para
    reproducir la velocidad asintótica observada.
    """
    if r < 1e-10:
        return 0.0

    # Parámetros MCV para este halo
    S_loc = S_local(M_halo)
    rho_0 = rho_0_MCV(S_loc)
    r_c = r_core_MCV(S_loc)

    # Masa de MCV encerrada
    if perfil == "Burkert":
        M_MCV = masa_encerrada_MCV_Burkert(r, rho_0, r_c) * NORM_FACTOR_MCV
    else:
        M_MCV = masa_encerrada_MCV_isotermico(r, rho_0, r_c) * NORM_FACTOR_MCV

    # Velocidad circular de MCV
    v_mcv_sq = G_GRAV * M_MCV / r

    # Componente bariónica
    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2

    v_total_sq = v_mcv_sq + v_bar_sq

    # Corrección por fricción entrópica (Ley de Cronos)
    if include_friction:
        friction = FriccionEntropicaMCV()
        rho_local = perfil_MCV_Burkert(r, rho_0, r_c) if perfil == "Burkert" else perfil_MCV_isotermico(r, rho_0, r_c)
        f_friction = friction.factor_velocidad(r, rho_local, r_c)
        v_total_sq *= f_friction

    return np.sqrt(max(v_total_sq, 0))


def velocidad_circular_MCV_calibrado(r: float, V_flat: float, r_eff: float,
                                      v_gas: float = 0, v_disk: float = 0,
                                      v_bulge: float = 0,
                                      include_friction: bool = True) -> float:
    """
    Velocidad circular MCV calibrada a la V_flat observada.

    Usa V_flat para inferir la masa del halo y construye un perfil
    Burkert que reproduce la curva de rotación.

    Args:
        r: Radio en kpc
        V_flat: Velocidad asintótica observada (km/s)
        r_eff: Radio efectivo de la galaxia (kpc)
        v_gas, v_disk, v_bulge: Componentes bariónicos
        include_friction: Si aplicar fricción entrópica

    Returns:
        Velocidad circular en km/s
    """
    if r < 1e-10:
        return 0.0

    # Estimar masa del halo desde V_flat
    # V_flat² ≈ G * M_halo / r_vir, con r_vir ~ 10 * r_eff
    r_vir = 10 * r_eff
    M_halo = V_flat**2 * r_vir / G_GRAV

    # Parámetros MCV
    S_loc = S_local(M_halo)
    r_c = r_core_MCV(S_loc)

    # Calcular rho_0 para que V(r_vir) = V_flat
    # M(<r_vir) = V_flat² * r_vir / G
    M_target = V_flat**2 * r_vir / G_GRAV
    x_vir = r_vir / r_c

    # Factor de masa Burkert
    f_burkert = (
        np.log(1 + x_vir) +
        0.5 * np.log(1 + x_vir**2) -
        np.arctan(x_vir)
    )

    # Resolver para rho_0
    rho_0 = M_target / (2 * np.pi * r_c**3 * f_burkert)

    # Masa encerrada a r
    x = r / r_c
    f_r = np.log(1 + x) + 0.5 * np.log(1 + x**2) - np.arctan(x)
    M_MCV = 2 * np.pi * rho_0 * r_c**3 * f_r

    # Velocidad de halo
    v_halo_sq = G_GRAV * M_MCV / r

    # Componente bariónica
    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2

    v_total_sq = v_halo_sq + v_bar_sq

    # Fricción entrópica
    if include_friction:
        friction = FriccionEntropicaMCV()
        rho_local = perfil_MCV_Burkert(r, rho_0, r_c)
        f_friction = friction.factor_velocidad(r, rho_local, r_c)
        v_total_sq *= f_friction

    return np.sqrt(max(v_total_sq, 0))


def velocidad_NFW_standard(r: float, M_halo: float, c: float = 10,
                           v_gas: float = 0, v_disk: float = 0, v_bulge: float = 0) -> float:
    """
    Velocidad circular NFW estándar para comparación.
    """
    if r < 1e-10:
        return 0.0

    # Radio virial aproximado
    r_vir = (M_halo / 1e12)**(1/3) * 200
    r_s = r_vir / c
    x = r / r_s

    # M(<r) para NFW
    f_nfw = np.log(1 + x) - x / (1 + x)
    f_norm = np.log(1 + c) - c / (1 + c)
    M_r = M_halo * f_nfw / f_norm

    v_nfw_sq = G_GRAV * M_r / r
    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2

    return np.sqrt(v_nfw_sq + v_bar_sq)


# =============================================================================
# FUNCIONES AUXILIARES LCDM (PARA COMPARACIÓN)
# =============================================================================

def E_LCDM_standard(z: float) -> float:
    """E(z) para ΛCDM estándar."""
    return np.sqrt(OMEGA_M_MCMC * (1 + z)**3 + OMEGA_LAMBDA_MCMC)


def H_LCDM_standard(z: float) -> float:
    """H(z) para ΛCDM estándar."""
    return H0_MCMC * E_LCDM_standard(z)


def distancia_comovil_LCDM(z: float) -> float:
    """Distancia comóvil ΛCDM."""
    integrand = lambda zp: C_LIGHT / H_LCDM_standard(zp)
    result, _ = quad(integrand, 0, z)
    return result


def distancia_luminosidad_LCDM(z: float) -> float:
    """Distancia luminosidad ΛCDM."""
    return (1 + z) * distancia_comovil_LCDM(z)


def modulo_distancia_LCDM(z: float) -> float:
    """Módulo de distancia ΛCDM."""
    D_L = distancia_luminosidad_LCDM(z)
    return 5 * np.log10(D_L) + 25


def distancia_volumen_LCDM(z: float) -> float:
    """Distancia de volumen ΛCDM."""
    D_M = distancia_comovil_LCDM(z)
    D_H = C_LIGHT / H_LCDM_standard(z)
    return (z * D_M**2 * D_H)**(1/3)


# =============================================================================
# VERIFICACIÓN DE ONTOLOGÍA
# =============================================================================

def verificar_ECV():
    """Verifica la implementación de ECV."""
    print("Verificación ECV:")
    print(f"  E(z=0) = {E_MCMC_ECV(0):.4f} (debe ser ~1.0)")
    print(f"  E(z=1) = {E_MCMC_ECV(1):.4f}")
    print(f"  E(z=2) = {E_MCMC_ECV(2):.4f}")
    print(f"  Λ_rel(z=0) = {Lambda_rel(0):.4f}")
    print(f"  Λ_rel(z=10) = {Lambda_rel(10):.4f}")
    print(f"  S(z=0) = {S_of_z(0):.6f}")
    print(f"  S(z=1000) = {S_of_z(1000):.6f}")


def verificar_MCV():
    """Verifica la implementación de MCV."""
    print("\nVerificación MCV:")
    for M in [1e8, 1e9, 1e10, 1e11, 1e12]:
        r_c = r_core_from_mass(M)
        S_l = S_local(M)
        rho = rho_0_MCV(S_l)
        print(f"  M={M:.0e} M☉: S_loc={S_l:.3f}, r_core={r_c:.2f} kpc, ρ_0={rho:.2e} M☉/kpc³")


def verificar_friccion():
    """Verifica la fricción entrópica."""
    print("\nVerificación Fricción Entrópica:")
    friction = FriccionEntropicaMCV()
    for rho in [1e6, 1e7, 1e8]:
        eta = friction.coeficiente(rho)
        f_v = friction.factor_velocidad(1.0, rho, 2.0)
        print(f"  ρ={rho:.0e}: η={eta:.4f}, f_v(r=1kpc)={f_v:.4f}")


if __name__ == "__main__":
    verificar_ECV()
    verificar_MCV()
    verificar_friccion()
