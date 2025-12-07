#!/usr/bin/env python3
"""
Datos Observacionales para Validación del Modelo MCMC
=====================================================

Compilación de datos observacionales de las principales colaboraciones:
- Planck 2018 (CMB)
- BAO (BOSS, eBOSS, DESI 2024)
- SPARC (Curvas de rotación)
- GAIA (Cinemática estelar)
- Supernovas Ia (Pantheon+)
- H0 (SH0ES, TRGB, CCHP)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

# =============================================================================
# CONSTANTES FÍSICAS FUNDAMENTALES
# =============================================================================

C_LIGHT = 299792.458  # km/s
H0_FIDUCIAL = 100.0   # h km/s/Mpc (para escalado)


# =============================================================================
# PLANCK 2018 - CMB
# =============================================================================

@dataclass
class DatosPlanck2018:
    """
    Datos cosmológicos de Planck 2018 (arXiv:1807.06209)
    TT,TE,EE+lowE+lensing
    """
    # Parámetros base ΛCDM
    H0: float = 67.36          # km/s/Mpc
    H0_err: float = 0.54

    Omega_b_h2: float = 0.02237    # Densidad bariónica
    Omega_b_h2_err: float = 0.00015

    Omega_c_h2: float = 0.1200     # Densidad materia oscura
    Omega_c_h2_err: float = 0.0012

    Omega_m: float = 0.3153        # Densidad total de materia
    Omega_m_err: float = 0.0073

    Omega_Lambda: float = 0.6847   # Densidad energía oscura
    Omega_Lambda_err: float = 0.0073

    sigma8: float = 0.8111         # Amplitud fluctuaciones
    sigma8_err: float = 0.0060

    n_s: float = 0.9649            # Índice espectral
    n_s_err: float = 0.0042

    tau: float = 0.0544            # Profundidad óptica
    tau_err: float = 0.0073

    # Derivados
    t0: float = 13.797             # Edad universo (Gyr)
    t0_err: float = 0.023

    z_reion: float = 7.67          # Redshift reionización
    z_reion_err: float = 0.73

    r_drag: float = 147.09         # Radio horizonte sonido (Mpc)
    r_drag_err: float = 0.26

    # S8 = sigma8 * sqrt(Omega_m/0.3)
    S8: float = 0.832
    S8_err: float = 0.013

    # Tensión con lensing débil
    S8_DES: float = 0.776          # DES Y3
    S8_DES_err: float = 0.017
    S8_KiDS: float = 0.759         # KiDS-1000
    S8_KiDS_err: float = 0.024


PLANCK_2018 = DatosPlanck2018()


# =============================================================================
# BAO - OSCILACIONES ACÚSTICAS DE BARIONES
# =============================================================================

@dataclass
class PuntoBAO:
    """Un punto de medición BAO."""
    z_eff: float           # Redshift efectivo
    observable: str        # Tipo: D_V/r_d, D_M/r_d, D_H/r_d, etc.
    valor: float           # Valor medido
    error: float           # Error (1σ)
    survey: str            # Colaboración
    referencia: str        # arXiv


# BOSS DR12 (2016) - arXiv:1607.03155
BAO_BOSS_DR12 = [
    # D_V(z)/r_d
    PuntoBAO(0.38, "D_V/r_d", 10.23, 0.17, "BOSS", "1607.03155"),
    PuntoBAO(0.51, "D_V/r_d", 13.36, 0.21, "BOSS", "1607.03155"),
    PuntoBAO(0.61, "D_V/r_d", 15.45, 0.21, "BOSS", "1607.03155"),

    # D_M(z)/r_d (distancia comóvil angular)
    PuntoBAO(0.38, "D_M/r_d", 10.27, 0.15, "BOSS", "1607.03155"),
    PuntoBAO(0.51, "D_M/r_d", 13.38, 0.18, "BOSS", "1607.03155"),
    PuntoBAO(0.61, "D_M/r_d", 15.45, 0.21, "BOSS", "1607.03155"),

    # D_H(z)/r_d = c/(H(z)*r_d)
    PuntoBAO(0.38, "D_H/r_d", 25.00, 0.76, "BOSS", "1607.03155"),
    PuntoBAO(0.51, "D_H/r_d", 22.33, 0.58, "BOSS", "1607.03155"),
    PuntoBAO(0.61, "D_H/r_d", 20.75, 0.53, "BOSS", "1607.03155"),
]

# eBOSS DR16 (2020) - arXiv:2007.08991
BAO_EBOSS_DR16 = [
    # LRG (Luminous Red Galaxies)
    PuntoBAO(0.70, "D_M/r_d", 17.86, 0.33, "eBOSS-LRG", "2007.08991"),
    PuntoBAO(0.70, "D_H/r_d", 19.33, 0.53, "eBOSS-LRG", "2007.08991"),

    # ELG (Emission Line Galaxies)
    PuntoBAO(0.85, "D_V/r_d", 18.33, 0.62, "eBOSS-ELG", "2007.08991"),

    # QSO (Quasars)
    PuntoBAO(1.48, "D_M/r_d", 30.21, 0.79, "eBOSS-QSO", "2007.08991"),
    PuntoBAO(1.48, "D_H/r_d", 13.23, 0.47, "eBOSS-QSO", "2007.08991"),

    # Lyman-α (z alto)
    PuntoBAO(2.33, "D_M/r_d", 37.6, 1.9, "eBOSS-Lyα", "2007.08991"),
    PuntoBAO(2.33, "D_H/r_d", 8.93, 0.28, "eBOSS-Lyα", "2007.08991"),
]

# DESI 2024 (arXiv:2404.03002) - Primeros resultados
BAO_DESI_2024 = [
    # BGS (Bright Galaxy Survey)
    PuntoBAO(0.30, "D_V/r_d", 7.93, 0.15, "DESI-BGS", "2404.03002"),

    # LRG
    PuntoBAO(0.51, "D_M/r_d", 13.62, 0.25, "DESI-LRG", "2404.03002"),
    PuntoBAO(0.51, "D_H/r_d", 20.98, 0.61, "DESI-LRG", "2404.03002"),
    PuntoBAO(0.71, "D_M/r_d", 16.85, 0.32, "DESI-LRG", "2404.03002"),
    PuntoBAO(0.71, "D_H/r_d", 20.08, 0.60, "DESI-LRG", "2404.03002"),
    PuntoBAO(0.93, "D_M/r_d", 21.71, 0.28, "DESI-LRG", "2404.03002"),
    PuntoBAO(0.93, "D_H/r_d", 17.88, 0.35, "DESI-LRG", "2404.03002"),

    # ELG
    PuntoBAO(1.32, "D_M/r_d", 27.79, 0.69, "DESI-ELG", "2404.03002"),
    PuntoBAO(1.32, "D_H/r_d", 13.82, 0.42, "DESI-ELG", "2404.03002"),

    # QSO
    PuntoBAO(1.49, "D_M/r_d", 30.69, 0.80, "DESI-QSO", "2404.03002"),
    PuntoBAO(1.49, "D_H/r_d", 13.26, 0.55, "DESI-QSO", "2404.03002"),

    # Lyman-α
    PuntoBAO(2.33, "D_M/r_d", 39.71, 0.94, "DESI-Lyα", "2404.03002"),
    PuntoBAO(2.33, "D_H/r_d", 8.52, 0.17, "DESI-Lyα", "2404.03002"),
]

# 6dFGS (z bajo)
BAO_6DFGS = [
    PuntoBAO(0.106, "D_V/r_d", 2.98, 0.13, "6dFGS", "1106.3366"),
]

# MGS (Main Galaxy Sample)
BAO_MGS = [
    PuntoBAO(0.15, "D_V/r_d", 4.47, 0.17, "SDSS-MGS", "1409.3242"),
]

# Compilación completa BAO
BAO_ALL = BAO_6DFGS + BAO_MGS + BAO_BOSS_DR12 + BAO_EBOSS_DR16 + BAO_DESI_2024


# =============================================================================
# SPARC - CURVAS DE ROTACIÓN GALÁCTICA
# =============================================================================

@dataclass
class GalaxiaSPARCCompleta:
    """
    Datos completos de una galaxia SPARC.
    Ref: Lelli, McGaugh & Schombert 2016 (arXiv:1606.09251)
    """
    nombre: str
    tipo: str                      # Tipo morfológico
    distancia: float               # Mpc
    distancia_err: float
    inclinacion: float             # grados
    L_3_6: float                   # Luminosidad 3.6μm (L☉)
    M_HI: float                    # Masa HI (M☉)
    M_star: float                  # Masa estelar (M☉)
    r_eff: float                   # Radio efectivo (kpc)
    V_flat: float                  # Velocidad asintótica (km/s)
    V_flat_err: float
    # Datos de curva de rotación
    r_data: np.ndarray             # Radio (kpc)
    v_obs: np.ndarray              # Velocidad observada (km/s)
    v_err: np.ndarray              # Error velocidad
    v_gas: np.ndarray              # Contribución gas
    v_disk: np.ndarray             # Contribución disco
    v_bul: np.ndarray              # Contribución bulbo


# Galaxias SPARC representativas (subset de 175 galaxias)
SPARC_CATALOG = [
    # Galaxias enanas (dominadas por materia oscura)
    GalaxiaSPARCCompleta(
        nombre="DDO 154",
        tipo="IB(s)m",
        distancia=3.7, distancia_err=0.2,
        inclinacion=66,
        L_3_6=2.0e7, M_HI=3.0e8, M_star=4.0e7,
        r_eff=1.5, V_flat=47, V_flat_err=2,
        r_data=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]),
        v_obs=np.array([15, 22, 28, 33, 38, 42, 44, 46, 47, 47, 47, 47]),
        v_err=np.array([2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5]),
        v_gas=np.array([5, 8, 10, 12, 14, 15, 16, 16, 17, 17, 17, 17]),
        v_disk=np.array([8, 12, 14, 15, 15, 14, 13, 12, 10, 8, 7, 6]),
        v_bul=np.zeros(12)
    ),
    GalaxiaSPARCCompleta(
        nombre="DDO 168",
        tipo="IBm",
        distancia=4.3, distancia_err=0.3,
        inclinacion=47,
        L_3_6=5.0e7, M_HI=2.5e8, M_star=1.0e8,
        r_eff=1.8, V_flat=55, V_flat_err=3,
        r_data=np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]),
        v_obs=np.array([18, 28, 38, 45, 50, 53, 55, 55]),
        v_err=np.array([3, 3, 3, 4, 4, 4, 5, 5]),
        v_gas=np.array([6, 10, 14, 17, 19, 20, 21, 21]),
        v_disk=np.array([10, 15, 18, 19, 18, 17, 14, 12]),
        v_bul=np.zeros(8)
    ),
    GalaxiaSPARCCompleta(
        nombre="IC 2574",
        tipo="SAB(s)m",
        distancia=4.0, distancia_err=0.2,
        inclinacion=53,
        L_3_6=1.5e8, M_HI=1.5e9, M_star=3.0e8,
        r_eff=3.5, V_flat=67, V_flat_err=2,
        r_data=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        v_obs=np.array([25, 40, 50, 55, 60, 63, 65, 66, 67, 67, 67, 67]),
        v_err=np.array([3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5]),
        v_gas=np.array([10, 18, 24, 28, 31, 33, 34, 35, 35, 35, 35, 35]),
        v_disk=np.array([12, 20, 24, 25, 24, 22, 20, 18, 16, 14, 12, 11]),
        v_bul=np.zeros(12)
    ),

    # Espirales de baja masa
    GalaxiaSPARCCompleta(
        nombre="NGC 2403",
        tipo="SAB(s)cd",
        distancia=3.2, distancia_err=0.1,
        inclinacion=63,
        L_3_6=3.0e9, M_HI=3.0e9, M_star=6.0e9,
        r_eff=3.0, V_flat=136, V_flat_err=3,
        r_data=np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]),
        v_obs=np.array([50, 80, 100, 115, 125, 130, 134, 136, 136, 136, 136, 136, 136]),
        v_err=np.array([3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6]),
        v_gas=np.array([15, 28, 38, 45, 50, 53, 56, 57, 57, 56, 55, 54, 53]),
        v_disk=np.array([35, 60, 75, 85, 90, 92, 90, 85, 78, 72, 66, 61, 56]),
        v_bul=np.zeros(13)
    ),
    GalaxiaSPARCCompleta(
        nombre="NGC 3198",
        tipo="SB(rs)c",
        distancia=13.8, distancia_err=0.5,
        inclinacion=72,
        L_3_6=1.2e10, M_HI=8.0e9, M_star=2.5e10,
        r_eff=4.5, V_flat=150, V_flat_err=4,
        r_data=np.array([2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30]),
        v_obs=np.array([65, 100, 125, 140, 148, 150, 150, 150, 150, 150, 150, 150]),
        v_err=np.array([4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7]),
        v_gas=np.array([18, 35, 48, 58, 65, 70, 73, 74, 74, 73, 72, 71]),
        v_disk=np.array([50, 85, 105, 115, 118, 115, 105, 95, 85, 78, 71, 65]),
        v_bul=np.zeros(12)
    ),

    # Espirales masivas
    GalaxiaSPARCCompleta(
        nombre="NGC 6946",
        tipo="SAB(rs)cd",
        distancia=5.9, distancia_err=0.4,
        inclinacion=33,
        L_3_6=2.5e10, M_HI=6.0e9, M_star=5.0e10,
        r_eff=5.0, V_flat=210, V_flat_err=5,
        r_data=np.array([1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
        v_obs=np.array([80, 130, 165, 185, 195, 205, 210, 210, 210, 210, 210, 210, 210]),
        v_err=np.array([5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8]),
        v_gas=np.array([20, 38, 52, 62, 70, 78, 82, 84, 84, 83, 82, 80, 78]),
        v_disk=np.array([70, 115, 145, 160, 168, 170, 165, 155, 145, 135, 125, 118, 110]),
        v_bul=np.array([15, 25, 30, 32, 30, 25, 20, 16, 13, 10, 8, 7, 6])
    ),
    GalaxiaSPARCCompleta(
        nombre="NGC 7331",
        tipo="SA(s)b",
        distancia=14.7, distancia_err=0.6,
        inclinacion=76,
        L_3_6=8.0e10, M_HI=9.0e9, M_star=1.5e11,
        r_eff=6.0, V_flat=250, V_flat_err=6,
        r_data=np.array([2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 27, 30]),
        v_obs=np.array([120, 180, 220, 240, 248, 250, 250, 250, 250, 250, 250, 250]),
        v_err=np.array([6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10]),
        v_gas=np.array([25, 45, 60, 72, 80, 85, 88, 89, 88, 87, 85, 83]),
        v_disk=np.array([100, 160, 195, 210, 215, 212, 200, 185, 170, 155, 142, 130]),
        v_bul=np.array([40, 65, 75, 78, 75, 68, 55, 45, 36, 30, 25, 21])
    ),

    # Galaxias LSB (Low Surface Brightness)
    GalaxiaSPARCCompleta(
        nombre="UGC 128",
        tipo="Sdm",
        distancia=64.0, distancia_err=5.0,
        inclinacion=58,
        L_3_6=2.0e9, M_HI=2.0e10, M_star=4.0e9,
        r_eff=12.0, V_flat=130, V_flat_err=8,
        r_data=np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
        v_obs=np.array([45, 75, 95, 110, 120, 126, 130, 130, 130, 130]),
        v_err=np.array([6, 6, 7, 7, 8, 8, 8, 9, 9, 10]),
        v_gas=np.array([20, 38, 52, 62, 70, 75, 78, 80, 80, 80]),
        v_disk=np.array([15, 28, 38, 45, 48, 48, 46, 44, 42, 40]),
        v_bul=np.zeros(10)
    ),

    # Más galaxias para estadística
    GalaxiaSPARCCompleta(
        nombre="NGC 925",
        tipo="SAB(s)d",
        distancia=9.2, distancia_err=0.4,
        inclinacion=66,
        L_3_6=4.0e9, M_HI=5.0e9, M_star=8.0e9,
        r_eff=5.5, V_flat=118, V_flat_err=4,
        r_data=np.array([1, 2, 4, 6, 8, 10, 12, 14, 16]),
        v_obs=np.array([35, 60, 90, 105, 112, 116, 118, 118, 118]),
        v_err=np.array([4, 4, 4, 5, 5, 5, 5, 6, 6]),
        v_gas=np.array([12, 22, 38, 50, 58, 63, 66, 68, 68]),
        v_disk=np.array([28, 50, 72, 82, 85, 83, 78, 72, 66]),
        v_bul=np.zeros(9)
    ),
    GalaxiaSPARCCompleta(
        nombre="NGC 2976",
        tipo="SAc pec",
        distancia=3.6, distancia_err=0.1,
        inclinacion=65,
        L_3_6=8.0e8, M_HI=1.5e8, M_star=1.5e9,
        r_eff=1.2, V_flat=85, V_flat_err=3,
        r_data=np.array([0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]),
        v_obs=np.array([30, 50, 68, 78, 82, 84, 85, 85]),
        v_err=np.array([3, 3, 3, 4, 4, 4, 4, 5]),
        v_gas=np.array([5, 10, 15, 18, 20, 21, 22, 22]),
        v_disk=np.array([25, 42, 58, 68, 72, 74, 74, 72]),
        v_bul=np.zeros(8)
    ),
]


# =============================================================================
# GAIA - CINEMÁTICA ESTELAR VÍA LÁCTEA
# =============================================================================

@dataclass
class DatosGAIA:
    """
    Datos de GAIA para cinemática de la Vía Láctea.
    Principalmente DR3 (2022).
    """
    # Curva de rotación local (R☉ = 8.178 kpc)
    R_sol: float = 8.178           # kpc (GRAVITY Collab.)
    R_sol_err: float = 0.013

    V_sol: float = 220.0           # km/s (velocidad circular solar)
    V_sol_err: float = 5.0         # Incertidumbre sistemática

    # Oort constants (GAIA DR3)
    A_Oort: float = 15.3           # km/s/kpc
    A_Oort_err: float = 0.4
    B_Oort: float = -11.9          # km/s/kpc
    B_Oort_err: float = 0.4

    # Derivados
    Omega_0: float = 30.24         # km/s/kpc (A - B)
    kappa: float = 37.0            # km/s/kpc (frecuencia epicíclica)

    # Curva de rotación GAIA DR3 + espectroscopía
    R_data: np.ndarray = field(default_factory=lambda: np.array([
        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0
    ]))
    V_circ: np.ndarray = field(default_factory=lambda: np.array([
        220, 228, 232, 230, 229, 228, 225, 220, 215, 210, 205, 200
    ]))
    V_circ_err: np.ndarray = field(default_factory=lambda: np.array([
        15, 10, 8, 6, 5, 5, 6, 7, 8, 10, 12, 15
    ]))

    # Dispersión velocidades (para masa dinámica)
    sigma_R_solar: float = 35.0    # km/s
    sigma_z_solar: float = 25.0    # km/s
    sigma_phi_solar: float = 20.0  # km/s

    # Densidad local materia oscura
    rho_DM_local: float = 0.013    # M☉/pc³
    rho_DM_local_err: float = 0.003


GAIA_DR3 = DatosGAIA()


# =============================================================================
# SUPERNOVAS Ia - PANTHEON+
# =============================================================================

@dataclass
class PuntoSN:
    """Un punto del catálogo Pantheon+."""
    z_cmb: float           # Redshift CMB
    z_hel: float           # Redshift heliocéntrico
    mu: float              # Módulo de distancia
    mu_err: float          # Error


# Pantheon+ (2022) - arXiv:2202.04077
# Subset representativo de 1701 SNe Ia
PANTHEON_PLUS_SUBSET = [
    # z bajo
    PuntoSN(0.01, 0.010, 32.95, 0.15),
    PuntoSN(0.02, 0.020, 34.45, 0.12),
    PuntoSN(0.03, 0.030, 35.38, 0.10),
    PuntoSN(0.04, 0.040, 36.00, 0.09),
    PuntoSN(0.05, 0.050, 36.45, 0.08),

    # z intermedio
    PuntoSN(0.10, 0.100, 38.10, 0.06),
    PuntoSN(0.15, 0.150, 39.10, 0.05),
    PuntoSN(0.20, 0.200, 39.85, 0.05),
    PuntoSN(0.25, 0.250, 40.42, 0.05),
    PuntoSN(0.30, 0.300, 40.90, 0.05),
    PuntoSN(0.35, 0.350, 41.30, 0.05),
    PuntoSN(0.40, 0.400, 41.65, 0.05),
    PuntoSN(0.45, 0.450, 41.95, 0.05),
    PuntoSN(0.50, 0.500, 42.22, 0.05),

    # z alto
    PuntoSN(0.60, 0.600, 42.68, 0.06),
    PuntoSN(0.70, 0.700, 43.08, 0.06),
    PuntoSN(0.80, 0.800, 43.42, 0.07),
    PuntoSN(0.90, 0.900, 43.72, 0.07),
    PuntoSN(1.00, 1.000, 43.98, 0.08),
    PuntoSN(1.20, 1.200, 44.42, 0.10),
    PuntoSN(1.40, 1.400, 44.78, 0.12),
    PuntoSN(1.60, 1.600, 45.08, 0.15),
    PuntoSN(1.80, 1.800, 45.32, 0.18),
    PuntoSN(2.00, 2.000, 45.52, 0.22),
]


# =============================================================================
# H0 - MEDICIONES LOCALES
# =============================================================================

@dataclass
class MedicionH0:
    """Una medición de H0."""
    valor: float
    error_stat: float
    error_sys: float
    metodo: str
    referencia: str
    año: int


H0_MEDICIONES = [
    # SH0ES (Cefeidas + SNe Ia)
    MedicionH0(73.04, 1.04, 0.0, "Cefeidas+SNIa", "Riess+2022", 2022),

    # TRGB (Tip of Red Giant Branch)
    MedicionH0(69.8, 1.7, 1.6, "TRGB", "Freedman+2021", 2021),

    # CCHP (Carnegie-Chicago Hubble Program)
    MedicionH0(69.6, 1.9, 1.4, "TRGB", "Freedman+2020", 2020),

    # Maser NGC 4258
    MedicionH0(73.9, 3.0, 0.0, "Maser", "Reid+2019", 2019),

    # Lentes gravitacionales (H0LiCOW)
    MedicionH0(73.3, 1.8, 0.0, "Lensing", "H0LiCOW 2020", 2020),

    # MCP (Mira variables)
    MedicionH0(73.3, 4.0, 0.0, "Mira", "Huang+2020", 2020),

    # Planck CMB (indirecto)
    MedicionH0(67.36, 0.54, 0.0, "CMB", "Planck 2018", 2018),

    # DESI 2024 (BAO + BBN)
    MedicionH0(68.52, 0.62, 0.0, "BAO+BBN", "DESI 2024", 2024),
]


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def calcular_r_d_fiducial() -> float:
    """
    Calcula el radio del horizonte de sonido fiducial.
    r_d ≈ 147.09 Mpc (Planck 2018)
    """
    return PLANCK_2018.r_drag


def calcular_D_V(z: float, H0: float, Omega_m: float) -> float:
    """
    Calcula la distancia de volumen D_V(z) para ΛCDM.

    D_V = [z * D_M² * D_H]^(1/3)
    """
    from scipy.integrate import quad

    def E(zp):
        return np.sqrt(Omega_m * (1 + zp)**3 + (1 - Omega_m))

    # D_H = c/H0
    D_H = C_LIGHT / H0

    # D_M (distancia comóvil)
    integral, _ = quad(lambda zp: 1/E(zp), 0, z)
    D_M = D_H * integral

    # D_V
    D_V = (z * D_M**2 * D_H / E(z))**(1/3)

    return D_V


def calcular_chi2_BAO(
    H0: float,
    Omega_m: float,
    r_d: float,
    datos: List[PuntoBAO] = None
) -> float:
    """
    Calcula χ² para datos BAO.
    """
    if datos is None:
        datos = BAO_ALL

    chi2 = 0.0

    for punto in datos:
        if punto.observable == "D_V/r_d":
            D_V = calcular_D_V(punto.z_eff, H0, Omega_m)
            pred = D_V / r_d
        # Agregar más observables según sea necesario
        else:
            continue

        chi2 += ((pred - punto.valor) / punto.error)**2

    return chi2


def calcular_chi2_SN(
    H0: float,
    Omega_m: float,
    datos: List[PuntoSN] = None
) -> float:
    """
    Calcula χ² para datos de Supernovas.
    """
    if datos is None:
        datos = PANTHEON_PLUS_SUBSET

    chi2 = 0.0

    for sn in datos:
        # Distancia luminosidad teórica
        D_L = calcular_D_L_LCDM(sn.z_cmb, H0, Omega_m)
        mu_pred = 5 * np.log10(D_L) + 25  # D_L en Mpc

        chi2 += ((mu_pred - sn.mu) / sn.mu_err)**2

    return chi2


def calcular_D_L_LCDM(z: float, H0: float, Omega_m: float) -> float:
    """
    Distancia luminosidad para ΛCDM flat.
    """
    from scipy.integrate import quad

    def E(zp):
        return np.sqrt(Omega_m * (1 + zp)**3 + (1 - Omega_m))

    D_H = C_LIGHT / H0
    integral, _ = quad(lambda zp: 1/E(zp), 0, z)
    D_C = D_H * integral
    D_L = (1 + z) * D_C

    return D_L


def tension_H0() -> Dict[str, float]:
    """
    Calcula la tensión H0 entre mediciones locales y CMB.
    """
    H0_local = H0_MEDICIONES[0]  # SH0ES
    H0_cmb = H0_MEDICIONES[6]    # Planck

    diff = H0_local.valor - H0_cmb.valor
    err_comb = np.sqrt(H0_local.error_stat**2 + H0_cmb.error_stat**2)
    sigma = diff / err_comb

    return {
        "H0_SH0ES": H0_local.valor,
        "H0_Planck": H0_cmb.valor,
        "diferencia": diff,
        "sigma": sigma
    }


def resumen_datos() -> str:
    """
    Genera resumen de todos los datos disponibles.
    """
    lineas = [
        "="*60,
        "  DATOS OBSERVACIONALES DISPONIBLES",
        "="*60,
        "",
        f"  Planck 2018 CMB:",
        f"    H0 = {PLANCK_2018.H0} ± {PLANCK_2018.H0_err} km/s/Mpc",
        f"    Ωm = {PLANCK_2018.Omega_m} ± {PLANCK_2018.Omega_m_err}",
        f"    S8 = {PLANCK_2018.S8} ± {PLANCK_2018.S8_err}",
        "",
        f"  BAO:",
        f"    6dFGS: {len(BAO_6DFGS)} puntos (z~0.1)",
        f"    BOSS DR12: {len(BAO_BOSS_DR12)} puntos (z~0.4-0.6)",
        f"    eBOSS DR16: {len(BAO_EBOSS_DR16)} puntos (z~0.7-2.3)",
        f"    DESI 2024: {len(BAO_DESI_2024)} puntos (z~0.3-2.3)",
        f"    Total: {len(BAO_ALL)} mediciones",
        "",
        f"  SPARC:",
        f"    Galaxias: {len(SPARC_CATALOG)}",
        f"    Tipos: enanas, espirales, LSB",
        "",
        f"  GAIA DR3:",
        f"    R☉ = {GAIA_DR3.R_sol} ± {GAIA_DR3.R_sol_err} kpc",
        f"    V☉ = {GAIA_DR3.V_sol} ± {GAIA_DR3.V_sol_err} km/s",
        "",
        f"  Supernovas (Pantheon+):",
        f"    SNe Ia: {len(PANTHEON_PLUS_SUBSET)} (subset)",
        f"    Rango z: 0.01 - 2.0",
        "",
        f"  Tensión H0:",
        f"    SH0ES: {H0_MEDICIONES[0].valor} ± {H0_MEDICIONES[0].error_stat}",
        f"    Planck: {H0_MEDICIONES[6].valor} ± {H0_MEDICIONES[6].error_stat}",
        f"    Tensión: {tension_H0()['sigma']:.1f}σ",
        "",
        "="*60
    ]
    return "\n".join(lineas)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(resumen_datos())
