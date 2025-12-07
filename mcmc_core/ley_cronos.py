"""
Ley de Cronos - Dilatación Temporal en Regiones Densas
=======================================================

La Ley de Cronos describe cómo el flujo del tiempo se modifica
en regiones de alta densidad, emergiendo en el sello S₄.

ACTIVACIÓN:
    La ley se activa en S ≥ S₄ = 1.001 (Big Bang).
    En S < S₄, el tiempo no está definido (fase pre-temporal).

DILATACIÓN TEMPORAL:
    Δt/Δt₀ = 1 + (ρ/ρc)^(3/2) / α

    donde:
    - ρc = 277.5 M☉/kpc³ (densidad crítica de Cronos)
    - α = 1.0 (parámetro de lapse)

CONSECUENCIAS:
    1. Núcleos planos (cored halos) en galaxias
    2. Supresión de satélites de baja masa
    3. Resolución del problema cusp-core

RELACIÓN MASA-NÚCLEO:
    r_core(M) = r★ × (M/M★)^α_r × (1+z)^β_r

    con r★ = 1.8 kpc, M★ = 10¹¹ M☉, α_r = 0.35, β_r = -0.5

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray


# =============================================================================
# Constantes de la Ley de Cronos
# =============================================================================

# Densidad crítica de Cronos (M☉/kpc³)
RHO_CRONOS: float = 277.5

# Parámetro de lapse
ALPHA_LAPSE: float = 1.0

# Sello de activación
S4_ACTIVACION: float = 1.001

# Parámetros de la relación masa-núcleo
R_STAR: float = 1.8       # kpc
M_STAR: float = 1e11      # M☉
ALPHA_R: float = 0.35     # Exponente de masa
BETA_R: float = -0.5      # Exponente de redshift
Z_STAR: float = 0.0       # Redshift de referencia


# =============================================================================
# Funciones de Dilatación Temporal
# =============================================================================

def dilatacion_temporal(
    rho: float,
    rho_c: float = RHO_CRONOS,
    alpha: float = ALPHA_LAPSE
) -> float:
    """
    Calcula el factor de dilatación temporal.

    Δt/Δt₀ = 1 + (ρ/ρc)^(3/2) / α

    En regiones densas, el tiempo fluye más lentamente,
    causando la formación de núcleos planos en halos.

    Args:
        rho: Densidad local (M☉/kpc³)
        rho_c: Densidad crítica de Cronos
        alpha: Parámetro de lapse

    Returns:
        Factor de dilatación Δt/Δt₀ ≥ 1
    """
    if rho <= 0:
        return 1.0

    x = rho / rho_c
    return 1.0 + (x ** 1.5) / alpha


def tiempo_propio(
    t_coordinado: float,
    rho: float,
    rho_c: float = RHO_CRONOS
) -> float:
    """
    Calcula el tiempo propio dado el tiempo coordinado.

    τ = t / (Δt/Δt₀)

    El tiempo propio es el experimentado localmente,
    que es menor en regiones densas.

    Args:
        t_coordinado: Tiempo coordinado (Gyr)
        rho: Densidad local

    Returns:
        Tiempo propio τ (Gyr)
    """
    factor = dilatacion_temporal(rho, rho_c)
    return t_coordinado / factor


def densidad_efectiva(
    rho: float,
    S: float
) -> float:
    """
    Densidad efectiva considerando la activación de Cronos.

    En S < S₄, la ley no está activa (pre-temporal).
    En S ≥ S₄, la densidad se modifica por efectos de Cronos.

    Args:
        rho: Densidad física
        S: Sello entrópico

    Returns:
        Densidad efectiva
    """
    if S < S4_ACTIVACION:
        return rho  # Pre-temporal, sin modificación

    factor = dilatacion_temporal(rho)
    return rho / factor


# =============================================================================
# Relación Masa-Núcleo
# =============================================================================

def radio_core(
    M_halo: float,
    z: float = 0.0,
    r_star: float = R_STAR,
    M_star: float = M_STAR,
    alpha_r: float = ALPHA_R,
    beta_r: float = BETA_R
) -> float:
    """
    Calcula el radio del núcleo según la Ley de Cronos.

    r_core(M,z) = r★ × (M/M★)^α_r × ((1+z)/(1+z★))^β_r

    Esta relación emerge de la dilatación temporal en
    las regiones centrales densas de los halos.

    Args:
        M_halo: Masa del halo (M☉)
        z: Redshift
        r_star: Radio característico (kpc)
        M_star: Masa característica (M☉)
        alpha_r: Exponente de masa
        beta_r: Exponente de redshift

    Returns:
        r_core en kpc
    """
    factor_masa = (M_halo / M_star) ** alpha_r
    factor_z = ((1 + z) / (1 + Z_STAR)) ** beta_r

    return r_star * factor_masa * factor_z


def tabla_radios_core() -> List[Dict]:
    """
    Genera tabla de radios de núcleo para diferentes masas.
    """
    masas = [1e10, 1e11, 1e12, 1e13]

    tabla = []
    for M in masas:
        tabla.append({
            "M_halo": M,
            "r_core_z0": radio_core(M, z=0),
            "r_core_z1": radio_core(M, z=1),
            "r_core_z2": radio_core(M, z=2),
        })

    return tabla


# =============================================================================
# Fricción Entrópica Extendida
# =============================================================================

def friccion_entropica_cronos(
    rho: float,
    v: float,
    S: float,
    alpha_f: float = 0.1
) -> float:
    """
    Fricción entrópica modificada por la Ley de Cronos.

    η(ρ,S) = α_f × (ρ/ρc)^(3/2) × f(S)

    donde f(S) = 1 si S ≥ S₄, 0 si S < S₄

    Args:
        rho: Densidad local
        v: Velocidad
        S: Sello entrópico
        alpha_f: Parámetro de fricción

    Returns:
        Fuerza de fricción por unidad de masa
    """
    if S < S4_ACTIVACION:
        return 0.0  # No hay fricción en fase pre-temporal

    x = rho / RHO_CRONOS
    eta = alpha_f * (x ** 1.5)

    return -eta * v


# =============================================================================
# Clase Principal: LeyCronos
# =============================================================================

@dataclass
class LeyCronos:
    """
    Implementación de la Ley de Cronos.

    La Ley de Cronos describe la emergencia del tiempo
    en S₄ y sus efectos en la formación de estructuras.

    EFECTOS PRINCIPALES:
        1. Dilatación temporal: Δt/Δt₀ = 1 + (ρ/ρc)^1.5
        2. Núcleos planos: r_core ∝ M^0.35
        3. Supresión de satélites

    Attributes:
        rho_c: Densidad crítica de Cronos
        alpha: Parámetro de lapse
        r_star: Radio característico del núcleo
        M_star: Masa característica
    """
    rho_c: float = RHO_CRONOS
    alpha: float = ALPHA_LAPSE
    r_star: float = R_STAR
    M_star: float = M_STAR

    def dilatacion(self, rho: float) -> float:
        """Factor de dilatación temporal."""
        return dilatacion_temporal(rho, self.rho_c, self.alpha)

    def tiempo_propio(self, t: float, rho: float) -> float:
        """Tiempo propio en función del coordinado."""
        return tiempo_propio(t, rho, self.rho_c)

    def r_core(self, M: float, z: float = 0.0) -> float:
        """Radio del núcleo para un halo."""
        return radio_core(M, z, self.r_star, self.M_star)

    def esta_activa(self, S: float) -> bool:
        """Determina si la ley está activa en el sello S."""
        return S >= S4_ACTIVACION

    def tabla_dilatacion(self) -> List[Dict]:
        """Genera tabla de dilatación para diferentes regiones."""
        regiones = [
            ("Vacío cósmico", 0.1 * self.rho_c),
            ("Periferia de halos", 1.0 * self.rho_c),
            ("Interior de halos", 10.0 * self.rho_c),
            ("Centro de halos", 100.0 * self.rho_c),
        ]

        tabla = []
        for nombre, rho in regiones:
            tabla.append({
                "region": nombre,
                "rho/rho_c": rho / self.rho_c,
                "Dt/Dt0": self.dilatacion(rho),
            })

        return tabla

    def resumen(self) -> str:
        """Genera resumen de la Ley de Cronos."""
        return (
            f"Ley de Cronos - Dilatación Temporal MCMC\n"
            f"{'='*55}\n"
            f"Activación: S ≥ {S4_ACTIVACION} (Big Bang)\n"
            f"{'='*55}\n"
            f"Parámetros:\n"
            f"  ρc = {self.rho_c:.1f} M☉/kpc³\n"
            f"  α = {self.alpha:.1f}\n"
            f"  r★ = {self.r_star:.1f} kpc\n"
            f"  M★ = {self.M_star:.2e} M☉\n"
            f"{'='*55}\n"
            f"Dilatación: Δt/Δt₀ = 1 + (ρ/ρc)^1.5 / α\n"
            f"{'='*55}\n"
            f"Ejemplos de dilatación:\n"
        )


# =============================================================================
# Mapeo S ↔ t ↔ z
# =============================================================================

# Parámetros del mapeo
BETA_MAPEO: float = 0.1   # Escala de mapeo S → t
T0_MAPEO: float = 0.01    # Tiempo de referencia (Gyr)


def S_desde_t(t: float, S4: float = 1.001, beta: float = BETA_MAPEO, t0: float = T0_MAPEO) -> float:
    """
    Mapea tiempo cósmico t a sello entrópico S.

    S(t) = S₄ + β × ln(1 + t/t₀)

    Args:
        t: Tiempo cósmico (Gyr)
        S4: Sello del Big Bang
        beta: Escala de mapeo
        t0: Tiempo de referencia

    Returns:
        S(t)
    """
    if t < 0:
        return S4
    return S4 + beta * np.log(1 + t / t0)


def t_desde_S(S: float, S4: float = 1.001, beta: float = BETA_MAPEO, t0: float = T0_MAPEO) -> float:
    """
    Mapea sello entrópico S a tiempo cósmico t.

    t(S) = t₀ × (exp[(S - S₄)/β] - 1)

    Args:
        S: Sello entrópico
        S4: Sello del Big Bang
        beta: Escala de mapeo
        t0: Tiempo de referencia

    Returns:
        t(S) en Gyr (0 si S ≤ S₄)
    """
    if S <= S4:
        return 0.0
    return t0 * (np.exp((S - S4) / beta) - 1)


def z_desde_t(t: float, t_universo: float = 13.8) -> float:
    """
    Mapea tiempo cósmico t a redshift z (aproximación).

    Usando modelo simplificado de EdS.
    """
    if t <= 0:
        return np.inf
    if t >= t_universo:
        return 0.0

    return (t_universo / t) ** (2/3) - 1


def tabla_correspondencias() -> List[Dict]:
    """
    Genera tabla de correspondencias S ↔ t ↔ z.
    """
    eventos = [
        ("Big Bang (S₄)", 0.0000, 1100),
        ("Recombinación", 0.00038, 1100),
        ("Primeras estrellas", 0.2, 20),
        ("Reionización", 0.5, 10),
        ("Formación del Sol", 9.2, 0.5),
        ("Presente", 13.8, 0.0),
    ]

    tabla = []
    for nombre, t, z in eventos:
        S = S_desde_t(t)
        tabla.append({
            "evento": nombre,
            "t_Gyr": t,
            "z": z,
            "S": S,
        })

    return tabla


# =============================================================================
# Tests
# =============================================================================

def _test_ley_cronos():
    """Verifica la implementación de la Ley de Cronos."""

    lc = LeyCronos()

    # Test 1: Dilatación = 1 para ρ = 0
    assert np.isclose(lc.dilatacion(0), 1.0), "Dilatación(0) = 1"

    # Test 2: Dilatación > 1 para ρ > 0
    assert lc.dilatacion(RHO_CRONOS) > 1.0, "Dilatación aumenta con ρ"

    # Test 3: Radio core crece con masa
    r1 = lc.r_core(1e10)
    r2 = lc.r_core(1e12)
    assert r2 > r1, "r_core debe crecer con M"

    # Test 4: Mapeo S ↔ t consistente
    for t in [0.1, 1.0, 5.0, 13.8]:
        S = S_desde_t(t)
        t_rec = t_desde_S(S)
        assert np.isclose(t, t_rec, rtol=0.01), f"Mapeo inconsistente: t={t}, t_rec={t_rec}"

    # Test 5: Ley activa solo en S ≥ S₄
    assert not lc.esta_activa(1.0), "No activa antes de S₄"
    assert lc.esta_activa(1.001), "Activa en S₄"
    assert lc.esta_activa(2.0), "Activa después de S₄"

    print("✓ Todos los tests de LeyCronos pasaron")
    return True


if __name__ == "__main__":
    _test_ley_cronos()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Ley de Cronos MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    lc = LeyCronos()
    print(lc.resumen())

    print("\nDilatación temporal por región:")
    print("-"*50)
    for row in lc.tabla_dilatacion():
        print(f"  {row['region']:20s}: ρ/ρc = {row['rho/rho_c']:5.1f}, "
              f"Δt/Δt₀ = {row['Dt/Dt0']:8.1f}")

    print("\n\nRadios de núcleo por masa:")
    print("-"*50)
    for row in tabla_radios_core():
        print(f"  M = {row['M_halo']:.0e} M☉: r_core = {row['r_core_z0']:.2f} kpc")

    print("\n\nCorrespondencias S ↔ t ↔ z:")
    print("-"*60)
    for row in tabla_correspondencias():
        print(f"  {row['evento']:20s}: t = {row['t_Gyr']:7.4f} Gyr, "
              f"z = {row['z']:6.1f}, S = {row['S']:.4f}")
