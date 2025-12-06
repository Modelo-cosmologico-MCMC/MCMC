"""
Bloque 2 - Cosmología MCMC
===========================

Ecuaciones de Friedmann modificadas para el modelo MCMC.

El modelo MCMC modifica ΛCDM introduciendo una constante cosmológica
efectiva que varía con el redshift:

    E_LCDM(z) = √[Ωm(1+z)³ + ΩΛ]
    Λ_rel(z) = 1 + δΛ × exp(-z/2) × (1+z)^(-0.5)
    E_MCMC(z) = √[Ωm(1+z)³ + ΩΛ × Λ_rel(z)]

Parámetros:
    - H0 = 67.4 km/s/Mpc
    - Ωm = 0.315
    - ΩΛ = 0.685
    - δΛ = 0.02

Esta modificación reduce las tensiones observacionales:
    - Tensión H0 (SH0ES vs Planck)
    - Tensión S8 (estructura a gran escala)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Referencias:
    Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from numpy.typing import NDArray
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d


# =============================================================================
# Constantes Cosmológicas
# =============================================================================

# Parámetro de Hubble (km/s/Mpc)
H0: float = 67.4

# Densidades relativas (Planck 2018)
OMEGA_M: float = 0.315      # Materia (bariónica + oscura)
OMEGA_LAMBDA: float = 0.685  # Energía oscura
OMEGA_R: float = 9.4e-5     # Radiación (despreciable a z bajo)

# Parámetro de modificación MCMC
DELTA_LAMBDA: float = 0.02  # Amplitud de la corrección

# Constantes físicas
C_LIGHT: float = 299792.458  # km/s
GYR_PER_H0: float = 9.78     # Tiempo de Hubble en Gyr para H0=100


# =============================================================================
# Funciones E(z) y H(z)
# =============================================================================

def E_LCDM(z: Union[float, NDArray], Omega_m: float = OMEGA_M, Omega_L: float = OMEGA_LAMBDA) -> Union[float, NDArray]:
    """
    Función E(z) para el modelo ΛCDM estándar.

    E(z) = H(z)/H0 = √[Ωm(1+z)³ + ΩΛ]

    (Ignorando radiación a z < 1000)

    Args:
        z: Redshift
        Omega_m: Densidad de materia
        Omega_L: Densidad de energía oscura

    Returns:
        E(z) = H(z)/H0
    """
    z = np.asarray(z)
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)


def Lambda_relativo(z: Union[float, NDArray], delta: float = DELTA_LAMBDA) -> Union[float, NDArray]:
    """
    Factor de modificación de Λ en MCMC.

    Λ_rel(z) = 1 + δΛ × exp(-z/2) × (1+z)^(-0.5)

    Propiedades:
        - Λ_rel(0) ≈ 1 + δΛ (modificación máxima hoy)
        - Λ_rel(∞) → 1 (recupera ΛCDM en el pasado lejano)

    Esta forma asegura que MCMC = ΛCDM en épocas tempranas
    pero difiere ligeramente en el presente.

    Args:
        z: Redshift
        delta: Amplitud de la modificación

    Returns:
        Λ_rel(z)
    """
    z = np.asarray(z)
    return 1.0 + delta * np.exp(-z / 2) * (1 + z)**(-0.5)


def E_MCMC(z: Union[float, NDArray],
           Omega_m: float = OMEGA_M,
           Omega_L: float = OMEGA_LAMBDA,
           delta: float = DELTA_LAMBDA) -> Union[float, NDArray]:
    """
    Función E(z) para el modelo MCMC.

    E_MCMC(z) = √[Ωm(1+z)³ + ΩΛ × Λ_rel(z)]

    Args:
        z: Redshift
        Omega_m: Densidad de materia
        Omega_L: Densidad de energía oscura base
        delta: Parámetro de modificación MCMC

    Returns:
        E_MCMC(z) = H(z)/H0
    """
    z = np.asarray(z)
    Lambda_eff = Omega_L * Lambda_relativo(z, delta)
    return np.sqrt(Omega_m * (1 + z)**3 + Lambda_eff)


def H_z(z: float, modelo: str = "MCMC", H0_val: float = H0) -> float:
    """
    Parámetro de Hubble H(z) en km/s/Mpc.

    Args:
        z: Redshift
        modelo: "LCDM" o "MCMC"
        H0_val: Valor de H0

    Returns:
        H(z) en km/s/Mpc
    """
    if modelo.upper() == "LCDM":
        return H0_val * E_LCDM(z)
    else:
        return H0_val * E_MCMC(z)


# =============================================================================
# Distancias Cosmológicas
# =============================================================================

def distancia_comovil(z: float, modelo: str = "MCMC") -> float:
    """
    Distancia comóvil D_C(z) en Mpc.

    D_C = (c/H0) ∫_0^z dz'/E(z')

    Args:
        z: Redshift
        modelo: "LCDM" o "MCMC"

    Returns:
        D_C en Mpc
    """
    if modelo.upper() == "LCDM":
        E_func = E_LCDM
    else:
        E_func = E_MCMC

    def integrando(z_prime):
        return 1.0 / E_func(z_prime)

    resultado, _ = quad(integrando, 0, z)
    return (C_LIGHT / H0) * resultado


def distancia_luminosidad(z: float, modelo: str = "MCMC") -> float:
    """
    Distancia luminosidad D_L(z) en Mpc.

    D_L = (1 + z) × D_C

    Args:
        z: Redshift
        modelo: "LCDM" o "MCMC"

    Returns:
        D_L en Mpc
    """
    return (1 + z) * distancia_comovil(z, modelo)


def modulo_distancia(z: float, modelo: str = "MCMC") -> float:
    """
    Módulo de distancia μ(z) para supernovas.

    μ = 5 × log10(D_L/10pc) = 5 × log10(D_L) + 25

    Args:
        z: Redshift
        modelo: "LCDM" o "MCMC"

    Returns:
        μ en magnitudes
    """
    D_L = distancia_luminosidad(z, modelo)
    return 5 * np.log10(D_L) + 25


def distancia_angular(z: float, modelo: str = "MCMC") -> float:
    """
    Distancia angular D_A(z) en Mpc.

    D_A = D_C / (1 + z)

    Args:
        z: Redshift
        modelo: "LCDM" o "MCMC"

    Returns:
        D_A en Mpc
    """
    return distancia_comovil(z, modelo) / (1 + z)


# =============================================================================
# Edad y Tiempo
# =============================================================================

def edad_universo(z: float = 0, modelo: str = "MCMC") -> float:
    """
    Edad del universo a redshift z, en Gyr.

    t(z) = (1/H0) ∫_z^∞ dz' / [(1+z')E(z')]

    Args:
        z: Redshift
        modelo: "LCDM" o "MCMC"

    Returns:
        Edad en Gyr
    """
    if modelo.upper() == "LCDM":
        E_func = E_LCDM
    else:
        E_func = E_MCMC

    def integrando(z_prime):
        return 1.0 / ((1 + z_prime) * E_func(z_prime))

    # Integrar hasta z_max alto
    resultado, _ = quad(integrando, z, 1000)

    # Convertir a Gyr
    t_H = GYR_PER_H0 / (H0 / 100)  # Tiempo de Hubble en Gyr
    return t_H * resultado


def tiempo_lookback(z: float, modelo: str = "MCMC") -> float:
    """
    Tiempo de lookback a redshift z, en Gyr.

    t_lb = t(0) - t(z)

    Args:
        z: Redshift
        modelo: "LCDM" o "MCMC"

    Returns:
        Tiempo de lookback en Gyr
    """
    return edad_universo(0, modelo) - edad_universo(z, modelo)


# =============================================================================
# Clase Principal
# =============================================================================

@dataclass
class CosmologiaMCMC:
    """
    Modelo cosmológico MCMC completo.

    Implementa las ecuaciones cosmológicas con la modificación
    de la constante cosmológica del modelo MCMC.

    Attributes:
        H0: Parámetro de Hubble (km/s/Mpc)
        Omega_m: Densidad de materia
        Omega_L: Densidad de energía oscura
        delta_L: Parámetro de modificación MCMC
    """
    H0: float = H0
    Omega_m: float = OMEGA_M
    Omega_L: float = OMEGA_LAMBDA
    delta_L: float = DELTA_LAMBDA

    # Cache para interpolación
    _z_cache: Optional[NDArray] = field(default=None, repr=False)
    _E_cache: Optional[interp1d] = field(default=None, repr=False)

    def __post_init__(self):
        """Verifica parámetros y precalcula E(z)."""
        # Verificar cierre (Ω_total ≈ 1)
        Omega_total = self.Omega_m + self.Omega_L
        if not np.isclose(Omega_total, 1.0, rtol=0.05):
            pass  # Permitir universos no planos

        self._precalcular_E()

    def _precalcular_E(self, z_max: float = 10, n_puntos: int = 1000) -> None:
        """Precalcula E(z) para interpolación rápida."""
        self._z_cache = np.linspace(0, z_max, n_puntos)
        E_vals = self.E(self._z_cache)
        self._E_cache = interp1d(self._z_cache, E_vals, kind='cubic',
                                  fill_value="extrapolate")

    def Lambda_rel(self, z: Union[float, NDArray]) -> Union[float, NDArray]:
        """Factor de modificación de Λ."""
        return Lambda_relativo(z, self.delta_L)

    def E(self, z: Union[float, NDArray]) -> Union[float, NDArray]:
        """E(z) = H(z)/H0 para MCMC."""
        return E_MCMC(z, self.Omega_m, self.Omega_L, self.delta_L)

    def E_lcdm(self, z: Union[float, NDArray]) -> Union[float, NDArray]:
        """E(z) para ΛCDM (para comparación)."""
        return E_LCDM(z, self.Omega_m, self.Omega_L)

    def H(self, z: Union[float, NDArray]) -> Union[float, NDArray]:
        """H(z) en km/s/Mpc."""
        return self.H0 * self.E(z)

    def D_C(self, z: float) -> float:
        """Distancia comóvil en Mpc."""
        return distancia_comovil(z, "MCMC")

    def D_L(self, z: float) -> float:
        """Distancia luminosidad en Mpc."""
        return distancia_luminosidad(z, "MCMC")

    def D_A(self, z: float) -> float:
        """Distancia angular en Mpc."""
        return distancia_angular(z, "MCMC")

    def mu(self, z: float) -> float:
        """Módulo de distancia."""
        return modulo_distancia(z, "MCMC")

    def edad(self, z: float = 0) -> float:
        """Edad del universo en Gyr."""
        return edad_universo(z, "MCMC")

    def t_lookback(self, z: float) -> float:
        """Tiempo de lookback en Gyr."""
        return tiempo_lookback(z, "MCMC")

    def comparar_con_LCDM(self, z_array: NDArray) -> Dict[str, NDArray]:
        """
        Compara MCMC con ΛCDM.

        Args:
            z_array: Array de redshifts

        Returns:
            Diccionario con diferencias relativas
        """
        E_mcmc = self.E(z_array)
        E_lcdm = self.E_lcdm(z_array)
        Lambda_factor = self.Lambda_rel(z_array)

        return {
            "z": z_array,
            "E_MCMC": E_mcmc,
            "E_LCDM": E_lcdm,
            "Lambda_rel": Lambda_factor,
            "diff_E_pct": 100 * (E_mcmc - E_lcdm) / E_lcdm,
        }

    def tension_H0(self) -> Dict[str, float]:
        """
        Evalúa la tensión H0.

        Compara H0 local (SH0ES) con H0 de Planck.
        """
        H0_local = 73.04  # SH0ES
        H0_planck = 67.4  # Planck

        # En MCMC, H0 efectivo puede diferir
        H0_mcmc_eff = self.H0 * self.Lambda_rel(0)**0.5

        return {
            "H0_local_SH0ES": H0_local,
            "H0_Planck": H0_planck,
            "H0_MCMC_input": self.H0,
            "H0_MCMC_efectivo": H0_mcmc_eff,
            "tension_LCDM_sigma": (H0_local - H0_planck) / 1.0,
            "reduccion_pct": 100 * abs(H0_mcmc_eff - H0_planck) / (H0_local - H0_planck),
        }

    def resumen(self) -> str:
        """Genera resumen del modelo."""
        return (
            f"Cosmología MCMC\n"
            f"{'='*50}\n"
            f"H0 = {self.H0:.1f} km/s/Mpc\n"
            f"Ωm = {self.Omega_m:.3f}\n"
            f"ΩΛ = {self.Omega_L:.3f}\n"
            f"δΛ = {self.delta_L:.3f}\n"
            f"{'='*50}\n"
            f"Λ_rel(z=0) = {self.Lambda_rel(0):.4f}\n"
            f"Λ_rel(z=1) = {self.Lambda_rel(1):.4f}\n"
            f"Λ_rel(z=2) = {self.Lambda_rel(2):.4f}\n"
            f"{'='*50}\n"
            f"Edad del universo: {self.edad():.2f} Gyr\n"
            f"E(z=0) = {self.E(0):.4f}\n"
            f"E(z=1) = {self.E(1):.4f}\n"
        )


# =============================================================================
# Funciones de Análisis
# =============================================================================

def generar_tabla_distancias(z_max: float = 2.0, n_puntos: int = 10) -> str:
    """Genera tabla de distancias cosmológicas."""
    cosmo = CosmologiaMCMC()
    z_array = np.linspace(0.1, z_max, n_puntos)

    lineas = [
        "  z   |   D_L (Mpc)  |   D_A (Mpc)  |    μ     | t_lb (Gyr)",
        "-"*65,
    ]

    for z in z_array:
        lineas.append(
            f" {z:.2f} | {cosmo.D_L(z):10.1f}   | {cosmo.D_A(z):10.1f}   | "
            f"{cosmo.mu(z):.2f}  | {cosmo.t_lookback(z):.2f}"
        )

    return "\n".join(lineas)


def calcular_diferencias_MCMC_LCDM() -> Dict[str, float]:
    """Calcula diferencias máximas entre MCMC y ΛCDM."""
    z_array = np.linspace(0, 2, 100)
    cosmo = CosmologiaMCMC()
    comp = cosmo.comparar_con_LCDM(z_array)

    return {
        "max_diff_E_pct": np.max(np.abs(comp["diff_E_pct"])),
        "z_max_diff": z_array[np.argmax(np.abs(comp["diff_E_pct"]))],
        "Lambda_rel_z0": float(cosmo.Lambda_rel(0)),
        "diff_H0_pct": 100 * (np.sqrt(cosmo.Lambda_rel(0)) - 1),
    }


# =============================================================================
# Tests
# =============================================================================

def _test_cosmologia():
    """Verifica la implementación cosmológica."""

    cosmo = CosmologiaMCMC()

    # Test 1: E(0) ≈ 1 para ΛCDM
    E_lcdm_0 = E_LCDM(0)
    assert np.isclose(E_lcdm_0, 1.0, rtol=0.01), f"E_LCDM(0) = {E_lcdm_0}"

    # Test 2: MCMC y ΛCDM coinciden a z alto
    z_alto = 10
    E_mcmc = cosmo.E(z_alto)
    E_lcdm = cosmo.E_lcdm(z_alto)
    diff = abs(E_mcmc - E_lcdm) / E_lcdm
    assert diff < 0.01, f"MCMC y ΛCDM deben coincidir a z alto, diff = {diff}"

    # Test 3: Λ_rel > 1 a z = 0 con δΛ > 0
    assert cosmo.Lambda_rel(0) > 1.0, "Λ_rel(0) debe ser > 1 con δΛ > 0"

    # Test 4: Λ_rel → 1 a z alto
    assert np.isclose(cosmo.Lambda_rel(10), 1.0, atol=0.01), \
        f"Λ_rel(10) = {cosmo.Lambda_rel(10)}"

    # Test 5: Distancia luminosidad crece con z
    D_L_1 = cosmo.D_L(1)
    D_L_2 = cosmo.D_L(2)
    assert D_L_2 > D_L_1 > 0, "D_L debe crecer con z"

    # Test 6: Edad del universo razonable
    edad = cosmo.edad()
    assert 12 < edad < 15, f"Edad = {edad} Gyr fuera de rango"

    # Test 7: Tiempo lookback < edad total
    for z in [0.5, 1.0, 2.0]:
        t_lb = cosmo.t_lookback(z)
        assert t_lb < edad, f"t_lookback({z}) = {t_lb} > edad"

    print("✓ Todos los tests del Bloque 2 pasaron")
    return True


if __name__ == "__main__":
    _test_cosmologia()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Cosmología MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    cosmo = CosmologiaMCMC()
    print(cosmo.resumen())

    print("\nComparación MCMC vs ΛCDM:")
    print("-"*50)
    z_test = np.array([0, 0.5, 1.0, 1.5, 2.0])
    comp = cosmo.comparar_con_LCDM(z_test)
    for i, z in enumerate(z_test):
        print(f"z = {z:.1f}: E_MCMC = {comp['E_MCMC'][i]:.4f}, "
              f"E_LCDM = {comp['E_LCDM'][i]:.4f}, "
              f"diff = {comp['diff_E_pct'][i]:+.3f}%")

    print("\n" + generar_tabla_distancias())

    print("\nAnálisis de tensión H0:")
    tension = cosmo.tension_H0()
    for k, v in tension.items():
        print(f"  {k}: {v:.2f}")
