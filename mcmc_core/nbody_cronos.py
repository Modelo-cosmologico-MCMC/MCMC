#!/usr/bin/env python3
"""
================================================================================
MÓDULO N-BODY CRONOS: Simulaciones con Dilatación Temporal
================================================================================

Implementa simulaciones N-body que incluyen los efectos de la Ley de Cronos:
1. Dilatación temporal en regiones densas: dt_propio = dt_coord * α(ρ)
2. Fricción entrópica: dv/dt = -η(ρ) * v
3. Emergencia de núcleos cored en halos

La Ley de Cronos establece que el tiempo transcurre más lento en
regiones de alta densidad, similar a la dilatación gravitacional
pero con origen entrópico.

Efectos principales:
- Reduce la velocidad de infall en centros de halos
- Produce perfiles de densidad cored (no cuspy)
- Modifica la función de masa de halos

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Importar funciones MCMC
from .ontologia_ecv_mcv import (
    G_GRAV, H0_MCMC, OMEGA_M_MCMC, OMEGA_LAMBDA_MCMC,
    ETA_FRICTION, GAMMA_FRICTION, RHO_CRIT_FRICTION,
)


# =============================================================================
# CONSTANTES N-BODY
# =============================================================================

# Unidades: kpc, M☉, Gyr, km/s
KPC_PER_MPC = 1000
GYR_PER_H0 = 9.78 / (H0_MCMC / 100)  # ~14.5 Gyr

# Densidad crítica actual
RHO_CRIT_0 = 2.775e11 * (H0_MCMC/100)**2  # M☉/Mpc³
RHO_CRIT_0_KPC = RHO_CRIT_0 / KPC_PER_MPC**3  # M☉/kpc³

# Conversión de tiempo
KMS_TO_KPC_GYR = 1.0227  # km/s → kpc/Gyr


# =============================================================================
# PARÁMETROS LEY DE CRONOS
# =============================================================================

@dataclass
class ParametrosCronosNBody:
    """
    Parámetros de la Ley de Cronos para N-body.

    NOTA: Estos parámetros están alineados con mcmc_core/bloque3_nbody.py
    y mcmc_core/ley_cronos.py para garantizar consistencia ontológica.
    """
    # Dilatación temporal - Fórmula: Δt/Δt₀ = 1 + (ρ/ρc)^β / α
    alpha_lapse: float = 1.0        # Parámetro de lapse (ALPHA_LAPSE)
    rho_cronos: float = 277.5       # M☉/kpc³ - Densidad crítica de Cronos (RHO_CRONOS)
    beta_lapse: float = 1.5         # Exponente de dilatación (BETA_ETA)

    # Fricción entrópica - Fórmula: η(ρ) = α × (ρ/ρc)^β
    eta_0: float = 1.0              # Coeficiente de fricción (igual a alpha_lapse)
    gamma_friction: float = 1.5     # Exponente de densidad (igual a beta_lapse)
    rho_friction: float = 277.5     # Densidad de referencia (igual a rho_cronos)

    # Núcleo cored - Relación masa-núcleo calibrada
    r_core_base: float = 1.8        # kpc - Radio característico (R_STAR)
    alpha_core: float = 0.35        # Exponente: r_core ∝ M^α_r (ALPHA_R)
    beta_core: float = -0.5         # Exponente de redshift (BETA_R)


PARAMS_CRONOS = ParametrosCronosNBody()


# =============================================================================
# FUNCIONES DE LA LEY DE CRONOS
# =============================================================================

def lapse_function(rho: float, params: ParametrosCronosNBody = None) -> float:
    """
    Factor de dilatación temporal según la Ley de Cronos.

    Δt/Δt₀ = 1 + (ρ/ρc)^β / α

    El factor devuelto indica cuánto se "estira" el tiempo coordinado
    respecto al tiempo propio en una región de densidad ρ.

    El tiempo propio se relaciona con el tiempo coordenado como:
    dτ = dt / factor_dilatacion

    En regiones densas (ρ >> ρ_c): factor >> 1 (tiempo más lento)
    En regiones vacías (ρ << ρ_c): factor → 1 (tiempo normal)

    NOTA: Esta fórmula es consistente con mcmc_core/ley_cronos.py:dilatacion_temporal()

    Args:
        rho: Densidad local en M☉/kpc³
        params: Parámetros de Cronos

    Returns:
        Factor de dilatación Δt/Δt₀ ≥ 1
    """
    p = params or PARAMS_CRONOS

    if rho <= 0:
        return 1.0

    ratio = rho / p.rho_cronos

    # Fórmula de dilatación: Δt/Δt₀ = 1 + (ρ/ρc)^β / α
    factor = 1.0 + (ratio ** p.beta_lapse) / p.alpha_lapse

    return factor


def friccion_entropica(rho: float, v: np.ndarray,
                       params: ParametrosCronosNBody = None) -> np.ndarray:
    """
    Fricción entrópica según la Ley de Cronos.

    F_friction = -η(ρ) × v

    donde η(ρ) = α × (ρ/ρc)^β

    Esta fórmula es consistente con mcmc_core/bloque3_nbody.py:FriccionEntropica.eta()

    Args:
        rho: Densidad local en M☉/kpc³
        v: Vector velocidad (vx, vy, vz) en km/s
        params: Parámetros de Cronos

    Returns:
        Aceleración de fricción en kpc/Gyr²
    """
    p = params or PARAMS_CRONOS

    if rho <= 0:
        return np.zeros_like(v)

    # Coeficiente de fricción según fórmula ontológica: η = α × (ρ/ρc)^β
    ratio = rho / p.rho_friction
    eta = p.eta_0 * (ratio ** p.gamma_friction)

    return -eta * v * KMS_TO_KPC_GYR  # Convertir a kpc/Gyr²


def radio_core_cronos(M_halo: float, z: float = 0.0,
                      params: ParametrosCronosNBody = None) -> float:
    """
    Radio core predicho por la Ley de Cronos.

    r_core(M,z) = r★ × (M/M★)^α_r × (1+z)^β_r

    Con parámetros calibrados:
        - r★ = 1.8 kpc (r_core_base)
        - M★ = 10¹¹ M☉
        - α_r = 0.35 (alpha_core)
        - β_r = -0.5 (beta_core)

    Esta fórmula es consistente con mcmc_core/ley_cronos.py:radio_core()

    Args:
        M_halo: Masa del halo en M☉
        z: Redshift (default=0)
        params: Parámetros de Cronos

    Returns:
        Radio core en kpc
    """
    p = params or PARAMS_CRONOS
    M_ref = 1e11  # M☉ (M_STAR)

    factor_masa = (M_halo / M_ref) ** p.alpha_core
    factor_z = (1 + z) ** p.beta_core

    return p.r_core_base * factor_masa * factor_z


# =============================================================================
# PERFIL DE DENSIDAD CON CRONOS
# =============================================================================

def perfil_NFW(r: float, M_vir: float, c: float = 10) -> float:
    """Perfil NFW estándar (sin Cronos)."""
    r_vir = (M_vir / (4/3 * np.pi * 200 * RHO_CRIT_0_KPC))**(1/3)
    r_s = r_vir / c

    if r < 1e-6:
        r = 1e-6

    x = r / r_s
    rho_s = M_vir / (4 * np.pi * r_s**3 * (np.log(1+c) - c/(1+c)))

    return rho_s / (x * (1 + x)**2)


def perfil_Cronos(r: float, M_vir: float, c: float = 10,
                  params: ParametrosCronosNBody = None) -> float:
    """
    Perfil de densidad modificado por Ley de Cronos.

    ρ_Cronos(r) = ρ_NFW(r) / [1 + (r_core/r)^2]

    El núcleo cored emerge de la dilatación temporal que reduce
    la acreción de materia en el centro.

    Args:
        r: Radio en kpc
        M_vir: Masa virial en M☉
        c: Concentración
        params: Parámetros de Cronos

    Returns:
        Densidad en M☉/kpc³
    """
    p = params or PARAMS_CRONOS

    rho_nfw = perfil_NFW(r, M_vir, c)
    r_core = radio_core_cronos(M_vir, p)

    # Modificación cored
    if r < 1e-6:
        r = 1e-6

    coring_factor = 1 + (r_core / r)**2

    return rho_nfw / coring_factor


# =============================================================================
# PARTÍCULA N-BODY
# =============================================================================

@dataclass
class ParticulaNBody:
    """Partícula en simulación N-body."""
    masa: float                          # M☉
    posicion: np.ndarray = field(default_factory=lambda: np.zeros(3))  # kpc
    velocidad: np.ndarray = field(default_factory=lambda: np.zeros(3)) # km/s
    id: int = 0

    def energia_cinetica(self) -> float:
        """Energía cinética en (km/s)² M☉."""
        return 0.5 * self.masa * np.sum(self.velocidad**2)


# =============================================================================
# INTEGRADOR N-BODY CRONOS
# =============================================================================

class IntegradorCronos:
    """
    Integrador N-body con efectos de la Ley de Cronos.

    Implementa:
    1. Gravedad Newtoniana modificada con softening
    2. Dilatación temporal dependiente de densidad
    3. Fricción entrópica
    """

    def __init__(self, params: ParametrosCronosNBody = None):
        self.params = params or PARAMS_CRONOS
        self.particulas: List[ParticulaNBody] = []
        self.tiempo = 0.0  # Gyr
        self.softening = 0.1  # kpc - softening gravitacional

    def agregar_particula(self, masa: float, pos: np.ndarray, vel: np.ndarray):
        """Agrega una partícula a la simulación."""
        p = ParticulaNBody(
            masa=masa,
            posicion=pos.copy(),
            velocidad=vel.copy(),
            id=len(self.particulas)
        )
        self.particulas.append(p)

    def crear_halo_NFW(self, M_vir: float, c: float = 10, N_particulas: int = 1000):
        """
        Crea partículas siguiendo un perfil NFW.
        """
        r_vir = (M_vir / (4/3 * np.pi * 200 * RHO_CRIT_0_KPC))**(1/3)
        r_s = r_vir / c

        masa_particula = M_vir / N_particulas

        for i in range(N_particulas):
            # Muestreo de radio con perfil NFW
            u = np.random.random()
            # Aproximación para r desde CDF de NFW
            r = r_s * np.tan(np.pi/2 * u**(1/3))
            r = min(r, r_vir)

            # Posición isotrópica
            theta = np.arccos(2*np.random.random() - 1)
            phi = 2*np.pi*np.random.random()

            pos = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])

            # Velocidad circular + dispersión
            v_circ = np.sqrt(G_GRAV * M_vir * self._masa_encerrada_NFW_frac(r, r_s, c) / r)
            sigma_v = v_circ * 0.3  # Dispersión

            vel = np.random.normal(0, sigma_v, 3)

            self.agregar_particula(masa_particula, pos, vel)

    def _masa_encerrada_NFW_frac(self, r: float, r_s: float, c: float) -> float:
        """Fracción de masa encerrada en r para NFW."""
        x = r / r_s
        f_x = np.log(1 + x) - x / (1 + x)
        f_c = np.log(1 + c) - c / (1 + c)
        return f_x / f_c

    def calcular_densidad(self, pos: np.ndarray) -> float:
        """
        Estima la densidad local usando vecinos cercanos.
        """
        r_suave = 2.0  # kpc - escala de suavizado

        masa_encerrada = 0.0
        for p in self.particulas:
            dist = np.linalg.norm(pos - p.posicion)
            if dist < r_suave:
                # Kernel SPH simplificado
                w = (1 - (dist/r_suave)**2)**2 if dist < r_suave else 0
                masa_encerrada += p.masa * w

        volumen = 4/3 * np.pi * r_suave**3
        return masa_encerrada / volumen

    def calcular_aceleracion(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Calcula la aceleración total incluyendo:
        1. Gravedad
        2. Fricción entrópica (Ley de Cronos)
        """
        accel = np.zeros(3)

        # Gravedad de todas las partículas
        for p in self.particulas:
            r_vec = p.posicion - pos
            r = np.linalg.norm(r_vec)

            if r > 0.01:  # Evitar autointeracción
                # Gravedad con softening
                r_soft = np.sqrt(r**2 + self.softening**2)
                accel += G_GRAV * p.masa * r_vec / r_soft**3

        # Convertir a kpc/Gyr²
        accel *= KMS_TO_KPC_GYR

        # Fricción entrópica
        rho_local = self.calcular_densidad(pos)
        accel_friction = friccion_entropica(rho_local, vel, self.params)
        accel += accel_friction

        return accel

    def paso_temporal(self, dt: float):
        """
        Avanza la simulación un paso temporal dt (en Gyr).

        Usa integrador leapfrog con dilatación temporal de Cronos.

        El tiempo propio es: dt_propio = dt_coordinado / factor_dilatacion
        En regiones densas, dt_propio < dt → la evolución es más lenta.
        """
        for p in self.particulas:
            # Densidad local para dilatación
            rho_local = self.calcular_densidad(p.posicion)

            # Factor de dilatación (tiempo coordinado / tiempo propio)
            factor_dilatacion = lapse_function(rho_local, self.params)

            # Paso temporal propio (menor en regiones densas)
            dt_propio = dt / factor_dilatacion

            # Leapfrog: kick
            accel = self.calcular_aceleracion(p.posicion, p.velocidad)
            p.velocidad += accel * dt_propio / KMS_TO_KPC_GYR  # Convertir de vuelta a km/s

            # Leapfrog: drift
            p.posicion += p.velocidad * KMS_TO_KPC_GYR * dt_propio

        self.tiempo += dt

    def simular(self, t_final: float, dt: float = 0.01) -> Dict:
        """
        Ejecuta la simulación hasta t_final.

        Args:
            t_final: Tiempo final en Gyr
            dt: Paso temporal en Gyr

        Returns:
            Dict con resultados
        """
        n_pasos = int(t_final / dt)
        historial = {
            'tiempo': [],
            'energia_cinetica': [],
            'energia_potencial': [],
            'r_medio': [],
            'perfil_densidad': []
        }

        for i in range(n_pasos):
            self.paso_temporal(dt)

            if i % 10 == 0:  # Guardar cada 10 pasos
                E_cin = sum(p.energia_cinetica() for p in self.particulas)
                r_medio = np.mean([np.linalg.norm(p.posicion) for p in self.particulas])

                historial['tiempo'].append(self.tiempo)
                historial['energia_cinetica'].append(E_cin)
                historial['r_medio'].append(r_medio)

        return historial


# =============================================================================
# ANÁLISIS DE HALOS CON CRONOS
# =============================================================================

def analizar_halo_cronos(M_vir: float, c: float = 10,
                         params: ParametrosCronosNBody = None,
                         verbose: bool = True) -> Dict:
    """
    Analiza las propiedades de un halo con efectos Cronos.

    Args:
        M_vir: Masa virial en M☉
        c: Concentración
        params: Parámetros de Cronos
        verbose: Mostrar resultados

    Returns:
        Dict con propiedades del halo
    """
    p = params or PARAMS_CRONOS

    # Radio virial
    r_vir = (M_vir / (4/3 * np.pi * 200 * RHO_CRIT_0_KPC))**(1/3)
    r_s = r_vir / c

    # Radio core de Cronos
    r_core = radio_core_cronos(M_vir, p)

    # Perfiles de densidad
    r_array = np.logspace(-1, np.log10(r_vir), 100)
    rho_nfw = np.array([perfil_NFW(r, M_vir, c) for r in r_array])
    rho_cronos = np.array([perfil_Cronos(r, M_vir, c, p) for r in r_array])

    # Densidad central
    rho_0_nfw = rho_nfw[0]
    rho_0_cronos = rho_cronos[0]

    # Dilatación temporal en el centro
    alpha_centro = lapse_function(rho_0_cronos, p)

    if verbose:
        print(f"\n  Halo: M_vir = {M_vir:.2e} M☉")
        print(f"    r_vir = {r_vir:.2f} kpc")
        print(f"    r_s (NFW) = {r_s:.2f} kpc")
        print(f"    r_core (Cronos) = {r_core:.2f} kpc")
        print(f"    ρ_0 (NFW) = {rho_0_nfw:.2e} M☉/kpc³")
        print(f"    ρ_0 (Cronos) = {rho_0_cronos:.2e} M☉/kpc³")
        print(f"    Reducción central: {100*(1-rho_0_cronos/rho_0_nfw):.1f}%")
        print(f"    α(centro) = {alpha_centro:.3f} (dilatación {100*(1-alpha_centro):.1f}%)")

    return {
        'M_vir': M_vir,
        'r_vir': r_vir,
        'r_s': r_s,
        'r_core': r_core,
        'c': c,
        'r_array': r_array,
        'rho_nfw': rho_nfw,
        'rho_cronos': rho_cronos,
        'rho_0_nfw': rho_0_nfw,
        'rho_0_cronos': rho_0_cronos,
        'alpha_centro': alpha_centro
    }


def comparar_perfiles_cronos(verbose: bool = True) -> Dict:
    """
    Compara perfiles NFW vs Cronos para varios halos.
    """
    masas = [1e10, 1e11, 1e12, 1e13]  # M☉

    if verbose:
        print("\n" + "="*65)
        print("  N-BODY CRONOS: Comparación de Perfiles")
        print("="*65)

    resultados = []
    for M in masas:
        res = analizar_halo_cronos(M, verbose=verbose)
        resultados.append(res)

    return {'halos': resultados}


# =============================================================================
# TEST N-BODY CRONOS
# =============================================================================

def test_NBody_Cronos(verbose: bool = True) -> Dict:
    """
    Test del módulo N-body Cronos.
    """
    if verbose:
        print("\n" + "="*65)
        print("  TEST N-BODY CRONOS: Simulaciones con Dilatación Temporal")
        print("="*65)

    params = PARAMS_CRONOS

    # 1. Test de función lapse
    if verbose:
        print(f"\n  1. Factor de Dilatacion Temporal:")
        print(f"     Parametros: alpha={params.alpha_lapse}, rho_c={params.rho_cronos:.1f}")
        for rho in [10, 100, 277.5, 1000]:
            factor = lapse_function(rho, params)
            print(f"     Dt/Dt0({rho:.1f}) = {factor:.3f} (tiempo local {100/factor:.1f}% del coordinado)")

    # 2. Test de radio core
    if verbose:
        print(f"\n  2. Radio Core vs Masa:")
        for M in [1e9, 1e10, 1e11, 1e12]:
            r_c = radio_core_cronos(M, params)
            print(f"     r_core({M:.0e} M☉) = {r_c:.2f} kpc")

    # 3. Comparar perfiles
    comparacion = comparar_perfiles_cronos(verbose=verbose)

    # 4. Mini-simulación (pocas partículas para test rápido)
    if verbose:
        print(f"\n  3. Mini-simulación N-body:")

    integrador = IntegradorCronos(params)
    integrador.crear_halo_NFW(1e11, c=10, N_particulas=100)

    if verbose:
        print(f"     Partículas: {len(integrador.particulas)}")
        print(f"     Masa total: {sum(p.masa for p in integrador.particulas):.2e} M☉")

    # Simular 0.1 Gyr
    historial = integrador.simular(0.1, dt=0.01)

    if verbose:
        print(f"     Simulación: 0.1 Gyr, {len(historial['tiempo'])} snapshots")
        print(f"     r_medio inicial: {historial['r_medio'][0]:.2f} kpc")
        print(f"     r_medio final: {historial['r_medio'][-1]:.2f} kpc")

    # Criterios de éxito
    passed = True

    # Factor de dilatacion debe ser >= 1 (tiempo mas lento en regiones densas)
    factor_test = lapse_function(277.5, params)  # En rho = rho_cronos
    passed &= (factor_test >= 1.0)

    # r_core debe escalar con masa
    r_c_10 = radio_core_cronos(1e10, z=0.0, params=params)
    r_c_12 = radio_core_cronos(1e12, z=0.0, params=params)
    passed &= (r_c_12 > r_c_10)

    # Densidad central Cronos < NFW
    halo = comparacion['halos'][1]  # 10^11 M☉
    passed &= (halo['rho_0_cronos'] < halo['rho_0_nfw'])

    if verbose:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  Estado: {status}")
        print("="*65)

    return {
        'params': {
            'alpha_lapse': params.alpha_lapse,
            'rho_cronos': params.rho_cronos,
            'beta_lapse': params.beta_lapse,
            'eta_0': params.eta_0,
            'r_core_base': params.r_core_base,
            'alpha_core': params.alpha_core
        },
        'comparacion': comparacion,
        'historial': historial,
        'passed': passed
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_NBody_Cronos(verbose=True)
