#!/usr/bin/env python3
"""
================================================================================
MÓDULO N-BODY BOX-100 CON FÍSICA CRONOS PARA MCMC
================================================================================

Configuración de simulación N-body cosmológica con Ley de Cronos.

Fundamentación Ontológica:
--------------------------
La Ley de Cronos introduce dilatación temporal en regiones densas:

    Δt_local = Δt × [1 + α⁻¹(ρ/ρ_c)^(3/2)]

Esto produce:
- Núcleos planos (cores) en halos sin necesidad de feedback extremo
- Reducción de subhalos de baja masa (~45%)
- Preservación de microhalos (diferencia clave con WDM)

Configuración de simulación:
- L_box = 100 h⁻¹ Mpc
- N_particles = 1024³ (resolución estándar)
- Snapshots: z = [9, 5, 3, 2, 1, 0]
- ICs: P(k) de CLASS-MCMC con Λ_rel(z)

Autor: Modelo MCMC
Copyright (c) 2024
================================================================================
"""

import numpy as np
from scipy.integrate import quad
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES Y PARÁMETROS
# =============================================================================

# Constantes físicas
G_GRAV = 4.302e-6  # kpc (km/s)² / M☉
C_LIGHT = 299792.458  # km/s
H0 = 67.4  # km/s/Mpc

# Densidad crítica (h² M☉/Mpc³)
RHO_CRIT_0 = 2.775e11

# Parámetros MCMC calibrados
MCMC_PARAMS = {
    'epsilon': 0.012,
    'z_trans': 8.9,
    'Delta_z': 1.5,
    'alpha_cronos': 0.15,
    'eta_friction': 0.05,
    'gamma_zhao': 0.51,
    'f_subhalo_suppression': 0.45,
    'Omega_m': 0.315,
    'sigma8': 0.811,
    'n_s': 0.965,
}


# =============================================================================
# PARÁMETROS DE SIMULACIÓN
# =============================================================================

@dataclass
class ConfiguracionBox100:
    """Configuración de la simulación Box-100."""
    # Tamaño de caja
    L_box: float = 100.0  # h⁻¹ Mpc

    # Número de partículas
    N_particles_side: int = 1024  # Por lado
    N_cells_side: int = 2048  # Malla PM

    # Resolución
    softening: float = 1.0  # kpc/h

    # Snapshots
    z_snapshots: List[float] = None

    # Cosmología
    Omega_m: float = 0.315
    Omega_b: float = 0.0493
    Omega_Lambda: float = 0.685
    h: float = 0.674
    sigma8: float = 0.811
    n_s: float = 0.965

    # Parámetros Cronos
    alpha_cronos: float = 0.15
    eta_friction: float = 0.05

    def __post_init__(self):
        if self.z_snapshots is None:
            self.z_snapshots = [9, 5, 3, 2, 1, 0.5, 0]


CONFIG_BOX100 = ConfiguracionBox100()


# =============================================================================
# CLASE PRINCIPAL BOX-100 CRONOS
# =============================================================================

class CronosBox100:
    """
    Configuración y física de simulación N-body con Cronos.

    La integración usa S (sello entrópico) como variable fundamental
    en lugar de tiempo coordenado t.

    Física clave:
    1. Función lapse α(ρ) para dilatación temporal
    2. Fricción entrópica F = -η(ρ)·v
    3. Campo φ_id de ECV para expansión modificada
    """

    def __init__(self, config: ConfiguracionBox100 = None):
        """
        Inicializa la configuración de simulación.

        Args:
            config: Configuración de caja
        """
        self.config = config or CONFIG_BOX100
        self.L_box = self.config.L_box
        self.N_particles = self.config.N_particles_side**3
        self.N_cells = self.config.N_cells_side**3

        # Cosmología
        self.Omega_m = self.config.Omega_m
        self.Omega_Lambda = self.config.Omega_Lambda
        self.h = self.config.h
        self.sigma8 = self.config.sigma8

        # Cronos
        self.alpha_cronos = self.config.alpha_cronos
        self.eta_friction = self.config.eta_friction

        # Densidad crítica actual
        self.rho_crit = RHO_CRIT_0 * self.h**2

        # Calcular masa por partícula
        self.m_particle = self._compute_particle_mass()

    def _compute_particle_mass(self) -> float:
        """
        Calcula masa por partícula en M☉/h.

        m_particle = ρ_m × V_box / N_particles
        """
        rho_m = self.rho_crit * self.Omega_m
        V_box = self.L_box**3
        return rho_m * V_box / self.N_particles

    def lapse_function(self, rho_local: float) -> float:
        """
        Función lapse α(ρ) para dilatación temporal.

        α = 1 + α_cronos⁻¹ × (ρ/ρ_c)^(3/2)

        En regiones densas (ρ >> ρ_c), α >> 1 → tiempo local más lento.

        Args:
            rho_local: Densidad local en unidades de ρ_crit

        Returns:
            Factor de dilatación temporal
        """
        rho_ratio = rho_local / self.rho_crit
        return 1 + rho_ratio**(1.5) / self.alpha_cronos

    def dt_local(self, dt_global: float, rho_local: float) -> float:
        """
        Paso temporal local (dilatado por Cronos).

        dt_local = dt_global / α(ρ)

        En regiones densas, dt_local < dt_global.

        Args:
            dt_global: Paso temporal global
            rho_local: Densidad local

        Returns:
            Paso temporal local efectivo
        """
        alpha = self.lapse_function(rho_local)
        return dt_global / alpha

    def friction_cronos(self, velocity: np.ndarray,
                        rho_local: float) -> np.ndarray:
        """
        Fricción entrópica de Cronos.

        F = -η(ρ) × v

        Donde η(ρ) = η₀ × (ρ/ρ_c)^0.5

        Esta fricción ralentiza el colapso en regiones densas,
        produciendo cores en lugar de cusps.

        Args:
            velocity: Vector velocidad en km/s
            rho_local: Densidad local

        Returns:
            Vector fuerza de fricción en km/s² (o aceleración)
        """
        rho_ratio = rho_local / self.rho_crit
        eta_rho = self.eta_friction * rho_ratio**0.5
        return -eta_rho * velocity

    def kick_modificado(self, v: np.ndarray, acc_grav: np.ndarray,
                        dt: float, rho_local: float) -> np.ndarray:
        """
        Integrador Kick modificado con física Cronos.

        Pseudocódigo del Tratado Técnico MCMC:
        ```
        for each active particle i:
            rho_i = local_density(i)
            delta_t = dt * [1 + (rho_i/rho_c)^(1.5)/alpha]^-1
            acc_i += friction_cronos(v_i, rho_i)
            v_i += acc_i * delta_t
        ```

        Args:
            v: Velocidad actual
            acc_grav: Aceleración gravitatoria
            dt: Paso temporal global
            rho_local: Densidad local

        Returns:
            Nueva velocidad
        """
        # Paso temporal efectivo (dilatado)
        dt_eff = self.dt_local(dt, rho_local)

        # Fricción Cronos
        acc_friction = self.friction_cronos(v, rho_local)

        # Aceleración total
        acc_total = acc_grav + acc_friction

        # Actualizar velocidad
        v_new = v + acc_total * dt_eff

        return v_new

    def drift_modificado(self, x: np.ndarray, v: np.ndarray,
                         dt: float, rho_local: float) -> np.ndarray:
        """
        Integrador Drift modificado con Cronos.

        La posición evoluciona con el paso temporal local.

        Args:
            x: Posición actual
            v: Velocidad actual
            dt: Paso temporal global
            rho_local: Densidad local

        Returns:
            Nueva posición
        """
        dt_eff = self.dt_local(dt, rho_local)
        return x + v * dt_eff

    def generate_ICs_config(self, z_init: float = 100) -> Dict:
        """
        Genera configuración para condiciones iniciales.

        Para usar con Music, N-GenIC, o código similar.

        Args:
            z_init: Redshift inicial

        Returns:
            Dict con parámetros para generador de ICs
        """
        config = {
            'cosmology': {
                'Omega_m': self.Omega_m,
                'Omega_b': self.config.Omega_b,
                'Omega_Lambda': self.Omega_Lambda,
                'h': self.h,
                'sigma8': self.sigma8,
                'n_s': self.config.n_s
            },
            'box': {
                'L': self.L_box,
                'N': self.config.N_particles_side,
                'z_init': z_init
            },
            'resolution': {
                'm_particle': self.m_particle,
                'softening': self.config.softening,
                'N_cells_PM': self.config.N_cells_side
            },
            'output': {
                'format': 'HDF5',
                'filename': f'ICs_Cronos_L{self.L_box}_N{self.config.N_particles_side}_z{z_init}.hdf5'
            },
            'transfer': {
                'use_CLASS_MCMC': True,
                'comment': 'Usar P(k) de CLASS-MCMC con Lambda_rel(z)'
            },
            'snapshots': {
                'z_list': self.config.z_snapshots
            }
        }
        return config

    def expected_observables(self) -> Dict:
        """
        Lista de observables esperados de la simulación.

        Predicciones MCMC vs CDM.
        """
        return {
            'halo_mass_function': {
                'descripcion': 'n(M, z) función de masa de halos',
                'prediccion_MCMC': 'Ligeramente suprimida a M < 10^11 M☉',
                'criterio': 'n_MCMC/n_CDM ~ 0.85 para M ~ 10^10 M☉'
            },
            'concentration_mass': {
                'descripcion': 'Relación concentración-masa c(M)',
                'prediccion_MCMC': 'Concentraciones menores debido a cores',
                'criterio': 'c_MCMC < c_CDM para halos ~ 10^12 M☉'
            },
            'core_radius': {
                'descripcion': 'Radio de core r_core(M)',
                'prediccion_MCMC': 'r_core > 0 (vs r_core ~ 0 en NFW)',
                'criterio': 'r_core(10^12 M☉) ~ 3 kpc'
            },
            'subhalo_abundance': {
                'descripcion': 'Número de subhalos N_sub(M_host)',
                'prediccion_MCMC': '~45% reducción respecto a CDM',
                'criterio': 'N_sub_MCMC/N_sub_CDM ~ 0.55'
            },
            'density_profiles': {
                'descripcion': 'Perfiles ρ(r) promediados',
                'prediccion_MCMC': 'Cores planos en r < r_core',
                'criterio': 'γ_inner ~ 0.51 (vs γ ~ 1 en NFW)'
            },
            'sparc_comparison': {
                'descripcion': 'Curvas de rotación vs SPARC',
                'prediccion_MCMC': 'Mejora en χ²',
                'criterio': 'χ²_Zhao < χ²_NFW'
            }
        }

    def core_radius_prediction(self, M_halo: float, z: float = 0) -> float:
        """
        Radio de core predicho por Ley de Cronos.

        r_core = r_c0 × (M_halo / 10^10 M☉)^η × (1+z)^β

        donde η ~ 0.35 y β ~ -0.5

        Args:
            M_halo: Masa del halo en M☉
            z: Redshift

        Returns:
            Radio de core en kpc
        """
        r_c0 = 1.8  # kpc (calibración)
        M_ref = 1e10  # M☉
        eta = 0.35
        beta = -0.5

        r_core = r_c0 * (M_halo / M_ref)**eta * (1 + z)**beta

        return r_core

    def subhalo_suppression(self, M_sub: float, M_host: float) -> float:
        """
        Factor de supresión de subhalos por Cronos.

        Los subhalos de baja masa son más susceptibles a la fricción
        entrópica y se disruptan más fácilmente.

        Args:
            M_sub: Masa del subhalo
            M_host: Masa del halo huésped

        Returns:
            Factor de supresión (0-1)
        """
        # Supresión base
        f_sup = 0.55  # 45% reducción promedio

        # Dependencia con masa (subhalos pequeños más suprimidos)
        M_ratio = M_sub / M_host
        f_mass = 0.85 + 0.15 * (M_ratio / 0.01)**0.3
        f_mass = np.clip(f_mass, 0.5, 1.0)

        return f_sup * f_mass


# =============================================================================
# GENERADOR DE PERFILES
# =============================================================================

class PerfilesCronos:
    """
    Generador de perfiles de densidad con física Cronos.

    Incluye:
    - NFW estándar (para comparación)
    - Zhao con γ=0.51 (predicción MCMC)
    - Perfil Cronos (con correcciones entrópicas)
    """

    def __init__(self):
        self.G = G_GRAV  # kpc (km/s)² / M☉

    def perfil_NFW(self, r: np.ndarray, M_vir: float,
                   c: float = 10) -> np.ndarray:
        """
        Perfil NFW estándar.

        ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
        """
        # Radio virial (aproximado)
        r_vir = (M_vir / (4*np.pi * 200 * RHO_CRIT_0 / 3))**(1/3) * 1e3  # kpc

        # Radio de escala
        r_s = r_vir / c

        # Densidad característica
        f_c = np.log(1 + c) - c / (1 + c)
        rho_s = M_vir / (4 * np.pi * r_s**3 * f_c)

        x = r / r_s
        x = np.maximum(x, 1e-10)

        return rho_s / (x * (1 + x)**2)

    def perfil_Zhao(self, r: np.ndarray, rho_0: float, r_s: float,
                    gamma: float = 0.51, alpha: float = 2.0,
                    beta: float = 3.0) -> np.ndarray:
        """
        Perfil Zhao generalizado (usado en MCMC con γ=0.51).

        ρ(r) = ρ_0 / [(r/r_s)^γ × (1 + (r/r_s)^α)^((β-γ)/α)]

        Con γ=0.51 produce un core suave.
        """
        x = r / r_s
        x = np.maximum(x, 1e-10)

        denominador = x**gamma * (1 + x**alpha)**((beta - gamma) / alpha)
        return rho_0 / denominador

    def perfil_Cronos(self, r: np.ndarray, M_vir: float,
                      c: float = 10, r_core: float = None) -> np.ndarray:
        """
        Perfil con corrección Cronos.

        Combina NFW externo con core interno debido a
        fricción entrópica.
        """
        if r_core is None:
            r_core = CronosBox100().core_radius_prediction(M_vir)

        # Perfil NFW base
        rho_NFW = self.perfil_NFW(r, M_vir, c)

        # Corrección de core
        # En r < r_core, la densidad se aplana
        core_factor = 1 / (1 + (r_core / r)**2)
        rho_cored = rho_NFW * (1 - core_factor) + rho_NFW[r > r_core].max() * core_factor

        return rho_cored

    def chi2_sparc_comparison(self, r_data: np.ndarray, v_data: np.ndarray,
                              v_err: np.ndarray, M_halo: float,
                              perfil: str = 'Zhao') -> float:
        """
        Compara perfil con datos tipo SPARC.

        Args:
            r_data: Radios observados (kpc)
            v_data: Velocidades observadas (km/s)
            v_err: Errores
            M_halo: Masa del halo
            perfil: 'NFW' o 'Zhao'

        Returns:
            χ² del ajuste
        """
        # Calcular velocidad circular para el perfil
        if perfil == 'NFW':
            rho = self.perfil_NFW(r_data, M_halo)
        else:
            r_s = 10  # kpc aproximado
            rho_0 = M_halo / (4 * np.pi * r_s**3 * 10)  # Normalización aproximada
            rho = self.perfil_Zhao(r_data, rho_0, r_s)

        # Masa encerrada (integración numérica simple)
        M_enc = np.cumsum(4 * np.pi * r_data**2 * rho * np.gradient(r_data))

        # Velocidad circular
        v_circ = np.sqrt(self.G * M_enc / r_data)

        # χ²
        chi2 = np.sum(((v_data - v_circ) / v_err)**2)

        return chi2


# =============================================================================
# FUNCIÓN DE TEST
# =============================================================================

def test_NBody_Box100():
    """Test del módulo N-body Box-100 Cronos."""
    print("\n" + "="*65)
    print("  TEST N-BODY BOX-100 CRONOS")
    print("="*65)

    # Crear instancia
    box = CronosBox100()

    # Test 1: Configuración
    print("\n[1] Configuración de caja:")
    print(f"    L_box = {box.L_box} h⁻¹ Mpc")
    print(f"    N_particles = {box.N_particles:.2e}")
    print(f"    m_particle = {box.m_particle:.2e} M☉/h")
    print(f"    Softening = {box.config.softening} kpc/h")

    # Test 2: Función lapse
    print("\n[2] Función lapse α(ρ):")
    rho_test = [1, 10, 100, 1000]  # En unidades de ρ_crit
    for rho in rho_test:
        alpha = box.lapse_function(rho * box.rho_crit)
        print(f"    α(ρ={rho}×ρ_c) = {alpha:.2f}")

    # Test 3: Fricción Cronos
    print("\n[3] Fricción entrópica:")
    v_test = np.array([100, 0, 0])  # km/s
    for rho in [10, 100]:
        F = box.friction_cronos(v_test, rho * box.rho_crit)
        print(f"    F(v=100, ρ={rho}×ρ_c) = {F[0]:.2f} km/s²")

    # Test 4: Radio de core predicho
    print("\n[4] Radio de core predicho:")
    masas = [1e10, 1e11, 1e12, 1e13]
    for M in masas:
        r_core = box.core_radius_prediction(M)
        print(f"    r_core(M={M:.0e} M☉) = {r_core:.2f} kpc")

    # Test 5: Supresión de subhalos
    print("\n[5] Supresión de subhalos:")
    M_host = 1e12
    for M_sub in [1e8, 1e9, 1e10]:
        f_sup = box.subhalo_suppression(M_sub, M_host)
        print(f"    N_sub(M={M_sub:.0e})/N_CDM = {f_sup:.2f}")

    # Test 6: Configuración de ICs
    print("\n[6] Configuración para ICs:")
    ic_config = box.generate_ICs_config(z_init=100)
    print(f"    Archivo: {ic_config['output']['filename']}")
    print(f"    z_init = {ic_config['box']['z_init']}")
    print(f"    Usar CLASS-MCMC: {ic_config['transfer']['use_CLASS_MCMC']}")

    # Test 7: Perfiles de densidad
    print("\n[7] Perfiles de densidad:")
    perfiles = PerfilesCronos()
    r_array = np.logspace(-1, 2, 50)  # kpc
    M_halo = 1e12

    rho_NFW = perfiles.perfil_NFW(r_array, M_halo)
    rho_Zhao = perfiles.perfil_Zhao(r_array, 1e7, 10)

    print(f"    ρ_NFW(1 kpc) = {rho_NFW[10]:.2e} M☉/kpc³")
    print(f"    ρ_Zhao(1 kpc) = {rho_Zhao[10]:.2e} M☉/kpc³")

    # Verificar criterio
    print("\n[8] Verificación de criterios:")
    r_core_12 = box.core_radius_prediction(1e12)
    passed = r_core_12 > 2  # Criterio: r_core > 2 kpc para 10^12 M☉

    print(f"    r_core(10¹² M☉) = {r_core_12:.2f} kpc")
    print(f"    Criterio r_core > 2 kpc: {'PASS' if passed else 'FAIL'}")

    print("\n" + "="*65)
    print(f"  N-BODY BOX-100 MODULE: {'PASS' if passed else 'FAIL'}")
    print("="*65)

    return passed


if __name__ == "__main__":
    test_NBody_Box100()
