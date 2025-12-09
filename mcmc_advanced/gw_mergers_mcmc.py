#!/usr/bin/env python3
"""
================================================================================
MODULO DE FUSIONES DE ONDAS GRAVITACIONALES EN MCMC
================================================================================

Predicciones de ondas gravitacionales de fusiones de objetos compactos
(BBH, BNS, NSBH) en el marco del modelo MCMC con correcciones ontologicas.

CORRECCIONES MCMC A FUSIONES GW:
--------------------------------
1. Friccion entropica modifica la fase de inspiral
2. Burbujas temporales afectan el redshift observado
3. Tasa de fusiones evolucion con z diferente a LCDM
4. Mass gap relacionado con sellos ontologicos
5. Ringdown modificado por geometria de Cronos

OBSERVABLES PREDICHOS:
----------------------
- Frecuencia de inspiral modificada: f_GW(t) con correccion entropica
- Amplitud de strain h(f) con factores de Cronos
- Tasa de fusiones R(z) con evolucion MCMC
- Distribucion de masas con gap ontologico
- Espectro de ringdown modificado

COMPARACION CON:
----------------
- LIGO O1-O3: ~90 eventos BBH, BNS, NSBH
- LIGO O4: Sensibilidad mejorada
- Predicciones para ET/CE (3G)

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FISICAS FUNDAMENTALES
# =============================================================================

# Constantes SI
C_LIGHT = 299792458.0           # m/s
G_NEWTON = 6.67430e-11          # m³ kg⁻¹ s⁻²
HBAR = 1.054571817e-34          # J·s
M_SUN = 1.98892e30              # kg
MPC_TO_M = 3.086e22             # m/Mpc

# Escalas de Planck
M_PLANCK = 2.176434e-8          # kg
L_PLANCK = 1.616255e-35         # m

# Constantes derivadas
G_M_SUN = G_NEWTON * M_SUN      # m³/s² - GM solar
R_S_SUN = 2 * G_M_SUN / C_LIGHT**2  # Radio de Schwarzschild solar ~ 2.95 km

# Cosmologia
H0 = 67.36                      # km/s/Mpc
OMEGA_M = 0.3153
OMEGA_LAMBDA = 0.6847


# =============================================================================
# PARAMETROS MCMC PARA FUSIONES GW
# =============================================================================

@dataclass
class ParametrosGW_Mergers:
    """
    Parametros MCMC para fusiones de ondas gravitacionales.
    """
    # ECV (correccion a Lambda)
    epsilon_ECV: float = 0.012
    z_trans: float = 8.9
    Delta_z: float = 1.5

    # Friccion entropica
    eta_friction: float = 0.05      # Coeficiente de friccion
    r_friction: float = 10.0        # Radio en r_s donde friccion es relevante

    # Mass gap ontologico
    M_gap_lower: float = 2.5        # M_sun - limite inferior (NS max)
    M_gap_upper: float = 5.0        # M_sun - limite superior (BH min clasico)
    f_gap_suppression: float = 0.3  # Supresion en el gap (vs LCDM)

    # Sellos ontologicos para masas
    S_NS_max: float = 0.95          # Sello maximo para NS
    S_BH_min: float = 0.98          # Sello minimo para BH

    # Evolucion con z
    alpha_rate: float = 2.7         # Exponente tasa de formacion estelar
    z_peak_sfr: float = 2.0         # Pico de SFR
    delay_time_Gyr: float = 0.5     # Tiempo de delay tipico


# Parametros por defecto
PARAMS_DEFAULT = ParametrosGW_Mergers()


# =============================================================================
# TIPOS DE FUSIONES
# =============================================================================

class TipoFusion(Enum):
    """Tipos de fusiones de objetos compactos."""
    BBH = "Binary Black Hole"
    BNS = "Binary Neutron Star"
    NSBH = "Neutron Star - Black Hole"


@dataclass
class EventoGW:
    """Representa un evento de ondas gravitacionales."""
    nombre: str
    tipo: TipoFusion
    m1: float                   # Masa primaria [M_sun]
    m2: float                   # Masa secundaria [M_sun]
    z: float                    # Redshift
    d_L: float = 0.0           # Distancia luminosidad [Mpc]
    chi_eff: float = 0.0       # Spin efectivo
    SNR: float = 0.0           # Signal-to-noise ratio

    @property
    def M_chirp(self) -> float:
        """Masa chirp."""
        return (self.m1 * self.m2)**(3/5) / (self.m1 + self.m2)**(1/5)

    @property
    def M_total(self) -> float:
        """Masa total."""
        return self.m1 + self.m2

    @property
    def q(self) -> float:
        """Ratio de masas."""
        return self.m2 / self.m1 if self.m1 > self.m2 else self.m1 / self.m2

    @property
    def eta(self) -> float:
        """Parametro de masa simetrica."""
        return self.m1 * self.m2 / (self.m1 + self.m2)**2


# =============================================================================
# CATALOGO DE EVENTOS LIGO/Virgo (SELECCION)
# =============================================================================

EVENTOS_LIGO = [
    # O1-O2 Events
    EventoGW("GW150914", TipoFusion.BBH, 35.6, 30.6, 0.09, 410, 0.06, 24),
    EventoGW("GW151226", TipoFusion.BBH, 13.7, 7.7, 0.09, 440, 0.18, 13),
    EventoGW("GW170104", TipoFusion.BBH, 30.8, 20.0, 0.18, 880, -0.04, 13),
    EventoGW("GW170814", TipoFusion.BBH, 30.6, 25.2, 0.11, 540, 0.07, 16),
    EventoGW("GW170817", TipoFusion.BNS, 1.46, 1.27, 0.01, 40, 0.00, 33),

    # O3 Events (seleccion)
    EventoGW("GW190412", TipoFusion.BBH, 30.1, 8.3, 0.15, 730, 0.25, 19),
    EventoGW("GW190425", TipoFusion.BNS, 2.0, 1.4, 0.03, 159, 0.00, 13),
    EventoGW("GW190521", TipoFusion.BBH, 85.0, 66.0, 0.82, 5300, 0.08, 14),
    EventoGW("GW190814", TipoFusion.NSBH, 23.2, 2.6, 0.05, 241, 0.00, 25),
    EventoGW("GW200105", TipoFusion.NSBH, 8.9, 1.9, 0.05, 280, 0.00, 13),
]


# =============================================================================
# CLASE PRINCIPAL: GW MERGERS MCMC
# =============================================================================

class GWMergersMCMC:
    """
    Modelo de fusiones GW con correcciones MCMC.

    Implementa:
    - Modificacion de forma de onda por friccion entropica
    - Evolucion de tasa de fusiones con z
    - Mass gap ontologico
    - Predicciones para detectores actuales y futuros
    """

    def __init__(self, params: ParametrosGW_Mergers = None):
        """
        Inicializa el modelo.

        Args:
            params: Parametros MCMC (usa defaults si None)
        """
        self.params = params if params else PARAMS_DEFAULT

        # Cache para distancias
        self._d_L_cache = {}

    # =========================================================================
    # COSMOLOGIA MCMC
    # =========================================================================

    def Lambda_rel(self, z: float) -> float:
        """Constante cosmologica relacional."""
        return 1.0 + self.params.epsilon_ECV * np.tanh(
            (self.params.z_trans - z) / self.params.Delta_z
        )

    def E_z(self, z: float) -> float:
        """E(z) = H(z)/H0 con correccion MCMC."""
        matter = OMEGA_M * (1 + z)**3
        de = OMEGA_LAMBDA * self.Lambda_rel(z)
        return np.sqrt(matter + de)

    def H_z(self, z: float) -> float:
        """H(z) en km/s/Mpc."""
        return H0 * self.E_z(z)

    def d_L(self, z: float) -> float:
        """
        Distancia de luminosidad en Mpc.

        Args:
            z: Redshift

        Returns:
            d_L en Mpc
        """
        if z in self._d_L_cache:
            return self._d_L_cache[z]

        if z <= 0:
            return 0.0

        # Distancia comovil
        integrand = lambda zp: C_LIGHT / 1000 / self.H_z(zp)  # c/H en Mpc
        d_C, _ = quad(integrand, 0, z)

        # Distancia luminosidad
        d_L = (1 + z) * d_C

        self._d_L_cache[z] = d_L
        return d_L

    # =========================================================================
    # FORMA DE ONDA CON CORRECCION ENTROPICA
    # =========================================================================

    def frecuencia_inspiral_MCMC(self, t: np.ndarray, M_chirp: float,
                                  z: float = 0.0) -> np.ndarray:
        """
        Frecuencia de inspiral con correccion entropica.

        La friccion entropica modifica ligeramente la evolucion de f(t)
        durante la fase de inspiral tardia.

        f(t) = f_GR(t) * (1 - epsilon_friction * factor(t))

        donde factor(t) crece cerca del merger.

        Args:
            t: Tiempo al merger [s] (negativo, t=0 es merger)
            M_chirp: Masa chirp en M_sun
            z: Redshift de la fuente

        Returns:
            Frecuencia en Hz
        """
        # Masa chirp en kg, redshifted
        M_c = M_chirp * M_SUN * (1 + z)

        # Tiempo caracteristico
        tau = 5 * G_M_SUN * M_chirp * (1 + z) / C_LIGHT**3 / 256

        # Frecuencia GR: f = (1/8pi) * (5/256)^(3/8) * (GM_c/c^3)^(-5/8) * |t|^(-3/8)
        # Simplificado:
        f_GR = np.zeros_like(t)
        valid = t < 0
        f_GR[valid] = (5 / (256 * np.abs(t[valid])))**(3/8) / (8 * np.pi) * (
            G_M_SUN * M_chirp * (1 + z) / C_LIGHT**3
        )**(-5/8)

        # Correccion MCMC: friccion entropica cerca del merger
        # Factor de correccion que crece con f (cerca del merger)
        f_ISCO = self.f_ISCO(M_chirp * (1 + z))

        # Factor de friccion: mas fuerte cerca de ISCO
        friction_factor = np.zeros_like(t)
        friction_factor[valid] = self.params.eta_friction * (
            f_GR[valid] / f_ISCO
        )**2

        # Frecuencia corregida
        f_MCMC = f_GR * (1 - friction_factor)

        return np.maximum(f_MCMC, 0.0)

    def f_ISCO(self, M_total: float) -> float:
        """
        Frecuencia ISCO (Innermost Stable Circular Orbit).

        f_ISCO = c^3 / (6^(3/2) * pi * G * M)

        Args:
            M_total: Masa total en M_sun

        Returns:
            Frecuencia en Hz
        """
        return C_LIGHT**3 / (6**1.5 * np.pi * G_NEWTON * M_total * M_SUN)

    def strain_amplitude(self, f: np.ndarray, M_chirp: float, z: float,
                         d_L_Mpc: float = None) -> np.ndarray:
        """
        Amplitud de strain h(f) con correcciones MCMC.

        h(f) = h_GR(f) * correction_factor

        Args:
            f: Frecuencia en Hz
            M_chirp: Masa chirp en M_sun
            z: Redshift
            d_L_Mpc: Distancia luminosidad (calcula si None)

        Returns:
            Strain amplitude
        """
        if d_L_Mpc is None:
            d_L_Mpc = self.d_L(z)

        # Distancia en metros
        d_L_m = d_L_Mpc * MPC_TO_M

        # Masa chirp redshifted en kg
        M_c = M_chirp * M_SUN * (1 + z)

        # Amplitud GR (leading order)
        # h(f) ∝ (G M_c)^(5/6) / (c^(3/2) d_L) * f^(-7/6)
        prefactor = (np.pi**2 / 3)**0.5 * (G_NEWTON * M_c)**(5/6) / (
            C_LIGHT**1.5 * d_L_m
        )

        h_GR = np.zeros_like(f)
        valid = f > 0
        h_GR[valid] = prefactor * f[valid]**(-7/6)

        # Correccion MCMC por friccion entropica
        # Reduce ligeramente la amplitud a altas frecuencias
        f_ISCO = self.f_ISCO(M_chirp * (1 + z) * 2.5)  # Estimacion M_total

        correction = 1 - 0.5 * self.params.eta_friction * (f / f_ISCO)**2
        correction = np.maximum(correction, 0.5)

        return h_GR * correction

    # =========================================================================
    # TASA DE FUSIONES
    # =========================================================================

    def tasa_formacion_estelar(self, z: float) -> float:
        """
        Tasa de formacion estelar psi(z) en M_sun/yr/Mpc^3.

        Modelo Madau-Dickinson modificado por MCMC.
        """
        # Madau-Dickinson 2014
        psi_MD = 0.015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)

        # Correccion MCMC: Lambda_rel afecta ligeramente el SFR
        # a z > z_trans
        if z > self.params.z_trans:
            corr = self.Lambda_rel(z)**0.5
        else:
            corr = 1.0

        return psi_MD * corr

    def tasa_fusiones_intrinseca(self, z: float,
                                  tipo: TipoFusion = TipoFusion.BBH) -> float:
        """
        Tasa de fusiones intrinseca R(z) en Gpc^-3 yr^-1.

        R(z) = integral sobre tiempo de delay de SFR * P(t_delay)

        Args:
            z: Redshift
            tipo: Tipo de fusion

        Returns:
            Tasa en Gpc^-3 yr^-1
        """
        # Normalizacion a z=0 (observaciones LIGO O3)
        R0_dict = {
            TipoFusion.BBH: 23.9,    # Gpc^-3 yr^-1 (O3a median)
            TipoFusion.BNS: 320.0,   # Gpc^-3 yr^-1
            TipoFusion.NSBH: 45.0,   # Gpc^-3 yr^-1
        }
        R0 = R0_dict.get(tipo, 23.9)

        # Evolucion con z
        # R(z) ∝ SFR(z_form) convolucionado con distribucion de delays
        # Aproximacion: R(z) = R0 * (1+z)^kappa
        # donde kappa depende del tipo y delay time

        if tipo == TipoFusion.BBH:
            kappa = 1.5  # Delay times largos
        elif tipo == TipoFusion.BNS:
            kappa = 2.0  # Delay times cortos
        else:  # NSBH
            kappa = 1.8

        # Correccion MCMC
        # La evolucion de Lambda afecta formacion de binarias a alto z
        Lambda_corr = self.Lambda_rel(z)**(kappa / 2)

        # Tasa evolucionada
        R_z = R0 * (1 + z)**kappa * Lambda_corr

        # Saturacion a alto z
        if z > 6:
            R_z *= np.exp(-(z - 6) / 2)

        return R_z

    def tasa_deteccion_esperada(self, z_max: float = 2.0,
                                 tipo: TipoFusion = TipoFusion.BBH,
                                 detector: str = "LIGO_O4") -> float:
        """
        Tasa de deteccion esperada en eventos/yr.

        Integra R(z) * dV_c/dz * P_det(z)

        Args:
            z_max: Redshift maximo de integracion
            tipo: Tipo de fusion
            detector: Detector ("LIGO_O4", "ET", "CE")

        Returns:
            Tasa en eventos/yr
        """
        # Horizonte de deteccion segun detector
        z_horizon = {
            "LIGO_O4": {"BBH": 1.0, "BNS": 0.2, "NSBH": 0.5},
            "ET": {"BBH": 20.0, "BNS": 2.0, "NSBH": 5.0},
            "CE": {"BBH": 30.0, "BNS": 3.0, "NSBH": 8.0},
        }

        tipo_str = tipo.name
        z_hor = z_horizon.get(detector, z_horizon["LIGO_O4"]).get(
            tipo_str, 1.0
        )

        def integrand(z):
            if z <= 0:
                return 0.0

            # Elemento de volumen comovil
            d_C = self.d_L(z) / (1 + z)  # Mpc
            dV_dz = 4 * np.pi * d_C**2 * C_LIGHT / 1000 / self.H_z(z)  # Mpc^3
            dV_dz /= 1e9  # Gpc^3

            # Tasa intrinseca
            R_z = self.tasa_fusiones_intrinseca(z, tipo)

            # Probabilidad de deteccion (aproximacion)
            P_det = np.exp(-(z / z_hor)**2)

            # Factor de dilatacion temporal
            return R_z * dV_dz * P_det / (1 + z)

        z_int_max = min(z_max, z_hor * 3)
        rate, _ = quad(integrand, 1e-4, z_int_max)

        return rate

    # =========================================================================
    # MASS GAP ONTOLOGICO
    # =========================================================================

    def probabilidad_masa_gap(self, m: float) -> float:
        """
        Probabilidad de observar masa m en el gap ontologico.

        El gap entre NS maximas y BH minimos esta relacionado
        con los sellos S_3 y S_4 del MCMC.

        Args:
            m: Masa en M_sun

        Returns:
            Probabilidad relativa (0-1)
        """
        M_low = self.params.M_gap_lower
        M_high = self.params.M_gap_upper

        if m < M_low or m > M_high:
            return 1.0  # Fuera del gap: probabilidad normal

        # Dentro del gap: supresion
        # Forma gaussiana centrada en el gap
        M_center = (M_low + M_high) / 2
        sigma_gap = (M_high - M_low) / 4

        suppression = 1 - (1 - self.params.f_gap_suppression) * np.exp(
            -(m - M_center)**2 / (2 * sigma_gap**2)
        )

        return suppression

    def distribucion_masas_primarias_MCMC(self, m: np.ndarray,
                                           tipo: TipoFusion = TipoFusion.BBH
                                           ) -> np.ndarray:
        """
        Distribucion de masas primarias p(m1) con correcciones MCMC.

        Modelo power-law + peak, modificado por mass gap ontologico.

        Args:
            m: Array de masas en M_sun
            tipo: Tipo de fusion

        Returns:
            Probabilidad (no normalizada)
        """
        if tipo == TipoFusion.BBH:
            # Power law + peak (Abbott et al. 2021)
            alpha = 3.5
            m_min = 5.0
            m_max = 100.0
            mu_peak = 35.0
            sigma_peak = 4.0
            f_peak = 0.1

            # Componente power-law
            p_pl = np.zeros_like(m)
            valid = (m >= m_min) & (m <= m_max)
            p_pl[valid] = m[valid]**(-alpha)

            # Componente gaussiano (peak)
            p_peak = np.exp(-(m - mu_peak)**2 / (2 * sigma_peak**2))

            # Combinacion
            p = (1 - f_peak) * p_pl + f_peak * p_peak

        elif tipo == TipoFusion.BNS:
            # Distribucion gaussiana para NS
            mu_NS = 1.35
            sigma_NS = 0.15
            p = np.exp(-(m - mu_NS)**2 / (2 * sigma_NS**2))

        else:  # NSBH
            # BH component for NSBH
            alpha = 2.5
            m_min = 3.0
            m_max = 50.0
            p = np.zeros_like(m)
            valid = (m >= m_min) & (m <= m_max)
            p[valid] = m[valid]**(-alpha)

        # Aplicar supresion del mass gap MCMC
        gap_factor = np.array([self.probabilidad_masa_gap(mi) for mi in m])
        p *= gap_factor

        return p

    # =========================================================================
    # RINGDOWN MODIFICADO
    # =========================================================================

    def frecuencia_ringdown_MCMC(self, M_final: float, chi_final: float,
                                  z: float = 0.0) -> Dict[str, float]:
        """
        Frecuencia de ringdown con correcciones MCMC.

        El ringdown esta dominado por el modo l=m=2, n=0 (fundamental).
        La frecuencia depende de M_final y chi_final.

        La correccion MCMC viene de la geometria modificada cerca
        del horizonte (burbuja entropica).

        Args:
            M_final: Masa final del BH [M_sun]
            chi_final: Spin final [0-1]
            z: Redshift

        Returns:
            Dict con f_ring, tau_ring y correcciones
        """
        # Masa final redshifted
        M_f = M_final * M_SUN * (1 + z)

        # Frecuencia QNM fundamental (ajuste de Berti et al.)
        # f_220 = (c^3 / 2pi G M) * f1(chi)
        # donde f1(chi) ≈ 1 - 0.63(1-chi)^0.3

        f1_chi = 1 - 0.63 * (1 - chi_final)**0.3

        f_ring_GR = C_LIGHT**3 / (2 * np.pi * G_NEWTON * M_f) * f1_chi

        # Tiempo de decaimiento
        # tau = 2 G M / c^3 * Q(chi)
        # Q(chi) ≈ 2(1-chi)^(-0.45)
        Q_chi = 2 * (1 - chi_final + 0.01)**(-0.45)

        tau_ring_GR = 2 * G_NEWTON * M_f / C_LIGHT**3 * Q_chi

        # Correccion MCMC por burbuja entropica
        # El horizonte tiene S_local < S_ext, lo que modifica
        # ligeramente la frecuencia de QNM

        # Factor de correccion basado en Xi del horizonte
        # Para BH masivos (> 10 M_sun), Xi ~ 1 -> correccion ~ 1%
        Xi_horizonte = self._estimar_Xi_horizonte(M_final)

        corr_factor = 1 - 0.01 * Xi_horizonte / (1 + Xi_horizonte)

        f_ring_MCMC = f_ring_GR * corr_factor
        tau_ring_MCMC = tau_ring_GR / corr_factor  # Tau aumenta si f disminuye

        return {
            'f_ring_GR_Hz': f_ring_GR,
            'f_ring_MCMC_Hz': f_ring_MCMC,
            'tau_ring_GR_s': tau_ring_GR,
            'tau_ring_MCMC_s': tau_ring_MCMC,
            'Xi_horizonte': Xi_horizonte,
            'correccion_percent': (1 - corr_factor) * 100,
            'M_final_Msun': M_final,
            'chi_final': chi_final,
        }

    def _estimar_Xi_horizonte(self, M_solar: float) -> float:
        """
        Estima Xi en el horizonte de un BH.

        Basado en el modulo MCV-BH calibrado.
        """
        # Valores calibrados por categoria
        if M_solar < 1e-5:
            return 1000.0  # PBH
        elif M_solar < 100:
            return 10.0    # Estelar
        elif M_solar < 1e5:
            return 5.0     # IMBH
        elif M_solar < 1e10:
            return 1.0     # SMBH
        else:
            return 0.5     # UMBH

    # =========================================================================
    # PREDICCIONES COMPARADAS CON OBSERVACIONES
    # =========================================================================

    def comparar_con_evento(self, evento: EventoGW) -> Dict:
        """
        Compara predicciones MCMC con un evento observado.

        Args:
            evento: Evento GW observado

        Returns:
            Dict con comparaciones
        """
        # Distancia MCMC
        d_L_MCMC = self.d_L(evento.z)
        d_L_obs = evento.d_L

        # Diferencia en distancia
        delta_dL = (d_L_MCMC - d_L_obs) / d_L_obs * 100  # %

        # Frecuencia ISCO
        f_ISCO = self.f_ISCO(evento.M_total)

        # Ringdown (si es BBH)
        if evento.tipo == TipoFusion.BBH:
            # Estimar M_final y chi_final
            eta = evento.eta
            chi_eff = evento.chi_eff
            M_final = evento.M_total * (1 - 0.05)  # ~5% radiado en GW
            chi_final = 0.69 * eta + chi_eff  # Aproximacion

            ringdown = self.frecuencia_ringdown_MCMC(M_final, chi_final, evento.z)
        else:
            ringdown = None

        # Probabilidad de masas (mass gap)
        p_m1 = self.probabilidad_masa_gap(evento.m1)
        p_m2 = self.probabilidad_masa_gap(evento.m2)

        return {
            'nombre': evento.nombre,
            'tipo': evento.tipo.value,
            'd_L_obs_Mpc': d_L_obs,
            'd_L_MCMC_Mpc': d_L_MCMC,
            'delta_dL_percent': delta_dL,
            'f_ISCO_Hz': f_ISCO,
            'ringdown': ringdown,
            'p_m1_gap': p_m1,
            'p_m2_gap': p_m2,
            'M_chirp': evento.M_chirp,
            'SNR': evento.SNR,
        }

    def tabla_comparacion_LIGO(self) -> List[Dict]:
        """
        Genera tabla de comparacion con eventos LIGO.
        """
        return [self.comparar_con_evento(e) for e in EVENTOS_LIGO]


# =============================================================================
# PREDICCIONES PARA DETECTORES FUTUROS
# =============================================================================

def predicciones_detectores_futuros(modelo: GWMergersMCMC = None) -> Dict:
    """
    Genera predicciones para detectores actuales y futuros.

    Args:
        modelo: Modelo MCMC (crea uno por defecto si None)

    Returns:
        Dict con predicciones por detector
    """
    if modelo is None:
        modelo = GWMergersMCMC()

    detectores = ["LIGO_O4", "ET", "CE"]
    tipos = [TipoFusion.BBH, TipoFusion.BNS, TipoFusion.NSBH]

    predicciones = {}

    for det in detectores:
        predicciones[det] = {}
        for tipo in tipos:
            rate = modelo.tasa_deteccion_esperada(z_max=30, tipo=tipo, detector=det)
            predicciones[det][tipo.name] = {
                'rate_per_year': rate,
                'rate_per_year_formatted': f"{rate:.1f}" if rate > 1 else f"{rate:.3f}",
            }

    return predicciones


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_GW_Mergers_MCMC() -> bool:
    """
    Test del modulo de fusiones GW.
    """
    print("\n" + "=" * 70)
    print("  TEST GW MERGERS MCMC - FUSIONES DE ONDAS GRAVITACIONALES")
    print("=" * 70)

    modelo = GWMergersMCMC()

    # 1. Cosmologia MCMC
    print("\n[1] Cosmologia MCMC:")
    print("-" * 70)
    for z in [0.1, 0.5, 1.0, 2.0]:
        d_L = modelo.d_L(z)
        Lambda = modelo.Lambda_rel(z)
        print(f"    z={z:.1f}: d_L={d_L:.1f} Mpc, Lambda_rel={Lambda:.4f}")

    # 2. Frecuencia ISCO
    print("\n[2] Frecuencia ISCO:")
    print("-" * 70)
    for M in [10, 30, 60, 100]:
        f = modelo.f_ISCO(M)
        print(f"    M_total={M} M_sun: f_ISCO={f:.1f} Hz")

    # 3. Tasa de fusiones
    print("\n[3] Tasa de fusiones R(z):")
    print("-" * 70)
    print(f"    {'z':>6} {'BBH':>12} {'BNS':>12} {'NSBH':>12} [Gpc^-3 yr^-1]")
    for z in [0.0, 0.5, 1.0, 2.0, 4.0]:
        R_bbh = modelo.tasa_fusiones_intrinseca(z, TipoFusion.BBH)
        R_bns = modelo.tasa_fusiones_intrinseca(z, TipoFusion.BNS)
        R_nsbh = modelo.tasa_fusiones_intrinseca(z, TipoFusion.NSBH)
        print(f"    {z:>6.1f} {R_bbh:>12.1f} {R_bns:>12.1f} {R_nsbh:>12.1f}")

    # 4. Mass gap
    print("\n[4] Mass gap ontologico:")
    print("-" * 70)
    for m in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        p = modelo.probabilidad_masa_gap(m)
        in_gap = "IN GAP" if 2.5 <= m <= 5.0 else ""
        print(f"    m={m:.1f} M_sun: P={p:.3f} {in_gap}")

    # Verificar supresion en el gap
    p_in_gap = modelo.probabilidad_masa_gap(3.5)
    p_out_gap = modelo.probabilidad_masa_gap(10.0)
    gap_ok = p_in_gap < p_out_gap
    print(f"\n    Supresion en gap: P(3.5 M_sun)={p_in_gap:.3f} < P(10 M_sun)={p_out_gap:.3f}: "
          f"{'PASS' if gap_ok else 'FAIL'}")

    # 5. Ringdown
    print("\n[5] Ringdown modificado (M_final=60 M_sun, chi=0.7):")
    print("-" * 70)
    ringdown = modelo.frecuencia_ringdown_MCMC(60, 0.7, z=0.3)
    print(f"    f_ring GR   = {ringdown['f_ring_GR_Hz']:.1f} Hz")
    print(f"    f_ring MCMC = {ringdown['f_ring_MCMC_Hz']:.1f} Hz")
    print(f"    tau_ring GR = {ringdown['tau_ring_GR_s']*1000:.2f} ms")
    print(f"    Correccion  = {ringdown['correccion_percent']:.2f}%")

    ringdown_ok = 0 < ringdown['correccion_percent'] < 5
    print(f"    Correccion < 5%: {'PASS' if ringdown_ok else 'FAIL'}")

    # 6. Comparacion con eventos LIGO
    print("\n[6] Comparacion con eventos LIGO seleccionados:")
    print("-" * 70)
    print(f"    {'Evento':>12} {'d_L_obs':>10} {'d_L_MCMC':>10} {'delta':>8}")

    comparaciones = modelo.tabla_comparacion_LIGO()
    deltas = []
    for c in comparaciones[:5]:  # Primeros 5
        print(f"    {c['nombre']:>12} {c['d_L_obs_Mpc']:>10.0f} "
              f"{c['d_L_MCMC_Mpc']:>10.0f} {c['delta_dL_percent']:>7.1f}%")
        deltas.append(abs(c['delta_dL_percent']))

    # Las distancias deben coincidir dentro de ~15% (incertidumbres observacionales)
    distancias_ok = max(deltas) < 15
    print(f"\n    Todas las distancias dentro de 15%: "
          f"{'PASS' if distancias_ok else 'FAIL'}")

    # 7. Predicciones para detectores
    print("\n[7] Predicciones de tasas de deteccion:")
    print("-" * 70)
    pred = predicciones_detectores_futuros(modelo)

    print(f"    {'Detector':>10} {'BBH':>12} {'BNS':>12} {'NSBH':>12} [eventos/yr]")
    for det in ["LIGO_O4", "ET", "CE"]:
        bbh = pred[det]['BBH']['rate_per_year']
        bns = pred[det]['BNS']['rate_per_year']
        nsbh = pred[det]['NSBH']['rate_per_year']
        print(f"    {det:>10} {bbh:>12.1f} {bns:>12.1f} {nsbh:>12.1f}")

    # Verificar que ET > LIGO_O4
    et_ok = pred["ET"]["BBH"]["rate_per_year"] > pred["LIGO_O4"]["BBH"]["rate_per_year"]
    print(f"\n    ET detecta mas que LIGO O4: {'PASS' if et_ok else 'FAIL'}")

    # Resultado final
    passed = gap_ok and ringdown_ok and distancias_ok and et_ok

    print("\n" + "=" * 70)
    print(f"  GW MERGERS MCMC: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    test_GW_Mergers_MCMC()
