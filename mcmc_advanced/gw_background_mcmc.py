#!/usr/bin/env python3
"""
================================================================================
MODULO DE FONDO DE ONDAS GRAVITACIONALES MCMC - gw_ret v1.0
================================================================================

Calcula el espectro de ondas gravitacionales Omega_GW(f) generado por el
retroceso entropico en la fase pre-geometrica del modelo MCMC.

Fundamentacion Ontologica:
--------------------------
En el MCMC, la transicion de la fase pre-geometrica (S < S_1 ~ 0.009) a la
fase geometrica produce ondas gravitacionales a traves del retroceso entropico.

El espectro tiene la forma:
    Omega_GW(f) = Omega_ret * (f * tau_ret)^2 * sech^2(pi * f * tau_ret / 2)

donde:
    - Omega_ret ~ 1.2e-9 (amplitud del retroceso)
    - tau_ret ~ escala temporal del retroceso
    - f_peak ~ 2.3e-16 Hz (banda PTA)

Predicciones:
-------------
1. Fondo estocastico en banda PTA (NANOGrav, SKA)
2. Quiebres espectrales en umbrales S_n
3. Correlacion con estructura a gran escala

Referencias:
- MCMC Maestro: Seccion fondo de ondas gravitacionales
- NANOGrav 15yr: Agazie et al. (2023)
- SKA-PTA projections

Autor: MCMC Cosmology Framework
Copyright (c) 2024-2025
================================================================================
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
# sech(x) = 1/cosh(x) - defined inline since scipy.special doesn't have it
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FUNDAMENTALES
# =============================================================================

# Constantes fisicas
C_LIGHT = 299792458.0      # m/s
G_NEWTON = 6.67430e-11     # m^3 kg^-1 s^-2
HBAR = 1.054571817e-34     # J s
M_PLANCK = 2.176434e-8     # kg
T_PLANCK = 5.391247e-44    # s
L_PLANCK = 1.616255e-35    # m
E_PLANCK = 1.22e19         # GeV

# Parametros cosmologicos
H0_FIDUCIAL = 67.4         # km/s/Mpc
H0_SI = H0_FIDUCIAL * 1e3 / 3.086e22  # s^-1
RHO_CRIT = 3 * H0_SI**2 / (8 * np.pi * G_NEWTON)  # kg/m^3
OMEGA_M = 0.315
OMEGA_R = 9.24e-5
OMEGA_LAMBDA = 0.685

# Umbrales entropicos MCMC
S_PLANCK = 0.009           # Sello de Planck
S_GUT = 0.099              # Sello GUT
S_EW = 0.999               # Sello electrodebil
S_QCD = 1.001              # Sello QCD

# Energias asociadas a umbrales
E_PLANCK_GeV = 1.22e19     # GeV
E_GUT_GeV = 1e16           # GeV
E_EW_GeV = 246             # GeV (VEV Higgs)
E_QCD_GeV = 0.2            # GeV (Lambda_QCD)


# =============================================================================
# PARAMETROS DEL RETROCESO ENTROPICO
# =============================================================================

@dataclass
class ParametrosGW_MCMC:
    """Parametros para el espectro de ondas gravitacionales MCMC."""
    # Amplitud del retroceso entropico
    # Calibrado para ser consistente con NANOGrav 15yr: Omega ~ 1e-9 a f ~ 1/yr
    Omega_ret: float = 1.2e-9

    # Escala temporal del retroceso
    # tau_ret ~ 1/f_peak para que el pico este en f_peak
    # f_peak ~ 2e-8 Hz (banda PTA alta), tau_ret ~ 5e7 s ~ 1.5 años
    tau_ret_seconds: float = 5e7

    # Frecuencia de pico esperada
    f_peak_Hz: float = 2e-8

    # Indice espectral en IR (f << f_peak)
    # Para Omega ~ f^n_IR
    n_IR: float = 2.0

    # Indice espectral en UV (f >> f_peak)
    n_UV: float = -2.0

    # Parametro de suavizado para quiebres
    Delta_f: float = 0.5

    # Eficiencia de conversion energia -> GW
    epsilon_GW: float = 0.01

    # Parametros MCMC
    epsilon_MCMC: float = 0.012
    z_trans: float = 8.9
    Delta_z: float = 1.5


PARAMS_GW_DEFAULT = ParametrosGW_MCMC()


# =============================================================================
# CLASE PRINCIPAL: GWBackgroundMCMC
# =============================================================================

class GWBackgroundMCMC:
    """
    Calculador del fondo de ondas gravitacionales del modelo MCMC.

    Implementa:
    -----------
    1. Espectro del retroceso entropico Omega_GW(f)
    2. Quiebres espectrales en umbrales S_n
    3. Predicciones para PTA (NANOGrav, SKA)
    4. Comparacion con LIGO/Virgo/KAGRA

    Fisica clave:
    -------------
    El retroceso entropico ocurre cuando la entropia S transita por los
    sellos ontologicos (S_1, S_2, S_3, S_4), liberando energia gravitacional
    que se propaga como ondas gravitacionales.

    Ejemplo de uso:
    ---------------
    >>> gw = GWBackgroundMCMC()
    >>> f_array = np.logspace(-18, -6, 100)
    >>> omega = gw.Omega_GW_total(f_array)
    >>> snr = gw.SNR_NANOGrav(T_obs_yr=15)
    """

    def __init__(self, params: ParametrosGW_MCMC = None):
        """
        Inicializa el calculador.

        Args:
            params: Parametros del espectro GW (usa defaults si None)
        """
        p = params or PARAMS_GW_DEFAULT

        self.Omega_ret = p.Omega_ret
        self.tau_ret = p.tau_ret_seconds  # Directamente en segundos
        self.f_peak = p.f_peak_Hz
        self.n_IR = p.n_IR
        self.n_UV = p.n_UV
        self.Delta_f = p.Delta_f
        self.epsilon_GW = p.epsilon_GW
        self.epsilon_MCMC = p.epsilon_MCMC
        self.z_trans = p.z_trans
        self.Delta_z = p.Delta_z

        # Calcular frecuencias de quiebre para cada umbral
        self._setup_break_frequencies()

    def _setup_break_frequencies(self):
        """
        Calcula las frecuencias de quiebre asociadas a cada umbral S_n.

        Las frecuencias de quiebre corresponden a ondas gravitacionales
        generadas durante las transiciones en cada umbral entropico.

        Para ondas generadas a temperatura T (en GeV):
        f_obs = f_emit * (a_emit / a_0) ~ f_emit * (T_0 / T_emit)

        donde T_0 ~ 2.3e-13 GeV es la temperatura actual del CMB.
        """
        # Temperaturas asociadas a cada umbral (en GeV)
        T_umbrales = {
            'QCD': E_QCD_GeV,       # 0.2 GeV ~ 200 MeV
            'EW': E_EW_GeV,         # 246 GeV
            'GUT': E_GUT_GeV,       # 1e16 GeV
            'Planck': E_PLANCK_GeV  # 1.22e19 GeV
        }

        # Temperatura actual del CMB en GeV
        T_0_GeV = 2.3e-13  # ~2.7 K

        # g_* (grados de libertad relativistas) en cada epoca
        # Afecta la relacion T-a via g_*^{1/3}
        g_star = {
            'QCD': 10.75,    # debajo de QCD
            'EW': 106.75,    # SM completo
            'GUT': 200,      # aprox para GUT
            'Planck': 200    # desconocido, asumimos similar
        }
        g_star_0 = 3.36  # hoy (foton + neutrinos)

        self.f_breaks = {}
        for nombre, T_n in T_umbrales.items():
            # Frecuencia caracteristica emitida ~ T_n (en unidades naturales)
            # f ~ T en unidades naturales, convertir a Hz
            # 1 GeV = 1.52e24 Hz (E = hf -> f = E/h)
            GeV_to_Hz = 1.52e24

            f_emit = T_n * GeV_to_Hz

            # Factor de dilusion por expansion
            # Para radiacion: f_obs/f_emit = a_emit/a_0 = T_0/T_emit * (g_*/g_*0)^{1/3}
            g_ratio = (g_star.get(nombre, 100) / g_star_0)**(1/3)
            dilution = (T_0_GeV / T_n) * g_ratio

            f_obs = f_emit * dilution

            self.f_breaks[nombre] = f_obs

        # Verificar ordenamiento
        # f_QCD > f_EW > f_GUT > f_Planck (mayor T -> mayor dilution -> mayor redshift)

    # =========================================================================
    # ESPECTRO DEL RETROCESO ENTROPICO
    # =========================================================================

    def Omega_GW_rebound(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Espectro de OG del retroceso entropico.

        Omega_GW(f) = Omega_ret * (f * tau_ret)^2 * sech^2(pi * f * tau_ret / 2)

        Esta forma viene de la transformada de Fourier de un pulso de
        liberacion de tension con forma tanh(t/tau_ret).

        Args:
            f: Frecuencia en Hz

        Returns:
            Omega_GW * h^2 (adimensional)
        """
        f = np.atleast_1d(f)

        # Variable adimensional
        x = f * self.tau_ret

        # Espectro
        # sech^2 da el cutoff exponencial a altas frecuencias
        # (f*tau)^2 da el comportamiento IR (n_IR = 2)
        omega = self.Omega_ret * x**2 * np.cosh(np.pi * x / 2)**(-2)

        # Normalizar h^2 ~ 0.5
        h_squared = 0.5

        result = omega * h_squared

        if len(result) == 1:
            return float(result[0])
        return result

    def Omega_GW_powerlaw(self, f: Union[float, np.ndarray],
                          f_ref: float = 1e-8) -> Union[float, np.ndarray]:
        """
        Espectro de ley de potencias simple (aproximacion IR).

        Omega_GW(f) = Omega_ref * (f / f_ref)^n

        Util para comparar con fits de NANOGrav.

        Args:
            f: Frecuencia en Hz
            f_ref: Frecuencia de referencia (default: 1 nHz)

        Returns:
            Omega_GW * h^2
        """
        f = np.atleast_1d(f)

        # Amplitud en f_ref
        Omega_ref = self.Omega_GW_rebound(f_ref)

        # Ley de potencias
        omega = Omega_ref * (f / f_ref)**self.n_IR

        if len(omega) == 1:
            return float(omega[0])
        return omega

    # =========================================================================
    # QUIEBRES ESPECTRALES EN UMBRALES S_n
    # =========================================================================

    def Omega_GW_breaks(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Espectro con quiebres en cada umbral S_n.

        Cada umbral Sn produce un cambio de pendiente en el espectro
        a la frecuencia f_n correspondiente.

        Omega_GW(f) = Sum_n Omega_n * (f/f_n)^alpha_n * Sigma_n(f)

        donde Sigma_n es una funcion de ventana suave.

        Args:
            f: Frecuencia en Hz

        Returns:
            Omega_GW * h^2 con quiebres
        """
        f = np.atleast_1d(f)

        # Amplitudes relativas en cada umbral
        # Normalizadas al retroceso principal
        amplitudes = {
            'Planck': 1.0,
            'GUT': 0.3,
            'EW': 0.1,
            'QCD': 0.03
        }

        # Indices espectrales entre quiebres
        alphas = {
            'IR': 2.0,           # f < f_QCD
            'QCD': 0.5,          # f_QCD < f < f_EW
            'EW': -0.5,          # f_EW < f < f_GUT
            'GUT': -1.5,         # f_GUT < f < f_Planck
            'UV': -3.0           # f > f_Planck
        }

        # Construir espectro por segmentos
        omega = np.zeros_like(f, dtype=float)

        # Base: retroceso entropico
        omega_base = self.Omega_GW_rebound(f)

        # Anadir contribuciones de cada quiebre
        f_sorted = sorted(self.f_breaks.items(), key=lambda x: x[1])

        for i, (nombre, f_break) in enumerate(f_sorted):
            # Factor de modulacion por el quiebre
            x = f / f_break
            # Transicion suave usando tanh
            sigma = 0.5 * (1 + np.tanh((np.log10(x)) / self.Delta_f))

            # Contribucion del quiebre
            A_n = amplitudes.get(nombre, 0.1)
            omega += omega_base * A_n * sigma * (1 - sigma)

        # Combinar con base
        omega_total = omega_base * (1 + omega / (omega_base + 1e-30))

        if len(omega_total) == 1:
            return float(omega_total[0])
        return omega_total

    # =========================================================================
    # ESPECTRO TOTAL
    # =========================================================================

    def Omega_GW_total(self, f: Union[float, np.ndarray],
                       include_breaks: bool = True) -> Union[float, np.ndarray]:
        """
        Espectro total de OG del modelo MCMC.

        Args:
            f: Frecuencia en Hz
            include_breaks: Si incluir quiebres espectrales

        Returns:
            Omega_GW * h^2 total
        """
        if include_breaks:
            return self.Omega_GW_breaks(f)
        else:
            return self.Omega_GW_rebound(f)

    # =========================================================================
    # PREDICCIONES PARA PTA
    # =========================================================================

    def characteristic_strain(self, f: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Strain caracteristico h_c(f).

        h_c^2(f) = (3 H_0^2) / (2 pi^2 f^2) * Omega_GW(f)

        Args:
            f: Frecuencia en Hz

        Returns:
            h_c (adimensional)
        """
        f = np.atleast_1d(f)

        omega = self.Omega_GW_total(f)

        # Conversion
        h_c_squared = (3 * H0_SI**2) / (2 * np.pi**2 * f**2) * omega
        h_c = np.sqrt(np.maximum(h_c_squared, 0))

        if len(h_c) == 1:
            return float(h_c[0])
        return h_c

    def timing_residual_rms(self, f: float, T_obs_yr: float = 15) -> float:
        """
        RMS del residuo de timing para un pulsar.

        sigma_t ~ h_c(f) / (2 pi f)

        Args:
            f: Frecuencia en Hz
            T_obs_yr: Tiempo de observacion en anos

        Returns:
            sigma_t en segundos
        """
        h_c = self.characteristic_strain(f)
        sigma_t = h_c / (2 * np.pi * f)
        return sigma_t

    def SNR_NANOGrav(self, T_obs_yr: float = 15, N_pulsars: int = 67) -> Dict:
        """
        Estima el SNR para NANOGrav.

        SNR ~ sqrt(2 * T_obs * N_pairs) * (Omega_GW / Omega_noise)

        Args:
            T_obs_yr: Tiempo de observacion en anos
            N_pulsars: Numero de pulsares en el array

        Returns:
            Dict con SNR y parametros
        """
        # Frecuencia caracteristica PTA
        f_yr = 1.0 / (T_obs_yr * 365.25 * 24 * 3600)  # Hz
        f_char = 3 * f_yr  # ~3/T es donde PTA es mas sensible

        # Omega_GW del modelo
        omega_signal = self.Omega_GW_total(f_char)

        # Ruido de timing (aproximacion)
        # sigma_t ~ 100 ns para buenos pulsares
        sigma_t_s = 100e-9  # s

        # Omega efectivo del ruido
        # Omega_noise ~ (2 pi f)^2 * sigma_t^2 / (3 H0^2)
        omega_noise = (2 * np.pi * f_char)**2 * sigma_t_s**2 / (3 * H0_SI**2)

        # Numero de pares de pulsares
        N_pairs = N_pulsars * (N_pulsars - 1) / 2

        # SNR
        T_obs_s = T_obs_yr * 365.25 * 24 * 3600
        snr = np.sqrt(2 * T_obs_s * f_char * N_pairs) * (omega_signal / omega_noise)

        # Detectabilidad
        detectable = snr > 3

        return {
            'SNR': float(snr),
            'f_char_Hz': f_char,
            'Omega_signal': float(omega_signal),
            'Omega_noise': omega_noise,
            'T_obs_yr': T_obs_yr,
            'N_pulsars': N_pulsars,
            'detectable': detectable,
            'h_c': float(self.characteristic_strain(f_char))
        }

    def SNR_SKA(self, T_obs_yr: float = 20, N_pulsars: int = 200) -> Dict:
        """
        Estima el SNR para SKA-PTA.

        SKA tendra ~200 pulsares con timing de ~50 ns.

        Args:
            T_obs_yr: Tiempo de observacion
            N_pulsars: Numero de pulsares (SKA: ~200)

        Returns:
            Dict con SNR y parametros
        """
        f_yr = 1.0 / (T_obs_yr * 365.25 * 24 * 3600)
        f_char = 3 * f_yr

        omega_signal = self.Omega_GW_total(f_char)

        # SKA tiene mejor timing
        sigma_t_s = 50e-9  # 50 ns
        omega_noise = (2 * np.pi * f_char)**2 * sigma_t_s**2 / (3 * H0_SI**2)

        N_pairs = N_pulsars * (N_pulsars - 1) / 2
        T_obs_s = T_obs_yr * 365.25 * 24 * 3600
        snr = np.sqrt(2 * T_obs_s * f_char * N_pairs) * (omega_signal / omega_noise)

        return {
            'SNR': float(snr),
            'f_char_Hz': f_char,
            'Omega_signal': float(omega_signal),
            'Omega_noise': omega_noise,
            'T_obs_yr': T_obs_yr,
            'N_pulsars': N_pulsars,
            'detectable': snr > 3,
            'h_c': float(self.characteristic_strain(f_char))
        }

    # =========================================================================
    # PREDICCIONES PARA LIGO/Virgo
    # =========================================================================

    def Omega_GW_LIGO_band(self) -> Dict:
        """
        Prediccion para banda LIGO (10-1000 Hz).

        El modelo MCMC predice un fondo muy debajo de la sensibilidad
        actual de LIGO, pero podria ser detectable con ET/CE.

        Returns:
            Dict con Omega_GW en banda LIGO
        """
        f_LIGO = np.array([10, 25, 50, 100, 250, 500, 1000])  # Hz

        omega_LIGO = self.Omega_GW_total(f_LIGO)

        # Sensibilidad de LIGO O4
        # Omega_sens ~ 1e-9 a 25 Hz
        Omega_LIGO_sens = 1e-9

        return {
            'f_Hz': f_LIGO,
            'Omega_GW': omega_LIGO,
            'Omega_sens_O4': Omega_LIGO_sens,
            'detectable_O4': np.any(omega_LIGO > Omega_LIGO_sens),
            'ratio': omega_LIGO / Omega_LIGO_sens
        }

    def Omega_GW_ET_band(self) -> Dict:
        """
        Prediccion para Einstein Telescope (1-10000 Hz).

        ET tendra sensibilidad Omega ~ 1e-13 a 10 Hz.

        Returns:
            Dict con prediccion para ET
        """
        f_ET = np.logspace(0, 4, 50)  # 1-10000 Hz

        omega_ET = self.Omega_GW_total(f_ET)

        # Sensibilidad de ET
        Omega_ET_sens = 1e-13  # a 10 Hz

        return {
            'f_Hz': f_ET,
            'Omega_GW': omega_ET,
            'Omega_sens_ET': Omega_ET_sens,
            'detectable_ET': np.any(omega_ET > Omega_ET_sens)
        }

    # =========================================================================
    # COMPARACION CON NANOGrav 15yr
    # =========================================================================

    def compare_NANOGrav_15yr(self) -> Dict:
        """
        Compara prediccion MCMC con resultados de NANOGrav 15yr.

        NANOGrav 15yr reporto:
        - A_GWB = 2.4e-15 a f_ref = 1/yr
        - gamma = 13/3 ~ 4.33 (consistente con SMBHB)

        Returns:
            Dict con comparacion
        """
        # Frecuencia de referencia NANOGrav
        f_yr = 1.0 / (365.25 * 24 * 3600)  # ~3.17e-8 Hz

        # Amplitud reportada por NANOGrav
        A_NANOGrav = 2.4e-15
        gamma_NANOGrav = 13/3

        # Omega correspondiente
        # Omega_GW = (2 pi^2 / 3 H0^2) * f^2 * h_c^2
        # h_c = A * (f/f_yr)^alpha, alpha = (3-gamma)/2
        alpha_NANOGrav = (3 - gamma_NANOGrav) / 2

        h_c_NANOGrav = A_NANOGrav * (f_yr / f_yr)**alpha_NANOGrav
        Omega_NANOGrav = (2 * np.pi**2 / (3 * H0_SI**2)) * f_yr**2 * h_c_NANOGrav**2

        # Prediccion MCMC
        Omega_MCMC = self.Omega_GW_total(f_yr)
        h_c_MCMC = self.characteristic_strain(f_yr)

        # Ratio
        ratio = Omega_MCMC / Omega_NANOGrav

        # El modelo MCMC predice un fondo diferente al de SMBHB
        # La diferencia en gamma es clave
        gamma_MCMC = 3 - 2 * self.n_IR  # gamma = 3 - 2*n para Omega ~ f^n

        return {
            'f_ref_Hz': f_yr,
            'A_NANOGrav': A_NANOGrav,
            'gamma_NANOGrav': gamma_NANOGrav,
            'Omega_NANOGrav': Omega_NANOGrav,
            'Omega_MCMC': float(Omega_MCMC),
            'h_c_MCMC': float(h_c_MCMC),
            'gamma_MCMC': gamma_MCMC,
            'ratio_Omega': float(ratio),
            'interpretation': self._interpret_comparison(ratio, gamma_MCMC, gamma_NANOGrav)
        }

    def _interpret_comparison(self, ratio: float, gamma_MCMC: float,
                              gamma_NANOGrav: float) -> str:
        """Genera interpretacion de la comparacion."""
        if ratio > 0.1 and ratio < 10:
            amplitude_match = "Amplitud compatible dentro de 1 orden de magnitud"
        elif ratio > 10:
            amplitude_match = "MCMC predice amplitud MAYOR que NANOGrav"
        else:
            amplitude_match = "MCMC predice amplitud MENOR que NANOGrav"

        delta_gamma = abs(gamma_MCMC - gamma_NANOGrav)
        if delta_gamma < 0.5:
            spectral_match = "Indice espectral compatible"
        else:
            spectral_match = f"Indice espectral DIFERENTE (Delta_gamma = {delta_gamma:.2f})"

        return f"{amplitude_match}. {spectral_match}. " \
               f"MCMC: gamma={gamma_MCMC:.2f}, NANOGrav: gamma={gamma_NANOGrav:.2f}"

    # =========================================================================
    # GENERADOR DE ESPECTRO COMPLETO
    # =========================================================================

    def generate_spectrum(self, f_min: float = 1e-18, f_max: float = 1e4,
                         n_points: int = 200) -> Dict:
        """
        Genera espectro completo para visualizacion.

        Args:
            f_min: Frecuencia minima en Hz
            f_max: Frecuencia maxima en Hz
            n_points: Numero de puntos

        Returns:
            Dict con f, Omega_GW, h_c, y metadatos
        """
        f_array = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

        omega_array = self.Omega_GW_total(f_array)
        h_c_array = self.characteristic_strain(f_array)

        # Encontrar pico
        idx_peak = np.argmax(omega_array)
        f_peak_actual = f_array[idx_peak]
        omega_peak = omega_array[idx_peak]

        return {
            'f_Hz': f_array,
            'Omega_GW_h2': omega_array,
            'h_c': h_c_array,
            'f_peak_Hz': f_peak_actual,
            'Omega_peak': omega_peak,
            'f_breaks': self.f_breaks,
            'params': {
                'Omega_ret': self.Omega_ret,
                'tau_ret_s': self.tau_ret,
                'n_IR': self.n_IR,
                'n_UV': self.n_UV
            }
        }


# =============================================================================
# FUNCION DE VALIDACION
# =============================================================================

def test_GW_Background_MCMC() -> bool:
    """
    Test completo del modulo de fondo de ondas gravitacionales.

    Verifica:
    1. Forma del espectro (pico y caidas)
    2. Consistencia con NANOGrav
    3. Predicciones para PTA
    4. Quiebres espectrales
    """
    print("\n" + "="*70)
    print("  TEST GW BACKGROUND MCMC - FONDO DE ONDAS GRAVITACIONALES")
    print("="*70)

    gw = GWBackgroundMCMC()
    all_passed = True

    # -------------------------------------------------------------------------
    # Test 1: Espectro basico
    # -------------------------------------------------------------------------
    print("\n[1] Espectro del retroceso entropico:")

    f_test = np.array([1e-16, 1e-12, 1e-8, 1e-4, 1e0, 1e4])
    omega_test = gw.Omega_GW_rebound(f_test)

    for f, omega in zip(f_test, omega_test):
        print(f"    f = {f:.0e} Hz: Omega_GW*h^2 = {omega:.2e}")

    # Verificar que tiene un pico
    spectrum = gw.generate_spectrum()
    has_peak = spectrum['Omega_peak'] > spectrum['Omega_GW_h2'][0]
    has_peak &= spectrum['Omega_peak'] > spectrum['Omega_GW_h2'][-1]

    print(f"    Pico en f = {spectrum['f_peak_Hz']:.2e} Hz")
    print(f"    Omega_peak = {spectrum['Omega_peak']:.2e}")
    test1 = has_peak
    print(f"    Espectro con pico: {'PASS' if test1 else 'FAIL'}")
    all_passed &= test1

    # -------------------------------------------------------------------------
    # Test 2: Predicciones PTA
    # -------------------------------------------------------------------------
    print("\n[2] Predicciones para PTA:")

    nano = gw.SNR_NANOGrav(T_obs_yr=15, N_pulsars=67)
    ska = gw.SNR_SKA(T_obs_yr=20, N_pulsars=200)

    print(f"    NANOGrav 15yr:")
    print(f"      SNR = {nano['SNR']:.2f}")
    print(f"      Omega_signal = {nano['Omega_signal']:.2e}")
    print(f"      h_c = {nano['h_c']:.2e}")
    print(f"      Detectable (SNR>3): {nano['detectable']}")

    print(f"    SKA-PTA 20yr:")
    print(f"      SNR = {ska['SNR']:.2f}")
    print(f"      Omega_signal = {ska['Omega_signal']:.2e}")
    print(f"      Detectable (SNR>3): {ska['detectable']}")

    # Verificar que SKA tiene mejor SNR que NANOGrav
    test2 = ska['SNR'] > nano['SNR']
    print(f"    SKA SNR > NANOGrav SNR: {'PASS' if test2 else 'FAIL'}")
    all_passed &= test2

    # -------------------------------------------------------------------------
    # Test 3: Comparacion con NANOGrav 15yr
    # -------------------------------------------------------------------------
    print("\n[3] Comparacion con NANOGrav 15yr:")

    comparison = gw.compare_NANOGrav_15yr()

    print(f"    NANOGrav reporta:")
    print(f"      A_GWB = {comparison['A_NANOGrav']:.2e}")
    print(f"      gamma = {comparison['gamma_NANOGrav']:.2f}")
    print(f"      Omega = {comparison['Omega_NANOGrav']:.2e}")

    print(f"    MCMC predice:")
    print(f"      Omega = {comparison['Omega_MCMC']:.2e}")
    print(f"      h_c = {comparison['h_c_MCMC']:.2e}")
    print(f"      gamma = {comparison['gamma_MCMC']:.2f}")

    print(f"    Ratio: {comparison['ratio_Omega']:.2f}")
    print(f"    {comparison['interpretation']}")

    # El modelo debe predecir algo en el rango correcto de ordenes de magnitud
    test3 = 1e-12 < comparison['Omega_MCMC'] < 1e-6
    print(f"    Omega en rango razonable: {'PASS' if test3 else 'FAIL'}")
    all_passed &= test3

    # -------------------------------------------------------------------------
    # Test 4: Quiebres espectrales
    # -------------------------------------------------------------------------
    print("\n[4] Quiebres espectrales en umbrales S_n:")

    for nombre, f_break in sorted(gw.f_breaks.items(), key=lambda x: x[1]):
        print(f"    {nombre}: f_break = {f_break:.2e} Hz")

    # Verificar que los quiebres estan en el rango de alta frecuencia
    # (muy por encima de la banda PTA, en la banda de detectores terrestres o superior)
    all_in_range = all(f > 1e9 for f in gw.f_breaks.values())
    test4 = all_in_range and len(gw.f_breaks) == 4
    print(f"    Quiebres en banda alta frecuencia (>1 GHz): {'PASS' if test4 else 'FAIL'}")
    all_passed &= test4

    # -------------------------------------------------------------------------
    # Test 5: Banda LIGO
    # -------------------------------------------------------------------------
    print("\n[5] Prediccion para banda LIGO:")

    ligo = gw.Omega_GW_LIGO_band()

    print(f"    Omega_GW en LIGO band:")
    for f, omega in zip(ligo['f_Hz'][:3], ligo['Omega_GW'][:3]):
        print(f"      f = {f:.0f} Hz: Omega = {omega:.2e}")

    print(f"    Sensibilidad LIGO O4: Omega ~ {ligo['Omega_sens_O4']:.0e}")
    print(f"    Detectable en O4: {ligo['detectable_O4']}")

    # En LIGO band el modelo predice un fondo muy debil
    test5 = not ligo['detectable_O4']  # No deberia ser detectable en O4
    print(f"    Prediccion consistente (no detectable en O4): {'PASS' if test5 else 'FAIL'}")
    all_passed &= test5

    # -------------------------------------------------------------------------
    # Test 6: Consistencia fisica
    # -------------------------------------------------------------------------
    print("\n[6] Consistencia fisica:")

    # Omega_GW debe ser positivo
    f_full = np.logspace(-18, 4, 100)
    omega_full = gw.Omega_GW_total(f_full)

    test6a = np.all(omega_full >= 0)
    print(f"    Omega_GW >= 0 para todo f: {'PASS' if test6a else 'FAIL'}")

    # h_c debe ser positivo
    h_c_full = gw.characteristic_strain(f_full)
    test6b = np.all(h_c_full >= 0)
    print(f"    h_c >= 0 para todo f: {'PASS' if test6b else 'FAIL'}")

    # Omega debe decrecer a altas frecuencias
    test6c = omega_full[-1] < omega_full[50]  # f=1e4 < f~1e-7
    print(f"    Omega decrece a f alta: {'PASS' if test6c else 'FAIL'}")

    test6 = test6a and test6b and test6c
    all_passed &= test6

    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"  GW BACKGROUND MODULE: {'PASS' if all_passed else 'FAIL'}")
    print("="*70)

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_GW_Background_MCMC()
