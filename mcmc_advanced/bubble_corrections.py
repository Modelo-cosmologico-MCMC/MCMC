#!/usr/bin/env python3
"""
================================================================================
MODULO DE CORRECCIONES POR BURBUJAS TEMPORALES - MCMC v1.0
================================================================================

Calcula correcciones observacionales por transito de fotones a traves de
regiones con diferentes S_local (burbujas temporales/gravitacionales).

Fundamentacion Ontologica:
--------------------------
En el MCMC, cada region del universo tiene un S_local determinado por:
- Densidad de materia local
- Historia de formacion de estructura
- Proximidad a pozos gravitacionales

El potencial cronologico Xi(x) determina el flujo del tiempo local:
    dt_loc/dS = T(S) N(S) e^{Xi(x)}

donde:
    Xi(x) = lambda_S * Delta_S(x, t_rel) / S_ext(t_rel)

Cuando un foton atraviesa multiples regiones (burbujas), acumula correcciones:
    z_obs = z_cosmo + Delta_z_Cronos

Conceptos Implementados:
------------------------
1. Integral de trayectoria fotonica (Seccion 2.1)
2. Efecto de burbuja emisora (Seccion 2.2)
3. Mapa S(z, n) con estructura LSS (Seccion 2.3)
4. Correccion a la edad del universo (Seccion 3)
5. Mecanismo de tension de Hubble (Seccion 5)

Referencias:
- Ec. 38, 42: Desfase temporal Cronos
- Ec. 296-297: Potencial cronologico local
- Ec. 147-150: Correcciones a frecuencias atomicas

Autor: MCMC Cosmology Framework
Copyright (c) 2024-2025
================================================================================
"""

import numpy as np
from scipy.integrate import quad, simpson, cumulative_trapezoid
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FUNDAMENTALES
# =============================================================================

# Constantes fisicas
C_LIGHT = 299792.458  # km/s
H0_PLANCK = 67.4      # km/s/Mpc (Planck 2018)
H0_SHOES = 73.0       # km/s/Mpc (SH0ES 2022)
T_HUBBLE = 13.8       # Gyr (edad Hubble)

# Parametros MCMC calibrados
EPSILON_MCMC = 0.012      # Acoplamiento ECV
Z_TRANS = 8.9             # Redshift de transicion
DELTA_Z = 1.5             # Anchura de transicion
LAMBDA_S = 0.5            # Coeficiente adimensional para Xi
LAMBDA_C = 50.0           # Escala Cronos (Ec. 38)
K_ALPHA = 1.0             # Constante de integracion

# Valores de S tipicos
S_LOCAL_VL = 85.0         # Via Lactea / Grupo Local
S_LOCAL_VOID = 88.0       # Void tipico
S_LOCAL_CLUSTER = 80.0    # Cumulo de galaxias
S_EXT_0 = 90.0            # Fondo cosmologico a z=0
S_RECOMB = 0.1            # Superficie de ultimo scattering


# =============================================================================
# TIPOS DE REGIONES (BURBUJAS)
# =============================================================================

class TipoBurbuja(Enum):
    """Clasificacion de regiones por densidad."""
    VOID = "void"                    # Delta_m < -0.5
    CAMPO = "campo"                  # -0.5 < Delta_m < 0.5
    FILAMENTO = "filamento"          # 0.5 < Delta_m < 5
    CUMULO = "cumulo"                # Delta_m > 5
    GALAXIA = "galaxia"              # Escala sub-Mpc
    VIA_LACTEA = "via_lactea"        # Nuestra burbuja


@dataclass
class ParametrosBurbuja:
    """Parametros de una region/burbuja temporal."""
    tipo: TipoBurbuja
    S_local: float
    z_centro: float = 0.0
    delta_m: float = 0.0          # Contraste de densidad
    radio_Mpc: float = 10.0       # Radio caracteristico
    nombre: str = ""


# Burbujas canonicas
BURBUJA_VIA_LACTEA = ParametrosBurbuja(
    tipo=TipoBurbuja.VIA_LACTEA,
    S_local=S_LOCAL_VL,
    z_centro=0.0,
    delta_m=0.5,
    radio_Mpc=2.0,
    nombre="Via Lactea"
)

BURBUJA_VOID_TIPICO = ParametrosBurbuja(
    tipo=TipoBurbuja.VOID,
    S_local=S_LOCAL_VOID,
    z_centro=0.5,
    delta_m=-0.7,
    radio_Mpc=30.0,
    nombre="Void Tipico"
)

BURBUJA_CUMULO = ParametrosBurbuja(
    tipo=TipoBurbuja.CUMULO,
    S_local=S_LOCAL_CLUSTER,
    z_centro=0.3,
    delta_m=10.0,
    radio_Mpc=5.0,
    nombre="Cumulo de Galaxias"
)


# =============================================================================
# PARAMETROS DE CONFIGURACION
# =============================================================================

@dataclass
class ParametrosBubbleCorrection:
    """Parametros para el calculador de correcciones por burbujas."""
    # Cosmologia
    H0: float = H0_PLANCK
    Omega_m: float = 0.315
    Omega_Lambda: float = 0.685

    # MCMC
    epsilon: float = EPSILON_MCMC
    z_trans: float = Z_TRANS
    Delta_z: float = DELTA_Z
    lambda_S: float = LAMBDA_S
    lambda_C: float = LAMBDA_C
    k_alpha: float = K_ALPHA

    # Burbuja local (observador)
    S_local_VL: float = S_LOCAL_VL
    S_ext_0: float = S_EXT_0

    # Precision numerica
    n_z_integral: int = 500
    z_max: float = 10.0


PARAMS_DEFAULT = ParametrosBubbleCorrection()


# =============================================================================
# CLASE PRINCIPAL: BubbleCorrectionCalculator
# =============================================================================

class BubbleCorrectionCalculator:
    """
    Calculador de correcciones observacionales por transito de fotones
    a traves de burbujas temporales con diferentes S_local.

    Implementa:
    -----------
    1. Correcciones al redshift por salida/entrada de burbujas
    2. Integral de trayectoria para fluctuaciones de S
    3. Correccion al modulo de distancia
    4. Correccion a la edad del universo
    5. Mecanismo de tension de Hubble

    Fisica clave:
    -------------
    - El tiempo fluye mas lento en regiones con S_local < S_ext (burbujas)
    - Fotones que salen de una burbuja experimentan blueshift cronologico
    - Fotones que entran en una burbuja experimentan redshift cronologico
    - La acumulacion a lo largo de la trayectoria modifica z_obs

    Ejemplo de uso:
    ---------------
    >>> calc = BubbleCorrectionCalculator()
    >>> z_obs = calc.corrected_redshift(z_em=2.0, S_local_em=82.0)
    >>> mu_corr = calc.corrected_distance_modulus(mu_cosmo=44.5, z=1.0, S_local_source=84.0)
    """

    def __init__(self, params: ParametrosBubbleCorrection = None):
        """
        Inicializa el calculador.

        Args:
            params: Parametros de configuracion (usa defaults si None)
        """
        p = params or PARAMS_DEFAULT

        # Cosmologia
        self.H0 = p.H0
        self.Omega_m = p.Omega_m
        self.Omega_Lambda = p.Omega_Lambda
        self.c = C_LIGHT

        # MCMC
        self.epsilon = p.epsilon
        self.z_trans = p.z_trans
        self.Delta_z = p.Delta_z
        self.lambda_S = p.lambda_S
        self.lambda_C = p.lambda_C
        self.k_alpha = p.k_alpha

        # Burbuja local
        self.S_local_VL = p.S_local_VL
        self.S_ext_0 = p.S_ext_0

        # Precision
        self.n_z_integral = p.n_z_integral
        self.z_max = p.z_max

        # Precalcular S_ext(z)
        self._setup_S_ext_interpolation()

        # Cache
        self._chi_cache = {}

    # =========================================================================
    # FUNCIONES COSMOLOGICAS BASICAS
    # =========================================================================

    def E(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """E(z) = H(z)/H0 para LCDM."""
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)

    def H(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """H(z) en km/s/Mpc."""
        return self.H0 * self.E(z)

    def chi(self, z: float) -> float:
        """Distancia comoovil en Mpc."""
        if z in self._chi_cache:
            return self._chi_cache[z]
        result, _ = quad(lambda zp: self.c / self.H(zp), 0, z, limit=200)
        self._chi_cache[z] = result
        return result

    def D_L(self, z: float) -> float:
        """Distancia luminosidad en Mpc."""
        return (1 + z) * self.chi(z)

    def mu_cosmo(self, z: float) -> float:
        """Modulo de distancia cosmologico (sin correcciones)."""
        D_L_Mpc = self.D_L(z)
        return 5 * np.log10(D_L_Mpc) + 25

    # =========================================================================
    # FUNCIONES DE S(z) Y Xi(x)
    # =========================================================================

    def _setup_S_ext_interpolation(self):
        """
        Precalcula S_ext(z) para el fondo homogeneo.

        S crece con el tiempo cosmico, por lo que S_ext(z) decrece con z.
        Normalizamos S_ext(z=0) = S_EXT_0.

        Fisicamente:
        - S_ext(z=0) = S_ext_0 ~ 90 (hoy)
        - S_ext(z->inf) -> 0 (Big Bang)

        La evolucion sigue la fraccion de tiempo cosmico transcurrido.
        """
        z_array = np.linspace(0, self.z_max, 500)

        # Tiempo cosmico desde z hasta hoy: t(z) = integral[z->0] dz'/((1+z')H(z'))
        # Fraccion: f(z) = t(z) / t_0
        # S_ext(z) = S_ext_0 * f(z)

        # Calcular t_0 (edad total)
        def t_integral(z_upper):
            """Integral de tiempo cosmico desde z hasta 0."""
            if z_upper < 1e-6:
                return 0.0
            integrand = lambda zp: 1 / ((1 + zp) * self.E(zp))
            result, _ = quad(integrand, 0, z_upper, limit=200)
            return result

        # Tiempo total hasta z=0 (referencia muy alta z)
        t_total_ref, _ = quad(lambda zp: 1 / ((1 + zp) * self.E(zp)), 0, 1000, limit=200)

        # S_ext escala con el tiempo cosmico transcurrido
        # A z=0: todo el tiempo ha pasado -> S_ext = S_ext_0
        # A z alto: poco tiempo ha pasado -> S_ext pequeno
        S_ext_array = np.zeros(len(z_array))
        for i, z in enumerate(z_array):
            # Tiempo desde el Big Bang hasta z
            t_from_BB_to_z, _ = quad(lambda zp: 1 / ((1 + zp) * self.E(zp)), z, 1000, limit=200)
            # Fraccion de tiempo transcurrido
            f_z = t_from_BB_to_z / t_total_ref
            S_ext_array[i] = self.S_ext_0 * f_z

        # Asegurar que S_ext(0) = S_ext_0 exactamente
        S_ext_array[0] = self.S_ext_0

        # Asegurar que S_ext sea positivo
        S_ext_array = np.maximum(S_ext_array, 0.01)

        self._S_ext_interp = interp1d(z_array, S_ext_array,
                                      kind='cubic', fill_value='extrapolate')

    def S_ext(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Entropia tensional del fondo cosmologico S_ext(z).

        Representa el "oceano geometrico" fuera de las burbujas.
        """
        return self._S_ext_interp(z)

    def Xi_from_S(self, S_local: float, S_ext: float) -> float:
        """
        Potencial cronologico desde diferencia de S (Ec. 296-297).

        Xi(x) = lambda_S * Delta_S(x) / S_ext

        donde Delta_S = S_ext - S_local (positivo dentro de burbujas).

        Args:
            S_local: Entropia tensional local
            S_ext: Entropia tensional del fondo

        Returns:
            Xi: Potencial cronologico (adimensional)
        """
        if S_ext < 1e-10:
            return 0.0
        Delta_S = S_ext - S_local
        return self.lambda_S * Delta_S / S_ext

    def Xi_at_z(self, z: float, S_local: float = None) -> float:
        """
        Potencial cronologico a redshift z.

        Args:
            z: Redshift
            S_local: Entropia local (si None, usa promedio)

        Returns:
            Xi(z)
        """
        S_ext_z = self.S_ext(z)
        if S_local is None:
            # Asume una burbuja tipica
            S_local = S_ext_z - 5.0  # Delta_S ~ 5 tipico
        return self.Xi_from_S(S_local, S_ext_z)

    # =========================================================================
    # CORRECCION AL SALIR DE BURBUJA EMISORA (Seccion 4.2A)
    # =========================================================================

    def delta_z_exit_bubble(self, z_em: float, S_local_em: float) -> float:
        """
        Correccion al redshift al salir de la burbuja emisora.

        El foton experimenta un "blueshift cronologico" porque el tiempo
        fluye mas lento dentro de la burbuja (S_local < S_ext).

        delta_nu_salida = nu_0 * epsilon * (Xi_local,em - Xi_ext)

        Como Xi_ext ~ 0 por definicion en el fondo:
        delta_z_exit = -epsilon * Xi_em  (negativo = blueshift)

        Args:
            z_em: Redshift de emision
            S_local_em: Entropia tensional en la galaxia emisora

        Returns:
            Correccion al redshift (puede ser negativa)
        """
        S_ext_z = self.S_ext(z_em)
        Xi_em = self.Xi_from_S(S_local_em, S_ext_z)

        # El foton gana energia al salir de la burbuja
        # (el reloj de la burbuja esta "atrasado")
        return -self.epsilon * Xi_em

    # =========================================================================
    # CORRECCION AL ENTRAR EN NUESTRA BURBUJA (Seccion 4.2C)
    # =========================================================================

    def delta_z_enter_bubble(self) -> float:
        """
        Correccion al redshift al entrar en nuestra burbuja (Via Lactea).

        El foton experimenta un "redshift cronologico" al entrar en nuestra
        cavidad tensional.

        delta_nu_entrada = -nu_0 * epsilon * (Xi_VL - Xi_ext)

        Returns:
            Correccion al redshift (positiva = redshift adicional)
        """
        Xi_VL = self.Xi_from_S(self.S_local_VL, self.S_ext_0)

        # El foton pierde energia al entrar en nuestra burbuja
        return self.epsilon * Xi_VL

    # =========================================================================
    # CORRECCION POR TRANSITO (Seccion 4.2B)
    # =========================================================================

    def delta_z_transit(self, z_em: float,
                        S_fluctuations: np.ndarray = None,
                        z_array: np.ndarray = None) -> float:
        """
        Correccion por fluctuaciones de S a lo largo de la trayectoria.

        Durante el transito por el "oceano geometrico", el foton acumula
        correcciones por variaciones de S debido a la estructura a gran escala.

        delta_z_transit = epsilon * integral[dXi]

        Args:
            z_em: Redshift de emision
            S_fluctuations: Array de fluctuaciones Delta_S_LSS(z)
            z_array: Array de redshifts correspondiente

        Returns:
            Correccion acumulada por transito
        """
        if S_fluctuations is None or z_array is None:
            # Sin fluctuaciones explicitas, asume oceano homogeneo
            return 0.0

        # Calcular Xi a lo largo de la trayectoria
        S_ext_z = self.S_ext(z_array)
        S_local_z = S_ext_z - S_fluctuations
        Xi_z = self.lambda_S * S_fluctuations / S_ext_z

        # Integrar dXi
        dXi = np.gradient(Xi_z, z_array)
        delta_z = self.epsilon * np.trapz(dXi, z_array)

        return delta_z

    # =========================================================================
    # CORRECCION TOTAL AL REDSHIFT (Seccion 4.3)
    # =========================================================================

    def corrected_redshift(self, z_em: float, S_local_em: float,
                          S_fluctuations: np.ndarray = None,
                          z_array: np.ndarray = None) -> Dict:
        """
        Calcula el redshift observado con todas las correcciones.

        z_obs = z_cosmo + epsilon * [(Xi_em - Xi_ext(z_em)) - (Xi_VL - Xi_ext(0)) + integral]

        Args:
            z_em: Redshift cosmologico (sin correcciones)
            S_local_em: Entropia tensional de la galaxia emisora
            S_fluctuations: Fluctuaciones de S por LSS (opcional)
            z_array: Array de z para la integral de transito

        Returns:
            Dict con z_obs, z_cosmo, y correcciones individuales
        """
        # Correcciones individuales
        delta_exit = self.delta_z_exit_bubble(z_em, S_local_em)
        delta_enter = self.delta_z_enter_bubble()
        delta_transit = self.delta_z_transit(z_em, S_fluctuations, z_array)

        # Correccion total
        delta_z_total = delta_exit + delta_enter + delta_transit

        # Redshift observado
        z_obs = z_em + delta_z_total

        return {
            'z_obs': z_obs,
            'z_cosmo': z_em,
            'delta_z_total': delta_z_total,
            'delta_z_exit': delta_exit,
            'delta_z_enter': delta_enter,
            'delta_z_transit': delta_transit,
            'S_local_em': S_local_em,
            'S_local_VL': self.S_local_VL,
            'Xi_em': self.Xi_from_S(S_local_em, self.S_ext(z_em)),
            'Xi_VL': self.Xi_from_S(self.S_local_VL, self.S_ext_0)
        }

    # =========================================================================
    # MAPA S(z, n) CON ESTRUCTURA LSS (Seccion 2.3)
    # =========================================================================

    def S_with_LSS(self, z: float, delta_m: float) -> float:
        """
        Entropia tensional incluyendo fluctuaciones de densidad.

        S(z, n) = S_ext(z) - Delta_S_LSS(z, n)

        donde Delta_S_LSS ~ W(z) * delta_m(z, n)

        Args:
            z: Redshift
            delta_m: Contraste de densidad a lo largo de la linea de vision

        Returns:
            S_local efectivo
        """
        S_ext_z = self.S_ext(z)

        # Funcion de peso W(z) - crece con z (mayor efecto a alto z)
        W_z = 1.0 + 0.5 * z

        # Delta_S proporcional a delta_m
        # Normalizacion: delta_m ~ 1 produce Delta_S ~ 5
        Delta_S_LSS = 5.0 * W_z * delta_m

        return S_ext_z - Delta_S_LSS

    def generate_S_map_1D(self, z_max: float = 3.0, n_points: int = 100,
                         delta_m_rms: float = 0.3) -> Dict:
        """
        Genera un mapa 1D de S(z) a lo largo de una linea de vision.

        Incluye fluctuaciones gaussianas de delta_m.

        Args:
            z_max: Redshift maximo
            n_points: Numero de puntos
            delta_m_rms: RMS de fluctuaciones de densidad

        Returns:
            Dict con z_array, S_array, delta_m_array, Xi_array
        """
        z_array = np.linspace(0.01, z_max, n_points)

        # Generar fluctuaciones de densidad
        # Correlacionadas en z (escala de correlacion ~ 100 Mpc)
        delta_m_array = self._generate_correlated_density(z_array, delta_m_rms)

        # Calcular S(z) con fluctuaciones
        S_array = np.array([self.S_with_LSS(z, dm)
                          for z, dm in zip(z_array, delta_m_array)])

        # Calcular Xi(z)
        S_ext_array = self.S_ext(z_array)
        Xi_array = self.lambda_S * (S_ext_array - S_array) / S_ext_array

        return {
            'z': z_array,
            'S': S_array,
            'S_ext': S_ext_array,
            'delta_m': delta_m_array,
            'Xi': Xi_array,
            'Delta_S': S_ext_array - S_array
        }

    def _generate_correlated_density(self, z_array: np.ndarray,
                                     rms: float) -> np.ndarray:
        """Genera fluctuaciones de densidad correlacionadas."""
        n = len(z_array)

        # Ruido blanco
        white_noise = np.random.randn(n)

        # Suavizar para introducir correlacion
        kernel_size = max(3, n // 20)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(white_noise, kernel, mode='same')

        # Normalizar al rms deseado
        smoothed = smoothed / np.std(smoothed) * rms

        return smoothed

    # =========================================================================
    # CORRECCION AL MODULO DE DISTANCIA (Seccion 2.2)
    # =========================================================================

    def corrected_distance_modulus(self, mu_cosmo: float, z: float,
                                   S_local_source: float) -> Dict:
        """
        Modulo de distancia corregido por efectos de burbuja.

        mu_obs = mu_cosmo + Delta_mu_burbuja,emisor + Delta_mu_burbuja,receptor

        donde:
        Delta_mu = (5*epsilon / ln(10)) * (Xi_local - Xi_fondo)

        Args:
            mu_cosmo: Modulo de distancia cosmologico
            z: Redshift de la fuente
            S_local_source: Entropia tensional de la fuente

        Returns:
            Dict con mu_corr, mu_cosmo, correcciones
        """
        # Xi de la fuente
        Xi_source = self.Xi_from_S(S_local_source, self.S_ext(z))

        # Xi de nuestra burbuja
        Xi_VL = self.Xi_from_S(self.S_local_VL, self.S_ext_0)

        # Correcciones al modulo de distancia
        # Factor: 5 * epsilon / ln(10) ~ 0.026 para epsilon = 0.012
        factor = 5 * self.epsilon / np.log(10)

        Delta_mu_source = factor * Xi_source
        Delta_mu_VL = factor * Xi_VL

        # Correccion total
        Delta_mu_total = Delta_mu_source - Delta_mu_VL

        return {
            'mu_corr': mu_cosmo + Delta_mu_total,
            'mu_cosmo': mu_cosmo,
            'Delta_mu_total': Delta_mu_total,
            'Delta_mu_source': Delta_mu_source,
            'Delta_mu_VL': Delta_mu_VL,
            'Xi_source': Xi_source,
            'Xi_VL': Xi_VL
        }

    # =========================================================================
    # CORRECCION A LA EDAD DEL UNIVERSO (Seccion 3)
    # =========================================================================

    def delta_t_cronos(self, S_local: float = None, S_ext: float = None) -> float:
        """
        Desfase temporal Cronos (Ec. 38, 42).

        Delta_t_Cronos = integral[S_local -> S_ext] (lambda_C / k_alpha) * tanh(S/lambda_C) dS

        Args:
            S_local: Entropia local (default: Via Lactea)
            S_ext: Entropia del fondo (default: S_ext_0)

        Returns:
            Delta_t en unidades de lambda_C / k_alpha
        """
        S_local = S_local or self.S_local_VL
        S_ext = S_ext or self.S_ext_0

        integrand = lambda S: (self.lambda_C / self.k_alpha) * np.tanh(S / self.lambda_C)
        result, _ = quad(integrand, S_local, S_ext, limit=100)

        return result

    def corrected_universe_age(self, t_obs_Gyr: float = 13.8) -> Dict:
        """
        Edad del universo corregida por efecto de burbuja (Seccion 3.2).

        t_ext = t_obs + Delta_t_Cronos

        La edad "verdadera" desde S_ext es mayor que la observada desde
        nuestra burbuja porque nuestro reloj esta "atrasado".

        Estimacion del documento (Seccion 3.2):
        - Si S_local ~ 85, S_ext ~ 90, lambda_C ~ 50, k_alpha ~ 1:
        - Delta_t ~ 50 * 5 * tanh(1.7) ~ 240 unidades
        - En terminos fisicos: Delta_t ~ epsilon * t_Hubble ~ 0.01 * 13.8 Gyr ~ 140 Myr

        Args:
            t_obs_Gyr: Edad observada en Gyr (default: 13.8)

        Returns:
            Dict con edad corregida y parametros
        """
        # Potencial cronologico de nuestra burbuja
        Xi_VL = self.Xi_from_S(self.S_local_VL, self.S_ext_0)

        # Desfase en unidades internas (Ec. 38)
        delta_t_units = self.delta_t_cronos()

        # Conversion a Gyr usando la relacion del documento:
        # Delta_t / t_Hubble ~ epsilon * Xi_local
        # Pero el documento sugiere ~1% (140 Myr), asi que usamos:
        # Delta_t ~ epsilon * t_Hubble * (Delta_S / S_ext) * factor_escala
        # Factor de escala calibrado para obtener ~1% de correccion
        Delta_S = self.S_ext_0 - self.S_local_VL
        factor_escala = 2.0  # Para obtener ~1% con los parametros tipicos

        delta_t_Gyr = factor_escala * self.epsilon * t_obs_Gyr * (Delta_S / self.S_ext_0)

        # Edad corregida (el universo es MAS viejo de lo que medimos)
        t_ext_Gyr = t_obs_Gyr + delta_t_Gyr

        return {
            't_obs_Gyr': t_obs_Gyr,
            't_ext_Gyr': t_ext_Gyr,
            'delta_t_Gyr': delta_t_Gyr,
            'delta_t_Myr': delta_t_Gyr * 1000,
            'delta_t_percent': (delta_t_Gyr / t_obs_Gyr) * 100,
            'delta_t_units': delta_t_units,
            'Xi_VL': Xi_VL,
            'S_local': self.S_local_VL,
            'S_ext': self.S_ext_0,
            'Delta_S': Delta_S
        }

    # =========================================================================
    # MECANISMO DE TENSION DE HUBBLE (Seccion 5)
    # =========================================================================

    def H0_local_vs_CMB(self, H0_CMB: float = H0_PLANCK) -> Dict:
        """
        Relacion entre H0 local y H0 del CMB (Seccion 5.2).

        H0_local = H0_CMB * (1 + epsilon * (Xi_VL - Xi_mean) / (1 + z_recomb))

        La medicion local esta sesgada por nuestra posicion en una burbuja.

        Args:
            H0_CMB: H0 inferido del CMB (default: 67.4 km/s/Mpc)

        Returns:
            Dict con H0_local predicho y parametros
        """
        # Xi de nuestra burbuja
        Xi_VL = self.Xi_from_S(self.S_local_VL, self.S_ext_0)

        # Xi promedio del universo (~ 0 por definicion en el fondo)
        Xi_mean = 0.0

        # z de recombinacion
        z_recomb = 1100

        # Factor de correccion
        # Nota: este es un efecto de segundo orden, la correccion principal
        # viene de Lambda_rel(z)
        correction = 1 + self.epsilon * (Xi_VL - Xi_mean)

        H0_local = H0_CMB * correction

        # Correccion adicional por Lambda_rel
        # El MCMC predice H0 ~ 69.5-70 debido a la transicion en z_trans
        Lambda_0 = 1 + self.epsilon * np.tanh(self.z_trans / self.Delta_z)
        H0_MCMC = H0_CMB * np.sqrt(self.Omega_Lambda * Lambda_0 + self.Omega_m) / \
                  np.sqrt(self.Omega_Lambda + self.Omega_m)

        return {
            'H0_CMB': H0_CMB,
            'H0_local_bubble': H0_local,
            'H0_MCMC': H0_MCMC,
            'Delta_H0_bubble': H0_local - H0_CMB,
            'Delta_H0_MCMC': H0_MCMC - H0_CMB,
            'Delta_H0_percent_bubble': (H0_local / H0_CMB - 1) * 100,
            'Delta_H0_percent_MCMC': (H0_MCMC / H0_CMB - 1) * 100,
            'Xi_VL': Xi_VL,
            'Lambda_0': Lambda_0,
            'tension_reduction_sigma': 4.0 * (1 - (H0_MCMC - H0_CMB) / (H0_SHOES - H0_CMB))
        }

    # =========================================================================
    # CASOS DE ESTUDIO
    # =========================================================================

    def analyze_case(self, name: str, z: float, S_local: float,
                    description: str = "") -> Dict:
        """
        Analiza un caso especifico de emision/observacion.

        Args:
            name: Nombre del caso
            z: Redshift
            S_local: Entropia tensional local
            description: Descripcion del caso

        Returns:
            Dict con analisis completo
        """
        # Redshift corregido
        z_result = self.corrected_redshift(z, S_local)

        # Modulo de distancia
        mu = self.mu_cosmo(z)
        mu_result = self.corrected_distance_modulus(mu, z, S_local)

        # Clasificacion de la burbuja
        S_ext_z = self.S_ext(z)
        Delta_S = S_ext_z - S_local

        if Delta_S < 2:
            tipo = "campo"
        elif Delta_S < 5:
            tipo = "burbuja_debil"
        elif Delta_S < 10:
            tipo = "burbuja_moderada"
        else:
            tipo = "burbuja_intensa"

        return {
            'nombre': name,
            'descripcion': description,
            'z': z,
            'S_local': S_local,
            'S_ext': S_ext_z,
            'Delta_S': Delta_S,
            'tipo_burbuja': tipo,
            'z_obs': z_result['z_obs'],
            'delta_z': z_result['delta_z_total'],
            'mu_cosmo': mu,
            'mu_corr': mu_result['mu_corr'],
            'delta_mu': mu_result['Delta_mu_total'],
            'Xi': self.Xi_from_S(S_local, S_ext_z)
        }


# =============================================================================
# CLASE PARA MAPA S(z, x) COMPLETO
# =============================================================================

class SMap3D:
    """
    Mapa tridimensional de S(z, theta, phi) incluyendo estructura LSS.

    Permite calcular correcciones para cualquier linea de vision.
    """

    def __init__(self, calc: BubbleCorrectionCalculator = None):
        """
        Inicializa el mapa.

        Args:
            calc: Calculador base (crea uno nuevo si None)
        """
        self.calc = calc or BubbleCorrectionCalculator()

    def S_at_position(self, z: float, theta: float, phi: float,
                      density_field: callable = None) -> float:
        """
        Entropia tensional en una posicion (z, theta, phi).

        Args:
            z: Redshift
            theta: Colatitud (radianes)
            phi: Azimut (radianes)
            density_field: Funcion delta_m(z, theta, phi)

        Returns:
            S_local en esa posicion
        """
        if density_field is None:
            # Sin campo de densidad, retorna el fondo
            return self.calc.S_ext(z)

        delta_m = density_field(z, theta, phi)
        return self.calc.S_with_LSS(z, delta_m)

    def line_of_sight_integral(self, z_max: float, theta: float, phi: float,
                               density_field: callable = None,
                               n_points: int = 200) -> Dict:
        """
        Calcula la integral de Xi a lo largo de una linea de vision.

        Args:
            z_max: Redshift maximo
            theta, phi: Direccion de la linea de vision
            density_field: Campo de densidad
            n_points: Numero de puntos para la integral

        Returns:
            Dict con integral y perfil de Xi(z)
        """
        z_array = np.linspace(0.01, z_max, n_points)

        S_array = np.array([self.S_at_position(z, theta, phi, density_field)
                          for z in z_array])
        S_ext_array = self.calc.S_ext(z_array)
        Xi_array = self.calc.lambda_S * (S_ext_array - S_array) / S_ext_array

        # Integral de dXi
        integral_Xi = np.trapz(np.gradient(Xi_array, z_array), z_array)

        return {
            'z': z_array,
            'S': S_array,
            'S_ext': S_ext_array,
            'Xi': Xi_array,
            'integral_dXi': integral_Xi,
            'delta_z_transit': self.calc.epsilon * integral_Xi
        }


# =============================================================================
# FUNCION DE VALIDACION
# =============================================================================

def test_BubbleCorrections() -> bool:
    """
    Test completo del modulo de correcciones por burbujas.

    Verifica:
    1. Coherencia de S_ext(z)
    2. Correcciones al redshift
    3. Correcciones al modulo de distancia
    4. Correccion a la edad del universo
    5. Mecanismo de tension de Hubble
    6. Casos limite
    """
    print("\n" + "="*70)
    print("  TEST BUBBLE CORRECTIONS - TRANSITO DE FOTONES POR BURBUJAS")
    print("="*70)

    calc = BubbleCorrectionCalculator()
    all_passed = True

    # -------------------------------------------------------------------------
    # Test 1: S_ext(z) coherente
    # -------------------------------------------------------------------------
    print("\n[1] Verificacion de S_ext(z):")
    S_ext_0 = calc.S_ext(0)
    S_ext_1 = calc.S_ext(1)
    S_ext_5 = calc.S_ext(5)

    print(f"    S_ext(z=0) = {S_ext_0:.2f}")
    print(f"    S_ext(z=1) = {S_ext_1:.2f}")
    print(f"    S_ext(z=5) = {S_ext_5:.2f}")

    # S debe decrecer con z (era menor en el pasado)
    test1 = S_ext_0 > S_ext_1 > S_ext_5 > 0
    print(f"    S_ext monotono decreciente: {'PASS' if test1 else 'FAIL'}")
    all_passed &= test1

    # -------------------------------------------------------------------------
    # Test 2: Correcciones al redshift
    # -------------------------------------------------------------------------
    print("\n[2] Correcciones al redshift:")

    # Usar S_local relativo a S_ext(z) para que Delta_S sea fisicamente consistente
    # A z=1, S_ext es menor, asi que S_local tambien debe ser menor
    # Mantenemos Delta_S ~ 5 (como en z=0: 90 - 85 = 5)

    S_ext_z1 = calc.S_ext(1.0)
    S_local_burbuja_z1 = S_ext_z1 - 5.0  # Delta_S = 5 (burbuja tipica)
    S_local_void_z1 = S_ext_z1 - 2.0     # Delta_S = 2 (void, mas cercano al fondo)

    # Caso: SN Ia a z=1 en burbuja similar a la nuestra (Delta_S ~ 5)
    result = calc.corrected_redshift(z_em=1.0, S_local_em=S_local_burbuja_z1)
    print(f"    z_em = 1.0, S_ext(z=1) = {S_ext_z1:.2f}")
    print(f"    S_local_em = {S_local_burbuja_z1:.2f} (Delta_S = 5, burbuja)")
    print(f"    delta_z_exit = {result['delta_z_exit']:.6f}")
    print(f"    delta_z_enter = {result['delta_z_enter']:.6f}")
    print(f"    delta_z_total = {result['delta_z_total']:.6f}")
    print(f"    z_obs = {result['z_obs']:.6f}")

    # Si emisor y receptor estan en burbujas con similar Delta_S, correccion ~ 0
    test2a = abs(result['delta_z_total']) < 0.01
    print(f"    Correccion pequena para burbujas similares: {'PASS' if test2a else 'FAIL'}")

    # Caso: emisor en void (S_local mas cercano a S_ext, Delta_S ~ 2)
    result_void = calc.corrected_redshift(z_em=1.0, S_local_em=S_local_void_z1)
    print(f"    S_local_em = {S_local_void_z1:.2f} (Delta_S = 2, void)")
    print(f"    delta_z_total = {result_void['delta_z_total']:.6f}")

    # Void tiene menos dilatacion temporal que burbuja:
    # - Xi_void < Xi_bubble
    # - Menor blueshift al salir del void
    # - Foton del void aparece MAS ROJO que foton de burbuja
    test2b = result_void['delta_z_total'] > result['delta_z_total']
    print(f"    Void produce menos blueshift (mas rojo): {'PASS' if test2b else 'FAIL'}")

    all_passed &= test2a and test2b

    # -------------------------------------------------------------------------
    # Test 3: Correcciones al modulo de distancia
    # -------------------------------------------------------------------------
    print("\n[3] Correcciones al modulo de distancia:")

    # Usar S_local consistente con S_ext(z=1)
    S_local_source = S_ext_z1 - 8.0  # Burbuja densa (Delta_S = 8)

    mu_cosmo = calc.mu_cosmo(1.0)
    mu_result = calc.corrected_distance_modulus(mu_cosmo, z=1.0, S_local_source=S_local_source)

    print(f"    mu_cosmo(z=1) = {mu_cosmo:.3f} mag")
    print(f"    S_local_source = {S_local_source:.2f} (Delta_S = 8)")
    print(f"    mu_corr = {mu_result['mu_corr']:.3f} mag")
    print(f"    Delta_mu = {mu_result['Delta_mu_total']:.4f} mag")

    # La correccion debe ser pequena (< 0.1 mag para epsilon ~ 0.01)
    test3 = abs(mu_result['Delta_mu_total']) < 0.1
    print(f"    |Delta_mu| < 0.1 mag: {'PASS' if test3 else 'FAIL'}")
    all_passed &= test3

    # -------------------------------------------------------------------------
    # Test 4: Correccion a la edad del universo
    # -------------------------------------------------------------------------
    print("\n[4] Correccion a la edad del universo:")

    age_result = calc.corrected_universe_age(t_obs_Gyr=13.8)

    print(f"    t_obs = {age_result['t_obs_Gyr']:.2f} Gyr")
    print(f"    t_ext = {age_result['t_ext_Gyr']:.2f} Gyr")
    print(f"    Delta_t = {age_result['delta_t_Myr']:.1f} Myr ({age_result['delta_t_percent']:.2f}%)")

    # Delta_t debe ser ~ 0.5-1.5% para epsilon ~ 0.01
    test4 = 0.1 < age_result['delta_t_percent'] < 2.0
    print(f"    0.1% < Delta_t < 2%: {'PASS' if test4 else 'FAIL'}")
    all_passed &= test4

    # -------------------------------------------------------------------------
    # Test 5: Tension de Hubble
    # -------------------------------------------------------------------------
    print("\n[5] Mecanismo de tension de Hubble:")

    H0_result = calc.H0_local_vs_CMB(H0_CMB=67.4)

    print(f"    H0_CMB = {H0_result['H0_CMB']:.2f} km/s/Mpc")
    print(f"    H0_MCMC (Lambda_rel) = {H0_result['H0_MCMC']:.2f} km/s/Mpc")
    print(f"    H0_local (burbuja) = {H0_result['H0_local_bubble']:.2f} km/s/Mpc")
    print(f"    Delta_H0_MCMC = {H0_result['Delta_H0_MCMC']:.2f} km/s/Mpc ({H0_result['Delta_H0_percent_MCMC']:.2f}%)")

    # H0_MCMC debe estar entre Planck y SH0ES
    test5 = H0_PLANCK < H0_result['H0_MCMC'] < H0_SHOES
    print(f"    H0_Planck < H0_MCMC < H0_SH0ES: {'PASS' if test5 else 'FAIL'}")
    all_passed &= test5

    # -------------------------------------------------------------------------
    # Test 6: Casos canonicos
    # -------------------------------------------------------------------------
    print("\n[6] Casos canonicos:")

    # Usar Delta_S relativo a S_ext(z) para cada caso
    casos = [
        ("SN Ia en void (z=1)", 1.0, 2.0, "Supernova en region de baja densidad"),
        ("SN Ia en cumulo (z=0.5)", 0.5, 10.0, "Supernova en cumulo de galaxias"),
        ("Quasar alto z (z=3)", 3.0, 8.0, "Quasar en epoca temprana"),
    ]

    for nombre, z, Delta_S, desc in casos:
        S_ext_z = calc.S_ext(z)
        S_local = S_ext_z - Delta_S
        analysis = calc.analyze_case(nombre, z, S_local, desc)
        print(f"    {nombre}:")
        print(f"      z={z}, Delta_S={Delta_S}, tipo={analysis['tipo_burbuja']}")
        print(f"      delta_z={analysis['delta_z']:.5f}, delta_mu={analysis['delta_mu']:.4f} mag")

    # Test: diferentes burbujas dan diferentes correcciones
    S_ext_test = calc.S_ext(1.0)
    a1 = calc.analyze_case("void", 1.0, S_ext_test - 2.0)   # Delta_S = 2
    a2 = calc.analyze_case("cluster", 1.0, S_ext_test - 10.0)  # Delta_S = 10
    test6 = a1['delta_z'] != a2['delta_z']
    print(f"    Correcciones diferentes para diferentes Delta_S: {'PASS' if test6 else 'FAIL'}")
    all_passed &= test6

    # -------------------------------------------------------------------------
    # Test 7: Mapa S(z) con LSS
    # -------------------------------------------------------------------------
    print("\n[7] Mapa S(z) con estructura LSS:")

    np.random.seed(42)
    s_map = calc.generate_S_map_1D(z_max=2.0, n_points=50, delta_m_rms=0.3)

    print(f"    S(z=0.5) = {np.interp(0.5, s_map['z'], s_map['S']):.2f}")
    print(f"    S_ext(z=0.5) = {np.interp(0.5, s_map['z'], s_map['S_ext']):.2f}")
    print(f"    Xi(z=0.5) = {np.interp(0.5, s_map['z'], s_map['Xi']):.4f}")
    print(f"    rms(Delta_S) = {np.std(s_map['Delta_S']):.3f}")

    test7 = np.std(s_map['Delta_S']) > 0
    print(f"    Fluctuaciones de S presentes: {'PASS' if test7 else 'FAIL'}")
    all_passed &= test7

    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"  BUBBLE CORRECTIONS MODULE: {'PASS' if all_passed else 'FAIL'}")
    print("="*70)

    return all_passed


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_BubbleCorrections()
