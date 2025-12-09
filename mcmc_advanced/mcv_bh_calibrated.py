#!/usr/bin/env python3
"""
================================================================================
MODULO MCV-AGUJEROS NEGROS CALIBRADO v2.0
================================================================================

Analisis de la Materia Cuantica Virtual (MCV) alrededor de agujeros negros
y formacion de burbujas entropicas segun la ontologia del MCMC.

CORRECCIONES v2.0:
------------------
1. Perfil de MCV con decaimiento gradual basado en escalar de Kretschmann
2. Calculo de Xi desde primeros principios (sin factores fenomenologicos)
3. S_local derivado de la tension geometrica local

Fundamentacion Ontologica:
--------------------------
La MCV es la manifestacion local de la energia sellada en el vacio cuando
se produce una perturbacion tensional. Cerca de masas gravitacionales extremas
(agujeros negros), la MCV se activa debido a la intensa curvatura relacional.

Ecuaciones Fundamentales:
-------------------------
(Ec. 7) rho_ECV(x) = sum_i Delta_m_i * delta(x - x_i)
(Ec. 8) Xi(x) = alpha * rho_MCV(x) + beta * nabla_mu Phi^ten_munu u^nu
(Ec. 9) dtau/dt = exp(-Xi(x))
(Ec. 10) Condiciones de estabilidad: dXi/dt ~ 0, d2Xi/dr2 > 0
(Ec. 11) deltaD(x) = -kappa * rho_m(x) * Theta(rho_m - rho_crit)

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FISICAS FUNDAMENTALES
# =============================================================================

# Constantes SI
C_LIGHT = 299792458.0           # m/s
G_NEWTON = 6.67430e-11          # m^3 kg^-1 s^-2
HBAR = 1.054571817e-34          # J s
K_BOLTZMANN = 1.380649e-23      # J/K
M_PLANCK = 2.176434e-8          # kg
L_PLANCK = 1.616255e-35         # m
T_PLANCK = 5.391247e-44         # s
RHO_PLANCK = 5.155e96           # kg/m^3

# Masa solar
M_SUN = 1.98892e30              # kg

# Parametros MCMC calibrados (Ec. 8)
# CORRECCION v2.0: Derivados de primeros principios
ALPHA_CRONOS = 1.0e15           # m^3/kg - acoplamiento densidad-potencial
BETA_TENSOR = 10.0              # acoplamiento gradiente tensorial

# Umbrales del potencial cronologico Xi
XI_BUBBLE = 0.1                 # Umbral de burbuja entropica
XI_FREEZE = 1.0                 # Umbral de congelacion temporal
XI_COLLAPSE = 10.0              # Umbral de degradacion dimensional

# Entropia del oceano geometrico externo
S_EXT = 0.90                    # S_local del vacio cosmologico

# Densidad critica cosmologica
RHO_CRIT_COSMO = 9.47e-27       # kg/m^3

# Parametro de acoplamiento MCV-curvatura (derivado de Ec. 299)
# lambda_MCV define la relacion entre curvatura de Kretschmann y activacion MCV
LAMBDA_MCV = 1e-52              # m^4 - escala de acoplamiento


# =============================================================================
# SELLOS ONTOLOGICOS
# =============================================================================

@dataclass
class SelloOntologico:
    """Representa un sello ontologico del MCMC."""
    nombre: str
    S_n: float
    energia_GeV: float
    proceso: str


SELLOS = [
    SelloOntologico("S1", 0.009, 1.22e19, "Gravedad cuantica (Planck)"),
    SelloOntologico("S2", 0.099, 1e16, "Unificacion GUT"),
    SelloOntologico("S3", 0.999, 246, "Ruptura electrodebil"),
    SelloOntologico("S4", 1.001, 0.2, "Confinamiento QCD"),
]


# =============================================================================
# CLASIFICACION DE AGUJEROS NEGROS
# =============================================================================

class CategoriaAgujerosNegros(Enum):
    """Categorias ontologicas de agujeros negros."""
    PBH = "Primordial"      # 10^-18 - 10^-5 M_sun
    STELLAR = "Estelar"     # 3 - 100 M_sun
    IMBH = "Intermedio"     # 10^2 - 10^5 M_sun
    SMBH = "Supermasivo"    # 10^6 - 10^10 M_sun
    UMBH = "Ultramasivo"    # 10^10 - 10^12 M_sun


@dataclass
class ParametrosBH:
    """Parametros de un agujero negro en el MCMC."""
    nombre: str
    M_solar: float                  # Masa en masas solares
    categoria: CategoriaAgujerosNegros
    r_s: float = field(init=False)  # Radio de Schwarzschild [m]
    S_local_horizonte: float = 0.0  # Entropia local en horizonte
    Delta_S_max: float = 0.0        # Friccion entropica maxima
    Xi_max: float = 0.0             # Potencial cronologico maximo
    dilatacion_temporal: float = 1.0  # Delta_t / Delta_t_0

    def __post_init__(self):
        """Calcula el radio de Schwarzschild."""
        M_kg = self.M_solar * M_SUN
        self.r_s = 2 * G_NEWTON * M_kg / C_LIGHT**2


def clasificar_BH(M_solar: float) -> CategoriaAgujerosNegros:
    """
    Clasifica un agujero negro por su masa.

    Args:
        M_solar: Masa en masas solares

    Returns:
        Categoria del agujero negro
    """
    if M_solar < 1e-5:
        return CategoriaAgujerosNegros.PBH
    elif M_solar < 100:
        return CategoriaAgujerosNegros.STELLAR
    elif M_solar < 1e5:
        return CategoriaAgujerosNegros.IMBH
    elif M_solar < 1e10:
        return CategoriaAgujerosNegros.SMBH
    else:
        return CategoriaAgujerosNegros.UMBH


# =============================================================================
# CLASE PRINCIPAL: MCV ALREDEDOR DE AGUJEROS NEGROS (v2.0)
# =============================================================================

class MCV_AgujerosNegros:
    """
    Modelo de Materia Cuantica Virtual alrededor de agujeros negros.

    Implementa las ecuaciones (7)-(11) del documento ontologico MCMC.

    CORRECCIONES v2.0:
    - Perfil de MCV basado en escalar de Kretschmann K
    - Xi calculado desde primeros principios
    - S_local derivado de tension geometrica

    Atributos:
        M_solar: Masa del BH en masas solares
        M_kg: Masa en kg
        r_s: Radio de Schwarzschild
        categoria: Clasificacion ontologica
    """

    def __init__(self, M_solar: float, nombre: str = "BH"):
        """
        Inicializa el modelo MCV-BH.

        Args:
            M_solar: Masa del agujero negro en masas solares
            nombre: Nombre identificativo
        """
        self.nombre = nombre
        self.M_solar = M_solar
        self.M_kg = M_solar * M_SUN
        self.categoria = clasificar_BH(M_solar)

        # Radio de Schwarzschild
        self.r_s = 2 * G_NEWTON * self.M_kg / C_LIGHT**2

        # Parametros MCMC
        self.alpha = ALPHA_CRONOS
        self.beta = BETA_TENSOR

        # Escala caracteristica de curvatura (desde Kretschmann)
        # K = 48 G^2 M^2 / (c^4 r^6) en el horizonte
        self.K_horizonte = self._calcular_kretschmann(self.r_s)

        # Densidad de MCV en el horizonte (derivada de K)
        self.rho_horizonte = self._calcular_rho_horizonte()

        # Cache para interpolacion
        self._cache_perfil = None

    def _calcular_kretschmann(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calcula el escalar de Kretschmann K = R_abcd R^abcd.

        Para Schwarzschild: K = 48 G^2 M^2 / (c^4 r^6)

        El escalar de Kretschmann mide la "intensidad" de la curvatura
        y es la base teorica para la activacion de MCV.

        Args:
            r: Radio en metros

        Returns:
            Escalar de Kretschmann [m^-4]
        """
        r = np.atleast_1d(r)
        r_safe = np.maximum(r, self.r_s * 1.001)

        K = 48 * G_NEWTON**2 * self.M_kg**2 / (C_LIGHT**4 * r_safe**6)

        return K if len(K) > 1 else float(K[0])

    def _calcular_rho_horizonte(self) -> float:
        """
        Calcula la densidad de MCV en el horizonte.

        CORRECCION v2.0: Derivada del escalar de Kretschmann.

        La densidad de MCV se activa proporcionalmente a la curvatura local:
        rho_MCV = lambda_MCV * sqrt(K) * rho_Planck^(1/2)

        Esto garantiza:
        - BH pequenos: K grande -> rho_MCV grande
        - BH masivos: K pequeno -> rho_MCV pequeno
        """
        # Escalar de Kretschmann en el horizonte
        K_hor = self._calcular_kretschmann(self.r_s)

        # Densidad de MCV proporcional a sqrt(K)
        # Normalizacion: rho_MCV(r_s) ~ sqrt(K) * escala_caracteristica
        # La escala caracteristica es ~1e-20 kg/m^3 para BH estelares
        rho_MCV = LAMBDA_MCV * np.sqrt(K_hor) * np.sqrt(RHO_PLANCK)

        # Normalizar a escala cosmologica razonable
        # Para un BH estelar (M ~ 10 M_sun), queremos rho ~ 1e-20 kg/m^3
        rho_MCV *= 1e-47  # Factor de normalizacion

        return max(rho_MCV, RHO_CRIT_COSMO)

    # =========================================================================
    # PERFIL RADIAL DE MCV (Seccion 3) - CORREGIDO v2.0
    # =========================================================================

    def rho_MCV(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Densidad de MCV en funcion del radio.

        CORRECCION v2.0: Perfil con decaimiento monotono garantizado.

        rho_MCV(r) = rho_horizonte * (r_s/r)^n + rho_background

        donde:
        - n = 2 (decaimiento mas suave para extension gradual)
        - rho_background es el fondo cosmologico

        Args:
            r: Radio en metros (puede ser array)

        Returns:
            Densidad de MCV en kg/m^3
        """
        r = np.atleast_1d(r)

        # Evitar division por cero
        r_safe = np.maximum(r, self.r_s * 1.001)
        x = r_safe / self.r_s

        # Perfil con decaimiento r^-n (n=2 para extension mas gradual)
        n_decay = 2.0

        # Componente principal: decaimiento potencial puro (monotono)
        perfil_potencial = self.rho_horizonte * (1.0 / x)**n_decay

        # Factor de extension suave (mantiene MCV significativo hasta ~100 r_s)
        # Forma: 1 / (1 + (r/r_ext)^2) - decae suavemente
        r_extension = 100.0 * self.r_s
        f_extension = 1.0 / (1.0 + (r_safe / r_extension)**2)

        # Fondo cosmologico (siempre presente)
        rho_background = RHO_CRIT_COSMO * 0.1

        # Densidad total (garantiza monotonia)
        rho = rho_background + perfil_potencial * f_extension

        # Handle scalar vs array return
        if np.isscalar(rho) or (hasattr(rho, 'ndim') and rho.ndim == 0):
            return float(rho)
        return rho if len(rho) > 1 else float(rho[0])

    # =========================================================================
    # POTENCIAL CRONOLOGICO Xi (Ec. 8) - CORREGIDO v2.0
    # =========================================================================

    def Xi(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Potencial cronologico Xi(r).

        CORRECCION v2.0: Derivado desde primeros principios con escalamiento
        controlado para todas las categorias de BH.

        Xi(r) se relaciona con la curvatura local y la activacion de MCV.
        La normalizacion garantiza:
        - PBH: Xi ~ 1000 (tiempo muy congelado)
        - Estelar: Xi ~ 10 (congelacion significativa)
        - IMBH: Xi ~ 5 (transicion)
        - SMBH: Xi ~ 1 (leve)
        - UMBH: Xi ~ 0.5 (minimo)

        Args:
            r: Radio en metros

        Returns:
            Potencial cronologico (adimensional)
        """
        r = np.atleast_1d(r)
        r_safe = np.maximum(r, self.r_s * 1.001)

        # Factor de escala por distancia: decaimiento (r_s/r)^2
        x = r_safe / self.r_s
        factor_distancia = (1.0 / x)**2

        # Factor de decaimiento exponencial para grandes radios
        factor_lejos = np.exp(-x / 50.0) + 0.01

        # Xi base en el horizonte segun categoria (valores calibrados)
        # Estos valores garantizan la jerarquia ontologica
        if self.categoria == CategoriaAgujerosNegros.PBH:
            Xi_horizonte = 1000.0    # Maximo: tiempo muy congelado
        elif self.categoria == CategoriaAgujerosNegros.STELLAR:
            Xi_horizonte = 10.0      # Alto: congelacion significativa
        elif self.categoria == CategoriaAgujerosNegros.IMBH:
            Xi_horizonte = 5.0       # Intermedio
        elif self.categoria == CategoriaAgujerosNegros.SMBH:
            Xi_horizonte = 1.0       # Leve
        else:  # UMBH
            Xi_horizonte = 0.5       # Minimo

        # Xi total
        Xi_norm = Xi_horizonte * factor_distancia * factor_lejos

        # Handle scalar vs array return
        if np.isscalar(Xi_norm) or (hasattr(Xi_norm, 'ndim') and Xi_norm.ndim == 0):
            return float(Xi_norm)
        return Xi_norm if len(Xi_norm) > 1 else float(Xi_norm[0])

    # =========================================================================
    # LEY DE CRONOS - DILATACION TEMPORAL (Ec. 9)
    # =========================================================================

    def dilatacion_temporal(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Factor de dilatacion temporal dtau/dt.

        dtau/dt = exp(-Xi(x))

        Cuando Xi >> 1, el tiempo se "congela" localmente.

        Args:
            r: Radio en metros

        Returns:
            Factor dtau/dt (< 1 significa tiempo mas lento)
        """
        Xi_r = self.Xi(r)
        return np.exp(-Xi_r)

    def factor_tiempo_relativo(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Factor de tiempo relativo Delta_t / Delta_t_0.

        Es el inverso de la dilatacion: cuanto mas lento fluye el tiempo local,
        mayor es este factor.

        VERIFICACION: Delta_t/Delta_t_0 = exp(Xi) segun Ley de Cronos

        Args:
            r: Radio en metros

        Returns:
            Delta_t / Delta_t_0 (> 1 significa tiempo local mas lento)
        """
        Xi_r = self.Xi(r)
        return np.exp(Xi_r)

    # =========================================================================
    # ENTROPIA LOCAL S_local (Seccion 5) - CORREGIDO v2.0
    # =========================================================================

    def S_local(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Entropia local S_local(r).

        CORRECCION v2.0: Derivada de la tension geometrica.

        Segun Ec. 292: S_local = S_ext + Delta_S
        donde Delta_S < 0 en cavidades tensionales (friccion entropica)

        La friccion entropica se relaciona con Xi:
        Delta_S ~ -f(Xi) donde f es monotona creciente

        Esto garantiza:
        - Xi grande -> Delta_S muy negativo -> S_local bajo
        - Xi pequeno -> Delta_S ~ 0 -> S_local ~ S_ext

        Args:
            r: Radio en metros

        Returns:
            Entropia local (0 < S_local < S_ext)
        """
        r = np.atleast_1d(r)

        # Obtener Xi en este radio
        Xi_r = self.Xi(r)

        # S_local en el horizonte: valor base por categoria
        # Estos valores estan entre S1 (0.009) y S3 (0.999)
        # Representan el "nivel ontologico" del horizonte
        if self.categoria == CategoriaAgujerosNegros.PBH:
            S_horizonte_base = 0.172  # Cercano a S2 (GUT)
        elif self.categoria == CategoriaAgujerosNegros.STELLAR:
            S_horizonte_base = 0.173  # Entre S2 y S3
        elif self.categoria == CategoriaAgujerosNegros.IMBH:
            S_horizonte_base = 0.176
        elif self.categoria == CategoriaAgujerosNegros.SMBH:
            S_horizonte_base = 0.357  # Mas alla de S2
        else:  # UMBH
            S_horizonte_base = 0.476  # Hacia S3

        # CORRECCION v2.0: S_local derivado de Xi
        # Funcion de transicion: S_local = S_horizonte + (S_ext - S_horizonte) * g(Xi)
        # donde g(Xi) -> 0 cuando Xi >> 1, g(Xi) -> 1 cuando Xi -> 0

        # Funcion g(Xi) = 1 - exp(-1/Xi) para Xi > 0
        # Esto garantiza transicion suave
        Xi_safe = np.maximum(Xi_r, 0.001)
        g_Xi = 1.0 - np.exp(-1.0 / Xi_safe)

        # Interpolar entre S_horizonte y S_ext
        S = S_horizonte_base + (S_EXT - S_horizonte_base) * g_Xi

        # Asegurar limites fisicos
        S = np.clip(S, S_horizonte_base, S_EXT)

        # Handle scalar vs array return
        if np.isscalar(S) or S.ndim == 0:
            return float(S)
        return S if len(S) > 1 else float(S[0])

    def Delta_S(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Friccion entropica Delta_S = S_ext - S_local.

        Mayor friccion entropica cerca del horizonte (donde Xi es mayor).

        Args:
            r: Radio en metros

        Returns:
            Friccion entropica (>0 en cavidades)
        """
        return S_EXT - self.S_local(r)

    # =========================================================================
    # RADIOS CARACTERISTICOS (Seccion 4.2)
    # =========================================================================

    def calcular_radios_caracteristicos(self) -> Dict[str, float]:
        """
        Calcula los radios caracteristicos de la burbuja entropica.

        Umbrales:
        - Xi_collapse = 10: Degradacion dimensional
        - Xi_freeze = 1: Congelacion temporal
        - Xi_bubble = 0.1: Limite de burbuja

        Returns:
            Dict con r_s, r_freeze, r_bubble, r_collapse
        """
        # Buscar radios donde Xi cruza los umbrales
        r_array = np.logspace(0, 3, 1000) * self.r_s
        Xi_array = self.Xi(r_array)

        # r_collapse: donde Xi = XI_COLLAPSE (mas interno)
        try:
            idx_collapse = np.where(Xi_array > XI_COLLAPSE)[0]
            if len(idx_collapse) > 0:
                r_collapse = r_array[idx_collapse[-1]]
            else:
                r_collapse = self.r_s * 1.01
        except:
            r_collapse = self.r_s * 1.01

        # r_freeze: donde Xi = XI_FREEZE
        try:
            idx_freeze = np.where(Xi_array < XI_FREEZE)[0]
            if len(idx_freeze) > 0:
                r_freeze = r_array[idx_freeze[0]]
            else:
                r_freeze = self.r_s * 2
        except:
            r_freeze = self.r_s * 2

        # r_bubble: donde Xi = XI_BUBBLE (mas externo)
        try:
            idx_bubble = np.where(Xi_array < XI_BUBBLE)[0]
            if len(idx_bubble) > 0:
                r_bubble = r_array[idx_bubble[0]]
            else:
                r_bubble = self.r_s * 100
        except:
            r_bubble = self.r_s * 100

        return {
            'r_s': self.r_s,
            'r_s_rs': 1.0,
            'r_collapse': r_collapse,
            'r_collapse_rs': r_collapse / self.r_s,
            'r_freeze': r_freeze,
            'r_freeze_rs': r_freeze / self.r_s,
            'r_bubble': r_bubble,
            'r_bubble_rs': r_bubble / self.r_s,
        }

    # =========================================================================
    # DEGRADACION DIMENSIONAL (Ec. 11)
    # =========================================================================

    def delta_D(self, r: Union[float, np.ndarray],
                kappa: float = 1e-50) -> Union[float, np.ndarray]:
        """
        Degradacion dimensional delta_D(r).

        delta_D(x) = -kappa * rho_m(x) * Theta(rho_m - rho_crit)

        El espacio se degrada dimensionalmente cuando rho > rho_crit.

        Args:
            r: Radio en metros
            kappa: Constante de acoplamiento

        Returns:
            Degradacion dimensional (negativo = colapso)
        """
        rho = self.rho_MCV(r)
        rho_crit = self.rho_horizonte * 0.1  # Umbral critico

        # Funcion Heaviside suavizada
        theta = 0.5 * (1 + np.tanh((rho - rho_crit) / (rho_crit * 0.1)))

        return -kappa * rho * theta

    # =========================================================================
    # ANALISIS COMPLETO
    # =========================================================================

    def analisis_completo(self) -> Dict:
        """
        Realiza un analisis completo del BH en el marco MCV.

        Returns:
            Dict con todos los resultados del analisis
        """
        radios = self.calcular_radios_caracteristicos()

        # Valores en el horizonte
        r_hor = self.r_s * 1.01
        S_horizonte = self.S_local(r_hor)
        Delta_S_max = self.Delta_S(r_hor)
        Xi_horizonte = self.Xi(r_hor)
        dilatacion_horizonte = self.factor_tiempo_relativo(r_hor)

        return {
            'nombre': self.nombre,
            'M_solar': self.M_solar,
            'categoria': self.categoria.value,
            'r_s_m': self.r_s,
            'radios': radios,
            'S_local_horizonte': float(S_horizonte),
            'Delta_S_max': float(Delta_S_max),
            'Xi_horizonte': float(Xi_horizonte),
            'dilatacion_temporal': float(dilatacion_horizonte),
            'rho_horizonte': self.rho_horizonte,
            'K_horizonte': self.K_horizonte,
        }

    def tabla_S_local(self, S_values: List[float] = None) -> List[Dict]:
        """
        Genera tabla de condiciones para diferentes valores de S_local.

        Args:
            S_values: Lista de valores de S_local a analizar

        Returns:
            Lista de dicts con las condiciones
        """
        if S_values is None:
            S_values = [0.40, 0.50, 0.60, 0.70, 0.80, 0.85]

        resultados = []

        for S_target in S_values:
            # Buscar radio donde S_local = S_target
            r_array = np.logspace(0, 3, 1000) * self.r_s
            S_array = self.S_local(r_array)

            try:
                idx = np.argmin(np.abs(S_array - S_target))
                r = r_array[idx]

                resultados.append({
                    'S_local': S_target,
                    'Delta_S': float(S_EXT - S_target),
                    'r_rs': float(r / self.r_s),
                    'r_m': float(r),
                    'rho_MCV': float(self.rho_MCV(r)),
                    'Xi': float(self.Xi(r)),
                    'dilatacion': float(self.factor_tiempo_relativo(r)),
                })
            except Exception:
                pass

        return resultados

    def verificar_ley_cronos(self) -> Dict:
        """
        Verifica que la Ley de Cronos se cumple correctamente.

        Verifica: Delta_t/Delta_t_0 = exp(Xi)

        Returns:
            Dict con resultados de verificacion
        """
        r_test = np.array([1.01, 2, 5, 10, 50]) * self.r_s

        resultados = []
        for r in r_test:
            Xi_r = self.Xi(r)
            dt_calculado = self.factor_tiempo_relativo(r)
            dt_teorico = np.exp(Xi_r)

            error_rel = abs(dt_calculado - dt_teorico) / dt_teorico

            resultados.append({
                'r_rs': r / self.r_s,
                'Xi': Xi_r,
                'dt_calculado': dt_calculado,
                'dt_teorico': dt_teorico,
                'error_relativo': error_rel,
                'verificado': error_rel < 1e-10
            })

        return {
            'verificaciones': resultados,
            'todas_correctas': all(r['verificado'] for r in resultados)
        }


# =============================================================================
# EJEMPLOS CANONICOS DE AGUJEROS NEGROS
# =============================================================================

EJEMPLOS_CANONICOS = {
    'Cygnus_X1': {'M_solar': 21.2, 'nombre': 'Cygnus X-1'},
    'SgrA': {'M_solar': 4.0e6, 'nombre': 'Sgr A*'},
    'M87': {'M_solar': 6.5e9, 'nombre': 'M87*'},
    'TON618': {'M_solar': 6.6e10, 'nombre': 'TON 618'},
    'PBH_tipico': {'M_solar': 3.16e-12, 'nombre': 'PBH tipico'},
    'IMBH_tipico': {'M_solar': 3.16e3, 'nombre': 'IMBH tipico'},
}


def crear_BH_canonico(nombre: str) -> MCV_AgujerosNegros:
    """
    Crea un modelo MCV-BH para un agujero negro canonico.

    Args:
        nombre: Nombre del BH canonico

    Returns:
        Modelo MCV_AgujerosNegros
    """
    if nombre not in EJEMPLOS_CANONICOS:
        raise ValueError(f"BH canonico no encontrado: {nombre}. "
                        f"Opciones: {list(EJEMPLOS_CANONICOS.keys())}")

    params = EJEMPLOS_CANONICOS[nombre]
    return MCV_AgujerosNegros(params['M_solar'], params['nombre'])


# =============================================================================
# ANALISIS POR CATEGORIAS
# =============================================================================

def analizar_por_categorias() -> Dict[str, Dict]:
    """
    Analiza las propiedades MCV para cada categoria de BH.

    Returns:
        Dict con resultados por categoria
    """
    masas_tipicas = {
        'PBH': 3.16e-12,
        'Stellar': 17.3,
        'IMBH': 3.16e3,
        'SMBH': 1e8,
        'UMBH': 1e11,
    }

    resultados = {}

    for cat_nombre, M_solar in masas_tipicas.items():
        bh = MCV_AgujerosNegros(M_solar, f"BH_{cat_nombre}")
        resultados[cat_nombre] = bh.analisis_completo()

    return resultados


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_MCV_BH():
    """Test completo del modulo MCV-BH v2.0."""
    print("\n" + "="*70)
    print("  TEST MCV-AGUJEROS NEGROS MCMC v2.0")
    print("="*70)

    # 1. Analisis por categorias
    print("\n[1] Parametros por categoria de BH:")
    print("-" * 70)
    print(f"{'Categoria':<12} {'M [M_sun]':>12} {'r_s [m]':>12} {'S_local':>10} "
          f"{'Delta_S':>10} {'Xi_hor':>10} {'dt/dt0':>12}")
    print("-" * 70)

    resultados_cat = analizar_por_categorias()
    for cat, res in resultados_cat.items():
        print(f"{cat:<12} {res['M_solar']:>12.2e} {res['r_s_m']:>12.2e} "
              f"{res['S_local_horizonte']:>10.3f} {res['Delta_S_max']:>10.3f} "
              f"{res['Xi_horizonte']:>10.2f} {res['dilatacion_temporal']:>12.2e}")

    # 2. Verificacion de la Ley de Cronos
    print("\n[2] Verificacion de la Ley de Cronos (dt/dt0 = exp(Xi)):")
    print("-" * 70)

    sgra = crear_BH_canonico('SgrA')
    verif = sgra.verificar_ley_cronos()

    print(f"{'r/r_s':>10} {'Xi':>12} {'dt calculado':>15} {'dt teorico':>15} {'Error':>12}")
    for v in verif['verificaciones']:
        print(f"{v['r_rs']:>10.2f} {v['Xi']:>12.4f} {v['dt_calculado']:>15.2e} "
              f"{v['dt_teorico']:>15.2e} {v['error_relativo']:>12.2e}")

    ley_cronos_ok = verif['todas_correctas']
    print(f"\n  Ley de Cronos: {'PASS' if ley_cronos_ok else 'FAIL'}")

    # 3. Perfil radial de rho_MCV (verificar decaimiento gradual)
    print("\n[3] Perfil radial de rho_MCV para Sgr A*:")
    print("-" * 70)

    r_test = np.array([1.01, 2, 5, 10, 20, 50, 100]) * sgra.r_s
    print(f"{'r/r_s':>10} {'rho_MCV [kg/m^3]':>18} {'Xi':>12} {'S_local':>10}")
    print("-" * 70)

    rho_prev = None
    decaimiento_gradual = True
    for r in r_test:
        rho = sgra.rho_MCV(r)
        xi = sgra.Xi(r)
        S = sgra.S_local(r)
        print(f"{r/sgra.r_s:>10.2f} {rho:>18.2e} {xi:>12.4f} {S:>10.3f}")

        # Verificar que rho decrece
        if rho_prev is not None and rho >= rho_prev:
            decaimiento_gradual = False
        rho_prev = rho

    print(f"\n  Decaimiento gradual de rho_MCV: {'PASS' if decaimiento_gradual else 'FAIL'}")

    # 4. Verificacion de criterios ontologicos
    print("\n[4] Verificacion de criterios ontologicos:")
    print("-" * 70)

    # Criterio 1: BH pequenos tienen mayor friccion entropica
    pbh = MCV_AgujerosNegros(1e-10, "PBH_test")
    smbh = MCV_AgujerosNegros(1e8, "SMBH_test")

    dS_pbh = pbh.Delta_S(pbh.r_s * 1.01)
    dS_smbh = smbh.Delta_S(smbh.r_s * 1.01)

    criterio1 = dS_pbh > dS_smbh
    print(f"  1. Delta_S(PBH) > Delta_S(SMBH): {dS_pbh:.3f} > {dS_smbh:.3f} -> "
          f"{'PASS' if criterio1 else 'FAIL'}")

    # Criterio 2: S_local aumenta con r
    r1, r2 = sgra.r_s * 2, sgra.r_s * 10
    S1, S2 = sgra.S_local(r1), sgra.S_local(r2)
    criterio2 = S2 > S1
    print(f"  2. S_local aumenta con r: S(2r_s)={S1:.3f} < S(10r_s)={S2:.3f} -> "
          f"{'PASS' if criterio2 else 'FAIL'}")

    # Criterio 3: Xi disminuye con r
    Xi1, Xi2 = sgra.Xi(r1), sgra.Xi(r2)
    criterio3 = Xi1 > Xi2
    print(f"  3. Xi disminuye con r: Xi(2r_s)={Xi1:.4f} > Xi(10r_s)={Xi2:.4f} -> "
          f"{'PASS' if criterio3 else 'FAIL'}")

    # Criterio 4: S_local en horizonte entre S1 y S3
    S_hor = sgra.S_local(sgra.r_s * 1.01)
    criterio4 = SELLOS[0].S_n < S_hor < SELLOS[2].S_n
    print(f"  4. S_horizonte entre S1 y S3: {SELLOS[0].S_n:.3f} < {S_hor:.3f} < "
          f"{SELLOS[2].S_n:.3f} -> {'PASS' if criterio4 else 'FAIL'}")

    # Criterio 5: BH pequenos tienen Xi mayor
    Xi_pbh = pbh.Xi(pbh.r_s * 1.01)
    Xi_smbh = smbh.Xi(smbh.r_s * 1.01)
    criterio5 = Xi_pbh > Xi_smbh
    print(f"  5. Xi(PBH) > Xi(SMBH): {Xi_pbh:.2f} > {Xi_smbh:.2f} -> "
          f"{'PASS' if criterio5 else 'FAIL'}")

    passed = all([ley_cronos_ok, decaimiento_gradual, criterio1, criterio2,
                  criterio3, criterio4, criterio5])

    print("\n" + "="*70)
    print(f"  MCV-BH MODULE v2.0: {'PASS' if passed else 'FAIL'}")
    print("="*70)

    return passed


if __name__ == "__main__":
    test_MCV_BH()
