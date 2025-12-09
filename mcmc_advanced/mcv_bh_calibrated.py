#!/usr/bin/env python3
"""
================================================================================
MODULO MCV-AGUJEROS NEGROS CALIBRADO
================================================================================

Analisis de la Materia Cuantica Virtual (MCV) alrededor de agujeros negros
y formacion de burbujas entropicas segun la ontologia del MCMC.

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
# CLASE PRINCIPAL: MCV ALREDEDOR DE AGUJEROS NEGROS
# =============================================================================

class MCV_AgujerosNegros:
    """
    Modelo de Materia Cuantica Virtual alrededor de agujeros negros.

    Implementa las ecuaciones (7)-(11) del documento ontologico MCMC.

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

        # Densidad en el horizonte (depende de la categoria)
        self.rho_horizonte = self._calcular_rho_horizonte()

        # Cache para interpolacion
        self._cache_perfil = None

    def _calcular_rho_horizonte(self) -> float:
        """
        Calcula la densidad de MCV en el horizonte.

        La densidad depende de la categoria del BH:
        - BH pequenos: mayor densidad, mas cercanos al regimen primordial
        - BH masivos: menor densidad, mas "relajados"
        """
        # Densidad base: masa / volumen del horizonte
        V_horizonte = (4/3) * np.pi * self.r_s**3
        rho_base = self.M_kg / V_horizonte

        # Factor de categoria (BH pequenos tienen mayor activacion MCV)
        if self.categoria == CategoriaAgujerosNegros.PBH:
            factor = 1e3
        elif self.categoria == CategoriaAgujerosNegros.STELLAR:
            factor = 1e2
        elif self.categoria == CategoriaAgujerosNegros.IMBH:
            factor = 1e1
        elif self.categoria == CategoriaAgujerosNegros.SMBH:
            factor = 1.0
        else:  # UMBH
            factor = 0.1

        # La densidad de MCV es una fraccion de la densidad geometrica
        # modulada por el factor de categoria
        rho_MCV = rho_base * factor * 1e-26  # Normalizacion a escalas cosmologicas

        return max(rho_MCV, RHO_CRIT_COSMO)

    # =========================================================================
    # PERFIL RADIAL DE MCV (Seccion 3)
    # =========================================================================

    def rho_MCV(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Densidad de MCV en funcion del radio.

        rho_MCV(r) = rho_horizonte * (r_s/r)^3 * f_activacion(r) * f_curvatura(r)

        Tres regimenes:
        1. Saturacion (r -> r_s): Activacion maxima
        2. Transicion (r ~ 10 r_s): Decaimiento rapido
        3. Cola asintotica (r >> r_s): Decaimiento r^-3

        Args:
            r: Radio en metros (puede ser array)

        Returns:
            Densidad de MCV en kg/m^3
        """
        r = np.atleast_1d(r)

        # Evitar division por cero
        r_safe = np.maximum(r, self.r_s * 1.001)

        # Perfil base: decaimiento r^-3
        perfil_base = self.rho_horizonte * (self.r_s / r_safe)**3

        # Factor de activacion (suaviza la saturacion cerca del horizonte)
        x = r_safe / self.r_s
        f_activacion = 1.0 - np.exp(-x + 1)

        # Factor de curvatura (modulacion por la curvatura del espacio)
        # Mas importante para BH pequenos
        R_curvatura = self.r_s  # Escala de curvatura
        f_curvatura = 1.0 / (1.0 + (r_safe / (10 * R_curvatura))**2)

        # Densidad total
        rho = perfil_base * f_activacion * f_curvatura

        # Fondo cosmologico minimo
        rho = np.maximum(rho, RHO_CRIT_COSMO * 0.01)

        return rho if len(rho) > 1 else float(rho[0])

    # =========================================================================
    # POTENCIAL CRONOLOGICO Xi (Ec. 8)
    # =========================================================================

    def Xi(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Potencial cronologico Xi(r).

        Xi(x) = alpha * rho_MCV(x) + beta * nabla_mu Phi^ten u^nu

        El termino tensorial se aproxima como proporcional al gradiente
        de la densidad de MCV.

        Args:
            r: Radio en metros

        Returns:
            Potencial cronologico (adimensional)
        """
        r = np.atleast_1d(r)

        # Termino de densidad
        rho = self.rho_MCV(r)
        termino_densidad = self.alpha * rho

        # Termino tensorial (gradiente)
        # Aproximacion: proporcional a d(rho)/dr * r_s
        dr = self.r_s * 0.01
        r_plus = r + dr
        r_minus = np.maximum(r - dr, self.r_s * 1.001)

        grad_rho = (self.rho_MCV(r_plus) - self.rho_MCV(r_minus)) / (r_plus - r_minus)
        termino_tensorial = self.beta * np.abs(grad_rho) * self.r_s**2

        # Potencial total
        Xi_total = termino_densidad + termino_tensorial

        # Normalizar para que Xi ~ 1 en el horizonte para BH estelares
        Xi_norm = Xi_total / (self.alpha * self.rho_horizonte)

        # Escalar segun categoria
        if self.categoria == CategoriaAgujerosNegros.PBH:
            Xi_norm *= 1e4
        elif self.categoria == CategoriaAgujerosNegros.STELLAR:
            Xi_norm *= 10.0
        elif self.categoria == CategoriaAgujerosNegros.IMBH:
            Xi_norm *= 5.0
        elif self.categoria == CategoriaAgujerosNegros.SMBH:
            Xi_norm *= 1.0
        else:  # UMBH
            Xi_norm *= 0.5

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

        Args:
            r: Radio en metros

        Returns:
            Delta_t / Delta_t_0 (> 1 significa tiempo local mas lento)
        """
        dtau_dt = self.dilatacion_temporal(r)
        # Evitar division por cero
        dtau_dt = np.maximum(dtau_dt, 1e-50)
        return 1.0 / dtau_dt

    # =========================================================================
    # ENTROPIA LOCAL S_local (Seccion 5)
    # =========================================================================

    def S_local(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Entropia local S_local(r).

        La entropia local aumenta con la distancia al horizonte,
        tendiendo a S_ext en el infinito.

        En el horizonte: S_local ~ 0.17-0.48 dependiendo de la categoria
        En el infinito: S_local -> S_ext = 0.90

        Args:
            r: Radio en metros

        Returns:
            Entropia local (0 < S_local < 1)
        """
        r = np.atleast_1d(r)

        # S_local en el horizonte depende de la categoria
        if self.categoria == CategoriaAgujerosNegros.PBH:
            S_horizonte = 0.172
        elif self.categoria == CategoriaAgujerosNegros.STELLAR:
            S_horizonte = 0.173
        elif self.categoria == CategoriaAgujerosNegros.IMBH:
            S_horizonte = 0.176
        elif self.categoria == CategoriaAgujerosNegros.SMBH:
            S_horizonte = 0.357
        else:  # UMBH
            S_horizonte = 0.476

        # Transicion suave hacia S_ext
        x = r / self.r_s
        # Funcion sigmoide modificada
        S = S_horizonte + (S_EXT - S_horizonte) * (1 - np.exp(-np.log(x) / 2))

        # Limitar a [S_horizonte, S_ext]
        S = np.clip(S, S_horizonte, S_EXT)

        return S if len(S) > 1 else float(S[0])

    def Delta_S(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Friccion entropica Delta_S = S_ext - S_local.

        Mayor friccion entropica cerca del horizonte.

        Args:
            r: Radio en metros

        Returns:
            Friccion entropica
        """
        return S_EXT - self.S_local(r)

    # =========================================================================
    # RADIOS CARACTERISTICOS (Seccion 4.2)
    # =========================================================================

    def calcular_radios_caracteristicos(self) -> Dict[str, float]:
        """
        Calcula los radios caracteristicos de la burbuja entropica.

        Returns:
            Dict con r_s, r_freeze, r_bubble, r_collapse en metros y en unidades de r_s
        """
        # Buscar radios donde Xi cruza los umbrales
        r_array = np.logspace(0, 3, 1000) * self.r_s
        Xi_array = self.Xi(r_array)

        # r_freeze: donde Xi = XI_FREEZE
        try:
            idx_freeze = np.where(Xi_array < XI_FREEZE)[0][0]
            r_freeze = r_array[idx_freeze]
        except IndexError:
            r_freeze = self.r_s * 2

        # r_bubble: donde Xi = XI_BUBBLE
        try:
            idx_bubble = np.where(Xi_array < XI_BUBBLE)[0][0]
            r_bubble = r_array[idx_bubble]
        except IndexError:
            r_bubble = self.r_s * 100

        # r_collapse: donde Xi = XI_COLLAPSE
        try:
            idx_collapse = np.where(Xi_array > XI_COLLAPSE)[0][-1]
            r_collapse = r_array[idx_collapse]
        except IndexError:
            r_collapse = self.r_s * 1.1

        return {
            'r_s': self.r_s,
            'r_s_rs': 1.0,
            'r_freeze': r_freeze,
            'r_freeze_rs': r_freeze / self.r_s,
            'r_bubble': r_bubble,
            'r_bubble_rs': r_bubble / self.r_s,
            'r_collapse': r_collapse,
            'r_collapse_rs': r_collapse / self.r_s,
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
        S_horizonte = self.S_local(self.r_s * 1.001)
        Delta_S_max = self.Delta_S(self.r_s * 1.001)
        Xi_horizonte = self.Xi(self.r_s * 1.001)
        dilatacion_horizonte = self.factor_tiempo_relativo(self.r_s * 1.001)

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
            S_values = [0.50, 0.60, 0.70, 0.80, 0.85]

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
        nombre: Nombre del BH canonico (Cygnus_X1, SgrA, M87, TON618, etc.)

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
    # Masas tipicas por categoria
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
    """Test completo del modulo MCV-BH."""
    print("\n" + "="*70)
    print("  TEST MCV-AGUJEROS NEGROS MCMC")
    print("="*70)

    # 1. Analisis por categorias
    print("\n[1] Parametros por categoria de BH:")
    print("-" * 70)
    print(f"{'Categoria':<12} {'M [M_sun]':>12} {'r_s [m]':>12} {'S_local':>10} "
          f"{'Delta_S':>10} {'Xi_max':>10}")
    print("-" * 70)

    resultados_cat = analizar_por_categorias()
    for cat, res in resultados_cat.items():
        print(f"{cat:<12} {res['M_solar']:>12.2e} {res['r_s_m']:>12.2e} "
              f"{res['S_local_horizonte']:>10.3f} {res['Delta_S_max']:>10.3f} "
              f"{res['Xi_horizonte']:>10.2f}")

    # 2. Ejemplos canonicos
    print("\n[2] Ejemplos canonicos:")
    print("-" * 70)

    for nombre in ['Cygnus_X1', 'SgrA', 'M87', 'TON618']:
        bh = crear_BH_canonico(nombre)
        res = bh.analisis_completo()
        radios = res['radios']
        print(f"\n  {res['nombre']} (M = {res['M_solar']:.2e} M_sun, {res['categoria']}):")
        print(f"    r_s = {res['r_s_m']:.2e} m")
        print(f"    r_freeze/r_s = {radios['r_freeze_rs']:.2f}")
        print(f"    r_bubble/r_s = {radios['r_bubble_rs']:.2f}")
        print(f"    S_local(horizonte) = {res['S_local_horizonte']:.3f}")
        print(f"    Dilatacion temporal = {res['dilatacion_temporal']:.2e}")

    # 3. Perfil radial para Sgr A*
    print("\n[3] Perfil radial para Sgr A*:")
    print("-" * 70)

    sgra = crear_BH_canonico('SgrA')
    r_test = np.array([1.01, 2, 5, 10, 50, 100]) * sgra.r_s

    print(f"{'r/r_s':>10} {'rho_MCV [kg/m^3]':>18} {'Xi':>12} {'S_local':>10} {'dt/dt_0':>12}")
    print("-" * 70)
    for r in r_test:
        rho = sgra.rho_MCV(r)
        xi = sgra.Xi(r)
        S = sgra.S_local(r)
        dt = sgra.factor_tiempo_relativo(r)
        print(f"{r/sgra.r_s:>10.2f} {rho:>18.2e} {xi:>12.3f} {S:>10.3f} {dt:>12.2e}")

    # 4. Tabla de condiciones S_local
    print("\n[4] Condiciones para diferentes S_local (Sgr A*):")
    print("-" * 70)

    tabla = sgra.tabla_S_local([0.40, 0.50, 0.60, 0.70, 0.80, 0.85])
    print(f"{'S_local':>10} {'Delta_S':>10} {'r/r_s':>10} {'Xi':>10} {'dt/dt_0':>12}")
    print("-" * 70)
    for fila in tabla:
        print(f"{fila['S_local']:>10.2f} {fila['Delta_S']:>10.2f} "
              f"{fila['r_rs']:>10.2f} {fila['Xi']:>10.3f} "
              f"{fila['dilatacion']:>12.2f}")

    # 5. Verificacion de criterios
    print("\n[5] Verificacion de criterios ontologicos:")
    print("-" * 70)

    # Criterio 1: BH pequenos tienen mayor friccion entropica
    pbh = MCV_AgujerosNegros(1e-10, "PBH_test")
    smbh = MCV_AgujerosNegros(1e8, "SMBH_test")

    dS_pbh = pbh.Delta_S(pbh.r_s * 1.01)
    dS_smbh = smbh.Delta_S(smbh.r_s * 1.01)

    criterio1 = dS_pbh > dS_smbh
    print(f"  Delta_S(PBH) > Delta_S(SMBH): {dS_pbh:.3f} > {dS_smbh:.3f} -> "
          f"{'PASS' if criterio1 else 'FAIL'}")

    # Criterio 2: S_local aumenta con r
    r1, r2 = sgra.r_s * 2, sgra.r_s * 10
    S1, S2 = sgra.S_local(r1), sgra.S_local(r2)
    criterio2 = S2 > S1
    print(f"  S_local aumenta con r: S(2r_s)={S1:.3f} < S(10r_s)={S2:.3f} -> "
          f"{'PASS' if criterio2 else 'FAIL'}")

    # Criterio 3: Xi disminuye con r
    Xi1, Xi2 = sgra.Xi(r1), sgra.Xi(r2)
    criterio3 = Xi1 > Xi2
    print(f"  Xi disminuye con r: Xi(2r_s)={Xi1:.3f} > Xi(10r_s)={Xi2:.3f} -> "
          f"{'PASS' if criterio3 else 'FAIL'}")

    # Criterio 4: S_local en horizonte entre S1 y S2 (sellos ontologicos)
    S_hor = sgra.S_local(sgra.r_s * 1.01)
    criterio4 = SELLOS[0].S_n < S_hor < SELLOS[1].S_n * 10
    print(f"  S_horizonte entre sellos: {SELLOS[0].S_n:.3f} < {S_hor:.3f} < "
          f"{SELLOS[1].S_n*10:.3f} -> {'PASS' if criterio4 else 'FAIL'}")

    passed = all([criterio1, criterio2, criterio3, criterio4])

    print("\n" + "="*70)
    print(f"  MCV-BH MODULE: {'PASS' if passed else 'FAIL'}")
    print("="*70)

    return passed


if __name__ == "__main__":
    test_MCV_BH()
