#!/usr/bin/env python3
"""
================================================================================
CICLO COSMICO MCMC: S_max -> S_0
================================================================================

Implementa el ciclo cosmico del modelo MCMC, describiendo la evolucion
de la entropia desde S_max (estado primordial) hasta S_0 (presente).

FUNDAMENTACION ONTOLOGICA:
--------------------------
El MCMC propone un ciclo cosmico donde:
1. S_max: Estado de maxima entropia (colapso previo o estado primordial)
2. Big Bang: Transicion S4 -> universo observable
3. Expansion: S decrece durante expansion cosmica
4. S_0: Estado actual (~90 en unidades MCMC)
5. Posible reciclaje: reconversion de energia en futuro

ECUACIONES DEL CICLO:
---------------------
(Ec. C1) dS/dt = -alpha_S * H(t) * (S - S_min)
(Ec. C2) S(t) = S_min + (S_max - S_min) * exp(-alpha_S * integral H dt)
(Ec. C3) t_ciclo ~ 1/H_0 * ln(S_max/S_min) / alpha_S

CANALES DE RECONVERSION:
------------------------
1. ECV -> MCV: Energia cuantica virtual se convierte en materia
2. Radiacion Hawking: BHs emiten energia al oceano geometrico
3. Decaimiento de vacio: Posible transicion de fase de vacio

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.integrate import odeint, quad
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES DEL CICLO COSMICO
# =============================================================================

# Entropia en unidades MCMC
S_MAX = 1000.0          # Entropia maxima (estado primordial)
S_MIN = 0.009           # Entropia minima (sello S1, Planck)
S_0 = 90.0              # Entropia actual (z=0)
S_BB = 1.001            # Entropia en el Big Bang (sello S4)

# Parametros de evolucion
ALPHA_S = 0.5           # Tasa de decaimiento de entropia
T_PLANCK = 5.391e-44    # Tiempo de Planck [s]
T_UNIVERSO = 13.8e9 * 3.156e7  # Edad del universo [s]

# Cosmologia
H0 = 67.36              # km/s/Mpc
OMEGA_M = 0.3153
OMEGA_LAMBDA = 0.6847


# =============================================================================
# CANALES DE RECONVERSION
# =============================================================================

class CanalReconversion(Enum):
    """Canales de reconversion de energia/entropia."""
    ECV_MCV = "ECV -> MCV"
    HAWKING = "Radiacion Hawking"
    DECAIMIENTO_VACIO = "Decaimiento de vacio"
    ANIQUILACION = "Aniquilacion materia-antimateria"
    FUSION_NUCLEAR = "Fusion nuclear estelar"


@dataclass
class PropiedadesCanal:
    """Propiedades de un canal de reconversion."""
    nombre: str
    eficiencia: float       # Fraccion de energia convertida
    escala_tiempo: float    # Tiempo caracteristico [Gyr]
    activo_hoy: bool        # Si esta activo en el presente
    contribucion_ciclo: float  # Contribucion al ciclo total [%]


CANALES = {
    CanalReconversion.ECV_MCV: PropiedadesCanal(
        "ECV -> MCV", 0.01, 0.1, True, 40.0
    ),
    CanalReconversion.HAWKING: PropiedadesCanal(
        "Radiacion Hawking", 1e-10, 1e67, False, 0.001
    ),
    CanalReconversion.DECAIMIENTO_VACIO: PropiedadesCanal(
        "Decaimiento vacio", 0.1, 1e10, False, 30.0
    ),
    CanalReconversion.ANIQUILACION: PropiedadesCanal(
        "Aniquilacion", 1.0, 1e-6, False, 20.0
    ),
    CanalReconversion.FUSION_NUCLEAR: PropiedadesCanal(
        "Fusion nuclear", 0.007, 10.0, True, 10.0
    ),
}


# =============================================================================
# EVOLUCION TEMPORAL DE S
# =============================================================================

class EvolucionEntropia:
    """
    Modela la evolucion temporal de la entropia S(t).

    La entropia del universo evoluciona segun:
    dS/dt = -alpha_S * H(t) * (S - S_min)

    Esto produce un decaimiento exponencial modulado por H(t).
    """

    def __init__(self, S_max: float = S_MAX, S_min: float = S_MIN,
                 alpha: float = ALPHA_S):
        """
        Inicializa el modelo de evolucion.

        Args:
            S_max: Entropia inicial (maxima)
            S_min: Entropia final (minima asintoticamente)
            alpha: Tasa de decaimiento
        """
        self.S_max = S_max
        self.S_min = S_min
        self.alpha = alpha

    def H(self, t_Gyr: float) -> float:
        """
        Parametro de Hubble H(t) en funcion del tiempo cosmico.

        Aproximacion simplificada.

        Args:
            t_Gyr: Tiempo desde Big Bang [Gyr]

        Returns:
            H en km/s/Mpc
        """
        if t_Gyr <= 0:
            return 1e10  # Muy grande cerca del Big Bang

        # Aproximacion: H ~ H0 * sqrt(Omega_m * (t_0/t)^2 + Omega_Lambda)
        t_0 = 13.8  # Gyr
        ratio = t_0 / t_Gyr

        if t_Gyr < 0.01:  # Era dominada por radiacion
            return H0 * ratio**2

        # Era de materia/Lambda
        return H0 * np.sqrt(OMEGA_M * ratio**3 + OMEGA_LAMBDA)

    def dS_dt(self, S: float, t_Gyr: float) -> float:
        """
        Tasa de cambio de entropia.

        dS/dt = -alpha * H(t) * (S - S_min)

        Args:
            S: Entropia actual
            t_Gyr: Tiempo [Gyr]

        Returns:
            dS/dt
        """
        H_t = self.H(t_Gyr)
        # Normalizar H a H0
        H_norm = H_t / H0
        return -self.alpha * H_norm * (S - self.S_min)

    def S(self, t_Gyr: float) -> float:
        """
        Entropia a tiempo t.

        Solucion aproximada:
        S(t) = S_min + (S_max - S_min) * exp(-alpha * integral H dt / H0)

        Args:
            t_Gyr: Tiempo desde Big Bang [Gyr]

        Returns:
            S(t)
        """
        if t_Gyr <= 0:
            return self.S_max

        # Integral aproximada de H/H0
        # Para simplificar, usamos t^p con p ajustado
        p = 0.5  # Exponente efectivo
        integral_H = (t_Gyr / 13.8)**p * self.alpha

        S = self.S_min + (self.S_max - self.S_min) * np.exp(-integral_H * 10)

        return max(S, self.S_min)

    def z_from_t(self, t_Gyr: float) -> float:
        """Convierte tiempo a redshift (aproximacion)."""
        if t_Gyr <= 0:
            return np.inf
        if t_Gyr >= 13.8:
            return 0.0
        return (13.8 / t_Gyr)**(2/3) - 1

    def t_from_z(self, z: float) -> float:
        """Convierte redshift a tiempo (aproximacion)."""
        if z < 0:
            return 13.8
        return 13.8 / (1 + z)**(3/2)

    def S_of_z(self, z: float) -> float:
        """
        Entropia en funcion del redshift.

        Args:
            z: Redshift

        Returns:
            S(z)
        """
        t = self.t_from_z(z)
        return self.S(t)

    def perfil_temporal(self, t_array: np.ndarray = None) -> Dict:
        """
        Genera perfil S(t) completo.

        Returns:
            Dict con t, S, z, H
        """
        if t_array is None:
            t_array = np.logspace(-6, np.log10(13.8), 100)

        S_array = np.array([self.S(t) for t in t_array])
        z_array = np.array([self.z_from_t(t) for t in t_array])
        H_array = np.array([self.H(t) for t in t_array])

        return {
            't_Gyr': t_array,
            'S': S_array,
            'z': z_array,
            'H': H_array,
        }


# =============================================================================
# CICLO COSMICO COMPLETO
# =============================================================================

@dataclass
class FaseCiclo:
    """Una fase del ciclo cosmico."""
    nombre: str
    t_inicio_Gyr: float
    t_fin_Gyr: float
    S_inicio: float
    S_fin: float
    descripcion: str


class CicloCosmico:
    """
    Modelo completo del ciclo cosmico MCMC.

    El ciclo tiene las siguientes fases:
    1. Estado primordial (S ~ S_max)
    2. Transicion al Big Bang (S4)
    3. Expansion y enfriamiento (S decrece)
    4. Era actual (S ~ S_0)
    5. Futuro: posible reciclaje
    """

    def __init__(self):
        """Inicializa el modelo de ciclo cosmico."""
        self.evolucion = EvolucionEntropia()
        self.fases = self._definir_fases()
        self.canales = CANALES

    def _definir_fases(self) -> List[FaseCiclo]:
        """Define las fases del ciclo cosmico."""
        return [
            FaseCiclo(
                "Primordial",
                -1e10, 0,
                S_MAX, S_BB,
                "Estado de maxima entropia antes del Big Bang"
            ),
            FaseCiclo(
                "Big Bang",
                0, 1e-6,
                S_BB, S_BB * 0.999,
                "Transicion S4: emergencia del espacio-tiempo"
            ),
            FaseCiclo(
                "Inflacion",
                1e-6, 1e-3,
                S_BB * 0.999, S_BB * 0.99,
                "Expansion exponencial (si aplica)"
            ),
            FaseCiclo(
                "Radiacion",
                1e-3, 0.05,
                S_BB * 0.99, 150.0,
                "Dominio de radiacion, formacion de nucleos"
            ),
            FaseCiclo(
                "Materia",
                0.05, 9.0,
                150.0, 95.0,
                "Dominio de materia, formacion de estructura"
            ),
            FaseCiclo(
                "Aceleracion",
                9.0, 13.8,
                95.0, S_0,
                "Dominio de ECV, expansion acelerada"
            ),
            FaseCiclo(
                "Futuro",
                13.8, 1e100,
                S_0, S_MIN,
                "Evolucion futura, posible reciclaje"
            ),
        ]

    def fase_actual(self, t_Gyr: float = 13.8) -> FaseCiclo:
        """
        Determina la fase del ciclo a un tiempo dado.

        Args:
            t_Gyr: Tiempo [Gyr]

        Returns:
            Fase correspondiente
        """
        for fase in self.fases:
            if fase.t_inicio_Gyr <= t_Gyr < fase.t_fin_Gyr:
                return fase
        return self.fases[-1]

    def entropia(self, t_Gyr: float) -> float:
        """
        Entropia a tiempo t.

        Args:
            t_Gyr: Tiempo [Gyr]

        Returns:
            S(t)
        """
        return self.evolucion.S(t_Gyr)

    def tasa_reconversion(self, canal: CanalReconversion,
                          t_Gyr: float = 13.8) -> float:
        """
        Tasa de reconversion de un canal dado.

        Args:
            canal: Canal de reconversion
            t_Gyr: Tiempo [Gyr]

        Returns:
            Tasa en unidades de S/Gyr
        """
        props = self.canales[canal]

        if not props.activo_hoy and t_Gyr > 1.0:
            return 0.0

        # Tasa base
        tasa_base = props.eficiencia / props.escala_tiempo

        # Modular por entropia disponible
        S_actual = self.entropia(t_Gyr)
        factor_S = S_actual / S_0

        return tasa_base * factor_S

    def duracion_ciclo_estimada(self) -> float:
        """
        Estima la duracion total del ciclo cosmico.

        El ciclo completo incluye:
        1. Expansion: S_max -> S_min (dominado por expansion cosmica)
        2. Reconversion: S_min -> S_max (dominado por proceso mas lento)

        La reconversion esta dominada por la evaporacion de agujeros negros
        via radiacion Hawking, con t ~ 10^67 Gyr para BHs estelares.

        Returns:
            Duracion en Gyr
        """
        # Fase de expansion: t_exp ~ (1/alpha) * ln(S_max/S_min) * t_Hubble
        t_Hubble = 14.4  # Gyr (1/H0)
        t_expansion = (1 / self.evolucion.alpha) * np.log(S_MAX / S_MIN) * t_Hubble

        # Fase de reconversion: dominada por el proceso mas lento
        # En este caso, Hawking radiation con t ~ 10^67 Gyr
        t_reconversion = max(props.escala_tiempo for props in self.canales.values())

        # Ciclo total dominado por reconversion
        return t_expansion + t_reconversion

    def energia_total_ciclo(self) -> Dict:
        """
        Estima la energia involucrada en el ciclo cosmico.

        Returns:
            Dict con energias por fase
        """
        # Energia de Planck
        E_Planck = 1.956e9  # J

        # Energia del universo observable ~ 10^70 J
        E_universo = 1e70  # J

        return {
            'E_Planck_J': E_Planck,
            'E_universo_J': E_universo,
            'E_universo_Planck': E_universo / E_Planck,
            'E_por_fase': {
                'Primordial': E_universo * 0.1,
                'Big Bang': E_universo * 0.3,
                'Inflacion': E_universo * 0.2,
                'Radiacion': E_universo * 0.2,
                'Materia': E_universo * 0.15,
                'Aceleracion': E_universo * 0.05,
            }
        }

    def resumen(self) -> str:
        """Genera resumen del ciclo cosmico."""
        lines = [
            "=" * 60,
            "  CICLO COSMICO MCMC",
            "=" * 60,
            "",
            f"  S_max (primordial) = {S_MAX:.1f}",
            f"  S_0 (actual)       = {S_0:.1f}",
            f"  S_min (asintótico) = {S_MIN:.4f}",
            "",
            "  FASES:",
            "-" * 60,
        ]

        for fase in self.fases:
            lines.append(f"  {fase.nombre:15s}: S = {fase.S_inicio:.1f} -> {fase.S_fin:.1f}")

        lines.extend([
            "",
            f"  Duración estimada: {self.duracion_ciclo_estimada():.1e} Gyr",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# CONSTANTES DEL CICLO
# =============================================================================

@dataclass
class ConstantesCiclo:
    """Constantes fundamentales del ciclo cosmico."""
    S_max: float = S_MAX
    S_min: float = S_MIN
    S_0: float = S_0
    S_BB: float = S_BB
    alpha_S: float = ALPHA_S
    t_universo_Gyr: float = 13.8
    t_Planck_s: float = T_PLANCK

    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return {
            'S_max': self.S_max,
            'S_min': self.S_min,
            'S_0': self.S_0,
            'S_BB': self.S_BB,
            'alpha_S': self.alpha_S,
            't_universo_Gyr': self.t_universo_Gyr,
            't_Planck_s': self.t_Planck_s,
        }


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_Cosmic_Cycle_MCMC() -> bool:
    """
    Test del modulo de ciclo cosmico.
    """
    print("\n" + "=" * 70)
    print("  TEST COSMIC CYCLE MCMC - CICLO COSMICO S_max -> S_0")
    print("=" * 70)

    # 1. Constantes del ciclo
    print("\n[1] Constantes del ciclo cosmico:")
    print("-" * 70)

    constantes = ConstantesCiclo()
    cdict = constantes.to_dict()

    print(f"    S_max = {cdict['S_max']:.1f}")
    print(f"    S_min = {cdict['S_min']:.4f}")
    print(f"    S_0   = {cdict['S_0']:.1f}")
    print(f"    S_BB  = {cdict['S_BB']:.4f}")
    print(f"    alpha = {cdict['alpha_S']:.2f}")

    # Verificar jerarquia
    const_ok = (cdict['S_max'] > cdict['S_0'] > cdict['S_min'])
    print(f"\n    Jerarquia S_max > S_0 > S_min: {'PASS' if const_ok else 'FAIL'}")

    # 2. Canales de reconversion
    print("\n[2] Canales de reconversion:")
    print("-" * 70)

    print(f"    {'Canal':25s} {'Eficiencia':>12} {'t_caract':>12} {'Activo':>8}")
    for canal, props in CANALES.items():
        print(f"    {props.nombre:25s} {props.eficiencia:>12.2e} "
              f"{props.escala_tiempo:>12.2e} {'Si' if props.activo_hoy else 'No':>8}")

    # Verificar que hay canales definidos
    canales_ok = len(CANALES) >= 3
    print(f"\n    Canales definidos: {'PASS' if canales_ok else 'FAIL'}")

    # 3. Evolucion de S(t)
    print("\n[3] Evolucion S(t):")
    print("-" * 70)

    ciclo = CicloCosmico()
    perfil = ciclo.evolucion.perfil_temporal()

    print(f"    {'t [Gyr]':>12} {'S':>10} {'z':>10}")
    indices = [0, 10, 30, 50, 70, 99]
    for i in indices:
        print(f"    {perfil['t_Gyr'][i]:>12.6f} {perfil['S'][i]:>10.2f} "
              f"{perfil['z'][i]:>10.2f}")

    # Verificar que S decrece con t
    S_decrece = perfil['S'][0] > perfil['S'][-1]
    print(f"\n    S decrece con t: {'PASS' if S_decrece else 'FAIL'}")

    # 4. Fases del ciclo
    print("\n[4] Fases del ciclo cosmico:")
    print("-" * 70)

    for fase in ciclo.fases[:5]:  # Primeras 5 fases
        print(f"    {fase.nombre:15s}: {fase.t_inicio_Gyr:.2e} - {fase.t_fin_Gyr:.2e} Gyr")

    # 5. Duracion del ciclo
    print("\n[5] Duracion estimada del ciclo:")
    print("-" * 70)

    duracion = ciclo.duracion_ciclo_estimada()
    print(f"    t_ciclo ~ {duracion:.2e} Gyr")

    duracion_ok = duracion > 1e10  # Mas de 10^10 Gyr
    print(f"    Duracion >> t_universo: {'PASS' if duracion_ok else 'FAIL'}")

    # Resultado final
    passed = const_ok and canales_ok and S_decrece and duracion_ok

    print("\n" + "=" * 70)
    print(f"  COSMIC CYCLE MCMC: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    test_Cosmic_Cycle_MCMC()
