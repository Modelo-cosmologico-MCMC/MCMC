"""
Bloque 1 - Pregeometría
========================

Define la estructura pregeométrica del modelo MCMC.

La pregeometría describe cómo el espacio-tiempo emerge de un sustrato
más fundamental. En MCMC, esto se caracteriza por:

    - Tasa de colapso k(S) = k0 × [1 + a1·sin(2πS) + a2·sin(4πS) + a3·sin(6πS)]
    - Parámetros: k0=6.307, a1=0.15, a2=0.25, a3=0.35
    - Integral entrópica: ε(S) = ∫k(s)ds / ∫k(s)ds|_{0→S4}
    - Evolución de masa/energía: Mp(S) = Mp0×(1-ε), Ep(S) = Mp0×ε

La función k(S) con sus armónicos sinusoidales captura las oscilaciones
en la tasa de conversión masa→energía durante los colapsos.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Referencias:
    Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from numpy.typing import NDArray
from scipy.integrate import quad

from .bloque0_estado_primordial import (
    SELLOS, SELLOS_ORDEN, Mp0, Ep0,
    calcular_P_ME, EstadoPrimordial
)


# =============================================================================
# Constantes de la Tasa de Colapso
# =============================================================================

# Tasa de colapso base
K0: float = 6.307

# Coeficientes de los armónicos
A1: float = 0.15  # Primer armónico (2πS)
A2: float = 0.25  # Segundo armónico (4πS)
A3: float = 0.35  # Tercer armónico (6πS)

# Sello final para normalización
S4: float = SELLOS["S4"]


# =============================================================================
# Funciones de Tasa de Colapso
# =============================================================================

def tasa_colapso_k(
    S: float,
    k0: float = K0,
    a1: float = A1,
    a2: float = A2,
    a3: float = A3
) -> float:
    """
    Calcula la tasa de colapso k(S).

    k(S) = k0 × [1 + a1·sin(2πS) + a2·sin(4πS) + a3·sin(6πS)]

    Esta función describe cómo la tasa de conversión masa→energía
    varía con el sello entrópico S. Los armónicos sinusoidales
    capturan las oscilaciones naturales del proceso de colapso.

    Args:
        S: Sello entrópico (valor continuo)
        k0: Tasa base
        a1, a2, a3: Coeficientes armónicos

    Returns:
        k(S): Tasa de colapso en el sello S
    """
    return k0 * (
        1.0
        + a1 * np.sin(2 * np.pi * S)
        + a2 * np.sin(4 * np.pi * S)
        + a3 * np.sin(6 * np.pi * S)
    )


def derivada_k(
    S: float,
    k0: float = K0,
    a1: float = A1,
    a2: float = A2,
    a3: float = A3
) -> float:
    """
    Derivada de la tasa de colapso dk/dS.

    dk/dS = k0 × [2πa1·cos(2πS) + 4πa2·cos(4πS) + 6πa3·cos(6πS)]

    Args:
        S: Sello entrópico

    Returns:
        dk/dS
    """
    return k0 * (
        2 * np.pi * a1 * np.cos(2 * np.pi * S)
        + 4 * np.pi * a2 * np.cos(4 * np.pi * S)
        + 6 * np.pi * a3 * np.cos(6 * np.pi * S)
    )


# =============================================================================
# Integral Entrópica
# =============================================================================

def integral_entropica(S_max: float, S_min: float = 0.0) -> float:
    """
    Calcula la integral entrópica ∫k(s)ds desde S_min hasta S_max.

    Esta integral representa la "entropía acumulada" hasta el sello S.

    Args:
        S_max: Límite superior
        S_min: Límite inferior (default 0)

    Returns:
        Valor de la integral
    """
    resultado, _ = quad(tasa_colapso_k, S_min, S_max)
    return resultado


def integral_total() -> float:
    """
    Calcula la integral total desde S0 hasta S4.

    Usada para normalizar ε(S).
    """
    return integral_entropica(S4, 0.0)


def calcular_epsilon(S: float) -> float:
    """
    Calcula la fracción de conversión ε(S).

    ε(S) = ∫₀ˢ k(s)ds / ∫₀^{S4} k(s)ds

    Propiedades:
        - ε(S0) = 0 (nada convertido)
        - ε(S4) = 1 (todo convertido)
        - ε es monótonamente creciente

    Args:
        S: Sello entrópico

    Returns:
        ε(S) ∈ [0, 1]
    """
    if S <= 0:
        return 0.0
    if S >= S4:
        return 1.0

    I_S = integral_entropica(S)
    I_total = integral_total()

    return I_S / I_total


def calcular_Mp_Ep(S: float) -> Tuple[float, float]:
    """
    Calcula Mp(S) y Ep(S) a partir del sello entrópico.

    Mp(S) = Mp0 × (1 - ε(S))
    Ep(S) = Mp0 × ε(S) + Ep0

    Args:
        S: Sello entrópico

    Returns:
        Tupla (Mp, Ep)
    """
    epsilon = calcular_epsilon(S)
    Mp = Mp0 * (1 - epsilon)
    Ep = Mp0 * epsilon + Ep0
    return Mp, Ep


# =============================================================================
# Clase Principal
# =============================================================================

@dataclass
class Pregeometria:
    """
    Describe la estructura pregeométrica del modelo MCMC.

    La pregeometría es el sustrato subyacente del cual emerge
    el espacio-tiempo. Se caracteriza por:
        - La tasa de colapso k(S)
        - La integral entrópica
        - La evolución de Mp, Ep, P_ME

    Attributes:
        k0: Tasa de colapso base
        a1, a2, a3: Coeficientes armónicos
        n_puntos: Resolución para cálculos numéricos
    """
    k0: float = K0
    a1: float = A1
    a2: float = A2
    a3: float = A3
    n_puntos: int = 1000

    # Cache para valores pre-calculados
    _S_array: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _k_array: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    _epsilon_array: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    def __post_init__(self):
        """Inicializa arrays pre-calculados."""
        self._calcular_arrays()

    def _calcular_arrays(self) -> None:
        """Pre-calcula arrays para interpolación eficiente."""
        self._S_array = np.linspace(0, S4, self.n_puntos)
        self._k_array = np.array([
            tasa_colapso_k(s, self.k0, self.a1, self.a2, self.a3)
            for s in self._S_array
        ])
        self._epsilon_array = np.array([
            calcular_epsilon(s) for s in self._S_array
        ])

    def k(self, S: float) -> float:
        """Tasa de colapso en S."""
        return tasa_colapso_k(S, self.k0, self.a1, self.a2, self.a3)

    def epsilon(self, S: float) -> float:
        """Fracción de conversión en S."""
        return calcular_epsilon(S)

    def Mp(self, S: float) -> float:
        """Masa potencial en S."""
        return Mp0 * (1 - self.epsilon(S))

    def Ep(self, S: float) -> float:
        """Energía en S."""
        return Mp0 * self.epsilon(S) + Ep0

    def P_ME(self, S: float) -> float:
        """Polarización masa-energía en S."""
        Mp = self.Mp(S)
        Ep = self.Ep(S)
        return calcular_P_ME(Mp, Ep)

    def trayectoria(
        self,
        n_puntos: Optional[int] = None
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Calcula la trayectoria completa S0 → S4.

        Args:
            n_puntos: Número de puntos (default: self.n_puntos)

        Returns:
            Diccionario con arrays S, k, epsilon, Mp, Ep, P_ME
        """
        if n_puntos is None:
            n_puntos = self.n_puntos

        S = np.linspace(0, S4, n_puntos)
        k_vals = np.array([self.k(s) for s in S])
        eps_vals = np.array([self.epsilon(s) for s in S])
        Mp_vals = Mp0 * (1 - eps_vals)
        Ep_vals = Mp0 * eps_vals + Ep0
        P_ME_vals = (Mp_vals - Ep_vals) / (Mp_vals + Ep_vals)

        return {
            "S": S,
            "k": k_vals,
            "epsilon": eps_vals,
            "Mp": Mp_vals,
            "Ep": Ep_vals,
            "P_ME": P_ME_vals,
        }

    def evaluar_en_sellos(self) -> Dict[str, Dict[str, float]]:
        """
        Evalúa todas las cantidades en los sellos discretos.

        Returns:
            Diccionario {sello: {variable: valor}}
        """
        resultados = {}

        for nombre, S in SELLOS.items():
            resultados[nombre] = {
                "S": S,
                "k": self.k(S),
                "epsilon": self.epsilon(S),
                "Mp": self.Mp(S),
                "Ep": self.Ep(S),
                "P_ME": self.P_ME(S),
            }

        return resultados

    def encontrar_S_por_P_ME(self, P_ME_objetivo: float) -> float:
        """
        Encuentra el sello S correspondiente a un valor de P_ME.

        Usa búsqueda binaria ya que P_ME es monótonamente decreciente.

        Args:
            P_ME_objetivo: Valor de P_ME buscado (-1 a +1)

        Returns:
            S correspondiente
        """
        if P_ME_objetivo > self.P_ME(0):
            return 0.0
        if P_ME_objetivo < self.P_ME(S4):
            return S4

        # Búsqueda binaria
        S_min, S_max = 0.0, S4
        while S_max - S_min > 1e-8:
            S_mid = (S_min + S_max) / 2
            P_mid = self.P_ME(S_mid)

            if P_mid > P_ME_objetivo:
                S_min = S_mid
            else:
                S_max = S_mid

        return (S_min + S_max) / 2

    def punto_equilibrio(self) -> float:
        """
        Encuentra el sello donde P_ME = 0 (Mp = Ep).

        Este es el punto de equilibrio masa-energía.
        """
        return self.encontrar_S_por_P_ME(0.0)

    def resumen(self) -> str:
        """Genera resumen de la pregeometría."""
        S_eq = self.punto_equilibrio()

        return (
            f"Pregeometría MCMC\n"
            f"{'='*50}\n"
            f"Parámetros: k0={self.k0:.3f}, a1={self.a1}, "
            f"a2={self.a2}, a3={self.a3}\n"
            f"{'='*50}\n"
            f"k(S0) = {self.k(0):.4f}\n"
            f"k(S4) = {self.k(S4):.4f}\n"
            f"Integral total = {integral_total():.4f}\n"
            f"Punto de equilibrio (P_ME=0): S = {S_eq:.4f}\n"
        )


# =============================================================================
# Funciones de Análisis
# =============================================================================

def analizar_armonicos() -> Dict[str, float]:
    """
    Analiza la contribución de cada armónico a k(S).

    Returns:
        Diccionario con amplitudes relativas
    """
    # Amplitud máxima de cada término
    amp_base = K0
    amp_1 = K0 * abs(A1)
    amp_2 = K0 * abs(A2)
    amp_3 = K0 * abs(A3)
    amp_total = amp_1 + amp_2 + amp_3

    return {
        "base": amp_base,
        "armonico_1": amp_1,
        "armonico_2": amp_2,
        "armonico_3": amp_3,
        "variacion_total": amp_total,
        "variacion_relativa": amp_total / amp_base,
    }


def generar_tabla_sellos() -> str:
    """Genera tabla formateada con valores en cada sello."""
    preg = Pregeometria()
    datos = preg.evaluar_en_sellos()

    lineas = [
        "Sello |    S    |   k(S)  | ε(S)  |   Mp    |    Ep     | P_ME",
        "-"*70,
    ]

    for nombre in SELLOS_ORDEN:
        d = datos[nombre]
        lineas.append(
            f"  {nombre}  | {d['S']:.3f}   | {d['k']:.3f}   | {d['epsilon']:.3f} | "
            f"{d['Mp']:.4f}  | {d['Ep']:.2e} | {d['P_ME']:+.4f}"
        )

    return "\n".join(lineas)


# =============================================================================
# Tests
# =============================================================================

def _test_pregeometria():
    """Verifica la implementación de la pregeometría."""

    preg = Pregeometria()

    # Test 1: k(S) siempre positiva
    for S in np.linspace(0, S4, 100):
        assert preg.k(S) > 0, f"k({S}) = {preg.k(S)} debe ser positiva"

    # Test 2: ε(0) = 0, ε(S4) = 1
    assert np.isclose(preg.epsilon(0), 0.0), f"ε(0) = {preg.epsilon(0)}"
    assert np.isclose(preg.epsilon(S4), 1.0, rtol=1e-3), f"ε(S4) = {preg.epsilon(S4)}"

    # Test 3: ε es monótonamente creciente
    eps_prev = 0
    for S in np.linspace(0, S4, 100):
        eps = preg.epsilon(S)
        assert eps >= eps_prev, f"ε no es creciente en S={S}"
        eps_prev = eps

    # Test 4: P_ME va de +1 a ~-1
    assert preg.P_ME(0) > 0.99, f"P_ME(0) = {preg.P_ME(0)}"
    assert preg.P_ME(S4) < -0.9, f"P_ME(S4) = {preg.P_ME(S4)}"

    # Test 5: P_ME es monótonamente decreciente
    P_prev = 2
    for S in np.linspace(0, S4, 100):
        P = preg.P_ME(S)
        assert P < P_prev, f"P_ME no decrece en S={S}"
        P_prev = P

    # Test 6: Punto de equilibrio existe
    S_eq = preg.punto_equilibrio()
    assert 0 < S_eq < S4, f"S_equilibrio = {S_eq} fuera de rango"
    assert np.isclose(preg.P_ME(S_eq), 0.0, atol=1e-4), \
        f"P_ME(S_eq) = {preg.P_ME(S_eq)} ≠ 0"

    # Test 7: Conservación de energía
    for S in np.linspace(0, S4, 50):
        Mp = preg.Mp(S)
        Ep = preg.Ep(S)
        E_total = Mp + Ep
        E_inicial = Mp0 + Ep0
        assert np.isclose(E_total, E_inicial, rtol=1e-6), \
            f"Energía no conservada en S={S}"

    print("✓ Todos los tests del Bloque 1 pasaron")
    return True


if __name__ == "__main__":
    _test_pregeometria()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Pregeometría MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    preg = Pregeometria()
    print(preg.resumen())

    print("\nAnálisis de armónicos:")
    for k, v in analizar_armonicos().items():
        print(f"  {k}: {v:.4f}")

    print("\n" + generar_tabla_sellos())

    # Mostrar trayectoria
    print("\n\nTrayectoria P_ME(S):")
    print("-"*40)
    traj = preg.trayectoria(n_puntos=11)
    for i in range(len(traj["S"])):
        print(f"S = {traj['S'][i]:.3f}: P_ME = {traj['P_ME'][i]:+.4f}")
