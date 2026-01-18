"""
Fase Pre-Geométrica - Evolución S ∈ [0, 1.001]
================================================

Implementación completa de la fase pre-geométrica del modelo MCMC,
desde la singularidad ontológica (S₀) hasta el nacimiento del tiempo (S₄).

TASA DE CONVERSIÓN k(S):
    k(S) = k₀ × ∏[1 - aₙ × θ(S - Sₙ)]

    Parámetros calibrados:
    - k₀ = 6.6252 Gyr⁻¹
    - a₁ = 0.1416 (S₁ = 0.010)
    - a₂ = 0.2355 (S₂ = 0.100)
    - a₃ = 0.3439 (S₃ = 1.000)

CONDICIÓN DE DISEÑO:
    ∫₀^S₄ k(S) dS = ln(1/ε) = 4.42
    con ε = 0.012 (fracción residual de masa primordial)

SELLOS ENTRÓPICOS:
    S₀ = 0.000: Singularidad ontológica (V₀D)
    S₁ = 0.010: Primera dimensionalización (V₁D)
    S₂ = 0.100: Segunda dimensionalización (V₂D)
    S₃ = 1.000: Tercera dimensionalización (V₃D)
    S₄ = 1.001: Big Bang / Nacimiento del tiempo (V₃₊₁D)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray
from scipy.integrate import quad


# =============================================================================
# Constantes Calibradas del Modelo
# =============================================================================

# Tasa inicial de conversión (Gyr⁻¹)
K0_CALIBRADO: float = 6.6252

# Coeficientes de reducción por sello
A1: float = 0.1416  # Reducción V₀D → V₁D
A2: float = 0.2355  # Reducción V₁D → V₂D
A3: float = 0.3439  # Reducción V₂D → V₃D

# Sellos entrópicos
S0: float = 0.000
S1: float = 0.009
S2: float = 0.099
S3: float = 0.999
S4: float = 1.001

# Fracción residual de masa (cristalizada en S₄)
EPSILON_RESIDUAL: float = 0.0112

# Integral objetivo: ln(1/ε)
INTEGRAL_OBJETIVO: float = np.log(1 / EPSILON_RESIDUAL)  # ≈ 4.42

# Dimensionalidades por sello
DIMENSIONES: Dict[str, str] = {
    "S0": "V₀D (0 dimensiones)",
    "S1": "V₁D (1 dimensión)",
    "S2": "V₂D (2 dimensiones)",
    "S3": "V₃D (3 dimensiones)",
    "S4": "V₃₊₁D (3+1 dimensiones, tiempo)",
}


# =============================================================================
# Función de Tasa de Conversión k(S)
# =============================================================================

def theta_heaviside(S: float, S_n: float) -> float:
    """Función escalón de Heaviside."""
    return 1.0 if S >= S_n else 0.0


def k_calibrado(
    S: float,
    k0: float = K0_CALIBRADO,
    a1: float = A1,
    a2: float = A2,
    a3: float = A3
) -> float:
    """
    Tasa de conversión masa→espacio calibrada.

    k(S) = k₀ × (1 - a₁θ(S-S₁)) × (1 - a₂θ(S-S₂)) × (1 - a₃θ(S-S₃))

    Esta función reduce escalonadamente en cada sello,
    modelando la "resistencia" creciente a la conversión
    conforme emergen las dimensiones espaciales.

    Args:
        S: Sello entrópico
        k0: Tasa inicial
        a1, a2, a3: Coeficientes de reducción

    Returns:
        k(S) en Gyr⁻¹
    """
    factor1 = 1 - a1 * theta_heaviside(S, S1)
    factor2 = 1 - a2 * theta_heaviside(S, S2)
    factor3 = 1 - a3 * theta_heaviside(S, S3)

    return k0 * factor1 * factor2 * factor3


def integral_k(S_max: float, S_min: float = 0.0) -> float:
    """
    Calcula ∫ₛₘᵢₙ^ₛₘₐₓ k(S) dS.

    Esta integral determina la fracción de masa convertida.
    """
    resultado, _ = quad(k_calibrado, S_min, S_max)
    return resultado


def verificar_calibracion() -> Dict[str, float]:
    """
    Verifica que la calibración cumple ∫k(S)dS = ln(1/ε).
    """
    I_total = integral_k(S4, 0.0)
    epsilon_calculado = np.exp(-I_total)

    return {
        "integral_total": I_total,
        "integral_objetivo": INTEGRAL_OBJETIVO,
        "error_relativo": abs(I_total - INTEGRAL_OBJETIVO) / INTEGRAL_OBJETIVO,
        "epsilon_calculado": epsilon_calculado,
        "epsilon_objetivo": EPSILON_RESIDUAL,
    }


# =============================================================================
# Evolución de Fracciones Mp(S) y Ep(S)
# =============================================================================

def Mp_fraccion(S: float) -> float:
    """
    Fracción de masa primordial Mp(S)/Mp₀.

    Mp(S)/Mp₀ = exp(-∫₀ˢ k(S') dS')

    Args:
        S: Sello entrópico

    Returns:
        Mp/Mp₀ ∈ [ε, 1]
    """
    if S <= 0:
        return 1.0
    if S >= S4:
        return EPSILON_RESIDUAL

    I_S = integral_k(S, 0.0)
    return np.exp(-I_S)


def Ep_fraccion(S: float) -> float:
    """
    Fracción de espacio primordial Ep(S)/Mp₀.

    Ep(S)/Mp₀ = 1 - Mp(S)/Mp₀

    Por conservación: Mp + Ep = Mp₀
    """
    return 1.0 - Mp_fraccion(S)


def P_ME_calibrado(S: float) -> float:
    """
    Polarización masa-espacio calibrada.

    P_ME = (Mp - Ep)/(Mp + Ep) = 2×Mp/Mp₀ - 1

    Args:
        S: Sello entrópico

    Returns:
        P_ME ∈ [-1, +1]
    """
    return 2 * Mp_fraccion(S) - 1


# =============================================================================
# Tabla de Estados en Sellos
# =============================================================================

def generar_tabla_sellos() -> List[Dict]:
    """
    Genera tabla completa de estados en cada sello.
    """
    sellos = [
        ("S0", S0),
        ("S1", S1),
        ("S2", S2),
        ("S3", S3),
        ("S4", S4),
    ]

    tabla = []
    for nombre, S in sellos:
        Mp = Mp_fraccion(S)
        Ep = Ep_fraccion(S)
        P = P_ME_calibrado(S)
        k_val = k_calibrado(S)

        tabla.append({
            "sello": nombre,
            "S": S,
            "Mp/Mp0": Mp,
            "Ep/Mp0": Ep,
            "P_ME": P,
            "k(S)": k_val,
            "geometria": DIMENSIONES.get(nombre, ""),
        })

    return tabla


# =============================================================================
# Secuencia de 10 Colapsos
# =============================================================================

def generar_secuencia_colapsos(n_pasos: int = 11) -> List[Dict]:
    """
    Genera la secuencia de n_pasos desde S₀ hasta S₄.

    Args:
        n_pasos: Número de puntos (default 11 para 10 colapsos)

    Returns:
        Lista de estados con todas las propiedades
    """
    S_valores = np.linspace(0, S4, n_pasos)

    secuencia = []
    for i, S in enumerate(S_valores):
        Mp = Mp_fraccion(S)
        Ep = Ep_fraccion(S)
        P = P_ME_calibrado(S)

        # Amplitudes del qubit tensorial
        alpha = np.sqrt(Mp)
        beta = np.sqrt(Ep)

        # Ángulo para gate RY
        theta = 2 * np.arccos(alpha) if Mp > 0 else np.pi

        # Entropía de entrelazamiento
        if Mp > 1e-10 and Ep > 1e-10:
            S_ent = -Mp * np.log2(Mp) - Ep * np.log2(Ep)
        else:
            S_ent = 0.0

        secuencia.append({
            "paso": i,
            "S": S,
            "|c0|^2": Mp,
            "|c1|^2": Ep,
            "P_ME": P,
            "alpha": alpha,
            "beta": beta,
            "theta_RY": theta,
            "S_ent": S_ent,
            "ZZ": 1.0,  # Siempre 1 para estados |00⟩ + |11⟩
        })

    return secuencia


# =============================================================================
# Clase Principal: FasePregeometrica
# =============================================================================

@dataclass
class FasePregeometrica:
    """
    Representa la fase pre-geométrica completa S ∈ [0, 1.001].

    Esta clase encapsula toda la evolución desde la singularidad
    ontológica hasta el nacimiento del tiempo (Big Bang).

    TRANSICIONES DIMENSIONALES:
        S₀ → S₁: Emerge la primera dimensión espacial
        S₁ → S₂: Emerge la segunda dimensión espacial
        S₂ → S₃: Emerge la tercera dimensión espacial
        S₃ → S₄: Emerge el tiempo (Big Bang)

    Attributes:
        k0: Tasa inicial de conversión
        a1, a2, a3: Coeficientes de reducción
        epsilon: Fracción residual de masa
    """
    k0: float = K0_CALIBRADO
    a1: float = A1
    a2: float = A2
    a3: float = A3
    epsilon: float = EPSILON_RESIDUAL

    def k(self, S: float) -> float:
        """Tasa de conversión en S."""
        return k_calibrado(S, self.k0, self.a1, self.a2, self.a3)

    def Mp(self, S: float) -> float:
        """Fracción de masa Mp/Mp₀."""
        return Mp_fraccion(S)

    def Ep(self, S: float) -> float:
        """Fracción de espacio Ep/Mp₀."""
        return Ep_fraccion(S)

    def P_ME(self, S: float) -> float:
        """Polarización masa-espacio."""
        return P_ME_calibrado(S)

    def alpha(self, S: float) -> float:
        """Amplitud de masa α = √(Mp/Mp₀)."""
        return np.sqrt(self.Mp(S))

    def beta(self, S: float) -> float:
        """Amplitud de espacio β = √(Ep/Mp₀)."""
        return np.sqrt(self.Ep(S))

    def theta_RY(self, S: float) -> float:
        """Ángulo de rotación RY para preparar |Φ(S)⟩."""
        alpha = self.alpha(S)
        if alpha > 1e-10:
            return 2 * np.arccos(alpha)
        return np.pi

    def entropia_entrelazamiento(self, S: float) -> float:
        """Entropía de entrelazamiento en bits."""
        Mp = self.Mp(S)
        Ep = self.Ep(S)
        if Mp > 1e-10 and Ep > 1e-10:
            return -Mp * np.log2(Mp) - Ep * np.log2(Ep)
        return 0.0

    def dimensionalidad(self, S: float) -> str:
        """Retorna la dimensionalidad en S."""
        if S < S1:
            return "V₀D"
        elif S < S2:
            return "V₁D"
        elif S < S3:
            return "V₂D"
        elif S < S4:
            return "V₃D"
        else:
            return "V₃₊₁D"

    def es_big_bang(self, S: float) -> bool:
        """Determina si S corresponde al Big Bang."""
        return np.isclose(S, S4, atol=1e-6)

    def trayectoria(self, n_puntos: int = 100) -> Dict[str, NDArray]:
        """Genera trayectoria completa S₀ → S₄."""
        S_vals = np.linspace(0, S4, n_puntos)

        return {
            "S": S_vals,
            "Mp": np.array([self.Mp(s) for s in S_vals]),
            "Ep": np.array([self.Ep(s) for s in S_vals]),
            "P_ME": np.array([self.P_ME(s) for s in S_vals]),
            "k": np.array([self.k(s) for s in S_vals]),
            "theta": np.array([self.theta_RY(s) for s in S_vals]),
            "S_ent": np.array([self.entropia_entrelazamiento(s) for s in S_vals]),
        }

    def secuencia_colapsos(self, n_pasos: int = 11) -> List[Dict]:
        """Genera secuencia de colapsos."""
        return generar_secuencia_colapsos(n_pasos)

    def verificar_calibracion(self) -> Dict[str, float]:
        """Verifica la calibración del modelo."""
        return verificar_calibracion()

    def resumen(self) -> str:
        """Genera resumen de la fase pre-geométrica."""
        cal = self.verificar_calibracion()

        return (
            f"Fase Pre-Geométrica MCMC\n"
            f"{'='*60}\n"
            f"Rango: S ∈ [0, {S4}]\n"
            f"Sellos: S₀=0, S₁=0.01, S₂=0.1, S₃=1.0, S₄=1.001\n"
            f"{'='*60}\n"
            f"Parámetros calibrados:\n"
            f"  k₀ = {self.k0:.4f} Gyr⁻¹\n"
            f"  a₁ = {self.a1:.4f} (S₁)\n"
            f"  a₂ = {self.a2:.4f} (S₂)\n"
            f"  a₃ = {self.a3:.4f} (S₃)\n"
            f"{'='*60}\n"
            f"Verificación:\n"
            f"  ∫k(S)dS = {cal['integral_total']:.4f} (objetivo: {cal['integral_objetivo']:.4f})\n"
            f"  ε = {cal['epsilon_calculado']:.6f} (objetivo: {cal['epsilon_objetivo']:.6f})\n"
            f"  Error: {cal['error_relativo']*100:.2f}%\n"
            f"{'='*60}\n"
            f"Estado inicial (S₀):\n"
            f"  Mp/Mp₀ = {self.Mp(0):.6f}\n"
            f"  P_ME = {self.P_ME(0):+.6f}\n"
            f"  Dimensionalidad: {self.dimensionalidad(0)}\n"
            f"{'='*60}\n"
            f"Estado final (S₄ = Big Bang):\n"
            f"  Mp/Mp₀ = {self.Mp(S4):.6f}\n"
            f"  P_ME = {self.P_ME(S4):+.6f}\n"
            f"  Dimensionalidad: {self.dimensionalidad(S4)}\n"
        )


# =============================================================================
# Tests
# =============================================================================

def _test_fase_pregeometrica():
    """Verifica la implementación de la fase pre-geométrica."""

    fp = FasePregeometrica()

    # Test 1: Mp(0) = 1, Ep(0) = 0
    assert np.isclose(fp.Mp(0), 1.0), f"Mp(0) = {fp.Mp(0)} ≠ 1"
    assert np.isclose(fp.Ep(0), 0.0), f"Ep(0) = {fp.Ep(0)} ≠ 0"

    # Test 2: P_ME(0) = +1
    assert np.isclose(fp.P_ME(0), 1.0), f"P_ME(0) = {fp.P_ME(0)} ≠ +1"

    # Test 3: P_ME(S4) ≈ -0.978
    P_final = fp.P_ME(S4)
    assert P_final < -0.9, f"P_ME(S4) = {P_final} debe ser < -0.9"

    # Test 4: k(S) siempre positiva
    for S in np.linspace(0, S4, 50):
        assert fp.k(S) > 0, f"k({S}) debe ser > 0"

    # Test 5: Mp + Ep = 1 (conservación)
    for S in np.linspace(0, S4, 50):
        total = fp.Mp(S) + fp.Ep(S)
        assert np.isclose(total, 1.0), f"Mp + Ep = {total} ≠ 1 en S={S}"

    # Test 6: Calibración correcta
    cal = fp.verificar_calibracion()
    assert cal['error_relativo'] < 0.05, f"Error de calibración: {cal['error_relativo']*100}%"

    print("✓ Todos los tests de FasePregeometrica pasaron")
    return True


if __name__ == "__main__":
    _test_fase_pregeometrica()

    # Demo
    print("\n" + "="*70)
    print("DEMO: Fase Pre-Geométrica MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*70 + "\n")

    fp = FasePregeometrica()
    print(fp.resumen())

    print("\n\nSecuencia de 10 Colapsos S₀ → S₄:")
    print("-"*70)
    print(f"{'Paso':>4} {'S':>6} {'|c₀|²':>8} {'|c₁|²':>8} {'P_ME':>8} {'θ_RY':>8} {'S_ent':>6}")
    print("-"*70)

    for estado in fp.secuencia_colapsos(11):
        print(f"{estado['paso']:4d} {estado['S']:6.3f} "
              f"{estado['|c0|^2']:8.4f} {estado['|c1|^2']:8.4f} "
              f"{estado['P_ME']:+8.4f} {estado['theta_RY']:8.4f} "
              f"{estado['S_ent']:6.3f}")
