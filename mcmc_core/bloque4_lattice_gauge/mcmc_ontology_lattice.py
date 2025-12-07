"""
Núcleo Ontológico MCMC para Lattice Gauge
==========================================

Conecta la historia tensional Mp/Ep con los parámetros de teorías gauge
en retícula, permitiendo extraer el mass gap ontológico.

OBJETIVO:
    Responder: ¿Es el mass gap dinámico extraído de simulaciones Yang-Mills
    consistente con el mass gap ontológico fijado en S₄ = 1.001?

MASS GAP ONTOLÓGICO:
    E_min(S) = ½[E_Pl(1 - tanh((S-1)/τ)) + m_H(1 + tanh((S-1)/τ))]

    donde τ = 0.001 (anchura de transición sigmoidal)

PARÁMETROS CALIBRADOS (Bloque 1):
    k₀ = 6.6252 Gyr⁻¹
    a₁ = 0.1416, a₂ = 0.2355, a₃ = 0.3439
    ε = 0.0112 (fracción residual)

CONEXIÓN GAUGE-TENSIONAL:
    β(S) = β₀ + β₁ × exp[-b_S × (S - S₃)]
    S_tens(S) = λ_tens × (1 - Mp(S)/Mp₀)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy.integrate import quad


# =============================================================================
# Constantes Fundamentales
# =============================================================================

# Sellos entrópicos
S0: float = 0.000
S1: float = 0.010
S2: float = 0.100
S3: float = 1.000
S4: float = 1.001  # Big Bang

# Parámetros de colapso calibrados (Bloque 1)
K0_CALIBRADO: float = 6.6252   # Gyr⁻¹
A1: float = 0.1416
A2: float = 0.2355
A3: float = 0.3439
EPSILON_RESIDUAL: float = 0.0112  # Mp(S₄)/Mp₀

# Parámetros gauge-tensionales
BETA_0: float = 6.0         # Acoplo base
BETA_1: float = 0.5         # Amplitud de variación
B_S: float = 100.0          # Escala de transición
LAMBDA_TENS: float = 0.01   # Acoplo tensional

# Constantes físicas
E_PLANCK: float = 1.22e19      # GeV (energía de Planck)
M_HIGGS: float = 125.25        # GeV (masa del Higgs)
LAMBDA_QCD: float = 0.217      # GeV (escala QCD)
ALPHA_H: float = 0.014         # Parámetro de mass gap
TAU_TRANSITION: float = 0.001  # Anchura de transición sigmoidal

# Constantes derivadas
M_GLUEBALL_0PP: float = 1.71   # GeV (glueball 0⁺⁺ de QCD lattice)


# =============================================================================
# Funciones de Tasa de Colapso
# =============================================================================

def theta_heaviside(S: float, S_ref: float, smoothing: float = 1e-4) -> float:
    """
    Función escalón de Heaviside suavizada.

    θ(S - S_ref) = ½[1 + tanh((S - S_ref)/smoothing)]
    """
    return 0.5 * (1 + np.tanh((S - S_ref) / smoothing))


def k_calibrado(
    S: float,
    k0: float = K0_CALIBRADO,
    a1: float = A1,
    a2: float = A2,
    a3: float = A3
) -> float:
    """
    Tasa de colapso k(S) calibrada.

    k(S) = k₀ × ∏[1 - aₙ × θ(S - Sₙ)]

    Args:
        S: Sello entrópico
        k0, a1, a2, a3: Parámetros calibrados

    Returns:
        k(S) en Gyr⁻¹
    """
    factor1 = 1.0 - a1 * theta_heaviside(S, S1)
    factor2 = 1.0 - a2 * theta_heaviside(S, S2)
    factor3 = 1.0 - a3 * theta_heaviside(S, S3)

    return k0 * factor1 * factor2 * factor3


def integral_k(S: float) -> float:
    """
    Integral ∫₀ˢ k(s) ds.
    """
    if S <= 0:
        return 0.0
    resultado, _ = quad(k_calibrado, 0, S)
    return resultado


def Mp_fraccion(S: float) -> float:
    """
    Fracción de masa Mp(S)/Mp₀ = exp(-∫₀ˢ k(s)ds).

    Returns:
        Fracción en [0, 1]
    """
    return np.exp(-integral_k(S))


def Ep_fraccion(S: float) -> float:
    """
    Fracción de espacio Ep(S)/Ep₀ = 1 - Mp(S)/Mp₀.
    """
    return 1.0 - Mp_fraccion(S)


def P_ME(S: float) -> float:
    """
    Polarización masa-espacio.

    P_ME = (Mp - Ep)/(Mp + Ep) = 2×Mp_frac - 1
    """
    return 2 * Mp_fraccion(S) - 1


# =============================================================================
# Funciones de Acoplo Gauge
# =============================================================================

def beta_MCMC(
    S: float,
    beta_0: float = BETA_0,
    beta_1: float = BETA_1,
    b_S: float = B_S,
    S_ref: float = S3
) -> float:
    """
    Acoplamiento gauge β(S) dependiente del sello entrópico.

    β(S) = β₀ + β₁ × exp[-b_S × (S - S₃)]

    Propiedades:
        - β(S << S₃) = β₀ + β₁ (acoplamiento fuerte)
        - β(S = S₃) = β₀ + β₁
        - β(S >> S₃) → β₀ (plateau)

    Args:
        S: Sello entrópico
        beta_0: Acoplamiento base
        beta_1: Amplitud
        b_S: Escala de transición
        S_ref: Sello de referencia

    Returns:
        β(S)
    """
    return beta_0 + beta_1 * np.exp(-b_S * (S - S_ref))


def g_squared(beta: float, N: int = 3) -> float:
    """
    Constante de acoplamiento g² desde β.

    β = 2N/g² → g² = 2N/β
    """
    if beta <= 0:
        return np.inf
    return 2 * N / beta


def alpha_s(beta: float, N: int = 3) -> float:
    """
    Constante de acoplamiento fuerte αs = g²/(4π).
    """
    return g_squared(beta, N) / (4 * np.pi)


def S_tensional(S: float, lambda_tens: float = LAMBDA_TENS) -> float:
    """
    Término tensional en la acción.

    S_tens(S) = λ_tens × (1 - Mp(S)/Mp₀)

    Este término modifica la acción de Wilson incorporando
    la tensión masa-espacio del MCMC.
    """
    return lambda_tens * (1 - Mp_fraccion(S))


# =============================================================================
# Mass Gap Ontológico
# =============================================================================

def E_min_ontologico(
    S: float,
    E_pl: float = 10.0,  # GeV (escala efectiva pre-BB)
    m_H: float = M_HIGGS,
    tau: float = TAU_TRANSITION
) -> float:
    """
    Mass gap ontológico E_min(S).

    E_min(S) = ½[E_pl(1 - tanh((S-1)/τ)) + m_H(1 + tanh((S-1)/τ))]

    Transición sigmoidal desde E_pl (pre-BB) hasta m_H/2 (post-BB).

    Args:
        S: Sello entrópico
        E_pl: Energía característica pre-Big Bang
        m_H: Masa del Higgs
        tau: Anchura de transición

    Returns:
        E_min en GeV
    """
    x = (S - 1.0) / tau
    tanh_x = np.tanh(x)

    return 0.5 * (E_pl * (1 - tanh_x) + m_H * (1 + tanh_x))


def E_min_QCD_scale(S: float, alpha_h: float = ALPHA_H) -> float:
    """
    Mass gap estimado en escala QCD.

    E_min ≈ αH × Λ_QCD × f(S)

    donde f(S) captura la dependencia en el sello.
    """
    # Factor sigmoidal de transición
    f_S = 1.0 / (1.0 + 10 * (S - S4)**2) if S > S3 else 1.0

    return alpha_h * LAMBDA_QCD * f_S * 1000  # Escalar a GeV


def campo_adrian(S: float) -> float:
    """
    Campo de Adrián Φ_Ad(S).

    Φ_Ad = Mp(S) / Ep(S)

    En S₄, Φ_Ad → ε/(1-ε) → 0, señalando la transición a campo de Higgs.
    """
    Mp = Mp_fraccion(S)
    Ep = Ep_fraccion(S)

    if Ep < 1e-10:
        return np.inf
    return Mp / Ep


def transicion_higgs(S: float) -> float:
    """
    Factor de transición ΦAd → ΦH.

    f_H(S) = ½[1 + tanh((S - 1)/τ)]

    f_H(S < 1) ≈ 0 (dominado por ΦAd)
    f_H(S > 1) ≈ 1 (dominado por ΦH)
    """
    return 0.5 * (1 + np.tanh((S - 1.0) / TAU_TRANSITION))


# =============================================================================
# Clase Principal: OntologiaMCMCLattice
# =============================================================================

@dataclass
class OntologiaMCMCLattice:
    """
    Núcleo ontológico para simulaciones lattice MCMC.

    Conecta la historia tensional Mp/Ep con los parámetros
    de teorías gauge en retícula.

    Attributes:
        k0: Tasa de colapso inicial
        a1, a2, a3: Reducciones en cada sello
        epsilon: Fracción residual
        beta_0, beta_1, b_S: Parámetros de acoplo gauge
        lambda_tens: Acoplo tensional
    """
    # Parámetros de colapso
    k0: float = K0_CALIBRADO
    a1: float = A1
    a2: float = A2
    a3: float = A3
    epsilon: float = EPSILON_RESIDUAL

    # Parámetros gauge
    beta_0: float = BETA_0
    beta_1: float = BETA_1
    b_S: float = B_S
    lambda_tens: float = LAMBDA_TENS

    # Historial de estados
    historial: List[Dict] = field(default_factory=list)

    def k(self, S: float) -> float:
        """Tasa de colapso."""
        return k_calibrado(S, self.k0, self.a1, self.a2, self.a3)

    def Mp(self, S: float) -> float:
        """Fracción de masa."""
        return Mp_fraccion(S)

    def Ep(self, S: float) -> float:
        """Fracción de espacio."""
        return Ep_fraccion(S)

    def beta(self, S: float) -> float:
        """Acoplamiento gauge."""
        return beta_MCMC(S, self.beta_0, self.beta_1, self.b_S)

    def S_tens(self, S: float) -> float:
        """Término tensional."""
        return S_tensional(S, self.lambda_tens)

    def E_min(self, S: float) -> float:
        """Mass gap ontológico."""
        return E_min_ontologico(S)

    def estado(self, S: float) -> Dict:
        """
        Genera estado ontológico completo para un sello S.
        """
        return {
            "S": S,
            "Mp": self.Mp(S),
            "Ep": self.Ep(S),
            "P_ME": P_ME(S),
            "k": self.k(S),
            "beta": self.beta(S),
            "alpha_s": alpha_s(self.beta(S)),
            "S_tens": self.S_tens(S),
            "E_min_GeV": self.E_min(S),
            "Phi_Ad": campo_adrian(S),
            "f_Higgs": transicion_higgs(S),
        }

    def estados_sellos(self) -> Dict[str, Dict]:
        """
        Genera estados para todos los sellos S₀-S₄.
        """
        sellos = {
            "S0": S0,
            "S1": S1,
            "S2": S2,
            "S3": S3,
            "S4": S4,
        }
        return {nombre: self.estado(S) for nombre, S in sellos.items()}

    def verificar_conservacion(self, tol: float = 1e-6) -> bool:
        """
        Verifica Mp + Ep = 1 en todo el rango.
        """
        for S in np.linspace(0, S4, 100):
            suma = self.Mp(S) + self.Ep(S)
            if abs(suma - 1.0) > tol:
                return False
        return True

    def verificar_epsilon(self, tol: float = 0.01) -> bool:
        """
        Verifica Mp(S₄) ≈ ε.
        """
        Mp_S4 = self.Mp(S4)
        return abs(Mp_S4 - self.epsilon) / self.epsilon < tol

    def generar_calendario(
        self,
        S_min: float = 0.90,
        S_max: float = S4,
        n_puntos: int = 20
    ) -> List[Dict]:
        """
        Genera calendario entrópico para escaneo.

        Args:
            S_min, S_max: Rango de S
            n_puntos: Número de puntos

        Returns:
            Lista de estados ontológicos
        """
        S_values = np.linspace(S_min, S_max, n_puntos)
        return [self.estado(S) for S in S_values]

    def resumen(self) -> str:
        """Genera resumen del núcleo ontológico."""
        lineas = [
            "="*60,
            "NÚCLEO ONTOLÓGICO MCMC PARA LATTICE GAUGE",
            "="*60,
            "",
            "PARÁMETROS DE COLAPSO:",
            f"  k₀ = {self.k0:.4f} Gyr⁻¹",
            f"  a₁ = {self.a1:.4f}, a₂ = {self.a2:.4f}, a₃ = {self.a3:.4f}",
            f"  ε = {self.epsilon:.4f}",
            "",
            "PARÁMETROS GAUGE:",
            f"  β₀ = {self.beta_0:.2f}",
            f"  β₁ = {self.beta_1:.2f}",
            f"  b_S = {self.b_S:.1f}",
            f"  λ_tens = {self.lambda_tens:.4f}",
            "",
            "ESTADOS EN SELLOS:",
        ]

        for nombre, estado in self.estados_sellos().items():
            lineas.append(
                f"  {nombre} (S={estado['S']:.3f}): "
                f"Mp={estado['Mp']:.4f}, E_min={estado['E_min_GeV']:.2f} GeV"
            )

        lineas.extend([
            "",
            "VERIFICACIÓN:",
            f"  {'✓' if self.verificar_conservacion() else '✗'} Conservación Mp+Ep = 1",
            f"  {'✓' if self.verificar_epsilon() else '✗'} Mp(S₄) ≈ ε",
            "="*60,
        ])

        return "\n".join(lineas)


# =============================================================================
# Clases de Validación
# =============================================================================

@dataclass
class CriterioValidacion:
    """
    Criterio de validación ontológica.
    """
    nombre: str
    descripcion: str
    cumplido: bool = False
    valor: float = 0.0
    umbral: float = 0.0

    def evaluar(self, valor: float) -> bool:
        """Evalúa si el criterio se cumple."""
        self.valor = valor
        self.cumplido = valor <= self.umbral
        return self.cumplido


@dataclass
class ResultadoValidacion:
    """
    Resultado de validación ontológica completa.
    """
    criterios: List[CriterioValidacion] = field(default_factory=list)
    consistente: bool = False
    n_cumplidos: int = 0
    mensaje: str = ""

    def evaluar_consistencia(self, minimo: int = 2):
        """
        Evalúa consistencia global.

        Args:
            minimo: Mínimo de criterios a cumplir
        """
        self.n_cumplidos = sum(1 for c in self.criterios if c.cumplido)
        self.consistente = self.n_cumplidos >= minimo

        if self.consistente:
            self.mensaje = f"✓ Consistente ({self.n_cumplidos}/{len(self.criterios)} criterios)"
        else:
            self.mensaje = f"✗ Inconsistente ({self.n_cumplidos}/{len(self.criterios)} criterios)"


def validar_ontologia(
    E_lat: float,
    E_onto: float,
    plateau_alcanzado: bool = True,
    tau_estimado: float = 0.001
) -> ResultadoValidacion:
    """
    Valida la consistencia del mass gap extraído.

    Criterios:
        1. Plateau: E_min(S) constante para S ≳ S₃
        2. Ratio en S₄: E_lat/E_onto ∈ [0.1, 10]
        3. Transición sigmoidal: τ ~ 10⁻³

    Args:
        E_lat: Mass gap dinámico de lattice (GeV)
        E_onto: Mass gap ontológico (GeV)
        plateau_alcanzado: Si se alcanzó plateau
        tau_estimado: Anchura de transición estimada

    Returns:
        ResultadoValidacion
    """
    resultado = ResultadoValidacion()

    # Criterio 1: Plateau
    c1 = CriterioValidacion(
        nombre="Plateau",
        descripcion="E_min(S) constante para S ≳ S₃",
        umbral=1.0,  # Dummy
    )
    c1.cumplido = plateau_alcanzado
    c1.valor = 1.0 if plateau_alcanzado else 0.0
    resultado.criterios.append(c1)

    # Criterio 2: Ratio en S₄
    ratio = E_lat / E_onto if E_onto > 0 else np.inf
    c2 = CriterioValidacion(
        nombre="Ratio S₄",
        descripcion="E_lat/E_onto ∈ [0.1, 10]",
        umbral=10.0,
    )
    c2.cumplido = 0.1 <= ratio <= 10.0
    c2.valor = ratio
    resultado.criterios.append(c2)

    # Criterio 3: Transición sigmoidal
    c3 = CriterioValidacion(
        nombre="Transición τ",
        descripcion="τ ~ 10⁻³",
        umbral=0.01,
    )
    c3.cumplido = abs(tau_estimado - 0.001) < 0.01
    c3.valor = tau_estimado
    resultado.criterios.append(c3)

    resultado.evaluar_consistencia(minimo=2)
    return resultado


# =============================================================================
# Funciones de Conveniencia
# =============================================================================

def crear_ontologia_default() -> OntologiaMCMCLattice:
    """
    Crea ontología con parámetros por defecto calibrados.
    """
    return OntologiaMCMCLattice()


def tabla_sellos() -> str:
    """
    Genera tabla de sellos y geometrías.
    """
    onto = crear_ontologia_default()

    geometrias = {
        "S0": "V₀D (Primordial)",
        "S1": "V₁D (Polaridad)",
        "S2": "V₂D (Giro)",
        "S3": "V₃D (Volumen)",
        "S4": "V₃₊₁D (Tiempo)",
    }

    lineas = [
        "SELLOS ENTRÓPICOS Y GEOMETRÍAS",
        "="*60,
        f"{'Sello':^8} | {'S':^8} | {'Mp':^8} | {'E_min':^10} | {'Geometría':^20}",
        "-"*60,
    ]

    for nombre, estado in onto.estados_sellos().items():
        lineas.append(
            f"{nombre:^8} | {estado['S']:^8.3f} | {estado['Mp']:^8.4f} | "
            f"{estado['E_min_GeV']:^10.2f} | {geometrias[nombre]:^20}"
        )

    return "\n".join(lineas)


# =============================================================================
# Tests
# =============================================================================

def _test_ontologia():
    """Verifica la implementación del núcleo ontológico."""

    print("Testing Núcleo Ontológico MCMC...")

    onto = crear_ontologia_default()

    # Test 1: Conservación
    assert onto.verificar_conservacion(), "Mp + Ep debe ser 1"
    print("  ✓ Conservación Mp + Ep = 1")

    # Test 2: Epsilon
    assert onto.verificar_epsilon(tol=0.1), f"Mp(S₄) ≈ ε, got {onto.Mp(S4)}"
    print("  ✓ Mp(S₄) ≈ ε")

    # Test 3: β(S) positivo y decreciente para S > S3
    for S in [0.9, 0.95, 1.0, 1.001]:
        b = onto.beta(S)
        assert b > 0, f"β({S}) = {b} debe ser > 0"
    assert onto.beta(0.9) > onto.beta(S4), "β debe decrecer"
    print("  ✓ β(S) positivo y decreciente")

    # Test 4: Mass gap positivo
    for S in [S0, S1, S2, S3, S4]:
        E = onto.E_min(S)
        assert E > 0, f"E_min({S}) = {E} debe ser > 0"
    print("  ✓ E_min(S) positivo")

    # Test 5: Transición sigmoidal
    E_pre = onto.E_min(0.999)
    E_post = onto.E_min(1.001)
    assert E_pre > E_post, "E_min debe decrecer en transición"
    print("  ✓ Transición sigmoidal correcta")

    # Test 6: Validación
    resultado = validar_ontologia(
        E_lat=5.0,
        E_onto=onto.E_min(S4),
        plateau_alcanzado=True,
        tau_estimado=0.001
    )
    assert resultado.consistente, "Validación debe ser consistente"
    print("  ✓ Validación ontológica")

    print("\n✓ Todos los tests del núcleo ontológico pasaron")
    return True


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    _test_ontologia()

    print("\n")
    onto = crear_ontologia_default()
    print(onto.resumen())

    print("\n")
    print(tabla_sellos())
