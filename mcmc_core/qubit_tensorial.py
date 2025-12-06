"""
Qubit Tensorial - Validación Cuántica del Modelo MCMC
======================================================

Este módulo implementa la representación cuántica de la dualidad
tensional MASA-ESPACIO mediante estados de 2-qubits entrelazados.

ESTADO TENSORIAL:
    |Φ(S)⟩ = α(S)|00⟩ + β(S)|11⟩

    donde:
        α(S) = √(1 - ε(S))  (amplitud de masa)
        β(S) = √ε(S)        (amplitud de espacio)
        ε(S) = integral entrópica normalizada

INTERPRETACIÓN ONTOLÓGICA:
    - |00⟩: Estado de "masa pura" - correlación perfecta en el polo masa
    - |11⟩: Estado de "espacio puro" - correlación perfecta en el polo espacio
    - La superposición captura la dualidad tensional

    El estado |Φ(S)⟩ es un estado de Bell generalizado que codifica
    la transición ontológica desde masa (S=S₀) hacia espacio (S=S₄).

ENTRELAZAMIENTO:
    La concurrencia C(S) mide el entrelazamiento cuántico:
    C(S) = 2|α(S)β(S)| = 2√[ε(S)(1-ε(S))]

    - C = 0 en S₀ y S₄ (estados puros, no entrelazados)
    - C = 1 en S_eq (máximo entrelazamiento, equilibrio tensional)

CONEXIÓN CON GEOMETRÍA:
    El estado |Φ(S)⟩ puede interpretarse como un "bit de geometría"
    en el sentido de Loop Quantum Gravity: la geometría emerge de
    las correlaciones cuánticas entre los polos tensionales.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Referencias:
    Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray


# =============================================================================
# Constantes Cuánticas
# =============================================================================

# Estados base computacionales
ESTADO_00 = np.array([1, 0, 0, 0], dtype=np.complex128)  # |00⟩
ESTADO_01 = np.array([0, 1, 0, 0], dtype=np.complex128)  # |01⟩
ESTADO_10 = np.array([0, 0, 1, 0], dtype=np.complex128)  # |10⟩
ESTADO_11 = np.array([0, 0, 0, 1], dtype=np.complex128)  # |11⟩

# Estados de Bell
BELL_PHI_PLUS = (ESTADO_00 + ESTADO_11) / np.sqrt(2)   # |Φ+⟩
BELL_PHI_MINUS = (ESTADO_00 - ESTADO_11) / np.sqrt(2)  # |Φ-⟩
BELL_PSI_PLUS = (ESTADO_01 + ESTADO_10) / np.sqrt(2)   # |Ψ+⟩
BELL_PSI_MINUS = (ESTADO_01 - ESTADO_10) / np.sqrt(2)  # |Ψ-⟩

# Matrices de Pauli
PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# =============================================================================
# Funciones de Estado Tensorial
# =============================================================================

def calcular_amplitudes(epsilon: float) -> Tuple[complex, complex]:
    """
    Calcula las amplitudes α y β del estado tensorial.

    α(ε) = √(1 - ε)  (amplitud de masa)
    β(ε) = √ε        (amplitud de espacio)

    Args:
        epsilon: Fracción entrópica ε ∈ [0, 1]

    Returns:
        Tupla (α, β) de amplitudes complejas
    """
    if epsilon < 0 or epsilon > 1:
        raise ValueError(f"epsilon debe estar en [0, 1], recibido: {epsilon}")

    alpha = np.sqrt(1 - epsilon)
    beta = np.sqrt(epsilon)

    return complex(alpha), complex(beta)


def estado_tensorial(epsilon: float) -> NDArray[np.complex128]:
    """
    Construye el estado tensorial |Φ(ε)⟩.

    |Φ(ε)⟩ = α(ε)|00⟩ + β(ε)|11⟩

    Este estado codifica la dualidad MASA-ESPACIO del modelo MCMC
    como una superposición cuántica de estados correlacionados.

    Args:
        epsilon: Fracción entrópica ε ∈ [0, 1]

    Returns:
        Vector de estado de 4 componentes (base computacional)
    """
    alpha, beta = calcular_amplitudes(epsilon)
    return alpha * ESTADO_00 + beta * ESTADO_11


def matriz_densidad(psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Calcula la matriz de densidad ρ = |ψ⟩⟨ψ|.

    Args:
        psi: Vector de estado

    Returns:
        Matriz de densidad (4×4 para 2 qubits)
    """
    return np.outer(psi, np.conj(psi))


def traza_parcial_A(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Calcula la traza parcial sobre el subsistema A.

    ρ_B = Tr_A(ρ)

    Args:
        rho: Matriz de densidad de 2 qubits (4×4)

    Returns:
        Matriz de densidad reducida del subsistema B (2×2)
    """
    # Reshape a tensor (2,2,2,2)
    rho_tensor = rho.reshape(2, 2, 2, 2)
    # Traza sobre el primer subsistema
    rho_B = np.trace(rho_tensor, axis1=0, axis2=2)
    return rho_B


def traza_parcial_B(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Calcula la traza parcial sobre el subsistema B.

    ρ_A = Tr_B(ρ)

    Args:
        rho: Matriz de densidad de 2 qubits (4×4)

    Returns:
        Matriz de densidad reducida del subsistema A (2×2)
    """
    # Reshape a tensor (2,2,2,2)
    rho_tensor = rho.reshape(2, 2, 2, 2)
    # Traza sobre el segundo subsistema
    rho_A = np.trace(rho_tensor, axis1=1, axis2=3)
    return rho_A


# =============================================================================
# Medidas de Entrelazamiento
# =============================================================================

def concurrencia(psi: NDArray[np.complex128]) -> float:
    """
    Calcula la concurrencia del estado de 2 qubits.

    Para el estado tensorial |Φ(ε)⟩ = α|00⟩ + β|11⟩:
    C = 2|αβ| = 2√[ε(1-ε)]

    La concurrencia mide el entrelazamiento cuántico:
    - C = 0: Estado separable (no entrelazado)
    - C = 1: Máximamente entrelazado (estado de Bell)

    En el contexto MCMC:
    - C(S₀) = 0: Masa pura, sin entrelazamiento
    - C(S_eq) = 1: Equilibrio tensional, máximo entrelazamiento
    - C(S₄) = 0: Espacio puro, sin entrelazamiento

    Args:
        psi: Vector de estado de 2 qubits

    Returns:
        Concurrencia C ∈ [0, 1]
    """
    # Matriz σ_y ⊗ σ_y
    sigma_yy = np.kron(PAULI_Y, PAULI_Y)

    # Estado conjugado spin-flip
    psi_tilde = sigma_yy @ np.conj(psi)

    # Concurrencia = |⟨ψ|ψ̃⟩|
    C = np.abs(np.vdot(psi, psi_tilde))

    return float(C)


def concurrencia_desde_epsilon(epsilon: float) -> float:
    """
    Calcula la concurrencia directamente desde ε.

    C(ε) = 2√[ε(1-ε)]

    Esta es la forma analítica para el estado tensorial.

    Args:
        epsilon: Fracción entrópica ε ∈ [0, 1]

    Returns:
        Concurrencia C ∈ [0, 1]
    """
    if epsilon < 0 or epsilon > 1:
        raise ValueError(f"epsilon debe estar en [0, 1]")

    return 2 * np.sqrt(epsilon * (1 - epsilon))


def entropia_von_neumann(rho: NDArray[np.complex128]) -> float:
    """
    Calcula la entropía de von Neumann.

    S(ρ) = -Tr(ρ log₂ ρ)

    Args:
        rho: Matriz de densidad

    Returns:
        Entropía en bits
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filtrar valores muy pequeños para evitar log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def entropia_entrelazamiento(psi: NDArray[np.complex128]) -> float:
    """
    Calcula la entropía de entrelazamiento.

    E = S(ρ_A) = S(ρ_B)

    Para estados puros, la entropía de entrelazamiento
    mide las correlaciones cuánticas entre subsistemas.

    Args:
        psi: Vector de estado de 2 qubits

    Returns:
        Entropía de entrelazamiento en bits
    """
    rho = matriz_densidad(psi)
    rho_A = traza_parcial_B(rho)
    return entropia_von_neumann(rho_A)


def entropia_desde_epsilon(epsilon: float) -> float:
    """
    Calcula la entropía de entrelazamiento desde ε.

    E(ε) = -ε log₂(ε) - (1-ε) log₂(1-ε)

    Esta es la función de entropía binaria H(ε).

    Args:
        epsilon: Fracción entrópica ε ∈ [0, 1]

    Returns:
        Entropía en bits
    """
    if epsilon <= 0 or epsilon >= 1:
        return 0.0

    return -(epsilon * np.log2(epsilon) + (1 - epsilon) * np.log2(1 - epsilon))


# =============================================================================
# Clase Principal
# =============================================================================

@dataclass
class QubitTensorial:
    """
    Representa el estado cuántico tensorial |Φ(S)⟩.

    Encapsula la dualidad MASA-ESPACIO como un estado de 2 qubits
    entrelazados, proporcionando una validación cuántica del modelo MCMC.

    INTERPRETACIÓN:
        - Qubit A: Representa el aspecto "masa" del campo tensional
        - Qubit B: Representa el aspecto "espacio" del campo tensional
        - El entrelazamiento codifica la correlación ontológica

    Attributes:
        epsilon: Fracción entrópica ε ∈ [0, 1]
        estado: Vector de estado |Φ(ε)⟩
    """
    epsilon: float
    _estado: Optional[NDArray[np.complex128]] = field(default=None, repr=False)

    def __post_init__(self):
        """Valida parámetros y construye el estado."""
        if self.epsilon < 0 or self.epsilon > 1:
            raise ValueError(f"epsilon debe estar en [0, 1]")
        self._estado = estado_tensorial(self.epsilon)

    @property
    def estado(self) -> NDArray[np.complex128]:
        """Vector de estado |Φ(ε)⟩."""
        if self._estado is None:
            self._estado = estado_tensorial(self.epsilon)
        return self._estado

    @property
    def alpha(self) -> complex:
        """Amplitud de masa α = √(1-ε)."""
        return complex(np.sqrt(1 - self.epsilon))

    @property
    def beta(self) -> complex:
        """Amplitud de espacio β = √ε."""
        return complex(np.sqrt(self.epsilon))

    @property
    def P_ME_cuantico(self) -> float:
        """
        Polarización masa-espacio cuántica.

        P_ME = |α|² - |β|² = (1-ε) - ε = 1 - 2ε

        Coincide con P_ME = (Mp-Ep)/(Mp+Ep) del modelo clásico.
        """
        return 1 - 2 * self.epsilon

    @property
    def concurrencia(self) -> float:
        """Concurrencia del estado (entrelazamiento)."""
        return concurrencia_desde_epsilon(self.epsilon)

    @property
    def entropia_entrelazamiento(self) -> float:
        """Entropía de entrelazamiento en bits."""
        return entropia_desde_epsilon(self.epsilon)

    @property
    def rho(self) -> NDArray[np.complex128]:
        """Matriz de densidad del estado."""
        return matriz_densidad(self.estado)

    @property
    def rho_A(self) -> NDArray[np.complex128]:
        """Matriz de densidad reducida del subsistema A (masa)."""
        return traza_parcial_B(self.rho)

    @property
    def rho_B(self) -> NDArray[np.complex128]:
        """Matriz de densidad reducida del subsistema B (espacio)."""
        return traza_parcial_A(self.rho)

    def probabilidad_00(self) -> float:
        """Probabilidad de medir |00⟩ (estado de masa)."""
        return float(np.abs(self.alpha)**2)

    def probabilidad_11(self) -> float:
        """Probabilidad de medir |11⟩ (estado de espacio)."""
        return float(np.abs(self.beta)**2)

    def medir(self) -> str:
        """
        Simula una medición del estado tensorial.

        Returns:
            "00" o "11" según el resultado de la medición
        """
        p_00 = self.probabilidad_00()
        if np.random.random() < p_00:
            return "00"
        else:
            return "11"

    def fidelidad_con_bell(self) -> float:
        """
        Calcula la fidelidad con el estado de Bell |Φ+⟩.

        F = |⟨Φ+|Φ(ε)⟩|²

        F = 1 cuando ε = 0.5 (máximo entrelazamiento).
        """
        overlap = np.vdot(BELL_PHI_PLUS, self.estado)
        return float(np.abs(overlap)**2)

    def verificar_normalizacion(self) -> bool:
        """Verifica que el estado está normalizado."""
        norma = np.linalg.norm(self.estado)
        return np.isclose(norma, 1.0)

    def resumen(self) -> str:
        """Genera resumen del estado tensorial."""
        return (
            f"Qubit Tensorial MCMC\n"
            f"{'='*50}\n"
            f"ε = {self.epsilon:.6f}\n"
            f"{'='*50}\n"
            f"Amplitudes:\n"
            f"  α = {self.alpha:.6f} (masa)\n"
            f"  β = {self.beta:.6f} (espacio)\n"
            f"{'='*50}\n"
            f"Probabilidades:\n"
            f"  P(|00⟩) = {self.probabilidad_00():.6f}\n"
            f"  P(|11⟩) = {self.probabilidad_11():.6f}\n"
            f"{'='*50}\n"
            f"Entrelazamiento:\n"
            f"  Concurrencia C = {self.concurrencia:.6f}\n"
            f"  Entropía E = {self.entropia_entrelazamiento:.6f} bits\n"
            f"{'='*50}\n"
            f"Polarización P_ME = {self.P_ME_cuantico:+.6f}\n"
            f"Fidelidad con |Φ+⟩ = {self.fidelidad_con_bell():.6f}\n"
        )

    @classmethod
    def desde_P_ME(cls, P_ME: float) -> QubitTensorial:
        """
        Crea QubitTensorial desde la polarización P_ME.

        ε = (1 - P_ME) / 2

        Args:
            P_ME: Polarización en [-1, +1]

        Returns:
            QubitTensorial correspondiente
        """
        if P_ME < -1 or P_ME > 1:
            raise ValueError(f"P_ME debe estar en [-1, +1]")
        epsilon = (1 - P_ME) / 2
        return cls(epsilon=epsilon)

    @classmethod
    def primordial(cls) -> QubitTensorial:
        """Crea estado primordial (ε ≈ 0, masa pura)."""
        return cls(epsilon=1e-10)

    @classmethod
    def equilibrio(cls) -> QubitTensorial:
        """Crea estado de equilibrio (ε = 0.5, Bell)."""
        return cls(epsilon=0.5)

    @classmethod
    def final(cls) -> QubitTensorial:
        """Crea estado final (ε ≈ 1, espacio puro)."""
        return cls(epsilon=1.0 - 1e-10)


# =============================================================================
# Funciones de Análisis
# =============================================================================

def trayectoria_cuantica(
    n_puntos: int = 100
) -> Dict[str, NDArray[np.float64]]:
    """
    Calcula la trayectoria cuántica completa ε: 0 → 1.

    Returns:
        Diccionario con arrays para ε, C, E, P_ME
    """
    epsilon = np.linspace(0.001, 0.999, n_puntos)
    concurrencias = np.array([concurrencia_desde_epsilon(e) for e in epsilon])
    entropias = np.array([entropia_desde_epsilon(e) for e in epsilon])
    P_ME = 1 - 2 * epsilon

    return {
        "epsilon": epsilon,
        "concurrencia": concurrencias,
        "entropia": entropias,
        "P_ME": P_ME,
    }


def punto_maximo_entrelazamiento() -> Dict[str, float]:
    """
    Encuentra el punto de máximo entrelazamiento.

    Para el estado tensorial, C_max = 1 ocurre en ε = 0.5.
    """
    epsilon_max = 0.5
    return {
        "epsilon": epsilon_max,
        "concurrencia": concurrencia_desde_epsilon(epsilon_max),
        "entropia": entropia_desde_epsilon(epsilon_max),
        "P_ME": 0.0,  # Equilibrio tensional
    }


def verificar_consistencia_cuantica_clasica(epsilon: float, Mp: float, Ep: float) -> bool:
    """
    Verifica la consistencia entre descripciones cuántica y clásica.

    El modelo MCMC es autoconsistente si:
    1. P_ME_cuantico = P_ME_clasico
    2. ε = Ep / (Mp + Ep)

    Args:
        epsilon: Fracción entrópica
        Mp: Masa potencial
        Ep: Energía

    Returns:
        True si las descripciones son consistentes
    """
    # P_ME clásico
    P_ME_clasico = (Mp - Ep) / (Mp + Ep)

    # P_ME cuántico
    P_ME_cuantico = 1 - 2 * epsilon

    # ε desde Mp, Ep
    epsilon_calculado = Ep / (Mp + Ep)

    return (
        np.isclose(P_ME_clasico, P_ME_cuantico, atol=1e-6) and
        np.isclose(epsilon, epsilon_calculado, atol=1e-6)
    )


# =============================================================================
# Tests
# =============================================================================

def _test_qubit_tensorial():
    """Verifica la implementación del qubit tensorial."""

    # Test 1: Estado normalizado
    for eps in [0.0, 0.25, 0.5, 0.75, 1.0]:
        psi = estado_tensorial(eps)
        assert np.isclose(np.linalg.norm(psi), 1.0), f"Estado no normalizado para ε={eps}"

    # Test 2: Límites de concurrencia
    C_0 = concurrencia_desde_epsilon(0.0)
    C_05 = concurrencia_desde_epsilon(0.5)
    C_1 = concurrencia_desde_epsilon(1.0)
    assert np.isclose(C_0, 0.0), f"C(0) = {C_0} ≠ 0"
    assert np.isclose(C_05, 1.0), f"C(0.5) = {C_05} ≠ 1"
    assert np.isclose(C_1, 0.0), f"C(1) = {C_1} ≠ 0"

    # Test 3: P_ME cuántico
    qt = QubitTensorial(epsilon=0.3)
    assert np.isclose(qt.P_ME_cuantico, 1 - 2*0.3), "P_ME cuántico incorrecto"

    # Test 4: Probabilidades suman 1
    for eps in [0.1, 0.5, 0.9]:
        qt = QubitTensorial(epsilon=eps)
        p_total = qt.probabilidad_00() + qt.probabilidad_11()
        assert np.isclose(p_total, 1.0), f"Probabilidades no suman 1 para ε={eps}"

    # Test 5: Fidelidad máxima en equilibrio
    qt_eq = QubitTensorial.equilibrio()
    assert np.isclose(qt_eq.fidelidad_con_bell(), 1.0), "Fidelidad máxima esperada en equilibrio"

    # Test 6: Consistencia cuántica-clásica
    epsilon = 0.3
    Mp = 0.7
    Ep = 0.3
    assert verificar_consistencia_cuantica_clasica(epsilon, Mp, Ep), \
        "Inconsistencia cuántica-clásica"

    print("✓ Todos los tests del Qubit Tensorial pasaron")
    return True


if __name__ == "__main__":
    _test_qubit_tensorial()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Qubit Tensorial MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    # Estado primordial
    print("Estado Primordial (ε ≈ 0):")
    qt_prim = QubitTensorial.primordial()
    print(f"  P_ME = {qt_prim.P_ME_cuantico:+.6f}")
    print(f"  Concurrencia = {qt_prim.concurrencia:.6f}")
    print()

    # Estado de equilibrio
    print("Estado de Equilibrio (ε = 0.5):")
    qt_eq = QubitTensorial.equilibrio()
    print(f"  P_ME = {qt_eq.P_ME_cuantico:+.6f}")
    print(f"  Concurrencia = {qt_eq.concurrencia:.6f}")
    print(f"  Fidelidad con Bell = {qt_eq.fidelidad_con_bell():.6f}")
    print()

    # Trayectoria
    print("Trayectoria ε: 0 → 1")
    print("-"*50)
    for eps in [0.0, 0.25, 0.5, 0.75, 1.0]:
        qt = QubitTensorial(epsilon=max(0.001, min(0.999, eps)))
        print(f"ε = {eps:.2f}: P_ME = {qt.P_ME_cuantico:+.2f}, "
              f"C = {qt.concurrencia:.3f}, E = {qt.entropia_entrelazamiento:.3f} bits")
