"""
Circuito Cuántico - Validación Experimental del Qubit Tensorial
================================================================

Implementación del circuito cuántico mínimo para validar el modelo MCMC
en hardware cuántico real (IonQ, IBM, etc.).

CIRCUITO MÍNIMO:
        ┌─────────┐
    |q₀⟩───┤ RY(θ)  ├───●───[M]
        └─────────┘   │
                      │
    |q₁⟩─────────────⊕───[M]

    donde θ = 2 × arccos(√ε) = 2.9299 rad

ESTADO PREPARADO:
    |Φ(S₄)⟩ = √ε |00⟩ + √(1-ε) |11⟩
            = 0.106 |00⟩ + 0.994 |11⟩

OBSERVABLES A MEDIR:
    - P_ME = ⟨Z⊗I⟩ = |α|² - |β|² = -0.9777
    - ZZ = ⟨Z⊗Z⟩ = 1.0 (correlación perfecta)
    - XX = ⟨X⊗X⟩ (coherencia)
    - YY = ⟨Y⊗Y⟩ (fase)
    - W(Bell) = 1/4 - (1 + ZZ + XX - YY)/4 < 0

PLATAFORMAS COMPATIBLES:
    - IonQ Aria (óptimo): T₂ = 100ms, error 2q = 0.1%
    - IBM Brisbane (viable): T₂ = 80μs, error 2q = 1%
    - Rigetti Aspen (marginal)
    - Google Sycamore (marginal)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from numpy.typing import NDArray


# =============================================================================
# Constantes del Circuito
# =============================================================================

# Fracción residual de masa (calibrada)
EPSILON: float = 0.0112

# Ángulo de rotación RY
THETA_RY: float = 2 * np.arccos(np.sqrt(EPSILON))  # ≈ 2.9299 rad

# Valores esperados del estado |Φ(S₄)⟩
P_ME_ESPERADO: float = 2 * EPSILON - 1  # ≈ -0.9777
ZZ_ESPERADO: float = 1.0
W_BELL_ESPERADO: float = -0.355  # Negativo = entrelazado

# Matrices de Pauli
PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# =============================================================================
# Generación del Circuito (Compatible con Qiskit)
# =============================================================================

def generar_codigo_qiskit(epsilon: float = EPSILON) -> str:
    """
    Genera código Qiskit para preparar el estado |Φ(S)⟩.

    Args:
        epsilon: Fracción de masa (default: ε calibrado)

    Returns:
        Código Python como string
    """
    theta = 2 * np.arccos(np.sqrt(epsilon))

    codigo = f'''"""
Circuito Qubit Tensorial MCMC
Prepara |Φ(S₄)⟩ = √ε |00⟩ + √(1-ε) |11⟩
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# Parámetros calibrados
epsilon = {epsilon}
theta = 2 * np.arccos(np.sqrt(epsilon))  # = {theta:.6f} rad

# Crear circuito
qc = QuantumCircuit(2, 2)
qc.ry(theta, 0)    # Rotación RY calibrada
qc.cx(0, 1)        # CNOT para entrelazamiento
qc.measure([0, 1], [0, 1])

# Simular
simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled, shots=8192).result()
counts = result.get_counts()

# Calcular observables
n_00 = counts.get('00', 0)
n_01 = counts.get('01', 0)
n_10 = counts.get('10', 0)
n_11 = counts.get('11', 0)
total = n_00 + n_01 + n_10 + n_11

P_ME = (n_00 + n_01 - n_10 - n_11) / total  # ⟨Z⊗I⟩
ZZ = (n_00 - n_01 - n_10 + n_11) / total    # ⟨Z⊗Z⟩

print(f"Resultados del Qubit Tensorial MCMC:")
print(f"  P_ME = {{P_ME:+.4f}} (esperado: {P_ME_ESPERADO:+.4f})")
print(f"  ZZ = {{ZZ:.4f}} (esperado: {ZZ_ESPERADO:.4f})")
print(f"  Estado preparado: |Φ(S₄)⟩ = 0.106|00⟩ + 0.994|11⟩")
'''
    return codigo


def circuito_como_matriz() -> Dict[str, NDArray]:
    """
    Genera las matrices del circuito.

    Returns:
        Diccionario con matrices RY, CNOT, U_total
    """
    # Gate RY(θ)
    c = np.cos(THETA_RY / 2)
    s = np.sin(THETA_RY / 2)
    RY = np.array([[c, -s], [s, c]], dtype=np.complex128)

    # Gate CNOT
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)

    # RY ⊗ I
    RY_I = np.kron(RY, PAULI_I)

    # Circuito total: CNOT @ (RY ⊗ I)
    U_total = CNOT @ RY_I

    return {
        "RY": RY,
        "CNOT": CNOT,
        "RY_I": RY_I,
        "U_total": U_total,
    }


def preparar_estado(epsilon: float = EPSILON) -> NDArray[np.complex128]:
    """
    Prepara el estado |Φ(ε)⟩ = √ε |00⟩ + √(1-ε) |11⟩.

    Args:
        epsilon: Fracción de masa

    Returns:
        Vector de estado (4 componentes)
    """
    alpha = np.sqrt(epsilon)
    beta = np.sqrt(1 - epsilon)

    # |00⟩ + |11⟩ base computacional
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = alpha  # |00⟩
    psi[3] = beta   # |11⟩

    return psi


# =============================================================================
# Medición de Observables
# =============================================================================

def valor_esperado(
    psi: NDArray[np.complex128],
    operador: NDArray[np.complex128]
) -> float:
    """
    Calcula ⟨ψ|O|ψ⟩.

    Args:
        psi: Vector de estado
        operador: Matriz del operador

    Returns:
        Valor esperado (real)
    """
    return float(np.real(np.vdot(psi, operador @ psi)))


def medir_P_ME(psi: NDArray[np.complex128]) -> float:
    """
    Mide P_ME = ⟨Z⊗I⟩.

    P_ME = P(|00⟩) + P(|01⟩) - P(|10⟩) - P(|11⟩)
         = |α|² - |β|² = 2ε - 1

    Args:
        psi: Vector de estado

    Returns:
        P_ME ∈ [-1, +1]
    """
    ZI = np.kron(PAULI_Z, PAULI_I)
    return valor_esperado(psi, ZI)


def medir_ZZ(psi: NDArray[np.complex128]) -> float:
    """
    Mide ⟨Z⊗Z⟩.

    Para |Φ⟩ = α|00⟩ + β|11⟩:
    ZZ = |α|² + |β|² = 1 (correlación perfecta)

    Args:
        psi: Vector de estado

    Returns:
        ZZ ∈ [-1, +1]
    """
    ZZ_op = np.kron(PAULI_Z, PAULI_Z)
    return valor_esperado(psi, ZZ_op)


def medir_XX(psi: NDArray[np.complex128]) -> float:
    """Mide ⟨X⊗X⟩."""
    XX_op = np.kron(PAULI_X, PAULI_X)
    return valor_esperado(psi, XX_op)


def medir_YY(psi: NDArray[np.complex128]) -> float:
    """Mide ⟨Y⊗Y⟩."""
    YY_op = np.kron(PAULI_Y, PAULI_Y)
    return valor_esperado(psi, YY_op)


def testigo_bell(psi: NDArray[np.complex128]) -> float:
    """
    Calcula el testigo de Bell.

    W(|Φ⁺⟩) = 1/4 - (1 + ZZ + XX - YY)/4

    W < 0 ⟹ Estado entrelazado genuino

    Args:
        psi: Vector de estado

    Returns:
        W (negativo si entrelazado)
    """
    ZZ = medir_ZZ(psi)
    XX = medir_XX(psi)
    YY = medir_YY(psi)

    return 0.25 - (1 + ZZ + XX - YY) / 4


def violacion_CHSH(psi: NDArray[np.complex128]) -> float:
    """
    Calcula el parámetro S de Bell-CHSH.

    S = |⟨ZZ⟩ + ⟨XX⟩| + |⟨ZZ⟩ - ⟨YY⟩|

    S > 2 ⟹ Violación de Bell (no-clásico)
    S_max = 2√2 ≈ 2.83 para estados de Bell

    Args:
        psi: Vector de estado

    Returns:
        S ∈ [0, 2√2]
    """
    ZZ = medir_ZZ(psi)
    XX = medir_XX(psi)
    YY = medir_YY(psi)

    return abs(ZZ + XX) + abs(ZZ - YY)


# =============================================================================
# Simulación de Medición con Ruido
# =============================================================================

def simular_medicion(
    psi: NDArray[np.complex128],
    shots: int = 8192,
    error_readout: float = 0.0
) -> Dict[str, int]:
    """
    Simula mediciones del estado.

    Args:
        psi: Vector de estado
        shots: Número de disparos
        error_readout: Error de lectura (0-1)

    Returns:
        Diccionario de conteos {'00': n00, '01': n01, ...}
    """
    probs = np.abs(psi) ** 2

    # Añadir error de lectura
    if error_readout > 0:
        probs = (1 - error_readout) * probs + error_readout / 4

    # Muestrear
    outcomes = np.random.choice(4, size=shots, p=probs)

    labels = ['00', '01', '10', '11']
    counts = {label: 0 for label in labels}

    for o in outcomes:
        counts[labels[o]] += 1

    return counts


def analizar_conteos(counts: Dict[str, int]) -> Dict[str, float]:
    """
    Analiza conteos de medición.

    Args:
        counts: Diccionario de conteos

    Returns:
        Diccionario con observables calculados
    """
    n00 = counts.get('00', 0)
    n01 = counts.get('01', 0)
    n10 = counts.get('10', 0)
    n11 = counts.get('11', 0)
    total = n00 + n01 + n10 + n11

    if total == 0:
        return {"error": "No hay conteos"}

    P_ME = (n00 + n01 - n10 - n11) / total
    ZZ = (n00 - n01 - n10 + n11) / total

    return {
        "P_ME": P_ME,
        "ZZ": ZZ,
        "P_00": n00 / total,
        "P_11": n11 / total,
        "P_01": n01 / total,
        "P_10": n10 / total,
    }


# =============================================================================
# Clase Principal: CircuitoTensorial
# =============================================================================

@dataclass
class CircuitoTensorial:
    """
    Circuito cuántico del Qubit Tensorial MCMC.

    Implementa el circuito mínimo para validar el modelo
    en hardware cuántico real.

    ARQUITECTURA:
        |00⟩ ─[RY(θ)]─[CNOT]─ |Φ(ε)⟩

    OBSERVABLES:
        - P_ME = ⟨Z⊗I⟩
        - ZZ = ⟨Z⊗Z⟩ = 1.0 (correlación perfecta)
        - W(Bell) < 0 (entrelazamiento)

    Attributes:
        epsilon: Fracción de masa (calibrada)
        theta: Ángulo de rotación RY
    """
    epsilon: float = EPSILON
    theta: float = field(init=False)
    _estado: Optional[NDArray[np.complex128]] = field(default=None, repr=False)

    def __post_init__(self):
        """Inicializa el circuito."""
        self.theta = 2 * np.arccos(np.sqrt(self.epsilon))
        self._estado = preparar_estado(self.epsilon)

    @property
    def estado(self) -> NDArray[np.complex128]:
        """Estado cuántico preparado."""
        if self._estado is None:
            self._estado = preparar_estado(self.epsilon)
        return self._estado

    @property
    def alpha(self) -> float:
        """Amplitud de masa √ε."""
        return np.sqrt(self.epsilon)

    @property
    def beta(self) -> float:
        """Amplitud de espacio √(1-ε)."""
        return np.sqrt(1 - self.epsilon)

    def P_ME(self) -> float:
        """Polarización masa-espacio."""
        return medir_P_ME(self.estado)

    def ZZ(self) -> float:
        """Correlación Z⊗Z."""
        return medir_ZZ(self.estado)

    def XX(self) -> float:
        """Coherencia X⊗X."""
        return medir_XX(self.estado)

    def YY(self) -> float:
        """Fase Y⊗Y."""
        return medir_YY(self.estado)

    def testigo_bell(self) -> float:
        """Testigo de Bell."""
        return testigo_bell(self.estado)

    def violacion_CHSH(self) -> float:
        """Parámetro S de Bell-CHSH."""
        return violacion_CHSH(self.estado)

    def es_entrelazado(self) -> bool:
        """Determina si el estado está entrelazado."""
        return self.testigo_bell() < 0

    def simular(self, shots: int = 8192, error: float = 0.0) -> Dict[str, float]:
        """
        Simula mediciones.

        Args:
            shots: Número de disparos
            error: Error de lectura

        Returns:
            Diccionario con observables
        """
        counts = simular_medicion(self.estado, shots, error)
        return analizar_conteos(counts)

    def codigo_qiskit(self) -> str:
        """Genera código Qiskit."""
        return generar_codigo_qiskit(self.epsilon)

    def matrices(self) -> Dict[str, NDArray]:
        """Matrices del circuito."""
        return circuito_como_matriz()

    def verificar_criterios(self) -> Dict[str, bool]:
        """
        Verifica criterios de éxito.

        Returns:
            Diccionario con resultados de cada criterio
        """
        P = self.P_ME()
        ZZ = self.ZZ()
        W = self.testigo_bell()

        return {
            "P_ME_en_rango": -1.0 <= P <= -0.9,
            "ZZ_mayor_07": ZZ > 0.7,
            "Bell_negativo": W < 0,
            "todo_ok": (-1.0 <= P <= -0.9) and (ZZ > 0.7) and (W < 0),
        }

    def resumen(self) -> str:
        """Genera resumen del circuito."""
        crit = self.verificar_criterios()

        return (
            f"Circuito Qubit Tensorial MCMC\n"
            f"{'='*55}\n"
            f"Parámetros:\n"
            f"  ε = {self.epsilon:.6f}\n"
            f"  θ_RY = {self.theta:.6f} rad ({np.degrees(self.theta):.2f}°)\n"
            f"  α = {self.alpha:.6f} (masa)\n"
            f"  β = {self.beta:.6f} (espacio)\n"
            f"{'='*55}\n"
            f"Estado: |Φ⟩ = {self.alpha:.3f}|00⟩ + {self.beta:.3f}|11⟩\n"
            f"{'='*55}\n"
            f"Observables:\n"
            f"  P_ME = {self.P_ME():+.6f} (esperado: {P_ME_ESPERADO:+.6f})\n"
            f"  ZZ = {self.ZZ():.6f} (esperado: {ZZ_ESPERADO:.6f})\n"
            f"  XX = {self.XX():.6f}\n"
            f"  YY = {self.YY():.6f}\n"
            f"{'='*55}\n"
            f"Entrelazamiento:\n"
            f"  W(Bell) = {self.testigo_bell():.6f}\n"
            f"  S_CHSH = {self.violacion_CHSH():.6f}\n"
            f"  Entrelazado: {'Sí' if self.es_entrelazado() else 'No'}\n"
            f"{'='*55}\n"
            f"Criterios de éxito:\n"
            f"  P_ME ∈ [-1, -0.9]: {'✓' if crit['P_ME_en_rango'] else '✗'}\n"
            f"  ZZ > 0.7: {'✓' if crit['ZZ_mayor_07'] else '✗'}\n"
            f"  W(Bell) < 0: {'✓' if crit['Bell_negativo'] else '✗'}\n"
        )


# =============================================================================
# Evolución de Estados (10 Colapsos)
# =============================================================================

def evolucionar_circuito(n_pasos: int = 11) -> List[Dict]:
    """
    Evoluciona el circuito desde S₀ hasta S₄.

    Args:
        n_pasos: Número de pasos

    Returns:
        Lista de estados con observables
    """
    from .fase_pregeometrica import FasePregeometrica

    fp = FasePregeometrica()
    resultados = []

    for estado in fp.secuencia_colapsos(n_pasos):
        eps = 1 - estado['|c1|^2']  # Mp = 1 - Ep
        if eps < 1e-10:
            eps = 1e-10
        if eps > 1 - 1e-10:
            eps = 1 - 1e-10

        ct = CircuitoTensorial(epsilon=eps)

        resultados.append({
            "paso": estado['paso'],
            "S": estado['S'],
            "|c0|^2": ct.alpha ** 2,
            "|c1|^2": ct.beta ** 2,
            "P_ME": ct.P_ME(),
            "ZZ": ct.ZZ(),
            "S_ent": estado['S_ent'],
            "W_Bell": ct.testigo_bell(),
        })

    return resultados


# =============================================================================
# Tests
# =============================================================================

def _test_circuito_tensorial():
    """Verifica la implementación del circuito."""

    ct = CircuitoTensorial()

    # Test 1: Estado normalizado
    norma = np.linalg.norm(ct.estado)
    assert np.isclose(norma, 1.0), f"Estado no normalizado: |ψ| = {norma}"

    # Test 2: P_ME ≈ -0.978
    P = ct.P_ME()
    assert -1.0 <= P <= -0.9, f"P_ME = {P} fuera de rango"

    # Test 3: ZZ = 1.0
    ZZ = ct.ZZ()
    assert np.isclose(ZZ, 1.0), f"ZZ = {ZZ} ≠ 1"

    # Test 4: Estado entrelazado
    assert ct.es_entrelazado(), "Estado debe estar entrelazado"

    # Test 5: Violación de Bell
    S = ct.violacion_CHSH()
    assert S > 2.0, f"S_CHSH = {S} no viola límite clásico"

    # Test 6: Criterios de éxito
    crit = ct.verificar_criterios()
    assert crit['todo_ok'], "No se cumplen todos los criterios"

    print("✓ Todos los tests del CircuitoTensorial pasaron")
    return True


if __name__ == "__main__":
    _test_circuito_tensorial()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Circuito Cuántico MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    ct = CircuitoTensorial()
    print(ct.resumen())

    print("\n\nCódigo Qiskit generado:")
    print("-"*60)
    print(ct.codigo_qiskit()[:500] + "...")

    print("\n\nSimulación con 8192 shots:")
    print("-"*40)
    sim = ct.simular(shots=8192)
    print(f"  P(|00⟩) = {sim['P_00']:.4f}")
    print(f"  P(|11⟩) = {sim['P_11']:.4f}")
    print(f"  P_ME = {sim['P_ME']:+.4f}")
    print(f"  ZZ = {sim['ZZ']:.4f}")
