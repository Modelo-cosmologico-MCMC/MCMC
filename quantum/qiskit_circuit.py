"""III.I — Conversión qudit d=5 → circuito de qubits para Qiskit.

El qudit d=5 (con base ontológica {|S0⟩..|S4⟩}) se codifica en
n_qubits = ⌈log₂(5)⌉ = 3 qubits con asignación binaria:

    |S0⟩ ↔ |000⟩
    |S1⟩ ↔ |001⟩
    |S2⟩ ↔ |010⟩
    |S3⟩ ↔ |011⟩
    |S4⟩ ↔ |100⟩

Las compuertas de colapso X_{n→n+1} se implementan como rotaciones
controladas. Si Qiskit no está instalado, el módulo expone la
representación matricial.
"""

from __future__ import annotations

import numpy as np


N_QUBITS = 3
DIM = 5


def basis_index_to_binary(n: int) -> str:
    """Codificación de |Sn⟩ en cadena binaria de 3 bits."""
    return format(n, f"0{N_QUBITS}b")


def collapse_unitary_matrix(n: int) -> np.ndarray:
    """Matriz unitaria 2^N × 2^N que implementa X_{n→n+1} en el subespacio físico.

    Fuera del subespacio {|S_n⟩, |S_{n+1}⟩} actúa como identidad.
    """
    if not 0 <= n < DIM - 1:
        raise ValueError(f"n fuera de rango: {n}")
    N = 2 ** N_QUBITS
    U = np.eye(N, dtype=complex)
    i, j = n, n + 1
    U[i, i] = 0.0; U[j, j] = 0.0
    U[i, j] = 1.0; U[j, i] = 1.0
    return U


def qudit_to_qubit_state(coeffs: np.ndarray) -> np.ndarray:
    """Codifica un estado qudit (5 coeficientes) en un statevector de 8 elementos."""
    if len(coeffs) != DIM:
        raise ValueError(f"Se esperan {DIM} coeficientes")
    psi = np.zeros(2 ** N_QUBITS, dtype=complex)
    psi[:DIM] = np.asarray(coeffs, dtype=complex)
    norm = np.linalg.norm(psi)
    return psi / norm if norm > 0 else psi


def build_qiskit_circuit():
    """Construye el circuito Qiskit equivalente.

    Si Qiskit no está disponible, lanza ImportError con instrucciones.
    """
    try:
        from qiskit import QuantumCircuit  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Qiskit no está instalado. `pip install qiskit qiskit-aer` "
            "o usa `collapse_unitary_matrix` en su lugar."
        ) from exc
    qc = QuantumCircuit(N_QUBITS)
    # Aplica las 4 compuertas de colapso X_{n→n+1} en orden.
    for n in range(DIM - 1):
        U = collapse_unitary_matrix(n)
        qc.unitary(U, list(range(N_QUBITS)), label=f"X_{n}_to_{n+1}")
    return qc
