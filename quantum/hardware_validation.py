"""III.I — Validación del qudit MCMC en hardware real.

Predicciones de fidelidad por plataforma (Tratado §3.20):
    IonQ Aria   (171Yb+, 5 niveles): F_global ≈ 0.82
    IBM Eagle   (3 transmones):       F_global ≈ 0.75
    Quantinuum H1-2:                  F_global ≈ 0.88

Métricas MDR a verificar:
    d_MDR(n) ∝ n^0.95   (métrica dual relativa)
    C_entanglement ≈ 0.65
"""

from __future__ import annotations

import numpy as np

from quantum.qudit import D, basis
from quantum.qiskit_circuit import collapse_unitary_matrix


HARDWARE_PROFILES = {
    "IonQ_Aria": {
        "platform":     "5 niveles en 171Yb+",
        "T_circuit_us": 40.0,
        "F_2Q_gate":    0.96,
        "F_global_pred": 0.82,
    },
    "IBM_Eagle": {
        "platform":     "3 qubits transmón",
        "T_circuit_us": 8.0,
        "F_2Q_gate":    0.995,
        "F_global_pred": 0.75,
    },
    "Quantinuum_H12": {
        "platform":     "H1-2 trapped ion",
        "T_circuit_us": 30.0,
        "F_2Q_gate":    0.999,
        "F_global_pred": 0.88,
    },
}


def fidelity_from_density_matrix(rho_meas: np.ndarray, psi_ideal: np.ndarray) -> float:
    """Fidelidad F = ⟨ψ|ρ|ψ⟩ (Uhlmann simplificada para estado puro objetivo)."""
    rho = np.asarray(rho_meas)
    psi = np.asarray(psi_ideal)
    return float(np.real(np.vdot(psi, rho @ psi)))


def predict_global_fidelity(F_gate: float, n_gates: int = 4) -> float:
    """Aproximación de fidelidad global encadenando n compuertas independientes.

        F_global ≈ F_gate^n_gates
    """
    return float(F_gate ** n_gates)


def simulate_with_noise(F_gate: float = 0.96, n_shots: int = 10_000,
                        seed: int = 0) -> dict:
    """Simulación clásica de la cadena S0→S1→S2→S3→S4 con ruido por compuerta.

    No requiere Qiskit; aplica un modelo depolarizing por compuerta sobre
    el estado puro inicial |S0⟩.
    """
    rng = np.random.default_rng(seed)
    state = basis(0)
    fidelities = []
    for n in range(D - 1):
        U = collapse_unitary_matrix(n)
        # Aplica unitaria sobre el subespacio computacional (5 dim → 8 dim)
        s8 = np.zeros(8, dtype=complex)
        s8[:D] = state
        s8 = U @ s8
        ideal = s8.copy()
        # Modelo depolarizing: con probabilidad (1-F_gate) reemplaza por estado max-mixed
        if rng.random() > F_gate:
            s8 = rng.normal(size=8) + 1j * rng.normal(size=8)
            s8 /= np.linalg.norm(s8)
        # Devuelve al subespacio físico
        state = s8[:D]
        state /= max(np.linalg.norm(state), 1e-30)
        f = float(abs(np.vdot(ideal[:D], state)) ** 2)
        fidelities.append(f)
    return {
        "transition_fidelities": fidelities,
        "F_global":              float(np.prod(fidelities)),
        "n_shots":               n_shots,
    }


def go_no_go(F_global: float) -> str:
    """Criterios go/no-go del Tratado:
        F_global > 0.70 → GO
        0.60 ≤ F_global ≤ 0.70 → MARGINAL
        F_global < 0.60 → NO-GO
    """
    if F_global > 0.70:
        return "GO"
    if F_global >= 0.60:
        return "MARGINAL"
    return "NO-GO"
