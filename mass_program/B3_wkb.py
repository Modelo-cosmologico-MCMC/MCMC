"""B3 — WKB espinorial.

Tres correcciones críticas (frente a un WKB ingenuo):

C1 — Coeficientes individuales:
    Los |T_n^(i)| son transmisiones INDIVIDUALES, no acumuladas. Si fueran
    acumuladas, |T_2^(mu)| = 1.000/4.3e-4 ≈ 2326 > 1, imposible.

C2 — WKB en espacio-S adimensional:
    m_P(S) ∈ [0,1]. El WKB opera sobre esa escala normalizada, no en GeV.

C3 — Túnel secuencial pre-emergencia:
    |T_n^(i)|_pre = Π_{k=n}^{n_e(i)-1} exp(-κ_k · ΔS_k / λ)
    κ_k = sqrt(m_P(S_k)^2 - E_{F_{k+1}}^2)

Calibración (Tratado, Tabla 3):
    κ_gap_12 = -ln(4.3e-4) / (0.090/0.01) ≈ 0.8613
    κ_gap_23 = -ln(2.1e-3) / (0.900/0.01) ≈ 0.0685
    E_F2 = sqrt(m_P(C1)^2 - κ_12^2) ≈ 0.4881
    E_F3 = sqrt(m_P(C2)^2 - κ_23^2) ≈ 0.9431
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from mcmc_ontology.S_map import m_P


_SEALS = ("C1", "C2", "C3", "C4")
_FAMILY_EMERGENCE_INDEX = {"F1": 0, "F2": 1, "F3": 2}  # índice 0-based en _SEALS


def kappa_gap_12() -> float:
    """κ del gap C1→C2 (calibrado por |T_1^(F2)| = 4.3e-4)."""
    dS = C.S_SEALS["C2"] - C.S_SEALS["C1"]
    return -np.log(C.T_UNIVERSAL["F2"][0]) / (dS / C.LAMBDA_ONT)


def kappa_gap_23() -> float:
    """κ del gap C2→C3 (calibrado por |T_2^(F3)| = 2.1e-3)."""
    dS = C.S_SEALS["C3"] - C.S_SEALS["C2"]
    return -np.log(C.T_UNIVERSAL["F3"][1]) / (dS / C.LAMBDA_ONT)


def E_F2() -> float:
    """Eigenvalor Dirac de F2: sqrt(m_P(C1)^2 - κ_12^2)."""
    return float(np.sqrt(m_P(C.S_SEALS["C1"]) ** 2 - kappa_gap_12() ** 2))


def E_F3() -> float:
    """Eigenvalor Dirac de F3: sqrt(m_P(C2)^2 - κ_23^2)."""
    return float(np.sqrt(m_P(C.S_SEALS["C2"]) ** 2 - kappa_gap_23() ** 2))


def transmission(family: str, seal: str) -> float:
    """|T_n^(family)| individual (Tabla 3 del Tratado).

    Devuelve directamente la tabla de transmisiones tabulada (universal,
    sin parámetros libres adicionales una vez calibrados κ y E_F).
    """
    n = _SEALS.index(seal)
    return C.T_UNIVERSAL[family][n]


def transmission_table() -> np.ndarray:
    """Tabla 3x4: filas = F1,F2,F3; columnas = C1..C4."""
    return np.array([C.T_UNIVERSAL[f] for f in ("F1", "F2", "F3")])


def sequential_tunnel(family: str, seal: str) -> float:
    """Túnel secuencial pre-emergencia (corrección C3).

    Aplica solo cuando el sello es anterior al sello de emergencia.
    Producto de exp(-κ_k · ΔS_k / λ) para k desde el sello dado hasta
    el sello justo antes de la emergencia.
    """
    n_seal = _SEALS.index(seal)
    n_emerge = _FAMILY_EMERGENCE_INDEX[family]
    if n_seal >= n_emerge:
        return 1.0
    kappas = [kappa_gap_12(), kappa_gap_23()]  # gap C1→C2, C2→C3
    dS = [C.S_SEALS[_SEALS[k + 1]] - C.S_SEALS[_SEALS[k]] for k in range(3)]
    out = 1.0
    for k in range(n_seal, n_emerge):
        out *= float(np.exp(-kappas[k] * dS[k] / C.LAMBDA_ONT))
    return out


def verify_calibration(tol: float = 1e-3) -> dict:
    """Verifica los parámetros WKB calibrados frente a los valores nominales."""
    return {
        "kappa_12": (kappa_gap_12(), C.KAPPA_GAP_12,
                     abs(kappa_gap_12() - C.KAPPA_GAP_12) < tol),
        "kappa_23": (kappa_gap_23(), C.KAPPA_GAP_23,
                     abs(kappa_gap_23() - C.KAPPA_GAP_23) < tol),
        "E_F2":     (E_F2(), C.E_F2, abs(E_F2() - C.E_F2) < tol),
        "E_F3":     (E_F3(), C.E_F3, abs(E_F3() - C.E_F3) < tol),
    }
