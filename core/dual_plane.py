"""El Plano Dual — el escenario de la unidad dual (v35, Cap. 2).

Realiza el Axioma 1. Definición 2.1: el espacio interno es D ≅ R² con
coordenadas (φM, φE) — las proyecciones efectivas de Masa Primordial y
Espacio Primordial — restringido al primer cuadrante (dominio físico).
Coordenadas derivadas (ec. 2.1):

    ρ² ≡ φM² + φE²             módulo tensional
    θ  ≡ arctan(φE/φM) ∈ [0, π/2]   fase de conversión
    χ  ≡ (φM − φE)/√2          modo de desequilibrio
    ς  ≡ (φM + φE)/√2          modo de carga

Definición 2.2: el intercambio Mp ↔ Ep es la reflexión Z₂ (φM ↔ φE,
equivalentemente χ → −χ, θ → π/2 − θ); su lugar fijo θ = π/4 (χ = 0)
es la DIAGONAL DUAL, el equilibrio Mp = Ep, que la trayectoria del
universo cruza en S ≃ 1 (Prop. 3.5).
"""

from __future__ import annotations

import numpy as np

THETA_DIAGONAL = np.pi / 4.0  # la diagonal dual (Def. 2.2)


def to_dual(phi_M: np.ndarray | float, phi_E: np.ndarray | float) -> dict:
    """(φM, φE) → coordenadas derivadas (ρ, θ, χ, ς) de la ec. (2.1)."""
    phi_M = np.asarray(phi_M, dtype=float)
    phi_E = np.asarray(phi_E, dtype=float)
    rho = np.sqrt(phi_M ** 2 + phi_E ** 2)
    theta = np.arctan2(phi_E, phi_M)
    chi = (phi_M - phi_E) / np.sqrt(2.0)
    sigma = (phi_M + phi_E) / np.sqrt(2.0)
    return {"rho": rho, "theta": theta, "chi": chi, "sigma": sigma}


def from_polar(rho: np.ndarray | float,
               theta: np.ndarray | float) -> tuple:
    """(ρ, θ) → (φM, φE)."""
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return rho * np.cos(theta), rho * np.sin(theta)


def dual_reflection(phi_M: np.ndarray | float,
                    phi_E: np.ndarray | float) -> tuple:
    """La reflexión Z₂ del intercambio Mp ↔ Ep (Def. 2.2)."""
    return phi_E, phi_M


def is_physical(phi_M: np.ndarray | float,
                phi_E: np.ndarray | float) -> np.ndarray | bool:
    """Dominio físico: primer cuadrante, φM ≥ 0 y φE ≥ 0."""
    out = (np.asarray(phi_M) >= 0.0) & (np.asarray(phi_E) >= 0.0)
    return bool(out) if np.ndim(out) == 0 else out


def on_diagonal(phi_M: float, phi_E: float, tol: float = 1e-12) -> bool:
    """¿Está el punto en la diagonal dual (χ = 0, equilibrio Mp = Ep)?"""
    return abs(to_dual(phi_M, phi_E)["chi"]) < tol
