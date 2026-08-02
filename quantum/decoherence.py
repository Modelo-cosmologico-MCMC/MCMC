"""La decoherencia como mini-colapso (v35, C.3 y C.5).

Disipación estructural tipo Lindblad (ec. C.4):

    dρ/dt = −i·[Ĥ, ρ] − λ_ten·[Φ_ten, [Φ_ten, ρ]]

con λ_ten la tasa de pérdida de coherencia del canal tensional. La
decoherencia no ocurre por medición sino por incompatibilidad
ontológica: el entorno resuelve la ambigüedad colapsando hacia el
estado más coherente con la geometría local.

Fidelidad: F(t) = ⟨Ψ0|ρ(t)|Ψ0⟩ ≈ exp(−∫ Γ_ont dt').

Firma falsable en hardware (ec. C.5):

    Γn/Γ0 = 1 + ξ_ten · η_void(Sn) · En/E4

si el hardware no mostrara ese exceso ontológico creciente con n, el
acoplo ξ_ten quedaría acotado superiormente.
"""

from __future__ import annotations

import numpy as np


def lindblad_rhs(rho: np.ndarray, H: np.ndarray, Phi_ten: np.ndarray,
                 lam_ten: float) -> np.ndarray:
    """Lado derecho de la ec. (C.4): −i[H,ρ] − λ_ten·[Φ,[Φ,ρ]]."""
    comm_H = H @ rho - rho @ H
    comm_P = Phi_ten @ rho - rho @ Phi_ten
    double = Phi_ten @ comm_P - comm_P @ Phi_ten
    return -1j * comm_H - lam_ten * double


def evolve(rho0: np.ndarray, H: np.ndarray, Phi_ten: np.ndarray,
           lam_ten: float, t_final: float, n_steps: int = 1000
           ) -> np.ndarray:
    """Integra (C.4) con RK4 de paso fijo. Devuelve ρ(t_final).

    Preserva traza y hermiticidad hasta el orden del integrador.
    """
    rho = np.asarray(rho0, dtype=complex).copy()
    dt = t_final / n_steps
    for _ in range(n_steps):
        k1 = lindblad_rhs(rho, H, Phi_ten, lam_ten)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, Phi_ten, lam_ten)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, Phi_ten, lam_ten)
        k4 = lindblad_rhs(rho + dt * k3, H, Phi_ten, lam_ten)
        rho = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        rho = 0.5 * (rho + rho.conj().T)  # re-hermitizar (error numérico)
    return rho


def fidelity_with(rho: np.ndarray, psi0: np.ndarray) -> float:
    """F = ⟨Ψ0|ρ|Ψ0⟩ (C.3)."""
    psi0 = np.asarray(psi0, dtype=complex)
    return float(np.real(psi0.conj() @ rho @ psi0))


def gamma_ratio(n: int, xi_ten: float, eta_void_n: float,
                E_n: float, E_4: float) -> float:
    """Γn/Γ0 = 1 + ξ_ten·η_void(Sn)·En/E4 — firma falsable (ec. C.5).

    Con ξ_ten = 0 no hay exceso ontológico y todas las tasas coinciden:
    la ausencia del patrón en hardware acota ξ_ten superiormente.
    """
    return 1.0 + xi_ten * eta_void_n * (E_n / E_4)
