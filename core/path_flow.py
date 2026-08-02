"""El Flujo del Camino — la dinámica fundamental (v35, Cap. 4).

Realiza el Axioma 4. Definición 4.4 (ec. 4.2): la dinámica pre-geométrica
es el flujo de gradiente disipativo

    dΦAd/dσ = −G⁻¹(S)·δV/δΦAd,     G ≻ 0 (rigidez)

sobre el Plano Dual con el Potencial Basal. La ecuación de segundo orden
del corpus queda descartada como dinámica física por ser reversible
(contradiría el Axioma 4).

Teoremas que los tests verifican numéricamente:
- Monotonía del Camino (Teo. 4.5): dV/dσ = −(∇V)ᵀG⁻¹(∇V) ≤ 0; la
  producción entrópica (∇V)ᵀG⁻¹∇V ≥ 0. «La flecha del tiempo se
  demuestra, no se postula.»
- Exclusión del Camino (Lema 4.7): un flujo de gradiente con rigidez
  positiva no admite órbitas periódicas — el Camino no vuelve.
- Salida de S0 (Prop. 3.5): el máximo de χ sobre el anillo ρ = ρ+ se
  alcanza en φE = 0 — la salida ocurre hacia el POLO DE MASA (θ ≈ 0);
  con Γ0(0) = 0: en δ0 = 0 no hay salida (inercia eterna del perfecto).

El dominio físico es el primer cuadrante (Cap. 2); el flujo se proyecta
sobre él (descenso de gradiente proyectado).
"""

from __future__ import annotations

import numpy as np

from .basal import B_BAR, C0_DEFAULT, E_BAR, M_BAR, scaled_params


def grad_V(phi: np.ndarray, delta0: float, m_bar: float = M_BAR,
           b_bar: float = B_BAR, e_bar: float = E_BAR,
           C0: float = C0_DEFAULT) -> np.ndarray:
    """∇V0 en coordenadas (φM, φE), analítico.

    Con x = φM²+φE² y χ = (φM−φE)/√2:
        ∂V/∂φM = (M0² − B·x + C0·x²)·φM − η/√2
        ∂V/∂φE = (M0² − B·x + C0·x²)·φE + η/√2
    """
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    phi_M, phi_E = float(phi[0]), float(phi[1])
    x = phi_M ** 2 + phi_E ** 2
    radial = p["M0_sq"] - p["B"] * x + C0 * x ** 2
    return np.array([radial * phi_M - p["eta"] / np.sqrt(2.0),
                     radial * phi_E + p["eta"] / np.sqrt(2.0)])


def flow(phi0: np.ndarray, delta0: float, *, G: np.ndarray | None = None,
         d_sigma: float = 1e-3, n_steps: int = 20000,
         m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
         C0: float = C0_DEFAULT, project: bool = True) -> dict:
    """Integra el Flujo del Camino (ec. 4.2) con Euler proyectado.

    Devuelve la trayectoria, V(σ) y la producción entrópica
    dS_prod/dσ = (∇V)ᵀG⁻¹(∇V) ≥ 0 (Teo. 4.5) en cada paso.
    """
    if G is None:
        G = np.eye(2)
    G_inv = np.linalg.inv(G)
    if not np.all(np.linalg.eigvalsh(0.5 * (G + G.T)) > 0):
        raise ValueError("La rigidez G debe ser definida positiva (G ≻ 0)")
    phi = np.asarray(phi0, dtype=float).copy()
    traj = [phi.copy()]
    V_hist = []
    S_prod_rate = []
    from .basal import V0
    from .dual_plane import to_dual
    for _ in range(n_steps):
        g = grad_V(phi, delta0, m_bar, b_bar, e_bar, C0)
        step = -G_inv @ g
        S_prod_rate.append(float(g @ G_inv @ g))
        phi = phi + d_sigma * step
        if project:  # dominio físico: primer cuadrante (Cap. 2)
            phi = np.clip(phi, 0.0, None)
        traj.append(phi.copy())
        c = to_dual(phi[0], phi[1])
        V_hist.append(float(V0(c["rho"], c["chi"], delta0,
                               m_bar, b_bar, e_bar, C0)))
        if np.linalg.norm(g) < 1e-12:
            break
    return {
        "trajectory": np.array(traj),
        "V": np.array(V_hist),
        "S_production_rate": np.array(S_prod_rate),
        "final": phi,
    }


def exits_to_mass_pole(result: dict, tol_theta: float = 0.35) -> bool:
    """Prop. 3.5: la salida de S0 ocurre hacia el polo de masa (θ ≈ 0).

    Comprueba que el estado final está en el semiplano de dominio de Mp
    (θ < π/4, por debajo de la diagonal dual) y cerca del eje φM.
    """
    from .dual_plane import to_dual
    phi = result["final"]
    theta = float(to_dual(phi[0], phi[1])["theta"])
    return theta < tol_theta
