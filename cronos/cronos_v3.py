"""Cronos v3 — esquema KDK del capítulo 11 del Tratado de Fundamentos (v35).

FRENTE ABIERTO Nº 5 (§13.6): la ejecución de simulaciones de producción
con este esquema es trabajo de investigación pendiente. Este módulo
implementa las primitivas del esquema vigente; NO está validado en
producción.

Ley de Cronos débil (Def. 11.1):
    N(x,t) = 1 + Φ_N/c² − ε_c(ρ)
    ε_c(ρ) = α0⁻¹ · (ρ/ρ_c)^(3/2)  ≥ 0

El signo (doble justificación): en regiones densas la descarga se estanca
→ N < 1; y los relojes van más lentos donde Φ_N < 0 — la corrección
Cronos suma en el mismo sentido. La fórmula del corpus anterior tenía el
signo invertido (ver Obs. 11.4 en cronos/__init__.py).

Las tres correcciones (Cor. 11.3):
  (a) Fricción con compuerta: Γ = (3/2)·(ρ̇/ρ)·ε_c·Θ(ρ̇)
      — unidades de 1/tiempo, activa solo mientras ρ̇ > 0, se apaga al
      virializar (firma falsable, fig. 11.1).
  (b) Cierre de caída libre: Γ = (3/2)·C_ff·sqrt(G·ρ_c)·α0⁻¹·(ρ/ρ_c)²
      — exponente 2 en la forma cerrada; sqrt(G·ρ_c) porta las unidades.
  (c) La fuerza omitida (MCV): F_extra = +c²·∇ε_c (atracción hacia picos
      de densidad).

Cota dura y falsable (ec. 11.5): α0⁻¹ ≲ (v_circ/c)² ~ 1e-6.
"""

from __future__ import annotations

import numpy as np

ALPHA0_INV_MAX = 1e-6   # Cota dura sobre la Amplitud de Cronos (ec. 11.5)
C_LIGHT = 1.0           # c en unidades internas del integrador (configurable)


def check_alpha0_inv(alpha0_inv: float) -> float:
    """Valida la cota α0⁻¹ ≲ 1e-6 (ec. 11.5). Lanza ValueError si se excede."""
    if alpha0_inv < 0.0:
        raise ValueError(f"α0⁻¹ debe ser ≥ 0 (recibido {alpha0_inv})")
    if alpha0_inv > ALPHA0_INV_MAX:
        raise ValueError(
            f"α0⁻¹ = {alpha0_inv} viola la cota dura {ALPHA0_INV_MAX} "
            "(Tratado de Fundamentos v35, ec. 11.5)"
        )
    return alpha0_inv


def epsilon_c(rho: np.ndarray, alpha0_inv: float,
              rho_c: float) -> np.ndarray:
    """ε_c(ρ) = α0⁻¹ · (ρ/ρ_c)^(3/2) ≥ 0 (Def. 11.1)."""
    check_alpha0_inv(alpha0_inv)
    rho = np.asarray(rho, dtype=float)
    return alpha0_inv * np.clip(rho / rho_c, 0.0, None) ** 1.5


def lapse(Phi_N: np.ndarray, rho: np.ndarray, alpha0_inv: float,
          rho_c: float, c: float = C_LIGHT) -> np.ndarray:
    """N = 1 + Φ_N/c² − ε_c(ρ) (Def. 11.1, signo corregido en v35)."""
    return 1.0 + np.asarray(Phi_N, dtype=float) / c ** 2 \
        - epsilon_c(rho, alpha0_inv, rho_c)


def gate_friction(rho: np.ndarray, rho_dot: np.ndarray, alpha0_inv: float,
                  rho_c: float) -> np.ndarray:
    """Γ = (3/2)·(ρ̇/ρ)·ε_c·Θ(ρ̇) (Cor. 11.3a).

    NO es ∝ ε_c a secas: tiene unidades de inverso de tiempo, está activa
    solo mientras ρ̇ > 0 y se apaga al virializar (fig. 11.1).
    """
    rho = np.asarray(rho, dtype=float)
    rho_dot = np.asarray(rho_dot, dtype=float)
    eps = epsilon_c(rho, alpha0_inv, rho_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = 1.5 * np.where(rho > 0.0, rho_dot / rho, 0.0) * eps
    return np.where(rho_dot > 0.0, gamma, 0.0)


def gate_friction_freefall(rho: np.ndarray, alpha0_inv: float, rho_c: float,
                           G: float = 1.0, C_ff: float = 1.0) -> np.ndarray:
    """Forma cerrada de caída libre: Γ = (3/2)·C_ff·√(G·ρ_c)·α0⁻¹·(ρ/ρ_c)²
    (Cor. 11.3b — exponente 2; √(G·ρ_c) porta las unidades)."""
    check_alpha0_inv(alpha0_inv)
    rho = np.asarray(rho, dtype=float)
    return 1.5 * C_ff * np.sqrt(G * rho_c) * alpha0_inv \
        * np.clip(rho / rho_c, 0.0, None) ** 2


def extra_force(grad_eps_c: np.ndarray, c: float = C_LIGHT) -> np.ndarray:
    """F_extra = +c²·∇ε_c (Cor. 11.3c — cara de campo débil de la MCV):
    atracción hacia los picos de densidad."""
    return c ** 2 * np.asarray(grad_eps_c, dtype=float)


def kdk_step_v3(x: np.ndarray, u: np.ndarray, dt: float, *,
                Phi_N: np.ndarray, grad_Phi_N: np.ndarray,
                rho: np.ndarray, rho_dot: np.ndarray,
                grad_eps_c: np.ndarray, alpha0_inv: float, rho_c: float,
                c: float = C_LIGHT) -> tuple[np.ndarray, np.ndarray]:
    """Un paso Kick-Drift-Kick de Cronos v3 (esquema 11.4 del tratado).

        1. Campos:  ε_c = α0⁻¹(ρ/ρ_c)^{3/2};  N = 1 + Φ_N/c² − ε_c
        2. Kick(½): u ← u·exp(−Γ·Δt/2), Γ = (3/2)(ρ̇/ρ)ε_c·Θ(ρ̇)
                    u ← u − ∇(Φ_N − c²·ε_c)·N·Δt/2
        3. Drift:   x ← x + u·N·Δt
        4. Refresco de campos y Kick espejo (aquí: mismos campos; el
           refresco corresponde al solver de Poisson del llamador)

    Los gradientes y ρ̇ los provee el llamador (solver de campos). Las
    correcciones frente al esquema v32: el kick lleva la lapse, la
    fricción lleva compuerta Θ(ρ̇), y aparece +c²∇ε_c en la fuerza.
    """
    N = lapse(Phi_N, rho, alpha0_inv, rho_c, c=c)
    Gamma = gate_friction(rho, rho_dot, alpha0_inv, rho_c)
    # −∇(Φ_N − c²·ε_c) = −∇Φ_N + c²·∇ε_c
    force = -np.asarray(grad_Phi_N, dtype=float) + extra_force(grad_eps_c, c=c)

    def half_kick(u_in: np.ndarray) -> np.ndarray:
        u_out = u_in * np.exp(-Gamma * dt / 2.0)[..., None]
        return u_out + force * N[..., None] * dt / 2.0

    u = half_kick(u)
    x = x + u * N[..., None] * dt
    u = half_kick(u)
    return x, u
