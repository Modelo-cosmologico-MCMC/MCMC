"""El Mecanismo del Discriminante y el flujo log-periódico (v35, Cap. 8).

Completa la Ley de la Década (Prop. 8.1, ya en constants.decade_thresholds)
con el mecanismo que la produce.

Discriminante basal y espinodal (Def. 8.4):

    D(S) ≡ B²(S) − 4·C0(S)·M0²(S);   la espinodal es el lugar D = 0.

Estructura tipo KLS del flujo radial (Prop. 8.5), con x = ρ²:

    dx/dσ = −(2·C0·x/G)·(x − x+)(x − x−),   x± = (B ± √D)/(2C0)

Los puntos fijos se FUSIONAN en la espinodal (D = 0) y se complejifican
para D < 0: x± = xR ± iΩ con xR = B/(2C0), Ω = √(−D)/(2C0) — el
mecanismo canónico de aniquilación de puntos fijos (KLS). En régimen
D < 0 el flujo es log-periódico (walking) con periodo (ec. 8.4):

    Δσ_walk ≃ (G/(2·C0·xR))·(π/Ω)

Exponente de Victoria (Def. 8.3): la cascada geométrica con razón λ es
invariancia de escala discreta por exponentes críticos complejos ±i·s0,
con λ = e^{π/s0}; para λ = 10, s0 = π/ln(10) ≈ 1.3644 (ec. 8.2).

ESTATUTO (Obs. 8.6-8.7): la metastabilidad (3.3) implica D(S0) > (4/3)·
C0·M0² > 0; la cascada exige que el flujo de acoplos hunda D(S) bajo
cero entre colapsos — el cruce de la espinodal (el CRUCE DE VICTORIA)
dispara cada colapso. λ = 10 es CALIBRADO; derivarlo de las funciones de
flujo es el frente abierto nº 2. Este módulo expone el mecanismo, no lo
resuelve.
"""

from __future__ import annotations

import numpy as np

from .basal import scaled_params, M_BAR, B_BAR, C0_DEFAULT


def discriminant_basal(delta0: float, m_bar: float = M_BAR,
                       b_bar: float = B_BAR,
                       C0: float = C0_DEFAULT) -> float:
    """D = B² − 4·C0·M0² con el escalado (3.2) (Def. 8.4)."""
    p = scaled_params(delta0, m_bar, b_bar)
    return p["B"] ** 2 - 4.0 * C0 * p["M0_sq"]


def fixed_points(B: float, M0_sq: float, C0: float = C0_DEFAULT) -> tuple:
    """x± = (B ± √D)/(2C0) si D ≥ 0; para D < 0, (xR, Ω) complejos
    (Prop. 8.5)."""
    D = B ** 2 - 4.0 * C0 * M0_sq
    xR = B / (2.0 * C0)
    if D >= 0.0:
        r = np.sqrt(D) / (2.0 * C0)
        return ("real", xR + r, xR - r)
    return ("complex", xR, np.sqrt(-D) / (2.0 * C0))  # (xR, Ω)


def radial_flow_rhs(x: np.ndarray | float, B: float, M0_sq: float,
                    C0: float = C0_DEFAULT,
                    G: float = 1.0) -> np.ndarray | float:
    """dx/dσ = −(2·C0·x/G)·(x − x+)(x − x−) (ec. 8.3).

    El producto (x−x+)(x−x−) = C0x² − Bx + M0² dividido por C0; se
    evalúa directamente con el polinomio para cubrir también D < 0.
    """
    x = np.asarray(x, dtype=float)
    poly = x ** 2 - (B / C0) * x + M0_sq / C0
    out = -(2.0 * C0 * x / G) * poly
    return float(out) if np.ndim(out) == 0 else out


def walk_period(B: float, M0_sq: float, C0: float = C0_DEFAULT,
                G: float = 1.0) -> float:
    """Δσ_walk ≃ (G/(2·C0·xR))·(π/Ω) — el periodo del walking (ec. 8.4).

    Solo definido en régimen D < 0 (puntos fijos complejos); diverge al
    acercarse a la espinodal (Ω → 0).
    """
    kind, xR, omega = fixed_points(B, M0_sq, C0)
    if kind != "complex":
        raise ValueError("El walking requiere D < 0 (puntos fijos complejos)")
    return (G / (2.0 * C0 * xR)) * np.pi / omega


def s0_from_lambda(lam: float) -> float:
    """s0 = π/ln(λ) (Def. 8.3). Para λ=10: 1.3644 (ec. 8.2)."""
    if lam <= 1.0:
        raise ValueError("La razón de la cascada debe ser λ > 1")
    return float(np.pi / np.log(lam))


def lambda_from_s0(s0: float) -> float:
    """λ = e^{π/s0} (Def. 8.3)."""
    return float(np.exp(np.pi / s0))


def spinodal_crossing(D_of_S, S_lo: float, S_hi: float,
                      tol: float = 1e-12) -> float:
    """El Cruce de Victoria: S* con D(S*) = 0, por bisección (Obs. 8.6).

    D_of_S: callable S → D(S) (el flujo de acoplos dλi/dS = βi del
    corpus provee la trayectoria; aquí se localiza el cruce dado el
    perfil). Requiere cambio de signo en [S_lo, S_hi].
    """
    d_lo, d_hi = D_of_S(S_lo), D_of_S(S_hi)
    if d_lo * d_hi > 0.0:
        raise ValueError("Sin cambio de signo de D en el intervalo dado")
    while S_hi - S_lo > tol:
        S_mid = 0.5 * (S_lo + S_hi)
        if D_of_S(S_mid) * d_lo <= 0.0:
            S_hi = S_mid
        else:
            S_lo = S_mid
    return 0.5 * (S_lo + S_hi)


def metastability_bound(delta0: float, m_bar: float = M_BAR,
                        b_bar: float = B_BAR,
                        C0: float = C0_DEFAULT) -> bool:
    """Obs. 8.6: la metastabilidad (3.3) implica D(S0) > (4/3)·C0·M0²."""
    p = scaled_params(delta0, m_bar, b_bar)
    return discriminant_basal(delta0, m_bar, b_bar, C0) \
        > (4.0 / 3.0) * C0 * p["M0_sq"]
