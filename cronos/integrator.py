"""Integrador Leapfrog Kick-Drift-Kick adaptado a S — ESQUEMA v32, SUPERADO.

La v35 (cap. 11, Obs. 11.4) sustituye este esquema por Cronos v3
(cronos/cronos_v3.py): a este kick le falta la lapse, la fricción no
lleva compuerta Θ(ρ̇) y falta la fuerza +c²∇ε_c. Se conserva como
referencia del corpus anterior.

Cronos integra en el índice entrópico S en vez de t o a:

    d ln a / dS = C(S)
    ΔS ≈ 1e-3   (paso discreto natural, granularidad LQG)

El paso típico:
    1. Kick: v += a · 0.5 ΔS · f_kick(ρ_local)
    2. Drift: x += v · ΔS
    3. Kick: v += a · 0.5 ΔS · f_kick(ρ_local)

donde f_kick incluye la fricción entrópica local (ver kronos_kick).
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from .kronos_kick import kick_factor, friction_acceleration


def C_of_S(S: float, C0: float = 1.0, alpha: float = 1.0) -> float:
    """d ln a / dS = C(S). Por defecto, C constante (Ley de Cronos lineal)."""
    return C0 * (S / C.S_SEALS["C4"]) ** (alpha - 1.0)


def kdk_step(x: np.ndarray, v: np.ndarray, accel_fn, S: float,
             dS: float = C.DELTA_S, rho_local: np.ndarray | None = None,
             alpha_cronos: float = C.ALPHA_CRONOS,
             rho_c0: float = C.RHO_C0, H_a: float = 1.0) -> tuple:
    """Un paso Leapfrog KDK en espacio S.

    Args:
        x : posiciones (N,3)
        v : velocidades (N,3)
        accel_fn : función x -> aceleración gravitatoria
        S : valor actual de S
        dS : paso ΔS
        rho_local : densidad local en cada partícula (si None, kick_factor=1)
        alpha_cronos, rho_c0 : parámetros Cronos
        H_a : H(a) para fricción

    Returns:
        (x_new, v_new, S_new)
    """
    a = accel_fn(x)
    if rho_local is None:
        f = np.ones(len(x))
    else:
        f = kick_factor(rho_local, alpha_cronos=alpha_cronos, rho_c0=rho_c0)
    a_eff = a * f[:, None] + friction_acceleration(
        v, rho_local if rho_local is not None else np.zeros(len(x)),
        H_a=H_a, alpha_cronos=alpha_cronos, rho_c0=rho_c0,
    )
    v = v + 0.5 * dS * a_eff
    x = x + dS * v
    a = accel_fn(x)
    a_eff = a * f[:, None] + friction_acceleration(
        v, rho_local if rho_local is not None else np.zeros(len(x)),
        H_a=H_a, alpha_cronos=alpha_cronos, rho_c0=rho_c0,
    )
    v = v + 0.5 * dS * a_eff
    return x, v, S + dS
