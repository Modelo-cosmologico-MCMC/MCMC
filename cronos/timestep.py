"""Paso temporal entrópico de Cronos v3 (v35, Apéndice B, ec. B.3).

    Δt_i = η/sqrt(|a_i|·a) · [1 + (1/α_cr)·(ρ_i/ρ_c)^(3/2)]⁻¹,   η = 0.025

En las regiones densas el reloj entrópico avanza más lento: el paso se
reduce y la dinámica se amortigua. Es la misma lapse N = e^{Φ_ten}
(ec. 11.1) actuando partícula a partícula; el signo es el corregido de
la Obs. 11.4 (sustituye al del esquema antiguo).
"""

from __future__ import annotations

import numpy as np

ETA_CRONOS = 0.025  # η del paso entrópico (v35, B.3)


def entropic_timestep(accel_mag: np.ndarray, a_scale: float,
                      rho: np.ndarray, rho_c: float, alpha_cr: float,
                      eta: float = ETA_CRONOS) -> np.ndarray:
    """Δt_i por partícula según la ec. (B.3).

    Args:
        accel_mag: |a_i| — módulo de la aceleración por partícula.
        a_scale: factor de escala a.
        rho: densidad local ρ_i por partícula.
        rho_c: densidad de referencia.
        alpha_cr: coeficiente tensional (> 0).
    """
    if alpha_cr <= 0.0:
        raise ValueError("alpha_cr debe ser > 0 (ec. B.3)")
    accel_mag = np.clip(np.asarray(accel_mag, dtype=float), 1e-30, None)
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, None)
    base = eta / np.sqrt(accel_mag * a_scale)
    return base / (1.0 + (rho / rho_c) ** 1.5 / alpha_cr)


def global_timestep(accel_mag: np.ndarray, a_scale: float,
                    rho: np.ndarray, rho_c: float, alpha_cr: float,
                    eta: float = ETA_CRONOS) -> float:
    """Paso global = mínimo del paso por partícula (sincronización)."""
    return float(np.min(entropic_timestep(accel_mag, a_scale, rho, rho_c,
                                          alpha_cr, eta=eta)))
