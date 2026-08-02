"""Refresco de los canales tensionales y dilatación local (v35, B.3).

Paso 3 del ciclo Cronos-KDK:
    ρ_lat^{n+1} = ρ_lat^n + Δt·(−κ_lat·ρ_lat^n + Γ_lat)
    ρ_id^{n+1}  = ρ_id^n  + Δt·(+η_dir·ρ_m^n + Γ_act)
con las tasas tomadas del mapa S(t).

Paso 2 (drift): dilatación local
    ζ(x) = ζ0 · ρ_lat(x) / (ρ_lat(x) + ρ*)

Los valores numéricos de κ_lat, Γ_lat, η_dir, Γ_act, ζ0 y ρ* no están
fijados en el tratado (estatuto calibrado; el mapa S(t) los provee en
producción): aquí son parámetros explícitos.
"""

from __future__ import annotations

import numpy as np


def refresh_channels(rho_lat: np.ndarray, rho_id: np.ndarray,
                     rho_m: np.ndarray, dt: float,
                     kappa_lat: float = 0.0, Gamma_lat: float = 0.0,
                     eta_dir: float = 0.0, Gamma_act: float = 0.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Un paso de Euler del refresco de mallas (B.3, paso 3)."""
    rho_lat = np.asarray(rho_lat, dtype=float)
    rho_id = np.asarray(rho_id, dtype=float)
    rho_m = np.asarray(rho_m, dtype=float)
    rho_lat_new = rho_lat + dt * (-kappa_lat * rho_lat + Gamma_lat)
    rho_id_new = rho_id + dt * (eta_dir * rho_m + Gamma_act)
    return np.clip(rho_lat_new, 0.0, None), np.clip(rho_id_new, 0.0, None)


def local_dilation(rho_lat: np.ndarray, zeta0: float,
                   rho_star: float) -> np.ndarray:
    """ζ(x) = ζ0·ρ_lat/(ρ_lat + ρ*) — dilatación local del drift (B.3).

    Saturante: ζ → 0 donde no hay canal latente, ζ → ζ0 donde domina.
    """
    if rho_star <= 0.0:
        raise ValueError("rho_star debe ser > 0")
    rho_lat = np.clip(np.asarray(rho_lat, dtype=float), 0.0, None)
    return zeta0 * rho_lat / (rho_lat + rho_star)
