"""Tabla ρ_id(z; S) para uso en runtime de simulaciones N-body."""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from cosmology.background import rho_id


def build_table(z_grid: np.ndarray | None = None,
                S_grid: np.ndarray | None = None) -> dict:
    """Construye la tabla 2D ρ_id(z, S) y la devuelve como dict."""
    if z_grid is None:
        z_grid = np.linspace(0.0, 30.0, 121)
    if S_grid is None:
        # Rango S del esquema v32: de C4 (1.001) al máximo 150.0 de la
        # parametrización del Unificado (bloque LEGACY_V32 en constants.py).
        S_grid = np.linspace(C.S_SEALS["C4"], 150.0, 121)
    z_grid = np.asarray(z_grid, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)
    grid = np.zeros((len(z_grid), len(S_grid)))
    for i, z in enumerate(z_grid):
        for j, S in enumerate(S_grid):
            grid[i, j] = float(rho_id(z, S=S))
    return {"z": z_grid, "S": S_grid, "rho_id": grid}


def interp_rho_id(z: float, S: float, table: dict) -> float:
    """Interpolación bilineal sobre la tabla."""
    z_grid = table["z"]; S_grid = table["S"]; rho = table["rho_id"]
    iz = np.clip(np.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 2)
    iS = np.clip(np.searchsorted(S_grid, S) - 1, 0, len(S_grid) - 2)
    fz = (z - z_grid[iz]) / (z_grid[iz + 1] - z_grid[iz])
    fS = (S - S_grid[iS]) / (S_grid[iS + 1] - S_grid[iS])
    v00 = rho[iz, iS];     v01 = rho[iz, iS + 1]
    v10 = rho[iz + 1, iS]; v11 = rho[iz + 1, iS + 1]
    return float(
        (1 - fz) * (1 - fS) * v00 + (1 - fz) * fS * v01
        + fz * (1 - fS) * v10 + fz * fS * v11
    )
