"""III.E — Validación Cronos vs catálogo SPARC.

Datos: Lelli, McGaugh & Schombert 2016, AJ 152, 157
URL:   http://astroweb.cwru.edu/SPARC/

Si los archivos del catálogo no se encuentran en `data/sparc/`, los métodos
caen sobre un mini-mock con las 5 galaxias diana del Tratado (Tabla 17),
permitiendo validar la pipeline sin dependencia de descarga externa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from cronos.halo_profile import cored_profile, nfw_profile, r_core


G_KPC = 4.302e-3   # G en kpc·(km/s)²/M_sun


@dataclass
class GalaxyRotation:
    name: str
    r_kpc: np.ndarray
    V_obs: np.ndarray
    dV_obs: np.ndarray
    V_baryons: np.ndarray  # gas + disk + bulge


# --- Tabla 17 del Tratado: 5 galaxias diana con χ² target ---
SPARC_TARGETS = {
    "NGC2403": {"chi2_NFW": 15.2, "chi2_MCMC": 6.1,  "improvement_pct": 60},
    "NGC3198": {"chi2_NFW": 18.4, "chi2_MCMC": 7.9,  "improvement_pct": 57},
    "F563-V2": {"chi2_NFW": 12.3, "chi2_MCMC": 4.2,  "improvement_pct": 66},
    "DDO154":  {"chi2_NFW":  9.5, "chi2_MCMC": 3.8,  "improvement_pct": 60},
    "UGC2885": {"chi2_NFW": 22.1, "chi2_MCMC": 10.4, "improvement_pct": 53},
}


def mock_galaxy(name: str, M_halo: float = 1e11, R_max: float = 30.0,
                seed: int = 0) -> GalaxyRotation:
    """Curva de rotación sintética coherente con el perfil cored."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.5, R_max, 30)
    rc = r_core(M_halo, z=0.0)
    rho0 = M_halo / (4 * np.pi * rc ** 3)
    M_enc = 4 * np.pi * rho0 * rc ** 2 * (r - rc * np.arctan(r / rc))
    V_dm = np.sqrt(G_KPC * np.maximum(M_enc, 0) / r)
    V_b = 0.3 * V_dm  # baryons sub-dominante
    V_obs = np.sqrt(V_dm ** 2 + V_b ** 2)
    dV = 0.05 * V_obs + 2.0
    V_obs_noisy = V_obs + dV * rng.normal(size=len(r))
    return GalaxyRotation(name=name, r_kpc=r, V_obs=V_obs_noisy,
                          dV_obs=dV, V_baryons=V_b)


def load_sparc_galaxy(name: str, sparc_dir: Path | str = "data/sparc") -> GalaxyRotation:
    """Carga una galaxia SPARC. Cae a `mock_galaxy` si el archivo no existe.

    Formato esperado (Lelli+2016): {name}_rotmod.dat con columnas
        r [kpc], V_obs [km/s], dV [km/s], V_gas, V_disk, V_bulge, ...
    """
    p = Path(sparc_dir) / f"{name}_rotmod.dat"
    if not p.exists():
        return mock_galaxy(name)
    data = np.loadtxt(p, comments=("#", "%"))
    r = data[:, 0]; V_obs = data[:, 1]; dV = data[:, 2]
    V_gas = data[:, 3] if data.shape[1] > 3 else np.zeros_like(r)
    V_dsk = data[:, 4] if data.shape[1] > 4 else np.zeros_like(r)
    V_bul = data[:, 5] if data.shape[1] > 5 else np.zeros_like(r)
    V_b = np.sqrt(V_gas ** 2 + V_dsk ** 2 + V_bul ** 2)
    return GalaxyRotation(name=name, r_kpc=r, V_obs=V_obs, dV_obs=dV, V_baryons=V_b)


def v_cored(r_kpc: np.ndarray, M_halo: float, rc_kpc: float | None = None,
            z: float = 0.0) -> np.ndarray:
    """Velocidad circular del perfil cored (Burkert-like) usado por Cronos."""
    if rc_kpc is None:
        rc_kpc = r_core(M_halo, z=z)
    rho0 = M_halo / (4 * np.pi * rc_kpc ** 3)
    M_enc = 4 * np.pi * rho0 * rc_kpc ** 2 * (
        r_kpc - rc_kpc * np.arctan(r_kpc / rc_kpc)
    )
    return np.sqrt(G_KPC * np.maximum(M_enc, 0) / np.maximum(r_kpc, 1e-3))


def v_NFW(r_kpc: np.ndarray, M200: float, c_concentration: float = 10.0) -> np.ndarray:
    """Velocidad circular NFW (referencia ΛCDM)."""
    R200 = (M200 / (4.0 / 3 * np.pi * 200 * 1.4e2)) ** (1.0 / 3)  # kpc, ρ_crit ~140
    rs = R200 / c_concentration
    f = lambda x: np.log(1 + x) - x / (1 + x)
    M_r = M200 * f(r_kpc / rs) / f(c_concentration)
    return np.sqrt(G_KPC * M_r / np.maximum(r_kpc, 1e-3))


def chi2_fit(galaxy: GalaxyRotation, V_model: np.ndarray) -> float:
    """χ² del ajuste."""
    return float(np.sum(((galaxy.V_obs - V_model) / galaxy.dV_obs) ** 2))


def fit_cored(galaxy: GalaxyRotation,
              M_halo_grid: np.ndarray | None = None) -> dict:
    """Ajuste mínimo: barrido en M_halo del perfil cored."""
    if M_halo_grid is None:
        M_halo_grid = np.logspace(9.5, 12.5, 50)
    chi2 = []
    for M in M_halo_grid:
        V_dm = v_cored(galaxy.r_kpc, M)
        V_tot = np.sqrt(V_dm ** 2 + galaxy.V_baryons ** 2)
        chi2.append(chi2_fit(galaxy, V_tot))
    chi2 = np.asarray(chi2)
    i = int(np.argmin(chi2))
    return {"M_halo": float(M_halo_grid[i]), "chi2": float(chi2[i]),
            "model": "cored"}


def fit_NFW(galaxy: GalaxyRotation,
            M200_grid: np.ndarray | None = None) -> dict:
    """Ajuste mínimo: barrido en M200 del perfil NFW."""
    if M200_grid is None:
        M200_grid = np.logspace(9.5, 12.5, 50)
    chi2 = []
    for M in M200_grid:
        V_dm = v_NFW(galaxy.r_kpc, M)
        V_tot = np.sqrt(V_dm ** 2 + galaxy.V_baryons ** 2)
        chi2.append(chi2_fit(galaxy, V_tot))
    chi2 = np.asarray(chi2)
    i = int(np.argmin(chi2))
    return {"M200": float(M200_grid[i]), "chi2": float(chi2[i]),
            "model": "NFW"}


def compare_cored_vs_NFW(galaxy_names: Iterable[str] | None = None,
                         sparc_dir: str = "data/sparc") -> list[dict]:
    """Comparativa χ²(cored) vs χ²(NFW) para una lista de galaxias.

    Si la lista no se da, usa las 5 galaxias diana de la Tabla 17.
    """
    if galaxy_names is None:
        galaxy_names = list(SPARC_TARGETS.keys())
    results = []
    for name in galaxy_names:
        gal = load_sparc_galaxy(name, sparc_dir=sparc_dir)
        r_nfw = fit_NFW(gal)
        r_cor = fit_cored(gal)
        improvement = (r_nfw["chi2"] - r_cor["chi2"]) / max(r_nfw["chi2"], 1e-12) * 100
        results.append({
            "galaxy":         name,
            "chi2_NFW":       r_nfw["chi2"],
            "chi2_cored":     r_cor["chi2"],
            "improvement_pct": improvement,
        })
    return results
