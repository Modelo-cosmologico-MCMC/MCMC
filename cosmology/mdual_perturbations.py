"""III.F — Perturbaciones MDR completas en variable S.

Tratado §3.19.8 Etapa III. Calcula fσ8(z) y S8 con la correción de
Cronos sobre el crecimiento de estructuras y compara con BOSS DR12,
eBOSS, KiDS-1000, DES Y3.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from cosmology.perturbations import f_sigma8, growth_factor


def f_sigma8_MCMC(z: np.ndarray | float,
                  sigma8_0: float = C.SIGMA8_MCMC,
                  Omega_m: float = 0.30) -> np.ndarray | float:
    """Re-exporta `f_sigma8` con valores MCMC nominales."""
    return f_sigma8(z, sigma8_0=sigma8_0, Omega_m=Omega_m)


def S8(sigma8: float = C.SIGMA8_MCMC, Omega_m: float = 0.30) -> float:
    """S₈ = σ₈ · √(Ω_m / 0.3)."""
    return float(sigma8 * np.sqrt(Omega_m / 0.30))


# --- Datasets observacionales ---
RSD_DATA = {
    "BOSS_DR12": {
        "z":     [0.38, 0.51, 0.61],
        "fsig8": [0.497, 0.458, 0.436],
        "err":   [0.045, 0.038, 0.034],
        "ref":   "Alam+2017 MNRAS 470 2617",
    },
    "eBOSS_QSO": {
        "z":     [1.48],
        "fsig8": [0.462],
        "err":   [0.045],
        "ref":   "Neveux+2020 MNRAS 499 210",
    },
    "VIPERS": {
        "z":     [0.6, 0.86],
        "fsig8": [0.48, 0.48],
        "err":   [0.12, 0.10],
        "ref":   "Pezzotta+2017 A&A 604 A33",
    },
}

WL_DATA = {
    "KiDS_1000": {"S8": 0.766, "err": 0.020,
                  "ref": "Heymans+2021 A&A 646 A140"},
    "DES_Y3":    {"S8": 0.776, "err": 0.017,
                  "ref": "Abbott+2022 PRD 105 023520"},
}


def chi2_fsigma8(dataset: str = "BOSS_DR12",
                 sigma8_0: float = C.SIGMA8_MCMC,
                 Omega_m: float = 0.30) -> float:
    """χ² de la predicción MCMC vs el dataset RSD dado."""
    info = RSD_DATA[dataset]
    z = np.asarray(info["z"], dtype=float)
    fs_obs = np.asarray(info["fsig8"], dtype=float)
    err = np.asarray(info["err"], dtype=float)
    fs_pred = np.asarray(f_sigma8_MCMC(z, sigma8_0=sigma8_0, Omega_m=Omega_m))
    return float(np.sum(((fs_obs - fs_pred) / err) ** 2))


def chi2_S8(dataset: str = "KiDS_1000",
            sigma8_0: float = C.SIGMA8_MCMC,
            Omega_m: float = 0.30) -> float:
    """χ² de S8_MCMC vs un dataset WL."""
    info = WL_DATA[dataset]
    s = S8(sigma8=sigma8_0, Omega_m=Omega_m)
    return float(((s - info["S8"]) / info["err"]) ** 2)


def summary_tensions() -> dict:
    """Resumen de tensiones cosmológicas."""
    return {
        "S8_MCMC":           S8(),
        "S8_KiDS_1000":      WL_DATA["KiDS_1000"]["S8"],
        "S8_DES_Y3":         WL_DATA["DES_Y3"]["S8"],
        "S8_tension_KiDS_sigma": abs(S8() - WL_DATA["KiDS_1000"]["S8"])
                                  / WL_DATA["KiDS_1000"]["err"],
        "S8_tension_DES_sigma":  abs(S8() - WL_DATA["DES_Y3"]["S8"])
                                  / WL_DATA["DES_Y3"]["err"],
        "fsigma8_chi2_BOSS_DR12": chi2_fsigma8("BOSS_DR12"),
    }
