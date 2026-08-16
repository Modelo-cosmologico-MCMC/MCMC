"""Comparador de perfiles binados σ_los(R) — instrumento listo, sin datos.

El perfil binado de Sculptor (Walker et al. 2009) NO está ingerido: los
archivos astronómicos están bloqueados por el proxy de la sesión de
trabajo (pendiente declarado en dynamics/dsph_data.py). Este módulo
deja el instrumento preparado para el momento de la ingesta: dado un
perfil binado (R_i, σ_i, e_i) y un modelo σ_los(R), el χ² y su
comparación entre modelos — con autoprueba sintética por construcción
(v0.1: la autoprueba NO es contraste con datos, igual que el
synthetic_selftest_dataset de cosmology/bayesian_fit.py).

Formato de ingesta esperado (data/dsph/sculptor_sigma_los.txt):
    # R_pc   sigma_kms   err_kms   [referencia por punto]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_FILE = (Path(__file__).resolve().parent.parent / "data" / "dsph"
             / "sculptor_sigma_los.txt")


def load_binned_profile(path: Path | None = None):
    """(R, σ, e) del fichero de ingesta; FileNotFoundError con mensaje
    de estatuto si aún no existe (la ausencia es el estado declarado)."""
    p = Path(path) if path is not None else DATA_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"{p} no existe: la ingesta del perfil binado de Walker "
            "et al. 2009 está pendiente (archivos bloqueados por el "
            "proxy — ver dynamics/dsph_data.py). Este instrumento "
            "queda listo para cuando llegue.")
    arr = np.loadtxt(p, usecols=(0, 1, 2))
    return arr[:, 0], arr[:, 1], arr[:, 2]


def chi2_profile(sigma_model_kms: np.ndarray, sigma_obs_kms: np.ndarray,
                 err_kms: np.ndarray) -> float:
    """χ² = Σ [(σ_mod − σ_obs)/e]² (bins independientes — la
    covarianza entre bins, si la tabla original la trae, se añadirá
    en la ingesta)."""
    resid = (np.asarray(sigma_model_kms, float)
             - np.asarray(sigma_obs_kms, float)) / np.asarray(err_kms, float)
    return float(np.sum(resid ** 2))


def synthetic_selftest(seed: int = 0) -> dict:
    """Autoprueba circular POR CONSTRUCCIÓN (no contraste): genera un
    perfil sintético desde un modelo, le añade ruido gaussiano y
    verifica que el χ² del modelo verdadero es ~n_bins y que un modelo
    desplazado lo empeora. Ejercita el instrumento, no valida física."""
    rng = np.random.default_rng(seed)
    R = np.linspace(50.0, 900.0, 12)
    sigma_true = 9.0 * (1.0 + R / 4000.0) ** -0.5
    err = np.full_like(R, 0.8)
    sigma_obs = sigma_true + rng.normal(0.0, 1.0, R.size) * err
    chi2_true = chi2_profile(sigma_true, sigma_obs, err)
    chi2_shifted = chi2_profile(sigma_true + 3.0, sigma_obs, err)
    return {"n_bins": R.size, "chi2_true": chi2_true,
            "chi2_shifted": chi2_shifted}
