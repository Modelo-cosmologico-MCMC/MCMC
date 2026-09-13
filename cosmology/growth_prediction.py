"""Predicción perturbativa out-of-sample: fσ8(z) desde el fondo
CC+BAO+SNe (rama theory/perturbations-linear-growth).

CONSTRUYE SOBRE el andamiaje existente — el crecimiento lineal
integrado EXACTO de cosmology/extended_likelihoods.growth_D_f (la ODE
D'' + (2 + dlnH/dlna)·D' − (3/2)·Ω_m(a)·D = 0 sobre el H(z) del
modelo, RK4; NO la aproximación γ de Linder de
cosmology/perturbations.f_sigma8, que queda como utilidades legacy).

PROTOCOLO (preinscrito en results/2026-09-12_perturbations_fsigma8/
preregistration.json ANTES de computar):

  - El posterior de fondo es el de CC+BAO+SNe SOLO (ajuste v1
    corregido, results/2026-08-10_production_fit/chains_mcmc.npz):
    el crecimiento NO entró en ese ajuste, así que la comparación con
    la compilación RSD es out-of-sample y aquí NO SE AJUSTA NADA
    (ni σ8 ni ningún otro parámetro; sin optimizadores ni samplers).

  - El número con CERO parámetros nuevos es la RAZÓN
        R(z) = [f·D](z; θ, ε) / [f·D](z; θ, ε = 0),
    donde σ8 se cancela exactamente. Para el contraste absoluto con
    los datos RSD, σ8 = SIGMA8_MCMC entra como CALIBRACIÓN EXTERNA
    DECLARADA (constante del corpus, estatuto Ap. F), idéntica en los
    dos brazos.

  - Los moduladores µ(a), η(a) NO se usan aquí (µ = η = 1, GR): el
    objetivo discriminante µ(k,z), η(k,z) ≠ 1 debe DERIVARSE de
    Atlas/ε_c (frentes 3 y 10) — declarado, no ejecutado. PROHIBICIÓN
    preinscrita: nunca elegir µ, η desde los datos de lensing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cosmology.extended_likelihoods import growth_D_f, load_fsigma8_data
from mcmc_ontology import constants as C

ROOT = Path(__file__).resolve().parent.parent
CHAINS_V1 = (ROOT / "results" / "2026-08-10_production_fit"
             / "chains_mcmc.npz")
OUTDIR = ROOT / "results" / "2026-09-12_perturbations_fsigma8"

Z_BAND = np.linspace(0.0, 2.0, 41)     # malla pública de la banda
N_DRAWS = 1000                          # submuestras del posterior
DRAW_SEED = 20260912
SIGMA8_EXTERNAL = float(C.SIGMA8_MCMC)  # calibración externa declarada


def load_background_posterior(path: Path = CHAINS_V1) -> np.ndarray:
    """Cadena (n, 4) = (H0, Ω_m, ε, z_trans) del ajuste v1 corregido
    (CC+BAO+SNe, SIN crecimiento). El fichero declara sus parámetros;
    se verifica el orden antes de usarlo."""
    d = np.load(path, allow_pickle=True)
    params = [str(p) for p in d["params"]]
    if params != ["H0", "Omega_m", "epsilon", "z_trans"]:
        raise RuntimeError(f"orden de parámetros inesperado: {params}")
    return np.asarray(d["chain"], float)


def fD_of_z(z: np.ndarray, theta: tuple) -> np.ndarray:
    """f(z)·D(z)/D(0) con el crecimiento integrado exacto."""
    D, f = growth_D_f(np.asarray(z, float), theta)
    return f * D


def ratio_fsigma8(z: np.ndarray, theta: tuple) -> np.ndarray:
    """R(z) = fσ8^MCMC/fσ8^ΛCDM al MISMO (H0, Ω_m): σ8 se cancela —
    la predicción con cero parámetros nuevos. Con ε = 0, R ≡ 1
    exactamente (mismo integrador, misma malla)."""
    H0, Om, eps, z_t = theta
    num = fD_of_z(z, (H0, Om, eps, z_t))
    den = fD_of_z(z, (H0, Om, 0.0, z_t))
    return num / den


def fsigma8_absolute(z: np.ndarray, theta: tuple,
                     sigma8: float = SIGMA8_EXTERNAL) -> np.ndarray:
    """fσ8(z) absoluta con σ8 externa (declarada), sin ajuste."""
    return sigma8 * fD_of_z(z, theta)


def posterior_band(chain: np.ndarray, z: np.ndarray = Z_BAND,
                   n_draws: int = N_DRAWS,
                   seed: int = DRAW_SEED) -> dict:
    """Banda predicha (p16/p50/p84) de R(z) y de fσ8(z) absoluta
    propagando el posterior de fondo — computación pura, sin datos."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chain), size=min(n_draws, len(chain)),
                     replace=False)
    R = np.empty((len(idx), len(z)))
    A = np.empty((len(idx), len(z)))
    for i, j in enumerate(idx):
        H0, Om, eps, z_t = chain[j]
        num = fD_of_z(z, (H0, Om, eps, z_t))
        den = fD_of_z(z, (H0, Om, 0.0, z_t))
        R[i] = num / den
        A[i] = SIGMA8_EXTERNAL * num
    p = [16, 50, 84]
    return {"z": z,
            "ratio_p16": np.percentile(R, p[0], axis=0),
            "ratio_p50": np.percentile(R, p[1], axis=0),
            "ratio_p84": np.percentile(R, p[2], axis=0),
            "abs_p16": np.percentile(A, p[0], axis=0),
            "abs_p50": np.percentile(A, p[1], axis=0),
            "abs_p84": np.percentile(A, p[2], axis=0),
            "n_draws": int(len(idx)), "seed": seed}


def prior_envelope(z: np.ndarray, theta_ref: tuple,
                   eps_1sigma: float = 0.05) -> np.ndarray:
    """Envolvente |R(z) − 1| al mover ε en ±1σ del prior del Ap. F
    (σ = 0.05) con el resto de θ en la referencia — la «banda del
    prior de ε_Λ» de la expectativa preinscrita."""
    H0, Om, _, z_t = theta_ref
    up = np.abs(ratio_fsigma8(z, (H0, Om, +eps_1sigma, z_t)) - 1.0)
    dn = np.abs(ratio_fsigma8(z, (H0, Om, -eps_1sigma, z_t)) - 1.0)
    return np.maximum(up, dn)


def out_of_sample_chi2(theta_ref: tuple,
                       sigma8: float = SIGMA8_EXTERNAL) -> dict:
    """χ² de fσ8 en los DOS brazos al θ de referencia (mediana del
    posterior de fondo), σ8 idéntica y fija: contraste out-of-sample,
    sin ajustar nada. El número que discrimina es Δχ² = MCMC − ΛCDM."""
    z, fs8, sig = load_fsigma8_data()
    H0, Om, eps, z_t = theta_ref
    pred_mcmc = sigma8 * fD_of_z(z, (H0, Om, eps, z_t))
    pred_lcdm = sigma8 * fD_of_z(z, (H0, Om, 0.0, z_t))
    chi2_m = float(np.sum(((fs8 - pred_mcmc) / sig) ** 2))
    chi2_l = float(np.sum(((fs8 - pred_lcdm) / sig) ** 2))
    return {"n_points": int(len(z)), "chi2_mcmc": chi2_m,
            "chi2_lcdm": chi2_l, "delta_chi2": chi2_m - chi2_l,
            "sigma8_external": sigma8}
