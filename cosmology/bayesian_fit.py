"""Ajuste bayesiano emcee para los parámetros MCMC.

Parámetros libres del modo básico: (H0, Omega_m, eps, z_trans).
Likelihoods disponibles: H(z) (cronómetros cósmicos), SNe Ia con matriz
de covarianza (Pantheon+) y BAO (D_M/r_d, D_H/r_d).

Priors alineados con la tabla F.3/F.4 del Tratado de Fundamentos (v35):
    H0      ~ N(67.4, 5²)
    ε       ~ gaussiano débil centrado en EPSILON_0 (εΛ = 0.012 ± 0.003
              es el valor de la transición, A.3; el prior del ajuste es
              deliberadamente más ancho)
    z_trans ~ gaussiano débil centrado en Z_TRANS (8.9 ± 0.4 en A.3)

Los datasets reales deben colocarse en `data/` (ver
scripts/download_data.py). Sin ellos, el único dataset disponible es la
AUTOPRUEBA SINTÉTICA (synthetic_selftest_dataset), que se genera desde el
propio modelo: sirve para verificar la maquinaria de inferencia y es
circular por construcción — no dice nada sobre el universo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mcmc_ontology import constants as C
from .background import H_of_z

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
C_KMS = 299792.458  # velocidad de la luz [km/s]

# Priors (v35, F.4): H0 calibrado con prior N(67.4, 5²)
PRIOR_H0_MEAN = 67.4
PRIOR_H0_SIGMA = 5.0


@dataclass
class HzData:
    z: np.ndarray
    H: np.ndarray
    sig: np.ndarray


@dataclass
class SNeData:
    z: np.ndarray
    mu: np.ndarray
    cov_inv: np.ndarray  # inversa de la covarianza STAT+SYS


@dataclass
class BAOData:
    z: np.ndarray
    kind: list[str]      # "DM_over_rd" | "DH_over_rd"
    value: np.ndarray
    sig: np.ndarray
    r_d: float = 147.09  # Mpc (fiducial; puede promoverse a parámetro)


# ---------------------------------------------------------------------
# Carga de datos reales (skip claro si data/ está vacío)
# ---------------------------------------------------------------------

def load_Hz_data(path: Path | str | None = None) -> HzData:
    """Carga H(z) de cronómetros cósmicos desde data/boss_eboss/hz_cc.txt.

    Lanza FileNotFoundError con instrucciones si el fichero no existe.
    """
    p = Path(path) if path else _DATA_DIR / "boss_eboss" / "hz_cc.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"No existe {p}. Genera las tablas con "
            "`python scripts/download_data.py boss_eboss`."
        )
    arr = np.loadtxt(p)
    return HzData(z=arr[:, 0], H=arr[:, 1], sig=arr[:, 2])


def load_bao_data(path: Path | str | None = None) -> BAOData:
    """Carga el consenso BAO desde data/boss_eboss/bao_dr12.txt."""
    p = Path(path) if path else _DATA_DIR / "boss_eboss" / "bao_dr12.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"No existe {p}. Genera las tablas con "
            "`python scripts/download_data.py boss_eboss`."
        )
    z, kind, value, sig = [], [], [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        z.append(float(parts[0])); kind.append(parts[1])
        value.append(float(parts[2])); sig.append(float(parts[3]))
    return BAOData(z=np.asarray(z), kind=kind,
                   value=np.asarray(value), sig=np.asarray(sig))


def load_sne_data(dat_path: Path | str | None = None,
                  cov_path: Path | str | None = None,
                  z_min: float = 0.023) -> SNeData:
    """Carga Pantheon+ (distancias + covarianza STAT+SYS) desde data/pantheon.

    Requiere haber ejecutado `python scripts/download_data.py pantheon`.
    Se filtran z < z_min (pico de velocidades peculiares) de forma estándar.
    """
    dat = Path(dat_path) if dat_path else _DATA_DIR / "pantheon" / "PantheonPlusSH0ES.dat"
    cov = Path(cov_path) if cov_path else _DATA_DIR / "pantheon" / "PantheonPlusSH0ES_STATSYS.cov"
    if not dat.exists() or not cov.exists():
        raise FileNotFoundError(
            f"Faltan {dat.name} y/o {cov.name} en data/pantheon/. "
            "Descárgalos con `python scripts/download_data.py pantheon`."
        )
    rows = dat.read_text(encoding="utf-8").splitlines()
    header = rows[0].split()
    iz = header.index("zHD")
    imu = header.index("MU_SH0ES")
    z_all, mu_all = [], []
    for line in rows[1:]:
        parts = line.split()
        z_all.append(float(parts[iz])); mu_all.append(float(parts[imu]))
    z_all = np.asarray(z_all); mu_all = np.asarray(mu_all)
    raw = np.loadtxt(cov)
    n = int(raw[0]); cov_m = raw[1:].reshape(n, n)
    mask = z_all >= z_min
    cov_m = cov_m[np.ix_(mask, mask)]
    return SNeData(z=z_all[mask], mu=mu_all[mask],
                   cov_inv=np.linalg.inv(cov_m))


# ---------------------------------------------------------------------
# Distancias
# ---------------------------------------------------------------------

def comoving_distance(z: np.ndarray, theta: Sequence[float]) -> np.ndarray:
    """D_C(z) [Mpc] integrando c/H(z') con trapecios (universo plano)."""
    H0, Om, eps, z_trans = theta
    z = np.atleast_1d(np.asarray(z, dtype=float))
    grid = np.linspace(0.0, float(z.max()), 2048)
    integrand = C_KMS / H_of_z(grid, H0=H0, Omega_m=Om, eps=eps, z_trans=z_trans)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
    return np.interp(z, grid, cum)


def distance_modulus(z: np.ndarray, theta: Sequence[float]) -> np.ndarray:
    """μ(z) = 5·log10(d_L / 10 pc), con d_L = (1+z)·D_C."""
    d_L = (1.0 + np.asarray(z)) * comoving_distance(z, theta)  # Mpc
    return 5.0 * np.log10(np.maximum(d_L, 1e-12) * 1e5)


# ---------------------------------------------------------------------
# Priors y likelihoods
# ---------------------------------------------------------------------

def log_prior(theta: Sequence[float]) -> float:
    H0, Om, eps, z_trans = theta
    if not (60.0 < H0 < 80.0):
        return -np.inf
    if not (0.20 < Om < 0.40):
        return -np.inf
    if not (-0.05 < eps < 0.10):
        return -np.inf
    if not (1.0 < z_trans < 20.0):
        return -np.inf
    # v35 F.4: H0 ~ N(67.4, 5²)
    lp = -0.5 * ((H0 - PRIOR_H0_MEAN) / PRIOR_H0_SIGMA) ** 2
    # Gaussianos débiles centrados en los valores calibrados de constants:
    lp += -0.5 * ((eps - C.EPSILON_0) / 0.05) ** 2
    lp += -0.5 * ((z_trans - C.Z_TRANS) / 5.0) ** 2
    return lp


def log_like_Hz(theta: Sequence[float], data: HzData) -> float:
    H0, Om, eps, z_trans = theta
    H_pred = H_of_z(data.z, H0=H0, Omega_m=Om, eps=eps, z_trans=z_trans)
    return float(-0.5 * np.sum(((data.H - H_pred) / data.sig) ** 2))


def log_like_sne(theta: Sequence[float], data: SNeData) -> float:
    """SNe Ia con covarianza completa; M_B se margina analíticamente
    (perfil sobre el offset constante)."""
    mu_pred = distance_modulus(data.z, theta)
    r = data.mu - mu_pred
    Cinv = data.cov_inv
    A = float(r @ Cinv @ r)
    B = float(np.sum(Cinv @ r))
    D = float(np.sum(Cinv))
    return -0.5 * (A - B * B / D)


def log_like_bao(theta: Sequence[float], data: BAOData) -> float:
    H0, Om, eps, z_trans = theta
    chi2 = 0.0
    for z_i, kind, val, sig in zip(data.z, data.kind, data.value, data.sig):
        if kind == "DM_over_rd":
            pred = comoving_distance(z_i, theta).item() / data.r_d
        elif kind == "DH_over_rd":
            pred = C_KMS / (float(H_of_z(z_i, H0=H0, Omega_m=Om, eps=eps,
                                         z_trans=z_trans)) * data.r_d)
        else:
            raise ValueError(f"Tipo BAO desconocido: {kind}")
        chi2 += ((val - pred) / sig) ** 2
    return -0.5 * chi2


def log_prob(theta: Sequence[float], data: HzData,
             sne: SNeData | None = None,
             bao: BAOData | None = None) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_like_Hz(theta, data)
    if sne is not None:
        ll += log_like_sne(theta, sne)
    if bao is not None:
        ll += log_like_bao(theta, bao)
    return lp + ll


# ---------------------------------------------------------------------
# ΛCDM sobre los mismos datos (comparación justa) y criterios de información
# ---------------------------------------------------------------------

def theta_lcdm_to_mcmc(theta2: Sequence[float]) -> tuple:
    """(H0, Ωm) de ΛCDM → (H0, Ωm, ε=0, z_trans) del MCMC.

    Con ε = 0 el fondo es EXACTAMENTE ΛCDM (Prop. A.1, verificado en
    tests/test_recovery_limit.py), así que ΛCDM se ajusta con la misma
    maquinaria — misma integración de distancias, mismos likelihoods,
    misma marginalización de M_B — y la comparación es justa.
    """
    H0, Om = theta2
    return (H0, Om, 0.0, C.Z_TRANS)


def log_prior_lcdm(theta2: Sequence[float]) -> float:
    H0, Om = theta2
    if not (60.0 < H0 < 80.0):
        return -np.inf
    if not (0.20 < Om < 0.40):
        return -np.inf
    return -0.5 * ((H0 - PRIOR_H0_MEAN) / PRIOR_H0_SIGMA) ** 2


def log_prob_lcdm(theta2: Sequence[float], data: HzData,
                  sne: SNeData | None = None,
                  bao: BAOData | None = None) -> float:
    lp = log_prior_lcdm(theta2)
    if not np.isfinite(lp):
        return -np.inf
    theta = theta_lcdm_to_mcmc(theta2)
    ll = log_like_Hz(theta, data)
    if sne is not None:
        ll += log_like_sne(theta, sne)
    if bao is not None:
        ll += log_like_bao(theta, bao)
    return lp + ll


def information_criteria(k: int, n: int, loglike_max: float) -> dict:
    """AIC y BIC con la fórmula explícita.

        AIC = 2k − 2·ln(L_max)
        BIC = k·ln(n) − 2·ln(L_max)

    k = nº de parámetros ajustados del modelo, n = nº total de puntos de
    datos. La constante M_B de SNe se marginaliza analíticamente en ambos
    modelos por igual y no se cuenta en k de ninguno.
    """
    return {
        "k": k,
        "n": n,
        "loglike_max": loglike_max,
        "AIC": 2.0 * k - 2.0 * loglike_max,
        "BIC": k * np.log(n) - 2.0 * loglike_max,
    }


def run_emcee(data: HzData, nwalkers: int = 32, nsteps: int = 2000,
              seed: int = 42, sne: SNeData | None = None,
              bao: BAOData | None = None):
    """Run emcee sobre el modelo MCMC. Requiere `emcee`."""
    import emcee  # type: ignore

    rng = np.random.default_rng(seed)
    p0_center = np.array([C.H0_MCMC, 0.300, C.EPSILON_0, C.Z_TRANS])
    p0 = p0_center + 1e-3 * rng.normal(size=(nwalkers, 4))
    sampler = emcee.EnsembleSampler(
        nwalkers, 4, log_prob, args=(data, sne, bao)
    )
    sampler.run_mcmc(p0, nsteps, progress=False)
    return sampler


def synthetic_selftest_dataset(n: int = 32, seed: int = 0) -> HzData:
    """AUTOPRUEBA: dataset H(z) sintético generado desde el modelo nominal.

    Los datos salen de H_of_z con los propios parámetros de referencia,
    de modo que el ajuste que los recupera es circular por construcción.
    Sirve exclusivamente para verificar que la maquinaria de inferencia
    funciona; NO constituye contraste observacional (para eso: data/ y
    load_Hz_data/load_sne_data/load_bao_data).
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(0.05, 2.5, n)
    H = H_of_z(z)
    sig = 0.03 * H
    H_obs = H + sig * rng.normal(size=n)
    return HzData(z=z, H=H_obs, sig=sig)
