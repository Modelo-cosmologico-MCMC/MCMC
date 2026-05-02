"""Ajuste bayesiano emcee para los parámetros MCMC.

Parámetros libres: (H0, Omega_m, eps, z_trans).
Likelihoods: gaussianas sobre H(z) (cosmic chronometers + BAO),
SNe Ia (Pantheon binned) y CMB (geometría compresada).

Esta es la implementación de referencia simplificada; los datasets reales
deben colocarse en `data/` antes de un run de producción.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mcmc_ontology import constants as C
from .background import H_of_z


@dataclass
class HzData:
    z: np.ndarray
    H: np.ndarray
    sig: np.ndarray


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
    # Prior gaussiano débil sobre eps, z_trans (centrados en valores nominales)
    lp = -0.5 * ((eps - 0.012) / 0.05) ** 2
    lp += -0.5 * ((z_trans - 8.9) / 5.0) ** 2
    return lp


def log_like_Hz(theta: Sequence[float], data: HzData) -> float:
    H0, Om, eps, z_trans = theta
    H_pred = H_of_z(data.z, H0=H0, Omega_m=Om, eps=eps, z_trans=z_trans)
    return float(-0.5 * np.sum(((data.H - H_pred) / data.sig) ** 2))


def log_prob(theta: Sequence[float], data: HzData) -> float:
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like_Hz(theta, data)


def run_emcee(data: HzData, nwalkers: int = 32, nsteps: int = 2000,
              seed: int = 42):
    """Run emcee sobre el modelo MCMC. Requiere `emcee`."""
    import emcee  # type: ignore

    rng = np.random.default_rng(seed)
    p0_center = np.array([C.H0_MCMC, 0.300, C.EPSILON_0, C.Z_TRANS])
    p0 = p0_center + 1e-3 * rng.normal(size=(nwalkers, 4))
    sampler = emcee.EnsembleSampler(
        nwalkers, 4, log_prob, args=(data,)
    )
    sampler.run_mcmc(p0, nsteps, progress=False)
    return sampler


def mock_Hz_dataset(n: int = 32, seed: int = 0) -> HzData:
    """Dataset H(z) sintético generado a partir del modelo nominal."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0.05, 2.5, n)
    H = H_of_z(z)
    sig = 0.03 * H
    H_obs = H + sig * rng.normal(size=n)
    return HzData(z=z, H=H_obs, sig=sig)
