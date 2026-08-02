"""Plots cosmológicos: H(z), Λ_rel(z), fσ_8(z)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from cosmology.background import H_of_z, Lambda_rel
from cosmology.perturbations import f_sigma8


def plot_Hz(z_max: float = 3.0, savepath: str | None = None):
    z = np.linspace(0, z_max, 200)
    fig, ax = plt.subplots()
    ax.plot(z, H_of_z(z), label="MCMC")
    ax.set_xlabel("z"); ax.set_ylabel("H(z) [km/s/Mpc]")
    ax.legend()
    if savepath: fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_Lambda(z_max: float = 20.0, savepath: str | None = None):
    z = np.linspace(0, z_max, 200)
    fig, ax = plt.subplots()
    ax.plot(z, Lambda_rel(z))
    ax.set_xlabel("z"); ax.set_ylabel(r"$\Omega_\Lambda^{\rm rel}(z)$")
    ax.set_title("Energía oscura dinámica MCMC")
    if savepath: fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_fsigma8(z_max: float = 2.0, savepath: str | None = None):
    z = np.linspace(0.05, z_max, 100)
    fig, ax = plt.subplots()
    ax.plot(z, f_sigma8(z))
    ax.set_xlabel("z"); ax.set_ylabel(r"$f\sigma_8(z)$")
    if savepath: fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
