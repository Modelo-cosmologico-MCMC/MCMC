"""Plot del espectro fermiónico predicho vs PDG."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mass_program.B4_masses import predict_fermion_masses


def plot_spectrum(savepath: str | None = None):
    """Compara m_MCMC vs m_PDG para los 9 fermiones cargados."""
    res = predict_fermion_masses()
    fermions = ["e", "mu", "tau", "u", "d", "s", "c", "b", "t"]
    m_mcmc = [res[f]["m_MCMC"] for f in fermions]
    m_pdg  = [res[f]["m_PDG"]  for f in fermions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fermions))
    ax.bar(x - 0.2, m_mcmc, width=0.4, label="MCMC", color="C0")
    ax.bar(x + 0.2, m_pdg,  width=0.4, label="PDG",  color="C1")
    ax.set_xticks(x); ax.set_xticklabels(fermions)
    ax.set_yscale("log")
    ax.set_ylabel("masa [GeV]")
    ax.legend()
    ax.set_title("Espectro fermiónico: MCMC vs PDG")
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
