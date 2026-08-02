"""Animación pre-geométrica S0 → S_{1.001}.

Genera una animación matplotlib mostrando la evolución del Campo de Adrián
y los sucesivos colapsos C0..C4. La salida puede exportarse a HTML interactivo.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from mcmc_ontology import constants as C
from mcmc_ontology.potential import V


def animate(out_html: str | None = None, frames: int = 120, lam: float = 0.01):
    """Animación 2D del potencial V(Φ; S) variando S.

    Si `out_html` se especifica, guarda la animación como HTML5 video.
    Devuelve la figura y la animación.
    """
    Phi = np.linspace(-2 * C.V_GEV["C3"], 2 * C.V_GEV["C3"], 400)
    S_grid = np.linspace(0.005, 1.05, frames)
    fig, ax = plt.subplots()
    line, = ax.plot(Phi, V(Phi, S_grid[0], lam=lam), lw=2)
    ax.set_xlabel("Φ_Ad [GeV]")
    ax.set_ylabel("V(Φ; S)")
    ax.set_yscale("symlog")
    title = ax.set_title(f"S = {S_grid[0]:.4f}")

    def _update(i):
        line.set_ydata(V(Phi, S_grid[i], lam=lam))
        title.set_text(f"S = {S_grid[i]:.4f}")
        return line, title

    anim = FuncAnimation(fig, _update, frames=len(S_grid), interval=50, blit=False)
    if out_html:
        with open(out_html, "w") as f:
            f.write(anim.to_jshtml())
    return fig, anim
