"""Visualización de transmisiones WKB |T_n^(i)|."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mass_program.B3_wkb import transmission_table


def plot_transmissions(savepath: str | None = None):
    """Heatmap log10 |T_n^(F_i)|."""
    T = transmission_table()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.log10(T), cmap="viridis", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["C1", "C2", "C3", "C4"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["F1", "F2", "F3"])
    ax.set_xlabel("Sello"); ax.set_ylabel("Familia")
    ax.set_title(r"$\log_{10}|T_n^{(F_i)}|$")
    fig.colorbar(im, ax=ax)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{T[i,j]:.2g}", ha="center", va="center",
                    color="w", fontsize=8)
    if savepath: fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
