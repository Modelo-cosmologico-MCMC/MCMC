#!/usr/bin/env python
"""El diagrama de fertilidad (frente nº 4, fig. H.2) — cartografía, no veredicto.

Monte Carlo sobre la región O(1) de las constantes de forma con γR
acotado, fracción fértil y figura análoga a la fig. H.2 del tratado.

Uso: python scripts/run_fertility_map.py [--n 40000] [--gamma-max 3.0]

Salida: figura y resumen en results/2026-08-01_fertility/ (versionados:
son la cartografía publicable del frente).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fertility_map import O1_RANGE, fertility_fraction  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "2026-08-01_fertility"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--gamma-max", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260801)
    opts = ap.parse_args()

    res = fertility_fraction(opts.n, seed=opts.seed,
                             gamma_max=opts.gamma_max)
    frac = res["fraction_fertile"]
    n_eff = res["n_effective"]
    print(f"Paisajes viables muestreados: {n_eff} "
          f"(formas O(1) en {O1_RANGE}, con (3.3) y c̄>0)")
    print(f"Fracción fértil (ν > 0) con γR ~ U(0, {opts.gamma_max}]: "
          f"{frac:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)

    # Figura análoga a la fig. H.2: plano (m̄²/ē, γR) con la frontera ν=0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = res["gamma_frontier"]          # γR* = m̄²/ē por muestra
        y = res["gamma_R"]
        fert = res["fertile"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x[~fert], y[~fert], s=3, alpha=0.25, color="#8a4a2f",
                   label="estéril (ν < 0) → Silencio")
        ax.scatter(x[fert], y[fert], s=3, alpha=0.25, color="#2f6b4f",
                   label="fértil (ν > 0) → eterno retorno")
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, grid, "k--", lw=1.5, label="frontera ν = 0 (γR = m̄²/ē)")
        ax.set_xlabel(r"$\bar{m}^2/\bar{e}$  (umbral de la ganancia)")
        ax.set_ylabel(r"$\gamma_R$  (ganancia del reinicio)")
        ax.set_title("Diagrama de fertilidad del Ciclo de Victoria "
                     "(frente nº 4; análogo a fig. H.2)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fertility_map.png", dpi=140)
        print(f"Figura: {OUT / 'fertility_map.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    (OUT / "summary.md").write_text(
        "# Diagrama de fertilidad (frente abierto nº 4)\n\n"
        f"Monte Carlo con semilla {opts.seed}: {n_eff} paisajes viables "
        f"(formas O(1) en {O1_RANGE} con casi-cancelación (3.3) y c̄>0), "
        f"γR ~ U(0, {opts.gamma_max}].\n\n"
        f"**Fracción fértil (ν > 0): {frac:.3f}.** Frontera ν = 0 en "
        "γR* = m̄²/ē (ec. H.7).\n\n"
        "Lectura honesta: esto CARTOGRAFÍA cuán genérico sería el eterno "
        "retorno sobre la región O(1) — no decide el signo de ν, que "
        "espera el valor real de γR desde la microdinámica del reinicio "
        "(frente nº 4). γ_max es un parámetro declarado de la "
        "cartografía, no una derivación del Techo.\n",
        encoding="utf-8")
    print(f"Resumen: {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
