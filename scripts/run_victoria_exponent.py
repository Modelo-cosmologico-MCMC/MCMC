#!/usr/bin/env python
"""Frente 2 instrumentado: la matriz de estabilidad de la ec. 14.2.

Tres medidas, con el hueco declarado:
(1) validación de la maquinaria — las dos rutas al exponente de
    Victoria (espectro ⟺ periodo del walking integrado) coinciden;
(2) genericidad de la cascada DSI con ansatz O(1) (robusta a la escala);
(3) la fracción en banda ±10% de s0 = π/ln10 — la medida de cuánto de
    «selección» es λ = 10 dentro del ansatz (y su dependencia declarada
    de la escala).

Uso: python scripts/run_victoria_exponent.py [--n 20000]

Salida: results/2026-08-05_victoria_exponent/report.md (+ figura).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.victoria_exponent import (  # noqa: E402
    S0_TARGET,
    STATUS_FRENTE2,
    o1_ansatz_scan,
    routes_coincide,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-05_victoria_exponent")
SCALES = (1.0, 1.5, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260805)
    opts = ap.parse_args()

    # (1) las dos rutas
    M3 = np.zeros((3, 3))
    M3[:2, :2] = np.array([[0.3, -S0_TARGET], [S0_TARGET, 0.3]])
    M3[2, 2] = -0.7
    M3[0, 2] = 0.4
    r = routes_coincide(M3)
    print(f"Dos rutas (3×3 con modo real acoplado): s0 espectral = "
          f"{r['s0_spectral']:.6f}, s0 dinámico = {r['s0_dynamic']:.6f} "
          f"(error {r['rel_error']:.2e}); λ = {r['lambda_spectral']:.4f}")

    # (2)-(3) barrido del ansatz
    scans = [o1_ansatz_scan(n=opts.n, scale=a, seed=opts.seed)
             for a in SCALES]
    for sc in scans:
        print(f"escala {sc['scale']}: DSI = {sc['frac_dsi']:.3f}; "
              f"s0 mediano = {sc['s0_median']:.3f}; banda ±10% = "
              f"{sc['frac_band']:.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for sc in scans:
            ax.hist(sc["s0_values"], bins=100, histtype="step", lw=1.5,
                    density=True,
                    label=(f"escala {sc['scale']}: DSI "
                           f"{sc['frac_dsi']:.0%}, banda "
                           f"{sc['frac_band']:.1%}"))
        ax.axvspan(0.9 * S0_TARGET, 1.1 * S0_TARGET, color="#2f6b4f",
                   alpha=0.15, label="banda ±10% de π/ln10")
        ax.axvline(S0_TARGET, ls="--", color="k", lw=1.2)
        ax.set_xlabel(r"$s_0 = |\mathrm{Im}\,\mu(M)|$")
        ax.set_ylabel("densidad (condicionada a DSI)")
        ax.set_xlim(0.0, 5.0)
        ax.set_title("Frente 2 (ec. 14.2): la DSI es genérica; "
                     "λ = 10 es una selección")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "victoria_exponent.png", dpi=140)
        print(f"Figura: {OUT / 'victoria_exponent.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    table = "\n".join(
        f"| {sc['scale']} | {sc['frac_dsi']:.3f} | {sc['s0_median']:.3f} "
        f"| {sc['frac_band']:.4f} |"
        for sc in scans)
    (OUT / "report.md").write_text(
        "# Frente 2 instrumentado: la matriz de estabilidad de la "
        "ec. 14.2\n\n"
        f"Semilla {opts.seed}, n = {opts.n} matrices por escala; ansatz "
        "DECLARADO: entradas i.i.d. U(−a, a) sobre λi = (M0², B, C0) — "
        "las β reales (Fokker-Planck, Def. 4.4) son el hueco del "
        "frente y este módulo queda listo para consumirlas.\n\n"
        "## 1. La maquinaria, validada\n\n"
        "Las dos rutas al exponente de Victoria coinciden sobre la "
        "matriz de prueba con modo real acoplado: s0 espectral = "
        f"{r['s0_spectral']:.6f} vs s0 dinámico (periodo del walking "
        f"del flujo integrado, unwrap del ángulo) = "
        f"{r['s0_dynamic']:.6f} — error relativo {r['rel_error']:.2e}; "
        f"λ = e^(π/s0) = {r['lambda_spectral']:.4f}.\n\n"
        "## 2-3. El barrido del ansatz O(1)\n\n"
        "| escala a | fracción DSI | s0 mediano | banda ±10% de "
        "π/ln10 |\n|---|---|---|---|\n"
        f"{table}\n\n"
        "- **La cascada DSI es GENÉRICA y robusta**: ~2/3 de las "
        "matrices O(1) complejifican sus exponentes, con independencia "
        "de la escala del ansatz (la complejidad del espectro es "
        "invariante de escala).\n"
        "- **λ = 10 es una SELECCIÓN, no una consecuencia**: la banda "
        "±10% de s0 = π/ln10 = 1.3644 captura solo unas unidades por "
        "ciento — y esa fracción DEPENDE de la escala del ansatz "
        "(s0 escala linealmente con a), así que ni siquiera es un "
        "número invariante: sin las β reales no hay predicción de λ. "
        "Exactamente la disyuntiva de §14.2: «si resulta λ ≠ 10, el "
        "diez era convención».\n\n"
        f"Estatuto: {STATUS_FRENTE2}.\n",
        encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
