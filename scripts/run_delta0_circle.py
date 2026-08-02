#!/usr/bin/env python
"""El círculo de δ₀ (ronda 5, prioridad nº 1) — el desenlace, sea cual sea.

¿Converge el mapa de retorno de Victoria (con Techo, en la región
fértil) al δ_H ≈ 0.0581 que el empalme C¹ midió (H.2.4, B7)? El cálculo
establece que en toda la región fértil el atractor es el Techo,
δ∞ = (W_max/c̄)^{1/3}, así que el círculo se cierra ⟺ W_max = c̄·δ_H³:
una ecuación de consistencia que liga el Teo. 10.6 con la ec. H.8 y
transfiere la pregunta a W_max (que el tratado no cuantifica).

Uso: python scripts/run_delta0_circle.py [--n 40000]

Salida: results/2026-08-02_delta0_circle/report.md (+ figura si hay
matplotlib) — versionados: son el desenlace publicable de la ronda 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.basal import c_bar  # noqa: E402
from core.delta0_circle import (  # noqa: E402
    STATUS_CIRCLE,
    W_max_required,
    attractor_numeric,
    closure_test,
    consistency_factors,
    delta0_H,
    landscape_scan,
)
from core.fertility_map import O1_RANGE  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-02_delta0_circle")

D_MIN = 1e-4          # suelo de activación (parámetro declarado del cálculo)
GAMMA_FERTILE = 1.5   # γR de demostración en la región fértil (A = 1.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--gamma-max", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260802)
    opts = ap.parse_args()

    d_H = delta0_H()
    W_close = W_max_required(d_H)
    print(f"δ_H (empalme, formas fiduciales) = {d_H:.5f}")
    print(f"W_max que cierra el círculo: c̄·δ_H³ = {W_close:.4e} "
          f"(c̄ = {c_bar():.4f})")

    # 1) El atractor, medido: desde el 0.012 heredado, con el Techo en
    #    el valor de cierre — ¿llega a δ_H y en cuántas vueltas?
    run = attractor_numeric(GAMMA_FERTILE, W_close, 0.012, delta_min=D_MIN)
    print(f"Atractor medido (γR={GAMMA_FERTILE}, δ_init=0.012): "
          f"{run['attractor']:.5f} en {run['cycles_to_settle']} vueltas")

    # 2) Control negativo: con otro Techo el atractor es OTRO número.
    bad = closure_test(0.5 * W_close, gamma_R=GAMMA_FERTILE,
                       delta_min=D_MIN, delta_init=0.012)
    print(f"Control (W_max/2): atractor {bad['attractor']:.5f} ≠ δ_H "
          f"→ cierra: {bad['closes']}")

    # 3) Factores de consistencia si δ₀ = δ_H (vs. el 0.012 heredado).
    f = consistency_factors()
    print(f"T₀ ×{f['T0_ratio']:.1f}; m_θ² ×{f['m_theta_sq_ratio']:.1f}")

    # 4) La contención por paisaje (fertilidad re-centrada en el círculo).
    scan = landscape_scan(opts.n, seed=opts.seed, gamma_max=opts.gamma_max)
    pct_d = scan["delta_H_percentiles"]
    pct_w = scan["W_required_percentiles"]
    cf = scan["containment_fraction"]
    print(f"Paisajes fértiles: {scan['n_fertile']} "
          f"(fracción fértil {scan['fraction_fertile']:.3f})")
    d_H_min = float(scan["delta_H"].min())
    print(f"δ_H mediana {pct_d[50]:.4f} [5-95%: {pct_d[5]:.4f}"
          f"-{pct_d[95]:.4f}]; mínimo muestreado {d_H_min:.4f}")
    print(f"Contención con W_max = {W_close:.3e}: {cf(W_close):.3f}")

    OUT.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        # Izquierda: la espiral hacia el Techo (historia del retorno)
        hist = run["history"]
        axes[0].plot(range(len(hist)), hist, "o-", color="#2f6b4f",
                     label=f"γR = {GAMMA_FERTILE}, W_max = c̄·δ_H³")
        axes[0].axhline(d_H, ls="--", color="k", lw=1.2,
                        label=f"δ_H (empalme) = {d_H:.4f}")
        axes[0].axhline(0.012, ls=":", color="#8a4a2f", lw=1.2,
                        label="0.012 (ε_Λ heredado como δ₀ en v32)")
        axes[0].set_xlabel("vuelta del ciclo")
        axes[0].set_ylabel(r"$\delta$")
        axes[0].set_title("El retorno de Victoria sube al Techo")
        axes[0].legend(fontsize=8)
        # Derecha: distribución de δ_H sobre paisajes fértiles
        axes[1].hist(scan["delta_H"], bins=60, color="#4a6b8a", alpha=0.8)
        axes[1].axvline(d_H, ls="--", color="k", lw=1.2,
                        label=f"fiducial {d_H:.4f}")
        axes[1].set_xlabel(r"$\delta_H$ = 0.130/$\sqrt{\bar b^2-4C_0\bar m^2}$")
        axes[1].set_ylabel("paisajes fértiles")
        axes[1].set_title("δ_H sobre el paisaje O(1) fértil")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "delta0_circle.png", dpi=140)
        print(f"Figura: {OUT / 'delta0_circle.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    w_grid = [0.1 * W_close, 0.5 * W_close, W_close, 2.0 * W_close,
              10.0 * W_close]
    containment_rows = "\n".join(
        f"| {w:.3e} | {cf(w):.3f} |" for w in w_grid)

    (OUT / "report.md").write_text(
        "# El círculo de δ₀ — ronda 5, prioridad nº 1\n\n"
        "La pregunta: ¿converge el mapa de retorno de Victoria (Teo. "
        "10.6, con Techo, en la región fértil) al δ_H ≈ 0.0581 que el "
        "empalme C¹ midió sin m_H como entrada (H.2.4, B7)?\n\n"
        "## Lo que el cálculo establece\n\n"
        "Con el mapa lineal + Techo de H.2.3, en TODA la región fértil "
        "(A > 1) el atractor es el Techo, independiente de γR y del δ "
        "inicial:\n\n"
        "    δ∞ = δ_sat = (W_max/c̄)^{1/3}\n\n"
        f"Medido iterando `core/victoria.py`: desde δ = 0.012 con γR = "
        f"{GAMMA_FERTILE} y W_max = c̄·δ_H³, el retorno llega a "
        f"δ∞ = {run['attractor']:.5f} en {run['cycles_to_settle']} "
        f"vueltas (δ_H = {d_H:.5f}). Control negativo: con W_max/2 el "
        f"atractor cae a {bad['attractor']:.5f} — el 0.058 NO es "
        "intrínseco al mapa.\n\n"
        "## El desenlace (publicable tal cual)\n\n"
        f"**El círculo se cierra ⟺ W_max = c̄·δ_H³ = {W_close:.4e}** "
        f"(formas fiduciales, c̄ = {c_bar():.4f}). Es una ECUACIÓN DE "
        "CONSISTENCIA que liga el Techo de Victoria (Lema 10.3 / Teo. "
        "10.6) con el empalme C¹ (ec. H.8). El tratado declara el Techo "
        "pero no asigna valor numérico a W_max, de modo que este "
        "cálculo no cierra ni rompe el círculo: TRANSFIERE la pregunta "
        "a W_max, que espera la microdinámica del reinicio (frente "
        "abierto nº 4). Ambos desenlaces (cierre o Silencio) siguen "
        "abiertos y ambos serían publicables.\n\n"
        f"Estatuto: {STATUS_CIRCLE}.\n\n"
        "## Factores de consistencia si δ₀ = δ_H\n\n"
        "Frente al 0.012 de ε_Λ que el v32 identificaba con δ₀ "
        "(identificación superada — regla canónica):\n\n"
        "| Magnitud | Ley | Factor |\n|---|---|---|\n"
        f"| T₀ (Prop. 3.4) | ×(δ_H/0.012)³ | ×{f['T0_ratio']:.1f} |\n"
        f"| m_θ² (ec. 10.1) | ×(δ_H/0.012)^{{5/2}} | "
        f"×{f['m_theta_sq_ratio']:.1f} |\n\n"
        "Estas dos escalas son las consecuencias a rastrear en el "
        "tratado si el círculo se cerrara en δ_H.\n\n"
        "## Fertilidad re-centrada: la contención por paisaje\n\n"
        "La condición de fertilidad ν > 0 ⟺ A > 1 (H.7) NO depende de "
        "δ₀ — el barrido del frente nº 4 no cambia. Lo que sí varía por "
        "paisaje es la CONTENCIÓN: W_max ≥ c̄(formas)·δ_H(formas)³.\n\n"
        f"Monte Carlo (semilla {opts.seed}, formas O(1) en {O1_RANGE}, "
        f"γR ~ U(0, {opts.gamma_max}]): {scan['n_fertile']} paisajes "
        f"fértiles (fracción {scan['fraction_fertile']:.3f}).\n\n"
        f"- δ_H: mediana {pct_d[50]:.4f}, [5%, 95%] = "
        f"[{pct_d[5]:.4f}, {pct_d[95]:.4f}], mínimo muestreado "
        f"{d_H_min:.4f} — la alternativa «formas no fiduciales» del "
        "desenlace de B7, cuantificada: en TODA la región O(1) "
        "muestreada δ_H = O(0.05); el discriminante máximo alcanzable "
        "acota δ_H ≳ 0.033, así que NINGUNA forma O(1) cierra el "
        "empalme en δ₀ = 0.012 — el 0.012 de ε_Λ queda fuera del rango "
        "alcanzable, en refuerzo independiente de la regla canónica.\n"
        f"- W_max requerido: mediana {pct_w[50]:.3e}, [5%, 95%] = "
        f"[{pct_w[5]:.3e}, {pct_w[95]:.3e}].\n\n"
        "Fracción de paisajes fértiles CONTENIDOS según W_max:\n\n"
        "| W_max | fracción contenida |\n|---|---|\n"
        f"{containment_rows}\n\n"
        "Parámetros declarados del cálculo: suelo δ_min = "
        f"{D_MIN:g} (Teo. 10.6 lo exige > 0 pero no lo cuantifica); "
        f"γ_max = {opts.gamma_max} (cartografía, no derivación).\n",
        encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
