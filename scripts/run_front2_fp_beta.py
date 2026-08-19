#!/usr/bin/env python
"""Frente 2 — el espectro de la matriz de estabilidad con las β
derivadas de Fokker-Planck (Def. 4.4), bajo la preinscripción
congelada (results/2026-08-19_front2_fp_beta/preregistration.json).

Éste es el PRIMER punto del frente donde se computa un autovalor. La
clasificación A/B/C/D sigue la regla preinscrita; el barrido entero se
publica siempre. La extracción de s0 usa la maquinaria YA validada
(core/victoria_exponent.routes_coincide: espectral ⟺ dinámica).

Salida: results/2026-08-19_front2_fp_beta/{fp_beta.json, fp_beta.png,
report.md}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fokker_planck_beta import (  # noqa: E402
    beta_functions,
    canonical_closure,
    spinodal_canonical_point,
    stability_matrix,
    stability_matrix_quartic,
)
from core.victoria_exponent import (  # noqa: E402
    has_complex_pair,
    routes_coincide,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-19_front2_fp_beta")


def spectrum_summary(M: np.ndarray) -> dict:
    """Tipo espectral + (si hay par complejo) s0 por las DOS rutas
    validadas, Q = |Im/Re| y autovalores completos."""
    eigs = np.linalg.eigvals(M)
    out = {"eigenvalues_re": np.real(eigs).tolist(),
           "eigenvalues_im": np.imag(eigs).tolist(),
           "complex_pair": bool(has_complex_pair(M))}
    if out["complex_pair"]:
        rc = routes_coincide(M)
        i = int(np.argmax(np.abs(np.imag(eigs))))
        re_pair = float(np.real(eigs[i]))
        out.update({
            "s0_spectral": rc["s0_spectral"],
            "s0_dynamic": rc["s0_dynamic"],
            "routes_rel_error": rc["rel_error"],
            "routes_coincide": rc["coincide"],
            "re_of_pair": re_pair,
            "Q_im_over_re": (abs(rc["s0_spectral"] / re_pair)
                             if re_pair != 0.0 else float("inf")),
        })
    return out


def dD_dt(M0_sq: float, B: float, C0: float, a: float, b: float,
          sign: int) -> float:
    """dD/dt = 2B·β_B − 4C0·β_{M0²} − 4M0²·β_{C0} (Def. 8.4), con el
    signo del flujo declarado — Obs. 8.6 exige que el flujo HUNDA D."""
    bm, bb, bc = beta_functions(M0_sq, B, C0, a=a, b=b)
    return float(sign * (2.0 * B * bb - 4.0 * C0 * bm
                         - 4.0 * M0_sq * bc))


def main() -> None:
    cl = canonical_closure()
    lam = spinodal_canonical_point()
    prereg = json.loads((OUT / "preregistration.json")
                        .read_text(encoding="utf-8"))

    # --- punto canónico (g = 1): la primera mirada al espectro ------
    M_can = cl.flow_sign * stability_matrix(*lam, a=cl.a, b=cl.b)
    canonical = spectrum_summary(M_can)
    canonical["dD_dt"] = dD_dt(*lam, a=cl.a, b=cl.b, sign=cl.flow_sign)

    # --- barrido preinscrito g = a/b (b = 1/2 fijo; la escala global
    # es el diccionario τ, declarado) --------------------------------
    scan = []
    for g in cl.g_grid:
        a_g = g * cl.b
        M_g = cl.flow_sign * stability_matrix(*lam, a=a_g, b=cl.b)
        row = {"g": float(g), **spectrum_summary(M_g),
               "dD_dt": dD_dt(*lam, a=a_g, b=cl.b, sign=cl.flow_sign)}
        row.pop("eigenvalues_re")
        row.pop("eigenvalues_im")
        scan.append(row)

    # --- sistemático del truncamiento: variante cuártica en E4 = 0 --
    M_q = cl.flow_sign * stability_matrix_quartic(*lam, 0.0,
                                                  a=cl.a, b=cl.b)
    quartic = spectrum_summary(M_q)

    # --- clasificación por la regla preinscrita ---------------------
    lo, hi = cl.robustness_window
    window = [r for r in scan if lo <= r["g"] <= hi]
    types_in_window = {r["complex_pair"] for r in window}
    if len(types_in_window) > 1:
        outcome = "D"
    elif not canonical["complex_pair"]:
        outcome = "A"
    else:
        s0 = canonical["s0_spectral"] * cl.tau
        in_band = abs(s0 - cl.s0_target) <= cl.s0_band * cl.s0_target
        outcome = "C" if in_band else "B"
    tau_star = (float(cl.s0_target / canonical["s0_spectral"])
                if canonical["complex_pair"] else None)

    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    doc = {
        "stage": "frente 2 — espectro bajo preinscripción",
        "code_commit": sha,
        "prereg_commit": prereg["code_commit"],
        "closure": {"a": cl.a, "b": cl.b, "flow_sign": cl.flow_sign,
                    "tau": cl.tau, "truncation": cl.truncation},
        "evaluation_point": {"M0_sq": lam[0], "B": lam[1],
                             "C0": lam[2]},
        "canonical": canonical,
        "scan": scan,
        "quartic_E4_0": quartic,
        "outcome": outcome,
        "tau_star_for_lambda10": tau_star,
        "s0_target": cl.s0_target,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fp_beta.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(f"punto canónico: par complejo = {canonical['complex_pair']}")
    if canonical["complex_pair"]:
        print(f"  s0(τ=1) = {canonical['s0_spectral']:.6f} "
              f"(dinámica: {canonical['s0_dynamic']:.6f}; objetivo "
              f"π/ln10 = {cl.s0_target:.4f})")
        print(f"  Re(par) = {canonical['re_of_pair']:.6f}; "
              f"Q = {canonical['Q_im_over_re']:.4f}; τ* = {tau_star:.4f}")
    print(f"  dD/dt = {canonical['dD_dt']:+.4f} "
          f"(Obs. 8.6 exige < 0 para hundir D)")
    print(f"cuártico (E4=0): par complejo = {quartic['complex_pair']}"
          + (f", s0 = {quartic['s0_spectral']:.6f}"
             if quartic["complex_pair"] else ""))
    n_cplx = sum(r["complex_pair"] for r in scan)
    print(f"barrido: {n_cplx}/{len(scan)} puntos con par complejo")
    print(f"DESENLACE PREINSCRITO: {outcome}")

    # --- figura -----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gs = [r["g"] for r in scan]
        s0s = [r.get("s0_spectral", np.nan) for r in scan]
        qs = [r.get("Q_im_over_re", np.nan) for r in scan]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.semilogx(gs, s0s, "o-", label="s0(τ=1)")
        ax1.axhline(cl.s0_target, color="crimson", ls="--",
                    label="π/ln10 (λ=10)")
        ax1.axvline(1.0, color="gray", lw=0.8, ls=":")
        ax1.axvspan(*cl.robustness_window, alpha=0.12, color="green",
                    label="ventana de robustez")
        ax1.set_xlabel("g = a/b")
        ax1.set_ylabel("s0 = |Im μ| (τ = 1)")
        ax1.set_title("Exponente de Victoria desde las β de F-P")
        ax1.legend(fontsize=8)
        ax2.semilogx(gs, qs, "s-", color="darkorange")
        ax2.axvline(1.0, color="gray", lw=0.8, ls=":")
        ax2.set_xlabel("g = a/b")
        ax2.set_ylabel("Q = |Im μ| / |Re μ| (invariante de τ)")
        ax2.set_title("Calidad de la rotación del walking")
        fig.tight_layout()
        fig.savefig(OUT / "fp_beta.png", dpi=140)
        print(f"Figura: {OUT / 'fp_beta.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible")
    print(f"Artefacto: {OUT / 'fp_beta.json'}")


if __name__ == "__main__":
    main()
