#!/usr/bin/env python
"""ADENDA del frente 2 (corrección de la revisión adversarial,
19-ago): el barrido físico en δ0 que el punto preinscrito escondía.

El hallazgo HIGH confirmado: las β cumplen la covarianza exacta
β(D_s·λ; a, b) = σ·D_s·β(λ; a, b·k) con D_s = diag(kσ, kσ², kσ³), de
modo que el eje g del barrido preinscrito ES el eje δ0 del escalado
(3.2): g_ef = δ0⁻³ (m̄ = 1, C0 = 1). El punto preinscrito (1, 2, 1)
fija implícitamente δ0 ≈ 1 — fuera del régimen perturbativo del
corpus. Esta adenda evalúa la MISMA derivación canónica congelada
(sin tocar coeficientes, banda ni malla — la preinscripción queda
intacta) sobre la familia espinodal FÍSICA (m̄²δ0², 2m̄√C0·δ0, C0):

- tipo espectral, s0(τ=1), Q y dD/dt a lo largo de δ0;
- la frontera de complejificación δ0* (vía las raíces del
  discriminante del polinomio característico — álgebra, no
  optimización);
- τ*(δ0) para λ = 10, publicado como dato (el corpus no fija τ);
- la clasificación CONTRAFACTUAL con las reglas preinscritas
  aplicadas al punto físico δ0 = 0.1 — declarada como contrafactual:
  el desenlace A del punto preinscrito NO se reclasifica.

Salida: results/2026-08-19_front2_fp_beta/fp_beta_delta0.{json,png}.
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
    g_effective,
    physical_spinodal_point,
    stability_matrix,
)
from core.victoria_exponent import has_complex_pair  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-19_front2_fp_beta")

DELTA0_GRID = np.round(np.concatenate([
    np.arange(0.05, 0.61, 0.05), [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]),
    10).tolist()
DELTA0_PHYSICAL = 0.1        # el punto físico de referencia declarado


def char_poly_coeffs_in_g():
    """Coeficientes (en g) del polinomio característico de
    M((1,2,1); a = g/2, b = 1/2): μ³ + p·μ² + q(g)·μ + r(g), con
    p = 12, q = 44 − 176g, r = 48 − 480g + 288g² (álgebra verificada
    por test contra np.linalg.eigvals)."""
    return {"p": np.array([12.0]),
            "q": np.array([-176.0, 44.0]),      # 44 − 176g
            "r": np.array([288.0, -480.0, 48.0])}  # 288g² − 480g + 48


def complexification_boundary_g() -> float:
    """g*: la raíz real del discriminante del cúbico característico
    (Δ = 18pqr − 4p³r + p²q² − 4q³ − 27r²) en g ∈ (0.1, 100) — por
    np.roots sobre el polinomio Δ(g), sin ningún optimizador."""
    P = np.polynomial.polynomial
    cp = char_poly_coeffs_in_g()
    p, q, r = np.array([12.0]), cp["q"][::-1], cp["r"][::-1]
    # (convención polynomial: coef ascendentes; q = [44, −176] etc.)
    term1 = 18.0 * P.polymul(P.polymul(p, q), r)
    term2 = -4.0 * P.polymul(P.polymul(P.polymul(p, p), p), r)
    term3 = P.polymul(P.polymul(p, p), P.polymul(q, q))
    term4 = -4.0 * P.polymul(P.polymul(q, q), q)
    term5 = -27.0 * P.polymul(r, r)
    n = max(len(t) for t in (term1, term2, term3, term4, term5))

    def pad(t):
        return np.pad(t, (0, n - len(t)))

    disc = pad(term1) + pad(term2) + pad(term3) + pad(term4) + pad(term5)
    roots = np.roots(disc[::-1])
    real = [float(z.real) for z in roots
            if abs(z.imag) < 1e-9 and 0.1 < z.real < 100.0]
    if len(real) != 1:
        raise RuntimeError(f"raíces reales en (0.1, 100): {real}")
    return real[0]


def point_summary(delta0: float, a: float, b: float,
                  sign: int) -> dict:
    lam = physical_spinodal_point(delta0)
    M = sign * stability_matrix(*lam, a=a, b=b)
    eigs = np.linalg.eigvals(M)
    bm, bb, bc = beta_functions(*lam, a=a, b=b)
    dD = float(sign * (2.0 * lam[1] * bb - 4.0 * lam[2] * bm
                       - 4.0 * lam[0] * bc))
    row = {"delta0": float(delta0),
           "g_effective": float(g_effective(delta0)),
           "complex_pair": bool(has_complex_pair(M)),
           "dD_dt": dD}
    if row["complex_pair"]:
        s0 = float(np.abs(eigs.imag).max())
        i = int(np.argmax(np.abs(eigs.imag)))
        row.update({"s0_tau1": s0,
                    "re_of_pair": float(eigs[i].real),
                    "Q_im_over_re": float(abs(s0 / eigs[i].real)),
                    "tau_star_for_lambda10": float(
                        (np.pi / np.log(10.0)) / s0)})
    return row


def covariance_check(rng_seed: int = 7) -> float:
    """La identidad de covarianza, verificada numéricamente:
    max |β(D_s·λ; a, b) − σ·D_s·β(λ; a, b·k)| sobre sorteos."""
    rng = np.random.default_rng(rng_seed)
    worst = 0.0
    for _ in range(200):
        lam = rng.uniform(0.2, 3.0, size=3)
        a, b, k, sigma = rng.uniform(0.2, 2.0, size=4)
        Ds = np.array([k * sigma, k * sigma ** 2, k * sigma ** 3])
        lhs = beta_functions(*(Ds * lam), a=a, b=b)
        rhs = sigma * Ds * beta_functions(*lam, a=a, b=b * k)
        worst = max(worst, float(np.max(np.abs(lhs - rhs))))
    return worst


def main() -> None:
    cl = canonical_closure()
    g_star = complexification_boundary_g()
    delta0_star = g_star ** (-1.0 / 3.0)
    cov = covariance_check()

    rows = [point_summary(d0, cl.a, cl.b, cl.flow_sign)
            for d0 in DELTA0_GRID]
    phys = point_summary(DELTA0_PHYSICAL, cl.a, cl.b, cl.flow_sign)

    # clasificación CONTRAFACTUAL: las reglas congeladas aplicadas al
    # punto físico (ventana g ∈ [0.5, 2] alrededor del cierre, con el
    # punto en δ0 = 0.1; la banda y la precedencia, las preinscritas):
    window = [point_summary(DELTA0_PHYSICAL, g * cl.b, cl.b,
                            cl.flow_sign)
              for g in cl.robustness_window]
    types = {r["complex_pair"] for r in window} | {phys["complex_pair"]}
    if not phys["complex_pair"]:
        counterfactual = "A"
    elif len(types) > 1:
        counterfactual = "D"
    else:
        in_band = (abs(phys["s0_tau1"] * cl.tau - cl.s0_target)
                   <= cl.s0_band * cl.s0_target)
        counterfactual = "C" if in_band else "B"

    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    doc = {
        "stage": "adenda δ0 (corrección de la revisión adversarial — "
                 "la preinscripción queda intacta; el desenlace A del "
                 "punto preinscrito NO se reclasifica)",
        "code_commit": sha,
        "closure": {"a": cl.a, "b": cl.b, "flow_sign": cl.flow_sign,
                    "tau": cl.tau},
        "covariance_identity_max_abs_dev": cov,
        "g_star_complexification": g_star,
        "delta0_star": delta0_star,
        "delta0_physical_reference": DELTA0_PHYSICAL,
        "physical_point": phys,
        "counterfactual_outcome_at_physical_point": counterfactual,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fp_beta_delta0.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(f"covarianza: máx desviación = {cov:.2e}")
    print(f"g* = {g_star:.5f} ⟺ δ0* = {delta0_star:.5f}")
    print(f"punto físico δ0 = {DELTA0_PHYSICAL}: complejo = "
          f"{phys['complex_pair']}"
          + (f", s0(τ=1) = {phys['s0_tau1']:.4f}, "
             f"Q = {phys['Q_im_over_re']:.4f}, "
             f"τ* = {phys['tau_star_for_lambda10']:.4f}, "
             f"dD/dt = {phys['dD_dt']:+.3f}"
             if phys["complex_pair"] else ""))
    print(f"contrafactual (reglas preinscritas en el punto físico): "
          f"{counterfactual}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d0s = [r["delta0"] for r in rows]
        s0s = [r.get("s0_tau1", np.nan) for r in rows]
        dds = [r["dD_dt"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.plot(d0s, s0s, "o-")
        ax1.axvline(delta0_star, color="crimson", ls="--",
                    label=f"δ0* = {delta0_star:.3f}")
        ax1.axhline(np.pi / np.log(10.0), color="gray", ls=":",
                    label="π/ln10")
        ax1.set_xlabel("δ0")
        ax1.set_ylabel("s0(τ=1)")
        ax1.set_title("Familia espinodal física, cierre canónico")
        ax1.legend(fontsize=8)
        ax2.plot(d0s, dds, "s-", color="darkorange")
        ax2.axhline(0.0, color="k", lw=0.8)
        ax2.axvline(delta0_star, color="crimson", ls="--")
        ax2.set_xlabel("δ0")
        ax2.set_ylabel("dD/dt")
        ax2.set_title("Obs. 8.6: dD/dt < 0 (hundir D) para δ0 < δ0_D")
        fig.tight_layout()
        fig.savefig(OUT / "fp_beta_delta0.png", dpi=140)
        print(f"Figura: {OUT / 'fp_beta_delta0.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible")
    print(f"Artefacto: {OUT / 'fp_beta_delta0.json'}")


if __name__ == "__main__":
    main()
