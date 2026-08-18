#!/usr/bin/env python
"""6A.5 — MCMC fondo-corregido vs ΛCDM sobre DESI DR2, con LOO completo.

PUERTA: exige benchmark.json con gate = PASS (6A.4) — si el ΛCDM
propio no reproduce el oficial, este contraste NO corre. Convenciones
IDÉNTICAS en ambos modelos: mismo sampler (emcee, semilla 42), mismos
priors comunes, mismo tratamiento de r_d (H0·rd común — el MCMC no
deriva r_d), misma likelihood (validada contra Cobaya), mismo criterio
de convergencia. Se ejecutan TODAS las configuraciones preinscritas
(DESI_ALL + 7 leave-one-bin-out) y se publican todas — sin selección
posterior del subconjunto favorable.

Salida: results/2026-08-18_desi_dr2_background/{contrast.csv,
contrast.json, report.md, figuras}.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.desi_background_fit import (  # noqa: E402
    chi2_at,
    log_prob_lcdm,
    log_prob_mcmc,
    run_fit,
)
from cosmology.desi_bao import (  # noqa: E402
    load_desi_dr2_all,
    loo_configurations,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-18_desi_dr2_background")


def refine_chi2(theta0, model, data, idx):
    from scipy.optimize import minimize
    res = minimize(lambda t: chi2_at(t, model, data, idx=idx), theta0,
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10,
                            "maxiter": 4000})
    return float(res.fun), res.x


def main() -> None:
    bench = OUT / "benchmark.json"
    if not bench.exists():
        raise SystemExit("PUERTA: falta benchmark.json — correr "
                         "run_desi_dr2_benchmark.py primero")
    bdoc = json.loads(bench.read_text(encoding="utf-8"))
    if bdoc["lcdm_benchmark"]["gate"] != "PASS" \
            or bdoc["equivalence"]["status"] != "PASS":
        raise SystemExit("PUERTA: benchmark o equivalencia sin PASS — "
                         "el contraste no corre (regla predeclarada)")

    data = load_desi_dr2_all()
    _, _, _, bins, _ = data
    configs = loo_configurations(bins)
    rows = []
    ref = {}
    for name, idx in configs.items():
        n = int(idx.size)
        fl = run_fit(log_prob_lcdm, (0.30, 10200.0), 2, data, idx=idx,
                     nwalkers=32, nsteps=3000)
        chi2_l, best_l = refine_chi2(fl["best"], "lcdm", data, idx)
        fm = run_fit(log_prob_mcmc, (0.30, 10200.0, 0.012, 8.9), 4,
                     data, idx=idx, nwalkers=32, nsteps=4000)
        chi2_m, best_m = refine_chi2(fm["best"], "mcmc", data, idx)
        om_l = np.percentile(fl["flat"][:, 0], [16, 50, 84])
        om_m = np.percentile(fm["flat"][:, 0], [16, 50, 84])
        eps = np.percentile(fm["flat"][:, 2], [16, 50, 84])
        row = {
            "config": name, "n": n,
            "chi2_lcdm": chi2_l, "chi2_mcmc": chi2_m,
            "delta_chi2": chi2_m - chi2_l,
            "aic_lcdm": 2 * 2 + chi2_l, "aic_mcmc": 2 * 4 + chi2_m,
            "bic_lcdm": 2 * np.log(n) + chi2_l,
            "bic_mcmc": 4 * np.log(n) + chi2_m,
            "delta_aic": (2 * 4 + chi2_m) - (2 * 2 + chi2_l),
            "delta_bic": (4 * np.log(n) + chi2_m)
                         - (2 * np.log(n) + chi2_l),
            "omega_m_lcdm": float(om_l[1]),
            "omega_m_mcmc": float(om_m[1]),
            "eps_median": float(eps[1]),
            "eps_minus": float(eps[1] - eps[0]),
            "eps_plus": float(eps[2] - eps[1]),
            "conv_lcdm": fl["converged"], "conv_mcmc": fm["converged"],
        }
        rows.append(row)
        if name == "DESI_ALL":
            ref = {"om": float(om_m[1]), "eps": float(eps[1]),
                   "chi2_l": chi2_l}
        print(f"{name:26s} n={n:2d}  χ²_Λ={chi2_l:7.3f}  "
              f"χ²_M={chi2_m:7.3f}  Δχ²={chi2_m - chi2_l:+7.3f}  "
              f"ΔBIC={row['delta_bic']:+7.3f}  "
              f"ε={eps[1]:+.4f}−{eps[1]-eps[0]:.4f}/+{eps[2]-eps[1]:.4f}")

    for row in rows:
        row["shift_omega_m_vs_ALL"] = row["omega_m_mcmc"] - ref["om"]
        row["shift_eps_vs_ALL"] = row["eps_median"] - ref["eps"]

    with open(OUT / "contrast.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    (OUT / "contrast.json").write_text(json.dumps(
        {"stage": "6A.5", "code_commit": sha,
         "gate_from": "benchmark.json (PASS verificado al arrancar)",
         "conventions": "idénticas ambos modelos: emcee semilla 42 "
                        "(Λ: 32×3000, MCMC: 32×4000), priors comunes, "
                        "H0·rd común (r_d no derivado por el MCMC), "
                        "likelihood validada contra Cobaya 3.6.2",
         "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # Figura: Δχ²/ΔBIC y ε por configuración
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["config"].replace("DESI_", "") for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.4))
        x = np.arange(len(rows))
        ax1.bar(x - 0.2, [r["delta_chi2"] for r in rows], 0.38,
                label="Δχ² (MCMC − ΛCDM)")
        ax1.bar(x + 0.2, [r["delta_bic"] for r in rows], 0.38,
                label="ΔBIC")
        ax1.axhline(0, color="k", lw=0.8)
        ax1.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
        ax1.set_title("DESI DR2: MCMC vs ΛCDM por configuración "
                      "(positivo favorece a ΛCDM)")
        ax1.legend(fontsize=8)
        ax2.errorbar(x, [r["eps_median"] for r in rows],
                     yerr=[[r["eps_minus"] for r in rows],
                           [r["eps_plus"] for r in rows]],
                     fmt="o", capsize=3)
        ax2.axhline(0, color="k", lw=0.8, ls="--")
        ax2.axhline(0.012, color="gray", lw=0.8, ls=":",
                    label="0.012 del corpus")
        ax2.set_xticks(x, names, rotation=45, ha="right", fontsize=7)
        ax2.set_title("Posterior de ε_Λ por configuración")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "contrast.png", dpi=140)
        print(f"Figura: {OUT / 'contrast.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible")

    print(f"Artefactos: {OUT}/contrast.{{csv,json}}")


if __name__ == "__main__":
    main()
