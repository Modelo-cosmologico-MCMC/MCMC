#!/usr/bin/env python
"""6A.5 — MCMC fondo-corregido vs ΛCDM sobre DESI DR2, con LOO completo.

PUERTA: exige benchmark.json con gate = PASS (6A.4) — si el ΛCDM
propio no reproduce el oficial, este contraste NO corre. Convenciones
IDÉNTICAS en ambos modelos: mismo sampler (emcee 32×4000, semilla
global 42 — cadenas bit-reproducibles), mismos priors comunes, mismo
tratamiento de r_d (H0·rd común — el MCMC no deriva r_d), misma
likelihood (validada contra Cobaya), mismo criterio de convergencia,
mismo minimizador de χ² (multistart acotado al soporte del prior;
argmin publicado, frontera declarada). Se ejecutan TODAS las
configuraciones preinscritas (DESI_ALL + 7 leave-one-bin-out) y se
publican todas — sin selección posterior del subconjunto favorable.

Además publica el bloque ESTRUCTURAL computado (revisión adversarial
6A): efecto bruto de ε sobre E(z) en el rango BAO — máximo sobre la
malla, no un punto — y residuo de χ² tras reabsorber (Ω_m, H0·rd);
y los percentiles del prior de ε TRUNCADO, la referencia correcta
para la afirmación de dominación por el prior.

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
    E_of_z,
    constrained_chi2_min,
    log_prob_lcdm,
    log_prob_mcmc,
    predict_desi_dr2_vector,
    run_fit,
)
from cosmology.desi_bao import (  # noqa: E402
    chi2_bao,
    load_desi_dr2_all,
    loo_configurations,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-18_desi_dr2_background")

EPS_PROBE = 0.05          # amplitud de sondeo del bloque estructural
Z_TRANS_REF = 8.9


def structural_block(data, om_ref: float, h0rd_ref: float) -> dict:
    """La razón estructural COMPUTADA, no afirmada: (a) máximo sobre
    z ∈ [0, 2.33] de la alteración relativa de E por ε = 0.05 (la
    primera versión de la ronda publicaba el valor en el único punto
    z = 2.33 como si fuera una cota del rango — hallazgo confirmado);
    (b) χ² entre los vectores DESI con y sin ε a parámetros fijos; y
    (c) el residuo tras reabsorber (Ω_m, H0·rd) — la degeneración que
    de verdad explica por qué BAO no mide ε en esta parametrización."""
    from scipy.optimize import minimize
    zg = np.linspace(0.0, 2.33, 4001)
    rel = np.abs(E_of_z(zg, om_ref, eps=EPS_PROBE, z_trans=Z_TRANS_REF)
                 / E_of_z(zg, om_ref, eps=0.0) - 1.0)
    _, _, _, _, C = data
    vec_eps = predict_desi_dr2_vector(om_ref, h0rd_ref, eps=EPS_PROBE,
                                      z_trans=Z_TRANS_REF, data=data)
    vec_0 = predict_desi_dr2_vector(om_ref, h0rd_ref, data=data)

    def cost(t):
        m = predict_desi_dr2_vector(t[0], t[1], data=data)
        return chi2_bao(m, vec_eps, C)

    res = minimize(cost, [om_ref, h0rd_ref], method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12,
                            "maxiter": 8000})
    return {
        "eps_probe": EPS_PROBE, "z_trans": Z_TRANS_REF,
        "omega_m_ref": om_ref, "H0rd_ref": h0rd_ref,
        "dE_rel_max_range": float(rel.max()),
        "z_at_max": float(zg[int(np.argmax(rel))]),
        "dE_rel_at_z233": float(rel[-1]),
        "chi2_distance_fixed_params": float(chi2_bao(vec_0, vec_eps, C)),
        "chi2_residual_after_reabsorbing": float(res.fun),
        "omega_m_compensating": float(res.x[0]),
        "H0rd_compensating": float(res.x[1]),
    }


def prior_eps_percentiles() -> dict:
    """Percentiles 16/50/84 del prior de ε REALMENTE muestreado
    (N(0.012, 0.05²) truncado a (−0.05, 0.10)) — la referencia para
    «dominado por el prior»; la mediana del truncado NO es 0.012."""
    from scipy.stats import truncnorm
    a = (-0.05 - 0.012) / 0.05
    b = (0.10 - 0.012) / 0.05
    p16, p50, p84 = truncnorm.ppf([0.16, 0.50, 0.84], a, b,
                                  loc=0.012, scale=0.05)
    return {"median": float(p50), "minus": float(p50 - p16),
            "plus": float(p84 - p50), "width_68": float(p84 - p16),
            "note": "N(0.012, 0.05²) truncada a (−0.05, 0.10)"}


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
    prior_eps = prior_eps_percentiles()
    rows = []
    ref = {}
    for name, idx in configs.items():
        n = int(idx.size)
        fl = run_fit(log_prob_lcdm, (0.30, 10200.0), 2, data, idx=idx,
                     nwalkers=32, nsteps=4000)
        min_l = constrained_chi2_min("lcdm", data, idx=idx,
                                     extra_starts=[fl["best"]])
        chi2_l = min_l["chi2"]
        fm = run_fit(log_prob_mcmc, (0.30, 10200.0, 0.012, 8.9), 4,
                     data, idx=idx, nwalkers=32, nsteps=4000)
        min_m = constrained_chi2_min("mcmc", data, idx=idx,
                                     extra_starts=[fm["best"]])
        chi2_m = min_m["chi2"]
        om_l = np.percentile(fl["flat"][:, 0], [16, 50, 84])
        om_m = np.percentile(fm["flat"][:, 0], [16, 50, 84])
        eps = np.percentile(fm["flat"][:, 2], [16, 50, 84])
        row = {
            "config": name, "n": n,
            "chi2_lcdm": chi2_l, "chi2_mcmc": chi2_m,
            "argmin_lcdm": min_l["theta"],
            "argmin_mcmc": min_m["theta"],
            "at_boundary_lcdm": min_l["at_boundary"],
            "at_boundary_mcmc": min_m["at_boundary"],
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
            "eps_width_ratio_vs_prior": float(
                (eps[2] - eps[0]) / prior_eps["width_68"]),
            "conv_lcdm": fl["converged"], "conv_mcmc": fm["converged"],
        }
        rows.append(row)
        if name == "DESI_ALL":
            ref = {"om_m": float(om_m[1]), "om_l": float(om_l[1]),
                   "eps": float(eps[1]), "chi2_l": chi2_l}
        print(f"{name:26s} n={n:2d}  χ²_Λ={chi2_l:7.3f}  "
              f"χ²_M={chi2_m:7.3f}  Δχ²={chi2_m - chi2_l:+7.3f}  "
              f"ΔBIC={row['delta_bic']:+7.3f}  "
              f"ε={eps[1]:+.4f}−{eps[1]-eps[0]:.4f}/+{eps[2]-eps[1]:.4f}"
              + ("  [frontera]" if min_m["at_boundary"] else ""))

    for row in rows:
        row["shift_omega_m_vs_ALL"] = row["omega_m_mcmc"] - ref["om_m"]
        row["shift_omega_m_lcdm_vs_ALL"] = (row["omega_m_lcdm"]
                                            - ref["om_l"])
        row["shift_eps_vs_ALL"] = row["eps_median"] - ref["eps"]

    # bloque estructural anclado en el benchmark (Ω_m, H0·rd medianos)
    bl = bdoc["lcdm_benchmark"]
    structural = structural_block(data, float(bl["Omega_m"]["median"]),
                                  float(bl["H0rd_kms"]["median"]))

    with open(OUT / "contrast.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    (OUT / "contrast.json").write_text(json.dumps(
        {"stage": "6A.5", "code_commit": sha,
         "gate_from": "benchmark.json (PASS verificado al arrancar)",
         "conventions": "idénticas ambos modelos: emcee 32×4000, "
                        "semilla global 42 (np.random.seed — cadenas "
                        "bit-reproducibles), priors comunes, H0·rd "
                        "común (r_d no derivado por el MCMC), "
                        "likelihood validada contra Cobaya 3.6.2, "
                        "χ²_min por multistart acotado al soporte del "
                        "prior (argmin y frontera publicados)",
         "prior_eps": prior_eps,
         "structural": structural,
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
