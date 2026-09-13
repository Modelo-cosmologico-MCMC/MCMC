#!/usr/bin/env python
"""PRIMERA APLICACIÓN del pipeline SN a Dovekie real, bajo la doble
barrera (PASS de mocks #15 + preinscripción de esta aplicación). Ejecuta
los cinco brazos, clasifica el desenlace con la regla congelada y
publica cadenas (con semilla), JSON y report.

Uso: python scripts/run_dovekie_real.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.dovekie_real_fit import (  # noqa: E402
    OUTDIR,
    PARAM_NAMES,
    classify_outcome,
    constrained_chi2_min,
    criteria,
    load_prereg,
    load_real_data,
    n_points,
    prereg_sha256,
    run_fit,
    summarize,
)
from validation.dovekie_mocks import (  # noqa: E402
    lcdm_grid_mu,
    lcdm_posterior_percentiles,
)


def main() -> int:
    t0 = time.time()
    prereg = load_prereg()
    psha = prereg_sha256()
    rules = prereg["outcomes_enumerated"]["rules"]
    smp = prereg["sampler"]
    off = prereg["official_reference"]["Omega_m_weighted"]
    print(f"[prereg] {psha[:12]}…  barreras: mocks "
          f"{prereg['precondition_mock_validation']['mock_validation_sha256'][:12]}…")

    # ---- primera lectura del HD real: SOLO aquí, tras las barreras ----
    data_ss = load_real_data("STAT+SYS")
    data_so = load_real_data("STATONLY")
    assert data_ss["mock_validation_sha256"] == \
        prereg["precondition_mock_validation"]["mock_validation_sha256"], \
        "el PASS de mocks citado por la preinscripción no es el del árbol"
    n_tot = n_points(data_ss)

    # ---- brazo 1: benchmark SN-only ΛCDM (malla determinista) ----
    print("[1] benchmark SN-only ΛCDM (malla)…", flush=True)
    import validation.dovekie_mocks as dm
    lo, hi = prereg["priors_frozen"]["Omega_m"]["support"]
    dm.OMEGA_GRID = np.linspace(lo, hi, 4001)
    grid = lcdm_grid_mu(data_ss["zHD"], data_ss["zHEL"])
    pct, chi2_sn_min, om_hat = lcdm_posterior_percentiles(grid, data_ss["W"],
                                                          data_ss["mu"])
    bench = {"Omega_m_p50": pct[0.50], "p16": pct[0.16], "p84": pct[0.84],
             "p025": pct[0.025], "p975": pct[0.975],
             "sigma_own": 0.5 * (pct[0.84] - pct[0.16]),
             "chi2_sn_min": chi2_sn_min, "official_mean": off["mean"],
             "official_sd": off["sd"],
             "delta_sigma_official": (pct[0.50] - off["mean"]) / off["sd"]}
    print(f"    Ω_m = {pct[0.50]:.4f} +{pct[0.84]-pct[0.50]:.4f} "
          f"−{pct[0.50]-pct[0.16]:.4f} | oficial {off['mean']:.4f} ± "
          f"{off['sd']:.4f} → Δ = {bench['delta_sigma_official']:+.3f}σ",
          flush=True)

    # ---- brazos 2–5: conjuntos ----
    arms = {}
    for tag, model, data in (("lcdm_statsys", "lcdm", data_ss),
                             ("mcmc_statsys", "mcmc", data_ss),
                             ("lcdm_statonly", "lcdm", data_so),
                             ("mcmc_statonly", "mcmc", data_so)):
        print(f"[{tag}] emcee {smp['nwalkers']}×{smp['nsteps']}…", flush=True)
        fit = run_fit(model, data, nwalkers=int(smp["nwalkers"]),
                      nsteps=int(smp["nsteps"]), seed=int(smp["seed"]))
        cmin = constrained_chi2_min(model, data, extra_starts=(fit["best"],))
        crit = criteria(cmin["chi2"], model, n_tot)
        np.savez(OUTDIR / f"chains_{tag}.npz", flat=fit["flat"],
                 logp=fit["logp"], params=np.array(PARAM_NAMES[model]),
                 seed=fit["seed"], nwalkers=fit["nwalkers"],
                 nsteps=fit["nsteps"])
        arms[tag] = {"model": model, "cov_kind": data["cov_kind"],
                     "posterior": summarize(fit["flat"], PARAM_NAMES[model]),
                     "acceptance": fit["acceptance"], "tau": fit["tau"],
                     "converged": fit["converged"],
                     "chi2_min": cmin["chi2"], "chi2_min_theta": cmin["theta"],
                     "chi2_blocks_at_min": cmin["blocks"],
                     "at_boundary": cmin["at_boundary"],
                     "chi2_nu": cmin["chi2"] / (n_tot - crit["k"]),
                     "AIC": crit["AIC"], "BIC": crit["BIC"], "k": crit["k"]}
        post = arms[tag]["posterior"]
        print(f"    χ²_min = {cmin['chi2']:.2f} (ν = {n_tot - crit['k']}), "
              f"Ω_m = {post['Omega_m']['p50']:.4f} ± {post['Omega_m']['sd']:.4f}"
              + (f", ε = {post['epsilon']['p50']:+.4f} ± {post['epsilon']['sd']:.4f}"
                 if model == "mcmc" else "")
              + f", acept {fit['acceptance']:.2f}, conv {fit['converged']}",
              flush=True)

    def contrast(cov):
        L, M = arms[f"lcdm_{cov}"], arms[f"mcmc_{cov}"]
        eps = M["posterior"]["epsilon"]
        return {"dAIC_mcmc_minus_lcdm": M["AIC"] - L["AIC"],
                "dBIC_mcmc_minus_lcdm": M["BIC"] - L["BIC"],
                "dchi2_min_gain": L["chi2_min"] - M["chi2_min"],
                "eps_p50": eps["p50"], "eps_sd": eps["sd"],
                "eps_ci95": [eps["p025"], eps["p975"]],
                "eps_ci95_contains_zero": bool(eps["p025"] <= 0.0 <= eps["p975"]),
                "chi2_nu_lcdm": L["chi2_nu"]}
    c_ss, c_so = contrast("statsys"), contrast("statonly")
    outcome = classify_outcome(bench["delta_sigma_official"], c_ss["chi2_nu_lcdm"],
                               c_ss["dBIC_mcmc_minus_lcdm"], c_ss["eps_sd"],
                               c_ss["eps_ci95_contains_zero"], rules)
    print(f"\n[STAT+SYS] ΔAIC = {c_ss['dAIC_mcmc_minus_lcdm']:+.2f}, ΔBIC = "
          f"{c_ss['dBIC_mcmc_minus_lcdm']:+.2f}, ε = {c_ss['eps_p50']:+.4f} ± "
          f"{c_ss['eps_sd']:.4f}, 0∈CI95 {c_ss['eps_ci95_contains_zero']}, "
          f"χ²_ν(ΛCDM) = {c_ss['chi2_nu_lcdm']:.3f}  →  DESENLACE {outcome}")

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {"outcome": outcome, "preregistration_sha256": psha,
           "mock_validation_sha256": data_ss["mock_validation_sha256"],
           "executed_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": sha, "runtime_seconds": round(time.time() - t0, 1),
           "n_points": {"sn": data_ss["n_sn"], "cc": int(len(data_ss["hz"].z)),
                        "bao": int(len(data_ss["bao"].z)), "total": n_tot},
           "benchmark_sn_only_lcdm": bench, "arms": arms,
           "contrast_statsys": c_ss, "contrast_statonly": c_so,
           "rules": rules}
    (OUTDIR / "dovekie_real.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    wording = {
        "A": "las SNe Dovekie no identifican ε_Λ; el fondo MCMC permanece "
             "consistente y no preferido",
        "B": "PREFERENCIA por la extensión — SOSPECHA DE ERROR PRIMERO: "
             "procedimiento preinscrito (STATONLY, fórmula oficial, 5 mocks "
             "de control) antes de publicar como indicio",
        "C": "TENSIÓN — resultado negativo de primera clase; prohibido "
             "reparametrizar a posteriori",
        "INDETERMINADO": "ningún desenlace preinscrito se cumple; se publica "
                         "como indeterminado sin ajuste",
    }[outcome]
    md = [
        f"# Primera aplicación a Dovekie real — desenlace **{outcome}**\n",
        f"Preinscripción `{psha[:12]}` (congelada antes de la primera lectura "
        f"del HD); PASS de mocks `{data_ss['mock_validation_sha256'][:12]}`; "
        f"ejecución commit `{sha[:9]}`, {doc['runtime_seconds']} s.",
        "", f"**Lectura obligatoria del desenlace**: {wording}.", "",
        "## Benchmark SN-only ΛCDM (like-for-like con el chain oficial)", "",
        f"Ω_m propio = {bench['Omega_m_p50']:.4f} (+{bench['p84']-bench['Omega_m_p50']:.4f} "
        f"−{bench['Omega_m_p50']-bench['p16']:.4f}) frente a oficial "
        f"{off['mean']:.4f} ± {off['sd']:.4f}: **Δ = "
        f"{bench['delta_sigma_official']:+.3f}σ_oficial**.",
        "", "## Contraste conjunto Dovekie + CC + BAO", "",
        "| brazo | χ²_min | χ²_ν | Ω_m | H0 | ε | AIC | BIC | conv |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for tag, a in arms.items():
        p = a["posterior"]
        eps = (f"{p['epsilon']['p50']:+.4f} ± {p['epsilon']['sd']:.4f}"
               if a["model"] == "mcmc" else "≡ 0")
        md.append(f"| {tag} | {a['chi2_min']:.2f} | {a['chi2_nu']:.3f} | "
                  f"{p['Omega_m']['p50']:.4f} ± {p['Omega_m']['sd']:.4f} | "
                  f"{p['H0']['p50']:.2f} ± {p['H0']['sd']:.2f} | {eps} | "
                  f"{a['AIC']:.2f} | {a['BIC']:.2f} | "
                  f"{'sí' if a['converged'] else 'no'} |")
    for label, c in (("STAT+SYS (principal)", c_ss), ("STATONLY (robustez)", c_so)):
        md += ["", f"**{label}**: ΔAIC(MCMC−ΛCDM) = {c['dAIC_mcmc_minus_lcdm']:+.2f}, "
               f"ΔBIC = {c['dBIC_mcmc_minus_lcdm']:+.2f}, Δχ²_min = "
               f"{c['dchi2_min_gain']:+.3f}; ε = {c['eps_p50']:+.4f} ± {c['eps_sd']:.4f}, "
               f"CI95 [{c['eps_ci95'][0]:+.4f}, {c['eps_ci95'][1]:+.4f}] "
               f"(0 ∈ CI95: {c['eps_ci95_contains_zero']}); χ²_ν(ΛCDM) = "
               f"{c['chi2_nu_lcdm']:.3f}."]
    md += ["", "## Regla congelada aplicada (orden C → B → A → INDETERMINADO)",
           "", f"`{json.dumps(rules)}`", "",
           "Roles: Dovekie real = primera aplicación · Unite = benchmark "
           "armonizado, NO replicación independiente · Union3 = contraste "
           "externo · Pantheon+ = disección. Esta corrida no confirma nada que "
           "Unite ya contenga. Estatuto: comprobación interna del pipeline "
           "sobre datos reales bajo preinscripción (E8) — no demostración "
           "física."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
