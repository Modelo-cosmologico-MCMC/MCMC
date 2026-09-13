#!/usr/bin/env python
"""Ejecuta las CUATRO puertas preinscritas del PR #15 sobre los 25
mocks a nivel HD y escribe el veredicto ejecutable.

Requiere la preinscripción congelada (run_dovekie_mocks_prereg.py);
falla cerrado sin ella. Produce en results/2026-09-12_dovekie_mocks/:
  - mock_validation.json  (status PASS/FAIL + sha256 de la prereg —
    la llave de la barrera del loader real de cosmology/dovekie_sn)
  - per_mock.csv          (tabla íntegra por mock)
  - report.md             (informe citando la prereg por hash)

Uso: python scripts/run_dovekie_mocks.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.dovekie_sn import (  # noqa: E402
    N_SN_EXPECTED,
    chi2_sn_marginalized,
    load_dovekie_inv_cov,
    mu_model,
)
from mcmc_ontology.data_registry import require_available  # noqa: E402
from validation.dovekie_independent import (  # noqa: E402
    chi2_marginalized_independent,
    mu_model_independent,
    parse_hd_independent,
    unpack_inv_cov_independent,
)
from validation.dovekie_mocks import (  # noqa: E402
    INJECTED,
    K_COSMOLOGIES,
    OUTDIR,
    chi2_min_mcmc_arm,
    generate_mocks,
    lcdm_grid_mu,
    lcdm_posterior_percentiles,
    official_cov_log_likelihood,
    run_mcmc_arm,
)

PREREG = OUTDIR / "preregistration.json"
LN_N = float(np.log(N_SN_EXPECTED))


def main() -> int:
    t0 = time.time()
    if not PREREG.exists():
        raise SystemExit("PREREGISTRATION_MISSING: congela primero "
                         "con scripts/run_dovekie_mocks_prereg.py")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    gates_cfg = prereg["gates"]
    k68 = gates_cfg["3_coverage_binomial_exact"]["accept_68"]["k_range"]
    k95 = gates_cfg["3_coverage_binomial_exact"]["accept_95"]["k_range"]
    kfp = int(prereg["frozen_numbers"]["false_preference_k_max"])
    om_true = float(prereg["injected_cosmology"]["our_parametrization"]
                    ["Omega_m"])

    print(f"[prereg] sha256 = {prereg_sha[:12]}…  "
          f"k68 ∈ {k68}, k95 ∈ {k95}, falsa pref ≤ {kfp}")

    # ------------------------------------------------------------------
    # Mocks y matrices
    # ------------------------------------------------------------------
    print("[mocks] generando 25 realizaciones a nivel HD…")
    mk = generate_mocks()
    zHD, zHEL, W = mk["zHD"], mk["zHEL"], mk["W"]
    mocks = mk["mu_mocks"]
    n_mocks = mocks.shape[0]

    # ------------------------------------------------------------------
    # Puerta 1a — equivalencia de la fórmula χ² + conteo
    # ------------------------------------------------------------------
    print("[1a] fórmula oficial (AST de los bytes del release)…")
    official = official_cov_log_likelihood()
    raw = require_available("des_dovekie")
    hd_b = parse_hd_independent(raw / "DES-Dovekie_HD.csv")
    counts = {"parser_A": int(len(zHD)), "parser_B": hd_b["n_sn"]}
    d1a_max = 0.0
    dW_max = 0.0
    for kind in ("STATONLY", "STAT+SYS"):
        W_A = load_dovekie_inv_cov(kind)
        W_B = unpack_inv_cov_independent(raw / f"{kind}.npz")
        counts[f"nsn_npz_{kind}"] = int(W_A.shape[0])
        dW_max = max(dW_max, float(np.max(np.abs(W_A - W_B))))
        for kcos in K_COSMOLOGIES:
            mu_k = mu_model(zHD, zHEL, kcos[0], eps=kcos[1],
                            z_trans=kcos[2])
            for m in range(n_mocks):
                d = mocks[m]
                chi_A = chi2_sn_marginalized(mu_k, d, W_A)
                chi_B = chi2_marginalized_independent(mu_k, d, W_B)
                chi_off = -2.0 * float(official(mu_k, d, W_A))
                d1a_max = max(d1a_max, abs(chi_A - chi_off),
                              abs(chi_B - chi_off))
    counts_ok = (len({counts["parser_A"], counts["parser_B"],
                      counts["nsn_npz_STATONLY"],
                      counts["nsn_npz_STAT+SYS"],
                      N_SN_EXPECTED}) == 1)
    gate1a = bool(d1a_max <= 1e-6 and dW_max == 0.0 and counts_ok)
    print(f"[1a] max|Δχ²| = {d1a_max:.3e}, max|ΔW| = {dW_max:.1e}, "
          f"conteos = {counts} → {'PASS' if gate1a else 'FAIL'}")

    # ------------------------------------------------------------------
    # Puerta 1b — equivalencia de integradores de distancia
    # ------------------------------------------------------------------
    print("[1b] integrador producción vs cuadratura independiente…")
    dmu_max = 0.0
    dchi_1b = 0.0
    d_fixed = mocks[0]
    for kcos in K_COSMOLOGIES:
        mu_A = mu_model(zHD, zHEL, kcos[0], eps=kcos[1],
                        z_trans=kcos[2])
        mu_B = mu_model_independent(zHD, zHEL, kcos[0], eps=kcos[1],
                                    z_trans=kcos[2])
        dmu_max = max(dmu_max, float(np.max(np.abs(mu_A - mu_B))))
        dchi_1b = max(dchi_1b,
                      abs(chi2_sn_marginalized(mu_A, d_fixed, W)
                          - chi2_sn_marginalized(mu_B, d_fixed, W)))
    gate1b = bool(dmu_max <= 1e-5 and dchi_1b <= 0.01)
    print(f"[1b] max|Δμ| = {dmu_max:.3e} mag, |Δχ̃²| = {dchi_1b:.3e} "
          f"→ {'PASS' if gate1b else 'FAIL'}")

    # ------------------------------------------------------------------
    # Puertas 2 y 3 — brazo ΛCDM determinista (malla)
    # ------------------------------------------------------------------
    print("[2-3] posterior ΛCDM en malla por mock…")
    mu_grid = lcdm_grid_mu(zHD, zHEL)
    rows = []
    pulls = []
    hit68 = hit95 = 0
    chi2_lcdm_min = []
    for m in range(n_mocks):
        pct, chi2m, om_hat = lcdm_posterior_percentiles(
            mu_grid, W, mocks[m])
        sig = 0.5 * (pct[0.84] - pct[0.16])
        pull = (pct[0.50] - om_true) / sig
        pulls.append(pull)
        in68 = pct[0.16] <= om_true <= pct[0.84]
        in95 = pct[0.025] <= om_true <= pct[0.975]
        hit68 += int(in68)
        hit95 += int(in95)
        chi2_lcdm_min.append(chi2m)
        rows.append({"mock": m, "om_p16": pct[0.16], "om_p50": pct[0.50],
                     "om_p84": pct[0.84], "om_p025": pct[0.025],
                     "om_p975": pct[0.975], "pull": pull,
                     "in68": int(in68), "in95": int(in95),
                     "chi2_lcdm": chi2m, "om_hat": om_hat})
    mean_pull = float(np.mean(pulls))
    sd_pull = float(np.std(pulls, ddof=1))
    gate2 = bool(abs(mean_pull) <= 0.6)
    gate3 = bool(k68[0] <= hit68 <= k68[1] and k95[0] <= hit95 <= k95[1])
    print(f"[2] media pulls = {mean_pull:+.3f} (sd = {sd_pull:.3f}) "
          f"→ {'PASS' if gate2 else 'FAIL'}")
    print(f"[3] k68 = {hit68}/25, k95 = {hit95}/25 "
          f"→ {'PASS' if gate3 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Puerta 4 — brazo MCMC: falsa preferencia y ε
    # ------------------------------------------------------------------
    print("[4] brazo MCMC por mock (emcee sembrado + χ²_min acotado)…")
    d_aic, d_bic = [], []
    eps_p50, eps_ci_hit = [], 0
    for m in range(n_mocks):
        chain = run_mcmc_arm(zHD, zHEL, W, mocks[m], seed=42 + m)
        eps_samples = chain["flat"][:, 1]
        lo, mid, hi = np.percentile(eps_samples, [2.5, 50.0, 97.5])
        eps_p50.append(float(mid))
        eps_ci_hit += int(lo <= 0.0 <= hi)
        chi2_mc = chi2_min_mcmc_arm(zHD, zHEL, W, mocks[m],
                                    rows[m]["om_hat"])
        # anidamiento: el mínimo MCMC no puede superar al ΛCDM (ε = 0
        # está en el soporte); tolerancia numérica del refinador
        chi2_mc = min(chi2_mc, chi2_lcdm_min[m])
        daic = (chi2_mc + 2 * 3) - (chi2_lcdm_min[m] + 2 * 1)
        dbic = (chi2_mc + 3 * LN_N) - (chi2_lcdm_min[m] + 1 * LN_N)
        d_aic.append(float(daic))
        d_bic.append(float(dbic))
        rows[m].update({"chi2_mcmc": chi2_mc, "delta_aic": daic,
                        "delta_bic": dbic, "eps_p50": float(mid),
                        "eps_ci95_lo": float(lo),
                        "eps_ci95_hi": float(hi),
                        "acceptance": chain["acceptance"]})
        print(f"    mock {m:02d}: ΔAIC = {daic:+.2f}, "
              f"ε_p50 = {mid:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    n_aic_neg = int(np.sum(np.array(d_aic) < 0))
    n_bic_neg = int(np.sum(np.array(d_bic) < 0))
    med_eps = float(np.median(eps_p50))
    gate4 = bool(np.median(d_aic) > 0 and n_aic_neg <= kfp
                 and np.median(d_bic) > 0 and n_bic_neg <= kfp
                 and eps_ci_hit >= k95[0] and abs(med_eps) <= 0.05)
    print(f"[4] mediana ΔAIC = {np.median(d_aic):+.2f} "
          f"(n<0: {n_aic_neg}), mediana ΔBIC = {np.median(d_bic):+.2f} "
          f"(n<0: {n_bic_neg}), 0∈CI95(ε): {eps_ci_hit}/25, "
          f"mediana p50(ε) = {med_eps:+.4f} "
          f"→ {'PASS' if gate4 else 'FAIL'}")

    # ------------------------------------------------------------------
    # Veredicto y artefactos
    # ------------------------------------------------------------------
    status = "PASS" if (gate1a and gate1b and gate2 and gate3
                        and gate4) else "FAIL"
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {
        "status": status,
        "preregistration_sha256": prereg_sha,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "runtime_seconds": round(time.time() - t0, 1),
        "mock_seed": mk["seed"], "cov_kind": mk["cov_kind"],
        "injected": INJECTED,
        "gates": {
            "1a_formula_equivalence": {
                "pass": gate1a, "max_abs_dchi2": d1a_max,
                "max_abs_dW": dW_max, "counts": counts},
            "1b_distance_equivalence": {
                "pass": gate1b, "max_abs_dmu_mag": dmu_max,
                "max_abs_dchi2": dchi_1b},
            "2_parameter_recovery": {
                "pass": gate2, "mean_pull": mean_pull,
                "sd_pull_diagnostic": sd_pull},
            "3_coverage": {
                "pass": gate3, "k68": hit68, "k95": hit95,
                "accept_68": k68, "accept_95": k95},
            "4_no_false_preference": {
                "pass": gate4,
                "median_delta_aic": float(np.median(d_aic)),
                "median_delta_bic": float(np.median(d_bic)),
                "n_delta_aic_neg": n_aic_neg,
                "n_delta_bic_neg": n_bic_neg,
                "max_false_pref": kfp,
                "eps_ci95_contains_zero": eps_ci_hit,
                "median_eps_p50": med_eps},
        },
        "per_mock": rows,
    }
    (OUTDIR / "mock_validation.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    cols = list(rows[0].keys())
    csv_lines = [",".join(cols)]
    for r in rows:
        csv_lines.append(",".join(str(r[c]) for c in cols))
    (OUTDIR / "per_mock.csv").write_text("\n".join(csv_lines) + "\n",
                                         encoding="utf-8")

    verdict_wording = (
        "no se detecta una descalibración incompatible con el tamaño "
        "de la muestra de mocks" if gate3 else
        "la cobertura observada cae FUERA de la región binomial "
        "preinscrita — descalibración detectada")
    md = [
        "# PR #15 — validación por mocks del pipeline SN Dovekie: "
        f"**{status}**\n",
        f"Preinscripción: `{prereg_sha[:12]}` (congelada ANTES de "
        f"ejecutar puerta alguna); ejecución: commit `{sha[:9]}`, "
        f"{doc['runtime_seconds']} s.",
        "",
        f"- **1a** fórmula χ²: max|Δχ²| = {d1a_max:.2e} (≤ 1e-6), "
        f"max|ΔW| = {dW_max:.0e}, conteo 1820 idéntico en parser A, "
        f"parser B y ambos npz → {'PASS' if gate1a else 'FAIL'}",
        f"- **1b** distancias: max|Δμ| = {dmu_max:.2e} mag (≤ 1e-5), "
        f"|Δχ̃²| = {dchi_1b:.2e} (≤ 0.01) → "
        f"{'PASS' if gate1b else 'FAIL'}",
        f"- **2** recovery: media de pulls = {mean_pull:+.3f} "
        f"(cota 0.6 = 3·SEM); sd = {sd_pull:.3f} (diagnóstico, sin "
        f"puerta) → {'PASS' if gate2 else 'FAIL'}",
        f"- **3** cobertura: k68 = {hit68}/25 ∈ {k68}, "
        f"k95 = {hit95}/25 ∈ {k95} → {'PASS' if gate3 else 'FAIL'}; "
        f"{verdict_wording}",
        f"- **4** falsa preferencia: mediana ΔAIC = "
        f"{np.median(d_aic):+.2f}, mediana ΔBIC = "
        f"{np.median(d_bic):+.2f}, n(ΔAIC<0) = {n_aic_neg} ≤ {kfp}, "
        f"0 ∈ CI95(ε) en {eps_ci_hit}/25 (≥ {k95[0]}), "
        f"mediana p50(ε) = {med_eps:+.4f} (|·| ≤ 0.05) → "
        f"{'PASS' if gate4 else 'FAIL'}",
        "",
        "Roles declarados: mocks = validación del pipeline · Dovekie "
        "real = primera aplicación (solo tras PASS) · Unite = "
        "benchmark armonizado de robustez, NO replicación "
        "independiente · Union3 = comprobación externa · Pantheon+ = "
        "disección de Unite.",
        "",
        "La tabla íntegra por mock está en per_mock.csv; el veredicto "
        "ejecutable (la llave de la barrera) en mock_validation.json.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n",
                                      encoding="utf-8")
    print(f"\nVEREDICTO: {status}  →  {OUTDIR / 'mock_validation.json'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
