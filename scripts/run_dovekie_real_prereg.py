#!/usr/bin/env python
"""Preinscripción de la PRIMERA APLICACIÓN del pipeline SN a Dovekie real
— OBLIGATORIA antes de la primera lectura del HD (columna MU).

Este script NO lee MU: consume el diseño (conteo de SNe verificado por
bytes), las tablas CC/BAO, el chain OFICIAL como referencia del
benchmark y el PASS de la validación por mocks (#15), y congela:
datos y alcance, brazos, priors, sampler, los TRES desenlaces con
umbrales numéricos y compromiso de no-ajuste, la redacción obligatoria
del desenlace A, prohibiciones, roles y artefactos exigidos.

Uso: python scripts/run_dovekie_real_prereg.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.bayesian_fit import load_bao_data, load_Hz_data  # noqa: E402
from cosmology.dovekie_real_fit import (  # noqa: E402
    K_PARAMS,
    OUTDIR,
    PRIOR_EPS,
    PRIOR_EPS_GAUSS,
    PRIOR_H0,
    PRIOR_H0_GAUSS,
    PRIOR_OMEGA_M,
    PRIOR_ZTRANS,
    official_omega_m,
)
from cosmology.dovekie_sn import (  # noqa: E402
    MOCK_VALIDATION,
    load_dovekie_design,
    require_mock_validation_pass,
)

RULES = {
    "C_bench_sigma": 3.0,      # |Ω_m^propio − Ω_m^oficial| > 3σ_oficial ⟹ tensión
    "C_chi2nu_max": 1.30,      # χ²_min/dof del brazo ΛCDM conjunto > 1.30 ⟹ tensión
    "B_dBIC_pro_mcmc": 2.0,    # BIC_ΛCDM − BIC_MCMC > 2 Y 0 ∉ CI95(ε) ⟹ preferencia
    "A_dBIC_min": 0.0,         # BIC_MCMC − BIC_ΛCDM > 0 (ΛCDM no desfavorecido)
    "A_eps_sd_min": 0.04,      # σ_post(ε) ≥ 0.8·σ_prior ⟹ dominada por el prior
    "A_bench_sigma": 1.0,      # benchmark SN-only dentro de 1σ_oficial
}
SAMPLER = {"nwalkers": 32, "nsteps": 3000, "seed": 42, "burn": "nsteps//2",
           "thin": 4}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    mock_doc = require_mock_validation_pass()
    mock_sha = hashlib.sha256(MOCK_VALIDATION.read_bytes()).hexdigest()
    design = load_dovekie_design()
    hz, bao = load_Hz_data(), load_bao_data()
    off = official_omega_m()
    n_total = design["n_sn"] + len(hz.z) + len(bao.z)

    doc = {
        "preregistered_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "front": ("primera aplicación del pipeline SN a Dovekie REAL — "
                  "ΛCDM vs MCMC-fondo sobre Dovekie STAT+SYS + CC + BAO"),
        "precondition_mock_validation": {
            "status": mock_doc["status"],
            "mock_validation_sha256": mock_sha,
            "mock_preregistration_sha256": mock_doc["preregistration_sha256"],
            "note": ("la barrera de #15 (require_mock_validation_pass) sigue "
                     "activa; el ejecutor cita este hash Y el de la presente "
                     "preinscripción")},
        "data_and_scope": {
            "sn": {"dataset": "des_dovekie @ c9a4fcaf", "n_sn": design["n_sn"],
                   "count_note": ("1829 líneas = 8 comentarios + cabecera + "
                                  "1820 filas SN; corte oficial zHD > 0 "
                                  "heredado idéntico, no elimina ninguna"),
                   "primary_cov": "STAT+SYS",
                   "robustness_cov": "STATONLY (jamás resultado principal)"},
            "cc": {"n": int(len(hz.z)), "source": "data/boss_eboss/hz_cc.txt"},
            "bao": {"n": int(len(bao.z)), "source": "data/boss_eboss/bao_dr12.txt",
                    "r_d_fiducial_Mpc": float(bao.r_d)},
            "n_total_for_BIC": int(n_total),
            "background": ("cosmology.background.H_of_z (clausura plana por "
                           "llamada, transición normalizada hoy); sin "
                           "reparametrización nueva en este PR"),
        },
        "arms": {
            "1_benchmark_sn_only_lcdm": ("posterior 1D determinista de Ω_m en "
                                         "malla (prior plano sobre el soporte "
                                         "oficial) sobre Dovekie STAT+SYS solo — "
                                         "comparación like-for-like con el chain "
                                         "oficial"),
            "2_joint_lcdm_statsys": "emcee, θ = (H0, Ω_m), k = 2",
            "3_joint_mcmc_statsys": "emcee, θ = (H0, Ω_m, ε, z_trans), k = 4",
            "4_joint_lcdm_statonly": "robustez",
            "5_joint_mcmc_statonly": "robustez",
            "chi2_min": ("multistart Powell acotado + pulido sobre el cierre "
                         "del soporte de los priors; at_boundary publicado"),
        },
        "priors_frozen": {
            "H0": {"support": list(PRIOR_H0), "gauss": list(PRIOR_H0_GAUSS),
                   "note": "F.4"},
            "Omega_m": {"support": list(PRIOR_OMEGA_M),
                        "note": ("IDÉNTICO al soporte del chain oficial; más "
                                 "ancho que bayesian_fit (0.20, 0.40) para no "
                                 "truncar el posterior SN-only")},
            "epsilon": {"support": list(PRIOR_EPS),
                        "gauss": list(PRIOR_EPS_GAUSS), "note": "Apéndice F, #12"},
            "z_trans": {"support": list(PRIOR_ZTRANS), "note": "#12"},
            "M_B": "marginalizada analíticamente en ambos brazos; no cuenta en k",
            "k": K_PARAMS,
        },
        "sampler": SAMPLER,
        "official_reference": {
            "dataset": "des_dovekie_chains (dovekie_lcdm_nautilus.txt @ c9a4fcaf)",
            "Omega_m_weighted": off,
            "role": "REFERENCIA del benchmark; no entra en ningún likelihood"},
        "outcomes_enumerated": {
            "order": "C (tensión) → B (preferencia) → A (aburrido) → INDETERMINADO",
            "A_expected_boring": {
                "rule": (f"BIC_MCMC − BIC_ΛCDM > {RULES['A_dBIC_min']} Y "
                         f"σ_post(ε) ≥ {RULES['A_eps_sd_min']} Y 0 ∈ CI95(ε) Y "
                         f"|ΔΩ_m|_benchmark ≤ {RULES['A_bench_sigma']}·σ_oficial"),
                "mandatory_wording": (
                    "«las SNe Dovekie no identifican ε_Λ; el fondo MCMC "
                    "permanece consistente y no preferido» — NUNCA «MCMC "
                    "compatible con SNe al X %» como evidencia positiva")},
            "B_preference": {
                "rule": (f"BIC_ΛCDM − BIC_MCMC > {RULES['B_dBIC_pro_mcmc']} Y "
                         "0 ∉ CI95(ε)"),
                "treatment": ("SOSPECHA DE ERROR PRIMERO: repetir con STATONLY, "
                              "con la fórmula oficial (equivalencia #15) y "
                              "sobre 5 mocks de control con la misma cadena; "
                              "solo si sobrevive a las tres se publica como "
                              "indicio con las comprobaciones adjuntas")},
            "C_tension": {
                "rule": (f"|ΔΩ_m|_benchmark > {RULES['C_bench_sigma']}·σ_oficial "
                         f"O χ²_min/dof(ΛCDM conjunto) > {RULES['C_chi2nu_max']}"),
                "treatment": ("resultado negativo de primera clase, publicado "
                              "con el mismo peso; PROHIBIDO reparametrizar ε_Λ, "
                              "z_trans o priors a posteriori")},
            "INDETERMINADO": "cualquier otro caso; se publica como tal",
            "rules": RULES,
            "no_adjustment_commitment": (
                "ningún parámetro nuevo ni cambio de prior tras la primera "
                "evaluación de la likelihood sobre datos reales; umbrales y "
                "sampler congelados; un cambio exige nueva preinscripción con "
                "el fallo previo publicado"),
        },
        "prohibitions": [
            "ningún parámetro nuevo ni cambio de prior tras la primera "
            "evaluación sobre datos reales",
            "ningún corte de SNe distinto del oficial (zHD > 0); conteo "
            "efectivo declarado: 1820",
            "roles fijados: Dovekie real = primera aplicación; Unite = "
            "benchmark armonizado NO independiente; Union3 = contraste "
            "externo; Pantheon+ = disección — esta corrida no confirma nada "
            "que Unite ya contenga",
            "el resultado entra en claims_registry.yaml con su categoría "
            "exacta (resultado-negativo / indicio / tensión) en el mismo PR",
        ],
        "artifacts_required": [
            "preregistration.json (esta, congelada; hash citado)",
            "chains_<brazo>.npz con semilla", "dovekie_real.json",
            "report.md con los tres desenlaces enumerados y el alcanzado",
            "comparación STATONLY/STAT+SYS", "fila(s) del registry",
            "CHANGELOG y Nota I (vocabulario v35.1: E8, comprobación interna "
            "≠ demostración física)",
        ],
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Preinscripción — primera aplicación del pipeline SN a Dovekie "
        "real\n",
        f"- **Commit**: `{sha}`", f"- **Fecha (UTC)**: {doc['preregistered_utc']}",
        f"- **Precondición**: PASS de mocks `{mock_sha[:12]}…` (prereg #15 "
        f"`{mock_doc['preregistration_sha256'][:12]}…`)",
        f"- **Datos**: Dovekie 1820 SNe (STAT+SYS principal; STATONLY "
        f"robustez) + {len(hz.z)} CC + {len(bao.z)} BAO; n = {n_total}",
        f"- **Referencia oficial**: Ω_m = {off['mean']:.4f} ± {off['sd']:.4f} "
        f"(chain nautilus flat-ΛCDM SN-only, ponderado)",
        f"- **Priors**: H0 ~ N(67.4, 5²) en (60, 80); Ω_m ~ U{PRIOR_OMEGA_M} "
        f"(= soporte oficial); ε ~ N(0.012, 0.05²) en {PRIOR_EPS}; z_trans ~ "
        f"U{PRIOR_ZTRANS}; M_B marginalizada",
        f"- **Sampler**: {SAMPLER}",
        "",
        "**Desenlaces (orden literal C → B → A → INDETERMINADO)**:",
        f"- C tensión: |ΔΩ_m| > {RULES['C_bench_sigma']}σ_of o χ²_ν(ΛCDM) > "
        f"{RULES['C_chi2nu_max']}.",
        f"- B preferencia: BIC_ΛCDM − BIC_MCMC > {RULES['B_dBIC_pro_mcmc']} y "
        "0 ∉ CI95(ε) — sospecha de error primero.",
        f"- A aburrido (esperado): ΔBIC > 0 pro-ΛCDM, σ(ε) ≥ "
        f"{RULES['A_eps_sd_min']}, 0 ∈ CI95(ε), |ΔΩ_m| ≤ 1σ_of. Redacción: "
        "«las SNe Dovekie no identifican ε_Λ; el fondo MCMC permanece "
        "consistente y no preferido».",
        "",
        "**Compromiso**: ningún parámetro ni prior cambia tras la primera "
        "evaluación sobre datos reales; el resultado entra en el registry "
        "con su categoría exacta.",
    ]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n",
                                               encoding="utf-8")
    print(f"Preinscripción congelada en {OUTDIR} (commit {sha[:9]}); "
          f"Ω_m oficial = {off['mean']:.4f} ± {off['sd']:.4f}; n = {n_total}")


if __name__ == "__main__":
    main()
