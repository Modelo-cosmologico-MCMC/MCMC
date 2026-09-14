#!/usr/bin/env python
"""Ejecuta el Contraste de los Residuos vía BBN bajo la preinscripción
congelada: carga la preinscripción (barrera), después los datos
registrados, corre los brazos 0–4, clasifica con la regla congelada
(brazo principal) y publica el artefacto con el aviso de procedencia.

Uso: python scripts/run_bbn_g.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.bbn_g import (  # noqa: E402
    OUTDIR,
    classify_outcome,
    delta_G_model,
    load_data,
    load_prereg,
    posterior_1d,
    posterior_2d,
    response_constants,
    sm_pulls,
)


def main() -> int:
    t0 = time.time()
    prereg, psha = load_prereg()
    rules = prereg["outcomes_enumerated"]["rules"]
    print(f"[prereg] {psha[:12]}…", flush=True)
    resp = response_constants()
    assert prereg["response_model"]["constants"]["a_G_YP"] == resp["a_G_YP"], \
        "la tabla de respuesta no es la citada por la preinscripción"
    dG_model = delta_G_model()

    # ---- primera lectura de los datos: SOLO tras la barrera ----
    data = load_data()
    v = data["values"]
    print(f"[datos] manifest {data['manifest_sha256'][:12]}…; bytes oficiales "
          f"verificados: {data['official_bytes_verified']}", flush=True)

    arm0 = sm_pulls(data, resp, rules)
    print(f"[0] SM: Y_P = {arm0['YP_sm']:.5f} (obs {v['Y_P_empress_xv']['value']:.4f}, "
          f"pull {arm0['pull_YP']:+.2f}σ); D/H = {arm0['DH_sm']*1e5:.4f}e-5 (obs "
          f"{v['DH_cooke_2018']['value']*1e5:.3f}e-5, pull {arm0['pull_DH']:+.2f}σ); "
          f"χ²_marg = {arm0['chi2_sm_marginalized']:.2f}", flush=True)

    arm1 = posterior_1d(data, resp, rules)
    outcome = classify_outcome(arm1["ci95"], dG_model)
    arm1["delta_G_model"] = dG_model
    arm1["model_in_ci95"] = bool(arm1["ci95"][0] <= dG_model <= arm1["ci95"][1])
    arm1["zero_in_ci95"] = bool(arm1["ci95"][0] <= 0.0 <= arm1["ci95"][1])
    arm1["pull_model_vs_p50_in_sd"] = (dG_model - arm1["p50"]) / arm1["sd"]
    print(f"[1] δ_G = {arm1['p50']:+.4f} ± {arm1['sd']:.4f}, CI95 "
          f"[{arm1['ci95'][0]:+.4f}, {arm1['ci95'][1]:+.4f}]; modelo {dG_model:+.4f} "
          f"{'∈' if arm1['model_in_ci95'] else '∉'} CI95; 0 "
          f"{'∈' if arm1['zero_in_ci95'] else '∉'} CI95  →  DESENLACE {outcome}",
          flush=True)

    arm2 = posterior_2d(data, resp, rules)
    m = arm2["marginal_delta_G"]
    print(f"[2] degeneración: δ_G marginal = {m['p50']:+.4f} ± {m['sd']:.4f} "
          f"(CI95 [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]); ΔN_eff marginal = "
          f"{arm2['marginal_dNeff']['p50']:+.3f} ± {arm2['marginal_dNeff']['sd']:.3f}; "
          f"cresta dΔN/dδ_G = {arm2['ridge_slope_dNeff_per_delta_G']:.2f}", flush=True)

    arm3 = posterior_1d(data, resp, rules, yp_key="Y_P_aver_2021")
    arm3["sm_pulls"] = sm_pulls(data, resp, rules, yp_key="Y_P_aver_2021")
    print(f"[3] control Aver+2021: δ_G = {arm3['p50']:+.4f} ± {arm3['sd']:.4f}", flush=True)

    arm4 = {"YP_only": posterior_1d(data, resp, rules, use_DH=False),
            "DH_only": posterior_1d(data, resp, rules, use_YP=False)}
    yo, do = arm4["YP_only"], arm4["DH_only"]
    arm4["tension_YP_vs_DH_sigma"] = (yo["p50"] - do["p50"]) / (yo["sd"] ** 2 + do["sd"] ** 2) ** 0.5
    print(f"[4] solo Y_P: {yo['p50']:+.4f} ± {yo['sd']:.4f}; solo D/H: {do['p50']:+.4f} "
          f"± {do['sd']:.4f}; tensión {arm4['tension_YP_vs_DH_sigma']:+.2f}σ", flush=True)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    caveat = ("" if data["official_bytes_verified"] else
              " — SOBRE VALORES TRANSCRITOS: bytes oficiales pendientes de "
              "verificación por el autor; el desenlace es provisional hasta esa "
              "verificación")
    wording = {
        "A": "banda de BBN compatible con G_cosmo/G_N − 1 = −1.8 % y con 0: sin "
             "identificación; el signo del central no se lee como señal (E13)",
        "B": "IDENTIFICACIÓN de G_cosmo ≠ G_N al 95 % — SOSPECHA DE ERROR PRIMERO "
             "(tasas nucleares, ξ_e, N_eff, transcripción); no es detección",
        "C": "el −1.8 % queda EXCLUIDO al 95 % por BBN — resultado negativo de "
             "primera clase; prohibido reparametrizar ε_K o invocar α_a a posteriori",
    }[outcome] + caveat
    doc = {"outcome": outcome, "outcome_provisional_pending_provenance":
           not data["official_bytes_verified"],
           "preregistration_sha256": psha,
           "response_table_sha256": prereg["response_model"]["response_table_sha256"],
           "data_manifest_sha256": data["manifest_sha256"],
           "official_bytes_verified": data["official_bytes_verified"],
           "executed_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": sha, "runtime_seconds": round(time.time() - t0, 1),
           "delta_G_model": dG_model, "mandatory_wording": wording,
           "arm0_sm": arm0, "arm1_principal": arm1, "arm2_degeneracy": arm2,
           "arm3_control_aver": arm3, "arm4_single_probe": arm4,
           "data_values_used": v, "rules": rules}
    (OUTDIR / "bbn_g.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2)
                                       + "\n", encoding="utf-8")

    md = [f"# Contraste de los Residuos vía BBN — desenlace **{outcome}**"
          f"{' (PROVISIONAL: procedencia pendiente)' if not data['official_bytes_verified'] else ''}\n",
          f"Preinscripción `{psha[:12]}` (congelada antes de leer los datos); tabla de "
          f"respuesta PRyMordial `{doc['response_table_sha256'][:12]}`; manifest de "
          f"datos `{data['manifest_sha256'][:12]}`; commit `{sha[:9]}`.", "",
          f"**Lectura obligatoria**: {wording}.", "",
          f"Predicción del modelo: δ_G = {dG_model:+.5f} (ε_K = 0.012, α_a = 0, ξ = 1).",
          "", "## Brazo 0 — SM (δ_G = 0, N_eff = 3.044)", "",
          f"Y_P^SM = {arm0['YP_sm']:.5f} frente a {v['Y_P_empress_xv']['value']:.4f} ± "
          f"{v['Y_P_empress_xv']['sigma']:.4f} (EMPRESS XV): pull {arm0['pull_YP']:+.2f}σ. "
          f"D/H^SM = {arm0['DH_sm']*1e5:.4f}e-5 frente a {v['DH_cooke_2018']['value']*1e5:.3f} "
          f"± {v['DH_cooke_2018']['sigma']*1e5:.3f} e-5 (σ_th = {arm0['sigma_th_DH_used']*1e5:.3f}e-5): "
          f"pull {arm0['pull_DH']:+.2f}σ. χ²_marg(SM) = {arm0['chi2_sm_marginalized']:.2f} (2 datos).",
          "", "## Brazo 1 — principal (δ_G libre, N_eff fijo)", "",
          f"δ_G = {arm1['p50']:+.4f} (p16 {arm1['p16']:+.4f}, p84 {arm1['p84']:+.4f}); "
          f"CI95 = [{arm1['ci95'][0]:+.4f}, {arm1['ci95'][1]:+.4f}]; MAP {arm1['map']:+.4f}. "
          f"Modelo (−1.8 %) {'∈' if arm1['model_in_ci95'] else '∉'} CI95 "
          f"(a {arm1['pull_model_vs_p50_in_sd']:+.2f} sd del central); 0 "
          f"{'∈' if arm1['zero_in_ci95'] else '∉'} CI95 (CDF(0) = {arm1['cdf_at_zero']:.3f}).",
          "", "## Brazo 2 — degeneración G ↔ N_eff (solo informa)", "",
          f"Marginal δ_G = {m['p50']:+.4f} ± {m['sd']:.4f}, CI95 [{m['ci95'][0]:+.3f}, "
          f"{m['ci95'][1]:+.3f}]; ΔN_eff marginal = {arm2['marginal_dNeff']['p50']:+.3f} ± "
          f"{arm2['marginal_dNeff']['sd']:.3f}; cresta dΔN_eff/dδ_G = "
          f"{arm2['ridge_slope_dNeff_per_delta_G']:.2f}.",
          "", "## Brazo 3 — control con Y_P de Aver+2021", "",
          f"δ_G = {arm3['p50']:+.4f} ± {arm3['sd']:.4f}, CI95 [{arm3['ci95'][0]:+.4f}, "
          f"{arm3['ci95'][1]:+.4f}] (pull Y_P SM {arm3['sm_pulls']['pull_YP']:+.2f}σ).",
          "", "## Brazo 4 — una sonda a la vez", "",
          f"Solo Y_P: {yo['p50']:+.4f} ± {yo['sd']:.4f}; solo D/H: {do['p50']:+.4f} ± "
          f"{do['sd']:.4f}; tensión interna {arm4['tension_YP_vs_DH_sigma']:+.2f}σ.",
          "", "## Regla congelada", "", f"`{json.dumps(rules)}`", "",
          "Fronteras: ξ_e no modelada; G ↔ N_eff casi degenerados; sistemático nuclear "
          "por |NACRE II − PRIMAT| y mínimo declarado; α_a = 0 en la predicción; "
          "transcripción de los datos pendiente de verificación por el autor. "
          "Estatuto: contraste interno bajo preinscripción (E8) sobre valores "
          "publicados transcritos — no demostración física."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
