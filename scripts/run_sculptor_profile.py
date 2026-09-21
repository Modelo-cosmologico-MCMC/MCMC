#!/usr/bin/env python
"""Preinscripción «perfil σ_los(R) de Sculptor» (frente 5d): la FORMA del
perfil de dispersión que la amplitud única del 5E predice, frente al
perfil binado de Walker et al. 2009 — cuando los bytes oficiales entren.

    python scripts/run_sculptor_profile.py prereg    # congela predicción y regla (sha256)
    python scripts/run_sculptor_profile.py analyze   # FALLA CERRADO mientras walker2009 no esté AVAILABLE

La predicción (dynamics.cronos_jeans.sculptor_profile_table, montaje
CONGELADO del 5E: Plummer con R_half = 260 pc, Υ⋆ = 2, A_Sculptor) es
un perfil σ_los(R) con pico central ≈ 19 km/s y caída a ≈ 2 km/s en
1 kpc, frente a un perfil observado ≈ 9–10 km/s plano hasta ≳ 1 kpc
(referencia de la literatura, NO ingerida). El brazo es la anisotropía
β ∈ {−0.5, 0, +0.3}. Regla congelada sobre los bins que la ingesta
preinscrita producirá (miembros con probabilidad ≥ 0.9; bins de igual
número, ≈ 150 estrellas, R ≤ 1 kpc; σ_los por bin con su error):

    A — χ²_ν ≤ 1.5 para algún β del brazo (forma compatible)
    B — 1.5 < χ²_ν ≤ 3 para el mejor β (tensión)
    C — el pico central queda excluido: en los dos bins más internos,
        (σ_pred − σ_obs)/σ_err > 3 para TODO β del brazo (la forma es
        refutada aunque la normalización global reproduzca σ_obs por
        construcción)
    INDETERMINADO — walker2009 no AVAILABLE (fallo cerrado), N_bins < 4 o
        esquema no confirmado

Estado al congelar: walker2009 está DATA_UNAVAILABLE (CDS/VizieR denegado
por el proxy, 403, comprobado el 21-sep-2026 14:10 UTC): el analizador
registra el fallo cerrado como resultado explícito hasta la ingesta.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.cronos_jeans import A_SCULPTOR, sculptor_profile_table  # noqa: E402
from mcmc_ontology.data_registry import require_available  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_sculptor_profile"
PREREG = OUTDIR / "preregistration.json"
BETA_ARM = (-0.5, 0.0, 0.3)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=OUTDIR.parent.parent).stdout.strip()


def prereg() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred = {str(b): sculptor_profile_table(beta=b) for b in BETA_ARM}
    doc = {
        "title": "Perfil σ_los(R) de Sculptor: la forma que predice la amplitud única del 5E frente al perfil binado de Walker et al. 2009",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_sculptor_profile.py prereg", "generator_commit_must_precede_freeze": True,
        "declared": {
            "frozen_setup": "montaje del 5E (dynamics.sculptor_transfer.frozen_config): Plummer con R_half = 260 pc, Υ⋆ = 2, A_Sculptor",
            "A_sculptor": A_SCULPTOR, "beta_arm": list(BETA_ARM),
            "dataset": "walker2009 (CDS J/AJ/137/3100, table4.dat: Sculptor, 1818 estrellas)",
            "dataset_state_at_freeze": "DATA_UNAVAILABLE (CDS/VizieR 403 vía proxy, 21-sep-2026 14:10 UTC)",
            "membership": "probabilidad de pertenencia ≥ 0.9 (columna del catálogo)",
            "binning": "bins de igual número (≈ 150 estrellas) en R proyectado ≤ 1 kpc; σ_los por bin = desviación típica corregida "
                       "por errores de medida (⟨e²⟩ restado); error del bin σ/√(2N)",
            "prediction_profiles_kms": {b: [{"R_pc": p["R_pc"], "newton": p["sigma_los_newton"], "newton_cronos": p["sigma_los_newton_cronos"]}
                                           for p in pred[b]["profile"]] for b in pred},
            "shape_prediction": pred["0.0"]["shape_prediction"],
            "observed_reference_not_ingested": pred["0.0"]["observed_reference"],
        },
        "rules": {
            "chi2": "χ² = Σ_bins [(σ_pred(R_bin; β) − σ_obs,bin)/σ_err,bin]², ν = N_bins (ningún parámetro ajustado)",
            "A": "χ²_ν ≤ 1.5 para algún β del brazo", "B": "1.5 < χ²_ν(mejor β) ≤ 3",
            "C": "en los dos bins más internos (σ_pred − σ_obs)/σ_err > 3 para TODO β del brazo",
            "INDETERMINADO": "walker2009 no AVAILABLE, N_bins < 4 o esquema no confirmado",
            "verdict": "una letra; ninguna es afirmación sobre el tratado: sitúa la Ley de Cronos débil con A_Sculptor frente a la forma "
                       "del perfil (E8). La normalización global reproduce σ_obs por construcción (A_Sculptor se obtuvo de ella): "
                       "solo la FORMA es predicción",
        },
        "what_this_cannot_decide": ["la forma de ε_c(ρ) (diccionario)", "A (el 5E la fijó; aquí es hipótesis)",
                                    "la anisotropía real (β es brazo)"],
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). "
          f"sha256 de `preregistration.json`: `{sha}`.", "",
          "## Declarado", ""] + [f"- **{k}**: {v}" for k, v in doc["declared"].items() if k != "prediction_profiles_kms"] + [
          "", "## Predicción congelada (σ_los en km/s; newtoniano / newtoniano + Cronos)", "",
          "| R [pc] | β = −0.5 | β = 0 | β = +0.3 |", "|---|---|---|---|"]
    for i, p in enumerate(pred["0.0"]["profile"]):
        cells = [f"{pred[b]['profile'][i]['sigma_los_newton']:.2f} / {pred[b]['profile'][i]['sigma_los_newton_cronos']:.2f}" for b in pred]
        md.append(f"| {p['R_pc']:.0f} | " + " | ".join(cells) + " |")
    md += ["", "## Reglas (congeladas)", ""] + [f"- **{k}**: {v}" for k, v in doc["rules"].items()] + [
          "", "## Lo que no puede decidir", ""] + [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def analyze() -> int:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    try:
        raw = require_available("walker2009")
    except (FileNotFoundError, RuntimeError) as exc:
        res = {"preregistration_sha256": _sha(PREREG), "code_commit_analysis": _git_head(),
               "analyzed_utc": datetime.now(timezone.utc).isoformat(), "verdict": "INDETERMINADO",
               "reason": f"fallo cerrado: {exc}", "next_step": "ingerir walker2009 (CDS J/AJ/137/3100) con checksum y schema_report; "
                                                                "después ejecutar el binning preinscrito y este analizador sin tocar la regla"}
        (OUTDIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (OUTDIR / "report.md").write_text(
            f"# Perfil σ_los(R) de Sculptor — {res['verdict']}\n\n{res['reason']}.\n\nSiguiente paso: {res['next_step']}.\n\n"
            f"La predicción y la regla quedan congeladas en `preregistration.json` (sha256 `{res['preregistration_sha256'][:12]}…`).\n",
            encoding="utf-8")
        print(f"INDETERMINADO — {exc}")
        return 0
    raise SystemExit(f"walker2009 AVAILABLE en {raw}: el binning preinscrito aún no está implementado — no se analiza sin él "
                     f"(regla: {pre['declared']['binning']})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"prereg": prereg, "analyze": analyze}.get(mode, lambda: print(__doc__) or 2)())
