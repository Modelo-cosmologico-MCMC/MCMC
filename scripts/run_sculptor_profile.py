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

Anexo de binning (23-oct-2026, PR-W del bloque P0 del autor): lo que la
regla congelada no fijaba (lectura del esquema, unidad «estrella» con
épocas repetidas, centro, distancia, R del bin, reparto del resto, orden
de las letras) se congela en binning_annex.json ANTES de los bytes; la
preinscripción no se toca. Modos añadidos:

    python scripts/run_sculptor_profile.py annex     # congela el anexo (se niega si ya existe o si hay bytes)
    python scripts/run_sculptor_profile.py schema    # día de los bytes: columnas y papeles para el schema_report
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.cronos_jeans import A_SCULPTOR, sculptor_profile_table  # noqa: E402
from dynamics.walker_binning import (  # noqa: E402
    SchemaError,
    assign_roles,
    binned_profile,
    frozen_letter,
    parse_readme,
    readme_record_count,
    stars_from_catalogue,
)
from mcmc_ontology.data_registry import manifest_path, require_available  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-09-21_sculptor_profile"
PREREG = OUTDIR / "preregistration.json"
ANNEX = OUTDIR / "binning_annex.json"
RAW = ROOT / "data" / "raw" / "walker2009"
TABLE = "table4.dat"
BETA_ARM = (-0.5, 0.0, 0.3)

# Lo que la regla congelada no fijaba: el anexo lo congela antes de los bytes (los umbrales de las letras se transcriben
# de las reglas congeladas y un candado comprueba que coinciden con su texto).
ANNEX_NUMBERS = {
    "pm_min": 0.9, "target_per_bin": 150, "R_max_pc": 1000.0,
    "distance_kpc": 79.0, "centre_ra_deg": 15.0 * (1.0 + 0.0 / 60.0 + 9.4 / 3600.0), "centre_dec_deg": -(33.0 + 42.0 / 60.0 + 33.0 / 3600.0),
    "merge_radius_arcsec": 1.0, "centroid_gate_arcmin": 10.0, "prediction_reproduction_tol_kms": 1e-6,
    "chi2_A_max": 1.5, "chi2_B_max": 3.0, "C_z_min": 3.0, "C_inner_bins": 2, "min_bins": 4,
    "distance_sensitivity_kpc": [75.0, 83.0],
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=OUTDIR.parent.parent).stdout.strip()


def prereg() -> int:
    if PREREG.exists():
        raise SystemExit(f"FALLO CERRADO: la preinscripción ya está congelada (sha256 {_sha(PREREG)}); no se sobrescribe")
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
    if not ANNEX.exists():
        raise SystemExit("FALLO CERRADO: falta el anexo de binning congelado (binning_annex.json)")
    ax = json.loads(ANNEX.read_text(encoding="utf-8"))
    if ax["of_preregistration"]["sha256"] != _sha(PREREG):
        raise SystemExit("FALLO CERRADO: el anexo no cita la preinscripción vigente")
    if _sha(ANNEX) not in (OUTDIR / "binning_annex.md").read_text(encoding="utf-8"):
        raise SystemExit("FALLO CERRADO: binning_annex.json no es el congelado (su sha256 no está en binning_annex.md)")
    if not (raw / "ReadMe").exists():
        raise SystemExit(f"FALLO CERRADO: falta el ReadMe del catálogo en {raw}")
    readme = (raw / "ReadMe").read_text(encoding="latin-1")
    res = analyze_catalogue(readme, raw / ax["table_file"], pre, ax)
    res = {"preregistration_sha256": _sha(PREREG), "annex_sha256": _sha(ANNEX), "code_commit_analysis": _git_head(),
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), **res}
    (OUTDIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTDIR / "report.md").write_text(_report_md(res), encoding="utf-8")
    print(f"{res['verdict']} — {res['reason']}")
    return 0


def _reproduction_gate(pre: dict) -> float:
    """Máxima desviación (km/s) entre el código congelado y la tabla de la preinscripción en su malla (ambas columnas,
    todo β del brazo); infinita si la A del código no es la congelada."""
    dev = 0.0
    for b in pre["declared"]["beta_arm"]:
        tab = sculptor_profile_table(beta=b)
        if tab["A"] != pre["declared"]["A_sculptor"]:
            return math.inf
        for p, f in zip(tab["profile"], pre["declared"]["prediction_profiles_kms"][str(b)], strict=True):
            dev = max(dev, abs(p["sigma_los_newton_cronos"] - f["newton_cronos"]), abs(p["sigma_los_newton"] - f["newton"]))
    return dev


def _prediction(R_pc: list[float], pre: dict, column: str) -> dict:
    return {str(b): [p[column] for p in sculptor_profile_table(beta=b, R_pc=tuple(R_pc))["profile"]]
            for b in pre["declared"]["beta_arm"]}


def analyze_catalogue(readme_text: str, table_path: Path, pre: dict, ax: dict) -> dict:
    """La regla congelada sobre el catálogo con los números del ANEXO (nunca constantes del módulo): lectura, unidad
    estrella, miembros, bins, puertas y letra; después, sin letra, la columna newtoniana y la sensibilidad a D."""
    N = ax["numbers"]
    try:
        stars = stars_from_catalogue(readme_text, table_path, ax["table_file"], N["merge_radius_arcsec"])
    except SchemaError as exc:
        return {"verdict": "INDETERMINADO", "reason": f"esquema no confirmado: {exc}"}
    out = {"roles": stars["roles"], "reading": stars["diagnostics"]}
    dev = _reproduction_gate(pre)
    out["prediction_reproduction_max_dev_kms"] = dev
    if not dev <= N["prediction_reproduction_tol_kms"]:
        return {**out, "verdict": "INDETERMINADO", "reason": f"predicción congelada no reproducida por el código (desviación {dev:.3g} km/s)"}

    def profile(D):
        return binned_profile(stars, N["centre_ra_deg"], N["centre_dec_deg"], D, N["pm_min"], N["R_max_pc"], N["target_per_bin"])

    prof = profile(N["distance_kpc"])
    out.update({k: prof[k] for k in ("members", "members_within_R_max", "stars_pm_straddling_threshold", "member_centroid_offset_arcmin")})
    out["bins"] = prof["bins"]
    if prof["members"] and not prof["member_centroid_offset_arcmin"] <= N["centroid_gate_arcmin"]:
        return {**out, "verdict": "INDETERMINADO",
                "reason": f"centroide de los miembros a {prof['member_centroid_offset_arcmin']:.3g}′ del centro fijado (> {N['centroid_gate_arcmin']}′): centro o tabla no confirmados"}

    def letter(bins):
        R_bins = [b["R_pc"] for b in bins]
        pred = _prediction(R_bins, pre, "sigma_los_newton_cronos") if len(bins) >= N["min_bins"] else {}
        return frozen_letter(bins, pred, N["chi2_A_max"], N["chi2_B_max"], N["C_z_min"], N["C_inner_bins"], N["min_bins"]), pred

    L, pred = letter(prof["bins"])
    out.update({"verdict": L["letter"], "reason": L["reason"], "letter_detail": L, "sigma_pred_kms": pred})
    if "per_beta" in L:
        newton = _prediction([b["R_pc"] for b in prof["bins"]], pre, "sigma_los_newton")
        sig = [b["sigma_kms"] for b in prof["bins"]]
        err = [b["sigma_err_kms"] for b in prof["bins"]]
        out["without_letter_newton_chi2_nu"] = {b: sum(((p - s) / e) ** 2 for p, s, e in zip(newton[b], sig, err)) / len(sig) for b in newton}
    sens = {}
    for D in N["distance_sensitivity_kpc"]:
        p = profile(D)
        Ld, _ = letter(p["bins"])
        sens[str(D)] = {"N_bins": len(p["bins"]),
                        "chi2_nu_best": Ld["per_beta"][Ld["best_beta"]]["chi2_nu"] if "per_beta" in Ld else None,
                        "C_condition": all(min(v["z_inner"]) > N["C_z_min"] for v in Ld["per_beta"].values()) if "per_beta" in Ld else None}
    out["without_letter_distance_sensitivity"] = sens
    return out


def _report_md(res: dict) -> str:
    md = [f"# Perfil σ_los(R) de Sculptor — {res['verdict']}", "", f"{res['reason']}.", "",
          f"Preinscripción `{res['preregistration_sha256'][:12]}…`; anexo de binning `{res['annex_sha256'][:12]}…`; "
          f"código del análisis `{res['code_commit_analysis'][:9]}`.", ""]
    if res.get("bins"):
        md += ["| bin | N | R mediana [pc] | σ_obs [km/s] | σ_err | " + " | ".join(f"σ_pred β={b}" for b in res.get("sigma_pred_kms", {})) + " |",
               "|---|---|---|---|---|" + "---|" * len(res.get("sigma_pred_kms", {}))]
        for i, b in enumerate(res["bins"]):
            pred = " | ".join(f"{v[i]:.2f}" for v in res.get("sigma_pred_kms", {}).values())
            md.append(f"| {i + 1} | {b['N']} | {b['R_pc']:.0f} | {b['sigma_kms']:.2f} | {b['sigma_err_kms']:.2f} | {pred} |")
    md += ["", "Ninguna letra es afirmación sobre el tratado (E8); el número no es señal (E13): A, B, C e INDETERMINADO se "
           "publican con el mismo peso. Columna newtoniana y sensibilidad a la distancia: sin letra (results.json)."]
    return "\n".join(md) + "\n"


def annex() -> int:
    """Congela el anexo de binning ANTES de los bytes: se niega si ya existe, si falta la preinscripción o si hay
    manifest o bytes de walker2009."""
    if ANNEX.exists():
        raise SystemExit(f"FALLO CERRADO: el anexo ya está congelado (sha256 {_sha(ANNEX)}); no se sobrescribe")
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada")
    present = [p.name for p in RAW.glob("*")] if RAW.exists() else []
    if manifest_path("walker2009").exists() or present:
        raise SystemExit(f"FALLO CERRADO: hay manifest o bytes de walker2009 ({present}); el anexo se congela antes de los bytes")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    d = pre["declared"]
    doc = {
        "title": "Anexo de binning de la preinscripción «perfil σ_los(R) de Sculptor»: lo que la regla congelada no fijaba, congelado antes de los bytes",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_sculptor_profile.py annex", "generator_commit_must_precede_freeze": True,
        "ordered_by": "bloque P0 del autor del 23-oct-2026 (PR-W): implementar el binning, congelar el anexo antes de los bytes",
        "of_preregistration": {"file": PREREG.name, "sha256": _sha(PREREG), "frozen_utc": pre["frozen_utc"],
                               "rules_untouched": "la preinscripción y sus reglas no se tocan; el anexo solo fija lo que no fijaban"},
        "dataset_state_at_freeze": "walker2009 sin manifest y sin bytes en data/raw/walker2009: ningún byte del catálogo visto",
        "frozen_rule_verbatim": {"membership": d["membership"], "binning": d["binning"], "rules": pre["rules"]},
        "table_file": TABLE,
        "numbers": ANNEX_NUMBERS,
        "pins": {
            "fichero": f"{TABLE} (Sculptor; 1818 registros según el ReadMe oficial) con su ReadMe, desde data/raw/walker2009 vía "
                       "require_available (manifest AVAILABLE, esquema confirmado, sha256 re-verificados)",
            "esquema": "manda la descripción byte a byte del ReadMe; papeles por las reglas de dynamics.walker_binning.assign_roles "
                       "(posición sexagesimal o en grados; velocidad = la única columna en km/s que no es error, la heliocéntrica si hay "
                       "varias; error = e_<velocidad>; pertenencia = la única columna real cuya explicación habla de pertenencia o "
                       "etiquetada PM/Pmem/Prob, con valores en [0, 1]; identificador opcional). El número de registros debe coincidir "
                       "con el File Summary. Cualquier ambigüedad: INDETERMINADO («esquema no confirmado», regla congelada)",
            "unidad_estrella": "dos registros son la misma estrella si comparten identificador o distan < 1″ (cierre transitivo); "
                               "las líneas idénticas byte a byte cuentan una vez; se descartan los registros sin velocidad, sin error "
                               "positivo o sin posición",
            "epocas": "velocidad de la estrella = media pesada por inverso de varianza de sus épocas, error (Σ e⁻²)^−½; sin recorte "
                      "de binarias ni de variables (la regla congelada no lo incluye)",
            "pertenencia": "P_M de la estrella = mínimo de los P_M presentes de sus épocas (conservador frente a contaminación); "
                           "miembro si P_M ≥ 0.9; sin P_M presente no es miembro; se publica cuántas estrellas tienen épocas a "
                           "ambos lados de 0.9",
            "centro": "(α, δ) J2000 = (01h00m09.4s, −33°42′33″) = (15.039167°, −33.709167°), McConnachie 2012 (AJ 144, 4). "
                      "Transcrito por el ejecutor sin acceso a la fuente desde el entorno (ADS y CDS denegados por el proxy): "
                      "verificación del autor pendiente ANTES de los bytes; si lo corrige, se congela un anexo v2 antes de los "
                      "bytes, nunca después",
            "distancia": "D = 79 kpc: la distancia con la que el montaje congelado fija R_half = 260 pc (r_h ≈ 11.3′; tabla 1 de "
                         "Walker et al. 2009, ApJ 704, 1274: D = 79 ± 4 kpc). θ → R usa la MISMA distancia que fijó la escala del "
                         "modelo: otra D cambiaría la forma comparada sin cambiar el modelo. Misma advertencia de verificación",
            "radio": "R = D·θ con θ la separación angular exacta al centro; radio CIRCULAR (el montaje congelado es esférico)",
            "corte": "R ≤ 1000 pc (43.5′ a 79 kpc)",
            "bins": "n_bins = ⌊N/150 + ½⌋ (mínimo 1 si N > 0) sobre los N miembros con R ≤ 1 kpc; orden estable por R (empates por "
                    "orden de aparición en el catálogo); np.array_split: los N mod n_bins bins internos llevan una estrella más",
            "R_bin": "mediana de R de las estrellas del bin",
            "sigma_bin": "σ² = s² − ⟨e²⟩ con s² la varianza muestral (ddof = 1) alrededor de la media PROPIA del bin y ⟨e²⟩ la media "
                         "de los errores combinados al cuadrado; σ_err = σ/√(2N); si s² ≤ ⟨e²⟩ en algún bin: INDETERMINADO (puerta)",
            "prediccion": "σ_pred(R_bin; β) = columna newton_cronos (la predicción del 5E) evaluada con el código congelado "
                          "(dynamics.cronos_jeans.sculptor_profile_table(beta=β, R_pc=R_bin)); puerta: ese código reproduce la tabla "
                          "de la preinscripción en su malla a ≤ 1e-6 km/s con la misma A_Sculptor, o INDETERMINADO",
            "puerta_centro": "el centroide de las posiciones de los miembros dista ≤ 10′ del centro fijado (protege frente a un centro "
                             "mal transcrito o una tabla equivocada), o INDETERMINADO",
            "letras": "orden puertas → C → B → A → INDETERMINADO (convención de la casa): C precede a A y B si sus condiciones "
                      "coinciden; si ninguna letra aplica (χ²_ν(mejor β) > 3 sin la exclusión del pico de C), INDETERMINADO con "
                      "las cifras publicadas, sin reasignar a B ni a C",
            "sin_letra": "χ²_ν de la columna newtoniana por β; sensibilidad a la distancia (D = 75 y 83 kpc, ±1σ): N_bins, "
                         "χ²_ν(mejor β) y la condición de C; diagnósticos de lectura (registros, épocas, fusiones, centroide)",
            "sin_correcciones": "ni rotación interna, ni rotación de perspectiva, ni recorte sigma, ni recorte de binarias; "
                                "velocidades heliocéntricas tal cual (la dispersión no depende de un desplazamiento constante)",
        },
        "expectation_E13": "C, ya declarada en la fila sculptor-perfil-sigma-los: el perfil de la literatura es plano ≈ 9–10 km/s "
                           "frente al pico predicho de 14–25 km/s. A, B, C e INDETERMINADO se publican con el mismo peso",
        "what_this_cannot_decide": ["la elipticidad de Sculptor (≈ 0.3) frente al montaje esférico",
                                    "la rotación interna o de perspectiva (no se corrigen)",
                                    "las binarias (no se recortan: inflan σ ligeramente en todos los bins)",
                                    "la exactitud del centro y de la distancia transcritos (verificación del autor pendiente)"],
    }
    ANNEX.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(ANNEX)
    md = [f"# {doc['title']}", "",
          f"Congelado {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). "
          f"sha256 de `binning_annex.json`: `{sha}`.", "",
          f"De la preinscripción `{PREREG.name}` (sha256 `{doc['of_preregistration']['sha256']}`), que no se toca.", "",
          f"Estado del dataset al congelar: {doc['dataset_state_at_freeze']}.", "",
          "## Lo que fija", ""] + [f"- **{k}**: {v}" for k, v in doc["pins"].items()] + [
          "", "## Números (los lee el analizador; los umbrales de las letras se transcriben de las reglas congeladas)", "",
          "| clave | valor |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in ANNEX_NUMBERS.items()] + [
          "", "## Expectativa (E13)", "", doc["expectation_E13"] + ".", "",
          "## Lo que no puede decidir", ""] + [f"- {x}" for x in doc["what_this_cannot_decide"]]
    (OUTDIR / "binning_annex.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"anexo congelado: {ANNEX} sha256 {sha}")
    return 0


def schema() -> int:
    """Día de los bytes: columnas, papeles y recuento de registros de table4.dat para el schema_report (no imprime
    velocidades ni perfiles)."""
    readme, table = RAW / "ReadMe", RAW / TABLE
    if not (readme.exists() and table.exists()):
        raise SystemExit(f"FALLO CERRADO: faltan ReadMe o {TABLE} en {RAW}")
    text = readme.read_text(encoding="latin-1")
    cols = parse_readme(text, TABLE)
    for c in cols:
        print(f"{c['start']:4d}-{c['end']:4d}  {c['kind']}  {c['unit']:8s} {c['label']:10s} {c['explanation']}")
    print("papeles:", json.dumps(assign_roles(cols), ensure_ascii=False))
    lines = sum(1 for ln in table.read_text(encoding="latin-1").splitlines() if ln.strip())
    print(f"registros: {lines} (declarados en el ReadMe: {readme_record_count(text, TABLE)})")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"prereg": prereg, "analyze": analyze, "annex": annex, "schema": schema}.get(mode, lambda: print(__doc__) or 2)())
