#!/usr/bin/env python
"""Preinscripción «Oort–K_z» (frente 5c): la vecindad solar acota la
amplitud A de la Ley de Cronos débil por su fuerza vertical.

    python scripts/run_oort_kz.py prereg    # congela la preinscripción (sha256)
    python scripts/run_oort_kz.py prereg --erratum-of <preregistration_v1.json>
                                            # v2 por ERRATUM: reglas copiadas VERBATIM de la v1,
                                            # manifest nuevo; el analizador exige números idénticos
    python scripts/run_oort_kz.py run       # calcula bajo la preinscripción (consume local_kz_bounds)
    python scripts/run_oort_kz.py analyze   # aplica la regla congelada; falla cerrado sin ella

El término de Cronos g_C = −c²dε_c/dz actúa sobre las estrellas
trazadoras como una fuerza vertical adicional y la cinemática lo lee
como densidad dinámica EFECTIVA: Δρ_eff(0) = (1/4πG)·dg_C/dz|₀ y ΔΣ_eff(1.1)
= g_C(1.1 kpc)/(2πG). La observación deja un margen ρ_dyn − ρ_bar y
Σ_dyn − Σ_bar (tabla transcrita `local_kz_bounds`, verificada por el autor
contra los resúmenes de arXiv el 22-sep-2026; bytes de las tablas no
descargados: aviso propagado) que Cronos comparte con la materia oscura
local; la regla le concede TODO el margen (la cota más laxa).

ERRATUM (22-sep-2026): la v1 (21-sep) citaba Σ(<1.1 kpc) = 68 ± 4 como
Bovy & Tremaine 2012; la cifra es de Bovy & Rix 2013 (ApJ 779, 115). El
valor no cambia. Un erratum NO es una preinscripción nueva: la v2 copia
las reglas de la v1 sin tocarlas (el analizador falla cerrado si difieren)
y publica que los números coinciden con los de la v1 (falla cerrado si no:
un cambio de valores exigiría preinscribir de nuevo, no corregir).

Brazos (amplitud): A_Sculptor (la amplitud única del 5E), 0.05·A_Sculptor
(el brazo del frente 5b) y la cota A_2σ derivada. Losa DECLARADA; su
sensibilidad (alturas ×½, ×2: Δρ_eff ∝ h⁻²) se publica como banda, no
como brazo.

Desenlaces por brazo, con la losa declarada (regla congelada):
    compatible — z_max ≤ 2 (el exceso sobre el margen cabe en 2σ)
    tensión    — 2 < z_max ≤ 5
    excluido   — z_max > 5
    INDETERMINADO — dataset no AVAILABLE, margen no positivo o sha alterado
Se publica además A_2σ/A_Sculptor (con su banda por la losa) y el
z-score de la columna por separado.

Piloto declarado (ANTES de congelar; ronda del criterio de Cronos–Jeans,
21-sep mañana, con la misma losa): límite de Oort efectivo de Cronos con
A_Sculptor ≈ 0.75 M☉/pc³ frente a ρ_dyn(0) ≈ 0.10 y cota A ≲ 0.05·A_S por
g_C ≤ 0.1·K_z en 300 pc. La estructura de desenlaces se escribe
conociéndolo; lo que la regla congelada añade es la comparación formal
con el margen observacional transcrito, la cota A_2σ, la columna a 1.1
kpc y la banda de la losa. Ningún umbral se ajusta al piloto.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.local_kz import (  # noqa: E402
    A_SCULPTOR,
    SLAB_DECLARED,
    load_bounds,
    room,
    significance,
)
from mcmc_ontology.data_registry import manifest_path  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_oort_kz"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=OUTDIR.parent.parent).stdout.strip()


def prereg(erratum_of: str | None = None) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    v1 = None
    if erratum_of is not None:
        v1_path = Path(erratum_of)
        if not v1_path.exists():
            raise SystemExit("FALLO CERRADO: la preinscripción v1 del erratum no existe")
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    bounds = load_bounds()
    doc = {
        "title": "Oort–K_z: la vecindad solar acota la amplitud de la Ley de Cronos débil por su fuerza vertical",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_oort_kz.py prereg", "generator_commit_must_precede_freeze": True,
        "declared": {
            "law": "ε_c = A·ρ^{3/2}; g_C = −c²dε_c/dz (Cor. 11.3, Ley de Cronos débil)",
            "effective_density": "Δρ_eff(0) = (1/4πG)·dg_C/dz|₀ = −(3/2)c²A√ρ(0)·ρ''(0)/(4πG); ΔΣ_eff(1.1) = g_C(1.1 kpc)/(2πG)",
            "slab": SLAB_DECLARED,
            "slab_sensitivity_band": "alturas de escala × 0.5 y × 2 (Δρ_eff ∝ h⁻²): banda publicada, no brazo",
            "dataset": ("local_kz_bounds (transcripción verificada por el autor contra los resúmenes de arXiv; bytes de las "
                        "tablas NO descargados — aviso propagado)" if bounds.get("values_verified_against_arxiv_abstracts")
                        else "local_kz_bounds (transcripción; bytes oficiales NO verificados — aviso propagado)"),
            "dataset_manifest_sha256": _sha(manifest_path("local_kz_bounds")),
            "room": "ρ_dyn(0) − ρ_bar(0) con ρ_dyn = media ponderada HF2000 + MPH2015; Σ_dyn(1.1) − Σ_bar(1.1) con "
                    "Bovy & Rix 2013 − MPH2015; errores en cuadratura; TODO el margen para Cronos",
            "arms": {"A_sculptor": A_SCULPTOR, "A_0p05": 0.05 * A_SCULPTOR, "A_2sigma": "derivada (lineal en A)"},
        },
        "rules": {
            "z_score": "z = (Δ_eff − margen)/σ_margen por observable; z_max = max(plano, columna)",
            "compatible": {"z_max_le": 2.0}, "tension": {"z_max_in": [2.0, 5.0]}, "excluded": {"z_max_gt": 5.0},
            "indeterminate": "dataset no AVAILABLE, margen ≤ 0 o sha del manifest distinto del congelado",
            "publish": ["A_2σ/A_Sculptor con banda de la losa", "z del plano y de la columna por separado",
                        "el aviso de procedencia del dataset"],
            "verdict": "letra por brazo con la losa declarada; ninguna letra es afirmación sobre el tratado: la Ley de "
                       "Cronos débil con A_Sculptor es la hipótesis del 5E y este contraste la sitúa frente a la dinámica local (E8)",
        },
        "pilot_declared": "ronda del criterio (21-sep mañana): límite de Oort efectivo de Cronos con A_Sculptor ≈ 0.75 M☉/pc³ "
                          "frente a ρ_dyn(0) ≈ 0.10; cota A ≲ 0.05·A_S por g_C ≤ 0.1·K_z en 300 pc. La estructura de desenlaces se "
                          "escribió conociéndolo; ningún umbral se ajustó.",
        "what_this_cannot_decide": ["si la Ley de Cronos débil es la ley correcta (forma ε_c(ρ): tarea del diccionario)",
                                    "la amplitud A del frente 5b (0.05·A_S es brazo, no resultado)",
                                    "la losa real (declarada; la banda mide su peso)"],
    }
    if v1 is not None:
        # ERRATUM: reglas, piloto y límites copiados VERBATIM de la v1 — nada se retoca
        doc["rules"] = v1["rules"]
        doc["pilot_declared"] = v1["pilot_declared"]
        doc["what_this_cannot_decide"] = v1["what_this_cannot_decide"]
        doc["erratum"] = {
            "of_file": v1_path.name, "of_sha256": _sha(v1_path), "of_frozen_utc": v1["frozen_utc"],
            "reason": ("cita de Σ(<1.1 kpc) corregida (Bovy & Tremaine 2012 → Bovy & Rix 2013, ApJ 779, 115); ρ_bar(0) "
                       "declarado como resta; valores verificados por el autor contra los resúmenes de arXiv; el "
                       "manifest del dataset cambia por esos metadatos, ningún valor ni error cambia"),
            "rules_identical_to_v1": True,
            "values_must_match_v1": "el analizador compara cada número con runs de la v1 y falla cerrado si difieren",
        }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). "
          f"sha256 de `preregistration.json`: `{sha}`.", ""]
    if v1 is not None:
        md += ["## Erratum", "", f"Versión 2 por erratum de `{doc['erratum']['of_file']}` (sha256 `{doc['erratum']['of_sha256']}`, "
               f"congelada {doc['erratum']['of_frozen_utc']}): {doc['erratum']['reason']}. Reglas, piloto y límites copiados "
               "verbatim de la v1; el analizador exige que los números coincidan con los de la v1.", ""]
    md += ["## Declarado", ""] + [f"- **{k}**: {v}" for k, v in doc["declared"].items()] + [
          "", "## Reglas (congeladas)", ""] + [f"- **{k}**: {v}" for k, v in doc["rules"].items()] + [
          "", "## Piloto declarado", "", doc["pilot_declared"], "",
          "## Lo que no puede decidir", ""] + [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _load_prereg() -> dict:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada")
    return json.loads(PREREG.read_text(encoding="utf-8"))


def run() -> int:
    pre = _load_prereg()
    if _sha(manifest_path("local_kz_bounds")) != pre["declared"]["dataset_manifest_sha256"]:
        raise SystemExit("FALLO CERRADO: el manifest del dataset cambió tras la congelación")
    bounds = load_bounds()          # falla cerrado si no está AVAILABLE
    arms = {"A_sculptor": A_SCULPTOR, "A_0p05": 0.05 * A_SCULPTOR}
    rows = {}
    for name, A in arms.items():
        rows[name] = {str(h): significance(A, bounds, h) for h in (0.5, 1.0, 2.0)}
    a2 = rows["A_sculptor"]["1.0"]["A_2sigma_over_A_sculptor"] * A_SCULPTOR
    rows["A_2sigma"] = {str(h): significance(a2, bounds, h) for h in (0.5, 1.0, 2.0)}
    RUNS.write_text(json.dumps({"preregistration_sha256": _sha(PREREG), "code_commit": _git_head(),
                                "executed_utc": datetime.now(timezone.utc).isoformat(),
                                "dataset_caveat": bounds["provenance_caveat"], "room": room(bounds), "arms": rows},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, r in rows.items():
        s = r["1.0"]
        print(f"{name}: A = {s['A']:.3e}; Δρ_eff(0) = {s['increments']['d_rho_eff_0_msun_pc3']:.4f}, z_plano = {s['z_rho_0']:.1f}, "
              f"z_columna = {s['z_Sigma_1p1']:.1f}; A_2σ/A_S = {s['A_2sigma_over_A_sculptor']:.4f}")
    return 0


def analyze() -> int:
    pre = _load_prereg()
    if not RUNS.exists():
        raise SystemExit("FALLO CERRADO: no hay corridas")
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    if runs["preregistration_sha256"] != _sha(PREREG):
        raise SystemExit("FALLO CERRADO: la preinscripción cambió después de las corridas")
    R = pre["rules"]
    rm = runs["room"]
    if rm["rho_room_0"] <= 0.0 or rm["Sigma_room_1p1"] <= 0.0:
        verdict = dict.fromkeys(runs["arms"], "INDETERMINADO")
    else:
        verdict = {}
        for name, r in runs["arms"].items():
            z = r["1.0"]["z_max"]
            verdict[name] = ("compatible" if z <= R["compatible"]["z_max_le"] else
                             "tensión" if z <= R["tension"]["z_max_in"][1] else "excluido")
    band = {name: {h: r[h]["A_2sigma_over_A_sculptor"] for h in ("0.5", "1.0", "2.0")} for name, r in runs["arms"].items()}
    erratum_check = None
    if "erratum" in pre:
        # ERRATUM: las reglas deben ser las de la v1 y los números deben coincidir con los de la v1
        v1_path = OUTDIR / pre["erratum"]["of_file"]
        if not v1_path.exists() or _sha(v1_path) != pre["erratum"]["of_sha256"]:
            raise SystemExit("FALLO CERRADO: la preinscripción v1 del erratum falta o cambió")
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        if v1["rules"] != R:
            raise SystemExit("FALLO CERRADO: las reglas de la v2 difieren de la v1 — eso no es un erratum, es otra preinscripción")
        runs_v1_path = OUTDIR / pre["erratum"]["of_file"].replace("preregistration", "runs")
        if not runs_v1_path.exists():
            raise SystemExit("FALLO CERRADO: faltan las corridas de la v1 con las que comparar")
        runs_v1 = json.loads(runs_v1_path.read_text(encoding="utf-8"))
        max_dev = 0.0
        for name, r in runs["arms"].items():
            for h in ("0.5", "1.0", "2.0"):
                for key in ("z_rho_0", "z_Sigma_1p1", "A_2sigma_over_A_sculptor"):
                    max_dev = max(max_dev, abs(r[h][key] - runs_v1["arms"][name][h][key]))
        for key in ("rho_room_0", "rho_room_0_err", "Sigma_room_1p1", "Sigma_room_1p1_err"):
            max_dev = max(max_dev, abs(rm[key] - runs_v1["room"][key]))
        if not max_dev <= 1e-9:
            raise SystemExit(f"FALLO CERRADO: los números difieren de la v1 (desviación máxima {max_dev:.3e}): un cambio de "
                             "valores exige preinscripción nueva, no erratum")
        erratum_check = {"v1_file": v1_path.name, "v1_sha256": pre["erratum"]["of_sha256"], "rules_identical": True,
                         "max_abs_deviation_from_v1": max_dev, "values_identical_to_v1": True}
    res = {"preregistration_sha256": runs["preregistration_sha256"], "code_commit_analysis": _git_head(),
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), "verdict_by_arm": verdict,
           "A_2sigma_over_A_sculptor_band_by_slab_h": band["A_sculptor"], "room": rm,
           "dataset_caveat": runs["dataset_caveat"], "erratum_check": erratum_check,
           "reading": ("el término de Cronos con la amplitud única del 5E excede el margen dinámico local en el plano; la "
                       "cota A_2σ es la que deja todo el margen a Cronos y depende de la losa declarada (∝ h²: banda "
                       "publicada); la columna a 1.1 kpc no constriñe. Contraste interno (E8) sobre valores transcritos; "
                       "su estado de verificación es el que declara el aviso de procedencia del dataset")}
    (OUTDIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Oort–K_z — resultado bajo la preinscripción `{runs['preregistration_sha256'][:12]}`", "",
          f"Corridas en `{runs['code_commit'][:9]}`; análisis en `{res['code_commit_analysis'][:9]}`.", "",
          f"**AVISO DE PROCEDENCIA**: {runs['dataset_caveat']}", ""] + ([
          f"**ERRATUM**: v2 de `{erratum_check['v1_file']}` (sha256 `{erratum_check['v1_sha256'][:12]}…`); reglas idénticas; "
          f"desviación máxima de los números respecto de la v1: {erratum_check['max_abs_deviation_from_v1']:.1e} (idénticos).", ""]
          if erratum_check else []) + [
          f"Margen observacional: plano ρ_dyn − ρ_bar = {rm['rho_room_0']:.4f} ± {rm['rho_room_0_err']:.4f} M☉/pc³; columna "
          f"Σ_dyn − Σ_bar = {rm['Sigma_room_1p1']:.1f} ± {rm['Sigma_room_1p1_err']:.1f} M☉/pc².", "",
          "| brazo | A | Δρ_eff(0) [M☉/pc³] | z plano | ΔΣ_eff(1.1) [M☉/pc²] | z columna | letra (losa declarada) | A_2σ/A_S (h×½ / h×1 / h×2) |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in runs["arms"].items():
        s = r["1.0"]
        md.append(f"| {name} | {s['A']:.3e} | {s['increments']['d_rho_eff_0_msun_pc3']:.4f} | {s['z_rho_0']:+.1f} | "
                  f"{s['increments']['d_sigma_eff_msun_pc2']:.2f} | {s['z_Sigma_1p1']:+.1f} | **{verdict[name]}** | "
                  f"{band[name]['0.5']:.4f} / {band[name]['1.0']:.4f} / {band[name]['2.0']:.4f} |")
    md += ["", res["reading"] + ".", "", "## Lo que no decide", ""] + [f"- {s}" for s in pre["what_this_cannot_decide"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"veredicto por brazo: {verdict}; A_2σ/A_S (banda): {band['A_sculptor']}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "prereg":
        err = sys.argv[sys.argv.index("--erratum-of") + 1] if "--erratum-of" in sys.argv else None
        raise SystemExit(prereg(err))
    raise SystemExit({"run": run, "analyze": analyze}.get(mode, lambda: print(__doc__) or 2)())
