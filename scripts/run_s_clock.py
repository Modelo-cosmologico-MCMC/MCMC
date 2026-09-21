#!/usr/bin/env python
"""El reloj S como simulador de consistencia — recorrido S₀ → S_{1,001}
con umbrales impuestos (Prop. 8.1), barrido en δ₀ y diagnóstico del
modo emergente. Publica results/2026-09-21_s_clock/.

No es un experimento con desenlaces preinscritos: es la comprobación
interna (E8) de que las estaciones del tramo pre-geométrico encajan
entre sí cuando se recorren en un solo bucle, con cada eslabón no
derivado declarado como tal (nucleación, umbrales, formas de sellado,
diagonal, cuantos de confirmación, diccionario).

Uso: python scripts/run_s_clock.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.s_clock import (  # noqa: E402
    DECLARED_FORMS,
    STATUS,
    delta0_metastability_max,
    emergent_diagnostic,
    radial_landscape,
    run_clock,
)
from mass_program.B7_empalme import delta0_required  # noqa: E402
from mcmc_ontology import constants as C  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_s_clock"
DELTA0_PRIMARY = 0.01
DELTA0_SCAN = [0.003, 0.01, 0.03, None, 0.1, 0.2]   # None → δ_H del empalme (H.2.4)
EMERGENT_DELTA0 = [0.01, 0.05]
EMERGENT_TAU = [0.0, 1e-3, 1e-2, 1e-1, 1.0]


def _slim(run: dict) -> dict:
    """El recorrido sin la trayectoria densa (se guarda aparte)."""
    return {k: v for k, v in run.items() if k != "trajectory"}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    d0max = delta0_metastability_max()
    delta_H = delta0_required()
    primary = run_clock(DELTA0_PRIMARY)
    scan = []
    for d0 in DELTA0_SCAN:
        d0v = delta_H if d0 is None else d0
        ld = radial_landscape(d0v)
        row = {"delta0": d0v, "label": "δ_H (empalme H.2.4, valor requerido)" if d0 is None else "valor de prueba",
               "metastable": ld["metastable"], "barrier_height_over_T0": (ld["barrier_height"] / ld["T0_full"]) if ld["metastable"] else None}
        if ld["metastable"] and ld["rho_esc"] is not None:
            r = run_clock(d0v)
            ev = {e["kind"]: e for e in r["events"]}
            row.update({"ran": True, "T0_full": r["T0"], "T0_law": r["T0_law_3_4"],
                        "tilt_correction": r["checks"]["S0"]["T0_scaling_3_4"]["tilt_correction"],
                        "sigma_hat_descent": r["checks"]["descent"]["descent_finished"]["sigma_hat_final"],
                        "S_equals_f_max_diff": r["checks"]["descent"]["S_equals_f_identity"]["max_abs_diff"],
                        "all_derived_checks_pass": all(
                            v["pass"] for sec in ("S0", "descent") for v in r["checks"][sec].values()
                            if isinstance(v, dict) and v.get("pass") is not None),
                        "sigma_events": {k: ev[k]["sigma"] for k in ("colapso_1D", "colapso_2D", "colapso_3D")},
                        "m_H_GeV": ev["Florencia"]["m_H_GeV"], "beta3": ev["Florencia"]["beta3_from_empalme"],
                        "theta_flow_final": r["checks"]["diagonal"]["flow"]["theta_final"],
                        "diagonal_crossed_by_flow": r["checks"]["diagonal"]["flow"]["crossed"]})
        else:
            row.update({"ran": False, "reason": f"sin falso vacío metastable (δ0 > δ0_max = {d0max:.4f}): la inclinación borra la barrera"})
        scan.append(row)
    emergent = [emergent_diagnostic(d0, tau) for d0 in EMERGENT_DELTA0 for tau in EMERGENT_TAU]

    doc = {"kind": "simulador de consistencia (E8), sin desenlaces preinscritos", "status": STATUS,
           "executed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "declared_forms": DECLARED_FORMS, "delta0_metastability_max_default_shapes": d0max,
           "delta_H_empalme": delta_H, "primary": _slim(primary), "delta0_scan": scan,
           "emergent_diagnostic": emergent,
           "what_is_not_claimed": [
               "ningún umbral emerge: los colapsos se disparan en 0.009/0.099/0.999 por la Ley de la Década (calibrada, frente 2)",
               "la nucleación (Γ₀) no se calcula: el reloj arranca en el punto de escape con σ = 0 declarado",
               "las leyes de sellado de c_eff y m_eff tienen forma paramétrica declarada, no la ec. (5.3)",
               "la diagonal θ = π/4 no la cruza el flujo del Basal: se impone para la entrega",
               "V3D y Florencia son cuantos declarados tras el residuo de descarga",
               "el estado entregado en S = 1,001 no es legible por la cosmología (diccionario ausente); m_H depende de δ₀ (input) y β₃ es condicional",
               "el modo emergente es diagnóstico: el diccionario τ no es derivable"]}
    (OUTDIR / "s_clock.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTDIR / "primary_trajectory.json").write_text(json.dumps(primary["trajectory"], ensure_ascii=False) + "\n", encoding="utf-8")

    p = primary
    ck = p["checks"]
    md = ["# El reloj S como simulador de consistencia — recorrido S₀ → S_{1,001} (21-sep-2026)\n",
          f"Commit `{sha[:9]}`. Modo con estatuto: umbrales IMPUESTOS (Prop. 8.1). δ₀ primario = {DELTA0_PRIMARY} "
          f"(valor de prueba: el tratado no asigna valor a δ₀). Formas del Basal: m̄ = {p['config']['m_bar']}, "
          f"b̄ = {p['config']['b_bar']}, ē = {p['config']['e_bar']}, C0 = {p['config']['C0']}.", "",
          f"**Lectura obligatoria**: {STATUS}.", "",
          "## Hallazgo del simulador: la metastabilidad tiene un δ₀ máximo", "",
          f"Con la inclinación −η·χ (η = ē·δ₀³) el falso vacío deja de existir por encima de **δ₀_max = {d0max:.4f}** "
          "para las formas por defecto: la Obs. 8.6 (D(S₀) > 0) es necesaria, no suficiente. Por debajo, el falso "
          f"vacío está en ρ_fv ≃ η/(√2·M0²) sobre el eje θ = 0 (residuo ⟨χ⟩ de la Prop. 3.4) y la barrera vale "
          f"{ck['S0']['metastability_with_tilt']['barrier_height_over_T0']:.3f}·T₀ en δ₀ = {DELTA0_PRIMARY}. "
          f"T₀ medida en el paisaje completo supera la ley c̄·δ₀³ en {ck['S0']['T0_scaling_3_4']['tilt_correction']*100:.1f} % "
          "(corrección de la inclinación, O(δ₀^{1/2}) relativa; la ley sin inclinación se reproduce a 1e-15).", "",
          "## Recorrido primario", "",
          "| estación | S (reloj f) | S = ∫Σ̇dσ/T₀ | σ | álgebra | firma | estatuto |", "|---|---|---|---|---|---|---|"]
    for e in p["events"]:
        s_int = f"{e['S_integrated']:.5f}" if "S_integrated" in e else "—"
        md.append(f"| {e['kind']} | {e['S']:.3f} | {s_int} | {e['sigma']:.1f} | "
                  f"{e.get('algebra', e.get('algebra_after', '—'))} | {e.get('signature', e.get('signature_after', '—'))} | {e['status']} |")
    dd = ck["descent"]
    md += ["", f"Identidad S ≡ f (Teo. 4.5, flujo proyectado): |S − f|_max = {dd['S_equals_f_identity']['max_abs_diff']:.1e}. "
           f"Monotonía: max ΔV/T₀ = {dd['monotonia_4_5']['max_dV_over_T0']:.1e}; producción entrópica mínima "
           f"{dd['produccion_entropica_4_5']['min']:.1e}; exclusión: {'sí' if dd['exclusion_4_7']['pass'] else 'NO'}; salida al polo de masa: "
           f"θ_final = {dd['exit_to_mass_pole_3_5']['theta_final']:.3f}. Descenso hasta f = 1 − ε_res en σ̂ = σ·δ₀² = "
           f"{dd['descent_finished']['sigma_hat_final']:.3f} ({dd['descent_finished']['steps']} pasos).", "",
           f"Sellados (forma declarada): c_eff congelado en S = {C.S_SEALS['C2']} con u = {ck['seals']['c_eff']['u_frozen']:.4f} "
           f"(máximo |du/dS| = {ck['seals']['c_eff']['c_eff_max']:.3f} en S = {ck['seals']['c_eff']['S_of_max']:.3f}); "
           f"m_eff congelado en S = {C.S_SEALS['C3']} con m = {ck['seals']['m_eff']['m_frozen']:.4f}.", "",
           f"Diagonal: el flujo NO la cruza (θ_max = {ck['diagonal']['flow']['theta_max']:.3f}); para la entrega se impone "
           f"θ_imp(S) con cruce en S = {ck['diagonal']['imposed']['S_at_crossing']}.", ""]
    fl = p["events"][-1]
    md += ["Florencia (S = 1,001): C(4,0) → C(3,1), firma " + str(fl["signature_after"]) + "; control negativo (dos giros): "
           + str(fl["negative_control_two_rotations"]) + f"; RP en la loncha: autovalor mínimo {fl['rp_slice_min_eig_J_plus']:.1e} "
           f"(J = +1) frente a {fl['rp_negative_control_J_minus']:.3f} (J = −1, control negativo); identidad m_H = √(2β₃)v₃ "
           f"comprobada con β₃ = {fl['beta3_from_empalme']:.4f} ⟹ m_H = {fl['m_H_GeV']:.1f} GeV en δ₀ = {DELTA0_PRIMARY} "
           "(depende de δ₀: no es predicción).", "",
           "Apretón de manos con la cosmología: Sello de Newton G_cosmo/G_N = "
           f"{ck['handshake_cosmology']['newton_seal_9_3']['ratio_at_seal']:.1f} en (1,1); Atlas sano en λ_K = 1 + ε_K; recuperación "
           f"de ΛCDM con ε = 0: desviación máxima {ck['handshake_cosmology']['lcdm_recovery_A_1']['max_rel_dev_eps0']:.1e}. "
           "**Diccionario primordial → cosmológico: NO EXISTE** — el estado entregado no es legible por `cosmology/`.", "",
           "## Barrido en δ₀", "",
           "| δ₀ | metastable | barrera/T₀ | corrección inclinación | σ̂ descenso | m_H [GeV] | diagonal por flujo |",
           "|---|---|---|---|---|---|---|"]
    for r in scan:
        if r["ran"]:
            md.append(f"| {r['delta0']:.4f} ({r['label']}) | sí | {r['barrier_height_over_T0']:.4f} | {r['tilt_correction']*100:+.1f} % | "
                      f"{r['sigma_hat_descent']:.3f} | {r['m_H_GeV']:.1f} | {'sí' if r['diagonal_crossed_by_flow'] else 'no'} |")
        else:
            md.append(f"| {r['delta0']:.4f} ({r['label']}) | NO | — | — | — | — | — ({r['reason']}) |")
    md += ["", "m_H(δ_H) reproduce el PDG por construcción de δ_H (valor requerido, auditoría de circularidad Obs. 12.2), no por "
           "predicción.", "",
           "## Modo emergente (diagnóstico, sin estatuto)", "",
           "Acoplos por las β de Fokker–Planck (cierre canónico) con diccionario τ declarado; reloj S = ∫Σ̇dσ/T₀; colapso donde D = 0.", "",
           "| δ₀ | τ | D inicial | D mínimo | cruces D = 0 (S) | S final | fin |", "|---|---|---|---|---|---|---|"]
    for r in emergent:
        md.append(f"| {r['delta0']} | {r['tau']:g} | {r['D_initial']:.2e} | {r['D_min']:.2e} | {r['S_crossings'] or '—'} | "
                  f"{r['S_final']:.4f} | {r['stop_reason']} |")
    md += ["", "Lectura: con el cierre canónico D se hunde (Obs. 8.6) pero en ninguna corrida cruza cero antes del fin del "
           "recorrido: para τ grande M0² cruza cero ANTES que D (el falso vacío se destabiliza por la masa, no por la "
           "espinodal) y para τ pequeño el descenso termina, o se agota el presupuesto de pasos con el flujo ya muy "
           "lento sobre un potencial aplanado, con D > 0. En el rango explorado los colapsos NO emergen del "
           "Cruce de Victoria — resultado diagnóstico que depende del diccionario τ (frente 2), publicado sin verdicto.", "",
           "## Lo que NO afirma", ""] + [f"- {s}" for s in doc["what_is_not_claimed"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"δ0_max = {d0max:.4f}; T0_full/T0_law − 1 = {ck['S0']['T0_scaling_3_4']['tilt_correction']:.3f}; "
          f"|S−f|max = {dd['S_equals_f_identity']['max_abs_diff']:.1e}; eventos: {[e['kind'] for e in p['events']]}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
