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

from core.basal import T0_analytic  # noqa: E402
from core.delta0_circle import (  # noqa: E402
    DECISION_A,
    W_max_required_both,
    W_max_required_decided,
)
from core.landscape_priors import PRIORS  # noqa: E402
from core.s_clock import (  # noqa: E402
    DECLARED_FORMS,
    STATUS,
    ClockConfig,
    SClock,
    delta0_metastability_max,
    delta0_metastability_max_analytic,
    emergent_diagnostic,
    kappa1_tilt,
    naturalness_sweep,
    radial_landscape,
    run_clock,
    tau_decade_table,
)
from mass_program.B7_empalme import delta0_required, delta0_required_full  # noqa: E402
from mcmc_ontology import constants as C  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_s_clock"
DELTA0_PRIMARY = 0.01
DELTA0_SCAN = [0.003, 0.01, 0.03, None, 0.1, 0.2]   # None → δ_H del empalme (H.2.4)
EMERGENT_DELTA0 = [0.01, 0.05]
EMERGENT_TAU = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
E_BAR_SWEEP = [0.5, 1.0, 2.0]
NATURALNESS_N = 4000
NATURALNESS_PROBES = (0.012, 0.0581)
# decisión B (22-sep): D_ent DECLARADA como fracción de la altura de la barrera
KRAMERS_D_ENT_OVER_BARRIER = (0.1, 1.0, 10.0)


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

    # --- v1 (21-sep, tarde): κ₁, las dos T₀ del Techo, δ₀_max analítico y
    #     naturalidad, inestabilidad de masa y la hipótesis τ_d ----------------
    kappa1 = kappa1_tilt()
    tilt_table = []
    for d0 in (0.003, 0.01, 0.03, delta_H, 0.1):
        ld = radial_landscape(d0)
        corr = ld["T0_full"] / T0_analytic(d0) - 1.0
        tilt_table.append({"delta0": d0, "measured": corr, "first_order": kappa1 * d0 ** 0.5,
                           "ratio": corr / (kappa1 * d0 ** 0.5), "residual_over_delta0": (corr - kappa1 * d0 ** 0.5) / d0})
    w_both = {"delta_H": W_max_required_both(delta_H), "delta_0.012": W_max_required_both(0.012)}
    d0max_e = [{"e_bar": e, "bisection": delta0_metastability_max(e_bar=e),
                "analytic": delta0_metastability_max_analytic(e_bar=e), "times_e_bar_sq": delta0_metastability_max_analytic(e_bar=e) * e ** 2}
               for e in E_BAR_SWEEP]
    naturalness = [naturalness_sweep(NATURALNESS_N, prior=p, delta0_probe=NATURALNESS_PROBES) for p in PRIORS]
    tau_table = {str(d0): tau_decade_table(d0) for d0 in EMERGENT_DELTA0}
    # hipótesis declarada τ_d = τ_k de la tabla (la que haría caer cada inestabilidad en su umbral)
    tau_hyp = [emergent_diagnostic(d0, tau_table[str(d0)]["tau_k"][0], tau_per_dim=tuple(tau_table[str(d0)]["tau_k"]))
               for d0 in EMERGENT_DELTA0]

    # --- v2 (22-sep): decisiones A, B, C del autor, derivadas del tratado ------
    delta_H_full = delta0_required_full()
    w_decided = W_max_required_decided()
    kramers_runs = []
    for d0 in (DELTA0_PRIMARY, delta_H_full):
        ld = radial_landscape(d0)
        for frac in KRAMERS_D_ENT_OVER_BARRIER:
            r = SClock(ClockConfig(delta0=d0, nucleation="kramers", D_ent=frac * ld["barrier_height"])).run()
            nuc = r["checks"]["S0"]["nucleation"]
            kramers_runs.append({"delta0": d0, "D_ent_over_barrier": frac, "D_ent": frac * ld["barrier_height"],
                                 "Gamma_K": nuc["Gamma0"], "sigma_wait": nuc["sigma_wait_before_nucleation"],
                                 "prefactor": nuc["kramers"]["prefactor"], "exponent": nuc["kramers"]["exponent"],
                                 "f0": r["checks"]["descent"]["descent_finished"].get("f0", 0.0),
                                 "events": [e["kind"] for e in r["events"]],
                                 "S_equals_f_max_diff": r["checks"]["descent"]["S_equals_f_identity"]["max_abs_diff"]})
    gamow_ref = {d0: SClock(ClockConfig(delta0=d0, nucleation="gamow")).run()["checks"]["S0"]["nucleation"]["Gamma0"]
                 for d0 in (DELTA0_PRIMARY, delta_H_full)}
    bounce_ctrl = SClock(ClockConfig(delta0=DELTA0_PRIMARY, nucleation="bounce")).run()["checks"]["S0"]["nucleation"]
    decisions = {
        "A": {"decision": DECISION_A, "delta_H_law": w_decided["delta_H_law"], "delta_H_full": delta_H_full,
              "delta_H_shift": w_decided["delta_H_shift"],
              "W_max_law_3_4_at_delta_H_law": w_decided["W_max_law_3_4_at_delta_H_law"],
              "W_max_full_at_delta_H_law": w_decided["W_max_full_at_delta_H_law"],
              "W_max_decided_T0_full_at_delta_H_full": w_decided["W_max_decided"],
              "reading": "el Lema 10.3 iguala W_max a T₀_full; con δ_H recalculado sobre el mismo paisaje (λ_Ad_full = λ_H) el "
                         "Techo requerido es T₀_full(δ_H_full); se publican las tres cadenas para la trazabilidad"},
        "B": {"decision": "n = 1: Kramers (Axioma 4, D_ent declarada) con Gamow como cota; bounce O(3)/O(4) solo control negativo",
              "kramers_runs": kramers_runs, "gamow_Gamma0_reference": gamow_ref,
              "bounce_negative_control": {"status": bounce_ctrl["status"], "f0": bounce_ctrl["bounce"]["f0"]},
              "reading": "Γ_K(0) = 0 por el prefactor ∝ δ₀²; D_ent no la fija el corpus (diccionario τ, frente 2): la ley "
                         "de Γ₀ es condicional a D_ent, no a n_dim"},
        "C": {"decision": DECLARED_FORMS["collapse_trigger"],
              "canonical_closure_crosses_D_zero": any(r["S_crossings"] for r in emergent),
              "mass_instability_events_in_emergent_runs": sum(1 for r in emergent if r["mass_instability"]),
              "reading": "bajo el cierre canónico D no cruza cero en ninguna corrida; M0² = 0 sí ocurre y se publica como "
                         "diagnóstico (no-evento para el estado ocupado); el colapso del tratado exige β del frente 2"},
    }
    doc = {"kind": "simulador de consistencia (E8), sin desenlaces preinscritos", "status": STATUS,
           "v2_decisions_22sep": decisions,
           "executed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "declared_forms": DECLARED_FORMS, "delta0_metastability_max_default_shapes": d0max,
           "delta_H_empalme": delta_H, "primary": _slim(primary), "delta0_scan": scan,
           "emergent_diagnostic": emergent,
           "v1_tilt_first_order": {"kappa1": kappa1, "form": "ē·√κ₊/(√2·c̄)", "table": tilt_table,
                                   "reading": "T₀_full/T₀_ley − 1 = κ₁√δ₀ + O(δ₀) con residuo ≈ −0.27·δ₀ (publicado)"},
           "v1_W_max_both_T0": w_both,
           "v1_delta0_max_vs_e_bar": {"table": d0max_e, "form": "δ₀_max = 2·g_max(m̄, b̄, C0)²/ē² (0.1028·ē⁻² por defecto)"},
           "v1_naturalness": naturalness,
           "v1_mass_instability": {"tau_decade_table": tau_table,
                                   "flips_in_emergent_runs": [{"delta0": r["delta0"], "tau": r["tau"], "mass_instability": r["mass_instability"],
                                                               "stop_reason": r["stop_reason"], "S_final": r["S_final"]} for r in emergent],
                                   "tau_per_dim_hypothesis": tau_hyp,
                                   "reading": "S_flip medido reproduce la forma cerrada (cociente 0.94–1.00); «Década ⟺ τ_k ≈ 11.8·δ₀·10⁻ᵏ» "
                                              "es el diccionario τ leído desde los umbrales calibrados, no una derivación (E13)"},
           "what_is_not_claimed": [
               "ningún umbral emerge: los colapsos se disparan en 0.009/0.099/0.999 por la Ley de la Década (calibrada, frente 2)",
               "la nucleación (Γ₀) no se calcula en la corrida primaria: el reloj arranca en el punto de escape con σ = 0 declarado; con la decisión B (n = 1) Γ₀ se publica en modo 'kramers' (D_ent declarada) o 'gamow' (cota), y 'bounce' es control negativo",
               "las leyes de sellado de c_eff y m_eff tienen forma paramétrica declarada, no la ec. (5.3)",
               "la diagonal θ = π/4 no la cruza el flujo del Basal: se impone para la entrega",
               "V3D y Florencia son cuantos declarados tras el residuo de descarga",
               "el estado entregado en S = 1,001 no es legible por la cosmología (diccionario ausente); m_H depende de δ₀ (input) y β₃ es condicional",
               "el modo emergente es diagnóstico: el diccionario τ no es derivable",
               "la inestabilidad de masa (M0² = 0) es un evento publicado del modo diagnóstico, no un colapso con estatuto; la tabla τ_k reformula el diccionario, no lo deriva (E13)",
               "W_max se publica con las tres cadenas (ley 3.4 en δ_H_ley; paisaje completo en δ_H_ley; paisaje completo en δ_H_full): la decisión A del autor (22-sep) nombra T₀_full, pero el valor de W_max sigue sin asignar en el tratado (frente 4)",
               "las decisiones A, B y C son del autor, derivadas del texto del tratado y declaradas en DECLARED_FORMS/DECISION_A: el código las ejecuta, no las demuestra (E8); D_ent y las β que hagan D → 0 siguen siendo del frente 2",
               "la unidad de S tras Florencia no se deriva de T₀ (hueco: fila diccionario-unidad-S-post-florencia)"]}
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
           "recorrido: para τ > 0 M0² cruza cero ANTES que D (el falso vacío se destabiliza por la masa, no por la "
           "espinodal; evento `inestabilidad_masa`, tabla siguiente) y el bucle sigue sobre un potencial aplanado hasta "
           "agotar el presupuesto de pasos con D > 0. En el rango explorado los colapsos NO emergen del "
           "Cruce de Victoria — resultado diagnóstico que depende del diccionario τ (frente 2), publicado sin verdicto.", "",
           "## v1 (21-sep, tarde): corrección de la inclinación a primer orden", "",
           f"κ₁ = ē·√κ₊/(√2·c̄) = **{kappa1:.3f}** con las formas por defecto; T₀_full/T₀_ley − 1 = κ₁·√δ₀ + O(δ₀).", "",
           "| δ₀ | medido | κ₁√δ₀ | cociente | residuo/δ₀ |", "|---|---|---|---|---|"]
    for r in tilt_table:
        md.append(f"| {r['delta0']:.4f} | {r['measured']:.4f} | {r['first_order']:.4f} | {r['ratio']:.3f} | {r['residual_over_delta0']:+.3f} |")
    wb = w_both["delta_H"]
    md += ["", "## v1: el Techo W_max con las dos T₀ (círculo de δ₀)", "",
           f"En δ_H = {delta_H:.4f}: ley 3.4 c̄·δ_H³ = **{wb['T0_law_3_4']:.4e}**; paisaje completo V_fv − V_tv = **{wb['T0_full_landscape']:.4e}** "
           f"(+{wb['tilt_correction']*100:.1f} %). En 0.012: {w_both['delta_0.012']['T0_law_3_4']:.4e} frente a "
           f"{w_both['delta_0.012']['T0_full_landscape']:.4e}. Cuál nombra el Lema 10.3 es decisión del autor (A); la fila "
           "circulo-delta0 hereda la corrección como condicional, no la resuelve.", "",
           "## v1: δ₀_max en forma cerrada y naturalidad", "",
           "δ₀_max = 2·g_max(m̄, b̄, C0)²/ē² ∝ ē⁻² (el origen es el falso vacío exacto sin inclinación: δ₀_max = ∞ con ē = 0, "
           "raíz que la malla de la v0 no veía).", "",
           "| ē | bisección | forma cerrada | δ₀_max·ē² |", "|---|---|---|---|"]
    for r in d0max_e:
        md.append(f"| {r['e_bar']} | {r['bisection']:.5f} | {r['analytic']:.5f} | {r['times_e_bar_sq']:.5f} |")
    md += ["", f"Sobre los paisajes viables de `landscape_priors` (n = {NATURALNESS_N} por prior; filtro fértil como en la cartografía):", "",
           "| prior | fracción con falso vacío en δ₀ = 0.012 | en δ₀ = δ_H = 0.0581 | mediana δ₀_max | forma cerrada vs bisección |",
           "|---|---|---|---|---|"]
    for r in naturalness:
        fr = r["fraction_metastable_at"]
        md.append(f"| {r['prior']} | {fr['0.012']:.3f} | {fr['0.0581']:.3f} | {r['delta0_max_percentiles'][50]:.4f} | {r['closed_form_vs_bisection_max_rel_err']:.1e} |")
    md += ["", "Lectura: la metastabilidad en δ_H NO es genérica sobre el prior (una fracción minoritaria de paisajes la admite); "
           "cartografía publicada sin veredicto — el prior es declarado.", "",
           "## v1: inestabilidad de masa (modo diagnóstico) y la tabla τ_k (E13)", "",
           "En modo emergente el cruce M0² = 0 se publica como evento y el bucle continúa (se detiene solo con C0 ≤ 0). "
           "Forma cerrada de primer orden: S_flip ≃ δ₀·[b̄ − √(b̄² − 6C0m̄²)]/(24·a·τ·C0) = "
           f"{tau_table[str(EMERGENT_DELTA0[0])]['coefficient_S_flip_tau_over_delta0']:.4f}·δ₀/τ.", "",
           "| δ₀ | τ | S_flip medido | S_flip 1er orden | cociente | S final | fin |", "|---|---|---|---|---|---|---|"]
    for r in emergent:
        mi = r["mass_instability"]
        if mi:
            md.append(f"| {r['delta0']} | {r['tau']:g} | {mi['S_flip']:.5f} | {mi['S_flip_first_order']:.5f} | {mi['ratio']:.3f} | {r['S_final']:.4f} | {r['stop_reason']} |")
        else:
            md.append(f"| {r['delta0']} | {r['tau']:g} | — | — | — | {r['S_final']:.4f} | {r['stop_reason']} |")
    md += ["", "Tabla τ_k — el τ que haría caer la inestabilidad de masa en cada umbral de la Década, τ_k = S_flip(τ = 1)·δ₀/S_k "
           "(≈ 11.8·δ₀·10⁻ᵏ): es el diccionario τ leído al revés desde los umbrales calibrados, **no una derivación** (E13: el número no es señal).", "",
           "| δ₀ | τ₀ (S = 0.009) | τ₁ (S = 0.099) | τ₂ (S = 0.999) |", "|---|---|---|---|"]
    for d0, t in tau_table.items():
        md.append(f"| {d0} | {t['tau_k'][0]:.4e} | {t['tau_k'][1]:.4e} | {t['tau_k'][2]:.4e} |")
    md += ["", "Hipótesis declarada τ_d (opcional): τ por dimensión igual a la tabla τ_k y M0² repuesto en cada inestabilidad — publica dónde caerían los colapsos:", ""]
    for r in tau_hyp:
        md.append(f"- δ₀ = {r['delta0']}: colapsos en S = {[round(s, 4) for s in r['S_crossings']]} con disparos {sorted(set(r['collapse_triggers']))}; fin: {r['stop_reason']}.")
    A, B, Cd = decisions["A"], decisions["B"], decisions["C"]
    md += ["", "## v2 (22-sep): las decisiones A, B y C del autor, ejecutadas", "",
           "**A — qué T₀ nombra el Lema 10.3: la del paisaje completo.** El contenido físico del Lema es T₀(δ′) ≤ W_max; "
           "c̄δ′³ es la expresión de la Prop. 3.4, cuya corrección el reloj midió. Con la inclinación en los dos lados "
           "(Techo y empalme H.8), δ_H se recalcula como raíz de λ_Ad_full(δ) = λ_H:", "",
           "| cadena | δ_H | W_max requerido |", "|---|---|---|",
           f"| ley 3.4 en δ_H_ley | {A['delta_H_law']:.5f} | {A['W_max_law_3_4_at_delta_H_law']:.4e} |",
           f"| paisaje completo en δ_H_ley | {A['delta_H_law']:.5f} | {A['W_max_full_at_delta_H_law']:.4e} |",
           f"| **decidida**: paisaje completo en δ_H_full | **{A['delta_H_full']:.5f}** ({A['delta_H_shift']*100:+.1f} %) | **{A['W_max_decided_T0_full_at_delta_H_full']:.4e}** |",
           "", A["reading"] + ".", "",
           "**B — n = 1 con Kramers (Axioma 4) y Gamow como cota; el bounce es control negativo.** D_ent DECLARADA como "
           "fracción de la altura de la barrera:", "",
           "| δ₀ | D_ent/ΔV_b | prefactor | ΔV_b/D_ent | Γ_K | σ_espera = 1/Γ_K | Γ₀ Gamow (ref.) | S ≡ f |", "|---|---|---|---|---|---|---|---|"]
    for r in B["kramers_runs"]:
        md.append(f"| {r['delta0']:.4f} | {r['D_ent_over_barrier']:g} | {r['prefactor']:.3e} | {r['exponent']:.2f} | {r['Gamma_K']:.3e} | "
                  f"{r['sigma_wait']:.3e} | {B['gamow_Gamma0_reference'][r['delta0']]:.3e} | {r['S_equals_f_max_diff']:.1e} |")
    md += ["", f"Control negativo: {B['bounce_negative_control']['status']} (f₀ = {B['bounce_negative_control']['f0']:.4f}: la descarga entera "
           "caería dentro de la nucleación). " + B["reading"] + ".", "",
           "**C — «el colapso» es D = 0 (Cruce de Victoria).** " + Cd["reading"] + f" (cruces D = 0 en las corridas emergentes: "
           f"{'sí' if Cd['canonical_closure_crosses_D_zero'] else 'ninguno'}; eventos M0² = 0: {Cd['mass_instability_events_in_emergent_runs']}).", "",
           "## Lo que NO afirma", ""] + [f"- {s}" for s in doc["what_is_not_claimed"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"δ0_max = {d0max:.4f}; T0_full/T0_law − 1 = {ck['S0']['T0_scaling_3_4']['tilt_correction']:.3f}; "
          f"|S−f|max = {dd['S_equals_f_identity']['max_abs_diff']:.1e}; eventos: {[e['kind'] for e in p['events']]}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
