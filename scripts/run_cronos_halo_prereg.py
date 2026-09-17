#!/usr/bin/env python
"""Congela la preinscripción del Nivel A del frente 5 ANTES de generar
ninguna condición inicial: halo aislado NFW de 1e11 M☉ con y sin Cronos
v3 a la amplitud única A_Sculptor (fila cronos-amplitud-unica), con el
cierre cosmológico como control de exclusión y 10·A_Sculptor como
control de respuesta.

Uso: python scripts/run_cronos_halo_prereg.py

DECLARACIONES (contrato): (i) el borrador del autor (17-sep) pedía N ≥ 1e6
y 5 Gyr; el presupuesto del entorno (4 núcleos, árbol Barnes–Hut en
numba) obliga a N = 4e5 y la duración fijada abajo — se declaran como
límite de recursos, no como elección post hoc; (ii) la maquinaria se
desarrolló con pilotos a N ≤ 5e4 (0.5 Gyr) y cortes de 0.1 Gyr a N = 4e5
para calibrar el estimador del campo, el paso mínimo y la conservación
de energía con el campo congelado — esos pilotos mostraron ya la
DIRECCIÓN del efecto (contracción del interior) y se declaran aquí como
observación de desarrollo, con el mismo papel que el control externo de
E5: no fijan umbrales; (iii) los umbrales se fijan por ruido entre
semillas y resolución, y las expectativas (E13) por el cálculo de
validez: D_F = 1 en r ≈ 0.9 kpc ⟹ contracción del interior, posiblemente
progresiva, sin núcleo kpc.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cronos.cronos_v3 import ALPHA0_INV_MAX  # noqa: E402
from cronos.halo_nbody import A_SCULPTOR, nfw_structural  # noqa: E402
from dynamics.cronos_amplitude_validity import (  # noqa: E402
    cosmological_closure_amplitude,
    force_dominance_ratio,
    nfw_halo,
    radius_where_force_ratio_is_one,
    rho_mean_matter_today,
)

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-17_cronos_halo_nivelA"
H0, OMEGA_M = 67.86705532886631, 0.32629884997700576   # θ de fondo congelado (12-sep)
SYSTEM = {"M200_msun": 1e11, "c": 10.0, "H0": H0, "N": 400000, "soft_plummer_kpc": 0.1,
          "r_decay_factor": 0.3, "ics": "NFW truncado exponencialmente (Kazantzidis+2004), "
          "distribución isótropa de Eddington; pares A/B con la misma semilla; sin bariones ni canales"}
INTEGRATOR = {"theta": 0.7, "eta_acc": 0.025, "eta_dyn": 0.02, "eta_cross": 0.1,
              "dt_max_gyr": 0.0512, "n_levels": 12, "brute_max": 500, "profile_every_ticks": 8,
              "field_tau_avg_myr": 50.0, "field_k_inner": 64, "field_smooth": 1.0,
              "scheme": "drift-all + kick en el punto medio del paso individual (potencias de 2); "
                        "gravedad Barnes–Hut (pytreegrav, spline h = 2.8·ε_soft) o fuerza bruta para "
                        "≤ brute_max activos; campo de Cronos de campo medio esférico (ln M(ln r) con "
                        "spline suavizante, media móvil τ_avg, radio suavizado r_s = sqrt(r² + ε_soft²))"}
SNAPSHOTS_GYR = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
T_END_GYR = 2.0
ARMS = {
    "a": {"label": "newtoniano (referencia)", "cronos": False, "amplitude": "A_sculptor",
          "seeds": [1, 2, 3], "t_end_gyr": T_END_GYR},
    "b": {"label": "Cronos v3 completo (fuerza + fricción con compuerta + lapse) a A_Sculptor",
          "cronos": True, "amplitude": "A_sculptor", "seeds": [1, 2, 3], "t_end_gyr": T_END_GYR},
    "b_static": {"label": "puerta numérica: Cronos a A_Sculptor con el campo CONGELADO en t = 0 (K + W + U_Cronos se conserva)",
                 "cronos": True, "amplitude": "A_sculptor", "seeds": [1], "t_end_gyr": 0.25, "field_static": True},
    "b_res": {"label": "control de convergencia del campo: como b con k_inner = 128 (radio interior del campo medio ×2 en partículas)",
              "cronos": True, "amplitude": "A_sculptor", "seeds": [1], "t_end_gyr": T_END_GYR, "field_k_inner": 128},
    "c": {"label": "control de exclusión: cierre cosmológico ρ_c = 200·ρ̄_m (A ≈ 41.5) con parada al violar el régimen débil",
          "cronos": True, "amplitude": "A_cosmological", "seeds": [1], "t_end_gyr": T_END_GYR,
          "stop_on_weak_violation": True},
    "cprime": {"label": "control de respuesta: 10·A_Sculptor (régimen débil válido; D_F = 1 en ≈ 2 kpc)",
               "cronos": True, "amplitude": "ten_A_sculptor", "seeds": [1], "t_end_gyr": 0.5},
}
GATES = {"weak_regime_eps_max": 1e-3, "energy_rel_tol_newton": 5e-3, "energy_rel_tol_frozen": 5e-3,
         "equilibrium_dlog10_max": 0.10}
STOP_RULES = {"runaway_well_speed_kms": 1000.0, "max_wall_hours_per_run": 3.0}
RULES = {"bands_kpc": {"inner_0p1_0p4": [0.1, 0.4], "inner_0p4_1": [0.4, 1.0], "core_1_2p3": [1.0, 2.3],
                       "fit_2p3_5": [2.3, 5.0], "outer_5_20": [5.0, 20.0]},
         "band_min_log10_shift": 0.10, "band_sigma_factor": 3.0,
         "fit_window_kpc": [0.5, 5.0], "cored_delta_rmse_min": 0.05, "cored_rc_min_kpc": 0.5,
         "runaway_final_ratio_min": 3.0, "interior_resolved_N_min": 2000,
         "control_min_log10_shift": 0.10, "resolution_control_max_log10": 0.15}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    p = nfw_structural(SYSTEM["M200_msun"], SYSTEM["c"], H0)
    h = nfw_halo(SYSTEM["M200_msun"], SYSTEM["c"], H0, SYSTEM["soft_plummer_kpc"])
    rho0 = rho_mean_matter_today(H0, OMEGA_M)
    A_cosmo = cosmological_closure_amplitude(ALPHA0_INV_MAX, rho0, 200.0)
    DF = force_dominance_ratio(h, A_SCULPTOR)
    import numpy as np
    at = lambda rk: float(DF[np.argmin(np.abs(h["r_kpc"] - rk))])  # noqa: E731
    D_phi_soft = float(299792.458 ** 2 * A_SCULPTOR * h["rho"][0] ** 1.5 / h["abs_phi"][0])
    doc = {
        "title": "Nivel A del frente 5 — halo aislado NFW (1e11 M☉, c = 10) con y sin Cronos v3 a la "
                 "amplitud única A_Sculptor; controles de exclusión (cierre cosmológico) y de respuesta (10·A_Sculptor)",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
        "generator": "scripts/run_cronos_halo_prereg.py",
        "question": "¿Produce Cronos v3, a la amplitud que la dinámica de las enanas exige (A_Sculptor, 5E), "
                    "núcleos kpc en un halo aislado en equilibrio — o modifica su interior en otro sentido?",
        "amplitudes": {"A_sculptor": A_SCULPTOR, "A_cosmological": A_cosmo,
                       "units": "(M_sol/pc^3)^(-3/2); epsilon_c = A * rho^(3/2)",
                       "source": "fila cronos-amplitud-unica (results/2026-09-17_cronos_amplitude_validity)"},
        "system": SYSTEM,
        "system_derived": {"r_s_kpc": p["r_s"], "r200_kpc": p["r200"], "rho_s_msun_kpc3": p["rho_s"],
                           "m_particle_approx_msun": p["M200"] * 1.35 / SYSTEM["N"],
                           "N_within_2p3kpc_approx": int(1.41e9 / (p["M200"] * 1.35 / SYSTEM["N"])),
                           "N_within_0p4kpc_approx": int(5.4e7 / (p["M200"] * 1.35 / SYSTEM["N"]))},
        "integrator": INTEGRATOR,
        "snapshots_gyr": SNAPSHOTS_GYR,
        "arms": ARMS,
        "physics_b": "Cor. 11.3 completo, los tres términos juntos: Γ = (3/2)(ρ̇/ρ)ε_c Θ(ρ̇) con ρ̇ "
                     "lagrangiana = ∂_tρ + v_r ∂_rρ sobre el perfil; F_extra = +c²∇ε_c radial; "
                     "N = 1 + Φ_N/c² − ε_c en kick y drift. c = 299792.458 km/s.",
        "validity_at_t0": {"D_phi_at_soft": D_phi_soft,
                           "D_F_at_soft": at(SYSTEM["soft_plummer_kpc"]), "D_F_0p4kpc": at(0.4), "D_F_1kpc": at(1.0),
                           "D_F_2p3kpc": at(2.3), "r_DF_equals_one_kpc": radius_where_force_ratio_is_one(h, A_SCULPTOR),
                           "eps_c_max_galactic": float(A_SCULPTOR * h["rho"][0] ** 1.5),
                           "eps_c_max_cosmological": float(A_cosmo * h["rho"][0] ** 1.5)},
        "gates": GATES,
        "stop_rules": STOP_RULES,
        "gates_text": {"regime": "régimen débil ε_c < 1e-3 en r ≥ ε_soft y N > 0 en todas las comprobaciones "
                                 "del brazo b; si falla → INDETERMINADO por régimen. D_Φ y D_F se PUBLICAN, no "
                                 "son puerta: la ley se integra completa aunque la fuerza de Cronos domine",
                       "numerical": "|ΔE/E| ≤ 5e-3 en el brazo a (K + W) y en b_static (K + W + U_Cronos con el "
                                    "campo congelado: la única corrida Cronos con energía conservada por "
                                    "construcción); en los brazos con campo dinámico ΔE se PUBLICA, no es puerta "
                                    "(el campo medio depende del tiempo). Equilibrio del brazo a: |Δlog10 ρ| ≤ "
                                    "0.10 dex en [1, 10] kpc (media de semillas)",
                       "stop_rules": "una corrida Cronos se detiene (y toma instantánea final) cuando la velocidad "
                                     "del pozo v_well = sqrt(2c²ε_c(ε_soft)) supera 1000 km/s — el modelo ha "
                                     "abandonado cualquier régimen de enana (σ ~ 10 km/s) y el resto de la corrida "
                                     "no añade información — o al superar 3 h de pared; la comparación b/a se hace "
                                     "en el instante final de cada b con la instantánea de a más cercana"},
        "rules": RULES,
        "outcomes": {
            "order": "puertas → C → B → A → INDETERMINADO; nunca se ajustan umbrales tras ver los números",
            "C_tension": "el brazo b (media de semillas) prefiere la forma cored en [0.5, 5] kpc con Δrmse ≥ 0.05 "
                         "dex y r_c ≥ 0.5 kpc, y el brazo a NO → núcleo kpc por Cronos: contrario a la expectativa "
                         "(D_F = 0.09 en 2.3 kpc) ⟹ sospecha de error primero (paso, estimador de ρ̇, unidades de c²)",
            "B_interior_modified": "sin núcleo kpc, pero |log10(ρ_b/ρ_a)| > max(0.10, 3σ_semillas(a)) en alguna banda; "
                                   "se publica el signo. Expectativa declarada (E13, no lectura favorable): «más "
                                   "concentrado» — D_F > 1 dentro de 0.9 kpc, y la fricción de compuerta drena "
                                   "energía cinética durante la contracción. Subcaso runaway: M_b(<0.4)/M_a(<0.4) "
                                   "creciente monótono tras 1 Gyr y > 3 al final (colapso progresivo). Si B ocurre, "
                                   "el modelo operativo (ΛCDM + Cronos galáctico) predice enanas más concentradas "
                                   "que ΛCDM en ≲ 1 kpc — predicción falsable, contraria a la lectura fuerte del corpus. "
                                   "El control b_res (k_inner = 128) decide si el RITMO del colapso está convergido en la "
                                   "resolución del campo (|log10(M_res/M_b)(<0.4 kpc)| ≤ 0.15 en el instante común): si no, "
                                   "se publica «colapso progresivo cuyo ritmo depende de la resolución del campo medio: "
                                   "la ley operativa no tiene predicción convergida con este instrumento»",
            "A_boring": "b indistinguible de a en todas las bandas (dentro del ruido) → «a la amplitud galáctica "
                        "congelada, Cronos v3 no produce núcleos kpc; el interior ≤ 0.4 kpc queda indeterminado "
                        "por resolución si N_a(<0.4 kpc) < 2000»",
            "INDETERMINADO": "puerta de régimen o numérica violada; ambos brazos cored (relajación numérica); "
                             "corridas ausentes; o control c′ sin efecto (|log10(ρ_c′/ρ_a)| < 0.10 en [0.4, 1) kpc "
                             "a 0.5 Gyr: implementación en duda, se retiene todo el resultado)",
            "controls": "c (cierre cosmológico) debe pararse en t = 0 por régimen débil (ε_c ≈ 17 en ε_soft): "
                        "registra la exclusión, no informa del núcleo; c′ (10·A_Sculptor, régimen débil válido, "
                        "D_F = 1 en ≈ 2 kpc) debe alterar el interior — si no, la implementación no responde",
        },
        "expectations_E13": {"direction": "contracción del interior (r ≲ 1 kpc), progresiva (runaway: ρ↑ ⟹ ε_c ∝ ρ^{3/2} ↑ "
                                          "⟹ pozo más profundo), probablemente hasta la regla de parada; sin núcleo kpc; "
                                          "ritmo probablemente dependiente de la resolución del campo",
                             "basis": "cálculo de validez: D_F(0.4 kpc) = 7.4, D_F(1 kpc) = 0.73; pilotos de "
                                      "desarrollo a N ≤ 5e4 (M(<0.4 kpc) ×4–5 en 0.5 Gyr) y cortes de 0.1 Gyr a N = 4e5 "
                                      "(campo congelado ×2.3; dinámico con k_inner = 64 ×4, con k_inner = 16 ×10 y "
                                      "energía −32 %: el ritmo del colapso depende del radio interior del campo) — "
                                      "observación de desarrollo, no fija umbrales",
                             "interior_resolution": "N(<0.4 kpc) ≈ 140 en t = 0 a N = 4e5: la banda [0.1, 0.4] kpc "
                                                    "se publica con su ruido; el veredicto fino del interior exige zoom"},
        "development_declaration": {
            "pilots": "N = 2e4 y 5e4 durante 0.3–0.5 Gyr (brazos a, b, b con campo congelado, 10·A_S); cortes de "
                      "0.1 Gyr a N = 4e5 (campo congelado: ΔE/E = −3.8e-4 con n_levels = 12, M(<0.4 kpc) ×2.3; "
                      "con n_levels = 9 el interior se calentaba y vaciaba: artefacto, no física) para fijar "
                      "Δt_min y la conservación de energía",
            "what_they_fixed": "estimador del campo (spline de ln M, k_inner = 64, τ_avg = 50 Myr, radio suavizado "
                               "r_s), Δt_min = 51.2 Myr/2^12 = 0.0125 Myr con criterio de cruce η_cross = 0.1 dentro "
                               "de 10 ε_soft y salto exacto entre eventos; las reglas de parada; NO se tocó ningún "
                               "umbral de desenlace con ellos",
            "budget": "N ≥ 1e6 y 5 Gyr del borrador del autor son inviables aquí (árbol ~2 s por evaluación a 1e6; "
                      "un brazo b a N = 4e5 cuesta ~220 s por 0.1 Gyr); N = 4e5 y 2 Gyr (≥ 17 t_dyn en 2.3 kpc, "
                      "≥ 100 en 0.4 kpc; c′ 0.5 Gyr) son el límite de recursos de una sesión",
        },
        "prohibitions": {"no_data": True, "no_threshold_tuning": True, "no_change_to_A_sculptor": True,
                         "no_change_to_prior_artifacts": True},
        "declared_frontiers": [
            "campo de Cronos en aproximación de campo medio esférico (la versión SPH por partícula es del Nivel B)",
            "halo sin bariones ni canales ρ_id/ρ_lat: aísla el efecto de Cronos; no es una enana completa",
            "el interior ≤ 0.4 kpc está marginalmente resuelto (~140 partículas en t = 0)",
            "el brazo c′ solo comprueba que la implementación responde; su amplitud no es del modelo",
            "la lectura fuerte (sin CDM) no se contrasta aquí: tiene otra amplitud y otra preinscripción",
        ],
    }
    (OUTDIR / "preregistration.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Preinscripción Nivel A — {doc['title']}\n",
          f"Congelada {doc['frozen_utc']} en el commit `{sha[:9]}` (contiene el generador), ANTES de generar ninguna "
          "condición inicial de las corridas preinscritas. Pilotos de desarrollo declarados en el JSON.", "",
          f"Sistema {SYSTEM['M200_msun']:.0e} M☉, c = {SYSTEM['c']}, N = {SYSTEM['N']}, ε_soft = {SYSTEM['soft_plummer_kpc']} kpc; "
          f"D_F(ε_soft) = {doc['validity_at_t0']['D_F_at_soft']:.0f}, D_F = 1 en {doc['validity_at_t0']['r_DF_equals_one_kpc']:.2f} kpc.", "",
          "Brazos: " + ", ".join(f"{k} ({v['label']}; semillas {v['seeds']}; {v['t_end_gyr']} Gyr)"
                                 for k, v in ARMS.items()) + ".", "",
          f"Puertas: {json.dumps(GATES)}. Reglas: {json.dumps(RULES, ensure_ascii=False)}.", "",
          "Desenlaces: C (núcleo por Cronos: sospecha de error) / B (interior modificado, signo publicado; subcaso runaway) / "
          "A (aburrido) / INDETERMINADO. Expectativa E13: contracción del interior, sin núcleo kpc.",
          "", "Sin datos; umbrales congelados; fronteras en el JSON."]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Preinscripción Nivel A congelada en {OUTDIR} (commit {sha[:9]})")


if __name__ == "__main__":
    main()
