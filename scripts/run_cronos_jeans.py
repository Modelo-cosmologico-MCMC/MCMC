#!/usr/bin/env python
"""Ronda de registro H1 (21-sep-2026) — el criterio de Cronos–Jeans:
reproducción en el repositorio de la derivación y de las tres tablas del
autor (halo NFW del Nivel A, vecindad solar, Sculptor congelado) y del
test 1D del umbral. Publica results/2026-09-21_cronos_jeans/.

No es un experimento con desenlaces preinscritos: es una derivación (E8)
con expectativas declaradas (E13) y sin datos ingeridos. El test 1D
reproduce el UMBRAL; la ley de la tasa ∝ k es la puerta del test
preinscrito del frente 5 refundado, no de este artefacto.

Uso: python scripts/run_cronos_jeans.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cronos.cronos_jeans_1d import threshold_scan  # noqa: E402
from dynamics.cronos_jeans import STATUS, full_report  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_cronos_jeans"
AUTHOR_CONTROL = {
    "source": "nota del autor «Verificación del brief 20-sep y criterio de Cronos–Jeans», 21-sep-2026 "
              "(cronos_scripts/cronos_jeans_criterion.py, cj_1d_test.py)",
    "r_CJ_kpc": 0.709, "M_within_r_CJ": 1.610e8, "r_CJ_0p1A": 0.247, "r_CJ_0p05A": 0.180,
    "oort_cronos": 0.75, "oort_stars_only": 0.115, "A_bound_Kz_over_A_S": 0.048, "A_bound_stab_over_A_S": 0.158,
    "sculptor_sigma_los_10pc": 19.14, "sculptor_sigma_los_500pc": 2.63,
    "cj1d_rms_q0p8_t0p5": 0.032, "cj1d_rms_q1p2_t0p5": 0.095, "cj1d_rms_q2_t0p5": 0.436}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    rep = full_report()
    cj = threshold_scan()
    doc = {"kind": "derivación y cálculos de consistencia (E8), sin desenlaces preinscritos ni datos ingeridos",
           "status": STATUS, "executed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "author_control": AUTHOR_CONTROL, "report": rep, "cj_1d_threshold_scan": cj,
           "expectations_E13": [
               "un N-cuerpos con la ley local y q > 1 no converge: refinar la resolución del campo aumenta la tasa",
               "la masa dentro de r_CJ(A_S) = 0.71 kpc del halo del Nivel A colapsa en ~t_cross hasta la regla de parada (ρ ≈ 4.3 M☉/pc³ ⟺ 1.6e8 M☉ dentro de ≈ 0.2 kpc)",
               "el propio montaje congelado de Sculptor tiene q ≈ 2–2.4 dentro de ≈ 430 pc: el equilibrio que calibra A_Sculptor es inestable según el criterio",
               "la vecindad solar exige A ≤ 0.048·A_Sculptor (g_C ≤ 0.1·K_z en 300 pc) y A < 0.16·A_Sculptor (q < 1 en el plano)"],
           "what_is_not_claimed": [
               "ningún resultado observacional: las referencias (Oort, K_z, perfil de Sculptor) se citan como contexto, no se ingieren",
               "la ley de la tasa ∝ k no se mide aquí (fase lineal cortísima, modos sembrados por Poisson)",
               "A_Sculptor no se toca; ninguna preinscripción cambia; el Nivel A sigue INDETERMINADO"]}
    (OUTDIR / "cronos_jeans.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    n, s, sc = rep["nfw_nivelA"], rep["solar_neighbourhood"], rep["sculptor_frozen_5E"]
    md = ["# El criterio de Cronos–Jeans (derivación del autor, 21-sep-2026; reproducida en el repo)\n",
          f"Commit `{sha[:9]}`. A = A_Sculptor = {rep['A_sculptor']:.4g} (M☉/pc³)^(−3/2), congelada en 5E.", "",
          "**Criterio**: q ≡ (3/2)·c²·ε_c(ρ)/σ_1D² ≥ 1 ⟹ inestable a TODA longitud de onda; en el límite fluido "
          "ω² = (σ² − (3/2)c²ε_c)k² − 4πGρ, tasa ∝ k·√(q − 1)·σ (ultravioleta, sin longitud de Jeans propia). "
          "Un N-cuerpos con q > 1 no puede converger: refinar el campo aumenta la tasa.", "",
          f"## (1) Halo del Nivel A (NFW {n['M200']:.0e} M☉, c = {n['c']:g}, Jeans isótropo)", "",
          "| r [kpc] | ρ [M☉/pc³] | (3/2)c²ε_c [km²/s²] | σ_r² | q | D_F |", "|---|---|---|---|---|---|"]
    for r in n["rows"]:
        md.append(f"| {r['r_kpc']:.2f} | {r['rho_msun_pc3']:.3e} | {r['cronos_term_kms2']:.0f} | {r['sigma_r2_kms2']:.0f} | "
                  f"**{r['q']:.2f}** | {r['D_F']:.2f} |")
    md += ["", f"**r_CJ (q = 1) = {n['r_CJ_kpc']:.3f} kpc; M(<r_CJ) = {n['M_within_r_CJ']:.2e} M☉.** Con 0.1·A_S: r_CJ = "
           f"{n['r_CJ_for_A_fraction']['0.1']:.3f} kpc; con 0.05·A_S: {n['r_CJ_for_A_fraction']['0.05']:.3f} kpc. La regla de parada "
           f"v_well = {n['stop_rule']['v_well_kms']:.0f} km/s equivale a ρ = {n['stop_rule']['rho_msun_pc3']:.2f} M☉/pc³: la masa dentro de "
           f"r_CJ cabría en ≈ {n['stop_rule']['r_kpc_holding_M_within_r_CJ']:.2f} kpc.", "",
           "## (2) Vecindad solar (losa MPH15: estrellas 0.043, gas 0.041, MO 0.013 M☉/pc³)", "",
           "| z [pc] | ρ | c²ε_c [km²/s²] | v_well | g_C [(km/s)²/kpc] | K_z(losa) | g_C/K_z | q (σ_z = 20) |", "|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        md.append(f"| {r['z_pc']:.0f} | {r['rho_msun_pc3']:.3f} | {r['c2_eps_c_kms2']:.0f} | {r['v_well_kms']:.1f} | {r['g_C_kms2_per_kpc']:.0f} | "
                  f"{r['K_z_slab_kms2_per_kpc']:.0f} | **{r['g_C_over_K_z']:.2f}** | {r['q']:.2f} |")
    md += ["", f"Límite de Oort efectivo de Cronos: **{s['oort_limit_cronos_msun_pc3']:.2f} M☉/pc³** (solo estrellas "
           f"{s['oort_limit_cronos_stars_only']:.3f}) frente al medido ρ_dyn(0) = 0.10 ± 0.01 (referencia, no ingerida). "
           f"Cotas: g_C ≤ 0.1·K_z en 300 pc ⟹ A ≤ {s['A_bound_over_A_sculptor']:.3f}·A_S; q < 1 en el plano ⟹ A < "
           f"{s['A_bound_stability_over_A_sculptor']:.3f}·A_S.", "",
           f"## (3) Sculptor con el montaje congelado del 5E (M* = {sc['M_star']:.2e} M☉, Υ = {sc['upsilon']:g}, a₀ = {sc['a0_pc']:.0f} pc, β = {sc['beta']:g})", "",
           "| r [pc] | ρ* | (3/2)c²ε_c | σ_r²(N) | σ_r²(N+C) | q | g_C/g_N |", "|---|---|---|---|---|---|---|"]
    for r in sc["rows"]:
        md.append(f"| {r['r_pc']} | {r['rho_star_msun_pc3']:.3e} | {r['cronos_term_kms2']:.1f} | {r['sigma_r2_newton']:.2f} | "
                  f"{r['sigma_r2_newton_cronos']:.2f} | **{r['q']:.2f}** | {r['g_C_over_g_N']:.2f} |")
    md += ["", "| R [pc] | σ_los Newton | σ_los Newton + Cronos (A_S) | observado (referencia) |", "|---|---|---|---|"]
    for p in sc["profile"]:
        md.append(f"| {p['R_pc']:.0f} | {p['sigma_los_newton']:.2f} | **{p['sigma_los_newton_cronos']:.2f}** | ≈ 9–10, plano |")
    md += ["", f"Inestable (q ≥ 1) hasta r ≈ {sc['r_unstable_max_pc']:.0f} pc. {sc['shape_prediction']}. El promedio pesado por "
           f"luminosidad reproduce σ_obs = {sc['sigma_obs_kms']} km/s por construcción.", "",
           "## Test 1D del umbral (láminas, sin gravedad, σ = ρ₀ = L = 1)", "",
           "| q | rms(δ) inicial (Poisson) | t = 0.05 | 0.1 | 0.2 | 0.5 | factor | √(q−1) |", "|---|---|---|---|---|---|---|---|"]
    for r in cj["scan"]:
        a = r["rms_at"]
        md.append(f"| {r['q']} | {r['rms_initial']:.4f} | {a['0.05']:.3f} | {a['0.1']:.3f} | {a['0.2']:.3f} | {a['0.5']:.3f} | "
                  f"×{r['growth_factor_final']:.1f} | {r['fluid_rate_over_k_sigma']:.2f} |")
    md += ["", cj["reading"] + ".", "",
           "## Control externo (nota del autor) reproducido", "",
           f"r_CJ {AUTHOR_CONTROL['r_CJ_kpc']} → {n['r_CJ_kpc']:.3f}; Oort {AUTHOR_CONTROL['oort_cronos']} → "
           f"{s['oort_limit_cronos_msun_pc3']:.3f}; cota K_z {AUTHOR_CONTROL['A_bound_Kz_over_A_S']} → {s['A_bound_over_A_sculptor']:.3f}; "
           f"σ_los(10 pc) {AUTHOR_CONTROL['sculptor_sigma_los_10pc']} → {sc['profile'][0]['sigma_los_newton_cronos']:.2f}.", "",
           "## Expectativas declaradas (E13) y lo que NO afirma", ""]
    md += [f"- {e}" for e in doc["expectations_E13"]] + [""] + [f"- {e}" for e in doc["what_is_not_claimed"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"r_CJ = {n['r_CJ_kpc']:.3f} kpc; Oort_C = {s['oort_limit_cronos_msun_pc3']:.2f}; A ≤ {s['A_bound_over_A_sculptor']:.3f}·A_S; "
          f"σ_los(10 pc) = {sc['profile'][0]['sigma_los_newton_cronos']:.1f} km/s; 1D: {[round(r['growth_factor_final'],1) for r in cj['scan']]}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
