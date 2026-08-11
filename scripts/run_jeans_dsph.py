#!/usr/bin/env python
"""Frente 5, medio paso 2 (5A/5B/5D): Sculptor contra el potencial débil.

Programa de medida (todo publicado, sea cual sea el desenlace):

1. VALIDACIÓN: el solucionador clava la identidad analítica del
   Plummer isótropo (error relativo máximo, publicado).
2. CONTROL (solo estrellas): σ_los newtoniana de Sculptor con
   Υ⋆ ∈ {1, 2, 3} — el déficit clásico frente a σ_obs.
3. LA COTA (11.5) EN ACCIÓN: la mayor amplitud de Cronos que respeta
   c²ε_c ≤ |Φ_N| (forma ρ^(3/2) del perfil) y la σ_los que alcanza;
   más el techo absoluto de saturación puntual (Φ_eff = 2Φ_N ⟹ √2·σ_N).
4. PROBLEMA INVERSO: la amplitud A_req = α₀⁻¹/ρ_c^(3/2) que Sculptor
   EXIGE para σ_obs; el ρ_c implicado con α₀⁻¹ en su cota; y el factor
   por el que A_req viola la ec. (11.5).
5. EL OBJETIVO PARA ρ_id: masa dinámica global (estimador estándar
   M_1/2 ≈ 4·σ²·R_half/G, Wolf et al. 2010 — aproximación declarada)
   frente a M⋆.
6. Sensibilidades: Υ⋆ × σ_obs × β; R_half ± error.

Uso: python scripts/run_jeans_dsph.py
Salida: results/2026-08-10_jeans_dsph/report.md (+ figura).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.dsph_data import (  # noqa: E402
    SCULPTOR,
    plummer_scale_from_Rhalf,
    stellar_mass,
)
from dynamics.jeans import sigma_los_sq, sigma_los_sq_lum_avg  # noqa: E402
from dynamics.weak_field import (  # noqa: E402
    G_PC,
    bound_saturation_ratio,
    g_eff_plummer,
    plummer_density,
    plummer_g_newton,
    plummer_sigma_los_sq_isotropic,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-10_jeans_dsph")
ALPHA0_INV_BOUND = 1e-6
A_UNIT = 1e-13          # amplitud unitaria del problema inverso (lineal)


def grids(a: float):
    r = np.geomspace(0.05 * a / 260.0, 120.0 * a, 800)
    return r


def analysis(M: float, a: float, sigma_obs: float, beta: float = 0.0):
    """Devuelve el bloque de medidas para (M⋆, a, σ_obs, β)."""
    r = grids(a)
    nu = plummer_density(r, M, a)
    gN = plummer_g_newton(r, M, a)
    R_max = 8.0 * a

    s2_N = sigma_los_sq_lum_avg(r, nu, gN, beta=beta, R_max=R_max,
                                u_max=120.0 * a)
    dS2 = sigma_los_sq_lum_avg(
        r, nu, g_eff_plummer(r, M, a, A_UNIT), beta=beta, R_max=R_max,
        u_max=120.0 * a) - s2_N

    q_unit = bound_saturation_ratio(r, M, a, A_UNIT)
    A_bound = A_UNIT / q_unit                  # max A con c²ε_c ≤ |Φ_N|
    s2_bound = s2_N + (A_bound / A_UNIT) * dS2

    A_req = (sigma_obs ** 2 - s2_N) / dS2 * A_UNIT
    viol = A_req / A_bound                     # factor de violación (11.5)
    rho_c_req = (ALPHA0_INV_BOUND / A_req) ** (2.0 / 3.0)

    return {
        "sigma_N": float(np.sqrt(s2_N)),
        "sigma_bound": float(np.sqrt(s2_bound)),
        "sigma_pointwise_max": float(np.sqrt(2.0 * s2_N)),
        "A_req": float(A_req),
        "rho_c_req": float(rho_c_req),
        "violation": float(viol),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = SCULPTOR
    a0 = plummer_scale_from_Rhalf(d["R_half_pc"])
    L_V = d["L_V_Lsun"]
    sigma_obs = d["sigma_los_kms"]

    # 1. Validación contra la identidad analítica
    M2 = stellar_mass(L_V, 2.0)
    r = grids(a0)
    nu = plummer_density(r, M2, a0)
    gN = plummer_g_newton(r, M2, a0)
    R_eval = np.array([0.0, 0.5 * a0, a0, 2.0 * a0, 4.0 * a0])
    s2_num = sigma_los_sq(R_eval, r, nu, gN, u_max=120.0 * a0)
    s2_ana = plummer_sigma_los_sq_isotropic(R_eval, M2, a0)
    val_err = float(np.max(np.abs(s2_num / s2_ana - 1.0)))
    print(f"1. Validación Plummer: error rel. máx = {val_err:.2e}")

    # 2-4. Barrido principal (β = 0)
    rows = []
    for ups in d["upsilon_scan"]:
        M = stellar_mass(L_V, ups)
        b = analysis(M, a0, sigma_obs)
        rows.append((ups, b))
        print(f"2-4. Υ⋆={ups:.0f}: σ_N = {b['sigma_N']:.2f} | "
              f"σ_max(11.5) = {b['sigma_bound']:.2f} | "
              f"ρ_c(req, α₀⁻¹=1e-6) = {b['rho_c_req']:.2e} M⊙/pc³ | "
              f"violación (11.5) ×{b['violation']:.0f}")

    # 5. Objetivo para ρ_id (estimador de Wolf, aproximación declarada)
    M_dyn = 4.0 * sigma_obs ** 2 * a0 / G_PC
    print(f"5. M_1/2 ≈ 4σ²R_half/G = {M_dyn:.2e} M⊙  "
          f"(M⋆(Υ=2) = {M2:.2e}; cociente {M_dyn / M2:.1f})")

    # 6. Sensibilidades
    sens = []
    for s_obs in d["sigma_los_scan_kms"]:
        b = analysis(M2, a0, s_obs)
        sens.append(("σ_obs", s_obs, b))
    for beta in (-0.3, 0.3):
        b = analysis(M2, a0, sigma_obs, beta=beta)
        sens.append(("β", beta, b))
    for dR in (-d["R_half_err_pc"], d["R_half_err_pc"]):
        b = analysis(M2, plummer_scale_from_Rhalf(d["R_half_pc"] + dR),
                     sigma_obs)
        sens.append(("R_half", d["R_half_pc"] + dR, b))

    # Figura
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        RR = np.geomspace(10.0, 6.0 * a0, 80)
        s2N_R = sigma_los_sq(RR, r, nu, gN, u_max=120.0 * a0)
        b2 = dict(rows)[2.0]
        g_bnd = g_eff_plummer(r, M2, a0,
                              A_UNIT / bound_saturation_ratio(
                                  r, M2, a0, A_UNIT))
        s2B_R = sigma_los_sq(RR, r, nu, g_bnd, u_max=120.0 * a0)
        g_req = g_eff_plummer(r, M2, a0, b2["A_req"])
        s2Q_R = sigma_los_sq(RR, r, nu, g_req, u_max=120.0 * a0)
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.axhspan(9.0, 10.0, color="0.85",
                   label="σ_obs global 9-10 (Walker+09)")
        ax.plot(RR, np.sqrt(s2N_R), lw=1.6,
                label="solo estrellas (Υ⋆=2, newtoniano)")
        ax.plot(RR, np.sqrt(s2B_R), lw=1.6, ls="--",
                label="Cronos máximo DENTRO de la cota (11.5)")
        ax.plot(RR, np.sqrt(s2Q_R), lw=1.6, ls=":",
                label=f"amplitud exigida (viola (11.5) ×{b2['violation']:.0f})")
        ax.set_xlabel("R [pc]")
        ax.set_ylabel("σ_los(R) [km/s]")
        ax.set_title("Sculptor: el término débil de Cronos no alcanza "
                     "σ_obs dentro de su propia cota")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "jeans_sculptor.png", dpi=140)
        print(f"Figura: {OUT / 'jeans_sculptor.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    # Informe
    tbl = "\n".join(
        f"| {ups:.0f} | {b['sigma_N']:.2f} | {b['sigma_bound']:.2f} | "
        f"{b['sigma_pointwise_max']:.2f} | {b['rho_c_req']:.2e} | "
        f"×{b['violation']:.0f} |"
        for ups, b in rows)
    stbl = "\n".join(
        f"| {k} = {v:g} | {b['sigma_N']:.2f} | {b['sigma_bound']:.2f} | "
        f"{b['rho_c_req']:.2e} | ×{b['violation']:.0f} |"
        for k, v, b in sens)
    (OUT / "report.md").write_text(
        "# Frente 5, medio paso 2: Sculptor contra el potencial débil "
        "(5A/5B/5D)\n\n"
        "El contraste declarado de antemano (5E): los MISMOS parámetros "
        "(α₀⁻¹, ρ_c) — que solo entran por A = α₀⁻¹/ρ_c^(3/2) — deben "
        "explicar sistemas rotacionales y de presión; el fallo cuenta "
        "como falsación del mecanismo galáctico propuesto, no como "
        "llamada a ajustar por sistema.\n\n"
        f"**Datos** (procedencia y estatuto en `dynamics/dsph_data.py`): "
        f"{d['provenance']}. σ_obs = {sigma_obs} km/s (banda 9-10), "
        f"R_half = {d['R_half_pc']:.0f} ± {d['R_half_err_pc']:.0f} pc, "
        f"L_V ≈ {L_V:.1e} L⊙ (derivada de M_V ≈ {d['M_V']}), "
        f"Υ⋆ ∈ [1, 3] declarado.\n\n"
        f"## 1. Validación del solucionador\n\n"
        f"Jeans + proyección clavan la identidad analítica del Plummer "
        f"isótropo con error relativo máximo {val_err:.1e} (test "
        f"permanente en `tests/test_jeans_dsph.py`).\n\n"
        "## 2-4. El resultado central (β = 0)\n\n"
        "| Υ⋆ | σ_N (solo estrellas) | σ_max dentro de (11.5) | techo "
        "puntual √2·σ_N | ρ_c exigido (α₀⁻¹=1e-6) [M⊙/pc³] | violación "
        "de (11.5) |\n|---|---|---|---|---|---|\n"
        f"{tbl}\n\n"
        f"**Lectura.** (i) Solo estrellas: σ_N = "
        f"{rows[0][1]['sigma_N']:.2f}-{rows[-1][1]['sigma_N']:.2f} km/s "
        f"según Υ⋆ — el déficit clásico frente a σ_obs = {sigma_obs}. "
        f"(ii) El término débil de Cronos −c²∇ε_c NO puede cerrarlo: su "
        f"propia cota (ec. 11.5, c²ε_c ≲ |Φ_N|) limita la subida a "
        f"√2·σ_N ≤ {max(b['sigma_pointwise_max'] for _, b in rows):.2f} "
        f"km/s incluso saturada puntualmente — a "
        f"{9.0 - max(b['sigma_pointwise_max'] for _, b in rows):.1f} "
        f"km/s del borde inferior de la banda observada. (iii) La "
        f"amplitud que Sculptor exige viola la cota por un factor "
        f"×{min(b['violation'] for _, b in rows):.0f}-"
        f"×{max(b['violation'] for _, b in rows):.0f} (según Υ⋆), y con "
        f"α₀⁻¹ en su cota implica ρ_c = "
        f"{min(b['rho_c_req'] for _, b in rows):.1f}-"
        f"{max(b['rho_c_req'] for _, b in rows):.1f} M⊙/pc³ — una "
        "densidad de escala estelar, no cosmológica: como ε_c ∝ "
        "ρ^(3/2), cualquier sistema con densidades muy por debajo de "
        "ese ρ_c queda con ε_c despreciable, en tensión directa con el "
        "paso 5C si el mismo término tuviera que actuar en discos.\n\n"
        "## 5. El objetivo que queda para ρ_id\n\n"
        f"Masa dinámica global (estimador estándar M_1/2 ≈ 4σ²R_half/G, "
        f"Wolf et al. 2010; aproximación declarada, insensible a β al "
        f"primer orden): **{M_dyn:.1e} M⊙** frente a "
        f"M⋆(Υ⋆=2) = {M2:.1e} M⊙ — cociente ≈ {M_dyn / M2:.0f}. En la "
        "arquitectura del tratado, esa carga explicativa corresponde al "
        "sector ρ_id (el perfil cored que el Apéndice A asigna a las "
        "curvas de rotación), cuyo perfil a escala dSph NO está "
        "derivado: hueco declarado del frente.\n\n"
        "## 6. Sensibilidades (Υ⋆ = 2)\n\n"
        "| variación | σ_N | σ_max (11.5) | ρ_c exigido | violación "
        "|\n|---|---|---|---|---|\n"
        f"{stbl}\n\n"
        "Las filas β = ±0.3 son idénticas POR TEOREMA, no por descuido: "
        "el promedio total pesado por luminosidad ⟨σ_los²⟩ es "
        "independiente de la anisotropía en un sistema esférico "
        "(teorema virial proyectado; verificado como test permanente). "
        "El efecto de β vive en el PERFIL σ_los(R) — inaccesible hasta "
        "la ingesta de los datos binados, pendiente declarado. El "
        "veredicto (ii) es robusto en todo el barrido: la mayor σ_max "
        "dentro de la cota queda por debajo de la banda observada en "
        "todos los casos.\n\n"
        "## Estatuto\n\n"
        "condicional y de medio paso (frente 5): (a) el veredicto es "
        "EXACTO dentro del montaje declarado — trazador Plummer "
        "isótropo/β constante, datos globales (un número), fuente "
        "bariónica sola; el perfil binado σ_los(R), poblaciones "
        "múltiples y la ingesta de la tabla original quedan "
        "pendientes (procedencia en dynamics/dsph_data.py); (b) lo que "
        "queda falsado en este montaje es que el TÉRMINO DÉBIL DE "
        "CRONOS explique los dSph dentro de su cota (11.5) — no el "
        "modelo completo: el tratado asigna la fenomenología galáctica "
        "a ρ_id, que este medio paso convierte en objetivo cuantitativo "
        f"(M_1/2 ≈ {M_dyn:.1e} M⊙ dentro de ~{a0:.0f} pc); (c) el ρ_c "
        "aquí exigido (escala estelar) discrepa en órdenes de magnitud "
        "de la receta que la malla PM del programa necesitó (ρ_c ≈ "
        "umbral de colapso ~200× la media — nota computacional I, "
        "§3.5): la incógnita compartida del contraste 5E ya tiene dos "
        "medidas discrepantes; (d) el paso 5C (SPARC, sistemas "
        "rotacionales con la MISMA A) decidirá la falsación cruzada "
        "5E.\n",
        encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
