#!/usr/bin/env python
"""Frente 5, paso 5C (mitad estructural) + curva objetivo ρ_id.

Programa de medida (publicado sea cual sea el desenlace):

1. A_req de Sculptor RECALCULADA con la misma maquinaria del medio
   paso 2 (no un número copiado): Υ⋆ ∈ {1, 2, 3}.
2. FORMA del término en discos (identidades, independientes de A):
   v_cronos² ∝ x·e^(−3x/2) — pico en x = 2/3, caída exterior.
3. La MISMA A sobre la malla declarada de discos exponenciales que
   barre la población SPARC (Σ0 × R_d × ζ): dónde es grande y dónde
   muere el término, frente a dónde vive la discrepancia de masa
   (x ≳ 3). La ingesta del catálogo real está BLOQUEADA por el proxy
   (pendiente declarado); esto es la mitad SIN datos del 5C.
4. La curva de degeneración ρ0(r_c) del objetivo ρ_id de Sculptor
   (M_1/2 − M⋆ dentro de r_1/2), con el punto del perfil cored del
   corpus (r_c = 0.30 kpc, tabla A del Tratado Unificado v32 —
   presentación, no derivación) marcado para el cruce 5E.

Uso: python scripts/run_sparc_structural.py
Salida: results/2026-08-14_sparc_structural/report.md (+ figuras).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.disc_cronos import (  # noqa: E402
    outer_decline_ratio,
    rho_midplane,
    v_bar_sq_freeman,
    v_cronos_sq,
    x_peak_v_cronos,
)
from dynamics.dsph_data import (  # noqa: E402
    SCULPTOR,
    plummer_scale_from_Rhalf,
    stellar_mass,
)
from dynamics.jeans import sigma_los_sq_lum_avg  # noqa: E402
from dynamics.rho_id_target import (  # noqa: E402
    rho0_required,
    sculptor_rho_id_curve,
)
from dynamics.weak_field import (  # noqa: E402
    G_PC,
    epsilon_c_of_rho,
    g_eff_plummer,
    plummer_density,
    plummer_g_newton,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-14_sparc_structural")
A_UNIT = 1e-13

# Malla declarada de discos: barre la población SPARC de LSB a HSB.
SIGMA0_GRID = (50.0, 200.0, 800.0)      # M⊙/pc² (central)
RD_GRID = (1000.0, 2000.0, 4000.0)      # pc
ZETA_GRID = (0.1, 0.2)                  # h/R_d declarado
X_OUTER = 4.0                           # R/R_d del «punto exterior»


def sculptor_A_req(upsilon: float) -> float:
    """A_req del medio paso 2, recalculada (misma maquinaria)."""
    d = SCULPTOR
    a0 = plummer_scale_from_Rhalf(d["R_half_pc"])
    M = stellar_mass(d["L_V_Lsun"], upsilon)
    r = np.geomspace(0.05, 120.0 * a0, 800)
    nu = plummer_density(r, M, a0)
    kw = {"R_max": 8.0 * a0, "u_max": 120.0 * a0}
    s2_N = sigma_los_sq_lum_avg(r, nu, plummer_g_newton(r, M, a0), **kw)
    dS2 = sigma_los_sq_lum_avg(
        r, nu, g_eff_plummer(r, M, a0, A_UNIT), **kw) - s2_N
    return float((d["sigma_los_kms"] ** 2 - s2_N) / dS2 * A_UNIT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. A_req recalculada
    A_by_ups = {u: sculptor_A_req(u) for u in SCULPTOR["upsilon_scan"]}
    A2 = A_by_ups[2.0]
    for u, A in A_by_ups.items():
        print(f"1. A_req(Υ⋆={u:.0f}) = {A:.3e} (M⊙/pc³)^(-3/2)")

    # 2. Identidades de forma
    xp = x_peak_v_cronos()
    decl = outer_decline_ratio(X_OUTER)
    print(f"2. forma: pico en x = {xp:.3f}; v²(x={X_OUTER:.0f})/v²(pico) "
          f"= {decl:.3f} — independiente de A, Σ0, R_d, ζ")

    # 3. La malla de discos con la A de Sculptor (Υ⋆ = 2)
    rows = []
    for S0 in SIGMA0_GRID:
        for Rd in RD_GRID:
            for z in ZETA_GRID:
                R = np.linspace(0.05 * Rd, 8.0 * Rd, 1600)
                v2b = v_bar_sq_freeman(R, S0, Rd)
                v2c = v_cronos_sq(R, S0, Rd, z, A2)
                i_pk = int(np.argmax(v2c))
                frac_peak = float(np.sqrt(v2c[i_pk] / v2b[i_pk]))
                i_out = int(np.argmin(np.abs(R - X_OUTER * Rd)))
                v_out = float(np.sqrt(v2c[i_out]))
                vbar_out = float(np.sqrt(v2b[i_out]))
                eps_max = float(epsilon_c_of_rho(
                    rho_midplane(0.0, S0, Rd, z), A2))
                rows.append((S0, Rd / 1000.0, z, frac_peak, v_out,
                             vbar_out, eps_max))
    tbl = "\n".join(
        f"| {S0:.0f} | {Rdk:.0f} | {z:.1f} | {fp:.2f} | {vo:.2f} | "
        f"{vb:.1f} | {em:.1e} |"
        for S0, Rdk, z, fp, vo, vb, em in rows)
    worst_inner = max(r[3] for r in rows)
    out_min = min(r[4] for r in rows)
    out_max = max(r[4] for r in rows)
    shape_cost = 1.0 / np.sqrt(outer_decline_ratio(X_OUTER))
    print(f"3. malla: max v_cronos/v_bar (pico interior) = "
          f"{worst_inner:.2f}; v_cronos(x=4) ∈ [{out_min:.2f}, "
          f"{out_max:.2f}] km/s; coste de forma ×{shape_cost:.1f}")

    # 4. Curva objetivo ρ_id de Sculptor
    d = SCULPTOR
    a0 = plummer_scale_from_Rhalf(d["R_half_pc"])
    r_half_3d = 4.0 / 3.0 * a0
    M_half = 4.0 * d["sigma_los_kms"] ** 2 * a0 / G_PC
    M2 = stellar_mass(d["L_V_Lsun"], 2.0)
    r_c_grid = np.geomspace(50.0, 2000.0, 200)
    m_id_target, rho0_curve = sculptor_rho_id_curve(
        r_c_grid, M_half, M2, a0, r_half_3d)
    rho0_300 = float(rho0_required(300.0, m_id_target, r_half_3d))
    print(f"4. objetivo ρ_id: M_id(<{r_half_3d:.0f} pc) = "
          f"{m_id_target:.2e} M⊙; en r_c = 0.30 kpc exacto ⟹ ρ0 = "
          f"{rho0_300:.3f} M⊙/pc³")

    # Figuras
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
        R = np.linspace(50.0, 8.0 * 2000.0, 800)
        v2b = v_bar_sq_freeman(R, 200.0, 2000.0)
        v2c = v_cronos_sq(R, 200.0, 2000.0, 0.15, A2)
        ax1.plot(R / 1000.0, np.sqrt(v2b), lw=1.6,
                 label="v_bar (Freeman, Σ0=200, R_d=2 kpc)")
        ax1.plot(R / 1000.0, np.sqrt(v2c), lw=1.6, ls="--",
                 label="v_cronos con la A de Sculptor (Υ⋆=2)")
        ax1.plot(R / 1000.0, np.sqrt(v2b + v2c), lw=1.2, ls=":",
                 label="total")
        ax1.axvspan(3.0 * 2.0, 8.0 * 2.0, color="0.9",
                    label="donde vive la discrepancia (x ≳ 3)")
        ax1.set_xlabel("R [kpc]")
        ax1.set_ylabel("v [km/s]")
        ax1.set_title("5C estructural (∝ x·e^(−3x/2)): aportar fuera "
                      "cuesta ×5 dentro —\nbulto interior incompatible con "
                      "discos internos bariónicos (literatura; ingesta "
                      "pendiente)")
        ax1.legend(fontsize=7)
        ax2.loglog(r_c_grid / 1000.0, rho0_curve, lw=1.6)
        ax2.scatter([0.30], [rho0_300], zorder=3,
                    label=f"r_c del corpus (0.30 kpc) ⟹ "
                          f"ρ0 = {rho0_300:.3f} M⊙/pc³")
        ax2.set_xlabel("r_c [kpc]")
        ax2.set_ylabel("ρ0 exigido [M⊙/pc³]")
        ax2.set_title("El objetivo ρ_id de Sculptor: curva de "
                      "degeneración ρ0(r_c)")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "sparc_structural.png", dpi=140)
        print(f"Figura: {OUT / 'sparc_structural.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    # Informe
    (OUT / "report.md").write_text(
        "# Frente 5, paso 5C (mitad estructural) + curva objetivo "
        "ρ_id\n\n"
        "La ingesta del catálogo SPARC real está BLOQUEADA por el "
        "proxy de la sesión (astroweb.cwru.edu fuera de la lista "
        "blanca; hueco de checksum abierto en "
        "`scripts/download_data.py`). Esta es la mitad del 5C que NO "
        "necesita datos: la FORMA del término de Cronos en discos es "
        "una identidad de la Def. 11.1, y la amplitud es la que "
        "Sculptor midió (medio paso 2) — recalculada aquí con la "
        "misma maquinaria, no copiada.\n\n"
        "## 1. La amplitud compartida (problema inverso de Sculptor)\n\n"
        "| Υ⋆ | A_req [(M⊙/pc³)^(-3/2)] |\n|---|---|\n"
        + "\n".join(f"| {u:.0f} | {A:.3e} |" for u, A in A_by_ups.items())
        + "\n\n(α₀⁻¹, ρ_c) solo entran por A — la degeneración está "
        "fijada por test. El contraste 5E es: la MISMA A en discos.\n\n"
        "## 2. La forma del término en discos (independiente de A)\n\n"
        f"v_cronos²(R) = (3c²/2)·(R/R_d)·A·ρ(R)^(3/2) ∝ x·e^(−3x/2) "
        f"con x = R/R_d: pico en x = {xp:.3f} y caída a "
        f"{100.0 * decl:.1f} % del pico en x = {X_OUTER:.0f}. La "
        "discrepancia de masa de los discos vive en x ≳ 3 (curvas "
        "planas con v_bar cayendo): **ninguna amplitud A convierte "
        "este término en una curva plana exterior** — su forma decae "
        "más deprisa que la propia v_bar². Esto es una identidad del "
        "término (ε_c alimentada por la ρ bariónica local en el disco "
        "exponencial declarado), no un ajuste. Que la discrepancia de "
        "masa vive en el exterior (curvas planas con v_bar cayendo) es "
        "el resultado observacional estándar que motiva el frente — "
        "sin ingesta aquí; hipótesis declarada.\n\n"
        "## 3. La A de Sculptor sobre la malla declarada de discos\n\n"
        "Discos exponenciales (Σ0 × R_d × ζ = h/R_d declarados) con "
        "Σ0 ∈ [50, 800] M⊙/pc² — un RANGO DECLARADO, no el catálogo: "
        "los LSB reales bajan de 50 (allí el término se hace pequeño "
        "y el argumento (ii) no excluye; excluye el (i): tampoco "
        "aporta nada). A = A_req(Υ⋆=2):\n\n"
        "| Σ0 [M⊙/pc²] | R_d [kpc] | ζ | v_cronos/v_bar en el pico | "
        "v_cronos(x=4) [km/s] | v_bar(x=4) [km/s] | ε_c máx |\n"
        "|---|---|---|---|---|---|---|\n"
        f"{tbl}\n\n"
        f"**Lectura.** (i) LA LEY DE FORMA ES EL VEREDICTO: como "
        f"v_cronos²(x={X_OUTER:.0f})/v_cronos²(pico) = "
        f"{outer_decline_ratio(X_OUTER):.3f}, aportar V km/s donde la "
        f"discrepancia vive cuesta {1.0 / np.sqrt(outer_decline_ratio(X_OUTER)):.1f}·V km/s "
        "en x = 2/3. Que las regiones internas de los discos masivos "
        "son bariónicas (sin hueco para bultos de esa escala) es un "
        "resultado ESTÁNDAR de la literatura de curvas de rotación — "
        "SIN INGESTA en este repositorio: aquí es hipótesis declarada, "
        "y su confrontación cuantitativa es lo que la ingesta SPARC "
        "pendiente hará por galaxia. La ley de coste ×5, en cambio, es "
        "identidad nuestra y vale para CUALQUIER amplitud A. (ii) Con "
        "la A concreta de Sculptor el término no es pequeño en "
        f"discos: v_cronos(x={X_OUTER:.0f}) va de {out_min:.1f} km/s "
        f"(celda menos densa de la malla) a {out_max:.1f} km/s (la más "
        f"densa), y en el pico interior llega a v_cronos/v_bar = "
        f"{worst_inner:.2f} en las celdas densas (en las menos densas "
        "baja de 1): la A compartida no produce curvas planas en "
        "ninguna celda — produce bultos interiores donde el disco es "
        "denso y nada donde no lo es. La confrontación por galaxia "
        "con SPARC real (pendiente de ingesta) convertirá esto en "
        "cotas superiores sobre A por sistema. (iii) ε_c ≤ "
        f"{max(r[6] for r in rows):.1e} en toda la malla (régimen "
        "débil de la Def. 11.1 intacto: lo que falla no es la validez "
        "de la expansión, es la fenomenología).\n\n"
        "**Veredicto estructural del 5E (parcial, sin datos por "
        "galaxia)**: el término débil de Cronos no puede ser el "
        "mecanismo galáctico común — en dSphs solo alcanza σ_obs "
        "como potencial dominante (medio paso 2), y en discos la ley "
        "de forma (aportar fuera cuesta ×5 dentro) le impide producir "
        "curvas planas con NINGUNA amplitud: en el extremo denso, la "
        "amplitud que Sculptor exige produce bultos interiores "
        "dominantes (×6 sobre v_bar) incompatibles con el carácter "
        "bariónico interior que la literatura reporta (hipótesis "
        "declarada, confrontación pendiente de ingesta); en el "
        "extremo difuso, el término se hace despreciable y no explica "
        "nada. Coherente con la arquitectura del tratado, que asigna "
        "la fenomenología galáctica a ρ_id; queda el cruce "
        "cuantitativo con el catálogo real.\n\n"
        "## 4. El objetivo ρ_id de Sculptor, como curva\n\n"
        f"M_1/2 = {M_half:.2e} M⊙ dentro de r_1/2 = {r_half_3d:.0f} "
        f"pc; parte estelar (Υ⋆=2, Plummer): la resta deja "
        f"**M_id(<r_1/2) = {m_id_target:.2e} M⊙**. Con el perfil "
        "cored del corpus ρ_id = ρ0/(1+(r/r_c)²), Sculptor no fija "
        "(ρ0, r_c): fija la CURVA ρ0(r_c) (figura). En el r_c = 0.30 "
        "kpc que la tabla A del corpus usa para galaxias (presentación "
        f"v32, no derivación), evaluado exacto: ρ0 = {rho0_300:.3f} "
        "M⊙/pc³. "
        "Cualquier derivación futura de ρ_id (o la relación "
        "rcore(M, z) de B.5 calibrada en SPARC) debe atravesar esta "
        "curva — el cruce 5E en el sector que el tratado sí señala "
        "como responsable.\n\n"
        "## Estatuto\n\n"
        "condicional y parcial: (a) la parte de FORMA (§2) es "
        "identidad matemática del término declarado (ε_c alimentada "
        "por ρ bariónica local; discos exponenciales); (b) la malla "
        "§3 es DECLARADA, no el catálogo real — la ingesta SPARC "
        "(bloqueada por el proxy) convertirá (ii) en cotas por "
        "galaxia; (c) la curva ρ_id (§4) hereda el estimador de Wolf "
        "y sus aproximaciones declaradas; (d) nada de esto falsa el "
        "modelo completo: acota QUÉ sector puede llevar la carga "
        "galáctica (ρ_id, no el término débil de Cronos) — la "
        "derivación de ρ_id a estas escalas sigue siendo el hueco "
        "declarado del frente.\n",
        encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
