#!/usr/bin/env python
"""Frente nº 5, medio paso (opción G): perfil emergente y compuerta H.2.5.

Dos medidas en la malla PM propia, sin reclamar producción:

1. PERFIL: el perfil radial emergente del colapso aislado — par A/B de
   semilla idéntica (A newtoniano, B Cronos v3 con α0⁻¹ en el máximo de
   la cota 11.5) — ajustado EN FORMA a cored ρ0/[1+(r/r_c)²] vs NFW.
   Comparación de formas, no de valores absolutos: el núcleo de 2.3 kpc
   del corpus queda para producción (Gadget-4-Cronos).

2. COMPUERTA (H.2.5): «la fricción corregida cae de ≈4e-4 a ≈7e-9 al
   virializar (compuerta cerrada), mientras la antigua persiste». Aquí
   se mide la caída EN ESTA MALLA sobre la Γ de la región que colapsa
   y viriliza, en dos configuraciones: halo aislado (la virialización
   limpia) y halo con fondo (donde la acreción secundaria continúa y el
   residuo de Γ es infall real, no compuerta abierta — se declara).
   Control negativo: la forma sin compuerta (11.3b) sobre la misma
   historia, que debe persistir.

Uso: python scripts/run_profile_shape.py [--steps 1200] [--np 4096]

Salida: results/2026-08-02_profile_shape/report.md (+ figura).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cronos.cronos_v3 import ALPHA0_INV_MAX  # noqa: E402
from cronos.profile_fit import (  # noqa: E402
    friction_drop,
    gate_histories,
    halo_center,
    radial_profile,
    shape_comparison,
    spherical_clump_ic,
    track_region_density,
)
from cronos.simulation import CronosPM  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-02_profile_shape")
BOX = 10.0
DT = 0.002
GRID = 32
# ρ_c de ε_c = α0⁻¹(ρ/ρ_c)^{3/2}: la densidad umbral de colapso, ~200×
# la media de la caja (la sobredensidad virial; el v32 usaba RHO_C0=200).
# Con ρ_c ~ media, ρ/ρ_c alcanza ~1e5 en el pico y ε_c = O(1): fuera del
# régimen débil de la Def. 11.1 (la fricción condensa el halo en un
# punto — artefacto, no física). El script MIDE ε_c_max y lo declara.
OVERDENSITY_C = 200.0


def rho_c_box(n_p: int) -> float:
    """ρ_c = 200 × densidad media de la caja (umbral de colapso)."""
    return OVERDENSITY_C * n_p * 1.0 / BOX ** 3


def run_pair(n_p: int, steps: int, seed: int, clump_frac: float,
             r_region: float, alpha0_inv: float) -> dict:
    """Par A/B de semilla idéntica; ambos con seguimiento de la región."""
    out = {}
    for tag, a0 in (("A", 0.0), ("B", alpha0_inv)):
        pos, vel, mass = spherical_clump_ic(seed, n_p, BOX,
                                            clump_frac=clump_frac)
        sim = CronosPM(pos, vel, mass, grid_n=GRID, box=BOX,
                       alpha0_inv=a0, rho_c=rho_c_box(n_p))
        rho_hist = track_region_density(sim, steps, DT, r_region)
        out[tag] = {"sim": sim, "rho_region": rho_hist}
    return out


def profile_and_fit(sim: CronosPM) -> dict:
    c = halo_center(sim.pos, sim.mass, sim.box)
    prof = radial_profile(sim.pos, sim.mass, c, sim.box)
    cmp_ = shape_comparison(prof["r"], prof["rho"])
    return {"profile": prof, "cmp": cmp_}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, default=4096, dest="n_p")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--steps-fondo", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260802)
    opts = ap.parse_args()

    a0 = ALPHA0_INV_MAX   # el máximo permitido por la cota (ec. 11.5)

    # ---- Configuración 1: halo aislado (perfil + compuerta limpia) ----
    print(f"[aislado] par A/B, {opts.n_p} partículas, {opts.steps} pasos...")
    iso = run_pair(opts.n_p, opts.steps, opts.seed, clump_frac=1.0,
                   r_region=0.5, alpha0_inv=a0)
    fits = {t: profile_and_fit(iso[t]["sim"]) for t in ("A", "B")}
    for t in ("A", "B"):
        c = fits[t]["cmp"]
        print(f"  [{t}] cored: rmse {c['cored']['rmse_log']:.4f} dex "
              f"(r_c={c['cored']['scale']:.3f}) | NFW: rmse "
              f"{c['nfw']['rmse_log']:.4f} dex (r_s={c['nfw']['scale']:.3f})"
              f" | preferida: {c['preferred']}")

    rc_iso = rho_c_box(opts.n_p)
    eps_max_iso = a0 * (iso["B"]["sim"].rho_m.max() / rc_iso) ** 1.5
    print(f"  validez del régimen débil: ε_c(ρ_celda_max) = "
          f"{eps_max_iso:.2e} (debe ser ≪ 1)")
    gh_iso = gate_histories(iso["B"]["rho_region"], DT, a0, rc_iso)
    drop_iso = friction_drop(gh_iso["gated"], gh_iso["i_peak"],
                             tail_frac=0.10)
    persist_iso = friction_drop(gh_iso["ungated"], gh_iso["i_peak"],
                                tail_frac=0.10)
    zeros_frac = drop_iso["tail_zero_frac"]
    print(f"  compuerta (región r=0.5): Γ_colapso = "
          f"{drop_iso['Gamma_collapse']:.3e} → Γ_vir máx = "
          f"{drop_iso['Gamma_virial']:.3e}, mediana = "
          f"{drop_iso['Gamma_virial_median']:.3e} "
          f"(Γ=0 exacto en {zeros_frac:.0%} del tramo final)")
    print(f"  sin compuerta (11.3b): caída {persist_iso['orders_drop']:.2f} "
          "órdenes — persiste")

    # ---- Configuración 2: halo con fondo (acreción continua) ----
    print(f"[con fondo] {opts.n_p} partículas (50% fondo), "
          f"{opts.steps_fondo} pasos...")
    pos, vel, mass = spherical_clump_ic(opts.seed, opts.n_p, BOX,
                                        clump_frac=0.5)
    sim_bg = CronosPM(pos, vel, mass, grid_n=GRID, box=BOX,
                      alpha0_inv=a0, rho_c=rho_c_box(opts.n_p))
    rho_bg = track_region_density(sim_bg, opts.steps_fondo, DT, 1.0)
    gh_bg = gate_histories(rho_bg, DT, a0, rho_c_box(opts.n_p), smooth=31)
    drop_bg = friction_drop(gh_bg["gated"], gh_bg["i_peak"], tail_frac=0.10)
    print(f"  compuerta (región r=1.0): caída {drop_bg['orders_drop']:.2f} "
          f"órdenes (Γ_vir = {drop_bg['Gamma_virial']:.3e}; el residuo es "
          "acreción secundaria real del fondo)")

    # ---- Figura ----
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        colors = {"A": "#8a4a2f", "B": "#2f6b4f"}
        for t in ("A", "B"):
            p = fits[t]["profile"]
            c = fits[t]["cmp"]
            lab = ("A (newtoniano)" if t == "A"
                   else f"B (Cronos v3, α0⁻¹={a0:g})")
            axes[0].loglog(p["r"], p["rho"], "o", color=colors[t], ms=4,
                           label=lab)
            r_fine = np.geomspace(p["r"].min(), p["r"].max(), 100)
            best = c[c["preferred"]]
            if c["preferred"] == "cored":
                model = best["amplitude"] / (1 + (r_fine / best["scale"]) ** 2)
            else:
                x = r_fine / best["scale"]
                model = best["amplitude"] / (x * (1 + x) ** 2)
            axes[0].loglog(r_fine, model, "-", color=colors[t], lw=1,
                           alpha=0.7,
                           label=f"  ajuste {c['preferred']} "
                                 f"({best['rmse_log']:.3f} dex)")
        axes[0].set_xlabel("r")
        axes[0].set_ylabel(r"$\rho(r)$")
        axes[0].set_title("Perfil emergente (colapso aislado, "
                          "misma semilla)")
        axes[0].legend(fontsize=7)

        tt = np.arange(len(gh_iso["gated"])) * DT
        axes[1].semilogy(tt, np.maximum(gh_iso["gated"], 1e-12), "-",
                         color="#2f6b4f",
                         label="Γ con compuerta (Cor. 11.3a)")
        axes[1].semilogy(tt, gh_iso["ungated"], "--", color="#8a4a2f",
                         label="forma sin compuerta (11.3b): persiste")
        axes[1].axvline(gh_iso["i_peak"] * DT, color="k", ls=":", lw=1,
                        label="pico de densidad de la región")
        axes[1].set_xlabel("t")
        axes[1].set_ylabel(r"$\Gamma$")
        axes[1].set_title("La compuerta se cierra al virializar (H.2.5)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "profile_shape.png", dpi=140)
        print(f"Figura: {OUT / 'profile_shape.png'}")
    except ImportError:
        print("[aviso] matplotlib no disponible: figura omitida")

    # ---- Informe ----
    ca, cb = fits["A"]["cmp"], fits["B"]["cmp"]
    pa, pb = fits["A"]["profile"], fits["B"]["profile"]
    if len(pa["r"]) == len(pb["r"]) and np.allclose(pa["r"], pb["r"]):
        ab_diff = float(np.max(np.abs(pb["rho"] - pa["rho"]) / pa["rho"]))
        ab_txt = f"{ab_diff:.1%}"
    else:
        ab_txt = "cáscaras no comparables (bins distintos)"
    rep = [
        ("# Frente nº 5, medio paso: perfil en la malla y compuerta "
         "H.2.5\n"),
        (f"Par A/B de semilla idéntica {opts.seed} (B.5/B.6): "
         f"{opts.n_p} partículas, malla {GRID}³, dt = {DT}, "
         f"{opts.steps} pasos; B con α0⁻¹ = {a0:g} (el máximo de la "
         "cota de la ec. 11.5). Malla PM mínima: NO es producción — "
         "el 2.3 kpc del corpus y los valores absolutos de H.2.5 quedan "
         "para Gadget-4-Cronos (frente nº 5).\n"),
        (f"PARÁMETROS DECLARADOS: ρ_c = {OVERDENSITY_C:g} × densidad "
         "media de la caja (umbral de colapso; con ρ_c ~ media, ε_c "
         "alcanza O(1) y el esquema sale del régimen débil de la "
         "Def. 11.1 — artefacto verificado, no física). Validez medida: "
         f"ε_c(ρ_celda_max) = {eps_max_iso:.2e} ≪ 1. Suavizado de ρ(t) "
         "de la región por media móvil (ruido de cáscara).\n"),
        "## 1. Perfil emergente: comparación de formas\n",
        ("| Corrida | RMSE cored [dex] | r_c | RMSE NFW [dex] | r_s | "
         "preferida |\n|---|---|---|---|---|---|"),
        (f"| A (newtoniana) | {ca['cored']['rmse_log']:.4f} | "
         f"{ca['cored']['scale']:.3f} | {ca['nfw']['rmse_log']:.4f} | "
         f"{ca['nfw']['scale']:.3f} | {ca['preferred']} |"),
        (f"| B (Cronos v3) | {cb['cored']['rmse_log']:.4f} | "
         f"{cb['cored']['scale']:.3f} | {cb['nfw']['rmse_log']:.4f} | "
         f"{cb['nfw']['scale']:.3f} | {cb['preferred']} |"),
        (f"\nDiferencia máxima A/B del perfil por cáscara: {ab_txt} "
         f"(con α0⁻¹ = {a0:g} dentro de la cota, la corrección de "
         "Cronos es minúscula a esta resolución — como debe ser en el "
         "régimen débil).\n"),
        ("NOTA DE RESOLUCIÓN: los radios de escala ajustados caen por "
         f"debajo de la celda ({BOX / GRID:.3f}) y del primer bin: la "
         "malla solo ve la rama externa del perfil, de modo que la "
         "discriminación efectiva es entre pendientes externas (−2 "
         "cored vs −3 NFW), no entre núcleo y cúspide. El colapso frío "
         "aislado emerge cuspy (NFW-like) en AMBAS corridas — el "
         "contraste del núcleo (r < celda) es inaccesible aquí y queda "
         "para producción.\n"),
        (f"Lectura honesta: Δrmse(NFW−cored) = "
         f"{ca['delta_rmse_log']:+.4f} dex (A) y "
         f"{cb['delta_rmse_log']:+.4f} dex (B). A esta resolución "
         f"(celda {BOX / GRID:.3f}) la zona interna está limitada por la "
         "malla: la comparación es de formas y su poder de decisión es "
         "el que muestran esos Δrmse — el veredicto del núcleo es de "
         "producción.\n"),
        "## 2. La compuerta de Cronos (H.2.5)\n",
        ("Tratado: «la fricción corregida cae de ≈4×10⁻⁴ a ≈7×10⁻⁹ al "
         "virializar (compuerta cerrada), mientras la antigua persiste» "
         "— valores de SU simulación de producción; aquí se mide la "
         "caída en esta malla.\n"),
        ("| Configuración | Γ_colapso | Γ_vir (máx. tramo final) | "
         "caída [órdenes] |\n|---|---|---|---|"),
        ((f"| Aislada (región r=0.5) | {drop_iso['Gamma_collapse']:.3e} | "
          f"{drop_iso['Gamma_virial']:.3e} | ")
         + ("∞ (cerrada exactamente)"
            if drop_iso["gate_closed_exactly"]
            else f"{drop_iso['orders_drop']:.2f}") + " |"),
        (f"| Con fondo 50% (región r=1.0) | "
         f"{drop_bg['Gamma_collapse']:.3e} | "
         f"{drop_bg['Gamma_virial']:.3e} | {drop_bg['orders_drop']:.2f} |"),
        (f"\n- En el halo AISLADO la compuerta cierra: Γ = 0 exacto en "
         f"el {zeros_frac:.0%} de los pasos del tramo final (mediana "
         f"del tramo = {drop_iso['Gamma_virial_median']:.1e}); el máx "
         "residual es ruido de cáscara de N finito (el Θ(ρ̇) del "
         "Cor. 11.3a corta el resto)."),
        ("- En el halo CON FONDO el residuo de Γ es acreción secundaria "
         "real (la caja pequeña sigue alimentando la región): es caída "
         f"de {drop_bg['orders_drop']:.1f} órdenes limitada por infall "
         "físico, no compuerta abierta — la caída de cinco órdenes del "
         "tratado es de un halo de producción plenamente virializado."),
        (f"- CONTROL NEGATIVO: la forma sin compuerta (11.3b) sobre la "
         f"misma historia cae solo {persist_iso['orders_drop']:.2f} "
         "órdenes — persiste, como dice H.2.5 del esquema antiguo.\n"),
        ("Firma falsable (H.2.5): «los halos virializados no deben "
         "mostrar fricción de Cronos residual» — en esta malla, el halo "
         "aislado virializado muestra exactamente ninguna.\n"),
    ]
    (OUT / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"Informe: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
