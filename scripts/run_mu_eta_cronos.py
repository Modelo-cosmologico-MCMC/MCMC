#!/usr/bin/env python
"""Computa µ − 1, η, Σ y la razón fσ8 del canal Cronos BAJO la
preinscripción congelada y publica el artefacto versionado. Sin datos.

Falla cerrado sin preinscripción. Recomputa la tabla de la nota de
teoría y publica las diferencias (ninguna cifra de la nota se cita
hasta aquí). Clasifica E1 con la regla congelada y verifica las
identidades E2.

Uso: python scripts/run_mu_eta_cronos.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.mu_eta_cronos import (  # noqa: E402
    ATLAS_STATUS,
    Sigma_cronos,
    eta_cronos,
    fsigma8_ratio_cronos,
    in_validity_window,
    mu_minus_one_cronos,
    mu_table,
)

OUTDIR = (Path(__file__).resolve().parent.parent / "results"
          / "2026-09-13_mu_eta_cronos")
PREREG = OUTDIR / "preregistration.json"


def main() -> int:
    if not PREREG.exists():
        raise SystemExit("PREREGISTRATION_MISSING: congela primero con "
                         "scripts/run_mu_eta_prereg.py")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    par = prereg["parameters_from_treatise_only"]
    alpha = float(par["alpha0_inv"]["value"])
    tr = par["background_theta_ref"]
    theta = (tr["H0"], tr["Omega_m"], tr["epsilon"], tr["z_trans"])
    k_grid = np.array(prereg["grids"]["k_hMpc"], float)
    z_grid = np.array(prereg["grids"]["z"], float)
    e1 = prereg["expectations"]["E1_boring_consistency_control"]
    e1_thr = float(e1["rule"].split("≤")[1].split("sobre")[0])
    k_max_e1 = float(prereg["closures_declared"]["3_validity_window"]
                     ["k_max_hMpc"])

    # ------------------------------------------------------------------
    # Tablas µ − 1 por cierre
    # ------------------------------------------------------------------
    tables = {mode: mu_table(theta, alpha, k_grid, z_grid, mode)
              for mode in ("comoving", "physical")}
    window = in_validity_window(k_grid)

    # ------------------------------------------------------------------
    # E2: identidades (se recalculan aquí; la suite las pinnea)
    # ------------------------------------------------------------------
    a_chk = 1.0 / (1.0 + z_grid)
    ident = {}
    for mode in ("comoving", "physical"):
        m1 = np.array([mu_minus_one_cronos(k_grid, a, theta, alpha, mode)
                       for a in a_chk])
        S = np.array([Sigma_cronos(k_grid, a, theta, alpha, mode)
                      for a in a_chk])
        E = np.array([eta_cronos(k_grid, a, theta, alpha, mode)
                      for a in a_chk])
        ident[mode] = {
            "max_abs_Sigma_identity": float(np.max(np.abs(
                (S - 1.0) - 0.5 * m1))),
            "max_abs_eta_second_order": float(np.max(np.abs(
                (E - 1.0) + m1) / np.maximum(m1 ** 2, 1e-300))),
            "k2_scaling_rel_err": float(np.max(np.abs(
                m1[:, 3] / m1[:, 2] / (k_grid[3] / k_grid[2]) ** 2 - 1.0))),
        }
    ratio_phys_com = tables["physical"] / tables["comoving"]
    expected = (1.0 + z_grid)[:, None] ** 4.5
    ident["physical_over_comoving_rel_err"] = float(np.max(np.abs(
        ratio_phys_com / expected - 1.0)))
    m_half = mu_table(theta, alpha / 2.0, k_grid, z_grid, "comoving")
    ident["linearity_alpha_rel_err"] = float(np.max(np.abs(
        m_half / (0.5 * tables["comoving"]) - 1.0)))
    ident["gr_limit_exact"] = bool(np.all(
        mu_table(theta, 0.0, k_grid, z_grid, "comoving") == 0.0))
    e2_pass = (all(ident[m]["max_abs_Sigma_identity"] <= 1e-15
                   for m in ("comoving", "physical"))
               and all(ident[m]["max_abs_eta_second_order"] <= 1.01
                       for m in ("comoving", "physical"))
               and all(ident[m]["k2_scaling_rel_err"] <= 1e-12
                       for m in ("comoving", "physical"))
               and ident["physical_over_comoving_rel_err"] <= 1e-10
               and ident["linearity_alpha_rel_err"] <= 1e-12
               and ident["gr_limit_exact"])

    # ------------------------------------------------------------------
    # E1: razón fσ8 con cero parámetros nuevos (ambos cierres; la regla
    # solo aplica al comoving)
    # ------------------------------------------------------------------
    z_e1 = z_grid[z_grid <= float(e1["rule"].split("z ≤")[1]
                                  .split(",")[0])]
    k_e1 = k_grid[k_grid <= k_max_e1]
    ratios = {}
    for mode in ("comoving", "physical"):
        R = np.array([fsigma8_ratio_cronos(z_e1, theta, float(k), alpha,
                                           mode) for k in k_e1])
        ratios[mode] = R                          # (len(k_e1), len(z_e1))
    e1_max = float(np.max(np.abs(ratios["comoving"] - 1.0)))
    e1_pass = bool(e1_max <= e1_thr)
    phys_max = float(np.max(np.abs(ratios["physical"] - 1.0)))

    # ------------------------------------------------------------------
    # Verificación de la tabla a mano de la nota
    # ------------------------------------------------------------------
    note = prereg["note_estimates_to_verify"]
    iz = {z: int(np.where(z_grid == z)[0][0]) for z in (0.0, 1.0, 3.0)}
    verification = []
    for row in note["rows"]:
        ik = int(np.where(np.isclose(k_grid, row["k_hMpc"]))[0][0])
        comp = {
            "k_hMpc": row["k_hMpc"],
            "z0_comoving": {"note": row["z0_comoving"],
                            "computed": float(tables["comoving"][iz[0.0], ik])},
            "z1_physical": {"note": row["z1_physical"],
                            "computed": float(tables["physical"][iz[1.0], ik])},
            "z3_physical": {"note": row["z3_physical"],
                            "computed": float(tables["physical"][iz[3.0], ik]),
                            "in_window": bool(window[ik])},
        }
        for key in ("z0_comoving", "z1_physical", "z3_physical"):
            n_, c_ = comp[key]["note"], comp[key]["computed"]
            comp[key]["ratio_computed_over_note"] = (
                None if n_ is None else float(c_ / n_))
        verification.append(comp)

    # ------------------------------------------------------------------
    # Artefactos
    # ------------------------------------------------------------------
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {
        "preregistration_sha256": prereg_sha,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "alpha0_inv_used": alpha,
        "alpha0_inv_role": "cota 11.5 saturada — límite superior, no medida",
        "theta_ref": tr,
        "k_hMpc": k_grid.tolist(), "z": z_grid.tolist(),
        "in_validity_window": window.tolist(),
        "mu_minus_one": {m: tables[m].tolist() for m in tables},
        "identities_E2": ident, "E2_pass": e2_pass,
        "fsigma8_ratio_E1": {
            "k_hMpc": k_e1.tolist(), "z": z_e1.tolist(),
            "comoving_R_minus_one": (ratios["comoving"] - 1.0).tolist(),
            "physical_R_minus_one": (ratios["physical"] - 1.0).tolist(),
            "comoving_max_abs": e1_max, "threshold": e1_thr,
            "E1_pass_comoving": e1_pass,
            "physical_max_abs_published_not_gated": phys_max},
        "note_table_verification": verification,
        "atlas_channel": {"status": ATLAS_STATUS,
                          "contribution_computed": False},
    }
    (OUTDIR / "mu_eta_cronos.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # CSV plano de la tabla µ − 1
    lines = ["closure,z,k_hMpc,mu_minus_one,in_window"]
    for mode in tables:
        for i, z in enumerate(z_grid):
            for j, k in enumerate(k_grid):
                lines.append(f"{mode},{z},{k},{tables[mode][i, j]:.6e},"
                             f"{int(window[j])}")
    (OUTDIR / "mu_minus_one.csv").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, mode in zip(axes, ("comoving", "physical")):
            for i, z in enumerate(z_grid):
                ax.loglog(k_grid, tables[mode][i], "o-", ms=4,
                          label=f"z = {z:g}")
            ax.axvline(k_max_e1, color="k", ls="--", lw=1,
                       label="ventana (cierre 3)")
            ax.set_title(f"cierre '{mode}' (α₀⁻¹ = {alpha:g}, cota)")
            ax.set_xlabel("k [h/Mpc]")
            ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("µ − 1")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTDIR / "mu_minus_one.png", dpi=140)
    except Exception as exc:            # noqa: BLE001 — solo la figura
        print(f"[aviso] figura no generada: {exc}")

    def fmt(x):
        return "—" if x is None else f"{x:.2e}"

    md = [
        "# µ(k,a), η, Σ desde la ontología — canal Cronos "
        f"(E1: **{'A' if e1_pass else 'B'}**, E2: "
        f"**{'PASS' if e2_pass else 'FAIL'}**)\n",
        f"Preinscripción: `{prereg_sha[:12]}` (congelada ANTES de "
        f"computar); ejecución: commit `{sha[:9]}`. Sin datos: es "
        "predicción.",
        "",
        f"- α₀⁻¹ = {alpha:g} — la COTA (11.5) saturada: límite superior, "
        "no medida. Fondo de referencia: H0 = "
        f"{tr['H0']:.2f}, Ω_m = {tr['Omega_m']:.4f}, ε = "
        f"{tr['epsilon']:+.4f}, z_trans = {tr['z_trans']:.2f}.",
        f"- **E1 (cierre comoving)**: max |R_µ − 1| = {e1_max:.2e} "
        f"(regla ≤ {e1_thr:g}) → {'desenlace A: el aburrido preinscrito' if e1_pass else 'desenlace B (a discriminar)'}. "
        f"Cierre physical (publicado, sin puerta): max |R_µ − 1| = "
        f"{phys_max:.2e}.",
        f"- **E2 identidades**: Σ−1 = (µ−1)/2 a "
        f"{ident['comoving']['max_abs_Sigma_identity']:.1e}; η−1 = "
        "−(µ−1) a segundo orden; k² y linealidad en α₀⁻¹ a "
        f"{ident['linearity_alpha_rel_err']:.1e}; physical/comoving = "
        f"(1+z)^(9/2) a {ident['physical_over_comoving_rel_err']:.1e}; "
        f"GR exacto: {ident['gr_limit_exact']}.",
        "",
        "## Tabla µ − 1 (α₀⁻¹ en su cota; k > ventana marcado ✗)",
        "",
        "| k [h/Mpc] | ventana | " + " | ".join(
            f"z={z:g} com." for z in z_grid) + " | " + " | ".join(
            f"z={z:g} phys." for z in z_grid) + " |",
        "|---|---|" + "---|" * (2 * len(z_grid)),
    ]
    for j, k in enumerate(k_grid):
        md.append(f"| {k:g} | {'✓' if window[j] else '✗'} | " + " | ".join(
            f"{tables['comoving'][i, j]:.2e}" for i in range(len(z_grid)))
            + " | " + " | ".join(
            f"{tables['physical'][i, j]:.2e}" for i in range(len(z_grid)))
            + " |")
    md += [
        "",
        "## Verificación de las estimaciones a mano de la nota (§B.3)",
        "",
        "| k | z=0 com. nota → calc. (×) | z=1 phys. nota → calc. (×) | "
        "z=3 phys. nota → calc. (×) |",
        "|---|---|---|---|",
    ]
    for c in verification:
        cells = []
        for key in ("z0_comoving", "z1_physical", "z3_physical"):
            n_, v_ = c[key]["note"], c[key]["computed"]
            r_ = c[key]["ratio_computed_over_note"]
            cells.append(f"{fmt(n_)} → {v_:.2e}"
                         + (f" (×{r_:.2f})" if r_ is not None else "")
                         + ("" if c["z3_physical"]["in_window"]
                            or key != "z3_physical" else " ✗"))
        md.append(f"| {c['k_hMpc']:g} | " + " | ".join(cells) + " |")
    md += [
        "",
        "Lectura: ×≈1 confirma la estimación; ×≠1 la corrige — las "
        "cifras citables son las calculadas, no las de la nota.",
        "",
        f"**Atlas**: {ATLAS_STATUS}. Contribución no computada.",
        "",
        "**Lo que este artefacto NO afirma**: que µ ≠ 1 esté detectado; "
        "que la amplitud sea la de la cota; que Atlas tenga "
        "coeficientes; que k² valga fuera de la ventana. Ningún dato "
        "de lensing ni de RSD entra aquí (prohibición preinscrita).",
    ]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n",
                                      encoding="utf-8")
    print(f"E1 comoving: max|R−1| = {e1_max:.2e} → "
          f"{'A' if e1_pass else 'B'}; physical (sin puerta): "
          f"{phys_max:.2e}; E2: {'PASS' if e2_pass else 'FAIL'}")
    print(f"Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
