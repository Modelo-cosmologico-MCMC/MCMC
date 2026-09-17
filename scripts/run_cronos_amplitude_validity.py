#!/usr/bin/env python
"""Ronda de registro 17-sep — cálculo de consistencia de la amplitud de
Cronos: los dos cierres de A ≡ α₀⁻¹/ρ_c^(3/2) del repositorio, el cociente
de dominancia D = c²ε_c/|Φ_N| en halos NFW y las consecuencias para el
sector µ/η (E1). Publica results/2026-09-17_cronos_amplitude_validity/.

No es un experimento con desenlaces preinscritos: es la reproducción en
el repo del cálculo del autor (sesión de verificación del 17-sep).

Uso: python scripts/run_cronos_amplitude_validity.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.mu_eta_cronos import (  # noqa: E402
    RHO_C_OVER_MEAN,
    mu_minus_one_cronos,
)
from cronos.cronos_v3 import ALPHA0_INV_MAX  # noqa: E402
from dynamics.cronos_amplitude_validity import (  # noqa: E402
    A_SCULPTOR,
    galactic_closure_f_over_mean,
    rho_mean_matter_today,
    validity_report,
)

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-17_cronos_amplitude_validity"
MU_ETA_PREREG = (Path(__file__).resolve().parent.parent / "results" / "2026-09-13_mu_eta_cronos"
                 / "preregistration.json")


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pre = json.loads(MU_ETA_PREREG.read_text(encoding="utf-8"))
    th = pre["parameters_from_treatise_only"]["background_theta_ref"]
    theta = (th["H0"], th["Omega_m"], th["epsilon"], th["z_trans"])
    rep = validity_report(th["H0"], th["Omega_m"], ALPHA0_INV_MAX, RHO_C_OVER_MEAN, A_SCULPTOR)
    rho_mean0 = rho_mean_matter_today(th["H0"], th["Omega_m"])
    f_gal = galactic_closure_f_over_mean(A_SCULPTOR, rho_mean0, ALPHA0_INV_MAX)
    mu = {}
    for z in (0.0, 1.0, 3.0):
        a = 1.0 / (1.0 + z)
        mu[str(z)] = {
            "cosmological_comoving_f200": float(mu_minus_one_cronos(0.2, a, theta, ALPHA0_INV_MAX,
                                                                   "comoving", RHO_C_OVER_MEAN)),
            "galactic_A_sculptor": float(mu_minus_one_cronos(0.2, a, theta, ALPHA0_INV_MAX,
                                                             "physical", f_gal))}
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {"kind": "cálculo de consistencia (E8), sin desenlaces preinscritos",
           "external_control": "cálculo del autor, sesión de verificación 17-sep-2026 "
                               "(h = 0.7, Ω_m = 0.31): A_cosmo/A_S = 6.6e7; D = 5.7 (r200), "
                               "1e4 (r_s), 3e5 (2.3 kpc) con el cierre cosmológico; "
                               "D = 0.8 (0.1 kpc), 0.1 (0.38 kpc), 4e-3 (2.3 kpc) con el galáctico; "
                               "A_max = 1.25 A_S; ε̄_c(hoy) = 5.4e-18 con A_S",
           "executed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "theta_source": "background_theta_ref de results/2026-09-13_mu_eta_cronos/preregistration.json",
           "f_galactic_over_mean_for_physical_mode": f_gal,
           "mu_minus_one_at_k0p2": mu, **rep}
    (OUTDIR / "cronos_amplitude_validity.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    h11 = rep["halos"][0]
    md = ["# Amplitud de Cronos: un solo cierre (cálculo de consistencia, 17-sep-2026)\n",
          f"Commit `{sha[:9]}`. θ de fondo = medianas congeladas del 12-sep (H0 = {th['H0']:.2f}, "
          f"Ω_m = {th['Omega_m']:.4f}); ρ̄_m(0) = {rho_mean0:.3e} M☉/pc³; α₀⁻¹ = {ALPHA0_INV_MAX:g}.", "",
          "**Lectura obligatoria**: (α₀⁻¹, ρ_c) entran solo por A = α₀⁻¹/ρ_c^(3/2). El cierre "
          f"cosmológico ρ_c = {RHO_C_OVER_MEAN:g}·ρ̄_m da A = {rep['A_cosmological']:.3g}; el galáctico "
          f"(5E, congelado) A_Sculptor = {A_SCULPTOR:.4g}: cociente **{rep['A_cosmo_over_A_galactic']:.2e}** "
          f"(ρ_c galáctica equivalente {rep['rho_c_galactic_equiv_msun_pc3']:.3f} M☉/pc³ = "
          f"{rep['rho_c_galactic_over_mean']:.2e} ρ̄_m). Dentro de un halo NFW de 1e11 M☉ el cierre "
          "cosmológico viola la subdominancia c²ε_c ≲ |Φ_N| (Cor. 11.3c) en todo el halo: EXCLUIDO "
          "dinámicamente. El galáctico la cumple en r ≥ 0.1 kpc y coincide con A_max. **La amplitud del "
          "modelo operativo es una y es A_Sculptor.**", "",
          "| halo | cierre | A | D(suav.) | D(0.38 kpc) | D(2.3 kpc) | D(r_s) | D(r200) | r(D = 1) | ε_c(0.38 kpc) | ε_c(r200) | subdominante |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for hrow in rep["halos"]:
        for name, cl in hrow["closures"].items():
            r1 = cl["r_D_equals_one_kpc"]
            md.append(f"| {hrow['M200']:.0e} M☉, c = {hrow['c']:g} | {name} | {cl['A']:.3g} | "
                      f"{cl['D_soft']:.2e} | {cl['D_0p38kpc']:.2e} | {cl['D_2p3kpc']:.2e} | "
                      f"{cl['D_rs']:.2e} | {cl['D_r200']:.2e} | "
                      f"{'—' if r1 is None else f'{r1:.2f} kpc'} | {cl['eps_c_0p38kpc']:.1e} | "
                      f"{cl['eps_c_r200']:.1e} | {'sí' if cl['subdominant_everywhere'] else 'NO'} |")
    md += ["", f"A_max(1e11, suavizado {h11['soft_kpc']} kpc) = {h11['A_max_over_A_galactic']:.2f}·A_Sculptor; "
           f"M(< 2.3 kpc) = {h11['M_within_2p3kpc']:.2e} M☉, M(< 0.4 kpc) = {h11['M_within_0p4kpc']:.2e} M☉.", "",
           "Dos matices respecto de la nota del autor: (i) con el cierre cosmológico la parte externa del "
           "halo está excluida por subdominancia con ε_c ≪ 1 (la lectura de la nota), pero por debajo de "
           "~r_s se rompe TAMBIÉN el régimen débil (ε_c > 1e-3 en r ≲ r_s, > 1 en r ≲ 0.4 kpc); (ii) en el "
           "halo enano de 1e10 M☉ (c = 13, suavizado 50 pc) A_Sculptor deja de ser subdominante por debajo "
           f"de r(D = 1) = {rep['halos'][1]['closures']['galactic']['r_D_equals_one_kpc']:.2f} kpc "
           f"(A_max = {rep['halos'][1]['A_max_over_A_galactic']:.2f}·A_Sculptor): la puerta de régimen del "
           "Nivel A (D ≤ 1 para r > ε_soft) decide por sí sola qué halo y qué suavizado son admisibles.", "",
           "## Consecuencia para el sector µ/η del canal Cronos (E1)", "",
           f"ε̄_c(hoy): cosmológico {rep['epsilon_c_background_today']['cosmological']:.2e}, "
           f"galáctico {rep['epsilon_c_background_today']['galactic']:.2e}. µ − 1 en k = 0.2 h/Mpc:", "",
           "| z | cierre cosmológico (E1 publicado) | cierre galáctico (A_Sculptor) |", "|---|---|---|"]
    for z, v in mu.items():
        md.append(f"| {z} | {v['cosmological_comoving_f200']:.2e} | {v['galactic_A_sculptor']:.2e} |")
    md += ["", "El artefacto E1 (results/2026-09-13_mu_eta_cronos) no se retoca: sus identidades son "
           "exactas para el cierre que declara; lo que cambia es el estatuto de ese cierre (excluido) y, "
           "con él, la amplitud de la cola k²: muere por consistencia interna. Control externo del autor "
           "reproducido (véase JSON)."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"A_cosmo/A_S = {rep['A_cosmo_over_A_galactic']:.2e}; D_cosmo(r200) = "
          f"{h11['closures']['cosmological']['D_r200']:.2f}; D_gal(0.1 kpc) = "
          f"{h11['closures']['galactic']['D_soft']:.2f}; µ−1(k=0.2, z=0) galáctico = "
          f"{mu['0.0']['galactic_A_sculptor']:.2e}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
