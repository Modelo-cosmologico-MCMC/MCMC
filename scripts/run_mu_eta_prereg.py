#!/usr/bin/env python
"""Preinscripción de la derivación µ(k,a), η(k,a), Σ(k,a) desde la
ontología (canal Cronos) — OBLIGATORIA antes de computar tabla alguna
y, sobre todo, antes de cualquier contraste con datos.

Congela: los cierres declarados, los parámetros (solo los del
tratado), el fondo de referencia (medianas del posterior CC+BAO+SNe ya
congelado por sha256), las mallas k/z, las TRES expectativas de la nota
(§B.5) con umbrales numéricos, la tabla de estimaciones a mano de la
nota como «a verificar» (ninguna cifra se cita hasta que la
implementación la recompute), la prohibición explícita sobre lensing y
el estatuto pendiente del canal Atlas.

Uso: python scripts/run_mu_eta_prereg.py
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

from cosmology.growth_prediction import (  # noqa: E402
    CHAINS_V1,
    load_background_posterior,
)
from cosmology.mu_eta_cronos import (  # noqa: E402
    ALPHA0_INV_MAX,
    ATLAS_STATUS,
    K_MAX_LINEAR_HMPC,
    RHO_C_MODES,
    RHO_C_OVER_MEAN,
)
from mcmc_ontology import constants as C  # noqa: E402

OUTDIR = (Path(__file__).resolve().parent.parent / "results"
          / "2026-09-13_mu_eta_cronos")

K_GRID_HMPC = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]   # 0.3 fuera de ventana
Z_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]
E1_THRESHOLD = 1e-3
E1_K_MAX = 0.2
E1_Z_MAX = 2.0

# Estimaciones A MANO de la nota de teoría (§B.3, «a verificar por la
# implementación antes de citar»): la implementación las recomputa y
# publica las diferencias; hasta entonces NINGUNA se cita.
NOTE_ESTIMATES_TO_VERIFY = {
    "alpha0_inv": 1e-6, "rho_c_over_mean": 200.0,
    "rows": [
        {"k_hMpc": 0.05, "z0_comoving": 3e-5, "z1_physical": 3e-4,
         "z3_physical": 1e-2},
        {"k_hMpc": 0.1, "z0_comoving": 1e-4, "z1_physical": 1e-3,
         "z3_physical": 5e-2},
        {"k_hMpc": 0.2, "z0_comoving": 4e-4, "z1_physical": 5e-3,
         "z3_physical": None},
    ],
}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    chains_sha = hashlib.sha256(CHAINS_V1.read_bytes()).hexdigest()
    chain = load_background_posterior()
    theta_ref = [float(v) for v in np.median(chain, axis=0)]

    doc = {
        "preregistered_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "front": ("sector perturbativo — µ(k,a), η(k,a), Σ(k,a) desde "
                  "la ontología (canal Cronos derivado; canal Atlas "
                  "declarado)"),
        "statute": (
            "derivación desde los objetos declarados del tratado (v35: "
            "ec. 11.1, 11.2, 9.1-9.4, A.3), con cada cierre nombrado. "
            "No introduce campos ni parámetros nuevos. Es PREDICCIÓN: "
            "este artefacto no carga datos ni ajusta nada."),
        "closures_declared": {
            "1_linearization": "δε_c = (3/2) ε̄_c(a) δ, válida para |δ| ≲ 1",
            "2_rho_c": {
                "modes": list(RHO_C_MODES),
                "rho_c_over_mean": RHO_C_OVER_MEAN,
                "note": ("el tratado NO fija ρ_c (Def. 11.1); las dos "
                         "lecturas dan tendencias en z opuestas y se "
                         "publican por separado — discriminador "
                         "interno declarado"),
            },
            "3_validity_window": {
                "k_max_hMpc": K_MAX_LINEAR_HMPC,
                "note": ("la extrapolación k² se corta donde la "
                         "linealización de ρ^{3/2} y la compuerta Θ(ρ̇) "
                         "toman el mando; fuera de ventana los números "
                         "se publican marcados y no se citan")},
        },
        "parameters_from_treatise_only": {
            "alpha0_inv": {"value": ALPHA0_INV_MAX,
                           "role": ("COTA (11.5) saturada — límite "
                                    "superior, NO una medida; la "
                                    "amplitud real la fijaría el sector "
                                    "galáctico (Jeans/5E), hoy "
                                    "DATA_UNAVAILABLE")},
            "rho_c_over_mean": RHO_C_OVER_MEAN,
            "epsilon_K": {"value": C.EPSILON_K,
                          "role": ("solo entra en el canal Atlas, cuyo "
                                   "coeficiente O(1) está pendiente")},
            "background_theta_ref": {
                "H0": theta_ref[0], "Omega_m": theta_ref[1],
                "epsilon": theta_ref[2], "z_trans": theta_ref[3],
                "source": ("medianas del posterior CC+BAO+SNe (v1 "
                           "corregido), cadenas ya congeladas por "
                           "sha256 en la preinscripción del 12-sep"),
                "chains_sha256": chains_sha},
        },
        "grids": {"k_hMpc": K_GRID_HMPC, "z": Z_GRID,
                  "growth_n_grid": 400},
        "expectations": {
            "E1_boring_consistency_control": {
                "statement": (
                    "con α₀⁻¹ en su cota y el cierre 'comoving', "
                    "fσ8(z) difiere de ΛCDM-con-fondo-MCMC en menos de "
                    "1e-3 en las escalas de los surveys — "
                    "indistinguible. NO es fracaso: el sector lineal no "
                    "debe inventar desviación donde el fondo casi no "
                    "cambia."),
                "rule": (f"max |R_µ(k,z) − 1| ≤ {E1_THRESHOLD} sobre "
                         f"k ≤ {E1_K_MAX} h/Mpc y z ≤ {E1_Z_MAX}, con "
                         "R_µ = [f·D](µ)/[f·D](µ≡1) al mismo fondo "
                         "(σ8 se cancela; cero parámetros nuevos)"),
                "scope": "cierre 'comoving' únicamente",
                "physical_closure_note": (
                    "el cierre 'physical' NO está sujeto a E1: por "
                    "construcción produce desviaciones mayores a z "
                    "alto (∝ (1+z)^{7/2}); sus números se publican "
                    "como el discriminador interno declarado en B.3"),
            },
            "E2_distinctive_signature_identities": {
                "statement": ("cola k² con µ−1 = 2(Σ−1) y η < 1, "
                              "amplitud ∝ α₀⁻¹; en el cierre 'physical' "
                              "el CMB-lensing a z ~ 2 aprieta α₀⁻¹"),
                "rules": [
                    "|(Σ − 1) − (µ − 1)/2| ≤ 1e-15 (exacto)",
                    "|(η − 1) + (µ − 1)| ≤ 1.01·(µ − 1)² (segundo orden)",
                    "(µ−1)(2k)/(µ−1)(k) = 4 con error relativo ≤ 1e-12",
                    "(µ−1)(α₀⁻¹/2) = (µ−1)(α₀⁻¹)/2 con error ≤ 1e-12",
                    "(µ−1)_physical/(µ−1)_comoving = (1+z)^{9/2} "
                    "con error relativo ≤ 1e-10",
                    "límite GR: α₀⁻¹ = 0 ⟹ µ = η = Σ = 1 exactamente",
                ],
            },
            "E3_out_of_sample_chain": {
                "statement": ("si el sector galáctico (Jeans/5E, sin "
                              "retuning) fija α₀⁻¹, el sector lineal "
                              "queda PREDICHO sin libertad: halos → "
                              "α₀⁻¹ → µ(k,z) → P(k)/fσ8/C_L^φφ"),
                "status": ("DECLARADA, NO EJECUTADA: α₀⁻¹ medido exige "
                           "los bytes de SPARC/Walker (DATA_UNAVAILABLE, "
                           "5E congelado); aquí solo la cota"),
            },
        },
        "note_estimates_to_verify": NOTE_ESTIMATES_TO_VERIFY,
        "atlas_channel": {
            "status": ATLAS_STATUS,
            "structural_statement": (
                "el mismo ε_K que produce G_cosmo/G_N − 1 ≈ −(3/2)ε_K "
                "(frente 6, BBN) fija la amplitud del offset de "
                "crecimiento a gran escala — dos observables, un solo "
                "residuo; consistencia cruzada falsable cuando el "
                "frente 3 cierre los coeficientes"),
            "route": ("perturbar S_Gea + ΔS_Gea en gauge newtoniano, "
                      "límite cuasi-estático, eliminar el khronon"),
        },
        "prohibitions": {
            "no_mu_eta_from_lensing": (
                "PROHIBICIÓN EXPLÍCITA: µ(k,z), η(k,z) NUNCA se eligen "
                "desde los datos de lensing — se derivan de "
                "Atlas/ε_c. Este artefacto no carga dato alguno "
                "(vigilado por test)."),
            "no_citing_out_of_window": ("ningún número con k > "
                                        f"{K_MAX_LINEAR_HMPC} h/Mpc se "
                                        "cita como predicción"),
            "no_adjustment": ("umbrales, mallas, cierres y parámetros "
                              "quedan congelados aquí; un cambio exige "
                              "nueva preinscripción con changelog"),
        },
        "what_this_does_not_claim": [
            "que µ ≠ 1 esté detectado (no lo está)",
            "que la amplitud sea la de la cota (es un límite superior)",
            "que el canal Atlas tenga coeficientes (esperan al frente 3)",
            "que la extrapolación k² valga en régimen no lineal",
        ],
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    md = [
        "# Preinscripción — µ(k,a), η(k,a), Σ(k,a) desde la ontología "
        "(canal Cronos)\n",
        f"- **Commit**: `{sha}`",
        f"- **Fecha (UTC)**: {doc['preregistered_utc']}",
        f"- **α₀⁻¹**: {ALPHA0_INV_MAX} (cota 11.5 saturada — límite "
        "superior, no medida); **ρ_c/ρ̄**: "
        f"{RHO_C_OVER_MEAN} (cierre 2, ambas lecturas); **ventana**: "
        f"k ≤ {K_MAX_LINEAR_HMPC} h/Mpc (cierre 3)",
        f"- **Fondo de referencia**: H0 = {theta_ref[0]:.2f}, Ω_m = "
        f"{theta_ref[1]:.4f}, ε = {theta_ref[2]:+.4f}, z_trans = "
        f"{theta_ref[3]:.2f} (cadenas `{chains_sha[:12]}…`)",
        "",
        f"**E1 (aburrido, control)**: max |R_µ − 1| ≤ {E1_THRESHOLD} en "
        f"k ≤ {E1_K_MAX}, z ≤ {E1_Z_MAX}, cierre comoving.",
        "**E2 (firma)**: Σ−1 = (µ−1)/2 exacto; η−1 = −(µ−1) a segundo "
        "orden; k²; lineal en α₀⁻¹; physical/comoving = (1+z)^{9/2}; "
        "GR exacto en α₀⁻¹ = 0.",
        "**E3 (cadena)**: declarada, no ejecutada (α₀⁻¹ medido exige "
        "5E).",
        "",
        "**Tabla de la nota**: estimaciones a mano, a verificar por la "
        "implementación; ninguna cifra se cita hasta entonces.",
        "",
        f"**Atlas**: {ATLAS_STATUS}.",
        "",
        "**Prohibición**: µ, η nunca se eligen desde los datos de "
        "lensing; este artefacto no carga datos.",
    ]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n",
                                               encoding="utf-8")
    print(f"Preinscripción congelada en {OUTDIR} (commit {sha[:9]})")


if __name__ == "__main__":
    main()
