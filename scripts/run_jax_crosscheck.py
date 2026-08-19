#!/usr/bin/env python
"""Crosscheck JAX ↔ NumPy del fondo (6A prioridad 3).

Compara la implementación independiente (validation/jax_background.py,
Gauss-Legendre, fórmulas del Apéndice A) contra las implementaciones
NumPy de producción, con PUERTAS predeclaradas:

    bloques 1-4 (E/H/D_H sobre malla, mapa S↔z, vector DESI, χ²):
        max |ΔX/X| < 1e-8  y  RMS |ΔX/X| < 1e-10 por bloque
        (χ²: |Δχ²| < 1e-8 absoluto)
    bloque 5 (distancias SNe de producción — trapecio 2048 puntos):
        SIN puerta: se publica el error medido tal cual — el
        crosscheck con integrador espectral MIDE el error de
        truncamiento del integrador de producción, y ese número es un
        resultado de la ronda, no un umbral que forzar.

Los θ de evaluación incluyen los argmin PUBLICADOS de los artefactos
DESI (benchmark.json + contrast.json): el crosscheck ata las dos
implementaciones exactamente donde viven los números publicados.

Salida: results/2026-08-19_jax_crosscheck/{crosscheck.json, report.md}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.background import H_of_z  # noqa: E402
from cosmology.bayesian_fit import (  # noqa: E402
    comoving_distance,
    distance_modulus,
)
from cosmology.desi_background_fit import (  # noqa: E402
    chi2_at,
    predict_desi_dr2_vector,
)
from cosmology.desi_bao import load_desi_dr2_all  # noqa: E402
from mcmc_ontology.S_map import a_to_z, s_to_a, z_to_a  # noqa: E402
from validation.jax_background import (  # noqa: E402
    DH_jax,
    E_jax,
    H_jax,
    a_to_s_jax,
    a_to_z_jax,
    chi2_jax,
    mu_jax,
    predict_desi_vector_jax,
    s_to_a_jax,
    z_to_a_jax,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-19_jax_crosscheck")
DESI_DIR = (Path(__file__).resolve().parent.parent / "results"
            / "2026-08-18_desi_dr2_background")

GATE_MAX = 1e-8
GATE_RMS = 1e-10
GATE_CHI2 = 1e-8

# θ de evaluación (H0 solo entra en H/D; E usa H0=1):
THETAS = {
    "fiducial_lcdm": (0.30, 0.0, 8.9, 1.5),
    "corpus_mcmc": (0.30, 0.012, 8.9, 1.5),
    "corner_lo": (0.20, -0.05, 1.0, 1.5),
    "corner_hi": (0.45, 0.10, 20.0, 1.5),
}


def rel_stats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rel = np.abs(a / b - 1.0)
    return {"max_rel": float(rel.max()),
            "rms_rel": float(np.sqrt(np.mean(rel ** 2)))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zg = np.linspace(0.0, 3.0, 1001)

    # --- bloque 1: E, H, D_H sobre la malla determinista ------------
    block1 = {}
    for name, (om, eps, zt, dzt) in THETAS.items():
        E_np = np.asarray(H_of_z(zg, H0=1.0, Omega_m=om, eps=eps,
                                 z_trans=zt, dz=dzt))
        E_jx = np.asarray(E_jax(zg, om, eps, zt, dzt))
        H_np = np.asarray(H_of_z(zg, H0=67.9, Omega_m=om, eps=eps,
                                 z_trans=zt, dz=dzt))
        H_jx = np.asarray(H_jax(zg, 67.9, om, eps, zt, dzt))
        DH_np = 299792.458 / H_np
        DH_jx = np.asarray(DH_jax(zg, 67.9, om, eps, zt, dzt))
        block1[name] = {"E": rel_stats(E_jx, E_np),
                        "H": rel_stats(H_jx, H_np),
                        "DH": rel_stats(DH_jx, DH_np)}
    b1_max = max(v[f]["max_rel"] for v in block1.values()
                 for f in ("E", "H", "DH"))
    b1_rms = max(v[f]["rms_rel"] for v in block1.values()
                 for f in ("E", "H", "DH"))

    # --- bloque 2: mapa S ↔ z ---------------------------------------
    Sg = np.linspace(1.001, 95.0, 500)
    a_np = np.asarray(s_to_a(Sg))
    a_jx = np.asarray(s_to_a_jax(Sg))
    zg2 = np.linspace(0.0, 3.0, 301)
    az_np = np.asarray(a_to_z(z_to_a(zg2)))
    az_jx = np.asarray(a_to_z_jax(z_to_a_jax(zg2)))
    roundtrip_S = np.asarray(a_to_s_jax(s_to_a_jax(Sg[Sg > 1.001])))
    block2 = {
        "s_to_a": rel_stats(a_jx, a_np),
        "z_roundtrip": {"max_abs": float(np.abs(az_jx - az_np).max()),
                        "rms_abs": float(np.sqrt(np.mean(
                            (az_jx - az_np) ** 2)))},
        "S_roundtrip_max_abs": float(
            np.abs(roundtrip_S - Sg[Sg > 1.001]).max()),
    }
    b2_max = max(block2["s_to_a"]["max_rel"],
                 block2["z_roundtrip"]["max_abs"])
    b2_rms = max(block2["s_to_a"]["rms_rel"],
                 block2["z_roundtrip"]["rms_abs"])

    # --- bloques 3-4: vector DESI y χ² en los argmin publicados -----
    data = load_desi_dr2_all()
    z_eff, quant, val, _, Cov = data
    bench = json.loads((DESI_DIR / "benchmark.json").read_text("utf-8"))
    contr = json.loads((DESI_DIR / "contrast.json").read_text("utf-8"))
    all_row = next(r for r in contr["rows"]
                   if r["config"] == "DESI_ALL")
    theta_sets = {
        "benchmark_argmin_lcdm":
            list(bench["lcdm_benchmark"]["argmin"]["theta"]) + [0.0,
                                                               8.9],
        "contrast_argmin_lcdm": list(all_row["argmin_lcdm"]) + [0.0,
                                                                8.9],
        "contrast_argmin_mcmc": list(all_row["argmin_mcmc"]),
        "corner": [0.25, 9000.0, -0.05, 1.0],
    }
    block3, block4 = {}, {}
    for name, th in theta_sets.items():
        om, h0rd, eps, zt = th[0], th[1], th[2], th[3]
        v_np = predict_desi_dr2_vector(om, h0rd, eps=eps, z_trans=zt,
                                       data=data)
        v_jx = np.asarray(predict_desi_vector_jax(om, h0rd, eps, zt,
                                                  1.5, z_eff, quant))
        block3[name] = rel_stats(v_jx, v_np)
        model = "mcmc" if "mcmc" in name or name == "corner" else "lcdm"
        th_model = th if model == "mcmc" else th[:2]
        c_np = chi2_at(th_model, model, data)
        c_jx = chi2_jax(v_jx, val, Cov)
        block4[name] = {"chi2_numpy": float(c_np),
                        "chi2_jax": float(c_jx),
                        "abs_diff": float(abs(c_np - c_jx))}
    b3_max = max(v["max_rel"] for v in block3.values())
    b3_rms = max(v["rms_rel"] for v in block3.values())
    b4_max = max(v["abs_diff"] for v in block4.values())

    # --- bloque 5: distancias SNe de producción (SIN puerta) --------
    z_sne = np.linspace(0.01, 2.26, 200)
    th_prod = (67.87, 0.326, 0.017, 9.1)
    dc_np = comoving_distance(z_sne, th_prod)
    mu_np = distance_modulus(z_sne, th_prod)
    from validation.jax_background import DM_jax
    dc_jx = np.asarray(DM_jax(z_sne, th_prod[0], th_prod[1],
                              th_prod[2], th_prod[3], 1.5))
    mu_jx = np.asarray(mu_jax(z_sne, th_prod[0], th_prod[1],
                              th_prod[2], th_prod[3], 1.5))
    block5 = {
        "theta": list(th_prod),
        "comoving_distance": rel_stats(dc_jx, dc_np),
        "distance_modulus_max_abs_mag": float(
            np.abs(mu_jx - mu_np).max()),
        "note": "el integrador de producción (trapecio, 2048 puntos) "
                "se compara con Gauss-Legendre espectral: la "
                "diferencia ES su error de truncamiento, publicado "
                "sin puerta",
    }

    gates = {
        "block1_background": bool(b1_max < GATE_MAX
                                  and b1_rms < GATE_RMS),
        "block2_s_map": bool(b2_max < GATE_MAX and b2_rms < GATE_RMS),
        "block3_desi_vector": bool(b3_max < GATE_MAX
                                   and b3_rms < GATE_RMS),
        "block4_chi2": bool(b4_max < GATE_CHI2),
    }
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    doc = {
        "stage": "6A prioridad 3 — crosscheck JAX ↔ NumPy",
        "code_commit": sha,
        "gates_declared": {"max_rel": GATE_MAX, "rms_rel": GATE_RMS,
                           "chi2_abs": GATE_CHI2,
                           "block5": "sin puerta (medición publicada)"},
        "summary": {"block1_max": b1_max, "block1_rms": b1_rms,
                    "block2_max": b2_max, "block2_rms": b2_rms,
                    "block3_max": b3_max, "block3_rms": b3_rms,
                    "block4_max_chi2_diff": b4_max},
        "gates": gates, "all_gates_pass": bool(all(gates.values())),
        "block1_background": block1,
        "block2_s_map": block2,
        "block3_desi_vector": block3,
        "block4_chi2": block4,
        "block5_sne_production": block5,
    }
    (OUT / "crosscheck.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(f"bloque 1 (E/H/DH):    max {b1_max:.2e}  rms {b1_rms:.2e}")
    print(f"bloque 2 (S↔z):       max {b2_max:.2e}  rms {b2_rms:.2e}")
    print(f"bloque 3 (vector):    max {b3_max:.2e}  rms {b3_rms:.2e}")
    print(f"bloque 4 (χ²):        max |Δχ²| {b4_max:.2e}")
    print(f"bloque 5 (SNe prod.): max rel "
          f"{block5['comoving_distance']['max_rel']:.2e} "
          f"(sin puerta — error del trapecio de producción)")
    print(f"PUERTAS: {gates} ⟹ "
          f"{'PASS' if doc['all_gates_pass'] else 'FAIL'}")
    print(f"Artefacto: {OUT / 'crosscheck.json'}")


if __name__ == "__main__":
    main()
