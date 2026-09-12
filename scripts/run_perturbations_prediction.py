#!/usr/bin/env python
"""Computa la banda predicha de fσ8 (out-of-sample) BAJO la
preinscripción congelada y publica el artefacto versionado.

Falla cerrado sin preinscripción o si las cadenas de fondo no son las
congeladas (sha256). Clasifica el desenlace con la regla preinscrita
(A esperado / B contrario) y lo publica sea cual sea.

Uso: python scripts/run_perturbations_prediction.py
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
    OUTDIR,
    Z_BAND,
    load_background_posterior,
    out_of_sample_chi2,
    posterior_band,
    prior_envelope,
)

PREREG = OUTDIR / "preregistration.json"


def main() -> int:
    if not PREREG.exists():
        raise SystemExit("PREREGISTRATION_MISSING: congela primero "
                         "con scripts/run_perturbations_prereg.py")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    chains_sha = hashlib.sha256(CHAINS_V1.read_bytes()).hexdigest()
    if chains_sha != prereg["method_frozen"]["chains_sha256"]:
        raise SystemExit("CHAINS_MISMATCH: las cadenas de fondo no "
                         "son las congeladas en la preinscripción")

    chain = load_background_posterior()
    theta_ref = tuple(np.median(chain, axis=0))
    print(f"[fondo] θ_ref (mediana) = H0 = {theta_ref[0]:.2f}, "
          f"Ω_m = {theta_ref[1]:.4f}, ε = {theta_ref[2]:+.4f}, "
          f"z_trans = {theta_ref[3]:.2f}")

    print("[banda] propagando el posterior…")
    band = posterior_band(chain)
    env = prior_envelope(Z_BAND, theta_ref)
    dev = np.abs(band["ratio_p50"] - 1.0)
    max_dev = float(np.max(dev))
    max_env = float(np.max(env))
    within = bool(np.all(dev <= env))

    print("[oos] contraste out-of-sample con la compilación RSD…")
    oos = out_of_sample_chi2(theta_ref)
    outcome = "A" if (within and abs(oos["delta_chi2"]) <= 2.0) else "B"

    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {
        "outcome": outcome,
        "preregistration_sha256": prereg_sha,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "theta_reference": {"H0": theta_ref[0], "Omega_m": theta_ref[1],
                            "epsilon": theta_ref[2],
                            "z_trans": theta_ref[3]},
        "band": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in band.items()},
        "prior_envelope_abs_dev": env.tolist(),
        "max_abs_ratio_dev_p50": max_dev,
        "max_prior_envelope": max_env,
        "ratio_within_prior_envelope": within,
        "out_of_sample": oos,
    }
    (OUTDIR / "prediction_band.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # figura de la banda (documental)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        from cosmology.extended_likelihoods import load_fsigma8_data

        zd, fs8, sig = load_fsigma8_data()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
                                       sharex=True)
        ax1.fill_between(Z_BAND, band["abs_p16"], band["abs_p84"],
                         alpha=0.35, label="banda 68 % (posterior de "
                         "fondo, σ8 externa)")
        ax1.plot(Z_BAND, band["abs_p50"], lw=1.5, label="p50")
        ax1.errorbar(zd, fs8, yerr=sig, fmt="o", ms=4, capsize=2,
                     label="compilación RSD (out-of-sample)")
        ax1.set_ylabel(r"$f\sigma_8(z)$")
        ax1.legend(fontsize=8)
        ax2.fill_between(Z_BAND, band["ratio_p16"], band["ratio_p84"],
                         alpha=0.35, label="R(z) 68 %")
        ax2.plot(Z_BAND, band["ratio_p50"], lw=1.5)
        ax2.plot(Z_BAND, 1.0 + env, "k--", lw=1,
                 label="envolvente prior ε = ±0.05")
        ax2.plot(Z_BAND, 1.0 - env, "k--", lw=1)
        ax2.axhline(1.0, color="gray", lw=0.7)
        ax2.set_xlabel("z")
        ax2.set_ylabel(r"$R = f\sigma_8^{MCMC}/f\sigma_8^{\Lambda CDM}$")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTDIR / "fsigma8_band.png", dpi=140)
    except Exception as exc:            # noqa: BLE001 — solo la figura
        print(f"[aviso] figura no generada: {exc}")

    md = [
        f"# Predicción perturbativa fσ8 out-of-sample — desenlace "
        f"**{outcome}**\n",
        f"Preinscripción: `{prereg_sha[:12]}` (congelada ANTES de "
        f"computar); ejecución: commit `{sha[:9]}`.",
        "",
        f"- θ de referencia (mediana del posterior CC+BAO+SNe): "
        f"H0 = {theta_ref[0]:.2f}, Ω_m = {theta_ref[1]:.4f}, "
        f"ε = {theta_ref[2]:+.4f}, z_trans = {theta_ref[3]:.2f}",
        f"- max_z |R_p50(z) − 1| = {max_dev:.2e} frente a la "
        f"envolvente del prior (máx {max_env:.2e}); dentro: "
        f"{within}",
        f"- Out-of-sample fσ8 ({oos['n_points']} puntos RSD, σ8 "
        f"externa = {oos['sigma8_external']}): χ²_MCMC = "
        f"{oos['chi2_mcmc']:.2f}, χ²_ΛCDM = {oos['chi2_lcdm']:.2f}, "
        f"Δχ² = {oos['delta_chi2']:+.3e} (regla |Δχ²| ≤ 2)",
        "",
        "**Lectura del desenlace A (preinscrita)**: el sector lineal "
        "NO inventa una desviación grande donde el fondo casi no "
        "cambia — es el control de consistencia esperado, no un "
        "fracaso ni una validación del modelo frente a ΛCDM.",
        "",
        "**Nota estructural (la misma de 6A, heredada)**: la "
        "magnitud ~1e-7 de |R − 1| y de la envolvente no es un "
        "accidente numérico: con z_trans ≈ 9 y la transición "
        "normalizada hoy, F(z)/F(0) es casi constante en todo "
        "z ≤ 2, así que ni el fondo (BAO, #12) ni el crecimiento "
        "lineal (aquí) pueden sondear ε en esta parametrización — "
        "el sector lineal hereda el resultado negativo estructural "
        "de 6A en vez de fabricar señal.",
        "",
        "µ = η = 1 aquí (límite GR). El objetivo discriminante "
        "µ(k,z), η(k,z) ≠ 1 queda DECLARADO: debe derivarse de "
        "Atlas/ε_c (frentes 3 y 10); prohibido elegirlos desde los "
        "datos de lensing (preinscripción).",
    ]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n",
                                      encoding="utf-8")
    print(f"\nDESENLACE {outcome} → {OUTDIR / 'prediction_band.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
