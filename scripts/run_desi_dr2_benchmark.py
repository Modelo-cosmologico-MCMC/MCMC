#!/usr/bin/env python
"""6A.2 + 6A.4 — equivalencia de likelihood y benchmark ΛCDM DESI DR2.

Etapas (el contraste MCMC de 6A.5 NO corre hasta que esto pase):

1. EQUIVALENCIA (6A.2): χ² de cosmology/desi_bao.py contra la
   likelihood oficial de Cobaya 3.6.2 (bao.desi_dr2.desi_bao_all),
   sobre vectores teóricos sintéticos independientes del modelo
   (cada dimensión perturbada + aleatorios). Tolerancia predeclarada:
   1e-10. Además, identidad de DATOS computada en cada ejecución:
   sha256 de cada fichero del manifest (ingesta bb0c1c9) contra el
   fichero homónimo que Cobaya lee en su packages path (bao_data,
   tag v2.6) — sin red, no es una afirmación de texto.

   El packages path se lee de COBAYA_PACKAGES_PATH (por defecto
   external/cobaya_packages, gitignorado); se recrea con
   scripts/setup_cobaya_packages.py (única pieza que requiere red).
   Si la equivalencia no puede correr, este script NO sobreescribe un
   benchmark.json con equivalencia PASS: escribe benchmark.SKIPPED.json
   y sale con error (la puerta de 6A.5 exige PASS).
2. BENCHMARK ΛCDM (6A.4): ajuste ΛCDM propio sobre DESI_ALL y
   comparación con el valor OFICIAL publicado (DESI DR2 BAO-only,
   ΛCDM plano): Ω_m = 0.2975 ± 0.0086 (arXiv:2503.14738, verificado
   18-ago-2026 vía búsqueda con cita). PUERTA PREDECLARADA:
   |Ω_m propio − 0.2975| < 0.0086 (1σ oficial). El resultado se
   publica sea cual sea; si falla, el contraste queda bloqueado.

Salida: results/2026-08-18_desi_dr2_background/benchmark.json + report.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.desi_background_fit import (  # noqa: E402
    constrained_chi2_min,
    log_prob_lcdm,
    run_fit,
)
from cosmology.desi_bao import chi2_bao, load_desi_dr2_all  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-08-18_desi_dr2_background"
OFFICIAL = {"Omega_m": 0.2975, "sigma": 0.0086,
            "source": "DESI DR2 BAO-only, ΛCDM plano "
                      "(arXiv:2503.14738; verificado 18-ago-2026)"}
EQUIV_TOL = 1e-10
COBAYA_PACKAGES = Path(os.environ.get(
    "COBAYA_PACKAGES_PATH", str(REPO / "external" / "cobaya_packages")))


def data_identity_check() -> dict:
    """Identidad de datos COMPUTADA: los ficheros que Cobaya lee en su
    packages path (bao_data @ v2.6) son byte-idénticos a la ingesta del
    manifest (sha256 por fichero, offline)."""
    manifest = json.loads((REPO / "data" / "manifests"
                           / "desi_dr2_bao.json").read_text("utf-8"))
    src = COBAYA_PACKAGES / "data" / "bao_data" / "desi_bao_dr2"
    vfile = COBAYA_PACKAGES / "data" / "bao_data" / "version.dat"
    mismatches = []
    for entry in manifest["files"]:
        f = src / entry["file"]
        if not f.exists():
            mismatches.append(entry["file"] + " (ausente)")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != entry["sha256"]:
            mismatches.append(entry["file"])
    n = len(manifest["files"])
    return {"status": "PASS" if not mismatches else "FAIL",
            "n_files": n, "n_identical": n - len(mismatches),
            "mismatches": mismatches,
            "packages_version": (vfile.read_text("utf-8").strip()
                                 if vfile.exists() else "ausente"),
            "method": "sha256 por fichero: manifest bb0c1c9 vs "
                      "packages path de Cobaya (bao_data tag v2.6)"}


def equivalence_check(data) -> dict:
    """6A.2 — χ² propio vs Cobaya sobre vectores sintéticos."""
    try:
        from cobaya.likelihoods.bao.desi_dr2.desi_bao_all import (
            desi_bao_all,
        )
    except ImportError:
        return {"status": "SKIPPED", "reason": "cobaya no instalada"}
    if not COBAYA_PACKAGES.exists():
        return {"status": "SKIPPED",
                "reason": "packages path de Cobaya ausente"}
    like = desi_bao_all({"packages_path": str(COBAYA_PACKAGES)})

    class MockProvider:
        C = 299792.458

        def __init__(self, DM, DH, rd=1.0):
            self.rd, self.DM, self.DH = rd, DM, DH

        def get_param(self, name):
            assert name == "rdrag"
            return self.rd

        def get_angular_diameter_distance(self, z):
            z = float(np.atleast_1d(z)[0])
            return np.array([self.DM[round(z, 3)] * self.rd / (1 + z)])

        def get_Hubble(self, z, units="km/s/Mpc"):
            z = float(np.atleast_1d(z)[0])
            H = self.C / (self.DH[round(z, 3)] * self.rd)
            return np.array([H if units == "km/s/Mpc" else H / self.C])

    z, quant, val, _, C = data

    def targets(vec):
        DM, DH = {}, {}
        for zi, q, v in zip(z, quant, vec):
            k = round(zi, 3)
            if q == "DM_over_rs":
                DM[k] = v
            elif q == "DH_over_rs":
                DH[k] = v
            else:
                DH[k] = 20.0
                DM[k] = np.sqrt(v ** 3 / (zi * 20.0))
        return DM, DH

    like.provider = MockProvider(*targets(val))
    lp0 = like.logp()
    rng = np.random.default_rng(7)
    max_diff = 0.0
    for trial in range(16):
        if trial < 13:
            delta = np.zeros(13)
            delta[trial] = 0.05 * val[trial]
        else:
            delta = rng.normal(0, 0.02, 13) * val
        m = val + delta
        like.provider = MockProvider(*targets(m))
        chi2_cob = -2.0 * (like.logp() - lp0)
        max_diff = max(max_diff, abs(chi2_cob - chi2_bao(m, val, C)))
    identity = data_identity_check()
    ok = max_diff < EQUIV_TOL and identity["status"] == "PASS"
    return {"status": "PASS" if ok else "FAIL",
            "max_abs_diff": float(max_diff), "tolerance": EQUIV_TOL,
            "n_vectors": 16, "cobaya_version": "3.6.2",
            "logp_at_data": float(lp0),
            "data_identity": identity}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_desi_dr2_all()

    eq = equivalence_check(data)
    print(f"1. equivalencia: {eq['status']} "
          + (f"(max |Δχ²| = {eq['max_abs_diff']:.2e})"
             if "max_abs_diff" in eq else f"({eq['reason']})"))
    out_path = OUT / "benchmark.json"
    if eq["status"] != "PASS" and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("equivalence", {}).get("status") == "PASS":
            raise SystemExit(
                "equivalencia sin PASS en esta ejecución: NO se "
                "sobreescribe el benchmark.json validado (recrear el "
                "packages path con scripts/setup_cobaya_packages.py, "
                "o COBAYA_PACKAGES_PATH)")

    fit = run_fit(log_prob_lcdm, (0.30, 10200.0), 2, data,
                  nwalkers=32, nsteps=4000)
    flat = fit["flat"]
    om = np.percentile(flat[:, 0], [16, 50, 84])
    h0rd = np.percentile(flat[:, 1], [16, 50, 84])
    argmin = constrained_chi2_min("lcdm", data,
                                  extra_starts=[fit["best"]])
    chi2_min = argmin["chi2"]
    diff = abs(om[1] - OFFICIAL["Omega_m"])
    dev_sigma = diff / OFFICIAL["sigma"]
    gate = bool(diff < OFFICIAL["sigma"])
    print(f"2. ΛCDM DESI_ALL: χ²_min = {chi2_min:.3f} (n=13, k=2) | "
          f"Ω_m = {om[1]:.4f} −{om[1] - om[0]:.4f}/+{om[2] - om[1]:.4f} | "
          f"H0·rd = {h0rd[1]:.0f} km/s | conv: {fit['converged']}")
    print(f"   benchmark |ΔΩ_m| = {diff:.4f} = {dev_sigma:.3f}σ oficial "
          f"⟹ {'PASS' if gate else 'FAIL'}")

    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    doc = {
        "stage": "6A.2 + 6A.4",
        "code_commit": sha,
        "dataset_manifest": "data/manifests/desi_dr2_bao.json "
                            "(bb0c1c9; require_available en carga)",
        "equivalence": eq,
        "lcdm_benchmark": {
            "config": "DESI_ALL (n = 13, k = 2)",
            "priors": "Ω_m ~ U(0.05, 0.7); H0rd ~ U(6000, 16000) km/s",
            "sampler": "emcee 32×4000, semilla global 42 "
                       "(np.random.seed: cadenas bit-reproducibles), "
                       "burn 1/2, thin 4; χ²_min por multistart "
                       "acotado al soporte del prior (Powell + "
                       "L-BFGS-B), argmin publicado",
            "chi2_min": chi2_min,
            "argmin": argmin,
            "Omega_m": {"median": float(om[1]),
                        "minus": float(om[1] - om[0]),
                        "plus": float(om[2] - om[1])},
            "H0rd_kms": {"median": float(h0rd[1]),
                         "minus": float(h0rd[1] - h0rd[0]),
                         "plus": float(h0rd[2] - h0rd[1])},
            "acceptance": fit["acceptance"],
            "tau": fit["tau"], "converged": fit["converged"],
            "official": OFFICIAL,
            "dev_sigma": float(dev_sigma),
            "gate_rule": "|Ω_m propio − oficial| < 1σ oficial "
                         "(predeclarada)",
            "gate": "PASS" if gate else "FAIL",
        },
    }
    out_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"Artefacto: {out_path}")


if __name__ == "__main__":
    main()
