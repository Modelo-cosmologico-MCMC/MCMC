#!/usr/bin/env python
"""6A.2 + 6A.4 — equivalencia de likelihood y benchmark ΛCDM DESI DR2.

Etapas (el contraste MCMC de 6A.5 NO corre hasta que esto pase):

1. EQUIVALENCIA (6A.2): χ² de cosmology/desi_bao.py contra la
   likelihood oficial de Cobaya 3.6.2 (bao.desi_dr2.desi_bao_all),
   sobre vectores teóricos sintéticos independientes del modelo
   (cada dimensión perturbada + aleatorios). Tolerancia predeclarada:
   1e-10. Además, identidad de DATOS: el tag v2.6 que Cobaya fija es
   git-idéntico (diff = 0) a la ingesta bb0c1c9.
2. BENCHMARK ΛCDM (6A.4): ajuste ΛCDM propio sobre DESI_ALL y
   comparación con el valor OFICIAL publicado (DESI DR2 BAO-only,
   ΛCDM plano): Ω_m = 0.2975 ± 0.0086 (arXiv:2503.14738, verificado
   18-ago-2026 vía búsqueda con cita). PUERTA PREDECLARADA:
   |Ω_m propio − 0.2975| < 0.0086 (1σ oficial). El resultado se
   publica sea cual sea; si falla, el contraste queda bloqueado.

Salida: results/2026-08-18_desi_dr2_background/benchmark.json + report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.desi_background_fit import (  # noqa: E402
    chi2_at,
    log_prob_lcdm,
    run_fit,
)
from cosmology.desi_bao import chi2_bao, load_desi_dr2_all  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-18_desi_dr2_background")
OFFICIAL = {"Omega_m": 0.2975, "sigma": 0.0086,
            "source": "DESI DR2 BAO-only, ΛCDM plano "
                      "(arXiv:2503.14738; verificado 18-ago-2026)"}
EQUIV_TOL = 1e-10
COBAYA_PACKAGES = Path(
    "/tmp/claude-0/-home-user-MCMC-unificado/"
    "995c4863-866f-584a-a5a9-a861b4efeb31/scratchpad/cobaya_packages")


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
    return {"status": "PASS" if max_diff < EQUIV_TOL else "FAIL",
            "max_abs_diff": float(max_diff), "tolerance": EQUIV_TOL,
            "n_vectors": 16, "cobaya_version": "3.6.2",
            "logp_at_data": float(lp0),
            "data_identity": "bao_data tag v2.6 (pin de Cobaya) "
                             "git-idéntico a la ingesta bb0c1c9 "
                             "(diff desi_bao_dr2/ = 0 líneas)"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_desi_dr2_all()

    eq = equivalence_check(data)
    print(f"1. equivalencia: {eq['status']} "
          + (f"(max |Δχ²| = {eq['max_abs_diff']:.2e})"
             if "max_abs_diff" in eq else f"({eq['reason']})"))

    fit = run_fit(log_prob_lcdm, (0.30, 10200.0), 2, data,
                  nwalkers=32, nsteps=3000)
    flat = fit["flat"]
    om = np.percentile(flat[:, 0], [16, 50, 84])
    h0rd = np.percentile(flat[:, 1], [16, 50, 84])
    from scipy.optimize import minimize
    res = minimize(lambda t: chi2_at(t, "lcdm", data), fit["best"],
                   method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10})
    chi2_min = float(res.fun)
    diff = abs(om[1] - OFFICIAL["Omega_m"])
    gate = bool(diff < OFFICIAL["sigma"])
    print(f"2. ΛCDM DESI_ALL: χ²_min = {chi2_min:.3f} (n=13, k=2) | "
          f"Ω_m = {om[1]:.4f} −{om[1] - om[0]:.4f}/+{om[2] - om[1]:.4f} | "
          f"H0·rd = {h0rd[1]:.0f} km/s | conv: {fit['converged']}")
    print(f"   benchmark |ΔΩ_m| = {diff:.4f} vs 1σ oficial "
          f"{OFFICIAL['sigma']} ⟹ {'PASS' if gate else 'FAIL'}")

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
            "sampler": "emcee 32×3000, semilla 42, burn 1/2, thin 4; "
                       "χ²_min refinado con Nelder-Mead desde el MAP",
            "chi2_min": chi2_min,
            "Omega_m": {"median": float(om[1]),
                        "minus": float(om[1] - om[0]),
                        "plus": float(om[2] - om[1])},
            "H0rd_kms": {"median": float(h0rd[1]),
                         "minus": float(h0rd[1] - h0rd[0]),
                         "plus": float(h0rd[2] - h0rd[1])},
            "acceptance": fit["acceptance"],
            "tau": fit["tau"], "converged": fit["converged"],
            "official": OFFICIAL,
            "gate_rule": "|Ω_m propio − oficial| < 1σ oficial "
                         "(predeclarada)",
            "gate": "PASS" if gate else "FAIL",
        },
    }
    (OUT / "benchmark.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"Artefacto: {OUT / 'benchmark.json'}")


if __name__ == "__main__":
    main()
