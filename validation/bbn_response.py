"""Frente 6 — respuesta de las abundancias primordiales a G_cosmo/G_N,
N_eff, ω_b y τ_n, DERIVADA con un código BBN público (PRyMordial,
Burns–Tait–Valli 2023, arXiv:2307.07061) a commit fijado.

Este módulo NO es ruta de producción: requiere el clone de PRyMordial en
external/PRyMordial (gitignorado) y sus dependencias opcionales
(numdifftools, numba). La suite no lo importa. Su salida es la tabla de
respuesta que cosmology/bbn_g.py fija como constantes y que el candado
compara contra el artefacto.

CÓMO ENTRA G_cosmo. En BBN la gravedad solo aparece en la tasa de
expansión, H² = 8πG_cosmo ρ/3; la física nuclear y débil no ve G. Con
G_cosmo = G_N(1 + δ_G) la implementación exacta es reescalar la masa de
Planck que PRyMordial usa en H (PRyMini.Mpl = 1/√(G_N(1+δ_G))) y DEJAR
FIJO η_b (que PRyMordial deriva de ω_b con la G_N de laboratorio en la
densidad crítica de hoy — reescalar GN a ciegas corrompería η_b). Así
δ_G es exactamente el observable del Contraste de los Residuos,
G_cosmo/G_N − 1 = (2ξ − α_a)/(3λ_K − 1) − 1.

Ajuste local: ln X = ln X₀ + a_G·ln(1+δ_G) + a_N·ΔN + b_N·ΔN² +
a_ω·ln(ω_b/ω_b⁰) + a_τ·ln(τ_n/τ_n⁰) para X ∈ {Y_P, D/H}; los términos
cruzados se miden y se publican como cota de validez.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRYM_DIR = ROOT / "external" / "PRyMordial"
PRYM_COMMIT = "725d8a8db3ad5ea2630580d825c9d0d69ed76533"
PRYM_URL = "https://github.com/vallima/PRyMordial"

FIDUCIAL = {"omega_b": 0.02237, "tau_n": 878.4, "Neff_SM": 3.044}
DESIGN = {
    "delta_G": [-0.06, -0.03, -0.015, 0.0, 0.015, 0.03, 0.06],
    "delta_Neff": [-1.0, -0.5, 0.0, 0.5, 1.0],
    "omega_b": [0.02207, 0.02222, 0.02237, 0.02252, 0.02267],
    "tau_n": [877.4, 878.4, 879.4],
}


def require_prymordial():
    """Fallo cerrado: el clone debe existir y estar en el commit fijado."""
    if not (PRYM_DIR / "PRyM" / "PRyM_main.py").exists():
        raise SystemExit(
            f"PRyMordial ausente. Clonar a commit fijado:\n"
            f"  git clone --filter=blob:none {PRYM_URL} {PRYM_DIR} && "
            f"git -C {PRYM_DIR} checkout {PRYM_COMMIT}\n"
            "  pip install numdifftools numba")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PRYM_DIR,
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != PRYM_COMMIT:
        raise SystemExit(f"PRyMordial en {head[:9]}, se exige {PRYM_COMMIT[:9]}")
    return head


class PRyMRunner:
    """Envoltorio mínimo: fija banderas, cachea las tasas débiles y
    expone run(delta_G, delta_Neff, omega_b, tau_n, nacre) → (Y_P, D/H)."""

    def __init__(self):
        require_prymordial()
        sys.path.insert(0, str(PRYM_DIR))
        os.chdir(PRYM_DIR)                      # PRyMordial resuelve sus tablas desde cwd
        import PRyM.PRyM_init as PRyMini
        self.ini = PRyMini
        try:
            import numba  # noqa: F401
        except ImportError:
            PRyMini.numba_flag = False
        PRyMini.smallnet_flag = True            # 12 reacciones: OK para Y_P y D/H
        PRyMini.compute_bckg_flag = True        # el fondo depende de G: recomputar
        PRyMini.save_bckg_flag = False
        need = not (PRYM_DIR / "PRyMrates" / "nTOp" / "nTOp_frwrd_HT.txt").exists()
        PRyMini.compute_nTOp_flag = need        # tasas débiles: microfísica, sin G
        PRyMini.save_nTOp_flag = need
        import PRyM.PRyM_main as PRyMmain
        self.main = PRyMmain
        self.GN0 = PRyMini.GN
        self.Mpl0 = PRyMini.Mpl
        self.nacre = False
        if need:                                # primera corrida: genera y guarda
            self.run()
            PRyMini.compute_nTOp_flag = False
            PRyMini.save_nTOp_flag = False

    def run(self, delta_G=0.0, delta_Neff=0.0, omega_b=FIDUCIAL["omega_b"],
            tau_n=FIDUCIAL["tau_n"], nacre=False):
        ini = self.ini
        if nacre != self.nacre:
            ini.nacreii_flag = nacre
            ini.ReloadKeyRates()
            self.nacre = nacre
        ini.Mpl = 1.0 / np.sqrt(self.GN0 * (1.0 + delta_G))   # solo H
        ini.DeltaNeff = delta_Neff
        ini.Omegabh2 = omega_b
        ini.eta0b = ini.Omegabh2_to_eta0b * omega_b            # con G_N de laboratorio
        ini.tau_n = tau_n
        try:
            res = self.main.PRyMclass().PRyMresults()
        finally:
            ini.Mpl = self.Mpl0
            ini.DeltaNeff = 0.0
        return {"Neff": float(res[0]), "YP_CMB": float(res[3]),
                "YP": float(res[4]), "DH": float(res[5]) * 1e-5,
                "He3H": float(res[6]) * 1e-5, "Li7H": float(res[7]) * 1e-10}


def _fit_powerlaw(xs, ys, x0):
    """a tal que ln y = ln y0 + a·ln(x/x0) (mínimos cuadrados); devuelve
    (a, max residuo relativo)."""
    lx = np.log(np.asarray(xs) / x0)
    ly = np.log(np.asarray(ys))
    A = np.vstack([np.ones_like(lx), lx]).T
    c, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = np.max(np.abs(np.exp(A @ c) / np.asarray(ys) - 1.0))
    return float(c[1]), float(resid)


def compute_response(verbose: bool = True) -> dict:
    T0 = time.time()
    R = PRyMRunner()

    def tick(msg):
        if verbose:
            print(f"{time.time() - T0:6.1f}s {msg}", flush=True)

    fid = R.run()
    tick(f"SM fiducial: Y_P = {fid['YP']:.5f}, D/H = {fid['DH']*1e5:.4f}e-5, "
         f"N_eff = {fid['Neff']:.4f}")
    out = {"prymordial": {"url": PRYM_URL, "commit": PRYM_COMMIT,
                          "flags": {"smallnet": True, "rates": "PRIMAT (key), "
                                    "NACRE II como sistemático",
                                    "compute_bckg": True, "nTOp": "cacheadas"}},
           "fiducial": dict(FIDUCIAL), "sm_prediction": fid, "design": DESIGN,
           "runs": {}, "exponents": {}}

    # --- δ_G: ley de potencias en (1+δ_G) ---
    rows = [R.run(delta_G=d) for d in DESIGN["delta_G"]]
    out["runs"]["delta_G"] = rows
    for key in ("YP", "DH"):
        a, res = _fit_powerlaw([1 + d for d in DESIGN["delta_G"]],
                               [r[key] for r in rows], 1.0)
        out["exponents"][f"a_G_{key}"] = a
        out["exponents"][f"a_G_{key}_max_rel_resid"] = res
    tick(f"δ_G: a_G(Y_P) = {out['exponents']['a_G_YP']:.4f}, "
         f"a_G(D/H) = {out['exponents']['a_G_DH']:.4f}")

    # --- ΔN_eff: cuadrático en ΔN ---
    rows = [R.run(delta_Neff=d) for d in DESIGN["delta_Neff"]]
    out["runs"]["delta_Neff"] = rows
    dN = np.array(DESIGN["delta_Neff"])
    for key in ("YP", "DH"):
        ly = np.log([r[key] for r in rows])
        c = np.polyfit(dN, ly - np.log(fid[key]), 2)
        out["exponents"][f"a_N_{key}"] = float(c[1])
        out["exponents"][f"b_N_{key}"] = float(c[0])
        out["exponents"][f"N_{key}_max_rel_resid"] = float(np.max(np.abs(
            np.exp(np.polyval(c, dN) + np.log(fid[key])) / np.array([r[key] for r in rows]) - 1)))
    tick(f"ΔN_eff: a_N(Y_P) = {out['exponents']['a_N_YP']:.4f}, "
         f"a_N(D/H) = {out['exponents']['a_N_DH']:.4f}")

    # --- ω_b y τ_n: leyes de potencias ---
    rows = [R.run(omega_b=w) for w in DESIGN["omega_b"]]
    out["runs"]["omega_b"] = rows
    for key in ("YP", "DH"):
        a, res = _fit_powerlaw(DESIGN["omega_b"], [r[key] for r in rows],
                               FIDUCIAL["omega_b"])
        out["exponents"][f"a_w_{key}"] = a
        out["exponents"][f"a_w_{key}_max_rel_resid"] = res
    rows = [R.run(tau_n=tn) for tn in DESIGN["tau_n"]]
    out["runs"]["tau_n"] = rows
    for key in ("YP", "DH"):
        a, res = _fit_powerlaw(DESIGN["tau_n"], [r[key] for r in rows],
                               FIDUCIAL["tau_n"])
        out["exponents"][f"a_tau_{key}"] = a
        out["exponents"][f"a_tau_{key}_max_rel_resid"] = res
    tick(f"ω_b: a_w(Y_P) = {out['exponents']['a_w_YP']:.4f}, a_w(D/H) = "
         f"{out['exponents']['a_w_DH']:.4f}; τ_n: {out['exponents']['a_tau_YP']:.3f}, "
         f"{out['exponents']['a_tau_DH']:.3f}")

    # --- término cruzado δ_G × ω_b (cota de validez del modelo separable) ---
    cross = R.run(delta_G=-0.03, omega_b=0.02252)
    pred = {}
    for key in ("YP", "DH"):
        pred[key] = fid[key] * (0.97 ** out["exponents"][f"a_G_{key}"]) * (
            (0.02252 / FIDUCIAL["omega_b"]) ** out["exponents"][f"a_w_{key}"])
    out["cross_term_check"] = {"point": {"delta_G": -0.03, "omega_b": 0.02252},
                               "run": cross, "separable_prediction": pred,
                               "rel_error": {k: cross[k] / pred[k] - 1 for k in pred}}

    # --- sistemático nuclear: NACRE II frente a PRIMAT en el fiducial ---
    nac = R.run(nacre=True)
    R.run(nacre=False)
    out["nuclear_rates_systematic"] = {
        "primat": {"YP": fid["YP"], "DH": fid["DH"]},
        "nacreii": {"YP": nac["YP"], "DH": nac["DH"]},
        "delta_DH": nac["DH"] - fid["DH"], "delta_YP": nac["YP"] - fid["YP"]}
    tick(f"NACRE II − PRIMAT: ΔD/H = {(nac['DH']-fid['DH'])*1e5:+.4f}e-5, "
         f"ΔY_P = {nac['YP']-fid['YP']:+.5f}")
    out["seconds"] = round(time.time() - T0, 1)
    return out


if __name__ == "__main__":
    out = compute_response()
    dest = ROOT / "results" / "2026-09-14_bbn_g"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "response_prymordial.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"tabla de respuesta escrita en {dest / 'response_prymordial.json'}")
