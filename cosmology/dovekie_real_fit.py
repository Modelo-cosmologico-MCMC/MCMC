"""Primera aplicación del pipeline SN Dovekie a los datos REALES:
ΛCDM vs MCMC-fondo sobre Dovekie (STAT+SYS) + CC + BAO, bajo
preinscripción congelada ANTES de la primera lectura del HD.

DOBLE BARRERA (ejecutable): el HD real solo es accesible vía
cosmology.dovekie_sn.load_dovekie_hd, que exige el PASS de la validación
por mocks (#15) citando su preinscripción por sha256; y el ejecutor
(scripts/run_dovekie_real.py) exige además la preinscripción de ESTA
aplicación (results/2026-09-13_dovekie_real/preregistration.json) y cita
ambos hashes en el artefacto. Nada de este módulo lee MU real por sí
mismo: load_real_data lo hace explícitamente y solo la llama el ejecutor.

COMPOSICIÓN (sin duplicar maquinaria validada):
  - Bloque SN: cosmology.dovekie_sn (parser, W oficial, μ en la
    convención del script oficial, χ̃² con M marginalizada — equivalencia
    con la likelihood oficial |Δχ²| = 1.5e-9 sobre mocks, #15). H0 queda
    FIJADA a 70 dentro del bloque SN: M absorbe exactamente 5·log10(H0),
    así que el bloque SN es independiente de H0 (testeado) y coherente
    con que CC y BAO lleven la H0 del vector de parámetros.
  - Bloques CC y BAO: cosmology.bayesian_fit (log_like_Hz, log_like_bao,
    las mismas tablas y la misma H_of_z de los ajustes de producción).

PARÁMETROS Y PRIORS (congelados en la preinscripción; Apéndice F, #12):
  ΛCDM: θ = (H0, Ω_m)              k = 2
  MCMC: θ = (H0, Ω_m, ε_Λ, z_trans) k = 4   (ε = 0 ⟹ ΛCDM exacto, Prop. A.1)
  H0 ~ N(67.4, 5²) truncada a (60, 80)  [F.4]
  Ω_m ~ U(0.10, 0.50) — soporte IDÉNTICO al prior del chain oficial de
        Dovekie (benchmark like-for-like); más ancho que el (0.20, 0.40)
        de bayesian_fit para no truncar el posterior SN-only (0.33 ± 0.015)
  ε ~ N(0.012, 0.05²) truncada a (−0.05, 0.10);  z_trans ~ U(1, 20)
  M (SNe) marginalizada analíticamente en ambos brazos; no cuenta en k.
  n = 1820 (SN) + 31 (CC) + 6 (BAO) = 1857 para el BIC.

ROLES DECLARADOS (traspaso 12-sep §2): Dovekie real = PRIMERA APLICACIÓN;
Unite = benchmark armonizado, NO replicación independiente; Union3 =
contraste externo; Pantheon+ = disección. Esta corrida no «confirma» nada
que Unite ya contenga.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cosmology.bayesian_fit import (
    information_criteria,
    load_bao_data,
    load_Hz_data,
    log_like_bao,
    log_like_Hz,
)
from cosmology.dovekie_sn import (
    MOCK_VALIDATION,
    N_SN_EXPECTED,
    chi2_sn_marginalized,
    load_dovekie_hd,
    load_dovekie_inv_cov,
    mu_model,
    require_mock_validation_pass,
)
from mcmc_ontology import constants as C
from mcmc_ontology.data_registry import require_available

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-09-13_dovekie_real"
PREREG = OUTDIR / "preregistration.json"

PRIOR_H0 = (60.0, 80.0)
PRIOR_H0_GAUSS = (67.4, 5.0)
PRIOR_OMEGA_M = (0.10, 0.50)
PRIOR_EPS = (-0.05, 0.10)
PRIOR_EPS_GAUSS = (0.012, 0.05)
PRIOR_ZTRANS = (1.0, 20.0)
K_PARAMS = {"lcdm": 2, "mcmc": 4}
PARAM_NAMES = {"lcdm": ["H0", "Omega_m"],
               "mcmc": ["H0", "Omega_m", "epsilon", "z_trans"]}
BOUNDS = {"lcdm": (PRIOR_H0, PRIOR_OMEGA_M),
          "mcmc": (PRIOR_H0, PRIOR_OMEGA_M, PRIOR_EPS, PRIOR_ZTRANS)}
START = {"lcdm": (67.4, 0.32), "mcmc": (67.4, 0.32, 0.012, 8.9)}


# ---------------------------------------------------------------------
# Referencia oficial (no es dato: no entra en ningún likelihood)
# ---------------------------------------------------------------------

def official_omega_m() -> dict:
    """Ω_m del chain oficial nautilus flat-ΛCDM DES-SN only, ponderado
    por exp(log_weight). Registro 6A.1: manifest + sha256 verificados."""
    raw = require_available("des_dovekie_chains")
    path = raw / "dovekie_lcdm_nautilus.txt"
    with open(path, encoding="utf-8") as fh:
        cols = fh.readline().lstrip("#").split()
    arr = np.loadtxt(path)
    om = arr[:, cols.index("cosmological_parameters--omega_m")]
    lw = arr[:, cols.index("log_weight")]
    w = np.exp(lw - lw.max())
    w /= w.sum()
    mean = float(np.sum(w * om))
    sd = float(np.sqrt(np.sum(w * (om - mean) ** 2)))
    order = np.argsort(om)
    cdf = np.cumsum(w[order])
    oms = om[order]

    def q(p_):
        return float(np.interp(p_, cdf, oms))

    return {"mean": mean, "sd": sd, "median": q(0.5),
            "p16": q(0.16), "p84": q(0.84), "p025": q(0.025),
            "p975": q(0.975), "n_rows": int(len(om)),
            "n_eff": float(1.0 / np.sum(w ** 2)),
            "prior_support": [float(om.min()), float(om.max())]}


# ---------------------------------------------------------------------
# Datos (el HD real solo aquí, y solo tras las barreras)
# ---------------------------------------------------------------------

def load_real_data(cov_kind: str = "STAT+SYS") -> dict:
    """Ensambla Dovekie (REAL, guardado por el PASS de mocks) + CC + BAO.
    Devuelve también los sha256 de las llaves de la barrera."""
    mock_doc = require_mock_validation_pass()
    hd = load_dovekie_hd()
    if hd["n_sn"] != N_SN_EXPECTED:
        raise RuntimeError(f"conteo SN {hd['n_sn']} != {N_SN_EXPECTED}")
    return {
        "zHD": hd["zHD"], "zHEL": hd["zHEL"], "mu": hd["MU"],
        "W": load_dovekie_inv_cov(cov_kind), "cov_kind": cov_kind,
        "hz": load_Hz_data(), "bao": load_bao_data(),
        "n_sn": int(hd["n_sn"]),
        "mock_validation_sha256": hashlib.sha256(
            MOCK_VALIDATION.read_bytes()).hexdigest(),
        "mock_prereg_sha256_cited": mock_doc["preregistration_sha256"],
    }


def n_points(data: dict, blocks=("sn", "cc", "bao")) -> int:
    n = 0
    if "sn" in blocks:
        n += int(len(data["zHD"]))
    if "cc" in blocks:
        n += int(len(data["hz"].z))
    if "bao" in blocks:
        n += int(len(data["bao"].z))
    return n


# ---------------------------------------------------------------------
# Modelo, priors, χ² por bloques
# ---------------------------------------------------------------------

def theta_full(theta, model: str) -> tuple:
    if model == "lcdm":
        H0, Om = theta
        return (float(H0), float(Om), 0.0, float(C.Z_TRANS))
    if model == "mcmc":
        H0, Om, eps, z_t = theta
        return (float(H0), float(Om), float(eps), float(z_t))
    raise ValueError(model)


def log_prior(theta, model: str) -> float:
    for value, (lo, hi) in zip(theta, BOUNDS[model]):
        if not (lo < value < hi):
            return -np.inf
    H0 = theta[0]
    lp = -0.5 * ((H0 - PRIOR_H0_GAUSS[0]) / PRIOR_H0_GAUSS[1]) ** 2
    if model == "mcmc":
        lp += -0.5 * ((theta[2] - PRIOR_EPS_GAUSS[0]) / PRIOR_EPS_GAUSS[1]) ** 2
    return float(lp)


def chi2_blocks(theta, model: str, data: dict,
                blocks=("sn", "cc", "bao")) -> dict:
    """χ² por bloque en θ (sin priors). El bloque SN es el χ̃² oficial
    con M marginalizada (incluye ln(S/2π), constante e idéntica en ambos
    brazos: no afecta a ΔAIC/ΔBIC)."""
    H0, Om, eps, z_t = theta_full(theta, model)
    out = {}
    if "sn" in blocks:
        m = mu_model(data["zHD"], data["zHEL"], Om, eps=eps, z_trans=z_t)
        out["sn"] = float(chi2_sn_marginalized(m, data["mu"], data["W"]))
    if "cc" in blocks:
        out["cc"] = float(-2.0 * log_like_Hz((H0, Om, eps, z_t), data["hz"]))
    if "bao" in blocks:
        out["bao"] = float(-2.0 * log_like_bao((H0, Om, eps, z_t), data["bao"]))
    out["total"] = float(sum(v for kk, v in out.items()))
    return out


def log_prob(theta, model: str, data: dict,
             blocks=("sn", "cc", "bao")) -> float:
    lp = log_prior(theta, model)
    if not np.isfinite(lp):
        return -np.inf
    try:
        chi2 = chi2_blocks(theta, model, data, blocks)["total"]
    except (ValueError, FloatingPointError):
        return -np.inf
    if not np.isfinite(chi2):
        return -np.inf
    return lp - 0.5 * chi2


# ---------------------------------------------------------------------
# Sampler (convenciones de #12) y χ²_min acotado
# ---------------------------------------------------------------------

def run_fit(model: str, data: dict, blocks=("sn", "cc", "bao"),
            nwalkers: int = 32, nsteps: int = 3000, seed: int = 42) -> dict:
    """emcee con semilla REAL: np.random.seed fija el stream interno del
    sampler (emcee 3 lo copia del estado global legacy) y default_rng la
    bola inicial — cadenas bit-idénticas entre procesos (#12)."""
    import emcee
    ndim = K_PARAMS[model]
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    center = np.asarray(START[model], float)
    p0 = center + 1e-3 * rng.normal(size=(nwalkers, ndim)) * center
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    args=(model, data, blocks))
    sampler.run_mcmc(p0, nsteps, progress=False)
    burn = nsteps // 2
    flat = sampler.get_chain(discard=burn, thin=4, flat=True)
    lp = sampler.get_log_prob(discard=burn, thin=4, flat=True)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        converged = bool(np.all(nsteps > 50 * np.asarray(tau)))
        tau = np.asarray(tau).tolist()
    except Exception:              # noqa: BLE001 — emcee lanza genérico
        tau, converged = None, False
    return {"flat": flat, "logp": lp, "best": flat[int(np.argmax(lp))],
            "acceptance": float(np.mean(sampler.acceptance_fraction)),
            "tau": tau, "converged": converged,
            "nwalkers": nwalkers, "nsteps": nsteps, "seed": seed,
            "burn": burn, "thin": 4}


def constrained_chi2_min(model: str, data: dict,
                         blocks=("sn", "cc", "bao"),
                         extra_starts=()) -> dict:
    """χ²_min sobre el CIERRE del soporte de los priors (multistart Powell
    acotado + pulido L-BFGS-B); el argmin puede caer en la frontera — se
    publica con la bandera at_boundary, no se oculta (#12)."""
    from scipy.optimize import minimize
    bounds = BOUNDS[model]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def f(t):
        try:
            v = chi2_blocks(t, model, data, blocks)["total"]
        except (ValueError, FloatingPointError):
            return 1e30
        return v if np.isfinite(v) else 1e30

    starts = [np.asarray(START[model], float)]
    for s in extra_starts:
        starts.append(np.clip(np.asarray(s, float), lo, hi))
    if model == "mcmc":
        for eps_c in (PRIOR_EPS[0], PRIOR_EPS[1]):
            for zt_c in (PRIOR_ZTRANS[0], PRIOR_ZTRANS[1]):
                starts.append(np.array([START[model][0], START[model][1],
                                        eps_c, zt_c]))
    best_val, best_x = np.inf, starts[0]
    for x0 in starts:
        res = minimize(f, x0, method="Powell", bounds=bounds,
                       options={"maxiter": 20000, "xtol": 1e-8,
                                "ftol": 1e-10})
        if float(res.fun) < best_val:
            best_val, best_x = float(res.fun), np.asarray(res.x)
    pol = minimize(f, best_x, method="L-BFGS-B", bounds=bounds)
    if float(pol.fun) < best_val:
        best_val, best_x = float(pol.fun), np.asarray(pol.x)
    span = hi - lo
    at_boundary = bool(np.any((best_x - lo < 1e-6 * span)
                              | (hi - best_x < 1e-6 * span)))
    return {"chi2": best_val, "theta": best_x.tolist(),
            "blocks": chi2_blocks(best_x, model, data, blocks),
            "at_boundary": at_boundary}


def summarize(flat: np.ndarray, names) -> dict:
    p = np.percentile(flat, [2.5, 16, 50, 84, 97.5], axis=0)
    return {n: {"p025": float(p[0, i]), "p16": float(p[1, i]),
                "p50": float(p[2, i]), "p84": float(p[3, i]),
                "p975": float(p[4, i]),
                "sd": float(np.std(flat[:, i], ddof=1))}
            for i, n in enumerate(names)}


def criteria(chi2_min: float, model: str, n: int) -> dict:
    return information_criteria(K_PARAMS[model], n, -0.5 * chi2_min)


# ---------------------------------------------------------------------
# Clasificación del desenlace (regla congelada en la preinscripción)
# ---------------------------------------------------------------------

def classify_outcome(bench_delta_sigma: float, chi2nu_lcdm: float,
                     dBIC_mcmc_minus_lcdm: float, eps_sd: float,
                     eps_ci95_contains_zero: bool, rules: dict) -> str:
    """Orden literal preinscrito: C (tensión) → B (preferencia) → A
    (aburrido) → INDETERMINADO."""
    if (abs(bench_delta_sigma) > rules["C_bench_sigma"]
            or chi2nu_lcdm > rules["C_chi2nu_max"]):
        return "C"
    if (-dBIC_mcmc_minus_lcdm > rules["B_dBIC_pro_mcmc"]
            and not eps_ci95_contains_zero):
        return "B"
    if (dBIC_mcmc_minus_lcdm > rules["A_dBIC_min"]
            and eps_sd >= rules["A_eps_sd_min"]
            and eps_ci95_contains_zero
            and abs(bench_delta_sigma) <= rules["A_bench_sigma"]):
        return "A"
    return "INDETERMINADO"


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def load_prereg() -> dict:
    if not PREREG.exists():
        raise RuntimeError("PREREGISTRATION_MISSING: la primera aplicación "
                           "a Dovekie real exige preinscripción congelada")
    return json.loads(PREREG.read_text(encoding="utf-8"))
