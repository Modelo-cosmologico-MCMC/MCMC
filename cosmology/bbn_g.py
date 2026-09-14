"""Contraste de los Residuos vía BBN (frente 6): G_cosmo/G_N libre sobre
las abundancias primordiales Y_P y D/H, con prior Planck en ω_b.

EL OBSERVABLE. El Sello de Newton (9.3) y el frente 3 (#18/#20) dan
G_cosmo/G_local = (2ξ − α_a)/(3λ_K − 1); con ξ = 1 (GW170817) y α_a = 0
declarado, δ_G ≡ G_cosmo/G_N − 1 = 2/(3λ_K − 1) − 1 ≈ −(3/2)ε_K = −1.8 %
(ε_K = 0.012). BBN solo ve G en la tasa de expansión H² = 8πG_cosmo ρ/3
(la física nuclear y débil no ve G), de modo que el mismo δ_G del fondo
es el que desplaza Y_P y D/H.

EL MODELO DE RESPUESTA (validation/bbn_response.py, PRyMordial @ commit
fijado, sin datos) se fija como tabla en results/2026-09-14_bbn_g/
response_prymordial.json y se lee aquí (fallo cerrado si falta):

    ln X = ln X₀ + a_G ln(1+δ_G) + a_N ΔN + b_N ΔN² + a_ω ln(ω_b/ω_b⁰)
           + a_τ ln(τ_n/τ_n⁰),          X ∈ {Y_P, D/H}.

La verosimilitud es gaussiana en ln X con los errores de medida y
teóricos en cuadratura; ω_b y τ_n (priors gaussianos) se marginalizan de
forma EXACTA (respuesta lineal en sus logaritmos) sumando a la
covarianza los términos a_ω a_ωᵀ σ²_{ln ω} + a_τ a_τᵀ σ²_{ln τ}.

REGLA PREINSCRITA (brazo principal: δ_G libre, N_eff = 3.044):
C = δ_G^model ∉ CI95 (exclusión) → B = 0 ∉ CI95 (identificación:
sospecha de error primero) → A = ambos dentro (banda compatible sin
identificación). El signo del central NUNCA se lee como señal (E13).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from cosmology.mu_eta_atlas import G_cosmo_over_G_local
from mcmc_ontology import constants as C
from mcmc_ontology.data_registry import require_available

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-09-14_bbn_g"
RESPONSE_TABLE = OUTDIR / "response_prymordial.json"
PREREG = OUTDIR / "preregistration.json"

RULES = {
    "sigma_th_YP": 0.0003,            # error teórico de Y_P (tasas débiles, τ_n)
    "sigma_th_DH_min": 0.05e-5,       # mínimo del sistemático nuclear en D/H
    "ci_level": 0.95,
    "delta_G_prior": [-0.30, 0.30],
    "delta_G_grid_n": 6001,
    "dNeff_prior": [-1.5, 1.5],
    "dNeff_grid_n": 301,
    "Neff_SM": 3.044,
}

__all__ = ["OUTDIR", "RESPONSE_TABLE", "RULES", "classify_outcome", "delta_G_model",
           "load_data", "load_prereg", "log_prob_grid", "posterior_1d",
           "posterior_2d", "predict", "response_constants", "sm_pulls",
           "summarize_1d"]


# --------------------------------------------------------------------
# Predicción del modelo y tabla de respuesta
# --------------------------------------------------------------------

def delta_G_model(eps_K: float = C.EPSILON_K, alpha_a: float = 0.0,
                  xi: float = 1.0) -> float:
    """δ_G = G_cosmo/G_local − 1 = (2ξ − α_a)/(3λ_K − 1) − 1 con λ_K = 1 + ε_K."""
    return G_cosmo_over_G_local(1.0 + eps_K, xi, alpha_a) - 1.0


@lru_cache(maxsize=1)
def response_constants() -> dict:
    """Constantes de la tabla PRyMordial (fallo cerrado si falta)."""
    if not RESPONSE_TABLE.exists():
        raise FileNotFoundError(
            f"tabla de respuesta ausente: {RESPONSE_TABLE} — generar con "
            "validation/bbn_response.py (requiere external/PRyMordial)")
    doc = json.loads(RESPONSE_TABLE.read_text(encoding="utf-8"))
    ex = doc["exponents"]
    sm = doc["sm_prediction"]
    nuc = doc["nuclear_rates_systematic"]
    return {
        "YP0": sm["YP"], "DH0": sm["DH"], "Neff_SM_prym": sm["Neff"],
        "omega_b0": doc["fiducial"]["omega_b"], "tau_n0": doc["fiducial"]["tau_n"],
        "a_G_YP": ex["a_G_YP"], "a_G_DH": ex["a_G_DH"],
        "a_N_YP": ex["a_N_YP"], "b_N_YP": ex["b_N_YP"],
        "a_N_DH": ex["a_N_DH"], "b_N_DH": ex["b_N_DH"],
        "a_w_YP": ex["a_w_YP"], "a_w_DH": ex["a_w_DH"],
        "a_tau_YP": ex["a_tau_YP"], "a_tau_DH": ex["a_tau_DH"],
        "max_rel_resid": max(v for k, v in ex.items() if k.endswith("resid")),
        "nuclear_delta_DH": nuc["delta_DH"], "nuclear_delta_YP": nuc["delta_YP"],
        "prymordial_commit": doc["prymordial"]["commit"],
    }


def predict(delta_G, delta_Neff=0.0, omega_b=None, tau_n=None, resp=None):
    """(Y_P, D/H) del modelo de respuesta; acepta arrays en delta_G/ΔN."""
    r = resp or response_constants()
    omega_b = r["omega_b0"] if omega_b is None else omega_b
    tau_n = r["tau_n0"] if tau_n is None else tau_n
    lg = np.log1p(np.asarray(delta_G, float))
    dN = np.asarray(delta_Neff, float)
    lw = np.log(omega_b / r["omega_b0"])
    lt = np.log(tau_n / r["tau_n0"])
    lnY = (np.log(r["YP0"]) + r["a_G_YP"] * lg + r["a_N_YP"] * dN
           + r["b_N_YP"] * dN ** 2 + r["a_w_YP"] * lw + r["a_tau_YP"] * lt)
    lnD = (np.log(r["DH0"]) + r["a_G_DH"] * lg + r["a_N_DH"] * dN
           + r["b_N_DH"] * dN ** 2 + r["a_w_DH"] * lw + r["a_tau_DH"] * lt)
    return np.exp(lnY), np.exp(lnD)


# --------------------------------------------------------------------
# Datos (tras el registro) y preinscripción
# --------------------------------------------------------------------

def load_data() -> dict:
    """Lee el dataset registrado bbn_abundances (require_available: manifest
    AVAILABLE, esquema confirmado, sha256 re-verificados). Devuelve los
    valores y el aviso de procedencia, que el artefacto debe propagar."""
    raw = require_available("bbn_abundances")
    doc = json.loads((raw / "abundances_2026.json").read_text(encoding="utf-8"))
    vals = doc["values"]
    return {"values": vals,
            "official_bytes_verified": bool(doc.get("official_bytes_verified", False)),
            "provenance_caveat": doc.get("provenance_caveat", ""),
            "manifest_sha256": _sha256(ROOT / "data" / "manifests" / "bbn_abundances.json")}


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise FileNotFoundError("FALLO CERRADO: falta la preinscripción BBN-G "
                                f"({PREREG})")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha256(PREREG)


# --------------------------------------------------------------------
# Verosimilitud con marginalización exacta de ω_b y τ_n
# --------------------------------------------------------------------

def _sigma_th_DH(resp: dict, rules: dict) -> float:
    return max(rules["sigma_th_DH_min"], abs(resp["nuclear_delta_DH"]))


def _covariance(data: dict, resp: dict, rules: dict, use_YP: bool, use_DH: bool,
                yp_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Residuos en ln X y covarianza total (medida + teoría + nuisance)."""
    v = data["values"]
    rows, obs, meas_var, aw, at = [], [], [], [], []
    if use_YP:
        y = v[yp_key]
        rows.append("YP")
        obs.append(np.log(y["value"]))
        meas_var.append((y["sigma"] / y["value"]) ** 2
                        + (rules["sigma_th_YP"] / resp["YP0"]) ** 2)
        aw.append(resp["a_w_YP"]); at.append(resp["a_tau_YP"])
    if use_DH:
        d = v["DH_cooke_2018"]
        rows.append("DH")
        obs.append(np.log(d["value"]))
        meas_var.append((d["sigma"] / d["value"]) ** 2
                        + (_sigma_th_DH(resp, rules) / resp["DH0"]) ** 2)
        aw.append(resp["a_w_DH"]); at.append(resp["a_tau_DH"])
    wb = v["omega_b_planck_2018"]
    tn = v["tau_n_pdg_2023"]
    s_lw = wb["sigma"] / wb["value"]
    s_lt = tn["sigma"] / tn["value"]
    aw, at = np.array(aw), np.array(at)
    cov = np.diag(meas_var) + np.outer(aw, aw) * s_lw ** 2 + np.outer(at, at) * s_lt ** 2
    return np.array(obs), cov, rows


def log_prob_grid(dG, dN, data: dict, resp: dict | None = None,
                  rules: dict = RULES, use_YP: bool = True, use_DH: bool = True,
                  yp_key: str = "Y_P_empress_xv") -> np.ndarray:
    """ln L(δ_G, ΔN) (arrays broadcastables) con ω_b, τ_n marginalizados
    exactamente y evaluados en los centros de sus priors."""
    resp = resp or response_constants()
    obs, cov, rows = _covariance(data, resp, rules, use_YP, use_DH, yp_key)
    wb = data["values"]["omega_b_planck_2018"]["value"]
    tn = data["values"]["tau_n_pdg_2023"]["value"]
    Y, D = predict(dG, dN, wb, tn, resp)
    preds = []
    if "YP" in rows:
        preds.append(np.log(Y))
    if "DH" in rows:
        preds.append(np.log(D))
    pred = np.stack(np.broadcast_arrays(*preds), axis=-1)
    r = pred - obs
    icov = np.linalg.inv(cov)
    chi2 = np.einsum("...i,ij,...j->...", r, icov, r)
    return -0.5 * chi2


def posterior_1d(data: dict, resp: dict | None = None, rules: dict = RULES,
                 dNeff: float = 0.0, **kw) -> dict:
    """Posterior de δ_G en malla con prior plano (brazo principal si
    dNeff = 0 y use_YP = use_DH = True)."""
    lo, hi = rules["delta_G_prior"]
    grid = np.linspace(lo, hi, int(rules["delta_G_grid_n"]))
    lp = log_prob_grid(grid, dNeff, data, resp, rules, **kw)
    return summarize_1d(grid, lp, rules)


def summarize_1d(grid: np.ndarray, logp: np.ndarray, rules: dict = RULES) -> dict:
    p = np.exp(logp - logp.max())
    p /= np.trapezoid(p, grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    q = lambda x: float(np.interp(x, cdf, grid))  # noqa: E731
    a = 0.5 * (1 - rules["ci_level"])
    return {"grid_lo": float(grid[0]), "grid_hi": float(grid[-1]),
            "map": float(grid[np.argmax(p)]), "p50": q(0.5),
            "p16": q(0.16), "p84": q(0.84), "sd": float(np.sqrt(
                np.trapezoid(p * (grid - np.trapezoid(p * grid, grid)) ** 2, grid))),
            "ci95": [q(a), q(1 - a)],
            "cdf_at_zero": float(np.interp(0.0, grid, cdf)),
            "prior_edge_hit": bool(p[0] > 1e-3 * p.max() or p[-1] > 1e-3 * p.max())}


def posterior_2d(data: dict, resp: dict | None = None, rules: dict = RULES,
                 **kw) -> dict:
    """Brazo de degeneración: (δ_G, ΔN_eff) con priors planos; marginal de
    δ_G y dirección degenerada (pendiente ΔN(δ_G) de la cresta)."""
    lo, hi = rules["delta_G_prior"]
    nlo, nhi = rules["dNeff_prior"]
    g = np.linspace(lo, hi, 601)
    n = np.linspace(nlo, nhi, int(rules["dNeff_grid_n"]))
    G, N = np.meshgrid(g, n, indexing="ij")
    lp = log_prob_grid(G, N, data, resp, rules, **kw)
    p = np.exp(lp - lp.max())
    marg_g = np.trapezoid(p, n, axis=1)
    marg_n = np.trapezoid(p, g, axis=0)
    ridge = n[np.argmax(p, axis=1)]
    sel = marg_g > 0.05 * marg_g.max()
    slope = float(np.polyfit(g[sel], ridge[sel], 1)[0]) if sel.sum() > 5 else float("nan")
    return {"marginal_delta_G": summarize_1d(g, np.log(marg_g + 1e-300), rules),
            "marginal_dNeff": summarize_1d(n, np.log(marg_n + 1e-300), rules),
            "ridge_slope_dNeff_per_delta_G": slope}


def sm_pulls(data: dict, resp: dict | None = None, rules: dict = RULES,
             yp_key: str = "Y_P_empress_xv") -> dict:
    """Brazo 0: pulls de Y_P y D/H respecto del SM (δ_G = 0, N_eff SM)."""
    resp = resp or response_constants()
    v = data["values"]
    Y, D = predict(0.0, 0.0, v["omega_b_planck_2018"]["value"],
                   v["tau_n_pdg_2023"]["value"], resp)
    sY = np.hypot(v[yp_key]["sigma"], rules["sigma_th_YP"])
    sD = np.hypot(v["DH_cooke_2018"]["sigma"], _sigma_th_DH(resp, rules))
    obs, cov, _ = _covariance(data, resp, rules, True, True, yp_key)
    pred = np.array([np.log(Y), np.log(D)])
    r = obs - pred
    return {"YP_sm": float(Y), "DH_sm": float(D),
            "pull_YP": float((v[yp_key]["value"] - Y) / sY),
            "pull_DH": float((v["DH_cooke_2018"]["value"] - D) / sD),
            "chi2_sm_marginalized": float(r @ np.linalg.solve(cov, r)),
            "sigma_th_DH_used": float(_sigma_th_DH(resp, rules))}


# --------------------------------------------------------------------
# Regla de desenlace
# --------------------------------------------------------------------

def classify_outcome(ci95, delta_model: float) -> str:
    """Orden C → B → A (exhaustivo)."""
    lo, hi = ci95
    if not (lo <= delta_model <= hi):
        return "C"
    if not (lo <= 0.0 <= hi):
        return "B"
    return "A"
