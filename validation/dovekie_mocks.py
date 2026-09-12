"""Validación por mocks del pipeline SN Dovekie (PR #15) — maquinaria.

NIVEL DE LA VALIDACIÓN (declarado en la preinscripción): nuestro
pipeline entra en el nivel HD + covarianza (4_DISTANCES_COVMAT). Los
25 mocks fotométricos de DES (1_SIMULATIONS @ c9a4fcaf) validan las
etapas DES aguas arriba (SALT3 + BBC), que no reimplementamos; la
validación del PIPELINE PROPIO usa 25 realizaciones sintéticas a nivel
HD: μ_mock = μ_model(z; cosmología inyectada de los mocks DES) +
ΔM_m + ruido ~ N(0, C), con los z del HD real (el diseño), N = 1820 y
la covarianza oficial. La columna MU real NO se lee en ninguna parte
de este módulo (barrera: load_dovekie_design no la devuelve).

Cosmología inyectada (README de 1_SIMULATIONS @ c9a4fcaf):
H0 = 70.0, Ω_m = 0.315, Ω_Λ = 0.685, w0 = −1, wa = 0 — en nuestra
parametrización: (Ω_m = 0.315, ε_Λ = 0, z_trans irrelevante con ε=0).

Cuatro puertas (tolerancias y enteros congelados en
results/2026-09-12_dovekie_mocks/preregistration.json):
  1a. equivalencia de la fórmula χ² (A y B vs cov_log_likelihood
      oficial ejecutada desde los bytes del release) sobre K
      cosmologías × {STATONLY, STAT+SYS};
  1b. equivalencia de integradores de distancia (A O(h⁴) vs B quad);
  2.  parameter recovery de Ω_m (brazo ΛCDM): |media de pulls| acotada
      en múltiplos de SEM = 1/√25;
  3.  coverage 68/95 % con enteros binomiales exactos para N = 25;
  4.  ausencia de falsa preferencia por la extensión (ΔAIC/ΔBIC) y
      ε_Λ consistente con cero (CI y mediana).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from cosmology.dovekie_sn import (
    N_SN_EXPECTED,
    chi2_sn_marginalized,
    load_dovekie_design,
    load_dovekie_inv_cov,
    mu_model,
)
from mcmc_ontology.data_registry import require_available

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-09-12_dovekie_mocks"

# --- constantes preinscritas (el generador de la preinscripción las
# --- serializa; el candado tests/test_dovekie_lock.py las fija) ------
INJECTED = {"H0": 70.0, "Omega_m": 0.315, "eps": 0.0}
N_MOCKS = 25
MOCK_SEED = 20260912
COV_KIND_PRODUCTION = "STAT+SYS"
# priors del brazo MCMC (los del Apéndice F, idénticos a #12)
PRIOR_OMEGA_M = (0.05, 0.7)
PRIOR_EPS = (-0.05, 0.10)
PRIOR_EPS_GAUSS = (0.012, 0.05)
PRIOR_ZTRANS = (1.0, 20.0)
# K cosmologías controladas de la puerta 1 (Ω_m, ε, z_trans)
K_COSMOLOGIES = (
    (0.10, 0.0, 8.9), (0.315, 0.0, 8.9), (0.50, 0.0, 8.9),
    (0.315, -0.05, 8.9), (0.315, 0.012, 8.9), (0.315, 0.10, 8.9),
    (0.315, 0.012, 1.0), (0.315, 0.012, 20.0),
)


# ---------------------------------------------------------------------
# La fórmula oficial, ejecutada desde los bytes del release
# ---------------------------------------------------------------------

def official_cov_log_likelihood():
    """Extrae y ejecuta cov_log_likelihood del script oficial del
    release (bytes verificados por sha256 vía require_available).

    LIMITACIÓN DECLARADA: el script oficial es un módulo cosmosis; el
    harness cosmosis no es ejecutable aquí. La parte que define la
    likelihood (la función pura cov_log_likelihood, ec. A9-A12) sí lo
    es, y es EXACTAMENTE la que se ejecuta — extraída por AST del
    fichero oficial, no copiada a mano."""
    raw = require_available("des_dovekie")
    src = (raw / "DES-Dovekie-SN_Likelihood.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "cov_log_likelihood"]
    if len(fn) != 1:
        raise RuntimeError(
            "cov_log_likelihood no encontrada en el script oficial")
    mod = ast.Module(body=fn, type_ignores=[])
    ns: dict = {"np": np}
    exec(compile(mod, "<official-dovekie-likelihood>", "exec"), ns)
    return ns["cov_log_likelihood"]


# ---------------------------------------------------------------------
# Generador de mocks (nivel HD; jamás lee MU real)
# ---------------------------------------------------------------------

def generate_mocks(n_mocks: int = N_MOCKS, seed: int = MOCK_SEED,
                   cov_kind: str = COV_KIND_PRODUCTION) -> dict:
    """25 realizaciones μ_mock = μ_model(z; inyectada) + ΔM_m + η_m,
    η_m ~ N(0, C) con C = W⁻¹ (Cholesky de W y resolución triangular,
    sin invertir explícitamente), ΔM_m ~ N(0, 1 mag) — el offset
    ejercita activamente la marginalización de M."""
    design = load_dovekie_design()
    if design["n_sn"] != N_SN_EXPECTED:
        raise RuntimeError(
            f"conteo de SNe {design['n_sn']} != {N_SN_EXPECTED}")
    W = load_dovekie_inv_cov(cov_kind)
    L = np.linalg.cholesky(W)
    mu_true = mu_model(design["zHD"], design["zHEL"],
                       INJECTED["Omega_m"], eps=INJECTED["eps"])
    rng = np.random.default_rng(seed)
    mocks = np.empty((n_mocks, design["n_sn"]))
    offsets = rng.normal(0.0, 1.0, size=n_mocks)
    for m in range(n_mocks):
        xi = rng.standard_normal(design["n_sn"])
        # cov(η) = L⁻ᵀ·I·L⁻¹ = (L·Lᵀ)⁻¹ = W⁻¹ = C
        eta = np.linalg.solve(L.T, xi)
        mocks[m] = mu_true + offsets[m] + eta
    return {"mu_mocks": mocks, "M_offsets": offsets,
            "zHD": design["zHD"], "zHEL": design["zHEL"],
            "W": W, "mu_true": mu_true, "seed": seed,
            "cov_kind": cov_kind}


# ---------------------------------------------------------------------
# Brazo ΛCDM: posterior 1D exacto en malla (determinista)
# ---------------------------------------------------------------------

OMEGA_GRID = np.linspace(PRIOR_OMEGA_M[0], PRIOR_OMEGA_M[1], 2601)


def lcdm_grid_mu(zHD: np.ndarray, zHEL: np.ndarray) -> np.ndarray:
    """Matriz (n_grid, n_sn) de μ_model ΛCDM sobre OMEGA_GRID."""
    return np.stack([mu_model(zHD, zHEL, om) for om in OMEGA_GRID])


def lcdm_posterior_percentiles(mu_grid: np.ndarray, W: np.ndarray,
                               mu_data: np.ndarray,
                               qs=(0.16, 0.50, 0.84, 0.025, 0.975)):
    """Percentiles del posterior 1D p(Ω_m) ∝ exp(−χ̃²/2) (prior plano)
    sobre la malla — determinista, sin sampler. Devuelve además el
    χ̃²_min sobre la malla (con refinado parabólico local)."""
    # χ̃²(p) por formas cuadráticas precomputables:
    #   Δ = μ_p − d;  ΔᵀWΔ = μᵀWμ − 2·dᵀWμ + dᵀWd;  B = 1ᵀWΔ
    WM = mu_grid @ W                       # (P, N)
    quad_mm = np.einsum("pn,pn->p", WM, mu_grid)
    S = float(np.sum(W))
    one_Wm = np.sum(WM, axis=1)            # 1ᵀWμ_p
    d = np.asarray(mu_data, float)
    Wd = W @ d
    dWd = float(d @ Wd)
    one_Wd = float(np.sum(Wd))
    cross = WM @ d                         # μ_pᵀWd
    chit2 = quad_mm - 2.0 * cross + dWd
    B = one_Wm - one_Wd
    chi2 = chit2 - B * B / S + np.log(S / (2.0 * np.pi))
    # percentiles del posterior normalizado
    ln_p = -0.5 * (chi2 - chi2.min())
    p = np.exp(ln_p)
    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (p[1:] + p[:-1]) * np.diff(OMEGA_GRID))])
    cdf /= cdf[-1]
    pct = {q: float(np.interp(q, cdf, OMEGA_GRID)) for q in qs}
    # χ²_min con refinado parabólico en torno al mínimo de malla
    k = int(np.argmin(chi2))
    if 0 < k < len(chi2) - 1:
        y0, y1, y2 = chi2[k - 1], chi2[k], chi2[k + 1]
        denom = (y0 - 2.0 * y1 + y2)
        chi2_min = float(y1 - 0.125 * (y0 - y2) ** 2 / denom) \
            if denom > 0 else float(y1)
    else:
        chi2_min = float(chi2[k])
    return pct, chi2_min, float(OMEGA_GRID[k])


# ---------------------------------------------------------------------
# Brazo MCMC (Ω_m, ε, z_trans): emcee sembrado + χ²_min acotado
# ---------------------------------------------------------------------

def log_prob_mcmc_sn(theta, zHD, zHEL, W, mu_data) -> float:
    om, eps, z_t = theta
    if not (PRIOR_OMEGA_M[0] < om < PRIOR_OMEGA_M[1]):
        return -np.inf
    if not (PRIOR_EPS[0] < eps < PRIOR_EPS[1]):
        return -np.inf
    if not (PRIOR_ZTRANS[0] < z_t < PRIOR_ZTRANS[1]):
        return -np.inf
    lp = -0.5 * ((eps - PRIOR_EPS_GAUSS[0]) / PRIOR_EPS_GAUSS[1]) ** 2
    m = mu_model(zHD, zHEL, om, eps=eps, z_trans=z_t)
    return lp - 0.5 * chi2_sn_marginalized(m, mu_data, W)


def run_mcmc_arm(zHD, zHEL, W, mu_data, nwalkers: int = 32,
                 nsteps: int = 800, seed: int = 42) -> dict:
    """emcee con la MISMA convención de siembra que la producción #12
    (np.random.seed fija el stream interno del sampler)."""
    import emcee
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    center = np.array([INJECTED["Omega_m"], PRIOR_EPS_GAUSS[0], 8.9])
    scale = np.array([0.02, 0.01, 1.0])
    p0 = center + scale * rng.normal(size=(nwalkers, 3))
    p0[:, 0] = np.clip(p0[:, 0], 0.06, 0.69)
    p0[:, 1] = np.clip(p0[:, 1], -0.049, 0.099)
    p0[:, 2] = np.clip(p0[:, 2], 1.01, 19.99)
    sampler = emcee.EnsembleSampler(
        nwalkers, 3, log_prob_mcmc_sn, args=(zHD, zHEL, W, mu_data))
    sampler.run_mcmc(p0, nsteps, progress=False)
    flat = sampler.get_chain(discard=nsteps // 2, thin=4, flat=True)
    return {"flat": flat,
            "acceptance": float(np.mean(sampler.acceptance_fraction))}


def chi2_min_mcmc_arm(zHD, zHEL, W, mu_data,
                      om_hat: float) -> float:
    """χ̃²_min del brazo MCMC sobre el CIERRE del soporte del prior.

    La estructura del problema hace el mínimo separable en la
    práctica: con los z del HD (z ≤ 1.13) y z_trans ∈ [1, 20], la
    transición normalizada hoy deja |Δμ| ≲ 1e-3·ε mag, así que el
    mínimo en (ε, z_trans) vive en la frontera con ganancia mínima.
    Se barre una malla predeclarada (ε × z_trans en las esquinas y el
    centro del soporte) y se refina Ω_m 1D por Brent acotado en cada
    punto — sin optimización libre fuera del soporte."""
    best = np.inf
    for eps_c in (PRIOR_EPS[0], 0.0, PRIOR_EPS_GAUSS[0], PRIOR_EPS[1]):
        for zt_c in (PRIOR_ZTRANS[0], 8.9, PRIOR_ZTRANS[1]):
            def f(om, e=eps_c, zt=zt_c):
                m = mu_model(zHD, zHEL, om, eps=e, z_trans=zt)
                return chi2_sn_marginalized(m, mu_data, W)
            res = minimize_scalar(
                f, bounds=(max(PRIOR_OMEGA_M[0], om_hat - 0.1),
                           min(PRIOR_OMEGA_M[1], om_hat + 0.1)),
                method="bounded",
                options={"xatol": 1e-6})
            if float(res.fun) < best:
                best = float(res.fun)
    return best
