"""Vector teórico DESI DR2 y ajustes de fondo (6A.3-6A.5).

CONTRATO r_d (decisión preinscrita de esta ronda): el MCMC NO deriva
la física pre-recombinación, así que r_d entra como CALIBRACIÓN COMÚN
a ΛCDM y MCMC — un solo parámetro H0·r_d [km/s] idéntico en ambos
modelos. Ni «ΛCDM con r_d derivado vs MCMC con r_d libre» ni lo
contrario: DESI evalúa la FORMA de la expansión E(z), sin atribuir al
MCMC física que aún no contiene.

Observables (adimensionales, orden EXACTO del release — pinneado por
test contra el fichero ingerido):
    DH/rd = c/(E(z)·H0rd)
    DM/rd = (c/H0rd)·∫₀ᶻ dz'/E(z')
    DV/rd = [z·(DM/rd)²·(DH/rd)]^(1/3)

E(z) = H(z)/H0 viene del fondo corregido (cosmology/background.H_of_z
con H0 = 1: clausura plana por llamada y transición normalizada hoy —
el invariante H(0) = H0 de la corrección a29e89f, FUSIONADA en esta
rama y vigilada por tests/test_physical_invariants.py; la revisión
adversarial 6A encontró que la primera versión de esta ronda afirmaba
esa corrección sin contenerla). ΛCDM ≡ ε = 0 exacto (Prop. A.1).

Parámetros:
    ΛCDM:  θ = (Ω_m, H0rd)                          k = 2
    MCMC:  θ = (Ω_m, H0rd, ε_Λ, z_trans)            k = 4
Priors (los del Apéndice F usados en los ajustes de producción):
    Ω_m ~ U(0.05, 0.7); H0rd ~ U(6000, 16000) km/s;
    ε ~ N(0.012, 0.05²) truncada a (−0.05, 0.10); z_trans ~ U(1, 20).
"""

from __future__ import annotations

import numpy as np

from cosmology.background import H_of_z
from cosmology.desi_bao import chi2_bao, load_desi_dr2_all

C_KMS = 299792.458
Z_GRID = np.linspace(0.0, 2.4, 4001)     # malla fija del integrador


def E_of_z(z, Omega_m: float, eps: float = 0.0,
           z_trans: float = 8.9, dz: float = 1.5):
    """E(z) = H(z)/H0 del fondo corregido (H_of_z con H0 = 1)."""
    return np.asarray(H_of_z(z, H0=1.0, Omega_m=Omega_m, eps=eps,
                             z_trans=z_trans, dz=dz))


def predict_desi_dr2_vector(Omega_m: float, H0rd_kms: float,
                            eps: float = 0.0, z_trans: float = 8.9,
                            dz: float = 1.5,
                            data=None) -> np.ndarray:
    """Las 13 componentes en el ORDEN EXACTO del release ingerido.

    data: (z, quant, val, bins, C) de load_desi_dr2_all (se puede
    pasar precargado para no re-verificar checksums en cada punto del
    sampler; la verificación ocurre al cargar)."""
    if data is None:
        data = load_desi_dr2_all()
    z_eff, quant, _, _, _ = data
    E = E_of_z(Z_GRID, Omega_m, eps, z_trans, dz)
    invE_int = np.concatenate(
        ([0.0], np.cumsum(0.5 * (1.0 / E[1:] + 1.0 / E[:-1])
                          * np.diff(Z_GRID))))
    out = np.empty(len(z_eff))
    for i, (zi, q) in enumerate(zip(z_eff, quant)):
        Ei = float(E_of_z(zi, Omega_m, eps, z_trans, dz))
        DH = C_KMS / (Ei * H0rd_kms)
        DM = C_KMS / H0rd_kms * float(np.interp(zi, Z_GRID, invE_int))
        if q == "DH_over_rs":
            out[i] = DH
        elif q == "DM_over_rs":
            out[i] = DM
        elif q == "DV_over_rs":
            out[i] = (zi * DM ** 2 * DH) ** (1.0 / 3.0)
        else:
            raise ValueError(q)
    return out


# ---------- priors y posterior (mismas convenciones ambos modelos) ----------

def log_prior_common(Omega_m: float, H0rd: float) -> float:
    if not (0.05 < Omega_m < 0.7) or not (6000.0 < H0rd < 16000.0):
        return -np.inf
    return 0.0


def log_prob_lcdm(theta, data, idx=None) -> float:
    Om, H0rd = theta
    lp = log_prior_common(Om, H0rd)
    if not np.isfinite(lp):
        return -np.inf
    m = predict_desi_dr2_vector(Om, H0rd, data=data)
    _, _, val, _, C = data
    return lp - 0.5 * chi2_bao(m, val, C, idx=idx)


def log_prob_mcmc(theta, data, idx=None) -> float:
    Om, H0rd, eps, z_t = theta
    lp = log_prior_common(Om, H0rd)
    if not (-0.05 < eps < 0.10) or not (1.0 < z_t < 20.0):
        return -np.inf
    if not np.isfinite(lp):
        return -np.inf
    lp += -0.5 * ((eps - 0.012) / 0.05) ** 2      # Apéndice F
    m = predict_desi_dr2_vector(Om, H0rd, eps=eps, z_trans=z_t,
                                data=data)
    _, _, val, _, C = data
    return lp - 0.5 * chi2_bao(m, val, C, idx=idx)


def run_fit(log_prob, p0_center, ndim, data, idx=None,
            nwalkers: int = 32, nsteps: int = 3000, seed: int = 42):
    """emcee con semilla REAL (la misma maquinaria para ambos modelos
    y todas las configuraciones LOO — convenciones idénticas).

    emcee 3 copia su RandomState interno del estado GLOBAL legacy de
    numpy al construir el sampler; sembrar solo la bola inicial NO
    reproduce las cadenas (hallazgo confirmado de la revisión
    adversarial 6A). np.random.seed(seed) fija el stream completo:
    cadenas bit-idénticas entre procesos con la misma semilla."""
    import emcee
    np.random.seed(seed)                 # stream interno del sampler
    rng = np.random.default_rng(seed)    # bola inicial
    p0 = np.asarray(p0_center) + 1e-3 * rng.normal(
        size=(nwalkers, ndim)) * np.asarray(p0_center)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    args=(data, idx))
    sampler.run_mcmc(p0, nsteps, progress=False)
    burn = nsteps // 2
    flat = sampler.get_chain(discard=burn, thin=4, flat=True)
    lp = sampler.get_log_prob(discard=burn, thin=4, flat=True)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        converged = bool(np.all(nsteps > 50 * np.asarray(tau)))
    except Exception:
        tau, converged = None, False
    return {"flat": flat, "logp": lp,
            "best": flat[int(np.argmax(lp))],
            "acceptance": float(np.mean(sampler.acceptance_fraction)),
            "tau": None if tau is None else np.asarray(tau).tolist(),
            "converged": converged}


def chi2_at(theta, model: str, data, idx=None) -> float:
    """χ² (sin priors) en θ — para χ²_min, AIC y BIC."""
    _, _, val, _, C = data
    if model == "lcdm":
        m = predict_desi_dr2_vector(theta[0], theta[1], data=data)
    else:
        m = predict_desi_dr2_vector(theta[0], theta[1], eps=theta[2],
                                    z_trans=theta[3], data=data)
    return chi2_bao(m, val, C, idx=idx)


# Cierre del soporte de los priors declarados arriba: el dominio sobre
# el que se define χ²_min de cada modelo.
PRIOR_SUPPORT = {
    "lcdm": ((0.05, 0.7), (6000.0, 16000.0)),
    "mcmc": ((0.05, 0.7), (6000.0, 16000.0), (-0.05, 0.10), (1.0, 20.0)),
}
_START_CENTER = {"lcdm": (0.30, 10200.0),
                 "mcmc": (0.30, 10200.0, 0.012, 8.9)}


def constrained_chi2_min(model: str, data, idx=None,
                         extra_starts=()) -> dict:
    """χ²_min del modelo sobre el CIERRE del soporte del prior
    (multistart acotado + pulido). El argmin puede caer en la frontera
    del soporte — se publica con la bandera at_boundary, no se oculta.

    Corrige el hallazgo de la revisión adversarial 6A: el refinado
    Nelder-Mead sin cotas escapaba del soporte del prior (ε ~ +7.5) y
    los χ²_M publicados no eran mínimos del modelo."""
    from scipy.optimize import minimize
    bounds = PRIOR_SUPPORT[model]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def f(t):
        return chi2_at(t, model, data, idx=idx)

    starts = [np.asarray(_START_CENTER[model], float)]
    for s in extra_starts:
        starts.append(np.clip(np.asarray(s, float), lo, hi))
    if model == "mcmc":
        # esquinas del soporte de (ε, z_trans), con el (Ω_m, H0rd) del
        # centro y el de cada start extra (el mínimo puede vivir ahí)
        heads = [np.asarray(_START_CENTER[model][:2], float)] + [
            np.clip(np.asarray(s, float)[:2], lo[:2], hi[:2])
            for s in extra_starts]
        for head in heads:
            for eps_c in (-0.05, 0.10):
                for zt_c in (1.0, 20.0):
                    starts.append(np.array([head[0], head[1],
                                            eps_c, zt_c]))
    best_val, best_x = np.inf, starts[0]
    for x0 in starts:
        res = minimize(f, x0, method="Powell", bounds=bounds,
                       options={"maxiter": 20000, "xtol": 1e-10,
                                "ftol": 1e-12})
        if float(res.fun) < best_val:
            best_val, best_x = float(res.fun), np.asarray(res.x)
    pol = minimize(f, best_x, method="L-BFGS-B", bounds=bounds)
    if float(pol.fun) < best_val:
        best_val, best_x = float(pol.fun), np.asarray(pol.x)
    span = hi - lo
    at_boundary = bool(np.any((best_x - lo < 1e-6 * span)
                              | (hi - best_x < 1e-6 * span)))
    return {"chi2": best_val, "theta": best_x.tolist(),
            "at_boundary": at_boundary}
