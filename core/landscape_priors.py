"""Sensibilidad de la naturalidad de δ_H a los priors del paisaje.

Tarea nacida de la auditoría v35.1: la afirmación de naturalidad —
«δ_H = O(0.05) es genérico en el paisaje O(1) fértil» (Nota I, §3.2) —
se midió con UN prior (uniforme) sobre UN dominio ((0.5, 2), con b̄
extendido a (0.5, 4)). Este módulo la condiciona correctamente: repite
la cartografía bajo priors distintos (uniforme, log-uniforme, normal
truncada), dominios O(1) distintos y con/sin el filtro de fertilidad,
y separa lo ROBUSTO de lo dependiente del prior.

Lo que es independiente del prior (analítico): dentro de un dominio
[m_lo, ·] × [b_·, b_max] × [C0_lo, ·], el discriminante está acotado
por D ≤ b_max² − 4·C0_lo·m_lo², luego

    δ_H ≥ λ_H / √(b_max² − 4·C0_lo·m_lo²)     (cota de dominio)

para CUALQUIER prior soportado en el dominio. La inaccesibilidad de
δ_H = 0.012 es una afirmación sobre dominios, no sobre priors.

Lo que sí depende del prior: la mediana y la banda central de δ_H.
Este módulo las mide y las publica por configuración — la afirmación
de naturalidad queda condicionada a lo que sobreviva al barrido.
"""

from __future__ import annotations

import numpy as np

from .basal import c_bar, quasi_cancellation_ok
from .delta0_circle import LAMBDA_H_SEAL
from .victoria import fertility_condition

PRIORS = ("uniform", "loguniform", "normal")
# Dominios O(1) del barrido: el fiducial y dos ensanchamientos honestos
DOMAINS = ((0.5, 2.0), (1.0 / 3.0, 3.0), (0.25, 4.0))


def _draw(rng: np.random.Generator, prior: str, lo: float, hi: float,
          size: int) -> np.ndarray:
    """Un lote del prior pedido, soportado en [lo, hi]."""
    if prior == "uniform":
        return rng.uniform(lo, hi, size=size)
    if prior == "loguniform":
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))
    if prior == "normal":
        # normal truncada por rechazo: μ centro del dominio, σ = ancho/4
        mu, sigma = 0.5 * (lo + hi), 0.25 * (hi - lo)
        out = rng.normal(mu, sigma, size=size)
        bad = (out < lo) | (out > hi)
        while bad.any():
            out[bad] = rng.normal(mu, sigma, size=int(bad.sum()))
            bad = (out < lo) | (out > hi)
        return out
    raise ValueError(f"prior desconocido: {prior}")


def sample_shapes_prior(n: int, seed: int, prior: str = "uniform",
                        o1_range: tuple = (0.5, 2.0),
                        b_extended: bool = True,
                        fertile_filter: bool = True,
                        gamma_max: float = 3.0,
                        max_batches: int = 200) -> dict:
    """Paisajes viables (casi-cancelación (3.3) y c̄ > 0) bajo el prior
    pedido; con fertile_filter, además A = γR·ē/m̄² > 1 con
    γR ~ U(0, γ_max] (γ_max declarado, como en fertility_map)."""
    rng = np.random.default_rng(seed)
    lo, hi = o1_range
    b_hi = 2.0 * hi if b_extended else hi
    out = {k: [] for k in ("m_bar", "b_bar", "C0", "e_bar")}
    n_attempted = 0
    for _ in range(max_batches):
        if len(out["m_bar"]) >= n:
            break
        batch = max(n, 1000)
        n_attempted += batch
        m = _draw(rng, prior, lo, hi, batch)
        b = _draw(rng, prior, lo, b_hi, batch)
        c0 = _draw(rng, prior, lo, hi, batch)
        e = _draw(rng, prior, lo, hi, batch)
        ok = np.array([quasi_cancellation_ok(mi, bi, ci)
                       for mi, bi, ci in zip(m, b, c0)])
        if fertile_filter:
            gamma_R = rng.uniform(0.0, gamma_max, size=batch)
            fert = np.array([fertility_condition(g, mi, ei)
                             for g, mi, ei in zip(gamma_R, m, e)])
            ok &= fert
        cb = np.array([c_bar(mi, bi, ci) if o else -1.0
                       for mi, bi, ci, o in zip(m, b, c0, ok)])
        ok &= cb > 0.0
        for k, arr in (("m_bar", m), ("b_bar", b), ("C0", c0), ("e_bar", e)):
            out[k].extend(arr[ok][:n - len(out[k]) + len(arr[ok])])
    shapes = {k: np.array(v[:n]) for k, v in out.items()}
    shapes["n_attempted"] = n_attempted
    return shapes


def delta_H_of(shapes: dict,
               lambda_target: float = LAMBDA_H_SEAL) -> np.ndarray:
    """δ_H = λ_H/√(b̄² − 4·C0·m̄²) por paisaje (H.8)."""
    disc = shapes["b_bar"] ** 2 - 4.0 * shapes["C0"] * shapes["m_bar"] ** 2
    return lambda_target / np.sqrt(disc)


def domain_bound(o1_range: tuple = (0.5, 2.0), b_extended: bool = True,
                 lambda_target: float = LAMBDA_H_SEAL) -> float:
    """Cota inferior ANALÍTICA de δ_H sobre el dominio — válida para
    cualquier prior soportado en él: δ_H ≥ λ_H/√(b_max²−4·C0_lo·m_lo²)."""
    lo, hi = o1_range
    b_max = 2.0 * hi if b_extended else hi
    disc_max = b_max ** 2 - 4.0 * lo * lo ** 2
    if disc_max <= 0.0:
        raise ValueError("Dominio sin paisajes viables (D ≤ 0 en todo él)")
    return lambda_target / float(np.sqrt(disc_max))


def sensitivity_row(n: int, seed: int, prior: str, o1_range: tuple,
                    b_extended: bool = True,
                    fertile_filter: bool = True) -> dict:
    """Una configuración del barrido: estadísticas de δ_H + cota."""
    shapes = sample_shapes_prior(n, seed, prior, o1_range, b_extended,
                                 fertile_filter)
    d_H = delta_H_of(shapes)
    return {
        "prior": prior, "o1_range": o1_range, "b_extended": b_extended,
        "fertile_filter": fertile_filter,
        "n": int(len(d_H)),
        "median": float(np.median(d_H)),
        "p5": float(np.percentile(d_H, 5)),
        "p95": float(np.percentile(d_H, 95)),
        "min": float(d_H.min()),
        "bound": domain_bound(o1_range, b_extended),
        "delta_H": d_H,
    }


def sensitivity_scan(n: int = 20000, seed: int = 20260804,
                     priors: tuple = PRIORS,
                     domains: tuple = DOMAINS) -> list[dict]:
    """El barrido completo: priors × dominios (b̄ extendido, filtro
    fértil), más la fila sin filtro fértil y la fila sin extensión de
    b̄ para el caso fiducial (los otros dos ejes de sensibilidad)."""
    rows = []
    for prior in priors:
        for dom in domains:
            rows.append(sensitivity_row(n, seed, prior, dom))
    rows.append(sensitivity_row(n, seed, "uniform", domains[0],
                                fertile_filter=False))
    rows.append(sensitivity_row(n, seed, "uniform", domains[0],
                                b_extended=False))
    return rows
