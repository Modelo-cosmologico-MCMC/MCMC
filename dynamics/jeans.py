"""Jeans esférico — 5D (frente 5; propuesta v36 §VII).

La ecuación del paso 5D con anisotropía constante β:

    d(ν·σ_r²)/dr + (2β/r)·ν·σ_r² = −ν·dΦ_eff/dr = −ν·g(r)

(g ≥ 0 hacia dentro). Solución por factor integrante (β constante):

    ν·σ_r²(r) = r^(−2β) · ∫_r^∞ ν(s)·g(s)·s^(2β) ds

Proyección en la línea de visión (β constante, kernel estándar):

    Σ(R)·σ_los²(R) = 2·∫_R^∞ (1 − β·R²/r²)·ν·σ_r² · r·dr/√(r² − R²)
    Σ(R)           = 2·∫_R^∞ ν · r·dr/√(r² − R²)

con la sustitución r = √(R² + u²) (elimina la singularidad del borde).

Control estructural del proyector: para un sistema esférico, el
promedio TOTAL pesado por luminosidad ⟨σ_los²⟩ = (1/3)·⟨σ²_3D⟩ es
INDEPENDIENTE de β (cada estrella proyecta 1/3 de su rapidez cuadrática
en cualquier dirección fija, por esfericidad del conjunto) — el test lo
verifica sobre β ∈ {−0.5, 0, 0.3}.

La solución es LINEAL en g: σ_los²[g_N + A·g_unit] = σ_los²[g_N] +
A·σ_los²[g_unit] — la premisa exacta del problema inverso de la
amplitud de Cronos (dsph_data / script del frente).
"""

from __future__ import annotations

import numpy as np


def sigma_r_sq_grid(r_grid: np.ndarray, nu: np.ndarray, g: np.ndarray,
                    beta: float = 0.0) -> np.ndarray:
    """ν·σ_r²·r^(2β) por cuadratura acumulada desde fuera (trapecio
    sobre la malla dada, que debe extenderse hasta r ≫ radio del
    sistema; el resto exterior se desprecia — declarado)."""
    r = np.asarray(r_grid, dtype=float)
    integrand = nu * g * r ** (2.0 * beta)
    acc = np.zeros_like(r)
    dr = np.diff(r)
    seg = 0.5 * (integrand[:-1] + integrand[1:]) * dr
    acc[:-1] = np.cumsum(seg[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(nu > 0.0, acc / (nu * r ** (2.0 * beta)), 0.0)
    return s2


def project_los(R: np.ndarray, r_of, nu_of, sig2_of,
                beta: float = 0.0, u_max: float = 1e4,
                n_u: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """(Σ(R), Σ·σ_los²(R)) con la sustitución r = √(R² + u²).

    r_of/nu_of/sig2_of son llamables (interpolantes o formas cerradas);
    u_max en pc debe cubrir el sistema (declarado en el llamador)."""
    R = np.atleast_1d(np.asarray(R, dtype=float))
    u = np.geomspace(1e-3, u_max, n_u)
    u = np.concatenate(([0.0], u))
    Sig = np.empty(R.size)
    Sig_s2 = np.empty(R.size)
    for i, Ri in enumerate(R):
        r = np.sqrt(Ri ** 2 + u ** 2)
        nu = nu_of(r)
        s2 = sig2_of(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            kern = np.where(r > 0.0, 1.0 - beta * Ri ** 2 / r ** 2, 1.0)
        Sig[i] = 2.0 * np.trapezoid(nu, u)
        Sig_s2[i] = 2.0 * np.trapezoid(kern * nu * s2, u)
    return Sig, Sig_s2


def sigma_los_sq(R: np.ndarray, r_grid: np.ndarray, nu: np.ndarray,
                 g: np.ndarray, beta: float = 0.0,
                 u_max: float = 1e4) -> np.ndarray:
    """σ_los²(R) resolviendo Jeans en r_grid y proyectando."""
    s2 = sigma_r_sq_grid(r_grid, nu, g, beta=beta)
    nu_of = lambda r: np.interp(r, r_grid, nu, right=0.0)  # noqa: E731
    s2_of = lambda r: np.interp(r, r_grid, s2, right=0.0)  # noqa: E731
    Sig, Sig_s2 = project_los(R, None, nu_of, s2_of, beta=beta,
                              u_max=u_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Sig > 0.0, Sig_s2 / Sig, 0.0)


def sigma_los_sq_lum_avg(r_grid: np.ndarray, nu: np.ndarray,
                         g: np.ndarray, beta: float = 0.0,
                         R_max: float = 3e3, n_R: int = 120,
                         u_max: float = 1e4) -> float:
    """⟨σ_los²⟩ pesado por luminosidad proyectada:
    ∫ Σ·σ_los²·2πR dR / ∫ Σ·2πR dR, hasta R_max (declarado)."""
    R = np.geomspace(r_grid[0], R_max, n_R)
    s2 = sigma_r_sq_grid(r_grid, nu, g, beta=beta)
    nu_of = lambda r: np.interp(r, r_grid, nu, right=0.0)  # noqa: E731
    s2_of = lambda r: np.interp(r, r_grid, s2, right=0.0)  # noqa: E731
    Sig, Sig_s2 = project_los(R, None, nu_of, s2_of, beta=beta,
                              u_max=u_max)
    num = np.trapezoid(Sig_s2 * 2.0 * np.pi * R, R)
    den = np.trapezoid(Sig * 2.0 * np.pi * R, R)
    return float(num / den)
