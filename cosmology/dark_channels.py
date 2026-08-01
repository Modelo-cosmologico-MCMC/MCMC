"""Los dos canales del sector oscuro en forma v35 (Apéndice A, ecs. A.4-A.7).

Friedmann normalizada (A.4):
    H²(z)/H0² = Ωb0(1+z)³ + Ωr0(1+z)⁴ + Ω_id(z) + Ω_lat(z) + Ωk0(1+z)²
    con Ω_id(z) = Ω_id,0 · f_id(S(z); α) y análogamente Ω_lat,
    normalizadas a f(z=0) = 1.

Escalones de los colapsos en la expansión tardía (A.5):
    f_id(S(z)) = 1 + Σ_n α_n · ½·[1 + tanh((S(z) − S_n^post)/ΔS_n)]
    con umbrales S_n^post POSTERIORES a S_1,001 (recombinación, formación
    galáctica, era tardía). El tratado no fija valores numéricos de α_n
    ni S_n^post: son parámetros del ajuste (estatuto CALIBRADO); el
    default α = () apaga los escalones (f ≡ 1).

Ecuaciones de estado efectivas (A.6):
    w_id(z) ≃ 0 para z ≫ z_trans;  ≃ −1 para z ≪ z_trans
    w_lat(z) ≃ −1 + (1/3)·d(ln ρ_lat)/d(ln(1+z))
    w_DE = (w_id·ρ_id + w_lat·ρ_lat)/(ρ_id + ρ_lat)

Puente S ↔ z (A.7): d(ln a)/dS = C(S); dt_rel/dS = T(S)·N(S). El mapa
operativo por defecto invierte el s_to_a heurístico de mcmc_ontology.S_map
(parametrización v32, LEGACY_V32).

Límite de recuperación (Prop. A.1): con α = () y canales constantes se
recupera exactamente el sector Λ de ΛCDM (verificado en tests).

Este módulo NO sustituye a cosmology.background (forma operativa del
ajuste): expone las formas v35 para uso analítico y de contraste.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mcmc_ontology import constants as C


def S_of_z(z: np.ndarray | float, S_today: float = 95.0,
           S_birth: float = C.S_SEALS["C4"]) -> np.ndarray | float:
    """Mapa operativo S(z) invirtiendo a(S) = exp(−(S_today−S)/(S_today−S4)).

    S(z) = S_today − (S_today − S_birth)·ln(1+z). Parametrización v32
    (S_today=95, LEGACY_V32); la forma exacta v35 requiere integrar C(S)
    de la ec. (A.7), no fijada numéricamente en el tratado.
    """
    z = np.asarray(z, dtype=float)
    out = S_today - (S_today - S_birth) * np.log1p(z)
    return float(out) if out.ndim == 0 else out


def f_steps(S: np.ndarray | float,
            alphas: Sequence[float] = (),
            S_post: Sequence[float] = (),
            dS_n: Sequence[float] | float = 1.0) -> np.ndarray | float:
    """Función de forma con escalones f(S) = 1 + Σ α_n·½[1+tanh((S−S_n)/ΔS_n)].

    Ec. (A.5). Con alphas=() devuelve 1 (sin escalones).
    """
    S = np.asarray(S, dtype=float)
    if len(alphas) != len(S_post):
        raise ValueError("alphas y S_post deben tener la misma longitud")
    widths = (np.full(len(alphas), float(dS_n))
              if np.isscalar(dS_n) else np.asarray(dS_n, dtype=float))
    out = np.ones_like(S)
    for a_n, S_n, w_n in zip(alphas, S_post, widths):
        out = out + a_n * 0.5 * (1.0 + np.tanh((S - S_n) / w_n))
    return float(out) if out.ndim == 0 else out


def Omega_channel(z: np.ndarray | float, Omega_0: float,
                  alphas: Sequence[float] = (),
                  S_post: Sequence[float] = (),
                  dS_n: Sequence[float] | float = 1.0,
                  S_today: float = 95.0) -> np.ndarray | float:
    """Ω_canal(z) = Ω_0 · f(S(z))/f(S(0)) — normalizada a f(z=0)=1 (A.4)."""
    S_z = S_of_z(z, S_today=S_today)
    S_0 = S_of_z(0.0, S_today=S_today)
    f_z = f_steps(S_z, alphas, S_post, dS_n)
    f_0 = f_steps(S_0, alphas, S_post, dS_n)
    return Omega_0 * f_z / f_0


def H2_normalized(z: np.ndarray | float,
                  Omega_b0: float = 0.0489,
                  Omega_cdm0: float = 0.2511,
                  Omega_r0: float = 9.2e-5,
                  Omega_id0: float = 0.65,
                  Omega_lat0: float = 0.05,
                  Omega_k0: float = 0.0,
                  alphas_id: Sequence[float] = (),
                  S_post_id: Sequence[float] = (),
                  alphas_lat: Sequence[float] = (),
                  S_post_lat: Sequence[float] = ()) -> np.ndarray | float:
    """H²(z)/H0² según la ec. (A.4), con materia = bariones + CDM.

    Con alphas vacíos los canales son constantes y la expresión coincide
    con ΛCDM plano (si Ωk0=0) con Ω_Λ = Ω_id0 + Ω_lat0 (Prop. A.1).
    """
    z = np.asarray(z, dtype=float)
    zp1 = 1.0 + z
    Om_m = (Omega_b0 + Omega_cdm0) * zp1 ** 3
    out = (Om_m + Omega_r0 * zp1 ** 4
           + Omega_channel(z, Omega_id0, alphas_id, S_post_id)
           + Omega_channel(z, Omega_lat0, alphas_lat, S_post_lat)
           + Omega_k0 * zp1 ** 2)
    return float(out) if out.ndim == 0 else out


def w_id(z: np.ndarray | float, z_trans: float = C.Z_TRANS,
         dz: float = C.DZ_TRANS) -> np.ndarray | float:
    """Interpolación operativa de los límites de la ec. (A.6):

        w_id → 0 para z ≫ z_trans;  w_id → −1 para z ≪ z_trans

    con la misma transición tanh y ancho Δz de (A.3).
    """
    z = np.asarray(z, dtype=float)
    out = -0.5 * (1.0 + np.tanh((z_trans - z) / dz))
    return float(out) if out.ndim == 0 else out


def w_lat(z: np.ndarray | float, rho_lat_fn, eps: float = 1e-4) -> np.ndarray | float:
    """w_lat(z) ≃ −1 + (1/3)·d(ln ρ_lat)/d(ln(1+z)) — ec. (A.6), numérica.

    rho_lat_fn: callable z → ρ_lat(z) (positiva).
    """
    z = np.asarray(z, dtype=float)
    lnzp1 = np.log1p(z)
    z_hi = np.expm1(lnzp1 + eps)
    z_lo = np.expm1(np.clip(lnzp1 - eps, 0.0, None))
    dln_rho = np.log(rho_lat_fn(z_hi)) - np.log(rho_lat_fn(z_lo))
    dln_zp1 = np.log1p(z_hi) - np.log1p(z_lo)
    out = -1.0 + dln_rho / (3.0 * dln_zp1)
    return float(out) if np.ndim(out) == 0 else out


def w_DE(z: np.ndarray | float, rho_id_z: np.ndarray | float,
         rho_lat_z: np.ndarray | float, w_id_z: np.ndarray | float,
         w_lat_z: np.ndarray | float) -> np.ndarray | float:
    """w_DE = (w_id·ρ_id + w_lat·ρ_lat)/(ρ_id + ρ_lat) — ec. (A.6)."""
    num = np.asarray(w_id_z) * np.asarray(rho_id_z) \
        + np.asarray(w_lat_z) * np.asarray(rho_lat_z)
    den = np.asarray(rho_id_z) + np.asarray(rho_lat_z)
    out = num / den
    return float(out) if np.ndim(out) == 0 else out
