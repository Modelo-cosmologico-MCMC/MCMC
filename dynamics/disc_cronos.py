"""5C estructural: el término débil de Cronos en discos (frente 5).

La falsación cruzada 5E exige que la MISMA amplitud A = α₀⁻¹/ρ_c^(3/2)
sirva a sistemas de presión (Sculptor, medio paso 2) y rotacionales
(SPARC, paso 5C). La ingesta del catálogo SPARC está BLOQUEADA por el
proxy de la sesión (astroweb.cwru.edu fuera de la lista blanca —
pendiente declarado en scripts/download_data.py); este módulo ejecuta
la mitad ESTRUCTURAL del 5C, que no necesita datos: sobre discos
exponenciales declarados que barren el rango de la población SPARC,
la forma del término está fijada por la propia Def. 11.1.

Disco exponencial declarado:
    Σ(R) = Σ0·e^(−R/R_d);   ρ(R) = Σ(R)/(2h),  h = ζ·R_d (ζ declarado)

Rotación bariónica (disco delgado, Freeman 1970):
    v_bar²(R) = 4πG·Σ0·R_d·y²·[I0(y)K0(y) − I1(y)K1(y)],  y = R/(2R_d)

Término de Cronos en el plano (ε_c = A·ρ^(3/2), Def. 11.1 reescrita):
    g_cronos(R) = −c²·dε_c/dR = (3c²/2R_d)·ε_c(R) ≥ 0  (hacia dentro)
    v_cronos²(R) = R·g_cronos = (3c²/2)·(R/R_d)·A·ρ(R)^(3/2)
                 ∝ x·e^(−3x/2),   x ≡ R/R_d

FORMA, no ajuste: v_cronos² alcanza su máximo en x = 2/3 y muere como
e^(−3x/2) — más deprisa que la propia v_bar². La consecuencia
estructural es independiente de A (y por tanto de α₀⁻¹ y ρ_c): NINGUNA
amplitud convierte el término en una curva de rotación plana exterior,
porque su forma decae donde la discrepancia de masa vive (x ≳ 3); y la
amplitud concreta que Sculptor exige lo convierte además en una
perturbación grande justo donde los discos masivos son bariónicos
(x ≲ 1). El script del frente cuantifica ambas cosas sobre la malla
declarada; la confrontación por galaxia con SPARC real queda pendiente
de la ingesta.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0, i1, k0, k1

from .weak_field import C_KMS, G_PC, epsilon_c_of_rho


def sigma_exponential(R, Sigma0: float, R_d: float):
    """Σ(R) = Σ0·e^(−R/R_d)  [M_sol/pc²]."""
    R = np.asarray(R, dtype=float)
    return Sigma0 * np.exp(-R / R_d)


def rho_midplane(R, Sigma0: float, R_d: float, zeta: float):
    """ρ(R) = Σ(R)/(2h) con h = ζ·R_d — la geometría vertical entra
    como parámetro DECLARADO ζ (no hay datos verticales aquí)."""
    if zeta <= 0.0:
        raise ValueError(f"ζ debe ser > 0 (recibido {zeta})")
    return sigma_exponential(R, Sigma0, R_d) / (2.0 * zeta * R_d)


def v_bar_sq_freeman(R, Sigma0: float, R_d: float):
    """v_bar²(R) del disco exponencial delgado (Freeman 1970)
    [(km/s)²]: 4πG·Σ0·R_d·y²·[I0K0 − I1K1], y = R/(2R_d)."""
    R = np.asarray(R, dtype=float)
    y = R / (2.0 * R_d)
    bessel = i0(y) * k0(y) - i1(y) * k1(y)
    return 4.0 * np.pi * G_PC * Sigma0 * R_d * y ** 2 * bessel


def v_cronos_sq(R, Sigma0: float, R_d: float, zeta: float, A: float):
    """v_cronos²(R) = (3c²/2)·(R/R_d)·A·ρ(R)^(3/2)  [(km/s)²] —
    la contribución rotacional del término −c²∇ε_c en el plano del
    disco exponencial (radial; el gradiente vertical no rota)."""
    R = np.asarray(R, dtype=float)
    rho = rho_midplane(R, Sigma0, R_d, zeta)
    return 1.5 * C_KMS ** 2 * (R / R_d) * epsilon_c_of_rho(rho, A)


def x_peak_v_cronos() -> float:
    """El máximo de v_cronos² ∝ x·e^(−3x/2) está en x = 2/3 —
    identidad de forma (d/dx[x·e^(−3x/2)] = 0 ⟺ x = 2/3)."""
    return 2.0 / 3.0


def outer_decline_ratio(x_outer: float) -> float:
    """v_cronos²(x_outer)/v_cronos²(x_peak) — cuánto ha muerto el
    término en el radio exterior x_outer = R/R_d (identidad de forma,
    independiente de A, Σ0, R_d, ζ)."""
    xp = x_peak_v_cronos()
    return float((x_outer * np.exp(-1.5 * x_outer))
                 / (xp * np.exp(-1.5 * xp)))
