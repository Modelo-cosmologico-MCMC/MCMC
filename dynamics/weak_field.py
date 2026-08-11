"""Potencial débil del MCMC — 5A/5B (frente 5; Def. 11.1, Prop. 11.2).

Una sola Φ_eff(r) desde la Ley de Cronos débil:

    N = 1 + Φ_N/c² − ε_c(ρ),      ε_c = α₀⁻¹·(ρ/ρ_c)^(3/2)
    u̇ = −(ȧ/a)·u − ∇(Φ_N − c²·ε_c) − ε̇_c·u        (Prop. 11.2)

En régimen estático (ȧ → 0; ρ̇ ≃ 0, la fricción de compuerta está
apagada en sistemas virializados — H.2.5):

    Φ_eff(r) = Φ_N(r) − c²·ε_c(ρ(r))
    g_eff(r) = dΦ_eff/dr = G·M(<r)/r² − c²·dε_c/dr ≥ 0 (hacia dentro)

El término −c²·ε_c es la «cara de campo débil de la MCV» (Cor. 11.3c):
atracción adicional hacia los picos de densidad. La cota dura (11.5)
α₀⁻¹ ≲ 1e-6 nace de exigir c²·ε_c ≲ |Φ_N| en halos con
v_circ ~ 220 km/s: el término no puede dominar sobre la gravedad
newtoniana donde ρ ~ ρ_c.

CONDICIONALES EXPUESTOS (nunca resueltos en silencio):
- α₀⁻¹: cota 1e-6 (reutiliza cronos.cronos_v3.check_alpha0_inv — una
  sola fuente para la cota en todo el repositorio);
- ρ_c: «densidad de referencia» (Def. 11.1) — el tratado NO la fija.
  El análisis dSph la trata como incógnita del problema inverso: los
  dos parámetros solo entran por la amplitud A ≡ α₀⁻¹/ρ_c^(3/2)
  (ε_c = A·ρ^(3/2)), así que medir A en un sistema es medir la
  combinación que TODOS los sistemas deben compartir (5E).

Unidades: pc, M_sol, km/s (G = 4.30091e-3 pc·(km/s)²/M_sol).
"""

from __future__ import annotations

import numpy as np

from cronos.cronos_v3 import check_alpha0_inv

G_PC = 4.30091e-3          # pc·(km/s)²/M_sol
C_KMS = 299792.458         # km/s


# ---------- Perfil estelar de Plummer (trazador y fuente bariónica) ----------

def plummer_density(r, M: float, a: float):
    """ρ(r) = (3M/4πa³)·(1 + r²/a²)^(−5/2)  [M_sol/pc³]."""
    r = np.asarray(r, dtype=float)
    return (3.0 * M / (4.0 * np.pi * a ** 3)) \
        * (1.0 + (r / a) ** 2) ** -2.5


def plummer_mass(r, M: float, a: float):
    """M(<r) = M·r³/(r² + a²)^(3/2)."""
    r = np.asarray(r, dtype=float)
    return M * r ** 3 / (r ** 2 + a ** 2) ** 1.5


def plummer_surface_density(R, M: float, a: float):
    """Σ(R) = M·a²/π · (a² + R²)^(−2)  [M_sol/pc²]."""
    R = np.asarray(R, dtype=float)
    return M * a ** 2 / np.pi / (a ** 2 + R ** 2) ** 2


def plummer_g_newton(r, M: float, a: float):
    """g_N(r) = G·M(<r)/r² = G·M·r/(r² + a²)^(3/2) ≥ 0 (hacia dentro)."""
    r = np.asarray(r, dtype=float)
    return G_PC * M * r / (r ** 2 + a ** 2) ** 1.5


def plummer_potential(r, M: float, a: float):
    """Φ_N(r) = −G·M/√(r² + a²)  [(km/s)²]."""
    r = np.asarray(r, dtype=float)
    return -G_PC * M / np.sqrt(r ** 2 + a ** 2)


def plummer_sigma_los_sq_isotropic(R, M: float, a: float):
    """Identidad analítica del Plummer isótropo autoconsistente:
    σ_los²(R) = (3π/64)·(G·M/a)·(1 + R²/a²)^(−1/2). Es el anclaje
    exacto contra el que se clava el solucionador numérico."""
    R = np.asarray(R, dtype=float)
    return (3.0 * np.pi / 64.0) * G_PC * M / a \
        / np.sqrt(1.0 + (R / a) ** 2)


# ---------- El término de Cronos (5A) ----------

def cronos_amplitude(alpha0_inv: float, rho_c: float) -> float:
    """A ≡ α₀⁻¹/ρ_c^(3/2), la ÚNICA combinación por la que (α₀⁻¹, ρ_c)
    entran en el potencial débil: ε_c = A·ρ^(3/2). Valida la cota
    (11.5) sobre α₀⁻¹."""
    check_alpha0_inv(alpha0_inv)
    if rho_c <= 0.0:
        raise ValueError(f"ρ_c debe ser > 0 (recibido {rho_c})")
    return alpha0_inv / rho_c ** 1.5


def epsilon_c_of_rho(rho, A: float):
    """ε_c = A·ρ^(3/2) (Def. 11.1 reescrita en la amplitud A)."""
    rho = np.asarray(rho, dtype=float)
    return A * np.clip(rho, 0.0, None) ** 1.5


def g_cronos_plummer(r, M: float, a: float, A: float):
    """−c²·dε_c/dr para el perfil de Plummer, analítico y ≥ 0
    (hacia dentro): dε_c/dr = (3/2)·ε_c·(dρ/dr)/ρ con
    dρ/dr = −5·ρ·r/(r² + a²), luego

        g_cronos(r) = c²·(15/2)·ε_c(ρ(r))·r/(r² + a²).
    """
    r = np.asarray(r, dtype=float)
    eps = epsilon_c_of_rho(plummer_density(r, M, a), A)
    return C_KMS ** 2 * 7.5 * eps * r / (r ** 2 + a ** 2)


def g_eff_plummer(r, M: float, a: float, A: float):
    """5B: g_eff = g_N + g_cronos ≥ 0 (hacia dentro), lineal en A."""
    return plummer_g_newton(r, M, a) + g_cronos_plummer(r, M, a, A)


def bound_saturation_ratio(r, M: float, a: float, A: float):
    """max_r c²·ε_c/|Φ_N| — el cociente que la ec. (11.5) exige ≲ 1.
    Con él se mide (a) la mayor amplitud A que respeta la cota con la
    forma ρ^(3/2) del perfil, y (b) cuánto la viola la amplitud que un
    sistema exige (problema inverso)."""
    r = np.asarray(r, dtype=float)
    eps = epsilon_c_of_rho(plummer_density(r, M, a), A)
    return float(np.max(C_KMS ** 2 * eps
                        / np.abs(plummer_potential(r, M, a))))
