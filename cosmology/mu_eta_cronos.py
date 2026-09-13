"""µ(k,a), η(k,a) y Σ(k,a) desde la ontología del MCMC — canal Cronos
DERIVADO, canal Atlas DECLARADO (nota de teoría 13-sep-2026, §B).

El tratado deja el hueco señalado (A.3): «a primer nivel basta con
µ(k,a) ≃ 1, Φ ≃ Ψ […] La faz tensorial puede incorporarse después
mediante µ(k,a), η(k,a) escala-dependientes». Este módulo es esa
incorporación, en el orden epistemológico preinscrito:
ε_c/Atlas → µ, η → observables → datos. NO introduce campos ni
parámetros nuevos: solo los que el tratado ya declara (α₀⁻¹, ρ_c,
ε_K, α_a).

OBJETOS DEL TRATADO (v35) que se usan tal cual:
  Ley de Cronos débil (11.1):  N = 1 + Φ_N/c² − ε_c(ρ),
                               ε_c = α₀⁻¹ (ρ/ρ_c)^{3/2} ≥ 0
      — la corrección vive SOLO en la lapse; q_ij no se toca.
  Geodésica (11.2):            Ψ_mat = Φ_N − c² ε_c
  Gea en campo débil (9.3):    ∇²Φ_N = 4πG_N a² ρ̄ δ   (Poisson estándar)
  Cota (11.5):                 α₀⁻¹ ≲ 1e-6 (cronos.cronos_v3.ALPHA0_INV_MAX)

DERIVACIÓN (canal Cronos, cerrada — sin frente pendiente):
  Cierre 1 (linealización, |δ| ≲ 1):  δε_c = (3/2) ε̄_c(a) δ,
      con ε̄_c(a) = ε_c(ρ̄_m(a)) por la función CANÓNICA
      cronos_v3.epsilon_c (una sola fuente para la fórmula y la cota).
  En Fourier, con Poisson para Φ_N:
      −k²Ψ = 4πG a² ρ̄ δ + (3/2) c² k² ε̄_c δ
  ⟹  µ(k,a) − 1 = (3/2) ε̄_c · c²k² / (4πG a² ρ̄_m)
                 = ε̄_c(a) · (c k/(a H))² / Ω_m(a)          [k físico]
  Como q_ij queda intacta, Φ = Φ_N, y de ahí las tres relaciones
  DISTINTIVAS del canal, sin parámetros adicionales:
      η = Φ/Ψ = 1/µ        ⟹ η − 1 = −(µ − 1) + O((µ−1)²)   [slip Φ/Ψ < 1]
      Σ = (Φ+Ψ)/(2Φ_N) = (1+µ)/2  ⟹ Σ − 1 = (µ − 1)/2        [exacto]
      µ − 1 ∝ k²                                             [cola cuadrática]
  Límite GR: α₀⁻¹ → 0 ⟹ µ = η = Σ = 1 exactamente (Prop. A.1).

  Cierre 2 (ρ_c: el tratado NO la fija — Def. 11.1 «densidad de
  referencia»; el frente 5 usó «~200× la media»). Dos lecturas con
  tendencias en z OPUESTAS, ambas declaradas y separadas en código:
      'comoving':  ρ_c = f·ρ̄_m(a)  ⟹ ε̄_c = α₀⁻¹ f^{−3/2}  (constante)
                   ⟹ µ − 1 ∝ (ck/aH)²: crece hacia z bajo
      'physical':  ρ_c = f·ρ̄_m(0)  ⟹ ε̄_c ∝ (1+z)^{9/2}
                   ⟹ µ − 1 ∝ k²(1+z)^{7/2} en era de materia: crece
                   fuerte hacia z alto (el kernel del CMB-lensing a
                   z ~ 2 pasa a ser el mejor constriñidor de α₀⁻¹)
      f = RHO_C_OVER_MEAN = 200 — parámetro del cierre, NO derivado.
      (constants.RHO_C0 = 200 es el esquema LEGACY v32 y NO es la
      fuente aquí: la coincidencia numérica es la misma convención
      «200× la media», declarada de nuevo.)
  Cierre 3 (ventana de validez): la extrapolación k² se corta donde
  la linealización de ρ^{3/2} y la compuerta Θ(ρ̇) toman el mando —
  K_MAX_LINEAR_HMPC = 0.2 h/Mpc declarado; fuera de la ventana los
  números se publican marcados, no se citan.

CANAL ATLAS (9.4–9.5) — DERIVADO el 13-sep-2026 (frente 3, cosmology/
mu_eta_atlas.py): en el régimen cuasi-estático sub-horizonte el offset
µ_Atlas − 1 = α_a/(2ξ − α_a) es independiente de k y de λ_K, y se
CANCELA exactamente contra la renormalización de la G local
(G_growth = G_local = G_B/(ξ − α_a/2)): medida respecto de la G de
laboratorio, µ_Atlas = 1 + O(e²) y η_Atlas = 1 + O(e²). La expectativa
anterior de este módulo («µ_Atlas − 1 = O(1)·ε_K») queda SUPERADA por la
derivación: ε_K no entra al orden dominante — entra en el fondo,
G_cosmo/G_local = (2ξ − α_a)/(3λ_K − 1) ≈ 1 − (3/2)ε_K − α_a/2, que BBN
acota como combinación (frente 6). Consecuencia: la cola k² del canal
Cronos es la ÚNICA firma sub-horizonte de (µ, η) del sector
perturbativo. atlas_offset() conserva la firma por compatibilidad y
devuelve 0 con estatuto derivado, no pendiente.

Todo cálculo de este módulo es PREDICCIÓN: no carga datos, no ajusta
nada. La prohibición preinscrita: µ, η nunca se eligen desde los datos
de lensing.
"""

from __future__ import annotations

import numpy as np

from cosmology.background import H_of_z
from cosmology.extended_likelihoods import growth_D_f
from cronos.cronos_v3 import ALPHA0_INV_MAX, check_alpha0_inv, epsilon_c
from mcmc_ontology import constants as C

C_KMS = 299792.458
RHO_C_OVER_MEAN = 200.0        # cierre 2: ρ_c = 200 × ρ̄_m (frente 5)
K_MAX_LINEAR_HMPC = 0.2        # cierre 3: ventana de validez declarada
RHO_C_MODES = ("comoving", "physical")
ATLAS_STATUS = ("DERIVADO-NULO al orden dominante (13-sep-2026, frente 3; "
                "ver cosmology/mu_eta_atlas.py y results/"
                "2026-09-13_mu_eta_atlas/): el offset α_a/(2ξ) se cancela "
                "contra la G local — la cola k² de Cronos es la ÚNICA firma "
                "sub-horizonte de (µ, η); colas O(e²) de Atlas: residuos "
                "del polo 1/(λ_K−1) derivados con velocidades (E3a = A, "
                "results/2026-09-13_mu_eta_atlas_tail/), partes regulares "
                "numéricas y confirmación numérica pendiente (E3b = C)")

__all__ = [
    "ALPHA0_INV_MAX", "ATLAS_STATUS", "K_MAX_LINEAR_HMPC",
    "RHO_C_MODES", "RHO_C_OVER_MEAN",
    "Sigma_cronos", "atlas_offset", "background_above_rho_c_z",
    "epsilon_c_background", "epsilon_c_nonperturbative_z",
    "eta_cronos", "fsigma8_ratio_cronos", "growth_Dk",
    "in_validity_window", "mu_cronos", "mu_minus_one_cronos",
    "mu_table", "omega_m_of_a", "z_where_mu_minus_one_reaches",
]


# ---------------------------------------------------------------------
# Fondo y ε̄_c(a)
# ---------------------------------------------------------------------

def omega_m_of_a(a, theta):
    """Ω_m(a) = Ω_m a⁻³ (H0/H)² y H(a) [km/s/Mpc] del fondo MCMC
    (cosmology.background.H_of_z: clausura plana por llamada, radiación
    incluida en la geometría). theta = (H0, Ω_m, ε, z_trans)."""
    H0, Om, eps, z_t = theta
    a = np.asarray(a, float)
    z = 1.0 / a - 1.0
    H = np.asarray(H_of_z(z, H0=H0, Omega_m=Om, eps=eps, z_trans=z_t))
    return Om * a ** -3.0 * (H0 / H) ** 2, H


def epsilon_c_background(a, alpha0_inv: float, rho_c_mode: str = "comoving",
                         rho_c_over_mean: float = RHO_C_OVER_MEAN):
    """ε̄_c(a) = ε_c(ρ̄_m(a)) con la función canónica del tratado
    (cronos_v3.epsilon_c valida la cota 11.5). Densidades en unidades
    de ρ̄_m(0): ρ̄_m(a) = a⁻³.

    'comoving': ρ_c = f·ρ̄_m(a) ⟹ ε̄_c = α₀⁻¹ f^{−3/2} (constante en a)
    'physical': ρ_c = f·ρ̄_m(0) ⟹ ε̄_c = α₀⁻¹ (a⁻³/f)^{3/2} ∝ (1+z)^{9/2}
    """
    if rho_c_mode not in RHO_C_MODES:
        raise ValueError(f"rho_c_mode debe ser uno de {RHO_C_MODES}: "
                         f"{rho_c_mode!r}")
    if rho_c_over_mean <= 0.0:
        raise ValueError("rho_c_over_mean debe ser > 0")
    a = np.asarray(a, float)
    rho_bar = a ** -3.0
    if rho_c_mode == "comoving":
        rho_c = rho_c_over_mean * rho_bar
    else:
        rho_c = rho_c_over_mean * np.ones_like(rho_bar)
    return np.asarray(epsilon_c(rho_bar, alpha0_inv, rho_c), float)


# ---------------------------------------------------------------------
# Canal Cronos: µ, η, Σ
# ---------------------------------------------------------------------

def mu_minus_one_cronos(k_hMpc, a, theta, alpha0_inv: float,
                        rho_c_mode: str = "comoving",
                        rho_c_over_mean: float = RHO_C_OVER_MEAN):
    """µ − 1 = ε̄_c(a) · (c k/(a H))² / Ω_m(a), con k físico = k[h/Mpc]·h.

    Devolver µ − 1 (y no µ) preserva la precisión de las identidades
    cuando µ − 1 ~ 1e-10 … 1e-2. k y a se difunden (broadcast)."""
    check_alpha0_inv(alpha0_inv)
    k = np.asarray(k_hMpc, float)
    a = np.asarray(a, float)
    if np.any(k < 0):
        raise ValueError("k debe ser ≥ 0")
    if np.any(a <= 0) or np.any(a > 1.0):
        raise ValueError("a debe estar en (0, 1]")
    H0 = theta[0]
    h = H0 / 100.0
    Om_a, H = omega_m_of_a(a, theta)
    eps_bar = epsilon_c_background(a, alpha0_inv, rho_c_mode,
                                   rho_c_over_mean)
    k_phys = k * h                                   # [1/Mpc]
    return eps_bar * (C_KMS * k_phys / (a * H)) ** 2 / Om_a


def mu_cronos(k_hMpc, a, theta, alpha0_inv: float,
              rho_c_mode: str = "comoving",
              rho_c_over_mean: float = RHO_C_OVER_MEAN):
    """µ(k,a) del canal Cronos."""
    return 1.0 + mu_minus_one_cronos(k_hMpc, a, theta, alpha0_inv,
                                     rho_c_mode, rho_c_over_mean)


def eta_cronos(k_hMpc, a, theta, alpha0_inv: float,
               rho_c_mode: str = "comoving",
               rho_c_over_mean: float = RHO_C_OVER_MEAN):
    """η = Φ/Ψ = 1/µ (exacto: Φ = Φ_N porque q_ij no se toca) ⟹
    η − 1 = −(µ − 1)/µ = −(µ − 1) + O((µ−1)²)."""
    return 1.0 / mu_cronos(k_hMpc, a, theta, alpha0_inv, rho_c_mode,
                           rho_c_over_mean)


def Sigma_cronos(k_hMpc, a, theta, alpha0_inv: float,
                 rho_c_mode: str = "comoving",
                 rho_c_over_mean: float = RHO_C_OVER_MEAN):
    """Σ = (Φ + Ψ)/(2 Φ_N) = (1 + µ)/2 ⟹ Σ − 1 = (µ − 1)/2 exacto: la
    lente recibe la mitad del impulso que el crecimiento."""
    return 0.5 * (1.0 + mu_cronos(k_hMpc, a, theta, alpha0_inv,
                                  rho_c_mode, rho_c_over_mean))


def in_validity_window(k_hMpc) -> np.ndarray:
    """Cierre 3: True donde la extrapolación k² está declarada válida."""
    return np.asarray(k_hMpc, float) <= K_MAX_LINEAR_HMPC


# ---------------------------------------------------------------------
# Canal Atlas: offset sub-horizonte derivado nulo (#18); colas O(e²) con
# residuos del polo derivados (E3a) y confirmación numérica pendiente (E3b)
# ---------------------------------------------------------------------

def atlas_offset(epsilon_K: float = C.EPSILON_K,
                 c_mu: float | None = None) -> float:
    """Offset sub-horizonte del canal Atlas en µ medida respecto de la G
    local: ≡ 0 por CANCELACIÓN (derivado, 13-sep-2026; ver
    cosmology.mu_eta_atlas.mu_atlas_relative_to_local). c_µ explícito es
    una hipótesis del llamador ajena al tratado (se conserva por
    compatibilidad de firma); ε_K NO entra al orden dominante."""
    if c_mu is None:
        return 0.0
    return float(c_mu) * float(epsilon_K)


# ---------------------------------------------------------------------
# Crecimiento con µ(k,a): D(k,a) y la razón fσ8 con cero parámetros nuevos
# ---------------------------------------------------------------------

def growth_Dk(z_eval, theta, k_hMpc: float, alpha0_inv: float,
              rho_c_mode: str = "comoving", n_grid: int = 400):
    """D(k,z)/D(k,0) y f(k,z) integrando la ODE exacta con la fuente
    (3/2)Ω_m(a)µ(k,a)D — generalización de extended_likelihoods.
    growth_D_f (misma malla, mismo RK4; con µ ≡ 1 reproduce D(a) bit a
    bit)."""
    def mu_of_a(a):
        return mu_cronos(k_hMpc, a, theta, alpha0_inv, rho_c_mode)
    return growth_D_f(np.asarray(z_eval, float), theta, n_grid=n_grid,
                      mu_of_a=mu_of_a)


def fsigma8_ratio_cronos(z_eval, theta, k_hMpc: float, alpha0_inv: float,
                         rho_c_mode: str = "comoving",
                         n_grid: int = 400) -> np.ndarray:
    """R_µ(k,z) = [f·D](µ) / [f·D](µ ≡ 1) al MISMO fondo: σ8 se cancela —
    la predicción del sector lineal con cero parámetros nuevos.

    Falla cerrado (FloatingPointError) si el crecimiento diverge: ocurre
    cuando el cierre deja de ser perturbativo dentro del intervalo de
    integración (µ − 1 ≳ 1 a z alto) — un NaN publicado sería un número
    inválido disfrazado de resultado; el diagnóstico correcto lo dan
    z_where_mu_minus_one_reaches y epsilon_c_nonperturbative_z."""
    z_eval = np.asarray(z_eval, float)
    with np.errstate(over="ignore", invalid="ignore"):
        D1, f1 = growth_Dk(z_eval, theta, k_hMpc, alpha0_inv, rho_c_mode,
                           n_grid)
    if not (np.all(np.isfinite(D1)) and np.all(np.isfinite(f1))):
        raise FloatingPointError(
            f"crecimiento divergente (k = {k_hMpc} h/Mpc, cierre "
            f"'{rho_c_mode}', α₀⁻¹ = {alpha0_inv:g}): µ − 1 deja de ser "
            "perturbativo dentro del intervalo de integración; ningún "
            "número de fσ8 es citable bajo este cierre")
    D0, f0 = growth_D_f(z_eval, theta, n_grid=n_grid)
    return (f1 * D1) / (f0 * D0)


# ---------------------------------------------------------------------
# Diagnósticos de no-perturbatividad (dónde se rompe cada cierre)
# ---------------------------------------------------------------------

def epsilon_c_nonperturbative_z(alpha0_inv: float,
                                rho_c_mode: str = "physical",
                                target: float = 1.0,
                                rho_c_over_mean: float = RHO_C_OVER_MEAN
                                ) -> float:
    """Menor z con ε̄_c(z) ≥ target — donde la corrección a la lapse
    (N = 1 + Φ_N/c² − ε_c) deja de ser pequeña y el ansatz del tratado
    deja de ser una corrección.

    'physical': ε̄_c = α₀⁻¹ f^{−3/2} (1+z)^{9/2} = target ⟹
                1 + z = (target · f^{3/2} / α₀⁻¹)^{2/9}   (analítico)
    'comoving': ε̄_c = α₀⁻¹ f^{−3/2} constante ⟹ inf si es < target."""
    check_alpha0_inv(alpha0_inv)
    if rho_c_mode not in RHO_C_MODES:
        raise ValueError(rho_c_mode)
    if alpha0_inv == 0.0:
        return float("inf")
    eps0 = alpha0_inv * rho_c_over_mean ** -1.5
    if rho_c_mode == "comoving":
        return 0.0 if eps0 >= target else float("inf")
    return float((target / eps0) ** (2.0 / 9.0) - 1.0)


def z_where_mu_minus_one_reaches(k_hMpc: float, theta, alpha0_inv: float,
                                 rho_c_mode: str = "physical",
                                 level: float = 1.0, z_max: float = 3000.0,
                                 n_grid: int = 4001) -> float:
    """Menor z ∈ [0, z_max] con µ(k,z) − 1 ≥ level (interpolado en
    log(1+z) sobre una malla fina); inf si no se alcanza. Con level = 1
    marca dónde la ecuación de Poisson efectiva se modifica en O(1) —
    el límite duro de la extrapolación lineal para ese k."""
    if level <= 0:
        raise ValueError("level debe ser > 0")
    lnopz = np.linspace(0.0, np.log1p(z_max), int(n_grid))
    a = np.exp(-lnopz)
    m1 = mu_minus_one_cronos(k_hMpc, a, theta, alpha0_inv, rho_c_mode)
    hit = np.nonzero(m1 >= level)[0]
    if hit.size == 0:
        return float("inf")
    i = int(hit[0])
    if i == 0:
        return 0.0
    # interpolación lineal en log(1+z) entre el último punto por debajo
    # y el primero por encima
    x0, x1 = lnopz[i - 1], lnopz[i]
    y0, y1 = m1[i - 1], m1[i]
    x = x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    return float(np.expm1(x))


def background_above_rho_c_z(rho_c_over_mean: float = RHO_C_OVER_MEAN
                             ) -> float:
    """Cierre 'physical': z a partir del cual el PROPIO fondo supera el
    umbral ρ_c = f·ρ̄_m(0): (1+z)³ = f. Con f = 200, z ≈ 4.85 — el
    universo medio queda «por encima del umbral de colapso», lo que
    muestra por qué el criterio «f × la media» es, por construcción,
    relativo a la media de cada época (cierre 'comoving')."""
    return float(rho_c_over_mean ** (1.0 / 3.0) - 1.0)


def mu_table(theta, alpha0_inv: float, k_grid, z_grid,
             rho_c_mode: str = "comoving") -> np.ndarray:
    """Tabla (len(z_grid), len(k_grid)) de µ − 1."""
    k_grid = np.asarray(k_grid, float)
    out = np.empty((len(z_grid), len(k_grid)))
    for i, z in enumerate(z_grid):
        a = 1.0 / (1.0 + float(z))
        out[i] = mu_minus_one_cronos(k_grid, a, theta, alpha0_inv,
                                     rho_c_mode)
    return out
