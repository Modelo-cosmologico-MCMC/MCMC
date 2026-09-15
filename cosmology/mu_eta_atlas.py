"""Canal Atlas de µ/η — formas cerradas del límite cuasi-estático,
DERIVADAS de la Acción de Gea (9.1) + Término de Cronos (9.4) (frente 3).

validation/atlas_derivation.py re-deriva estos resultados desde la
acción (sympy) y el candado E1 (tests/test_mu_eta_atlas.py +
tests/test_mu_eta_atlas_lock.py) compara fórmula fijada contra
derivación. Este módulo no carga datos ni ajusta nada.

Parámetros del tratado: λ_K (ε_K ≡ λ_K − 1, constants.EPSILON_K), ξ y
α_a; GR = (1, 1, 0). Normalización: G_N ≡ G_B/ξ (la del Sello 9.3).
e ≡ aH/(ck) es el parámetro de la expansión sub-horizonte.

RESULTADOS (gauge unitario, materia = polvo, un modo escalar, QS):

  µ_Atlas(e)  = 2ξ / [(2ξ − α_a) + 3(3λ_K − 1) e²]
  η_Atlas(e)  = −(9e²λ² − 9e²λξ − 6e²λ + 3e²ξ + e² + 2λξ − 2ξ)
                 / [ξ (9e²λ − 3e² − 2λ + 2)]                 (λ ≡ λ_K)
  sub-horizonte estricto (e → 0): µ = 1/(1 − α_a/2ξ),  η = 1
  G_growth = G_B/(ξ − α_a/2) = G_local (H → 0 en las mismas ecuaciones)
      ⟹ CANCELACIÓN: µ medida respecto de la G de laboratorio vale
        1 + O(e²), η = 1 + O(e²). El canal Atlas no deja firma
        sub-horizonte en (µ, η) al orden dominante.
  G_cosmo = 2G_B/(3λ_K − 1)  (Sello 9.3, reproducido por la acción)
      ⟹ G_cosmo/G_local = (2ξ − α_a)/(3λ_K − 1)
         ≈ 1 − (3/2)ε_K − α_a/2   (ξ = 1, primer orden): BBN acota la
         COMBINACIÓN; el −1.8 % de (9.5) es el caso α_a ≪ ε_K.
  khronon: coef. cinético ∝ (3λ_K−1)/(λ_K−1) ⟹ no-fantasma ⟺ λ_K > 1
      (o λ_K < 1/3); c_s² = ξ(2ξ − α_a)(λ_K − 1)/[α_a(3λ_K − 1)] > 0
      ⟺ α_a < 2ξ (λ_K > 1) — la ventana declarada en (9.4).
  tensores: c_T² = ξ (GW170817 ⟹ ξ = 1 a 1e-15: ancla externa).

DOS PRECISIONES que el candado hace explícitas:
  (i) η_Atlas(e) es 0/0 en λ_K = 1 exacto: a e finito la fórmula da
      1/3 en GR estricto (el khronon es gauge en λ_K = 1 y el sistema QS
      degenera). El límite GR correcto toma e → 0 PRIMERO: η → 1.
  (ii) La cola de η NO es O(e²) a secas: η − 1 = e²·(A + ξB)/(2ξ(λ_K−1))
      + O(e⁴), con A = 9λ² − 9λξ − 6λ + 3ξ + 1 y B = 9λ − 3. El
      1/(λ_K − 1) — herencia de c_s² ∝ (λ_K − 1) — hace que el parámetro
      pequeño efectivo sea e/√(λ_K−1) ~ aH/(c_s k): con ε_K = 0.012 el
      coeficiente es ~170 y la ventana sub-horizonte se estrecha ×~9.
      Además el coeficiente QS de esa cola NO es el completo (la
      aproximación QS descarta ∂_t y velocidades del mismo orden e²).
      E3_Atlas (validation/atlas_tail_derivation.py) lo cierra al nivel
      del polo: con el sector de velocidades, η − 1 = [P_η/(λ_K−1) + Q_η]e²
      y µ_loc − 1 = [P_µ/(λ_K−1) + Q_µ]e², P_η = 3α_a/(2−α_a),
      P_µ = −P_η·p(2p−1)/3 (tail_pole_residues). El polo es FÍSICO (no un
      artefacto de la truncación, como #18 sugería al nivel medido): la
      truncación QS exagera su residuo (2 en vez de P_η) y pierde la
      dependencia en α_a. La cola de µ_loc SÍ tiene el polo (la forma QS
      1 − 3(3λ_K−1)e²/(2ξ−α_a) no lo tiene y tampoco es física). Las
      partes regulares Q son numéricas y dominan en α_a ≈ λ_K − 1.

Erratum candidata (v36, H.2.2): el apéndice escribe «c_s² = α/(2−α) → 0
cuando α → 0»; la derivación da c_s² = (2−α_a)(λ_K−1)/(α_a(3λ_K−1))
(ξ = 1): se anula cuando λ_K → 1 y a λ_K fijo DIVERGE cuando α_a → 0.
Lo que sí se mantiene de H.2.2: Λ_sc ~ M_P√α_a → 0. Precisión, no
retractación: la ventana de salud y la identificación Cronos = khronon
quedan intactas y ahora derivadas.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C

__all__ = [
    "ATLAS_STATUS", "G_cosmo_over_G_local", "G_cosmo_over_GB",
    "G_growth_over_GB", "G_local_over_GB", "c_T2", "eta_atlas_qs",
    "eta_tail_coefficient", "eta_tail_physical", "growth_index_matter_era",
    "is_healthy", "khronon_cs2", "khronon_kinetic_sign", "mu_atlas_qs",
    "mu_atlas_relative_to_local", "mu_atlas_subhorizon",
    "mu_local_tail_physical", "residues_ratio_first_order",
    "tail_pole_residues", "ppn_alpha1", "ppn_alpha2", "alpha_a_max_from_ppn",
]

ATLAS_STATUS = (
    "DERIVADO-NULO al orden dominante (frente 3, desenlace A publicado el "
    "13-sep-2026 en results/2026-09-13_mu_eta_atlas/): el offset "
    "sub-horizonte α_a/(2ξ) de µ_Atlas se cancela exactamente contra la "
    "renormalización de la G local (G_growth = G_local = G_B/(ξ − α_a/2), "
    "re-derivado desde la acción y confirmado por integración completa al "
    "nivel 5e-4); η_Atlas → 1. La firma sub-horizonte de (µ, η) es SOLO "
    "la del canal Cronos. Colas O(e²) (E3_Atlas, "
    "results/2026-09-13_mu_eta_atlas_tail/): los residuos del polo "
    "1/(λ_K−1) están DERIVADOS con el sector de velocidades (E3a = A: "
    "P_η = 3α_a/(2−α_a), P_µ = −P_η·p(2p−1)/3, escalera exacta; el polo es "
    "físico — horizonte de sonido del khronon — y la cola QS truncada lo "
    "sobreestima ×2(2−α_a)/(3α_a)); partes regulares Q numéricas. "
    "Confirmación numérica E3b = C bajo la regla congelada: pendiente de "
    "confirmación numérica independiente (las seis pendientes están al "
    "0.9–6.8 % de la escalera y ninguna es compatible con la QS truncada, "
    "pero el brazo e ≤ 0.02 falla el exponente log-log, 2.14 ∉ [1.9, 2.1], "
    "por curvatura O(e⁴) que el umbral no calibró; no se retoca). PPN "
    "(E4_Atlas, results/2026-09-15_mu_eta_atlas_ppn/): α₁ = −4α_a, "
    "α₂ = −α_a/2 + O(α_a²) en el límite khronométrico con el mapeo β = 0 "
    "verificado ⟹ α_a ≤ 8e-7 (giro solar; cotas transcritas) y "
    "G_cosmo/G_N − 1 → −(3/2)ε_K en forma limpia; PERO la escalera exacta "
    "da η − 1 = 4.6·e² en (λ_K, α_a) = (1.012, 1e-6), coeficiente que no se "
    "apaga con α_a → 0 (E4a = B, hallazgo estructural; la expectativa "
    "«colas inobservables» usaba solo el residuo del polo). Condicionado a "
    "la confirmación numérica independiente de la escalera, que sigue "
    "abierta (E4b = C: el brazo k/H0 = 66.7 no separa e² de e⁴)")


def _check_params(lamK: float, xi: float, alpha_a: float) -> None:
    if xi <= 0:
        raise ValueError("ξ debe ser > 0")
    if alpha_a < 0:
        raise ValueError("α_a debe ser ≥ 0")
    if 3 * lamK - 1 == 0:
        raise ValueError("λ_K = 1/3 es singular (3λ_K − 1 = 0)")


# --------------------------------------------------------------------
# µ, η en el límite cuasi-estático
# --------------------------------------------------------------------

def mu_atlas_qs(e, lamK: float, xi: float = 1.0, alpha_a: float = 0.0):
    """µ_Atlas(e) respecto de G_N ≡ G_B/ξ, e = aH/(ck)."""
    _check_params(lamK, xi, alpha_a)
    e = np.asarray(e, float)
    return 2.0 * xi / ((2.0 * xi - alpha_a) + 3.0 * (3.0 * lamK - 1.0) * e ** 2)


def eta_atlas_qs(e, lamK: float, xi: float = 1.0, alpha_a: float = 0.0):
    """η_Atlas(e) = Φ/Ψ (QS). Para λ_K = 1 exacto la forma cerrada es
    0/0 (precisión (i) del módulo): se devuelve el orden dominante,
    η ≡ 1, que es el límite e → 0."""
    _check_params(lamK, xi, alpha_a)
    e = np.asarray(e, float)
    if abs(lamK - 1.0) < 1e-13:
        return np.ones_like(e)
    e2 = e * e
    num = -(9 * e2 * lamK ** 2 - 9 * e2 * lamK * xi - 6 * e2 * lamK
            + 3 * e2 * xi + e2 + 2 * lamK * xi - 2 * xi)
    den = xi * (9 * e2 * lamK - 3 * e2 - 2 * lamK + 2)
    return num / den


def mu_atlas_subhorizon(lamK: float, xi: float = 1.0,
                        alpha_a: float = 0.0) -> float:
    """e → 0: µ = 1/(1 − α_a/2ξ) respecto de G_B/ξ — independiente de k
    y de λ_K (ε_K no entra al orden dominante)."""
    _check_params(lamK, xi, alpha_a)
    if alpha_a >= 2 * xi:
        raise ValueError("α_a ≥ 2ξ: fuera de la ventana de salud (c_s² ≤ 0)")
    return 1.0 / (1.0 - alpha_a / (2.0 * xi))


def mu_atlas_relative_to_local(e, lamK: float, xi: float = 1.0,
                               alpha_a: float = 0.0):
    """µ respecto de la G medida en laboratorio: µ_QS·(1 − α_a/2ξ) =
    1/[1 + 3(3λ_K−1)e²/(2ξ−α_a)] = 1 − 3(3λ_K−1)e²/(2ξ−α_a) + O(e⁴).
    Sin polo en λ_K = 1: la cancelación es exacta al orden dominante."""
    return mu_atlas_qs(e, lamK, xi, alpha_a) * (1.0 - alpha_a / (2.0 * xi))


def eta_tail_coefficient(lamK: float, xi: float = 1.0) -> float:
    """Coeficiente c tal que η_Atlas − 1 = c·e² + O(e⁴) en la forma QS
    TRUNCADA: c = (A + ξB)/(2ξ(λ_K − 1)), A = 9λ²−9λξ−6λ+3ξ+1, B = 9λ−3.
    Diverge en λ_K → 1: el parámetro pequeño efectivo es e/√(λ_K−1).

    NO ES FÍSICO: la truncación QS descarta ∂_t y velocidades del mismo
    orden e². El residuo verdadero del polo lo da tail_pole_residues
    (E3_Atlas); este coeficiente lo sobreestima en ×2(2−α_a)/(3α_a) y no
    depende de α_a. Se conserva solo como referencia de la corrección."""
    _check_params(lamK, xi, 0.0)
    if abs(lamK - 1.0) < 1e-13:
        return float("inf")
    A = 9 * lamK ** 2 - 9 * lamK * xi - 6 * lamK + 3 * xi + 1
    B = 9 * lamK - 3
    return (A + xi * B) / (2.0 * xi * (lamK - 1.0))


# --------------------------------------------------------------------
# Cola física O(e²) — residuos del polo, CON el sector de velocidades
# (E3_Atlas, validation/atlas_tail_derivation.py; ξ = 1)
# --------------------------------------------------------------------

def _check_xi_one(xi: float) -> None:
    if abs(xi - 1.0) > 1e-12:
        raise ValueError("la cola física está derivada solo para ξ = 1 "
                         "(GW170817); frontera declarada")


def tail_pole_residues(alpha_a: float, xi: float = 1.0) -> dict:
    """Residuos del polo 1/(λ_K − 1) de las colas O(e²), derivados de la
    escalera exacta del modo creciente sobre el sistema lineal completo:

        η − 1     = [P_η/(λ_K−1) + Q_η(α_a) + O(λ_K−1)]·e²,
        µ_loc − 1 = [P_µ/(λ_K−1) + Q_µ(α_a) + O(λ_K−1)]·e²,

        P_η = 3α_a/(2 − α_a),   P_µ = −P_η · p(2p−1)/3,

    con p el índice de crecimiento en λ_K = 1, p(p+½) = 3/(2−α_a) (GR:
    p = 1). Las partes regulares Q son numéricas (artefacto E3). El polo es
    físico: el parámetro pequeño es aH/(c_s k) (horizonte de sonido del
    khronon)."""
    _check_params(1.0, xi, alpha_a)
    _check_xi_one(xi)
    if alpha_a <= 0.0 or alpha_a >= 2.0:
        raise ValueError("0 < α_a < 2 (ventana de salud con ξ = 1)")
    p = growth_index_matter_era(1.0, 1.0, alpha_a)
    P_eta = 3.0 * alpha_a / (2.0 - alpha_a)
    P_mu = -P_eta * p * (2.0 * p - 1.0) / 3.0
    return {"P_eta": P_eta, "P_mu_local": P_mu, "p_at_lamK_1": p,
            "ratio_P_mu_over_P_eta": -p * (2.0 * p - 1.0) / 3.0,
            "qs_over_physical_eta": 2.0 * (2.0 - alpha_a) / (3.0 * alpha_a)}


def eta_tail_physical(e, lamK: float, alpha_a: float, xi: float = 1.0):
    """Residuo del polo de la cola de η, en la variable física:
    η − 1 = (3/2)·(aH/(c_s k))²·[1 + O(λ_K−1)] = (3/2) e²/c_s².

    Devuelve SOLO el residuo del polo (término dominante en λ_K − 1); la
    parte regular Q_η(α_a) no está en forma cerrada y NO es despreciable:
    en (λ_K, α_a) = (1.05, 0.3) el coeficiente completo de e² es 17.07
    frente a 10.59 del polo, y en α_a ≈ λ_K − 1 (p. ej. 0.012) la parte
    regular domina. Los coeficientes completos por punto viven en el
    artefacto E3 (results/2026-09-13_mu_eta_atlas_tail/). ξ = 1."""
    _check_xi_one(xi)
    e = np.asarray(e, float)
    cs2 = khronon_cs2(lamK, 1.0, alpha_a)
    return 1.5 * e ** 2 / cs2


def mu_local_tail_physical(e, lamK: float, alpha_a: float, xi: float = 1.0):
    """Residuo del polo de la cola de µ respecto de la G local:
    µ_loc − 1 = −[p(2p−1)/2]·(aH/(c_s k))²·[1 + O(λ_K−1)], p en λ_K = 1.
    Solo el término dominante en λ_K − 1; ξ = 1."""
    _check_xi_one(xi)
    e = np.asarray(e, float)
    cs2 = khronon_cs2(lamK, 1.0, alpha_a)
    p = growth_index_matter_era(1.0, 1.0, alpha_a)
    return -0.5 * p * (2.0 * p - 1.0) * e ** 2 / cs2


# --------------------------------------------------------------------
# Las tres G y el Contraste de los Residuos refinado
# --------------------------------------------------------------------

def G_growth_over_GB(lamK: float, xi: float = 1.0,
                     alpha_a: float = 0.0) -> float:
    """G_growth/G_B = 1/(ξ − α_a/2)."""
    _check_params(lamK, xi, alpha_a)
    return 1.0 / (xi - alpha_a / 2.0)


def G_local_over_GB(lamK: float, xi: float = 1.0,
                    alpha_a: float = 0.0) -> float:
    """G_local/G_B (Cavendish, H → 0) = 1/(ξ − α_a/2) = G_growth/G_B:
    la cancelación. Coincide con la G_N conocida de la clase
    khronométrica, G/(1 − α/2), en la normalización correspondiente."""
    return G_growth_over_GB(lamK, xi, alpha_a)


def G_cosmo_over_GB(lamK: float) -> float:
    """G_cosmo/G_B = 2/(3λ_K − 1) — el Sello de Newton (9.3), que la
    acción reproduce (etapa 2 de la derivación)."""
    _check_params(lamK, 1.0, 0.0)
    return 2.0 / (3.0 * lamK - 1.0)


def G_cosmo_over_G_local(lamK: float, xi: float = 1.0,
                         alpha_a: float = 0.0) -> float:
    """(2ξ − α_a)/(3λ_K − 1): la razón que BBN acota (frente 6)."""
    return G_cosmo_over_GB(lamK) / G_local_over_GB(lamK, xi, alpha_a)


def residues_ratio_first_order(eps_K: float = C.EPSILON_K,
                               alpha_a: float = 0.0) -> float:
    """G_cosmo/G_N ≈ 1 − (3/2)ε_K − α_a/2 (ξ = 1, primer orden). Con
    α_a = 0 reproduce la ec. (9.5) usada por cosmology/residues_test
    (1 − 1.5·ε_K = 0.982); en general BBN mide la combinación."""
    return 1.0 - 1.5 * eps_K - 0.5 * alpha_a


# --------------------------------------------------------------------
# PPN de marco preferido — límite khronométrico de Einstein-aether
# (E4_Atlas, validation/atlas_ppn_derivation.py). Mapeo: α_a = α,
# λ_K = 1 + λ, ξ = 1 ⇔ β = 0 (c_T² = 1/(1−β) en la forma covariante).
# --------------------------------------------------------------------

def ppn_alpha1(alpha_a: float, beta: float = 0.0) -> float:
    """α₁ = 4(α_a − 2β)/(β − 1); con β = 0 (ξ = 1): α₁ = −4α_a."""
    return 4.0 * (alpha_a - 2.0 * beta) / (beta - 1.0)


def ppn_alpha2(alpha_a: float, lamK: float, beta: float = 0.0) -> float:
    """α₂ = (α−2β)(αβ + 2αλ + α − β² − 3βλ − 3β − λ)/((α−2)(β−1)(β+λ)),
    λ ≡ λ_K − 1; con β = 0: α_a(2α_aλ + α_a − λ)/(λ(2 − α_a)) = −α_a/2 + O(α_a²)."""
    lam = lamK - 1.0
    if lam + beta == 0.0:
        raise ValueError("λ_K = 1 con β = 0: α₂ singular (khronon no dinámico)")
    num = (alpha_a - 2 * beta) * (alpha_a * beta + 2 * alpha_a * lam + alpha_a
                                   - beta ** 2 - 3 * beta * lam - 3 * beta - lam)
    return num / ((alpha_a - 2.0) * (beta - 1.0) * (beta + lam))


def alpha_a_max_from_ppn(bound_alpha1: float, bound_alpha2: float,
                         lamK: float) -> dict:
    """Cota sobre α_a (β = 0): α_a ≤ b₁/4 de |α₁| y la raíz de
    |α₂(α_a, λ_K)| = b₂ (bisección en (0, 1), donde |α₂| es monótona
    creciente para α_a ≪ 1). La más restrictiva gobierna."""
    from_a1 = bound_alpha1 / 4.0
    f = lambda x: abs(ppn_alpha2(x, lamK)) - bound_alpha2  # noqa: E731
    lo, hi = 1e-12, 1.0
    if f(hi) < 0:
        from_a2 = float("inf")
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        from_a2 = 0.5 * (lo + hi)
    amax = min(from_a1, from_a2)
    return {"alpha_a_max_from_alpha1": from_a1, "alpha_a_max_from_alpha2": from_a2,
            "alpha_a_max": amax,
            "governing": "alpha2" if from_a2 < from_a1 else "alpha1",
            "leading_order_alpha2_bound": 2.0 * bound_alpha2}


# --------------------------------------------------------------------
# Salud del khronon y sector tensorial
# --------------------------------------------------------------------

def khronon_kinetic_sign(lamK: float) -> float:
    """Signo del coeficiente cinético del khronon ∝ (3λ_K−1)/(λ_K−1):
    no-fantasma ⟺ λ_K > 1 o λ_K < 1/3."""
    _check_params(lamK, 1.0, 0.0)
    if abs(lamK - 1.0) < 1e-13:
        return 0.0                       # khronon no dinámico (GR)
    return float(np.sign((3 * lamK - 1) / (lamK - 1)))


def khronon_cs2(lamK: float, xi: float = 1.0, alpha_a: float = 0.0) -> float:
    """c_s² = ξ(2ξ − α_a)(λ_K − 1)/[α_a(3λ_K − 1)]; con α_a = 0 y λ_K ≠ 1
    diverge (acoplamiento fuerte, Λ_sc ~ M_P√α_a → 0 — H.2.2)."""
    _check_params(lamK, xi, alpha_a)
    if alpha_a == 0.0:
        return float("inf") if lamK != 1.0 else float("nan")
    return xi * (2 * xi - alpha_a) * (lamK - 1.0) / (alpha_a * (3 * lamK - 1.0))


def is_healthy(lamK: float, xi: float = 1.0, alpha_a: float = 0.0) -> bool:
    """Ventana (9.4): no-fantasma (λ_K > 1) y c_s² > 0 (0 < α_a < 2ξ)."""
    _check_params(lamK, xi, alpha_a)
    if alpha_a == 0.0:
        return False
    return lamK > 1.0 and 0.0 < alpha_a < 2.0 * xi


def c_T2(xi: float = 1.0) -> float:
    """c_T² = ξ para las ondas gravitacionales (etapa 6)."""
    if xi <= 0:
        raise ValueError("ξ debe ser > 0")
    return float(xi)


def growth_index_matter_era(lamK: float, xi: float = 1.0,
                            alpha_a: float = 0.0) -> float:
    """δ ∝ a^p en materia dominante con la G de crecimiento del canal:
    p(p + ½) = (3/2)·G_growth/G_cosmo = (3/2)(3λ_K−1)/(2ξ−α_a); GR: p = 1.
    Es el observable que el arnés E2 mide sin depender de la
    normalización de µ."""
    _check_params(lamK, xi, alpha_a)
    g = (3 * lamK - 1.0) / (2 * xi - alpha_a)
    return (-0.5 + np.sqrt(0.25 + 6.0 * g)) / 2.0
