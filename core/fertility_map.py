"""El diagrama de fertilidad del Ciclo de Victoria — frente nº 4 (H.2.3, fig. H.2).

Cartografía Monte Carlo de la pregunta más ontológica del modelo: ¿cuán
genérico es que el retorno se gane? La fertilidad exige (ec. H.7)

    A = γR·ē/m̄² > 1   ⟺   ν > 0

con las constantes de forma (m̄, b̄, C0, ē) de orden uno (F.2) y γR
acotado por el Techo de Victoria: T0(δ0') = c̄·δ0'³ ≤ W_max implica
0 < γR ≤ γ_max (H.2.3, «esa ganancia no es libre»).

LO QUE ESTE MÓDULO HACE Y NO HACE: cartografía la fracción fértil del
espacio de parámetros O(1) y la frontera ν = 0 (γR* = m̄²/ē) — la figura
análoga al diagrama de fertilidad de la fig. H.2. NO decide el signo de
ν: el valor real de γR espera la microdinámica del reinicio (frente
abierto nº 4). El Silencio de Victoria sigue siendo posibilidad real.
"""

from __future__ import annotations

import numpy as np

from .basal import c_bar, quasi_cancellation_ok
from .victoria import fertility_condition, lydia_gain

# Región O(1) de las constantes de forma (F.2: m̄, b̄, ē = O(1), C0 > 0)
O1_RANGE = (0.5, 2.0)


def sample_shapes(n: int, seed: int = 0,
                  o1_range: tuple = O1_RANGE) -> dict:
    """Muestrea (m̄, b̄, C0, ē) uniformes en la región O(1), quedándose
    con los paisajes viables: casi-cancelación (3.3) y c̄ > 0."""
    rng = np.random.default_rng(seed)
    lo, hi = o1_range
    out = {"m_bar": [], "b_bar": [], "C0": [], "e_bar": []}
    trials = 0
    while len(out["m_bar"]) < n and trials < 200 * n:
        trials += 1
        m, b, c0, e = rng.uniform(lo, hi, size=4)
        # (3.3) exige b̄² > (16/3)C0m̄²; en la región O(1) obliga b̄ alto
        b = rng.uniform(lo, 2.0 * hi)  # b̄ puede rozar el límite superior
        if not quasi_cancellation_ok(m, b, c0):
            continue
        try:
            if c_bar(m, b, c0) <= 0.0:
                continue
        except ValueError:
            continue
        out["m_bar"].append(m); out["b_bar"].append(b)
        out["C0"].append(c0); out["e_bar"].append(e)
    return {k: np.array(v) for k, v in out.items()}


def delta_saturation(W_max: float, m_bar: float, b_bar: float,
                     C0: float) -> float:
    """δ_sat = (W_max/c̄)^{1/3} — el Techo de Victoria acota el residuo
    reinvertido: T0(δ') = c̄·δ'³ ≤ W_max (Lema 10.3 / H.2.3)."""
    return (W_max / c_bar(m_bar, b_bar, C0)) ** (1.0 / 3.0)


def fertility_fraction(n: int = 20000, seed: int = 0,
                       gamma_max: float = 3.0) -> dict:
    """Fracción fértil del espacio (formas O(1) viables × γR ∈ (0, γ_max]).

    γ_max es un PARÁMETRO DECLARADO de la cartografía: el tratado acota
    0 < γR ≤ γ_max por el Techo (H.2.3) pero no da su fórmula — el valor
    real espera la microdinámica del reinicio (frente nº 4). El default
    3.0 cubre con holgura el entorno O(1) de la frontera γR* = m̄²/ē.
    Devuelve la fracción, las muestras y la frontera ν=0 por muestra.
    """
    shapes = sample_shapes(n, seed=seed)
    n_eff = len(shapes["m_bar"])
    rng = np.random.default_rng(seed + 1)
    gamma_R = rng.uniform(0.0, gamma_max, size=n_eff)
    fertile = np.array([
        fertility_condition(g, m, e)
        for g, m, e in zip(gamma_R, shapes["m_bar"], shapes["e_bar"])
    ])
    frontier = shapes["m_bar"] ** 2 / shapes["e_bar"]  # γR* con ν = 0
    gains = np.array([
        lydia_gain(g, m, e)
        for g, m, e in zip(gamma_R, shapes["m_bar"], shapes["e_bar"])
    ])
    return {
        "fraction_fertile": float(fertile.mean()) if n_eff else float("nan"),
        "n_effective": n_eff,
        "shapes": shapes,
        "gamma_R": gamma_R,
        "gamma_frontier": frontier,
        "gains": gains,
        "fertile": fertile,
    }
