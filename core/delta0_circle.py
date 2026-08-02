"""El círculo de δ₀ — ¿converge el retorno de Victoria al valor del empalme?

La pregunta de la ronda 5 (prioridad nº 1 del análisis de la ronda 4):
el empalme C¹ (H.2.4, mass_program/B7) MIDE δ_H ≈ 0.0581 con las formas
fiduciales — el δ₀ que hace λ_Ad = λ_H = 0.130 y m_H ≈ 125 GeV sin la
masa medida como entrada. El Teo. 10.6 dice que el mapa de retorno de
Victoria tiene un atractor δ₀*. ¿Son el mismo número? Si el ciclo
converge por sí solo al valor que el empalme exige, el círculo
[imperfección → ciclo → atractor → curvatura sellada → Higgs] se cierra.

LO QUE EL CÁLCULO ESTABLECE (con el mapa lineal + Techo de H.2.3):
en TODA la región fértil (A > 1), desde cualquier δ inicial por encima
del suelo, el atractor es el Techo de Victoria:

    δ∞ = δ_sat = (W_max/c̄)^{1/3}    — independiente de γR y de δ_init.

Por tanto el círculo se cierra  ⟺  W_max = c̄·δ_H³  (≈ 1.65e-4 con las
formas fiduciales): una ECUACIÓN DE CONSISTENCIA que liga el Techo
(Lema 10.3 / Teo. 10.6) con el empalme C¹ (H.8). El tratado declara el
Techo pero NO asigna valor numérico a W_max, de modo que este cálculo
no cierra ni rompe el círculo por sí mismo: TRANSFIERE la pregunta al
trabajo máximo reinvertible W_max — condicional expuesto como parámetro,
nunca resuelto en silencio (F.2). Ambos desenlaces siguen abiertos.

FACTORES DE CONSISTENCIA si δ₀ = δ_H (frente al 0.012 de ε_Λ que el
v32 identificaba con δ₀ — identificación superada, regla canónica):

    T₀   escala ×(δ_H/0.012)³      ≈ 114   (Prop. 3.4: T₀ = c̄·δ₀³)
    m_θ² escala ×(δ_H/0.012)^{5/2} ≈ 52    (ec. 10.1: m_θ² = O(δ₀^{5/2}))

FERTILIDAD RE-CENTRADA: la condición de fertilidad ν > 0 ⟺ A > 1 (H.7)
NO depende de δ₀ — el barrido de core/fertility_map.py no cambia. Lo que
sí depende del paisaje es la CONTENCIÓN: que el Techo alcance el valor
del empalme, W_max ≥ c̄(formas)·δ_H(formas)³. landscape_scan cartografía
δ_H y ese W_max requerido sobre los paisajes fértiles O(1), cuantificando
la alternativa «formas no fiduciales» del desenlace de B7.
"""

from __future__ import annotations

import numpy as np

from .basal import (
    M_BAR, B_BAR, E_BAR, C0_DEFAULT, T0_analytic,
)
from .victoria import iterate_cycles, lydia_gain, memory_mode_mass_sq
from .fertility_map import delta_saturation, fertility_fraction

# λ_H del cierre del empalme (H.8 / convenio 12.1; = BETA3_CONVENIO_12_1
# de mass_program/B7 — core no importa mass_program, el valor se declara):
LAMBDA_H_SEAL = 0.130

STATUS_CIRCLE = (
    "condicional (F.2): el círculo se cierra ⟺ W_max = c̄·δ_H³; el "
    "tratado declara el Techo (Lema 10.3) pero no asigna valor a W_max "
    "— la ecuación de consistencia liga Teo. 10.6 con H.8 y transfiere "
    "la pregunta a W_max"
)


def delta0_H(m_bar: float = M_BAR, b_bar: float = B_BAR,
             C0: float = C0_DEFAULT,
             lambda_target: float = LAMBDA_H_SEAL) -> float:
    """δ_H = λ_H/√(b̄²−4C0m̄²) — el δ₀ que el cierre del empalme exige
    (H.8; espejo de mass_program.B7_empalme.delta0_required).

    Con las formas fiduciales (m̄=1, b̄=3, C0=1): δ_H = 0.130/√5 ≈ 0.0581.
    """
    disc = b_bar ** 2 - 4.0 * C0 * m_bar ** 2
    if disc <= 0.0:
        raise ValueError("Sin vacío sellado: D ≤ 0 (b̄² ≤ 4C0m̄²)")
    return lambda_target / float(np.sqrt(disc))


def W_max_required(delta_star: float, m_bar: float = M_BAR,
                   b_bar: float = B_BAR, C0: float = C0_DEFAULT) -> float:
    """El Techo que cierra el círculo en δ*: W_max = c̄·δ*³ = T₀(δ*).

    Es la ecuación de consistencia del círculo (Lema 10.3 ↔ Prop. 3.4):
    el trabajo reinvertible máximo debe coincidir con la Tensión
    Primordial del δ₀ del empalme. Fiducial: c̄·(0.0581)³ ≈ 1.65e-4.
    """
    return T0_analytic(delta_star, m_bar, b_bar, C0)


def attractor_analytic(gamma_R: float, W_max: float, *, delta_min: float,
                       m_bar: float = M_BAR, b_bar: float = B_BAR,
                       e_bar: float = E_BAR,
                       C0: float = C0_DEFAULT) -> float:
    """Atractor del mapa lineal + Techo (Teo. 10.6), en forma cerrada.

    - A ≤ 1 (región infértil): las iteradas decaen bajo el suelo →
      0.0 (Silencio de Victoria).
    - A > 1: el atractor es el Techo, δ_sat = (W_max/c̄)^{1/3} —
      salvo que el Techo quede BAJO el suelo (δ_sat < δ_min), en cuyo
      caso la vuelta siguiente no arranca → 0.0 (Silencio).
    """
    A = lydia_gain(gamma_R, m_bar, e_bar)
    if A <= 1.0:
        return 0.0
    d_sat = delta_saturation(W_max, m_bar, b_bar, C0)
    return d_sat if d_sat >= delta_min else 0.0


def attractor_numeric(gamma_R: float, W_max: float, delta_init: float, *,
                      delta_min: float, n_cycles: int = 200,
                      m_bar: float = M_BAR, b_bar: float = B_BAR,
                      e_bar: float = E_BAR,
                      C0: float = C0_DEFAULT) -> dict:
    """El atractor MEDIDO iterando core.victoria (verificación cruzada).

    Devuelve el punto final, la historia completa y el número de vueltas
    hasta estacionarse (la espiral del Teo. 10.6, contada).
    """
    d_sat = delta_saturation(W_max, m_bar, b_bar, C0)
    hist = iterate_cycles(delta_init, gamma_R, n_cycles,
                          delta_min=delta_min, delta_sat=d_sat,
                          m_bar=m_bar, e_bar=e_bar)
    # primera vuelta en que la historia se estaciona:
    settle = next((i for i in range(1, len(hist))
                   if hist[i] == hist[-1]), len(hist) - 1)
    return {"attractor": float(hist[-1]), "history": hist,
            "cycles_to_settle": int(settle)}


def consistency_factors(delta_star: float | None = None,
                        delta_inherited: float = 0.012,
                        m_bar: float = M_BAR, b_bar: float = B_BAR,
                        e_bar: float = E_BAR,
                        C0: float = C0_DEFAULT) -> dict:
    """Factores de escala si δ₀ = δ_H en vez del 0.012 heredado del v32.

    Calculados con las funciones reales (no solo las leyes de potencia):
        T₀(δ_H)/T₀(0.012)     = (δ_H/0.012)³      ≈ 114
        m_θ²(δ_H)/m_θ²(0.012) = (δ_H/0.012)^{5/2} ≈ 52
    Son las consecuencias a verificar del análisis de la ronda 4.
    """
    d_star = delta0_H(m_bar, b_bar, C0) if delta_star is None else delta_star
    r = d_star / delta_inherited
    t0_ratio = (T0_analytic(d_star, m_bar, b_bar, C0)
                / T0_analytic(delta_inherited, m_bar, b_bar, C0))
    mth_ratio = (memory_mode_mass_sq(d_star, m_bar, b_bar, e_bar, C0)
                 / memory_mode_mass_sq(delta_inherited, m_bar, b_bar,
                                       e_bar, C0))
    return {
        "delta_star": d_star,
        "delta_inherited": delta_inherited,
        "ratio": r,
        "T0_ratio": float(t0_ratio),          # = r³
        "m_theta_sq_ratio": float(mth_ratio),  # = r^{5/2}
    }


def closure_test(W_max: float, *, gamma_R: float, delta_min: float,
                 delta_init: float, rtol: float = 1e-3,
                 m_bar: float = M_BAR, b_bar: float = B_BAR,
                 e_bar: float = E_BAR, C0: float = C0_DEFAULT) -> dict:
    """¿Cierra el círculo con ESTE W_max? Compara el atractor medido
    (iterando Victoria) con el δ_H del empalme, para el mismo paisaje."""
    d_H = delta0_H(m_bar, b_bar, C0)
    num = attractor_numeric(gamma_R, W_max, delta_init,
                            delta_min=delta_min, m_bar=m_bar,
                            b_bar=b_bar, e_bar=e_bar, C0=C0)
    ana = attractor_analytic(gamma_R, W_max, delta_min=delta_min,
                             m_bar=m_bar, b_bar=b_bar, e_bar=e_bar, C0=C0)
    att = num["attractor"]
    closes = att > 0.0 and abs(att - d_H) <= rtol * d_H
    return {"delta_H": d_H, "attractor": att, "attractor_analytic": ana,
            "closes": closes, "cycles_to_settle": num["cycles_to_settle"],
            "W_max": W_max, "W_max_closing": W_max_required(d_H, m_bar,
                                                            b_bar, C0)}


def landscape_scan(n: int = 20000, seed: int = 0,
                   gamma_max: float = 3.0) -> dict:
    """δ_H y el W_max requerido sobre los paisajes fértiles O(1).

    Re-centra la cartografía de fertilidad en el círculo: la condición
    A > 1 no depende de δ₀ (H.7), así que lo que distingue a un paisaje
    fértil es cuánto Techo necesita para contener su δ_H. Devuelve las
    distribuciones y `containment_fraction(W)` = fracción de paisajes
    fértiles con W_req ≤ W (la curva que la microdinámica del reinicio
    tendrá que atravesar cuando fije W_max — frente abierto nº 4).
    """
    res = fertility_fraction(n, seed=seed, gamma_max=gamma_max)
    sh = res["shapes"]
    fertile = res["fertile"]
    m, b, c0 = (sh["m_bar"][fertile], sh["b_bar"][fertile],
                sh["C0"][fertile])
    d_H = np.array([delta0_H(mi, bi, ci) for mi, bi, ci in zip(m, b, c0)])
    w_req = np.array([W_max_required(di, mi, bi, ci)
                      for di, mi, bi, ci in zip(d_H, m, b, c0)])

    def containment_fraction(W: float) -> float:
        return float(np.mean(w_req <= W)) if len(w_req) else float("nan")

    return {
        "n_fertile": int(fertile.sum()),
        "fraction_fertile": res["fraction_fertile"],
        "delta_H": d_H,
        "W_required": w_req,
        "delta_H_percentiles": {
            p: float(np.percentile(d_H, p)) for p in (5, 25, 50, 75, 95)
        } if len(d_H) else {},
        "W_required_percentiles": {
            p: float(np.percentile(w_req, p)) for p in (5, 25, 50, 75, 95)
        } if len(w_req) else {},
        "containment_fraction": containment_fraction,
    }
