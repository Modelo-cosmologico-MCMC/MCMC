"""B8 — El WKB de primeros principios sobre el Potencial Basal (H.2.4).

El peso c_in del Funcional del Camino es la amplitud de tunelaje del
modo. En aproximación WKB (H.2.4):

    c(E) = e^{−W(E)},   W(E) = κ·∫ √(V0(ρ) − E) dρ
    (entre los puntos de retorno de la barrera del Basal)

Las tres verificaciones que el tratado declara y este módulo ejecuta:
1. SENSIBILIDAD EXPONENCIAL: una variación O(1) en la profundidad E
   produce órdenes de magnitud en c.
2. UNA SOLA ESCALA DE ACCIÓN κ debe cubrir los trece órdenes del rango
   m_t/m_ν — aquí se calcula la κ mínima que lo consigue y se construye
   la escalera con κ = 1.2·κ_min (elección declarada del barrido).
3. LA ESCALERA: la inversión c_i ∝ m_i ordena el espectro en
   profundidades monótonas — el más pesado el más somero (cerca del
   borde de la barrera), el neutrino el más profundo (enlace con el
   seesaw, Prop. 12.5).

La barrera es la del Basal radial (χ = 0; la inclinación es
subdominante, O(δ0^{7/2})): mínimo falso en ρ=0, techo en ρ_bar = √x−,
vacío verdadero en ρ+. Los modos viven a profundidad E ∈ (0, V_max).

ESTATUTO (Estado de H.2.4): derivado el MECANISMO (jerarquía exponencial
desde entrada lineal, escalera tensional); CONDICIONAL la ejecución de
los c_in con la correspondencia modo↔familia sellada. La comparación
con T_UNIVERSAL pasa de entrada a CONTRASTE: este módulo produce pesos
desde el potencial; la tabla calibrada es lo que deben reproducir cuando
la correspondencia esté fijada. Anclaje de la escalera: el modo más
pesado en E_top = 0.9·V_max (convención declarada del barrido).
"""

from __future__ import annotations

import numpy as np

from core.basal import scaled_params, M_BAR, B_BAR, C0_DEFAULT

# Espectro de la Tabla 12.1 (valores del corpus, GeV) para la escalera:
MASS_LADDER_GEV = {
    "t": 173.0, "b": 4.2, "tau": 1.77, "c": 1.28, "s": 0.096,
    "mu": 0.106, "d": 4.8e-3, "u": 2.3e-3, "e": 5.2e-4,
    "nu_tau": 6e-12,   # ~6 meV
}


def _potential_x(x: np.ndarray, delta0: float, m_bar: float,
                 b_bar: float, C0: float) -> np.ndarray:
    p = scaled_params(delta0, m_bar, b_bar)
    return 0.5 * p["M0_sq"] * x - 0.25 * p["B"] * x ** 2 + (C0 / 6.0) * x ** 3


def barrier(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
            C0: float = C0_DEFAULT) -> dict:
    """Geometría de la barrera del Basal: x− (techo), V_max, ρ_bar."""
    p = scaled_params(delta0, m_bar, b_bar)
    disc = p["B"] ** 2 - 4.0 * C0 * p["M0_sq"]
    if disc <= 0.0:
        raise ValueError("Sin barrera: D ≤ 0")
    x_minus = (p["B"] - np.sqrt(disc)) / (2.0 * C0)
    V_max = float(_potential_x(np.array([x_minus]), delta0, m_bar,
                               b_bar, C0)[0])
    return {"x_barrier": float(x_minus), "rho_barrier": float(np.sqrt(x_minus)),
            "V_max": V_max}


def W_integral(E: float, delta0: float, kappa: float = 1.0,
               m_bar: float = M_BAR, b_bar: float = B_BAR,
               C0: float = C0_DEFAULT, n_grid: int = 4000) -> float:
    """W(E) = κ·∫√(V0(ρ)−E)dρ entre los puntos de retorno de la barrera.

    E ∈ (0, V_max). Integración sobre malla densa de ρ con la raíz
    truncada en cero (los extremos son los puntos de retorno).
    """
    bar = barrier(delta0, m_bar, b_bar, C0)
    if not 0.0 < E < bar["V_max"]:
        raise ValueError(f"E fuera de la barrera: (0, {bar['V_max']:.3e})")
    # la región prohibida está en torno a ρ_bar; malla hasta pasado ρ+
    p = scaled_params(delta0, m_bar, b_bar)
    disc = p["B"] ** 2 - 4.0 * C0 * p["M0_sq"]
    x_plus = (p["B"] + np.sqrt(disc)) / (2.0 * C0)
    rho = np.linspace(1e-12, np.sqrt(x_plus), n_grid)
    V = _potential_x(rho ** 2, delta0, m_bar, b_bar, C0)
    integrand = np.sqrt(np.clip(V - E, 0.0, None))
    return float(kappa * np.trapezoid(integrand, rho))


def survival_weight(E: float, delta0: float, kappa: float,
                    **kw) -> float:
    """c(E) = e^{−W(E)} — el peso de supervivencia del modo."""
    return float(np.exp(-W_integral(E, delta0, kappa, **kw)))


def kappa_minimum(delta0: float, mass_ratio: float = None,
                  anchor_frac: float = 0.9, m_bar: float = M_BAR,
                  b_bar: float = B_BAR, C0: float = C0_DEFAULT) -> float:
    """La κ mínima que cubre el rango de masas con una sola escala.

    El rango exige ΔW = ln(m_max/m_min) entre el ancla (E = anchor_frac·
    V_max) y el fondo de la barrera (E → 0⁺):
        κ_min = ΔW / [W₁(0⁺) − W₁(E_ancla)],  W₁ ≡ W con κ=1.
    Para la Tabla 12.1: m_t/m_ν ≈ 3×10¹³ → ΔW ≈ 31 (trece órdenes).
    """
    if mass_ratio is None:
        mass_ratio = MASS_LADDER_GEV["t"] / MASS_LADDER_GEV["nu_tau"]
    bar = barrier(delta0, m_bar, b_bar, C0)
    W1_bottom = W_integral(1e-9 * bar["V_max"], delta0, 1.0,
                           m_bar=m_bar, b_bar=b_bar, C0=C0)
    W1_anchor = W_integral(anchor_frac * bar["V_max"], delta0, 1.0,
                           m_bar=m_bar, b_bar=b_bar, C0=C0)
    return float(np.log(mass_ratio) / (W1_bottom - W1_anchor))


def ladder(delta0: float, kappa: float | None = None,
           masses: dict | None = None, anchor_frac: float = 0.9,
           m_bar: float = M_BAR, b_bar: float = B_BAR,
           C0: float = C0_DEFAULT) -> dict:
    """La escalera de profundidades: E_i tal que c(E_i) ∝ m_i.

    Ancla el modo más pesado en E = anchor_frac·V_max y resuelve cada
    E_i por bisección de W(E_i) = W_ancla + ln(m_max/m_i). Devuelve
    {modo: E_i} más la κ usada (1.2·κ_min si no se da).
    """
    masses = dict(MASS_LADDER_GEV if masses is None else masses)
    bar = barrier(delta0, m_bar, b_bar, C0)
    if kappa is None:
        kappa = 1.2 * kappa_minimum(delta0, anchor_frac=anchor_frac,
                                    m_bar=m_bar, b_bar=b_bar, C0=C0)
    heaviest = max(masses, key=masses.get)
    E_anchor = anchor_frac * bar["V_max"]
    W_anchor = W_integral(E_anchor, delta0, kappa, m_bar=m_bar,
                          b_bar=b_bar, C0=C0)
    out = {}
    for name, m in masses.items():
        W_target = W_anchor + np.log(masses[heaviest] / m)
        lo, hi = 1e-9 * bar["V_max"], E_anchor
        W_lo = W_integral(lo, delta0, kappa, m_bar=m_bar, b_bar=b_bar, C0=C0)
        if W_target > W_lo:
            raise ValueError(
                f"κ={kappa:.3g} no cubre el rango hasta {name}: subir κ")
        for _ in range(200):  # bisección (W decrece con E)
            mid = 0.5 * (lo + hi)
            if W_integral(mid, delta0, kappa, m_bar=m_bar, b_bar=b_bar,
                          C0=C0) > W_target:
                lo = mid
            else:
                hi = mid
        out[name] = 0.5 * (lo + hi)
    return {"E": out, "kappa": float(kappa), "V_max": bar["V_max"],
            "anchor": heaviest}
