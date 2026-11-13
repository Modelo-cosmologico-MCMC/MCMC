"""El Mecanismo del Discriminante y el flujo log-periódico (v35, Cap. 8).

Completa la Ley de la Década (Prop. 8.1, ya en constants.decade_thresholds)
con el mecanismo que la produce.

Discriminante basal y espinodal (Def. 8.4):

    D(S) ≡ B²(S) − 4·C0(S)·M0²(S);   la espinodal es el lugar D = 0.

Estructura tipo KLS del flujo radial (Prop. 8.5), con x = ρ²:

    dx/dσ = −(2·C0·x/G)·(x − x+)(x − x−),   x± = (B ± √D)/(2C0)

Los puntos fijos se FUSIONAN en la espinodal (D = 0) y se complejifican
para D < 0: x± = xR ± iΩ con xR = B/(2C0), Ω = √(−D)/(2C0) — el
mecanismo canónico de aniquilación de puntos fijos (KLS). En régimen
D < 0 el flujo es log-periódico (walking) con periodo (ec. 8.4):

    Δσ_walk ≃ (G/(2·C0·xR))·(π/Ω)

Exponente de Victoria (Def. 8.3): la cascada geométrica con razón λ es
invariancia de escala discreta por exponentes críticos complejos ±i·s0,
con λ = e^{π/s0}; para λ = 10, s0 = π/ln(10) ≈ 1.3644 (ec. 8.2).

ESTATUTO (Obs. 8.6-8.7): la metastabilidad (3.3) implica D(S0) > (4/3)·
C0·M0² > 0; la cascada exige que el flujo de acoplos hunda D(S) bajo
cero entre colapsos — el cruce de la espinodal (el CRUCE DE VICTORIA)
dispara cada colapso. λ = 10 es CALIBRADO; derivarlo de las funciones de
flujo es el frente abierto nº 2. Este módulo expone el mecanismo, no lo
resuelve.

EL CRUCE DE VICTORIA DEL ESTADO OCUPADO ES UN RETORNO (revisión del
10-nov-2026, hallazgo 1; comprobación interna, E8). Con r ≡ B²/(4·C0·M0²)
(D = 4·C0·M0²·(r − 1)) y E0 ≡ M0³/√C0, el vacío exterior vale

    V(x+) = −E0·t(r),   t(r) = r^{3/2}/3 − √r/2 + (r − 1)^{3/2}/3

(`outer_vacuum_energy`). Tres regímenes: para r > 4/3, x+ es el vacío
verdadero y hay Tensión Primordial T0 = E0·t(r) (t(9/4) = c̄ con las formas
por defecto); para 1 < r < 4/3 los papeles se invierten (x+ es el falso
vacío y el origen el verdadero); para r < 1 (D < 0) el único mínimo es el
origen. En la espinodal D = 0 el vacío ocupado vale +E0/6, POR ENCIMA del
origen: llegar a ella desde el estado que dejó la salida de S0 exige que
los acoplos re-tensen el estado en W_ret = T0 + E0′/6 > T0
(`retension_cost`), contra la Monotonía (Teo. 4.5, que vale con acoplos
fijos), y lo que el cruce dispara es la vuelta del campo a ρ = 0
(`adiabatic_B_ramp`), no un colapso hacia fuera. Con un solo Potencial
Basal séxtico el Cruce de Victoria es un mecanismo de RETORNO; dónde vive
(reinicio del ciclo, gatillo con fuente, un Basal por tramo o cascada
latente) es la decisión 1 del autor, que este módulo no toma.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .basal import B_BAR, C0_DEFAULT, M_BAR, scaled_params


def discriminant_basal(delta0: float, m_bar: float = M_BAR,
                       b_bar: float = B_BAR,
                       C0: float = C0_DEFAULT) -> float:
    """D = B² − 4·C0·M0² con el escalado (3.2) (Def. 8.4)."""
    p = scaled_params(delta0, m_bar, b_bar)
    return p["B"] ** 2 - 4.0 * C0 * p["M0_sq"]


def fixed_points(B: float, M0_sq: float, C0: float = C0_DEFAULT) -> tuple:
    """x± = (B ± √D)/(2C0) si D ≥ 0; para D < 0, (xR, Ω) complejos
    (Prop. 8.5)."""
    D = B ** 2 - 4.0 * C0 * M0_sq
    xR = B / (2.0 * C0)
    if D >= 0.0:
        r = np.sqrt(D) / (2.0 * C0)
        return ("real", xR + r, xR - r)
    return ("complex", xR, np.sqrt(-D) / (2.0 * C0))  # (xR, Ω)


def radial_flow_rhs(x: np.ndarray | float, B: float, M0_sq: float,
                    C0: float = C0_DEFAULT,
                    G: float = 1.0) -> np.ndarray | float:
    """dx/dσ = −(2·C0·x/G)·(x − x+)(x − x−) (ec. 8.3).

    El producto (x−x+)(x−x−) = C0x² − Bx + M0² dividido por C0; se
    evalúa directamente con el polinomio para cubrir también D < 0.
    """
    x = np.asarray(x, dtype=float)
    poly = x ** 2 - (B / C0) * x + M0_sq / C0
    out = -(2.0 * C0 * x / G) * poly
    return float(out) if np.ndim(out) == 0 else out


def walk_period(B: float, M0_sq: float, C0: float = C0_DEFAULT,
                G: float = 1.0) -> float:
    """Δσ_walk ≃ (G/(2·C0·xR))·(π/Ω) — el periodo del walking (ec. 8.4).

    Solo definido en régimen D < 0 (puntos fijos complejos); diverge al
    acercarse a la espinodal (Ω → 0).
    """
    kind, xR, omega = fixed_points(B, M0_sq, C0)
    if kind != "complex":
        raise ValueError("El walking requiere D < 0 (puntos fijos complejos)")
    return (G / (2.0 * C0 * xR)) * np.pi / omega


def s0_from_lambda(lam: float) -> float:
    """s0 = π/ln(λ) (Def. 8.3). Para λ=10: 1.3644 (ec. 8.2)."""
    if lam <= 1.0:
        raise ValueError("La razón de la cascada debe ser λ > 1")
    return float(np.pi / np.log(lam))


def lambda_from_s0(s0: float) -> float:
    """λ = e^{π/s0} (Def. 8.3)."""
    return float(np.exp(np.pi / s0))


def spinodal_crossing(D_of_S, S_lo: float, S_hi: float,
                      tol: float = 1e-12) -> float:
    """El Cruce de Victoria: S* con D(S*) = 0, por bisección (Obs. 8.6).

    D_of_S: callable S → D(S) (el flujo de acoplos dλi/dS = βi del
    corpus provee la trayectoria; aquí se localiza el cruce dado el
    perfil). Requiere cambio de signo en [S_lo, S_hi].
    """
    d_lo, d_hi = D_of_S(S_lo), D_of_S(S_hi)
    if d_lo * d_hi > 0.0:
        raise ValueError("Sin cambio de signo de D en el intervalo dado")
    while S_hi - S_lo > tol:
        S_mid = 0.5 * (S_lo + S_hi)
        if D_of_S(S_mid) * d_lo <= 0.0:
            S_hi = S_mid
        else:
            S_lo = S_mid
    return 0.5 * (S_lo + S_hi)


def metastability_bound(delta0: float, m_bar: float = M_BAR,
                        b_bar: float = B_BAR,
                        C0: float = C0_DEFAULT) -> bool:
    """Obs. 8.6: la metastabilidad (3.3) implica D(S0) > (4/3)·C0·M0²."""
    p = scaled_params(delta0, m_bar, b_bar)
    return discriminant_basal(delta0, m_bar, b_bar, C0) \
        > (4.0 / 3.0) * C0 * p["M0_sq"]


# ---------------------------------------------------------------------------
# El Cruce de Victoria del estado ocupado como retorno (revisión del 10-nov-2026)
# ---------------------------------------------------------------------------

# Los tres caminos de un solo acoplo hasta la espinodal D = 0 desde r0 > 1: solo baja B, solo sube M0² o solo sube C0
RETENSION_PATHS = ("B", "M0_sq", "C0")


def tension_ratio(B: float, M0_sq: float, C0: float = C0_DEFAULT) -> float:
    """r ≡ B²/(4·C0·M0²): la única combinación adimensional del Basal radial
    (la R de la nota computacional §3.40). D = 4·C0·M0²·(r − 1); r =
    b̄²/(4·C0·m̄²) = 9/4 con las formas por defecto."""
    if M0_sq <= 0.0 or C0 <= 0.0:
        raise ValueError("r exige M0² > 0 y C0 > 0")
    return float(B ** 2 / (4.0 * C0 * M0_sq))


def basal_energy_scale(M0_sq: float, C0: float = C0_DEFAULT) -> float:
    """E0 ≡ M0³/√C0: la escala de energía del Basal radial (V = E0·v(y; r)
    con y = x·√C0/M0)."""
    if M0_sq <= 0.0 or C0 <= 0.0:
        raise ValueError("E0 exige M0² > 0 y C0 > 0")
    return float(M0_sq ** 1.5 / np.sqrt(C0))


def tension_function(r: np.ndarray | float) -> np.ndarray | float:
    """t(r) = r^{3/2}/3 − √r/2 + (r − 1)^{3/2}/3, para r ≥ 1 (D ≥ 0).

    La Tensión Primordial del Basal radial es T0 = E0·t(r) cuando r > 4/3.
    Límites: t(1) = −1/6 (espinodal), t(4/3) = 0 (vacíos degenerados) y
    t(9/4) = c̄ = 0.840847 con las formas por defecto."""
    r = np.asarray(r, dtype=float)
    if np.any(r < 1.0):
        raise ValueError("Sin vacío exterior: r < 1 (D < 0), el único mínimo es el origen")
    out = r ** 1.5 / 3.0 - np.sqrt(r) / 2.0 + (r - 1.0) ** 1.5 / 3.0
    return float(out) if np.ndim(out) == 0 else out


def outer_vacuum_energy(r: np.ndarray | float) -> np.ndarray | float:
    """Energía del vacío exterior x+ en unidades de E0: V(x+)/E0 = −t(r).

    Negativa para r > 4/3 (x+ es el vacío verdadero), nula en r = 4/3 y
    POSITIVA para 1 ≤ r < 4/3: el vacío ocupado queda por encima del origen
    y vale +1/6 en la espinodal D = 0."""
    out = -np.asarray(tension_function(r))
    return float(out) if np.ndim(out) == 0 else out


def vacuum_regime(r: float) -> str:
    """El régimen del Basal radial según r: 'vacio_exterior_verdadero'
    (r > 4/3: hay Tensión Primordial y la transición cuesta abajo es la
    salida, origen → x+), 'vacio_exterior_falso' (1 ≤ r ≤ 4/3: papeles
    invertidos, la transición cuesta abajo es el retorno, x+ → origen) o
    'solo_origen' (r < 1, D < 0)."""
    if r > 4.0 / 3.0:
        return "vacio_exterior_verdadero"
    return "vacio_exterior_falso" if r >= 1.0 else "solo_origen"


def retension_cost(r0: float, camino: str = "B") -> float:
    """Coste de re-tensar, W_ret/T0: lo que los acoplos tienen que SUBIR la
    energía del vacío ocupado para llevarlo desde el estado que dejó la
    salida de S0 (r0 > 4/3, V = −T0) hasta la espinodal D = 0 (V = +E0′/6):

        W_ret = T0 + E0′/6   ⟹   W_ret/T0 = 1 + (E0′/E0)/(6·t(r0)) > 1

    con E0′ la escala del Basal en el cruce. Por los tres caminos de un
    solo acoplo: E0′/E0 = 1 si solo baja B, r0^{3/2} si solo sube M0² y
    r0^{−1/2} si solo sube C0 (1.198, 1.669 y 1.132 con r0 = 9/4). El
    retorno al origen libera después W_ret − T0 = E0′/6."""
    if r0 <= 4.0 / 3.0:
        raise ValueError("El estado de partida exige r0 > 4/3 (Tensión Primordial positiva)")
    if camino not in RETENSION_PATHS:
        raise ValueError(f"camino: {' | '.join(RETENSION_PATHS)}")
    scale = {"B": 1.0, "M0_sq": r0 ** 1.5, "C0": r0 ** -0.5}[camino]
    return float(1.0 + scale / (6.0 * tension_function(r0)))


def adiabatic_B_ramp(r0: float = B_BAR ** 2 / (4.0 * C0_DEFAULT * M_BAR ** 2), r_end: float = 0.81,
                     ramp_time: float = 3200.0, hold_time: float = 40.0, n_samples: int = 40001) -> dict:
    """Demostración dinámica del retorno: B baja despacio con el campo en el
    vacío ocupado x+, a M0² y C0 fijos, hasta cruzar la espinodal.

    Variables adimensionales: y = x·√C0/M0, v = V/E0 y s = σ·M0²/G, con
    v(y; r) = y/2 − (√r/2)·y² + y³/6 y el Flujo del Camino radial (Prop.
    8.5) dy/ds = −2y·(1 − 2√r·y + y²). √r (∝ B) baja linealmente de √r0 a
    √r_end < 1 en `ramp_time` y se mantiene `hold_time` más. El campo sigue
    al vacío ocupado, cuya energía SUBE desde −t(r0) hasta un máximo junto a
    +1/6 (el valor exacto en r = 1; una rampa de duración finita lo rebasa
    algo y lo alcanza con r algo menor que 1, por el frenado crítico junto
    a la fusión de puntos fijos), y cuando el vacío exterior desaparece el
    campo cae a ρ = 0. Integración LSODA con tolerancias 1e-10 / 1e-14."""
    if r0 <= 4.0 / 3.0 or not (0.0 < r_end < 1.0):
        raise ValueError("La rampa va de r0 > 4/3 a 0 < r_end < 1 (más allá de la espinodal)")
    if ramp_time <= 0.0 or hold_time < 0.0 or n_samples < 3:
        raise ValueError("ramp_time > 0, hold_time ≥ 0 y n_samples ≥ 3")
    a0, a1 = np.sqrt(r0), np.sqrt(r_end)

    def sqrt_r(s):
        return a0 + (a1 - a0) * np.minimum(s / ramp_time, 1.0)

    def rhs(s, y):
        return -2.0 * y * (1.0 - 2.0 * sqrt_r(s) * y + y * y)
    s = np.linspace(0.0, ramp_time + hold_time, n_samples)
    sol = solve_ivp(rhs, (0.0, ramp_time + hold_time), [a0 + np.sqrt(r0 - 1.0)], method="LSODA", t_eval=s,
                    rtol=1e-10, atol=1e-14, max_step=ramp_time / 200.0)
    if not sol.success:
        raise RuntimeError(f"La rampa no se integró: {sol.message}")
    y = np.clip(sol.y[0], 0.0, None)
    a = sqrt_r(s)
    v = y / 2.0 - a / 2.0 * y ** 2 + y ** 3 / 6.0
    i = int(np.argmax(v))
    t0 = float(tension_function(r0))
    return {"s": s, "r": a ** 2, "y": y, "V_over_E0": v, "r0": float(r0), "r_end": float(r_end),
            "ramp_time": float(ramp_time), "hold_time": float(hold_time),
            "V_start_over_E0": float(v[0]), "V_max_over_E0": float(v[i]), "r_at_max": float(a[i] ** 2),
            "V_at_spinodal_closed_form": 1.0 / 6.0, "rise_over_T0": float((v[i] - v[0]) / t0),
            "retension_cost_closed_form": retension_cost(r0, "B"),
            "y_final": float(y[-1]), "y_final_over_y_start": float(y[-1] / y[0]),
            "returns_to_origin": bool(y[-1] < 1e-9 * y[0]),
            "status": "comprobación interna (E8): con acoplos en movimiento V no es función de Lyapunov; el cruce D = 0 "
                      "del estado ocupado devuelve el campo al origen"}
