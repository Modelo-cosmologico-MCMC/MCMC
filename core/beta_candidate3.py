"""Candidato 3 del cribado de β, «sentido del flujo» (bloque P1 del autor del 23-oct): las β canónicas del reloj (cierre de
Polchinski en dimensión cero, core/fokker_planck_beta.py) recorridas hacia el UV, τ < 0, es decir FPClosure(flow_sign=−1)
en el modo emergente. Brazo nuevo del cribado de PR-6, en fichero propio: la preinscripción del frente 2 (flow_sign = +1,
candado de tests/test_front2_fp_lock.py) no se toca.

El flujo de acoplos del reloj es dλ/dS = flow_sign·τ·β(λ): autónomo en S y sin dependencia de la trayectoria del campo. En
s = |τ|·S los cruces D = B² − 4C₀M₀² = 0 y sus cocientes S_{k+1}/S_k no dependen de τ. `flow_crossings` integra ese flujo en
s hasta tres cruces o s = s_max, con o sin el escalón del potencial (M₀² ← M₀²(0) en cada cruce, como
reset_mass_on_collapse del reloj), con el truncamiento cúbico o con la variante cuártica (beta_functions_quartic, E₄ como
cuarto acoplo, E₄(0) = 0; D sigue siendo el discriminante del Basal), que mide el coste del truncamiento. Hacia el UV la
ecuación de Polchinski completa es una difusión hacia atrás: el resultado vale para el séxtico truncado.

`cell_letter` aplica la letra del autor sobre los dos primeros cocientes: A, los dos en [7, 14]; B, tres cruces con algún
cociente fuera de [7, 14] pero dentro de [3, 30]; C, menos de tres cruces o algún cociente fuera de [3, 30]; INDETERMINADO,
puertas numéricas o truncamiento > 10 % en algún cociente. La letra global es la peor (orden A < B < INDETERMINADO < C).
Estatuto: cribado diagnóstico (E8, E13); ninguna letra decide λ ni el sentido del flujo, que es una decisión del autor.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from core.beta_candidates import integral_criterion
from core.fokker_planck_beta import beta_functions, beta_functions_quartic

LETTER_ORDER = ("A", "B", "INDETERMINADO", "C")      # de mejor a peor: la letra global es la peor


def discriminant(y) -> float:
    return float(y[1] ** 2 - 4.0 * y[2] * y[0])


def flow_crossings(lam0, flow_sign: int = -1, quartic: bool = False, reset_mass: bool = False, s_max: float = 50.0,
                   n_max: int = 3, a: float = 0.5, b: float = 0.5, rtol: float = 1e-10, atol_rel: float = 1e-14,
                   blowup: float = 1e8, method: str = "LSODA", rhs=None, n_grid: int = 2000) -> dict:
    """Cruces D = 0 (de D > 0 a D ≤ 0) del flujo dλ/ds = flow_sign·β(λ) desde lam0 (M₀², B, C₀), hasta n_max cruces, s_max,
    la divergencia de un acoplo (|λ_i| > blowup·escala_i) o C₀ ≤ 0. `rhs` sustituye a las β (pruebas con flujos de juguete)."""
    lam0 = np.asarray(lam0, dtype=float)
    y = np.concatenate([lam0, [0.0]]) if quartic else lam0.copy()
    scale = np.abs(np.concatenate([lam0, [lam0[2]]])) if quartic else np.abs(lam0)
    scale = np.where(scale > 0.0, scale, 1.0)
    if rhs is None:
        def rhs(_s, yy):
            if quartic:
                return flow_sign * beta_functions_quartic(yy[0], yy[1], yy[2], yy[3], a, b)
            return flow_sign * beta_functions(yy[0], yy[1], yy[2], a, b)

    def ev_D(_s, yy):
        return discriminant(yy)
    ev_D.terminal, ev_D.direction = True, -1

    def ev_blow(_s, yy):
        return float(np.max(np.abs(yy) / scale)) - blowup
    ev_blow.terminal, ev_blow.direction = True, 1

    def ev_C(_s, yy):
        return float(yy[2])
    ev_C.terminal, ev_C.direction = True, -1

    s, crossings, lam_at, n_resets = 0.0, [], [], 0
    first_segment = None
    stop, status = None, "ok"
    while True:
        sol = solve_ivp(rhs, (s, s_max), y, method=method, rtol=rtol, atol=atol_rel * scale, events=(ev_D, ev_blow, ev_C),
                        dense_output=first_segment is None)
        if first_segment is None:
            first_segment = sol
        if sol.status < 0:
            status, stop = "solver_failed", f"fallo del integrador: {sol.message}"
            break
        if len(sol.t_events[0]):
            s = float(sol.t_events[0][0])
            y = np.array(sol.y_events[0][0], dtype=float)
            crossings.append(s)
            lam_at.append(y.tolist())
            if len(crossings) >= n_max:
                stop = f"{n_max} cruces"
                break
            if reset_mass:
                y[0] = lam0[0]
                n_resets += 1
            else:
                # sin escalón, D sale del cruce por debajo de cero: un paso corto sin eventos evita volver a detectar el
                # mismo cruce por redondeo (el siguiente exige que D vuelva a ser positivo)
                nudge = solve_ivp(rhs, (s, s + 1e-9 * (1.0 + s)), y, method=method, rtol=rtol, atol=atol_rel * scale)
                s, y = float(nudge.t[-1]), np.array(nudge.y[:, -1], dtype=float)
            continue
        if len(sol.t_events[1]):
            s, stop = float(sol.t_events[1][0]), f"divergencia del flujo truncado (|λ_i| > {blowup:g}·escala)"
            y = np.array(sol.y_events[1][0], dtype=float)
            break
        if len(sol.t_events[2]):
            s, stop = float(sol.t_events[2][0]), "acoplos fuera del dominio (C₀ ≤ 0)"
            y = np.array(sol.y_events[2][0], dtype=float)
            break
        s, stop = float(sol.t[-1]), "s = s_max"
        y = np.array(sol.y[:, -1], dtype=float)
        break
    out = {"status": status, "stop": stop, "s_end": s, "crossings": crossings, "lam_at_crossings": lam_at,
           "lam_end": y.tolist(), "n_resets": n_resets, "quartic": quartic, "reset_mass": reset_mass, "flow_sign": flow_sign}
    if status == "ok" and first_segment is not None and first_segment.sol is not None:
        s_hi = crossings[0] if crossings else float(first_segment.t[-1])
        grid = np.linspace(0.0, s_hi, n_grid)
        Y = first_segment.sol(grid)
        D = Y[1] ** 2 - 4.0 * Y[2] * Y[0]
        out["criterion_i"] = {"D_monotone_decreasing_to_first_crossing": bool(np.all(np.diff(D) <= 1e-12 * abs(D[0]))),
                              "M0_sq_positive_at_first_crossing": bool(crossings and lam_at[0][0] > 0.0)}
        if Y[0][0] > 0.0:
            out["criterion_i_integral"] = integral_criterion(grid, Y[0], Y[1], Y[2])
    return out


def ratios(crossings: list[float]) -> list[float]:
    return [crossings[k + 1] / crossings[k] for k in range(len(crossings) - 1)]


def cell_letter(cubic: dict, tight: dict, quartic: dict, rules: dict) -> dict:
    """Letra de una celda (δ₀, sub-brazo) con las puertas numéricas y la del truncamiento (reglas de la preinscripción)."""
    for name, run in (("cúbico", cubic), ("cúbico con tolerancia estricta", tight), ("cuártico", quartic)):
        if run["status"] != "ok":
            return {"letter": "INDETERMINADO", "reason": f"puerta numérica: el flujo {name} no terminó ({run['stop']})"}
    c, t = cubic["crossings"], tight["crossings"]
    if len(c) != len(t) or cubic["stop"] != tight["stop"] or \
            any(abs(x / y - 1.0) > rules["convergence_rel"] for x, y in zip(c, t)):
        return {"letter": "INDETERMINADO", "reason": "puerta numérica: los cruces no convergen con la tolerancia estricta"}
    r, q = ratios(c)[:2], ratios(quartic["crossings"])[:2]
    lo_a, hi_a = rules["A_interval"]
    lo_b, hi_b = rules["C_outside"]
    out = {"n_crossings": len(c), "ratios": r, "ratios_quartic": q}
    if len(c) >= 3:
        trunc = [abs(y / x - 1.0) for x, y in zip(r, q)] if len(q) >= 2 else None
        out["truncation_rel"] = trunc
        if trunc is None or max(trunc) > rules["truncation_max_rel"]:
            return {**out, "letter": "INDETERMINADO", "reason": "truncamiento: la variante cuártica mueve algún cociente más de "
                                                                f"{rules['truncation_max_rel']:.0%} o no da tres cruces"}
    elif len(quartic["crossings"]) >= 3:
        return {**out, "letter": "INDETERMINADO", "reason": "truncamiento: el cúbico da menos de tres cruces y el cuártico tres"}
    if len(c) < 3:
        return {**out, "letter": "C", "reason": f"{len(c)} cruces (< 3)"}
    if any(not lo_b <= x <= hi_b for x in r):
        return {**out, "letter": "C", "reason": f"algún cociente fuera de [{lo_b:g}, {hi_b:g}]"}
    if all(lo_a <= x <= hi_a for x in r):
        return {**out, "letter": "A", "reason": f"los dos cocientes en [{lo_a:g}, {hi_a:g}]"}
    return {**out, "letter": "B", "reason": f"tres cruces; algún cociente fuera de [{lo_a:g}, {hi_a:g}] pero dentro de [{lo_b:g}, {hi_b:g}]"}


def worst(letters) -> str:
    return max(letters, key=LETTER_ORDER.index)
