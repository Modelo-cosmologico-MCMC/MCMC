"""β a un lazo del potencial séxtico del Basal en dimensión d — candidato 2 del cribado de β del frente 2.

ORIGEN. Revisión del autor del 30-sep (Ap. L, Ruta II, «grupo de renormalización entrópico en S»): las β a un lazo de un
potencial ½M₀²φ² − (B/4)φ⁴ + (C₀/6)φ⁶ en dimensión d = 1, 2, 3 dependen de d por construcción y deben pasar el mismo cribado
que el candidato 1 (core/beta_candidates.py), con el criterio (i) escrito como desigualdad integral y los umbrales T₀ = 0 y
D = 0 publicados. Este módulo es la derivación cerrada; no ejecuta el reloj.

EL OBJETO. Potencial O(N) en x = |φ|² con la convención del reloj: u(x) = ½M·x − (B/4)·x² + (C/6)·x³ (c1 = M/2, c2 = −B/4,
c3 = C/6). N = 2 son las dos componentes (φ_M, φ_E) del reloj; N = 1 es la lectura literal «un escalar». Aproximación de
potencial local (LPA), variables adimensionales y tiempo de RG hacia el infrarrojo s = ln(Λ/k). Dos esquemas declarados:

  Wetterich con regulador de Litim, con c_d = 1/((4π)^{d/2} Γ(d/2 + 1)) absorbido en la normalización (el «a un lazo» que
  reproduce en d = 4 el coeficiente de libro ∂_t λ = 3λ²/(16π²)):
      ∂_s u = d·u − (d − 2)·x·u' − [1/(1 + 2u' + 4x·u'') + (N − 1)/(1 + 2u')]
  Polchinski (Wilson), normalizado con coeficientes a, b > 0:
      ∂_s u = d·u − (d − 2)·x·u' + a·Δu − b·(∇u)²,   Δu = 2N·u' + 4x·u'',   (∇u)² = 4x·u'²
  El Polchinski con N = 2 y sin los términos de escala ES el flujo canónico del reloj (core.fokker_planck_beta.beta_functions).

β (desarrollo a O(x³) en el origen, el truncamiento séxtico del reloj; derivadas con álgebra simbólica y fijadas por los
tests contra los coeficientes de Taylor del lado derecho exacto):
  Wetterich:  dM/ds = 2M − 2(N+2)·B/(1+M)²
              dB/ds = (4−d)·B + 4(N+8)·B²/(1+M)³ − 4(N+4)·C/(1+M)²
              dC/ds = (6−2d)·C + 12(N+14)·B·C/(1+M)³ − 6(N+26)·B³/(1+M)⁴
  Polchinski: dM/ds = 2M − 2(N+2)·a·B − 2b·M²
              dB/ds = (4−d)·B − 8b·M·B − 4(N+4)·a·C
              dC/ds = (6−2d)·C − 6b·B² − 12b·M·C

LO QUE DECIDE EL CRUCE. D = B² − 4CM = 4CM(R − 1) con R = B²/(4CM). R es invariante bajo toda reescala (dimensión canónica,
dimensión anómala, normalización del campo y del potencial): los términos de escala se cancelan en d ln R/ds, así que d no
entra en el flujo INSTANTÁNEO de R, solo en la trayectoria. Formas cerradas:
  Polchinski: d ln R/ds = (8aC/B)·[(N+2)R − (N+4)] + 2bM·(12R − 1)
  Wetterich:  d ln R/ds = 8C/(B(1+M)⁴) · F(R, M; N),
              F = (N+2)R − (N+4)(1+M)² − 48RM − (N+50)RM² + 12(N+26)R²M²

TEOREMA DE SIGNO (fijado por los tests). Con B, C > 0:
  * Polchinski: R > (N+4)/(N+2) y M > 0 ⟹ d ln R/ds > 0 para todo a, b > 0 y toda d (N = 2: R > 3/2).
  * Wetterich: R > R_c(N) = (13N + 74)/(12(N + 2)) ⟹ F > 0 para TODO M real (el discriminante de F en M es
    −4R²(N+26)(12(N+2)R − (13N+74)) < 0 y el coeficiente de M² es positivo) y toda d (N = 2: R_c = 25/12; N = 1: 29/12).
CONSECUENCIA para las formas por defecto (m̄ = 1, b̄ = 3, C₀ = 1 ⟹ R₀ = 9/4) y el campo del reloj (N = 2): R crece en los dos
esquemas y en toda d, D no puede cruzar cero y el criterio (i) en forma integral, ∫[2β_B/B − β_C/C − β_M/M]dS = −ln R₀ =
−0.811 antes de M₀² = 0, no se alcanza (la integral es positiva). Con un escalar (N = 1) existe una ventana de masa en la que
R decrece (en R = 9/4, m̃² ∈ (0.0200, 0.0577)); entrar en ella exige que m̃² crezca, 2m̃² > 6B̃/(1+m̃²)², lo que el escalado (3.2)
(M₀² ~ δ₀², B ~ δ₀) no da: la inestabilidad de masa llega antes.

Estatuto: derivación (E8) con esquema, regulador, truncamiento y N declarados; no decide λ ni la ley; no es una corrida del reloj.
"""

from __future__ import annotations

import math

import numpy as np

SCHEMES = ("wetterich_litim", "polchinski")


def c_d(d: float) -> float:
    """Factor de lazo del regulador de Litim: c_d = 1/((4π)^{d/2} Γ(d/2 + 1)) (absorbido en la normalización)."""
    return 1.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0 + 1.0))


def ratio_R(M, B, C):
    """R = B²/(4CM): D = 4CM(R − 1); T₀ = 0 en R = 4/3 (b̄² = (16/3)C₀m̄²) y la espinodal D = 0 en R = 1."""
    return np.asarray(B, dtype=float) ** 2 / (4.0 * np.asarray(C, dtype=float) * np.asarray(M, dtype=float))


def beta_wetterich(M: float, B: float, C: float, d: float, N: int = 2) -> np.ndarray:
    """(dM/ds, dB/ds, dC/ds) del esquema de Wetterich con regulador de Litim (LPA, adimensional, hacia el IR)."""
    p = 1.0 + M
    return np.array([
        2.0 * M - 2.0 * (N + 2) * B / p ** 2,
        (4.0 - d) * B + 4.0 * (N + 8) * B ** 2 / p ** 3 - 4.0 * (N + 4) * C / p ** 2,
        (6.0 - 2.0 * d) * C + 12.0 * (N + 14) * B * C / p ** 3 - 6.0 * (N + 26) * B ** 3 / p ** 4,
    ])


def beta_polchinski(M: float, B: float, C: float, d: float, N: int = 2, a: float = 0.5, b: float = 0.5,
                    scaling: bool = True) -> np.ndarray:
    """(dM/ds, dB/ds, dC/ds) del esquema de Polchinski normalizado (LPA, adimensional, hacia el IR). Con N = 2 y
    scaling=False es exactamente core.fokker_planck_beta.beta_functions (el flujo canónico del reloj)."""
    sc = np.array([2.0 * M, (4.0 - d) * B, (6.0 - 2.0 * d) * C]) if scaling else np.zeros(3)
    return sc + np.array([
        -2.0 * (N + 2) * a * B - 2.0 * b * M ** 2,
        -8.0 * b * M * B - 4.0 * (N + 4) * a * C,
        -6.0 * b * B ** 2 - 12.0 * b * M * C,
    ])


def lnR_flow(beta: np.ndarray, M: float, B: float, C: float) -> float:
    """d ln R/ds = 2β_B/B − β_C/C − β_M/M: el integrando del criterio (i) en forma integral."""
    return float(2.0 * beta[1] / B - beta[2] / C - beta[0] / M)


def F_wetterich(R, M, N: int = 2):
    """Factor de signo del flujo de ln R en Wetterich: d ln R/ds = 8C/(B(1+M)⁴)·F(R, M; N)."""
    R = np.asarray(R, dtype=float)
    M = np.asarray(M, dtype=float)
    return ((N + 2) * R - (N + 4) * (1.0 + M) ** 2 - 48.0 * R * M - (N + 50) * R * M ** 2
            + 12.0 * (N + 26) * R ** 2 * M ** 2)


def lnR_flow_polchinski_closed(M: float, B: float, C: float, N: int = 2, a: float = 0.5, b: float = 0.5) -> float:
    R = float(ratio_R(M, B, C))
    return 8.0 * a * C / B * ((N + 2) * R - (N + 4)) + 2.0 * b * M * (12.0 * R - 1.0)


def R_c_wetterich(N: int = 2) -> float:
    """Por encima de R_c(N) = (13N + 74)/(12(N + 2)), F(R, M; N) > 0 para todo M: R solo puede crecer."""
    return (13.0 * N + 74.0) / (12.0 * (N + 2.0))


def R_star_polchinski(N: int = 2) -> float:
    """Raíz de la parte a del flujo de ln R en Polchinski: por encima (y con M > 0) R solo puede crecer."""
    return (N + 4.0) / (N + 2.0)


def wetterich_window(R: float, N: int = 2):
    """Intervalo de m̃² en el que F(R, M; N) < 0 (R decrece), o None si no existe (R ≥ R_c)."""
    a2 = 12.0 * (N + 26) * R ** 2 - (N + 50) * R - (N + 4)
    a1 = -48.0 * R - 2.0 * (N + 4)
    a0 = (N + 2) * R - (N + 4)
    disc = a1 ** 2 - 4.0 * a2 * a0
    if disc <= 0.0 or a2 <= 0.0:
        return None
    r = sorted(((-a1 - math.sqrt(disc)) / (2.0 * a2), (-a1 + math.sqrt(disc)) / (2.0 * a2)))
    return (r[0], r[1])


def rhs_potential(x, M: float, B: float, C: float, d: float, N: int = 2, scheme: str = "wetterich_litim",
                  a: float = 0.5, b: float = 0.5):
    """Lado derecho EXACTO de la LPA para el potencial séxtico (sin truncar): sirve para fijar las β por sus coeficientes de
    Taylor en x = 0 (tests). Sin la constante c0 del potencial (no entra en ninguna β)."""
    x = np.asarray(x, dtype=float)
    u = 0.5 * M * x - 0.25 * B * x ** 2 + C / 6.0 * x ** 3
    up = 0.5 * M - 0.5 * B * x + 0.5 * C * x ** 2
    upp = -0.5 * B + C * x
    scale = d * u - (d - 2.0) * x * up
    if scheme == "wetterich_litim":
        return scale - (1.0 / (1.0 + 2.0 * up + 4.0 * x * upp) + (N - 1) / (1.0 + 2.0 * up))
    if scheme == "polchinski":
        return scale + a * (2.0 * N * up + 4.0 * x * upp) - b * 4.0 * x * up ** 2
    raise ValueError(f"esquema: {' | '.join(SCHEMES)}")


def _clean(v: float, tol: float = 1e-12) -> float:
    """El ruido de redondeo de la salida densa en s = 0 (|v| < tol) se publica como 0."""
    return 0.0 if abs(v) < tol else float(v)


def integrate_flow(scheme: str, lam0, d: float, N: int = 2, s_max: float = 40.0, a: float = 0.5, b: float = 0.5,
                   n_out: int = 400) -> dict:
    """Integra el flujo (M, B, C) desde lam0 hasta s_max o hasta un evento terminal (M = 0: inestabilidad de masa; D = 0;
    C = 0). Publica s de T₀ = 0 (R = 4/3) y de D = 0 (R = 1), la integral del criterio (i) y su fracción del valor exigido."""
    from scipy.integrate import solve_ivp

    if scheme not in SCHEMES:
        raise ValueError(f"esquema: {' | '.join(SCHEMES)}")

    def f(_s, y):
        if scheme == "wetterich_litim":
            return beta_wetterich(y[0], y[1], y[2], d, N)
        return beta_polchinski(y[0], y[1], y[2], d, N, a, b)

    def ev_mass(_s, y):
        return y[0]
    ev_mass.terminal, ev_mass.direction = True, -1

    def ev_D(_s, y):
        return y[1] ** 2 - 4.0 * y[2] * y[0]
    ev_D.terminal, ev_D.direction = True, -1

    def ev_T0(_s, y):
        return 3.0 * y[1] ** 2 - 16.0 * y[2] * y[0]            # R = 4/3 ⟺ 3B² = 16CM
    ev_T0.terminal, ev_T0.direction = False, -1

    def ev_C(_s, y):
        return y[2]
    ev_C.terminal, ev_C.direction = True, -1

    lam0 = np.asarray(lam0, dtype=float)
    sol = solve_ivp(f, (0.0, s_max), lam0, method="LSODA", rtol=1e-10, atol=1e-14, dense_output=True,
                    events=(ev_mass, ev_D, ev_T0, ev_C))
    s_end = float(sol.t[-1])
    grid = np.linspace(0.0, s_end, n_out)
    Y = sol.sol(grid)
    R0 = float(ratio_R(*lam0))
    with np.errstate(divide="ignore", invalid="ignore"):
        lnR = np.log(ratio_R(Y[0], Y[1], Y[2]) / R0)
    positive = Y[0] > 0.0
    I_min = float(np.nanmin(lnR[positive])) if positive.any() else float("nan")
    # ln R diverge cuando M₀² → 0⁺: se publica si crece monótonamente, no su máximo (que dependería del muestreo)
    monotone_up = bool(positive.sum() > 1 and np.all(np.diff(lnR[positive]) >= -1e-12))
    required = -math.log(R0)

    def first(ev):
        return float(sol.t_events[ev][0]) if len(sol.t_events[ev]) else None
    return {"scheme": scheme, "d": d, "N": N, "lam0": lam0.tolist(), "R0": R0, "s_end": s_end, "status": sol.message,
            "s_mass_zero": first(0), "s_D_zero": first(1), "s_T0_zero": first(2), "s_C_zero": first(3),
            "lam_end": [float(v) for v in sol.y[:, -1]],
            "R_end": float(ratio_R(*sol.y[:, -1])) if sol.y[0, -1] > 0 else None,
            "integral_criterion": {"required": required, "min_integral_while_M_positive": I_min,
                                   "R_monotone_increasing_while_M_positive": monotone_up,
                                   "fraction_of_required": (_clean(I_min / required) if required != 0 else None),
                                   "reached": bool(first(1) is not None)}}
