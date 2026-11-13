"""La cuarta ecuación, brazo «término angular»: puntos críticos interiores, collado de salida y umbral A_c (revisión
profunda del 10-nov-2026, hallazgo 2; PR R2 del plan de acción del 11-nov).

El término vive en el reloj S (`ClockConfig.angular_A`, decisión del autor del 23-oct): al Potencial Basal completo se
le suma la masa del modo de desequilibrio,

    V = V0(ρ, χ) + ½·μ²·χ²,    μ² = 4·A·(c̄/κ₊)·δ0²,    χ = (φ_M − φ_E)/√2.

Este módulo da las formas cerradas del paisaje que ese término deja en el Plano Dual. Con R(x) ≡ M0² − B·x + C0·x²
(x = ρ²), ∇V = 0 en el interior del cuadrante físico exige R(x) = 0 y χ = η/μ²: los puntos críticos interiores están
sobre los anillos ρ² = κ₊·δ0 y ρ² = κ₋·δ0, con χ̂ = ē·√δ0/μ̂² en variables escaladas (ρ = √δ0·ρ̂, μ² = μ̂²·δ0²). Con el
peso de la inclinación t = η·ρ₊/T₀ = ē·√(κ₊·δ0)/c̄:

- el del anillo κ₊ es el VACÍO, a arcsin(t/4A) de la diagonal dual (el θ* del registro, fila
  cuarta-ecuacion-termino-angular);
- el del anillo κ₋ es el COLLADO de salida de S0, a arcsin((t/4A)·√(κ₊/κ₋)) de la diagonal;
- el collado interior existe si A ≥ A_c = (t/4)·√(2κ₊/κ₋). Por debajo, el collado está sobre la frontera φ_E = 0 y
  la salida es por el polo de masa; por encima, la barrera del rayo del polo deja de ser un collado (allí ∂V/∂φ_E < 0)
  y es más alta que la del collado interior;
- con salida por el polo (A ≤ A_c) el vacío queda a arcsin√(κ₋/2κ₊) o más de la diagonal, sea cual sea δ0: 0.2735 rad
  (15.7°) con las formas por defecto.

LO QUE DICE Y LO QUE NO (comprobación interna, E8, sin letra). Con ½μ²χ² la salida por el polo de masa y la llegada
junto a la diagonal se excluyen: las cifras del 23-oct (θ* = 0.759 / 0.737 / 0.672 con A = 1) son correctas como
descenso desde el rayo θ = 0, que el reloj impone como forma declarada (`theta_nuc = 0.0`), pero con A = 1 ese rayo no
es el collado. El comportamiento por defecto del reloj no cambia; `saddle_ray_run` es una corrida de consistencia con
`theta_nuc` en el ángulo del collado. Qué mitad de la Prop. 3.5 se conserva (salida por el polo, llegada a la
diagonal o un segundo colapso angular) es la decisión 2 del autor, que este módulo no toma.
"""

from __future__ import annotations

import math

import numpy as np

from .basal import B_BAR, C0_DEFAULT, E_BAR, M_BAR, c_bar, kappa_plus, scaled_params
from .dual_plane import THETA_DIAGONAL
from .s_clock import ClockConfig, SClock, radial_landscape


def kappa_minus(m_bar: float = M_BAR, b_bar: float = B_BAR, C0: float = C0_DEFAULT) -> float:
    """κ₋ = [b̄ − √(b̄² − 4C0m̄²)]/(2C0): el anillo de la barrera del Basal radial, x₋ = κ₋·δ0 (Prop. 3.2)."""
    disc = b_bar ** 2 - 4.0 * C0 * m_bar ** 2
    if disc < 0.0:
        raise ValueError("Sin anillo de barrera: b̄² < 4C0m̄² (D < 0)")
    return (b_bar - math.sqrt(disc)) / (2.0 * C0)


def tilt_weight(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
                C0: float = C0_DEFAULT) -> float:
    """t = η·ρ₊/T₀ con la ley T₀ = c̄·δ0³ y ρ₊ = √(κ₊·δ0): el peso de la inclinación frente a la Tensión Primordial
    (0.105 / 0.192 / 0.453 para δ0 = 0.003 / 0.01 / 0.0554)."""
    return float(e_bar * math.sqrt(kappa_plus(m_bar, b_bar, C0) * delta0) / c_bar(m_bar, b_bar, C0))


def angular_mu2(delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR, C0: float = C0_DEFAULT) -> float:
    """μ² = 4·A·(c̄/κ₊)·δ0²: la masa del modo de desequilibrio (la misma que `SClock.mu2`)."""
    if A <= 0.0:
        raise ValueError("A > 0 (masa del modo de desequilibrio)")
    return float(4.0 * A * c_bar(m_bar, b_bar, C0) / kappa_plus(m_bar, b_bar, C0) * delta0 ** 2)


def critical_A(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
               C0: float = C0_DEFAULT) -> float:
    """A_c = (t/4)·√(2κ₊/κ₋): por encima, el collado de salida de S0 es interior al cuadrante y se despega del polo
    de masa (0.098 / 0.178 / 0.419 para δ0 = 0.003 / 0.01 / 0.0554)."""
    kp, km = kappa_plus(m_bar, b_bar, C0), kappa_minus(m_bar, b_bar, C0)
    return float(0.25 * tilt_weight(delta0, m_bar, b_bar, e_bar, C0) * math.sqrt(2.0 * kp / km))


def min_diagonal_distance_with_pole_exit(m_bar: float = M_BAR, b_bar: float = B_BAR, C0: float = C0_DEFAULT) -> float:
    """arcsin√(κ₋/2κ₊): lo más cerca de la diagonal que queda el vacío si la salida de S0 es por el polo de masa
    (A ≤ A_c). No depende de δ0 ni de ē: 0.2735 rad (15.7°) con las formas por defecto; entre 24.1° (b̄² = (16/3)·C0m̄²,
    el borde de la casi-cancelación) y 4.1° (b̄² = 100·C0m̄²) según las formas."""
    return float(math.asin(math.sqrt(kappa_minus(m_bar, b_bar, C0) / (2.0 * kappa_plus(m_bar, b_bar, C0)))))


def saddle_distance_for_vacuum_distance(d_vacuum: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                                        C0: float = C0_DEFAULT) -> float | None:
    """Distancia del collado a la diagonal cuando el vacío queda a `d_vacuum` de ella: los dos comparten χ = η/μ², así
    que sin(d_collado) = √(κ₊/κ₋)·sin(d_vacío), sin δ0, A ni ē. None si el collado ya no es interior (sin(d_collado) >
    1/√2, es decir d_vacío > arcsin√(κ₋/2κ₊)): entonces la salida es por el polo. Acercar el vacío a la diagonal aleja
    el collado del polo de masa: con el vacío a 0.05 rad, el collado está a 0.131 rad de la diagonal (0.654 rad del
    polo) con las formas por defecto."""
    s = math.sqrt(kappa_plus(m_bar, b_bar, C0) / kappa_minus(m_bar, b_bar, C0)) * math.sin(d_vacuum)
    return None if s > math.sin(THETA_DIAGONAL) else float(math.asin(s))


def _interior_point(delta0: float, A: float, kappa: float, m_bar: float, b_bar: float, e_bar: float, C0: float) -> dict | None:
    """El punto crítico interior del anillo ρ² = κ·δ0 (χ = η/μ²), o None si cae fuera del cuadrante físico."""
    eta = scaled_params(delta0, m_bar, b_bar, e_bar)["eta"]
    rho = float(math.sqrt(kappa * delta0))
    chi = float(eta / angular_mu2(delta0, A, m_bar, b_bar, C0))
    s = chi / rho                                  # sin(π/4 − θ) = χ/ρ
    if s > math.sin(THETA_DIAGONAL):
        return None
    theta = THETA_DIAGONAL - math.asin(s)
    return {"rho": rho, "chi": chi, "theta": theta, "distance_to_diagonal": THETA_DIAGONAL - theta,
            "phi": [rho * math.cos(theta), rho * math.sin(theta)]}


def interior_vacuum(delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
                    C0: float = C0_DEFAULT) -> dict | None:
    """El vacío interior: anillo ρ² = κ₊·δ0, a arcsin(t/4A) de la diagonal. None si t/4A > 1/√2: entonces el vacío
    sigue sobre la frontera φ_E = 0 (el polo de masa)."""
    return _interior_point(delta0, A, kappa_plus(m_bar, b_bar, C0), m_bar, b_bar, e_bar, C0)


def interior_saddle(delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
                    C0: float = C0_DEFAULT) -> dict | None:
    """El collado interior: anillo ρ² = κ₋·δ0, a arcsin((t/4A)·√(κ₊/κ₋)) de la diagonal. None si A < A_c: entonces el
    collado de salida está sobre la frontera φ_E = 0."""
    return _interior_point(delta0, A, kappa_minus(m_bar, b_bar, C0), m_bar, b_bar, e_bar, C0)


def potential(phi: np.ndarray, delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
              e_bar: float = E_BAR, C0: float = C0_DEFAULT) -> float:
    """V = V0 + ½μ²χ² en coordenadas (φ_M, φ_E): el mismo paisaje que integra el reloj con `angular_A = A`."""
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    x = float(phi[0]) ** 2 + float(phi[1]) ** 2
    chi = (float(phi[0]) - float(phi[1])) / math.sqrt(2.0)
    return (0.5 * p["M0_sq"] * x - 0.25 * p["B"] * x ** 2 + (C0 / 6.0) * x ** 3 - p["eta"] * chi
            + 0.5 * angular_mu2(delta0, A, m_bar, b_bar, C0) * chi ** 2)


def gradient(phi: np.ndarray, delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
             e_bar: float = E_BAR, C0: float = C0_DEFAULT) -> np.ndarray:
    """∇V = R(x)·Φ + (μ²χ − η)·(1, −1)/√2."""
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    q = np.asarray(phi, dtype=float)
    x = float(q @ q)
    chi = (q[0] - q[1]) / math.sqrt(2.0)
    radial = p["M0_sq"] - p["B"] * x + C0 * x ** 2
    return radial * q + (angular_mu2(delta0, A, m_bar, b_bar, C0) * chi - p["eta"]) * np.array([1.0, -1.0]) / math.sqrt(2.0)


def hessian(phi: np.ndarray, delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
            C0: float = C0_DEFAULT) -> np.ndarray:
    """∂²V = R(x)·1 + 2R′(x)·Φ⊗Φ + (μ²/2)·[[1, −1], [−1, 1]] (la inclinación es lineal y no entra)."""
    p = scaled_params(delta0, m_bar, b_bar)
    q = np.asarray(phi, dtype=float)
    x = float(q @ q)
    radial = p["M0_sq"] - p["B"] * x + C0 * x ** 2
    d_radial = -p["B"] + 2.0 * C0 * x
    return (radial * np.eye(2) + 2.0 * d_radial * np.outer(q, q)
            + 0.5 * angular_mu2(delta0, A, m_bar, b_bar, C0) * np.array([[1.0, -1.0], [-1.0, 1.0]]))


def pole_ray(delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
             C0: float = C0_DEFAULT) -> dict:
    """El rayo del polo de masa (θ = 0) con el término angular: falso vacío, barrera y punto de escape del rayo, y el
    signo de ∂V/∂φ_E = η/√2 − μ²ρ/2 en su barrera. Si es ≥ 0 la frontera sostiene el campo y esa barrera es el collado
    (salida por el polo); si es < 0 el campo baja hacia el interior y la barrera del rayo no es un collado."""
    mu2 = angular_mu2(delta0, A, m_bar, b_bar, C0)
    ld = radial_landscape(delta0, 0.0, m_bar, b_bar, e_bar, C0, mass_extra=0.5 * mu2)
    out = {"metastable": ld["metastable"], "rho_fv": ld.get("rho_fv"), "V_fv": ld.get("V_fv"),
           "rho_barrier": ld.get("rho_barrier"), "V_barrier": ld.get("V_barrier"), "rho_esc": ld.get("rho_esc"),
           "barrier_height": ld.get("barrier_height") if ld["metastable"] else None, "dV_dphiE_at_barrier": None}
    if ld["metastable"]:
        eta = scaled_params(delta0, m_bar, b_bar, e_bar)["eta"]
        out["dV_dphiE_at_barrier"] = eta / math.sqrt(2.0) - 0.5 * mu2 * ld["rho_barrier"]
    return out


def exit_route(delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
               C0: float = C0_DEFAULT) -> str:
    """Por dónde sale el campo de S0 si lo hace por el collado más bajo del paisaje: 'polo' (A ≤ A_c, collado sobre la
    frontera φ_E = 0) o 'collado interior' (A > A_c)."""
    return "collado interior" if A > critical_A(delta0, m_bar, b_bar, e_bar, C0) else "polo"


def barrier_ratio(delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
                  C0: float = C0_DEFAULT) -> dict:
    """Barrera del rayo del polo sobre barrera del collado interior, las dos medidas desde el falso vacío del
    cuadrante (que está sobre el rayo del polo): 2.58 / 2.34 / 1.78 con A = 1 para δ0 = 0.003 / 0.01 / 0.0554.

    La energía del collado tiene forma cerrada: V_c = V0_radial(x₋) − η²/(2μ²). `ratio` es None si no hay collado
    interior (A < A_c) o si el rayo del polo ya no tiene barrera."""
    ray = pole_ray(delta0, A, m_bar, b_bar, e_bar, C0)
    sad = interior_saddle(delta0, A, m_bar, b_bar, e_bar, C0)
    out = {"pole_ray_barrier": ray["barrier_height"], "saddle_barrier": None, "ratio": None, "V_saddle": None,
           "V_false_vacuum": ray["V_fv"]}
    if sad is None or not ray["metastable"]:
        return out
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    xm = kappa_minus(m_bar, b_bar, C0) * delta0
    V_saddle = (0.5 * p["M0_sq"] * xm - 0.25 * p["B"] * xm ** 2 + (C0 / 6.0) * xm ** 3
                - p["eta"] ** 2 / (2.0 * angular_mu2(delta0, A, m_bar, b_bar, C0)))
    out.update({"V_saddle": V_saddle, "saddle_barrier": V_saddle - ray["V_fv"],
                "ratio": ray["barrier_height"] / (V_saddle - ray["V_fv"])})
    return out


def saddle_landscape(delta0: float, A: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
                     C0: float = C0_DEFAULT) -> dict:
    """Todo lo anterior junto para un (δ0, A): t, A_c, vacío y collado interiores (ángulo y distancia a la diagonal),
    signatura del hessiano en cada uno y el cociente de barreras."""
    vac = interior_vacuum(delta0, A, m_bar, b_bar, e_bar, C0)
    sad = interior_saddle(delta0, A, m_bar, b_bar, e_bar, C0)

    def eigs(pt):
        return None if pt is None else np.linalg.eigvalsh(hessian(np.array(pt["phi"]), delta0, A, m_bar, b_bar, C0)).tolist()

    def gnorm(pt):
        return None if pt is None else float(np.linalg.norm(gradient(np.array(pt["phi"]), delta0, A, m_bar, b_bar, e_bar, C0)))
    ray = pole_ray(delta0, A, m_bar, b_bar, e_bar, C0)
    return {"delta0": delta0, "A": A, "t": tilt_weight(delta0, m_bar, b_bar, e_bar, C0),
            "A_c": critical_A(delta0, m_bar, b_bar, e_bar, C0), "exit_route": exit_route(delta0, A, m_bar, b_bar, e_bar, C0),
            "vacuum": vac, "saddle": sad, "hessian_eigenvalues_vacuum": eigs(vac), "hessian_eigenvalues_saddle": eigs(sad),
            "gradient_norm_vacuum": gnorm(vac), "gradient_norm_saddle": gnorm(sad),
            "pole_ray_barrier_is_a_saddle": None if ray["dV_dphiE_at_barrier"] is None else bool(ray["dV_dphiE_at_barrier"] >= 0.0),
            "barriers": barrier_ratio(delta0, A, m_bar, b_bar, e_bar, C0),
            "min_distance_with_pole_exit": min_diagonal_distance_with_pole_exit(m_bar, b_bar, C0)}


def saddle_ray_run(delta0: float, A: float = 1.0, **clock_kwargs) -> dict:
    """Corrida de CONSISTENCIA del reloj S con `theta_nuc` en el ángulo del collado interior en lugar de 0 (sin
    letra): qué queda de la rotación del 23-oct si la salida de S0 es por el collado más bajo.

    El rayo del collado pasa por él (allí χ = η/μ² anula también la derivada a lo largo del rayo), así que la barrera
    del rayo ES el collado y el reloj arranca en su punto de escape. Publica θ inicial y final, el θ* del vacío, la
    rotación que queda frente a la del rayo del polo (θ*), la monotonía de θ y las identidades del descenso. No cambia
    el comportamiento por defecto del reloj (θ_nuc = 0 sigue siendo la forma declarada)."""
    sad = interior_saddle(delta0, A)
    if sad is None:
        raise ValueError(f"Sin collado interior con A = {A:g} (A_c = {critical_A(delta0):.4g}): la salida es por el polo")
    clk = SClock(ClockConfig(delta0=delta0, angular_A=A, theta_nuc=sad["theta"], **clock_kwargs))
    out = clk.run()
    tr, chk = out["trajectory"], out["checks"]["descent"]
    theta = np.asarray(tr["theta"])
    theta_star = clk.true_vacuum_2d["theta"]
    return {"delta0": delta0, "A": A, "theta_saddle": sad["theta"], "theta_start": float(theta[0]),
            "theta_final": float(theta[-1]), "theta_star": float(theta_star),
            "rotation_done": float(theta[-1] - theta[0]),
            "rotation_left_from_saddle": float(theta_star - sad["theta"]),
            "rotation_from_pole_ray": float(theta_star),
            "fraction_of_pole_rotation_left": float((theta_star - sad["theta"]) / theta_star),
            "theta_max_drop": float(np.max(np.maximum.accumulate(theta) - theta)),
            "crosses_diagonal": bool(theta.max() > THETA_DIAGONAL),
            "rho_barrier_on_ray_over_rho_saddle": float(clk.land["rho_barrier"] / sad["rho"]),
            "S_final": float(tr["S"][-1]), "f_final": float(tr["f"][-1]), "T0_full_over_law": float(clk.T0 / clk.T0_law),
            "S_equals_f_pass": chk["S_equals_f_identity"]["pass"], "monotonia_pass": chk["monotonia_4_5"]["pass"],
            "finished": out["finished"], "stop_reason": out["stop_reason"], "steps": out["steps"],
            "derived_checks_failed": out["ledger"]["derived_checks_failed"],
            "status": "corrida de consistencia (E8), sin letra: θ_nuc en el collado es una lectura, no la forma declarada"}
