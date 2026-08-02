"""El flujo KLS integrado: la ley del walking, medida (frente E; v35 Cap. 8).

Medio paso cuantitativo del frente abierto nº 2 (Obs. 8.7: ¿λ = 10 es
física o convención de calibre?). Lo que este módulo HACE:

1. Integra el flujo radial (ec. 8.3) de forma EXACTA (cuadratura de la
   ODE separable) y con RK4 (verificación cruzada), y MIDE la ley del
   periodo del walking (ec. 8.4):

       Δσ_walk ≃ (G/(2·C0·xR))·(π/Ω)

   El «≃» de la ecuación queda cuantificado: el error del prefactor
   x ≈ xR se mide en función de Ω/xR.

2. Mide el exponente de divergencia en la espinodal: Δσ_walk ∝ |D|^ν
   con ν = −1/2 (Ω = √(−D)/2C0), ajustado sobre un barrido.

3. Ejecuta el CRUCE DE VICTORIA como bifurcación dinámica: con D(σ)
   hundiéndose (Obs. 8.6), el colapso se dispara DESPUÉS del cruce
   D = 0, con un retraso que escala como (ritmo de hundimiento)^(-1/3)
   — la ley de la bifurcación silla-nodo con deriva (teoría de
   bifurcaciones dinámicas; RESULTADO DEL PROGRAMA, no del tratado, y
   se declara como tal). Control negativo: sin hundimiento (D > 0
   constante) el flujo aparca en x+ y no hay colapso.

LO QUE NO HACE (estatuto, frente 2): convertir el periodo medido en un
λ derivado exige las funciones de flujo de los acoplos (dλi/dS), que el
tratado no da. El módulo valida cuantitativamente el mecanismo sobre el
que descansa la Def. 8.3 (λ = e^{π/s0}); no decide λ = 10.
"""

from __future__ import annotations

import numpy as np

from .decade import fixed_points, radial_flow_rhs, walk_period

STATUS_LAMBDA = (
    "condicional (frente 2, Obs. 8.7): el flujo valida las ec. 8.3-8.4 "
    "cuantitativamente; convertir el periodo del walking en λ = e^{π/s0} "
    "requiere las funciones de flujo de los acoplos, que el tratado no "
    "da — este módulo mide todo lo medible sin ellas"
)


def integrate_flow(x0: float, B: float, M0_sq, C0: float = 1.0,
                   G: float = 1.0, d_sigma: float = 1e-3,
                   n_steps: int = 100000,
                   stop_below: float | None = None) -> dict:
    """Integra dx/dσ = −(2C0x/G)(x−x+)(x−x−) con RK4 (ec. 8.3).

    M0_sq puede ser un número (acoplos estáticos) o un callable M0²(σ)
    (acoplos con deriva — el hundimiento de D del Obs. 8.6). Si
    stop_below se da, la integración termina al cruzarlo (colapso).
    """
    m_of = M0_sq if callable(M0_sq) else (lambda s: M0_sq)
    xs = np.empty(n_steps + 1)
    xs[0] = x0
    x = x0
    sigma = 0.0
    n_done = n_steps
    for i in range(n_steps):
        k1 = radial_flow_rhs(x, B, m_of(sigma), C0, G)
        k2 = radial_flow_rhs(x + 0.5 * d_sigma * k1, B,
                             m_of(sigma + 0.5 * d_sigma), C0, G)
        k3 = radial_flow_rhs(x + 0.5 * d_sigma * k2, B,
                             m_of(sigma + 0.5 * d_sigma), C0, G)
        k4 = radial_flow_rhs(x + d_sigma * k3, B,
                             m_of(sigma + d_sigma), C0, G)
        x = x + (d_sigma / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        sigma += d_sigma
        xs[i + 1] = x
        if stop_below is not None and x < stop_below:
            n_done = i + 1
            break
    return {"x": xs[:n_done + 1], "sigma_final": sigma,
            "d_sigma": d_sigma, "collapsed": stop_below is not None
            and x < stop_below}


def walking_time_measured(B: float, M0_sq: float, C0: float = 1.0,
                          G: float = 1.0, K: float = 50.0,
                          n_grid: int = 4001) -> float:
    """σ EXACTO para atravesar la ventana de walking [xR+KΩ, xR−KΩ],
    por cuadratura de la ODE separable (sin aproximaciones):

        σ = (G/(2C0Ω)) ∫ du / x(u),   x(u) = xR + Ω·tan(u)

    con u ∈ [−arctan K, +arctan K]. Es el flujo (8.3) integrado en
    forma cerrada salvo cuadratura — la referencia contra la que se
    mide el «≃» de la ec. 8.4."""
    kind, xR, omega = fixed_points(B, M0_sq, C0)
    if kind != "complex":
        raise ValueError("El walking requiere D < 0")
    if K * omega >= 0.9 * xR:
        raise ValueError(
            f"Ventana inválida: K·Ω = {K * omega:.3g} alcanza el polo "
            f"x = 0 (xR = {xR:.3g}); reduce K u Ω")
    u = np.linspace(-np.arctan(K), np.arctan(K), n_grid)
    integrand = 1.0 / (xR + omega * np.tan(u))
    return float(G / (2.0 * C0 * omega) * np.trapezoid(integrand, u))


def walking_time_analytic(B: float, M0_sq: float, C0: float = 1.0,
                          G: float = 1.0,
                          K: float | None = None) -> float:
    """La ley de la ec. 8.4. Con K=None, la forma del tratado
    (π/Ω, ventana infinita); con K finito, la misma aproximación
    x ≈ xR restringida a la ventana (factor 2·arctan(K)/π) — para
    comparar contra walking_time_measured con la MISMA ventana."""
    if K is None:
        return walk_period(B, M0_sq, C0, G)
    kind, xR, omega = fixed_points(B, M0_sq, C0)
    if kind != "complex":
        raise ValueError("El walking requiere D < 0")
    return float((G / (2.0 * C0 * xR)) * 2.0 * np.arctan(K) / omega)


def divergence_exponent(B: float, C0: float = 1.0, G: float = 1.0,
                        depths: np.ndarray | None = None) -> dict:
    """Δσ_walk ∝ |D|^ν cerca de la espinodal: ν ajustado sobre un
    barrido de profundidades D < 0 (esperado ν = −1/2, pues
    Ω = √(−D)/2C0 y Δσ ∝ 1/Ω)."""
    xR = B / (2.0 * C0)
    if depths is None:
        depths = np.geomspace(1e-8, 1e-4, 8) * (2.0 * C0 * xR) ** 2
    times = []
    for d in depths:
        M0_sq = (B ** 2 + d) / (4.0 * C0)   # D = −d
        omega = np.sqrt(d) / (2.0 * C0)
        K_eff = min(50.0, 0.5 * xR / omega)
        times.append(walking_time_measured(B, M0_sq, C0, G, K=K_eff))
    slope = float(np.polyfit(np.log(depths), np.log(times), 1)[0])
    return {"exponent": slope, "depths": np.asarray(depths),
            "times": np.array(times)}


def cruce_de_victoria(B: float, M0_sq0: float, rate: float,
                      C0: float = 1.0, G: float = 1.0,
                      d_sigma: float = 1e-2,
                      max_steps: int = 2000000) -> dict:
    """El Cruce de Victoria como bifurcación dinámica (Obs. 8.6).

    M0²(σ) = M0²(0) + rate·σ hunde D(σ) = B² − 4C0·M0²(σ) linealmente
    (representa el flujo de acoplos entre colapsos; las β-funciones
    reales son lo que el frente 2 no tiene). El flujo parte del vacío
    x+(0) y lo sigue adiabáticamente; en σ* (D = 0) los puntos fijos se
    aniquilan y x cae — el colapso. Devuelve σ*, el σ del colapso
    (x < xR/4) y el retraso Δσ = σ_colapso − σ*."""
    if rate <= 0.0:
        raise ValueError("El cruce requiere hundimiento: rate > 0")
    _, x_plus, _ = fixed_points(B, M0_sq0, C0)
    sigma_star = (B ** 2 / (4.0 * C0) - M0_sq0) / rate
    xR = B / (2.0 * C0)
    res = integrate_flow(x_plus, B, lambda s: M0_sq0 + rate * s, C0, G,
                         d_sigma=d_sigma, n_steps=max_steps,
                         stop_below=0.25 * xR)
    if not res["collapsed"]:
        raise RuntimeError("El flujo no colapsó: aumenta max_steps")
    sigma_collapse = res["sigma_final"]
    return {"sigma_star": float(sigma_star),
            "sigma_collapse": float(sigma_collapse),
            "delay": float(sigma_collapse - sigma_star),
            "x": res["x"], "d_sigma": d_sigma}


def delay_scaling(B: float, M0_sq0: float, rates: np.ndarray,
                  C0: float = 1.0, G: float = 1.0) -> dict:
    """El retraso del colapso tras el cruce escala como rate^ν con
    ν = −1/3 (silla-nodo con deriva — teoría de bifurcaciones
    dinámicas; resultado del programa, no del tratado). El paso de
    integración se escala con el ritmo para resolver cada retraso."""
    delays = []
    for r in rates:
        d_sigma = min(1e-2, 2e-3 * r ** (-1.0 / 3.0))
        delays.append(cruce_de_victoria(B, M0_sq0, r, C0, G,
                                        d_sigma=d_sigma)["delay"])
    slope = float(np.polyfit(np.log(rates), np.log(delays), 1)[0])
    return {"exponent": slope, "rates": np.asarray(rates),
            "delays": np.array(delays)}
