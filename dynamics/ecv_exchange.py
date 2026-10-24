"""Intercambio de masa entre la burbuja de MCV y la ECV en el borde R₀ (documento vivo del 24-oct-2026, PR-64): las leyes
de borde por halo, barrido y contacto, con las afueras del NFW prolongado. Sin datos y sin letra.

LA LEY DE BORDE (documento del 24-oct). M es la masa gravitatoria total dentro de R₀ (MCV, bariones y lo ya atrapado):

  Ṁ_x = 4πR₀²·[ξ_s·ρ_E·Ṙ₀ + (ν_b/√(2π))·(ξ·ρ_E − ρ_b)],   Ṙ₀/R₀ = (1/3)·(Ṁ/M − ρ̇_E/ρ_E).

  - Barrido (ξ_s): la ECV que engulle el borde al avanzar. Como M incluye lo atrapado, se alimenta a sí mismo: con ρ_E
    constante vale (ξ_s/2)·Ṁ, y lo barrido es (ξ_s/2)/(1 − ξ_s/2) veces el resto del crecimiento. En un halo aislado, sin
    acreción, solo lo mueve el propio contacto.
  - Contacto (ξ): la ECV se condensa en MCV y la MCV se evapora hacia la ECV con la escala cinética de la interfaz (forma de
    Hertz–Knudsen, flujo unidireccional ρν/√(2π)). Atrapa si ρ_b < ξ·ρ_E y escapa si ρ_b > ξ·ρ_E; ξ_eq = ρ_b/ρ_E.
  - Forma de Onsager: el contacto es el caso lineal de j = L_b·(μ_E − μ_M)/T_b con L_b = ρ_b·ν_b/√(2π) ≥ 0 y
    μ_E − μ_M = T_b·ln(ξ·ρ_E/ρ_b). La producción de entropía j·ln(ξρ_E/ρ_b) es ≥ 0 en las dos formas, también lejos del
    equilibrio, porque ξρ_E − ρ_b y el logaritmo tienen siempre el mismo signo.
  - Los brazos: el A, solo atrapamiento, es barrido más condensación (sin evaporación); el B, dos sentidos, es cualquier
    combinación con escape (decisión del autor del 24-oct: se prueban todas). La evaporación sola (E2) no tiene constante.

LAS AFUERAS (decisión 6 del documento). La recomendada: ρ_b y ν_b de la capa del borde, que con el NFW prolongado son
ρ_NFW(R₀) y la dispersión hidrostática del NFW completo en R₀. La alternativa: la media del entorno en R₀ (ρ̄_m y
ν = H·R₀/√3). ν_b es una escala cinética de interfaz, no una velocidad medida: es un cierre declarado, como ξ.

Unidades: las del código gravotérmico (r_s, ρ_s, v₀ = √(4πGρ_s)·r_s); las tasas por unidad de masa se dan en unidades de
H(z), con H en unidades del código √(m(c)/(100c³)) para un halo definido a 200ρ_c(z). Cosmología: la de
dynamics/ecv_border.py (Planck 2018).
"""

from __future__ import annotations

import math

from scipy.integrate import quad

from dynamics.ecv_border import PLANCK18, E_of_z, r0_selfconsistent, rho_s_over_rho_ecv

HK = 1.0 / math.sqrt(2.0 * math.pi)          # flujo unidireccional de una maxwelliana: ρν/√(2π)
CONTACT_FORMS = ("hertz-knudsen", "onsager-lineal")


def _m(x: float) -> float:
    return math.log1p(x) - x / (1.0 + x)


def _rho(x: float) -> float:
    return 1.0 / (x * (1.0 + x) ** 2)


def _nu2(x: float) -> float:
    """ν²/v₀² hidrostática del NFW completo en x (presión integrada hasta el infinito)."""
    return quad(lambda y: _rho(y) * _m(y) / y ** 2, x, math.inf, limit=400)[0] / _rho(x)


def hertz_knudsen_flux(rho: float, nu: float) -> float:
    """Flujo de masa unidireccional por unidad de área de un gas maxwelliano: ρ·ν/√(2π)."""
    return HK * rho * nu


def contact_flux(rho_b: float, nu_b: float, xi: float, rho_E: float, form: str = "hertz-knudsen") -> float:
    """Flujo neto de masa por unidad de área hacia la burbuja (> 0 atrapa, < 0 escapa)."""
    if form == "hertz-knudsen":
        return HK * nu_b * (xi * rho_E - rho_b)
    if form == "onsager-lineal":
        return HK * rho_b * nu_b * math.log(xi * rho_E / rho_b)
    raise ValueError(f"forma {form!r} fuera de {CONTACT_FORMS}")


def entropy_production(rho_b: float, nu_b: float, xi: float, rho_E: float, form: str = "hertz-knudsen") -> float:
    """j·(μ_E − μ_M)/T_b = j·ln(ξρ_E/ρ_b): no negativa en las dos formas."""
    return contact_flux(rho_b, nu_b, xi, rho_E, form) * math.log(xi * rho_E / rho_b)


def nfw_outskirts(c: float, z: float = 0.0) -> dict:
    """Las afueras recomendadas: el NFW prolongado hasta su R₀ autoconsistente, en unidades del código."""
    x = r0_selfconsistent(c, z)
    q_E = 1.0 / rho_s_over_rho_ecv(c, z)
    return {"c": c, "z": z, "x_R0": x, "M_R0": _m(x), "rho_b": _rho(x), "nu_b": math.sqrt(_nu2(x)), "q_E": q_E,
            "H": math.sqrt(_m(c) / (100.0 * c ** 3))}


def environment_outskirts(c: float, z: float = 0.0, cosmo: dict = PLANCK18) -> dict:
    """La alternativa de la decisión 6: la media del entorno en R₀, ρ_b = ρ̄_m(z) y ν_b = H·R₀/√3."""
    o = nfw_outskirts(c, z)
    rho_m_over_E = cosmo["Omega_m"] * (1.0 + z) ** 3 / (E_of_z(z, cosmo) ** 2 - cosmo["Omega_m"] * (1.0 + z) ** 3)
    return {**o, "rho_b": rho_m_over_E * o["q_E"], "nu_b": o["H"] * o["x_R0"] / math.sqrt(3.0)}


def border_rates(outskirts: dict, xi: float = 0.0, xi_s: float = 0.0, evaporation: bool = True,
                 form: str = "hertz-knudsen", accretion: float = 0.0) -> dict:
    """Tasas por unidad de la masa dentro de R₀, en unidades de H, de la ley de borde con unas afueras dadas.

    accretion: Ṁ_acr/M en unidades de H (0 en un halo aislado); el barrido se alimenta del crecimiento total."""
    o = outskirts
    area = o["x_R0"] ** 2                                   # 4πR₀² en unidades del código, sin el 4π (masa en 4πρ_s r_s³)
    per = 1.0 / (o["M_R0"] * o["H"])
    condensation = area * HK * o["nu_b"] * xi * o["q_E"] * per
    evaporated = area * HK * o["nu_b"] * o["rho_b"] * per if evaporation else 0.0
    if form == "onsager-lineal" and evaporation and xi > 0.0:
        contact = area * contact_flux(o["rho_b"], o["nu_b"], xi, o["q_E"], form) * per
    else:
        contact = condensation - evaporated
    total = (contact + 0.5 * xi_s * accretion) / (1.0 - 0.5 * xi_s)          # barrido autoalimentado
    return {"condensation": condensation, "evaporation": evaporated, "contact": contact, "sweep": total - contact,
            "net": total, "xi_eq": o["rho_b"] / o["q_E"]}
