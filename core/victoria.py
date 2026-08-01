"""El Ciclo de Victoria con suelo de activación (v35, Cap. 10 y H.2.3).

Realiza el Axioma 8 («el retorno no está garantizado: se gana»).

Modo de Memoria (Lema 10.2 / ec. 10.1): en el anillo de vacíos, la fase
de conversión θ es el pseudo-Goldstone de la simetría O(2) rota por la
inclinación; su masa es paramétricamente ligera:

    m_θ² = η/ρ+ = O(δ0^{5/2})

— lo que casi no pesa es lo que sobrevive entre ciclos.

Exponente de Lydia (Def. 10.4, derivado en H.2.3, ec. H.7): la
aplicación de retorno tiene pendiente en el origen

    A = γR·ē/m̄² = F'_Vic(0),     ν = (1/ΔS)·ln A

    ν > 0  ⟺  γR·ē > m̄²      ← CONDICIONAL (F.2): el signo espera el
                                 valor de γR desde la microdinámica del
                                 reinicio (frente abierto nº 4).

Retorno con suelo (Teo. 10.6): F_Vic actúa sobre [δ0_min, δ_sat] — el
colapso C0 solo dispara si T0(δ0) supera la barrera (suelo δ0_min; por
debajo, el ciclo no arranca: el SILENCIO DE VICTORIA) y el Techo W_max
acota el trabajo reinvertido (T0(δ') = c̄·δ'³ ≤ W_max ⟹ δ_sat). Con
A > 1 el punto fijo δ0* es atractor: la imperfección se reconfigura
vuelta a vuelta — espiral, no círculo; la perfección es un repulsor.

Este módulo EXPONE ν como parámetro con su condición; no decide su signo.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from .basal import scaled_params, kappa_plus, M_BAR, B_BAR, E_BAR, C0_DEFAULT

STATUS_LYDIA = ("condicional (F.2): ν > 0 ⟺ γR·ē > m̄²; γR pendiente de la "
                "microdinámica del reinicio (frente abierto nº 4)")


def memory_mode_mass_sq(delta0: float, m_bar: float = M_BAR,
                        b_bar: float = B_BAR, e_bar: float = E_BAR,
                        C0: float = C0_DEFAULT) -> float:
    """m_θ² = η/ρ+ (ec. 10.1) — el Modo de Memoria, pseudo-Goldstone O(2).

    Con η = ē·δ0³ y ρ+ = √(κ+·δ0): m_θ² = (ē/√κ+)·δ0^{5/2}.
    """
    if delta0 == 0.0:
        return 0.0
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    rho_plus = np.sqrt(kappa_plus(m_bar, b_bar, C0) * delta0)
    return float(p["eta"] / rho_plus)


def lydia_gain(gamma_R: float, m_bar: float = M_BAR,
               e_bar: float = E_BAR) -> float:
    """A = γR·ē/m̄² = F'_Vic(0) — la pendiente del retorno (ec. H.7)."""
    return gamma_R * e_bar / m_bar ** 2


def lydia_exponent(gamma_R: float, m_bar: float = M_BAR,
                   e_bar: float = E_BAR,
                   delta_S: float = C.DELTA_S) -> float:
    """ν = (1/ΔS)·ln A (ec. H.7). ν > 0 ⟺ γR·ē > m̄² — CONDICIONAL."""
    A = lydia_gain(gamma_R, m_bar, e_bar)
    if A <= 0.0:
        raise ValueError("La ganancia A debe ser positiva")
    return float(np.log(A) / delta_S)


def fertility_condition(gamma_R: float, m_bar: float = M_BAR,
                        e_bar: float = E_BAR) -> bool:
    """ν > 0 ⟺ γR·ē > m̄² (H.7): el eterno retorno se gana si la
    inclinación (memoria) supera a la curvatura (supresión)."""
    return gamma_R * e_bar > m_bar ** 2


def return_map(delta: float, gamma_R: float, *, delta_min: float,
               delta_sat: float, m_bar: float = M_BAR,
               e_bar: float = E_BAR) -> float:
    """F_Vic con suelo y techo (Teo. 10.6).

    - δ < δ0_min → 0 (SILENCIO DE VICTORIA: el ciclo no arranca)
    - lineal A·δ cerca del origen (H.7)
    - saturación en δ_sat (Techo de Victoria: T0 ≤ W_max)
    """
    if delta < delta_min:
        return 0.0
    A = lydia_gain(gamma_R, m_bar, e_bar)
    return float(min(A * delta, delta_sat))


def iterate_cycles(delta0: float, gamma_R: float, n_cycles: int, *,
                   delta_min: float, delta_sat: float,
                   m_bar: float = M_BAR, e_bar: float = E_BAR) -> np.ndarray:
    """Itera el mapa de retorno n_cycles veces (la historia entre ciclos)."""
    out = [delta0]
    d = delta0
    for _ in range(n_cycles):
        d = return_map(d, gamma_R, delta_min=delta_min,
                       delta_sat=delta_sat, m_bar=m_bar, e_bar=e_bar)
        out.append(d)
        if d == 0.0:  # Silencio: terminal
            break
    return np.array(out)


def is_silence(history: np.ndarray) -> bool:
    """¿Terminó la historia en el Silencio de Victoria (residuo bajo el
    suelo, la cadena no rearranca)?"""
    return bool(history[-1] == 0.0)
