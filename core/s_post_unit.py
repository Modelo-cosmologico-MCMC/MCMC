"""Diccionario, ecuaciones 2 y 3 (orden del autor del 22-sep, §10):
la UNIDAD de S tras Florencia y la corriente de conversión C(S).

Ecuación 2 — la unidad de S post-Florencia (fila
diccionario-unidad-S-post-florencia, hasta ahora hueco). Antes de
Florencia el reloj S mide fracciones de la Tensión Primordial:
dS_pre = Σ̇ dσ / T₀ (S = 1 ⟺ descarga completa; C1, Teo. 4.5). Después,
la magnitud que alimenta los canales (ρ_id + ρ_lat, la «E sellada» del
guion v32) es el depósito sellado T_sellada, y la producción entrópica
que cuenta es la de los canales, Σ̇_post (corrientes J_id, J_lat, B.1.2
del Maestro). La ecuación que fija la unidad, en las MISMAS unidades ΔS,
es

    dS_post = Σ̇_post dσ / T_sellada.                          (E2)

Esta es la forma DECLARADA de la unidad; lo que el repositorio no tiene
todavía son Σ̇_post y T_sellada como magnitudes derivadas (cronos/channels
provee el refresco de los canales con tasas CALIBRADAS por el mapa S(t),
no la producción entrópica). Hasta entonces (E2) es una ecuación con sus
dos entradas declaradas, no un cálculo: el hueco se ESTRECHA a dos
magnitudes nombradas, no se cierra.

Ecuación 3 — C(S) y la corriente de conversión. Si la preinscripción del
vacío 2D adopta la corriente de conversión dΦ/dσ = −G⁻¹∇V + κ·Σ̇·ê_E
(§5 del texto del autor), el espacio acumulado en la descarga es
∫κΣ̇dσ = κT₀f, y tras Florencia la MISMA corriente es la ley de expansión

    d ln a / dS = C(S)   con   C(S) = κ · T_sellada · (dS_post/dS) / …   (E3, forma)

es decir, C(S) deja de ser convención (S_hoy = 95 en
cosmology.dark_channels.S_of_z) y se deriva de κ y del depósito sellado.
Aquí solo se declara la relación y se publica la convención vigente;
el número de κ es el resultado de la preinscripción del vacío 2D
(brazo conversion_current) y no existe todavía.

Estatuto: declaraciones (E8); ningún número nuevo; el hueco de la fila
se reformula en dos magnitudes nombradas y una relación por escribir.
"""

from __future__ import annotations

E2_FORM = ("dS_post = Σ̇_post·dσ / T_sellada — Σ̇_post: producción entrópica de los canales (J_id, J_lat; B.1.2); "
           "T_sellada: depósito sellado ρ_id + ρ_lat que alimenta los canales; misma unidad ΔS que dS_pre = Σ̇dσ/T₀")
E3_FORM = ("d ln a/dS = C(S), con C(S) la continuación post-Florencia de la corriente de conversión κ·Σ̇·ê_E "
           "(κ: el único número calibrado por el cruce de la diagonal en S = 1, preinscripción del vacío 2D); "
           "hasta que κ exista, S_hoy = 95 sigue siendo convención (cosmology.dark_channels.S_of_z)")

STATUS = {"E2": "declarada: unidad de S post-Florencia como fracción del depósito sellado; Σ̇_post y T_sellada nombradas, no derivadas",
          "E3": "declarada: C(S) desde κ; κ pendiente de la preinscripción del vacío 2D (brazo conversion_current)"}


def dS_post(Sigma_dot_post: float, T_sealed: float, d_sigma: float) -> float:
    """(E2): incremento de S tras Florencia en unidades ΔS. Falla cerrado
    con un depósito no positivo (no hay unidad que medir)."""
    if T_sealed <= 0.0:
        raise ValueError("T_sellada ≤ 0: sin depósito sellado no hay unidad de S post-Florencia")
    return float(Sigma_dot_post * d_sigma / T_sealed)


def dS_pre(Sigma_dot: float, T0: float, d_sigma: float) -> float:
    """La unidad pre-geométrica del reloj S (C1, Teo. 4.5), para comparar."""
    if T0 <= 0.0:
        raise ValueError("T₀ ≤ 0")
    return float(Sigma_dot * d_sigma / T0)


def C_of_S_from_kappa(kappa: float | None, T_sealed: float | None) -> dict:
    """(E3): publica la relación y su estado. Sin κ (la preinscripción del
    vacío 2D no ha corrido) devuelve la convención vigente etiquetada."""
    if kappa is None or T_sealed is None:
        return {"C_of_S": None, "status": STATUS["E3"], "convention_in_force": "S_hoy = 95 (cosmology.dark_channels.S_of_z)",
                "form": E3_FORM}
    return {"C_of_S": "por escribir: requiere el diccionario t ↔ σ además de κ y T_sellada", "kappa": kappa, "T_sealed": T_sealed,
            "status": "κ disponible; la relación C(S) = f(κ, T_sellada, dt/dσ) exige el diccionario τ (frente 2)", "form": E3_FORM}
