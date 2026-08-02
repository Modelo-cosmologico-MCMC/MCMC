"""Los ocho axiomas ontológicos del MCMC (Tratado de Fundamentos v35, §1.2).

Única entrada conceptual del modelo: todo lo demás es definición, teorema
o conjetura declarada. Este módulo ancla el repositorio al tratado vigente;
cada módulo del paquete declara en su docstring qué axioma(s) realiza.

La cadena deductiva es lineal:
    axiomas 1-3 → cinemática interna (Plano Dual, Potencial Basal)
    axioma  4   → dinámica (Flujo del Camino)
    axiomas 5-6 → tramo euclidiano (sellos, atemporalidad pre-geométrica)
    axioma  7   → nacimiento del tiempo (Ley de Cronos)
    axioma  8   → cierre cíclico (Victoria)
"""

from __future__ import annotations

AXIOMS: dict[int, tuple[str, str]] = {
    1: ("Unidad dual",
        "La realidad es una sola unidad con dos capacidades inseparables: "
        "condensar (Mp) y acoger (Ep). Ninguna existe sin la otra, y su "
        "distinción es interna, no espacial."),
    2: ("Imperfección heredada",
        "Todo ciclo hereda una imperfección δ₀ > 0. El caso δ₀ = 0 define "
        "el estado perfecto: simétrico, inerte y estéril."),
    3: ("Tensión Primordial",
        "Existe una energía liberable del ciclo, T₀ ≥ 0, con "
        "T₀ = 0 ⟺ δ₀ = 0."),
    4: ("El Camino",
        "La evolución fundamental es descarga de tensión por la vía de "
        "mínima resistencia; produce estructura y no admite reversión."),
    5: ("Colapsos y sellos",
        "La descarga procede por umbrales discretos; en cada sello una "
        "regla del mundo queda fijada irreversiblemente."),
    6: ("Atemporalidad pre-geométrica",
        "Antes del umbral final no existe tiempo: solo correlaciones "
        "ordenadas por la estructuración S."),
    7: ("Ley de Cronos",
        "El tiempo nace como proyección causal del flujo de S; su caudal "
        "local es la tasa local de descarga."),
    8: ("Victoria",
        "El residuo de cada descarga funda el ciclo siguiente. El retorno "
        "no está garantizado: se gana."),
}

# Realización EJECUTABLE de cada axioma (módulo que lo implementa y test
# que verifica su teorema — la cadena deductiva como código, ronda 3).
REALIZATION: dict[int, str] = {
    1: "Plano Dual (ρ,θ,χ,ς) — core.dual_plane (Defs. 2.1-2.2; tests: "
       "reflexión Z₂, diagonal dual); v35 §2",
    2: "δ₀ input de ciclo con escalado canónico (3.2) — core.basal."
       "scaled_params (test: rigidez marginal M0² ∝ δ0²); v35 §3",
    3: "T₀ = c̄·δ₀³ — core.basal.T0_numeric (test: exponente 3 medido; "
       "T0=0 ⟺ δ0=0); v35 Prop. 3.4",
    4: "Flujo del Camino — core.path_flow (tests: Monotonía Teo. 4.5, "
       "Exclusión Lema 4.7, salida al polo de masa Prop. 3.5); v35 §4",
    5: "Sellos y Ley de la Década — constants.decade_thresholds "
       "(Prop. 8.1) y core.decade (Discriminante, Def. 8.4/Prop. 8.5, "
       "Cruce de Victoria); v35 §8, Tabla F.1",
    6: "Tramo euclidiano C(d+1,0) — core.florencia.chain_generators "
       "(tests: firmas +1, elipticidad Lema 5.2); v35 §5",
    7: "Nacimiento del tiempo — core.florencia.florencia_rotation "
       "(test: γ⁰=iγS, firma (−,+,+,+), un solo giro); Ley de Cronos: "
       "mcmc_ontology.S_map, cronos/ (cap. 11); v35 §6",
    8: "Ciclo de Victoria — core.victoria (tests: m_θ² ∝ δ0^{5/2}, "
       "espiral con ν>0, Silencio con ν<0; ν EXPUESTO como condicional, "
       "frente nº 4); v35 §10, H.2.3",
}


def axiom(n: int) -> tuple[str, str]:
    """Devuelve (nombre, enunciado literal) del axioma n (1..8)."""
    if n not in AXIOMS:
        raise KeyError(f"Axioma fuera de rango: {n} (válidos: 1..8)")
    return AXIOMS[n]


def realization(n: int) -> str:
    """Dónde se realiza el axioma n en el paquete/tratado."""
    if n not in REALIZATION:
        raise KeyError(f"Axioma fuera de rango: {n} (válidos: 1..8)")
    return REALIZATION[n]
