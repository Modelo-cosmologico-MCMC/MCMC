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

# Realización de cada axioma en el paquete / el tratado (referencia cruzada).
REALIZATION: dict[int, str] = {
    1: "Plano Dual Mp/Ep — mcmc_ontology.potential (Potencial Basal); v35 §2-3",
    2: "δ₀ ≡ EPSILON_0 (input de ciclo) — mcmc_ontology.constants; v35 §3, Teo. 10.6",
    3: "T₀ = c̄·δ₀³ — v35 §3.4 (estructura de vacíos)",
    4: "Flujo del Camino, monotonía — v35 §4.5; potencial escalonado V(Φ;S)",
    5: "Sellos S_n y Ley de la Década — mcmc_ontology.constants (S_SEALS), "
       "mcmc_ontology.seals; v35 Prop. 8.1, Tabla F.1",
    6: "Tramo euclidiano C(d+1,0) — mcmc_ontology.clifford_algebra; v35 §5.1-5.2",
    7: "Nacimiento del tiempo en S=1.001 — mcmc_ontology.S_map (s_to_t_rel), "
       "cronos/; v35 §7 (Rotación de Florencia), cap. 11 (Ley de Cronos)",
    8: "Ciclo de Victoria, s0 = π/ln(10) — v35 §10 (Retorno de Victoria); "
       "frente abierto nº 4",
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
