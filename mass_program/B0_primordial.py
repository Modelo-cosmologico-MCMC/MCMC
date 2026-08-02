"""B0 — Imperfección primordial δ₀.

Define la asimetría primordial que da origen a la jerarquía de escalas.

REGLA CANÓNICA (ronda 5): no identificar δ₀ con ε_Λ en ningún material.
En el corpus v32 este bloque operaba con la identificación δ₀ = ε =
0.012 ± 0.003 (herencia superada); la v35 (F.2) lista δ₀ como «input de
ciclo; atractor δ₀* (Teo. 10.6)» SIN valor numérico asignado. El empalme
C¹ (H.2.4, mass_program/B7) lo mide: δ₀* ≈ 0.0581 con formas fiduciales.

El VEV del nivel cero v₀ es proporcional al primero:
    v₀ = δ₀ · v₁

Nota de versión: v₁ = escala Planck es la asignación por sello del
Tratado Unificado (v32, bloque LEGACY_V32 de constants.py); la v35 no
asigna la escala Planck a ningún umbral.
"""

from mcmc_ontology import constants as C


def delta_0() -> float:
    """δ₀ operativo del programa de masas v32 (devuelve 0.012).

    ADVERTENCIA (regla canónica): 0.012 es el valor de ε_Λ que el v32
    identificaba con δ₀; la identificación está superada y la v35 no
    asigna valor numérico a δ₀ (F.2). Se conserva porque los bloques
    v32 del programa de masas operan con ella. El candidato medido por
    el empalme C¹ es δ₀* ≈ 0.0581 (mass_program/B7.delta0_required).
    """
    return C.EPSILON_0


def v0() -> float:
    """v₀ = δ₀ · v₁ [GeV] (con el δ₀ operativo v32 y v₁ de LEGACY_V32)."""
    return C.EPSILON_0 * C.V_GEV["C1"]
