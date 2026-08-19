"""Las β del frente 2 desde la jerarquía de Fokker-Planck (Def. 4.4).

EL OBJETO. El Flujo del Camino (ec. 4.2) es dΦ/dσ = −G⁻¹∇V. Su
completación estadística — la jerarquía de Fokker-Planck de la Def.
4.4 — evoluciona la distribución P(Φ) con deriva −G⁻¹∇V y difusión
entrópica. Integrar las fluctuaciones hasta varianza t renormaliza los
acoplos del Basal λ = (M0², B, C0): ESA es la β que el frente 2
necesita (§14.2) y que `core/victoria_exponent.py` está listo para
consumir.

LA REDUCCIÓN (cierres DECLARADOS, nunca resueltos en silencio):

Ecuación de Polchinski en dimensión cero (campos homogéneos (φM, φE),
kernel isótropo de ritmo unidad — G ∝ 1, temperatura absorbida):

    dV/dt = a·ΔV − b·(∇V)²,      canónico: a = b = 1/2

- El término a (difusión) es EXACTO sobre polinomios: con x = ρ² en
  2D, Δx = 4, Δx² = 16x, Δx³ = 36x², Δx⁴ = 64x³ (Δr^{2k} = 4k²r^{2k−2}).
- El término b (deriva) genera la JERARQUÍA: (∇V)² = 4x·(dV/dx)² sube
  el grado; el cierre canónico TRUNCA al grado cúbico del Basal
  (Def. 3.1) — la aproximación se declara y se mide con la variante
  cuártica (truncation='quartic').
- El sector η·χ es subdominante por el escalado (3.2): η = ē·δ0³, su
  retroalimentación en (M0², B, C0) es O(δ0⁶) — despreciada con
  declaración; los operadores χ·xⁿ generados se truncan (jerarquía).

LAS β DERIVADAS (base c: V = c1·x + c2·x² + c3·x³ con c1 = M0²/2,
c2 = −B/4, c3 = C0/6; los flujos dc/dt del a-término son 16a·c2,
36a·c3, 0 y del b-término −4b·c1², −16b·c1·c2, −b·(16c2² + 24c1·c3);
traducidos a acoplos físicos y CONGELADOS en la preinscripción ANTES
de computar ningún autovalor):

        β_{M0²} = −8·a·B  − 2·b·M0⁴
        β_B     = −24·a·C0 − 8·b·M0²·B
        β_{C0}  =          − 6·b·(B² + 2·M0²·C0)

El DICCIONARIO t ↔ ln S (τ = dt/d ln S > 0) no es derivable del corpus
recogido en este repositorio: reescala las tres β por un factor común.
Consecuencia declarada: el TIPO espectral (par complejo o no) y los
cocientes (p. ej. |Im μ|/|Re μ|) son invariantes de τ; el VALOR de s0
es proporcional a τ. Se publica s0 en unidades canónicas τ = 1 y el τ*
que λ = 10 exigiría (Def. 8.3).

ESTATUTO: derivación con cierres declarados (a/b, signo del flujo,
truncamiento, diccionario τ) — los cuatro desenlaces posibles están
preinscritos en results/2026-08-19_front2_fp_beta/ ANTES del primer
espectro. «No ajustar coeficientes después de conocer s0» es un
candado ejecutable (tests/test_front2_fp_lock.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- coeficientes derivados (enteros del álgebra; congelados) --------
# dλi/dt = Σ términos; tabla = {acoplo: [(coef, monomio)]} con
# monomios en (M0², B, C0). El candado compara esta tabla con la
# preinscripción.
BETA_COEFFICIENTS = {
    "M0_sq": [(-8.0, "a*B"), (-2.0, "b*M0_sq**2")],
    "B": [(-24.0, "a*C0"), (-8.0, "b*M0_sq*B")],
    "C0": [(-6.0, "b*B**2"), (-12.0, "b*M0_sq*C0")],
}


@dataclass(frozen=True)
class FPClosure:
    """El cierre DECLARADO de la reducción — inmutable; el análisis lo
    recibe, no lo estima ni lo ajusta."""
    a: float = 0.5                 # difusión (Polchinski canónico)
    b: float = 0.5                 # deriva (Polchinski canónico)
    flow_sign: int = +1            # t crece con ln S (τ > 0)
    truncation: str = "cubic"      # cierre de la jerarquía
    tau: float = 1.0               # diccionario t ↔ lnS (unidades canónicas)
    # espacio de robustez preinscrito (g = a/b):
    g_grid: tuple = field(default=tuple(np.round(
        np.logspace(-1.0, 1.0, 21), 10).tolist()))
    robustness_window: tuple = (0.5, 2.0)   # ventana para el desenlace D
    s0_band: float = 0.10          # banda ±10% alrededor de π/ln10
    s0_target: float = float(np.pi / np.log(10.0))


def canonical_closure() -> FPClosure:
    return FPClosure()


def beta_functions(M0_sq: float, B: float, C0: float,
                   a: float = 0.5, b: float = 0.5) -> np.ndarray:
    """(β_{M0²}, β_B, β_{C0}) por unidad de t — la derivación, tal
    cual; el diccionario τ y el signo se aplican fuera."""
    return np.array([
        -8.0 * a * B - 2.0 * b * M0_sq ** 2,
        -24.0 * a * C0 - 8.0 * b * M0_sq * B,
        -6.0 * b * (B ** 2 + 2.0 * M0_sq * C0),
    ])


def stability_matrix(M0_sq: float, B: float, C0: float,
                     a: float = 0.5, b: float = 0.5) -> np.ndarray:
    """M_ij = ∂βi/∂λj (ec. 14.2), analítica — el objeto del frente 2.

    Filas/columnas en el orden (M0², B, C0)."""
    return np.array([
        [-4.0 * b * M0_sq, -8.0 * a, 0.0],
        [-8.0 * b * B, -8.0 * b * M0_sq, -24.0 * a],
        [-12.0 * b * C0, -12.0 * b * B, -12.0 * b * M0_sq],
    ])


def spinodal_canonical_point() -> tuple[float, float, float]:
    """El punto de evaluación preinscrito: la espinodal D = 0
    (Def. 8.4) con M0² = 1, C0 = 1 ⟹ B = +2.

    CORRECCIÓN (revisión adversarial 19-ago, hallazgo HIGH): la
    versión original de este docstring afirmaba que «la dependencia
    física en δ0 entra por el diccionario τ(δ0)». Es FALSO: las β
    cumplen la covarianza exacta

        β(D_s·λ; a, b) = σ·D_s·β(λ; a, b·k),
        D_s = diag(k·σ, k·σ², k·σ³),

    así que la familia espinodal FÍSICA del escalado (3.2),
    (m̄²δ0², b̄δ0, C0), es espectralmente equivalente (salvo factor
    global positivo, absorbible en τ) a (1, 2, 1) con
    g_ef = g·δ0⁻³ — el eje g del barrido ES el eje δ0, y el TIPO
    espectral (invariante de τ) SÍ depende de δ0. Este punto fija
    implícitamente el invariante B³/C0² = 8, es decir δ0 ≈ 1 (con
    m̄ = 1, b̄ = 2, C0 = 1): fuera del régimen perturbativo δ0 ≪ 1
    del corpus. El punto físico vive en physical_spinodal_point y su
    barrido en scripts/run_front2_fp_delta0.py (adenda)."""
    return (1.0, 2.0, 1.0)


def physical_spinodal_point(delta0: float, m_bar: float = 1.0,
                            C0: float = 1.0) -> tuple:
    """La espinodal FÍSICA del escalado (3.2): con b̄² = 4·C0·m̄²
    (forma espinodal), (M0², B, C0) = (m̄²δ0², 2·m̄·√C0·δ0, C0) —
    D = 0 exacto para todo δ0."""
    return (m_bar ** 2 * delta0 ** 2,
            2.0 * m_bar * np.sqrt(C0) * delta0, C0)


def g_effective(delta0: float, g: float = 1.0) -> float:
    """El g equivalente del punto físico bajo la covarianza:
    g_ef = g·δ0⁻³ (con m̄ = 1, C0 = 1; el caso general reescala por
    b̄/(2m̄⁴)). El barrido en g y el barrido en δ0 son el mismo eje."""
    return g * delta0 ** -3


# --- variante cuártica: el coste del truncamiento, medido -----------

def beta_functions_quartic(M0_sq: float, B: float, C0: float, E4: float,
                           a: float = 0.5, b: float = 0.5) -> np.ndarray:
    """Nivel siguiente de la jerarquía: V += c4·x⁴ con E4 = 8·c4.

    Δx⁴ = 64x³ alimenta el flujo de C0, y el b-término da
    dc4/dt = −b·(32·c1·c4 + 48·c2·c3). En acoplos físicos:
        β_{C0} += 48·a·E4          (de 6·64·a·c4 = 6·64·a·E4/8)
        β_{E4}  = −16·b·M0²·E4 + 16·b·B·C0
    Evaluar en E4 = 0 mide el coste del truncamiento cúbico
    (test de álgebra por diferencias finitas en la suite)."""
    base = beta_functions(M0_sq, B, C0, a, b)
    dc4 = -b * (32.0 * (M0_sq / 2.0) * (E4 / 8.0)
                + 48.0 * (-B / 4.0) * (C0 / 6.0))
    return np.array([
        base[0],
        base[1],
        base[2] + 48.0 * a * E4,
        8.0 * dc4,
    ])


def stability_matrix_quartic(M0_sq: float, B: float, C0: float,
                             E4: float = 0.0, a: float = 0.5,
                             b: float = 0.5) -> np.ndarray:
    """∂β/∂λ de la variante cuártica (4×4, orden M0², B, C0, E4) —
    evaluada en E4 = 0 mide el coste del truncamiento cúbico."""
    m = np.zeros((4, 4))
    m[:3, :3] = stability_matrix(M0_sq, B, C0, a, b)
    m[2, 3] = 48.0 * a
    # de β_{E4} = −16·b·M0²·E4 + 16·b·B·C0:
    m[3, 0] = -16.0 * b * E4
    m[3, 1] = 16.0 * b * C0
    m[3, 2] = 16.0 * b * B
    m[3, 3] = -16.0 * b * M0_sq
    return m
