"""Candidatos a segundo nivel del Camino que entran en el flujo de acoplos — CRIBADO, no derivación (frente 2).

EL PROBLEMA. Bajo el cierre canónico de Fokker–Planck (core/fokker_planck_beta.py) el discriminante
D = B² − 4C₀M₀² se hunde pero NO cruza cero antes del fin del recorrido: M₀² cambia de signo antes y D vuelve
a crecer (fila decada-discriminante; §3.20). Los colapsos del tratado exigen β que hagan D → 0. Cualquier
término nuevo que se proponga para el Camino y que entre en las β debe someterse al mismo cribado
(preinscrito, orden del 28-sep, PR-6) ANTES de discutir λ:
    (i)   reduce D de forma monótona SIN llevar M₀² a cero antes del primer cruce;
    (ii)  produce tres cruces D = 0 en el recorrido;
    (iii) se publican los S de los cruces frente a 0.009 / 0.099 / 0.999 y los cocientes entre cruces sucesivos
          (la Década sería 10).
Nada de esto decide λ: el cribado dice si el candidato PUEDE ser la β del frente 2, no que lo sea (E13).

CANDIDATO 1 — «conversion_current»: la contribución del término κΣ̇ê_E (brazo (ii) del vacío 2D, §3.33) a las
β vía la Def. 4.4. En la ecuación de Polchinski en dimensión cero, dV/dt = a·ΔV − b·(∇V)², el término de
deriva es −b veces la producción entrópica del Flujo del Camino, Σ̇ = −∇V·dΦ/dσ = (∇V)² (G = 1). Con la
corriente de conversión dΦ/dσ = −∇V + u, u = κ·Σ̇·ê_E, la producción total es Σ̇_total = (∇V)² − ∇V·u, y la
deriva del Polchinski pasa a ser −b·Σ̇_total. LECTURA DECLARADA (no derivada del corpus): el término de deriva
acompaña a la producción entrópica TOTAL, luego

    Δ(dV/dt) = + b · ∇V·u = b · κ · (∇V)² · ∂_E V = b · κ · r(x)³ · x · φ_E ,      r(x) = M₀² − B·x + C₀·x²

(sector η·χ despreciado, como en la derivación canónica: O(δ₀⁶)). El término es IMPAR en φ_E: no es función
de x = ρ² y no vive en la base cúbica del Basal {x, x², x³}. Para escribirlo como Δβ hace falta un cierre
de proyección, y se declaran dos como brazos:

    'plane'     — promedio angular sobre el plano completo: ⟨φ_E⟩ = 0 ⟹ Δβ ≡ 0. El candidato NO entra en las β
                  a este orden: brazo de control negativo (el cribado falla por construcción).
    'quadrant'  — promedio angular sobre el dominio físico φ_M, φ_E ≥ 0 (Cap. 2): ⟨φ_E⟩_Q = (2/π)·√x, de modo
                  que Δ(dV/dt) → (2bκ/π)·r(x)³·x^{3/2}, y proyección por mínimos cuadrados sobre span{x, x², x³}
                  en x ∈ [0, x₊] (x₊ = ρ₊² del paisaje inicial, fijo). Los coeficientes (d₁, d₂, d₃) de la
                  proyección se traducen a acoplos con c₁ = M₀²/2, c₂ = −B/4, c₃ = C₀/6:
                      Δβ_{M₀²} = 2·d₁,   Δβ_B = −4·d₂,   Δβ_{C₀} = 6·d₃ .

κ = κ̂·ρ₊/T₀ como en el reloj (κ̂ adimensional, declarado). El diccionario τ y el signo del flujo se aplican
fuera, igual que a las β canónicas. Todo lo anterior son cierres DECLARADOS: el cribado publica qué pasa con
ellos, no afirma que sean los del tratado.
"""

from __future__ import annotations

import numpy as np

CANDIDATES = ("conversion_current",)
CLOSURES = ("plane", "quadrant")

# base cúbica del Basal: V = c1·x + c2·x² + c3·x³ con c1 = M0²/2, c2 = −B/4, c3 = C0/6
_ACOPLO_FROM_COEF = np.array([2.0, -4.0, 6.0])


def _project_halfinteger_series(coefs: np.ndarray, x_plus: float) -> np.ndarray:
    """Proyección L²([0, x₊]) de f(x) = Σ_k coefs[k]·x^{k + 3/2} sobre span{x, x², x³}.

    Trabaja en y = x/x₊ ∈ [0, 1] para acondicionar el sistema normal (3×3, tipo Hilbert) y devuelve los
    coeficientes (d₁, d₂, d₃) en la variable x."""
    k = np.arange(len(coefs))
    g = coefs * x_plus ** (k + 1.5)                    # f(y) = Σ g_k y^{k+3/2}
    G = np.array([[1.0 / (i + j + 1.0) for j in (1, 2, 3)] for i in (1, 2, 3)])
    rhs = np.array([np.sum(g / (k + i + 2.5)) for i in (1, 2, 3)])
    e = np.linalg.solve(G, rhs)                        # f(y) ≈ Σ e_i y^i
    return e / x_plus ** np.array([1.0, 2.0, 3.0])     # en x: d_i = e_i / x₊^i


def delta_beta_conversion_current(M0_sq: float, B: float, C0: float, kappa: float, x_plus: float,
                                  closure: str = "quadrant", b: float = 0.5) -> np.ndarray:
    """Δβ = (Δβ_{M0²}, Δβ_B, Δβ_{C0}) por unidad de t del candidato 1 bajo el cierre declarado."""
    if closure not in CLOSURES:
        raise ValueError(f"closure: {' | '.join(CLOSURES)}")
    if closure == "plane" or kappa == 0.0:
        return np.zeros(3)
    r = np.polynomial.polynomial.Polynomial([M0_sq, -B, C0])       # r(x) = M0² − B x + C0 x²
    r3 = (r ** 3).coef                                              # grado 6 en x
    coefs = (2.0 * b * kappa / np.pi) * r3                          # Δ(dV/dt) = Σ coefs[k] x^{k+3/2}
    d = _project_halfinteger_series(coefs, x_plus)
    return _ACOPLO_FROM_COEF * d


def delta_beta(candidate: str, M0_sq: float, B: float, C0: float, *, kappa: float, x_plus: float,
               closure: str = "quadrant", b: float = 0.5) -> np.ndarray:
    """Despacho por nombre declarado; falla cerrado ante un candidato desconocido."""
    if candidate == "conversion_current":
        return delta_beta_conversion_current(M0_sq, B, C0, kappa, x_plus, closure, b)
    raise ValueError(f"candidato a β desconocido: {candidate!r} (declarados: {', '.join(CANDIDATES)})")


def screening_report(S_crossings: list, D: np.ndarray, S: np.ndarray, M0_sq: np.ndarray, thresholds: list,
                     tol_monotone: float = 0.0) -> dict:
    """Los tres criterios del cribado sobre una trayectoria (S, D, M₀²) y sus cruces D = 0.

    (i)   monótono: D no sube más de tol_monotone·|D₀| entre muestras antes del primer cruce, y M₀² > 0 en el
          primer cruce (si no hay cruce, (i) es False: no hay nada que cribar);
    (ii)  n_cruces ≥ 3;
    (iii) tabla S_cruce frente a los umbrales y cocientes S_{k+1}/S_k (Década = 10).
    No decide λ."""
    D = np.asarray(D, dtype=float); S = np.asarray(S, dtype=float); M = np.asarray(M0_sq, dtype=float)
    n = len(S_crossings)
    if n:
        upto = S <= S_crossings[0]
        rises = np.diff(D[upto]) if upto.sum() > 1 else np.array([0.0])
        monotone = bool(rises.max() <= tol_monotone * abs(D[0])) if rises.size else True
        i_first = int(np.searchsorted(S, S_crossings[0]))
        m_at_first = float(M[min(i_first, len(M) - 1)])
        crit_i = bool(monotone and m_at_first > 0.0)
    else:
        monotone, m_at_first, crit_i = None, None, False
    ratios = [S_crossings[k + 1] / S_crossings[k] for k in range(n - 1)] if n > 1 else []
    return {"n_crossings": n, "S_crossings": list(S_crossings), "thresholds": list(thresholds),
            "S_over_threshold": [S_crossings[k] / thresholds[k] for k in range(min(n, len(thresholds)))],
            "ratios_successive": ratios, "decade_would_be": 10.0,
            "D_monotone_before_first_crossing": monotone, "M0_sq_at_first_crossing": m_at_first,
            "criterion_i": crit_i, "criterion_ii": bool(n >= 3), "passes_i_and_ii": bool(crit_i and n >= 3),
            "status": "cribado diagnóstico (E13): pasa/no pasa (i)–(ii); ninguna letra sobre λ"}
