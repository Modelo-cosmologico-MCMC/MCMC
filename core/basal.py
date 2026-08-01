"""El Potencial Basal — el paisaje de la imperfección (v35, Cap. 3).

Realiza los Axiomas 2 y 3. Definición 3.1, sobre D con x ≡ ρ²:

    V0(ρ, χ; δ0) = ½·M0²·ρ² − (B/4)·ρ⁴ + (C0/6)·ρ⁶ − η·χ,   C0 > 0

con el escalado canónico de la imperfección (ec. 3.2):

    M0² = m̄²·δ0²,   B = b̄·δ0,   η = ē·δ0³,   m̄, b̄, ē > 0 de orden uno.

Estructura de vacíos (§3.2): los extremos no triviales resuelven
C0·x² − B·x + M0² = 0, con discriminante D ≡ B² − 4·C0·M0²; el mínimo
no trivial es más profundo que el origen si y solo si (ec. 3.3)

    b̄² > (16/3)·C0·m̄²      (condición de casi-cancelación, de cocientes)

Ley de escala (Prop. 3.4): x+ = κ+·δ0 con κ+ = [b̄ + √(b̄²−4C0m̄²)]/(2C0), y

    T0 = c̄·δ0³,   c̄ = b̄·κ+²/4 − C0·κ+³/6 − m̄²·κ+/2 > 0

con T0 = 0 ⟺ δ0 = 0 (Axioma 3, Def. 3.3). La inclinación en el vacío
verdadero es η·ρ+ = O(δ0^{7/2}) (subdominante) y el residuo de
desequilibrio ⟨χ⟩ ≃ (ē/m̄²)·δ0.

Control negativo (§3.2): con δ0 = 0 el paisaje es plano hasta sexto
orden (V0 = C0·ρ⁶/6) y el estado es estrictamente inerte — no hay salida.
"""

from __future__ import annotations

import numpy as np

# Constantes de forma O(1) por defecto (inputs, v35 F.2). Cumplen (3.3):
# b̄² = 9 > (16/3)·C0·m̄² = 5.33
M_BAR = 1.0
B_BAR = 3.0
E_BAR = 1.0
C0_DEFAULT = 1.0


def scaled_params(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                  e_bar: float = E_BAR) -> dict:
    """Escalado canónico (3.2): M0² = m̄²δ0², B = b̄δ0, η = ēδ0³."""
    return {
        "M0_sq": m_bar ** 2 * delta0 ** 2,
        "B": b_bar * delta0,
        "eta": e_bar * delta0 ** 3,
    }


def V0(rho: np.ndarray | float, chi: np.ndarray | float, delta0: float,
       m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
       C0: float = C0_DEFAULT) -> np.ndarray | float:
    """El Potencial Basal (ec. 3.1)."""
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    rho = np.asarray(rho, dtype=float)
    x = rho ** 2
    out = (0.5 * p["M0_sq"] * x - 0.25 * p["B"] * x ** 2
           + (C0 / 6.0) * x ** 3 - p["eta"] * np.asarray(chi, dtype=float))
    return float(out) if np.ndim(out) == 0 else out


def quasi_cancellation_ok(m_bar: float = M_BAR, b_bar: float = B_BAR,
                          C0: float = C0_DEFAULT) -> bool:
    """Condición de casi-cancelación (3.3): b̄² > (16/3)·C0·m̄²."""
    return b_bar ** 2 > (16.0 / 3.0) * C0 * m_bar ** 2


def discriminant(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                 C0: float = C0_DEFAULT) -> float:
    """D ≡ B² − 4·C0·M0² (Def. 8.4). Con (3.2): D = δ0²·(b̄² − 4C0m̄²)."""
    p = scaled_params(delta0, m_bar, b_bar)
    return p["B"] ** 2 - 4.0 * C0 * p["M0_sq"]


def kappa_plus(m_bar: float = M_BAR, b_bar: float = B_BAR,
               C0: float = C0_DEFAULT) -> float:
    """κ+ = [b̄ + √(b̄² − 4C0m̄²)]/(2C0) — constante O(1) (Prop. 3.4)."""
    disc = b_bar ** 2 - 4.0 * C0 * m_bar ** 2
    if disc < 0.0:
        raise ValueError("Sin vacío no trivial: b̄² < 4C0m̄² (D < 0)")
    return (b_bar + np.sqrt(disc)) / (2.0 * C0)


def c_bar(m_bar: float = M_BAR, b_bar: float = B_BAR,
          C0: float = C0_DEFAULT) -> float:
    """c̄ = b̄κ+²/4 − C0κ+³/6 − m̄²κ+/2 (Prop. 3.4)."""
    k = kappa_plus(m_bar, b_bar, C0)
    return b_bar * k ** 2 / 4.0 - C0 * k ** 3 / 6.0 - m_bar ** 2 * k / 2.0


def T0_analytic(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
                C0: float = C0_DEFAULT) -> float:
    """Ley de escala de la Tensión Primordial (Prop. 3.4): T0 = c̄·δ0³."""
    if delta0 == 0.0:
        return 0.0
    return c_bar(m_bar, b_bar, C0) * delta0 ** 3


def T0_numeric(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR,
               C0: float = C0_DEFAULT) -> float:
    """T0 medida numéricamente: V0(falso vacío) − V0(vacío verdadero)
    (Def. 3.3), con la inclinación η apagada (subdominante, O(δ0^{7/2}))."""
    if delta0 == 0.0:
        return 0.0
    p = scaled_params(delta0, m_bar, b_bar)
    disc = p["B"] ** 2 - 4.0 * C0 * p["M0_sq"]
    if disc < 0.0:
        return 0.0  # sin vacío no trivial
    x_plus = (p["B"] + np.sqrt(disc)) / (2.0 * C0)
    v_true = (0.5 * p["M0_sq"] * x_plus - 0.25 * p["B"] * x_plus ** 2
              + (C0 / 6.0) * x_plus ** 3)
    return float(-v_true)  # V0(0) = 0; T0 = −V(x+)


def chi_residue(delta0: float, m_bar: float = M_BAR,
                e_bar: float = E_BAR) -> float:
    """Residuo de desequilibrio del falso vacío: ⟨χ⟩ ≃ (ē/m̄²)·δ0
    (Prop. 3.4) — la semilla del Modo de Memoria (Cap. 10)."""
    return (e_bar / m_bar ** 2) * delta0
