"""Mapa S ↔ t ↔ z ↔ a.

Realiza los axiomas 6 (atemporalidad pre-geométrica) y 7 (Ley de Cronos)
del Tratado de Fundamentos (v35, §1.2).

S es el índice entrópico (parámetro de estructuración tensional). NO es tiempo.
El tiempo emerge SOLO en S ≥ 1.001 mediante la Ley de Cronos:

    t_rel(S) = C · S^alpha,   alpha ≈ 1

En forma operativa (para integración):

    d ln a / dS = C(S)
    dt_rel / dS = T(S) · N(S),   N(S) = exp[Phi_ten(S)]
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def s_to_t_rel(S: np.ndarray | float, alpha: float = 1.0,
               C_const: float = 1.0, S_birth: float | None = None) -> np.ndarray | float:
    """Tiempo relativo emergente para S ≥ S_birth (≡ C4).

    Para S < S_birth devuelve 0 (tiempo aún no nacido).
    """
    if S_birth is None:
        S_birth = C.S_SEALS["C4"]
    S_arr = np.asarray(S, dtype=float)
    out = np.where(S_arr >= S_birth, C_const * (S_arr - S_birth) ** alpha, 0.0)
    return float(out) if np.isscalar(S) else out


def m_P(S: np.ndarray | float) -> np.ndarray | float:
    """Perfil lineal de masa primordial m_P(S) entre C1 y C4.

    m_P(S) = M_P0 - (M_P0 - M_P_eq) * (S - S1) / (S4 - S1)
    """
    S1 = C.S_SEALS["C1"]
    S4 = C.S_SEALS["C4"]
    S_arr = np.asarray(S, dtype=float)
    frac = (S_arr - S1) / (S4 - S1)
    out = C.MP_0 - (C.MP_0 - C.MP_EQ) * frac
    out = np.clip(out, 0.0, 1.0)
    return float(out) if np.isscalar(S) else out


def s_to_a(S: np.ndarray | float, S_today: float | None = None) -> np.ndarray | float:
    """Mapa monótono S→a en el régimen post-geométrico (S ≥ C4).

    Aproximación operativa: a(S) = exp(integral d ln a / dS).
    Para uso heurístico fuera del integrador completo de Cronos.
    El default S_today = 95.0 es la parametrización del Unificado
    (bloque LEGACY_V32 en constants.py); la v35 no la contiene.
    """
    if S_today is None:
        S_today = 95.0  # valor operativo v32 (LEGACY_V32)
    S4 = C.S_SEALS["C4"]
    S_arr = np.asarray(S, dtype=float)
    out = np.where(
        S_arr >= S4,
        np.exp(-(S_today - S_arr) / (S_today - S4)),
        0.0,
    )
    return float(out) if np.isscalar(S) else out


def a_to_z(a: np.ndarray | float) -> np.ndarray | float:
    """Factor de escala → redshift."""
    return 1.0 / np.asarray(a, dtype=float) - 1.0


def z_to_a(z: np.ndarray | float) -> np.ndarray | float:
    """Redshift → factor de escala."""
    return 1.0 / (1.0 + np.asarray(z, dtype=float))


def alpha3_inv(S: np.ndarray | float) -> np.ndarray | float:
    """α₃⁻¹(S) por interpolación lineal sobre la Tabla 41 del Tratado."""
    table = np.asarray(C.ALPHA3_INV_TABLE)
    return np.interp(np.asarray(S, dtype=float), table[:, 0], table[:, 1])
