"""III.D — Lattice SU(3) con acoplamiento dependiente de S.

Tratado §I (pp. 261-313), Ecs. 734-736.

Acoplamiento de Wilson dependiente del entrópico (Ec. 736):
    β(S) = β₀ + β₁ · exp(-b_S (S - S_3))

Acción modificada (Ec. 735):
    S_YM[U] = β(S) · Σ_□ [1 - (1/3) Re Tr U_□]

Predicciones de espectro glueball (Tabla del Tratado):
    0⁺⁺ : 1755 ± 50 MeV   (vs HotQCD 2024: 1730 ± 50, BMW 2023: 1685 ± 45)
    2⁺⁺ : 2415 ± 90 MeV   (vs HotQCD: 2390 ± 90)
    0⁻⁺ : 2575 ± 110 MeV  (vs HotQCD: 2560 ± 110)

Este módulo provee la estructura de cálculo (β(S), extracción de mass gap,
extrapolación al continuo) sin invocar simulaciones de retículo completas
(que requieren MILC/Chroma). El uso típico es analítico: dado un correlador
medido externamente, extraer m_gap y compararlo con la predicción MCMC.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def beta_S(S: float, beta0: float = 6.10, beta1: float = 0.50,
           b_S: float = 10.0, S3: float = 1.000) -> float:
    """Acoplamiento Wilson β(S) (Ec. 736).

    β(S) = β₀ + β₁ · exp(-b_S (S - S_3)).

    En S = S_3 = 1.000 → β = β₀ + β₁ (régimen pre-Big Bang).
    En S >> S_3 → β → β₀ (running QCD estándar).
    """
    return float(beta0 + beta1 * np.exp(-b_S * (S - S3)))


def mass_gap_extraction(correlators: np.ndarray, a_fm: float,
                        T_lat: int, t_min: int = 4) -> dict:
    """Extracción del mass gap del correlador C(t).

    Modelo: C(t) = A · [exp(-m·t) + exp(-m·(T-t))]   (BC periódicas)

    Args:
        correlators: array C(t) (t = 0..T-1).
        a_fm: paso del retículo en fm.
        T_lat: extensión temporal (debe coincidir con len(correlators)).
        t_min: t mínimo para evitar contaminación de estados excitados.

    Returns:
        dict con `m_lat` (en unidades de retículo) y `m_GeV`.
    """
    correlators = np.asarray(correlators, dtype=float)
    t_arr = np.arange(t_min, T_lat // 2)
    C_t = correlators[t_min: T_lat // 2]

    def model(t, A, m):
        return A * (np.exp(-m * t) + np.exp(-m * (T_lat - t)))

    popt, _ = curve_fit(model, t_arr, C_t, p0=[abs(C_t[0]), 0.5])
    m_lat = float(popt[1])
    # Conversión: m [GeV] = m_lat / (a_fm · 5.068e-3 fm·GeV⁻¹)
    m_GeV = m_lat / (a_fm * 5.068e-3)
    return {"m_lat": m_lat, "m_GeV": m_GeV, "a_fm": a_fm}


def continuum_extrapolation(m_gap_arr: np.ndarray,
                            a_fm_arr: np.ndarray) -> dict:
    """Extrapolación al continuo m(a → 0) con corrección O(a²).

    m(a) = m(0) + c1 · a²
    """
    m_gap = np.asarray(m_gap_arr, dtype=float)
    a2 = np.asarray(a_fm_arr, dtype=float) ** 2
    coeffs = np.polyfit(a2, m_gap, 1)  # [pendiente c1, intercepto m(0)]
    return {"m_continuum": float(coeffs[1]), "c1_a2": float(coeffs[0])}


# --- Configuraciones del Tratado (Apéndice I) ---
LATTICE_CONFIGS = {
    "32x32x32x64": {"a_inv_GeV": 2.0, "beta_W": 6.10, "sweeps": 100_000},
    "48x48x48x128": {"a_inv_GeV": 3.0, "beta_W": 6.30, "sweeps":  50_000},
    "64x64x64x128": {"a_inv_GeV": 4.0, "beta_W": 6.50, "sweeps":  20_000},
}


# --- Predicción y comparativa observacional ---
GLUEBALL_PREDICTIONS_MCMC = {
    "0++": {"m_MeV": 1755, "sigma_MeV": 50},
    "2++": {"m_MeV": 2415, "sigma_MeV": 90},
    "0-+": {"m_MeV": 2575, "sigma_MeV": 110},
}

GLUEBALL_DATA = {
    "HotQCD_2024": {
        "0++": {"m_MeV": 1730, "sigma_MeV": 50},
        "2++": {"m_MeV": 2390, "sigma_MeV": 90},
        "0-+": {"m_MeV": 2560, "sigma_MeV": 110},
        "ref": "Bazavov+2024 arXiv:2402",
    },
    "BMW_2023": {
        "0++": {"m_MeV": 1685, "sigma_MeV": 45, "sigma_sys_MeV": 85},
        "ref": "BMW Collaboration arXiv:2311.18229",
    },
    "Lucini_Teper_2001": {
        "0++_over_sqrt_sigma": 3.256,
        "0++_err":            0.032,
        "ref": "Lucini+Teper 2001 PRD 64 014501",
    },
}


def compare_with_data(state: str = "0++") -> dict:
    """Predicción MCMC vs HotQCD/BMW para un estado dado."""
    pred = GLUEBALL_PREDICTIONS_MCMC[state]
    out = {"state": state, "MCMC_MeV": pred["m_MeV"],
           "MCMC_sigma": pred["sigma_MeV"]}
    for src in ("HotQCD_2024", "BMW_2023"):
        info = GLUEBALL_DATA[src]
        if state in info:
            obs = info[state]
            dev = abs(pred["m_MeV"] - obs["m_MeV"]) / obs["m_MeV"] * 100
            out[src] = {"m_MeV": obs["m_MeV"], "deviation_pct": dev,
                         "ref": info["ref"]}
    return out


def E_min_lattice_units(a_fm: float, k_GeV: float = 1.22e19,
                       delta_S: float = 1e-3) -> float:
    """E_min = k·ΔS expresado en unidades de retículo (a·m).

    E_min en GeV → E_min·a en lattice.
    """
    E_min_GeV = k_GeV * delta_S
    return float(E_min_GeV * a_fm * 5.068e-3)
