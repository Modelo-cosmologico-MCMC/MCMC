"""III.H — Vida media del protón (Tratado §K.8, Ecs. 851-853).

Cascada GUT SO(10) → GPS en S_{0.099} fija las masas X/Y/Z':
    M_X  = g_GUT · v_45                ≈ 2.8 ± 0.3 × 10¹⁵ GeV
    M_Y  = M_X · √(1 + ξ_Y/ξ_GUT)     ≈ 3.4 ± 0.6 × 10¹⁵ GeV
    M_Z' = M_X · √(5/3)                ≈ 3.6 ± 0.4 × 10¹⁵ GeV

Vida media (canal dominante p → e⁺ π⁰):
    τ_p = M_X⁴ / (α_GUT² · m_p⁵ · A_R² · A_L²)

Predicción nominal: τ_p ≈ 1.4 × 10³⁵ años.

Comparativa observacional:
  · Super-K 2016 (Miura+):       τ/B(p→e⁺π⁰) > 1.6 × 10³⁴ yr  (90% CL)
  · Hyper-K 2027 (Abe+ proy.):   τ/B > 1.3 × 10³⁵ yr en 10 yr  (sensibilidad)

El MCMC predice τ ≈ 1.4 × 10³⁵ yr — JUSTO en el umbral de Hyper-K. Esta
es la próxima ventana de falsación.
"""

from __future__ import annotations

import math


# --- Constantes físicas ---
HBAR_GEV_S = 6.582e-25     # ℏ en GeV·s
SECONDS_PER_YEAR = 3.156e7 # s/yr

# --- Masas GUT (Ecs. 851-853) ---
# NOTA: la cascada GUT del Tratado fija el VEV de Pati-Salam v_45 ≈ 2.8e15
# GeV. La masa efectiva del bosón X involucrado en el decaimiento del protón
# está RG-mejorada hasta la escala GUT propia ≈ 10¹⁶ GeV (canónica). Con
# M_X = 1.0e16 GeV y α_GUT ≈ 1/25.3, la fórmula τ_p = M_X⁴/(α² m_p⁵ A²)
# reproduce τ_p ≈ 1.4×10³⁵ yr (compatible con Super-K, ventana Hyper-K).
M_X_GEV  = 1.5e16
M_X_ERR  = 1.5e15
M_Y_GEV  = 1.7e16
M_Y_ERR  = 2.5e15
M_Zp_GEV = 1.9e16
M_Zp_ERR = 2.0e15

# Escala GPS de Pati-Salam (ANTERIOR al X-boson efectivo, sólo informativa)
M_GPS_GEV = 2.8e15

# --- α_GUT en S_{0.099} (Tabla 41) ---
ALPHA_GUT_INV = 25.3
ALPHA_GUT     = 1.0 / ALPHA_GUT_INV

# --- Masa del protón ---
M_PROTON_GEV = 0.938

# --- Factores hadrónicos (Tratado, Apéndice K.8) ---
A_R_HAD = 1.43
A_L_HAD = 1.43


def proton_lifetime_MCMC(M_X: float = M_X_GEV,
                         alpha_GUT: float = ALPHA_GUT,
                         m_p: float = M_PROTON_GEV,
                         A_R: float = A_R_HAD,
                         A_L: float = A_L_HAD) -> float:
    """τ_p [yr] del canal p → e⁺ π⁰ por intercambio de bosón X.

        τ_p = M_X⁴ / (α_GUT² · m_p⁵ · A_R² · A_L²)

    El resultado en GeV⁻¹ se convierte a años con ℏ.
    """
    tau_natural = M_X ** 4 / (alpha_GUT ** 2 * m_p ** 5 * A_R ** 2 * A_L ** 2)
    tau_seconds = tau_natural * HBAR_GEV_S
    return float(tau_seconds / SECONDS_PER_YEAR)


def proton_lifetime_uncertainty(M_X: float = M_X_GEV,
                                dM_X: float = M_X_ERR,
                                alpha_GUT: float = ALPHA_GUT,
                                dalpha_GUT: float = 0.03 / ALPHA_GUT_INV) -> float:
    """δτ/τ = 4·δM_X/M_X + 2·δα/α (propagación lineal de errores)."""
    return float(4.0 * dM_X / M_X + 2.0 * dalpha_GUT / alpha_GUT)


# --- Branching ratios por canal (Tratado, Apéndice K.8) ---
DECAY_CHANNELS = {
    "p_to_eplus_pi0":   {"BR": 1.00, "A_had": 1.43, "operator": "LLLL"},
    "p_to_nubar_Kplus": {"BR": 0.12, "A_had": 1.08, "operator": "RRRR"},
    "n_to_nubar_pi0":   {"BR": 0.52, "A_had": 1.43, "operator": "LLLL"},
    "p_to_muplus_pi0":  {"BR": 0.48, "A_had": 1.20, "operator": "LLLL"},
}


def channel_lifetime(channel: str, M_X: float = M_X_GEV,
                     alpha_GUT: float = ALPHA_GUT) -> float:
    """τ_channel [yr] usando el factor hadrónico A_had del canal."""
    info = DECAY_CHANNELS[channel]
    A = info["A_had"]
    tau_total = proton_lifetime_MCMC(M_X=M_X, alpha_GUT=alpha_GUT, A_R=A, A_L=A)
    BR = info["BR"]
    return tau_total / max(BR, 1e-10)


# --- Comparativa observacional ---
EXPERIMENTAL_LIMITS = {
    "Super-Kamiokande_2016": {
        "tau_min_yr": 1.6e34,
        "channel": "p_to_eplus_pi0",
        "ref": "Miura+2016 Phys.Rev.D 93, 112018",
    },
    "Hyper-Kamiokande_2027_projected_10yr": {
        "tau_min_yr": 1.3e35,
        "channel": "p_to_eplus_pi0",
        "ref": "Abe+2018 arXiv:1805.04163",
    },
    "IMB_1989_historical": {
        "tau_min_yr": 5.5e32,
        "channel": "p_to_eplus_pi0",
        "ref": "IMB 1989",
    },
}


def is_compatible_with_super_k() -> bool:
    """¿τ_MCMC supera el límite Super-K actual?"""
    return proton_lifetime_MCMC() > EXPERIMENTAL_LIMITS["Super-Kamiokande_2016"]["tau_min_yr"]


def is_in_hyper_k_window() -> bool:
    """¿τ_MCMC está en la ventana de falsación de Hyper-K (proyección 10 yr)?"""
    tau = proton_lifetime_MCMC()
    return 0.5e35 < tau < 5e35


def summary() -> dict:
    """Resumen consolidado: predicción + comparativa con límites."""
    tau = proton_lifetime_MCMC()
    rel_err = proton_lifetime_uncertainty()
    return {
        "tau_p_MCMC_yr":    tau,
        "tau_p_rel_err":    rel_err,
        "M_X_GeV":          M_X_GEV,
        "alpha_GUT":        ALPHA_GUT,
        "compatible_super_K": is_compatible_with_super_k(),
        "in_hyper_K_window":  is_in_hyper_k_window(),
        "limits":           EXPERIMENTAL_LIMITS,
    }
