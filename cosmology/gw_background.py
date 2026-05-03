"""III.G — Fondo de ondas gravitacionales del retroceso entrópico.

Tratado §C, Ecs. 469-470. Extiende `mcmc_ontology.lqg_geometry` con:
  · h_c(f) (characteristic strain)
  · SNR PTA proyectada (NANOGrav/PPTA/CPTA/EPTA/SKA)
  · Comparativa con datasets públicos
  · Espectro SMBH (power-law) para ajuste competitivo

Predicción MCMC (Ec. 470):
    Ω_GW(f) = Ω_ret · (f τ_signature)² / cosh²(π f τ_signature / 2)

con f_pico ≈ 1/(π τ_signature) ≈ 2 × 10⁻⁸ Hz (banda PTA), Ω_ret ≈ 1.2×10⁻⁹.
La firma sech² es ESTRUCTURALMENTE DISTINTA del power-law de fusiones SMBH
(α ≈ 2/3) y permite distinguir los dos fondos en un análisis conjunto.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from mcmc_ontology.lqg_geometry import (
    Omega_GW_return,
    TAU_SIGNATURE_S,
)


# --- Constantes para PTA ---
H0_SI = C.H0_MCMC * 1e3 / 3.086e22   # H₀ en s⁻¹
SECONDS_PER_YEAR = 3.156e7


def Omega_GW_MCMC(f_Hz, tau_signature_s: float = TAU_SIGNATURE_S,
                  Omega_ret: float = 1.2e-9):
    """Re-exporta `mcmc_ontology.lqg_geometry.Omega_GW_return` (Ec. 470)."""
    return Omega_GW_return(f_Hz, tau_signature_s=tau_signature_s,
                            Omega_ret_amp=Omega_ret)


def Omega_GW_SMBH(f_Hz, A_cp: float = 2.4e-15, gamma: float = 13.0 / 3.0):
    """Fondo SMBH como power-law (referencia para discriminación PTA).

        Ω_GW^SMBH(f) = (2π²/3 H₀²) · A_cp² · f² · (f/f_yr)^(5-γ) · f_yr²
    """
    f = np.asarray(f_Hz, dtype=float)
    f_yr = 1.0 / SECONDS_PER_YEAR
    return (2.0 * np.pi ** 2 / 3.0) * A_cp ** 2 * (f / f_yr) ** (5.0 - gamma) * f_yr ** 2


def characteristic_strain(f_Hz, **kwargs):
    """h_c(f) = √(3 H₀² / (2π²) · Ω_GW(f) / f²)  (Apéndice C)."""
    f = np.asarray(f_Hz, dtype=float)
    Omega = Omega_GW_MCMC(f, **kwargs)
    return np.sqrt(3.0 * H0_SI ** 2 / (2.0 * np.pi ** 2) * Omega / f ** 2)


def snr_pta(N_pulsars: int = 150, T_yr: float = 15.0,
            sigma_ns: float = 50.0,
            tau_signature_s: float = TAU_SIGNATURE_S,
            Omega_ret: float = 1.2e-9) -> float:
    """SNR proyectada para un PTA con N_pulsars pulsares y timing rms σ.

    Estimación heurística (Tratado, Apéndice C):

        SNR ≈ √(N_p · T_obs) · (h_c(f_pico) / σ_timing) · √(B_eff / f_pico)

    Para SKA-PTA con 150 pulsares, 15 yr, σ ≈ 50 ns: SNR ≈ 2.5 (Tratado).
    """
    f_grid = np.logspace(-10, -5, 4000)
    Omega = np.array([
        Omega_GW_MCMC(fi, tau_signature_s=tau_signature_s, Omega_ret=Omega_ret)
        for fi in f_grid
    ])
    h_c = np.sqrt(3.0 * H0_SI ** 2 / (2.0 * np.pi ** 2) * Omega
                  / np.maximum(f_grid, 1e-30) ** 2)
    i_peak = int(np.argmax(Omega))
    f_peak = f_grid[i_peak]
    h_peak = h_c[i_peak]
    T_s = T_yr * SECONDS_PER_YEAR
    sigma_s = sigma_ns * 1e-9
    # Banda efectiva ≈ 1/T_obs (resolución espectral)
    B_eff = 1.0 / T_s
    return float(np.sqrt(N_pulsars * T_s) * (h_peak * f_peak / sigma_s)
                 * np.sqrt(B_eff / f_peak))


# --- Datasets públicos PTA ---
PTA_DATASETS = {
    "NANOGrav_15yr": {
        "A_cp_central": 2.4e-15,
        "gamma":        13.0 / 3.0,
        "T_yr":         15,
        "N_pulsars":    67,
        "ref":          "Agazie+2023 ApJL 951 L8",
        "url":          "https://nanograv.org/15yr-data-release",
    },
    "PPTA_DR3": {
        "A_cp_central": 2.2e-15,
        "gamma":        13.0 / 3.0,
        "T_yr":         18,
        "N_pulsars":    32,
        "ref":          "Reardon+2023 ApJL 951 L6",
        "url":          "https://www.atnf.csiro.au/research/pulsar/ppta/",
    },
    "EPTA_DR2": {
        "A_cp_central": 2.5e-15,
        "gamma":        13.0 / 3.0,
        "T_yr":         24.7,
        "N_pulsars":    25,
        "ref":          "EPTA+InPTA 2023 A&A 678 A50",
        "url":          "https://www.epta.eu.org/aed.html",
    },
    "CPTA_DR1": {
        "A_cp_central": 2.0e-15,
        "gamma":        13.0 / 3.0,
        "T_yr":         3.5,
        "N_pulsars":    57,
        "ref":          "Xu+2023 RAA 23 075024",
        "url":          "http://cpta.bao.ac.cn/",
    },
    "SKA_PTA_projected_15yr": {
        "T_yr":      15,
        "N_pulsars": 150,
        "sigma_ns":  50.0,
        "Omega_h2_sensitivity": 2e-12,
        "ref": "Janssen+2015 PoS AASKA14 037",
        "projection": True,
    },
}


def predict_for_pta(name: str) -> dict:
    """Predicción MCMC para un PTA dado (señal vs ruido proyectado)."""
    info = PTA_DATASETS[name]
    if name.startswith("SKA"):
        snr = snr_pta(N_pulsars=info["N_pulsars"], T_yr=info["T_yr"],
                       sigma_ns=info["sigma_ns"])
        return {"name": name, "SNR_MCMC_proj": snr,
                "ref": info["ref"], "projection": True}
    # Para PTAs activos: comparar Ω_GW MCMC con A_cp observado
    f_yr = 1.0 / SECONDS_PER_YEAR
    Omega_obs = (2.0 * np.pi ** 2 / 3.0) * info["A_cp_central"] ** 2 * f_yr ** 2
    Omega_MCMC_at_fyr = float(Omega_GW_MCMC(f_yr))
    return {
        "name":               name,
        "Omega_obs_at_fyr":   Omega_obs,
        "Omega_MCMC_at_fyr":  Omega_MCMC_at_fyr,
        "ratio_MCMC_obs":     (Omega_MCMC_at_fyr / Omega_obs
                                if Omega_obs > 0 else float("inf")),
        "ref": info["ref"],
    }


def discriminate_sech2_vs_powerlaw(f_Hz_arr) -> dict:
    """Diagnóstico de distinguibilidad: ratio Ω_MCMC / Ω_SMBH y log-pendientes.

    Para un análisis PTA conjunto, el cociente Ω_MCMC/Ω_SMBH varía fuertemente
    con f en torno al pico sech², lo que permite identificar la firma MCMC.
    """
    f = np.asarray(f_Hz_arr, dtype=float)
    O_mcmc = np.array([Omega_GW_MCMC(fi) for fi in f])
    O_smbh = Omega_GW_SMBH(f)
    return {
        "f_Hz":          f,
        "Omega_MCMC":    O_mcmc,
        "Omega_SMBH":    O_smbh,
        "ratio":         O_mcmc / np.maximum(O_smbh, 1e-30),
    }
