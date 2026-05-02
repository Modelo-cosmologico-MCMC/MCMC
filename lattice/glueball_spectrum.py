"""Espectro glueball SU(3) en el modelo MCMC.

m^MCMC_J^PC(a, L; λ_IR) = m_J^PC(a, L) + Δm(λ_IR)
con Δm → 0 cuando λ_IR → 0.

Tabla nominal MCMC (vs. QCD puro lattice):

|  Estado J^PC | QCD puro [GeV]   | MCMC [GeV]      | |Δm|/m_std |
|--------------|------------------|-----------------|------------|
|  0^++        | 1.73 ± 0.05      | 1.75 ± 0.05     |   1.2%     |
|  2^++        | 2.39 ± 0.09      | 2.41 ± 0.09     |   0.8%     |
|  0^-+        | 2.56 ± 0.11      | 2.57 ± 0.11     |   0.4%     |
"""

from __future__ import annotations

GLUEBALL_TABLE = {
    "0++": {"m_qcd": 1.73, "sig_qcd": 0.05, "m_mcmc": 1.75, "sig_mcmc": 0.05},
    "2++": {"m_qcd": 2.39, "sig_qcd": 0.09, "m_mcmc": 2.41, "sig_mcmc": 0.09},
    "0-+": {"m_qcd": 2.56, "sig_qcd": 0.11, "m_mcmc": 2.57, "sig_mcmc": 0.11},
}


def relative_shift(state: str) -> float:
    """|Δm|/m_std para un estado dado."""
    row = GLUEBALL_TABLE[state]
    return abs(row["m_mcmc"] - row["m_qcd"]) / row["m_qcd"]


def continuum_extrapolation(m_a_L: float, lambda_IR: float, c: float = 0.02) -> float:
    """m_continuum = m(a,L) + c · λ_IR (modelo lineal)."""
    return m_a_L + c * lambda_IR
