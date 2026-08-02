"""M1 — Running QCD entrópico.

Flujo entrópico (Ec. 847, Tratado Unificado v32):
    d g_i^{-2} / d S = b_i / (8π² λ)
con b_1=-41/10, b_2=+19/6, b_3=+7.

RGE 1-loop QCD (n_f=6, b0=7, γ_m=4):
    K_QCD(S1→S2) = [α_s(S2) / α_s(S1)]^(4/7)

α_s(S) = 1 / α_s^{-1}(S), interpolado de la Tabla 41 (Tratado Unificado v32).

Efecto principal: el pre-emergente del quark u en C2 usa
y_u^(GUT) = y_u × K_QCD(EW→GUT) = y_u × 0.5645, eliminando la
contaminación cruzada 133% → 0.1%.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from mcmc_ontology.S_map import alpha3_inv

# Coeficientes RGE 1-loop SM
B_COEFF = {"U1": -41.0 / 10.0, "SU2": 19.0 / 6.0, "SU3": 7.0}
GAMMA_M_QCD = 4.0  # dimensión anómala de masa, n_f=6
B0_QCD = 7.0


def alpha_s(S: float | np.ndarray) -> float | np.ndarray:
    """α_s(S) = 1 / α_3^{-1}(S)."""
    return 1.0 / alpha3_inv(S)


def K_QCD(S_from: float, S_to: float) -> float:
    """Factor de running de Yukawa por QCD (1-loop).

    K = [α_s(S_to) / α_s(S_from)]^(γ_m / b0) = [α_s(S_to)/α_s(S_from)]^(4/7).
    """
    a_from = alpha_s(S_from)
    a_to = alpha_s(S_to)
    return float((a_to / a_from) ** (GAMMA_M_QCD / B0_QCD))


def K_EW_to_GUT() -> float:
    """K_QCD(EW → GUT) — debe valer ≈ 0.5645."""
    return K_QCD(C.S_SEALS["C3"], C.S_SEALS["C2"])


def yukawa_run(y_ref: float, S_from: float, S_to: float) -> float:
    """Aplica el running QCD a un Yukawa entre dos escalas S."""
    return y_ref * K_QCD(S_from, S_to)
