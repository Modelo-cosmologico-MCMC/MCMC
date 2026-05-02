"""B1 — Escalas v_n y pesos ϑ_n.

Pesos naïve (logarítmicos) y efectivos, con correcciones de curvatura
P3 y QCD running (M1).
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C


def naive_weight(seal: str, ref: str = "C3") -> float:
    """ϑ_naïve(n) = v_n / v_ref (escala lineal)."""
    return C.V_GEV[seal] / C.V_GEV[ref]


def log_weight(seal: str, ref: str = "C3") -> float:
    """log ratio entre la escala n y la de referencia."""
    return float(np.log(C.V_GEV[seal] / C.V_GEV[ref]))


def vev_table() -> dict[str, float]:
    """Tabla {sello: v_n} en GeV."""
    return dict(C.V_GEV)
