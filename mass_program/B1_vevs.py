"""B1 — Escalas v_n y pesos ϑ_n.

Pesos naïve (logarítmicos) y efectivos, con correcciones de curvatura
P3 y QCD running (M1).

Nota de versión: V_GEV (Planck/GUT/EW/QCD por sello) es la asignación
del Tratado Unificado (v32, bloque LEGACY_V32 de constants.py). En la
v35, v3 = 246 GeV está sellada por V3+1D (S=1.001) y la escala QCD
emerge en S3 = 1.000 (D.1); Planck/GUT no tienen umbral asignado.
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
