"""P3 — Curvatura V''_total y Δm_eff.

Curvatura total en el vacío Phi* = v_n:
    V''_total(v_n) = 8 β_n v_n^2 + d_n (m_P(S_n) v_n sqrt(d_n))^2

Δm_eff,n = sqrt(V''_total) / (m_P(S_n) · K_norm)
con K_norm = 679.14 GeV calibrado desde C4 (donde β_4=10^7 domina I_dD).

IMPORTANTE: La fórmula numérica cruda sólo es válida en C4 (ancla de
calibración). En C1 (escala Planck) y C2 (escala GUT) las VEVs en GeV
hacen que I_dD diverja a ~1e16/1e13, lo cual NO es físico en S-space.
Los Δm_eff que entran en la fórmula maestra de masas son los CALIBRADOS
(`mcmc_ontology.constants.DELTA_M_EFF_CAL`), expuestos por
`mcmc_ontology.potential.delta_m_eff`.

Ver `mcmc_ontology.potential` para las primitivas.
"""

from mcmc_ontology import constants as C
from mcmc_ontology.potential import (
    V_pp_quartic,
    V_pp_kinetic,
    V_pp_total,
    delta_m_eff,         # calibrado (canónico)
    delta_m_eff_raw,     # crudo numérico (sólo válido en C4)
)


def K_norm_from_C4() -> float:
    """K_norm calibrado desde C4 (donde I_dD es despreciable frente al cuártico).

    Por construcción coincide con C.K_NORM (= 679.14 GeV) en la calibración
    nominal del modelo.
    """
    return C.K_NORM


def curvature_table() -> dict:
    """Tabla por sello con V''_quartic, V''_kinetic, V''_total, Δm_eff (cal y raw)."""
    out = {}
    for seal in ("C1", "C2", "C3", "C4"):
        out[seal] = {
            "V_pp_quartic":    V_pp_quartic(seal),
            "V_pp_kinetic":    V_pp_kinetic(seal),
            "V_pp_total":      V_pp_total(seal),
            "delta_m_eff":     delta_m_eff(seal),       # calibrado (canónico)
            "delta_m_eff_raw": delta_m_eff_raw(seal),   # crudo (sólo C4 físico)
        }
    return out


def calibrated_table() -> dict:
    """Tabla canónica de Δm_eff por sello (Tratado, Tabla P3)."""
    return dict(C.DELTA_M_EFF_CAL)
