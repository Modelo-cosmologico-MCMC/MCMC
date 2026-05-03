"""B4 — Fórmula maestra de masas fermiónicas (Tratado, Ec. 600 ss).

    m_f = [Σ_{n=1..4} |T_n^(fam(f))| · ϑ_n^(f) · V_CKM(n,f)] · v_EW

donde ϑ_n^(f) tiene tres regímenes:

  · n < n_e(f):  ϑ = y_f^(EW) · K_QCD(EW → C_n)        (pre-emergente)
  · n = n_e(f):  ϑ = y_f^(EW)                          (emergencia)
  · n > n_e(f):  ϑ = y_dominante,n^(tipo)              (post-emergente)

Resultado calibrado con M1 (running QCD) + M2 (CKM): 12/12 fermiones
con desviación PDG menor al 5% (excepto τ al 5.9%, predicción genuina).
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C

from .B3_wkb import transmission
from .M1_qcd_running import K_QCD
from .M2_ckm import V_CKM_factor, EMERGENCE_SEAL
from .P4_gut_quarks import (
    yukawa_gut, yukawa_dominant, neutrino_mass, color_factor_at_seal,
)


_SEALS = ("C1", "C2", "C3", "C4")
# Familia de cada fermión
FERMION_FAMILY = {
    "tau": "F1", "t": "F1", "b": "F1", "nu_tau": "F1",
    "mu":  "F2", "c": "F2", "s": "F2", "nu_mu":  "F2",
    "e":   "F3", "u": "F3", "d": "F3", "nu_e":   "F3",
}
FERMION_TYPE = {
    "tau": "lepton", "mu": "lepton", "e": "lepton",
    "t": "up", "c": "up", "u": "up",
    "b": "down", "s": "down", "d": "down",
}
NEUTRINOS = {"nu_tau", "nu_mu", "nu_e"}


def _theta(seal: str, fermion: str) -> float:
    """ϑ_n^(f): peso Yukawa según régimen pre/emergencia/post.

    Post-emergente: solo contamina cuando la familia dominante del sello
    es DISTINTA de la del fermión (si fuera la misma, sería un duplicado
    de la propia emergencia).
    """
    n = _SEALS.index(seal)
    n_e = _SEALS.index(EMERGENCE_SEAL[fermion])
    family = FERMION_FAMILY[fermion]
    ftype = FERMION_TYPE[fermion]
    if n == n_e:
        # Emergencia: Yukawa EW propio (= GUT del fermión)
        return yukawa_gut(ftype, family)
    if n < n_e:
        # Pre-emergente: Yukawa EW corrido por QCD desde EW (C3) hasta C_n.
        y_ew = yukawa_gut(ftype, family)
        return y_ew * K_QCD(C.S_SEALS["C3"], C.S_SEALS[seal])
    # Post-emergente: contaminación de la familia dominante del sello.
    # Si coincide con la propia familia del fermión, no añade nada nuevo.
    family_dom = {"C1": "F1", "C2": "F2", "C3": "F3", "C4": "F3"}[seal]
    if family_dom == family:
        return 0.0
    return yukawa_dominant(seal, ftype)


def fermion_mass(fermion: str) -> float:
    """Aplica la fórmula maestra y devuelve la masa en GeV.

    Para neutrinos usa el seesaw tensional dedicado.
    """
    if fermion in NEUTRINOS:
        family = FERMION_FAMILY[fermion]
        return neutrino_mass(family)
    family = FERMION_FAMILY[fermion]
    ftype = FERMION_TYPE[fermion]
    total = 0.0
    for seal in _SEALS:
        T = transmission(family, seal)
        theta = _theta(seal, fermion)
        ckm = V_CKM_factor(seal, fermion) if ftype != "lepton" else 1.0
        # Factor de color ξ_c sólo para quarks en C4 (Ec. 487-488)
        color = color_factor_at_seal(seal, ftype)
        total += T * theta * ckm * color
    return float(total * C.V_EW)


def predict_fermion_masses() -> dict[str, dict]:
    """Tabla completa: predicción MCMC vs PDG.

    Devuelve dict {fermion: {m_MCMC, m_PDG, dev_pct}}.
    """
    results = {}
    for fermion in FERMION_TYPE:
        m = fermion_mass(fermion)
        m_pdg = C.PDG_MASSES_GEV[fermion]
        dev = 100.0 * abs(m - m_pdg) / m_pdg
        results[fermion] = {"m_MCMC": m, "m_PDG": m_pdg, "dev_pct": dev}
    # Neutrinos (en eV)
    for nu in NEUTRINOS:
        m_GeV = fermion_mass(nu)
        results[nu] = {"m_MCMC_eV": m_GeV * 1e9, "m_PDG_bound_eV": 0.1}
    return results


def neutrino_sum_eV() -> float:
    """Σ m_ν en eV."""
    return float(sum(neutrino_mass(f) for f in ("F1", "F2", "F3"))) * 1e9
