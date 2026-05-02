"""B2 — N_gen = 3 algebraico, modos espinoriales F1, F2, F3.

La estructura algebraica Cl(3,0) en V₃D admite exactamente tres modos
espinoriales independientes. Esto fija N_gen = 3 SIN introducir parámetros
libres.

Cada familia emerge en un sello concreto:
  F1 emerge en C1 (S=0.009): tau, t, b, nu_tau
  F2 emerge en C2 (S=0.099): mu,  c, s, nu_mu
  F3 emerge en C3 (S=0.999): e,   u, d, nu_e
"""

from mcmc_ontology import constants as C


def n_generations() -> int:
    """Número de familias fermiónicas (algebraico)."""
    return C.N_GEN


def emergence_seal(family: str) -> str:
    """Sello en el que emerge la familia dada (F1→C1, F2→C2, F3→C3)."""
    return {"F1": "C1", "F2": "C2", "F3": "C3"}[family]


def family_fermions(family: str) -> dict:
    """Mapa familia → fermiones (lepton, up, down, nu)."""
    return dict(C.FAMILY_FERMIONS[family])
