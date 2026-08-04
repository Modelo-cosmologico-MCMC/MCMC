"""B2 — N_gen = 3 algebraico, modos espinoriales F1, F2, F3.

N_gen = 3 se identifica con la dimensión del módulo espinorial de la
Cadena de Álgebras en V3+1D (v35, Prop. 12.4; en v35.1, E4, se
reclasifica como IDENTIFICACIÓN estructural: la derivación que excluya
otras multiplicidades de sabor queda pendiente — sub-frente del
frente 7).

La asociación de cada familia a un sello de emergencia (F1↔C1, F2↔C2,
F3↔C3, con los fermiones de FAMILY_FERMIONS) es la presentación del
Tratado Unificado (v32) que este módulo implementa; en la v35 la
supervivencia de cada modo a través de los sellos la codifican los pesos
c_in del Funcional del Camino (Def. 12.3).
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
