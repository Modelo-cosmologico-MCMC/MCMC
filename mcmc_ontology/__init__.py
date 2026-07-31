"""mcmc_ontology — Núcleo ontológico del MCMC.

Contiene los ocho axiomas del Tratado de Fundamentos (v35, §1.2), los
parámetros globales, el mapa S↔t↔z↔a, el potencial del Campo de Adrián,
las álgebras de Clifford por régimen y los sellos ontológicos.
"""

from . import axioms
from . import constants
from . import S_map
from . import potential
from . import clifford_algebra
from . import seals

__all__ = ["axioms", "constants", "S_map", "potential", "clifford_algebra",
           "seals"]
