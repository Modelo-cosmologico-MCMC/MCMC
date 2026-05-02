"""mcmc_ontology — Núcleo ontológico del MCMC.

Contiene los parámetros globales, el mapa S↔t↔z↔a, el potencial del Campo de
Adrián, las álgebras de Clifford por régimen y los sellos ontológicos.
"""

from . import constants
from . import S_map
from . import potential
from . import clifford_algebra
from . import seals

__all__ = ["constants", "S_map", "potential", "clifford_algebra", "seals"]
