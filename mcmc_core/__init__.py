"""
MCMC Core - Modelo Cosmológico de Múltiples Colapsos
=====================================================

Implementación del modelo MCMC que describe la evolución del universo
desde un estado primordial de máxima tensión hasta el estado actual.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Módulos:
    - bloque0_estado_primordial: Estado inicial Mp0, Ep0, tensión máxima
    - bloque1_pregeometria: Tasa de colapso k(S), integral entrópica
    - bloque2_cosmologia: Ecuaciones de Friedmann modificadas MCMC

Ontología MCMC:
    - Sellos entrópicos: S0=0.000, S1=0.010, S2=0.100, S3=1.000, S4=1.001
    - Polarización masa-energía: P_ME = (Mp - Ep)/(Mp + Ep)
    - Trayectoria: P_ME evoluciona de +1 (masa domina) a -1 (energía domina)
"""

from .bloque0_estado_primordial import (
    EstadoPrimordial,
    SELLOS,
    Mp0,
    Ep0,
    calcular_tension,
    calcular_P_ME,
)

from .bloque1_pregeometria import (
    Pregeometria,
    tasa_colapso_k,
    integral_entropica,
    calcular_epsilon,
    calcular_Mp_Ep,
    K0, A1, A2, A3,
)

from .bloque2_cosmologia import (
    CosmologiaMCMC,
    E_LCDM,
    E_MCMC,
    Lambda_relativo,
    H0, OMEGA_M, OMEGA_LAMBDA, DELTA_LAMBDA,
)

__version__ = "1.0.0"
__author__ = "Adrián Martínez Estellés"
__email__ = "adrianmartinezestelles92@gmail.com"
__license__ = "Propietaria - Ver LICENSE"

__all__ = [
    # Bloque 0
    "EstadoPrimordial",
    "SELLOS",
    "Mp0",
    "Ep0",
    "calcular_tension",
    "calcular_P_ME",
    # Bloque 1
    "Pregeometria",
    "tasa_colapso_k",
    "integral_entropica",
    "calcular_epsilon",
    "calcular_Mp_Ep",
    "K0", "A1", "A2", "A3",
    # Bloque 2
    "CosmologiaMCMC",
    "E_LCDM",
    "E_MCMC",
    "Lambda_relativo",
    "H0", "OMEGA_M", "OMEGA_LAMBDA", "DELTA_LAMBDA",
]
