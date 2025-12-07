"""
MCMC Core - Modelo Cosmológico de Múltiples Colapsos
=====================================================

Implementación del modelo MCMC que describe la evolución del universo
desde un estado primordial de máxima tensión hasta el estado actual,
basado en la DUALIDAD TENSIONAL MASA-ESPACIO.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

MÓDULOS PRINCIPALES:
    - bloque0_estado_primordial: Estado inicial Mp0, Ep0, Campo de Adrián
    - bloque1_pregeometria: Tasa de colapso k(S), integral entrópica
    - bloque2_cosmologia: Ecuaciones de Friedmann modificadas MCMC

MÓDULOS DE VALIDACIÓN CUÁNTICA:
    - qubit_tensorial: Estado cuántico |Φ(S)⟩ = α|00⟩ + β|11⟩
    - spin_network_lqg: Conexión con Loop Quantum Gravity

ONTOLOGÍA MCMC:
    - Sellos entrópicos: S0=0.000, S1=0.010, S2=0.100, S3=1.000, S4=1.001
    - Campo de Adrián: Φ_ten(S) = Mp(S)/Ep(S)
    - Polarización masa-espacio: P_ME = (Mp - Ep)/(Mp + Ep)
    - Trayectoria: P_ME evoluciona de +1 (masa domina) a -1 (espacio domina)

CONEXIONES TEÓRICAS:
    - Cosmología: Reducción de tensiones H0 y S8 mediante Λ_rel(z)
    - Física de partículas: Mass gap ontológico en Yang-Mills
    - Gravedad cuántica: Emergencia geométrica vía spin networks
"""

# Bloque 0: Estado Primordial
from .bloque0_estado_primordial import (
    EstadoPrimordial,
    SELLOS,
    SELLOS_ORDEN,
    NOMBRES_SELLOS,
    Mp0,
    Ep0,
    K_PRE,
    PHI_TEN_0,
    calcular_Phi_ten,
    calcular_tension,
    calcular_P_ME,
    Phi_ten_desde_P_ME,
    verificar_conservacion,
    verificar_P_ME_monotonico,
)

# Bloque 1: Pregeometría
from .bloque1_pregeometria import (
    Pregeometria,
    tasa_colapso_k,
    integral_entropica,
    integral_total,
    calcular_epsilon,
    calcular_Mp_Ep,
    derivada_k,
    K0, A1, A2, A3,
)

# Bloque 2: Cosmología
from .bloque2_cosmologia import (
    CosmologiaMCMC,
    E_LCDM,
    E_MCMC,
    Lambda_relativo,
    distancia_luminosidad,
    distancia_comovil,
    distancia_angular,
    modulo_distancia,
    edad_universo,
    tiempo_lookback,
    H0, OMEGA_M, OMEGA_LAMBDA, DELTA_LAMBDA,
)

# Validación Cuántica: Qubit Tensorial
from .qubit_tensorial import (
    QubitTensorial,
    estado_tensorial,
    calcular_amplitudes,
    concurrencia,
    concurrencia_desde_epsilon,
    entropia_entrelazamiento,
    entropia_von_neumann,
    entropia_desde_epsilon,
    matriz_densidad,
    traza_parcial_A,
    traza_parcial_B,
    verificar_consistencia_cuantica_clasica,
    trayectoria_cuantica,
    punto_maximo_entrelazamiento,
    # Estados base
    ESTADO_00, ESTADO_11,
    BELL_PHI_PLUS, BELL_PHI_MINUS,
    PAULI_I, PAULI_X, PAULI_Y, PAULI_Z,
)

# Conexión LQG: Spin Networks
from .spin_network_lqg import (
    SpinNetwork,
    Nodo,
    Enlace,
    area_enlace,
    volumen_nodo,
    operador_area_total,
    calcular_probabilidad_percolacion,
    encontrar_punto_critico,
    interpretar_transicion_geometrica,
    # Constantes de Planck
    L_PLANCK, A_PLANCK, V_PLANCK,
    GAMMA_IMMIRZI,
    P_C_2D, P_C_3D,
)

__version__ = "1.1.0"
__author__ = "Adrián Martínez Estellés"
__email__ = "adrianmartinezestelles92@gmail.com"
__license__ = "Propietaria - Ver LICENSE"

__all__ = [
    # Bloque 0: Estado Primordial
    "EstadoPrimordial",
    "SELLOS",
    "SELLOS_ORDEN",
    "NOMBRES_SELLOS",
    "Mp0",
    "Ep0",
    "K_PRE",
    "PHI_TEN_0",
    "calcular_Phi_ten",
    "calcular_tension",
    "calcular_P_ME",
    "Phi_ten_desde_P_ME",
    "verificar_conservacion",
    "verificar_P_ME_monotonico",

    # Bloque 1: Pregeometría
    "Pregeometria",
    "tasa_colapso_k",
    "integral_entropica",
    "integral_total",
    "calcular_epsilon",
    "calcular_Mp_Ep",
    "derivada_k",
    "K0", "A1", "A2", "A3",

    # Bloque 2: Cosmología
    "CosmologiaMCMC",
    "E_LCDM",
    "E_MCMC",
    "Lambda_relativo",
    "distancia_luminosidad",
    "distancia_comovil",
    "distancia_angular",
    "modulo_distancia",
    "edad_universo",
    "tiempo_lookback",
    "H0", "OMEGA_M", "OMEGA_LAMBDA", "DELTA_LAMBDA",

    # Qubit Tensorial
    "QubitTensorial",
    "estado_tensorial",
    "calcular_amplitudes",
    "concurrencia",
    "concurrencia_desde_epsilon",
    "entropia_entrelazamiento",
    "entropia_von_neumann",
    "entropia_desde_epsilon",
    "matriz_densidad",
    "traza_parcial_A",
    "traza_parcial_B",
    "verificar_consistencia_cuantica_clasica",
    "trayectoria_cuantica",
    "punto_maximo_entrelazamiento",
    "ESTADO_00", "ESTADO_11",
    "BELL_PHI_PLUS", "BELL_PHI_MINUS",
    "PAULI_I", "PAULI_X", "PAULI_Y", "PAULI_Z",

    # Spin Networks LQG
    "SpinNetwork",
    "Nodo",
    "Enlace",
    "area_enlace",
    "volumen_nodo",
    "operador_area_total",
    "calcular_probabilidad_percolacion",
    "encontrar_punto_critico",
    "interpretar_transicion_geometrica",
    "L_PLANCK", "A_PLANCK", "V_PLANCK",
    "GAMMA_IMMIRZI",
    "P_C_2D", "P_C_3D",
]
