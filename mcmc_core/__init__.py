"""
MCMC Core - Modelo Cosmológico de Múltiples Colapsos
=====================================================

Implementación del modelo MCMC que describe la evolución del universo
desde un estado primordial de máxima tensión hasta el estado actual,
basado en la DUALIDAD TENSIONAL MASA-ESPACIO.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

MÓDULOS PRINCIPALES (5 BLOQUES):
    - bloque0_estado_primordial: Estado inicial Mp0, Ep0, Campo de Adrián
    - bloque1_pregeometria: Tasa de colapso k(S), integral entrópica
    - bloque2_cosmologia: Ecuaciones de Friedmann modificadas MCMC
    - bloque3_nbody (simulations): Fricción entrópica, perfiles Burkert
    - bloque4_ym_lattice (lattice): Mass gap, Yang-Mills

MÓDULOS DE VALIDACIÓN CUÁNTICA:
    - qubit_tensorial: Estado cuántico |Φ(S)⟩ = α|00⟩ + β|11⟩
    - spin_network_lqg: Conexión con Loop Quantum Gravity
    - fase_pregeometrica: Evolución calibrada S ∈ [0, 1.001]
    - ley_cronos: Dilatación temporal en regiones densas
    - circuito_cuantico: Validación experimental (Qiskit)

ONTOLOGÍA MCMC:
    - Sellos entrópicos: S0=0.000, S1=0.010, S2=0.100, S3=1.000, S4=1.001
    - Campo de Adrián: Φ_ten(S) = Mp(S)/Ep(S)
    - Polarización masa-espacio: P_ME = (Mp - Ep)/(Mp + Ep)
    - Trayectoria: P_ME evoluciona de +1 (masa domina) a -1 (espacio domina)
    - ε = 0.0112: Fracción residual cristalizada en S₄ (Big Bang)

CONEXIONES TEÓRICAS:
    - Cosmología: Reducción de tensiones H0 (56%) y S8 (62%)
    - Física de partículas: Mass gap ontológico en Yang-Mills
    - Gravedad cuántica: Emergencia geométrica vía spin networks
    - Hardware cuántico: Validación en IonQ, IBM
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

# Alias para compatibilidad
FasePregeometrica = Pregeometria

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

# Bloque 3: N-Body y Formación de Estructuras
from .bloque3_nbody import (
    # Funciones ontológicas
    k_S,
    Mp_frac,
    Ep_frac,
    P_ME as P_ME_nbody,
    S_to_z,
    z_to_S,
    Lambda_rel,
    H_MCMC,
    H_MCMC_kpc_gyr,
    # Fricción entrópica
    ParametrosCronos,
    FriccionEntropica,
    # Perfiles de densidad
    ParametrosHalo,
    PerfilDensidad,
    perfil_NFW,
    perfil_Burkert,
    perfil_Zhao_MCMC,
    radio_core_MCMC,
    densidad_central_Burkert,
    # Integrador Cronos
    ConfiguracionCronos,
    ParticulaCronos,
    ResultadoSimulacion,
    IntegradorCronos,
    # Análisis de halos
    Halo,
    AnalizadorHalos,
    # Comparación SPARC
    GalaxiaSPARC,
    ComparadorSPARC,
    cargar_datos_SPARC_ejemplo,
    # Función de masa
    funcion_masa_MCMC,
    ratio_MCMC_LCDM,
    # Validación ontológica
    ValidadorOntologico,
    # Constantes
    RHO_CRONOS as RHO_CRONOS_NBODY,
    ALPHA_LAPSE,
    BETA_ETA,
    GAMMA_FRICCION,
    PHI0_ADRIAN,
    LAMBDA_PHI,
    G_NEWTON,
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
    L_PLANCK, A_PLANCK, V_PLANCK,
    GAMMA_IMMIRZI,
    P_C_2D, P_C_3D,
)

# Fase Pre-Geométrica Calibrada
from .fase_pregeometrica import (
    FasePregeometrica as FasePregeometricaCalibrada,
    k_calibrado,
    Mp_fraccion,
    Ep_fraccion,
    P_ME_calibrado,
    generar_secuencia_colapsos,
    generar_tabla_sellos,
    verificar_calibracion,
    K0_CALIBRADO, A1 as A1_CAL, A2 as A2_CAL, A3 as A3_CAL,
    S0, S1, S2, S3, S4,
    EPSILON_RESIDUAL,
)

# Ley de Cronos: Dilatación Temporal
from .ley_cronos import (
    LeyCronos,
    dilatacion_temporal,
    tiempo_propio,
    radio_core,
    friccion_entropica_cronos,
    S_desde_t,
    t_desde_S,
    tabla_correspondencias,
    RHO_CRONOS,
    R_STAR, M_STAR, ALPHA_R, BETA_R,
)

# Circuito Cuántico: Validación Experimental
from .circuito_cuantico import (
    CircuitoTensorial,
    generar_codigo_qiskit,
    preparar_estado,
    medir_P_ME,
    medir_ZZ,
    medir_XX,
    medir_YY,
    testigo_bell,
    violacion_CHSH,
    simular_medicion,
    evolucionar_circuito,
    EPSILON as EPSILON_CALIBRADO,
    THETA_RY,
    P_ME_ESPERADO,
    ZZ_ESPERADO,
)

__version__ = "2.1.0"
__author__ = "Adrián Martínez Estellés"
__email__ = "adrianmartinezestelles92@gmail.com"
__license__ = "Propietaria - Ver LICENSE"

__all__ = [
    # Bloque 0: Estado Primordial
    "EstadoPrimordial",
    "SELLOS", "SELLOS_ORDEN", "NOMBRES_SELLOS",
    "Mp0", "Ep0", "K_PRE", "PHI_TEN_0",
    "calcular_Phi_ten", "calcular_tension", "calcular_P_ME",
    "Phi_ten_desde_P_ME",
    "verificar_conservacion", "verificar_P_ME_monotonico",

    # Bloque 1: Pregeometría
    "Pregeometria", "FasePregeometrica",
    "tasa_colapso_k", "integral_entropica", "integral_total",
    "calcular_epsilon", "calcular_Mp_Ep", "derivada_k",
    "K0", "A1", "A2", "A3",

    # Bloque 2: Cosmología
    "CosmologiaMCMC",
    "E_LCDM", "E_MCMC", "Lambda_relativo",
    "distancia_luminosidad", "distancia_comovil", "distancia_angular",
    "modulo_distancia", "edad_universo", "tiempo_lookback",
    "H0", "OMEGA_M", "OMEGA_LAMBDA", "DELTA_LAMBDA",

    # Bloque 3: N-Body y Estructuras
    "k_S", "Mp_frac", "Ep_frac", "P_ME_nbody",
    "S_to_z", "z_to_S", "Lambda_rel", "H_MCMC", "H_MCMC_kpc_gyr",
    "ParametrosCronos", "FriccionEntropica",
    "ParametrosHalo", "PerfilDensidad",
    "perfil_NFW", "perfil_Burkert", "perfil_Zhao_MCMC",
    "radio_core_MCMC", "densidad_central_Burkert",
    "ConfiguracionCronos", "ParticulaCronos", "ResultadoSimulacion",
    "IntegradorCronos",
    "Halo", "AnalizadorHalos",
    "GalaxiaSPARC", "ComparadorSPARC", "cargar_datos_SPARC_ejemplo",
    "funcion_masa_MCMC", "ratio_MCMC_LCDM",
    "ValidadorOntologico",
    "RHO_CRONOS_NBODY", "ALPHA_LAPSE", "BETA_ETA", "GAMMA_FRICCION",
    "PHI0_ADRIAN", "LAMBDA_PHI", "G_NEWTON",

    # Qubit Tensorial
    "QubitTensorial",
    "estado_tensorial", "calcular_amplitudes",
    "concurrencia", "concurrencia_desde_epsilon",
    "entropia_entrelazamiento", "entropia_von_neumann", "entropia_desde_epsilon",
    "matriz_densidad", "traza_parcial_A", "traza_parcial_B",
    "verificar_consistencia_cuantica_clasica",
    "trayectoria_cuantica", "punto_maximo_entrelazamiento",
    "ESTADO_00", "ESTADO_11", "BELL_PHI_PLUS", "BELL_PHI_MINUS",
    "PAULI_I", "PAULI_X", "PAULI_Y", "PAULI_Z",

    # Spin Networks LQG
    "SpinNetwork", "Nodo", "Enlace",
    "area_enlace", "volumen_nodo", "operador_area_total",
    "calcular_probabilidad_percolacion", "encontrar_punto_critico",
    "interpretar_transicion_geometrica",
    "L_PLANCK", "A_PLANCK", "V_PLANCK", "GAMMA_IMMIRZI",
    "P_C_2D", "P_C_3D",

    # Fase Pre-Geométrica Calibrada
    "FasePregeometricaCalibrada",
    "k_calibrado", "Mp_fraccion", "Ep_fraccion", "P_ME_calibrado",
    "generar_secuencia_colapsos", "generar_tabla_sellos",
    "verificar_calibracion",
    "K0_CALIBRADO", "S0", "S1", "S2", "S3", "S4",
    "EPSILON_RESIDUAL",

    # Ley de Cronos
    "LeyCronos",
    "dilatacion_temporal", "tiempo_propio", "radio_core",
    "friccion_entropica_cronos",
    "S_desde_t", "t_desde_S", "tabla_correspondencias",
    "RHO_CRONOS", "R_STAR", "M_STAR", "ALPHA_R", "BETA_R",

    # Circuito Cuántico
    "CircuitoTensorial",
    "generar_codigo_qiskit", "preparar_estado",
    "medir_P_ME", "medir_ZZ", "medir_XX", "medir_YY",
    "testigo_bell", "violacion_CHSH",
    "simular_medicion", "evolucionar_circuito",
    "EPSILON_CALIBRADO", "THETA_RY", "P_ME_ESPERADO", "ZZ_ESPERADO",
]
