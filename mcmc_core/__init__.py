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
    - bloque4_lattice_gauge: Mass gap ontológico, Yang-Mills en retícula

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

# Bloque 4: Lattice Gauge y Mass Gap Ontológico
from .bloque4_lattice_gauge import (
    # Clases ontológicas
    OntologiaMCMCLattice,
    CriterioValidacion,
    ResultadoValidacion,
    crear_ontologia_default,
    validar_ontologia,
    tabla_sellos,

    # Funciones ontológicas
    beta_MCMC,
    S_tensional,
    E_min_ontologico,
    E_min_QCD_scale,
    campo_adrian,
    transicion_higgs,

    # Constantes ontológicas
    BETA_0, BETA_1, B_S, LAMBDA_TENS,
    E_PLANCK, M_HIGGS, TAU_TRANSITION, M_GLUEBALL_0PP,
    ALPHA_H,
)

# Renombrar para evitar conflicto con LAMBDA_QCD del legacy module
from .bloque4_lattice_gauge import LAMBDA_QCD as LAMBDA_QCD_GAUGE

# Yang-Mills Lattice
from .bloque4_lattice_gauge.lattice import (
    # Grupos y algoritmos
    GrupoGauge,
    AlgoritmoMC,
    AlgebraLie,
    ConfiguracionLattice,
    ReticulaYangMills,
    SimuladorMonteCarlo,
    crear_simulacion_MCMC,

    # Correladores y mass gap
    OperadorGlueball,
    Correlador,
    ResultadoMassGap,
    calcular_correlador,
    extraer_mass_gap,
    medir_mass_gap,
    masa_efectiva,
    encontrar_plateau,
    A_LATTICE_FM,
    SCALE_FACTOR,

    # Escaneo entrópico
    ConfiguracionEscaneo,
    PuntoEscaneo,
    ResultadoEscaneo,
    EscaneoEntropico,
    escaneo_rapido,
    escaneo_completo,
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

# Datos Observacionales
from .datos_observacionales import (
    # Constantes
    C_LIGHT, H0_FIDUCIAL,
    # Planck 2018
    DatosPlanck2018, PLANCK_2018,
    # BAO
    PuntoBAO,
    BAO_6DFGS, BAO_MGS, BAO_BOSS_DR12, BAO_EBOSS_DR16, BAO_DESI_2024,
    BAO_ALL,
    # SPARC
    GalaxiaSPARCCompleta, SPARC_CATALOG,
    # GAIA
    DatosGAIA, GAIA_DR3,
    # Supernovas
    PuntoSN, PANTHEON_PLUS_SUBSET,
    # H0
    MedicionH0, H0_MEDICIONES,
    # Funciones de utilidad
    calcular_r_d_fiducial,
    calcular_D_V,
    calcular_chi2_BAO,
    calcular_chi2_SN,
    calcular_D_L_LCDM,
    tension_H0,
    resumen_datos,
)

# Ontología ECV y MCV
from .ontologia_ecv_mcv import (
    # Constantes ECV
    EPSILON_ECV, Z_TRANS, DELTA_Z, S_BB, S_0, P_ENTROPIC,
    # Constantes MCV
    RHO_STAR, R_STAR, S_STAR, ALPHA_RHO, ALPHA_R, NORM_FACTOR_MCV,
    # Fricción
    ETA_FRICTION, GAMMA_FRICTION, RHO_CRIT_FRICTION,
    # Funciones ECV
    S_of_z, F_transition, rho_ECV, Omega_ECV, Lambda_rel,
    E_MCMC_ECV, E_MCMC_Lambda_rel, H_MCMC_ECV,
    distancia_comovil_ECV, distancia_luminosidad_ECV,
    modulo_distancia_ECV, distancia_volumen_ECV,
    # Funciones MCV
    S_local, rho_0_MCV, r_core_MCV, r_core_from_mass,
    perfil_MCV_Burkert, perfil_MCV_isotermico,
    masa_encerrada_MCV_Burkert, masa_encerrada_MCV_isotermico,
    perfil_Zhao_MCV,
    # Fricción entrópica
    ParametrosFriccion, FriccionEntropicaMCV,
    # Velocidad circular
    velocidad_circular_MCV, velocidad_circular_MCV_calibrado, velocidad_NFW_standard,
    # LCDM para comparación
    E_LCDM_standard, H_LCDM_standard,
    distancia_comovil_LCDM, distancia_luminosidad_LCDM,
    modulo_distancia_LCDM, distancia_volumen_LCDM,
    # Verificación
    verificar_ECV, verificar_MCV, verificar_friccion,
)

# Validación Estadística Robusta
from .validacion_estadistica import (
    # Constantes
    G_KPC_KMS2_MSUN, H0_FIDUCIAL as H0_FIDUCIAL_VALID,
    # SNe Ia con marginalización
    mu_theory_noM, mu_MCMC, mu_LCDM,
    sigma_mu_peculiar, add_peculiar_velocity_to_cov,
    chi2_sne_marginalized_M, chi2_sne_simple,
    validar_sne_ia_robusto, ResultadoSNeIa,
    # SPARC con perfiles corregidos
    v_circ_from_Menc, menc_burkert, v_burkert_dm,
    menc_nfw, v_nfw_dm,
    v_model_total, chi2_rotation_curve,
    ajustar_curva_rotacion_sparc, ResultadoSPARC,
    # Datos de prueba
    generar_datos_sne_ejemplo, generar_galaxia_sparc_ejemplo,
)

__version__ = "2.6.0"  # Bump for robust validation
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

    # Bloque 4: Lattice Gauge y Mass Gap
    "OntologiaMCMCLattice", "CriterioValidacion", "ResultadoValidacion",
    "crear_ontologia_default", "validar_ontologia", "tabla_sellos",
    "beta_MCMC", "S_tensional", "E_min_ontologico", "E_min_QCD_scale",
    "campo_adrian", "transicion_higgs",
    "BETA_0", "BETA_1", "B_S", "LAMBDA_TENS",
    "E_PLANCK", "M_HIGGS", "LAMBDA_QCD_GAUGE", "ALPHA_H",
    "TAU_TRANSITION", "M_GLUEBALL_0PP",
    "GrupoGauge", "AlgoritmoMC", "AlgebraLie",
    "ConfiguracionLattice", "ReticulaYangMills", "SimuladorMonteCarlo",
    "crear_simulacion_MCMC",
    "OperadorGlueball", "Correlador", "ResultadoMassGap",
    "calcular_correlador", "extraer_mass_gap", "medir_mass_gap",
    "masa_efectiva", "encontrar_plateau",
    "A_LATTICE_FM", "SCALE_FACTOR",
    "ConfiguracionEscaneo", "PuntoEscaneo", "ResultadoEscaneo",
    "EscaneoEntropico", "escaneo_rapido", "escaneo_completo",

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

    # Datos Observacionales
    "C_LIGHT", "H0_FIDUCIAL",
    "DatosPlanck2018", "PLANCK_2018",
    "PuntoBAO",
    "BAO_6DFGS", "BAO_MGS", "BAO_BOSS_DR12", "BAO_EBOSS_DR16", "BAO_DESI_2024",
    "BAO_ALL",
    "GalaxiaSPARCCompleta", "SPARC_CATALOG",
    "DatosGAIA", "GAIA_DR3",
    "PuntoSN", "PANTHEON_PLUS_SUBSET",
    "MedicionH0", "H0_MEDICIONES",
    "calcular_r_d_fiducial", "calcular_D_V",
    "calcular_chi2_BAO", "calcular_chi2_SN", "calcular_D_L_LCDM",
    "tension_H0", "resumen_datos",

    # Ontología ECV y MCV
    "EPSILON_ECV", "Z_TRANS", "DELTA_Z", "S_BB", "S_0", "P_ENTROPIC",
    "RHO_STAR", "R_STAR", "S_STAR", "ALPHA_RHO", "ALPHA_R",
    "ETA_FRICTION", "GAMMA_FRICTION", "RHO_CRIT_FRICTION",
    "S_of_z", "F_transition", "rho_ECV", "Omega_ECV", "Lambda_rel",
    "E_MCMC_ECV", "E_MCMC_Lambda_rel", "H_MCMC_ECV",
    "distancia_comovil_ECV", "distancia_luminosidad_ECV",
    "modulo_distancia_ECV", "distancia_volumen_ECV",
    "S_local", "rho_0_MCV", "r_core_MCV", "r_core_from_mass",
    "perfil_MCV_Burkert", "perfil_MCV_isotermico",
    "masa_encerrada_MCV_Burkert", "masa_encerrada_MCV_isotermico",
    "perfil_Zhao_MCV",
    "ParametrosFriccion", "FriccionEntropicaMCV",
    "velocidad_circular_MCV", "velocidad_NFW_standard",
    "E_LCDM_standard", "H_LCDM_standard",
    "distancia_comovil_LCDM", "distancia_luminosidad_LCDM",
    "modulo_distancia_LCDM", "distancia_volumen_LCDM",
    "verificar_ECV", "verificar_MCV", "verificar_friccion",

    # SPARC Zhao MCMC
    "ParametrosZhaoMCMC", "PARAMS_ZHAO", "PerfilZhaoMCMC", "PerfilNFW",
    "AjustadorSPARC", "test_SPARC_Zhao_MCMC", "verificar_SPARC_Zhao",

    # GAIA Zhao MCMC
    "ParametrosMWMCMC", "PARAMS_MW", "AjustadorGAIA", "test_GAIA_Zhao_MCMC",

    # CLASS-MCMC
    "ParametrosCLASS", "PARAMS_CLASS",
    "E_MCMC_full", "H_MCMC_full",
    "calcular_D_MCMC", "calcular_f_MCMC",
    "calcular_sigma8_MCMC", "calcular_S8_MCMC",
    "P_k_MCMC", "calcular_Pk_array",
    "C_l_TT_approx", "calcular_Cl_array",
    "theta_star_MCMC", "l_acoustic_MCMC",
    "comparar_con_LCDM", "test_CLASS_MCMC",

    # N-body Cronos
    "ParametrosCronosNBody", "PARAMS_CRONOS",
    "lapse_function", "friccion_entropica", "radio_core_cronos",
    "perfil_Cronos", "IntegradorCronos",
    "analizar_halo_cronos", "test_NBody_Cronos",

    # Lensing MCV
    "ParametrosLensing", "PARAMS_LENSING",
    "Sigma_crit", "Sigma_NFW", "Sigma_Zhao",
    "kappa_NFW", "kappa_Zhao", "gamma_tangencial",
    "calcular_S8_lensing", "test_Lensing_MCV",

    # DESI Y3
    "PuntoDESI", "DESI_Y3_DATA",
    "calcular_chi2_DESI", "calcular_chi2_LCDM_DESI",
    "ajustar_epsilon_z_trans", "comparar_DESI_detallado",
    "analizar_tensiones_DESI", "test_DESI_Y3",
]

# SPARC Zhao MCMC (late import to avoid circular)
from .sparc_zhao import (
    ParametrosZhaoMCMC, PARAMS_ZHAO,
    PerfilZhaoMCMC, PerfilNFW,
    AjustadorSPARC,
    test_SPARC_Zhao_MCMC, verificar_SPARC_Zhao,
    # GAIA (Vía Láctea)
    ParametrosMWMCMC, PARAMS_MW,
    AjustadorGAIA,
    test_GAIA_Zhao_MCMC,
)

# CLASS-MCMC
from .class_mcmc import (
    ParametrosCLASS, PARAMS_CLASS,
    E_MCMC_full, H_MCMC_full,
    calcular_D_MCMC, calcular_f_MCMC,
    calcular_sigma8_MCMC, calcular_S8_MCMC,
    P_k_MCMC, calcular_Pk_array,
    C_l_TT_approx, calcular_Cl_array,
    theta_star_MCMC, l_acoustic_MCMC,
    comparar_con_LCDM, test_CLASS_MCMC,
)

# N-body Cronos
from .nbody_cronos import (
    ParametrosCronosNBody, PARAMS_CRONOS,
    lapse_function, friccion_entropica, radio_core_cronos,
    perfil_Cronos, IntegradorCronos,
    analizar_halo_cronos, test_NBody_Cronos,
)

# Lensing MCV
from .lensing_mcv import (
    ParametrosLensing, PARAMS_LENSING,
    Sigma_crit, Sigma_NFW, Sigma_Zhao,
    kappa_NFW, kappa_Zhao, gamma_tangencial,
    calcular_S8_lensing, test_Lensing_MCV,
)

# DESI Y3
from .desi_y3 import (
    PuntoDESI, DESI_Y3_DATA,
    calcular_chi2_DESI, calcular_chi2_LCDM_DESI,
    ajustar_epsilon_z_trans, comparar_DESI_detallado,
    analizar_tensiones_DESI, test_DESI_Y3,
)
