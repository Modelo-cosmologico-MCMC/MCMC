"""
Bloque 4 - Lattice Gauge y Mass Gap Ontológico
===============================================

Implementación de teorías de gauge en retícula para verificar
el mass gap ontológico predicho por el modelo MCMC.

OBJETIVO:
    Responder: ¿Es el mass gap dinámico extraído de simulaciones Yang-Mills
    consistente con el mass gap ontológico fijado en S₄ = 1.001?

COMPONENTES:
    - mcmc_ontology_lattice: Núcleo ontológico (~500 líneas)
    - lattice/yang_mills_lattice: Retícula y Monte Carlo (~820 líneas)
    - lattice/correlators_massgap: Correladores y extracción (~650 líneas)
    - lattice/lattice_sscan: Escaneo entrópico (~480 líneas)
    - bloque4_main: Script integrador (~320 líneas)

VALORES CLAVE:
    E_min(S₄) ontológico: 5.44 GeV
    m_glueball 0⁺⁺: 1.71 GeV (QCD lattice)
    αH: 0.014
    β(S₄): 6.45

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

# Núcleo ontológico
from .mcmc_ontology_lattice import (
    # Constantes
    S0, S1, S2, S3, S4,
    K0_CALIBRADO, A1, A2, A3, EPSILON_RESIDUAL,
    BETA_0, BETA_1, B_S, LAMBDA_TENS,
    E_PLANCK, M_HIGGS, LAMBDA_QCD, ALPHA_H,
    TAU_TRANSITION, M_GLUEBALL_0PP,

    # Funciones ontológicas
    k_calibrado,
    Mp_fraccion,
    Ep_fraccion,
    P_ME,
    beta_MCMC,
    S_tensional,
    E_min_ontologico,
    E_min_QCD_scale,
    campo_adrian,
    transicion_higgs,

    # Clases
    OntologiaMCMCLattice,
    CriterioValidacion,
    ResultadoValidacion,

    # Funciones de conveniencia
    crear_ontologia_default,
    validar_ontologia,
    tabla_sellos,
)

# Submódulos de lattice
from .lattice import (
    # Yang-Mills
    GrupoGauge,
    AlgoritmoMC,
    AlgebraLie,
    ConfiguracionLattice,
    ReticulaYangMills,
    SimuladorMonteCarlo,
    crear_simulacion_MCMC,

    # Correladores
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

    # Escaneo
    ConfiguracionEscaneo,
    PuntoEscaneo,
    ResultadoEscaneo,
    EscaneoEntropico,
    escaneo_rapido,
    escaneo_completo,
)

__all__ = [
    # Constantes
    "S0", "S1", "S2", "S3", "S4",
    "K0_CALIBRADO", "A1", "A2", "A3", "EPSILON_RESIDUAL",
    "BETA_0", "BETA_1", "B_S", "LAMBDA_TENS",
    "E_PLANCK", "M_HIGGS", "LAMBDA_QCD", "ALPHA_H",
    "TAU_TRANSITION", "M_GLUEBALL_0PP",

    # Funciones ontológicas
    "k_calibrado", "Mp_fraccion", "Ep_fraccion", "P_ME",
    "beta_MCMC", "S_tensional",
    "E_min_ontologico", "E_min_QCD_scale",
    "campo_adrian", "transicion_higgs",

    # Clases ontológicas
    "OntologiaMCMCLattice", "CriterioValidacion", "ResultadoValidacion",
    "crear_ontologia_default", "validar_ontologia", "tabla_sellos",

    # Yang-Mills
    "GrupoGauge", "AlgoritmoMC", "AlgebraLie",
    "ConfiguracionLattice", "ReticulaYangMills", "SimuladorMonteCarlo",
    "crear_simulacion_MCMC",

    # Correladores
    "OperadorGlueball", "Correlador", "ResultadoMassGap",
    "calcular_correlador", "extraer_mass_gap", "medir_mass_gap",
    "masa_efectiva", "encontrar_plateau",
    "A_LATTICE_FM", "SCALE_FACTOR",

    # Escaneo
    "ConfiguracionEscaneo", "PuntoEscaneo", "ResultadoEscaneo",
    "EscaneoEntropico", "escaneo_rapido", "escaneo_completo",
]
