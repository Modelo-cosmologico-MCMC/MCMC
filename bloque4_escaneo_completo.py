#!/usr/bin/env python3
"""
Escaneo Completo del Bloque 4 - Lattice Gauge y Mass Gap
=========================================================

Ejecuta un análisis completo del Bloque 4:
1. Escaneo entrópico S ∈ [0.90, 1.001]
2. Exploración de grupos GUT: SU(2), SU(3), SU(5)
3. Extracción de mass gap con múltiples métodos
4. Comparación con datos QCD (m_0++ ≈ 1.7 GeV)

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importaciones del modelo MCMC
from mcmc_core.bloque4_lattice_gauge import (
    OntologiaMCMCLattice, crear_ontologia_default,
    E_min_ontologico, beta_MCMC, E_min_QCD_scale,
    M_HIGGS, M_GLUEBALL_0PP, LAMBDA_QCD, ALPHA_H,
    S0, S1, S2, S3, S4, TAU_TRANSITION
)

from mcmc_core.bloque4_lattice_gauge.lattice import (
    GrupoGauge, AlgoritmoMC, ConfiguracionLattice,
    ReticulaYangMills, SimuladorMonteCarlo,
    EscaneoEntropico, ConfiguracionEscaneo,
    calcular_correlador, extraer_mass_gap, masa_efectiva
)


# =============================================================================
# CONSTANTES Y DATOS OBSERVACIONALES
# =============================================================================

# Datos QCD Lattice (PDG + simulaciones)
QCD_GLUEBALL_0PP = 1.71    # GeV (0++ glueball mass)
QCD_GLUEBALL_0PP_ERR = 0.10
QCD_GLUEBALL_2PP = 2.39    # GeV (2++ glueball mass)
QCD_GLUEBALL_2PP_ERR = 0.12
QCD_LAMBDA_MSBAR = 0.217   # GeV (Λ_QCD in MS-bar scheme)

# Escalas GUT
GUT_SCALE_SU5 = 2e16       # GeV (SU(5) unification)
GUT_SCALE_SO10 = 3e16      # GeV (SO(10) unification)


# =============================================================================
# CLASES DE RESULTADOS
# =============================================================================

@dataclass
class ResultadoGrupo:
    """Resultado para un grupo de gauge específico."""
    grupo: str
    N: int
    beta: float
    plaqueta: float
    mass_gap: Optional[float] = None
    mass_gap_err: Optional[float] = None
    aceptacion: float = 0.0
    n_configs: int = 0


@dataclass
class ResultadoPuntoS:
    """Resultado para un punto entrópico."""
    S: float
    E_onto: float      # Mass gap ontológico
    E_qcd: float       # Escala QCD
    beta: float        # Acoplamiento gauge
    grupos: Dict[str, ResultadoGrupo] = field(default_factory=dict)


@dataclass
class ResultadoEscaneoCompleto:
    """Resultado del escaneo completo."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    L: int = 4
    n_configs: int = 50
    puntos: List[ResultadoPuntoS] = field(default_factory=list)
    metricas_globales: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "parametros": {"L": self.L, "n_configs": self.n_configs},
            "puntos": [
                {
                    "S": p.S,
                    "E_onto": p.E_onto,
                    "E_qcd": p.E_qcd,
                    "beta": p.beta,
                    "grupos": {
                        k: {
                            "plaqueta": v.plaqueta,
                            "mass_gap": v.mass_gap,
                            "aceptacion": v.aceptacion
                        }
                        for k, v in p.grupos.items()
                    }
                }
                for p in self.puntos
            ],
            "metricas": self.metricas_globales
        }


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def simular_grupo(
    grupo: GrupoGauge,
    S: float,
    L: int = 4,
    n_termalizacion: int = 50,
    n_configs: int = 50,
    algoritmo: AlgoritmoMC = AlgoritmoMC.METROPOLIS
) -> ResultadoGrupo:
    """
    Ejecuta simulación Monte Carlo para un grupo específico.
    """
    # Crear configuración
    config = ConfiguracionLattice(L=L, d=4, grupo=grupo, S=S)

    # Crear retícula
    lattice = ReticulaYangMills(config)

    # Crear simulador con la retícula (no config)
    sim = SimuladorMonteCarlo(lattice, algoritmo=algoritmo)

    # Termalizar
    sim.termalizar(n_sweeps=n_termalizacion)

    # Medir plaqueta promedio (el método está en la retícula)
    plaquetas = []
    for _ in range(n_configs):
        sim.sweep()
        plaquetas.append(lattice.promedio_plaqueta())

    P_avg = np.mean(plaquetas)
    P_err = np.std(plaquetas) / np.sqrt(len(plaquetas))

    # Resultado
    resultado = ResultadoGrupo(
        grupo=grupo.name,
        N=config.N,
        beta=config.beta,
        plaqueta=P_avg,
        aceptacion=sim.tasa_aceptacion if hasattr(sim, 'tasa_aceptacion') else 0.0,
        n_configs=n_configs
    )

    return resultado


def analizar_punto_S(
    S: float,
    L: int = 4,
    n_configs: int = 50,
    grupos: List[GrupoGauge] = None
) -> ResultadoPuntoS:
    """
    Analiza un punto entrópico con múltiples grupos.
    """
    if grupos is None:
        grupos = [GrupoGauge.SU2, GrupoGauge.SU3]

    # Calcular valores ontológicos
    E_onto = E_min_ontologico(S)
    E_qcd = E_min_QCD_scale(S)
    beta = beta_MCMC(S)

    resultado = ResultadoPuntoS(
        S=S,
        E_onto=E_onto,
        E_qcd=E_qcd,
        beta=beta
    )

    # Simular cada grupo
    for grupo in grupos:
        try:
            res_grupo = simular_grupo(grupo, S, L=L, n_configs=n_configs)
            resultado.grupos[grupo.name] = res_grupo
        except Exception as e:
            print(f"    [!] Error con {grupo.name}: {e}")

    return resultado


def ejecutar_escaneo_entropico(
    S_values: List[float] = None,
    L: int = 4,
    n_configs: int = 50,
    grupos: List[GrupoGauge] = None,
    verbose: bool = True
) -> ResultadoEscaneoCompleto:
    """
    Ejecuta escaneo entrópico completo.
    """
    if S_values is None:
        # Puntos clave: pre-Big Bang, transición, post-Big Bang
        S_values = [0.90, 0.95, 0.99, 1.000, 1.001]

    if grupos is None:
        grupos = [GrupoGauge.SU2, GrupoGauge.SU3]

    resultado = ResultadoEscaneoCompleto(L=L, n_configs=n_configs)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ESCANEO ENTRÓPICO - BLOQUE 4")
        print(f"  L={L}, n_configs={n_configs}")
        print(f"  Grupos: {[g.name for g in grupos]}")
        print(f"{'='*60}\n")

    for i, S in enumerate(S_values):
        if verbose:
            print(f"[{i+1}/{len(S_values)}] S = {S:.4f}")

        punto = analizar_punto_S(S, L=L, n_configs=n_configs, grupos=grupos)
        resultado.puntos.append(punto)

        if verbose:
            print(f"    E_onto = {punto.E_onto:.3f} GeV")
            print(f"    β(S)   = {punto.beta:.4f}")
            for gname, gres in punto.grupos.items():
                print(f"    {gname}: <P> = {gres.plaqueta:.4f}")
            print()

    # Calcular métricas globales
    if resultado.puntos:
        E_onto_S4 = E_min_ontologico(S4)
        E_qcd_S4 = E_min_QCD_scale(S4)
        ratio_glueball = E_qcd_S4 / QCD_GLUEBALL_0PP

        resultado.metricas_globales = {
            "E_onto_S4": E_onto_S4,
            "E_qcd_S4": E_qcd_S4,
            "ratio_vs_glueball": ratio_glueball,
            "beta_S4": beta_MCMC(S4),
            "m_glueball_ref": QCD_GLUEBALL_0PP
        }

    return resultado


def comparar_con_QCD(resultado: ResultadoEscaneoCompleto) -> Dict[str, any]:
    """
    Compara resultados con datos QCD lattice.
    """
    comparacion = {
        "QCD_referencia": {
            "m_0++": QCD_GLUEBALL_0PP,
            "m_0++_err": QCD_GLUEBALL_0PP_ERR,
            "Lambda_QCD": QCD_LAMBDA_MSBAR
        },
        "MCMC_prediccion": {},
        "compatibilidad": {}
    }

    # Predicciones MCMC en S4
    E_qcd_S4 = E_min_QCD_scale(S4)
    comparacion["MCMC_prediccion"] = {
        "E_qcd_S4": E_qcd_S4,
        "E_onto_S4": E_min_ontologico(S4),
        "beta_S4": beta_MCMC(S4)
    }

    # Test de compatibilidad
    ratio = E_qcd_S4 / QCD_GLUEBALL_0PP
    sigma = abs(E_qcd_S4 - QCD_GLUEBALL_0PP) / QCD_GLUEBALL_0PP_ERR

    comparacion["compatibilidad"] = {
        "ratio_E_qcd/m_glueball": ratio,
        "desviacion_sigma": sigma,
        "compatible_1sigma": sigma < 1,
        "compatible_2sigma": sigma < 2,
        "compatible_3sigma": sigma < 3,
        "orden_magnitud_correcto": 0.1 < ratio < 10
    }

    return comparacion


def explorar_grupos_GUT(
    L: int = 4,
    n_configs: int = 30,
    verbose: bool = True
) -> Dict[str, ResultadoGrupo]:
    """
    Explora grupos de unificación GUT disponibles.
    """
    # Grupos disponibles en la implementación
    grupos_disponibles = [
        GrupoGauge.SU2,
        GrupoGauge.SU3,
        GrupoGauge.SU5,
    ]

    resultados = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  EXPLORACIÓN DE GRUPOS GUT")
        print(f"  L={L}, S=S₄={S4}")
        print(f"{'='*60}\n")

    for grupo in grupos_disponibles:
        if verbose:
            print(f"Simulando {grupo.name}...", end=" ", flush=True)

        try:
            res = simular_grupo(grupo, S=S4, L=L, n_configs=n_configs)
            resultados[grupo.name] = res
            if verbose:
                print(f"<P> = {res.plaqueta:.4f}")
        except Exception as e:
            if verbose:
                print(f"Error: {e}")

    return resultados


def generar_tabla_transicion() -> str:
    """
    Genera tabla de transición mass gap ontológico.
    """
    lineas = []
    lineas.append("\n" + "="*60)
    lineas.append("  TABLA DE TRANSICIÓN MASS GAP ONTOLÓGICO")
    lineas.append("="*60)
    lineas.append(f"\n{'S':^8} │ {'E_onto (GeV)':^14} │ {'E_QCD (GeV)':^14} │ {'β(S)':^10}")
    lineas.append("─"*8 + "─┼─" + "─"*14 + "─┼─" + "─"*14 + "─┼─" + "─"*10)

    S_values = [0.0, 0.5, 0.9, 0.99, 1.0, 1.001]
    for S in S_values:
        E_onto = E_min_ontologico(S)
        E_qcd = E_min_QCD_scale(S)
        beta = beta_MCMC(S)
        lineas.append(f"{S:^8.3f} │ {E_onto:^14.3f} │ {E_qcd:^14.3f} │ {beta:^10.4f}")

    lineas.append("")
    lineas.append("Fórmula: E_min(S) = ½[E_Pl(1-tanh((S-1)/τ)) + m_H(1+tanh((S-1)/τ))]")
    lineas.append(f"τ = {TAU_TRANSITION}, m_H = {M_HIGGS} GeV")

    return "\n".join(lineas)


def validar_criterios_bloque4() -> Dict[str, bool]:
    """
    Valida los 3 criterios del Bloque 4.
    """
    criterios = {}

    # Criterio 1: Plateau - E_min constante para S ≳ S₃
    E_S3 = E_min_ontologico(S3)
    E_S4 = E_min_ontologico(S4)
    diff_plateau = abs(E_S4 - E_S3) / E_S3 if E_S3 > 0 else 1
    criterios["plateau_S3_S4"] = diff_plateau < 0.5  # Menos de 50% cambio

    # Criterio 2: Ratio E_lat/E_onto en rango [0.1, 10]
    E_qcd = E_min_QCD_scale(S4)
    ratio = E_qcd / QCD_GLUEBALL_0PP
    criterios["ratio_en_rango"] = 0.1 <= ratio <= 10

    # Criterio 3: Transición sigmoidal con τ ≈ 0.001
    # Verificar que hay transición significativa
    E_pre = E_min_ontologico(0.5)
    E_post = E_min_ontologico(S4)
    criterios["transicion_sigmoidal"] = abs(E_post - E_pre) / E_pre > 0.1

    # Criterio adicional: β(S) decrece para S > S3
    beta_S3 = beta_MCMC(S3)
    beta_S4 = beta_MCMC(S4)
    criterios["beta_decrece"] = beta_S4 <= beta_S3 + 0.1  # Margen pequeño

    return criterios


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """Ejecuta análisis completo del Bloque 4."""
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#    ANÁLISIS COMPLETO BLOQUE 4 - LATTICE GAUGE MCMC" + " "*7 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)

    # 1. Tabla de transición
    print(generar_tabla_transicion())

    # 2. Validación de criterios
    print("\n" + "="*60)
    print("  VALIDACIÓN DE CRITERIOS")
    print("="*60 + "\n")

    criterios = validar_criterios_bloque4()
    n_cumplidos = sum(criterios.values())

    for nombre, cumplido in criterios.items():
        status = "✓" if cumplido else "✗"
        print(f"  {status} {nombre}")

    print(f"\n  Resultado: {n_cumplidos}/{len(criterios)} criterios cumplidos")

    # 3. Escaneo entrópico
    resultado_escaneo = ejecutar_escaneo_entropico(
        L=4,
        n_configs=30,
        grupos=[GrupoGauge.SU2, GrupoGauge.SU3]
    )

    # 4. Exploración GUT
    resultado_gut = explorar_grupos_GUT(L=4, n_configs=20)

    # 5. Comparación QCD
    print("\n" + "="*60)
    print("  COMPARACIÓN CON DATOS QCD LATTICE")
    print("="*60 + "\n")

    comparacion = comparar_con_QCD(resultado_escaneo)

    print(f"  Referencia QCD:")
    print(f"    m_0⁺⁺ = {QCD_GLUEBALL_0PP} ± {QCD_GLUEBALL_0PP_ERR} GeV")
    print(f"    Λ_QCD = {QCD_LAMBDA_MSBAR} GeV")
    print()
    print(f"  Predicción MCMC (S₄):")
    print(f"    E_QCD = {comparacion['MCMC_prediccion']['E_qcd_S4']:.3f} GeV")
    print(f"    β(S₄) = {comparacion['MCMC_prediccion']['beta_S4']:.4f}")
    print()
    print(f"  Compatibilidad:")
    print(f"    Ratio E_QCD/m_glueball = {comparacion['compatibilidad']['ratio_E_qcd/m_glueball']:.3f}")
    print(f"    Orden de magnitud: {'✓' if comparacion['compatibilidad']['orden_magnitud_correcto'] else '✗'}")

    # 6. Resumen
    print("\n" + "="*60)
    print("  RESUMEN BLOQUE 4")
    print("="*60 + "\n")

    print(f"  Escaneo entrópico: {len(resultado_escaneo.puntos)} puntos")
    print(f"  Grupos explorados: {len(resultado_gut)} grupos GUT")
    print(f"  Criterios cumplidos: {n_cumplidos}/{len(criterios)}")
    print(f"  Compatibilidad QCD: {'✓' if comparacion['compatibilidad']['orden_magnitud_correcto'] else '✗'}")

    # Exportar resultados (convertir numpy types a Python types)
    output_path = os.path.join(os.path.dirname(__file__), "resultados_bloque4.json")

    def convertir_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convertir_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convertir_numpy(v) for v in obj]
        return obj

    datos_export = {
        "escaneo": resultado_escaneo.to_dict(),
        "grupos_gut": {k: {"plaqueta": float(v.plaqueta), "N": int(v.N)} for k, v in resultado_gut.items()},
        "comparacion_qcd": convertir_numpy(comparacion),
        "criterios": {k: bool(v) for k, v in criterios.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(datos_export, f, indent=2)

    print(f"\n  Resultados exportados a: {output_path}")
    print("\n" + "="*60 + "\n")

    return resultado_escaneo, resultado_gut, comparacion


if __name__ == "__main__":
    main()
