#!/usr/bin/env python3
"""
Bloque 4 - Script Principal Integrador
=======================================

Interfaz de línea de comandos para el Bloque 4 del MCMC:
Lattice Gauge y Mass Gap Ontológico.

MODOS DE EJECUCIÓN:
    --validate : Tests de verificación
    --test     : Escaneo rápido de prueba
    --full     : Escaneo completo de producción

EJEMPLO:
    python bloque4_main.py --validate
    python bloque4_main.py --test
    python bloque4_main.py --full

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from __future__ import annotations

import sys
import argparse
import os

# Asegurar imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcmc_ontology_lattice import (
    OntologiaMCMCLattice, crear_ontologia_default, validar_ontologia,
    tabla_sellos, S0, S1, S2, S3, S4
)

from lattice import (
    GrupoGauge, AlgoritmoMC,
    ConfiguracionLattice, ReticulaYangMills, SimuladorMonteCarlo,
    crear_simulacion_MCMC,
    ConfiguracionEscaneo, EscaneoEntropico,
    escaneo_rapido, escaneo_completo,
)


# =============================================================================
# Funciones de Validación
# =============================================================================

def ejecutar_validacion() -> bool:
    """
    Ejecuta tests de validación del Bloque 4.

    Returns:
        True si todos los tests pasan
    """
    print("="*70)
    print("VALIDACIÓN DEL BLOQUE 4: LATTICE GAUGE Y MASS GAP ONTOLÓGICO")
    print("="*70)

    resultados = []

    # Test 1: Núcleo Ontológico
    print("\n[1/4] Núcleo Ontológico...")
    try:
        onto = crear_ontologia_default()
        assert onto.verificar_conservacion(), "Conservación fallida"
        assert onto.verificar_epsilon(tol=0.1), "Epsilon fallido"

        print("  PARÁMETROS DE COLAPSO:")
        print(f"    k₀ = {onto.k0:.4f} Gyr⁻¹")
        print(f"    ε = {onto.epsilon:.4f}")

        print("\n  ESTADOS EN SELLOS:")
        for nombre, estado in onto.estados_sellos().items():
            print(f"    {nombre}: Mp={estado['Mp']:.4f}, E_min={estado['E_min_GeV']:.2f} GeV")

        print("\n  VERIFICACIÓN:")
        print("    ✓ Conservación Mp+Ep = 1")
        print("    ✓ Mp(S₄) ≈ ε")
        resultados.append(True)
        print("  → PASÓ")

    except Exception as e:
        print(f"  ✗ FALLÓ: {e}")
        resultados.append(False)

    # Test 2: Álgebra de Lie
    print("\n[2/4] Álgebra de Lie (SU(2), SU(3))...")
    try:
        from lattice.yang_mills_lattice import AlgebraLie

        for grupo in [GrupoGauge.SU2, GrupoGauge.SU3]:
            algebra = AlgebraLie(grupo)
            U = algebra.elemento_aleatorio()
            err_unit, err_det = algebra.verificar_unitariedad(U)

            assert err_unit < 1e-10, f"Unitariedad {grupo.value}: {err_unit}"
            assert err_det < 1e-10, f"Determinante {grupo.value}: {err_det}"

            print(f"  {grupo.value}:")
            print(f"    Error unitariedad: {err_unit:.2e} ✓")
            print(f"    Error determinante: {err_det:.2e} ✓")

        resultados.append(True)
        print("  → PASÓ")

    except Exception as e:
        print(f"  ✗ FALLÓ: {e}")
        resultados.append(False)

    # Test 3: Retícula Yang-Mills
    print("\n[3/4] Retícula Yang-Mills...")
    try:
        config = ConfiguracionLattice(L=4, grupo=GrupoGauge.SU3, S=S4)
        lattice = ReticulaYangMills(config)

        P_cold = lattice.promedio_plaqueta()
        assert abs(P_cold - 1.0) < 1e-10, f"Cold start: <P> = {P_cold}"

        print(f"  Retícula {config.L}⁴, {config.grupo.value}, S={S4:.4f}:")
        print(f"    β(S) = {config.beta:.4f}")
        print(f"    S_tens = {config.S_tens:.6f}")
        print(f"    ⟨P⟩ (cold start) = {P_cold:.6f} ✓")

        resultados.append(True)
        print("  → PASÓ")

    except Exception as e:
        print(f"  ✗ FALLÓ: {e}")
        resultados.append(False)

    # Test 4: Monte Carlo
    print("\n[4/4] Simulación Monte Carlo...")
    try:
        lattice, sim = crear_simulacion_MCMC(
            S=S4, L=4, grupo=GrupoGauge.SU2,
            n_term=10, verbose=False
        )
        lattice._hot_start()

        P_before = lattice.promedio_plaqueta()
        sim.termalizar(n_sweeps=10, verbose=False)
        P_after = lattice.promedio_plaqueta()

        print(f"  Hot start → Termalización:")
        print(f"    ⟨P⟩ inicial: {P_before:.4f}")
        print(f"    ⟨P⟩ final:   {P_after:.4f}")
        print(f"    Tasa aceptación: {sim.tasa_aceptacion:.3f}")

        resultados.append(True)
        print("  → PASÓ")

    except Exception as e:
        print(f"  ✗ FALLÓ: {e}")
        resultados.append(False)

    # Resumen
    n_pass = sum(resultados)
    n_total = len(resultados)

    print("\n" + "="*70)
    print(f"RESULTADO: {n_pass}/{n_total} tests pasaron")
    print("="*70)

    return all(resultados)


# =============================================================================
# Funciones de Escaneo
# =============================================================================

def ejecutar_test() -> None:
    """Ejecuta escaneo de prueba."""
    print("="*70)
    print("ESCANEO DE PRUEBA")
    print("="*70)

    resultado = escaneo_rapido(verbose=True)

    # Guardar
    config = ConfiguracionEscaneo(prefijo="escaneo_test")
    escaneo = EscaneoEntropico(config)
    escaneo.resultado = resultado
    filepath = escaneo.guardar()
    print(f"\nResultados guardados en: {filepath}")


def ejecutar_full() -> None:
    """Ejecuta escaneo completo."""
    print("="*70)
    print("ESCANEO COMPLETO DE PRODUCCIÓN")
    print("="*70)
    print("\n¡ADVERTENCIA! Este proceso puede tardar varias horas.\n")

    confirmar = input("¿Continuar? [y/N]: ")
    if confirmar.lower() != 'y':
        print("Cancelado.")
        return

    resultado = escaneo_completo(verbose=True)


# =============================================================================
# Función Principal de Información
# =============================================================================

def mostrar_info() -> None:
    """Muestra información del Bloque 4."""
    print("="*70)
    print("BLOQUE 4: LATTICE GAUGE Y MASS GAP ONTOLÓGICO")
    print("="*70)

    onto = crear_ontologia_default()
    print(onto.resumen())

    print("\n")
    print(tabla_sellos())

    print("\n" + "-"*70)
    print("PREGUNTA ONTOLÓGICA CENTRAL:")
    print("-"*70)
    print("  ¿Es el mass gap dinámico extraído de simulaciones Yang-Mills")
    print("  consistente con el mass gap ontológico fijado en S₄ = 1.001?")
    print("-"*70)

    print("\nUSO:")
    print("  python bloque4_main.py --validate  # Tests de verificación")
    print("  python bloque4_main.py --test      # Escaneo rápido")
    print("  python bloque4_main.py --full      # Escaneo completo")
    print("  python bloque4_main.py --info      # Esta información")


# =============================================================================
# Main
# =============================================================================

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Bloque 4 MCMC: Lattice Gauge y Mass Gap Ontológico"
    )

    parser.add_argument(
        "--validate", action="store_true",
        help="Ejecutar tests de validación"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Ejecutar escaneo de prueba"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Ejecutar escaneo completo"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Mostrar información del bloque"
    )

    args = parser.parse_args()

    if args.validate:
        success = ejecutar_validacion()
        sys.exit(0 if success else 1)
    elif args.test:
        ejecutar_test()
    elif args.full:
        ejecutar_full()
    else:
        mostrar_info()


if __name__ == "__main__":
    main()
