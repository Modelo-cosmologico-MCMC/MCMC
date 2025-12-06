#!/usr/bin/env python3
"""
Ejemplo Completo: Los 5 Bloques del Modelo MCMC
================================================

Este script demuestra el uso integrado de los 5 bloques del
Modelo Cosmológico de Múltiples Colapsos (MCMC).

Bloques:
    0. Estado Primordial - Mp0, Ep0, tensión máxima
    1. Pregeometría - Tasa de colapso k(S), integral entrópica
    2. Cosmología - Ecuaciones de Friedmann modificadas
    3. N-body - Fricción entrópica, perfiles Burkert
    4. Lattice-Gauge - Yang-Mills, mass gap

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Uso:
    python ejemplo_5_bloques.py
"""

import numpy as np
import sys
import os

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcmc_core.bloque0_estado_primordial import (
    EstadoPrimordial, SELLOS, Mp0, Ep0
)
from mcmc_core.bloque1_pregeometria import (
    Pregeometria, tasa_colapso_k, K0, A1, A2, A3
)
from mcmc_core.bloque2_cosmologia import (
    CosmologiaMCMC, E_LCDM, E_MCMC, H0, OMEGA_M, OMEGA_LAMBDA
)
from simulations.bloque3_nbody import (
    SimulacionNBody, friccion_entropica, perfil_burkert, radio_core
)
from lattice.bloque4_ym_lattice import (
    LatticeYM, beta_MCMC, mass_gap, BETA_0, LAMBDA_QCD
)


def separador(titulo: str) -> None:
    """Imprime separador con título."""
    print("\n" + "="*70)
    print(f"  {titulo}")
    print("="*70 + "\n")


def bloque_0_demo():
    """Demuestra el Bloque 0: Estado Primordial."""
    separador("BLOQUE 0: Estado Primordial")

    print("El universo comienza en un estado de máxima tensión:")
    print(f"  Mp0 = {Mp0}")
    print(f"  Ep0 = {Ep0:.0e}")
    print(f"  Tensión = Mp0/Ep0 = {Mp0/Ep0:.0e}")
    print()

    # Crear estado primordial
    estado = EstadoPrimordial.crear_primordial()
    print(f"Estado en S0:")
    print(f"  P_ME = {estado.P_ME:+.6f} (masa domina)")
    print()

    # Evolucionar a través de los sellos
    print("Sellos entrópicos:")
    for nombre, valor in SELLOS.items():
        print(f"  {nombre} = {valor:.3f}")
    print()

    # Trayectoria completa
    print("Evolución P_ME (de +1 a -1):")
    epsilons = {"S1": 0.01, "S2": 0.10, "S3": 0.50, "S4": 0.99}
    historial = estado.trayectoria_completa(epsilons)

    for h in historial:
        print(f"  {h['sello']}: P_ME = {h['P_ME']:+.4f}")


def bloque_1_demo():
    """Demuestra el Bloque 1: Pregeometría."""
    separador("BLOQUE 1: Pregeometría")

    print("Tasa de colapso k(S):")
    print(f"  k(S) = k0 × [1 + a1·sin(2πS) + a2·sin(4πS) + a3·sin(6πS)]")
    print(f"  k0 = {K0:.3f}, a1 = {A1}, a2 = {A2}, a3 = {A3}")
    print()

    preg = Pregeometria()

    print("Valores de k(S) en los sellos:")
    for nombre, S in SELLOS.items():
        k = preg.k(S)
        eps = preg.epsilon(S)
        print(f"  {nombre} (S={S:.3f}): k = {k:.4f}, ε = {eps:.4f}")
    print()

    # Punto de equilibrio
    S_eq = preg.punto_equilibrio()
    print(f"Punto de equilibrio (P_ME = 0): S = {S_eq:.4f}")


def bloque_2_demo():
    """Demuestra el Bloque 2: Cosmología."""
    separador("BLOQUE 2: Cosmología MCMC")

    print("Parámetros cosmológicos:")
    print(f"  H0 = {H0} km/s/Mpc")
    print(f"  Ωm = {OMEGA_M}")
    print(f"  ΩΛ = {OMEGA_LAMBDA}")
    print()

    cosmo = CosmologiaMCMC()

    print("E(z) = H(z)/H0:")
    print(f"  {'z':>5} | {'E_ΛCDM':>10} | {'E_MCMC':>10} | {'Diff (%)':>10}")
    print("  " + "-"*45)

    for z in [0, 0.5, 1.0, 1.5, 2.0]:
        E_l = E_LCDM(z)
        E_m = E_MCMC(z)
        diff = 100 * (E_m - E_l) / E_l
        print(f"  {z:5.2f} | {E_l:10.4f} | {E_m:10.4f} | {diff:+10.3f}")
    print()

    print(f"Edad del universo: {cosmo.edad():.2f} Gyr")
    print()

    print("Distancias luminosidad:")
    for z in [0.5, 1.0, 2.0]:
        D_L = cosmo.D_L(z)
        print(f"  z = {z}: D_L = {D_L:.1f} Mpc")


def bloque_3_demo():
    """Demuestra el Bloque 3: N-body."""
    separador("BLOQUE 3: N-body con Fricción Entrópica")

    print("Fricción entrópica:")
    print("  η(ρ) = α × (ρ/ρc)^1.5")
    print()

    print("Perfil de Burkert:")
    print("  ρ(r) = ρ0 / [(1 + r/rc)(1 + (r/rc)²)]")
    print()

    print("Relación radio core - masa:")
    print("  r_core(M) = 1.8 × (M/10¹¹ M☉)^0.35 kpc")
    print()

    print("Ejemplos de r_core:")
    for log_M in [10, 11, 12]:
        M = 10**log_M
        r_c = radio_core(M)
        print(f"  M = 10^{log_M} M☉: r_core = {r_c:.2f} kpc")
    print()

    # Crear simulación pequeña
    print("Simulación de halo (10 partículas, demo):")
    sim = SimulacionNBody.crear_halo_inicial(
        n_particulas=10,
        M_total=1e11,
        alpha=0.1
    )
    print(f"  ρ0 = {sim.rho_0:.2e} M☉/kpc³")
    print(f"  r_c = {sim.r_c:.2f} kpc")


def bloque_4_demo():
    """Demuestra el Bloque 4: Lattice-Gauge."""
    separador("BLOQUE 4: Yang-Mills Lattice")

    print("Acoplamiento gauge dependiente de S:")
    print("  β(S) = β0 + β1 × exp[-bS × (S - S3)]")
    print(f"  β0 = {BETA_0}, β1 = 2.0, bS = 10.0, S3 = 1.0")
    print()

    print("β(S) vs Sello S:")
    for S in [0.0, 0.5, 1.0, 1.5, 2.0]:
        beta = beta_MCMC(S)
        print(f"  S = {S:.1f}: β = {beta:.3f}")
    print()

    print("Mass gap:")
    print(f"  E_min = αH × ΛQCD")
    print(f"  αH = 0.1, ΛQCD = {LAMBDA_QCD} GeV")
    print()

    print("E_min(S) vs Sello S:")
    for S in [0.0, 0.5, 1.0, 1.5, 2.0]:
        E = mass_gap(S)
        print(f"  S = {S:.1f}: E_min = {E:.4f} GeV")
    print()

    # Crear lattice pequeño
    print("Simulación lattice 2⁴ (demo):")
    lattice = LatticeYM(L=2, S=1.0)
    P = lattice.calcular_plaqueta_promedio()
    print(f"  Cold start <P> = {P:.4f}")


def resumen_mcmc():
    """Resumen integrado del modelo."""
    separador("RESUMEN: Modelo MCMC Integrado")

    print("El Modelo Cosmológico de Múltiples Colapsos (MCMC) describe")
    print("la evolución del universo desde un estado primordial de")
    print("máxima tensión hasta el estado actual.\n")

    print("Características principales:")
    print("  • Ontología basada en sellos entrópicos S0→S4")
    print("  • Tasa de colapso k(S) con armónicos sinusoidales")
    print("  • Cosmología que reduce tensiones H0 y S8")
    print("  • Fricción entrópica que explica núcleos tipo Burkert")
    print("  • Conexión con QCD y el problema del mass gap")
    print()

    print("Predicciones verificables:")
    print("  1. Modificación de E(z) respecto a ΛCDM")
    print("  2. Relación masa-núcleo en galaxias")
    print("  3. Mass gap dependiente de la escala")
    print()

    print("Autor: Adrián Martínez Estellés (2024)")
    print("Contacto: adrianmartinezestelles92@gmail.com")


def main():
    """Ejecuta todas las demos."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#    MODELO COSMOLÓGICO DE MÚLTIPLES COLAPSOS (MCMC)" + " "*17 + "#")
    print("#    Demostración de los 5 Bloques" + " "*32 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    print("\n    Autor: Adrián Martínez Estellés")
    print("    Copyright (c) 2024. Todos los derechos reservados.\n")

    # Ejecutar demos
    bloque_0_demo()
    bloque_1_demo()
    bloque_2_demo()
    bloque_3_demo()
    bloque_4_demo()
    resumen_mcmc()

    print("\n" + "="*70)
    print("  Demo completada exitosamente")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
