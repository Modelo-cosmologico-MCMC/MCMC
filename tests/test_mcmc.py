#!/usr/bin/env python3
"""
Tests del Modelo MCMC
=====================

Tests unitarios para verificar la consistencia del modelo.

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
import sys
import os
import unittest

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcmc_core.bloque0_estado_primordial import (
    EstadoPrimordial, SELLOS, Mp0, Ep0,
    calcular_P_ME, calcular_tension,
    verificar_conservacion, verificar_P_ME_monotonico
)
from mcmc_core.bloque1_pregeometria import (
    Pregeometria, tasa_colapso_k, calcular_epsilon,
    integral_total, K0, A1, A2, A3
)
from mcmc_core.bloque2_cosmologia import (
    CosmologiaMCMC, E_LCDM, E_MCMC, Lambda_relativo,
    H0, OMEGA_M, OMEGA_LAMBDA, DELTA_LAMBDA
)
from simulations.bloque3_nbody import (
    friccion_entropica, perfil_burkert, radio_core,
    ALPHA_FRICCION, RHO_CRITICA
)
from lattice.bloque4_ym_lattice import (
    beta_MCMC, mass_gap, su2_elemento_aleatorio,
    BETA_0, BETA_1, B_S, LAMBDA_QCD
)


class TestBloque0(unittest.TestCase):
    """Tests para el Bloque 0: Estado Primordial."""

    def test_constantes_primordiales(self):
        """Verifica constantes iniciales."""
        self.assertEqual(Mp0, 1.0)
        self.assertEqual(Ep0, 1e-10)

    def test_sellos_ordenados(self):
        """Verifica que los sellos están ordenados."""
        valores = list(SELLOS.values())
        self.assertEqual(valores, sorted(valores))

    def test_P_ME_rango(self):
        """P_ME debe estar en [-1, +1]."""
        estado = EstadoPrimordial.crear_primordial()
        self.assertGreaterEqual(estado.P_ME, -1)
        self.assertLessEqual(estado.P_ME, 1)

    def test_P_ME_inicial_positivo(self):
        """P_ME inicial debe ser ~+1."""
        estado = EstadoPrimordial.crear_primordial()
        self.assertGreater(estado.P_ME, 0.99)

    def test_tension_alta(self):
        """Tensión inicial debe ser muy alta."""
        tension = calcular_tension(Mp0, Ep0)
        self.assertGreater(tension, 1e9)

    def test_conservacion_energia(self):
        """La energía total debe conservarse."""
        estado = EstadoPrimordial.crear_primordial()
        epsilons = {"S1": 0.01, "S2": 0.1, "S3": 0.5, "S4": 0.99}
        historial = estado.trayectoria_completa(epsilons)

        conservada, error = verificar_conservacion(historial)
        self.assertTrue(conservada, f"Error de conservación: {error}")

    def test_P_ME_decreciente(self):
        """P_ME debe decrecer monótonamente."""
        estado = EstadoPrimordial.crear_primordial()
        epsilons = {"S1": 0.01, "S2": 0.1, "S3": 0.5, "S4": 0.99}
        historial = estado.trayectoria_completa(epsilons)

        self.assertTrue(verificar_P_ME_monotonico(historial))


class TestBloque1(unittest.TestCase):
    """Tests para el Bloque 1: Pregeometría."""

    def test_k_positiva(self):
        """k(S) debe ser siempre positiva."""
        for S in np.linspace(0, 2, 50):
            k = tasa_colapso_k(S)
            self.assertGreater(k, 0, f"k({S}) = {k}")

    def test_epsilon_rango(self):
        """ε(S) debe estar en [0, 1]."""
        preg = Pregeometria()
        for S in np.linspace(0, SELLOS["S4"], 50):
            eps = preg.epsilon(S)
            self.assertGreaterEqual(eps, 0)
            self.assertLessEqual(eps, 1)

    def test_epsilon_limites(self):
        """ε(0) = 0, ε(S4) = 1."""
        eps_0 = calcular_epsilon(0)
        eps_S4 = calcular_epsilon(SELLOS["S4"])

        self.assertAlmostEqual(eps_0, 0.0, places=5)
        self.assertAlmostEqual(eps_S4, 1.0, places=2)

    def test_epsilon_monotono(self):
        """ε(S) debe ser monótonamente creciente."""
        preg = Pregeometria()
        eps_prev = -1
        for S in np.linspace(0, SELLOS["S4"], 50):
            eps = preg.epsilon(S)
            self.assertGreaterEqual(eps, eps_prev)
            eps_prev = eps

    def test_integral_positiva(self):
        """La integral entrópica debe ser positiva."""
        I = integral_total()
        self.assertGreater(I, 0)


class TestBloque2(unittest.TestCase):
    """Tests para el Bloque 2: Cosmología."""

    def test_E_LCDM_hoy(self):
        """E_LCDM(0) ≈ 1."""
        E = E_LCDM(0)
        self.assertAlmostEqual(E, 1.0, places=2)

    def test_E_crece_con_z(self):
        """E(z) debe crecer con z."""
        for modelo in [E_LCDM, E_MCMC]:
            E_0 = modelo(0)
            E_1 = modelo(1)
            E_2 = modelo(2)
            self.assertLess(E_0, E_1)
            self.assertLess(E_1, E_2)

    def test_Lambda_rel_limites(self):
        """Λ_rel(0) > 1, Λ_rel(∞) → 1."""
        Lambda_0 = Lambda_relativo(0)
        Lambda_alto = Lambda_relativo(10)

        self.assertGreater(Lambda_0, 1.0)
        self.assertAlmostEqual(Lambda_alto, 1.0, places=2)

    def test_MCMC_LCDM_coinciden_z_alto(self):
        """MCMC y ΛCDM deben coincidir a z alto."""
        z = 10
        E_mcmc = E_MCMC(z)
        E_lcdm = E_LCDM(z)
        diff_rel = abs(E_mcmc - E_lcdm) / E_lcdm
        self.assertLess(diff_rel, 0.01)

    def test_edad_universo_razonable(self):
        """Edad del universo debe ser 12-15 Gyr."""
        cosmo = CosmologiaMCMC()
        edad = cosmo.edad()
        self.assertGreater(edad, 12)
        self.assertLess(edad, 15)


class TestBloque3(unittest.TestCase):
    """Tests para el Bloque 3: N-body."""

    def test_friccion_positiva(self):
        """Fricción entrópica debe ser ≥ 0."""
        for rho in [1e6, 1e8, 1e10]:
            eta = friccion_entropica(rho)
            self.assertGreaterEqual(eta, 0)

    def test_friccion_normalizada(self):
        """η(ρc) = α."""
        eta = friccion_entropica(RHO_CRITICA)
        self.assertAlmostEqual(eta, ALPHA_FRICCION, places=5)

    def test_burkert_finito_centro(self):
        """Perfil Burkert finito en r=0."""
        rho_0 = 1e8
        r_c = 2.0
        rho_centro = perfil_burkert(0, rho_0, r_c)
        self.assertEqual(rho_centro, rho_0)
        self.assertTrue(np.isfinite(rho_centro))

    def test_burkert_decrece(self):
        """Perfil Burkert debe decrecer con r."""
        rho_0 = 1e8
        r_c = 2.0
        rho_1 = perfil_burkert(1, rho_0, r_c)
        rho_5 = perfil_burkert(5, rho_0, r_c)
        self.assertGreater(rho_0, rho_1)
        self.assertGreater(rho_1, rho_5)

    def test_radio_core_crece_con_masa(self):
        """r_core debe crecer con la masa."""
        r_c_1 = radio_core(1e10)
        r_c_2 = radio_core(1e12)
        self.assertGreater(r_c_2, r_c_1)


class TestBloque4(unittest.TestCase):
    """Tests para el Bloque 4: Lattice-Gauge."""

    def test_beta_positivo(self):
        """β(S) debe ser siempre positivo."""
        for S in np.linspace(0, 2, 50):
            beta = beta_MCMC(S)
            self.assertGreater(beta, 0)

    def test_beta_decrece(self):
        """β debe decrecer cuando S aumenta (para S > S3)."""
        beta_05 = beta_MCMC(0.5)
        beta_15 = beta_MCMC(1.5)
        self.assertGreater(beta_05, beta_15)

    def test_mass_gap_positivo(self):
        """Mass gap debe ser positivo."""
        for S in [0.0, 0.5, 1.0, 1.5]:
            E = mass_gap(S)
            self.assertGreater(E, 0)

    def test_su2_unitario(self):
        """Elementos SU(2) deben ser unitarios."""
        for _ in range(10):
            U = su2_elemento_aleatorio()
            # U U† = I
            producto = U @ U.conj().T
            self.assertTrue(np.allclose(producto, np.eye(2)))

    def test_su2_det_uno(self):
        """det(U) = 1 para SU(2)."""
        for _ in range(10):
            U = su2_elemento_aleatorio()
            det = np.linalg.det(U)
            self.assertAlmostEqual(np.abs(det), 1.0, places=5)


class TestIntegracion(unittest.TestCase):
    """Tests de integración entre bloques."""

    def test_sellos_consistentes(self):
        """Los sellos deben ser consistentes entre bloques."""
        from mcmc_core.bloque0_estado_primordial import SELLOS as S0
        from mcmc_core.bloque1_pregeometria import SELLOS as S1

        self.assertEqual(S0, S1)

    def test_trayectoria_completa(self):
        """La trayectoria S0→S4 debe ser consistente."""
        preg = Pregeometria()

        # P_ME inicial
        P_inicial = preg.P_ME(0)
        self.assertGreater(P_inicial, 0.99)

        # P_ME final
        P_final = preg.P_ME(SELLOS["S4"])
        self.assertLess(P_final, -0.9)


def run_all_tests():
    """Ejecuta todos los tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Añadir todos los tests
    suite.addTests(loader.loadTestsFromTestCase(TestBloque0))
    suite.addTests(loader.loadTestsFromTestCase(TestBloque1))
    suite.addTests(loader.loadTestsFromTestCase(TestBloque2))
    suite.addTests(loader.loadTestsFromTestCase(TestBloque3))
    suite.addTests(loader.loadTestsFromTestCase(TestBloque4))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegracion))

    # Ejecutar
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("="*70)
    print("  TESTS DEL MODELO MCMC")
    print("  Autor: Adrián Martínez Estellés (2024)")
    print("="*70)
    print()

    exito = run_all_tests()

    print()
    if exito:
        print("✓ Todos los tests pasaron correctamente")
    else:
        print("✗ Algunos tests fallaron")

    sys.exit(0 if exito else 1)
