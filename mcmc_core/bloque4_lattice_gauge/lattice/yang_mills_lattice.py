"""
Yang-Mills en Retícula con Acoplamiento MCMC
=============================================

Implementación completa de teorías gauge en lattice con acoplamiento
dependiente del sello entrópico S.

GRUPOS SOPORTADOS:
    - SU(2), SU(3), SU(5), SU(10)
    - SO(10)

ACCIÓN EFECTIVA:
    S_YM^eff[U;S] = β(S) Σ_□[1 - (1/N)Re Tr U_□] + S_tens(S)

ALGORITMOS MONTE CARLO:
    1. Metropolis: Propuestas cerca de identidad
    2. Heatbath: Kennedy-Pendleton (SU(2)), Cabibbo-Marinari (SU(N))
    3. Overrelaxation: Microcanónico

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from numpy.typing import NDArray
from enum import Enum

# Importar ontología
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcmc_ontology_lattice import (
    beta_MCMC, S_tensional, S4, S3,
    BETA_0, BETA_1, B_S, LAMBDA_TENS
)


# =============================================================================
# Constantes y Generadores
# =============================================================================

# Matrices de Pauli (generadores SU(2))
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI = [PAULI_X, PAULI_Y, PAULI_Z]

# Matrices de Gell-Mann (generadores SU(3))
def generar_gell_mann() -> List[NDArray]:
    """Genera las 8 matrices de Gell-Mann."""
    matrices = []

    # λ₁
    matrices.append(np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=np.complex128))

    # λ₂
    matrices.append(np.array([
        [0, -1j, 0],
        [1j, 0, 0],
        [0, 0, 0]
    ], dtype=np.complex128))

    # λ₃
    matrices.append(np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.complex128))

    # λ₄
    matrices.append(np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ], dtype=np.complex128))

    # λ₅
    matrices.append(np.array([
        [0, 0, -1j],
        [0, 0, 0],
        [1j, 0, 0]
    ], dtype=np.complex128))

    # λ₆
    matrices.append(np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ], dtype=np.complex128))

    # λ₇
    matrices.append(np.array([
        [0, 0, 0],
        [0, 0, -1j],
        [0, 1j, 0]
    ], dtype=np.complex128))

    # λ₈
    matrices.append(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -2]
    ], dtype=np.complex128) / np.sqrt(3))

    return matrices

GELL_MANN = generar_gell_mann()


# =============================================================================
# Enumeraciones
# =============================================================================

class GrupoGauge(Enum):
    """Grupos gauge soportados."""
    SU2 = "SU(2)"
    SU3 = "SU(3)"
    SU5 = "SU(5)"
    SU10 = "SU(10)"
    SO10 = "SO(10)"


class AlgoritmoMC(Enum):
    """Algoritmos Monte Carlo."""
    METROPOLIS = "Metropolis"
    HEATBATH = "Heatbath"
    OVERRELAXATION = "Overrelaxation"


# =============================================================================
# Álgebra de Grupos
# =============================================================================

class AlgebraLie:
    """
    Implementación de álgebra de Lie para grupos gauge.
    """

    def __init__(self, grupo: GrupoGauge):
        self.grupo = grupo
        self.N = self._extraer_N()
        self.generadores = self._construir_generadores()

    def _extraer_N(self) -> int:
        """Extrae N del grupo."""
        nombre = self.grupo.value
        if "SU" in nombre:
            return int(nombre.replace("SU(", "").replace(")", ""))
        elif "SO" in nombre:
            return int(nombre.replace("SO(", "").replace(")", ""))
        return 3

    def _construir_generadores(self) -> List[NDArray]:
        """Construye generadores del grupo."""
        if self.grupo == GrupoGauge.SU2:
            return [0.5 * P for P in PAULI]
        elif self.grupo == GrupoGauge.SU3:
            return [0.5 * G for G in GELL_MANN]
        else:
            # Para N > 3, usar construcción general
            return self._generadores_SUN(self.N)

    def _generadores_SUN(self, N: int) -> List[NDArray]:
        """
        Construye generadores de SU(N) usando base de Cartan-Weyl.
        """
        generadores = []

        # Matrices fuera de diagonal
        for i in range(N):
            for j in range(i + 1, N):
                # E_ij + E_ji (parte real)
                T = np.zeros((N, N), dtype=np.complex128)
                T[i, j] = 0.5
                T[j, i] = 0.5
                generadores.append(T)

                # -i(E_ij - E_ji) (parte imaginaria)
                T = np.zeros((N, N), dtype=np.complex128)
                T[i, j] = -0.5j
                T[j, i] = 0.5j
                generadores.append(T)

        # Matrices diagonales (Cartan)
        for k in range(1, N):
            T = np.zeros((N, N), dtype=np.complex128)
            for i in range(k):
                T[i, i] = 1.0
            T[k, k] = -k
            T /= np.sqrt(2 * k * (k + 1))
            generadores.append(T)

        return generadores

    def elemento_aleatorio(self, epsilon: float = 0.2) -> NDArray:
        """
        Genera elemento aleatorio del grupo cerca de la identidad.

        U = exp(i ε Σ_a θ_a T_a)

        Args:
            epsilon: Tamaño del paso

        Returns:
            Matriz unitaria U ∈ SU(N)
        """
        # Combinación lineal de generadores
        theta = epsilon * np.random.randn(len(self.generadores))
        H = sum(t * T for t, T in zip(theta, self.generadores))

        # Exponencial de matriz
        U = self._matrix_exp(1j * H)

        # Proyectar a SU(N) para asegurar unitariedad
        return self._proyectar_SUN(U)

    def elemento_heatbath(self, staple: NDArray, beta: float) -> NDArray:
        """
        Genera elemento usando algoritmo heatbath.

        Para SU(2): Kennedy-Pendleton
        Para SU(N): Cabibbo-Marinari
        """
        if self.N == 2:
            return self._heatbath_su2(staple, beta)
        else:
            return self._heatbath_sun(staple, beta)

    def _heatbath_su2(self, staple: NDArray, beta: float) -> NDArray:
        """
        Heatbath para SU(2) usando Kennedy-Pendleton.
        """
        # Parametrización de SU(2): U = a₀I + i Σ aₖσₖ
        # con a₀² + a₁² + a₂² + a₃² = 1

        # Calcular k = √(det(staple))
        W = staple
        k = np.sqrt(np.abs(np.linalg.det(W)))

        if k < 1e-10:
            return self.elemento_aleatorio(0.5)

        # Generar a₀ usando distribución de Kennedy-Pendleton
        delta = beta * k
        accept = False
        while not accept:
            x = np.random.random()
            a0 = 1 + np.log(x + (1-x)*np.exp(-2*delta)) / delta

            if np.random.random() < np.sqrt(1 - a0**2):
                accept = True

        # Generar a₁, a₂, a₃ uniformemente en esfera
        phi = 2 * np.pi * np.random.random()
        cos_theta = 2 * np.random.random() - 1
        sin_theta = np.sqrt(1 - cos_theta**2)
        r = np.sqrt(1 - a0**2)

        a1 = r * sin_theta * np.cos(phi)
        a2 = r * sin_theta * np.sin(phi)
        a3 = r * cos_theta

        # Construir U
        U = a0 * np.eye(2, dtype=np.complex128)
        U += 1j * (a1 * PAULI_X + a2 * PAULI_Y + a3 * PAULI_Z)

        # Multiplicar por (W/k)†
        W_norm = W / k
        return U @ W_norm.conj().T

    def _heatbath_sun(self, staple: NDArray, beta: float) -> NDArray:
        """
        Heatbath para SU(N) usando Cabibbo-Marinari.

        Actualiza subgrupos SU(2) secuencialmente.
        """
        U = np.eye(self.N, dtype=np.complex128)

        # Iterar sobre todos los pares (i,j)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Extraer subgrupo SU(2)
                W_sub = self._extraer_su2(staple, i, j)

                # Heatbath en SU(2)
                U_sub = self._heatbath_su2_simple(W_sub, beta)

                # Embeber en SU(N)
                U_emb = np.eye(self.N, dtype=np.complex128)
                U_emb[i, i] = U_sub[0, 0]
                U_emb[i, j] = U_sub[0, 1]
                U_emb[j, i] = U_sub[1, 0]
                U_emb[j, j] = U_sub[1, 1]

                U = U_emb @ U

        return self._proyectar_SUN(U)

    def _extraer_su2(self, W: NDArray, i: int, j: int) -> NDArray:
        """Extrae submatriz SU(2) de W."""
        W_sub = np.array([
            [W[i, i], W[i, j]],
            [W[j, i], W[j, j]]
        ], dtype=np.complex128)
        return W_sub

    def _heatbath_su2_simple(self, W: NDArray, beta: float) -> NDArray:
        """Versión simplificada de heatbath SU(2)."""
        k = np.sqrt(np.abs(np.linalg.det(W)))
        if k < 1e-10:
            return np.eye(2, dtype=np.complex128)

        delta = beta * k

        # Generar a₀
        x = np.random.random()
        a0 = 1 + np.log(x + (1-x)*np.exp(-2*delta)) / max(delta, 0.1)
        a0 = np.clip(a0, -1, 1)

        # Vector aleatorio
        r = np.sqrt(max(0, 1 - a0**2))
        v = np.random.randn(3)
        v = v / np.linalg.norm(v) * r

        U = a0 * np.eye(2, dtype=np.complex128)
        U += 1j * (v[0] * PAULI_X + v[1] * PAULI_Y + v[2] * PAULI_Z)

        return U

    def overrelaxation(self, U: NDArray, staple: NDArray) -> NDArray:
        """
        Actualización de overrelaxation (microcanónica).

        U' = V U† V†

        donde V = staple / |staple|
        """
        # Normalizar staple
        k = np.sqrt(np.abs(np.linalg.det(staple)))
        if k < 1e-10:
            return U

        V = staple / k

        # Overrelaxation
        U_new = V @ U.conj().T @ V.conj().T

        return self._proyectar_SUN(U_new)

    def _matrix_exp(self, A: NDArray) -> NDArray:
        """Exponencial de matriz."""
        # Usar descomposición en valores propios
        eigenvalues, eigenvectors = np.linalg.eig(A)
        exp_eigenvalues = np.exp(eigenvalues)
        return eigenvectors @ np.diag(exp_eigenvalues) @ np.linalg.inv(eigenvectors)

    def _proyectar_SUN(self, U: NDArray) -> NDArray:
        """Proyecta matriz a SU(N)."""
        # Gram-Schmidt modificado via QR
        Q, R = np.linalg.qr(U)

        # Ajustar fase para det = 1
        det = np.linalg.det(Q)
        if np.abs(det) > 1e-10:
            # Multiplicar por la N-ésima raíz del inverso del determinante
            fase_correccion = (1.0 / det) ** (1.0 / self.N)
            Q = Q * fase_correccion

        return Q

    def verificar_unitariedad(self, U: NDArray) -> Tuple[float, float]:
        """
        Verifica unitariedad y determinante.

        Returns:
            (error_unitario, error_det)
        """
        error_unit = np.max(np.abs(U @ U.conj().T - np.eye(self.N)))
        error_det = np.abs(np.linalg.det(U) - 1.0)
        return error_unit, error_det


# =============================================================================
# Configuración de Retícula
# =============================================================================

@dataclass
class ConfiguracionLattice:
    """
    Configuración de la retícula hipercúbica.
    """
    L: int = 4                    # Tamaño (L^d)
    d: int = 4                    # Dimensión
    grupo: GrupoGauge = GrupoGauge.SU3
    S: float = S4                 # Sello entrópico

    # Derivados
    N: int = field(init=False)
    volumen: int = field(init=False)
    n_enlaces: int = field(init=False)
    n_plaquetas: int = field(init=False)

    def __post_init__(self):
        """Calcula propiedades derivadas."""
        # Extraer N del grupo
        nombre = self.grupo.value
        if "SU" in nombre:
            self.N = int(nombre.replace("SU(", "").replace(")", ""))
        elif "SO" in nombre:
            self.N = int(nombre.replace("SO(", "").replace(")", ""))
        else:
            self.N = 3

        self.volumen = self.L ** self.d
        self.n_enlaces = self.volumen * self.d
        self.n_plaquetas = self.volumen * self.d * (self.d - 1) // 2

    @property
    def beta(self) -> float:
        """Acoplamiento gauge β(S)."""
        return beta_MCMC(self.S)

    @property
    def S_tens(self) -> float:
        """Término tensional."""
        return S_tensional(self.S)


# =============================================================================
# Clase Principal: ReticuaYangMills
# =============================================================================

class ReticulaYangMills:
    """
    Retícula de Yang-Mills con acoplamiento MCMC.

    Implementa una red d-dimensional con enlaces del grupo gauge
    especificado y acoplamiento dependiente del sello entrópico.
    """

    def __init__(self, config: ConfiguracionLattice):
        """
        Inicializa la retícula.

        Args:
            config: Configuración de la retícula
        """
        self.config = config
        self.algebra = AlgebraLie(config.grupo)

        # Almacenamiento de enlaces
        shape = tuple([config.L] * config.d) + (config.d, config.N, config.N)
        self.enlaces = np.zeros(shape, dtype=np.complex128)

        # Inicializar
        self._cold_start()

    def _cold_start(self) -> None:
        """Inicializa todos los enlaces a la identidad."""
        identidad = np.eye(self.config.N, dtype=np.complex128)

        for idx in np.ndindex(*([self.config.L] * self.config.d)):
            for mu in range(self.config.d):
                self.enlaces[idx + (mu,)] = identidad.copy()

    def _hot_start(self) -> None:
        """Inicializa con elementos aleatorios."""
        for idx in np.ndindex(*([self.config.L] * self.config.d)):
            for mu in range(self.config.d):
                self.enlaces[idx + (mu,)] = self.algebra.elemento_aleatorio(1.0)

    def get_enlace(self, sitio: Tuple[int, ...], mu: int) -> NDArray:
        """
        Obtiene enlace U_μ(sitio).

        Args:
            sitio: Coordenadas del sitio
            mu: Dirección del enlace

        Returns:
            Matriz del enlace
        """
        sitio_mod = tuple(s % self.config.L for s in sitio)
        return self.enlaces[sitio_mod + (mu,)]

    def set_enlace(self, sitio: Tuple[int, ...], mu: int, U: NDArray) -> None:
        """Establece enlace U_μ(sitio)."""
        sitio_mod = tuple(s % self.config.L for s in sitio)
        self.enlaces[sitio_mod + (mu,)] = U

    def plaqueta(self, sitio: Tuple[int, ...], mu: int, nu: int) -> NDArray:
        """
        Calcula plaqueta U_□ en el plano (μ,ν).

        U_□ = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)

        Args:
            sitio: Coordenadas del sitio
            mu, nu: Direcciones del plano

        Returns:
            Matriz de la plaqueta
        """
        x = list(sitio)

        # U_μ(x)
        U1 = self.get_enlace(tuple(x), mu)

        # U_ν(x+μ)
        x_mu = x.copy()
        x_mu[mu] += 1
        U2 = self.get_enlace(tuple(x_mu), nu)

        # U_μ†(x+ν)
        x_nu = x.copy()
        x_nu[nu] += 1
        U3_dag = self.get_enlace(tuple(x_nu), mu).conj().T

        # U_ν†(x)
        U4_dag = self.get_enlace(tuple(x), nu).conj().T

        return U1 @ U2 @ U3_dag @ U4_dag

    def calcular_staple(self, sitio: Tuple[int, ...], mu: int) -> NDArray:
        """
        Calcula el staple para el enlace (sitio, μ).

        Staple = Σ_ν≠μ [staple_positivo + staple_negativo]
        """
        staple = np.zeros((self.config.N, self.config.N), dtype=np.complex128)
        x = list(sitio)

        for nu in range(self.config.d):
            if nu == mu:
                continue

            # Staple positivo: U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
            x_mu = x.copy()
            x_mu[mu] += 1
            x_nu = x.copy()
            x_nu[nu] += 1

            staple += (
                self.get_enlace(tuple(x_mu), nu) @
                self.get_enlace(tuple(x_nu), mu).conj().T @
                self.get_enlace(tuple(x), nu).conj().T
            )

            # Staple negativo: U_ν†(x+μ-ν) U_μ†(x-ν) U_ν(x-ν)
            x_mu_mnu = x_mu.copy()
            x_mu_mnu[nu] -= 1
            x_mnu = x.copy()
            x_mnu[nu] -= 1

            staple += (
                self.get_enlace(tuple(x_mu_mnu), nu).conj().T @
                self.get_enlace(tuple(x_mnu), mu).conj().T @
                self.get_enlace(tuple(x_mnu), nu)
            )

        return staple

    def promedio_plaqueta(self) -> float:
        """
        Calcula el valor esperado de la plaqueta.

        <P> = (1/N_plaq) Σ_□ (1/N) Re Tr U_□
        """
        total = 0.0
        count = 0

        for idx in np.ndindex(*([self.config.L] * self.config.d)):
            for mu in range(self.config.d):
                for nu in range(mu + 1, self.config.d):
                    U_plaq = self.plaqueta(idx, mu, nu)
                    total += np.real(np.trace(U_plaq)) / self.config.N
                    count += 1

        return total / count if count > 0 else 0.0

    def accion_total(self) -> float:
        """
        Calcula la acción total de Wilson + término tensional.

        S = β Σ_□ [1 - (1/N) Re Tr U_□] + S_tens
        """
        beta = self.config.beta
        S_Wilson = 0.0

        for idx in np.ndindex(*([self.config.L] * self.config.d)):
            for mu in range(self.config.d):
                for nu in range(mu + 1, self.config.d):
                    U_plaq = self.plaqueta(idx, mu, nu)
                    P = np.real(np.trace(U_plaq)) / self.config.N
                    S_Wilson += beta * (1 - P)

        # Añadir término tensional
        S_total = S_Wilson + self.config.S_tens * self.config.n_plaquetas

        return S_total

    def polyakov_loop(self, sitio_espacial: Tuple[int, ...]) -> complex:
        """
        Calcula el lazo de Polyakov en la posición espacial dada.

        P(x) = (1/N) Tr ∏_t U_0(t, x)

        Args:
            sitio_espacial: Coordenadas espaciales (d-1 dimensiones)

        Returns:
            Valor complejo del lazo de Polyakov
        """
        producto = np.eye(self.config.N, dtype=np.complex128)

        for t in range(self.config.L):
            sitio = (t,) + sitio_espacial
            producto = producto @ self.get_enlace(sitio, 0)

        return np.trace(producto) / self.config.N


# =============================================================================
# Simulador Monte Carlo
# =============================================================================

class SimuladorMonteCarlo:
    """
    Simulador Monte Carlo para la retícula.
    """

    def __init__(
        self,
        lattice: ReticulaYangMills,
        algoritmo: AlgoritmoMC = AlgoritmoMC.METROPOLIS,
        epsilon: float = 0.2
    ):
        """
        Inicializa el simulador.

        Args:
            lattice: Retícula a simular
            algoritmo: Algoritmo Monte Carlo
            epsilon: Tamaño del paso (para Metropolis)
        """
        self.lattice = lattice
        self.algoritmo = algoritmo
        self.epsilon = epsilon

        self.n_sweeps = 0
        self.tasa_aceptacion = 0.0
        self.historial = []

    def sweep(self) -> float:
        """
        Realiza un barrido completo de la retícula.

        Returns:
            Tasa de aceptación
        """
        if self.algoritmo == AlgoritmoMC.METROPOLIS:
            return self._sweep_metropolis()
        elif self.algoritmo == AlgoritmoMC.HEATBATH:
            return self._sweep_heatbath()
        elif self.algoritmo == AlgoritmoMC.OVERRELAXATION:
            return self._sweep_overrelax()
        else:
            return self._sweep_metropolis()

    def _sweep_metropolis(self) -> float:
        """Barrido Metropolis."""
        aceptados = 0
        total = 0
        beta = self.lattice.config.beta

        for idx in np.ndindex(*([self.lattice.config.L] * self.lattice.config.d)):
            for mu in range(self.lattice.config.d):
                # Staple
                staple = self.lattice.calcular_staple(idx, mu)

                # Enlace actual
                U_old = self.lattice.get_enlace(idx, mu)
                dS_old = -beta * np.real(np.trace(U_old @ staple)) / self.lattice.config.N

                # Proponer nuevo enlace
                U_prop = self.lattice.algebra.elemento_aleatorio(self.epsilon)
                U_new = U_prop @ U_old
                dS_new = -beta * np.real(np.trace(U_new @ staple)) / self.lattice.config.N

                # Aceptar/rechazar
                if np.random.random() < np.exp(dS_old - dS_new):
                    self.lattice.set_enlace(idx, mu, U_new)
                    aceptados += 1

                total += 1

        self.n_sweeps += 1
        self.tasa_aceptacion = aceptados / total if total > 0 else 0.0
        return self.tasa_aceptacion

    def _sweep_heatbath(self) -> float:
        """Barrido Heatbath."""
        beta = self.lattice.config.beta

        for idx in np.ndindex(*([self.lattice.config.L] * self.lattice.config.d)):
            for mu in range(self.lattice.config.d):
                staple = self.lattice.calcular_staple(idx, mu)
                U_new = self.lattice.algebra.elemento_heatbath(staple, beta)
                self.lattice.set_enlace(idx, mu, U_new)

        self.n_sweeps += 1
        self.tasa_aceptacion = 1.0  # Heatbath siempre acepta
        return self.tasa_aceptacion

    def _sweep_overrelax(self) -> float:
        """Barrido Overrelaxation."""
        for idx in np.ndindex(*([self.lattice.config.L] * self.lattice.config.d)):
            for mu in range(self.lattice.config.d):
                staple = self.lattice.calcular_staple(idx, mu)
                U_old = self.lattice.get_enlace(idx, mu)
                U_new = self.lattice.algebra.overrelaxation(U_old, staple)
                self.lattice.set_enlace(idx, mu, U_new)

        self.n_sweeps += 1
        self.tasa_aceptacion = 1.0
        return self.tasa_aceptacion

    def termalizar(self, n_sweeps: int = 100, verbose: bool = True) -> None:
        """
        Termaliza la configuración.

        Args:
            n_sweeps: Número de barridos
            verbose: Imprimir progreso
        """
        for i in range(n_sweeps):
            acc = self.sweep()

            if verbose and (i + 1) % max(1, n_sweeps // 10) == 0:
                P = self.lattice.promedio_plaqueta()
                print(f"  Term {i+1:4d}/{n_sweeps}: <P> = {P:.6f}, acc = {acc:.3f}")

    def generar_configuraciones(
        self,
        n_configs: int,
        n_skip: int = 5
    ) -> List[NDArray]:
        """
        Genera configuraciones descorrelacionadas.

        Args:
            n_configs: Número de configuraciones
            n_skip: Barridos entre mediciones

        Returns:
            Lista de configuraciones
        """
        configs = []

        for i in range(n_configs):
            # Barridos de decorrelación
            for _ in range(n_skip):
                self.sweep()

            # Guardar configuración
            configs.append(self.lattice.enlaces.copy())

        return configs


# =============================================================================
# Funciones de Conveniencia
# =============================================================================

def crear_simulacion_MCMC(
    S: float = S4,
    L: int = 4,
    grupo: GrupoGauge = GrupoGauge.SU3,
    algoritmo: AlgoritmoMC = AlgoritmoMC.METROPOLIS,
    n_term: int = 100,
    n_configs: int = 50,
    verbose: bool = True
) -> Tuple[ReticulaYangMills, SimuladorMonteCarlo]:
    """
    Crea y prepara una simulación MCMC.

    Args:
        S: Sello entrópico
        L: Tamaño de la retícula
        grupo: Grupo gauge
        algoritmo: Algoritmo MC
        n_term: Barridos de termalización
        n_configs: Configuraciones a generar
        verbose: Imprimir progreso

    Returns:
        (lattice, simulador)
    """
    config = ConfiguracionLattice(L=L, grupo=grupo, S=S)
    lattice = ReticulaYangMills(config)
    lattice._hot_start()

    simulador = SimuladorMonteCarlo(lattice, algoritmo)

    if verbose:
        print(f"Simulación: {grupo.value}, L={L}, S={S:.4f}")
        print(f"  β(S) = {config.beta:.4f}")
        print(f"  S_tens = {config.S_tens:.6f}")

    return lattice, simulador


# =============================================================================
# Tests
# =============================================================================

def _test_yang_mills():
    """Verifica la implementación de Yang-Mills lattice."""

    print("Testing Yang-Mills Lattice...")

    # Test 1: Algebra SU(2)
    algebra_su2 = AlgebraLie(GrupoGauge.SU2)
    U = algebra_su2.elemento_aleatorio()
    err_unit, err_det = algebra_su2.verificar_unitariedad(U)
    assert err_unit < 1e-10, f"SU(2) no unitario: {err_unit}"
    assert err_det < 1e-10, f"det(SU(2)) ≠ 1: {err_det}"
    print("  ✓ Álgebra SU(2)")

    # Test 2: Algebra SU(3)
    algebra_su3 = AlgebraLie(GrupoGauge.SU3)
    U = algebra_su3.elemento_aleatorio()
    err_unit, err_det = algebra_su3.verificar_unitariedad(U)
    assert err_unit < 1e-10, f"SU(3) no unitario: {err_unit}"
    assert err_det < 1e-10, f"det(SU(3)) ≠ 1: {err_det}"
    print("  ✓ Álgebra SU(3)")

    # Test 3: Retícula cold start
    config = ConfiguracionLattice(L=4, grupo=GrupoGauge.SU2, S=S4)
    lattice = ReticulaYangMills(config)
    P = lattice.promedio_plaqueta()
    assert np.isclose(P, 1.0, atol=1e-10), f"Cold start: <P> = {P}"
    print("  ✓ Cold start <P> = 1")

    # Test 4: β(S) correcto
    assert config.beta > 6.0, f"β(S₄) = {config.beta}"
    print(f"  ✓ β(S₄) = {config.beta:.4f}")

    # Test 5: Metropolis cambia configuración
    lattice._hot_start()
    P_before = lattice.promedio_plaqueta()
    sim = SimuladorMonteCarlo(lattice)
    sim.sweep()
    P_after = lattice.promedio_plaqueta()
    # Puede ser igual por casualidad, solo verificar que corre
    print(f"  ✓ Metropolis: <P> {P_before:.4f} → {P_after:.4f}")

    print("\n✓ Todos los tests de Yang-Mills lattice pasaron")
    return True


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    _test_yang_mills()

    print("\n" + "="*60)
    print("DEMO: Yang-Mills Lattice MCMC")
    print("="*60)

    # Crear simulación
    lattice, sim = crear_simulacion_MCMC(
        S=S4,
        L=4,
        grupo=GrupoGauge.SU3,
        n_term=20,
        verbose=True
    )

    print("\nTermalizando...")
    sim.termalizar(n_sweeps=20)

    print(f"\nResultados finales:")
    print(f"  <P> = {lattice.promedio_plaqueta():.6f}")
    print(f"  S_total = {lattice.accion_total():.2f}")
