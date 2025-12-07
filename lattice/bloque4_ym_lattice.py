"""
Bloque 4 - Yang-Mills Lattice Gauge MCMC
=========================================

Teoría de gauge en la red con acoplamiento dependiente del sello entrópico.

En el modelo MCMC, el acoplamiento gauge varía con S:
    β(S) = β0 + β1 × exp[-bS × (S - S3)]

Parámetros:
    - β0 = 6.0 (acoplamiento base)
    - β1 = 2.0 (amplitud de variación)
    - bS = 10.0 (escala de transición)
    - S3 = 1.0 (sello de referencia)

El mass gap se estima como:
    E_min = αH × ΛQCD

donde αH ≈ 0.1 y ΛQCD ≈ 0.2 GeV.

Este módulo conecta la ontología MCMC con:
    - El problema del mass gap en Yang-Mills
    - Confinamiento de quarks
    - QCD a bajas energías

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
Contacto: adrianmartinezestelles92@gmail.com

Referencias:
    - Martínez Estellés, A. (2024). "Modelo Cosmológico de Múltiples Colapsos"
    - Wilson, K. (1974). "Confinement of Quarks"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray


# =============================================================================
# Constantes del Modelo
# =============================================================================

# Parámetros de β(S)
BETA_0: float = 6.0    # Acoplamiento base
BETA_1: float = 2.0    # Amplitud de variación
B_S: float = 10.0      # Escala de transición
S3: float = 1.0        # Sello de referencia

# Parámetros de QCD
LAMBDA_QCD: float = 0.2    # GeV
ALPHA_H: float = 0.1       # Factor para mass gap

# Matrices de Pauli (generadores SU(2))
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# =============================================================================
# Funciones de Acoplamiento MCMC
# =============================================================================

def beta_MCMC(
    S: float,
    beta_0: float = BETA_0,
    beta_1: float = BETA_1,
    b_S: float = B_S,
    S_ref: float = S3
) -> float:
    """
    Calcula el acoplamiento gauge dependiente del sello entrópico.

    β(S) = β0 + β1 × exp[-bS × (S - S3)]

    Propiedades:
        - β(S→-∞) = β0 + β1 (acoplamiento fuerte, confinamiento)
        - β(S=S3) = β0 + β1 (punto de transición)
        - β(S→+∞) = β0 (acoplamiento débil)

    Args:
        S: Sello entrópico
        beta_0: Acoplamiento base
        beta_1: Amplitud
        b_S: Escala de transición
        S_ref: Sello de referencia

    Returns:
        β(S)
    """
    return beta_0 + beta_1 * np.exp(-b_S * (S - S_ref))


def constante_acoplamiento(beta: float, N: int = 3) -> float:
    """
    Convierte β a constante de acoplamiento g².

    β = 2N/g² → g² = 2N/β

    Args:
        beta: Parámetro de acoplamiento lattice
        N: Dimensión del grupo (3 para SU(3))

    Returns:
        g²
    """
    if beta <= 0:
        return np.inf
    return 2 * N / beta


def alpha_s(beta: float, N: int = 3) -> float:
    """
    Constante de acoplamiento fuerte αs = g²/(4π).

    Args:
        beta: Parámetro de acoplamiento lattice
        N: Dimensión del grupo

    Returns:
        αs
    """
    g_sq = constante_acoplamiento(beta, N)
    return g_sq / (4 * np.pi)


# =============================================================================
# Mass Gap
# =============================================================================

def mass_gap(
    S: float,
    alpha_h: float = ALPHA_H,
    Lambda_qcd: float = LAMBDA_QCD
) -> float:
    """
    Estima el mass gap E_min(S) del modelo MCMC.

    E_min = αH × ΛQCD × f(S)

    donde f(S) captura la dependencia en el sello entrópico.

    El mass gap es la diferencia de energía entre el vacío
    y el primer estado excitado. En Yang-Mills puro, E_min > 0
    es una predicción fundamental (problema del milenio).

    Args:
        S: Sello entrópico
        alpha_h: Factor de proporcionalidad
        Lambda_qcd: Escala QCD (GeV)

    Returns:
        E_min en GeV
    """
    # Factor que depende del sello
    beta = beta_MCMC(S)
    g_sq = constante_acoplamiento(beta)

    # En el régimen de acoplamiento fuerte, E_min ~ ΛQCD
    # El factor f(S) modula esto según el sello
    f_S = 1.0 / (1.0 + 0.1 * (S - S3)**2)

    return alpha_h * Lambda_qcd * f_S


def tension_cuerda(beta: float, a: float = 1.0) -> float:
    """
    Estima la tensión de cuerda σ.

    En el régimen de confinamiento:
    σ ≈ ΛQCD² / β

    Args:
        beta: Acoplamiento lattice
        a: Espaciado de la red (fm)

    Returns:
        σ en GeV/fm
    """
    if beta <= 0:
        return np.inf

    return LAMBDA_QCD**2 / beta


# =============================================================================
# Funciones de Lattice
# =============================================================================

def su2_elemento_aleatorio(epsilon: float = 0.2) -> NDArray[np.complex128]:
    """
    Genera elemento aleatorio de SU(2) cerca de la identidad.

    U = cos(θ)I + i sin(θ) n̂·σ

    Args:
        epsilon: Tamaño del paso

    Returns:
        Matriz 2×2 unitaria con det = 1
    """
    n = np.random.randn(3)
    n = n / np.linalg.norm(n)
    theta = epsilon * np.pi * np.random.random()

    U = np.cos(theta) * np.eye(2, dtype=np.complex128)
    U += 1j * np.sin(theta) * (n[0]*PAULI_X + n[1]*PAULI_Y + n[2]*PAULI_Z)

    return U


def accion_wilson(plaqueta: NDArray[np.complex128], beta: float, N: int = 2) -> float:
    """
    Acción de Wilson para una plaqueta.

    S_□ = β × (1 - (1/N) Re Tr U_□)

    Args:
        plaqueta: Producto de enlaces U_□
        beta: Acoplamiento
        N: Dimensión del grupo

    Returns:
        Contribución a la acción
    """
    trace = np.trace(plaqueta)
    return beta * (1 - np.real(trace) / N)


def promedio_plaqueta(U_plaq: NDArray[np.complex128], N: int = 2) -> float:
    """
    Valor esperado de la plaqueta.

    <P> = (1/N) Re Tr U_□

    Args:
        U_plaq: Plaqueta
        N: Dimensión del grupo

    Returns:
        <P>
    """
    return np.real(np.trace(U_plaq)) / N


# =============================================================================
# Clase Principal
# =============================================================================

@dataclass
class LatticeYM:
    """
    Simulación de Yang-Mills en la red con acoplamiento MCMC.

    Implementa una red 4D con enlaces SU(2) y acoplamiento
    que depende del sello entrópico.

    Attributes:
        L: Tamaño de la red (L⁴)
        N: Dimensión del grupo
        S: Sello entrópico actual
        enlaces: Array de enlaces
    """
    L: int = 4
    N: int = 2
    S: float = 1.0
    enlaces: Optional[NDArray[np.complex128]] = None

    # Estadísticas
    plaqueta_promedio: float = 0.0
    historial: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Inicializa la red."""
        if self.enlaces is None:
            self._cold_start()

    def _cold_start(self) -> None:
        """Inicializa todos los enlaces a la identidad."""
        shape = (self.L, self.L, self.L, self.L, 4, self.N, self.N)
        self.enlaces = np.zeros(shape, dtype=np.complex128)

        identidad = np.eye(self.N, dtype=np.complex128)
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for t in range(self.L):
                        for mu in range(4):
                            self.enlaces[x, y, z, t, mu] = identidad

    def _hot_start(self) -> None:
        """Inicializa con elementos aleatorios."""
        shape = (self.L, self.L, self.L, self.L, 4, self.N, self.N)
        self.enlaces = np.zeros(shape, dtype=np.complex128)

        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for t in range(self.L):
                        for mu in range(4):
                            self.enlaces[x, y, z, t, mu] = su2_elemento_aleatorio(1.0)

    @property
    def beta(self) -> float:
        """Acoplamiento actual según el sello S."""
        return beta_MCMC(self.S)

    @property
    def volumen(self) -> int:
        """Volumen de la red."""
        return self.L**4

    @property
    def n_plaquetas(self) -> int:
        """Número de plaquetas."""
        return self.volumen * 6  # 6 planos por sitio

    def get_enlace(self, x: int, y: int, z: int, t: int, mu: int) -> NDArray[np.complex128]:
        """Obtiene enlace U_μ(x)."""
        return self.enlaces[x % self.L, y % self.L, z % self.L, t % self.L, mu]

    def set_enlace(self, x: int, y: int, z: int, t: int, mu: int,
                   U: NDArray[np.complex128]) -> None:
        """Establece enlace U_μ(x)."""
        self.enlaces[x % self.L, y % self.L, z % self.L, t % self.L, mu] = U

    def plaqueta(self, x: int, y: int, z: int, t: int, mu: int, nu: int) -> NDArray[np.complex128]:
        """
        Calcula plaqueta U_□ en el plano (μ,ν).

        U_□ = U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
        """
        sitio = [x, y, z, t]

        # U_μ(x)
        U1 = self.get_enlace(*sitio, mu)

        # U_ν(x+μ)
        sitio_mu = sitio.copy()
        sitio_mu[mu] += 1
        U2 = self.get_enlace(*sitio_mu, nu)

        # U_μ†(x+ν)
        sitio_nu = sitio.copy()
        sitio_nu[nu] += 1
        U3_dag = self.get_enlace(*sitio_nu, mu).conj().T

        # U_ν†(x)
        U4_dag = self.get_enlace(*sitio, nu).conj().T

        return U1 @ U2 @ U3_dag @ U4_dag

    def calcular_plaqueta_promedio(self) -> float:
        """Calcula el promedio de plaquetas."""
        total = 0.0
        count = 0

        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for t in range(self.L):
                        for mu in range(4):
                            for nu in range(mu + 1, 4):
                                U_plaq = self.plaqueta(x, y, z, t, mu, nu)
                                total += promedio_plaqueta(U_plaq, self.N)
                                count += 1

        self.plaqueta_promedio = total / count
        return self.plaqueta_promedio

    def calcular_accion(self) -> float:
        """Calcula la acción total."""
        S_total = 0.0

        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for t in range(self.L):
                        for mu in range(4):
                            for nu in range(mu + 1, 4):
                                U_plaq = self.plaqueta(x, y, z, t, mu, nu)
                                S_total += accion_wilson(U_plaq, self.beta, self.N)

        return S_total

    def actualizar_metropolis(self, n_barridos: int = 1) -> float:
        """
        Realiza actualizaciones Metropolis.

        Args:
            n_barridos: Número de barridos

        Returns:
            Tasa de aceptación
        """
        aceptados = 0
        total = 0

        for _ in range(n_barridos):
            for x in range(self.L):
                for y in range(self.L):
                    for z in range(self.L):
                        for t in range(self.L):
                            for mu in range(4):
                                # Calcular staple
                                staple = self._calcular_staple(x, y, z, t, mu)

                                # Enlace actual
                                U_old = self.get_enlace(x, y, z, t, mu)
                                dS_old = -self.beta * np.real(np.trace(U_old @ staple)) / self.N

                                # Proponer nuevo enlace
                                U_new = su2_elemento_aleatorio(0.2) @ U_old
                                dS_new = -self.beta * np.real(np.trace(U_new @ staple)) / self.N

                                # Aceptar/rechazar
                                if np.random.random() < np.exp(dS_old - dS_new):
                                    self.set_enlace(x, y, z, t, mu, U_new)
                                    aceptados += 1

                                total += 1

        return aceptados / total if total > 0 else 0.0

    def _calcular_staple(self, x: int, y: int, z: int, t: int, mu: int) -> NDArray[np.complex128]:
        """Calcula el staple para el enlace (x,μ)."""
        staple = np.zeros((self.N, self.N), dtype=np.complex128)
        sitio = [x, y, z, t]

        for nu in range(4):
            if nu == mu:
                continue

            # Staple positivo
            sitio_mu = sitio.copy()
            sitio_mu[mu] += 1

            sitio_nu = sitio.copy()
            sitio_nu[nu] += 1

            U_nu_xmu = self.get_enlace(*sitio_mu, nu)
            U_mu_xnu_dag = self.get_enlace(*sitio_nu, mu).conj().T
            U_nu_x_dag = self.get_enlace(*sitio, nu).conj().T

            staple += U_nu_xmu @ U_mu_xnu_dag @ U_nu_x_dag

            # Staple negativo
            sitio_mnu = sitio.copy()
            sitio_mnu[nu] -= 1

            sitio_mu_mnu = sitio_mu.copy()
            sitio_mu_mnu[nu] -= 1

            U_nu_xmu_mnu_dag = self.get_enlace(*sitio_mu_mnu, nu).conj().T
            U_mu_xmnu_dag = self.get_enlace(*sitio_mnu, mu).conj().T
            U_nu_xmnu = self.get_enlace(*sitio_mnu, nu)

            staple += U_nu_xmu_mnu_dag @ U_mu_xmnu_dag @ U_nu_xmnu

        return staple

    def termalizar(self, n_barridos: int = 100) -> None:
        """Termaliza la configuración."""
        for i in range(n_barridos):
            acc = self.actualizar_metropolis(1)
            if (i + 1) % 10 == 0:
                P = self.calcular_plaqueta_promedio()
                print(f"Term {i+1}/{n_barridos}: <P> = {P:.6f}, acc = {acc:.3f}")

    def medir_mass_gap(self) -> float:
        """Estima el mass gap basado en el sello S actual."""
        return mass_gap(self.S)

    def resumen(self) -> str:
        """Genera resumen de la simulación."""
        P = self.calcular_plaqueta_promedio()
        S_val = self.calcular_accion()
        E_min = self.medir_mass_gap()

        return (
            f"Lattice Yang-Mills MCMC\n"
            f"{'='*50}\n"
            f"Tamaño: {self.L}⁴\n"
            f"Grupo: SU({self.N})\n"
            f"Sello S: {self.S:.3f}\n"
            f"β(S): {self.beta:.3f}\n"
            f"αs: {alpha_s(self.beta):.4f}\n"
            f"{'='*50}\n"
            f"Promedio plaqueta: {P:.6f}\n"
            f"Acción total: {S_val:.4f}\n"
            f"Mass gap estimado: {E_min:.4f} GeV\n"
        )


# =============================================================================
# Tests
# =============================================================================

def _test_lattice_ym():
    """Verifica la implementación de lattice YM."""

    # Test 1: β(S) es positivo
    for S in [0.0, 0.5, 1.0, 1.5]:
        b = beta_MCMC(S)
        assert b > 0, f"β({S}) = {b} debe ser > 0"

    # Test 2: β decrece con S (para S > S3)
    b1 = beta_MCMC(0.5)
    b2 = beta_MCMC(1.5)
    assert b1 > b2, "β debe decrecer cuando S aumenta"

    # Test 3: Mass gap positivo
    E = mass_gap(1.0)
    assert E > 0, f"Mass gap = {E} debe ser > 0"

    # Test 4: SU(2) elemento es unitario
    U = su2_elemento_aleatorio()
    assert np.allclose(U @ U.conj().T, np.eye(2)), "U debe ser unitario"
    assert np.isclose(np.linalg.det(U), 1.0), "det(U) = 1"

    # Test 5: Cold start tiene <P> = 1
    lattice = LatticeYM(L=2, S=1.0)
    P = lattice.calcular_plaqueta_promedio()
    assert np.isclose(P, 1.0), f"Cold start: <P> = {P}"

    # Test 6: Acción mínima para cold start
    S_cold = lattice.calcular_accion()
    assert S_cold < 0.1, f"Acción cold start debe ser ~0"

    print("✓ Todos los tests del Bloque 4 pasaron")
    return True


if __name__ == "__main__":
    _test_lattice_ym()

    # Demo
    print("\n" + "="*60)
    print("DEMO: Yang-Mills Lattice MCMC")
    print("Autor: Adrián Martínez Estellés (2024)")
    print("="*60 + "\n")

    # Crear lattice
    print("Creando lattice 4⁴...")
    lattice = LatticeYM(L=4, S=1.0)
    lattice._hot_start()

    print("\nAntes de termalizar:")
    print(f"  <P> = {lattice.calcular_plaqueta_promedio():.6f}")

    # Termalizar
    print("\nTermalizando...")
    lattice.termalizar(n_barridos=20)

    print("\n" + lattice.resumen())

    # Variación con S
    print("\nMass gap vs Sello S:")
    print("-"*40)
    for S in [0.0, 0.5, 1.0, 1.5, 2.0]:
        E = mass_gap(S)
        b = beta_MCMC(S)
        print(f"  S = {S:.1f}: β = {b:.3f}, E_min = {E:.4f} GeV")
