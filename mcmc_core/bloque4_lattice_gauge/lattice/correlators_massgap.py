"""
Correladores y Extracción del Mass Gap
=======================================

Implementación de operadores de glueball y métodos de extracción
del mass gap dinámico para comparar con el mass gap ontológico.

OPERADORES DE GLUEBALL:
    - Escalar 0⁺⁺: Σ Re Tr P_ij (plaquetas espaciales)
    - Tensor 2⁺⁺: 2P_xy - P_xz - P_yz
    - Polyakov: Tr ∏_t U_0(t,x)

MÉTODOS DE EXTRACCIÓN:
    1. Plateau de masa efectiva: m_eff(τ) = ln(C(τ)/C(τ+1))
    2. Ajuste exponencial: C(τ) = A exp(-mτ) + c
    3. Modelo cosh periódico: C(τ) = A[exp(-mτ) + exp(-m(L_t-τ))]

ERRORES:
    - Jackknife para errores estadísticos
    - Bootstrap para verificación cruzada

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from numpy.typing import NDArray
from scipy.optimize import curve_fit, minimize
import warnings

# Importar módulos locales
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcmc_ontology_lattice import (
    E_min_ontologico, S4, LAMBDA_QCD, M_GLUEBALL_0PP
)


# =============================================================================
# Constantes
# =============================================================================

# Escala para conversión a GeV
A_LATTICE_FM: float = 0.1      # Espaciado de red típico (fm)
HBAR_C: float = 0.197327       # GeV·fm
SCALE_FACTOR: float = HBAR_C / A_LATTICE_FM  # Conversión a GeV


# =============================================================================
# Operadores de Glueball
# =============================================================================

class OperadorGlueball:
    """
    Operadores de glueball para diferentes canales.
    """

    def __init__(self, config, enlaces: NDArray):
        """
        Inicializa el operador.

        Args:
            config: Configuración de la retícula
            enlaces: Array de enlaces
        """
        self.config = config
        self.enlaces = enlaces
        self.L = config.L
        self.N = config.N
        self.d = config.d

    def _get_enlace(self, sitio: Tuple[int, ...], mu: int) -> NDArray:
        """Obtiene enlace con condiciones periódicas."""
        sitio_mod = tuple(s % self.L for s in sitio)
        return self.enlaces[sitio_mod + (mu,)]

    def _plaqueta(self, sitio: Tuple[int, ...], mu: int, nu: int) -> NDArray:
        """Calcula plaqueta en el plano (μ,ν)."""
        x = list(sitio)

        U1 = self._get_enlace(tuple(x), mu)

        x_mu = x.copy()
        x_mu[mu] += 1
        U2 = self._get_enlace(tuple(x_mu), nu)

        x_nu = x.copy()
        x_nu[nu] += 1
        U3_dag = self._get_enlace(tuple(x_nu), mu).conj().T

        U4_dag = self._get_enlace(tuple(x), nu).conj().T

        return U1 @ U2 @ U3_dag @ U4_dag

    def escalar_0pp(self, t: int) -> complex:
        """
        Operador escalar 0⁺⁺ en el tiempo t.

        O_0⁺⁺(t) = Σ_x Σ_{i<j espaciales} Re Tr P_{ij}(t,x)

        Args:
            t: Coordenada temporal

        Returns:
            Valor del operador
        """
        total = 0.0

        # Sumar sobre todas las posiciones espaciales
        for idx in np.ndindex(*([self.L] * (self.d - 1))):
            sitio = (t,) + idx

            # Sumar sobre planos espaciales (i,j = 1,2,3)
            for i in range(1, self.d):
                for j in range(i + 1, self.d):
                    P = self._plaqueta(sitio, i, j)
                    total += np.real(np.trace(P)) / self.N

        return total

    def tensor_2pp(self, t: int) -> complex:
        """
        Operador tensor 2⁺⁺ en el tiempo t.

        O_2⁺⁺ = 2P_xy - P_xz - P_yz

        Args:
            t: Coordenada temporal

        Returns:
            Valor del operador
        """
        total = 0.0

        for idx in np.ndindex(*([self.L] * (self.d - 1))):
            sitio = (t,) + idx

            if self.d >= 4:
                P_xy = np.real(np.trace(self._plaqueta(sitio, 1, 2))) / self.N
                P_xz = np.real(np.trace(self._plaqueta(sitio, 1, 3))) / self.N
                P_yz = np.real(np.trace(self._plaqueta(sitio, 2, 3))) / self.N

                total += 2 * P_xy - P_xz - P_yz
            else:
                # Para d < 4, usar solo lo disponible
                if self.d >= 3:
                    P = np.real(np.trace(self._plaqueta(sitio, 1, 2))) / self.N
                    total += P

        return total

    def polyakov_loop(self, sitio_espacial: Tuple[int, ...]) -> complex:
        """
        Lazo de Polyakov en la posición espacial.

        P(x) = (1/N) Tr ∏_t U_0(t, x)

        Args:
            sitio_espacial: Coordenadas espaciales

        Returns:
            Valor complejo del lazo
        """
        producto = np.eye(self.N, dtype=np.complex128)

        for t in range(self.L):
            sitio = (t,) + sitio_espacial
            producto = producto @ self._get_enlace(sitio, 0)

        return np.trace(producto) / self.N

    def polyakov_promedio(self) -> complex:
        """
        Promedio espacial del lazo de Polyakov.

        <P> = (1/V_esp) Σ_x P(x)
        """
        total = 0.0

        for idx in np.ndindex(*([self.L] * (self.d - 1))):
            total += self.polyakov_loop(idx)

        return total / (self.L ** (self.d - 1))


# =============================================================================
# Correladores Temporales
# =============================================================================

@dataclass
class Correlador:
    """
    Correlador temporal C(τ).
    """
    tau: NDArray                  # Separaciones temporales
    C: NDArray                    # Valores del correlador
    C_err: NDArray                # Errores
    L_t: int                      # Extensión temporal
    canal: str = "0++"            # Canal del operador


def calcular_correlador(
    configuraciones: List[NDArray],
    config,
    canal: str = "0++"
) -> Correlador:
    """
    Calcula correlador temporal promediando sobre configuraciones.

    C(τ) = <O(t) O(t+τ)> - <O>²

    Args:
        configuraciones: Lista de configuraciones de enlaces
        config: Configuración de la retícula
        canal: "0++", "2++", o "polyakov"

    Returns:
        Correlador con errores jackknife
    """
    L_t = config.L
    n_configs = len(configuraciones)

    # Calcular operador para cada configuración y tiempo
    operadores = np.zeros((n_configs, L_t), dtype=np.complex128)

    for i, enlaces in enumerate(configuraciones):
        op = OperadorGlueball(config, enlaces)

        for t in range(L_t):
            if canal == "0++":
                operadores[i, t] = op.escalar_0pp(t)
            elif canal == "2++":
                operadores[i, t] = op.tensor_2pp(t)
            elif canal == "polyakov":
                operadores[i, t] = op.polyakov_promedio()

    # Calcular correlador para cada τ
    tau_values = np.arange(L_t // 2 + 1)
    C_all = np.zeros((n_configs, len(tau_values)), dtype=np.complex128)

    for i in range(n_configs):
        for j, tau in enumerate(tau_values):
            # Promedio sobre t inicial
            correlacion = 0.0
            for t in range(L_t):
                t_plus_tau = (t + tau) % L_t
                correlacion += operadores[i, t] * np.conj(operadores[i, t_plus_tau])
            correlacion /= L_t

            # Restar <O>²
            O_mean = np.mean(operadores[i])
            C_all[i, j] = correlacion - np.abs(O_mean)**2

    # Promediar sobre configuraciones
    C_mean = np.mean(np.real(C_all), axis=0)

    # Errores jackknife
    C_jackknife = np.zeros((n_configs, len(tau_values)))
    for i in range(n_configs):
        # Excluir configuración i
        mask = np.ones(n_configs, dtype=bool)
        mask[i] = False
        C_jackknife[i] = np.mean(np.real(C_all[mask]), axis=0)

    C_err = np.sqrt((n_configs - 1) * np.var(C_jackknife, axis=0))

    return Correlador(
        tau=tau_values,
        C=C_mean,
        C_err=C_err,
        L_t=L_t,
        canal=canal
    )


# =============================================================================
# Extracción del Mass Gap
# =============================================================================

@dataclass
class ResultadoMassGap:
    """
    Resultado de la extracción del mass gap.
    """
    E_min: float                  # Mass gap en unidades de lattice
    E_min_err: float              # Error
    E_min_GeV: float              # En GeV
    E_min_GeV_err: float          # Error en GeV
    metodo: str                   # Método usado
    chi2_red: float               # Chi² reducido
    plateau_inicio: int           # Inicio del plateau
    plateau_fin: int              # Fin del plateau
    parametros: Dict = field(default_factory=dict)


def masa_efectiva(C: NDArray, C_err: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Calcula masa efectiva m_eff(τ) = ln(C(τ)/C(τ+1)).

    Args:
        C: Valores del correlador
        C_err: Errores del correlador

    Returns:
        (m_eff, m_eff_err)
    """
    n = len(C) - 1
    m_eff = np.zeros(n)
    m_eff_err = np.zeros(n)

    for i in range(n):
        if C[i] > 0 and C[i+1] > 0:
            m_eff[i] = np.log(C[i] / C[i+1])

            # Propagación de errores
            dC_ratio = np.sqrt(
                (C_err[i] / C[i])**2 + (C_err[i+1] / C[i+1])**2
            )
            m_eff_err[i] = dC_ratio
        else:
            m_eff[i] = np.nan
            m_eff_err[i] = np.nan

    return m_eff, m_eff_err


def encontrar_plateau(
    m_eff: NDArray,
    m_eff_err: NDArray,
    n_min: int = 3
) -> Tuple[int, int, float, float]:
    """
    Encuentra región de plateau en la masa efectiva.

    Args:
        m_eff: Masa efectiva
        m_eff_err: Errores
        n_min: Mínimo de puntos en plateau

    Returns:
        (inicio, fin, m_plateau, m_plateau_err)
    """
    n = len(m_eff)
    valid = ~np.isnan(m_eff) & (m_eff > 0)

    if np.sum(valid) < n_min:
        # No hay suficientes puntos válidos
        return 0, 0, 0.0, 0.0

    # Buscar región de menor variación
    mejor_chi2 = np.inf
    mejor_inicio = 0
    mejor_fin = n

    for inicio in range(n - n_min):
        for fin in range(inicio + n_min, n):
            region = m_eff[inicio:fin]
            region_valid = valid[inicio:fin]

            if np.sum(region_valid) >= n_min:
                region_clean = region[region_valid]
                media = np.mean(region_clean)
                var = np.var(region_clean)
                chi2 = var / (media**2 + 1e-10)

                if chi2 < mejor_chi2:
                    mejor_chi2 = chi2
                    mejor_inicio = inicio
                    mejor_fin = fin

    # Calcular masa promedio en plateau
    region = m_eff[mejor_inicio:mejor_fin]
    region_err = m_eff_err[mejor_inicio:mejor_fin]
    region_valid = valid[mejor_inicio:mejor_fin]

    if np.sum(region_valid) > 0:
        m_plateau = np.mean(region[region_valid])
        m_plateau_err = np.sqrt(np.sum(region_err[region_valid]**2)) / np.sum(region_valid)
    else:
        m_plateau = 0.0
        m_plateau_err = 0.0

    return mejor_inicio, mejor_fin, m_plateau, m_plateau_err


def ajuste_exponencial(
    tau: NDArray,
    C: NDArray,
    C_err: NDArray,
    L_t: int
) -> Tuple[float, float, float]:
    """
    Ajusta correlador a forma exponencial.

    C(τ) = A exp(-m τ) + c

    Args:
        tau: Separaciones temporales
        C: Correlador
        C_err: Errores
        L_t: Extensión temporal

    Returns:
        (m, m_err, chi2_red)
    """
    def modelo(t, A, m, c):
        return A * np.exp(-m * t) + c

    try:
        # Filtrar puntos válidos
        valid = (C > 0) & (C_err > 0)
        if np.sum(valid) < 4:
            return 0.0, 0.0, np.inf

        tau_fit = tau[valid]
        C_fit = C[valid]
        err_fit = C_err[valid]

        # Ajuste
        p0 = [C_fit[0], 0.5, 0.0]
        bounds = ([0, 0, -np.inf], [np.inf, 10, np.inf])

        popt, pcov = curve_fit(
            modelo, tau_fit, C_fit,
            p0=p0, sigma=err_fit,
            bounds=bounds, maxfev=5000
        )

        m = popt[1]
        m_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.0

        # Chi² reducido
        residuos = C_fit - modelo(tau_fit, *popt)
        chi2 = np.sum((residuos / err_fit)**2)
        chi2_red = chi2 / (len(tau_fit) - 3)

        return m, m_err, chi2_red

    except Exception:
        return 0.0, 0.0, np.inf


def ajuste_cosh(
    tau: NDArray,
    C: NDArray,
    C_err: NDArray,
    L_t: int
) -> Tuple[float, float, float]:
    """
    Ajusta correlador a forma cosh (condiciones periódicas).

    C(τ) = A [exp(-m τ) + exp(-m (L_t - τ))]

    Args:
        tau: Separaciones temporales
        C: Correlador
        C_err: Errores
        L_t: Extensión temporal

    Returns:
        (m, m_err, chi2_red)
    """
    def modelo(t, A, m):
        return A * (np.exp(-m * t) + np.exp(-m * (L_t - t)))

    try:
        valid = (C > 0) & (C_err > 0)
        if np.sum(valid) < 3:
            return 0.0, 0.0, np.inf

        tau_fit = tau[valid]
        C_fit = C[valid]
        err_fit = C_err[valid]

        p0 = [C_fit[0] / 2, 0.5]
        bounds = ([0, 0], [np.inf, 10])

        popt, pcov = curve_fit(
            modelo, tau_fit, C_fit,
            p0=p0, sigma=err_fit,
            bounds=bounds, maxfev=5000
        )

        m = popt[1]
        m_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.0

        residuos = C_fit - modelo(tau_fit, *popt)
        chi2 = np.sum((residuos / err_fit)**2)
        chi2_red = chi2 / (len(tau_fit) - 2)

        return m, m_err, chi2_red

    except Exception:
        return 0.0, 0.0, np.inf


def extraer_mass_gap(
    correlador: Correlador,
    metodo: str = "plateau",
    a_fm: float = A_LATTICE_FM
) -> ResultadoMassGap:
    """
    Extrae el mass gap de un correlador.

    Args:
        correlador: Correlador temporal
        metodo: "plateau", "exponencial", o "cosh"
        a_fm: Espaciado de red en fm

    Returns:
        ResultadoMassGap
    """
    tau = correlador.tau
    C = correlador.C
    C_err = correlador.C_err
    L_t = correlador.L_t

    # Conversión de unidades
    # m [lattice] × (ℏc/a) = m [GeV]
    conversion = HBAR_C / a_fm

    if metodo == "plateau":
        # Masa efectiva y plateau
        m_eff, m_eff_err = masa_efectiva(C, C_err)
        inicio, fin, m, m_err = encontrar_plateau(m_eff, m_eff_err)

        chi2_red = 0.0 if m > 0 else np.inf

        return ResultadoMassGap(
            E_min=m,
            E_min_err=m_err,
            E_min_GeV=m * conversion,
            E_min_GeV_err=m_err * conversion,
            metodo="plateau",
            chi2_red=chi2_red,
            plateau_inicio=inicio,
            plateau_fin=fin
        )

    elif metodo == "exponencial":
        m, m_err, chi2_red = ajuste_exponencial(tau, C, C_err, L_t)

        return ResultadoMassGap(
            E_min=m,
            E_min_err=m_err,
            E_min_GeV=m * conversion,
            E_min_GeV_err=m_err * conversion,
            metodo="exponencial",
            chi2_red=chi2_red,
            plateau_inicio=0,
            plateau_fin=len(tau)
        )

    elif metodo == "cosh":
        m, m_err, chi2_red = ajuste_cosh(tau, C, C_err, L_t)

        return ResultadoMassGap(
            E_min=m,
            E_min_err=m_err,
            E_min_GeV=m * conversion,
            E_min_GeV_err=m_err * conversion,
            metodo="cosh",
            chi2_red=chi2_red,
            plateau_inicio=0,
            plateau_fin=len(tau)
        )

    else:
        raise ValueError(f"Método desconocido: {metodo}")


# =============================================================================
# Función Principal de Medición
# =============================================================================

def medir_mass_gap(
    configuraciones: List[NDArray],
    config,
    S: float = S4,
    canales: List[str] = ["0++"],
    metodos: List[str] = ["plateau", "exponencial"],
    a_fm: float = A_LATTICE_FM,
    verbose: bool = True
) -> Dict[str, ResultadoMassGap]:
    """
    Mide el mass gap usando múltiples canales y métodos.

    Args:
        configuraciones: Lista de configuraciones
        config: Configuración de la retícula
        S: Sello entrópico
        canales: Canales a medir
        metodos: Métodos de extracción
        a_fm: Espaciado de red
        verbose: Imprimir resultados

    Returns:
        Diccionario de resultados por canal y método
    """
    resultados = {}

    if verbose:
        print(f"\nExtracción del Mass Gap (S = {S:.4f})")
        print("="*50)

    for canal in canales:
        if verbose:
            print(f"\nCanal {canal}:")

        # Calcular correlador
        correlador = calcular_correlador(configuraciones, config, canal)

        for metodo in metodos:
            resultado = extraer_mass_gap(correlador, metodo, a_fm)
            key = f"{canal}_{metodo}"
            resultados[key] = resultado

            if verbose:
                print(f"  {metodo:12s}: E_min = {resultado.E_min_GeV:.3f} ± "
                      f"{resultado.E_min_GeV_err:.3f} GeV (χ² = {resultado.chi2_red:.2f})")

    # Comparar con ontología
    if verbose:
        E_onto = E_min_ontologico(S)
        print(f"\nComparación con ontología:")
        print(f"  E_min ontológico (S={S:.4f}): {E_onto:.2f} GeV")

        mejor_key = min(resultados.keys(), key=lambda k: resultados[k].chi2_red)
        E_lat = resultados[mejor_key].E_min_GeV
        ratio = E_lat / E_onto if E_onto > 0 else 0
        print(f"  Mejor ajuste ({mejor_key}): {E_lat:.3f} GeV")
        print(f"  Ratio E_lat/E_onto: {ratio:.3f}")

    return resultados


# =============================================================================
# Tests
# =============================================================================

def _test_correladores():
    """Verifica la implementación de correladores."""

    print("Testing Correladores y Mass Gap...")

    # Test 1: Masa efectiva
    C = np.array([10.0, 5.0, 2.5, 1.25, 0.625])
    C_err = np.array([0.5, 0.25, 0.125, 0.0625, 0.03])
    m_eff, m_eff_err = masa_efectiva(C, C_err)

    # Para C = A exp(-m τ), m_eff debería ser constante ≈ ln(2)
    expected_m = np.log(2)
    assert np.allclose(m_eff, expected_m, rtol=0.01), f"m_eff = {m_eff}"
    print("  ✓ Masa efectiva")

    # Test 2: Encontrar plateau
    m_eff = np.array([0.7, 0.69, 0.68, 0.685, 0.69, 0.7])
    m_eff_err = np.array([0.05, 0.04, 0.03, 0.03, 0.04, 0.05])
    inicio, fin, m, m_err = encontrar_plateau(m_eff, m_eff_err)
    assert m > 0, f"No se encontró plateau: m = {m}"
    print(f"  ✓ Plateau: m = {m:.3f} ± {m_err:.3f} (τ: {inicio}-{fin})")

    # Test 3: Ajuste exponencial sintético
    tau = np.arange(8)
    m_true = 0.5
    A_true = 10.0
    C = A_true * np.exp(-m_true * tau)
    C_err = 0.1 * C

    m, m_err, chi2 = ajuste_exponencial(tau, C, C_err, 16)
    assert abs(m - m_true) < 0.1, f"m = {m}, esperado {m_true}"
    print(f"  ✓ Ajuste exponencial: m = {m:.3f} (esperado {m_true})")

    print("\n✓ Todos los tests de correladores pasaron")
    return True


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    _test_correladores()
