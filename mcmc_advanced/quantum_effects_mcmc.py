#!/usr/bin/env python3
"""
================================================================================
EFECTOS CUANTICOS MCMC - QUBIT TENSORIAL
================================================================================

Implementa los efectos cuanticos del MCMC basados en la teoria del
Qubit Tensorial, que describe la estructura cuantica fundamental del
espacio-tiempo como una red de qubits entrelazados.

FUNDAMENTACION ONTOLOGICA:
--------------------------
El Qubit Tensorial propone:
1. El espacio-tiempo emerge de una red de qubits cuanticos
2. El entrelazamiento cuantico genera la conectividad espacial
3. La entropia S esta relacionada con los grados de libertad de qubits
4. Las correlaciones tensoriales codifican la geometria

ECUACIONES FUNDAMENTALES:
-------------------------
(Ec. Q1) |Psi> = sum_i alpha_i |q_i> (Estado del universo)
(Ec. Q2) S = -Tr(rho * log(rho)) (Entropia de von Neumann)
(Ec. Q3) E_ent = integral d^3x * T_ij * g^ij (Energia de entrelazamiento)
(Ec. Q4) C_AB = <A B> - <A><B> (Correlaciones cuanticas)

PREDICCIONES CUANTICAS:
-----------------------
- Decoherencia a escala macroscopica
- Correlatores cuanticos en CMB
- Efectos de tunelamiento cosmico

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.linalg import expm, logm, sqrtm
from scipy.integrate import quad, dblquad
from scipy.special import factorial
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES CUANTICAS
# =============================================================================

# Constantes fundamentales
HBAR = 1.054571817e-34       # J*s
C = 299792458                # m/s
K_B = 1.380649e-23           # J/K
L_PLANCK = 1.616255e-35      # m
T_PLANCK = 5.391247e-44      # s
E_PLANCK = 1.956086e9        # J

# Parametros del modelo Qubit Tensorial
N_QUBITS_UNIVERSO = 10**122  # Numero estimado de qubits (~ exp(S_Bekenstein))
D_HILBERT = 2                # Dimension de un qubit individual
EPSILON_DECOHER = 1e-40      # Tasa de decoherencia base [s^-1]


# =============================================================================
# ESTADOS CUANTICOS BASICOS
# =============================================================================

# Estados de qubit
ESTADO_0 = np.array([1, 0], dtype=complex)
ESTADO_1 = np.array([0, 1], dtype=complex)
ESTADO_PLUS = (ESTADO_0 + ESTADO_1) / np.sqrt(2)
ESTADO_MINUS = (ESTADO_0 - ESTADO_1) / np.sqrt(2)

# Matrices de Pauli
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTIDAD = np.array([[1, 0], [0, 1]], dtype=complex)


# =============================================================================
# QUBIT CUANTICO
# =============================================================================

@dataclass
class QubitCosmico:
    """
    Representa un qubit cosmico fundamental.

    Cada qubit es un grado de libertad cuantico que contribuye
    a la estructura del espacio-tiempo.
    """
    alpha: complex = 1.0       # Amplitud de |0>
    beta: complex = 0.0        # Amplitud de |1>

    def __post_init__(self):
        """Normaliza el estado."""
        norm = np.sqrt(abs(self.alpha)**2 + abs(self.beta)**2)
        if norm > 0:
            self.alpha /= norm
            self.beta /= norm

    @property
    def estado(self) -> np.ndarray:
        """Vector de estado."""
        return np.array([self.alpha, self.beta], dtype=complex)

    @property
    def matriz_densidad(self) -> np.ndarray:
        """Matriz densidad |psi><psi|."""
        psi = self.estado.reshape(-1, 1)
        return psi @ psi.conj().T

    def probabilidad_0(self) -> float:
        """Probabilidad de medir |0>."""
        return abs(self.alpha)**2

    def probabilidad_1(self) -> float:
        """Probabilidad de medir |1>."""
        return abs(self.beta)**2

    def entropia_von_neumann(self) -> float:
        """
        Entropia de von Neumann (0 para estados puros).

        S = -Tr(rho * log(rho))
        """
        # Para estado puro, S = 0
        return 0.0

    def aplicar_operador(self, U: np.ndarray) -> 'QubitCosmico':
        """
        Aplica operador unitario al qubit.

        Args:
            U: Matriz unitaria 2x2

        Returns:
            Nuevo QubitCosmico
        """
        nuevo_estado = U @ self.estado
        return QubitCosmico(nuevo_estado[0], nuevo_estado[1])


# =============================================================================
# ENTRELAZAMIENTO CUANTICO
# =============================================================================

class EntrelazamientoCuantico:
    """
    Modela el entrelazamiento cuantico entre qubits cosmicos.

    El entrelazamiento es fundamental para la emergencia
    del espacio-tiempo en el Qubit Tensorial.
    """

    def __init__(self, n_qubits: int = 2):
        """
        Inicializa sistema entrelazado.

        Args:
            n_qubits: Numero de qubits (2 por defecto)
        """
        self.n_qubits = n_qubits
        self.dim = D_HILBERT ** n_qubits

        # Estado inicial: producto |00...0>
        self.estado = np.zeros(self.dim, dtype=complex)
        self.estado[0] = 1.0

    def crear_bell_state(self, tipo: str = 'phi_plus') -> np.ndarray:
        """
        Crea un estado de Bell.

        |Phi+> = (|00> + |11>) / sqrt(2)
        |Phi-> = (|00> - |11>) / sqrt(2)
        |Psi+> = (|01> + |10>) / sqrt(2)
        |Psi-> = (|01> - |10>) / sqrt(2)

        Args:
            tipo: 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'

        Returns:
            Vector de estado
        """
        if tipo == 'phi_plus':
            estado = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        elif tipo == 'phi_minus':
            estado = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
        elif tipo == 'psi_plus':
            estado = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
        elif tipo == 'psi_minus':
            estado = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
        else:
            estado = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        self.estado = estado
        return estado

    def matriz_densidad(self) -> np.ndarray:
        """Matriz densidad del sistema."""
        psi = self.estado.reshape(-1, 1)
        return psi @ psi.conj().T

    def matriz_densidad_reducida(self, subsistema: int = 0) -> np.ndarray:
        """
        Matriz densidad reducida de un subsistema.

        rho_A = Tr_B(rho_AB)

        Args:
            subsistema: 0 o 1 (primer o segundo qubit)

        Returns:
            Matriz densidad 2x2
        """
        rho = self.matriz_densidad()

        if self.n_qubits != 2:
            raise ValueError("Solo implementado para 2 qubits")

        rho_reduced = np.zeros((2, 2), dtype=complex)

        if subsistema == 0:
            # Traza sobre segundo qubit
            rho_reduced[0, 0] = rho[0, 0] + rho[1, 1]
            rho_reduced[0, 1] = rho[0, 2] + rho[1, 3]
            rho_reduced[1, 0] = rho[2, 0] + rho[3, 1]
            rho_reduced[1, 1] = rho[2, 2] + rho[3, 3]
        else:
            # Traza sobre primer qubit
            rho_reduced[0, 0] = rho[0, 0] + rho[2, 2]
            rho_reduced[0, 1] = rho[0, 1] + rho[2, 3]
            rho_reduced[1, 0] = rho[1, 0] + rho[3, 2]
            rho_reduced[1, 1] = rho[1, 1] + rho[3, 3]

        return rho_reduced

    def entropia_entrelazamiento(self) -> float:
        """
        Entropia de entrelazamiento (von Neumann de subsistema).

        S_ent = -Tr(rho_A * log(rho_A))

        Returns:
            Entropia de entrelazamiento [0, log(2)]
        """
        rho_A = self.matriz_densidad_reducida(0)

        # Valores propios
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]

        if len(eigenvalues) == 0:
            return 0.0

        return -np.sum(eigenvalues * np.log(eigenvalues))

    def concurrencia(self) -> float:
        """
        Concurrencia (medida de entrelazamiento).

        C = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)

        Returns:
            Concurrencia [0, 1]
        """
        rho = self.matriz_densidad()

        # Matriz sigma_y tensor sigma_y
        Y = np.array([[0, 0, 0, -1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [-1, 0, 0, 0]], dtype=complex)

        # rho_tilde = Y * rho* * Y
        rho_tilde = Y @ np.conj(rho) @ Y

        # R = sqrt(sqrt(rho) * rho_tilde * sqrt(rho))
        sqrt_rho = sqrtm(rho)
        R = sqrtm(sqrt_rho @ rho_tilde @ sqrt_rho)

        # Valores propios de R
        eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]

        concurrencia = max(0, eigenvalues[0] - eigenvalues[1] -
                          eigenvalues[2] - eigenvalues[3])

        return concurrencia


# =============================================================================
# TENSOR DE CORRELACION CUANTICA
# =============================================================================

class TensorCorrelacion:
    """
    Tensor de correlaciones cuanticas del espacio-tiempo.

    En el Qubit Tensorial, la geometria emerge de las correlaciones
    entre qubits: g_ij ~ <sigma_i sigma_j> - <sigma_i><sigma_j>
    """

    def __init__(self, n_qubits: int = 4):
        """
        Inicializa tensor de correlacion.

        Args:
            n_qubits: Numero de qubits en la red
        """
        self.n_qubits = n_qubits
        self.correlaciones = np.zeros((n_qubits, n_qubits))

    def calcular_correlacion(self, rho: np.ndarray, i: int, j: int,
                             op_i: np.ndarray = SIGMA_Z,
                             op_j: np.ndarray = SIGMA_Z) -> float:
        """
        Calcula correlacion cuantica C_ij = <A_i B_j> - <A_i><B_j>.

        Args:
            rho: Matriz densidad del sistema
            i, j: Indices de qubits
            op_i, op_j: Operadores locales

        Returns:
            Correlacion C_ij
        """
        # Simplificacion para 2 qubits
        if self.n_qubits != 2:
            return 0.0

        # <A_i B_j>
        AB = np.kron(op_i, op_j)
        expectation_AB = np.real(np.trace(rho @ AB))

        # <A_i> y <B_j>
        A_ext = np.kron(op_i, IDENTIDAD)
        B_ext = np.kron(IDENTIDAD, op_j)

        expectation_A = np.real(np.trace(rho @ A_ext))
        expectation_B = np.real(np.trace(rho @ B_ext))

        return expectation_AB - expectation_A * expectation_B

    def tensor_metrico_emergente(self, entrelazamiento: EntrelazamientoCuantico) -> np.ndarray:
        """
        Calcula tensor metrico emergente de las correlaciones.

        g_ij ~ correlacion(sigma_i, sigma_j)

        Args:
            entrelazamiento: Sistema de qubits entrelazados

        Returns:
            Tensor metrico 2x2 (simplificado)
        """
        rho = entrelazamiento.matriz_densidad()

        # Componentes usando diferentes operadores de Pauli
        g = np.zeros((3, 3))

        operadores = [SIGMA_X, SIGMA_Y, SIGMA_Z]

        for i, op_i in enumerate(operadores):
            for j, op_j in enumerate(operadores):
                g[i, j] = self.calcular_correlacion(rho, 0, 1, op_i, op_j)

        return g


# =============================================================================
# DECOHERENCIA COSMICA
# =============================================================================

class DecoherenciaCosmica:
    """
    Modela la decoherencia cuantica a escala cosmica.

    La decoherencia explica la transicion cuantico -> clasico
    en el universo temprano.
    """

    def __init__(self, gamma_0: float = EPSILON_DECOHER):
        """
        Inicializa modelo de decoherencia.

        Args:
            gamma_0: Tasa de decoherencia base [s^-1]
        """
        self.gamma_0 = gamma_0

    def tasa_decoherencia(self, T: float, M: float) -> float:
        """
        Tasa de decoherencia dependiente de temperatura y masa.

        Gamma ~ gamma_0 * (T/T_Planck)^3 * (M/M_Planck)^2

        Args:
            T: Temperatura [K]
            M: Masa caracteristica [kg]

        Returns:
            Gamma [s^-1]
        """
        T_Pl = E_PLANCK / K_B  # ~1.4e32 K
        M_Pl = np.sqrt(HBAR * C / (6.674e-11))  # ~2.2e-8 kg

        return self.gamma_0 * (T / T_Pl)**3 * (M / M_Pl)**2

    def tiempo_decoherencia(self, T: float, M: float) -> float:
        """
        Tiempo de decoherencia.

        t_dec = 1 / Gamma

        Args:
            T: Temperatura [K]
            M: Masa [kg]

        Returns:
            t_dec [s]
        """
        gamma = self.tasa_decoherencia(T, M)
        if gamma <= 0:
            return np.inf
        return 1 / gamma

    def evolucion_coherencia(self, t: float, gamma: float) -> float:
        """
        Evolucion temporal de la coherencia.

        C(t) = C_0 * exp(-gamma * t)

        Args:
            t: Tiempo [s]
            gamma: Tasa de decoherencia [s^-1]

        Returns:
            Coherencia relativa [0, 1]
        """
        return np.exp(-gamma * t)

    def escala_clasica(self, T: float) -> float:
        """
        Escala espacial donde el sistema se vuelve clasico.

        L_cl ~ sqrt(hbar * t_dec / M)

        Args:
            T: Temperatura [K]

        Returns:
            L_cl [m]
        """
        M = 1e-20  # Masa tipica de sistema mesoscopico [kg]
        t_dec = self.tiempo_decoherencia(T, M)

        return np.sqrt(HBAR * t_dec / M)


# =============================================================================
# EFECTOS CUANTICOS EN CMB
# =============================================================================

class EfectosCuanticosCMB:
    """
    Efectos cuanticos observables en el CMB.

    El Qubit Tensorial predice sutiles desviaciones de la
    estadistica gaussiana debido a correlaciones cuanticas primordiales.
    """

    def __init__(self):
        """Inicializa calculador de efectos CMB."""
        self.T_CMB = 2.725  # K

    def correlacion_primordial(self, k1: float, k2: float) -> float:
        """
        Correlacion cuantica primordial en espacio de momentos.

        <delta_k1 delta_k2> con contribucion cuantica extra

        Args:
            k1, k2: Numeros de onda [Mpc^-1]

        Returns:
            Correlacion
        """
        # Espectro de potencia base (aproximacion)
        k_pivot = 0.05  # Mpc^-1
        A_s = 2.1e-9
        n_s = 0.9649

        P_k1 = A_s * (k1 / k_pivot)**(n_s - 1)
        P_k2 = A_s * (k2 / k_pivot)**(n_s - 1)

        # Correlacion cuantica extra (modelo simplificado)
        xi_q = 1e-6  # Parametro de correlacion cuantica
        delta_k = abs(k1 - k2)
        corr_cuantica = xi_q * np.exp(-delta_k / k_pivot)

        return np.sqrt(P_k1 * P_k2) + corr_cuantica

    def no_gaussianidad(self, k: float) -> float:
        """
        Parametro de no-gaussianidad f_NL efectivo.

        El Qubit Tensorial predice f_NL pequeno pero no nulo.

        Args:
            k: Numero de onda [Mpc^-1]

        Returns:
            f_NL(k)
        """
        # Prediccion del modelo
        f_NL_0 = 0.5  # Valor central predicho
        k_pivot = 0.05

        # Dependencia en escala (running)
        return f_NL_0 * (1 + 0.01 * np.log(k / k_pivot))

    def espectro_B_modes(self, l: int) -> float:
        """
        Espectro de modos B con contribucion cuantica.

        C_l^BB = C_l^{BB,tensor} + C_l^{BB,quantum}

        Args:
            l: Multipolo

        Returns:
            C_l^BB
        """
        # Contribucion tensorial estandar
        r = 0.004  # Ratio tensor/escalar (prediccion MCMC)
        l_pivot = 80

        C_tensor = r * 1e-10 * (l / l_pivot)**(-0.5)

        # Contribucion cuantica adicional
        C_quantum = 1e-14 * (l / 100)**(-2)

        return C_tensor + C_quantum


# =============================================================================
# TUNELAMIENTO COSMICO
# =============================================================================

class TunelamientoCosmico:
    """
    Modela efectos de tunelamiento cuantico a escala cosmica.

    El tunelamiento puede ser relevante para:
    - Transiciones de fase de vacio
    - Nucleacion de burbujas
    - Inflacion eterna
    """

    def __init__(self):
        """Inicializa modelo de tunelamiento."""
        pass

    def probabilidad_tunelamiento(self, V_barrera: float, E: float,
                                   masa_eff: float, ancho: float) -> float:
        """
        Probabilidad de tunelamiento cuantico.

        P ~ exp(-2 * kappa * ancho)
        kappa = sqrt(2*m*(V-E)) / hbar

        Args:
            V_barrera: Altura de barrera [J]
            E: Energia del sistema [J]
            masa_eff: Masa efectiva [kg]
            ancho: Ancho de la barrera [m]

        Returns:
            Probabilidad de tunelamiento [0, 1]
        """
        if E >= V_barrera:
            return 1.0

        kappa = np.sqrt(2 * masa_eff * (V_barrera - E)) / HBAR

        # Limitar exponente para evitar underflow
        exponente = min(2 * kappa * ancho, 700)

        return np.exp(-exponente)

    def tasa_nucleacion_burbuja(self, S_E: float, T: float) -> float:
        """
        Tasa de nucleacion de burbujas de vacio verdadero.

        Gamma ~ T^4 * exp(-S_E / T)

        donde S_E es la accion euclidea del instanton.

        Args:
            S_E: Accion del instanton [GeV]
            T: Temperatura [GeV]

        Returns:
            Tasa de nucleacion [GeV^4]
        """
        if T <= 0:
            return 0.0

        return T**4 * np.exp(-S_E / T)


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_Quantum_Effects_MCMC() -> bool:
    """
    Test del modulo de efectos cuanticos.
    """
    print("\n" + "=" * 70)
    print("  TEST QUANTUM EFFECTS MCMC - QUBIT TENSORIAL")
    print("=" * 70)

    # 1. Qubit cosmico
    print("\n[1] Qubit cosmico basico:")
    print("-" * 70)

    q = QubitCosmico(alpha=1/np.sqrt(2), beta=1/np.sqrt(2))
    print(f"    Estado: |psi> = {q.alpha:.3f}|0> + {q.beta:.3f}|1>")
    print(f"    P(|0>) = {q.probabilidad_0():.4f}")
    print(f"    P(|1>) = {q.probabilidad_1():.4f}")
    print(f"    S_vN = {q.entropia_von_neumann():.4f}")

    qubit_ok = abs(q.probabilidad_0() + q.probabilidad_1() - 1.0) < 1e-10
    print(f"\n    Normalizacion correcta: {'PASS' if qubit_ok else 'FAIL'}")

    # 2. Entrelazamiento
    print("\n[2] Entrelazamiento cuantico:")
    print("-" * 70)

    ent = EntrelazamientoCuantico(n_qubits=2)
    ent.crear_bell_state('phi_plus')

    S_ent = ent.entropia_entrelazamiento()
    C = ent.concurrencia()

    print(f"    Estado de Bell: |Phi+> = (|00> + |11>)/sqrt(2)")
    print(f"    S_entrelazamiento = {S_ent:.4f} (max = ln(2) = {np.log(2):.4f})")
    print(f"    Concurrencia = {C:.4f} (max = 1)")

    # Bell state debe tener maximal entanglement
    ent_ok = S_ent > 0.6 and C > 0.9
    print(f"\n    Entrelazamiento maximo: {'PASS' if ent_ok else 'FAIL'}")

    # 3. Tensor de correlacion
    print("\n[3] Tensor de correlacion cuantica:")
    print("-" * 70)

    tensor = TensorCorrelacion(n_qubits=2)
    g = tensor.tensor_metrico_emergente(ent)

    print("    Componentes g_ij (correlaciones Pauli):")
    print(f"    g_xx = {g[0,0]:>8.4f}   g_xy = {g[0,1]:>8.4f}   g_xz = {g[0,2]:>8.4f}")
    print(f"    g_yx = {g[1,0]:>8.4f}   g_yy = {g[1,1]:>8.4f}   g_yz = {g[1,2]:>8.4f}")
    print(f"    g_zx = {g[2,0]:>8.4f}   g_zy = {g[2,1]:>8.4f}   g_zz = {g[2,2]:>8.4f}")

    # Para Phi+, correlacion zz debe ser ~1
    tensor_ok = abs(g[2, 2] - 1.0) < 0.1
    print(f"\n    Correlacion ZZ ~ 1: {'PASS' if tensor_ok else 'FAIL'}")

    # 4. Decoherencia cosmica
    print("\n[4] Decoherencia cosmica:")
    print("-" * 70)

    decoh = DecoherenciaCosmica()

    T_test = [3000, 1e10, 1e20]  # K
    M_test = 1e-20  # kg

    print(f"    {'T [K]':>12} {'t_dec [s]':>15} {'Coherencia(t=1s)':>15}")
    for T in T_test:
        gamma = decoh.tasa_decoherencia(T, M_test)
        t_dec = decoh.tiempo_decoherencia(T, M_test)
        coh = decoh.evolucion_coherencia(1.0, gamma)
        print(f"    {T:>12.2e} {t_dec:>15.2e} {coh:>15.6f}")

    # Alta T debe dar decoherencia rapida
    decoh_ok = decoh.tiempo_decoherencia(1e20, M_test) < decoh.tiempo_decoherencia(3000, M_test)
    print(f"\n    T mayor -> decoherencia mas rapida: {'PASS' if decoh_ok else 'FAIL'}")

    # 5. Efectos cuanticos en CMB
    print("\n[5] Efectos cuanticos en CMB:")
    print("-" * 70)

    cmb = EfectosCuanticosCMB()

    k_test = [0.01, 0.05, 0.1]
    print(f"    {'k [Mpc^-1]':>12} {'f_NL':>10} {'C_l^BB (l=100)':>15}")
    for k in k_test:
        f_NL = cmb.no_gaussianidad(k)
        C_BB = cmb.espectro_B_modes(100)
        print(f"    {k:>12.3f} {f_NL:>10.4f} {C_BB:>15.2e}")

    # f_NL debe ser pequeno
    cmb_ok = abs(cmb.no_gaussianidad(0.05)) < 10
    print(f"\n    f_NL pequeno (< 10): {'PASS' if cmb_ok else 'FAIL'}")

    # 6. Tunelamiento cosmico
    print("\n[6] Tunelamiento cosmico:")
    print("-" * 70)

    tunel = TunelamientoCosmico()

    # Probabilidad de tunelamiento para diferentes barreras
    V_test = [1e-30, 1e-25, 1e-20]  # J
    E = 1e-35  # J
    M = 1e-25  # kg
    ancho = 1e-15  # m

    print(f"    {'V_barrera [J]':>15} {'P_tunel':>15}")
    for V in V_test:
        P = tunel.probabilidad_tunelamiento(V, E, M, ancho)
        print(f"    {V:>15.2e} {P:>15.2e}")

    # Barrera mas alta -> menos tunelamiento
    P_low = tunel.probabilidad_tunelamiento(V_test[0], E, M, ancho)
    P_high = tunel.probabilidad_tunelamiento(V_test[2], E, M, ancho)
    tunel_ok = P_low > P_high
    print(f"\n    Barrera mayor -> P menor: {'PASS' if tunel_ok else 'FAIL'}")

    # Resultado final
    passed = qubit_ok and ent_ok and tensor_ok and decoh_ok and cmb_ok and tunel_ok

    print("\n" + "=" * 70)
    print(f"  QUANTUM EFFECTS MCMC: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    test_Quantum_Effects_MCMC()
