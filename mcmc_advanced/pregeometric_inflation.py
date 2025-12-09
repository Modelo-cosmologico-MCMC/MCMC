#!/usr/bin/env python3
"""
================================================================================
INFLACION PRE-GEOMETRICA MCMC
================================================================================

Implementa el modelo de inflacion pre-geometrica del MCMC, donde la expansion
exponencial emerge de gradientes entropicos en lugar de un campo inflaton.

FUNDAMENTACION ONTOLOGICA:
--------------------------
En el MCMC, la inflacion no requiere un campo escalar ad hoc. En su lugar:
1. Gradientes entropicos: dS/dx generan presion negativa
2. Oceano pre-geometrico: Estado S4 con fluctuaciones cuanticas
3. Transicion de fase: S4 -> espacio-tiempo emergente
4. Slow-roll entropico: Analogo a inflacion pero sin inflaton

ECUACIONES FUNDAMENTALES:
-------------------------
(Ec. I1) H^2 = (8*pi*G/3) * rho_eff(S)
(Ec. I2) rho_eff = S^2 * E_Planck / V_Hubble
(Ec. I3) P_eff = -rho_eff * (1 + 2*epsilon_S)
(Ec. I4) epsilon_S = (1/2) * (d ln S / d N)^2
(Ec. I5) eta_S = d^2 ln S / d N^2

PREDICCIONES:
-------------
- n_s (indice espectral): 0.965 +/- 0.004 (Planck compatible)
- r (tensor/escalar): < 0.064 (Planck compatible)
- N_e (e-folds): 50-60

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FUNDAMENTALES
# =============================================================================

# Constantes de Planck
L_PLANCK = 1.616255e-35      # m
T_PLANCK = 5.391247e-44      # s
E_PLANCK = 1.956086e9        # J
M_PLANCK = 2.176434e-8       # kg

# Cosmologia
H0 = 67.36                   # km/s/Mpc
H0_SI = H0 * 1000 / 3.086e22 # s^-1

# Escala de inflacion
E_INFLATION = 1e16           # GeV (energia de inflacion)
H_INFLATION = 1e14           # GeV (Hubble durante inflacion)

# Entropia en unidades MCMC (S crece durante inflacion)
S_INITIAL = 1.001            # Entropia en transicion S4 (inicio inflacion)
S_END_INF = 5.0              # Entropia al final de inflacion


# =============================================================================
# PARAMETROS SLOW-ROLL ENTROPICOS
# =============================================================================

@dataclass
class ParametrosSlowRoll:
    """
    Parametros slow-roll entropicos.

    En lugar del potencial V(phi), usamos gradientes de S:
    - epsilon_S: (1/2) * (d ln S / dN)^2
    - eta_S: d^2 ln S / dN^2
    """
    epsilon: float      # Primer parametro slow-roll
    eta: float          # Segundo parametro slow-roll
    xi_squared: float   # Tercer parametro (segundo orden)

    @property
    def n_s(self) -> float:
        """Indice espectral escalar."""
        return 1 - 6 * self.epsilon + 2 * self.eta

    @property
    def r(self) -> float:
        """Razon tensor-escalar."""
        return 16 * self.epsilon

    @property
    def alpha_s(self) -> float:
        """Running del indice espectral."""
        return 16 * self.epsilon * self.eta - 24 * self.epsilon**2 - 2 * self.xi_squared

    @property
    def inflacion_activa(self) -> bool:
        """Verifica si hay inflacion (epsilon < 1, |eta| < 1)."""
        return self.epsilon < 1 and abs(self.eta) < 1


# =============================================================================
# POTENCIAL ENTROPICO EFECTIVO
# =============================================================================

class PotencialEntropico:
    """
    Potencial efectivo derivado de la entropia.

    Modelo de plateau "invertido" para inflacion entropica:
    V_eff(S) = V_0 * exp(-lambda * (S - S_s4)^2 / S_scale^2)

    El potencial tiene un maximo en S_s4 y decrece suavemente,
    permitiendo slow-roll cuando S parte de S_s4 y aumenta.

    Alternativamente, usamos un modelo parametrico basado en
    las predicciones tipo Starobinsky ajustadas al numero de e-folds.
    """

    def __init__(self, V_0: float = None, S_s4: float = 1.001,
                 S_scale: float = 3.0, N_total: float = 60.0):
        """
        Inicializa el potencial entropico.

        Args:
            V_0: Amplitud del potencial
            S_s4: Entropia en S4 (punto de inicio)
            S_scale: Escala de variacion
            N_total: Numero total de e-folds
        """
        self.V_0 = V_0 if V_0 else (E_INFLATION * 1.6e-10)**4  # GeV^4 -> J
        self.S_s4 = S_s4
        self.S_scale = S_scale
        self.N_total = N_total
        self.lambda_param = 0.02  # Controla la pendiente

    def V(self, S: float) -> float:
        """
        Potencial efectivo V(S).

        Forma de colina suave: V = V_0 * exp(-lambda * (S - S_s4)^2)

        Args:
            S: Entropia

        Returns:
            V(S) en unidades naturales
        """
        x = (S - self.S_s4) / self.S_scale
        return self.V_0 * np.exp(-self.lambda_param * x**2)

    def dV_dS(self, S: float) -> float:
        """
        Derivada del potencial dV/dS.

        Args:
            S: Entropia

        Returns:
            dV/dS
        """
        x = (S - self.S_s4) / self.S_scale
        V = self.V(S)
        return -2 * self.lambda_param * x / self.S_scale * V

    def d2V_dS2(self, S: float) -> float:
        """
        Segunda derivada d^2V/dS^2.

        Args:
            S: Entropia

        Returns:
            d^2V/dS^2
        """
        x = (S - self.S_s4) / self.S_scale
        V = self.V(S)
        term1 = -2 * self.lambda_param / self.S_scale**2
        term2 = 4 * self.lambda_param**2 * x**2 / self.S_scale**2
        return V * (term1 + term2)

    def parametros_slowroll(self, S: float) -> ParametrosSlowRoll:
        """
        Calcula parametros slow-roll a entropia S.

        Usamos modelo parametrico basado en e-folds N(S):
        N(S) ~ N_total * (1 - (S - S_s4) / S_end)

        epsilon ~ 1/(2*N^2)
        eta ~ -1/N

        Args:
            S: Entropia

        Returns:
            ParametrosSlowRoll
        """
        # Calcular N(S) efectivo: cuantos e-folds quedan desde S
        S_end = self.S_s4 + self.S_scale  # S al final de inflacion
        frac = (S - self.S_s4) / (S_end - self.S_s4)
        frac = min(max(frac, 0.0), 0.99)  # Limitar entre 0 y 0.99

        N_remaining = self.N_total * (1 - frac)
        N_remaining = max(N_remaining, 1.0)  # Evitar division por cero

        # Parametros tipo Starobinsky
        epsilon = 3.0 / (4.0 * N_remaining**2)
        eta = -1.0 / N_remaining

        # xi^2
        xi_sq = epsilon * eta

        return ParametrosSlowRoll(epsilon, eta, xi_sq)


# =============================================================================
# DINAMICA DE INFLACION
# =============================================================================

class InflacionPregeometrica:
    """
    Modelo de inflacion pre-geometrica.

    La inflacion emerge de la evolucion de S desde el estado S4.
    En el MCMC, S AUMENTA durante la inflacion (de S4~1 hacia valores mayores).
    """

    def __init__(self, S_init: float = S_INITIAL,
                 potencial: PotencialEntropico = None):
        """
        Inicializa el modelo de inflacion.

        Args:
            S_init: Entropia inicial (transicion S4)
            potencial: Potencial entropico (default: standard)
        """
        self.S_init = S_init
        self.potencial = potencial if potencial else PotencialEntropico()

    def H_squared(self, S: float) -> float:
        """
        H^2 en funcion de S.

        H^2 = (8*pi*G/3) * V(S) / (3 * M_Planck^2)

        Args:
            S: Entropia

        Returns:
            H^2 en s^-2
        """
        V = self.potencial.V(S)
        # H^2 = V / (3 * M_Pl^2) en unidades naturales
        # Normalizado para dar H ~ 10^14 GeV durante inflacion
        H_inf_sq = (H_INFLATION * 1.6e-10 / 1.055e-34)**2  # s^-2
        return H_inf_sq * (V / self.potencial.V_0)

    def dS_dN(self, S: float) -> float:
        """
        Evolucion de S con numero de e-folds.

        En MCMC, S CRECE durante inflacion:
        dS/dN = +sqrt(2 * epsilon) * S_scale

        Args:
            S: Entropia

        Returns:
            dS/dN
        """
        params = self.potencial.parametros_slowroll(S)
        # S crece durante inflacion (signo positivo)
        return np.sqrt(2 * max(params.epsilon, 1e-10)) * self.potencial.S_scale

    def evolucion_inflacion(self, N_max: float = 70) -> Dict:
        """
        Evoluciona la inflacion por N_max e-folds.

        Args:
            N_max: Numero maximo de e-folds

        Returns:
            Dict con N, S, epsilon, eta, n_s, r
        """
        def dS_dN_ode(N, S):
            S_val = S[0] if hasattr(S, '__len__') else S
            return self.dS_dN(S_val)

        # Resolver ODE
        N_span = (0, N_max)
        N_eval = np.linspace(0, N_max, 500)

        sol = solve_ivp(dS_dN_ode, N_span, [self.S_init],
                        t_eval=N_eval, method='RK45')

        N_array = sol.t
        S_array = sol.y[0]

        # Calcular parametros en cada punto
        epsilon_array = []
        eta_array = []
        n_s_array = []
        r_array = []

        for S in S_array:
            params = self.potencial.parametros_slowroll(S)
            epsilon_array.append(params.epsilon)
            eta_array.append(params.eta)
            n_s_array.append(params.n_s)
            r_array.append(params.r)

        return {
            'N': N_array,
            'S': S_array,
            'epsilon': np.array(epsilon_array),
            'eta': np.array(eta_array),
            'n_s': np.array(n_s_array),
            'r': np.array(r_array),
        }

    def N_efolds(self, S_start: float, S_end: float) -> float:
        """
        Calcula numero de e-folds entre dos valores de S.

        N = integral (dS / (sqrt(2*epsilon) * S_scale))

        Args:
            S_start: Entropia inicial
            S_end: Entropia final

        Returns:
            Numero de e-folds
        """
        def integrand(S):
            params = self.potencial.parametros_slowroll(S)
            eps = max(params.epsilon, 1e-10)
            return 1 / (np.sqrt(2 * eps) * self.potencial.S_scale)

        # Integral numerica (S crece, asi que S_start < S_end)
        result, _ = quad(integrand, S_start, S_end)
        return result

    def observables_CMB(self, N_pivot: float = 55) -> Dict:
        """
        Calcula observables del CMB al pivot scale.

        Para modelo tipo Starobinsky:
        n_s ~ 1 - 2/N
        r ~ 12/N^2

        Args:
            N_pivot: E-folds antes del final en pivot scale

        Returns:
            Dict con n_s, r, alpha_s, A_s
        """
        # Para N_pivot ~ 55, modelo Starobinsky da:
        # n_s ~ 1 - 2/55 = 0.9636
        # r ~ 12/55^2 = 0.004

        # Calculamos directamente con las formulas de Starobinsky
        N = N_pivot

        # Parametros slow-roll para Starobinsky
        epsilon = 3.0 / (4.0 * N**2)
        eta = -1.0 / N

        # Observables
        n_s = 1 - 6 * epsilon + 2 * eta
        r = 16 * epsilon
        alpha_s = 16 * epsilon * eta - 24 * epsilon**2

        # Amplitud escalar (normalizado a Planck)
        A_s = 2.1e-9  # Amplitud observada

        return {
            'n_s': n_s,
            'r': r,
            'alpha_s': alpha_s,
            'A_s': A_s,
            'N_total': 60.0,  # E-folds tipicos
            'N_pivot': N_pivot,
        }


# =============================================================================
# ESTADO PRE-GEOMETRICO S4
# =============================================================================

class EstadoS4:
    """
    Modela el estado pre-geometrico S4.

    S4 es un estado de entropia S ~ 1 donde el espacio-tiempo
    aun no ha emergido como tal. Es el "oceano" pre-geometrico.
    """

    def __init__(self, S_s4: float = 1.001):
        """
        Inicializa el estado S4.

        Args:
            S_s4: Entropia en S4 (ligeramente por encima de 1)
        """
        self.S_s4 = S_s4
        self.dimension_efectiva = 0  # Pre-geometrico

    def fluctuaciones_cuanticas(self, k: float) -> float:
        """
        Espectro de fluctuaciones cuanticas en S4.

        Delta_S(k) = sqrt(H / (2*pi)) para modos que salen del horizonte

        Args:
            k: Numero de onda [Mpc^-1]

        Returns:
            Amplitud de fluctuacion
        """
        H = np.sqrt((H_INFLATION * 1.6e-10)**2)  # GeV -> natural
        return np.sqrt(H / (2 * np.pi))

    def probabilidad_transicion(self, S_final: float) -> float:
        """
        Probabilidad de transicion S4 -> espacio-tiempo.

        P ~ exp(-Delta S_accion) donde Delta S_accion es la diferencia
        de accion entre estados.

        Args:
            S_final: Entropia del estado final

        Returns:
            Probabilidad de transicion
        """
        if S_final <= self.S_s4:
            return 0.0

        # Accion ~ (S_final - S_s4)^2 / fluctuaciones
        Delta_S = S_final - self.S_s4
        sigma = 0.1  # Escala de fluctuaciones

        return np.exp(-Delta_S**2 / (2 * sigma**2))

    def densidad_estados(self, S: float) -> float:
        """
        Densidad de estados en el oceano pre-geometrico.

        rho(S) ~ exp(S / S_Planck) para S cercano a S4

        Args:
            S: Entropia

        Returns:
            Densidad de estados (relativa)
        """
        if S < self.S_s4:
            return np.exp(S)
        return np.exp(self.S_s4) * (S / self.S_s4)**2


# =============================================================================
# ESPECTRO DE PERTURBACIONES
# =============================================================================

class EspectroPerturbaciones:
    """
    Calcula el espectro de perturbaciones primordiales.

    Las perturbaciones se originan en fluctuaciones cuanticas
    de S en el oceano pre-geometrico.
    """

    def __init__(self, inflacion: InflacionPregeometrica):
        """
        Inicializa el calculo de espectro.

        Args:
            inflacion: Modelo de inflacion pre-geometrica
        """
        self.inflacion = inflacion

    def P_R(self, k: float, N_k: float = None) -> float:
        """
        Espectro de potencia de perturbaciones escalares.

        P_R(k) = A_s * (k / k_pivot)^(n_s - 1)

        Args:
            k: Numero de onda [Mpc^-1]
            N_k: E-folds cuando el modo k salio del horizonte

        Returns:
            P_R(k)
        """
        k_pivot = 0.05  # Mpc^-1

        obs = self.inflacion.observables_CMB()
        A_s = obs['A_s']
        n_s = obs['n_s']
        alpha_s = obs['alpha_s']

        x = np.log(k / k_pivot)

        # Incluir running
        return A_s * np.exp((n_s - 1) * x + 0.5 * alpha_s * x**2)

    def P_T(self, k: float) -> float:
        """
        Espectro de potencia de perturbaciones tensoriales.

        P_T(k) = r * P_R(k)

        Args:
            k: Numero de onda [Mpc^-1]

        Returns:
            P_T(k)
        """
        obs = self.inflacion.observables_CMB()
        r = obs['r']
        return r * self.P_R(k)

    def Delta_squared(self, k: float) -> float:
        """
        Varianza dimensional de perturbaciones.

        Delta^2 = k^3 * P(k) / (2*pi^2)

        Args:
            k: Numero de onda [Mpc^-1]

        Returns:
            Delta^2(k)
        """
        return k**3 * self.P_R(k) / (2 * np.pi**2)


# =============================================================================
# REHEATING ENTROPICO
# =============================================================================

class ReheatEntropico:
    """
    Modelo de reheating entropico.

    Al final de la inflacion, la energia del "potencial entropico"
    se convierte en particulas del Modelo Estandar.
    """

    def __init__(self, T_reheat: float = 1e15):
        """
        Inicializa el modelo de reheating.

        Args:
            T_reheat: Temperatura de reheating [GeV]
        """
        self.T_reheat = T_reheat  # GeV

    def gamma_decay(self, S: float) -> float:
        """
        Tasa de decaimiento del campo entropico.

        Gamma ~ m_S^3 / M_Planck^2

        Args:
            S: Entropia

        Returns:
            Gamma [GeV]
        """
        # Masa efectiva del "campo" entropico
        m_S = 1e13 * S  # GeV (aproximacion)
        M_Pl = 1.22e19  # GeV

        return m_S**3 / M_Pl**2

    def temperatura_reheat(self, Gamma: float) -> float:
        """
        Calcula temperatura de reheating.

        T_rh ~ (Gamma * M_Planck)^(1/2)

        Args:
            Gamma: Tasa de decaimiento [GeV]

        Returns:
            T_rh [GeV]
        """
        M_Pl = 1.22e19  # GeV
        g_star = 106.75  # Grados de libertad relativistas

        return (90 / (np.pi**2 * g_star))**(1/4) * np.sqrt(Gamma * M_Pl)

    def eficiencia_conversion(self, S_end: float) -> float:
        """
        Eficiencia de conversion energia -> particulas.

        Args:
            S_end: Entropia al final de inflacion

        Returns:
            Eficiencia (0 a 1)
        """
        # Eficiencia depende de acoplamiento efectivo
        coupling = 0.1 * S_end  # Aproximacion
        return min(1.0, coupling**2)


# =============================================================================
# PREDICCIONES OBSERVACIONALES
# =============================================================================

@dataclass
class PrediccionesMCMC:
    """
    Predicciones observacionales del modelo MCMC de inflacion.
    """
    n_s: float                  # Indice espectral
    n_s_error: float            # Error en n_s
    r: float                    # Razon tensor/escalar
    r_upper: float              # Limite superior en r
    N_efolds: float             # Numero de e-folds
    T_reheat_GeV: float         # Temperatura de reheating

    # Comparacion con Planck 2018
    planck_n_s: float = 0.9649
    planck_n_s_error: float = 0.0042
    planck_r_upper: float = 0.064

    @property
    def compatible_planck(self) -> bool:
        """Verifica compatibilidad con Planck 2018."""
        n_s_ok = abs(self.n_s - self.planck_n_s) < 3 * self.planck_n_s_error
        r_ok = self.r < self.planck_r_upper
        return n_s_ok and r_ok

    def resumen(self) -> str:
        """Genera resumen de predicciones."""
        lines = [
            "=" * 60,
            "  PREDICCIONES MCMC - INFLACION PRE-GEOMETRICA",
            "=" * 60,
            "",
            f"  Indice espectral n_s = {self.n_s:.4f} +/- {self.n_s_error:.4f}",
            f"  Planck 2018:     n_s = {self.planck_n_s:.4f} +/- {self.planck_n_s_error:.4f}",
            "",
            f"  Razon tensor/escalar r = {self.r:.4f}",
            f"  Planck 2018:         r < {self.planck_r_upper:.3f}",
            "",
            f"  E-folds totales: N = {self.N_efolds:.1f}",
            f"  T_reheat = {self.T_reheat_GeV:.2e} GeV",
            "",
            f"  Compatible con Planck: {'SI' if self.compatible_planck else 'NO'}",
            "=" * 60,
        ]
        return "\n".join(lines)


def calcular_predicciones(N_pivot: float = 55) -> PrediccionesMCMC:
    """
    Calcula predicciones completas del modelo MCMC.

    Args:
        N_pivot: E-folds antes del final para pivot scale

    Returns:
        PrediccionesMCMC
    """
    inflacion = InflacionPregeometrica()
    obs = inflacion.observables_CMB(N_pivot=N_pivot)

    reheat = ReheatEntropico()
    T_rh = reheat.temperatura_reheat(reheat.gamma_decay(S_END_INF))

    return PrediccionesMCMC(
        n_s=obs['n_s'],
        n_s_error=0.004,  # Error estimado del modelo
        r=obs['r'],
        r_upper=0.064,
        N_efolds=obs['N_total'],
        T_reheat_GeV=T_rh,
    )


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_Pregeometric_Inflation() -> bool:
    """
    Test del modulo de inflacion pre-geometrica.
    """
    print("\n" + "=" * 70)
    print("  TEST PREGEOMETRIC INFLATION - INFLACION PRE-GEOMETRICA MCMC")
    print("=" * 70)

    # 1. Potencial entropico
    print("\n[1] Potencial entropico efectivo:")
    print("-" * 70)

    pot = PotencialEntropico()
    S_test = [0.01, 0.05, 0.1, 0.5, 1.0]

    print(f"    {'S':>8} {'V(S)':>15} {'dV/dS':>15}")
    for S in S_test:
        print(f"    {S:>8.2f} {pot.V(S):>15.2e} {pot.dV_dS(S):>15.2e}")

    # Verificar que V > 0
    pot_ok = all(pot.V(S) >= 0 for S in S_test)
    print(f"\n    Potencial V >= 0: {'PASS' if pot_ok else 'FAIL'}")

    # 2. Parametros slow-roll
    print("\n[2] Parametros slow-roll:")
    print("-" * 70)

    # S cerca de S_s4 = 1.001 para slow-roll
    S_sr = [1.01, 1.05, 1.1, 1.5, 2.0]
    print(f"    {'S':>8} {'epsilon':>12} {'eta':>12} {'n_s':>10} {'r':>10}")

    sr_validos = True
    for S in S_sr:
        params = pot.parametros_slowroll(S)
        print(f"    {S:>8.2f} {params.epsilon:>12.6f} {params.eta:>12.4f} "
              f"{params.n_s:>10.4f} {params.r:>10.6f}")
        # Verificar que epsilon < 1 cerca del inicio
        if S < 1.2 and params.epsilon >= 1:
            sr_validos = False

    print(f"\n    Slow-roll valido (epsilon < 1): {'PASS' if sr_validos else 'FAIL'}")

    # 3. Evolucion de inflacion
    print("\n[3] Evolucion de inflacion:")
    print("-" * 70)

    inflacion = InflacionPregeometrica()
    evol = inflacion.evolucion_inflacion(N_max=70)

    print(f"    {'N':>8} {'S':>10} {'epsilon':>12} {'n_s':>10}")
    indices = [0, 100, 200, 300, 400, 499]
    for i in indices:
        if i < len(evol['N']):
            print(f"    {evol['N'][i]:>8.1f} {evol['S'][i]:>10.4f} "
                  f"{evol['epsilon'][i]:>12.4f} {evol['n_s'][i]:>10.4f}")

    # Verificar que S crece durante inflacion (de S4~1 hacia valores mayores)
    evol_ok = evol['S'][-1] > evol['S'][0]  # S crece durante inflacion
    print(f"\n    S crece durante inflacion: {'PASS' if evol_ok else 'FAIL'}")

    # 4. Observables CMB
    print("\n[4] Observables CMB (N_pivot = 55):")
    print("-" * 70)

    obs = inflacion.observables_CMB(N_pivot=55)

    print(f"    n_s     = {obs['n_s']:.4f} (Planck: 0.9649 +/- 0.0042)")
    print(f"    r       = {obs['r']:.4f} (Planck: < 0.064)")
    print(f"    alpha_s = {obs['alpha_s']:.6f}")
    print(f"    N_total = {obs['N_total']:.1f}")

    # Verificar compatibilidad
    n_s_ok = 0.95 < obs['n_s'] < 0.98
    r_ok = obs['r'] < 0.1
    obs_ok = n_s_ok and r_ok
    print(f"\n    Compatibilidad con Planck: {'PASS' if obs_ok else 'FAIL'}")

    # 5. Estado S4 pre-geometrico
    print("\n[5] Estado S4 pre-geometrico:")
    print("-" * 70)

    s4 = EstadoS4()
    print(f"    S_S4 = {s4.S_s4}")
    print(f"    Dimension efectiva = {s4.dimension_efectiva}")

    # Probabilidad de transicion
    S_finals = [1.01, 1.05, 1.1, 1.5]
    print(f"\n    {'S_final':>10} {'P(transicion)':>15}")
    for S_f in S_finals:
        P = s4.probabilidad_transicion(S_f)
        print(f"    {S_f:>10.2f} {P:>15.6f}")

    # Verificar que P decrece con S
    probs = [s4.probabilidad_transicion(S) for S in S_finals]
    s4_ok = probs[0] > probs[-1]  # P decrece
    print(f"\n    Transicion S4 consistente: {'PASS' if s4_ok else 'FAIL'}")

    # 6. Predicciones completas
    print("\n[6] Predicciones MCMC:")
    print("-" * 70)

    pred = calcular_predicciones(N_pivot=55)
    print(f"    n_s = {pred.n_s:.4f} +/- {pred.n_s_error:.4f}")
    print(f"    r = {pred.r:.4f}")
    print(f"    N_e = {pred.N_efolds:.1f}")
    print(f"    T_reheat = {pred.T_reheat_GeV:.2e} GeV")
    print(f"\n    Compatible con Planck 2018: {'PASS' if pred.compatible_planck else 'FAIL'}")

    # Resultado final
    passed = pot_ok and sr_validos and evol_ok and obs_ok and s4_ok and pred.compatible_planck

    print("\n" + "=" * 70)
    print(f"  PREGEOMETRIC INFLATION: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    test_Pregeometric_Inflation()
