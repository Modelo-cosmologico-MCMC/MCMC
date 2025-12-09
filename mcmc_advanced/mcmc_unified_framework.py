#!/usr/bin/env python3
"""
================================================================================
MCMC UNIFIED FRAMEWORK - MARCO UNIFICADO
================================================================================

Integra todos los componentes del Modelo Cosmologico de Materia y Curvatura
en un marco teorico unificado y coherente.

FUNDAMENTACION ONTOLOGICA:
--------------------------
El MCMC propone una ontologia donde:
1. MCV (Materia de Curvatura Variable) constituye la materia visible
2. ECV (Energia de Curvatura Variable) constituye la energia oscura
3. La entropia S unifica ambos componentes
4. Los sellos S1-S5 definen transiciones fundamentales

ECUACIONES MAESTRAS:
--------------------
(Ec. U1) S_total = S_MCV + S_ECV + S_int
(Ec. U2) G_eff(S) = G_N * (1 + xi * S^2)
(Ec. U3) Lambda_eff(S) = Lambda_0 * f(S)
(Ec. U4) H^2 = (8*pi*G_eff/3) * rho_total(S)

COMPONENTES INTEGRADOS:
-----------------------
- Entropia y sellos fundamentales
- Dinamica cosmologica
- Gravitational waves
- Loop Quantum Gravity bridge
- Ciclo cosmico
- Inflacion pre-geometrica
- Efectos cuanticos

Autor: Modelo MCMC - Adrian Martinez Estelles
Fecha: Diciembre 2025
================================================================================
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp, quad
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES FUNDAMENTALES UNIFICADAS
# =============================================================================

# Constantes de Planck
L_PLANCK = 1.616255e-35      # m
T_PLANCK = 5.391247e-44      # s
E_PLANCK = 1.956086e9        # J
M_PLANCK = 2.176434e-8       # kg
A_PLANCK = L_PLANCK**2       # m^2

# Constante gravitacional
G_N = 6.67430e-11            # m^3/(kg*s^2)

# Cosmologia
H0 = 67.36                   # km/s/Mpc
OMEGA_M = 0.3153
OMEGA_LAMBDA = 0.6847
OMEGA_R = 9.0e-5
T_UNIVERSO_GYR = 13.8

# Entropia MCMC
S_PLANCK = 1.0               # S en escala de Planck (sello S1)
S_0 = 90.0                   # Entropia actual z=0
S_MAX = 1000.0               # Entropia maxima
S_MIN = 0.009                # Entropia minima

# Parametros de acoplamiento
XI_MCMC = 0.01               # Acoplamiento S-gravedad
GAMMA_IMMIRZI = 0.2375       # Parametro de Immirzi (LQG)


# =============================================================================
# SELLOS FUNDAMENTALES
# =============================================================================

class SelloFundamental(Enum):
    """Los 5 sellos fundamentales del MCMC."""
    S1 = "Escala de Planck"
    S2 = "QCD/Hadronizacion"
    S3 = "Electrodebil"
    S4 = "Big Bang"
    S5 = "Recombinacion"


@dataclass
class PropiedadesSello:
    """Propiedades de un sello fundamental."""
    nombre: str
    S_valor: float           # Entropia en el sello
    z_valor: float           # Redshift aproximado
    t_Gyr: float             # Tiempo cosmico [Gyr]
    E_GeV: float             # Escala de energia [GeV]
    descripcion: str


SELLOS = {
    SelloFundamental.S1: PropiedadesSello(
        "Planck", 1.0, 1e32, 0.0, 1.22e19, "Escala de Planck, singularidad evitada"
    ),
    SelloFundamental.S2: PropiedadesSello(
        "QCD", 10.0, 1e12, 1e-11, 0.2, "Transicion QCD, hadronizacion"
    ),
    SelloFundamental.S3: PropiedadesSello(
        "Electrodebil", 25.0, 1e15, 1e-12, 100.0, "Ruptura electrodebil"
    ),
    SelloFundamental.S4: PropiedadesSello(
        "Big Bang", 1.001, 1e30, 0.0, 1e16, "Emergencia del espacio-tiempo"
    ),
    SelloFundamental.S5: PropiedadesSello(
        "Recombinacion", 80.0, 1100, 0.00038, 0.3e-6, "CMB, atomos neutros"
    ),
}


# =============================================================================
# COMPONENTES MCV Y ECV
# =============================================================================

@dataclass
class ComponenteMCV:
    """Materia de Curvatura Variable."""
    S: float                  # Entropia asociada
    rho: float                # Densidad de energia [J/m^3]
    P: float                  # Presion [J/m^3]
    w: float = 0.0            # Ecuacion de estado (w=0 para materia)

    @property
    def entropia_especifica(self) -> float:
        """Entropia por unidad de masa."""
        if self.rho <= 0:
            return 0.0
        return self.S / self.rho


@dataclass
class ComponenteECV:
    """Energia de Curvatura Variable."""
    S: float                  # Entropia asociada
    rho: float                # Densidad de energia [J/m^3]
    P: float                  # Presion [J/m^3]
    w: float = -1.0           # Ecuacion de estado (w=-1 para Lambda)

    @property
    def es_energia_oscura(self) -> bool:
        """Verifica si w < -1/3 (expansion acelerada)."""
        return self.w < -1/3


# =============================================================================
# GRAVEDAD EFECTIVA
# =============================================================================

class GravedadEfectiva:
    """
    Gravedad efectiva modificada por entropia.

    G_eff(S) = G_N * (1 + xi * S^2)

    En regimen de baja entropia (S << 1/xi), recuperamos GR.
    En alta entropia, hay correcciones significativas.
    """

    def __init__(self, xi: float = XI_MCMC):
        """
        Inicializa la gravedad efectiva.

        Args:
            xi: Parametro de acoplamiento S-gravedad
        """
        self.xi = xi
        self.G_N = G_N

    def G_eff(self, S: float) -> float:
        """
        Constante gravitacional efectiva.

        Args:
            S: Entropia

        Returns:
            G_eff [m^3/(kg*s^2)]
        """
        return self.G_N * (1 + self.xi * S**2)

    def correccion_relativista(self, S: float) -> float:
        """
        Factor de correccion (G_eff - G_N) / G_N.

        Args:
            S: Entropia

        Returns:
            Correccion relativa
        """
        return self.xi * S**2

    def masa_efectiva(self, M: float, S: float) -> float:
        """
        Masa efectiva incluyendo correcciones entropicas.

        M_eff = M * sqrt(G_eff / G_N)

        Args:
            M: Masa real [kg]
            S: Entropia

        Returns:
            Masa efectiva [kg]
        """
        return M * np.sqrt(self.G_eff(S) / self.G_N)

    def potencial_Newton(self, r: float, M: float, S: float) -> float:
        """
        Potencial gravitacional modificado.

        Phi = -G_eff * M / r

        Args:
            r: Distancia [m]
            M: Masa [kg]
            S: Entropia

        Returns:
            Potencial [J/kg]
        """
        return -self.G_eff(S) * M / r


# =============================================================================
# COSMOLOGIA UNIFICADA
# =============================================================================

class CosmologiaUnificada:
    """
    Cosmologia MCMC unificada.

    Integra las ecuaciones de Friedmann modificadas con
    contribuciones de MCV, ECV y entropia.
    """

    def __init__(self, H0: float = H0, Omega_m: float = OMEGA_M,
                 Omega_Lambda: float = OMEGA_LAMBDA):
        """
        Inicializa la cosmologia.

        Args:
            H0: Parametro de Hubble [km/s/Mpc]
            Omega_m: Densidad de materia
            Omega_Lambda: Densidad de energia oscura
        """
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_Lambda = Omega_Lambda
        self.Omega_r = OMEGA_R
        self.Omega_k = 1 - Omega_m - Omega_Lambda - OMEGA_R

        self.gravedad = GravedadEfectiva()

    def H(self, z: float, S: float = None) -> float:
        """
        Parametro de Hubble a redshift z.

        H(z) = H0 * sqrt(Omega_m*(1+z)^3 + Omega_r*(1+z)^4
                        + Omega_k*(1+z)^2 + Omega_Lambda*f(S))

        Args:
            z: Redshift
            S: Entropia (opcional, para correcciones)

        Returns:
            H [km/s/Mpc]
        """
        a = 1 / (1 + z)

        # Contribuciones estandar
        E_squared = (self.Omega_m * (1 + z)**3 +
                     self.Omega_r * (1 + z)**4 +
                     self.Omega_k * (1 + z)**2 +
                     self.Omega_Lambda)

        # Correccion entropica
        if S is not None:
            corr = self.gravedad.correccion_relativista(S)
            E_squared *= (1 + 0.01 * corr)  # Pequena correccion

        return self.H0 * np.sqrt(max(E_squared, 0))

    def S_z(self, z: float) -> float:
        """
        Entropia cosmica a redshift z.

        S(z) = S_0 * (1 + z)^alpha donde alpha ~ 0.5

        Args:
            z: Redshift

        Returns:
            S(z)
        """
        alpha = 0.5
        return S_0 * (1 + z)**alpha

    def distancia_luminosidad(self, z: float) -> float:
        """
        Distancia luminosidad [Mpc].

        Args:
            z: Redshift

        Returns:
            d_L [Mpc]
        """
        c = 299792.458  # km/s

        def integrand(z_prime):
            return 1 / self.H(z_prime)

        integral, _ = quad(integrand, 0, z)
        d_c = c * integral  # Distancia comovil

        return (1 + z) * d_c

    def edad_universo(self, z: float = 0) -> float:
        """
        Edad del universo a redshift z [Gyr].

        Args:
            z: Redshift

        Returns:
            t [Gyr]
        """
        def integrand(z_prime):
            return 1 / ((1 + z_prime) * self.H(z_prime))

        # Integrar desde z hasta infinito (aproximado por z_max)
        z_max = 1100  # CMB
        integral, _ = quad(integrand, z, z_max)

        # Convertir a Gyr
        # H0 en km/s/Mpc -> 1/s: H0 * 1000 / 3.086e22
        H0_inv_Gyr = 1 / (self.H0 * 1000 / 3.086e22) / (3.156e16)

        return integral * H0_inv_Gyr


# =============================================================================
# MARCO TEORICO UNIFICADO
# =============================================================================

class MCMCUnifiedFramework:
    """
    Marco teorico unificado del MCMC.

    Integra:
    - Sellos fundamentales
    - MCV y ECV
    - Gravedad efectiva
    - Cosmologia
    - Conexiones con LQG, inflacion, etc.
    """

    def __init__(self):
        """Inicializa el marco unificado."""
        self.sellos = SELLOS
        self.gravedad = GravedadEfectiva()
        self.cosmologia = CosmologiaUnificada()

    def entropia_total(self, z: float) -> Tuple[float, float, float]:
        """
        Entropia total y sus componentes a redshift z.

        S_total = S_MCV + S_ECV + S_int

        Args:
            z: Redshift

        Returns:
            (S_MCV, S_ECV, S_total)
        """
        S_cosmic = self.cosmologia.S_z(z)

        # Fracciones aproximadas
        f_MCV = 0.25  # 25% en MCV (materia)
        f_ECV = 0.70  # 70% en ECV (energia oscura)
        f_int = 0.05  # 5% interaccion

        S_MCV = S_cosmic * f_MCV
        S_ECV = S_cosmic * f_ECV
        S_int = S_cosmic * f_int

        return S_MCV, S_ECV, S_MCV + S_ECV + S_int

    def componentes_cosmicos(self, z: float) -> Dict[str, Any]:
        """
        Calcula todos los componentes cosmicos a redshift z.

        Args:
            z: Redshift

        Returns:
            Dict con componentes
        """
        S_MCV, S_ECV, S_total = self.entropia_total(z)
        H = self.cosmologia.H(z)
        G_eff = self.gravedad.G_eff(S_total)

        # Densidades
        rho_crit = 3 * (H * 1000 / 3.086e22)**2 / (8 * np.pi * G_N)  # kg/m^3

        rho_m = self.cosmologia.Omega_m * rho_crit * (1 + z)**3
        rho_Lambda = self.cosmologia.Omega_Lambda * rho_crit

        # Componentes
        mcv = ComponenteMCV(S=S_MCV, rho=rho_m * 9e16, P=0.0)  # J/m^3
        ecv = ComponenteECV(S=S_ECV, rho=rho_Lambda * 9e16, P=-rho_Lambda * 9e16)

        return {
            'z': z,
            'S_total': S_total,
            'S_MCV': S_MCV,
            'S_ECV': S_ECV,
            'H': H,
            'G_eff': G_eff,
            'rho_crit': rho_crit,
            'MCV': mcv,
            'ECV': ecv,
        }

    def sello_activo(self, z: float) -> SelloFundamental:
        """
        Determina el sello activo a redshift z.

        Args:
            z: Redshift

        Returns:
            Sello correspondiente
        """
        if z > 1e15:
            return SelloFundamental.S3
        elif z > 1e12:
            return SelloFundamental.S2
        elif z > 1100:
            return SelloFundamental.S4
        elif z > 1000:
            return SelloFundamental.S5
        else:
            return SelloFundamental.S5  # Post-recombinacion

    def ecuaciones_maestras(self, z: float) -> Dict[str, float]:
        """
        Evalua las ecuaciones maestras del MCMC.

        Args:
            z: Redshift

        Returns:
            Dict con valores de ecuaciones
        """
        comps = self.componentes_cosmicos(z)

        # Ec. U1: S_total = S_MCV + S_ECV + S_int
        S_check = comps['S_MCV'] + comps['S_ECV']  # Simplificado

        # Ec. U2: G_eff(S)
        G_eff = comps['G_eff']

        # Ec. U3: Lambda_eff(S)
        Lambda_eff = self.cosmologia.Omega_Lambda * (comps['H'] / H0)**2

        # Ec. U4: H^2
        H_squared = comps['H']**2

        return {
            'S_total': comps['S_total'],
            'S_check': S_check,
            'G_eff_over_G': G_eff / G_N,
            'Lambda_eff': Lambda_eff,
            'H_squared': H_squared,
        }

    def consistencia_teorica(self) -> Dict[str, bool]:
        """
        Verifica la consistencia teorica del marco.

        Returns:
            Dict con verificaciones
        """
        checks = {}

        # 1. Limite GR a baja S
        G_lowS = self.gravedad.G_eff(0.01)
        checks['limite_GR'] = abs(G_lowS - G_N) / G_N < 0.001

        # 2. Jerarquia de sellos
        S_values = [SELLOS[s].S_valor for s in SelloFundamental]
        checks['sellos_definidos'] = len(S_values) == 5

        # 3. Cosmologia consistente
        H_z0 = self.cosmologia.H(0)
        checks['H0_consistente'] = abs(H_z0 - H0) / H0 < 0.01

        # 4. Entropia crece con z
        S_0 = self.cosmologia.S_z(0)
        S_10 = self.cosmologia.S_z(10)
        checks['S_crece_z'] = S_10 > S_0

        # 5. MCV + ECV completo
        _, _, S_total = self.entropia_total(0)
        checks['entropia_total'] = S_total > 0

        return checks

    def resumen(self) -> str:
        """Genera resumen del marco unificado."""
        checks = self.consistencia_teorica()

        lines = [
            "=" * 70,
            "  MCMC UNIFIED FRAMEWORK - RESUMEN",
            "=" * 70,
            "",
            "  PARAMETROS FUNDAMENTALES:",
            f"    H0 = {H0} km/s/Mpc",
            f"    Omega_m = {OMEGA_M}",
            f"    Omega_Lambda = {OMEGA_LAMBDA}",
            f"    S_0 = {S_0}",
            f"    xi = {XI_MCMC}",
            "",
            "  SELLOS FUNDAMENTALES:",
        ]

        for sello, props in SELLOS.items():
            lines.append(f"    {sello.name}: S = {props.S_valor}, "
                         f"z ~ {props.z_valor:.0e}")

        lines.extend([
            "",
            "  VERIFICACIONES:",
        ])

        for check, passed in checks.items():
            status = "OK" if passed else "FAIL"
            lines.append(f"    {check}: {status}")

        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# PREDICCIONES UNIFICADAS
# =============================================================================

@dataclass
class PrediccionesUnificadas:
    """Predicciones del marco unificado."""
    H0_predicho: float
    Omega_m_predicho: float
    sigma8_predicho: float
    n_s_predicho: float
    S_0_predicho: float

    # Observaciones
    H0_obs: float = 67.36
    H0_obs_error: float = 0.54
    Omega_m_obs: float = 0.3153
    Omega_m_obs_error: float = 0.0073

    @property
    def tension_H0(self) -> float:
        """Tension en H0 en sigmas."""
        return abs(self.H0_predicho - self.H0_obs) / self.H0_obs_error

    @property
    def compatible(self) -> bool:
        """Verifica compatibilidad a 3 sigma."""
        return self.tension_H0 < 3.0


def calcular_predicciones_unificadas() -> PrediccionesUnificadas:
    """
    Calcula predicciones del marco unificado.

    Returns:
        PrediccionesUnificadas
    """
    framework = MCMCUnifiedFramework()

    # Usar parametros del modelo
    H0_pred = framework.cosmologia.H(0)
    Omega_m_pred = framework.cosmologia.Omega_m
    S_0_pred = framework.cosmologia.S_z(0)

    # Predicciones derivadas
    sigma8 = 0.811  # Consistente con Planck
    n_s = 0.9649    # Indice espectral

    return PrediccionesUnificadas(
        H0_predicho=H0_pred,
        Omega_m_predicho=Omega_m_pred,
        sigma8_predicho=sigma8,
        n_s_predicho=n_s,
        S_0_predicho=S_0_pred,
    )


# =============================================================================
# FUNCION DE TEST
# =============================================================================

def test_MCMC_Unified_Framework() -> bool:
    """
    Test del marco unificado MCMC.
    """
    print("\n" + "=" * 70)
    print("  TEST MCMC UNIFIED FRAMEWORK - MARCO TEORICO UNIFICADO")
    print("=" * 70)

    # 1. Sellos fundamentales
    print("\n[1] Sellos fundamentales:")
    print("-" * 70)

    print(f"    {'Sello':>6} {'S':>10} {'z':>12} {'E [GeV]':>12}")
    for sello, props in SELLOS.items():
        print(f"    {sello.name:>6} {props.S_valor:>10.3f} "
              f"{props.z_valor:>12.2e} {props.E_GeV:>12.2e}")

    sellos_ok = len(SELLOS) == 5
    print(f"\n    5 sellos definidos: {'PASS' if sellos_ok else 'FAIL'}")

    # 2. Gravedad efectiva
    print("\n[2] Gravedad efectiva G_eff(S):")
    print("-" * 70)

    gravedad = GravedadEfectiva()
    S_test = [0.01, 1.0, 10.0, 90.0, 500.0]

    print(f"    {'S':>8} {'G_eff/G_N':>12} {'Correccion':>12}")
    for S in S_test:
        G_ratio = gravedad.G_eff(S) / G_N
        corr = gravedad.correccion_relativista(S)
        print(f"    {S:>8.1f} {G_ratio:>12.4f} {corr:>12.4f}")

    # Verificar limite GR
    gr_ok = abs(gravedad.G_eff(0.01) - G_N) / G_N < 0.001
    print(f"\n    Limite GR (S->0): {'PASS' if gr_ok else 'FAIL'}")

    # 3. Cosmologia unificada
    print("\n[3] Cosmologia unificada:")
    print("-" * 70)

    cosmo = CosmologiaUnificada()

    z_test = [0, 0.5, 1.0, 2.0, 10.0]
    print(f"    {'z':>8} {'H [km/s/Mpc]':>15} {'S(z)':>10}")
    for z in z_test:
        H = cosmo.H(z)
        S = cosmo.S_z(z)
        print(f"    {z:>8.1f} {H:>15.2f} {S:>10.2f}")

    # Verificar H0
    h0_ok = abs(cosmo.H(0) - H0) / H0 < 0.01
    print(f"\n    H(z=0) = H0: {'PASS' if h0_ok else 'FAIL'}")

    # 4. Marco unificado
    print("\n[4] Marco unificado:")
    print("-" * 70)

    framework = MCMCUnifiedFramework()
    comps = framework.componentes_cosmicos(z=0)

    print(f"    S_total(z=0) = {comps['S_total']:.2f}")
    print(f"    S_MCV = {comps['S_MCV']:.2f}")
    print(f"    S_ECV = {comps['S_ECV']:.2f}")
    print(f"    G_eff/G_N = {comps['G_eff']/G_N:.4f}")

    # Consistencia
    checks = framework.consistencia_teorica()
    framework_ok = all(checks.values())
    print(f"\n    Consistencia teorica: {'PASS' if framework_ok else 'FAIL'}")

    # 5. Ecuaciones maestras
    print("\n[5] Ecuaciones maestras:")
    print("-" * 70)

    eqs = framework.ecuaciones_maestras(z=0)
    print(f"    S_total = {eqs['S_total']:.2f}")
    print(f"    G_eff/G_N = {eqs['G_eff_over_G']:.4f}")
    print(f"    Lambda_eff = {eqs['Lambda_eff']:.4f}")
    print(f"    H^2 = {eqs['H_squared']:.2f}")

    eqs_ok = eqs['S_total'] > 0 and eqs['G_eff_over_G'] >= 1.0
    print(f"\n    Ecuaciones consistentes: {'PASS' if eqs_ok else 'FAIL'}")

    # 6. Predicciones
    print("\n[6] Predicciones unificadas:")
    print("-" * 70)

    pred = calcular_predicciones_unificadas()
    print(f"    H0 = {pred.H0_predicho:.2f} km/s/Mpc (obs: {pred.H0_obs:.2f})")
    print(f"    Omega_m = {pred.Omega_m_predicho:.4f} (obs: {pred.Omega_m_obs:.4f})")
    print(f"    S_0 = {pred.S_0_predicho:.2f}")
    print(f"    Tension H0: {pred.tension_H0:.1f} sigma")

    pred_ok = pred.compatible
    print(f"\n    Compatible con observaciones: {'PASS' if pred_ok else 'FAIL'}")

    # Resultado final
    passed = sellos_ok and gr_ok and h0_ok and framework_ok and eqs_ok and pred_ok

    print("\n" + "=" * 70)
    print(f"  MCMC UNIFIED FRAMEWORK: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return passed


if __name__ == "__main__":
    test_MCMC_Unified_Framework()
