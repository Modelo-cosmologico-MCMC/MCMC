#!/usr/bin/env python3
"""
================================================================================
MÓDULO ISW-LSS CROSS-CORRELATION PARA MCMC
================================================================================

Calcula el espectro cruzado C_ℓ^Tg entre CMB (efecto ISW tardío) y
trazadores LSS (galaxias, QSOs).

Fundamentación Ontológica:
--------------------------
El efecto Sachs-Wolfe Integrado (ISW) tardío surge de la evolución temporal
de los potenciales gravitatorios:

    (ΔT/T)_ISW ∝ ∫[η_⋆ → η₀] dη · d/dη(Φ + Ψ)

En el MCMC:
- La ECV emerge dinámicamente mediante ρ_id(z)
- La transición suave en z_trans ≈ 8.9 modifica la amplitud ISW
- El canal latente ρ_lat produce un ISW ligeramente diferente

Predicción MCMC:
- Si ε > 0: ISW tardío reducido (potenciales más estables)
- Correlación CMB×LSS discrimina entre escenarios

Autor: Modelo MCMC
Copyright (c) 2024
================================================================================
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTES Y PARÁMETROS
# =============================================================================

# Constantes físicas
C_LIGHT = 299792.458  # km/s

# Parámetros MCMC calibrados
MCMC_PARAMS = {
    'epsilon': 0.012,
    'z_trans': 8.9,
    'Delta_z': 1.5,
    'H0': 67.4,
    'Omega_m': 0.315,
    'sigma8': 0.811,
    'n_s': 0.965,
}


# =============================================================================
# CLASE PRINCIPAL ISW-LSS
# =============================================================================

@dataclass
class ParametrosISWLSS:
    """Parámetros para correlación ISW-LSS."""
    H0: float = 67.4
    Omega_m: float = 0.315
    epsilon: float = 0.012
    z_trans: float = 8.9
    Delta_z: float = 1.5
    sigma8: float = 0.811
    n_s: float = 0.965


PARAMS_ISW = ParametrosISWLSS()


class ISWLSS_MCMC:
    """
    Correlación cruzada ISW-galaxias para el modelo MCMC.

    Física clave:
    - ISW: d(Φ+Ψ)/dη a z < 1
    - LSS: bias b(z) × δ_m(z)

    El MCMC modifica la evolución de potenciales via ρ_id(z).
    """

    def __init__(self, params: ParametrosISWLSS = None):
        """
        Inicializa el módulo ISW-LSS.

        Args:
            params: Parámetros cosmológicos y MCMC
        """
        p = params or PARAMS_ISW
        self.H0 = p.H0
        self.Omega_m = p.Omega_m
        self.Omega_Lambda = 1 - p.Omega_m
        self.epsilon = p.epsilon
        self.z_trans = p.z_trans
        self.Delta_z = p.Delta_z
        self.sigma8 = p.sigma8
        self.n_s = p.n_s
        self.c = C_LIGHT

        # Cache de distancias
        self._chi_cache = {}

    def Lambda_rel(self, z: float) -> float:
        """
        Λ_rel(z) = 1 + ε·tanh((z_trans - z)/Δz)

        Modificación ECV de la constante cosmológica.
        """
        return 1 + self.epsilon * np.tanh((self.z_trans - z) / self.Delta_z)

    def E(self, z: float) -> float:
        """
        E(z) = H(z)/H₀ para MCMC.

        Incluye Λ_rel que modifica la energía oscura.
        """
        Lambda_z = self.Lambda_rel(z)
        Omega_DE_z = self.Omega_Lambda * Lambda_z
        return np.sqrt(self.Omega_m * (1 + z)**3 + Omega_DE_z)

    def H(self, z: float) -> float:
        """H(z) en km/s/Mpc."""
        return self.H0 * self.E(z)

    def chi(self, z: float) -> float:
        """
        Distancia comóvil χ(z) en Mpc.

        χ(z) = c ∫₀^z dz'/H(z')
        """
        if z in self._chi_cache:
            return self._chi_cache[z]

        integrand = lambda zp: self.c / self.H(zp)
        result, _ = quad(integrand, 0, z, limit=100)

        self._chi_cache[z] = result
        return result

    def D_growth_MCMC(self, z: float) -> float:
        """
        Factor de crecimiento D(z) con modulación MCMC.

        La ECV suprime ligeramente el crecimiento a z < 1.
        Usa aproximación de Carroll et al. (1992) con corrección MCMC.
        """
        # Ω_m(z) y Ω_Λ(z)
        E_z = self.E(z)
        Om_z = self.Omega_m * (1 + z)**3 / E_z**2
        OL_z = 1 - Om_z

        # Fórmula de Carroll
        g = 2.5 * Om_z / (Om_z**(4/7) - OL_z + (1 + Om_z/2) * (1 + OL_z/70))

        # Normalizar a z=0
        Om_0 = self.Omega_m
        OL_0 = self.Omega_Lambda
        g0 = 2.5 * Om_0 / (Om_0**(4/7) - OL_0 + (1 + Om_0/2) * (1 + OL_0/70))

        # Modulación MCMC: supresión del crecimiento por ECV
        # El canal latente reduce ligeramente el crecimiento
        modulation = 1 - 0.3 * self.epsilon * np.exp(-z)

        return (g / g0) / (1 + z) * modulation

    def dPhi_deta(self, z: float) -> float:
        """
        Derivada del potencial gravitatorio respecto a tiempo conforme.

        dΦ/dη ∝ (1 - 3w_eff)/(1+z) × H × Ω_DE(z) × D(z)

        En MCMC, la evolución de Φ depende de ρ_id(z).
        """
        # w efectivo ligeramente diferente de -1 debido a ECV
        w_eff = -1 + 0.01 * self.epsilon

        # Fracción de energía oscura
        E_z = self.E(z)
        Omega_DE_z = 1 - self.Omega_m * (1 + z)**3 / E_z**2

        # Derivada del potencial
        dPhi = (1 - 3*w_eff) * self.H(z) * Omega_DE_z * self.D_growth_MCMC(z) / (1 + z)

        return dPhi

    def W_ISW(self, z: float, ell: int) -> float:
        """
        Kernel ISW para multipolo ℓ.

        W_ISW(z) ∝ d(Φ+Ψ)/dη

        El factor 2 viene de Φ + Ψ = 2Φ (sin anisotropic stress).
        """
        chi_z = self.chi(z)
        if chi_z < 1e-6:
            return 0.0

        # Derivada del potencial
        dPhi = self.dPhi_deta(z)

        # Normalización: 2/(c²) convierte a ΔT/T
        return 2 * dPhi / (self.c**2)

    def W_gal(self, z: float, z_mean: float = 0.5,
              sigma_z: float = 0.1, bias: float = 1.5) -> float:
        """
        Kernel de galaxias (distribución Gaussiana).

        W_g(z) = b(z) × dN/dz

        Args:
            z: Redshift
            z_mean: Redshift medio del survey
            sigma_z: Dispersión en redshift
            bias: Bias lineal de galaxias
        """
        # Distribución de redshift (Gaussiana)
        dN_dz = np.exp(-0.5 * ((z - z_mean) / sigma_z)**2)
        dN_dz /= (sigma_z * np.sqrt(2*np.pi))

        # Bias lineal (simplificado, podría depender de z)
        b_z = bias

        return b_z * dN_dz

    def P_matter(self, k: float, z: float) -> float:
        """
        Espectro de potencia de materia P(k,z).

        P(k) ∝ k^n_s × T²(k) × D²(z)

        Usa transfer function BBKS simplificada.
        """
        # Escala de igualdad materia-radiación
        k_eq = 0.01  # h/Mpc aproximado

        # Transfer function BBKS
        q = k / k_eq
        if q < 1e-10:
            T_k = 1.0
        else:
            T_k = np.log(1 + 2.34*q) / (2.34*q)
            T_k *= (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)

        # Factor de crecimiento
        D_z = self.D_growth_MCMC(z)

        # Espectro primordial × transfer × crecimiento
        # Normalización con σ₈
        P_k = (k / 0.05)**self.n_s * T_k**2 * D_z**2
        P_k *= (self.sigma8 / 0.811)**2 * 1e4  # Normalización aproximada

        return P_k

    def C_Tg_ell(self, ell: int, z_gal: float = 0.5,
                 sigma_z: float = 0.1, bias: float = 1.5) -> float:
        """
        Espectro cruzado ISW-galaxias C_ℓ^Tg.

        C_ℓ^Tg = ∫ dz/χ² × W_ISW(z) × W_g(z) × P(k=ℓ/χ, z)

        Usa aproximación de Limber.

        Args:
            ell: Multipolo
            z_gal: Redshift medio del survey de galaxias
            sigma_z: Dispersión en redshift
            bias: Bias de galaxias

        Returns:
            C_ℓ^Tg en unidades de (μK × sr)
        """
        # Rango de integración en z
        z_min = max(0.01, z_gal - 4*sigma_z)
        z_max = min(3.0, z_gal + 4*sigma_z)
        z_array = np.linspace(z_min, z_max, 100)

        integrand = []
        for z in z_array:
            chi_z = self.chi(z)
            if chi_z < 1:
                integrand.append(0)
                continue

            # Aproximación de Limber: k = ℓ/χ
            k = ell / chi_z

            # Espectro de potencia
            P_k = self.P_matter(k, z)

            # Kernels
            W_I = self.W_ISW(z, ell)
            W_g = self.W_gal(z, z_gal, sigma_z, bias)

            # dχ/dz = c/H(z)
            dchi_dz = self.c / self.H(z)

            integrand.append(W_I * W_g * P_k * dchi_dz / chi_z**2)

        return simpson(integrand, x=z_array)

    def compute_spectrum(self, ell_range: np.ndarray = None,
                        z_gal: float = 0.5, sigma_z: float = 0.1,
                        bias: float = 1.5) -> Dict:
        """
        Calcula espectro C_ℓ^Tg completo.

        Args:
            ell_range: Array de multipolos
            z_gal: Redshift medio del survey
            sigma_z: Dispersión
            bias: Bias de galaxias

        Returns:
            Dict con ell, C_Tg, y parámetros
        """
        if ell_range is None:
            ell_range = np.arange(2, 100)

        C_Tg = [self.C_Tg_ell(ell, z_gal, sigma_z, bias) for ell in ell_range]

        return {
            'ell': ell_range,
            'C_Tg': np.array(C_Tg),
            'z_gal': z_gal,
            'sigma_z': sigma_z,
            'bias': bias,
            'epsilon': self.epsilon,
            'z_trans': self.z_trans
        }

    def compare_LCDM(self, ell_range: np.ndarray = None,
                    z_gal: float = 0.5, sigma_z: float = 0.1,
                    bias: float = 1.5) -> Dict:
        """
        Compara ISW-LSS para MCMC vs ΛCDM.

        Returns:
            Dict con espectros y ratios
        """
        if ell_range is None:
            ell_range = np.arange(2, 100)

        # ΛCDM (ε = 0)
        epsilon_orig = self.epsilon
        self.epsilon = 0
        C_Tg_LCDM = [self.C_Tg_ell(ell, z_gal, sigma_z, bias) for ell in ell_range]

        # MCMC
        self.epsilon = epsilon_orig
        C_Tg_MCMC = [self.C_Tg_ell(ell, z_gal, sigma_z, bias) for ell in ell_range]

        C_LCDM = np.array(C_Tg_LCDM)
        C_MCMC = np.array(C_Tg_MCMC)

        # Evitar división por cero
        ratio = np.where(np.abs(C_LCDM) > 1e-30, C_MCMC / C_LCDM, 1.0)

        return {
            'ell': ell_range,
            'C_Tg_LCDM': C_LCDM,
            'C_Tg_MCMC': C_MCMC,
            'ratio': ratio,
            'delta_percent': (ratio - 1) * 100,
            'z_gal': z_gal,
            'epsilon': self.epsilon
        }

    def significance_forecast(self, z_gal_array: List[float] = None,
                             f_sky: float = 0.4,
                             n_gal: float = 1e-3) -> Dict:
        """
        Pronóstico de significancia para detección MCMC vs ΛCDM.

        Args:
            z_gal_array: Redshifts de los surveys
            f_sky: Fracción de cielo
            n_gal: Densidad de galaxias (Mpc⁻³)

        Returns:
            Dict con significancia esperada por bin de z
        """
        if z_gal_array is None:
            z_gal_array = [0.5, 1.0, 1.5]

        results = []
        for z_gal in z_gal_array:
            comparison = self.compare_LCDM(z_gal=z_gal)

            # Diferencia promedio en el rango ℓ = 10-50 (máxima señal ISW)
            mask = (comparison['ell'] >= 10) & (comparison['ell'] <= 50)
            delta_mean = np.mean(np.abs(comparison['delta_percent'][mask]))

            # Error estadístico aproximado (cosmic variance dominado)
            ell_eff = 30
            sigma_CV = np.sqrt(2 / ((2*ell_eff + 1) * f_sky))

            # Significancia
            significance = delta_mean / (sigma_CV * 100)

            results.append({
                'z_gal': z_gal,
                'delta_percent_mean': delta_mean,
                'sigma_CV': sigma_CV * 100,
                'significance_sigma': significance,
                'detectable': significance > 1.0
            })

        return {
            'forecasts': results,
            'f_sky': f_sky,
            'summary': f"Máxima sensibilidad en z ~ {z_gal_array[np.argmax([r['significance_sigma'] for r in results])]}"
        }


# =============================================================================
# DATOS OBSERVACIONALES
# =============================================================================

# Trazadores LSS típicos para correlación ISW
LSS_TRACERS = {
    'SDSS_LRG': {'z_mean': 0.35, 'sigma_z': 0.08, 'bias': 2.0, 'f_sky': 0.25},
    'SDSS_QSO': {'z_mean': 1.5, 'sigma_z': 0.3, 'bias': 2.5, 'f_sky': 0.25},
    '2MASS': {'z_mean': 0.07, 'sigma_z': 0.02, 'bias': 1.2, 'f_sky': 0.65},
    'NVSS': {'z_mean': 1.0, 'sigma_z': 0.5, 'bias': 1.8, 'f_sky': 0.75},
    'WISE': {'z_mean': 0.3, 'sigma_z': 0.15, 'bias': 1.4, 'f_sky': 0.70},
}


# =============================================================================
# FUNCIÓN DE TEST
# =============================================================================

def test_ISW_LSS_MCMC():
    """Test completo del módulo ISW-LSS."""
    print("\n" + "="*65)
    print("  TEST ISW-LSS CROSS-CORRELATION MCMC")
    print("="*65)

    # Crear instancia
    isw = ISWLSS_MCMC()

    # Test 1: Funciones básicas
    print("\n[1] Funciones cosmológicas básicas:")
    print(f"    E(z=0) = {isw.E(0):.4f}")
    print(f"    E(z=1) = {isw.E(1):.4f}")
    print(f"    χ(z=1) = {isw.chi(1):.2f} Mpc")
    print(f"    D(z=0) = {isw.D_growth_MCMC(0):.4f}")
    print(f"    D(z=1) = {isw.D_growth_MCMC(1):.4f}")

    # Test 2: Espectro C_ℓ^Tg
    print("\n[2] Espectro cruzado C_ℓ^Tg:")
    ell_test = [10, 30, 50, 100]
    for ell in ell_test:
        C_Tg = isw.C_Tg_ell(ell, z_gal=0.5)
        print(f"    C_{ell}^Tg = {C_Tg:.4e}")

    # Test 3: Comparación MCMC vs ΛCDM
    print("\n[3] Comparación MCMC vs ΛCDM:")
    comparison = isw.compare_LCDM(np.array([10, 30, 50]), z_gal=0.5)
    for i, ell in enumerate([10, 30, 50]):
        ratio = comparison['ratio'][i]
        delta = comparison['delta_percent'][i]
        print(f"    ℓ={ell}: ratio = {ratio:.4f}, δ = {delta:.2f}%")

    # Test 4: Pronóstico de significancia
    print("\n[4] Pronóstico de significancia:")
    forecast = isw.significance_forecast([0.5, 1.0, 1.5])
    for f in forecast['forecasts']:
        print(f"    z={f['z_gal']}: δ={f['delta_percent_mean']:.2f}%, "
              f"σ={f['significance_sigma']:.2f}")

    # Verificar criterio de éxito
    print("\n[5] Verificación de criterios:")
    comp = isw.compare_LCDM(np.arange(10, 60), z_gal=0.5)
    ratio_mean = np.mean(comp['ratio'])

    passed = 0.9 <= ratio_mean <= 1.0
    print(f"    C_Tg^MCMC / C_Tg^ΛCDM = {ratio_mean:.4f}")
    print(f"    Criterio [0.9, 1.0]: {'PASS' if passed else 'FAIL'}")

    print("\n" + "="*65)
    print(f"  ISW-LSS MODULE: {'PASS' if passed else 'FAIL'}")
    print("="*65)

    return passed


if __name__ == "__main__":
    test_ISW_LSS_MCMC()
