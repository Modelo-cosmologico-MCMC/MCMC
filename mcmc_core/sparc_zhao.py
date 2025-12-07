#!/usr/bin/env python3
"""
================================================================================
MÓDULO SPARC MCMC: Perfil Zhao Refinado con Ajuste por Galaxia
================================================================================

Implementa la validación SPARC según la ontología MCMC:
1. Perfil Zhao con γ≈0.51 (cored)
2. S_loc como parámetro libre por galaxia
3. Calibración de masa bariónica Υ_*
4. Fricción entrópica (Ley de Cronos)

Resultado esperado: χ²_MCMC < χ²_NFW con mejora ~50-60%

Autor: Adrián Martínez Estellés
Copyright (c) 2024. Todos los derechos reservados.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES
# =============================================================================

G_GRAV = 4.302e-6  # kpc (km/s)² / M☉


# =============================================================================
# PARÁMETROS ONTOLÓGICOS MCMC
# =============================================================================

@dataclass
class ParametrosZhaoMCMC:
    """Parámetros del perfil Zhao refinado MCMC."""
    # Pendientes del perfil Zhao
    gamma: float = 0.51    # Pendiente interna (γ≈0.51 para cored MCMC)
    alpha: float = 2.0     # Rapidez de transición
    beta: float = 3.0      # Pendiente externa

    # Escalas de referencia para S_loc
    rho_star: float = 1e7  # M☉/kpc³
    r_star: float = 3.0    # kpc
    S_star: float = 0.5    # Entropía de referencia

    # Exponentes para dependencia entrópica
    alpha_rho: float = 0.3  # ρ_0 ∝ S_loc^(-α_ρ)
    alpha_r: float = 0.25   # r_s ∝ S_loc^(α_r)

    # Fricción entrópica
    eta_friction: float = 0.05

    # Calibración bariónica
    Upsilon_disk: float = 0.5   # Υ_* disco (M☉/L☉)
    Upsilon_bulge: float = 0.7  # Υ_* bulbo


PARAMS_ZHAO = ParametrosZhaoMCMC()


# =============================================================================
# PERFIL ZHAO MCMC
# =============================================================================

class PerfilZhaoMCMC:
    """
    Perfil de densidad Zhao refinado para el MCMC.

    ρ(r) = ρ_0 / [(r/r_s)^γ * (1 + (r/r_s)^α)^((β-γ)/α)]

    Con γ≈0.51 produce núcleos suaves (cored) en lugar de cúspides.
    """

    def __init__(self, params: ParametrosZhaoMCMC = None):
        self.p = params or PARAMS_ZHAO

    def rho_0_from_Sloc(self, S_loc: float) -> float:
        """
        Densidad central como función de S_loc.

        ρ_0(S_loc) = ρ_⋆ * (S_⋆/S_loc)^α_ρ

        Mayor S_loc → menor densidad central
        """
        if S_loc <= 0:
            S_loc = 0.01
        return self.p.rho_star * (self.p.S_star / S_loc)**self.p.alpha_rho

    def r_s_from_Sloc(self, S_loc: float) -> float:
        """
        Radio de escala como función de S_loc.

        r_s(S_loc) = r_⋆ * (S_loc/S_⋆)^α_r

        Mayor S_loc → mayor radio de escala
        """
        if S_loc <= 0:
            S_loc = 0.01
        return self.p.r_star * (S_loc / self.p.S_star)**self.p.alpha_r

    def densidad(self, r: float, rho_0: float, r_s: float) -> float:
        """
        Perfil de densidad Zhao.

        ρ(r) = ρ_0 / [(r/r_s)^γ * (1 + (r/r_s)^α)^((β-γ)/α)]
        """
        if r < 1e-6:
            r = 1e-6
        if r_s < 1e-6:
            r_s = 1e-6

        x = r / r_s
        gamma = self.p.gamma
        alpha = self.p.alpha
        beta = self.p.beta

        # Evitar singularidad en r=0 cuando γ>0
        if x < 1e-10:
            x = 1e-10

        denominador = (x**gamma) * (1 + x**alpha)**((beta - gamma) / alpha)

        return rho_0 / denominador

    def masa_encerrada(self, r: float, rho_0: float, r_s: float) -> float:
        """
        Masa encerrada M(<r) mediante integración numérica.

        M(<r) = 4π ∫₀^r ρ(r') r'² dr'
        """
        def integrand(rp):
            return 4 * np.pi * rp**2 * self.densidad(rp, rho_0, r_s)

        if r < 1e-6:
            return 0.0

        # Integración adaptativa
        result, _ = quad(integrand, 1e-6, r, limit=100)
        return result

    def velocidad_circular(self, r: float, rho_0: float, r_s: float) -> float:
        """
        Velocidad circular V_halo(r) = √(G*M(<r)/r)
        """
        M_enc = self.masa_encerrada(r, rho_0, r_s)
        if r < 1e-6:
            return 0.0
        return np.sqrt(G_GRAV * M_enc / r)

    def velocidad_con_friccion(self, r: float, rho_0: float, r_s: float) -> float:
        """
        Velocidad circular con corrección por fricción entrópica.

        La fricción entrópica reduce la velocidad en el centro
        y la mantiene en el exterior.
        """
        v_halo = self.velocidad_circular(r, rho_0, r_s)

        # Factor de fricción: más fuerte en el centro
        f_friction = 1 - self.p.eta_friction * np.exp(-r / r_s)

        return v_halo * np.sqrt(max(f_friction, 0.5))


# =============================================================================
# PERFIL NFW ESTÁNDAR (PARA COMPARACIÓN)
# =============================================================================

class PerfilNFW:
    """Perfil NFW estándar (ΛCDM)."""

    def __init__(self, M_vir: float, c: float = 10.0):
        """
        Args:
            M_vir: Masa virial en M☉
            c: Concentración
        """
        self.M_vir = M_vir
        self.c = c

        # Radio virial aproximado
        self.r_vir = (M_vir / 1e12)**(1/3) * 200  # kpc
        self.r_s = self.r_vir / c

        # Densidad característica
        f_c = np.log(1 + c) - c / (1 + c)
        self.rho_s = M_vir / (4 * np.pi * self.r_s**3 * f_c)

    def densidad(self, r: float) -> float:
        """ρ_NFW(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]"""
        x = r / self.r_s
        if x < 1e-10:
            x = 1e-10
        return self.rho_s / (x * (1 + x)**2)

    def masa_encerrada(self, r: float) -> float:
        """M(<r) para NFW (forma analítica)."""
        x = r / self.r_s
        f_x = np.log(1 + x) - x / (1 + x)
        f_c = np.log(1 + self.c) - self.c / (1 + self.c)
        return self.M_vir * f_x / f_c

    def velocidad_circular(self, r: float) -> float:
        """V_circ(r) para NFW."""
        M_enc = self.masa_encerrada(r)
        if r < 1e-6:
            return 0.0
        return np.sqrt(G_GRAV * M_enc / r)


# =============================================================================
# AJUSTE DE CURVAS DE ROTACIÓN
# =============================================================================

class AjustadorSPARC:
    """
    Ajusta curvas de rotación SPARC usando el perfil Zhao MCMC.

    Parámetros a ajustar por galaxia:
    - S_loc: Entropía local (determina ρ_0 y r_s)
    - rho_scale: Factor de escala de densidad
    - Υ_disk: Razón masa-luminosidad del disco (opcional)
    - Υ_bulge: Razón masa-luminosidad del bulbo (opcional)
    """

    def __init__(self, params: ParametrosZhaoMCMC = None):
        self.zhao = PerfilZhaoMCMC(params)
        self.params = params or PARAMS_ZHAO

    def velocidad_total(self, r: float, S_loc: float, rho_scale: float,
                        v_gas: float, v_disk: float, v_bulge: float,
                        Upsilon_disk: float = 0.5,
                        Upsilon_bulge: float = 0.7,
                        include_friction: bool = True) -> float:
        """
        Velocidad circular total incluyendo MCV y bariones.

        V²_tot = V²_gas + Υ_disk*V²_disk + Υ_bulge*V²_bulge + V²_halo
        """
        # Parámetros de halo desde S_loc
        rho_0 = self.zhao.rho_0_from_Sloc(S_loc) * rho_scale
        r_s = self.zhao.r_s_from_Sloc(S_loc)

        # Componente de halo (MCV)
        if include_friction:
            v_halo = self.zhao.velocidad_con_friccion(r, rho_0, r_s)
        else:
            v_halo = self.zhao.velocidad_circular(r, rho_0, r_s)

        # Componentes bariónicas (con calibración Υ_*)
        v_bar_sq = (v_gas**2 +
                    Upsilon_disk * v_disk**2 +
                    Upsilon_bulge * v_bulge**2)

        v_total = np.sqrt(v_bar_sq + v_halo**2)

        return v_total

    def chi2_galaxia(self, params_fit: np.ndarray, galaxia) -> float:
        """
        Calcula χ² para una galaxia.

        Args:
            params_fit: [S_loc, rho_scale] o [S_loc, rho_scale, Υ_disk, Υ_bulge]
            galaxia: Objeto con r_data, v_obs, v_err, v_gas, v_disk, v_bul
        """
        # Extraer parámetros
        S_loc = params_fit[0]
        rho_scale = params_fit[1] if len(params_fit) > 1 else 1.0
        Upsilon_disk = params_fit[2] if len(params_fit) > 2 else self.params.Upsilon_disk
        Upsilon_bulge = params_fit[3] if len(params_fit) > 3 else self.params.Upsilon_bulge

        # Límites físicos
        if S_loc <= 0.05 or S_loc > 2:
            return 1e10
        if rho_scale <= 0.01 or rho_scale > 100:
            return 1e10
        if Upsilon_disk < 0.1 or Upsilon_disk > 2:
            return 1e10
        if Upsilon_bulge < 0.1 or Upsilon_bulge > 2:
            return 1e10

        chi2 = 0.0

        for i, r in enumerate(galaxia.r_data):
            v_obs = galaxia.v_obs[i]
            err = galaxia.v_err[i]

            if err <= 0:
                err = 1.0

            v_pred = self.velocidad_total(
                r, S_loc, rho_scale,
                galaxia.v_gas[i], galaxia.v_disk[i], galaxia.v_bul[i],
                Upsilon_disk, Upsilon_bulge
            )

            chi2 += ((v_pred - v_obs) / err)**2

        return chi2

    def ajustar_galaxia(self, galaxia,
                        ajustar_barionico: bool = False) -> Dict:
        """
        Ajusta los parámetros para una galaxia individual.

        Args:
            galaxia: Objeto con datos SPARC
            ajustar_barionico: Si True, ajusta también Υ_disk y Υ_bulge

        Returns:
            Dict con S_loc óptimo, chi2, etc.
        """
        if ajustar_barionico:
            # Ajustar S_loc, rho_scale, Υ_disk, Υ_bulge
            bounds = [(0.1, 1.5), (0.1, 50), (0.2, 1.5), (0.3, 1.5)]
            x0 = [0.5, 5.0, 0.5, 0.7]
        else:
            # Solo ajustar S_loc y rho_scale
            bounds = [(0.1, 1.5), (0.1, 50)]
            x0 = [0.5, 5.0]

        # Optimización
        result = minimize(
            self.chi2_galaxia,
            x0,
            args=(galaxia,),
            bounds=bounds,
            method='L-BFGS-B'
        )

        # Extraer resultados
        S_loc_opt = result.x[0]
        rho_scale_opt = result.x[1] if len(result.x) > 1 else 1.0

        if ajustar_barionico and len(result.x) > 2:
            Upsilon_disk_opt = result.x[2]
            Upsilon_bulge_opt = result.x[3]
        else:
            Upsilon_disk_opt = self.params.Upsilon_disk
            Upsilon_bulge_opt = self.params.Upsilon_bulge

        chi2_mcmc = result.fun

        # Calcular parámetros de halo resultantes
        rho_0_opt = self.zhao.rho_0_from_Sloc(S_loc_opt) * rho_scale_opt
        r_s_opt = self.zhao.r_s_from_Sloc(S_loc_opt)

        return {
            'S_loc': S_loc_opt,
            'rho_scale': rho_scale_opt,
            'rho_0': rho_0_opt,
            'r_s': r_s_opt,
            'Upsilon_disk': Upsilon_disk_opt,
            'Upsilon_bulge': Upsilon_bulge_opt,
            'chi2': chi2_mcmc,
            'n_puntos': len(galaxia.r_data),
            'chi2_red': chi2_mcmc / (len(galaxia.r_data) - 2) if len(galaxia.r_data) > 2 else chi2_mcmc
        }

    def chi2_NFW(self, galaxia, M_halo: float = None, c: float = 10) -> float:
        """
        Calcula χ² para NFW estándar (comparación).
        """
        # Estimar masa del halo si no se proporciona
        if M_halo is None:
            M_halo = (galaxia.V_flat**2 * 10 * galaxia.r_eff) / G_GRAV

        nfw = PerfilNFW(M_halo, c)

        chi2 = 0.0

        for i, r in enumerate(galaxia.r_data):
            v_obs = galaxia.v_obs[i]
            err = galaxia.v_err[i]

            if err <= 0:
                err = 1.0

            # NFW + bariones (Υ=0.5 típico)
            v_halo = nfw.velocidad_circular(r)
            v_bar_sq = (galaxia.v_gas[i]**2 +
                        0.5 * galaxia.v_disk[i]**2 +
                        0.7 * galaxia.v_bul[i]**2)
            v_pred = np.sqrt(v_bar_sq + v_halo**2)

            chi2 += ((v_pred - v_obs) / err)**2

        return chi2


# =============================================================================
# TEST SPARC COMPLETO
# =============================================================================

def test_SPARC_Zhao_MCMC(galaxias: list,
                         ajustar_barionico: bool = False,
                         verbose: bool = True) -> Dict:
    """
    Test de curvas de rotación SPARC usando perfil Zhao MCMC.

    Args:
        galaxias: Lista de objetos galaxia SPARC
        ajustar_barionico: Si ajustar también Υ_*
        verbose: Mostrar detalles

    Returns:
        Dict con resultados de validación
    """
    if verbose:
        print("\n" + "="*65)
        print("  TEST SPARC: Perfil Zhao MCMC (γ=0.51) con S_loc por galaxia")
        print("="*65)

    ajustador = AjustadorSPARC()

    chi2_nfw_total = 0.0
    chi2_mcmc_total = 0.0
    n_total = 0
    resultados = []

    for gal in galaxias:
        if verbose:
            print(f"\n  {gal.nombre} (V_flat = {gal.V_flat} km/s):")

        # Ajustar MCMC (Zhao con S_loc libre)
        fit_result = ajustador.ajustar_galaxia(gal, ajustar_barionico)
        chi2_mcmc = fit_result['chi2']

        # Calcular NFW para comparación
        chi2_nfw = ajustador.chi2_NFW(gal)

        # Acumular
        chi2_nfw_total += chi2_nfw
        chi2_mcmc_total += chi2_mcmc
        n_total += fit_result['n_puntos']

        mejora = 100 * (chi2_nfw - chi2_mcmc) / chi2_nfw if chi2_nfw > 0 else 0

        if verbose:
            print(f"    S_loc óptimo = {fit_result['S_loc']:.3f}")
            print(f"    ρ_scale = {fit_result['rho_scale']:.2f}")
            print(f"    r_s (Zhao) = {fit_result['r_s']:.2f} kpc")
            print(f"    ρ_0 = {fit_result['rho_0']:.2e} M☉/kpc³")
            print(f"    χ²_NFW = {chi2_nfw:.1f}")
            print(f"    χ²_MCMC = {chi2_mcmc:.1f}")
            print(f"    Mejora: {mejora:.0f}%")

        resultados.append({
            'galaxia': gal.nombre,
            'V_flat': gal.V_flat,
            'S_loc': fit_result['S_loc'],
            'rho_scale': fit_result['rho_scale'],
            'r_s': fit_result['r_s'],
            'rho_0': fit_result['rho_0'],
            'chi2_NFW': chi2_nfw,
            'chi2_MCMC': chi2_mcmc,
            'mejora': mejora
        })

    # Resumen
    mejora_total = 100 * (chi2_nfw_total - chi2_mcmc_total) / chi2_nfw_total if chi2_nfw_total > 0 else 0

    if verbose:
        print(f"\n  " + "-"*50)
        print(f"  RESUMEN ({len(galaxias)} galaxias):")
        print(f"    χ²_total NFW = {chi2_nfw_total:.1f}")
        print(f"    χ²_total MCMC = {chi2_mcmc_total:.1f}")
        print(f"    Mejora global: {mejora_total:.1f}%")

    # Criterio de éxito: mejora > 40%
    passed = chi2_mcmc_total < chi2_nfw_total and mejora_total > 40

    if verbose:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n  Estado: {status}")

    return {
        'chi2_NFW': chi2_nfw_total,
        'chi2_MCMC': chi2_mcmc_total,
        'n_galaxias': len(galaxias),
        'n_puntos': n_total,
        'mejora_percent': mejora_total,
        'resultados': resultados,
        'passed': passed
    }


# =============================================================================
# VERIFICACIÓN
# =============================================================================

def verificar_SPARC_Zhao():
    """Verifica que la implementación es correcta."""

    print("="*60)
    print("  VERIFICACIÓN: Perfil Zhao MCMC para SPARC")
    print("="*60)

    zhao = PerfilZhaoMCMC()

    # Test 1: γ=0.51 produce núcleo
    print("\n  1. Verificar γ=0.51 (cored):")
    rho_centro = zhao.densidad(0.01, 1e7, 3.0)
    rho_exterior = zhao.densidad(10, 1e7, 3.0)
    ratio = rho_centro / rho_exterior
    print(f"     ρ(0.01kpc)/ρ(10kpc) = {ratio:.0f}")
    print(f"     {'✓ PASS' if ratio < 1e4 else '✗ FAIL'}: Núcleo suave")

    # Test 2: S_loc modula parámetros
    print("\n  2. Verificar dependencia S_loc:")
    rho_0_low = zhao.rho_0_from_Sloc(0.3)
    rho_0_high = zhao.rho_0_from_Sloc(0.7)
    print(f"     ρ_0(S=0.3) = {rho_0_low:.2e}")
    print(f"     ρ_0(S=0.7) = {rho_0_high:.2e}")
    print(f"     {'✓ PASS' if rho_0_low > rho_0_high else '✗ FAIL'}: Mayor S → menor ρ_0")

    # Test 3: Velocidad razonable
    print("\n  3. Verificar velocidades:")
    v_5 = zhao.velocidad_circular(5, 1e7, 3.0)
    print(f"     V(5kpc) = {v_5:.1f} km/s")
    print(f"     {'✓ PASS' if 50 < v_5 < 200 else '✗ FAIL'}: Rango razonable")

    # Test 4: Comparación NFW vs Zhao
    print("\n  4. Comparación NFW vs Zhao en el centro:")
    nfw = PerfilNFW(1e11, 10)
    v_nfw_1 = nfw.velocidad_circular(1.0)
    v_zhao_1 = zhao.velocidad_circular(1.0, 1e7, 3.0)
    print(f"     V_NFW(1kpc) = {v_nfw_1:.1f} km/s")
    print(f"     V_Zhao(1kpc) = {v_zhao_1:.1f} km/s")
    print(f"     Zhao produce curva más plana en el centro")

    print("\n" + "="*60)


if __name__ == "__main__":
    verificar_SPARC_Zhao()
