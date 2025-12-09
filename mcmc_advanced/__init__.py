"""
MCMC Advanced Validation Modules
================================

Modulos avanzados para validacion del modelo cosmologico MCMC
con correcciones ontologicas ECV (Lambda(z)) y MCV (Zhao gamma=0.51).

Modules:
--------
1. mcmc_isw_lss: ISW-LSS Cross-Correlation C_ell^Tg
2. mcmc_cmb_lensing: CMB Lensing C_L^phiphi
3. mcmc_desi_y3_real: DESI Year 3 BAO Real Data
4. mcmc_nbody_box100: N-body Box 100 h^-1Mpc with Cronos
5. mcmc_zoom_MW: Zoom-in Milky Way Subhalo Analysis
6. mcmc_jwst_highz: JWST High-z Galaxy Comparison

Parameters:
-----------
ECV: epsilon=0.012, z_trans=8.9, Delta_z=1.5
MCV: gamma_zhao=0.51, alpha=2, beta=3
Cronos: alpha_cronos=0.15, eta_friction=0.05

Usage:
------
>>> from mcmc_advanced import ISWLSS_MCMC, CMBLensing_MCMC
>>> isw = ISWLSS_MCMC()
>>> result = isw.validate()
"""

__version__ = "1.0.0"
__author__ = "MCMC Cosmology Team"

# ISW-LSS Cross-Correlation
from .mcmc_isw_lss import (
    ISWLSS_MCMC,
    test_ISW_LSS_MCMC
)

# CMB Lensing
from .mcmc_cmb_lensing import (
    CMBLensing_MCMC,
    test_CMB_Lensing_MCMC
)

# DESI Y3 Real Data
from .mcmc_desi_y3_real import (
    DESI_Y3_MCMC,
    PuntoDESI,
    DESI_Y3_DATA,
    test_DESI_Y3_MCMC
)

# N-body Box-100 Cronos
from .mcmc_nbody_box100 import (
    CronosBox100,
    PerfilesCronos,
    test_NBody_Box100
)

# Zoom-in MW Subhalos
from .mcmc_zoom_MW import (
    ZoomMW_Cronos,
    test_Zoom_MW_Cronos
)

# JWST High-z
from .mcmc_jwst_highz import (
    JWST_HighZ_MCMC,
    GalaxiaJWST,
    JWST_SAMPLE,
    test_JWST_HighZ_MCMC
)

# MCV-Black Holes (Burbujas Entropicas)
from .mcv_bh_calibrated import (
    MCV_AgujerosNegros,
    CategoriaAgujerosNegros,
    ParametrosBH,
    SelloOntologico,
    SELLOS,
    EJEMPLOS_CANONICOS,
    crear_BH_canonico,
    analizar_por_categorias,
    test_MCV_BH,
)

# Bubble Corrections (Transito de Fotones por Burbujas Temporales)
from .bubble_corrections import (
    BubbleCorrectionCalculator,
    SMap3D,
    TipoBurbuja,
    ParametrosBurbuja,
    ParametrosBubbleCorrection,
    BURBUJA_VIA_LACTEA,
    BURBUJA_VOID_TIPICO,
    BURBUJA_CUMULO,
    test_BubbleCorrections,
)

# GW Background (Fondo de Ondas Gravitacionales del Retroceso Entropico)
from .gw_background_mcmc import (
    GWBackgroundMCMC,
    ParametrosGW_MCMC,
    test_GW_Background_MCMC,
)

# First Principles Derivation (Derivacion de α y β)
from .first_principles_derivation import (
    DerivacionPrimerosPrincipios,
    derivar_alpha_primera_principios,
    derivar_beta_primera_principios,
    parametros_actualizados_MCV_BH,
    test_first_principles_derivation,
)

# GW Mergers (Fusiones de Objetos Compactos - LIGO/Virgo)
from .gw_mergers_mcmc import (
    GWMergersMCMC,
    ParametrosGW_Mergers,
    TipoFusion,
    EventoGW,
    EVENTOS_LIGO,
    predicciones_detectores_futuros,
    test_GW_Mergers_MCMC,
)

# All exported classes
__all__ = [
    # ISW-LSS
    'ISWLSS_MCMC',
    'test_ISW_LSS_MCMC',
    # CMB Lensing
    'CMBLensing_MCMC',
    'test_CMB_Lensing_MCMC',
    # DESI Y3
    'DESI_Y3_MCMC',
    'PuntoDESI',
    'DESI_Y3_DATA',
    'test_DESI_Y3_MCMC',
    # N-body Cronos
    'CronosBox100',
    'PerfilesCronos',
    'test_NBody_Box100',
    # Zoom MW
    'ZoomMW_Cronos',
    'test_Zoom_MW_Cronos',
    # JWST High-z
    'JWST_HighZ_MCMC',
    'GalaxiaJWST',
    'JWST_SAMPLE',
    'test_JWST_HighZ_MCMC',
    # MCV-Black Holes
    'MCV_AgujerosNegros',
    'CategoriaAgujerosNegros',
    'ParametrosBH',
    'SelloOntologico',
    'SELLOS',
    'EJEMPLOS_CANONICOS',
    'crear_BH_canonico',
    'analizar_por_categorias',
    'test_MCV_BH',
    # Bubble Corrections
    'BubbleCorrectionCalculator',
    'SMap3D',
    'TipoBurbuja',
    'ParametrosBurbuja',
    'ParametrosBubbleCorrection',
    'BURBUJA_VIA_LACTEA',
    'BURBUJA_VOID_TIPICO',
    'BURBUJA_CUMULO',
    'test_BubbleCorrections',
    # GW Background
    'GWBackgroundMCMC',
    'ParametrosGW_MCMC',
    'test_GW_Background_MCMC',
    # First Principles Derivation
    'DerivacionPrimerosPrincipios',
    'derivar_alpha_primera_principios',
    'derivar_beta_primera_principios',
    'parametros_actualizados_MCV_BH',
    'test_first_principles_derivation',
    # GW Mergers
    'GWMergersMCMC',
    'ParametrosGW_Mergers',
    'TipoFusion',
    'EventoGW',
    'EVENTOS_LIGO',
    'predicciones_detectores_futuros',
    'test_GW_Mergers_MCMC',
]


def run_all_validations(verbose: bool = True) -> dict:
    """
    Execute all validation tests for the 11 advanced modules.

    Returns:
        dict: Results for each module with pass/fail status
    """
    results = {}

    if verbose:
        print("=" * 60)
        print("MCMC Advanced Validation Suite v{}".format(__version__))
        print("=" * 60)

    # 1. ISW-LSS
    if verbose:
        print("\n[1/11] ISW-LSS Cross-Correlation...")
    try:
        passed = test_ISW_LSS_MCMC()
        results['isw_lss'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['isw_lss'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 2. CMB Lensing
    if verbose:
        print("\n[2/11] CMB Lensing C_L^phiphi...")
    try:
        passed = test_CMB_Lensing_MCMC()
        results['cmb_lensing'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['cmb_lensing'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 3. DESI Y3
    if verbose:
        print("\n[3/11] DESI Y3 Real Data...")
    try:
        passed = test_DESI_Y3_MCMC()
        results['desi_y3'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['desi_y3'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 4. N-body Cronos
    if verbose:
        print("\n[4/11] N-body Box-100 Cronos...")
    try:
        passed = test_NBody_Box100()
        results['nbody_cronos'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['nbody_cronos'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 5. Zoom MW
    if verbose:
        print("\n[5/11] Zoom-in MW Subhalos...")
    try:
        passed = test_Zoom_MW_Cronos()
        results['zoom_mw'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['zoom_mw'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 6. JWST High-z
    if verbose:
        print("\n[6/11] JWST High-z Galaxies...")
    try:
        passed = test_JWST_HighZ_MCMC()
        results['jwst_highz'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['jwst_highz'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 7. MCV-Black Holes (Burbujas Entropicas)
    if verbose:
        print("\n[7/11] MCV-Black Holes (Burbujas Entropicas)...")
    try:
        passed = test_MCV_BH()
        results['mcv_bh'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['mcv_bh'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 8. Bubble Corrections (Transito de Fotones)
    if verbose:
        print("\n[8/11] Bubble Corrections (Transito de Fotones)...")
    try:
        passed = test_BubbleCorrections()
        results['bubble_corrections'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['bubble_corrections'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 9. GW Background (Fondo de Ondas Gravitacionales)
    if verbose:
        print("\n[9/11] GW Background (Ondas Gravitacionales)...")
    try:
        passed = test_GW_Background_MCMC()
        results['gw_background'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['gw_background'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 10. First Principles Derivation (Derivacion de α y β)
    if verbose:
        print("\n[10/11] First Principles Derivation (α, β)...")
    try:
        passed = test_first_principles_derivation()
        results['first_principles'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['first_principles'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # 11. GW Mergers (Fusiones LIGO/Virgo)
    if verbose:
        print("\n[11/11] GW Mergers (LIGO/Virgo)...")
    try:
        passed = test_GW_Mergers_MCMC()
        results['gw_mergers'] = {'passed': passed}
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"      {status}")
    except Exception as e:
        results['gw_mergers'] = {'passed': False, 'error': str(e)}
        if verbose:
            print(f"      ERROR: {e}")

    # Summary
    n_passed = sum(1 for r in results.values() if r.get('passed', False))
    total = len(results)

    if verbose:
        print("\n" + "=" * 60)
        print(f"RESULTS: {n_passed}/{total} validations passed")
        print("=" * 60)

    results['summary'] = {
        'passed': n_passed,
        'total': total,
        'all_passed': n_passed == total
    }

    return results


if __name__ == "__main__":
    run_all_validations(verbose=True)
