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
]


def run_all_validations(verbose: bool = True) -> dict:
    """
    Execute all validation tests for the 6 advanced modules.

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
        print("\n[1/6] ISW-LSS Cross-Correlation...")
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
        print("\n[2/6] CMB Lensing C_L^phiphi...")
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
        print("\n[3/6] DESI Y3 Real Data...")
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
        print("\n[4/6] N-body Box-100 Cronos...")
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
        print("\n[5/6] Zoom-in MW Subhalos...")
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
        print("\n[6/6] JWST High-z Galaxies...")
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
