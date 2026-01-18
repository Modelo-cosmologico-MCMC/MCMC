"""
Test Suite for MCMC Advanced Validation Modules
================================================

Tests all 6 advanced cosmological validation modules:
1. ISW-LSS Cross-Correlation
2. CMB Lensing C_L^phiphi
3. DESI Y3 Real Data
4. N-body Box-100 Cronos
5. Zoom-in MW Subhalos
6. JWST High-z Galaxies

Run with: pytest tests/test_mcmc_advanced.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestISWLSS:
    """Tests for ISW-LSS Cross-Correlation module."""

    def test_import(self):
        """Test module imports correctly."""
        from mcmc_advanced.mcmc_isw_lss import ISWLSS_MCMC
        isw = ISWLSS_MCMC()
        assert isw is not None

    def test_hubble_parameter(self):
        """Test E(z) Hubble parameter."""
        from mcmc_advanced.mcmc_isw_lss import ISWLSS_MCMC
        isw = ISWLSS_MCMC()
        # E(0) should be close to 1
        assert abs(isw.E_MCMC(0) - 1.0) < 0.02

    def test_growth_factor(self):
        """Test growth factor D(z)."""
        from mcmc_advanced.mcmc_isw_lss import ISWLSS_MCMC
        isw = ISWLSS_MCMC()
        # D(0) should be normalized, D(z>0) < D(0)
        D0 = isw.D_growth_MCMC(0)
        D1 = isw.D_growth_MCMC(1)
        assert D0 > D1  # Growth decreases at higher z

    def test_isw_window(self):
        """Test ISW window function."""
        from mcmc_advanced.mcmc_isw_lss import ISWLSS_MCMC
        isw = ISWLSS_MCMC()
        W = isw.W_ISW(0.5)
        assert isinstance(W, float)

    def test_cross_spectrum(self):
        """Test C_ell^Tg cross-correlation spectrum."""
        from mcmc_advanced.mcmc_isw_lss import ISWLSS_MCMC
        isw = ISWLSS_MCMC()
        C_Tg = isw.C_Tg_ell(100)
        assert C_Tg >= 0  # Should be non-negative

    def test_validation_criterion(self):
        """Test ISW-LSS validation: ratio in [0.9, 1.0]."""
        from mcmc_advanced.mcmc_isw_lss import test_ISW_LSS_MCMC
        passed = test_ISW_LSS_MCMC()
        assert passed, "ISW-LSS validation failed"


class TestCMBLensing:
    """Tests for CMB Lensing module."""

    def test_import(self):
        """Test module imports correctly."""
        from mcmc_advanced.mcmc_cmb_lensing import CMBLensing_MCMC
        lens = CMBLensing_MCMC()
        assert lens is not None

    def test_kappa_window(self):
        """Test convergence window function."""
        from mcmc_advanced.mcmc_cmb_lensing import CMBLensing_MCMC
        lens = CMBLensing_MCMC()
        W = lens.W_kappa(0.5)
        assert W >= 0

    def test_kappa_spectrum(self):
        """Test C_L^kappa kappa spectrum."""
        from mcmc_advanced.mcmc_cmb_lensing import CMBLensing_MCMC
        lens = CMBLensing_MCMC()
        C_kk = lens.C_kappa_L(100)
        assert C_kk >= 0

    def test_phi_spectrum(self):
        """Test C_L^phi phi lensing potential spectrum."""
        from mcmc_advanced.mcmc_cmb_lensing import CMBLensing_MCMC
        lens = CMBLensing_MCMC()
        C_phi = lens.C_phi_L(100)
        assert C_phi >= 0

    def test_A_lens(self):
        """Test A_lens parameter."""
        from mcmc_advanced.mcmc_cmb_lensing import CMBLensing_MCMC
        lens = CMBLensing_MCMC()
        A = lens.A_lens_parameter()
        # A_lens should be close to 1
        assert 0.95 < A < 1.05

    def test_validation_criterion(self):
        """Test CMB Lensing validation: |delta C_L| < 1%."""
        from mcmc_advanced.mcmc_cmb_lensing import test_CMB_Lensing_MCMC
        passed = test_CMB_Lensing_MCMC()
        assert passed, "CMB Lensing validation failed"


class TestDESIY3:
    """Tests for DESI Y3 Real Data module."""

    def test_import(self):
        """Test module imports correctly."""
        from mcmc_advanced.mcmc_desi_y3_real import DESI_Y3_MCMC, DESI_Y3_DATA
        desi = DESI_Y3_MCMC()
        assert desi is not None
        assert len(DESI_Y3_DATA) == 13

    def test_distance_measures(self):
        """Test DM, DH, DV distance calculations."""
        from mcmc_advanced.mcmc_desi_y3_real import DESI_Y3_MCMC
        desi = DESI_Y3_MCMC()
        z = 0.5
        DM = desi.DM_over_rd(z)
        DH = desi.DH_over_rd(z)
        DV = desi.DV_over_rd(z)
        assert DM > 0
        assert DH > 0
        assert DV > 0

    def test_chi2_calculation(self):
        """Test chi-squared calculation."""
        from mcmc_advanced.mcmc_desi_y3_real import DESI_Y3_MCMC
        desi = DESI_Y3_MCMC()
        chi2 = desi.chi2_MCMC()
        assert chi2 >= 0
        assert chi2 < 100  # Should be reasonable

    def test_validation_criterion(self):
        """Test DESI Y3 validation: chi2_MCMC <= chi2_LCDM."""
        from mcmc_advanced.mcmc_desi_y3_real import test_DESI_Y3_MCMC
        passed = test_DESI_Y3_MCMC()
        assert passed, "DESI Y3 validation failed"


class TestNBodyCronos:
    """Tests for N-body Box-100 Cronos module."""

    def test_import(self):
        """Test module imports correctly."""
        from mcmc_advanced.mcmc_nbody_box100 import CronosBox100, PerfilesCronos
        cronos = CronosBox100()
        perfiles = PerfilesCronos()
        assert cronos is not None
        assert perfiles is not None

    def test_lapse_function(self):
        """Test Cronos lapse function."""
        from mcmc_advanced.mcmc_nbody_box100 import CronosBox100
        cronos = CronosBox100()
        # At mean density, lapse should be > 1
        alpha = cronos.lapse_function(cronos.rho_crit)
        assert alpha > 1

    def test_friction(self):
        """Test entropic friction."""
        from mcmc_advanced.mcmc_nbody_box100 import CronosBox100
        cronos = CronosBox100()
        F = cronos.friction_cronos(100.0, cronos.rho_crit)  # v=100 km/s
        assert F < 0  # Friction opposes motion

    def test_zhao_profile(self):
        """Test Zhao profile with gamma=0.51."""
        from mcmc_advanced.mcmc_nbody_box100 import PerfilesCronos
        perfiles = PerfilesCronos()
        r = np.logspace(-1, 2, 50)  # 0.1 to 100 kpc
        rho_nfw = perfiles.NFW_profile(r, 1e12, 10)
        rho_zhao = perfiles.Zhao_profile(r, 1e12, 10)
        # At small r, Zhao should be less dense (cored)
        assert rho_zhao[0] < rho_nfw[0]

    def test_core_radius(self):
        """Test core radius calculation."""
        from mcmc_advanced.mcmc_nbody_box100 import PerfilesCronos
        perfiles = PerfilesCronos()
        r_core = perfiles.core_radius(1e12, 10)
        assert r_core > 0

    def test_validation_criterion(self):
        """Test N-body Cronos validation: r_core > 2 kpc."""
        from mcmc_advanced.mcmc_nbody_box100 import test_NBody_Box100
        passed = test_NBody_Box100()
        assert passed, "N-body validation failed"


class TestZoomMW:
    """Tests for Zoom-in MW Subhalo module."""

    def test_import(self):
        """Test module imports correctly."""
        from mcmc_advanced.mcmc_zoom_MW import ZoomMW_Cronos
        zoom = ZoomMW_Cronos()
        assert zoom is not None

    def test_subhalo_count(self):
        """Test expected subhalo count."""
        from mcmc_advanced.mcmc_zoom_MW import ZoomMW_Cronos
        zoom = ZoomMW_Cronos()
        result = zoom.expected_subhalos()
        assert result['N_CDM'] > result['N_Cronos']  # Should suppress
        assert result['suppression'] < 1.0

    def test_mass_function(self):
        """Test subhalo mass function."""
        from mcmc_advanced.mcmc_zoom_MW import ZoomMW_Cronos
        zoom = ZoomMW_Cronos()
        M_array = np.logspace(8, 11, 20)
        dN_dM_CDM, dN_dM_Cronos = zoom.subhalo_mass_function(M_array)
        # Cronos should suppress at low masses
        assert np.all(dN_dM_Cronos <= dN_dM_CDM)

    def test_vmax_reduction(self):
        """Test V_max reduction for TBTF."""
        from mcmc_advanced.mcmc_zoom_MW import ZoomMW_Cronos
        zoom = ZoomMW_Cronos()
        V_CDM = 50.0  # km/s
        V_Cronos = zoom.Vmax_reduction(V_CDM)
        assert V_Cronos < V_CDM

    def test_tbtf_test(self):
        """Test Too Big To Fail resolution."""
        from mcmc_advanced.mcmc_zoom_MW import ZoomMW_Cronos
        zoom = ZoomMW_Cronos()
        result = zoom.too_big_to_fail_test()
        assert result['resolved']

    def test_validation_criterion(self):
        """Test Zoom MW validation: N_sub in [40, 60]."""
        from mcmc_advanced.mcmc_zoom_MW import test_Zoom_MW_Cronos
        passed = test_Zoom_MW_Cronos()
        assert passed, "Zoom MW validation failed"


class TestJWSTHighz:
    """Tests for JWST High-z Galaxies module."""

    def test_import(self):
        """Test module imports correctly."""
        from mcmc_advanced.mcmc_jwst_highz import JWST_HighZ_MCMC, JWST_SAMPLE
        jwst = JWST_HighZ_MCMC()
        assert jwst is not None
        assert len(JWST_SAMPLE) >= 6

    def test_halo_mass_function(self):
        """Test halo mass function."""
        from mcmc_advanced.mcmc_jwst_highz import JWST_HighZ_MCMC
        jwst = JWST_HighZ_MCMC()
        M_array = np.logspace(10, 13, 20)
        n_M = jwst.halo_mass_function(M_array, z=10)
        assert np.all(n_M >= 0)

    def test_uv_luminosity_function(self):
        """Test UV luminosity function."""
        from mcmc_advanced.mcmc_jwst_highz import JWST_HighZ_MCMC
        jwst = JWST_HighZ_MCMC()
        M_UV = np.linspace(-22, -18, 10)
        phi = jwst.UV_luminosity_function(M_UV, z=10)
        assert np.all(phi >= 0)

    def test_mcmc_vs_lcdm_ratio(self):
        """Test MCMC vs LCDM abundance ratio."""
        from mcmc_advanced.mcmc_jwst_highz import JWST_HighZ_MCMC
        jwst = JWST_HighZ_MCMC()
        M_array = np.logspace(10, 13, 20)
        ratio = jwst.MCMC_vs_LCDM_ratio(M_array, z=10)
        # Ratio should be close to 1 but slightly less
        assert np.all(ratio > 0.8)
        assert np.all(ratio <= 1.0)

    def test_wdm_comparison(self):
        """Test WDM comparison."""
        from mcmc_advanced.mcmc_jwst_highz import JWST_HighZ_MCMC
        jwst = JWST_HighZ_MCMC()
        result = jwst.WDM_comparison(z=10)
        # MCMC should suppress less than WDM
        assert result['MCMC_suppression'] < result['WDM_suppression']

    def test_validation_criterion(self):
        """Test JWST High-z validation: ratio > 0.85."""
        from mcmc_advanced.mcmc_jwst_highz import test_JWST_HighZ_MCMC
        passed = test_JWST_HighZ_MCMC()
        assert passed, "JWST validation failed"


class TestIntegration:
    """Integration tests for all modules together."""

    def test_all_imports(self):
        """Test all modules can be imported from package."""
        from mcmc_advanced import (
            ISWLSS_MCMC,
            CMBLensing_MCMC,
            DESI_Y3_MCMC,
            CronosBox100,
            ZoomMW_Cronos,
            JWST_HighZ_MCMC
        )
        assert all([
            ISWLSS_MCMC,
            CMBLensing_MCMC,
            DESI_Y3_MCMC,
            CronosBox100,
            ZoomMW_Cronos,
            JWST_HighZ_MCMC
        ])

    def test_run_all_validations(self):
        """Test run_all_validations function."""
        from mcmc_advanced import run_all_validations
        results = run_all_validations(verbose=False)
        assert 'summary' in results
        assert results['summary']['total'] == 6

    def test_consistent_parameters(self):
        """Test that all modules use consistent MCMC parameters."""
        from mcmc_advanced import (
            ISWLSS_MCMC,
            CMBLensing_MCMC,
            DESI_Y3_MCMC
        )
        isw = ISWLSS_MCMC()
        lens = CMBLensing_MCMC()
        desi = DESI_Y3_MCMC()

        # Check epsilon consistency
        assert isw.epsilon == lens.epsilon == desi.epsilon == 0.012

        # Check z_trans consistency
        assert isw.z_trans == lens.z_trans == desi.z_trans == 1.0

    def test_all_validations_pass(self):
        """Test that all 6 validations pass their criteria."""
        from mcmc_advanced import run_all_validations
        results = run_all_validations(verbose=False)

        failed = []
        for name, result in results.items():
            if name != 'summary' and not result.get('passed', False):
                failed.append(name)

        assert len(failed) == 0, f"Validations failed: {failed}"
        assert results['summary']['all_passed']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
