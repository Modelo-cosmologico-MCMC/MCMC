"""Tests scaffold para Phase 2-4 (III.D, III.E, III.C, III.F, III.B, III.I).

Validan la API y consistencia interna sin dependencia de datos reales
externos (SPARC, NANOGrav, HotQCD, etc.).
"""

import numpy as np

# III.E — SPARC
from cronos.validate_sparc import (
    mock_galaxy, v_cored, v_NFW, fit_cored, fit_NFW,
    compare_cored_vs_NFW, SPARC_TARGETS,
)

# III.D — Lattice SU(3)
from lattice.su3_lattice_S import (
    beta_S, mass_gap_extraction, continuum_extrapolation,
    compare_with_data, GLUEBALL_PREDICTIONS_MCMC, E_min_lattice_units,
)

# III.C — Gravedad modificada
from cosmology.perturbations_modified import (
    mu_effective, eta_slip, growth_factor_modified, growth_exponent,
)

# III.F — MDR perturbaciones
from cosmology.mdual_perturbations import (
    f_sigma8_MCMC, S8, chi2_fsigma8, chi2_S8, summary_tensions,
)

# III.B — Boltzmann S
from cosmology.boltzmann_S import BoltzmannS

# III.I — Quantum hardware
from quantum.qiskit_circuit import (
    collapse_unitary_matrix, qudit_to_qubit_state, basis_index_to_binary,
)
from quantum.hardware_validation import (
    HARDWARE_PROFILES, predict_global_fidelity, simulate_with_noise, go_no_go,
)


# ─────────── III.E SPARC ───────────

class TestSPARC:
    def test_mock_galaxy_shape(self):
        gal = mock_galaxy("test", M_halo=1e11)
        assert len(gal.r_kpc) == len(gal.V_obs) == len(gal.dV_obs)

    def test_v_cored_positive(self):
        r = np.linspace(0.5, 30, 30)
        V = v_cored(r, M_halo=1e11)
        assert np.all(V >= 0.0)

    def test_v_NFW_positive(self):
        r = np.linspace(0.5, 30, 30)
        V = v_NFW(r, M200=1e12)
        assert np.all(V >= 0.0)

    def test_fit_returns_finite_chi2(self):
        gal = mock_galaxy("X", M_halo=1e11)
        r = fit_cored(gal); n = fit_NFW(gal)
        assert np.isfinite(r["chi2"]) and np.isfinite(n["chi2"])

    def test_compare_with_targets(self):
        out = compare_cored_vs_NFW(list(SPARC_TARGETS.keys())[:2])
        assert len(out) == 2
        for r in out:
            assert "improvement_pct" in r


# ─────────── III.D Lattice ───────────

class TestLatticeSU3:
    def test_beta_S_at_S3(self):
        """β(S=S3) = β₀ + β₁."""
        b = beta_S(1.000)
        assert abs(b - (6.10 + 0.50)) < 1e-9

    def test_beta_S_asymptotic(self):
        """β(S>>S3) → β₀."""
        assert abs(beta_S(10.0) - 6.10) < 1e-3

    def test_mass_gap_extraction_recovers(self):
        """Recupera m de un correlador sintético."""
        T = 32; m_true = 0.4
        t = np.arange(T)
        C_t = np.exp(-m_true * t) + np.exp(-m_true * (T - t))
        out = mass_gap_extraction(C_t, a_fm=0.1, T_lat=T, t_min=4)
        assert abs(out["m_lat"] - m_true) < 0.05

    def test_continuum_extrapolation(self):
        """Extrapolación lineal en a²."""
        a = np.array([0.05, 0.075, 0.1])
        m_true = 1.5
        m_meas = m_true + 2.0 * a ** 2
        out = continuum_extrapolation(m_meas, a)
        assert abs(out["m_continuum"] - m_true) < 1e-6

    def test_glueball_prediction_table(self):
        """Predicciones MCMC para 0++/2++/0-+."""
        for state in ("0++", "2++", "0-+"):
            assert GLUEBALL_PREDICTIONS_MCMC[state]["m_MeV"] > 1500

    def test_compare_with_HotQCD(self):
        """Comparativa MCMC vs HotQCD entrega desv. < 5% en 0++."""
        out = compare_with_data("0++")
        assert "HotQCD_2024" in out
        assert out["HotQCD_2024"]["deviation_pct"] < 5.0


# ─────────── III.C Gravedad modificada ───────────

class TestGravityModified:
    def test_mu_one_at_high_z(self):
        """µ(a→0) → 1 (régimen materia dominante, fuera de la transición)."""
        assert abs(mu_effective(0.001) - 1.0) < 0.05

    def test_eta_zero_today(self):
        """η(a=1) = 0 (sin slip hoy)."""
        assert abs(eta_slip(1.0)) < 1e-9

    def test_growth_exponents(self):
        """Exponentes p del Cuadro 15."""
        assert growth_exponent("matter") == -1.5
        assert growth_exponent("lambda") == 0.0

    def test_growth_factor_finite(self):
        """D(a) finito y normalizado a 1 hoy."""
        a = np.linspace(0.01, 1.0, 100)
        D = growth_factor_modified(a)
        assert np.isfinite(D[-1]) and abs(D[-1] - 1.0) < 1e-9


# ─────────── III.F MDR perturbations ───────────

class TestMDRPerturbations:
    def test_S8_value(self):
        """S₈ ≈ 0.795 (valor MCMC)."""
        assert 0.78 < S8() < 0.82

    def test_fsigma8_finite(self):
        z = np.array([0.38, 0.51, 0.61])
        fs = f_sigma8_MCMC(z)
        assert np.all(np.isfinite(fs)) and np.all(fs > 0)

    def test_chi2_BOSS(self):
        """χ² BOSS DR12 dentro de un rango razonable."""
        chi2 = chi2_fsigma8("BOSS_DR12")
        assert 0 < chi2 < 50

    def test_summary_tensions_keys(self):
        s = summary_tensions()
        assert "S8_MCMC" in s and "S8_tension_KiDS_sigma" in s


# ─────────── III.B Boltzmann ───────────

class TestBoltzmannS:
    def test_kappa_lat_peaks_at_seals(self):
        """κ_lat(S) tiene picos en los sellos S_n."""
        bs = BoltzmannS()
        peak_C2 = bs.kappa_lat(0.099)
        away = bs.kappa_lat(0.500)
        assert peak_C2 > away

    def test_rho_lat_evolution_finite(self):
        """ρ_lat(S) finito y positivo a lo largo de la integración.

        La evolución incluye fuente Γ_lat (crecimiento exponencial leve
        β_lat = 0.01) y sumideros κ_lat picudos en los sellos. La
        propiedad básica es finitud + positividad, no monotonía global.
        """
        bs = BoltzmannS()
        S = np.linspace(0.5, 1.5, 200)
        rho = bs.rho_lat_evolution(S, rho_lat_0=0.05)
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0.0)

    def test_transfer_function_not_implemented(self):
        """transfer_function() requiere Boltzmann externo."""
        import pytest
        bs = BoltzmannS()
        with pytest.raises(NotImplementedError):
            bs.transfer_function(np.array([0.1]), np.array([0.0]))


# ─────────── III.I Quantum hardware ───────────

class TestQuantumHardware:
    def test_basis_binary_encoding(self):
        assert basis_index_to_binary(0) == "000"
        assert basis_index_to_binary(4) == "100"

    def test_collapse_unitary_unitary(self):
        """U†U = I para cada compuerta de colapso."""
        for n in range(4):
            U = collapse_unitary_matrix(n)
            assert np.allclose(U.conj().T @ U, np.eye(8), atol=1e-12)

    def test_qudit_to_qubit_norm(self):
        """Estado codificado normalizado."""
        c = np.array([1, 0, 0, 1, 0]) / np.sqrt(2)
        s = qudit_to_qubit_state(c)
        assert abs(np.linalg.norm(s) - 1.0) < 1e-12

    def test_predict_global_fidelity(self):
        """F_global = F_gate^n decreciente con n."""
        F1 = predict_global_fidelity(0.96, 4)
        F2 = predict_global_fidelity(0.96, 6)
        assert F1 > F2

    def test_simulate_with_noise_fields(self):
        out = simulate_with_noise(F_gate=0.96, n_shots=100, seed=0)
        assert "F_global" in out and "transition_fidelities" in out
        assert 0.0 <= out["F_global"] <= 1.0

    def test_go_no_go(self):
        assert go_no_go(0.85) == "GO"
        assert go_no_go(0.65) == "MARGINAL"
        assert go_no_go(0.50) == "NO-GO"

    def test_hardware_profiles_complete(self):
        for k in HARDWARE_PROFILES:
            for field in ("platform", "T_circuit_us", "F_global_pred"):
                assert field in HARDWARE_PROFILES[k]
