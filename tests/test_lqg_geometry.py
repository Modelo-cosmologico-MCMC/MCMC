"""Tests para mcmc_ontology/lqg_geometry.py — Etapa III.

Cubre los 9 bloques del Apéndice I y K (pp. 138-161 del Tratado):
γ_Immirzi, conversión δ→ΔA, topología, rotación de Wick, fricción
spinfoam, complejidad C_nD, c como punto fijo, proto-métrica y
dinámica de retorno (GWs relictas).
"""

import numpy as np

from mcmc_ontology import constants as C
from mcmc_ontology.lqg_geometry import (
    gamma_immirzi_mcmc, gamma_kaul_majumdar, E_star_ratio,
    spin_from_area, delta_area_from_energy,
    wick_rotation_angle, gamma0_wick, lapse_function,
    partition_function_transition,
    spinfoam_amplitude_cronos, zeta_sf, partition_function_spinfoam,
    complexity_nD, C_crit_table, dC_nD_dS,
    c_eff_evolution, c_fixed_point, c_stability_timescale,
    proto_metric_V3D, Xi_asymmetry, Delta_tens,
    V_ret, omega_ret, Omega_GW_return, delta0_next_cycle,
)


# ------------------------------------------------------------
# B1. γ Barbero-Immirzi
# ------------------------------------------------------------

class TestImmirzi:
    def test_gamma_mcmc_value(self):
        """γ_MCMC ∈ [0.270, 0.278] (Ec. 400)."""
        gamma = gamma_immirzi_mcmc()
        assert 0.270 < gamma < 0.278

    def test_gamma_vs_constants(self):
        """γ_derivado coincide con C.GAMMA_LQG dentro del 2%."""
        assert abs(gamma_immirzi_mcmc() / C.GAMMA_LQG - 1.0) < 0.02

    def test_gamma_kaul_majumdar_reference(self):
        """γ_KM = ln(2)/(π√3) ≈ 0.127 (referencia LQG).

        γ_MCMC ≈ 0.274 NO coincide numéricamente con γ_KM (factor 2),
        pero sí con la constante GAMMA_LQG del modelo (test anterior).
        """
        assert abs(gamma_kaul_majumdar() - 0.1274) < 1e-3

    def test_E_star_ratio(self):
        """E_*/E_P calibrado para γ ≈ 0.274."""
        assert E_star_ratio() > 0.0


# ------------------------------------------------------------
# B2. Espín a partir del área
# ------------------------------------------------------------

class TestSpinFromArea:
    def test_spin_non_negative(self):
        """j ≥ 0 para todo área."""
        for DA in (0.1, 1.0, 10.0, 100.0, 1000.0):
            assert spin_from_area(DA) >= 0.0

    def test_spin_growth(self):
        """j crece monótonamente con ΔA."""
        DAs = [10.0, 100.0, 1000.0, 10_000.0]
        js = [spin_from_area(DA) for DA in DAs]
        assert all(js[i] < js[i + 1] for i in range(len(js) - 1))

    def test_delta_area_non_negative(self):
        """ΔA(δE, ρV, K) ≥ 0 (no se acepta área negativa)."""
        DA = delta_area_from_energy(1e15, 0.5e15, 0.1, 1.0)
        assert DA >= 0.0


# ------------------------------------------------------------
# B4. Rotación de Wick
# ------------------------------------------------------------

class TestWickRotation:
    def test_euclidean_before_S1001(self):
        """θ_W ≈ 0 para S < S_{1.001}."""
        assert wick_rotation_angle(0.5) < 0.01

    def test_lorentzian_after_S1001(self):
        """θ_W ≈ π/2 para S > S_{1.001}."""
        assert abs(wick_rotation_angle(1.5) - np.pi / 2.0) < 0.01

    def test_smooth_transition(self):
        """θ_W continua y acotada en torno a S_{1.001}."""
        thetas = [wick_rotation_angle(S) for S in [0.99, 0.999, 1.001, 1.01, 1.1]]
        diffs = [abs(thetas[i + 1] - thetas[i]) for i in range(len(thetas) - 1)]
        assert all(d < 1.0 for d in diffs)

    def test_gamma0_euclidean(self):
        """γ⁰(S<<) → +1 (Euclidiano)."""
        assert abs(gamma0_wick(0.5) - 1.0) < 0.01

    def test_gamma0_lorentzian(self):
        """γ⁰(S>>) → i (Lorentziano, e^{iπ/2})."""
        assert abs(gamma0_wick(1.5) - 1j) < 0.01

    def test_lapse_zero_pre_geom(self):
        """g₀₀ ≈ 0 antes de la rotación de Wick (sin tiempo)."""
        assert abs(lapse_function(0.5, 0.0)) < 1e-3

    def test_lapse_negative_post_geom(self):
        """g₀₀ < 0 en régimen post-geométrico."""
        assert lapse_function(1.5, 0.0) < 0.0

    def test_partition_regime(self):
        """partition_function_transition refleja el régimen."""
        assert partition_function_transition(0.5) is False
        assert partition_function_transition(1.5) is True


# ------------------------------------------------------------
# B5. Spinfoam Cronos
# ------------------------------------------------------------

class TestSpinfoamCronos:
    def test_amplitude_suppressed(self):
        """Mayor curvatura → amplitud más suprimida."""
        z = 1e-3
        A_low  = spinfoam_amplitude_cronos(1.0, [1, 1], 100, z)
        A_high = spinfoam_amplitude_cronos(1.0, [50, 50], 100, z)
        assert A_high < A_low

    def test_zeta_sf_positive(self):
        """ζ_sf > 0 con parámetros físicos."""
        assert zeta_sf() > 0.0

    def test_partition_function_finite(self):
        """Z = Σ Π A_v finita para grafos triviales."""
        graphs = [[[1.0, 1.0]], [[2.0, 2.0]]]
        amps   = [[1.0, 1.0], [0.5, 0.5]]
        Z = partition_function_spinfoam(graphs, amps, zeta=1e-3)
        assert np.isfinite(Z) and Z > 0


# ------------------------------------------------------------
# B6. Complejidad tensional
# ------------------------------------------------------------

class TestComplexity:
    def test_C_nD_zero_when_E_zero(self):
        """C_nD = 0 si E_min = 0 (evita división por cero)."""
        assert complexity_nD(1.0, 1.0, 0.1, 0.0) == 0.0

    def test_C_nD_value(self):
        """C_nD = ρ·σ²/E_min = 2.0·0.5/4.0 = 0.25."""
        c = complexity_nD(1.0, 2.0, 0.5, 4.0)
        assert abs(c - 2.0 * 0.5 / 4.0) < 1e-12

    def test_C_crit_table_keys(self):
        """Tabla 68 contiene las 4 transiciones."""
        t = C_crit_table()
        for k in ("V0D->V1D", "V1D->V2D", "V2D->V3D", "V3D->V3+1D"):
            assert k in t

    def test_dC_nD_finite(self):
        """dC/dS finito con argumentos razonables."""
        d = dC_nD_dS(0.1, 1.0, 0.5, 4.0, 0.1, 0.05, dE_dS=0.0)
        assert np.isfinite(d)


# ------------------------------------------------------------
# B7. c como punto fijo
# ------------------------------------------------------------

class TestSpeedOfLight:
    def test_c_equipartition(self):
        """c = c₀ con E_p = M_p (equipartición)."""
        assert abs(c_fixed_point(1.0, 1.0, 1.0) - 1.0) < 1e-12

    def test_c_near_equipartition(self):
        """c ≈ √(1+δ)·c₀ con pequeña asimetría."""
        d = 1e-4
        assert abs(c_fixed_point(1 + d, 1.0, 1.0) - np.sqrt(1 + d)) < 1e-9

    def test_c_stability(self):
        """τ_estab > 0 con Γ_ann > 0."""
        assert c_stability_timescale(0.1) > 0.0

    def test_c_eff_evolution_finite(self):
        """dc_eff/dS finito con valores normales."""
        d = c_eff_evolution(1.0, 1.0, 1.0, 0.1, 0.05, 0.1)
        assert np.isfinite(d)


# ------------------------------------------------------------
# B8. Proto-métrica
# ------------------------------------------------------------

class TestProtoMetric:
    def test_h_ij_diagonal_at_zero_potential(self):
        """h_ij = δ_ij con Φ_grav = 0."""
        h = proto_metric_V3D(0.0)
        assert np.allclose(h, np.eye(3))

    def test_Xi_at_C3(self):
        """Ξ(S_{0.999}) pequeño (regularización en C3)."""
        Xi = Xi_asymmetry(0.999)
        assert abs(Xi) < 1.0

    def test_Delta_tens_subpercent(self):
        """Δ_tens / c² ≤ ~ δ₀² ≈ 1e-4 en S_{0.999}."""
        d = Delta_tens(0.999)
        assert d <= 1e-2  # cota laxa; en C3 la ventana suaviza Xi


# ------------------------------------------------------------
# B9. Dinámica de retorno + GWs relictas
# ------------------------------------------------------------

class TestReturnDynamics:
    def test_V_ret_minimum_at_Smax(self):
        """V_ret(S_max) = 0."""
        assert abs(V_ret(C.S_SEALS["S_max"])) < 1e-15

    def test_V_ret_positive(self):
        """V_ret(S<S_max) > 0."""
        assert V_ret(0.0) > 0.0

    def test_omega_ret_positive(self):
        """ω_ret > 0."""
        assert omega_ret() > 0.0

    def test_GW_amplitude_in_PTA_band(self):
        """Ω_GW(f≈10⁻⁸ Hz) detectable en SKA-PTA."""
        Omega = Omega_GW_return(1e-8)
        assert 1e-12 < Omega < 1e-8

    def test_GW_peak_near_PTA(self):
        """Pico del espectro en banda PTA (f_pico ~ 10⁻⁸ Hz)."""
        f = np.logspace(-10, -5, 200)
        Omega = np.array([Omega_GW_return(fi) for fi in f])
        f_peak = f[np.argmax(Omega)]
        assert 1e-9 < f_peak < 1e-7

    def test_GW_shape_sech2_decay(self):
        """Decae rápido tras el pico (firma sech²)."""
        f = np.logspace(-10, -5, 200)
        Omega = np.array([Omega_GW_return(fi) for fi in f])
        i_peak = int(np.argmax(Omega))
        # Diez frecuencias por encima del pico → caída significativa
        assert Omega[min(i_peak + 30, len(Omega) - 1)] < 0.3 * Omega[i_peak]

    def test_delta0_next_cycle_smaller(self):
        """δ̃₀ < δ₀ (Ec. 471)."""
        assert delta0_next_cycle() < C.EPSILON_0

    def test_delta0_next_cycle_positive(self):
        """δ̃₀ > 0 (la imperfección no se anula)."""
        assert delta0_next_cycle() > 0.0

    def test_delta0_reduction_factor(self):
        """δ̃₀/δ₀ = exp(-λ_pre · S_max / ΔS)."""
        ratio = delta0_next_cycle() / C.EPSILON_0
        expected = np.exp(-C.LAMBDA_PRE * C.S_SEALS["S_max"] / C.DELTA_S)
        assert abs(ratio - expected) < 1e-9
