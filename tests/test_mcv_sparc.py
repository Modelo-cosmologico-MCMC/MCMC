"""PR-S, brazo MCV (c) para SPARC (dynamics/mcv_sparc.py), sin datos: el mapa inverso frente al empalme directo, las
cifras de los documentos del 26 y del 27-oct (r_½ por σ_x/m y por t_edad, el control (σ_x/m)·t_edad, J.6.2 ≈ 0.22 cm²/g),
la ECV por debajo de 1e-4 de la atracción, la rejilla vectorizada frente al mapa exacto, el punto fijo de la edad desde
la concentración, el ajuste de una galaxia sintética declarada y las lecturas sin letra."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from dynamics import mcv_baryons as mb
from dynamics import mcv_sparc as ms
from dynamics.epsilon_c_saturating import deps_c_drho, eps_c, exponential_disc_g_bar
from dynamics.mcv_equilibrium import jeans_match, nfw_mass

M11 = 1e11
C11 = float(ms.c200_dm14(M11))


def _disc(M_d=3e9, R_d=2.0, radii=None):
    """Disco exponencial delgado declarado (Freeman), tabulado: los bariones de la comprobación del 26-oct."""
    R = np.geomspace(0.1, 40.0, 60) if radii is None else radii
    return mb.TabulatedBaryons(R, exponential_disc_g_bar(R, M_d / (2.0 * math.pi * R_d ** 2) / 1e6, R_d) * R)


def _synthetic(M200=1e11, som=1.0, R_d=2.0, n=20, noise=False, seed=0):
    """Galaxia SINTÉTICA declarada (no observacional): halo con la c–M mediana, M⋆ de Moster et al. (2013), gas = M⋆/2 en
    un disco de 2R_d, Υ_d = 0.5, rama con núcleo del empalme y errores del 4 % (mínimo 2 km/s)."""
    c = float(ms.c200_dm14(M200))
    x = M200 / 10 ** 11.590
    Ms = 2 * 0.0351 * M200 / (x ** -1.376 + x ** 0.608)
    R = np.geomspace(0.5 * R_d, 5 * R_d, n)
    v2s = exponential_disc_g_bar(R, Ms / (2 * math.pi * R_d ** 2) / 1e6, R_d) * R
    v2g = exponential_disc_g_bar(R, 0.5 * Ms / (2 * math.pi * (2 * R_d) ** 2) / 1e6, 2 * R_d) * R
    bar = mb.TabulatedBaryons(R, v2s + v2g)
    br = mb.match_branches(M200, c, som, baryons=bar, prefilter=True)["branches"][0]
    h = mb.nfw_halo(M200, c, mb.H0)
    M_dm = np.where(R < br["r1"], mb.core_profile(br["rho0"], br["sigma0"], R, bar)["M_M"], nfw_mass(R, h["rho_s"], h["r_s"]))
    v = np.sqrt(v2s + v2g + mb.G_KPC * M_dm / R - 8 / 3 * math.pi * mb.G_KPC * mb.RHO_E * R ** 2)
    e = np.maximum(2.0, 0.04 * v)
    if noise:
        v = v + np.random.default_rng(seed).normal(0.0, e)
    gal = ms.Galaxy("SINTÉTICA", R, v, e, np.sqrt(v2g), np.sqrt(v2s / 0.5), np.zeros_like(R), 10.0, 1.0, 60.0, 4.0,
                    R_d=R_d, SB_disk0=Ms / 0.5 / (2 * math.pi * (R_d * 1e3) ** 2), synthetic=True)
    return gal, br, c


SMALL = {**ms.FIT_SPEC, "log_rho0": [6.0, 9.5], "log_sigma0": [1.3, 2.1], "k_grid": [-1.0, 0.0, 1.0]}


def test_generic_match_reproduces_jeans_match_with_plummer():
    a = jeans_match(M11, C11, 3.0, plummer={"M": 3e9, "a": 1.5}, rho_E=mb.RHO_E, prefilter=True)["branches"]
    b = mb.match_branches(M11, C11, 3.0, baryons=mb.PlummerBaryons(3e9, 1.5), rho_E=mb.RHO_E, prefilter=True)["branches"]
    assert [x["branch"] for x in a] == [x["branch"] for x in b] and len(a) == 3
    for x, y in zip(a, b):
        assert y["sigma0"] == pytest.approx(x["sigma0"], rel=1e-9) and y["rho0"] == pytest.approx(x["rho0"], rel=1e-8)


def test_inverse_map_returns_the_envelope_of_the_match_with_a_disc():
    """Documento del 26-oct: 1e11 M☉ con disco y σ_x/m = 1: M200 a 2e-4 y c a 1e-4 (aquí, mucho mejor)."""
    bar = _disc()
    br = mb.match_branches(M11, C11, 1.0, baryons=bar, prefilter=True)["branches"]
    assert [b["branch"] for b in br] == ["con núcleo"]
    m = mb.inverse_map(br[0]["rho0"], br[0]["sigma0"], 1.0, 10.0, bar)
    assert abs(m["M200"] / M11 - 1.0) < 2e-4 * 1e-3 and abs(m["c"] / C11 - 1.0) < 1e-4 * 1e-3
    assert m["r1"] == pytest.approx(br[0]["r1"], rel=1e-7)


def test_inverse_map_covers_the_dense_branches():
    pl = mb.PlummerBaryons(3e9, 1.5)
    for b in mb.match_branches(M11, C11, 3.0, baryons=pl, prefilter=True)["branches"]:
        m = mb.inverse_map(b["rho0"], b["sigma0"], 3.0, 10.0, pl)
        assert m["M200"] == pytest.approx(M11, rel=1e-8) and m["c"] == pytest.approx(C11, rel=1e-8)


@pytest.mark.parametrize("som,r_half", [(0.3, 1.29), (1.0, 2.56), (3.0, 3.75), (10.0, 4.87)])
def test_core_half_density_radius_without_baryons(som, r_half):
    """26-oct: sin bariones, r_½(1e11 M☉) = 1.29 / 2.56 / 3.75 / 4.87 kpc con σ_x/m = 0.3 / 1 / 3 / 10 cm²/g."""
    br = mb.match_branches(M11, C11, som, prefilter=True)["branches"]
    assert br[0]["branch"] == "con núcleo" and round(br[0]["r_c"], 2) == r_half


@pytest.mark.parametrize("t_age,r_half", [(5.0, 1.81), (10.0, 2.56), (12.5, 2.81)])
def test_core_depends_on_the_age(t_age, r_half):
    """27-oct, C4 y enmienda 8: r_½(1e11, σ_x/m = 1) = 1.81 / 2.56 / 2.81 kpc con t_edad = 5 / 10 / 12.5 Gyr."""
    assert round(mb.match_branches(M11, C11, 1.0, t_age, prefilter=True)["branches"][0]["r_c"], 2) == r_half


def test_only_the_product_of_cross_section_and_age_counts():
    """Control de C4: (σ_x/m, t_edad) = (1, 5 Gyr) da el mismo núcleo que (0.5, 10 Gyr)."""
    a = mb.match_branches(M11, C11, 1.0, 5.0, prefilter=True)["branches"][0]
    b = mb.match_branches(M11, C11, 0.5, 10.0, prefilter=True)["branches"][0]
    assert a["r_c"] == pytest.approx(b["r_c"], abs=1e-4) and a["rho0"] == pytest.approx(b["rho0"], rel=1e-6)


def test_j62_core_asks_for_about_0p22_cm2_per_g():
    """C6: el r_c = 1.8 kpc de Burkert de J.6.2 tiene la media densidad en 0.979 kpc, que pide σ_x/m ≈ 0.22 cm²/g."""
    r_half = ms.burkert_half_density_radius(ms.J62_R_STAR_KPC)
    assert round(r_half, 3) == 0.979

    def f(lsom):
        return mb.match_branches(M11, C11, math.exp(lsom), prefilter=True)["branches"][0]["r_c"] - r_half
    assert round(math.exp(brentq(f, math.log(0.1), math.log(0.5), xtol=1e-6)), 2) == 0.22


@pytest.mark.parametrize("M200,R0_kpc", [(1e10, 240.0), (1e11, 517.0), (1e12, 1114.0)])
def test_ecv_weight_and_border_position(M200, R0_kpc):
    """C1 del 27-oct: (8π/3)ρ_E·r³/M(r) ≤ 6e-5 hasta 0.1·r200 y 1.3–1.4e-3 en r200/2 (envoltura NFW con la c mediana);
    R₀ = 5.27·r200 (dynamics/ecv_border.py). Dentro de 10·R_d depende del tamaño del disco: con R_d = 2 kpc en 1e11 M☉,
    10·R_d = 0.21·r200 y el peso llega a 1.7e-4 (lo publica scripts/run_pr_s.py checks)."""
    from dynamics.ecv_border import r0_over_r200
    h = mb.nfw_halo(M200, float(ms.c200_dm14(M200)), mb.H0)
    r = np.linspace(1e-3, 0.1 * h["r200"], 400)
    assert np.max(mb.ecv_force_fraction(r, nfw_mass(r, h["rho_s"], h["r_s"]))) < 6.1e-5
    half = float(mb.ecv_force_fraction(0.5 * h["r200"], nfw_mass(0.5 * h["r200"], h["rho_s"], h["r_s"])))
    assert 1.25e-3 < half < 1.4e-3
    assert r0_over_r200(0.0) * h["r200"] == pytest.approx(R0_kpc, abs=1.0)


def test_vectorized_grid_agrees_with_the_exact_inverse_map():
    bar = _disc(radii=np.geomspace(0.3, 30.0, 25))
    rng = np.random.default_rng(1)
    rho0, sig0 = 10 ** rng.uniform(6.0, 10.5, 60), 10 ** rng.uniform(1.0, 2.3, 60)
    mp = mb.map_table(mb.core_table(rho0, sig0, bar, mb.RHO_E, mb.radial_nodes(h=0.02)), 1.0, 10.0)
    n = 0
    for j in range(rho0.size):
        ex = mb.inverse_map(rho0[j], sig0[j], 1.0, 10.0, bar)
        assert (ex is None) == (not np.isfinite(mp["c"][j]))
        if ex is not None:
            n += 1
            # los quiebros de V_bar² (lineal entre puntos) dejan el RK4 de paso fijo en ~1e-5 cerca de ellos
            assert mp["M200"][j] == pytest.approx(ex["M200"], rel=3e-5) and mp["c"][j] == pytest.approx(ex["c"], rel=3e-5)
    assert n > 20


def test_core_mass_between_nodes_by_hermite():
    bar = _disc()
    tab = mb.core_table(np.array([3e7, 1e9]), np.array([50.0, 60.0]), bar, mb.RHO_E, mb.radial_nodes())
    R = np.array([0.37, 1.9, 7.3, 22.0])
    got = tab.mass_at(np.log(R))
    for i, (r0, s0) in enumerate(zip(tab.rho0, tab.sigma0)):
        assert got[i] == pytest.approx(mb.core_profile(r0, s0, R, bar)["M_M"], rel=2e-5)


def test_formation_age_from_concentration():
    assert mb.T_UNIVERSE == pytest.approx(13.80, abs=0.01)
    c = np.array([3.0, 4.1, 5.0, 10.0, 20.0])
    t = mb.formation_age_gyr(c, 4.1)
    assert np.isnan(t[0]) and np.isnan(t[1]) and np.all(np.diff(t[2:]) > 0.0)
    assert t[3] == pytest.approx(mb.T_UNIVERSE - float(mb.cosmic_time_gyr(0.41)), rel=1e-12)


def test_concentration_age_fixed_points_are_exact_and_all_found():
    gal, _, _ = _synthetic()
    _, _, _, bar = ms.galaxy_baryons(gal, ms.nuisance_point(gal, 0.0, 0.0, 0.0))
    rng = np.random.default_rng(3)
    tab = mb.core_table(10 ** rng.uniform(6.5, 9.0, 300), 10 ** rng.uniform(1.4, 2.0, 300), bar, mb.RHO_E,
                        mb.radial_nodes())
    maps = mb.concentration_age_maps(tab, [1.0], 0.5, 4.1)[0]
    som = mb.sigma_over_m_power(1.0, 0.5)
    ts = np.geomspace(0.05, mb.T_UNIVERSE, 600)
    g = np.array([mb.formation_age_gyr(mb.map_table(tab, som, t)["c"], 4.1) - t for t in ts])
    fin = np.isfinite(g)
    dense = ((np.sign(g[:-1]) != np.sign(g[1:])) & fin[:-1] & fin[1:]).sum(axis=0)
    found = sum(np.isfinite(m["t_age"]) for m in maps)
    assert np.all(found >= dense)                     # el borde de validez no esconde raíces
    for m in maps:
        ok = np.isfinite(m["t_age"])
        assert np.all(np.abs(mb.formation_age_gyr(m["c"][ok], 4.1) - m["t_age"][ok]) < 1e-6 * m["t_age"][ok])


def test_profiled_inclination_matches_a_brute_force_minimum():
    rng = np.random.default_rng(5)
    v = rng.uniform(50, 150, 15)
    e = rng.uniform(2, 6, 15)
    v2m = (v * rng.uniform(0.85, 1.1)) ** 2
    best, i_best = ms.profile_inclination(v, e, 55.0, 5.0, v2m[None, :])

    def total(i):
        u = math.sin(math.radians(i)) / math.sin(math.radians(55.0))
        return float(np.sum((v - u * np.sqrt(v2m)) ** 2 / e ** 2) + ((i - 55.0) / 5.0) ** 2)
    ref = minimize_scalar(total, bounds=(35.0, 75.0), method="bounded", options={"xatol": 1e-10})
    assert best[0] == pytest.approx(ref.fun, rel=1e-9, abs=1e-9) and i_best[0] == pytest.approx(ref.x, abs=1e-5)


def test_negative_baryon_points_are_zeroed_and_counted():
    v2, n = ms.baryon_v2([-20.0, 5.0, 30.0], [10.0, 10.0, 10.0], [0.0, 0.0, 0.0], 0.5, 0.7)
    assert n == 1 and v2[0] == 0.0 and v2[2] == pytest.approx(950.0)
    bar = mb.TabulatedBaryons([1.0, 2.0, 4.0], [100.0, 400.0, 400.0])
    assert bar.mass(0.5) == pytest.approx(bar.mass(1.0) / 8.0) and bar.mass(9.0) == pytest.approx(bar.mass(4.0))


def test_fit_recovers_a_noiseless_synthetic_galaxy():
    gal, br, c = _synthetic()
    st = {"sigma100": 1.0, "a": 0.0}
    fit = ms.fit_mcv(gal, [st], SMALL, zoom=True)[0]
    assert fit["chi2"] < 1e-3 and fit["stage"] == "rejilla + refinado"
    assert fit["M200"] == pytest.approx(M11, rel=1e-3) and fit["c"] == pytest.approx(c, rel=1e-3)
    assert fit["rho0"] == pytest.approx(br["rho0"], rel=1e-3) and fit["inc"] == pytest.approx(60.0, abs=0.01)
    assert ms.classify_branch(gal, fit, st, SMALL)["branch"] == "con núcleo"


def test_readings_without_letter():
    gal, br, c = _synthetic()
    st = {"sigma100": 1.0, "a": 0.0}
    fit = ms.fit_mcv(gal, [st], SMALL, zoom=True)[0]
    col = ms.collapse_reading(fit, st)
    assert col["t_c_gyr"]["0.45"] == pytest.approx(col["t_c_gyr"]["0.75"] * 0.75 / 0.45)
    assert col["t_c_exceeds_age"]["0.75"]             # C7: con 1 cm²/g, 1e11 M☉ no colapsa en 10 Gyr
    lap = ms.lapse_depth(fit, gal)
    assert 1e-7 < lap["Delta_N0"] < 1e-6 and lap["Delta_N0"] > lap["Delta_N0_nfw_only"]
    A = 6.2e-7
    q = ms.q_cj_cross_reading(fit, gal, lambda r: A * r ** 1.5, lambda r: 1.5 * A * r ** 0.5)
    assert q["q_CJ0"] == pytest.approx(q["q_CJ0_general"], rel=1e-12)          # ley débil: los dos q coinciden
    pt = (8.132006605754103e-08, 0.001)
    qs = ms.q_cj_cross_reading(fit, gal, lambda r: eps_c(r, *pt), lambda r: deps_c_drho(r, *pt))
    assert qs["q_CJ0_general"] < 0.01 * qs["q_CJ0"]          # forma saturante saturada: q de estabilidad ≪ 1.5c²ε_c/σ₀²


def test_black_hole_cusp_is_out_of_reach():
    """Enmienda 6 (C2): r_h ≤ 43 pc en las combinaciones de la comprobación."""
    rows = ms.bh_influence_table()
    assert max(r["r_h_pc"] for r in rows) == pytest.approx(43.0, abs=0.05)
    assert min(r["r_h_pc"] for r in rows) == pytest.approx(0.17, abs=0.005)


def test_rc_m200_reading_and_its_references():
    M = np.geomspace(1e10, 1e12, 9)
    rd = ms.rc_m200_reading(0.979 * (M / 1e11) ** 0.45, M)
    assert rd["slope"] == pytest.approx(0.45) and rd["r_half_at_1e11_kpc"] == pytest.approx(0.979)
    assert rd["references"]["J.6.2 (M 767), control"]["r_half_at_1e11_kpc"] == pytest.approx(0.9786, abs=1e-4)
    assert rd["U_table_45"] == "pendiente de transcripción del autor"


def test_fixed_and_declared_cosmology():
    """Punto 9 del bloque del 28-oct: PLANCK18 de dynamics/ecv_border.py, ρ_E = Ω_Λ·ρ_crit,0 constante, H0 = 67.4 y el t(a)
    de ese fondo; ninguna ley de Q ni de H."""
    from scipy.integrate import quad

    from dynamics.ecv_border import E_of_z, r0_over_r200, rho_crit, rho_ecv
    from dynamics.mcv_equilibrium import H0_DEFAULT
    assert rho_ecv() == pytest.approx(0.685 * rho_crit(0.0), rel=1e-15) and mb.RHO_E == rho_ecv()
    assert mb.RHO_CRIT == rho_crit(0.0) and mb.H0 == H0_DEFAULT == 67.4
    assert round(r0_over_r200(0.0), 2) == 5.27
    lookback = mb.T_UNIVERSE - float(mb.cosmic_time_gyr(0.5))
    assert round(lookback, 2) == 7.95
    num = quad(lambda z: 1.0 / ((1.0 + z) * E_of_z(z)), 0.0, 1.0, epsabs=1e-13, epsrel=1e-13)[0] * 977.792221 / H0_DEFAULT
    assert lookback == pytest.approx(num, rel=1e-9)
    assert ms.PLANCK18_DECLARED == {"h": 0.674, "Omega_m": 0.315, "Omega_L": 0.685}
