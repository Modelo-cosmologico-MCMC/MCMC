"""Criterio de Cronos–Jeans (dynamics/cronos_jeans.py, cronos/cronos_jeans_1d.py).

Se comprueba la derivación (forma de q, relación de dispersión, tasa
∝ k), la reproducción del control externo del autor (r_CJ, Oort, cotas,
perfil de Sculptor) y el umbral del test 1D. Nada es demostración
física ni resultado observacional (E8/E13)."""

import numpy as np
import pytest

from cronos.cronos_jeans_1d import run_sheets
from dynamics.cronos_jeans import (
    A_SCULPTOR,
    fluid_dispersion_omega2,
    growth_rate_over_k,
    nfw_jeans_table,
    q_criterion,
    sculptor_profile_table,
    solar_neighbourhood_table,
)
from dynamics.weak_field import C_KMS


def test_q_formula_and_dispersion():
    rho, s2 = 0.1, 400.0
    q = q_criterion(rho, s2)
    assert q == pytest.approx(1.5 * C_KMS ** 2 * A_SCULPTOR * rho ** 1.5 / s2)
    # sin gravedad: ω² cambia de signo exactamente en q = 1 para todo k
    A1 = s2 / (1.5 * C_KMS ** 2 * rho ** 1.5)          # q = 1
    assert fluid_dispersion_omega2(1.0, rho, s2, A=A1, with_gravity=False) == pytest.approx(0.0, abs=1e-9)
    assert fluid_dispersion_omega2(3.0, rho, s2, A=0.5 * A1, with_gravity=False) > 0.0
    assert fluid_dispersion_omega2(3.0, rho, s2, A=2.0 * A1, with_gravity=False) < 0.0
    # tasa por unidad de k: ultravioleta
    assert growth_rate_over_k(0.8) == 0.0 and growth_rate_over_k(2.0) == pytest.approx(1.0)


def test_nfw_nivelA_reproduces_author_control():
    n = nfw_jeans_table()
    assert n["r_CJ_kpc"] == pytest.approx(0.709, abs=0.01)
    assert n["M_within_r_CJ"] == pytest.approx(1.61e8, rel=0.02)
    assert n["r_CJ_for_A_fraction"]["0.1"] == pytest.approx(0.247, abs=0.01)
    assert n["r_CJ_for_A_fraction"]["0.05"] == pytest.approx(0.180, abs=0.01)
    q = {r["r_kpc"]: r["q"] for r in n["rows"]}
    assert q[0.1] > 50 and q[0.4] > 3 and q[0.9] < 1 and q[2.3] < 0.1
    assert n["stop_rule"]["rho_msun_pc3"] == pytest.approx(4.32, rel=0.02)


def test_solar_neighbourhood_reproduces_author_control():
    s = solar_neighbourhood_table()
    assert s["oort_limit_cronos_msun_pc3"] == pytest.approx(0.75, abs=0.02)
    assert s["oort_limit_cronos_stars_only"] == pytest.approx(0.115, abs=0.01)
    assert s["A_bound_over_A_sculptor"] == pytest.approx(0.048, abs=0.005)
    assert s["A_bound_stability_over_A_sculptor"] == pytest.approx(0.158, abs=0.01)
    ratio = {r["z_pc"]: r["g_C_over_K_z"] for r in s["rows"]}
    assert ratio[50] > 5 and ratio[300] > 1.5 and ratio[1100] < 0.1


def test_sculptor_frozen_profile_shape():
    sc = sculptor_profile_table()
    prof = {p["R_pc"]: p for p in sc["profile"]}
    assert prof[10.0]["sigma_los_newton_cronos"] == pytest.approx(19.14, abs=0.2)
    assert prof[500.0]["sigma_los_newton_cronos"] == pytest.approx(2.63, abs=0.1)
    assert prof[1000.0]["sigma_los_newton_cronos"] == pytest.approx(prof[1000.0]["sigma_los_newton"], rel=0.01)
    q_in = {r["r_pc"]: r["q"] for r in sc["rows"]}
    assert 2.0 < q_in[5] < 2.6 and q_in[1000] < 0.05
    assert 380.0 < sc["r_unstable_max_pc"] < 480.0
    assert sc["A"] == pytest.approx(A_SCULPTOR)


@pytest.mark.parametrize("q,grows", [(0.8, False), (2.0, True)])
def test_cj_1d_threshold(q, grows):
    res = run_sheets(q, N=100_000, ng=128, T=0.3, dt=2e-3, seed=3, sample_every=25)
    r0 = res["samples"][0]["rms_delta"]
    r_end = res["samples"][-1]["rms_delta"]
    if grows:
        assert r_end > 5.0 * r0
    else:
        assert r_end < 2.0 * r0


def test_1d_stability_within_poisson_noise_is_seed_independent():
    a = run_sheets(0.8, N=50_000, ng=64, T=0.2, seed=1)["samples"][-1]["rms_delta"]
    b = run_sheets(0.8, N=50_000, ng=64, T=0.2, seed=2)["samples"][-1]["rms_delta"]
    assert abs(a - b) / a < 0.5
    assert np.isfinite(a)
