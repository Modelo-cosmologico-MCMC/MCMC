"""Diccionario, ecuación 1 (dynamics/epsilon_c_saturating.py) y
ecuaciones 2–3 (core/s_post_unit.py): la forma saturante recupera la ley
débil a baja densidad y satura a alta; q generaliza el criterio; la cota
de Oort reproduce el 0.060·A_S de §3.26 en el límite de ley débil; la
calibración de Sculptor fija q_S independientemente de la forma; la
predicción RAR es un escalón. Nada es física: consistencia (E8)."""

import numpy as np
import pytest

from dynamics.epsilon_c_saturating import (
    A_SCULPTOR,
    constraint_map,
    deps_c_drho,
    eps_c,
    oort_bound_on_deps,
    q_local,
    rar_step_prediction,
    sculptor_central_density,
    sculptor_force_calibration_eps_max,
    weak_law_amplitude,
)


def test_form_limits_and_derivative():
    e_max, rs = 1e-6, 0.1
    A = weak_law_amplitude(e_max, rs)
    rho = np.array([1e-4, 1e-3])
    assert np.allclose(eps_c(rho, e_max, rs), A * rho ** 1.5, rtol=1e-3)          # ley débil para ρ ≪ ρ*
    assert eps_c(1e3, e_max, rs) == pytest.approx(e_max, rel=1e-4)                # techo
    h = 1e-8
    num = (eps_c(0.05 + h, e_max, rs) - eps_c(0.05 - h, e_max, rs)) / (2 * h)
    assert deps_c_drho(0.05, e_max, rs) == pytest.approx(num, rel=1e-5)
    # q generaliza (3/2)c²ε_c/σ² en la ley débil
    from dynamics.cronos_jeans import q_criterion
    assert q_local(1e-3, 400.0, e_max, rs) == pytest.approx(q_criterion(1e-3, 400.0, A), rel=2e-3)


def test_oort_bound_matches_weak_law_result():
    o = oort_bound_on_deps()
    assert o["power_law_equivalent_A_max"] / A_SCULPTOR == pytest.approx(0.0601, abs=2e-3)   # §3.26: A_2σ = 0.060·A_S


def test_sculptor_calibration_is_force_matching_and_q_form_independent():
    rs = np.array([1e-3, 1e-1, 10.0])
    for ups in (1.0, 2.0):
        e = sculptor_force_calibration_eps_max(rs, ups)
        rho_s = sculptor_central_density(ups)
        assert np.allclose(deps_c_drho(rho_s, e, rs), 1.5 * A_SCULPTOR * np.sqrt(rho_s), rtol=1e-12)
    m = constraint_map(log_eps_max=(-12.0, -4.0, 41), log_rho_star=(-4.0, 1.0, 26))
    cal = m["sculptor_calibration"]
    assert all(c["q_sculptor_self_form_independent"] > 1.0 for c in cal.values())    # Sculptor inestable bajo su calibración
    assert all(not c["both_ok_any"] for c in cal.values())
    assert cal["1.0"]["oort_and_others_ok_any"] and not cal["3.0"]["oort_and_others_ok_any"]
    assert 0.0 < m["allowed_fraction"] < 1.0 and "fallo cerrado" in m["sculptor_profile_constraint"]


def test_rar_step_is_localized_where_rho_crosses_rho_star():
    r = rar_step_prediction(1e-7, 0.03, Sigma0_msun_pc2=300.0, h_kpc=3.0, h_z_kpc=0.3)
    R = np.array(r["R_kpc"])
    boost = np.array(r["g_C"]) / np.array(r["g_bar"])
    assert r["R_where_rho_equals_rho_star_kpc"] is not None
    assert abs(r["R_of_max_boost_kpc"] - r["R_where_rho_equals_rho_star_kpc"]) < 2.0
    assert boost[0] < 0.1 * boost.max() and boost[-1] < 0.1 * boost.max()       # escalón: pequeño lejos de ρ ≈ ρ*
    assert np.all(np.array(r["g_bar"]) > 0.0) and R[0] < r["R_of_max_boost_kpc"] < R[-1]


def test_s_post_unit_declarations():
    from core.s_post_unit import STATUS, C_of_S_from_kappa, dS_post, dS_pre
    assert dS_post(2.0, 4.0, 0.5) == pytest.approx(0.25) and dS_pre(2.0, 4.0, 0.5) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        dS_post(1.0, 0.0, 1.0)
    c = C_of_S_from_kappa(None, None)
    assert c["C_of_S"] is None and "S_hoy = 95" in c["convention_in_force"]
    assert "declarada" in STATUS["E2"] and "κ" in STATUS["E3"]
