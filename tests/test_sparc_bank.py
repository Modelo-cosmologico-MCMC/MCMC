"""PR-S, banco de mecanismos (dynamics/sparc_bank.py), sin datos: reparto A/B, submuestra de calidad, los modelos de los
brazos frente a sus fórmulas, el ajuste de un brazo rápido, la calibración, las letras y sus puertas, E8-Q, la
resolución y las lecturas sin letra. Las galaxias son SINTÉTICAS y declaradas."""

import math

import numpy as np
import pytest

from dynamics import mcv_baryons as mb
from dynamics import mcv_sparc as ms
from dynamics import sparc_bank as sb
from dynamics.mcv_equilibrium import nfw_halo, nfw_mass
from dynamics.sculptor_transfer import frozen_config
from dynamics.sparc_crossfalsification import predict_cronos_fixed_A


def _nfw_galaxy(M200=10 ** 11.3, c=9.0, inc=55.0):
    """Galaxia SINTÉTICA declarada: disco + gas de Freeman y un halo NFW, sin ruido (V a Υ = 1 en el disco)."""
    from dynamics.epsilon_c_saturating import exponential_disc_g_bar
    R = np.geomspace(0.6, 18.0, 18)
    v2s = exponential_disc_g_bar(R, 3e9 / (2 * math.pi * 4.0) / 1e6, 2.0) * R
    v2g = exponential_disc_g_bar(R, 1.5e9 / (2 * math.pi * 16.0) / 1e6, 4.0) * R
    h = nfw_halo(M200, c, mb.H0)
    v = np.sqrt(v2s + v2g + mb.G_KPC * nfw_mass(R, h["rho_s"], h["r_s"]) / R - sb.ECV_TERM * R ** 2)
    return ms.Galaxy("NFW-SINT", R, v, np.maximum(2.0, 0.03 * v), np.sqrt(v2g), np.sqrt(v2s / 0.5), np.zeros_like(R),
                     12.0, 1.2, inc, 3.0, R_d=2.0, SB_disk0=3e9 / 0.5 / (2 * math.pi * 4e6), quality=1, synthetic=True)


def test_split_by_sha256_is_fixed_before_the_bytes():
    names = [f"G{i:03d}" for i in range(31)]
    a = sb.split_halves(names)
    b = sb.split_halves(list(reversed(names)))
    assert a == b and len(a["A"]) == 16 and len(a["B"]) == 15 and not set(a["A"]) & set(a["B"])
    assert sb.split_halves(names, "semilla", seed=7) == sb.split_halves(names, "semilla", seed=7)
    with pytest.raises(ValueError):
        sb.split_halves(names, "semilla")


def test_quality_sample_uses_only_catalogue_metadata():
    g = _nfw_galaxy()
    bad_q = ms.Galaxy(**{**g.__dict__, "name": "Q3", "quality": 3})
    low_i = ms.Galaxy(**{**g.__dict__, "name": "I25", "inc": 25.0})
    few = ms.Galaxy(**{**g.__dict__, "name": "N4", "R": g.R[:4], "v_obs": g.v_obs[:4], "e_v": g.e_v[:4],
                       "v_gas": g.v_gas[:4], "v_disk": g.v_disk[:4], "v_bul": g.v_bul[:4]})
    assert [x.name for x in sb.quality_sample([g, bad_q, low_i, few])] == ["NFW-SINT"]


def test_arm_models_against_their_formulas():
    R = np.array([[0.5, 2.0, 10.0, 40.0]])
    h = nfw_halo(1e11, 10.0, mb.H0)
    assert sb.nfw_v2(np.array([11.0]), np.array([1.0]), R)[0] == pytest.approx(
        mb.G_KPC * nfw_mass(R[0], h["rho_s"], h["r_s"]) / R[0], rel=1e-10)
    r = np.linspace(0.1, 20.0, 4001)
    M = sb.burkert_mass(r, 2e7, 3.0)
    dens = 2e7 * 27.0 / ((r + 3.0) * (r * r + 9.0))
    assert np.gradient(M, r)[100:-100] == pytest.approx((4 * math.pi * r * r * dens)[100:-100], rel=1e-4)
    M200 = 10 ** 11.5
    r200 = (3 * M200 / (4 * math.pi * 200 * mb.RHO_CRIT)) ** (1 / 3)
    v2 = sb.j62_v2(np.array([11.5]), 1.8, np.array([[r200]]))
    assert v2[0, 0] == pytest.approx(mb.G_KPC * M200 / r200, rel=1e-12)
    # MOND: límite newtoniano y profundo (V⁴ = G·M·a₀)
    gb = np.array([1e3 * sb.A0_MOND, 1e-4 * sb.A0_MOND])
    v2m = sb.mond_v2(gb * 5.0, np.array([5.0, 5.0]))
    assert v2m[0] / (gb[0] * 5.0) == pytest.approx(1.0, abs=1e-12)
    assert v2m[1] ** 2 == pytest.approx(25.0 * gb[1] * sb.A0_MOND, rel=2e-2)
    assert sb.A0_MOND == pytest.approx(3702.8, abs=0.1)


def test_mcmc_c_is_the_frozen_m1_of_the_5e():
    g = _nfw_galaxy()
    cfg = ms.nuisance_point(g, 0.0, 0.0, 0.0)
    ref = predict_cronos_fixed_A(1e3 * g.R, 0.5 * g.SB_disk0, 1e3 * g.R_d, frozen_config())
    assert sb.cronos_v2(g, cfg, g.R) == pytest.approx(ref, rel=1e-12)


def test_fast_arm_recovers_a_noiseless_nfw_galaxy():
    g = _nfw_galaxy()
    f = sb.fit_arm(g, "ΛCDM-1", zoom=True)
    assert f["chi2"] < 1e-3 and f["params"][0] == pytest.approx(11.3, abs=1e-3)
    assert 10 ** f["params"][1] == pytest.approx(9.0, rel=2e-3) and f["k_params"] == 5
    f2 = sb.fit_arm(g, "ΛCDM-2", zoom=True)
    assert f2["chi2"] >= f2["chi2_data"] > -1e-12


def test_calibration_region_and_grid_edge():
    chi2 = {(1.0, 0.0): [10.0, 12.0], (2.0, 0.0): [9.0, 11.0], (3.0, 0.5): [12.0, 14.5], (5.0, 0.0): [np.inf, 1.0]}
    cal = sb.calibrate(chi2)
    assert cal["best"] == (2.0, 0.0) and cal["region"] == [(1.0, 0.0), (2.0, 0.0)]
    assert sb.on_grid_edge((0.1, 0.0)) and sb.on_grid_edge((1.0, 2.0)) and not sb.on_grid_edge((1.0, 0.5))


GATES_OK = {"bytes_con_manifiesto": True, "cualificacion_E8Q": True, "frac_B_sin_empalme": 0.0,
            "calibracion_en_borde": False}


def test_letters_and_their_order():
    rng = np.random.default_rng(0)
    assert sb.letters_mcv_vs_lcdm2(-3.0 + rng.normal(0, 1, 60), GATES_OK)["letter"] == "A"
    assert sb.letters_mcv_vs_lcdm2(3.0 + rng.normal(0, 1, 60), GATES_OK)["letter"] == "C"
    assert sb.letters_mcv_vs_lcdm2(rng.normal(0, 1, 60), GATES_OK)["letter"] == "B"
    ind = sb.letters_mcv_vs_lcdm2(-3.0 + rng.normal(0, 1, 30), {**GATES_OK, "cualificacion_E8Q": False})
    assert ind["letter"] == "INDETERMINADO" and len(ind["reasons"]) == 2
    assert sb.letters_mcv_vs_lcdm2(-np.ones(60), {**GATES_OK, "frac_B_sin_empalme": 0.11})["letter"] == "INDETERMINADO"
    assert sb.letters_mcv_vs_lcdm2(-np.ones(60), {**GATES_OK, "calibracion_en_borde": True})["letter"] == "INDETERMINADO"
    assert sb.letters_mcv_vs_lcdm2(-np.ones(60), {})["letter"] == "INDETERMINADO"


def test_e8q_fails_closed_and_decides():
    assert not sb.e8q_gate({}, None, None)["pass"]
    gals = [f"G{i}" for i in range(20)]
    ref = {"NFW": dict.fromkeys(gals, 1.0), "Burkert": dict.fromkeys(gals, 2.0)}
    ours = {"NFW": dict.fromkeys(gals, 1.05), "Burkert": {g: (2.1 if i < 18 else 3.0) for i, g in enumerate(gals)}}
    assert sb.e8q_gate(ours, ref, (1.14e-10, 0.02e-10))["pass"]
    assert not sb.e8q_gate(ours, ref, (1.25e-10, 0.02e-10))["pass"]
    ours["Burkert"]["G0"] = 3.0
    assert not sb.e8q_gate(ours, ref, (1.13e-10, 0.02e-10))["pass"]


def test_resolution_check_and_readings_without_letter():
    rc = sb.resolution_check({f"G{i}": 10.0 for i in range(20)}, {f"G{i}": 10.0 + (0.6 if i == 0 else 0.1) for i in range(20)})
    assert rc["pass"] and rc["frac_within"] == pytest.approx(0.95)
    assert not sb.resolution_check({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 2.0})["pass"]
    assert sb.globality({"best": (1.0, 0.0), "region": [(1.0, 0.0), (2.0, 0.0)]}, {"best": (2.0, 0.0)})["B_dentro_region_A"]
    sc = sb.sculptor_cross_reading((1.0, 1.0))
    assert sc["v_rel_kms"] == pytest.approx(20.76, abs=0.01) and sc["sigma_over_m_at_sculptor"] == pytest.approx(100 / 20.76, rel=1e-3)
    pp = sb.pairwise_preferences({"X": {"g1": 1.0, "g2": 5.0}, "Y": {"g1": 2.0, "g2": 3.0}})
    assert pp == {"X < Y": 1, "Y < X": 1}
    assert sb.bic(10.0, 5, 20) == pytest.approx(10.0 + 5 * math.log(20))


def test_letter_scope_is_declared_for_every_arm_and_reading():
    """Enmienda 1 del 27-oct: el alcance de cada letra, como en «Qué prueba PR-S y qué no»."""
    assert set(sb.LETTER_SCOPE) == {"MCV (c), letra sobre la mitad B", "MCMC-C (5E)", "Escalón de la RAR", "MCMC-J62",
                                    "Lecturas sin letra"}
    mcv = sb.LETTER_SCOPE["MCV (c), letra sobre la mitad B"]
    assert "Ren et al." in mcv["no_puede_decir"] and "ECV" in mcv["C"]
