"""PR-S de principio a fin (dynamics/sparc_pr_s.py) sobre una muestra SINTÉTICA y declarada, con rejillas pequeñas: el
análisis corre entero, la letra queda INDETERMINADO por sus puertas sin bytes ni E8-Q, y salen las lecturas sin letra
(memoria del halo, rama densa y colapso, lapse, r_c(M200), SPARC → Sculptor, q_CJ(0) cruzado, banco). También: las decisiones
pendientes y los hallazgos están escritos, y la a₀ poblacional se recupera en galaxias MOND sintéticas."""

import math

import numpy as np
import pytest

from dynamics import mcv_sparc as ms
from dynamics import sparc_bank as sb
from dynamics import sparc_pr_s as pr
from dynamics.epsilon_c_saturating import exponential_disc_g_bar
from tests.test_mcv_sparc import _synthetic

SPEC_MCV = {**ms.FIT_SPEC, "log_rho0": [6.0, 9.5], "dlog_rho0": 0.125, "log_sigma0": [1.2, 2.1], "dlog_sigma0": 0.075,
            "k_grid": [0.0], "nodes": {"r_lo": 1e-5, "r_hi": 1e3, "h": 0.08}}
SPEC_BANK = {**sb.BANK_SPEC, "nfw": {"log_M200": [9.0, 13.5], "dlog_M200": 0.1, "log_c": [0.0, 2.0], "dlog_c": 0.05},
             "burkert": {"log_rho": [5.0, 11.0], "dlog_rho": 0.1, "log_rc": [-1.0, 2.0], "dlog_rc": 0.05},
             "j62": {"log_M200": [9.0, 13.5], "dlog_M200": 0.05, "log_rstar": [-0.5, 1.5], "dlog_rstar": 0.5},
             "mond_free": {"log_a0": [-11.0, -9.0], "dlog_a0": 0.05}, "sigma100_grid": [0.5, 1.0, 2.0],
             "a_grid": [0.0, 0.5], "letters": {**sb.BANK_SPEC["letters"], "n_B_min": 2, "n_boot": 200}}


@pytest.fixture(scope="module")
def result():
    gals = []
    for k, (M, som) in enumerate([(10 ** 10.6, 1.0), (10 ** 10.9, 0.5), (1e11, 1.0), (10 ** 11.2, 2.0)]):
        g, _, _ = _synthetic(M, som, R_d=1.5 + 0.2 * k, noise=True, seed=k)
        g.name, g.quality = f"S{k}", 1
        gals.append(g)
    plan = {"spec_mcv": SPEC_MCV, "spec_bank": SPEC_BANK, "j62_rstars": [0.5, 1.8, 5.0],
            "modes": {"t10": pr.settings_grid(SPEC_BANK), "concentración": pr.settings_grid(SPEC_BANK, "concentración", c1=4.1),
                      "t6": pr.settings_grid(SPEC_BANK, "fijo", t_age=6.0)}}
    A = 6.2e-7
    return pr.analysis(gals, plan, local_law_A={"5E": (lambda r: A * r ** 1.5, lambda r: 1.5 * A * r ** 0.5)})


def test_the_letter_waits_for_bytes_and_qualification(result):
    assert result["letra"]["letter"] == "INDETERMINADO"
    assert {"bytes sin manifiesto", "cualificación del ajustador (E8-Q) no superada"} <= set(result["letra"]["reasons"])
    assert result["n_muestra"] == 4 and len(result["mitades"]["A"]) == 2 and result["calibracion_A"]["best"] is not None
    assert "MCV (c), letra sobre la mitad B" in result["alcance"]


def test_readings_without_letter_are_there(result):
    lec = result["lecturas"]
    rows = lec["por_galaxia"]
    assert len(rows) == 4 and all(r["branch"] in ("con núcleo", "densa") for r in rows)
    for r in rows:
        assert set(r["t_c_gyr"]) == {"0.75", "0.45"} and 1e-8 < r["Delta_N0"] < 1e-5 and r["sigma0"] > 0.0
        assert "restringido" in r and r["q_CJ0_cruzado"]["5E"]["q_CJ0"] == pytest.approx(r["q_CJ0_cruzado"]["5E"]["q_CJ0_general"])
    assert set(result["memoria_del_halo"]) == {"concentración", "t6"}
    assert "delta_chi2_B" in result["memoria_del_halo"]["t6"]
    assert lec["sparc_sculptor"]["v_rel_kms"] == pytest.approx(20.76, abs=0.01)
    assert "43 pc" in lec["cuspide_agujero_negro"] and "MCV (c)" in result["banco"]["delta_BIC_frente_a_LCDM2"]


def test_pending_decisions_and_findings_are_written():
    ids = [d["id"] for d in pr.PENDING_DECISIONS]
    assert ids == ["27oct-1", "27oct-2", "27oct-3", "27oct-4", "26oct-1", "26oct-2", "26oct-3", "26oct-4", "26oct-5",
                   "26oct-6"]
    assert [h["id"] for h in pr.EXECUTOR_FINDINGS] == [f"H{i}" for i in range(1, 10)]


def test_population_a0_is_recovered_on_mond_galaxies():
    """Galaxias SINTÉTICAS que siguen la RAR de un parámetro con a₀ = 1.2e-10 m/s²."""
    gals = []
    for k in range(3):
        R = np.geomspace(0.5, 15.0, 15)
        v2b = exponential_disc_g_bar(R, 300.0 / (k + 1), 2.0 + k) * R
        v = np.sqrt(sb.mond_v2(v2b, R))
        gals.append(ms.Galaxy(f"M{k}", R, v, 0.03 * v, np.zeros_like(R), np.sqrt(v2b / 0.5), np.zeros_like(R), 10.0, 1.0,
                              60.0, 4.0, synthetic=True))
    a0 = pr.population_a0(gals, np.geomspace(0.8e-10, 1.6e-10, 81), {**ms.FIT_SPEC, "k_grid": [-1.0, 0.0, 1.0]})
    assert a0["a0"] == pytest.approx(1.2e-10, rel=0.01) and not a0["on_edge"] and math.isfinite(a0["sigma"])
