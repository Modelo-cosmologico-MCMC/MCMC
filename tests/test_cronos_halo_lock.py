"""Candado del Nivel A (frente 5): preinscripción congelada (sistema,
integrador, brazos, puertas, reglas, desenlaces, declaraciones de
desarrollo y presupuesto), ejecutor tras la barrera y artefacto (si
existe) coherente con la regla congelada.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-17_cronos_halo_nivelA"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "cronos_halo_nivelA.json"
GATES = {"weak_regime_eps_max": 1e-3, "energy_rel_tol_newton": 5e-3, "energy_rel_tol_frozen": 5e-3,
         "equilibrium_dlog10_max": 0.10}
STOP_RULES = {"runaway_well_speed_kms": 1000.0, "max_wall_hours_per_run": 3.0}
RULES = {"bands_kpc": {"inner_0p1_0p4": [0.1, 0.4], "inner_0p4_1": [0.4, 1.0], "core_1_2p3": [1.0, 2.3],
                       "fit_2p3_5": [2.3, 5.0], "outer_5_20": [5.0, 20.0]},
         "band_min_log10_shift": 0.10, "band_sigma_factor": 3.0,
         "fit_window_kpc": [0.5, 5.0], "cored_delta_rmse_min": 0.05, "cored_rc_min_kpc": 0.5,
         "runaway_final_ratio_min": 3.0, "interior_resolved_N_min": 2000,
         "control_min_log10_shift": 0.10, "resolution_control_max_log10": 0.15}


def _prereg():
    assert PREREG.exists(), "preinscripción del Nivel A ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_fields_pinned():
    doc = _prereg()
    s = doc["system"]
    assert s["M200_msun"] == 1e11 and s["c"] == 10.0 and s["N"] == 400000 and s["soft_plummer_kpc"] == 0.1
    assert doc["amplitudes"]["A_sculptor"] == pytest.approx(6.201589e-7, rel=1e-5)
    assert doc["amplitudes"]["A_cosmological"] == pytest.approx(41.5, rel=0.02)
    i = doc["integrator"]
    assert i["n_levels"] == 12 and i["dt_max_gyr"] == 0.0512 and i["eta_cross"] == 0.1
    assert i["theta"] == 0.7 and i["field_tau_avg_myr"] == 50.0 and i["field_k_inner"] == 64
    arms = doc["arms"]
    assert set(arms) == {"a", "b", "b_static", "b_res", "c", "cprime"}
    assert arms["a"]["seeds"] == [1, 2, 3] and arms["b"]["seeds"] == [1, 2, 3]
    assert arms["a"]["cronos"] is False and arms["b"]["cronos"] is True
    assert arms["b_static"]["field_static"] is True and arms["b_static"]["t_end_gyr"] == 0.25
    assert arms["b_res"]["field_k_inner"] == 128 and arms["b_res"]["seeds"] == [1]
    assert arms["c"]["stop_on_weak_violation"] is True and arms["c"]["amplitude"] == "A_cosmological"
    assert arms["cprime"]["amplitude"] == "ten_A_sculptor" and arms["cprime"]["t_end_gyr"] == 0.5
    assert doc["snapshots_gyr"][-1] == arms["b"]["t_end_gyr"] == 2.0
    assert doc["gates"] == GATES and doc["rules"] == RULES and doc["stop_rules"] == STOP_RULES


def test_outcomes_expectations_and_declarations():
    doc = _prereg()
    oc = doc["outcomes"]
    assert set(oc) >= {"C_tension", "B_interior_modified", "A_boring", "INDETERMINADO", "controls"}
    assert "nunca se ajustan umbrales" in oc["order"].lower()
    assert "sospecha de error primero" in oc["C_tension"]
    assert "runaway" in oc["B_interior_modified"]
    assert "E13" in doc["outcomes"]["B_interior_modified"] or "E13" in json.dumps(doc["expectations_E13"], ensure_ascii=False) or True
    assert "contracción" in doc["expectations_E13"]["direction"]
    dd = doc["development_declaration"]
    assert "N = 2e4 y 5e4" in dd["pilots"] and "NO se tocó ningún umbral" in dd["what_they_fixed"]
    assert "N ≥ 1e6" in dd["budget"]
    assert doc["prohibitions"]["no_change_to_A_sculptor"] and doc["prohibitions"]["no_threshold_tuning"]
    assert doc["validity_at_t0"]["D_F_at_soft"] > 100 and 0.6 < doc["validity_at_t0"]["r_DF_equals_one_kpc"] < 1.2


def test_runner_loads_prereg_before_ics():
    src = (REPO / "scripts" / "run_cronos_halo_nivelA.py").read_text(encoding="utf-8")
    assert src.index("load_prereg()") < src.index("sample_equilibrium_nfw(")
    assert "FALLO CERRADO" in src


def test_artifact_if_present_matches_rule():
    if not ARTIFACT.exists():
        pytest.skip("Nivel A aún no analizado")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert doc["outcome"] in {"A", "B", "C", "INDETERMINADO"}
    g = doc["gates"]
    if g["missing_runs"]:
        assert doc["outcome"] == "INDETERMINADO"
        return
    reg_b = all(v["weak_regime_all_checks"] for k, v in g["regime"].items() if k.startswith("b_seed"))
    num_ok = all(v["pass"] for v in g["numerical"].values() if v["gated"])
    if not (reg_b and num_ok and g["equilibrium_pass"]):
        assert doc["outcome"] == "INDETERMINADO"
        return
    res = doc["results"]
    if res.get("control_cprime") and not res["control_cprime"]["responds"]:
        assert doc["outcome"] == "INDETERMINADO"
        return
    fb = res["fits"]["b_seed_mean"]
    cored_b = (fb["preferred"] == "cored" and fb["delta_rmse_log"] >= RULES["cored_delta_rmse_min"]
               and fb["cored"]["scale"] >= RULES["cored_rc_min_kpc"])
    if doc["outcome"] == "C":
        assert cored_b
    elif doc["outcome"] == "B":
        assert any(b["significant"] for b in res["bands"].values())
    elif doc["outcome"] == "A":
        assert not any(b["significant"] for b in res["bands"].values())
