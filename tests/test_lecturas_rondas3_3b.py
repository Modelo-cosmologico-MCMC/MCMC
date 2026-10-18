"""Candado de las lecturas a posteriori de las rondas 3 y 3b (sin letra, sin corridas): el artefacto no lleva letras, sus
cifras son las de la comprobación del autor del 18-oct y el bloque de láminas coincide con la regla de banda de
validation.qualification."""

import json
from pathlib import Path

import pytest

from validation.qualification import FORBIDDEN_KEYS, admissible_k_cut

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-10-18_lecturas_rondas3_3b" / "lecturas.json"

pytestmark = pytest.mark.skipif(not OUT.exists(), reason="lecturas no escritas")


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def _doc():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_no_letters_in_the_readings():
    d = _doc()
    assert not {str(k).lower() for k in _keys(d)} & FORBIDDEN_KEYS
    assert "SIN LETRA" in d["status"]


def test_sheet_readings_match_the_author_check():
    s = _doc()["sheets"]
    assert {q: (b["n_unstable_max"], b["n_fastest"]) for q, b in s["band"].items()} == {"1.2": (59, 34), "1.5": (88, 51), "2.0": (114, 66)}
    assert s["implied_floor"]["median"] == pytest.approx(1.0e-10, rel=0.1)
    assert [tuple(c) for c in s["band_rule"]["excluded_margin10"]] == [(1.2, 4), (1.5, 4), (2.0, 4), (2.0, 8)]
    for c in s["band_rule"]["cells"]:
        assert c["excluded_margin10"] == c["excluded_no_margin"]                        # el margen ×10 no cambia qué celdas caen
    uv = {tuple(u["cell"]): u for u in s["uv_rule_at_close"]}
    assert uv[(1.5, 4)]["excluded_at_close"] and not uv[(1.2, 4)]["excluded_at_close"] and not uv[(2.0, 8)]["excluded_at_close"]
    assert s["admissible_n_c"] == {f"{q}_{n}": admissible_k_cut(q, n, 512, s["band_rule"]["a_floor"])["n_c"]
                                   for q in (1.2, 1.5, 2.0) for n in (4, 8, 16, 32)}
    assert s["seed_argument"]["floor_needed_no_margin"] == pytest.approx(3.1e-15, rel=0.05)
    nine = s["round3_robustness"]["nueve"]
    assert nine["1.2"]["max_abs_dev"] == pytest.approx(0.027, abs=5e-4) and nine["1.2"]["rel_spread"] == pytest.approx(0.025, abs=5e-4)
    assert nine["2.0"]["max_abs_dev"] == pytest.approx(0.011, abs=5e-4) and nine["2.0"]["rel_spread"] == pytest.approx(0.008, abs=5e-4)
    assert max(s["round3_robustness"]["q0.8_growth_factors"]) <= 3.0


def test_halo_readings_match_the_author_check():
    h = _doc()["halo"]
    l1 = h["L1_saturating_responds_to_added_force"]
    assert [round(r["ratio_01"], 2) for r in l1["series"][:2]] == [3.34, 3.02] and l1["eps_max_ratio"] == pytest.approx(3.0, rel=1e-3)
    assert max(abs(r["sub_01"]) for r in l1["early_below_0p5_myr"]) <= 2e-4
    assert (round(l1["D_F_at_soft"]["sub"], 2), round(l1["D_F_at_soft"]["above"], 2)) == (1.25, 3.75)
    assert l1["above_q_newton_gt1_below_kpc"] == pytest.approx(0.118, abs=1e-3)
    assert (round(l1["q_equilibrium"]["sub"]["q_eq_max"], 2), round(l1["q_equilibrium"]["above"]["q_eq_max"], 2)) == (0.33, 0.67)
    l2 = h["L2_no_viable_unstable_point"]
    assert l2["rho_nfw_0p1"] == pytest.approx(0.5482, abs=1e-4) and l2["sigma_n_0p1"] == pytest.approx(21.5, abs=0.05)
    assert not any(r["unstable_resolved_0.1"] or r["unstable_resolved_0.05"] for r in l2["rows"])
    assert max(r["q_eq_max_r_ge_3eps_0.1"] for r in l2["rows"]) == pytest.approx(0.25, abs=0.01)
    assert max(r["q_eq_max_r_ge_3eps_0.05"] for r in l2["rows"]) == pytest.approx(0.55, abs=0.01)
    l3 = h["L3_t_weak_not_common_milestone"]
    assert l3["density_factor"] == pytest.approx(7.37, abs=0.01)
    assert (round(l3["by_eps"]["0.05"]["ratio_t_weak"]), round(l3["by_eps"]["0.025"]["ratio_t_weak"])) == (128, 45)
    assert not l3["by_eps"]["0.2"]["b005_exits"] and not l3["by_eps"]["0.1"]["b005_exits"]
    l4 = h["L4_global_energy_gate_inverts"]
    assert l4["sweep_b005_eps0.2"]["dE_rel"] < l4["sweep_AS_eps0.2"]["dE_rel"] and l4["sweep_b005_eps0.2"]["eta_E"] > l4["sweep_AS_eps0.2"]["eta_E"]
    assert l4["sweep_AS_eps0.025"]["eta_E"] == pytest.approx(0.634, abs=1e-3)
    assert l4["sweep_b005_eps0.05"]["eta_E_first_segment"] == pytest.approx(0.80, abs=0.01)
    l5 = h["L5_newtonian_breathes"]
    assert l5["dlogM01_min"] == pytest.approx(-0.222, abs=1e-3) and l5["newtonian_arms"] == ["newton_N1e6"]
    l6 = h["L6_uv_signature_A_S"]
    assert [round(v, 2) for v in l6["p_eps_local"].values()] == [2.29, 2.98, 2.84]
    assert [round(v, 2) for v in l6["p_ds_local"].values()] == [0.97, 2.14]
