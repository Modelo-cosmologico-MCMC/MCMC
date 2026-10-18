"""Regla de banda de las láminas (decisión del autor del 18-oct; preinscripciones nuevas, no retroactiva): la banda
inestable bajo k_c, el cierre de la ventana desde t = 0, el k_c admisible por celda, la regla UV con el cierre y el
candado de alcanzabilidad de las letras. Las cifras de referencia son las de la comprobación a posteriori del autor
sobre las rondas 3 y 3b (A_suelo ≈ 1e-10 estimado de esos mismos datos)."""

import json
import math
from pathlib import Path

import pytest

from validation.qualification import (
    admissible_k_cut,
    assert_letters_reachable,
    band_growth_rate,
    band_rule_cell,
    sheet_cell_excluded,
    sheet_cell_excluded_at_close,
    window_close_time,
)

REPO = Path(__file__).resolve().parent.parent
NG, A_FLOOR = 512, 1e-10
CELLS = [(q, n) for q in (1.2, 1.5, 2.0) for n in (4, 8, 16, 32)]


def test_band_below_kc_and_its_fastest_mode():
    for q, edge, fastest, ratio in ((1.2, 59, 34, 5.76), (1.5, 88, 51, 8.46), (2.0, 114, 66, 10.86)):
        b = band_growth_rate(q, NG // 4, NG)
        g4 = math.log(30.0) / window_close_time(q, 4, NG)
        assert b["n_unstable_max"] == edge and b["n_fastest"] == fastest
        assert b["gamma_band"] / g4 == pytest.approx(ratio, abs=0.01)
    assert band_growth_rate(0.8, NG // 4, NG)["gamma_band"] == 0.0           # q < 1: no hay banda


def test_close_time_is_relative_and_matches_the_instrument_tables():
    # t_cierre = ln 30/γ_inst: (1.5, 4) prevista en 0.396 en la preinscripción de la ronda 3b, con cualquier siembra
    assert window_close_time(1.5, 4, NG) == pytest.approx(0.396, abs=1e-3)
    pre = json.loads((REPO / "results/2026-10-18_cj_criterion_round3b_q1p5/preregistration.json").read_text(encoding="utf-8"))
    for key, w in pre["windows_and_T"].items():
        q, n = float(key[1:4]), int(key.split("_n")[1])
        assert window_close_time(q, n, NG) == pytest.approx(w["t_window_close"], rel=1e-9)
    assert math.isinf(window_close_time(0.8, 4, NG))


def test_band_rule_marks_exactly_the_four_contaminated_cells():
    rows = {c: band_rule_cell(*c, NG, A_FLOOR) for c in CELLS}
    assert {c for c, r in rows.items() if r["excluded"]} == {(1.2, 4), (1.5, 4), (2.0, 4), (2.0, 8)}
    for c, ef in (((1.2, 4), 19.6), ((2.0, 8), 18.5), ((1.5, 4), 28.8), ((2.0, 4), 36.9)):
        assert rows[c]["efolds_to_close"] == pytest.approx(ef, abs=0.05)
    assert max(r["efolds_to_close"] for c, r in rows.items() if not r["excluded"]) == pytest.approx(14.5, abs=0.05)
    assert rows[(1.5, 4)]["ln_budget"] == pytest.approx(math.log(1e-2 / (10 * A_FLOOR)))
    # banda prevista en el cierre frente a la medida en el diagnóstico de la ronda 3b: (1.5, 8) 2.0e-4 frente a 2.1e-4
    assert rows[(1.5, 8)]["predicted_band_at_close"] == pytest.approx(2.0e-4, rel=0.05)


def test_admissible_k_cut_per_cell():
    want = {(1.2, 4): 21, (1.5, 4): 19, (2.0, 4): 19, (2.0, 8): 44}
    for c in CELLS:
        assert admissible_k_cut(*c, NG, A_FLOOR)["n_c"] == want.get(c, NG // 4), c
    # el corte admisible pasa la regla y el siguiente ya no
    for c, n_c in want.items():
        assert not band_rule_cell(*c, NG, A_FLOOR, n_c=n_c)["excluded"]
        assert band_rule_cell(*c, NG, A_FLOOR, n_c=n_c + 1)["excluded"]
    assert admissible_k_cut(0.8, 4, NG, A_FLOOR)["n_c"] is None                # sin ventana que medir


def test_seed_amplitude_cannot_rescue_the_slow_cell():
    r = band_rule_cell(1.5, 4, NG, A_FLOOR)
    need = 1e-2 * math.exp(-r["efolds_to_close"])                             # suelo que haría falta, sin margen
    assert need == pytest.approx(3.1e-15, rel=0.05) and need < 1e-14


def test_uv_rule_with_the_close_time():
    noise = max(2.3e-16, 2e-5 ** 2)
    expect = {(1.2, 4, 5.833): (5.31, False), (2.0, 8, 128.8): (14.31, False), (1.5, 4, 43.45): (17.19, True),
              (1.5, 8, 43.45): (8.65, False)}
    for (q, n, g_uv), (prod, excl) in expect.items():
        r = sheet_cell_excluded_at_close(g_uv, q, n, NG, 1e-2, noise)
        assert r["gamma_uv_times_t_close"] == pytest.approx(prod, abs=0.01) and r["excluded"] is excl
        assert r["ln_budget"] == pytest.approx(17.03, abs=0.01)
    # con la duración de la ventana (como en las rondas 3 y 3b) la misma celda (1.5, 4) no se excluía
    assert not sheet_cell_excluded(43.45, math.log(10.0) / (math.log(30.0) / window_close_time(1.5, 4, NG)), 1e-2, noise)["excluded"]


def test_band_rule_needs_a_measured_floor():
    with pytest.raises(ValueError):
        band_rule_cell(1.5, 4, NG, 0.0)
    with pytest.raises(ValueError):
        admissible_k_cut(1.5, 4, NG, -1.0)
    assert band_rule_cell(1.5, 4, NG, A_FLOOR, n_c=4)["excluded"]             # el corte deja fuera el modo sembrado


def test_letters_reachable_lock():
    # ronda 3b: B exigía la celda n = 4 medida (fuera de tolerancia); excluirla sin más la dejaba sin celda
    req_3b = {"A": [{(1.5, 8), (1.5, 16), (1.5, 32)}], "B": [{(1.5, 4), (1.5, 8)}], "C": [{(1.5, 8)}]}
    assert assert_letters_reachable(req_3b, {(1.5, 4), (1.5, 8), (1.5, 16), (1.5, 32)})["reachable"]["B"]
    with pytest.raises(SystemExit):
        assert_letters_reachable(req_3b, {(1.5, 8), (1.5, 16), (1.5, 32)})
    # ronda 3c (opcional): B con alguna n = 4 y (2.0, 8) medidas
    req_3c = {"B": [{(1.2, 4), (2.0, 8)}, {(1.5, 4), (2.0, 8)}, {(2.0, 4), (2.0, 8)}], "C": [{(2.0, 8)}]}
    assert assert_letters_reachable(req_3c, {(1.5, 4), (2.0, 8)})["reachable"] == {"B": True, "C": True}
    with pytest.raises(SystemExit):
        assert_letters_reachable(req_3c, {(2.0, 8)})
