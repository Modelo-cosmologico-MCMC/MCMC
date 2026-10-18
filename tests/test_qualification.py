"""Tipo de artefacto «cualificación de instrumento» (E8-Q): sin letras; puertas alcanzables (fallo cerrado);
criterio de exclusión de las láminas; los generadores de las dos cualificaciones recomponen sus artefactos
desde las corridas guardadas con una regla única."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from validation.qualification import (
    ARTIFACT_TYPE,
    assert_cell_gate_attainable,
    assert_cell_stationarity_attainable,
    assert_gate_attainable,
    assert_wall_budget_attainable,
    cell_report,
    load_qualification,
    sheet_cell_excluded,
    write_qualification,
)

REPO = Path(__file__).resolve().parent.parent
SHELLS = REPO / "results" / "2026-09-28_qualification_shells"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_qualification_artifact_has_no_letters_and_gate_rule_fails_closed(tmp_path):
    sha = write_qualification(tmp_path, "prueba", "0" * 40, {"x": [1]}, [{"key": "a"}],
                              {"energy_floor": {"value": 1e-3}}, {"rule": "r"}, ["n"])
    q = load_qualification(tmp_path)
    assert q["artifact_type"] == ARTIFACT_TYPE and q["_sha256"] == sha and len(sha) == 64
    rec = assert_gate_attainable(q, "energy_floor", 3e-3)                  # 3× el suelo: alcanzable
    assert rec["measured_floor"] == 1e-3 and rec["qualification_sha256"] == sha
    with pytest.raises(SystemExit):
        assert_gate_attainable(q, "energy_floor", 2e-3)                    # < 3× el suelo: no
    with pytest.raises(SystemExit):
        load_qualification(tmp_path / "no-existe")                         # falta la cualificación: fallo cerrado
    with pytest.raises(ValueError):
        write_qualification(tmp_path / "b", "prueba", "0" * 40, {}, [], {"verdict": "A"}, {}, [])
    assert (tmp_path / "qualification.md").read_text(encoding="utf-8").startswith("# Cualificación")


def _cell(arm, N, ds=0.04, eps=0.1, ru=True):
    return {"arm": arm, "N": N, "ds": ds, "eps_soft": eps, "rank_update": ru}


def _synthetic(tmp_path):
    runs = [  # la misma celda con dos dt_min: una sale del régimen débil, la otra se para en el suelo de dt
        {"key": "AS_a", **_cell("AS", 100_000), "dt_min_myr": 1e-5, "dE_rel_max": 1e-3, "t_weak_myr": 0.55, "t_final_myr": 0.55, "wall_s": 100.0},
        {"key": "AS_b", **_cell("AS", 100_000), "dt_min_myr": 1e-4, "dE_rel_max": 5e-4, "t_weak_myr": None, "t_final_myr": 0.54, "wall_s": 90.0},
        {"key": "AS_cut", **_cell("AS", 1_000_000), "dt_min_myr": 1e-4, "dE_rel_max": 1e-4, "t_weak_myr": None, "t_final_myr": 0.17, "wall_s": 480.0},
        {"key": "N_a", **_cell("newton", 100_000), "dt_min_myr": 1e-4, "dE_rel_max": 2e-6, "stationarity_dex": 0.05, "t_weak_myr": None, "t_final_myr": 50.0, "wall_s": 300.0},
        {"key": "N_big", **_cell("newton", 1_000_000), "dt_min_myr": 1e-4, "dE_rel_max": 1e-7, "stationarity_dex": 0.016, "t_weak_myr": None, "t_final_myr": 14.85, "wall_s": 480.0},
    ]
    write_qualification(tmp_path, "capas (sintética)", "0" * 40, {}, runs, {}, {}, [])
    return load_qualification(tmp_path)


def test_cell_gate_rule_uses_same_cell_at_same_evaluation_instant(tmp_path):
    """Revisión del autor del 30-sep: puerta = 3 × el suelo de la MISMA celda, evaluada en t_weak; las corridas que no
    llegaron a la salida no miden ese suelo; sin celda equivalente o sin llegada, fallo cerrado."""
    q = _synthetic(tmp_path)
    rec = assert_cell_gate_attainable(q, _cell("AS", 100_000), 3e-3)
    assert rec["measured_floor"] == 1e-3 and rec["runs_reaching_evaluation"] == ["AS_a"] and len(rec["runs_same_cell"]) == 2
    with pytest.raises(SystemExit):
        assert_cell_gate_attainable(q, _cell("AS", 100_000), 2.9e-3)       # < 3 × el suelo de la celda
    with pytest.raises(SystemExit):
        assert_cell_gate_attainable(q, _cell("AS", 1_000_000), 1.0)        # la celda se cortó antes de salir
    with pytest.raises(SystemExit):
        assert_cell_gate_attainable(q, _cell("AS", 100_000, eps=0.025), 1.0)  # sin celda equivalente
    assert cell_report(q, _cell("AS", 1_000_000))["status"] == "no llegó al instante de evaluación"
    assert cell_report(q, _cell("b005", 100_000))["status"] == "sin celda equivalente"
    with pytest.raises(ValueError):
        cell_report(q, {"arm": "AS", "N": 100_000})                         # la celda debe declarar todos sus campos


def test_wall_budget_and_stationarity_per_cell(tmp_path):
    q = _synthetic(tmp_path)
    assert assert_wall_budget_attainable(q, _cell("AS", 100_000), 150.0, margin=1.5)["wall_s_measured"] == 100.0
    with pytest.raises(SystemExit):
        assert_wall_budget_attainable(q, _cell("AS", 100_000), 140.0, margin=1.5)
    with pytest.raises(SystemExit):
        assert_wall_budget_attainable(q, _cell("AS", 1_000_000), 1e9)        # no salió con ningún presupuesto medido
    assert assert_cell_stationarity_attainable(q, _cell("AS", 100_000), 0.05, t_eval_myr=50.0)["measured_dex"] == 0.05
    with pytest.raises(SystemExit):
        assert_cell_stationarity_attainable(q, _cell("AS", 100_000), 0.04, t_eval_myr=50.0)
    with pytest.raises(SystemExit):                                         # N = 1e6 solo llegó a 14.85 Myr
        assert_cell_stationarity_attainable(q, _cell("AS", 1_000_000), 1.0, t_eval_myr=50.0)
    assert assert_cell_stationarity_attainable(q, _cell("AS", 1_000_000), 0.016, t_eval_myr=14.0)["measured_dex"] == 0.016


def test_cell_rule_on_the_frozen_shells_qualification():
    """Consecuencia publicada de la regla por celda sobre la cualificación de capas del 28-sep (artefacto intacto): la
    celda de producción a A_Sculptor (N = 1e6) se cortó a 0.167 Myr y no mide su suelo en t_weak; a N = 1e5 la celda
    (ds 0.04, ε 0.1, rango actualizado) sí, por la corrida con dt_min = 1e-5; 0.05·A_Sculptor no salió en ninguna celda;
    la estacionariedad newtoniana a N = 1e5 es por celda (0.0994 dex con ε_soft = 0.025, 0.0086 con 0.2)."""
    q = load_qualification(SHELLS)
    assert cell_report(q, _cell("AS", 1_000_000))["status"] == "no llegó al instante de evaluación"
    rep = cell_report(q, _cell("AS", 100_000))
    assert rep["status"] == "cualificada" and rep["runs_reaching_evaluation"] == ["AS_N100000_ds0.04_eps0.1_ru1_dtmin1e-05"]
    assert rep["floor"] == pytest.approx(4.31e-3, rel=0.01)
    assert all(cell_report(q, _cell("b005", r["N"], r["ds"], r["eps_soft"], r["rank_update"]))["status"] != "cualificada"
               for r in q["runs"] if r["arm"] == "b005")
    st = {e: cell_report(q, _cell("newton", 100_000, eps=e), "stationarity_dex", "t_end", 50.0)["floor"] for e in (0.025, 0.2)}
    assert st[0.025] == pytest.approx(0.0994, abs=1e-3) and st[0.2] == pytest.approx(0.0086, abs=1e-3)


def test_sheet_exclusion_criterion():
    c = sheet_cell_excluded(gamma_uv=50.0, t_window=0.5, amp_nonlinear=1e-2, amp_noise=1e-15)   # 25 < ln(1e13) = 29.9
    assert not c["excluded"]
    c = sheet_cell_excluded(gamma_uv=100.0, t_window=0.5, amp_nonlinear=1e-2, amp_noise=1e-15)  # 50 > 29.9
    assert c["excluded"]


def test_uv_growth_fit_rule():
    m = _mod("qualify_sheets")
    t = np.linspace(0.0, 0.4, 81)
    quiet = [[float(x), 1e-16 * (1 + 0.1 * np.sin(7 * x))] for x in t]
    assert m.fit_uv_growth(quiet)["gamma_uv"] == 0.0                       # nunca supera 10× el ruido
    growing = [[float(x), 1e-16 * np.exp(20.0 * x)] for x in t]
    fit = m.fit_uv_growth(growing)
    assert fit["gamma_uv"] == pytest.approx(20.0, rel=1e-6) and fit["n_points"] >= 4 and fit["r2"] > 0.999
    jump = [[float(x), 1e-16 if x < 0.39 else 1e-3] for x in t]              # creció sin racha ajustable: NaN (fallo cerrado)
    assert np.isnan(m.fit_uv_growth(jump)["gamma_uv"])


def test_shells_qualification_keys_are_deterministic():
    m = _mod("qualify_shells")
    cfg = {"N": 100_000, "ds": 0.04, "eps_soft": 0.1, "rank_update": True, "dt_min_myr": 1e-4}
    assert m.key_of(cfg, "newton") == "newton_N100000_ds0.04_eps0.1_ru1_dtmin0.0001"
    keys = {m.key_of(c, a) for c in m.configs() for a in m.GRID["arms"]}
    assert len(keys) == len(m.configs()) * len(m.GRID["arms"])
