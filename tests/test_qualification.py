"""Tipo de artefacto «cualificación de instrumento» (E8-Q): sin letras; puertas alcanzables (fallo cerrado);
criterio de exclusión de las láminas; los generadores de las dos cualificaciones recomponen sus artefactos
desde las corridas guardadas con una regla única."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from validation.qualification import (
    ARTIFACT_TYPE,
    assert_gate_attainable,
    load_qualification,
    sheet_cell_excluded,
    write_qualification,
)

REPO = Path(__file__).resolve().parent.parent


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
