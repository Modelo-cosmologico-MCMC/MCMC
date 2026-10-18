"""Candado de la preinscripción de la ronda 3 del halo (capas esféricas): el .md cita el sha256 del .json congelado;
el analizador lee las puertas del JSON; las corridas presentes citan el sha congelado; falla cerrado sin
preinscripción; y la tabla por celda anunciada antes de la segunda ejecución no tiene letra ni escribe el desenlace."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-28_halo_round3"
PREREG = OUT / "preregistration.json"
SCRIPT = REPO / "scripts" / "run_halo_round3.py"
FROZEN_SHA = "6fba6b53d92910db86e21ed5ae9c8943d9c50f32dcb33b52e3d6eae01bf513ec"

pytestmark = pytest.mark.skipif(not PREREG.exists(), reason="preinscripción aún no congelada")


def _module():
    spec = importlib.util.spec_from_file_location("run_halo_round3_mod", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_prereg_is_the_frozen_one_and_md_cites_it():
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert sha == FROZEN_SHA
    assert sha in (OUT / "preregistration.md").read_text(encoding="utf-8")
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert len(doc["arms"]) == 17 and doc["instrument"]["max_wall_hours_per_run"] == 4.0
    assert doc["rules"]["energy_gate_factor"] == 3.0 and doc["rules"]["order"].startswith("puertas")


def test_analyzer_reads_gates_from_json():
    body = SCRIPT.read_text(encoding="utf-8").split("def cmd_analyze(")[1].split("def _same_cell_runs(")[0]
    assert 'G["tol_E_by_class"]' in body and 'G["stationarity_dex"]' in body and 'R["p_power_law_min"]' in body


def test_runs_if_present_cite_frozen_sha():
    runs = sorted((OUT / "runs").glob("*.json")) if (OUT / "runs").exists() else []
    if not runs:
        pytest.skip("sin corridas")
    for p in runs:
        assert json.loads(p.read_text(encoding="utf-8"))["preregistration_sha256"] == FROZEN_SHA, p.name


def test_analyze_fails_closed_without_prereg(tmp_path):
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)" % (str(REPO), str(SCRIPT), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


def _fake_run(path: Path, cronos: bool, dE: float, t_exit_gyr=None, t_final_gyr=0.05):
    key = "E_self" if cronos else "E_grav_only"
    t_last = t_exit_gyr if t_exit_gyr is not None else t_final_gyr
    snaps = [{"t_gyr": 0.0, "energy": {key: -1.0}, "eps_c_max": 0.0, "r_eps_max_kpc": 0.1, "M_within": {"0.1": 1.0, "0.4": 2.0}},
             {"t_gyr": t_last, "energy": {key: -1.0 - dE}, "eps_c_max": 2e-3 if t_exit_gyr is not None else 1e-4,
              "r_eps_max_kpc": 0.1, "M_within": {"0.1": 1.0, "0.4": 2.0}}]
    events = [{"kind": "régimen débil violado"}] if t_exit_gyr is not None else []
    path.write_text(json.dumps({"snapshots": snaps, "events": events, "t_final_gyr": t_last, "stopped_early": True, "stop_reason": "prueba",
                                "n_steps": 2, "wall_s": 1.0, "preregistration_sha256": FROZEN_SHA, "N_within_0p4_initial": 2000}))


def test_per_cell_table_has_no_letter_and_uses_the_same_cell_at_t_weak(tmp_path):
    m = _module()
    runs = tmp_path / "runs"
    runs.mkdir()
    _fake_run(runs / "ctrlN_AS_N1e5.json", True, 1e-3, t_exit_gyr=5e-4)       # la cualificación la vio salir: suelo propio
    _fake_run(runs / "sweep_AS_eps0.1.json", True, 1e-3, t_final_gyr=2e-3)    # su celda se cortó antes de salir
    _fake_run(runs / "sat_sub_ds0.04.json", True, 1e-5)                       # la forma saturante no tiene cualificación
    _fake_run(runs / "newton_N1e6.json", False, 3e-6)                         # la celda newtoniana N = 1e6 no llegó a 50 Myr
    m.RUNS, m.OUTDIR = runs, tmp_path
    assert m.cmd_cells(None) == 0
    res = json.loads((tmp_path / "per_cell_energy.json").read_text(encoding="utf-8"))
    rows = res["rows"]
    assert rows["ctrlN_AS_N1e5"]["qualification_status"] == "cualificada"
    assert rows["ctrlN_AS_N1e5"]["cell_floor"] == pytest.approx(4.31e-3, rel=0.01)
    assert rows["sweep_AS_eps0.1"]["qualification_status"] == "no llegó al instante de evaluación"
    assert rows["sat_sub_ds0.04"]["qualification_status"] == "sin celda equivalente"
    assert rows["newton_N1e6"]["qualification_status"] == "no llegó al instante de evaluación"
    assert len(res["missing_runs"]) == 13
    assert not (tmp_path / "halo_round3.json").exists() and "verdict" not in res
    assert "No es una puerta ni cambia el desenlace" in (tmp_path / "per_cell_energy.md").read_text(encoding="utf-8")
