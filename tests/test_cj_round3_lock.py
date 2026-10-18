"""Candado de la ronda 3 del test del criterio de Cronos–Jeans: la preinscripción (si está congelada) cita la
cualificación E8-Q de las láminas y excluye a priori las celdas que ella excluye; el analizador lee las reglas del
JSON y falla cerrado; la ronda 2 y la cualificación quedan intactas."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-28_cj_criterion_round3"
PREREG = OUT / "preregistration.json"
ROUND2 = REPO / "results" / "2026-09-22_cj_criterion_round2"
QUAL = REPO / "results" / "2026-09-28_qualification_sheets" / "qualification.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _mod():
    spec = importlib.util.spec_from_file_location("cj3", REPO / "scripts" / "run_cj_round3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_exclusion_comes_from_the_qualification_and_names_round2_failure():
    """La celda que la cualificación excluye a priori con el instrumento de la ronda 2 es la que falló en la ronda 2."""
    m = _mod()
    qual = json.loads(QUAL.read_text(encoding="utf-8"))
    excluded = [f"q{q}_n{n}" for q in (1.2, 2.0) for n in m.MODES if m._qual_cell(qual, q, n)["excluded"]]
    assert excluded == ["q2.0_n4"]
    r2 = json.loads((ROUND2 / "cj_criterion_round2.json").read_text(encoding="utf-8"))
    assert r2["outcome"] == "INDETERMINADO" and "q2.0_n4" in " ".join(r2["reasons"])
    assert m._qual_t_uv(qual, 2.0) < 0.3 and m._qual_t_uv(qual, 1.2) > 1.0
    with pytest.raises(SystemExit, match="FALLO CERRADO"):
        m._qual_cell(qual, 7.0, 4)


@pytest.mark.skipif(not PREREG.exists(), reason="preinscripción de la ronda 3 no congelada")
def test_round3_frozen_fields_pinned():
    d = json.loads(PREREG.read_text(encoding="utf-8"))
    assert _sha(PREREG) in (OUT / "preregistration.md").read_text(encoding="utf-8")
    assert d["qualification"]["sha256"] == _sha(QUAL)                      # la cualificación citada es la del repositorio, intacta
    assert d["q_grid"] == [0.8, 1.2, 2.0] and d["modes"] == [4, 8, 16, 32]
    ins = d["instrument"]
    assert ins["ng"] == 512 and ins["n_beams"] == 1024 and ins["cells_per_beam"] == 4 and ins["k_cut_frac"] == 0.5
    assert d["excluded_cells_a_priori"] == ["q2.0_n4"]
    r = d["rules"]
    assert r["rate_rel_tol"] == 0.25 and r["min_points_linear"] == 8 and r["min_r2"] == 0.98 and r["uv_amp_nonlinear"] == 1e-2
    for key, w in d["windows_and_T"].items():
        if w["window"] is not None and not w["excluded_a_priori"]:
            assert w["T"] <= w["t_uv_nonlinear"] and w["window_closes_before_uv_nonlinear"], key
    assert d["prohibitions"]["round2_untouched"] and d["prohibitions"]["qualification_untouched"]


def test_round2_untouched():
    assert (ROUND2 / "preregistration.json").exists()
    assert json.loads((ROUND2 / "cj_criterion_round2.json").read_text(encoding="utf-8"))["outcome"] == "INDETERMINADO"


def test_analyzer_reads_rules_from_json_and_fails_closed(tmp_path):
    body = (REPO / "scripts" / "run_cj_round3.py").read_text(encoding="utf-8").split("def cmd_analyze(")[1]
    assert 'R["rate_rel_tol"]' in body and 'R["min_r2"]' in body and 'R["uv_amp_nonlinear"]' in body
    assert re.search(r"r2\s*>=\s*0\.98", body) is None
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)"
            % (str(REPO), str(REPO / "scripts" / "run_cj_round3.py"), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


@pytest.mark.skipif(not (OUT / "runs").exists(), reason="sin corridas de la ronda 3")
def test_runs_cite_frozen_sha_lattice_and_filter():
    sha = _sha(PREREG)
    for p in (OUT / "runs").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["preregistration_sha256"] == sha and d["lattice_exact"] is True and d["k_cut_frac"] == 0.5, p.name
