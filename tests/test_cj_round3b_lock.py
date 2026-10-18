"""Candado de la ronda 3b (ampliación a q = 1.5) del test del criterio de Cronos–Jeans: la adenda de cualificación es
una cualificación sin letras con la misma regla que la base; la preinscripción (si está congelada) usa el instrumento,
las tolerancias y la regla de exclusión de la ronda 3 y cita la adenda; el analizador lee las reglas del JSON y falla
cerrado; la ronda 3 queda intacta."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from validation.qualification import load_qualification

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-10-18_cj_criterion_round3b_q1p5"
PREREG = OUT / "preregistration.json"
ROUND3 = REPO / "results" / "2026-09-28_cj_criterion_round3"
QUAL = REPO / "results" / "2026-10-18_qualification_sheets_q1p5" / "qualification.json"
QUAL_BASE = REPO / "results" / "2026-09-28_qualification_sheets" / "qualification.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_q1p5_profile_changes_only_the_declared_grid_fields():
    m = _mod("qualify_sheets")
    changed = {k for k in m.GRID if m.GRID[k] != m.GRID_Q15[k]}
    assert changed == {"ng", "cells_per_beam", "q", "n_beams", "round3_cells"} and set(m.GRID) == set(m.GRID_Q15)
    assert m.GRID_Q15["q"] == [1.5] and m.GRID_Q15["n_beams"] == [1024] and m.GRID_Q15["ng"] == [512]
    assert m.PROFILES["base"][1] == m.OUTDIR and "2026-10-18" in str(m.PROFILES["q1p5"][1])


@pytest.mark.skipif(not QUAL.exists(), reason="adenda de cualificación en q = 1.5 no escrita")
def test_q1p5_addendum_is_a_qualification_with_the_base_rule():
    q, base = load_qualification(QUAL), load_qualification(QUAL_BASE)
    assert q["exclusion_criteria"]["rule"] == base["exclusion_criteria"]["rule"]
    assert q["exclusion_criteria"]["amp_nonlinear_declared"] == base["exclusion_criteria"]["amp_nonlinear_declared"]
    cells = q["exclusion_criteria"]["cells"]
    assert {f"ng512_nb1024_kc{kc}_q1.5_n{n}" for kc in (None, 0.5) for n in (4, 8, 16, 32)} == set(cells)
    assert {r["key"] for r in q["runs"]} == {"ng512_nb1024_p4_q1.5_kcNone", "ng512_nb1024_p4_q1.5_kc0.5"}
    assert all(r["lattice_exact"] for r in q["runs"])


@pytest.mark.skipif(not PREREG.exists(), reason="preinscripción de la ronda 3b no congelada")
def test_round3b_frozen_fields_pinned_and_equal_to_round3():
    d = json.loads(PREREG.read_text(encoding="utf-8"))
    r3 = json.loads((ROUND3 / "preregistration.json").read_text(encoding="utf-8"))
    assert _sha(PREREG) in (OUT / "preregistration.md").read_text(encoding="utf-8")
    assert d["qualification"]["sha256"] == _sha(QUAL)
    assert d["q_grid"] == [1.5] and d["modes"] == [4, 8, 16, 32]
    ins, ins3 = dict(d["instrument"]), dict(r3["instrument"])
    for k in ("N", "ng", "dt", "sample_every", "full_modes_every", "quiet_start", "n_beams", "cells_per_beam", "k_cut_frac",
              "uv_band_frac", "seed_eigenmode", "seed_amp_by_mode", "seed", "T_safety", "delta_end"):
        assert ins[k] == ins3[k], k
    assert ins["beam_convergence_run"] == {"q": 1.5, "mode": 8, "n_beams_alt": 512, "cells_per_beam_alt": 8}
    for k in ("linear_window_lo_factor", "linear_window_hi_factor", "min_points_linear", "min_r2", "rate_rel_tol",
              "k_independence_rel_spread", "stable_max_growth_factor", "beam_convergence_rel_tol", "uv_amp_nonlinear"):
        assert d["rules"][k] == r3["rules"][k], k
    for key, w in d["windows_and_T"].items():
        if not w["excluded_a_priori"]:
            assert w["T"] <= w["t_uv_nonlinear"] and w["window_closes_before_uv_nonlinear"], key
    assert d["round3"]["outcome"] == "A" and d["prohibitions"]["round3_untouched"]


def test_round3_untouched():
    assert json.loads((ROUND3 / "cj_criterion_round3.json").read_text(encoding="utf-8"))["outcome"] == "A"


def test_analyzer_reads_rules_from_json_and_fails_closed(tmp_path):
    body = (REPO / "scripts" / "run_cj_round3b.py").read_text(encoding="utf-8").split("def cmd_analyze(")[1]
    assert 'R["rate_rel_tol"]' in body and 'R["min_r2"]' in body and 'R["uv_amp_nonlinear"]' in body
    assert re.search(r"r2\s*>=\s*0\.98", body) is None
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)"
            % (str(REPO), str(REPO / "scripts" / "run_cj_round3b.py"), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


@pytest.mark.skipif(not (OUT / "runs").exists(), reason="sin corridas de la ronda 3b")
def test_runs_cite_frozen_sha_lattice_and_filter():
    sha = _sha(PREREG)
    for p in (OUT / "runs").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["preregistration_sha256"] == sha and d["lattice_exact"] is True and d["k_cut_frac"] == 0.5, p.name
