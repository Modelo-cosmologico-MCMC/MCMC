"""Candado de la ronda 2 del test del criterio de Cronos–Jeans:
preinscripción congelada (instrumento con retículo exacto y siembra del
modo propio, modos 4/8/16/32, reglas, desenlaces), analizador con fallo
cerrado que lee las reglas del JSON, corridas que citan el sha, y la
ronda 1 intacta."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-22_cj_criterion_round2"
PREREG = OUT / "preregistration.json"
ROUND1 = REPO / "results" / "2026-09-21_cj_criterion_test"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.skipif(not PREREG.exists(), reason="preinscripción de la ronda 2 no congelada")
def test_round2_frozen_fields_pinned():
    d = json.loads(PREREG.read_text(encoding="utf-8"))
    assert _sha(PREREG) in (OUT / "preregistration.md").read_text(encoding="utf-8")
    assert d["q_grid"] == [0.8, 1.2, 2.0] and d["modes"] == [4, 8, 16, 32]
    ins = d["instrument"]
    assert ins["N"] == ins["n_beams"] * ins["ng"] * ins["cells_per_beam"] and ins["N"] % (ins["n_beams"] * ins["ng"]) == 0
    assert ins["ng"] == 512 and ins["n_beams"] == 1024 and ins["cells_per_beam"] == 4 and ins["seed_eigenmode"] is True
    assert ins["seed_amp_by_mode"] == {"4": 2e-5, "8": 2e-5, "16": 2e-6, "32": 2e-6}
    bc = ins["beam_convergence_run"]
    assert bc == {"q": 1.2, "mode": 8, "n_beams_alt": 512, "cells_per_beam_alt": 8}
    r = d["rules"]
    assert r["rate_rel_tol"] == 0.25 and r["k_independence_rel_spread"] == 0.25 and r["beam_convergence_rel_tol"] == 0.02
    assert r["min_points_linear"] == 8 and r["min_r2"] == 0.98 and r["stable_max_growth_factor"] == 3.0
    assert set(d["outcomes"]) >= {"C_criterion_fails", "B_finite_size", "A_kinetic_reproduced", "INDETERMINADO"}
    w = d["windows_and_T"]["q2.0_n32"]
    assert w["window"] == [pytest.approx(3e-6), pytest.approx(3e-5)] and 0.9 < w["W_k"] < 1.0
    assert d["prohibitions"]["round1_untouched"] is True


def test_round1_untouched():
    """La ronda 1 conserva su preinscripción y su INDETERMINADO."""
    assert (ROUND1 / "preregistration.json").exists()
    art = json.loads((ROUND1 / "cj_criterion_test.json").read_text(encoding="utf-8"))
    assert art["outcome"] == "INDETERMINADO"


def test_analyzer_reads_rules_from_json_and_fails_closed(tmp_path):
    body = (REPO / "scripts" / "run_cj_round2.py").read_text(encoding="utf-8").split("def cmd_analyze(")[1]
    assert 'R["rate_rel_tol"]' in body and 'R["min_r2"]' in body and 'R["beam_convergence_rel_tol"]' in body
    assert re.search(r"r2\s*>=\s*0\.98", body) is None
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)"
            % (str(REPO), str(REPO / "scripts" / "run_cj_round2.py"), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


@pytest.mark.skipif(not (OUT / "runs").exists(), reason="sin corridas de la ronda 2")
def test_runs_cite_frozen_sha_and_exact_lattice():
    sha = _sha(PREREG)
    for p in (OUT / "runs").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["preregistration_sha256"] == sha, p.name
        assert d["lattice_exact"] is True, p.name


def test_lattice_rule_and_eigen_seed_are_instrument_options():
    """La regla del retículo y la siembra del modo propio existen en el
    instrumento y se publican en la corrida (piloto mínimo)."""
    from cronos.cronos_jeans_1d import run_sheets
    r = run_sheets(2.0, N=16 * 32 * 2, ng=32, T=0.01, dt=1e-3, seed=1, nmodes=4, sample_every=1, quiet_start=True, n_beams=16,
                   seed_mode=2, seed_amp=2e-5, seed_eigen_gamma_over_k=0.6, q_seed=2.0, full_modes_every=5)
    assert r["lattice_exact"] is True and r["eigen_dispersion_residual"] is not None
    assert r["samples"][1]["delta_k"][0] is None and r["samples"][1]["delta_k"][1] is not None
    r2 = run_sheets(2.0, N=16 * 33, ng=32, T=0.002, dt=1e-3, seed=1, nmodes=2, quiet_start=True, n_beams=16, seed_mode=1, seed_amp=2e-5)
    assert r2["lattice_exact"] is False
