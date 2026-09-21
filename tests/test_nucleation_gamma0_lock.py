"""Candado de la ronda 2 de la nucleación (Γ₀ condicional a n_dim).

Vigila: (1) que la preinscripción congelada existe y que el .md cita el
sha256 del .json vigente (cambiar una tolerancia tras ver los números
rompe aquí); (2) que el analizador falla cerrado sin preinscripción;
(3) que las reglas que aplica el analizador salen del JSON, no de
constantes del script; (4) que el conjunto n_dim declarado del módulo
coincide con el congelado.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-21_nucleation_gamma0"
PREREG = OUT / "preregistration.json"
SCRIPT = REPO / "scripts" / "run_nucleation_gamma0.py"

pytestmark = pytest.mark.skipif(not PREREG.exists(), reason="preinscripción aún no congelada")


def test_md_cites_current_json_sha():
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    md = (OUT / "preregistration.md").read_text(encoding="utf-8")
    assert sha in md, "el .md cita otro sha256: la preinscripción se tocó tras congelarla"


def test_declared_n_dim_matches_module():
    from core.nucleation import N_DIM_DECLARED
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert doc["declared"]["n_dim"] == list(N_DIM_DECLARED)
    assert doc["rules"]["A"]["tol_A"] == 0.05 and doc["rules"]["B"]["tol_B"] == 0.25
    assert doc["rules"]["C"]["ratio_threshold"] == 0.5 and doc["rules"]["gates"]["tol_fit"] == 0.10


def test_analyzer_reads_rules_from_json_not_constants():
    src = SCRIPT.read_text(encoding="utf-8")
    body = src.split("def analyze()")[1]
    # ninguna tolerancia numérica escrita a mano en el analizador
    assert re.search(r"tol_A\s*=\s*0\.", body) is None and re.search(r"tol_B\s*=\s*0\.", body) is None
    assert 'R["A"]["tol_A"]' in body and 'R["B"]["tol_B"]' in body and 'R["gates"]["tol_fit"]' in body
    assert 'R["fit_window_n_points"]' in body


def test_runs_if_present_cite_frozen_sha():
    runs = OUT / "runs.json"
    if not runs.exists():
        pytest.skip("sin corridas")
    doc = json.loads(runs.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()


def test_analyze_fails_closed_without_prereg(tmp_path, monkeypatch):
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.analyze()"
            % (str(REPO), str(SCRIPT), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)
