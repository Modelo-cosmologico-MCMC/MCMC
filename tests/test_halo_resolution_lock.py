"""Candado de la preinscripción del frente 5 (b) (serie de resolución del
campo de Cronos): el .md cita el sha256 del .json vigente; el analizador
lee tolerancias del JSON; falla cerrado sin preinscripción; las corridas
presentes citan el sha congelado."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-21_halo_resolution"
PREREG = OUT / "preregistration.json"
SCRIPT = REPO / "scripts" / "run_halo_resolution.py"

pytestmark = pytest.mark.skipif(not PREREG.exists(), reason="preinscripción aún no congelada")


def test_md_cites_current_json_sha_and_rules_pinned():
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert sha in (OUT / "preregistration.md").read_text(encoding="utf-8")
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert doc["rules"]["tol_conv"] == 0.15 and doc["gates"]["tol_E_newton"] == 5e-3
    assert doc["integrator"]["field_follow_particles"] is True and doc["integrator"]["field_tau_avg_myr"] == 0.0
    assert set(doc["arms"]) == {"a", "bAS_k64", "bAS_k128", "bAS_k256", "b005_k64", "b005_k128", "b005_k256"}


def test_analyzer_reads_tolerances_from_json():
    body = SCRIPT.read_text(encoding="utf-8").split("def cmd_analyze(")[1]
    assert 'R["tol_conv"]' in body and 'G["tol_E_self"]' in body and 'G["tol_E_newton"]' in body
    assert re.search(r"<=\s*0\.15", body) is None


def test_runs_if_present_cite_frozen_sha():
    runs = sorted((OUT / "runs").glob("*.json")) if (OUT / "runs").exists() else []
    if not runs:
        pytest.skip("sin corridas")
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    for p in runs:
        assert json.loads(p.read_text(encoding="utf-8"))["preregistration_sha256"] == sha, p.name


def test_analyze_fails_closed_without_prereg(tmp_path):
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)" % (str(REPO), str(SCRIPT), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)
