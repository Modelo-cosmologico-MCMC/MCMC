"""Candado de la preinscripción del |J| requerido (circulación J∇C).

(1) el .md cita el sha256 del .json vigente; (2) el analizador lee las
reglas del JSON (tolerancias y ventana) y no de constantes; (3) falla
cerrado sin preinscripción; (4) las corridas, si existen, citan el sha
congelado; (5) la forma declarada del reloj coincide con la congelada.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-21_j_circulation"
PREREG = OUT / "preregistration.json"
SCRIPT = REPO / "scripts" / "run_j_circulation.py"

pytestmark = pytest.mark.skipif(not PREREG.exists(), reason="preinscripción aún no congelada")


def test_md_cites_current_json_sha():
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert sha in (OUT / "preregistration.md").read_text(encoding="utf-8")


def test_frozen_form_matches_clock_and_rules_pinned():
    from core.s_clock import DECLARED_FORMS
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert doc["declared"]["form"] == DECLARED_FORMS["J_circulation"]
    assert doc["declared"]["window"] == [0.95, 1.05] and doc["declared"]["C_primary"] == "V"
    assert doc["rules"]["scale_invariance"]["tol"] == 0.10
    assert doc["rules"]["gates"]["identity_interior_C_V"] == 1e-6


def test_analyzer_reads_rules_from_json():
    body = SCRIPT.read_text(encoding="utf-8").split("def analyze()")[1]
    assert 'R["gates"]["identity_interior_C_V"]' in body and 'R["scale_invariance"]["tol"]' in body
    assert 'D["window"]' in body
    assert re.search(r"w_lo\s*=\s*0\.9", body) is None


def test_runs_if_present_cite_frozen_sha():
    runs = OUT / "runs.json"
    if not runs.exists():
        pytest.skip("sin corridas")
    assert json.loads(runs.read_text(encoding="utf-8"))["preregistration_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()


def test_analyze_fails_closed_without_prereg(tmp_path):
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.analyze()" % (str(REPO), str(SCRIPT), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)
