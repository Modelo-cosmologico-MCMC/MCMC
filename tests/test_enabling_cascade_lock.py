"""Candado del cribado de la cascada de habilitación (R6).

Vigila: (1) que el .md cita el sha256 del .json vigente (tocar una regla tras ver los números rompe aquí); (2) que el acta
de las decisiones del autor es la que se congeló; (3) que lo declarado coincide con el módulo y con el canon; (4) que el
analizador toma las tolerancias del JSON y falla cerrado sin preinscripción; (5) que las corridas citan la preinscripción
vigente y que las letras publicadas son las que da la regla sobre esas corridas.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.enabling_cascade import CascadeConfig, cascade_letter, scale_law_shares
from mcmc_ontology import constants as C

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-11-13_enabling_cascade"
PREREG = OUT / "preregistration.json"
SCRIPT = REPO / "scripts" / "run_enabling_cascade.py"
DECISIONS = REPO / "docs" / "decisiones_autor_2026-11-13.md"

pytestmark = pytest.mark.skipif(not PREREG.exists(), reason="preinscripción aún no congelada")


def _doc():
    return json.loads(PREREG.read_text(encoding="utf-8"))


def _module():
    spec = importlib.util.spec_from_file_location("run_enabling_cascade", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_md_cites_current_json_sha():
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    md = (OUT / "preregistration.md").read_text(encoding="utf-8")
    assert sha in md, "el .md cita otro sha256: la preinscripción se tocó tras congelarla"


def test_author_decisions_are_the_frozen_ones():
    doc = _doc()
    assert doc["author_decisions"]["file"] == "docs/decisiones_autor_2026-11-13.md"
    assert hashlib.sha256(DECISIONS.read_bytes()).hexdigest() == doc["author_decisions"]["sha256"], \
        "el acta de las decisiones del autor cambió tras congelar"
    text = DECISIONS.read_text(encoding="utf-8")
    assert "Latente primero" in text and "parámetros de orden asociados a modos de estructuración emergente" in text
    assert "progresión geométrica de razón" in text and "no constituye una demostración de emergencia dimensional" in text


def test_declared_matches_module_and_canon():
    doc = _doc()
    D, R = doc["declared"], doc["rules"]
    cfg = CascadeConfig()
    assert D["delta0_grid"] == [0.003, 0.01, 0.0554]
    assert D["base"]["lam_scale"] == C.LAMBDA_DECADE == cfg.lam_scale
    assert D["base"]["gate_theta"] == cfg.gate_theta and D["base"]["gate_Delta"] == cfg.gate_Delta
    assert tuple(D["base"]["seeds"]) == cfg.seeds and D["base"]["dS_step"] == cfg.dS_step
    assert D["base"]["noise_floor"] == 0.0 and D["base"]["tension_reference"] == "measured" and D["base"]["gated"] is True
    assert D["letter_arms"] == ["base"] and D["arms"]["base"] == {"model": {}, "clock": {}}
    assert str(list(scale_law_shares(C.LAMBDA_DECADE))) in D["scale_law"]
    assert R["A_interval"] == [7.0, 14.0] and R["n_collapses_required"] == 3 and R["no_letter_on_lambda"] is True
    assert R["gates"]["monotonia_max_dV_over_T_total"] == 1e-12 and R["gates"]["convergence_rel"] == 1e-4
    assert R["controls"]["seed_rel"] == 1e-3 and R["controls"]["no_seed_rel"] == 1e-6
    assert doc["prohibitions"] == {"no_threshold_tuning": True, "no_runs_before_freeze": True, "default_clock_untouched": True}


def test_analyzer_reads_rules_from_json_not_constants():
    src = SCRIPT.read_text(encoding="utf-8")
    body = src.split("def _cell(")[1].split("def cmd_analyze(")[0]
    for key in ('G["monotonia_max_dV_over_T_total"]', 'G["identity_max_abs_diff"]', 'G["final_V_tol"]', 'G["convergence_rel"]',
                'K["no_seed_rel"]', 'K["seed_rel"]'):
        assert key in body, f"el analizador no lee {key} de la preinscripción"
    assert re.search(r"\b1e-\d+\b", body) is None, "hay una tolerancia escrita a mano en el analizador"
    assert "cascade_letter(base[\"S_triggers\"], R)" in body


def test_prereg_refuses_to_overwrite():
    m = _module()
    with pytest.raises(SystemExit, match="FALLO CERRADO"):
        m.prereg(None)


def test_analyze_fails_closed_without_prereg(tmp_path):
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)"
            % (str(REPO), str(SCRIPT), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


def test_runs_if_present_cite_frozen_sha():
    runs = sorted((OUT / "runs").glob("*.json")) if (OUT / "runs").exists() else []
    if not runs:
        pytest.skip("sin corridas")
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    for p in runs:
        assert json.loads(p.read_text(encoding="utf-8"))["preregistration_sha256"] == sha, p.name


def test_published_letters_are_what_the_rule_gives_on_the_runs():
    analysis = OUT / "analysis.json"
    if not analysis.exists():
        pytest.skip("sin análisis")
    doc, an = _doc(), json.loads(analysis.read_text(encoding="utf-8"))
    assert an["prereg_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    order = ["A", "B", "INDETERMINADO", "C"]
    letters = []
    for cell in an["cells"]:
        base = json.loads((OUT / "runs" / f"d{cell['delta0']}_base.json").read_text(encoding="utf-8"))
        assert cell["S_triggers"] == base["S_triggers"]
        rule = cascade_letter(base["S_triggers"], doc["rules"])["letter"]
        failed = [k for k, v in {**cell["gates"], **cell["controls"]}.items() if not v]
        assert cell["letter"] == ("INDETERMINADO" if failed else rule)
        letters.append(cell["letter"])
    assert an["letter_global"] == max(letters, key=order.index)
