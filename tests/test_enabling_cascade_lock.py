"""Candado del cribado de la cascada de habilitación (R6), por rondas.

Vigila, en cada ronda congelada: (1) que el .md cita el sha256 del .json vigente (tocar una regla tras ver los números
rompe aquí); (2) que el acta de las decisiones del autor es la que se congeló; (3) que lo declarado coincide con el módulo
y con el canon; (4) que el analizador toma las tolerancias del JSON y falla cerrado sin preinscripción; (5) que las
corridas citan la preinscripción vigente y que las letras publicadas son las que da la regla sobre esas corridas. Y, para
la ronda 2: que sus reglas y sus brazos son los de la ronda 1 y que solo cambia la representación de las amplitudes.
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
ROUND_DIRS = {1: "2026-11-13_enabling_cascade", 2: "2026-11-13_enabling_cascade_round2"}
SCRIPT = REPO / "scripts" / "run_enabling_cascade.py"
DECISIONS = REPO / "docs" / "decisiones_autor_2026-11-13.md"


def _out(n: int) -> Path:
    return REPO / "results" / ROUND_DIRS[n]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(params=sorted(ROUND_DIRS))
def rnd(request):
    if not (_out(request.param) / "preregistration.json").exists():
        pytest.skip(f"ronda {request.param}: preinscripción aún no congelada")
    return request.param


def _doc(n: int) -> dict:
    return json.loads((_out(n) / "preregistration.json").read_text(encoding="utf-8"))


def _module(n: int = 1):
    spec = importlib.util.spec_from_file_location("run_enabling_cascade", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.select_round(n)
    return m


def test_md_cites_current_json_sha(rnd):
    md = (_out(rnd) / "preregistration.md").read_text(encoding="utf-8")
    assert _sha(_out(rnd) / "preregistration.json") in md, "el .md cita otro sha256: la preinscripción se tocó tras congelarla"


def test_author_decisions_are_the_frozen_ones(rnd):
    doc = _doc(rnd)
    assert doc["author_decisions"]["file"] == "docs/decisiones_autor_2026-11-13.md"
    assert _sha(DECISIONS) == doc["author_decisions"]["sha256"], "el acta de las decisiones del autor cambió tras congelar"
    text = DECISIONS.read_text(encoding="utf-8")
    assert "Latente primero" in text and "parámetros de orden asociados a modos de estructuración emergente" in text
    assert "progresión geométrica de razón" in text and "no constituye una demostración de emergencia dimensional" in text


def test_declared_matches_module_and_canon(rnd):
    doc = _doc(rnd)
    D, R = doc["declared"], doc["rules"]
    cfg = CascadeConfig()
    assert D["delta0_grid"] == [0.003, 0.01, 0.0554]
    assert D["base"]["lam_scale"] == C.LAMBDA_DECADE == cfg.lam_scale
    assert D["base"]["gate_theta"] == cfg.gate_theta and D["base"]["gate_Delta"] == cfg.gate_Delta
    assert tuple(D["base"]["seeds"]) == cfg.seeds and D["base"]["dS_step"] == cfg.dS_step
    assert D["base"]["noise_floor"] == 0.0 and D["base"]["tension_reference"] == "measured" and D["base"]["gated"] is True
    # la ronda 1 integra las amplitudes tal cual (el valor por defecto del módulo); la ronda 2, su logaritmo
    assert D["base"].get("amplitude_representation", cfg.amplitude_representation) == ("linear" if rnd == 1 else "log")
    assert cfg.amplitude_representation == "linear"
    assert D["letter_arms"] == ["base"] and D["arms"]["base"] == {"model": {}, "clock": {}}
    assert str(list(scale_law_shares(C.LAMBDA_DECADE))) in D["scale_law"]
    assert R["A_interval"] == [7.0, 14.0] and R["n_collapses_required"] == 3 and R["no_letter_on_lambda"] is True
    assert R["gates"]["monotonia_max_dV_over_T_total"] == 1e-12 and R["gates"]["convergence_rel"] == 1e-4
    assert R["controls"]["seed_rel"] == 1e-3 and R["controls"]["no_seed_rel"] == 1e-6
    assert doc["prohibitions"] == {"no_threshold_tuning": True, "no_runs_before_freeze": True, "default_clock_untouched": True}


def test_round2_changes_only_the_amplitude_representation():
    if not (_out(2) / "preregistration.json").exists():
        pytest.skip("ronda 2: preinscripción aún no congelada")
    d1, d2 = _doc(1), _doc(2)
    assert d2["rules"] == d1["rules"], "la ronda 2 debe llevar las reglas de la ronda 1, sin tocar"
    assert d2["declared"]["arms"] == d1["declared"]["arms"] and d2["declared"]["delta0_grid"] == d1["declared"]["delta0_grid"]
    assert d2["declared"]["base_clock"] == d1["declared"]["base_clock"]
    b1, b2 = dict(d1["declared"]["base"]), dict(d2["declared"]["base"])
    assert b2.pop("amplitude_representation") == "log" and b2 == b1
    od = d2["round2_opening_decision"]
    assert od["file"] == "docs/decision_autor_2026-11-13_r6_ronda2.md" and od["sha256"] == _sha(REPO / od["file"])
    literal = " ".join(od["literal"])
    assert "La única modificación admitida es sustituir la integración directa" in literal
    assert "esta segunda ronda no es ciega" in literal and "no sustituirá el brazo principal" in literal
    assert set(d2["author_conditions"]["verified_before_run"]) == {"equivalencia de las ecuaciones",
                                                                   "conservación de los estados físicamente nulos",
                                                                   "convergencia numérica"}
    assert d2["declared"]["letter_arms"] == ["base"] and "suelo_1e-18" in d2["declared"]["control_arms"]
    pr = d2["previous_round"]
    assert pr["preregistration_sha256"] == _sha(_out(1) / "preregistration.json")
    assert pr["analysis_sha256"] == _sha(_out(1) / "analysis.json"), "el análisis de la ronda 1 cambió tras congelar la ronda 2"
    an1 = json.loads((_out(1) / "analysis.json").read_text(encoding="utf-8"))
    assert pr["letter_global"] == an1["letter_global"] == "INDETERMINADO"
    known = d2["known_before_freeze"]["round1_base_cells"]
    for c in an1["cells"]:
        assert known[str(c["delta0"])]["letter"] == c["letter"] and known[str(c["delta0"])]["S_triggers"] == c.get("S_triggers", [])
    assert "NO es ciega" in d2["known_before_freeze"]["note"]


def test_analyzer_reads_rules_from_json_not_constants():
    src = SCRIPT.read_text(encoding="utf-8")
    body = src.split("def _cell(")[1].split("def cmd_analyze(")[0]
    for key in ('G["monotonia_max_dV_over_T_total"]', 'G["identity_max_abs_diff"]', 'G["final_V_tol"]', 'G["convergence_rel"]',
                'K["no_seed_rel"]', 'K["seed_rel"]'):
        assert key in body, f"el analizador no lee {key} de la preinscripción"
    assert re.search(r"\b1e-\d+\b", body) is None, "hay una tolerancia escrita a mano en el analizador"
    assert "cascade_letter(base[\"S_triggers\"], R)" in body


def test_prereg_refuses_to_overwrite(rnd):
    m = _module(rnd)
    with pytest.raises(SystemExit, match="FALLO CERRADO"):
        m.prereg(None)


def test_analyze_fails_closed_without_prereg(tmp_path):
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)"
            % (str(REPO), str(SCRIPT), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


def test_runs_if_present_cite_frozen_sha(rnd):
    runs = sorted((_out(rnd) / "runs").glob("*.json")) if (_out(rnd) / "runs").exists() else []
    if not runs:
        pytest.skip("sin corridas")
    sha = _sha(_out(rnd) / "preregistration.json")
    for p in runs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec["preregistration_sha256"] == sha, p.name
        assert rec.get("amplitude_representation", "linear") == ("linear" if rnd == 1 else "log"), p.name


def test_published_letters_are_what_the_rule_gives_on_the_runs(rnd):
    analysis = _out(rnd) / "analysis.json"
    if not analysis.exists():
        pytest.skip("sin análisis")
    doc, an = _doc(rnd), json.loads(analysis.read_text(encoding="utf-8"))
    assert an["prereg_sha256"] == _sha(_out(rnd) / "preregistration.json")
    order = ["A", "B", "INDETERMINADO", "C"]
    letters = []
    for cell in an["cells"]:
        base = json.loads((_out(rnd) / "runs" / f"d{cell['delta0']}_base.json").read_text(encoding="utf-8"))
        assert cell["S_triggers"] == base["S_triggers"]
        rule = cascade_letter(base["S_triggers"], doc["rules"])["letter"]
        failed = [k for k, v in {**cell["gates"], **cell["controls"]}.items() if not v]
        assert cell["letter"] == ("INDETERMINADO" if failed else rule)
        letters.append(cell["letter"])
    assert an["letter_global"] == max(letters, key=order.index)


def test_round1_outcome_is_recorded_as_it_came_out():
    an = json.loads((_out(1) / "analysis.json").read_text(encoding="utf-8")) if (_out(1) / "analysis.json").exists() else None
    if an is None:
        pytest.skip("sin análisis de la ronda 1")
    assert an["letter_global"] == "INDETERMINADO"
    assert {str(c["delta0"]): c["letter"] for c in an["cells"]} == {"0.003": "A", "0.01": "INDETERMINADO", "0.0554": "INDETERMINADO"}
    assert (_out(1) / "lectura_a_posteriori.md").exists()
