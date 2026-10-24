"""Candidato 3 del cribado de β, «sentido del flujo» (bloque P1 del autor del 23-oct): el integrador de cruces D = 0 sobre
flujos de JUGUETE con cruces analíticos (con y sin escalón, divergencia, C₀ ≤ 0), el control hacia el IR (resultado ya
publicado: sin cruces), la letra con sus puertas y el orden de «la peor», y el candado de la preinscripción (se activa al
congelarla). Antes de congelar no se calcula ningún cruce del flujo hacia el UV."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from core.basal import B_BAR, C0_DEFAULT, M_BAR
from core.beta_candidate3 import (
    LETTER_ORDER,
    cell_letter,
    flow_crossings,
    ratios,
    worst,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-10-23_beta_candidate3"
RULES = {"A_interval": [7.0, 14.0], "C_outside": [3.0, 30.0], "truncation_max_rel": 0.10, "convergence_rel": 1e-6}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _linear(_s, y):
    return np.array([1.0, 0.0, 0.0])                     # D = B² − 4C(M₀ + s): cruce en s* = (B² − 4CM₀)/(4C)


def test_toy_flow_crossings_with_and_without_the_step():
    lam0 = (1.0, 4.0, 1.0)                                # D₀ = 12, s* = 3
    with_step = flow_crossings(lam0, rhs=_linear, reset_mass=True, s_max=50.0)
    assert with_step["crossings"] == pytest.approx([3.0, 6.0, 9.0], rel=1e-8) and with_step["stop"] == "3 cruces"
    assert ratios(with_step["crossings"]) == pytest.approx([2.0, 1.5]) and with_step["n_resets"] == 2
    without = flow_crossings(lam0, rhs=_linear, reset_mass=False, s_max=50.0)
    assert without["crossings"] == pytest.approx([3.0], rel=1e-8) and without["stop"] == "s = s_max"
    assert without["criterion_i"]["D_monotone_decreasing_to_first_crossing"]


def test_toy_flow_blowup_and_domain_ends():
    blow = flow_crossings((1.0, 10.0, 1.0), rhs=lambda s, y: np.array([y[0] ** 2, 0.0, 0.0]), blowup=1e3, s_max=50.0)
    assert blow["crossings"] == pytest.approx([1.0 - 1.0 / 25.0], rel=1e-6)          # M = 1/(1 − s) cruza 25 en s = 0.96
    assert blow["stop"].startswith("divergencia") and blow["s_end"] == pytest.approx(1.0 - 1e-3, rel=1e-6)
    dom = flow_crossings((1.0, 10.0, 1.0), rhs=lambda s, y: np.array([0.0, 0.0, -1.0]), s_max=50.0)
    assert dom["stop"].startswith("acoplos fuera del dominio") and dom["s_end"] == pytest.approx(1.0, rel=1e-8)


def test_ir_control_has_no_crossing():
    """Control con el sentido del Maestro (flow_sign = +1, IR): sin cruces D = 0 (resultado ya publicado, #45 y #51)."""
    for d0 in (0.003, 0.01, 0.0554):
        lam0 = (M_BAR ** 2 * d0 ** 2, B_BAR * d0, C0_DEFAULT)
        r = flow_crossings(lam0, flow_sign=+1, s_max=50.0)
        assert r["status"] == "ok" and r["crossings"] == []


def _run(cross, status="ok", stop="3 cruces"):
    return {"status": status, "crossings": cross, "stop": stop}


def test_letter_rules_and_gates():
    geo = [1.0, 10.0, 100.0]
    assert cell_letter(_run(geo), _run(geo), _run([1.0, 10.5, 110.0]), RULES)["letter"] == "A"
    assert cell_letter(_run([1.0, 5.0, 100.0]), _run([1.0, 5.0, 100.0]), _run([1.0, 5.0, 100.0]), RULES)["letter"] == "B"
    assert cell_letter(_run([1.0, 2.0, 3.0]), _run([1.0, 2.0, 3.0]), _run([1.0, 2.0, 3.0]), RULES)["letter"] == "C"
    assert cell_letter(_run([1.0, 40.0, 400.0]), _run([1.0, 40.0, 400.0]), _run([1.0, 40.0, 400.0]), RULES)["letter"] == "C"
    two = _run([1.0, 10.0], stop="s = s_max")
    assert cell_letter(two, two, two, RULES)["letter"] == "C"
    # puertas: truncamiento > 10 %, cuártico sin tres cruces, cúbico C que el cuártico no confirma, convergencia, integrador
    assert cell_letter(_run(geo), _run(geo), _run([1.0, 12.0, 120.0]), RULES)["letter"] == "INDETERMINADO"
    assert cell_letter(_run(geo), _run(geo), _run([1.0, 10.0], stop="s = s_max"), RULES)["letter"] == "INDETERMINADO"
    assert cell_letter(two, two, _run(geo), RULES)["letter"] == "INDETERMINADO"
    assert cell_letter(_run(geo), _run([1.0, 10.0, 100.1]), _run(geo), RULES)["letter"] == "INDETERMINADO"
    assert cell_letter(_run(geo, status="solver_failed", stop="x"), _run(geo), _run(geo), RULES)["letter"] == "INDETERMINADO"
    assert LETTER_ORDER == ("A", "B", "INDETERMINADO", "C")
    assert worst(["A", "B"]) == "B" and worst(["A", "INDETERMINADO"]) == "INDETERMINADO" and worst(["INDETERMINADO", "C"]) == "C"


def test_quartic_variant_starts_from_the_cubic_point():
    lam0 = (1e-4, 3e-2, 1.0)
    r = flow_crossings(lam0, flow_sign=+1, quartic=True, s_max=1e-3)
    assert len(r["lam_end"]) == 4 and r["status"] == "ok" and math.isfinite(r["lam_end"][3])


@pytest.mark.skipif(not (OUT / "preregistration.json").exists(), reason="preinscripción del candidato 3 aún no congelada")
def test_frozen_prereg_lock():
    pre = json.loads((OUT / "preregistration.json").read_text(encoding="utf-8"))
    assert _sha(OUT / "preregistration.json") in (OUT / "preregistration.md").read_text(encoding="utf-8")
    r = pre["rules"]
    assert r["A_interval"] == [7.0, 14.0] and r["C_outside"] == [3.0, 30.0] and r["truncation_max_rel"] == 0.10
    assert r["s_max"] == 50.0 and pre["declared"]["delta0_grid"] == [0.003, 0.01, 0.0554]
    assert pre["declared"]["flow_sign"] == -1 and pre["generator_commit_must_precede_freeze"]
    # la preinscripción del frente 2 no se toca
    assert (REPO / "results" / "2026-08-19_front2_fp_beta" / "preregistration.json").exists()
    spec = importlib.util.spec_from_file_location("c3", REPO / "scripts" / "run_beta_candidate3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.RULES == r
