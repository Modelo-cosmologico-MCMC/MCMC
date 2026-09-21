"""Candado del test del criterio de Cronos–Jeans: preinscripción
congelada (sistema, instrumento, predicción cinética, reglas,
desenlaces), ejecutor con fallo cerrado y artefacto (si existe)
coherente con la regla congelada."""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-21_cj_criterion_test"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "cj_criterion_test.json"


def _prereg():
    assert PREREG.exists(), "preinscripción del test del criterio ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_fields_pinned():
    d = _prereg()
    assert d["q_grid"] == [0.5, 0.8, 1.2, 2.0] and d["modes"] == [4, 8, 16]
    ins = d["instrument"]
    assert ins["N"] == 2_000_000 and ins["ng"] == 512 and ins["n_beams"] == 1024 and ins["seed_amp"] == 2e-5
    assert ins["sample_every"] == 2 and ins["dt"] == 1e-3 and ins["beam_convergence_run"] == {"q": 1.2, "mode": 8, "n_beams_alt": 512}
    r = d["rules"]
    assert r["rate_rel_tol"] == 0.25 and r["k_independence_rel_spread"] == 0.25 and r["beam_convergence_rel_tol"] == 0.10
    assert r["min_points_linear"] == 8 and r["min_r2"] == 0.98 and r["stable_max_growth_factor"] == 3.0
    assert r["linear_window_lo_factor"] == 3.0 and r["linear_window_hi_factor"] == 30.0
    assert d["delta_k_initial"] == pytest.approx(1e-5)
    pred = {p["q"]: p for p in d["prediction_kinetic"]}
    assert pred[1.2]["gamma_over_k_kinetic"] == pytest.approx(0.1492, abs=1e-3)
    assert pred[2.0]["gamma_over_k_kinetic"] == pytest.approx(0.6120, abs=1e-3)
    assert pred[0.8]["gamma_over_k_kinetic"] == 0.0
    w = d["windows_and_T"]["q1.2_n8"]
    assert w["window"] == [pytest.approx(3e-5), pytest.approx(3e-4)] and 0.98 < w["W_k"] < 1.0
    assert set(d["outcomes"]) >= {"C_criterion_or_instrument_broken", "B_threshold_ok_rate_off", "A_kinetic_reproduced", "INDETERMINADO"}
    assert "nunca se ajustan umbrales" in d["outcomes"]["order"]
    assert d["prohibitions"]["no_threshold_tuning"] and "tolerancia" in d["development_declaration"]["what_they_fixed"]


def test_runner_fails_closed_and_uses_prereg():
    src = (REPO / "scripts" / "run_cj_test.py").read_text(encoding="utf-8")
    assert "FALLO CERRADO" in src and src.index("load_prereg()") < src.index("run_sheets(")


def test_artifact_if_present_matches_rule():
    if not ARTIFACT.exists():
        pytest.skip("test del criterio aún no analizado")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert doc["outcome"] in {"A", "B", "C", "INDETERMINADO"}
    if doc["missing_runs"]:
        assert doc["outcome"] == "INDETERMINADO"
        return
    rows = doc["table"]
    stable = [r for r in rows if r["q"] < 1.0]
    unstable = [r for r in rows if r["q"] > 1.0]
    if any(not r["stable"] for r in stable):
        assert doc["outcome"] == "C"
        return
    if doc["beam_convergence"] is None or not doc["beam_convergence"]["pass"]:
        assert doc["outcome"] == "INDETERMINADO"
        return
    if any(not r["linear_resolved"] for r in unstable):
        assert doc["outcome"] == "INDETERMINADO"
        return
    rates_ok = all(r["rate_within_tol"] for r in unstable)
    kind_ok = all(v["pass"] for v in doc["k_independence"].values())
    assert doc["outcome"] == ("A" if rates_ok and kind_ok else "B")
