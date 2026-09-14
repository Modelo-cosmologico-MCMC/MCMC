"""Candado del Contraste de los Residuos vía BBN: preinscripción
congelada (regla, brazos, prohibiciones, tabla de respuesta citada por
sha256), dataset registrado con su aviso de procedencia, ejecutor tras la
barrera y artefacto (si existe) consistente con la regla y con el aviso.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cosmology.bbn_g import RESPONSE_TABLE, RULES, classify_outcome, delta_G_model

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-14_bbn_g"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "bbn_g.json"
MANIFEST = REPO / "data" / "manifests" / "bbn_abundances.json"


def _prereg():
    assert PREREG.exists(), "preinscripción BBN-G ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_rules_and_model_pinned():
    doc = _prereg()
    assert doc["outcomes_enumerated"]["rules"] == RULES
    assert doc["model_prediction"]["delta_G_model"] == pytest.approx(delta_G_model(), rel=1e-12)
    assert doc["model_prediction"]["alpha_a"] == 0.0 and doc["model_prediction"]["epsilon_K"] == 0.012
    assert doc["response_model"]["response_table_sha256"] == hashlib.sha256(
        RESPONSE_TABLE.read_bytes()).hexdigest()
    assert doc["outcomes_enumerated"]["order"].startswith("C → B → A")
    assert "Nunca se ajustan umbrales" in doc["outcomes_enumerated"]["rule"]
    assert "SOSPECHA DE ERROR PRIMERO" in doc["outcomes_enumerated"]["B_identification"]
    assert "PROHIBIDO reparametrizar" in doc["outcomes_enumerated"]["C_exclusion"]
    assert "E13" in doc["outcomes_enumerated"]["A_compatible"]


def test_data_declared_with_provenance_caveat():
    doc = _prereg()
    d = doc["data_declared"]
    assert d["dataset"] == "bbn_abundances"
    assert d["principal"] == {"Y_P": "Y_P_empress_xv", "DH": "DH_cooke_2018"}
    assert "TRANSCRITOS" in d["provenance_caveat"]
    assert doc["prohibitions"]["no_Neff_from_empress_as_data"]
    assert doc["prohibitions"]["no_sign_as_signal"]
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert man["state"] == "AVAILABLE"
    assert any("NO verificados" in t for t in man["transformations"])
    assert any("esquema confirmado" in t for t in man["transformations"])
    raw = json.loads((REPO / "data" / "raw" / "bbn_abundances" /
                      "abundances_2026.json").read_text(encoding="utf-8"))
    assert raw["official_bytes_verified"] is False
    assert raw["values"]["Y_P_empress_xv"]["value"] == 0.2402


def test_runner_loads_prereg_before_data():
    src = (REPO / "scripts" / "run_bbn_g.py").read_text(encoding="utf-8")
    assert src.index("load_prereg()") < src.index("load_data()")


def test_artifact_if_present_matches_rule_and_carries_caveat():
    if not ARTIFACT.exists():
        pytest.skip("contraste BBN-G aún no ejecutado")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert doc["data_manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert doc["response_table_sha256"] == hashlib.sha256(RESPONSE_TABLE.read_bytes()).hexdigest()
    a1 = doc["arm1_principal"]
    assert doc["outcome"] == classify_outcome(a1["ci95"], doc["delta_G_model"])
    assert doc["delta_G_model"] == pytest.approx(delta_G_model(), rel=1e-12)
    assert not a1["prior_edge_hit"]
    if not doc["official_bytes_verified"]:
        assert doc["outcome_provisional_pending_provenance"] is True
        assert "TRANSCRITOS" in doc["mandatory_wording"]
    for key in ("arm0_sm", "arm2_degeneracy", "arm3_control_aver", "arm4_single_probe"):
        assert key in doc
