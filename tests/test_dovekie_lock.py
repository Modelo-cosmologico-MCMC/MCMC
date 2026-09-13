"""Candados del PR #15: la barrera de datos reales Dovekie y la
preinscripción congelada de la validación por mocks.

Patrón 5E: estos tests vigilan el PROTOCOLO — fallan si la barrera se
relaja, si la preinscripción cambia tras el congelado o si el
verificador independiente deja de ser independiente.
"""

import hashlib
import json
from pathlib import Path

import pytest

import cosmology.dovekie_sn as dsn

REPO = Path(__file__).resolve().parent.parent
PREREG = REPO / "results" / "2026-09-12_dovekie_mocks" / "preregistration.json"
MOCKVAL = REPO / "results" / "2026-09-12_dovekie_mocks" / "mock_validation.json"


def test_prereg_exists_and_frozen_fields_pinned():
    """CANDADO HISTÓRICO: los enteros y semillas de la preinscripción
    quedan fijados aquí con sus valores congelados — una regeneración
    que los cambiara rompe la suite."""
    assert PREREG.exists(), "preinscripción ausente (PR #15)"
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    fz = doc["frozen_numbers"]
    assert fz["n_sn"] == 1820
    assert fz["n_mocks"] == 25
    assert fz["mock_seed"] == 20260912
    assert fz["cov_kind_production"] == "STAT+SYS"
    assert fz["coverage_k68_range"] == [12, 21]
    assert fz["coverage_k95_range"] == [21, 25]
    assert fz["false_preference_k_max"] == 7
    assert fz["priors"]["eps_gauss"] == [0.012, 0.05]
    assert fz["priors"]["z_trans"] == [1.0, 20.0]
    assert doc["injected_cosmology"]["our_parametrization"][
        "Omega_m"] == 0.315
    assert "no_adjustment_commitment" in doc["outcomes_enumerated"]
    assert "mandatory_wording_on_pass" in doc["gates"][
        "3_coverage_binomial_exact"]


def test_barrier_fails_closed_without_validation(monkeypatch, tmp_path):
    """Sin mock_validation.json el HD real es inaccesible (falla
    cerrado), exista o no la preinscripción."""
    monkeypatch.setattr(dsn, "MOCK_VALIDATION",
                        tmp_path / "no_existe.json")
    with pytest.raises(RuntimeError, match="MOCK_VALIDATION_REQUIRED"):
        dsn.require_mock_validation_pass()


def test_barrier_fails_closed_on_fail_status(monkeypatch, tmp_path):
    """Un mock_validation con status FAIL no abre la barrera."""
    bad = tmp_path / "mock_validation.json"
    bad.write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
    monkeypatch.setattr(dsn, "MOCK_VALIDATION", bad)
    with pytest.raises(RuntimeError, match="MOCK_VALIDATION_FAIL"):
        dsn.require_mock_validation_pass()


def test_barrier_fails_closed_on_prereg_mismatch(monkeypatch, tmp_path):
    """Un PASS que cite un sha256 distinto del de la preinscripción en
    el árbol (preinscripción alterada tras el PASS) falla cerrado."""
    ok = tmp_path / "mock_validation.json"
    ok.write_text(json.dumps({"status": "PASS",
                              "preregistration_sha256": "0" * 64}),
                  encoding="utf-8")
    prereg = tmp_path / "preregistration.json"
    prereg.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(dsn, "MOCK_VALIDATION", ok)
    monkeypatch.setattr(dsn, "PREREG", prereg)
    with pytest.raises(RuntimeError, match="PREREGISTRATION_MISMATCH"):
        dsn.require_mock_validation_pass()


def test_committed_validation_if_present_is_consistent():
    """Si el veredicto ejecutable está commiteado, debe citar EXACTA
    la preinscripción del árbol y declarar sus puertas."""
    if not MOCKVAL.exists():
        pytest.skip("mock_validation.json aún no ejecutado")
    doc = json.loads(MOCKVAL.read_text(encoding="utf-8"))
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert doc["preregistration_sha256"] == sha
    assert set(doc["gates"]) == {"1a_formula_equivalence",
                                 "1b_distance_equivalence",
                                 "2_parameter_recovery", "3_coverage",
                                 "4_no_false_preference"}
    if doc["status"] == "PASS":
        assert all(g["pass"] for g in doc["gates"].values())


def test_independent_b_is_isolated():
    """La implementación B no importa la ruta de producción: ni
    cosmology.dovekie_sn ni cosmology.background (patrón #14)."""
    src = (REPO / "validation" / "dovekie_independent.py").read_text(
        encoding="utf-8")
    assert "cosmology.dovekie_sn" not in src
    assert "cosmology.background" not in src
    assert "from cosmology" not in src


def test_mock_machinery_never_touches_real_mu():
    """El generador de mocks y el runner de puertas no consumen la
    columna MU real: prohibido load_dovekie_hd en ambos, y el diseño
    (load_dovekie_design) no la devuelve."""
    for rel in ("validation/dovekie_mocks.py",
                "scripts/run_dovekie_mocks.py",
                "validation/dovekie_independent.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "load_dovekie_hd" not in src, rel
    design = dsn.load_dovekie_design()
    assert "MU" not in design and "MUERR" not in design
    assert design["n_sn"] == 1820


def test_real_loader_is_guarded():
    """load_dovekie_hd pasa por la barrera (verificación textual del
    orden: la guardia se invoca antes de tocar el fichero)."""
    src = (REPO / "cosmology" / "dovekie_sn.py").read_text(
        encoding="utf-8")
    body = src.split("def load_dovekie_hd")[1]
    assert body.index("require_mock_validation_pass()") \
        < body.index("_parse_hd")
