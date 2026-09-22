"""Candados de las preinscripciones del frente 5 (c) «Oort–K_z» y (d)
«perfil σ_los(R) de Sculptor»: el .md cita el sha256 del .json vigente;
los analizadores leen las reglas del JSON y fallan cerrado sin él; las
corridas, si existen, citan el sha congelado; el manifest del dataset
citado no cambió; walker2009 sigue fallando cerrado hasta su ingesta."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
KZ = REPO / "results" / "2026-09-21_oort_kz"
SC = REPO / "results" / "2026-09-21_sculptor_profile"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.skipif(not (KZ / "preregistration.json").exists(), reason="preinscripción Oort–K_z no congelada")
def test_oort_kz_prereg_frozen_and_rules_from_json():
    doc = json.loads((KZ / "preregistration.json").read_text(encoding="utf-8"))
    assert _sha(KZ / "preregistration.json") in (KZ / "preregistration.md").read_text(encoding="utf-8")
    assert doc["rules"]["compatible"]["z_max_le"] == 2.0 and doc["rules"]["excluded"]["z_max_gt"] == 5.0
    assert doc["declared"]["dataset_manifest_sha256"] == _sha(REPO / "data" / "manifests" / "local_kz_bounds.json")
    body = (REPO / "scripts" / "run_oort_kz.py").read_text(encoding="utf-8").split("def analyze()")[1]
    assert 'R["compatible"]["z_max_le"]' in body and 'R["tension"]["z_max_in"]' in body
    assert re.search(r"z\s*<=\s*2\.0", body) is None
    runs = KZ / "runs.json"
    if runs.exists():
        assert json.loads(runs.read_text(encoding="utf-8"))["preregistration_sha256"] == _sha(KZ / "preregistration.json")
    if "erratum" in doc:
        # un erratum copia las reglas verbatim de la v1 (que sigue en el repositorio, con su sha) y el
        # analizador comprueba que los números coinciden con los de la v1
        v1 = KZ / doc["erratum"]["of_file"]
        assert v1.exists() and _sha(v1) == doc["erratum"]["of_sha256"]
        assert json.loads(v1.read_text(encoding="utf-8"))["rules"] == doc["rules"]
        assert 'v1["rules"] != R' in body and "values_identical_to_v1" in body
        res = KZ / "results.json"
        if res.exists():
            chk = json.loads(res.read_text(encoding="utf-8"))["erratum_check"]
            assert chk["rules_identical"] and chk["values_identical_to_v1"] and chk["max_abs_deviation_from_v1"] <= 1e-9


@pytest.mark.skipif(not (SC / "preregistration.json").exists(), reason="preinscripción Sculptor no congelada")
def test_sculptor_prereg_frozen_and_fails_closed_without_data():
    doc = json.loads((SC / "preregistration.json").read_text(encoding="utf-8"))
    assert _sha(SC / "preregistration.json") in (SC / "preregistration.md").read_text(encoding="utf-8")
    assert doc["declared"]["beta_arm"] == [-0.5, 0.0, 0.3]
    assert "3" in doc["rules"]["C"] and "1.5" in doc["rules"]["A"]
    res = SC / "results.json"
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8"))
        manifest = REPO / "data" / "manifests" / "walker2009.json"
        if not manifest.exists():
            assert r["verdict"] == "INDETERMINADO" and "fallo cerrado" in r["reason"]


@pytest.mark.parametrize("script", ["run_oort_kz.py", "run_sculptor_profile.py"])
def test_analyzers_fail_closed_without_prereg(tmp_path, script):
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.analyze()"
            % (str(REPO), str(REPO / "scripts" / script), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)
