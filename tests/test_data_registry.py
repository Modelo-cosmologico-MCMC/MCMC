"""Tests offline de la capa única de datos (sin red; los datos
commiteados hacen esto ejecutable desde un clone limpio)."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from data_registry import (  # noqa: E402
    STATES,
    load_sources,
    manifest_path,
    sha256_of,
    status,
    verify,
)


def test_sources_registry_parses_and_states_valid():
    """sources.yaml parsea y todo estado declarado es del vocabulario
    (o pending_ingest transitorio, que no debe sobrevivir a una
    ingesta)."""
    src = load_sources()["datasets"]
    assert {"sparc", "walker2009", "desi_dr2_bao",
            "pantheonplus", "union3", "union3_1",
            "des_dovekie"} <= set(src)
    for name, st in status().items():
        assert st in STATES or st == "pending_ingest", (name, st)


def test_manifests_have_computed_checksums():
    """Todo manifest lista sha256 hexadecimales de 64 chars con bytes
    coherentes — y verify() los reproduce sobre el disco (nunca a
    mano)."""
    for ds in ("desi_dr2_bao", "union3", "union3_1", "des_dovekie"):
        doc = json.loads(manifest_path(ds).read_text(encoding="utf-8"))
        assert doc["state"] == "AVAILABLE"
        assert doc["version"], ds
        for e in doc["files"]:
            assert len(e["sha256"]) == 64
            assert e["bytes"] > 0
        assert verify(ds) == "AVAILABLE", ds


def test_verify_detects_corruption(tmp_path, monkeypatch):
    """INVALID_CHECKSUM: un byte cambiado se detecta y el fichero se
    aparta (.REJECTED) — con una copia sintética, sin tocar el raw."""
    import shutil

    import data_registry as dr
    fake_root = tmp_path
    (fake_root / "data" / "raw" / "desi_dr2").mkdir(parents=True)
    (fake_root / "data" / "manifests").mkdir(parents=True)
    src = REPO / "data" / "raw" / "desi_dr2"
    f = "desi_gaussian_bao_ALL_GCcomb_mean.txt"
    shutil.copy2(src / f, fake_root / "data" / "raw" / "desi_dr2" / f)
    doc = {"dataset": "desi_dr2_bao", "state": "AVAILABLE",
           "files": [{"file": f, "sha256": sha256_of(src / f),
                      "bytes": (src / f).stat().st_size}]}
    (fake_root / "data" / "manifests" / "desi_dr2_bao.json").write_text(
        json.dumps(doc))
    monkeypatch.setattr(dr, "ROOT", fake_root)
    monkeypatch.setattr(dr, "MANIFESTS", fake_root / "data" / "manifests")
    monkeypatch.setattr(dr, "SOURCES", REPO / "data" / "sources.yaml")
    assert dr.verify("desi_dr2_bao") == "AVAILABLE"
    target = fake_root / "data" / "raw" / "desi_dr2" / f
    target.write_text(target.read_text() + "x")
    assert dr.verify("desi_dr2_bao") == "INVALID_CHECKSUM"
    assert not target.exists()
