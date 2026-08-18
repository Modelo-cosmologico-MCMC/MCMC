"""Tests offline de la capa única de datos (sin red; los datos
commiteados hacen esto ejecutable desde un clone limpio)."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

from mcmc_ontology.data_registry import (  # noqa: E402
    STATES,
    load_sources,
    manifest_path,
    require_available,
    require_schema,
    sha256_of,
    status,
    verify,
    verify_all_available,
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

    import mcmc_ontology.data_registry as dr
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


def _fake_registry(tmp_path, monkeypatch, state="AVAILABLE",
                   with_schema=True, corrupt=False):
    """Registro sintético aislado para probar los guards 6A.1 sin
    tocar el raw real."""
    import shutil

    import mcmc_ontology.data_registry as dr
    (tmp_path / "data" / "raw" / "desi_dr2").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    src = REPO / "data" / "raw" / "desi_dr2"
    f = "desi_gaussian_bao_ALL_GCcomb_mean.txt"
    shutil.copy2(src / f, tmp_path / "data" / "raw" / "desi_dr2" / f)
    transforms = []
    if with_schema:
        rep = tmp_path / "data" / "manifests" / "x.schema_report.md"
        rep.write_text("# schema sintético (fixture)")
        transforms.append("esquema confirmado: data/manifests/x.schema_report.md")
    doc = {"dataset": "desi_dr2_bao", "state": state,
           "transformations": transforms,
           "files": [{"file": f, "sha256": sha256_of(src / f),
                      "bytes": (src / f).stat().st_size}]}
    (tmp_path / "data" / "manifests" / "desi_dr2_bao.json").write_text(
        json.dumps(doc))
    if corrupt:
        t = tmp_path / "data" / "raw" / "desi_dr2" / f
        t.write_text(t.read_text() + "x")
    monkeypatch.setattr(dr, "ROOT", tmp_path)
    monkeypatch.setattr(dr, "MANIFESTS", tmp_path / "data" / "manifests")
    monkeypatch.setattr(dr, "SOURCES", REPO / "data" / "sources.yaml")
    return dr


def test_likelihood_refuses_corrupt_dataset(tmp_path, monkeypatch):
    """6A.1: un fichero presente pero alterado NO puede consumirse —
    require_available re-verifica sha256 en tiempo de ejecución y
    falla cerrado."""
    dr = _fake_registry(tmp_path, monkeypatch, corrupt=True)
    import pytest as _pt
    with _pt.raises(RuntimeError, match="INVALID_CHECKSUM"):
        dr.require_available("desi_dr2_bao")


def test_likelihood_refuses_schema_unverified(tmp_path, monkeypatch):
    """6A.1: sin esquema confirmado en el manifest, el consumo por
    likelihood falla cerrado aunque los bytes sean válidos."""
    dr = _fake_registry(tmp_path, monkeypatch, with_schema=False)
    import pytest as _pt
    with _pt.raises(RuntimeError, match="SCHEMA_UNVERIFIED"):
        dr.require_available("desi_dr2_bao")
    dr2 = _fake_registry(tmp_path, monkeypatch, state="SCHEMA_UNVERIFIED")
    with _pt.raises(RuntimeError, match="SCHEMA_UNVERIFIED"):
        dr2.require_available("desi_dr2_bao")


def test_require_available_happy_path_and_verify_all():
    """Sobre el registro REAL: require_available devuelve el raw_dir
    de desi_dr2_bao (estado sano) y verify_all_available re-verifica
    todo sin corrupciones."""
    raw = require_available("desi_dr2_bao")
    assert (raw / "desi_gaussian_bao_ALL_GCcomb_mean.txt").exists()
    require_schema("desi_dr2_bao")
    states = verify_all_available()
    assert states["desi_dr2_bao"] == "AVAILABLE"
    assert "INVALID_CHECKSUM" not in states.values()


def test_pantheon_legacy_gap_is_enforced():
    """El hueco declarado de Pantheon+ (manifest sin esquema
    confirmado al estándar nuevo) lo hace INCONSUMIBLE por loaders
    nuevos hasta su migración — el guard lo documenta ejecutablemente."""
    import pytest as _pt
    with _pt.raises(RuntimeError, match="SCHEMA_UNVERIFIED"):
        require_schema("pantheonplus")
