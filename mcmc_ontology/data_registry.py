#!/usr/bin/env python
"""Capa única de datos observacionales reproducibles (data/sources.yaml).

NÚCLEO IMPORTABLE (6A.1): los loaders científicos consumen datos SOLO
a través de require_available() — manifest presente, estado AVAILABLE,
esquema confirmado y TODOS los sha256 re-verificados en tiempo de
ejecución; cualquier otra cosa falla cerrado. El CLI vive en
scripts/data_registry.py.

Contrato:
- Los SHA256 NUNCA se escriben a mano: se computan al ingerir y
  quedan en data/manifests/<dataset>.json junto con fuente, versión
  (commit git si procede), fecha UTC, bytes, cita y transformaciones.
- Estados: AVAILABLE (bytes verificados y esquema confirmado),
  SCHEMA_UNVERIFIED (bytes verificados, schema_report pendiente —
  ningún likelihood puede consumirlo), DATA_UNAVAILABLE (fuente
  inalcanzable; se falla cerrado), INVALID_CHECKSUM (verificación
  fallida; el fichero se renombra .REJECTED).
- Ingesta git: ingest_from_clone copia artefactos de un clone local
  cuyo HEAD queda registrado como versión — la procedencia es el
  commit, no una URL suelta.

Uso:
    python scripts/data_registry.py status
    python scripts/data_registry.py verify <dataset>
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources.yaml"
MANIFESTS = ROOT / "data" / "manifests"

STATES = ("AVAILABLE", "DATA_UNAVAILABLE", "INVALID_CHECKSUM",
          "SCHEMA_UNVERIFIED")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources() -> dict:
    return yaml.safe_load(SOURCES.read_text(encoding="utf-8"))


def manifest_path(dataset: str) -> Path:
    return MANIFESTS / f"{dataset}.json"


def write_manifest(dataset: str, entries: list[dict], source: str,
                   version: str, citation: str, state: str,
                   transformations: list[str] | None = None) -> Path:
    """Escribe el manifest de un dataset tras una ingesta REAL (los
    sha256 de `entries` deben venir de sha256_of, jamás a mano)."""
    assert state in STATES, state
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    doc = {
        "dataset": dataset,
        "source": source,
        "version": version,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "citation": citation,
        "state": state,
        "transformations": transformations or [],
        "files": entries,
    }
    p = manifest_path(dataset)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


def ingest_from_clone(dataset: str, clone_dir: Path, subdir: str,
                      dest_dir: Path, citation: str,
                      state: str = "SCHEMA_UNVERIFIED") -> Path:
    """Ingesta desde un clone git local: copia los ficheros de
    clone_dir/subdir a dest_dir, computa sha256 y registra el commit
    HEAD del clone como versión. El estado por defecto es
    SCHEMA_UNVERIFIED: los bytes están, el esquema aún no está
    confirmado por schema_report."""
    clone_dir = Path(clone_dir)
    src = clone_dir / subdir
    if not src.is_dir():
        raise FileNotFoundError(f"{src} no existe (¿clone incompleto?)")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone_dir,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    remote = subprocess.run(["git", "remote", "get-url", "origin"],
                            cwd=clone_dir, capture_output=True,
                            text=True, check=True).stdout.strip()
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        dest = dest_dir / f.name
        shutil.copy2(f, dest)
        entries.append({"file": f.name, "sha256": sha256_of(dest),
                        "bytes": dest.stat().st_size})
    return write_manifest(dataset, entries,
                          source=f"{remote} @ {subdir}", version=head,
                          citation=citation, state=state,
                          transformations=[
                              "copia literal del clone (sin "
                              "transformación de contenido)"])


def verify(dataset: str) -> str:
    """Re-verifica los sha256 del manifest contra los ficheros en
    disco. INVALID_CHECKSUM renombra el fichero a .REJECTED."""
    p = manifest_path(dataset)
    if not p.exists():
        return "DATA_UNAVAILABLE"
    doc = json.loads(p.read_text(encoding="utf-8"))
    src = load_sources()["datasets"].get(dataset, {})
    raw_dir = ROOT / src.get("raw_dir", f"data/raw/{dataset}")
    for e in doc["files"]:
        f = raw_dir / e["file"]
        if not f.exists():
            return "DATA_UNAVAILABLE"
        if sha256_of(f) != e["sha256"]:
            f.rename(f.with_suffix(f.suffix + ".REJECTED"))
            return "INVALID_CHECKSUM"
    return doc["state"]


def status() -> dict:
    out = {}
    for name in load_sources()["datasets"]:
        p = manifest_path(name)
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))["state"]
        else:
            out[name] = load_sources()["datasets"][name].get(
                "state", "DATA_UNAVAILABLE")
    return out


def require_schema(dataset: str) -> None:
    """Exige que el manifest registre la confirmación de esquema
    (transformación «esquema confirmado: <schema_report>») y que ese
    fichero exista. Sin esquema confirmado, ningún likelihood consume
    el dataset — falla cerrado."""
    p = manifest_path(dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"DATA_UNAVAILABLE: {dataset} sin manifest ({p})")
    doc = json.loads(p.read_text(encoding="utf-8"))
    refs = [t for t in doc.get("transformations", [])
            if "esquema confirmado" in t]
    if not refs:
        raise RuntimeError(
            f"SCHEMA_UNVERIFIED: {dataset} no tiene esquema confirmado "
            "en el manifest — el schema_report es prerequisito de "
            "cualquier consumo por likelihood")
    for t in refs:
        rep = ROOT / t.split("esquema confirmado:")[1].strip()
        if not rep.exists():
            raise RuntimeError(
                f"SCHEMA_UNVERIFIED: {dataset} referencia un "
                f"schema_report inexistente ({rep})")


def require_available(dataset: str) -> Path:
    """El guard runtime de 6A.1 — obligatorio para todo loader
    científico: manifest presente → estado AVAILABLE → esquema
    confirmado → TODOS los sha256 re-verificados AHORA contra el
    disco. Solo entonces devuelve el raw_dir. Cualquier otro caso
    falla cerrado (una corrida individual no puede consumir un
    fichero presente pero alterado)."""
    p = manifest_path(dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"DATA_UNAVAILABLE: {dataset} sin manifest ({p})")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("state") != "AVAILABLE":
        raise RuntimeError(
            f"{doc.get('state', 'DESCONOCIDO')}: {dataset} no está "
            "AVAILABLE — el consumo científico falla cerrado")
    require_schema(dataset)
    state = verify(dataset)
    if state != "AVAILABLE":
        raise RuntimeError(
            f"{state}: verificación sha256 de {dataset} fallida — "
            "fichero apartado como .REJECTED; re-ingerir de la fuente")
    src = load_sources()["datasets"].get(dataset, {})
    return ROOT / src.get("raw_dir", f"data/raw/{dataset}")


def verify_all_available() -> dict:
    """Re-verifica TODOS los datasets con manifest; devuelve
    {dataset: estado} y lanza si alguno presente está corrupto."""
    out = {}
    bad = []
    for name in load_sources()["datasets"]:
        if manifest_path(name).exists():
            st = verify(name)
            out[name] = st
            if st == "INVALID_CHECKSUM":
                bad.append(name)
        else:
            out[name] = "DATA_UNAVAILABLE"
    if bad:
        raise RuntimeError(f"INVALID_CHECKSUM en: {bad}")
    return out
