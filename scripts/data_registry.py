#!/usr/bin/env python
"""Capa única de datos observacionales reproducibles (data/sources.yaml).

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
import sys
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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        for k, v in status().items():
            print(f"{k:20s} {v}")
    elif cmd == "verify":
        print(verify(sys.argv[2]))
