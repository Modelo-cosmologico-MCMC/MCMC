#!/usr/bin/env python
"""CLI de la capa de datos (el núcleo importable vive en
mcmc_ontology/data_registry.py — 6A.1)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcmc_ontology.data_registry import (  # noqa: E402, F401
    STATES,
    ingest_from_clone,
    load_sources,
    manifest_path,
    require_available,
    require_schema,
    sha256_of,
    status,
    verify,
    verify_all_available,
    write_manifest,
)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        for k, v in status().items():
            print(f"{k:20s} {v}")
    elif cmd == "verify":
        print(verify(sys.argv[2]))
    elif cmd == "verify_all":
        for k, v in verify_all_available().items():
            print(f"{k:20s} {v}")
