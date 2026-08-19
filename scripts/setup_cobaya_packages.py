#!/usr/bin/env python
"""Recrea el packages path de Cobaya para la equivalencia 6A.2.

La validación externa (scripts/run_desi_dr2_benchmark.py) necesita
cobaya == 3.6.2 (pip) y su dataset bao_data en el tag v2.6. Este
script deja el dataset en external/cobaya_packages (gitignorado) — o
en la ruta que indique COBAYA_PACKAGES_PATH. Es la ÚNICA pieza de la
cadena que requiere red; la suite de tests nunca lo llama (el test de
equivalencia hace skip declarado si el path no existe).

Documentación ejecutable de lo que automatiza:

    pip install cobaya==3.6.2
    git clone https://github.com/CobayaSampler/bao_data \\
        <packages>/data/bao_data
    git -C <packages>/data/bao_data checkout v2.6
    echo v2.6 > <packages>/data/bao_data/version.dat   # Cobaya lo lee

La identidad de los ficheros descargados contra la ingesta del repo
(manifest bb0c1c9) la computa después run_desi_dr2_benchmark.py por
sha256 — aquí no se afirma nada.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGES = Path(os.environ.get(
    "COBAYA_PACKAGES_PATH", str(REPO / "external" / "cobaya_packages")))
BAO_URL = "https://github.com/CobayaSampler/bao_data"
TAG = "v2.6"


def main() -> None:
    dest = PACKAGES / "data" / "bao_data"
    if (dest / "version.dat").exists():
        print(f"ya existe: {dest} "
              f"(versión {(dest / 'version.dat').read_text().strip()})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    (PACKAGES / "code").mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", BAO_URL, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", TAG],
                   check=True)
    (dest / "version.dat").write_text(TAG + "\n", encoding="utf-8")
    print(f"packages path listo: {PACKAGES} (bao_data @ {TAG})")


if __name__ == "__main__":
    sys.exit(main())
