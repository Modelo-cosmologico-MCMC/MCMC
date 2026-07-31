#!/usr/bin/env python
"""Descarga de los catálogos observacionales públicos a data/.

Uso:
    python scripts/download_data.py [pantheon|sparc|boss_eboss|planck2018|all]

Solo usa la biblioteca estándar (urllib + hashlib). Cada descarga se
verifica por SHA-256: si el registro tiene un checksum fijado, la descarga
se rechaza en caso de no coincidir; si aún no lo tiene (None), el script
imprime el checksum calculado para fijarlo en este registro (trust on
first use, documentado).

Catálogos y licencias/condiciones:

- Pantheon+ (SNe Ia): repositorio público PantheonPlusSH0ES/DataRelease.
  Citar Scolnic et al. 2022 (ApJ 938, 113) y Brout et al. 2022
  (ApJ 938, 110).
- SPARC (curvas de rotación): http://astroweb.cwru.edu/SPARC/ — uso
  académico con cita a Lelli, McGaugh & Schombert 2016 (AJ 152, 157).
- Cronómetros cósmicos + BAO: tablas compiladas de la literatura
  (referencias en cada fichero generado); los valores BAO DR12 proceden
  del consenso de Alam et al. 2017 (MNRAS 470, 2617).
- Planck 2018: DESVIACIÓN DECLARADA — en lugar de descargar los
  productos oficiales (pesados, PLA), este script escribe la geometría
  comprimida (R, l_A, ω_b) publicada para Planck 2018 TT,TE,EE+lowE
  (Chen, Huang & Wang 2019, JCAP 02, 028). Es una elección de tamaño,
  no de fuente: los productos completos están en
  https://pla.esac.esa.int/ y pueden colocarse en data/planck2018/.

Los ficheros compilados que este script ESCRIBE (no descarga) llevan
cabecera con la referencia bibliográfica de cada punto.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# --- Registro de descargas: (url, destino, sha256 esperado o None) ---
DOWNLOADS = {
    "pantheon": [
        (
            "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
            "main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
            DATA / "pantheon" / "PantheonPlusSH0ES.dat",
            # Verificado el 31-jul-2026 (579 283 bytes):
            "1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
        ),
        (
            "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
            "main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
            "Pantheon%2BSH0ES_STAT%2BSYS.cov",
            DATA / "pantheon" / "PantheonPlusSH0ES_STATSYS.cov",
            # Verificado el 31-jul-2026 (33 284 960 bytes):
            "abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc",
        ),
    ],
    "sparc": [
        (
            "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip",
            DATA / "sparc" / "Rotmod_LTG.zip",
            None,  # pendiente: fijar en la primera descarga verificada
        ),
    ],
}

# --- Cronómetros cósmicos: compilación estándar (31 puntos) ---
# Formato: z, H [km/s/Mpc], sigma_H. Referencias por bloque.
CC_TABLE = """\
# Cronómetros cósmicos H(z) — compilación de la literatura.
# Columnas: z  H[km/s/Mpc]  sigma_H
# Refs: Zhang et al. 2014 (RAA 14, 1221); Simon et al. 2005 (PRD 71, 123001);
#       Moresco et al. 2012 (JCAP 08, 006); Moresco et al. 2016 (JCAP 05, 014);
#       Stern et al. 2010 (JCAP 02, 008); Moresco 2015 (MNRAS 450, L16);
#       Ratsimbazafy et al. 2017 (MNRAS 467, 3239); Borghi et al. 2022 (ApJL 928, L4).
0.070   69.0  19.6
0.090   69.0  12.0
0.120   68.6  26.2
0.170   83.0   8.0
0.179   75.0   4.0
0.199   75.0   5.0
0.200   72.9  29.6
0.270   77.0  14.0
0.280   88.8  36.6
0.352   83.0  14.0
0.380   83.0  13.5
0.400   95.0  17.0
0.4004  77.0  10.2
0.425   87.1  11.2
0.445   92.8  12.9
0.470   89.0  49.6
0.4783  80.9   9.0
0.480   97.0  62.0
0.593  104.0  13.0
0.680   92.0   8.0
0.750   98.8  33.6
0.781  105.0  12.0
0.875  125.0  17.0
0.880   90.0  40.0
0.900  117.0  23.0
1.037  154.0  20.0
1.300  168.0  17.0
1.363  160.0  33.6
1.430  177.0  18.0
1.530  140.0  14.0
1.750  202.0  40.0
"""

# --- BAO BOSS DR12 (consenso, Alam et al. 2017) ---
# D_M/r_d y H·r_d (convertido a D_H/r_d = c/(H·r_d)) por bin de z.
BAO_TABLE = """\
# BAO — consenso BOSS DR12 (Alam et al. 2017, MNRAS 470, 2617).
# Columnas: z  tipo  valor  sigma
# tipo: DM_over_rd (distancia comóvil transversal / r_d)
#       DH_over_rd (c / (H(z)·r_d))
0.38  DM_over_rd  10.27  0.15
0.38  DH_over_rd  24.89  0.58
0.51  DM_over_rd  13.38  0.18
0.51  DH_over_rd  22.43  0.48
0.61  DM_over_rd  15.45  0.22
0.61  DH_over_rd  20.86  0.45
"""

# --- Planck 2018: geometría comprimida (desviación declarada arriba) ---
PLANCK_TABLE = """\
# Planck 2018 TT,TE,EE+lowE — geometría comprimida.
# Fuente: Chen, Huang & Wang 2019 (JCAP 02, 028), tabla 1.
# Columnas: parametro  valor  sigma
R        1.75020  0.00460
l_A    301.47100  0.09000
omega_b  0.02236  0.00015
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} ya existe")
    else:
        print(f"[get ] {url}")
        urllib.request.urlretrieve(url, dest)  # noqa: S310 — URLs del registro
    digest = _sha256(dest)
    if expected_sha256 is None:
        print(f"[hash] {dest.name}: sha256={digest}")
        print("       (sin checksum fijado: verifícalo y fíjalo en DOWNLOADS)")
    elif digest != expected_sha256:
        dest.rename(dest.with_suffix(dest.suffix + ".REJECTED"))
        raise RuntimeError(
            f"Checksum inválido para {dest.name}: {digest} != {expected_sha256}"
        )
    else:
        print(f"[ok  ] {dest.name}: sha256 verificado")


def write_compiled_tables() -> None:
    """Escribe las tablas compiladas (con referencias) que no se descargan."""
    targets = {
        DATA / "boss_eboss" / "hz_cc.txt": CC_TABLE,
        DATA / "boss_eboss" / "bao_dr12.txt": BAO_TABLE,
        DATA / "planck2018" / "compressed_geometry.txt": PLANCK_TABLE,
    }
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[write] {path.relative_to(DATA.parent)}  sha256={_sha256(path)}")


def main(argv: list[str]) -> None:
    what = argv[1] if len(argv) > 1 else "all"
    if what in ("boss_eboss", "planck2018", "all"):
        write_compiled_tables()
    for key, entries in DOWNLOADS.items():
        if what not in (key, "all"):
            continue
        for url, dest, sha in entries:
            download(url, dest, sha)
    print("\nHecho. Recuerda citar cada catálogo según su cabecera/licencia.")


if __name__ == "__main__":
    main(sys.argv)
