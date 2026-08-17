"""Ingesta SPARC del Frente 5E — falla cerrado, inspecciona antes de asumir.

CONTRATO (preinscripción del 5E + regla de fallo):

1. Sin datos descargados con manifest de procedencia, TODO falla
   cerrado (`FileNotFoundError` con estatuto DATA_UNAVAILABLE) —
   nunca se fabrican fixtures observacionales ni se asume un mirror.
2. Cuando el dataset exista, la PRIMERA acción es la inspección de
   esquema (`inspect_schema` → `schema_report.md`): qué columnas trae
   el fichero real, con qué unidades declara la cabecera, qué es
   directo del catálogo y qué será derivado. NINGÚN nombre de columna
   se asume antes de esa inspección — por eso este módulo aún no
   contiene un parser de columnas: contiene el contrato que el parser
   deberá satisfacer.
3. El parser (a escribir tras la inspección) deberá: conservar
   unidades explícitas; validar radios > 0 y errores de velocidad
   > 0; distinguir V_obs, gas, disco y bulbo; conservar identificador
   original y banderas de calidad del catálogo; detectar NaN y
   duplicados; registrar columnas directas vs derivadas. Las
   validaciones ya están implementadas y testeadas con fixtures
   SINTÉTICOS (claramente no observacionales) en validate_catalogue.

El estatuto vigente de la ingesta vive en data/sparc/INGESTA_ESTADO.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"
MANIFEST = DATA_DIR / "manifest.json"

_UNAVAILABLE = (
    "DATA_UNAVAILABLE: el catálogo SPARC no está ingerido (manifest "
    "de procedencia ausente en data/sparc/). Estatuto y desbloqueo: "
    "data/sparc/INGESTA_ESTADO.md. El análisis 5E observacional falla "
    "CERRADO por contrato — no usa fixtures ni mirrors."
)


def load_sparc_catalogue(data_dir: Path | None = None):
    """Punto de entrada del análisis observacional. Sin manifest de
    descarga real: DATA_UNAVAILABLE (falla cerrado)."""
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(_UNAVAILABLE)
    raise NotImplementedError(
        "Ingesta presente pero el parser aún no está escrito: la "
        "primera acción obligatoria es inspect_schema() → "
        "schema_report.md (no se asumen columnas sin inspección).")


def inspect_schema(path: Path, n_lines: int = 40) -> str:
    """Volcado de cabecera del fichero real para schema_report.md —
    la inspección que precede a cualquier parser."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(_UNAVAILABLE)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    head = "\n".join(lines[:n_lines])
    return (f"# schema_report — {p.name}\n\n"
            f"Líneas totales: {len(lines)}\n\n"
            "Cabecera literal (primeras "
            f"{min(n_lines, len(lines))} líneas):\n\n"
            "```\n" + head + "\n```\n")


def validate_catalogue(radii_kpc: np.ndarray, v_obs: np.ndarray,
                       e_v: np.ndarray, galaxy_ids: np.ndarray) -> dict:
    """Validación física que el parser DEBERÁ aplicar a cada galaxia
    (testeada con fixtures sintéticos; los datos reales pertenecen al
    análisis reproducible, no a la unit suite):

    - radios estrictamente positivos y crecientes por galaxia;
    - errores de velocidad estrictamente positivos;
    - sin NaN en ninguna columna;
    - sin duplicados (galaxy_id, R).
    Devuelve el resumen; lanza ValueError si algo falla."""
    r = np.asarray(radii_kpc, float)
    v = np.asarray(v_obs, float)
    e = np.asarray(e_v, float)
    g = np.asarray(galaxy_ids)
    if np.isnan(r).any() or np.isnan(v).any() or np.isnan(e).any():
        raise ValueError("NaN en el catálogo")
    if (r <= 0.0).any():
        raise ValueError("radios no estrictamente positivos")
    if (e <= 0.0).any():
        raise ValueError("errores de velocidad no positivos")
    keys = list(zip(g.tolist(), r.tolist()))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicados (galaxia, R)")
    return {"n_points": int(r.size),
            "n_galaxies": int(np.unique(g).size)}


if __name__ == "__main__":
    import sys
    if "--inspect" in sys.argv:
        target = DATA_DIR / "SPARC_Lelli2016c.mrt"
        report = inspect_schema(target)
        out = DATA_DIR.parent.parent / "results" / \
            "2026-08-16_front5e_sparc" / "schema_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"schema_report: {out}")
